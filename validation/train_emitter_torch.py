"""Train the 450M program emitter (H-E15) in torch: LoRA over the frozen base, on SLOTTED programs.

The same adapter mechanics as the talk adapter (`train_talk_torch.py`: LoRA on every projection of every
layer, loss on the target tokens only, AdamW warmup+cosine, bf16 autocast) and the same adapter file
layout, so the adapter loads into grilly2 on the local card (`standin/emitter.py: Cubby450mEmitter`).
What differs is the data (`emitter_data.py`: the emitter SFT set in slot form -- the emitter writes
`bind frame, H1_CAPITAL, "$E2"` and `assign s0 = $N1;`, never a name or a number the host could have
placed) and the read: after training, the val split is GENERATED (greedy, no repetition guards: a
program repeats `create frame: number;` on purpose), each program is checked against its slot table
(unknown slot / copied literal = a refusal), filled, and written to `val_generations.json` in the
shape `standin/eval_emitter_vm.py --val-generations` replays through the real VM. With a cubelang
binary on the machine (`--vm`) the VM read runs here too.

    python validation/train_emitter_torch.py --ckpt <base450m_final.pt> --tokenizer <bbpe128k.json> \
        --data <emitter_sft_v12e_slots.jsonl> --out <adapter dir> [--epochs 2] [--batch 16] [--tag _colab]
    python validation/train_emitter_torch.py --smoke --tokenizer <bbpe128k.json> --data <slots.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if "--smoke" in sys.argv:                    # train_base reads its shape from the environment at import
    os.environ.update(CB_D="64", CB_L="4", CB_HEADS="4", CB_WINDOW="16", CB_GEN_BASIS="4", CB_GEN_RANK="2")

import torch  # noqa: E402

from emitter_data import decode_program, encode_records, fill, table_of, val_records  # noqa: E402
from talk_data import batches, lora_targets, make_batch  # noqa: E402
from train_talk_torch import apply_lora, load_adapter, masked_loss, save_adapter  # noqa: E402

ADAPTER = "emitter_lora"
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def emit(model, tk, eos, prompt: str, dev, max_new: int) -> str:
    """Greedy program generation with the base's own step(): no repetition penalty, no n-gram ban."""
    from cubbyllm.core.decoding import DecodeConfig, generate
    cfg = DecodeConfig(max_new_tokens=max_new, temperature=0.0, repetition_penalty=1.0, no_repeat_ngram=0,
                       eos_id=eos)
    ids = tk.encode(prompt).ids

    def step_fn(tok, state):
        return model.step(tok.to(dev), state)
    with torch.no_grad():
        out, _ = generate(model, ids, cfg, state=model.init_state(1, dev), step=step_fn)
    return decode_program(tk, out, eos)          # keeps the role tokens (ACTION/AGENT/OBJECT are special)


def generate_val(model, tk, eos, records, dev, max_new, out_path, tag, vm=False) -> dict:
    """Generate every val record, check and fill its slots, write the replay file, summarise."""
    t0 = time.time()
    outputs, stats = [], Counter()
    for i, r in enumerate(records, 1):
        gen = emit(model, tk, eos, r["prompt"], dev, max_new)
        table = table_of(r["spans"], r["question"])
        verdict = table.check(gen)
        filled = table.fill(gen)
        stats[f"{r['task']}:n"] += 1
        stats[f"{r['task']}:slots_ok"] += int(verdict.ok)
        stats[f"{r['task']}:copied"] += int(bool(verdict.copied))
        stats[f"{r['task']}:unknown_slot"] += int(verdict.reason.startswith("unknown slot"))
        stats[f"{r['task']}:text_exact"] += int(" ".join(gen.split()) == " ".join(r["program"].split()))
        outputs.append({"id": r["id"], "task": r["task"], "subtype": r.get("subtype", ""), "prompt": r["prompt"],
                        "reference": r["reference"], "gold": r.get("gold"), "generated": filled,
                        "generated_slotted": gen, "slots_ok": verdict.ok, "slot_reason": verdict.reason,
                        "unused_slots": verdict.unused_entities})
        if i % 20 == 0:
            log(f"  generated {i}/{len(records)} ({time.time() - t0:.0f}s)")
    summary = {}
    for task in sorted({r["task"] for r in records}):
        n = stats[f"{task}:n"]
        summary[task] = {"n": n, "slots_ok": stats[f"{task}:slots_ok"] / n, "copied": stats[f"{task}:copied"] / n,
                         "unknown_slot": stats[f"{task}:unknown_slot"] / n, "text_exact": stats[f"{task}:text_exact"] / n}
        log(f"  {task:13s} n={n:4d} slots_ok={summary[task]['slots_ok']:.3f} copied={summary[task]['copied']:.3f} "
            f"unknown_slot={summary[task]['unknown_slot']:.3f} text_exact={summary[task]['text_exact']:.3f}")
    payload = {"model": f"cubby450m{tag}", "adapter": ADAPTER, "n": len(records), "summary": summary,
               "outputs": outputs, "wall_s": time.time() - t0,
               "note": "generated = slots filled by the host (what the VM runs); generated_slotted = as emitted"}
    if vm:
        payload["vm"] = vm_read(outputs)
    json.dump(payload, open(out_path, "w", encoding="utf-8"), indent=1)
    log(f"wrote {out_path}")
    return payload


def vm_read(outputs) -> dict:
    """The VM read, when a cubelang binary is reachable: executes / gold_match per family."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "standin", "data"))
    from build_emitter_sft import answer_fn, gold_matches, shim_isolver
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.slots import _norm
    st = Counter()
    for o in outputs:
        src = shim_isolver(o["generated"])
        try:
            out = cc.run_program_proto(src, fn=answer_fn(src))
            ok, res = bool(out.get("ok")), out.get("result")
        except Exception:
            ok, res = False, None
        o["executes"], o["vm_result"] = ok, None if res is None else str(res)[:120]
        gm = None
        if ok and o.get("gold") is not None:
            gm = gold_matches(res, o["gold"]) or (_norm(str(res)) == _norm(str(o["gold"])))
        o["gold_match"] = gm
        st[f"{o['task']}:n"] += 1
        st[f"{o['task']}:executes"] += int(ok)
        if o.get("gold") is not None:
            st[f"{o['task']}:with_gold"] += 1
            st[f"{o['task']}:gold_match"] += int(bool(gm))
    summary = {}
    for task in sorted({o["task"] for o in outputs}):
        n, wg = st[f"{task}:n"], st[f"{task}:with_gold"]
        summary[task] = {"n": n, "executes": st[f"{task}:executes"] / n,
                         "gold_match": (st[f"{task}:gold_match"] / wg) if wg else None}
        gm = summary[task]["gold_match"]
        log(f"  VM {task:13s} n={n:4d} executes={summary[task]['executes']:.3f} "
            f"gold_match={'—' if gm is None else f'{gm:.3f}'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True, help="the slot-form jsonl (emitter_data.py convert)")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--gen-per-task", type=int, default=64, help="val records generated per family at the end (0 = all)")
    ap.add_argument("--gen-max-new", type=int, default=640)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--gen-extra", default="", help="a second slot-form jsonl whose val records are generated too "
                                                    "(H-E18: the program-first held-out eval, written by another writer)")
    ap.add_argument("--gen-extra-per-task", type=int, default=0, help="records per family from --gen-extra (0 = all)")
    ap.add_argument("--gen-extra-tag", default="_pfheld", help="suffix of the second generations file")
    ap.add_argument("--vm", action="store_true", help="also run the VM read here (needs cubelang on this machine)")
    ap.add_argument("--resume-adapter", default="", help="start from this adapter dir instead of zero")
    ap.add_argument("--gen-only", action="store_true", help="no training: regenerate with --resume-adapter "
                                                            "(the previous generations file is kept as .prev.json)")
    ap.add_argument("--gen-tasks", default="", help="comma-separated families to generate (default: all); with "
                                                    "--gen-only the other families of the previous file are kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = dev.type == "cuda"
    if amp:
        torch.backends.cuda.matmul.allow_tf32 = True

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    eos = tk.token_to_id("</s>")
    t0 = time.time()
    if args.smoke:
        from train_base import arch_meta, build_model
        meta = arch_meta(tk.get_vocab_size(), ((tk.get_vocab_size() + 127) // 128) * 128)
        model = build_model(meta, dev)
        args.steps = args.steps or 3
        args.batch, args.eval_n, args.gen_per_task, args.gen_max_new = 2, 4, 2, 24
        args.gen_extra_per_task = min(args.gen_extra_per_task or 2, 2)
    else:
        from train_base import load_base
        model = load_base(args.ckpt, str(dev))
        meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)["meta"]
    for p in model.parameters():
        p.requires_grad_(False)
    if args.resume_adapter:
        ameta = load_adapter(model, args.resume_adapter, ADAPTER)
        targets, wrapped = ameta["targets"], {t: _find(model, t) for t in ameta["targets"]}
    else:
        targets = lora_targets(meta["L"], meta["attn_every"])
        wrapped = apply_lora(model, targets, args.rank, args.alpha)
    params = [p for w in wrapped.values() for p in (w.lora_A.weight, w.lora_B.weight)]
    for p in params:
        p.requires_grad_(True)
    logs = os.path.join(HERE, "logs")
    os.makedirs(logs, exist_ok=True)
    if args.gen_only:                        # regenerate from a saved adapter; the training logs stay untouched
        if not args.resume_adapter:
            raise SystemExit("--gen-only needs --resume-adapter (the adapter to generate with)")
        log(f"train_emitter_torch --gen-only | {dev} | adapter {args.resume_adapter} | load {time.time() - t0:.0f}s")
        run_generation(args, model, tk, eos, dev, logs, keep_previous=True)
        with open(os.path.join(logs, f"train_emitter_torch{args.tag}_regen.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(LOG) + "\n")
        return

    train = encode_records(args.data, tk, eos, "train", args.max_len)
    held = encode_records(args.data, tk, eos, "val", args.max_len)
    rng.shuffle(held)
    held = held[:args.eval_n]
    if args.smoke:
        train = train[:16]
    total = args.steps or int(math.ceil(args.epochs * len(train) / args.batch))
    fam = Counter(r["family"] for r in train)
    log(f"train_emitter_torch | {dev} | {len(targets)} LoRA modules, {sum(p.numel() for p in params) / 1e6:.2f}M "
        f"trainable (r={args.rank}, alpha={args.alpha}) | train {len(train):,} rows {dict(fam)} | held-eval {len(held)} | "
        f"{total} steps of {args.batch} | lr {args.lr} warmup {args.warmup} | max_len {args.max_len} | load {time.time() - t0:.0f}s")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, **({"fused": True} if amp else {}))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        frac = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    def evaluate():
        tot, n = 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(held), args.batch):
                x, y, m, _ = make_batch(held[s:s + args.batch])
                k = float(m.sum())
                tot += float(masked_loss(model, x, y, m, dev, amp)) * k
                n += k
        return tot / max(n, 1.0)

    def meta_out(step):
        return {"step": step, "rank": args.rank, "alpha": args.alpha, "targets": targets, "trainer": "torch",
                "adapter": ADAPTER, "ckpt": os.path.basename(args.ckpt) if args.ckpt else "smoke",
                "data": os.path.basename(args.data), "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                "max_len": args.max_len, "tag": args.tag}

    mf = open(os.path.join(logs, f"train_emitter_torch{args.tag}.jsonl"), "w", encoding="utf-8")
    ev = evaluate()
    log(f"  step 0 held program loss {ev:.4f}")
    mf.write(json.dumps({"step": 0, "held_loss": ev}) + "\n")
    order, step, tokens, t_start, run = [], 0, 0, time.time(), []
    while step < total:
        if not order:
            order = batches(train, args.batch, rng)
        x, y, m, _ = make_batch([train[i] for i in order.pop()])
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = masked_loss(model, x, y, m, dev, amp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        lv = float(loss.detach())
        if not math.isfinite(lv):
            log(f"  step {step}: loss {lv} -- stopping without saving")
            break
        run.append(lv)
        tokens += int((x != 0).sum())
        step += 1
        if step % 20 == 0 or step == total:
            el = time.time() - t_start
            mf.write(json.dumps({"step": step, "loss": float(np.mean(run)), "gnorm": gn, "lr": lr_at(step - 1),
                                 "s_per_step": el / step, "tok_s": tokens / el}) + "\n")
            mf.flush()
            log(f"  step {step:5d}/{total}  loss {np.mean(run):.4f}  gnorm {gn:.3f}  lr {lr_at(step - 1):.2e}  "
                f"{el / step:.2f} s/step  {tokens / el:,.0f} tok/s  eta {(total - step) * el / step / 60:.1f} min")
            run = []
        if step % args.eval_every == 0 or step == total:
            ev = evaluate()
            log(f"  step {step} held program loss {ev:.4f}")
            mf.write(json.dumps({"step": step, "held_loss": ev}) + "\n")
        if args.out and (step % args.save_every == 0 or step == total):
            save_adapter(wrapped, args.out, meta_out(step), ADAPTER)
    mf.close()
    log(f"trained: {step} steps in {(time.time() - t_start) / 60:.1f} min" + (f" -> {args.out}" if args.out else ""))

    if not args.no_gen:
        run_generation(args, model, tk, eos, dev, logs)
    with open(os.path.join(logs, f"train_emitter_torch{args.tag}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


def run_generation(args, model, tk, eos, dev, logs, keep_previous=False) -> None:
    """The val split's generations file, and the --gen-extra one beside it when asked."""
    jobs = [(args.data, args.gen_per_task, args.tag, "val records")]
    if args.gen_extra:                           # the same adapter on a second eval set, one more file beside the first
        jobs.append((args.gen_extra, args.gen_extra_per_task, args.tag + args.gen_extra_tag,
                     f"records of {os.path.basename(args.gen_extra)}"))
    tasks = {t for t in (args.gen_tasks or "").split(",") if t}
    for path, per_task, tag, what in jobs:
        recs = [r for r in val_records(path, per_task) if not tasks or r["task"] in tasks]
        if not recs:
            continue
        out_path = os.path.join(args.out or logs, f"val_generations{tag}.json")
        prev = None
        if keep_previous and os.path.exists(out_path):
            prev_path = out_path[:-len(".json")] + ".prev.json"
            os.replace(out_path, prev_path)
            prev = json.load(open(prev_path, encoding="utf-8"))
            log(f"kept the previous file as {os.path.basename(prev_path)}")
        log(f"generating {len(recs)} {what}{' (' + ','.join(sorted(tasks)) + ')' if tasks else ''} "
            f"(greedy, up to {args.gen_max_new} tokens each) ...")
        payload = generate_val(model, tk, eos, recs, dev, args.gen_max_new, out_path, tag, vm=args.vm)
        if tasks and prev is not None:          # only these families regenerated: the rest of the file is kept
            payload["outputs"] = [o for o in prev["outputs"] if o["task"] not in tasks] + payload["outputs"]
            payload["summary"] = {**{k: v for k, v in prev["summary"].items() if k not in tasks}, **payload["summary"]}
            payload["n"] = len(payload["outputs"])
            payload["note"] += f" | regenerated {','.join(sorted(tasks))} with the decode fix; the rest as before"
            payload.pop("vm", None)
            json.dump(payload, open(out_path, "w", encoding="utf-8"), indent=1)
            log(f"merged into {out_path}: {payload['n']} records")


def _find(model, target):
    from talk_data import torch_attr
    i, group, attr = torch_attr(target)
    return getattr(getattr(model.backbone, group)[i], attr)


if __name__ == "__main__":
    main()
