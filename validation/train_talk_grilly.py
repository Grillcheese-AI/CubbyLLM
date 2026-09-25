"""Train the talk adapter on the 450M base, on grilly2 (the local AMD GPU), LoRA over a frozen base.

The job (H-E6): say what the VM returned -- the asked entity's value, the facts over the weights,
and "The facts don't say." when they hold no answer. The data is `standin/data/build_ground_sft.py`'s
(the talk format `Facts: ... Question: ... Answer:`); the loss is on the answer tokens only (the
answer and its end-of-text), never on the facts or the question.

LoRA on every projection of every layer -- the attention's qkv and output, MinGRU's gate, value and
decay, the FFN's three -- rank 16, alpha 32; the base is frozen and never merged, so the adapter is
a separate file (the "two adapters on one base" rule: talk here, programs later). AdamW, warmup then
cosine to 10%, gradient norm clipped at 1.0.

    python validation/train_talk_grilly.py --export <export dir> --tokenizer <bbpe128k.json> \
        --data standin/data/out/ground_sft.jsonl --out <adapter dir> --steps 100 --tag _pilot
Writes <out>/talk_lora.safetensors (+ .json), validation/logs/train_talk_grilly<tag>.{log,jsonl}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import grilly  # noqa: E402  (grilly2)
from grilly.infer.lora import LoRALinear, apply_lora_, lora_parameters  # noqa: E402
from grilly.nn import functional as GF  # noqa: E402
from grilly.nn.utils import clip_grad_norm_  # noqa: E402
from grilly.optim import AdamW  # noqa: E402

from exp_e5_base450m_probes import load_grilly  # noqa: E402
from talk_data import batches, encode_records, make_batch  # noqa: E402
from talk_data import lora_targets as _targets  # noqa: E402

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def lora_targets(model) -> list[str]:
    c = model.config
    return _targets(c.num_hidden_layers, c.attn_every)


def masked_loss(model, x, y, m, keep):
    logits = model(grilly.from_numpy(x), logits_to_keep=keep).logits
    lp = GF.log_softmax(logits, -1)
    tgt = grilly.from_numpy(np.ascontiguousarray(y[:, -keep:])).unsqueeze(-1)
    nll = -lp.gather(-1, tgt).squeeze(-1)
    mask = grilly.from_numpy(np.ascontiguousarray(m[:, -keep:]))
    return (nll * mask).sum() / float(m.sum())


def gpu_mem_mb():
    """This process's GPU memory as Windows counts it: (dedicated MB, shared MB), or None elsewhere.

    Shared is the part that spilled into system RAM over PCIe. 2026-09-24: batch 8 reached 11.4 GB
    dedicated + 15.3 GB shared, ran 4x slower (5.8 s/step), and the machine went down during the retry,
    so the trainer watches this and stops (saving) before a spill instead of grinding through one."""
    if os.name != "nt":
        return None
    import subprocess
    pid = os.getpid()
    ps = ("$s=(Get-Counter -ErrorAction SilentlyContinue "
          f"'\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage','\\GPU Process Memory(pid_{pid}_*)\\Shared Usage')"
          ".CounterSamples; "
          "$d=($s | Where-Object {$_.Path -like '*dedicated usage'} | Measure-Object CookedValue -Sum).Sum; "
          "$h=($s | Where-Object {$_.Path -like '*shared usage'} | Measure-Object CookedValue -Sum).Sum; "
          "Write-Output ([int]($d/1MB)); Write-Output ([int]($h/1MB))")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=30).stdout.split()
        return int(out[0]), int(out[1])
    except Exception:                                # noqa: BLE001 -- a missing reading never stops training
        return None


def save_adapter(model, out_dir, meta):
    from safetensors.numpy import save_file
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            tensors[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().numpy().astype(np.float32)
            tensors[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().numpy().astype(np.float32)
    save_file(tensors, os.path.join(out_dir, "talk_lora.safetensors"))
    with open(os.path.join(out_dir, "talk_lora.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="adapter directory (off-repo)")
    ap.add_argument("--steps", type=int, default=0, help="0 = --epochs over the train split")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4, help="sequences per micro-batch")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=160)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-shared-mb", type=int, default=1024,
                    help="stop (saving) once this much GPU memory has spilled into system RAM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    eos = tk.token_to_id("</s>")
    if eos is None:
        raise SystemExit("the tokenizer has no </s>")
    t0 = time.time()
    model, cfg = load_grilly(args.export)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = lora_targets(model)
    apply_lora_(model, args.rank, args.alpha, targets)
    params = lora_parameters(model)
    for p in params:
        p.requires_grad_(True)
    n_lora = sum(p.numel() for p in params)
    train = encode_records(args.data, tk, eos, "train", args.max_len)
    held = encode_records(args.data, tk, eos, "held", args.max_len)
    rng.shuffle(held)
    held = held[:args.eval_n]
    total = args.steps or int(math.ceil(args.epochs * len(train) / (args.batch * args.accum)))
    log(f"train_talk_grilly | {len(targets)} LoRA modules, {n_lora / 1e6:.2f}M trainable (r={args.rank}, "
        f"alpha={args.alpha}) | train {len(train):,} held-eval {len(held)} | {total} steps of "
        f"{args.batch} x {args.accum} | "
        f"lr {args.lr} warmup {args.warmup} | load {time.time() - t0:.0f}s")
    opt = AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        frac = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    def evaluate():
        tot, n = 0.0, 0.0
        with grilly.no_grad():
            for s in range(0, len(held), args.batch):
                x, y, m, keep = make_batch(held[s:s + args.batch])
                tot += float(masked_loss(model, x, y, m, keep).item()) * float(m.sum())
                n += float(m.sum())
        return tot / max(n, 1.0)

    metrics_path = os.path.join(HERE, "logs", f"train_talk_grilly{args.tag}.jsonl")
    mf = open(metrics_path, "w", encoding="utf-8")
    ev0 = evaluate()
    mem0 = gpu_mem_mb()
    log(f"  step 0 held answer loss {ev0:.4f} ({time.time() - t0:.0f}s)"
        + (f"  gpu {mem0[0]:,} MB + shared {mem0[1]:,} MB" if mem0 else ""))
    mf.write(json.dumps({"step": 0, "held_loss": ev0}) + "\n")
    order = []
    step, tokens, t_start = 0, 0, time.time()
    run_loss, run_n = 0.0, 0
    while step < total:
        if not order:
            order = batches(train, args.batch, rng)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        lv = 0.0
        for _ in range(args.accum):              # micro-batches: activation memory is per micro-batch
            if not order:
                order = batches(train, args.batch, rng)
            rows = [train[i] for i in order.pop()]
            x, y, m, keep = make_batch(rows)
            loss = masked_loss(model, x, y, m, keep)
            (loss * (1.0 / args.accum)).backward()
            lv += float(loss.item()) / args.accum
            tokens += int((x != 0).sum())
            del loss
        gn = float(clip_grad_norm_(params, 1.0).item())
        opt.step()
        if not math.isfinite(lv):
            log(f"  step {step}: loss {lv} -- stopping without saving")
            break
        run_loss += lv
        run_n += 1
        step += 1
        if step % args.log_every == 0 or step == total:
            el = time.time() - t_start
            mem = gpu_mem_mb()
            rec = {"step": step, "loss": run_loss / run_n, "gnorm": gn, "lr": lr_at(step - 1),
                   "s_per_step": el / step, "tok_s": tokens / el, "gpu_mb": mem}
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            log(f"  step {step:5d}/{total}  loss {run_loss / run_n:.4f}  gnorm {gn:.3f}  lr {lr_at(step - 1):.2e}  "
                f"{el / step:.2f} s/step  {tokens / el:,.0f} tok/s  eta {(total - step) * el / step / 3600:.2f} h"
                + (f"  gpu {mem[0]:,} MB + shared {mem[1]:,} MB" if mem else ""))
            run_loss, run_n = 0.0, 0
            if mem and mem[1] > args.max_shared_mb:
                log(f"  STOP: {mem[1]:,} MB spilled into shared memory (limit {args.max_shared_mb:,}); "
                    f"saving at step {step} and stopping rather than running over PCIe")
                save_adapter(model, args.out, {"step": step, "rank": args.rank, "alpha": args.alpha,
                                               "targets": targets, "stopped": "shared_memory",
                                               "export": os.path.basename(args.export.rstrip("\\/")),
                                               "data": os.path.basename(args.data), "tag": args.tag})
                break
        if step % args.eval_every == 0 or step == total:
            ev = evaluate()
            log(f"  step {step} held answer loss {ev:.4f}")
            mf.write(json.dumps({"step": step, "held_loss": ev}) + "\n")
        if step % args.save_every == 0 or step == total:
            save_adapter(model, args.out, {"step": step, "rank": args.rank, "alpha": args.alpha,
                                           "targets": targets, "export": os.path.basename(args.export.rstrip("\\/")),
                                           "data": os.path.basename(args.data), "tag": args.tag})
    mf.close()
    with open(os.path.join(HERE, "logs", f"train_talk_grilly{args.tag}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    log(f"done: {step} steps in {(time.time() - t_start) / 3600:.2f} h -> {args.out}")


if __name__ == "__main__":
    main()
