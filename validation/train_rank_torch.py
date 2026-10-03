"""H-E19, listwise fine-tune of the emitter adapter: train the PICK, not only the writing.

The ranking read (exp_he19_rank.py) found the right next step on the adapter's short list but not at its top;
a frozen-feature pointer head did worse (exp_he19_pointer.py). So the pick is trained where the knowledge is:
the adapter itself, with a listwise loss over the host's menu.

Per step row: the right emission (the gold step, or the gold stop) and K wrong ones from the same menu --
hard first (the same two values under another op, the stop when a step is due and a step when the stop is
due), then random -- each scored by the sum log-prob of its tokens and </s> after the shared prompt. Loss:

    softmax cross-entropy over [right, wrong_1 .. wrong_K]  +  sft_weight * the right one's per-token NLL

The second term keeps the adapter a writer (the loop still writes the step out); the first moves probability
from the wrong menu entries to the right one. Menu entries that compute the gold value are never negatives.

    python validation/train_rank_torch.py --ckpt <base .pt> --tokenizer <bbpe128k> --adapter <write-back adapter>
        --pool emitter_sft_v12e_w_tg30_step.jsonl --out <dir>
Read afterwards, locally and free: exp_he19_rank.py (the ranking) and exp_he19_step_loop.py (the 236 loop).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if "--smoke" in sys.argv:
    os.environ.update(CB_D="64", CB_L="4", CB_HEADS="4", CB_WINDOW="16", CB_GEN_BASIS="4", CB_GEN_RANK="2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.reasoning.step_loop import render_step, render_stop  # noqa: E402

ADAPTER = "emitter_lora"
OPS = ("add", "sub", "mul", "div")
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# ── the list: the right emission and K wrong ones from the same menu ─────────

def build_list(row: dict, K: int, rng: random.Random):
    """(right_text, [wrong_text * <=K]) or None when the menu has no right entry. Hard negatives first: the
    gold's two values under another op (and the same op with one value swapped), the stop when a step is
    due, the other stops when the stop is due; then random wrong entries. One spelling per wrong value."""
    from exp_he19_pointer import structure
    from cubbyllm.reasoning.step_loop import parse_emission
    ids, menu, facts = structure(row)
    right_groups = {c[4] for c in menu if c[3]}
    if not right_groups:
        return None
    k = facts["k"]

    def text(c):
        if c[0] == 4:
            return render_stop(ids[c[1]])
        return render_step(k, [("assign", ids[c[1]]), (OPS[c[0]], ids[c[2]])])

    seen, wrong = set(), []
    for c in menu:
        if c[4] in right_groups or c[4] in seen:
            continue
        seen.add(c[4])
        wrong.append(c)
    gold_ops, gold_op = set(), None
    kind, what = parse_emission(row["program"])
    if kind == "step" and what is not None:
        gold_ops = {v for _, v in what.ops}
        gold_op = next((o for o, _ in what.ops if o != "assign"), None)
    hard = []
    for c in wrong:
        if facts["is_stop"]:
            hard_c = c[0] == 4
        else:
            pair = {ids[c[1]], ids[c[2]]}
            hard_c = c[0] == 4 or pair == gold_ops or (gold_op in OPS and OPS[c[0]] == gold_op and pair & gold_ops)
        if hard_c:
            hard.append(c)
    rng.shuffle(hard)
    pick = hard[:max(1, K // 2)]
    rest = [c for c in wrong if c not in pick]
    rng.shuffle(rest)
    pick += rest[:K - len(pick)]
    return row["program"], [text(c) for c in pick]


def load_pool(paths: list[str], split: str, n: int, seed: int) -> list[dict]:
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("task") == "arithmetic" and r.get("split") == split and "#" in r.get("id", ""):
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


# ── scoring: sum log-prob of each target after its prompt, with the gradient ──

def seq_logprobs(model, tk, eos, pairs, dev, amp):
    """(sum, count) of log p(target tokens, </s> | prompt) per pair -- training's encoding: prompt and
    target separately, target = text.rstrip() + newline."""
    from talk_data import make_batch
    rows = [{"p": tk.encode(p).ids, "t": tk.encode(t.rstrip() + "\n").ids + [eos]} for p, t in pairs]
    x, y, m, _ = make_batch(rows)
    x, y, sel = (torch.from_numpy(a).to(dev) for a in (x, y, m > 0))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        h = model.features(x)
        logits = model.logits_from(h[sel].unsqueeze(0))[0]
    lp = -F.cross_entropy(logits.float(), y[sel], reduction="none")
    row = sel.nonzero()[:, 0]
    tot = torch.zeros(len(rows), device=dev).index_add_(0, row, lp)
    cnt = torch.zeros(len(rows), device=dev).index_add_(0, row, torch.ones_like(lp))
    return tot, cnt


def list_loss(model, tk, eos, lists, dev, amp, sft_weight: float):
    """lists: [(prompt, right, [wrong..])], every list the same length. Returns (loss, listwise top-1)."""
    width = 1 + max(len(w) for _, _, w in lists)            # a small menu gives fewer wrong entries: pad, masked
    pairs, where = [], []
    for i, (p, right, wrong) in enumerate(lists):
        for j, t in enumerate([right] + wrong):
            pairs.append((p, t))
            where.append(i * width + j)
    tot, cnt = seq_logprobs(model, tk, eos, pairs, dev, amp)
    idx = torch.tensor(where, device=dev)
    scores = torch.full((len(lists) * width,), float("-inf"), device=dev).index_put((idx,), tot).view(len(lists), width)
    ce = F.cross_entropy(scores, torch.zeros(len(lists), dtype=torch.long, device=dev))
    first = torch.tensor([i * width for i in range(len(lists))], device=dev)
    pos = torch.tensor([where.index(f) for f in first.tolist()], device=dev)
    sft = -(tot[pos] / cnt[pos]).mean()
    acc = float((scores.argmax(-1) == 0).float().mean())
    return ce + sft_weight * sft, acc, float(ce.detach()), float(sft.detach())


def read_lists(model, tk, eos, lists, dev, amp, per: int):
    """Listwise top-1 on fixed held-out lists, no gradient."""
    acc, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(lists), per):
            part = lists[s:s + per]
            _, a, _, _ = list_loss(model, tk, eos, part, dev, amp, 0.0)
            acc += a * len(part)
            n += len(part)
    return acc / max(1, n)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True, help="the adapter to start from (the write-back adapter)")
    ap.add_argument("--pool", nargs="+", required=True, help="*_step.jsonl, train split")
    ap.add_argument("--held", default="", help="held-out *_step.jsonl for a sampled-list read (val split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--k", type=int, default=11, help="wrong entries per list")
    ap.add_argument("--per-step", type=int, default=4, help="lists per optimizer step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--sft-weight", type=float, default=0.5)
    ap.add_argument("--held-n", type=int, default=200)
    ap.add_argument("--every", type=int, default=250, help="log, read the held lists and save every N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.smoke else "cpu")
    amp = dev.type == "cuda"

    from tokenizers import Tokenizer
    from train_talk_torch import load_adapter, save_adapter
    from train_emitter_torch import _find
    tk = Tokenizer.from_file(a.tokenizer)
    eos = tk.token_to_id("</s>")
    if a.smoke:
        from train_base import arch_meta, build_model
        from train_talk_torch import apply_lora
        from talk_data import lora_targets
        meta = arch_meta(tk.get_vocab_size(), ((tk.get_vocab_size() + 127) // 128) * 128)
        model = build_model(meta, dev)
        a.rows, a.k, a.held_n, a.every = 6, 3, 4, 2
    else:
        from train_base import load_base
        model = load_base(a.ckpt, str(dev))
    for p in model.parameters():
        p.requires_grad_(False)
    start = a.out if os.path.exists(os.path.join(a.out, f"{ADAPTER}.safetensors")) else a.adapter
    if a.smoke and not os.path.exists(os.path.join(start, f"{ADAPTER}.safetensors")):
        targets, rank, alpha = lora_targets(meta["L"], meta["attn_every"]), 4, 8.0
        wrapped = apply_lora(model, targets, rank, alpha)
    else:
        ameta = load_adapter(model, start, ADAPTER)
        targets, rank, alpha = ameta["targets"], ameta["rank"], ameta["alpha"]
        wrapped = {t: _find(model, t) for t in targets}
    done = int(ameta.get("rank_step", 0)) if start == a.out and not a.smoke else 0   # a restarted run resumes
    params = [p for w in wrapped.values() for p in (w.lora_A.weight, w.lora_B.weight)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    pool = load_pool(a.pool, "train", 0, a.seed)
    lists = []
    for r in pool:
        if len(lists) >= a.rows:
            break
        built = build_list(r, a.k, rng)
        if built and built[1]:
            lists.append((r["prompt"], built[0], built[1]))
    held = []
    if a.held:
        hrng = random.Random(1)
        for r in load_pool([a.held], "val", 0, 1):
            built = build_list(r, a.k, hrng)
            if built and built[1]:
                held.append((r["prompt"], built[0], built[1]))
            if len(held) >= a.held_n:
                break
    steps = math.ceil(len(lists) / a.per_step)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    log(f"rank | {dev} | {len(lists)} lists of 1 right + <= {a.k} wrong | {steps} steps of {a.per_step} | "
        f"lr {a.lr} sft_weight {a.sft_weight} | from {os.path.basename(os.path.normpath(start))}")
    if held:
        log(f"  step 0: held lists top-1 {read_lists(model, tk, eos, held, dev, amp, a.per_step):.3f} ({len(held)} lists)")
    t0, run = time.time(), {"loss": 0.0, "acc": 0.0, "ce": 0.0, "sft": 0.0, "n": 0}
    order = list(range(len(lists)))
    rng.shuffle(order)
    for step in range(1, steps + 1):
        if step <= done:
            sched.step()
            continue
        part = [lists[i] for i in order[(step - 1) * a.per_step: step * a.per_step]]
        loss, acc, ce, sft = list_loss(model, tk, eos, part, dev, amp, a.sft_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        for k_, v in (("loss", float(loss)), ("acc", acc), ("ce", ce), ("sft", sft)):
            run[k_] += v
        run["n"] += 1
        if step % a.every == 0 or step == steps:
            n = run["n"]
            msg = (f"  step {step}/{steps}: loss {run['loss'] / n:.3f} (list {run['ce'] / n:.3f}, write {run['sft'] / n:.3f}) "
                   f"| train lists top-1 {run['acc'] / n:.3f}")
            if held:
                msg += f" | held lists top-1 {read_lists(model, tk, eos, held, dev, amp, a.per_step):.3f}"
            log(msg + f" | {time.time() - t0:.0f}s")
            run = {"loss": 0.0, "acc": 0.0, "ce": 0.0, "sft": 0.0, "n": 0}
            save_adapter(wrapped, a.out, {"targets": targets, "rank": rank, "alpha": alpha, "adapter": ADAPTER,
                                          "rank_step": step, "from": os.path.basename(os.path.normpath(a.adapter))},
                         ADAPTER)
    with open(os.path.join(a.out, "rank.log"), "a", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
