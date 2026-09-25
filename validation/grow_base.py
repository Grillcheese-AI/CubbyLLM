"""Grow a trained base and check that the grown model computes what the base does.

Step 0 of the growth plan (docs/research/2026-09-25-grow-450m.md). Loads a
``train_base`` checkpoint, builds the grown shape, fills it with
``cubbyllm.model.grow.grow_into``, and scores base and grown on the same
held-out windows (the tails ``train_base.Corpus`` validates on), in float32 on
the CPU. The grown model passes when its loss equals the base's to within
float rounding and its logits agree; the plain copy (``--exit copy``, G_stack's
operator) is not meant to pass, and its loss jump is recorded.

    python validation/grow_base.py --ckpt <base450m_final.pt> --cache <token_cache_base> \
        --depth 2 --ffn 2 [--save <grown.pt>] [--tag d2f2]

Writes validation/logs/grow_step0_<tag>.{json,log}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.model.grow import grow_into, grown_meta  # noqa: E402
from train_base import build_model, load_base  # noqa: E402

SOURCES = "arxiv,books,fineweb_edu,nemotron_code,unified,wikipedia"


def val_windows(cache: str, name: str, n: int, seq: int):
    """``n`` evenly spaced windows from ``name``'s held-out tail, read with a
    seek per window (the shards are GB; only the tail is touched)."""
    holds = json.load(open(os.path.join(cache, "_holdout.json"), encoding="utf-8"))
    path = os.path.join(cache, f"{name}.u32")
    total = os.path.getsize(path) // 4
    s0 = total - int(holds[name])
    span = total - s0 - seq - 1
    starts = [s0 + (span * i) // max(n - 1, 1) for i in range(n)] if span > 0 else [s0] * n
    out = []
    with open(path, "rb") as f:
        for s in starts:
            f.seek(4 * s)
            out.append(np.frombuffer(f.read(4 * (seq + 1)), dtype=np.uint32).astype(np.int64))
    t = torch.from_numpy(np.stack(out))
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def score(model, x, y):
    logits = model.forward(x).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    return float(loss), logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="the base run's token cache (with _holdout.json)")
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--ffn", type=int, default=1)
    ap.add_argument("--exit", default="zero", choices=("zero", "copy"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--save", default="", help="write the grown checkpoint here")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    tag = args.tag or f"w{args.width}d{args.depth}f{args.ffn}{'' if args.exit == 'zero' else '_copy'}"
    logp = os.path.join(HERE, "logs", f"grow_step0_{tag}")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    open(logp + ".log", "w", encoding="utf-8").close()
    lines = []

    def say(s=""):
        """Printed, and appended to the log at once, so a run that dies keeps
        what it measured."""
        print(s, flush=True)
        lines.append(s)
        with open(logp + ".log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta, step = ck["meta"], ck.get("step")
    del ck
    src = load_base(args.ckpt, "cpu")
    gmeta = grown_meta(meta, args.width, args.depth, args.ffn)
    dst = build_model(gmeta, torch.device("cpu"), seed=args.seed)
    rep = grow_into(src, dst, width=args.width, depth=args.depth, ffn=args.ffn,
                    exit=args.exit, seed=args.seed)
    say(f"grow_base step 0 | {os.path.basename(args.ckpt)} (step {step}) -> width x{args.width}, "
        f"depth x{args.depth}, ffn x{args.ffn}, exit {args.exit}")
    say(f"  base  D{meta['D']} L{meta['L']} heads {meta['heads']} ffn_mult {meta.get('ffn_mult', 2)}: "
        f"{rep['params_src'] / 1e6:,.1f}M parameters")
    say(f"  grown D{gmeta['D']} L{gmeta['L']} heads {gmeta['heads']} ffn_mult {gmeta.get('ffn_mult', 2)}: "
        f"{rep['params_dst'] / 1e6:,.1f}M parameters  (built and filled in {time.time() - t0:.0f}s)")
    new = sum(1 for _, _, c in rep["layers"] if c)
    if new:
        say(f"  layers: {len(rep['layers'])}, of which {new} copies ({args.exit} exit); "
            f"attention at {sum(1 for j in range(gmeta['L']) if j % gmeta['attn_every'] == 0)}")

    rows, worst = [], 0.0
    for name in [s for s in args.sources.split(",") if s]:
        t1 = time.time()
        x, y = val_windows(args.cache, name, args.windows, args.seq)
        ls, a = score(src, x, y)
        lg, b = score(dst, x, y)
        d = float((a - b).abs().max())
        worst = max(worst, d)
        rows.append({"source": name, "base": ls, "grown": lg, "max_logit_diff": d})
        say(f"  {name:14s} base {ls:.5f}  grown {lg:.5f}  diff {lg - ls:+.2e}  "
            f"max |logit diff| {d:.2e}  ({time.time() - t1:.0f}s)")
    mb = sum(r["base"] for r in rows) / len(rows)
    mg = sum(r["grown"] for r in rows) / len(rows)
    exact = all(abs(r["grown"] - r["base"]) <= 1e-4 for r in rows) and worst <= 1e-2
    say(f"  mean over {len(rows)} sources x {args.windows} windows x {args.seq} tokens: "
        f"base {mb:.5f}  grown {mg:.5f}  diff {mg - mb:+.2e}  max |logit diff| {worst:.2e}")
    if args.exit == "zero":
        say(f"verdict: {'EXACT' if exact else 'NOT EXACT'} (bar: |loss diff| <= 1e-4 per source, "
            f"max |logit diff| <= 1e-2, float32)")
    else:
        say(f"verdict: plain copy, not exact by design; loss jump {mg - mb:+.4f} nats")

    if args.save:
        payload = {"step": 0, "meta": gmeta, "sched": None, "opt": None,
                   "params": [p.detach() for p in dst.parameters()],
                   "router": [p.detach() for p in dst.context_source.parameters()],
                   "grown_from": {"checkpoint": os.path.basename(args.ckpt), "step": step,
                                  "meta": meta, "width": args.width, "depth": args.depth,
                                  "ffn": args.ffn, "exit": args.exit, "seed": args.seed}}
        torch.save(payload, args.save)
        say(f"saved {args.save} ({os.path.getsize(args.save) / 1e9:.2f} GB)")
        import gc
        del payload, dst, src, a, b
        gc.collect()                  # the reload needs the room the two models held
        back = load_base(args.save, "cpu")
        lb, _ = score(back, *val_windows(args.cache, rows[0]["source"], args.windows, args.seq))
        say(f"  reloaded through train_base.load_base: {rows[0]['source']} loss {lb:.5f} "
            f"(before saving {rows[0]['grown']:.5f})")

    with open(logp + ".json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "meta": meta, "grown_meta": gmeta, "rows": rows,
                   "mean_base": mb, "mean_grown": mg, "max_logit_diff": worst,
                   "exact": exact, "layers": rep["layers"],
                   "params_src": rep["params_src"], "params_dst": rep["params_dst"]}, f, indent=1)
    say(f"wrote {logp}.json / .log  ({time.time() - t0:.0f}s total)")
    if not math.isfinite(mg):
        raise SystemExit("grown loss is not finite")


if __name__ == "__main__":
    main()
