"""validation/head_dump.py -- order 0050, stage 1: what the shortlist index is built and read against, from the trained
450M on grilly2.

For the 13 pretraining shards' held-out tails (``_holdout.json``; windows evenly spaced, as ``train_base.val_batch``
draws them) and, apart, held-in windows from their training region (the learned router trains on these, never on the
tails), one forward of each window through the base export, keeping per position:
  hidden   the state the head reads (after the adapter), fp16
  target   the next token; prev: this position's own token (arm c's successors read it)
  top      the full head's top --top token ids and their logits
  lse      the full head's log-sum-exp; tlogit: the target's logit (so NLL = lse - tlogit)
The head's own rows are read from the export when the index is built (``head_shortlist_eval.py``).

    uv run --project <grilly2> python validation/head_dump.py --split heldout --windows 4 --len 1024
    uv run --project <grilly2> python validation/head_dump.py --split heldin --windows 32 --len 1024
    uv run --project <grilly2> python validation/head_dump.py --split heldin --windows 4 --phase 0.5 --name heldin_val
-> $CUBBY_DATA_ROOT/head_shortlist/<split>.npz

Wired: STANDALONE (an experiment; feeds ``head_shortlist_eval.py``). Imports grilly (grilly2), never torch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

__wiring__ = "STANDALONE"

SHARDS = ("arxiv", "books", "fineweb_edu", "kbtxt", "nemotron_code", "nemotron_legal", "nemotron_qa", "pretrain_ext",
          "unified", "verified_facts", "wiki_full", "wikibooks", "wikipedia")


def windows(cache: str, name: str, split: str, n: int, length: int, phase: float = 0.0) -> np.ndarray:
    """(n, length + 1) token windows: evenly spaced over the held-out tail, or over the training region (held-in);
    `phase` shifts the starts by that share of their spacing (a held-in tuning split apart from the build's)."""
    arr = np.memmap(os.path.join(cache, f"{name}.u32"), dtype=np.uint32, mode="r")
    hold = int(json.load(open(os.path.join(cache, "_holdout.json"), encoding="utf-8"))[name])
    lo, hi = (len(arr) - hold, len(arr)) if split == "heldout" else (0, len(arr) - hold)
    span = hi - lo - length - 1
    step = span // max(n - 1, 1) if span > 0 else 0
    starts = [min(lo + span, lo + step * i + int(phase * step)) for i in range(n)] if span > 0 else [lo] * n
    return np.stack([np.array(arr[s:s + length + 1], dtype=np.int64) for s in starts])


class Capture:
    """Wraps ``lm_head.forward``: keeps the hidden state the head is given, returns the real head's logits."""

    def __init__(self, head) -> None:
        self.h, orig = None, head.forward

        def forward(x):
            self.h = x
            return orig(x)
        head.forward = forward


def main() -> None:
    import grilly as g

    from cubbyllm.local_paths import expand
    from cubbyllm.ops.grilly_seats import load_export
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="$CUBBY_MODELS/base450m_cont/grilly")
    ap.add_argument("--cache", default="$CUBBY_TRAINING_DATA/token_cache_cont")
    ap.add_argument("--out", default="$CUBBY_DATA_ROOT/head_shortlist")
    ap.add_argument("--split", choices=("heldout", "heldin"), required=True)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--phase", type=float, default=0.0, help="shift the held-in windows (a tuning split: 0.5)")
    ap.add_argument("--name", default="", help="the file's name (default: the split's)")
    a = ap.parse_args()
    model, cfg = load_export(expand(a.export))
    cap = Capture(model.lm_head)
    cache, keep = expand(a.cache), {k: [] for k in ("hidden", "target", "prev", "top", "toplogit", "lse", "tlogit", "shard")}
    for si, name in enumerate(SHARDS):
        for w in windows(cache, name, a.split, a.windows, a.len, a.phase):
            x, y = w[:-1], w[1:]
            with g.no_grad():
                logits = model(g.tensor(x.reshape(1, -1).tolist())).logits.reshape(len(x), -1)
                lp = g.log_softmax(logits, dim=-1)
                tv, ti = g.topk(logits, a.top, dim=-1)
                tlp = g.gather(lp, 1, g.tensor(y.reshape(-1, 1).tolist())).numpy()[:, 0]
                top1lp = g.gather(lp, 1, ti.narrow(1, 0, 1)).numpy()[:, 0]
            tv, ti = tv.numpy(), ti.numpy()
            lse = tv[:, 0] - top1lp
            keep["hidden"].append(cap.h.reshape(len(x), -1).numpy().astype(np.float16))
            keep["target"].append(y.astype(np.int32))
            keep["prev"].append(x.astype(np.int32))
            keep["top"].append(ti.astype(np.int32))
            keep["toplogit"].append(tv.astype(np.float32))
            keep["lse"].append(lse.astype(np.float32))
            keep["tlogit"].append((tlp + lse).astype(np.float32))
            keep["shard"].append(np.full(len(x), si, dtype=np.int8))
        nll = float(np.mean(np.concatenate(keep["lse"][-a.windows:]) - np.concatenate(keep["tlogit"][-a.windows:])))
        print(f"{name}: nll {nll:.3f}", flush=True)
    out = expand(a.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{a.name or a.split}.npz")
    np.savez(path, **{k: np.concatenate(v) for k, v in keep.items()}, shards=np.array(SHARDS))
    print("->", path, sum(len(t) for t in keep["target"]), "positions")


if __name__ == "__main__":
    main()
