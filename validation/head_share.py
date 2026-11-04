"""validation/head_share.py -- order 0050, stage 1, step 1: the 128k output head's share of a decode step, on grilly2.

Measured, not computed (the order: "measure today first"), on the card this runs on:
  step     one decode step of the base export (``validation/export_base.py``), the greedy token read back to the host
           (what a seat waits for), averaged over --steps after a prompt of --prompt tokens
  head     the head alone on a hidden state of the step's shape: ``lm_head(x)``, then the argmax read back
  head32k  a full 32k head (1,024 x 32,000) the same way: the gate's speed line ("no slower than a full 32k head")
  K rows   a shortlist head of K rows already gathered (K in --ks): what scoring a shortlist costs, the index's own
           cost left out (measured by the index's own report)
At batch 1 and 16 (--batches), with the head in fp32, then in int8 (grilly2's ``quantize_``, weight-only, the head
only, so the step's other weights stay as they were). Memory: the head's bytes in each.

    uv run --project <grilly2> python validation/head_share.py [--export $CUBBY_MODELS/base450m_cont/grilly]
-> validation/logs/head_share_grilly_<date>.json

Wired: STANDALONE (an experiment; 0050's report reads it). Imports grilly (grilly2), never torch.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

__wiring__ = "STANDALONE"


def _ms(fn, reps: int, warm: int = 3) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"p50_ms": round(statistics.median(ts), 3), "mean_ms": round(statistics.fmean(ts), 3),
            "p90_ms": round(ts[int(0.9 * (len(ts) - 1))], 3), "n": reps}


def _greedy(g, logits):
    return g.argmax(logits.reshape(logits.shape[0], -1), dim=-1).numpy()


def decode_step(g, model, batch: int, prompt: int, steps: int, seed: int) -> dict:
    """The mean decode step after a random prompt (timing does not depend on the text)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ids = g.tensor(rng.integers(100, 30000, size=(batch, prompt)).tolist())
    with g.no_grad():
        cache = model.make_cache(batch, prompt + steps + 8)
        out = model(ids, past_key_values=cache, logits_to_keep=1)
        nxt = _greedy(g, out.logits)
        ts = []
        for _ in range(steps):
            t0 = time.perf_counter()
            out = model(g.tensor(nxt.reshape(batch, 1).tolist()), past_key_values=cache, logits_to_keep=1)
            nxt = _greedy(g, out.logits)
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = ts[2:] or ts
    return {"p50_ms": round(statistics.median(ts), 3), "mean_ms": round(statistics.fmean(ts), 3), "n": len(ts)}


def head_time(g, weight, batch: int, hidden: int, reps: int) -> dict:
    x = g.randn(batch, 1, hidden)
    F = g.nn.functional
    with g.no_grad():
        return _ms(lambda: _greedy(g, F.linear(x, weight)), reps)


def nbytes(w) -> int:
    for attr in ("nbytes",):
        try:
            return int(getattr(w, attr))
        except Exception:                                   # noqa: BLE001 -- a quantized tensor reports differently
            pass
    try:
        return int(w.numel() * w.element_size())
    except Exception:                                       # noqa: BLE001
        return -1


def main() -> None:
    import grilly as g
    from grilly.infer.quantization import Int8WeightOnlyConfig, quantize_

    from cubbyllm.local_paths import expand
    from cubbyllm.ops.grilly_seats import load_export
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="$CUBBY_MODELS/base450m_cont/grilly")
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--ks", default="256,1024,4096")
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--no-int8", action="store_true")
    a = ap.parse_args()
    model, cfg = load_export(expand(a.export))
    hid, vocab = cfg["hidden_size"], cfg["vocab_size"]
    batches, ks = [int(x) for x in a.batches.split(",")], [int(x) for x in a.ks.split(",")]
    g.manual_seed(0)
    head32k = g.randn(32000, hid) * 0.02
    rows = {k: g.randn(k, hid) * 0.02 for k in ks}
    res = {"what": "order 0050 stage 1: the output head's share of a decode step (grilly2)",
           "date": _dt.date.today().isoformat(), "export": os.path.basename(os.path.dirname(expand(a.export))),
           "vocab": vocab, "hidden": hid, "machine": platform.platform(), "prompt": a.prompt, "runs": []}
    for prec in (["fp32"] if a.no_int8 else ["fp32", "int8"]):
        if prec == "int8":
            quantize_(model, Int8WeightOnlyConfig(), filter_fn=lambda m, n: n == "lm_head")
        for b in batches:
            step = decode_step(g, model, b, a.prompt, a.steps, seed=b)
            head = head_time(g, model.lm_head.weight, b, hid, a.reps)
            h32 = head_time(g, head32k, b, hid, a.reps)
            short = {str(k): head_time(g, w, b, hid, a.reps) for k, w in rows.items()}
            run = {"precision": prec, "batch": b, "step": step, "head": head, "head32k_fp32": h32,
                   "shortlist_rows_fp32": short, "head_bytes": nbytes(model.lm_head.weight),
                   "head_share_of_step": round(head["p50_ms"] / max(1e-9, step["p50_ms"]), 4)}
            res["runs"].append(run)
            print(json.dumps({k: run[k] for k in ("precision", "batch", "head_share_of_step")}
                             | {"step_ms": step["p50_ms"], "head_ms": head["p50_ms"], "head32k_ms": h32["p50_ms"]}),
                  flush=True)
    path = os.path.join(HERE, "logs", f"head_share_grilly_{res['date']}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
