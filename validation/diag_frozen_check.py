"""diag_frozen_check -- which tensors a run moved: its checkpoint against the one it started from
(order 0005: a ``CB_TRAIN_ONLY=copy`` run must move the copy path's tensors and nothing else).

The run's checkpoint holds its whole model. A start without the copy path (``CB_INIT_FROM`` added it)
has the copy's tensors missing at the end of its list; a start with it (a resumed slot) has them last
too. Every other tensor, and the router's, must be bit-identical; the copy's are printed against the
start's (or, when the start has none, against their init: lam 0.1, tau and beta from the run's meta).
Exits 1 if anything outside the copy moved.

    CB_RUN=<run ckpt> CB_START=<start ckpt> [CB_OUT=<json>] python validation/diag_frozen_check.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch


def main() -> None:
    run = torch.load(os.environ["CB_RUN"], map_location="cpu", weights_only=False, mmap=True)
    start = torch.load(os.environ["CB_START"], map_location="cpu", weights_only=False, mmap=True)
    sp = run["meta"]["spatial"]
    names = ["tau"] + (["beta"] if sp.get("copy_beta") is not None else []) + ["gate.weight", "gate.bias"]
    pr, ps, k = run["params"], start["params"], len(names)        # SpatialCopy's parameters() order; they come last
    added = len(pr) - len(ps)
    if added not in (0, k):
        raise SystemExit(f"the run has {len(pr)} tensors, the start {len(ps)}: not the same model, with or without the copy")
    rest = len(pr) - k
    moved = [i for i in range(rest) if not torch.equal(pr[i], ps[i])]
    moved_router = [i for i, (a, b) in enumerate(zip(run["router"], start["router"])) if not torch.equal(a, b)]
    init = {"tau": sp.get("copy_tau", 3.0), "beta": sp.get("copy_beta"), "gate.bias": math.log(0.1 / 0.9), "gate.weight": 0.0}
    copy = {}
    for j, n in enumerate(names):
        end = pr[rest + j].float()
        was = ps[rest + j].float() if not added else torch.full_like(end, init[n])
        copy[n] = {"start": float(was.flatten()[0]) if was.numel() == 1 else float(was.norm()),
                   "end": float(end.flatten()[0]) if end.numel() == 1 else float(end.norm()),
                   "max_change": float((end - was).abs().max()), "of": "norm" if end.numel() > 1 else "value"}
    report = {"run_step": run["step"], "start_step": start["step"], "copy_added_at_start": bool(added),
              "tensors_checked": rest, "router_checked": len(run["router"]), "moved": moved, "router_moved": moved_router,
              "copy": copy}
    print(f"run step {run['step']} vs start step {start['step']}: {rest} tensors and {len(run['router'])} router tensors"
          f" checked -- {len(moved)} moved, {len(moved_router)} router moved"
          f"{' (the copy was added at the start: compared to its init)' if added else ''}")
    for n, c in copy.items():
        print(f"  copy {n:12s} {c['of']} {c['start']:+.4f} -> {c['end']:+.4f}   max change {c['max_change']:.4f}")
    if os.environ.get("CB_OUT"):
        json.dump(report, open(os.environ["CB_OUT"], "w", encoding="utf-8"), indent=1)
    sys.exit(1 if moved or moved_router else 0)


if __name__ == "__main__":
    main()
