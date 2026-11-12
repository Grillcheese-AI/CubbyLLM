"""validation/head_lab_wired.py -- order 0068: the wired "+ham" arm against the lab's reading of route 5.

Route 5 passed in the lab (``head_lab_coarse.py``: 0050's context list of K rows and the best N rows of a Hamming
pass, read as two hit tests). This reads the same positions through the model code as wired
(``cubbyllm/model/vocab/hamming.py`` and the "+ham" arm of ``shortlist.py``):
  lists   every position: is the full head's best token in the wired arm's list, against the lab's two hit tests
          position by position; and how many rows the list holds (the rows scored exactly)
  head    every `--head-every`th position through the torch head with ``CB_HEAD_SHORTLIST=1``: its best token against
          the full head's, its kept rows against the list, its kept logits against the full head's on those rows
  seconds per position on this machine's CPU for each part of the numpy reference. Not a card's time.
  distance alone   an exploration: the pass ranked by Hamming distance with no norm (what grilly2's packed top-k
          computes today), with the ties at the Nth place counted for and against
Held-in positions only. The held-out ones were read once per route (order 0068) and this refuses to read them.

    python validation/head_lab_wired.py --export <base export> --dumps <head_shortlist folder>
-> validation/logs/head_lab_wired_<date><tag>.json

Wired: STANDALONE (a check; order 0068's report reads it).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

__wiring__ = "STANDALONE"


def _lab():
    spec = importlib.util.spec_from_file_location("head_lab_coarse", os.path.join(HERE, "head_lab_coarse.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has(ids: np.ndarray, t: int) -> bool:
    pos = int(np.searchsorted(ids, t))
    return pos < len(ids) and ids[pos] == t


def lists(idx, w: np.ndarray, d: dict, k: int, n: int, arm: str, chunk: int = 256, log=print) -> dict:
    """Every position: the wired list's hit against the lab's, the rows it holds, the seconds of each part."""
    from cubbyllm.model.vocab.shortlist import HAM
    lab = _lab()
    coarse = lab.coarse_head(w, "hamming")
    scale = np.maximum(np.linalg.norm(w, axis=1) / np.sqrt(w.shape[1]), 1e-30).astype(np.float32)
    hid, prev, base = d["hidden"], d["prev"], arm[:-len(HAM)]
    m = len(hid)
    wired, ctx_hit, lab_hit = np.zeros(m, dtype=bool), np.zeros(m, dtype=bool), np.zeros(m, dtype=bool)
    size, u_for, u_against = np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)
    secs = {"full_head": 0.0, "context_list": 0.0, "hamming_pass": 0.0}
    for s in range(0, m, chunk):
        h, p = hid[s:s + chunk].astype(np.float32), prev[s:s + chunk]
        rows = np.arange(len(h))
        t0 = time.perf_counter()
        best = (h @ w.T).argmax(1)
        secs["full_head"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        ctx = idx.shortlist(h, k, base, p)
        secs["context_list"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        idx.ham.top(h, n)
        secs["hamming_pass"] += time.perf_counter() - t0
        both = idx.shortlist(h, k, arm, p)                               # the wired arm itself
        co = lab.coarse_scores(coarse, h)
        rank = (co > co[rows, best][:, None]).sum(1)                     # the lab's reading: rows ranked above it
        co = np.rint(co / scale)                                         # d - 2 * distance, no norm
        mine = co[rows, best][:, None]
        u_for[s:s + chunk], u_against[s:s + chunk] = (co > mine).sum(1), (co >= mine).sum(1) - 1
        for i in rows:
            wired[s + i], ctx_hit[s + i], size[s + i] = _has(both[i], best[i]), _has(ctx[i], best[i]), len(both[i])
        lab_hit[s:s + chunk] = ctx_hit[s:s + chunk] | (rank < n)
        if (s // chunk) % 10 == 0:
            log(f"lists {min(s + chunk, m)}/{m}: wired {wired[:s + len(h)].mean():.5f} lab {lab_hit[:s + len(h)].mean():.5f}")
    return {"positions": m,
            "wired_agree": round(float(wired.mean()), 5), "wired_misses": int((~wired).sum()),
            "lab_agree": round(float(lab_hit.mean()), 5), "lab_misses": int((~lab_hit).sum()),
            "positions_where_they_differ": int((wired != lab_hit).sum()),
            "context_list_alone": round(float(ctx_hit.mean()), 5),
            "rows_scored_exactly": {"mean": round(float(size.mean()), 1), "p50": int(np.median(size)),
                                    "p99": int(np.quantile(size, 0.99)), "max": int(size.max())},
            "rows_by_bytes": {"index_pass": int(len(idx.screen.centroids)), "hamming_pass": round(len(w) / 32.0, 1),
                              "exact_mean": round(float(size.mean()), 1),
                              "total_mean": round(len(idx.screen.centroids) + len(w) / 32.0 + float(size.mean()), 1)},
            "zeros": {"head_weights": int((w == 0).sum()), "state_values": int((hid == 0).sum())},
            "seconds_per_position_cpu": {a: round(b / m, 6) for a, b in secs.items()},
            "distance_alone": {str(x): {"ties_for": round(float((ctx_hit | (u_for < x)).mean()), 5),
                                        "ties_against": round(float((ctx_hit | (u_against < x)).mean()), 5)}
                               for x in (n, 2 * n, 4 * n)}}


def head_check(idx, w: np.ndarray, d: dict, k: int, n: int, arm: str, every: int, chunk: int = 64, log=print) -> dict:
    """Every `every`th position through ``TopKRetrievalHead`` with the flag on, against the same head with it off."""
    import torch
    from cubbyllm.model.vocab import TopKRetrievalHead
    from cubbyllm.model.vocab.hamming import pack
    head = TopKRetrievalHead(torch.from_numpy(w), learnable=True)
    head.attach_shortlist(idx, k=k, arm=arm, ham_n=n)
    sel = np.arange(0, len(d["hidden"]), every)
    agree = hit = same_rows = 0
    worst, secs, was = 0.0, {"full": 0.0, "shortlisted": 0.0}, os.environ.get("CB_HEAD_SHORTLIST")
    try:
        with torch.no_grad():
            for s in range(0, len(sel), chunk):
                ii = sel[s:s + chunk]
                q = torch.from_numpy(d["hidden"][ii].astype(np.float32))
                p = torch.from_numpy(d["prev"][ii].astype(np.int64))
                os.environ.pop("CB_HEAD_SHORTLIST", None)
                t0 = time.perf_counter()
                full = head.logits(q, head.vocab, p)
                secs["full"] += time.perf_counter() - t0
                os.environ["CB_HEAD_SHORTLIST"] = "1"
                t0 = time.perf_counter()
                out = head.logits(q, head.vocab, p)
                secs["shortlisted"] += time.perf_counter() - t0
                kept, best = torch.isfinite(out), full.argmax(1)
                agree += int((out.argmax(1) == best).sum())
                worst = max(worst, float((out[kept] - full[kept]).abs().max()))
                for i, ids in enumerate(idx.shortlist(q.numpy(), k, arm, p.numpy())):
                    same_rows += int(np.array_equal(kept[i].nonzero().reshape(-1).numpy(), ids))
                    hit += int(_has(ids, int(best[i])))
                log(f"head {min(s + chunk, len(sel))}/{len(sel)}: agree {agree}")
    finally:
        os.environ.pop("CB_HEAD_SHORTLIST", None)
        if was is not None:
            os.environ["CB_HEAD_SHORTLIST"] = was
    return {"positions": int(len(sel)), "every": every,
            "bits_are_the_heads_rows": bool(np.array_equal(idx.ham.bits, pack(w))),
            "best_token_is_the_full_heads": agree, "best_token_in_the_list": hit,
            "kept_rows_are_the_list": same_rows, "max_abs_logit_difference_on_kept_rows": round(worst, 6),
            "seconds_per_position_cpu_torch": {a: round(b / len(sel), 6) for a, b in secs.items()}}


def main() -> None:
    from cubbyllm.model.vocab.shortlist import HAM, ShortlistIndex
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--index", default="index_router_succ.npz")
    ap.add_argument("--dump", default="heldin_val")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--arm", default="screen+succ" + HAM)
    ap.add_argument("--head-every", type=int, default=13)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if not a.dump.startswith("heldin"):
        raise SystemExit("held-in dumps only: the held-out positions were read once per route (order 0068)")
    if not a.arm.endswith(HAM):
        raise SystemExit(f"an arm ending in {HAM!r}")
    from head_shortlist_eval import head_rows
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    w = head_rows(a.export)
    d = _lab().cut(dict(np.load(os.path.join(a.dumps, f"{a.dump}.npz"))), a.skip, a.stride, a.offset)
    idx = ShortlistIndex.load(os.path.join(a.dumps, a.index)).attach_hamming(w, a.n)
    log(f"{a.dump}: {len(d['target'])} positions (skip {a.skip}, every {a.stride}th, offset {a.offset}); {a.arm}, "
        f"K = {a.k}, N = {a.n}")
    rep = {"what": "order 0068: the wired '+ham' arm against the lab's reading of route 5, on held-in positions",
           "date": _dt.date.today().isoformat(), "dump": a.dump, "skip": a.skip, "stride": a.stride, "offset": a.offset,
           "arm": a.arm, "k": a.k, "n": a.n, "threads": os.environ.get("OMP_NUM_THREADS"),
           "lists": lists(idx, w, d, a.k, a.n, a.arm, log=log),
           "head": head_check(idx, w, d, a.k, a.n, a.arm, a.head_every, log=log)}
    path = os.path.join(HERE, "logs", f"head_lab_wired_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    log(json.dumps(rep, indent=1))
    log("->", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
