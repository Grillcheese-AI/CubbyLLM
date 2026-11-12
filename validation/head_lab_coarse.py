"""validation/head_lab_coarse.py -- order 0068: a coarse pass over every row, then the exact rows of its best N.

The shortlists of routes 2 and 3 choose rows without looking at them. A coarse pass looks at all 128,000, cheaply,
and the exact head then scores only the N rows the coarse pass ranks best. If the full head's best token is among
those N, the exact pass returns it: the agreement is the recall of the full head's best token in the coarse top N.
The coarse heads, each built from the trained rows with no training:
  sign      one bit per weight: the row's signs, scaled by its mean absolute weight; the state stays as it is
  sign_c    the same after removing the mean row
  hamming   one bit per weight and one bit per state dimension (a Hamming distance), scaled by the row's norm
  pcaR      the rows and the state projected on the rows' first R principal directions (R = 64, 128)
  exact     the trained rows themselves (the check: recall 1)
Read per coarse head and per N, against the full head on the same states: the recall of its best token, and the
probability the full head puts inside the coarse top N. Costs are given in rows of the head's width by bytes read per
token (a 1-bit pass over 128,000 rows of 1,024 reads what 4,000 float rows do; rank R reads 128,000 * R / 1,024).
Times are not computed here.
With --screen-k, 0050's frozen context shortlist (screen plus successors, K rows) is scored as well: the rows are the
union, and the agreement is the full head's best token being in either. The union's cost counts both in full (the
index's pass, K, the coarse pass and N), with no discount for rows they share.

    python validation/head_lab_coarse.py --export <base export> --dumps <head_shortlist folder> [--schemes sign,pca64]
-> validation/logs/head_lab_coarse_<date><tag>.json

Wired: STANDALONE (an experiment; order 0068's report reads it). numpy only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

__wiring__ = "STANDALONE"

SCHEMES = ("sign", "sign_c", "hamming", "pca64", "pca128", "exact")


def coarse_head(w: np.ndarray, scheme: str) -> dict:
    """{"g": the coarse rows, "state": how the state is read ("raw" | "sign" | a projection), "rows_eq": its cost}."""
    v, d = w.shape
    if scheme == "exact":
        return {"g": w, "state": "raw", "rows_eq": float(v)}
    if scheme in ("sign", "sign_c"):
        x = w - w.mean(0) if scheme == "sign_c" else w
        return {"g": (np.sign(x) * np.abs(x).mean(1, keepdims=True)).astype(np.float32), "state": "raw", "rows_eq": v / 32.0}
    if scheme == "hamming":
        return {"g": (np.sign(w) * (np.linalg.norm(w, axis=1, keepdims=True) / np.sqrt(d))).astype(np.float32),
                "state": "sign", "rows_eq": v / 32.0}
    if scheme.startswith("pca"):
        r = int(scheme[3:])
        x = w - w.mean(0)
        _vals, vecs = np.linalg.eigh((x.T @ x).astype(np.float64))
        p = np.ascontiguousarray(vecs[:, ::-1][:, :r].T, dtype=np.float32)          # (r, d), the largest first
        return {"g": np.ascontiguousarray(x @ p.T), "state": p, "rows_eq": v * r / float(d)}
    raise ValueError(scheme)


def coarse_scores(head: dict, h: np.ndarray) -> np.ndarray:
    s = head["state"]
    x = h if isinstance(s, str) and s == "raw" else np.sign(h) if isinstance(s, str) else h @ s.T
    return x @ head["g"].T


def screen_hits(idx, w: np.ndarray, d: dict, k: int, arm: str = "screen+succ", chunk: int = 256, log=print) -> np.ndarray:
    """Per state: is the full head's best token in the context shortlist of K rows."""
    hid, prev = d["hidden"], d["prev"]
    hit = np.zeros(len(hid), dtype=bool)
    for s in range(0, len(hid), chunk):
        h = hid[s:s + chunk].astype(np.float32)
        best = (h @ w.T).argmax(1)
        for i, ids in enumerate(idx.shortlist(h, k, arm, prev[s:s + chunk])):
            pos = int(np.searchsorted(ids, best[i]))
            hit[s + i] = pos < len(ids) and ids[pos] == best[i]
        if (s // chunk) % 20 == 0:
            log(f"screen {min(s + chunk, len(hid))}/{len(hid)}")
    return hit


def read(w: np.ndarray, d: dict, schemes: list, ns: list, chunk: int = 256, log=print, screen=None, screen_rows: float = 0.0) -> dict:
    """Per coarse head and N: the recall of the full head's best token in the coarse top N, and the full head's
    probability inside it."""
    hid = d["hidden"]
    n = len(hid)
    out = {}
    for sch in schemes:
        head = coarse_head(w, sch)
        rank, mass = np.zeros(n, dtype=np.int64), {k: 0.0 for k in ns}
        for s in range(0, n, chunk):
            h = hid[s:s + chunk].astype(np.float32)
            full = h @ w.T
            best = full.argmax(1)
            co = coarse_scores(head, h)
            rank[s:s + chunk] = (co > co[np.arange(len(h)), best][:, None]).sum(1)   # coarse rows ranked above it
            m = full.max(1, keepdims=True)
            p = np.exp(full - m)
            p /= p.sum(1, keepdims=True)
            for k in ns:
                if k >= full.shape[1]:
                    mass[k] += float(len(h))
                    continue
                cutoff = np.partition(co, -k, axis=1)[:, -k][:, None]
                mass[k] += float((p * (co >= cutoff)).sum())
            if (s // chunk) % 20 == 0:
                log(f"{sch} {min(s + chunk, n)}/{n}")
        out[sch] = {"rows_eq_coarse": round(head["rows_eq"], 1),
                    "n": {str(k): {"recall": round(float((rank < k).mean()), 5), "mass_inside": round(mass[k] / n, 5),
                                   "rows_eq_total": round(head["rows_eq"] + k, 1)} for k in ns},
                    **({} if screen is None else {"union": {str(k): {
                        "agree": round(float((screen | (rank < k)).mean()), 5),
                        "recall_where_the_shortlist_misses": round(float((rank[~screen] < k).mean()), 5) if (~screen).any() else None,
                        "rows_eq_total": round(screen_rows + head["rows_eq"] + k, 1)} for k in ns}}),
                    "rank_p50": int(np.median(rank)), "rank_p99": int(np.quantile(rank, 0.99)),
                    "rank_p999": int(np.quantile(rank, 0.999)), "rank_max": int(rank.max())}
        log(sch, json.dumps(out[sch]))
    return out


def cut(d: dict, skip: int, stride: int, offset: int) -> dict:
    return {k: (v if k == "shards" else v[skip:][offset::stride]) for k, v in d.items()}


def main() -> None:
    from head_shortlist_eval import head_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--schemes", default="sign,sign_c,hamming,pca64,pca128")
    ap.add_argument("--ns", default="1024,4096,8192,16384")
    ap.add_argument("--dump", default="heldin_val", help="heldin_val to explore; heldout is read once (order 0068)")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--screen-k", type=int, default=0, help="also score 0050's context shortlist of K rows (the union)")
    ap.add_argument("--index", default="index_router_succ.npz")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    w = head_rows(a.export)
    d = cut(dict(np.load(os.path.join(a.dumps, f"{a.dump}.npz"))), a.skip, a.stride, a.offset)
    log(f"{a.dump}: {len(d['target'])} positions (skip {a.skip}, every {a.stride}th, offset {a.offset})")
    screen, screen_rows = None, 0.0
    if a.screen_k:
        from cubbyllm.model.vocab.shortlist import ShortlistIndex
        idx = ShortlistIndex.load(os.path.join(a.dumps, a.index))
        screen = screen_hits(idx, w, d, a.screen_k, log=log)
        screen_rows = float(len(idx.screen.centroids) + a.screen_k)
        log(f"the context shortlist alone (K = {a.screen_k}): {screen.mean():.5f} at {screen_rows:.0f} rows")
    rep = {"what": "order 0068: a coarse pass over every row, then the exact rows of its best N",
           "date": _dt.date.today().isoformat(), "dump": a.dump, "skip": a.skip, "stride": a.stride, "offset": a.offset,
           "positions": len(d["target"]),
           "screen_k": a.screen_k, "screen_alone": None if screen is None else round(float(screen.mean()), 5),
           "screen_rows": screen_rows,
           "schemes": read(w, d, a.schemes.split(","), [int(x) for x in a.ns.split(",")], log=log, screen=screen,
                           screen_rows=screen_rows)}
    path = os.path.join(HERE, "logs", f"head_lab_coarse_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    log("->", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
