"""validation/head_lab_distance.py -- order 0068: route 5 with the Hamming pass ranked by distance alone, read once on
fresh held-out states.

Route 5 passed on held-out with the pass weighting each row by its norm (``hamming.py``). grilly2's packed top-k
(``cubbyllm/ops/packed.py``) ranks by Hamming distance alone, and on held-in the norm bought nothing. This reads the
union of 0050's frozen context list (screen plus successors, K rows) and the N rows nearest the state by Hamming
distance, with no norm, on a dump no route has read (``head_share_torch.py dump --split heldout --phase 0.5
--avoid-phase 0``: the held-out tails' windows halfway between the earlier ones, none sharing a token with them).
  agreement   the full head's best token is in the union. A tie at the Nth distance is counted against the pass
              (the conservative count: a card's top-k may break it either way); the count with ties for is reported
  rows        by bytes read per token: the index's pass (its context centroids), the Hamming pass (the vocabulary / 32)
              and the union's rows scored exactly (the lists themselves, ties at the Nth place to the lower id)
  interval    over the dump's windows (1,024 positions each), 2,000 resamples, seed 68, the 51st to the 1,950th
Lines (written in the order before the dump): agreement at least 0.995; rows at most 32,000. Kill: the agreement.
Held-out positions are read once: the script refuses a dump it already has a log for.

    python validation/head_lab_distance.py --export <base export> --dumps <head_shortlist folder> --cache <token cache>
-> validation/logs/head_lab_distance_<date>_<dump>.json

Wired: STANDALONE (a read; order 0068's report reads it). numpy only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

__wiring__ = "STANDALONE"

WINDOW = 1024


def nearest(dist: np.ndarray, n: int) -> np.ndarray:
    """The N rows nearest by Hamming distance, ties at the Nth place to the lower id (sorted ids)."""
    v = len(dist)
    if n >= v:
        return np.arange(v, dtype=np.int64)
    key = dist.astype(np.int64) * v + np.arange(v, dtype=np.int64)
    return np.sort(np.argpartition(key, n - 1)[:n]).astype(np.int64)


def hit_counts(dist: np.ndarray, best: int, n: int) -> tuple[bool, bool]:
    """(hit with ties counted against, hit with ties counted for): is the best row among the N nearest."""
    d0 = dist[best]
    closer, level = int((dist < d0).sum()), int((dist <= d0).sum()) - 1
    return level < n, closer < n


def interval(hits: np.ndarray, size: int = WINDOW, reps: int = 2000, seed: int = 68) -> list:
    """The agreement over resampled windows: the 51st and the 1,950th of `reps` (a 95% interval)."""
    per = hits[:len(hits) // size * size].reshape(-1, size).mean(1)
    rng = np.random.default_rng(seed)
    draws = np.sort(per[rng.integers(0, len(per), (reps, len(per)))].mean(1))
    return [round(float(draws[50]), 5), round(float(draws[reps - 51]), 5)]


def fresh(d: dict, cache: str, length: int = WINDOW) -> dict:
    """Every window of the dump against the held-out windows read before (phase 0, 4 a shard): none may share a token."""
    from head_dump import SHARDS, starts
    hold = json.load(open(os.path.join(cache, "_holdout.json"), encoding="utf-8"))
    worst = None
    for si, s in zip(d["win_shard"], d["win_start"]):
        name = SHARDS[int(si)]
        total = os.path.getsize(os.path.join(cache, f"{name}.u32")) // 4
        old = starts(total, int(hold[name]), "heldout", 4, length, 0.0)
        gap = min(abs(int(s) - o) for o in old)
        worst = gap if worst is None else min(worst, gap)
    return {"windows": int(len(d["win_start"])), "closest_gap_to_an_earlier_window": worst,
            "shares_no_token": bool(worst is not None and worst > length)}


def read(idx, w: np.ndarray, d: dict, k: int, n: int, chunk: int = 256, log=print) -> dict:
    from cubbyllm.model.vocab.hamming import pack, popcount
    hid, prev, m = d["hidden"], d["prev"], len(d["hidden"])
    bits = pack(w)
    against, for_, ctx, size = (np.zeros(m, dtype=bool), np.zeros(m, dtype=bool), np.zeros(m, dtype=bool),
                                np.zeros(m, dtype=np.int64))
    for s in range(0, m, chunk):
        h = hid[s:s + chunk].astype(np.float32)
        best = (h @ w.T).argmax(1)
        lists = idx.shortlist(h, k, "screen+succ", prev[s:s + chunk])
        for i, q in enumerate(pack(h)):
            dist = popcount(bits ^ q).sum(1, dtype=np.int32)
            a, f = hit_counts(dist, int(best[i]), n)
            pos = int(np.searchsorted(lists[i], best[i]))
            c = pos < len(lists[i]) and lists[i][pos] == best[i]
            against[s + i], for_[s + i], ctx[s + i] = a or c, f or c, c
            size[s + i] = len(np.union1d(lists[i], nearest(dist, n)))
        if (s // chunk) % 10 == 0:
            log(f"{min(s + chunk, m)}/{m}: agree {against[:s + len(h)].mean():.5f}")
    by_bytes = len(idx.screen.centroids) + len(w) / 32.0 + float(size.mean())
    shard = d["shard"]
    return {"positions": m,
            "agree": round(float(against.mean()), 5), "misses": int((~against).sum()),
            "agree_ties_for": round(float(for_.mean()), 5),
            "interval_95": interval(against),
            "context_list_alone": round(float(ctx.mean()), 5),
            "caught_where_the_list_misses": round(float((against & ~ctx).sum() / max(1, (~ctx).sum())), 5),
            "rows_exact": {"mean": round(float(size.mean()), 1), "p99": int(np.quantile(size, 0.99)), "max": int(size.max())},
            "rows_by_bytes": {"index_pass": int(len(idx.screen.centroids)), "hamming_pass": round(len(w) / 32.0, 1),
                              "mean_total": round(by_bytes, 1)},
            "by_shard": {str(name): round(float(against[shard == i].mean()), 5)
                         for i, name in enumerate(d["shards"]) if (shard == i).any()},
            "lines": {"agree_at_least_0.995": bool(against.mean() >= 0.995), "rows_at_most_32000": bool(by_bytes <= 32000)}}


def main() -> None:
    from head_shortlist_eval import head_rows

    from cubbyllm.model.vocab.shortlist import ShortlistIndex
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--cache", required=True, help="the token cache, to check the dump's windows are fresh")
    ap.add_argument("--dump", default="heldout_b")
    ap.add_argument("--index", default="index_router_succ.npz")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    if glob.glob(os.path.join(HERE, "logs", f"head_lab_distance_*_{a.dump}.json")):
        raise SystemExit(f"{a.dump} was read already: held-out positions are read once (order 0068)")
    path = os.path.join(a.dumps, f"{a.dump}.npz")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    d = dict(np.load(path))
    check = fresh(d, a.cache)
    log(f"{a.dump}: {len(d['target'])} positions, sha256 {sha[:12]}..., {check}")
    if not check["shares_no_token"]:
        raise SystemExit("the dump shares tokens with the held-out windows read before: not read")
    w = head_rows(a.export)
    idx = ShortlistIndex.load(os.path.join(a.dumps, a.index))
    rep = {"what": "order 0068: route 5 with the Hamming pass by distance alone, fresh held-out, read once",
           "date": _dt.date.today().isoformat(), "dump": a.dump, "sha256": sha, "fresh": check,
           "k": a.k, "n": a.n, "threads": os.environ.get("OMP_NUM_THREADS"),
           "read": read(idx, w, d, a.k, a.n, log=log)}
    out = os.path.join(HERE, "logs", f"head_lab_distance_{rep['date']}_{a.dump}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    log(json.dumps(rep, indent=1))
    log("->", os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
