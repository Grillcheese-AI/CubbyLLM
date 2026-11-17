"""validation/head_lab_time.py -- order 0068, step 2: route 5's time per token on a card (grilly2), against a full 32k
head and the full 128k head.

Route 5 by distance alone (``head_lab_distance.py``) end to end per token. Two builds:
  --route host  (step 2) today's grilly2 ops, the list on the host:
    readback   the state read back from the card (4 KB), its signs packed on the host (``hamming.pack``)
    hamming    grilly2's packed top-k (``grilly.vsa.packed.topk``): the N codes of the head's rows nearest the state
    context    the context pass on the card (the screen's centroids), the clusters ranked on the host
    union      the context list (nearest clusters until K tokens, plus the previous token's successors) joined with
               the N nearest, on the host
    score      the union's rows gathered on the card (``index_select``), scored (a linear), the best token read back
  --route card  (step 2b) grilly2's ``grilly.infer.shortlist.ShortlistHead``: the same union and the best of it on
               the card in eight dispatches, the token read back (16 bytes); each dispatch's time is the p50 of the
               step stopped after it, less the one before
Against, on the same card and states: the full head and a full 32k head (its first 32,000 rows), each a linear and an
argmax read back, as 0050 timed them. Batch 1, fp32, p50 of --reps tokens after 3 warm-ups, each token its own state.
The line (0050's): route 5 at most a full 32k head's time. Also checked: route 5's token is the full head's wherever
the union holds it.

    uv run --project <grilly2> python validation/head_lab_time.py --export <base export> --dumps <head_shortlist folder>
-> validation/logs/head_lab_time_grilly_<date>.json (host), head_lab_time_grilly_card_<date>.json (card)

Wired: STANDALONE (a timing; order 0068's report reads it). Imports grilly (grilly2), never torch.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

__wiring__ = "STANDALONE"

STAGES = ("context", "hamming", "plan", "mark", "count", "cut", "score", "best")


def context_ids(screen, order: np.ndarray, k: int, succ_row: np.ndarray | None) -> np.ndarray:
    """One query's context list as ``ScreenIndex.shortlist`` builds it: the nearest clusters' tokens until K distinct,
    plus the successors; sorted, unique."""
    got, n = [], 0
    seen = np.zeros(0, dtype=np.int64)
    for c in order:
        lst = screen.ids[screen.offsets[c]:screen.offsets[c + 1]]
        new = lst[~np.isin(lst, seen)] if len(seen) else lst
        take = new[:k - n]
        got.append(take)
        n += len(take)
        seen = np.concatenate([seen, new])
        if n >= k:
            break
    ids = np.concatenate(got) if got else np.zeros(0, dtype=np.int64)
    if succ_row is not None:
        ids = np.concatenate([ids, succ_row[succ_row >= 0]])
    return np.unique(ids.astype(np.int64))


def _p50(ts: list) -> dict:
    s = sorted(ts)
    return {"p50_ms": round(statistics.median(s), 3), "mean_ms": round(statistics.fmean(s), 3),
            "p90_ms": round(s[int(0.9 * (len(s) - 1))], 3), "n": len(s)}


def _grilly_commit(g) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(g.__file__),
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    import grilly as g
    import grilly.vsa.packed as packed
    from head_shortlist_eval import head_rows

    from cubbyllm.model.vocab.hamming import pack
    from cubbyllm.model.vocab.shortlist import ShortlistIndex
    from cubbyllm.ops.grilly_seats import load_export
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--dump", default="heldout_b")
    ap.add_argument("--index", default="index_router_succ.npz")
    ap.add_argument("--route", choices=("host", "card"), default="host")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ranks", type=int, default=256, help="card: clusters scanned at most")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    F = g.nn.functional
    model, cfg = load_export(a.export)
    w = head_rows(a.export)
    v, dim = w.shape
    W = model.lm_head.weight
    W32 = g.from_numpy(np.ascontiguousarray(w[:32000]))
    idx = ShortlistIndex.load(os.path.join(a.dumps, a.index))
    screen, succ = idx.screen, idx.succ
    C = g.from_numpy(np.ascontiguousarray(screen.centroids, dtype=np.float32))
    d = np.load(os.path.join(a.dumps, f"{a.dump}.npz"))
    hs, prevs = d["hidden"][:a.reps + 3].astype(np.float32), d["prev"][:a.reps + 3]
    xs = [g.from_numpy(h.reshape(1, 1, dim)) for h in hs]
    card = None
    if a.route == "card":
        from grilly.infer.shortlist import ShortlistHead
        card = ShortlistHead(W, screen.centroids, screen.ids, screen.offsets, succ, near=a.n, context=a.k,
                             ranks=a.ranks)
    else:
        book = g.from_numpy(np.ascontiguousarray(pack(w).view(np.int32)))

    def best(logits) -> int:
        return int(g.argmax(logits.reshape(1, -1), dim=-1).numpy().reshape(-1)[0])

    names = ("head128k", "head32k", "route5") + (("readback", "hamming", "context", "union", "score")
                                                 if card is None else ())
    parts = {p: [] for p in names}
    agree = held = short = 0
    rows = []
    with g.no_grad():
        for i, x in enumerate(xs):
            warm = i < 3
            prev = int(prevs[i])
            t0 = time.perf_counter()
            full = best(F.linear(x, W))
            t1 = time.perf_counter()
            best(F.linear(x, W32))
            t2 = time.perf_counter()
            if card is not None:
                tok = card.argmax(x, prev)
                t7 = time.perf_counter()
                if warm:
                    continue
                ids, was_short = card.union(x, prev)
                short += was_short
                rows.append(card.last.rows)
                for p, s, e in (("head128k", t0, t1), ("head32k", t1, t2), ("route5", t2, t7)):
                    parts[p].append((e - s) * 1e3)
            else:
                h = x.numpy().reshape(1, dim)                                   # route 5 starts here
                q = pack(h).view(np.int32)
                t3 = time.perf_counter()
                near = packed.topk(g.from_numpy(np.ascontiguousarray(q)), book, k=a.n, dim=dim)[0].numpy()
                near = near.astype(np.int64)
                t4 = time.perf_counter()
                order = np.argsort(-F.linear(x, C).numpy().reshape(-1), kind="stable")
                t5 = time.perf_counter()
                ids = np.union1d(context_ids(screen, order, a.k, None if succ is None else succ[prev]), near)
                t6 = time.perf_counter()
                sel = g.index_select(W, 0, g.from_numpy(ids.astype(np.int64)))
                tok = int(ids[best(F.linear(x, sel))])
                t7 = time.perf_counter()
                if warm:
                    continue
                for p, s, e in (("head128k", t0, t1), ("head32k", t1, t2), ("readback", t2, t3), ("hamming", t3, t4),
                                ("context", t4, t5), ("union", t5, t6), ("score", t6, t7), ("route5", t2, t7)):
                    parts[p].append((e - s) * 1e3)
            inside = bool(np.isin(full, ids))
            held += inside
            agree += inside and tok == full
        stages = None
        if card is not None:
            # each dispatch's share: the step stopped after it, less the step stopped before it
            per = {s: [] for s in range(len(STAGES) + 1)}
            for i, x in enumerate(xs):
                for s in range(len(STAGES) + 1):
                    t0 = time.perf_counter()
                    card.profile(x, int(prevs[i]), s)
                    if i >= 3:
                        per[s].append((time.perf_counter() - t0) * 1e3)
            p50 = {s: statistics.median(ts) for s, ts in per.items()}
            stages = {"wait_alone_ms": round(p50[0], 3),
                      "added_ms": {name: round(p50[s + 1] - p50[s], 3) for s, name in enumerate(STAGES)},
                      "all_eight_then_wait_ms": round(p50[len(STAGES)], 3)}
    res = {"what": "order 0068 step 2" + ("b" if card is not None else "")
                   + f": route 5 (distance alone, {a.route} build) per token on grilly2, against full 32k and 128k heads",
           "date": _dt.date.today().isoformat(), "machine": platform.platform(), "grilly2": _grilly_commit(g),
           "route": a.route, "batch": 1, "precision": "fp32",
           "vocab": v, "hidden": dim, "k": a.k, "n": a.n, "reps": a.reps,
           "times": {p: _p50(ts) for p, ts in parts.items()},
           "check": {"states": a.reps, "full_heads_token_in_the_union": held,
                     "route5_token_is_the_full_heads_where_held": agree}}
    if card is not None:
        res["ranks"] = a.ranks
        res["stages"] = stages
        res["check"].update({"short_lists": short, "union_rows_mean": round(float(np.mean(rows)), 1),
                             "union_rows_max": int(np.max(rows))})
    t = res["times"]
    res["line"] = {"route5_p50_ms": t["route5"]["p50_ms"], "head32k_p50_ms": t["head32k"]["p50_ms"],
                   "at_most_a_32k_heads_time": t["route5"]["p50_ms"] <= t["head32k"]["p50_ms"]}
    tag = "grilly_card" if card is not None else "grilly"
    path = os.path.join(HERE, "logs", f"head_lab_time_{tag}_{res['date']}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
    print("->", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
