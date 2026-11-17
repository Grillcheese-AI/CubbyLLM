"""validation/head_lab_card_union.py -- order 0068, step 2b: does the card build of route 5 score the union the lab read?

grilly2's ``ShortlistHead`` computes route 5 (distance alone) on the card. Its clusters are ranked from the card's fp32
scores, so a near tie may rank differently from numpy's and change the context list. This compares, on held-in states
(never held-out), every position's card union with numpy's as ``head_lab_distance.py`` read it: ``ScreenIndex.shortlist``
(screen plus successors, K) joined with the N rows nearest by Hamming distance, ties to the lower id. Where they differ,
it also asks whether the chosen token differs (numpy's best over its union against the card's). The log keeps every
difference that is not a short list, and the first 200 that are. Its ``grilly2`` field is the grilly checkout's commit
when the check ends: do not move that checkout during a run.

    python validation/head_lab_card_union.py --export <base export> --dumps <head_shortlist folder> [--limit n]
-> validation/logs/head_lab_card_union_<date>_<dump>.json

Wired: STANDALONE (a check; order 0068's report reads it). numpy, and grilly2 for the card.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

__wiring__ = "STANDALONE"


def grilly_commit(g) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(g.__file__),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def compare(ref: np.ndarray, got: np.ndarray) -> tuple[int, int]:
    """(rows only numpy's union holds, rows only the card's holds)."""
    return len(np.setdiff1d(ref, got, assume_unique=True)), len(np.setdiff1d(got, ref, assume_unique=True))


def main() -> None:
    import grilly as g
    from grilly.infer.shortlist import ShortlistHead
    from head_lab_distance import nearest
    from head_shortlist_eval import head_rows

    from cubbyllm.model.vocab.hamming import pack, popcount
    from cubbyllm.model.vocab.shortlist import ShortlistIndex
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--dump", default="heldin_val")
    ap.add_argument("--index", default="index_router_succ.npz")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ranks", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first n positions only (a trial; the report reads all)")
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    if a.dump.startswith("heldout"):
        raise SystemExit("held-out positions are not for this check (order 0068): use a held-in dump")
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    w = head_rows(a.export)
    idx = ShortlistIndex.load(os.path.join(a.dumps, a.index))
    head = ShortlistHead(g.from_numpy(w), idx.screen.centroids, idx.screen.ids, idx.screen.offsets, idx.succ,
                         near=a.n, context=a.k, ranks=a.ranks)
    d = np.load(os.path.join(a.dumps, f"{a.dump}.npz"))
    hid, prev = d["hidden"], d["prev"]
    m = len(hid) if not a.limit else min(a.limit, len(hid))
    bits = pack(w)
    same = short = 0
    only_ref, only_card, token_differs, sizes, differ_at, details = [], [], 0, [], [], []
    t_start = time.time()
    with g.no_grad():
        for s in range(0, m, a.chunk):
            h = hid[s:min(m, s + a.chunk)].astype(np.float32)
            lists = idx.shortlist(h, a.k, "screen+succ", prev[s:s + len(h)])
            for i, q in enumerate(pack(h)):
                dist = popcount(bits ^ q).sum(1, dtype=np.int32)
                ref = np.union1d(lists[i], nearest(dist, a.n))
                x = g.from_numpy(h[i])
                got, was_short = head.union(x, int(prev[s + i]))
                short += was_short
                sizes.append(len(got))
                a_ref, a_card = compare(ref, got.astype(np.int64))
                if a_ref == 0 and a_card == 0:
                    same += 1
                    continue
                differ_at.append(s + i)
                only_ref.append(a_ref)
                only_card.append(a_card)
                want = int(ref[np.argmax(w[ref] @ h[i])])
                differs = head.argmax(x, int(prev[s + i])) != want
                token_differs += differs
                details.append({"position": s + i, "short": bool(was_short), "rows_only_numpy_has": a_ref,
                                "rows_only_the_card_has": a_card, "token_differs": bool(differs)})
            log(f"{s + len(h)}/{m}: same {same}, differ {len(differ_at)}, token differs {token_differs}, "
                f"short {short}, {time.time() - t_start:.0f} s")
    rep = {"what": "order 0068 step 2b: the card build's union against numpy's, held-in, every position",
           "date": _dt.date.today().isoformat(), "grilly2": grilly_commit(g), "dump": a.dump, "positions": m, "k": a.k, "n": a.n, "ranks": a.ranks,
           "same_union": same, "union_differs": len(differ_at), "token_differs_where_the_union_does": token_differs,
           "short_lists": short,
           "rows_only_numpy_has": {"total": int(np.sum(only_ref)), "max": int(np.max(only_ref)) if only_ref else 0},
           "rows_only_the_card_has": {"total": int(np.sum(only_card)), "max": int(np.max(only_card)) if only_card else 0},
           "union_differs_where_not_short": sum(1 for x in details if not x["short"]),
           "differences": [x for x in details if not x["short"]] + [x for x in details if x["short"]][:200],
           "card_union_rows": {"mean": round(float(np.mean(sizes)), 1), "max": int(np.max(sizes))},
           "seconds": round(time.time() - t_start)}
    tag = f"_first{m}" if a.limit else ""
    out = os.path.join(HERE, "logs", f"head_lab_card_union_{rep['date']}_{a.dump}_{rep['grilly2']}{tag}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    log(json.dumps(rep, indent=1))
    log("->", os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
