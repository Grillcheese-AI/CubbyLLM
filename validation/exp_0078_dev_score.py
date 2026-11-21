"""exp_0078_dev_score - order 0078's dev read scored (Cowork's ruling, 2026-11-20), from the rows
`exp_0078_dev_read.py` wrote, with no second pass:

- **the speak rule (deciding):** for k in {8, 16} (k = 8 is the first 8 tries), m chosen by `speak_vote.choose_m` (the
  smallest m keeping dev's wrong at or under 1% of the questions); right / wrong / refused at that (k, m), and the
  table for every m. The arm's dev right is the better k's right at its own m.
- **one greedy try (reported, not deciding):** right / wrong / refused, the gold within 0.01 (the speak rule's).

`drop`: a json file with a "drop" list (or a bare list) of ids to leave out (S0's footnote).

Wiring: STANDALONE (order 0078).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cubbyllm.reasoning.speak_vote import choose_m  # noqa: E402

__wiring__ = "STANDALONE"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(path: str, drop: str = "") -> list[dict]:
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    if drop:
        d = json.load(open(drop, encoding="utf-8"))
        ids = set(d["drop"] if isinstance(d, dict) else d)
        rows = [r for r in rows if r["id"] not in ids]
    return rows


def greedy(rows: list[dict]) -> dict:
    c = {"right": 0, "wrong": 0, "refused": 0}
    for r in rows:
        v, gold = num(r["greedy"]["answer"]), float(r["gold"])
        c["refused" if v is None else "right" if abs(v - gold) <= 0.01 else "wrong"] += 1
    return c


def score(path: str, drop: str = "") -> dict:
    rows = load_rows(path, drop)
    dev = [([num(t["answer"]) for t in r["tries"]], float(r["gold"])) for r in rows]
    by_k = {}
    for k in (8, 16):
        ch = choose_m(dev, k)
        at = ch["by m"].get(ch["m"]) if ch["m"] else None
        by_k[k] = {"m": ch["m"], "at m": at, "by m": ch["by m"]}
    best = max((k for k in by_k if by_k[k]["at m"]), key=lambda k: (by_k[k]["at m"]["right"], -k), default=None)
    secs = [r["seconds"]["greedy"] + r["seconds"]["tries"] for r in rows]
    return {"rows": path, "dropped": drop or None, "questions": len(rows),
            "speak rule": {"k": best, "m": by_k[best]["m"] if best else None,
                           "right": by_k[best]["at m"]["right"] if best else 0, "by k": by_k},
            "greedy": greedy(rows),
            "any try right (the oracle)": sum(any(v is not None and abs(v - g) <= 0.01 for v in vs) for vs, g in dev),
            "seconds a question": {"mean": round(sum(secs) / max(1, len(secs)), 1),
                                   "median": sorted(secs)[len(secs) // 2] if secs else None}}


__all__ = ["greedy", "load_rows", "score"]
