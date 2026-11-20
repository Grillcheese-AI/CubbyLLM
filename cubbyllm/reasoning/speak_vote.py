"""cubbyllm/reasoning/speak_vote.py -- when an arithmetic answer is spoken (order 0078, step 5): k tries, each run step by
step in the VM, and the answer is said only when at least m of them land on the same VM value; otherwise it is refused.

- ``vote(values, m)``: the values the k tries reached (None for a try that did not run to a value: it counts as a
  try, never as a vote) -> the value said, or a refusal and why. Values within ``TOL`` (relative) are one value. The
  most common value is said when it has at least m votes; if another value ties it for first, the tries disagree and
  the answer is refused: a tie is never broken by a guess.
- ``choose_m(dev, k)``: m chosen on dev only, then frozen in the order's report before test: the smallest m whose
  wrong answers stay at or under ``max_wrong`` of all dev questions (a smaller m speaks more). Returns the m and the
  right / wrong / refused counts for every m from 1 to k, so the report shows what each would have done.

A refusal is a result, never a wrong answer. No model judges a value here: the values are the VM's, and the gold only
enters ``choose_m``, on dev.

Wiring: STANDALONE (order 0078's read: `validation/`; nothing in the serve path until Nick's word on the kept arm).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

TOL = 1e-6


def same(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def vote(values: list, m: int, tol: float = TOL) -> dict:
    """{"answer": the value said or None, "votes": its count, "k": tries, "why": None | "too few agree" | "tie"}."""
    groups: list[list[float]] = []
    for v in values:
        if v is None:
            continue
        g = next((g for g in groups if same(v, g[0], tol)), None)
        if g is None:
            groups.append([v])
        else:
            g.append(v)
    groups.sort(key=len, reverse=True)
    top = len(groups[0]) if groups else 0
    if top < m:
        return {"answer": None, "votes": top, "k": len(values), "why": "too few agree"}
    if len(groups) > 1 and len(groups[1]) == top:
        return {"answer": None, "votes": top, "k": len(values), "why": "tie"}
    return {"answer": groups[0][0], "votes": top, "k": len(values), "why": None}


def outcome(values: list, gold: float, m: int, tol_gold: float = 0.01) -> str:
    """right / wrong / refused for one dev question (the gold within 0.01, as GSM8K's checker reads it)."""
    v = vote(values, m)
    if v["answer"] is None:
        return "refused"
    return "right" if abs(v["answer"] - gold) <= tol_gold else "wrong"


def choose_m(dev: list[tuple[list, float]], k: int, max_wrong: float = 0.01) -> dict:
    """``dev``: [(the k tries' values, the gold)] -> {"m": the smallest m within the wrong line (None: none is),
    "by m": {m: {"right", "wrong", "refused", "wrong share"}}}."""
    table = {}
    for m in range(1, k + 1):
        c = {"right": 0, "wrong": 0, "refused": 0}
        for values, gold in dev:
            c[outcome(values[:k], gold, m)] += 1
        c["wrong share"] = round(c["wrong"] / max(1, len(dev)), 4)
        table[m] = c
    ok = [m for m in table if table[m]["wrong"] <= max_wrong * len(dev)]
    return {"m": min(ok) if ok else None, "k": k, "max wrong": max_wrong, "questions": len(dev), "by m": table}


__all__ = ["TOL", "choose_m", "outcome", "same", "vote"]
