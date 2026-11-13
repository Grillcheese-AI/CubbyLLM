"""r3g_guards -- order 0070, B: three guards in front of a relation arm. A guard either lets the arm's list through,
unchanged, or refuses the row ("missing fact"). No guard adds a property, a fact or a score to the list.

  G1  overall against own: the arm's own scorer over the whole property codebook and over the row's candidates (the
      properties its items carry). Refuse when the best property overall is one the items do not carry and beats the
      items' best by the margin m: best_absent > best_own and (best_absent - best_own) / best_absent >= m.
  G2  expected but absent: the properties that at least a share s of the items of the row's classes carry (``r3g_classes``,
      counts from the local store, statistics, never facts) join the candidates when the items lack them; the arm scores
      the joined set. Refuse when the best score's run (every property at that score) holds one of them. A row whose
      classes are all too small in the store (fewer than ``N_MIN`` items) or unknown is "unavailable": the list passes,
      and the row is counted apart, never as "none expected".
  G3  independent support (qFHRR and the 450M only): the list keeps only the properties the lexicon (``lexicon_content``,
      any wording it finds among the candidates) or the grammar also names; an empty list is a refusal. Agreement only
      filters; it never certifies.
  G1+G2  refuse when either refuses.

Thresholds (pre-registered with the split, 2026-11-13, before any guard ran): G1's m on ``MARGINS``, G2's s on ``SHARES``,
set per arm on guard-dev only (``choose``):
  1. the present line: on guard-dev's ids with the gold carried, recall@5 down by at most 0.02 against the same arm
     without the guard, and SimpleQuestions recall@5 at least 0.85 where the arm without the guard had it;
  2. among the thresholds that keep it, the fewest handovers on guard-dev's hidden rows; then the smaller recall loss;
     then the threshold that refuses least (the larger m, the larger s);
  3. none keeps it: the threshold with the smallest recall loss (then the fewest handovers), flagged "present line not
     met on guard-dev".

Wired: STANDALONE (order 0070's bench).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SHARES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_MIN = 50                       # a class with fewer items in the local store is "unavailable" for G2
LOSS = 0.02                      # the present line: recall@5 down by at most 2 points
SQ_LINE = 0.85                   # SimpleQuestions recall@5 kept where the arm had it
REFUSED = {"read": False, "top": [], "undecided": False, "refused": "missing fact"}


def bests(scores: dict, carried: set) -> tuple[float, float]:
    """(the best score among properties not carried, the best among those carried); 0 for none."""
    absent = max((s for p, s in scores.items() if p not in carried), default=0.0)
    own = max((s for p, s in scores.items() if p in carried), default=0.0)
    return absent, own


def g1(best_absent: float, best_own: float, margin: float) -> bool:
    """True: refuse (a property the items do not carry beats theirs by the margin)."""
    return best_absent > 0 and best_absent > best_own and (best_absent - best_own) / best_absent >= margin


def best_run(scores: dict) -> set:
    if not scores:
        return set()
    top = max(scores.values())
    return {p for p, s in scores.items() if s == top} if top > 0 else set()


def g2(scores_joined: dict, expected_absent: set | None) -> bool:
    """True: refuse (the best run over candidates + expected-but-absent holds an absent one). None: unavailable."""
    if not expected_absent:
        return False
    return bool(best_run(scores_joined) & set(expected_absent))


def g3(out: dict, named: set) -> dict:
    """The arm's list kept to what the lexicon or the grammar names; refused when nothing is left."""
    if not out.get("read"):
        return out
    keep = [p for p in out.get("top") or () if p in named]
    return {**out, "top": keep, "undecided": False} if keep else dict(REFUSED)


def apply(out: dict, refuse: bool) -> dict:
    """The arm's list unchanged, or the refusal."""
    return dict(REFUSED) if refuse else out


def grid(guard: str) -> list:
    if guard == "G1":
        return [{"m": m} for m in MARGINS]
    if guard == "G2":
        return [{"s": s} for s in SHARES]
    if guard == "G1+G2":
        return [{"m": m, "s": s} for m in MARGINS for s in SHARES]
    return [{}]


def _lenience(t: dict) -> tuple:
    return (t.get("m", 0.0), t.get("s", 0.0))


def choose(points: list[dict]) -> dict:
    """The threshold, by the pre-registered rule. A point: {"t": thresholds, "handovers": n, "loss": recall drop,
    "sq_ok": bool} (guard-dev only)."""
    ok = [p for p in points if p["loss"] <= LOSS + 1e-12 and p["sq_ok"]]
    if ok:
        best = min(ok, key=lambda p: (p["handovers"], p["loss"], tuple(-x for x in _lenience(p["t"]))))
        return {**best, "flag": None}
    best = min(points, key=lambda p: (p["loss"], p["handovers"], tuple(-x for x in _lenience(p["t"]))))
    return {**best, "flag": "present line not met on guard-dev"}


__all__ = ["LOSS", "MARGINS", "N_MIN", "REFUSED", "SHARES", "SQ_LINE", "apply", "best_run", "bests", "choose", "g1", "g2",
           "g3", "grid"]
