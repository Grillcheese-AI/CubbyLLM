"""Order 0073, step 3: replies that can tell a right reading from a wrong one, and how much a row's replies tell.

**One-reversal mutants** of a row, on its folded forms (arm M) and its groups, the evaluator only:
- of a part (`part_mutants`): a comparison negated (``<`` for ``>=``), a bound mirrored (``<`` for ``>``), ``any``
  and ``none`` exchanged, ``absent`` toggled;
- of a group (`group_mutants`): the pair untied (both must hold), its join changed (``or`` for ``if`` and back, the
  ``if`` turned round), or tied to the other neighbour.
`told(parts, groups, views)` counts the mutants and those some reply tells apart from the gold, by the row's verdict
and by the verdict of what was changed run alone.

**Replies that can tell** (`extras`): from the row's pool, for each part whose own comparison has a number N (a
view's size, a count, a position's numeric value) one reply measuring N - 1, one N and one N + 1, and for each tied
pair one reply for each of the four combinations of its two parts' truths, where the replies kept so far have none
and the pool has one. A reply on which the row's verdict turns on that part or pair alone is taken first: a reply
that another part fails shows nothing at the row's verdict. Nothing is edited; `reach` says which were reached and
how (2: on such a reply; 1: on another; 0: not).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
import view_wording as W  # noqa: E402

from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

__wiring__ = "STANDALONE"

MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _ops(op: str) -> list[str]:
    return [VS.INV[op]] + ([MIRROR[op]] if op in MIRROR else [])


def part_mutants(m: dict) -> list[dict]:
    """`m` (a folded form) with one slot reversed, each once, never `m` itself."""
    out, q = [], m["q"]
    if q in ("n", "count"):
        out += [dict(m, op=o) for o in _ops(m["op"])]
    elif q in ("any", "none"):
        out.append(dict(m, q="none" if q == "any" else "any"))
    elif q == "at":
        out += [dict(m, op=o) for o in _ops(m["op"])]
        out.append({k: v for k, v in m.items() if k != "absent"} if m.get("absent") else dict(m, absent="pass"))
    for i, c in enumerate(m.get("where", [])):
        for o in _ops(c["op"]):
            out.append(dict(m, where=[dict(c, op=o) if j == i else x for j, x in enumerate(m["where"])]))
    seen, uniq = [m], []
    for x in out:
        if x not in seen:
            seen.append(x)
            uniq.append(x)
    return uniq


def group_mutants(groups: list, n_parts: int) -> list[list]:
    """`groups` with one tied pair's join changed, each once."""
    out = []
    tied = {x for g in groups if g[0] != "one" for x in g[1:]}
    for k, g in enumerate(groups):
        if g[0] == "one":
            continue
        a, b = g[1], g[2]
        joins = {("either", a, b), ("if", a, b), ("if", b, a)}
        same = ("either", a, b) if g[0] in ("either", "unless") else tuple(g)
        for alt in sorted(joins - {same}):
            out.append(groups[:k] + [alt] + groups[k + 1:])
        out.append(groups[:k] + [("one", a), ("one", b)] + groups[k + 1:])
        lo, hi = min(a, b), max(a, b)
        for new, free in ((lo - 1, hi), (hi + 1, lo)):       # tied to the other neighbour, which takes the freed
            if 0 <= new < n_parts and new not in tied:       # part's place in the join; the freed part stands alone
                moved = (g[0], *[new if x == free else x for x in g[1:]])
                out.append([h for h in groups if h not in (g, ("one", new))] + [("one", free), moved])
    return out


def told(parts: list, groups: list, views: list) -> dict:
    """How many one-reversal mutants the row has and how many `views` (its replies') tell apart from the gold: by the
    row's verdict (``told``), and by the verdict of what was changed run alone (``told alone``: the part, or the
    groups the change touches)."""
    groups = [tuple(g) for g in groups]

    def run(ps, gs):
        return [VS.evaluate(VJ.spec(ps, gs), v) for v in views]
    gold = run(parts, groups)
    t = {"parts": 0, "parts told": 0, "parts told alone": 0, "groups": 0, "groups told": 0, "groups told alone": 0}
    for i, m in enumerate(parts):
        own = [VS.evaluate(m, v) for v in views]
        for x in part_mutants(m):
            t["parts"] += 1
            t["parts told"] += int(run(parts[:i] + [x] + parts[i + 1:], groups) != gold)
            t["parts told alone"] += int([VS.evaluate(x, v) for v in views] != own)
    for alt in group_mutants(groups, len(parts)):
        t["groups"] += 1
        t["groups told"] += int(run(parts, alt) != gold)
        t["groups told alone"] += int(run(parts, [g for g in alt if g not in groups])
                                      != run(parts, [g for g in groups if g not in alt]))
    return t


def number(m: dict):
    """The number a folded form's own comparison is made with, or None (a filter's numbers are not its own)."""
    if m["q"] in ("n", "count"):
        return m["v"]
    if m["q"] == "at" and m["col"] in W.NUMERIC:
        return m["val"]
    return None


def measured(m: dict, views: dict):
    """What a reply measures where `m` compares: the view's size, the rows matching, the value at the position."""
    rows = views[m["view"]]
    if m["q"] == "n":
        return len(rows)
    if m["q"] == "count":
        return sum(1 for r in rows if VS.match(r, m["where"]))
    idx = m["pos"] - 1 if m["pos"] > 0 else len(rows) - 1
    return rows[idx][m["col"]] if 0 <= idx < len(rows) else None


def extras(rng, parts: list, groups: list, kept: list, pool: list) -> tuple[list, dict]:
    """(indices into `pool` to add, what was reached). `kept` and `pool` are lists of views; `pool` only usable ones.
    Reached is 2 on a reply where the row's verdict turns on this part or group alone (the row's other groups hold
    and, for a tied part, its partner leaves the pair's verdict to it), so the reply can tell at the row's verdict;
    1 on another reply; 0 when there is none."""
    have, add = list(kept), []
    reach = {"numbers": [], "groups": []}
    order = list(range(len(pool)))
    rng.shuffle(order)
    own = [VJ.spec(parts, [tuple(g)]) for g in groups]
    group_of = {x: k for k, g in enumerate(groups) for x in g[1:]}

    def turns(i: int):
        """On which replies the row's verdict is part `i`'s."""
        k, g = group_of[i], groups[group_of[i]]
        j = None if g[0] == "one" else g[2] if g[1] == i else g[1]
        need = g[0] == "if" and g[1] == j                  # an `if`'s condition must hold; any other partner must fail
        return lambda v: (all(VS.evaluate(s, v) for x, s in enumerate(own) if x != k)
                          and (j is None or VS.evaluate(parts[j], v) == need))

    def find(test, alone) -> int:
        for level, ok in ((2, lambda v: test(v) and alone(v)), (1, test)):
            if any(ok(v) for v in have):
                return level
            for j in order:
                if j not in add and ok(pool[j]):
                    add.append(j)
                    have.append(pool[j])
                    return level
        return 0
    for i, m in enumerate(parts):
        n = number(m)
        if n is not None:
            reach["numbers"].append([i, [find(lambda v, d=d: measured(m, v) == n + d, turns(i)) for d in (-1, 0, 1)]])
    for k, g in enumerate(groups):
        if g[0] != "one":
            a, b = parts[g[1]], parts[g[2]]
            rest = lambda v, k=k: all(VS.evaluate(s, v) for x, s in enumerate(own) if x != k)   # noqa: E731
            reach["groups"].append([k, [find(lambda v, x=x, y=y: (VS.evaluate(a, v), VS.evaluate(b, v)) == (x, y), rest)
                                        for x in (True, False) for y in (True, False)]])
    return add, reach


__all__ = ["extras", "group_mutants", "measured", "number", "part_mutants", "told"]
