"""Order 0073, step 3: the draw of a long row's parts and groups (the design is in the order's Report).

Against order 0067's draw (`build_long_rows.draw_parts`, `groups_for`):
- **bounds balanced.** A part's said form states bounds with a number other than 0 at four places: a view's size, a
  count, a position's value, a filter (`bounds`). The draw is handed the running tally of the split and, among the
  candidates it already tries for a part (the atom and its negation, six tries), takes the best one whose bounds are
  on the side that is behind at their place, when taking it leaves enough passing replies for the parts still to
  draw; otherwise the candidate that passes most replies, as before.
- **`not all` of two conditions is not drawn** (it has no description: `view_said.said` is None), nor a part whose
  folded said form is of a kind kept for the test (`held`).
- **groups evenly placed** (`place`): with two to four parts each adjacent pair is tied in the same share of the
  rows kept (the build hands back the count of what it kept); with five to eight the places are tried in a random
  order.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
import view_programs as VP  # noqa: E402
import view_said as S  # noqa: E402
import view_wording as W  # noqa: E402
import view_wording_inv as WI  # noqa: E402

__wiring__ = "STANDALONE"

SIDE = {">=": "lower", ">": "lower", "<=": "upper", "<": "upper", "==": "exactly", "!=": "other"}
OTHER = {"lower": "upper", "upper": "lower"}


def bounds(form: dict) -> list[tuple[str, str]]:
    """(place, side) of each comparison a said form makes with a number other than 0."""
    out = []
    if form["q"] in ("n", "count") and form["v"] != 0:
        out.append(("size" if form["q"] == "n" else "count", SIDE[form["op"]]))
    if form["q"] == "at" and form["col"] in W.NUMERIC and form["val"] != 0:
        out.append(("position", SIDE[form["op"]]))
    out += [("filter", SIDE[c["op"]]) for c in form.get("where", []) if c["col"] in W.NUMERIC and c["v"] != 0]
    return out


def lower_share(tally: Counter, place: str | None = None) -> float | None:
    """The lower bounds' share of the bounds in `tally` (at one place, or over all four)."""
    lo = sum(n for (p, s), n in tally.items() if s == "lower" and place in (None, p))
    up = sum(n for (p, s), n in tally.items() if s == "upper" and place in (None, p))
    return lo / (lo + up) if lo + up else None


def wanted(form: dict, tally: Counter) -> int:
    """1 when a bound of `form` is on the side that is behind at its place, -1 when one is ahead, else 0."""
    score = 0
    for place, side in bounds(form):
        if side in OTHER:
            gap = tally[(place, OTHER[side])] - tally[(place, side)]
            score += (gap > 0) - (gap < 0)
    return (score > 0) - (score < 0)


def held(cand: dict, form: dict | None = None) -> bool:
    """A part of a kind kept out of every row (`view_programs.held_out`): as the atom it equals, or as its folded
    said form. 0067's draw looked at the first only, and a negated ``at`` or ``all`` has no such atom."""
    form = form or S.said(cand, "en")
    eq = WI.positive(cand) if "not" in cand else cand
    return bool((eq is not None and VP.held_out(eq)) or (form is not None and VP.held_out(S.VS.fold(form))))


def draw_parts(rng, views: list, k: int, tally: Counter) -> list[dict] | None:
    passing, parts = list(range(len(views))), []
    for n in range(k):
        best, floor = None, 4 + 6 * (k - n - 1)            # replies a wanted candidate must leave for the rest
        for _ in range(6):
            a = VP.sample_atom(rng, [views[j] for j in passing])
            if a is None:
                continue
            for cand in (a, {"not": a}):
                form = S.said(cand, "en")
                if form is None or held(cand, form):       # not all of two conditions; a kind kept for the test
                    continue
                share = sum(VP.evaluate(cand, views[j]) for j in passing) / len(passing)
                if share >= 0.95:
                    continue
                want = wanted(form, tally) if share * len(passing) >= floor else 0
                if best is None or (want, share) > best[:2]:
                    best = (want, share, cand)
        if best is None:
            return None
        parts.append(best[2])
        passing = [j for j in passing if VP.evaluate(best[2], views[j])]
        if len(passing) < 4:
            return None
    return parts


SHARES = {                                                 # (parts, a lone part kept) -> {tied places: share of rows}
    (2, False): {(0,): 0.3, (): 0.7},
    (3, False): {(0,): 0.255, (1,): 0.255, (): 0.49},
    (4, False): {(0,): 0.157, (1,): 0.247, (2,): 0.157, (0, 2): 0.09, (): 0.349},   # each place in 0.247 of the rows
    (2, True): {(): 1.0},
    (3, True): {(0,): 0.255, (1,): 0.255, (): 0.49},
    (4, True): {(0,): 0.217, (1,): 0.217, (2,): 0.217, (): 0.349},                  # 0.651 tied, as above
}


def placement(groups: list) -> tuple:
    """Where a row's tied pairs sit: the first part of each, in order."""
    return tuple(sorted(min(g[1:]) for g in groups if g[0] != "one"))


def place(rng, k: int, lone: bool = False, seen: Counter | None = None) -> list[int]:
    """The first part of each tied pair (the pair is ``i, i + 1``), in order. ``lone``: a part at least stays untied
    (a minimal pair flips a lone part), so two parts are never tied and four have one pair at most. ``seen`` counts
    the rows kept so far by ``(parts, placement)``: with it the placement furthest behind its share is taken, so the
    rows kept have the shares whatever the build drops (a row with a pair is dropped more often); without it the
    placement is drawn."""
    if k < 2:
        return []
    if k <= 4:
        shares = SHARES[(k, lone)]
        if seen is None:
            return list(rng.choices(list(shares), weights=list(shares.values()))[0])
        total = sum(seen[(k, p)] for p in shares)
        short = {p: s * total - seen[(k, p)] for p, s in shares.items()}
        return list(rng.choice(sorted(p for p, n in short.items() if n >= max(short.values()) - 1e-9)))
    order, used, out = list(range(k - 1)), set(), []
    rng.shuffle(order)
    for i in order:
        if i not in used and i + 1 not in used and rng.random() < 0.3:
            out.append(i)
            used |= {i, i + 1}
    return sorted(out)


def groups_for(rng, parts: list[dict], lone: bool = False, seen: Counter | None = None) -> list[tuple]:
    tied, out, i = set(place(rng, len(parts), lone, seen)), [], 0
    while i < len(parts):
        if i not in tied:
            out.append(("one", i))
            i += 1
            continue
        kind = rng.choice(["unless", "if", "either"])
        plain = [x for x in (i + 1, i) if "not" not in parts[x]]    # a condition is said of a plain atom
        if kind == "either" or not plain:
            out.append(("either", i, i + 1))
        else:
            c = plain[0]
            other = i if c == i + 1 else i + 1
            out.append(("unless", other, c) if kind == "unless" else ("if", c, other))
        i += 2
    return out


def tally_row(tally: Counter, forms: list[dict]) -> None:
    for form in forms:
        tally.update(bounds(form))


__all__ = ["bounds", "draw_parts", "groups_for", "held", "lower_share", "place", "placement", "tally_row", "wanted"]
