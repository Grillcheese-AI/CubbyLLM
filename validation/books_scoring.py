"""books_scoring -- how the books bench (order 0012) scores an answer against the world's values.

`agree(rel, value, gold, within)`: 'agree' (the same: a date's year; one name's words in the other's; a spelling of
it, 'Wiewerd' / 'Wieuwerd'), 'finer' (a place inside the world's: Carthage within the Vandal Kingdom -- agreement,
Cowork's review), 'coarser' (a place that holds the world's: 'Rutland' for Oakham -- right only when hedged), 'other'
(another value of a relation with several), 'contradict'; None when nothing compares. `gazetteer` gives `within`:
the wiki world's LOCATED_IN / PART_OF and its capitals (a capital lies in its state), and the hdc world's country,
located-in, part-of and capital, up to 3 steps.
"""
from __future__ import annotations

import collections
import difflib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import wikikg as wk  # noqa: E402
from books import norm  # noqa: E402

DATES = {"date of birth", "date of death", "inception", "point in time"}


def year(v: str):
    m = re.search(r"(-?\d{3,4})(?:-\d\d)?(?:-\d\d)?\s*(BC|BCE)?", str(v))
    return None if not m else (-int(m.group(1)) if m.group(2) else int(m.group(1)))


MANY = {"founded by", "location", "part of"}


def agree(rel: str, value: str, gold: set[str], within=None) -> str | None:
    """Against the world's values: 'agree' (the same: a year; one name's words in the other's; a spelling of it,
    'Wiewerd' / 'Wieuwerd'), 'coarser' (a place that holds the world's: 'Rutland' for Oakham, `within`),
    'other' (another value of a relation with several: a co-founder), 'contradict'; None when nothing compares."""
    if rel in DATES:
        ys = {year(g) for g in gold} - {None}
        return None if not ys or year(value) is None else "agree" if year(value) in ys else "contradict"
    v = set(norm(value).split()) - {"the"}
    gs = [g for g in (set(norm(g).split()) - {"the"} for g in gold) if g]
    if not gs or not v:
        return None
    if any(v <= g or g <= v for g in gs):
        return "agree"
    sv = " ".join(sorted(v))
    if any(difflib.SequenceMatcher(None, sv, " ".join(sorted(g))).ratio() >= 0.8 or
           difflib.SequenceMatcher(None, norm(value).split()[0], sorted(g, key=len)[-1]).ratio() >= 0.85 for g in gs):
        return "agree"
    if any(len(w) >= 6 and any(w in x or x in w for x in g if len(x) >= 6) for w in v for g in gs):
        return "agree"                                   # 'Borgo San Sepolcro' / 'Sansepolcro': one name, two spellings
    if within is not None and any(within(norm(g), norm(value)) for g in gold):
        return "finer"
    if within is not None and any(within(norm(value), norm(g)) for g in gold):
        return "coarser"
    return "other" if rel in MANY else "contradict"


def gazetteer(hdc_path: str | None = None):
    """`within(big, small)`: does the place `big` hold `small`, by the wiki world's LOCATED_IN / PART_OF facts (and the
    hdc world's 'country' / 'located in ...' when given), up to 3 steps."""
    up = collections.defaultdict(set)
    for s, r, o in wk.load_triples():
        if r in ("LOCATED_IN", "PART_OF", "CAPITAL_OF"):
            up[norm(wk.decamel(s))].add(norm(wk.decamel(o)))
    db = None
    if hdc_path and os.path.exists(hdc_path):
        import sqlite3
        db = sqlite3.connect(f"file:{hdc_path}?mode=ro", uri=True)

    def parents(x):
        out = set(up.get(x, ()))
        if db is not None:
            out |= {o for (o,) in db.execute("SELECT okey FROM facts WHERE skey = ? AND rel IN ('country', "
                                             "'located in the administrative territorial entity', 'part of')", (x,))}
            out |= {o for (o,) in db.execute("SELECT skey FROM facts WHERE okey = ? AND rel = 'capital'", (x,))}
        return out

    def within(big, small):
        seen, frontier = {small}, {small}
        for _ in range(3):
            frontier = {p for x in frontier for p in parents(x)} - seen
            if big in frontier or any(big in f.split() and len(big) > 3 and f.startswith(big) for f in frontier):
                return True
            seen |= frontier
        return False
    return within
