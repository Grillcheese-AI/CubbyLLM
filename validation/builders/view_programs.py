"""Order 0067, the second reader's step 2: checks over the reply's views (`cubbyllm/speech/reply_views.py`), drawn as
specs a compiler turns into CubeLang. Rows are built in reverse: a spec is drawn, then run; no model writes a check.

**The spec language and its evaluator live in the core** since order 0073's step 1 (`cubbyllm/speech/view_spec.py`:
`FILTERS`, `cmp`, `evaluate`, `atoms`), so the host and these builders use one implementation. This module keeps
what draws a spec, and re-exports the core's names under the old ones for one cycle.

**Drawn from the replies.** `sample(rng, views)` draws a spec's values and thresholds from the views of a pool of
replies built for it (`view_replies.py`), so the spec holds for some and fails for others (at least a fifth each).

**Held out** (`held_out`): forms that state the line-3 test kinds (IFEval's held-out kinds; IFBench's kinds the
grammar can state) are never drawn: no commas or brackets, a word absent, the number of list items, a paragraph's first
word, capitalised words counted, the reply's or a sentence's last word, punctuation counted, every word title-cased, a
word at a given position, indents and list levels (not filter columns at all).
"""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cubbyllm.speech.view_spec import FILTERS, NUM_OPS, STR_OPS, atoms, cmp, evaluate  # noqa: E402
from cubbyllm.speech.view_spec import match as _match  # noqa: E402

__wiring__ = "STANDALONE"

VIEW_WEIGHTS = {"char": 1, "word": 4, "sentence": 3, "line": 2, "paragraph": 2, "item": 2, "json": 1}


def held_out(spec: dict) -> str | None:
    """The line-3 test kind an atom of `spec` states, or None."""
    for a in atoms(spec):
        conds, v = a.get("where", []), a["view"]
        cv = {(c["col"], c["v"]) for c in conds}
        if v == "char" and ({("char", ","), ("char", "["), ("char", "]")} & cv or ("kind", "punct") in cv):
            return "commas, placeholders or punctuation counted (IFEval no_comma / placeholders; IFBench punctuation)"
        if v == "word" and any(c["col"] == "lower" for c in conds) and (
                a["q"] == "none" or (a["q"] == "count" and ((a["op"] in ("==", "<=") and a["v"] == 0) or (a["op"] == "<" and a["v"] <= 1)))):
            return "a word absent (IFEval forbidden_words)"
        if v == "item" and a["q"] in ("n", "count"):
            return "the number of list items (IFEval number_bullet_lists)"
        if a["q"] == "at" and v == "paragraph" and a["col"] == "first_word":
            return "a paragraph's first word (IFEval nth_paragraph_first_word)"
        if v == "word" and ("case", "upper") in cv and a["q"] == "count":
            return "capitalised words counted (IFEval capital_word_frequency)"
        if a["q"] == "at" and a["pos"] == -1 and (v == "word" or (v == "sentence" and a["col"] == "last_word")):
            return "the last word (IFEval end_checker)"
        if v == "word" and ("case", "title") in cv and a["q"] == "all":
            return "every word title-cased (IFBench title_case)"
        if a["q"] == "at" and v == "word":
            return "a word at a position (IFBench keywords_specific_position)"
    return None


def _values(views: list[dict], view: str, col: str) -> list:
    """The values a column takes, the ones found in a fifth to four fifths of the replies first."""
    seen = Counter()
    for vs in views:                                      # no value a person would not write in one line
        seen.update({r[col] for r in vs[view] if not isinstance(r[col], str) or (r[col].strip() and not
                                                                                 re.search(r"[\n\r\t]", r[col]))})
    n = max(1, len(views))
    order = sorted(seen, key=lambda x: (str(type(x)), x))  # set order follows the process's string hashing: sorted,
    mid = [x for x in order if 0.2 <= seen[x] / n <= 0.8]   # the same seed draws the same programs in every run
    return mid or order


def _cond(rng, views, view) -> dict | None:
    col, typ = rng.choice(sorted(FILTERS[view].items()))
    vals = _values(views, view, col)
    if not vals:
        return None
    return {"col": col, "op": rng.choice(NUM_OPS if typ == "num" else STR_OPS if rng.random() < 0.3 else ("==",)),
            "v": rng.choice(vals)}


def _mixed(spec, views, lo=0.2) -> bool:
    k = sum(1 for vs in views if evaluate(spec, vs))
    return lo * len(views) <= k <= (1 - lo) * len(views)


def sample_atom(rng: random.Random, views: list[dict], tries: int = 40) -> dict | None:
    names = list(VIEW_WEIGHTS)
    for _ in range(tries):
        view = rng.choices(names, weights=[VIEW_WEIGHTS[x] for x in names])[0]
        q = rng.choice(["count", "count", "all", "any", "none", "n", "at"])
        if q == "n":
            stats = [len(vs[view]) for vs in views]
            atom = {"q": "n", "view": view, "op": rng.choice(NUM_OPS), "v": rng.choice(stats)}
        elif q == "at":
            col, typ = rng.choice(sorted(FILTERS[view].items()))
            pos = rng.choice([1, 2, 3, -1])
            at = [vs[view][pos - 1 if pos > 0 else -1][col] for vs in views
                  if (len(vs[view]) >= pos if pos > 0 else vs[view])]
            at = [x for x in at if not isinstance(x, str) or (x.strip() and not re.search(r"[\n\r\t]", x))]
            if not at:
                continue
            atom = {"q": "at", "view": view, "pos": pos, "col": col,
                    "op": rng.choice(NUM_OPS if typ == "num" else STR_OPS), "val": rng.choice(at)}
        else:
            conds = [c for c in (_cond(rng, views, view) for _ in range(2 if rng.random() < 0.25 else 1)) if c]
            if not conds:
                continue
            atom = {"q": q, "view": view, "where": conds}
            if q == "count":
                stats = [sum(1 for r in vs[view] if _match(r, conds)) for vs in views]
                atom.update(op=rng.choice(NUM_OPS), v=rng.choice(stats))
        if not held_out(atom) and _mixed(atom, views, 0.1):
            return atom
    return None


def sample(rng: random.Random, views: list[dict], tries: int = 30) -> dict | None:
    """A spec of one or two atoms, holding for between a fifth and four fifths of the pool's replies."""
    for _ in range(tries):
        a = sample_atom(rng, views)
        if a is None:
            continue
        r = rng.random()
        if r < 0.45:
            spec = a
        elif r < 0.55:
            spec = {"not": a}
        else:
            b = sample_atom(rng, views)
            if b is None:
                continue
            spec = rng.choice([{"and": [a, b]}, {"or": [a, b]}, {"if": a, "then": b}, {"unless": [a, b]}])
        if not held_out(spec) and _mixed(spec, views):
            return spec
    return None


__all__ = ["FILTERS", "atoms", "cmp", "evaluate", "held_out", "sample", "sample_atom"]
