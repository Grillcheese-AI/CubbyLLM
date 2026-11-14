"""Order 0073, step 2: what a wording says. Each sentence `view_wording` and `view_wording_inv` write for a part has
one said form: the part in the spec language's slots as the words put it (`cubbyllm/speech/view_spec.py`, "said
forms"). It follows the wording's own branches, so it is the sentence's form and not the stored part with a flag:

- **Amounts as said** (`amount`'s own table): a count ``> 0`` or ``!= 0`` is "at least one", so the quantifier ``any``;
  ``< 1`` or ``<= 0`` is "no", so ``none``; any other count keeps its comparison and number. A view's size and a
  numeric column keep the comparison said ("at most 3" for ``not > 3``): numbers never carry a ``not``.
- **A ``not`` only where a verb is negated** ("not", "do not", "ne ... pas", "n'... pas"): on a condition ("that does
  not begin with"), or on the quantifier ("Do not include any", "Do not use", "N'utilise pas"). "No" and "aucun" are
  the quantifier ``none``; "without" and "other than" are the comparison said.
- **"if there is one"** is ``absent: pass`` on ``at``.
- A list line is said of ``item == "yes"`` ("is part of a list"), negated or not.

`said(part, lang, how)` is arm W's form (None: the wording has no description, ``not all`` of two conditions; or
`how` has no such wording); arm M's is `view_spec.fold` of it. ``how``: ``"imp"`` (`say` / `say_not`), ``"cond"``
(`say_cond`, a plain atom), ``"inverted"`` (`view_wording_inv.say_inverted`). `say(part, lang, rng, how)` returns the
sentence with both forms, drawn by the same call. `negations(text, lang)` counts the words that make a ``not``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
import view_wording as W  # noqa: E402
import view_wording_inv as WI  # noqa: E402

from cubbyllm.speech import view_spec as VS  # noqa: E402

__wiring__ = "STANDALONE"

NORM = {(">", 0): (">=", 1), ("!=", 0): (">=", 1), ("<", 1): ("==", 0), ("<=", 0): ("==", 0)}   # `view_wording.amount`


def _amount(op: str, n) -> tuple:
    return NORM.get((op, n), (op, n))


def cond(c: dict, neg: bool = False) -> dict:
    """A condition as `view_wording.pred(c, lang, neg)` says it."""
    col, op, v = c["col"], c["op"], c["v"]
    if col in W.NUMERIC:                                   # the complement is said: "at most", "no"
        o, n = _amount(VS.INV[op] if neg else op, v)
        return {"col": col, "op": o, "v": n}
    neg = neg != (op == "!=")
    if col == "end_mark" and v == "none":                  # "without closing punctuation" / "with closing punctuation"
        return {"col": col, "op": "!=" if neg else "==", "v": v}
    if col == "item":                                      # "is part of a list" / "is not part of a list"
        neg, v = neg != (v == "no"), "yes"
    out = {"col": col, "op": "==", "v": v}
    return dict(out, **{"not": True}) if neg else out


def _at(a: dict, neg: bool) -> dict:
    c = cond({"col": a["col"], "op": a["op"], "v": a["val"]}, neg)
    out = {"q": "at", "view": a["view"], "pos": a["pos"], "col": c["col"], "op": c["op"], "val": c["v"]}
    if c.get("not"):
        out["not"] = True
    return dict(out, absent="pass") if neg else out


def _quantified(q: str, a: dict, where: list, verb: bool) -> dict:
    """``count`` / ``any`` / ``none`` as said. ``verb``: "no" is said with a negated verb ("Do not include any")."""
    view = a["view"]
    if q == "count":
        op, v = _amount(a["op"], a["v"])
        if (op, v) not in ((">=", 1), ("==", 0)):
            return {"q": "count", "view": view, "op": op, "v": v, "where": where}
        q = "any" if op == ">=" else "none"
    if q == "any":
        return {"q": "any", "view": view, "where": where}
    return {"q": "any", "view": view, "where": where, "not": True} if verb else {"q": "none", "view": view, "where": where}


def _n(a: dict) -> dict:
    op, v = _amount(a["op"], a["v"])
    return {"q": "n", "view": a["view"], "op": op, "v": v}


def _imp(a: dict, lang: str) -> dict:
    """`view_wording.say(a)`."""
    q, view = a["q"], a["view"]
    if q == "n":
        return _n(a)
    if q == "at":
        return _at(a, False)
    conds = a["where"]
    where = [cond(c) for c in conds]
    one = conds[0] if len(conds) == 1 else None
    noun = bool(one and one["op"] == "==" and (one["col"] == "lower" or (view == "char" and W._char_noun(a, lang, False))))
    if q != "all":                                         # "Do not use <a noun>" in both languages; "Do not include any"
        return _quantified(q, a, where, noun or lang == "en")   # in English, "Ne mets aucun" in French
    if lang == "en":
        if len(where) == 1 and where[0].get("not"):         # "No sentence should begin with ..."
            return {"q": "none", "view": view, "where": [{k: v for k, v in where[0].items() if k != "not"}]}
        where = sorted(where, key=lambda c: bool(c.get("not")))   # the sentence's order: the plain ones first
    return {"q": "all", "view": view, "where": where}


def _imp_not(a: dict, lang: str) -> dict | None:
    """`view_wording.say_not(a)`."""
    q = a["q"]
    if q in ("count", "n"):
        return _imp(dict(a, op=VS.INV[a["op"]]), lang)
    if q in ("any", "none"):
        return _imp(dict(a, q="none" if q == "any" else "any"), lang)
    if q == "at":
        return _at(a, True)
    if len(a["where"]) != 1:                               # "... or ...": an or-filter, no description
        return None
    return {"q": "any", "view": a["view"], "where": [cond(a["where"][0], True)]}


def _cond_form(a: dict, lang: str) -> dict:
    """`view_wording.say_cond(a)`."""
    q, view = a["q"], a["view"]
    if q == "n":
        return _n(a)
    if q == "at":
        return _at(a, False)
    conds = a["where"]
    word = len(conds) == 1 and conds[0]["col"] == "lower" and conds[0]["op"] == "=="
    if q == "all":
        return {"q": "all", "view": view, "where": [cond(c) for c in conds]}
    return _quantified(q, a, [cond(c) for c in conds], word)   # "does not use the word"; else "has no ..."


def _inverted(part: dict, lang: str) -> dict | None:
    """`view_wording_inv.say_inverted(part)`."""
    a = WI.positive(part)
    if a is None:
        return None
    if a["q"] == "n":
        n = _n(a)
        return None if lang == "fr" and (n["op"], n["v"]) == ("==", 0) else n
    conds = a.get("where") or []
    if len(conds) == 1 and conds[0]["op"] == "==" and (conds[0]["col"] == "lower" or a["view"] == "char"):
        return None
    return _quantified(a["q"], a, [cond(c) for c in conds], False)   # "No ... should be included"


def said(part: dict, lang: str, how: str = "imp") -> dict | None:
    """Arm W's form of the sentence `how` writes for `part`, or None."""
    if how == "inverted":
        return _inverted(part, lang)
    if how == "cond":
        return _cond_form(part, lang)
    return _imp_not(part["not"], lang) if "not" in part else _imp(part, lang)


def say(part: dict, lang: str, rng, how: str = "imp") -> tuple:
    """(the sentence, arm W's form, arm M's form); the forms are None where the wording has no description."""
    if how == "inverted":
        text = WI.say_inverted(part, lang, rng)
    elif how == "cond":
        text = W.say_cond(part, lang)
    else:
        text = W.say_not(part["not"], lang, rng) if "not" in part else W.say(part, lang, rng)
    w = None if text is None else said(part, lang, how)
    return text, w, (None if w is None else VS.fold(w))


def negations(text: str, lang: str) -> int:
    """How many words of `text` make a ``not`` ("not"; "pas" after "ne" or "n'"), quoted values aside."""
    bare = re.sub(r'"(?:"|[^"]*)"|« (?:»|[^»]*?) »', " ", text)   # a value can be the quote mark itself
    return len(re.findall(r"\bnot\b" if lang == "en" else r"\bpas\b", bare))


def marks(form: dict) -> int:
    """How many ``not`` a said form carries."""
    return int(bool(form.get("not"))) + sum(1 for c in form.get("where", []) if c.get("not"))


__all__ = ["cond", "marks", "negations", "said", "say"]
