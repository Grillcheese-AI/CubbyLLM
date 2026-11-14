"""Order 0067, the view reader's round 2 (Nick, 2026-11-13: inverted phrasing, "trap" examples whose wording changes and
whose program does not): a check said goal first or in the passive, by hand-written templates over the same atom, so
the meaning is the plain wording's by construction (no model rewrites an instruction: a rewrite can change what it
says while the target stays, which teaches a wrong answer).

- `say_inverted(part, lang, rng)`: the part (an atom or ``{"not": atom}``) with its amount first: "Fewer than 48 words:
  that is the length to keep." / "At least 2 sentences that end with a question mark should be included." / "Moins de
  48 mots : voilà la longueur à tenir." / "Aucune phrase qui finit par un point ne doit figurer dans ta réponse.";
  None when the part has no such wording (a position, "every", a word or a character said as its own noun): the caller
  keeps the plain wording then.

Wiring: STANDALONE (`build_pair_rows.py`).
"""
from __future__ import annotations

import random

import view_wording as W

__wiring__ = "STANDALONE"


def positive(part: dict) -> dict | None:
    """The atom a part is (a ``{"not": a}`` carried into the atom), or None when it is not a counted one."""
    a = part.get("not", part)
    if "not" in part:
        if a["q"] in ("count", "n"):
            a = dict(a, op=W.INV[a["op"]])
        elif a["q"] in ("any", "none"):
            a = dict(a, q="none" if a["q"] == "any" else "any")
        else:
            return None
    return a if a["q"] in ("n", "count", "any", "none") else None


def _up(s: str) -> str:
    return s[:1].upper() + s[1:]


def say_inverted(part: dict, lang: str, rng: random.Random) -> str | None:
    a = positive(part)
    if a is None:
        return None
    en_sg, en_pl, fr_sg, fr_pl, g = W.NOUNS[a["view"]]
    sg, pl = (en_sg, en_pl) if lang == "en" else (fr_sg, fr_pl)
    if a["q"] == "n":
        what = W.counted(a["op"], a["v"], lang, sg, pl, g)
        if lang == "en":
            return rng.choice(["{}: that is the length to keep.", "{} in all: that is what your reply should have."]).format(_up(what))
        if what.startswith("aucun"):
            return None
        return rng.choice(["{} : voilà la longueur à tenir.", "{} en tout : voilà ce que ta réponse doit avoir."]).format(_up(what))
    conds = a.get("where") or []
    one = conds[0] if len(conds) == 1 else None
    if one and one["op"] == "==" and (one["col"] == "lower" or a["view"] == "char"):
        return None                                       # a word or a character said as its own noun: kept plain
    if a["q"] == "count" and W.amount(a["op"], a["v"], lang, g)[0] in ("no", "aucun", "aucune"):
        a = dict(a, q="none")
    if a["q"] == "count":
        ph, plural = W.amount(a["op"], a["v"], lang, g)
        plural = plural or (a["op"] == "!=" and a["v"] != 0)
        what = W.counted(a["op"], a["v"], lang, sg, pl, g)
    elif a["q"] == "any":
        plural, what = False, (f"at least one {sg}" if lang == "en" else f"au moins {'une' if g == 'f' else 'un'} {sg}")
    else:
        plural, what = False, (f"no {sg}" if lang == "en" else f"{'aucune' if g == 'f' else 'aucun'} {sg}")
    rels = (" and " if lang == "en" else " et ").join(W.rel(c, lang, plural) for c in conds)
    if lang == "en":
        return f"{_up(what)} {rels} should be included."           # "No sentence that ... should be included."
    if a["q"] == "none":
        return f"{_up(what)} {rels} ne doit figurer dans ta réponse."
    return f"{_up(what)} {rels} {'doivent' if plural else 'doit'} figurer dans ta réponse."


__all__ = ["positive", "say_inverted"]
