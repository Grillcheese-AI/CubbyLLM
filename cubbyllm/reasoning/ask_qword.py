"""cubbyllm/reasoning/ask_word_fits.py -- the answer fits the question word (order 0054, Cowork's proposal 1 from the
read of the "after", approved by Nick 2026-11-11), behind ``CB_ANSWER_FITS=1`` (off by default; off, nothing changes).

"Where was the remake of Dirty Dancing filmed?" was answered "Dirty Dancing: Havana Nights": the plan read another
relation (derivative work) and the VM certified that fact faithfully. The question word says what kind of thing the
answer is, whatever the plan read:
- **where**: a place -- the value's item states coordinates, a "located in", a country, a terrain feature or a
  continent;
- **who** (whom, whose): a person (a human, a sex or gender, a birth or death date) or an organisation (headquarters,
  founder, industry, legal form, parent, chair, CEO, employees, members, a league, a coach, a venue, a record label);
- **when**: a date -- "1952", "March 1952", "11 March 1952", "500 BC", an ISO date.
Otherwise the answer is refused (`word_mismatch`), naming what was found. It is the class check that met the Marie
Curie test (`answer_values`), with the class taken from the question word. What the value's item is comes from the
world that handed the fact out (`wd_store_names.Named.value_kinds`, the hot Wikidata store); a value that is no item
there is left as it was (nothing to check against), unless it is a date or a number answering "where" or "who".

Wired: WIRED (``ask_memory.AskMemory._checked``, after the class check).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

_WORD = re.compile(r"^\s*(?:and\s+|so\s+)?(where|who|whom|whose|when)\b", re.I)
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATE = re.compile(rf"^(?:(?:\d{{1,2}}\s+)?{_MONTH}\s+)?\d{{1,4}}(?:\s*BC)?$|^-?\d{{1,4}}-\d\d(?:-\d\d)?$", re.I)
NUMBER = re.compile(r"^[+-]?[\d.,]+(?:\s+\S+)?$")
WANT = {"where": {"place"}, "who": {"person", "organisation"}}


def on() -> bool:
    return os.environ.get("CB_ANSWER_FITS") == "1"


def asked(question: str) -> str | None:
    """'where' | 'who' | 'when' the question opens with, or None."""
    m = _WORD.match(question or "")
    if not m:
        return None
    w = m.group(1).lower()
    return "who" if w in ("whom", "whose") else w


def _kinds(world, triple) -> set | None:
    for w in [world, *getattr(world, "others", [])]:
        fn = getattr(w, "value_kinds", None)
        if fn is not None:
            got = fn(triple)
            if got is not None:
                return got
    return None


def fits(word: str, value: str, kinds: set | None) -> bool | None:
    """True / False when the value is / is not of the word's kind; None when there is nothing to check against."""
    v = " ".join(str(value or "").split())
    if word == "when":
        return bool(DATE.match(v))
    if kinds is None:
        return False if (DATE.match(v) or NUMBER.match(v)) else None
    return bool(kinds & WANT[word])


def check(loop, question: str, lr, rec: dict) -> bool:
    """True when the answer stands (or there is nothing to check); False when `rec` now refuses."""
    word = asked(question) if on() else None
    h = lr.result.trace[-1] if lr.result.trace else None
    if word is None or h is None or h.triple is None:
        return True
    value = str(rec.get("answer") or "")
    kinds = None if word == "when" else _kinds(getattr(loop, "world", None), h.triple)
    ok = fits(word, value, kinds)
    if ok is None or ok:
        return True
    what = {"where": "a place", "who": "a person or an organisation", "when": "a date"}[word]
    rec.update(answer=None, verified=False, reason="word_mismatch",
               refused={"answer": value, "why": f"the question asks '{word}', and {value} is not {what}"})
    return False


__all__ = ["asked", "check", "fits", "on"]
