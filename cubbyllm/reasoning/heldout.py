"""cubbyllm/reasoning/heldout.py -- order 0047's guarantees, found by 0046's R0 on held-out questions (six wrong answers
spoken on today's serving path), and their switches.

- ``CB_HELDOUT_FIXES=1``: the name kept as asked (``asked_span`` here; the source's side is
  ``sources.wikidata_asked``), a quantity said with its unit (``sources.wikidata_units``), and an answer of the class
  the question asks or a ``class_mismatch`` refusal (``ask_class``).
- ``CB_CLASS_WALK=1`` (with the fixes on): a value of the wrong class is walked up ``located in`` to the first item of
  the class asked ("Dallas, where Parkland Memorial Hospital is"), instead of refused.

Off (the default until the order's acceptance holds), every answer is bit-identical to before.

Wired: WIRED (``ask.AskLoop._ask``, ``ask_memory.AskMemory._checked``, ``sources.wikidata``).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

# the articles a name can begin with, English and French; "du"/"des" are contractions of "de" + an article, so the
# name after them carries no article of its own ("du Canada" is "Canada")
_ART = re.compile(r"(?:(?<![\w'’])(the|le|la|les)\s+|(?<![\w'’])(l)['’]\s*)$", re.I)


def on() -> bool:
    return os.environ.get("CB_HELDOUT_FIXES") == "1"


def walk_on() -> bool:
    return on() and os.environ.get("CB_CLASS_WALK") == "1"


def asked_span(question: str, seed: str) -> str | None:
    """The name with the article the asker wrote before it ("The Last Dragon" where the seed is "Last Dragon";
    "l'interprétation des rêves" where it is "interprétation des rêves"), or None when no article stands before it."""
    q, s = question or "", (seed or "").strip()
    if not s:
        return None
    i = q.lower().find(s.lower())
    while i >= 0:
        m = _ART.search(q[:i])
        if m:
            return q[m.start():i + len(s)]
        i = q.lower().find(s.lower(), i + 1)
    return None


__all__ = ["asked_span", "on", "walk_on"]
