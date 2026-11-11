"""cubbyllm/reasoning/ask_qword.py -- order 0066, check 1a: the answer is of the kind the question word asks, behind
``CB_QWORD_KIND=1`` (off by default; off, nothing changes). An extension of `ask_class` (order 0047): read from there
when the question names no class, with the same `class_mismatch` refusal.

"Where was the remake of Dirty Dancing filmed?" was answered "Dirty Dancing: Havana Nights": the plan read another
relation, and the VM certified that fact faithfully. The word that opens the question (after an optional
preposition) says what kind of thing the answer is:
- **where / où**: a place;
- **who, whom, whose / qui**: a person, a group of people, an organisation or a fictional character;
- **when / quand**: a date.
The kind is read from the store, never from the label's words (`wd_store_names.Named.value_kind`: a time value is a
date; the value's item is a place by coordinates or by a class under "geographic location"; a person, a group, an
organisation or a character by its class, through the class-roots table). For a fact another world gave, the item the
source linked its value to is read in the store the same way. A value whose kind no world holds (a plain string, an
item with no class, no link) is not checked; the record says so (`kind_unknown`), and the run counts it apart.

Check 1b, the head noun of a class the source has no class for ("what culture region" read as "region"), is in
`ask_class.check` itself.

Wired: WIRED (``ask_class.check``).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

_LEAD = re.compile(r"^\s*(?:(?:in|at|on|from|for|of|to|by|with|since|until|till|dans|de|depuis|en|à|a|pour|par|sur|"
                   r"avec|chez|jusqu'à|jusqu’à)\s+)?(where|who|whom|whose|when|où|qui|quand)\b", re.I)
WORD = {"where": "where", "où": "where", "who": "who", "whom": "who", "whose": "who", "qui": "who",
        "when": "when", "quand": "when"}
WANT = {"where": {"place"}, "who": {"person", "group", "organisation", "character"}, "when": {"time"}}
WHAT = {"where": "a place", "who": "a person, a group or an organisation", "when": "a date"}


def on() -> bool:
    return os.environ.get("CB_QWORD_KIND") == "1"


def asked(question: str) -> str | None:
    """'where' | 'who' | 'when' the question opens with (English or French), or None."""
    m = _LEAD.match(question or "")
    return WORD[m.group(1).lower()] if m else None


def _stores(world) -> list:
    return [w for w in [world, *getattr(world, "others", [])] if hasattr(type(w), "value_kind")]


def store_item(loop, triple) -> str | None:
    """The item a store world linked this fact's value to, when it handed the fact out (else None). With
    ``CB_QWORD_KIND=1`` the class check reads the value through it, not through a label search: "Beatrice" by label
    is a name, a character and a dozen places, and the ten first hits may miss the one the fact is about."""
    from cubbyllm.worlds.wd_store_names import key3
    for w in _stores(getattr(loop, "world", None)):
        q = w._rec()["obj"].get(key3(triple.subj, triple.rel, triple.obj)) if triple is not None else None
        if q:
            return q
    return None


def store_is_a(loop, item: str, cls: set) -> bool | None:
    """`ask_class.is_a` read in the store that linked the value: the item's classes and one subclass-of step above
    them (the same depth as the class check's own rule), from the dump, not from live Wikidata, which may have been
    edited since ("city in the United States" is no longer a subclass of "city" there). None: no store holds it."""
    for w in _stores(getattr(loop, "world", None)):
        if w._entity(item)[0] is None and item not in w._p31:
            continue
        kinds = set(w._p31.get(item, set()))
        up = set()
        for k in kinds:
            e = w.ask({"op": "get", "id": k}) or {}
            up |= {s["value"]["entity"] for s in e.get("statements", []) if s.get("pid") == 279 and s.get("best")
                   and s.get("rank") != 0 and isinstance(s.get("value"), dict) and "entity" in s["value"]}
        return bool((kinds | up) & set(cls))
    return None


def kind_of(loop, fact: str, triple) -> set | None:
    """The kinds of the answer's value as the store reads them, or None (unknown)."""
    stores = _stores(getattr(loop, "world", None))
    for w in stores:
        got = w.value_kind(triple)
        if got is not None:
            return got
    from .planner import normalize
    linked = list(((getattr(getattr(loop, "source", None), "links", None) or {}).get(normalize(fact or ""))) or ())
    for w in stores:
        if linked and getattr(w, "_roots", None) is not None:
            out = set()
            for q in linked:
                out |= w.item_roots(q) - {"role", "work"}
                out |= {"place"} if 625 in w._rec()["pids"].get(q, set()) else set()
            return out
    return None


def check(loop, question: str, lr, rec: dict) -> bool | None:
    """None: not this check's question (no question word, or the flag off). True: the answer stands (or its kind is
    unknown, recorded). False: `rec` now refuses (`class_mismatch`)."""
    word = asked(question) if on() else None
    h = lr.result.trace[-1] if lr.result.trace else None
    if word is None or h is None or h.triple is None:
        return None
    value = str(rec.get("answer") or "")
    kinds = kind_of(loop, h.fact, h.triple)
    if kinds is None:
        rec["kind_unknown"] = word
        return True
    if kinds & WANT[word]:
        return True
    rec.update(answer=None, verified=False, reason="class_mismatch",
               refused={"answer": value, "why": f"the question asks '{word}', and {value} is not {WHAT[word]}"},
               qword=word)
    return False


__all__ = ["asked", "check", "kind_of", "on"]
