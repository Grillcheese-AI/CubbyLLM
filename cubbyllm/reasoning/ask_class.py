"""cubbyllm/reasoning/ask_class.py -- an answer of the class the question asks (order 0047, item 2;
``CB_HELDOUT_FIXES``, on by default; ``=0`` off).

0046's R0: "In which city did John F. Kennedy die?" was answered "Parkland Memorial Hospital" -- his place of death,
a hospital, not a city. When the question names the kind of its answer ("which city", "what country", "quelle
ville", "quel pays"):
- the class word is read through Wikidata's own classes: the items labelled or aliased exactly so, in the question's
  language, that are classes (they have a ``subclass of``, P279) -- data, not a hand list;
- the value is spoken only if it is an instance of one: its P31, or one ``subclass of`` step above it (as
  ``sources.wikidata_namesakes`` reads kinds);
- otherwise ``class_mismatch``, naming what was found ("Parkland Memorial Hospital, a hospital");
- with ``CB_CLASS_WALK=1``: the value is walked up ``located in`` (P131), at most three steps, to the first item of
  the class asked, each step a statement the source makes ("Dallas, where Parkland Memorial Hospital is").

The value's item: the one the source linked the fact's object to, else the items labelled exactly so. A date or a
number asked by a "year" or "date" is checked to be one; a class word the source has no class for, and a value no
item carries, are left as they were (nothing to check against). A class word the reader's own table reads as the
relation the plan asks ("what nationality" -> country of citizenship, "what language" -> official language) is the
relation, not a class: the relation already names the answer's kind, and no check is made (the scoreboard, 2026-10-31:
Turkish and Icelandic are a "language" two ``subclass of`` steps up, and a country is no "nationality").

Wired: WIRED (``ask_memory.AskMemory._checked``).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"

_EN = re.compile(r"^\s*(?:in|at|on|from|for|of|to|by|into)?\s*(?:which|what)\s+(?P<c>[a-z][a-z' -]*?)\s+"
                 r"(?:did|does|do|is|was|were|are|has|have|had|can|could|will|would)\b", re.I)
_FR = re.compile(r"^\s*(?:dans|de|en|à|a|pour|par|sur)?\s*(?:quel|quelle|quels|quelles)\s+(?P<c>[\w'-]+)", re.I)
META = frozenset("kind type sort form name way part number amount genre de".split())       # words about a class, not one
DATES = frozenset("year date day month century décennie année date jour mois siècle".split())
_DATE = re.compile(r"^-?\d{1,4}(?:-\d\d){0,2}(?:\s*BC)?$")
UP = 3


def class_word(question: str) -> tuple[str, str] | None:
    """(the class phrase, its language) the question names for its answer, or None."""
    for rx, lang in ((_EN, "en"), (_FR, "fr")):
        m = rx.match(question or "")
        if m:
            c = " ".join(m.group("c").lower().split())
            words = c.split()
            if words and words[0] not in META and words[-1] not in META:      # "kind of government": about kinds
                return c, lang
    return None


def _ids(claims: dict, pid: str) -> list[str]:
    return [((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
            for st in claims.get(pid, []) if ((st.get("mainsnak") or {}).get("datavalue") or {}).get("type") == "wikibase-entityid"]


def classes(src, phrase: str, lang: str) -> set:
    """The class items the phrase names: exact label or alias in `lang`, with a ``subclass of`` of their own."""
    from ..sources.wikidata_asked import _same
    s = src._get({"action": "wbsearchentities", "search": phrase, "language": lang, "uselang": lang, "type": "item",
                  "limit": 10}) or {}
    hits = [h["id"] for h in s.get("search") or []
            if _same(h.get("label"), phrase) or _same((h.get("match") or {}).get("text"), phrase)]
    return {q for q in hits if _ids(src._claims(q), "P279")}


def is_a(src, qid: str, cls: set) -> bool:
    kinds = set(_ids(src._claims(qid), "P31"))
    up = {u for k in kinds for u in _ids(src._claims(k), "P279")}
    return bool((kinds | up) & cls)


def value_items(loop, fact: str, value: str) -> list[str]:
    from .planner import normalize
    src = loop.source
    linked = list((getattr(src, "links", None) or {}).get(normalize(fact), ()))
    if linked:
        return linked
    s = src._get({"action": "wbsearchentities", "search": value, "language": "en", "limit": 10}) or {}
    return [h["id"] for h in s.get("search") or [] if normalize(h.get("label") or "") == normalize(value)]


def walk_up(loop, start: str, cls: set) -> list[tuple[str, str]] | None:
    """[(item, label)] from `start` up ``located in`` to the first item of the class, or None."""
    src, path, cur = loop.source, [], start
    for _ in range(UP):
        nxt = _ids(src._claims(cur), "P131")
        if not nxt:
            return None
        cur = nxt[0]
        path.append((cur, src._labels_for([cur]).get(cur) or cur))
        if is_a(src, cur, cls):
            return path
    return None


def check(loop, question: str, lr, rec: dict) -> bool:
    """True when the answer stands (or there is nothing to check); False when `rec` now refuses (or walked)."""
    from .heldout import on, walk_on
    src = getattr(loop, "source", None)
    cw = class_word(question) if on() else None
    h = lr.result.trace[-1] if lr.result.trace else None
    if not cw or h is None or h.triple is None or not hasattr(src, "_claims"):
        return True
    table = getattr(loop, "_everyday", None)
    if table is not None and set(rec.get("plan") or ()) & set(table.relations(cw[0])):
        return True                     # the class word IS the relation asked ("what nationality", "what language"):
                                        # the relation already names the answer's kind (0047's scoreboard run)
    value = str(rec.get("answer") or "")
    if cw[0].split()[-1] in DATES:
        if _DATE.match(value.strip()):
            return True
        rec.update(answer=None, verified=False, reason="class_mismatch",
                   refused={"answer": value, "why": f"the question asks for a {cw[0]}, and {value} is not one"})
        return False
    cls = classes(src, cw[0], cw[1])
    items = value_items(loop, h.fact, value) if cls else []
    if not cls or not items or any(is_a(src, q, cls) for q in items):
        return True
    kinds = [src._labels_for([k]).get(k) for k in _ids(src._claims(items[0]), "P31")][:1]
    if walk_on():
        path = walk_up(loop, items[0], cls)
        if path:
            subj = value
            facts = []
            for q, lab in path:
                facts.append(f"{lab} is the located in the administrative territorial entity of {subj}")
                subj = lab
            rec.update(answer=path[-1][1], reply=f"{path[-1][1]}, where {value} is.", trace=list(rec.get("trace") or []) + facts,
                       class_walk={"from": value, "path": [q for q, _l in path], "class": cw[0]})
            return False
    rec.update(answer=None, verified=False, reason="class_mismatch",
               refused={"answer": value, "why": f"the question asks for a {cw[0]}; {value} is "
                        + (f"a {kinds[0]}" if kinds and kinds[0] else "not one")})
    return False


__all__ = ["check", "class_word", "classes", "is_a", "walk_up"]
