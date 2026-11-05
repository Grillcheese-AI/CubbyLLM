"""r4_valuetype -- order 0046, R4 rule 3 as Cowork read it (2026-11-04, evening): keep the serving path's exemption, and
check every spoken value against the property's own value-type constraint, as Wikidata states it.

  the constraint  the property's P2302 statements whose value is Q21510865 ("value-type constraint"): the classes
                  (qualifier P2308) and the relation (P2309: Q21503252 "instance of", Q21514624 "subclass of",
                  Q30208840 "instance or subclass of"); deprecated statements left out
  the check       a value passes when a class is reached: from its ``instance of`` (P31) and/or the value itself (as
                  the relation says), then up ``subclass of`` (P279), at most UP steps, every step a statement the
                  source makes (read through the source's cache)
  no constraint   a property without one: nothing to check, the values stand
  the direction   a value read from the statements naming the item (``in``: CronQuestions' incoming rows) is the
                  statement's subject, so it is checked against the property's subject-type constraint (Q21503250),
                  the same way; the value-type constraint is the other end's (the first dev reading checked these
                  against it and refused every incoming P39 row: the holders of a post are people, not posts)
A value that reaches no class is dropped; none left: refused ("value_type").

Wired: STANDALONE (R4's bench).
"""
from __future__ import annotations

import re

__wiring__ = "STANDALONE"

VALUE_TYPE = "Q21510865"
SUBJECT_TYPE = "Q21503250"
INSTANCE, SUBCLASS, EITHER = "Q21503252", "Q21514624", "Q30208840"
UP = 8
_Q = re.compile(r"Q\d+")


def _ids(claims: dict, pid: str) -> list[str]:
    out = []
    for st in claims.get(pid) or ():
        if st.get("rank") == "deprecated":
            continue
        v = ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if isinstance(v, dict) and v.get("id"):
            out.append(v["id"])
    return out


def constraint(src, pid: str, cache: dict, kind: str = VALUE_TYPE) -> tuple[set, str] | None:
    """(classes, relation) of the property's value-type (or subject-type) constraint, or None when it states none."""
    key = (kind, pid)
    if key not in cache:
        classes, rel = set(), None
        for st in src._claims(pid).get("P2302") or ():
            if st.get("rank") == "deprecated":
                continue
            if ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id") != kind:
                continue
            q = st.get("qualifiers") or {}
            classes |= {x["datavalue"]["value"]["id"] for x in q.get("P2308", []) if x.get("datavalue")}
            rels = [x["datavalue"]["value"]["id"] for x in q.get("P2309", []) if x.get("datavalue")]
            rel = rel or (rels[0] if rels else INSTANCE)
        cache[key] = (classes, rel) if classes else None
    return cache[key]


def reaches(src, q: str, classes: set, rel: str, cache: dict) -> bool:
    """Whether item q reaches one of `classes` as `rel` says, up ``subclass of`` at most UP steps."""
    key = ("reach", q, frozenset(classes), rel)
    if key in cache:
        return cache[key]
    start = set()
    if rel in (INSTANCE, EITHER):
        start |= set(_ids(src._claims(q), "P31"))
    if rel in (SUBCLASS, EITHER):
        start.add(q)
    seen, frontier, ok = set(start), set(start), bool(start & classes)
    for _ in range(UP):
        if ok or not frontier:
            break
        nxt = set()
        for c in frontier:
            nxt |= set(_ids(src._claims(c), "P279"))
        nxt -= seen
        seen |= nxt
        frontier = nxt
        ok = bool(nxt & classes)
    cache[key] = ok
    return ok


def check(src, pid: str, res: dict, cache: dict, direction: str = "out") -> dict:
    """Rule 3's value-type reading on one arm's result (``direction`` "in": the values are the statements' subjects)."""
    sp = res.get("spoken")
    if not sp:
        return res
    c = constraint(src, pid, cache, SUBJECT_TYPE if direction == "in" else VALUE_TYPE)
    if c is None:
        return {**res, "value_type": "no constraint"}
    items = [x for x in sp if isinstance(x, str) and _Q.fullmatch(x)]
    if len(items) < len(sp):
        return {**res, "value_type": "a literal among the values"}
    keep = [x for x in sp if reaches(src, x, c[0], c[1], cache)]
    if len(keep) == len(sp):
        return {**res, "value_type": "every value of the type"}
    if not keep:
        return {"spoken": None, "why": "value_type: no value of the property's type", "type_dropped": sp}
    return {**res, "spoken": keep, "type_dropped": sorted(set(sp) - set(keep))}


__all__ = ["check", "constraint", "reaches"]
