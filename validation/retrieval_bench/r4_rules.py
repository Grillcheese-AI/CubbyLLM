"""r4_rules -- order 0046, R4: Cowork's four rules after dev (2026-11-04, evening), read once on CronQuestions' test.

1. ended   An open end is placed only if nothing on record says it ended. A value with no stated end holds at a year
           (or now) only if:
           - the subject's own end (death P570, dissolution P576, end time P582) is not before that year;
           - the value item's own end (the same properties) is not before it;
           - where the property holds one value at a time (a single-value constraint, or a "replaced by" P1366
             qualifier on the statement), no later value (a start after this one's) starts before it.
           "No stated end" never means "still holds". Both items are checked whichever way the statement points (the
           asked item and the spoken one), so the rule reads the same for "who held X" and "what did X hold".
2. bound   A spoken set has at most BOUND values. Above that: the count, and ASK to narrow it (judged "asked": not
           spoken, not wrong). The share of rows that hit the bound is reported.
3. class   The class asked applies as on the serving path (0047's check, ``cubbyllm.reasoning.ask_class``): a
           question that names its answer's kind ("what country") keeps only the values whose P31, or one ``subclass
           of`` step above it, is a class so named. ``exempt``: the serving path's exemption (the class word the
           everyday table reads as the relation asked: "what nationality", "what language") is kept or not.
4. merges  The gold ids are resolved through Wikidata's merge redirects before judging; a different item with the same
           label stays wrong.

Order of application: 1 (placement), 3 (class), 2 (the bound) -- the count is of what would be spoken.

Wired: STANDALONE (R4's bench).
"""
from __future__ import annotations

import re

__wiring__ = "STANDALONE"

BOUND = 10
END_PROPS = ("P570", "P576", "P582")
_Q = re.compile(r"Q\d+")


def _y(t: dict | None) -> int | None:
    return t["y"] if t else None


def why_ended(v: dict, vals: list[dict], year: int, asked: str, ends: dict, single: bool) -> str | None:
    """Why value `v` (no stated end) cannot be placed at `year` under rule 1, or None if nothing says it ended."""
    if v.get("end") or v.get("point"):
        return None
    for who, q in (("the asked item", asked), ("the value", v.get("v"))):
        e = ends.get(q)
        if e == "unread":
            return f"{who}'s end could not be read"
        if e is not None and e < year:
            return f"{who} ended in {e}"
    s = _y(v.get("start"))
    if s is not None and (single or v.get("replaced")):
        later = [_y(w.get("start")) for w in vals if w is not v and w.get("v") != v.get("v") and w.get("start")]
        nxt = [x for x in later if s < x < year]
        if nxt:
            return f"a later value started in {min(nxt)}"
    return None


def drop_ended(vals: list[dict] | None, year: int | None, asked: str, ends: dict, single: bool) -> tuple[list, list]:
    """(the values rule 1 lets stand at `year`, [(value, why)] dropped). No year: unchanged."""
    if not vals or year is None:
        return list(vals or ()), []
    keep, dropped = [], []
    for v in vals:
        why = why_ended(v, vals, year, asked, ends, single)
        (dropped.append((v.get("v"), why)) if why else keep.append(v))
    return keep, dropped


def ended_ids(vals: list[dict] | None, asked: str) -> list[str]:
    """The items whose own end rule 1 reads: the asked item and every item value with no stated end."""
    return [asked] + sorted({v["v"] for v in vals or () if isinstance(v.get("v"), str) and _Q.fullmatch(v["v"])
                             and not v.get("end") and not v.get("point")})


def bound(res: dict) -> dict:
    """Rule 2: more than BOUND values -> the count and ASK, nothing spoken."""
    sp = res.get("spoken")
    if sp and len(sp) > BOUND:
        return {"spoken": None, "ask": len(sp), "why": f"{len(sp)} values: the count and ASK"}
    return res


def class_filter(src, question: str, res: dict, plan_relations: set | None = None, exempt: bool = True,
                 cache: dict | None = None) -> dict:
    """Rule 3: keep the spoken items of the class the question names; none left -> refused (class_mismatch)."""
    from cubbyllm.reasoning import ask_class as C
    sp = res.get("spoken")
    cw = C.class_word(question)
    if not sp or not cw:
        return res
    if exempt and plan_relations and {r.lower() for r in plan_relations} & {r.lower() for r in _table().relations(cw[0])}:
        return {**res, "class": "the class word is the relation asked"}
    cache = {} if cache is None else cache
    key = ("cls", cw)
    if key not in cache:
        cache[key] = C.classes(src, cw[0], cw[1])
    cls = cache[key]
    if not cls:
        return {**res, "class": f"no class named {cw[0]!r}"}
    items = [x for x in sp if isinstance(x, str) and _Q.fullmatch(x)]
    if len(items) < len(sp):
        return {**res, "class": "a literal among the values: nothing to check"}
    keep = [x for x in sp if C.is_a(src, x, cls)]
    if len(keep) == len(sp):
        return {**res, "class": "every value of the class"}
    if not keep:
        return {"spoken": None, "why": f"class_mismatch: none is a {cw[0]}", "class_dropped": sp}
    return {"spoken": keep, "why": res.get("why", "") + f"; of the class {cw[0]}",
            "class_dropped": sorted(set(sp) - set(keep))}


_TABLE = []


def _table():
    if not _TABLE:
        from cubbyllm.reasoning.everyday import EverydayForms
        _TABLE.append(EverydayForms())
    return _TABLE[0]


def merged(src, ids) -> dict:
    """Rule 4: {id: the id it redirects to} for the ids Wikidata merged (``wbgetentities`` reports the target)."""
    from retrieval_sets.common import entities
    out = {}
    for q, e in entities(src, sorted(set(ids)), props="info", langs="en").items():
        to = e.get("id") or (e.get("redirects") or {}).get("to")
        if to and to != q:
            out[q] = to
    return out


def gold_through_merges(gold: set, merges: dict) -> set:
    return set(gold) | {merges[g] for g in gold if g in merges}


__all__ = ["BOUND", "bound", "class_filter", "drop_ended", "ended_ids", "gold_through_merges", "merged", "why_ended"]
