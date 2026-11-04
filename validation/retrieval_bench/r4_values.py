"""r4_values -- order 0046, R4: which of a relation's values to speak, by the source's own order and by time.

Arms (the order: "the source's own order (preferred rank, then single-value constraints, then the values with no
stated end); + an interval filter for an explicit time; + 'present tense means now', run only as Nick's decision 2
says" -- an arm here, reported, never served):
  order     the values not deprecated. Preferred-rank values, if any, are the answer (all of them). Else one value is
            the answer. Else, for a single-value property, the one value with no stated end, if exactly one; for a
            property that may hold several, every value with no stated end (a set). Otherwise refused.
  interval  an explicit time in the question (a year): the values whose own time holds at it (start <= year <= end,
            an open end holding from its start; a point in time in that year). A value with no time, or a start
            unknown, cannot be placed at the year, so it is never spoken under a time; none placed: refused.
            No explicit time: as `order`.
  now       a present-tense question with no explicit time: the values (the preferred-rank ones, if any) that hold
            this year: no stated end, or an end not yet passed, and no start still to come. A single-value property
            with several holding refuses; several values are a set. Otherwise as `interval`.
Nothing is ranked by fame or popularity; ties are never broken: several values are spoken as a set or refused.

Wired: STANDALONE (R4's bench).
"""
from __future__ import annotations

import datetime as _dt
import re

__wiring__ = "STANDALONE"

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
PRESENT = re.compile(r"\b(is|are|does|do|has|have|today|now|currently|nowadays|present)\b|'s\b", re.I)


def asked_year(question: str, times: list | None = None) -> int | None:
    """The explicit year: the set's own (CronQuestions' `times`), else the one year the question writes."""
    if times:
        return int(str(times[0])[:4])
    ys = set(YEAR.findall(question or ""))
    return int(next(iter(ys))) if len(ys) == 1 else None


def present(question: str) -> bool:
    return bool(PRESENT.search(question or ""))


def _y(t: dict | None) -> int | None:
    return t["y"] if t else None


def holds_at(v: dict, year: int) -> bool | None:
    """True if the value's own time places it at `year`; False if it places it elsewhere; None if it cannot say."""
    if v.get("point"):
        return _y(v["point"]) == year
    s, e = _y(v.get("start")), _y(v.get("end"))
    if s is None:
        return False if (e is not None and year > e) else None
    return s <= year and (e is None or year <= e)


def live(vals: list[dict] | None) -> list[dict]:
    return [v for v in vals or () if v.get("v") is not None and v.get("rank") != "deprecated"]


def _set(vs: list[dict], why: str) -> dict:
    return {"spoken": sorted({v["v"] for v in vs}), "why": why}


def _refuse(why: str) -> dict:
    return {"spoken": None, "why": why}


def order(vals: list[dict] | None, single: bool) -> dict:
    lv = live(vals)
    if not lv:
        return _refuse("no value")
    pref = [v for v in lv if v.get("rank") == "preferred"]
    if pref:
        return _set(pref, "preferred rank")
    if len({v["v"] for v in lv}) == 1:
        return _set(lv, "one value")
    open_ = [v for v in lv if not v.get("end")]
    if single:
        return _set(open_, "single-value: the one with no end") if len({v["v"] for v in open_}) == 1 \
            else _refuse("single-value: several with no end" if open_ else "single-value: every value ended")
    return _set(open_, "several: every value with no end") if open_ else _refuse("every value ended")


def interval(vals: list[dict] | None, single: bool, year: int | None) -> dict:
    if year is None:
        return order(vals, single)
    placed = [v for v in live(vals) if holds_at(v, year)]
    if not placed:
        return _refuse("none placed at the year")
    if single and len({v["v"] for v in placed}) > 1:
        return _refuse("single-value: several at the year")
    return _set(placed, "placed at the year")


def now(vals: list[dict] | None, single: bool, year: int | None, question: str, today: _dt.date | None = None) -> dict:
    if year is not None or not present(question):
        return interval(vals, single, year)
    y = (today or _dt.date.today()).year
    lv = live(vals)
    pref = [v for v in lv if v.get("rank") == "preferred"]
    held = [v for v in (pref or lv) if not ((v.get("end") and _y(v["end"]) < y) or (v.get("start") and _y(v["start"]) > y))]
    if not held:
        return _refuse("now: no value holds this year")
    if single and len({v["v"] for v in held}) > 1:
        return _refuse("now: single-value, several hold")
    return _set(held, "now: " + ("preferred rank" if pref else "holds this year"))


def judge(spoken: list | None, gold: set) -> str:
    """refused; right (every value spoken is gold); partly (some gold, some not); wrong (none gold)."""
    if not spoken:
        return "refused"
    s = set(spoken)
    if s <= gold:
        return "right"
    return "partly" if s & gold else "wrong"


__all__ = ["asked_year", "holds_at", "interval", "judge", "now", "order", "present"]
