"""cubbyllm/worlds/history_time.py -- the history graph's times and events: the relations, dates parsed (BC/AD, day,
month, year), an event's key, the slack and the overlap of two times, an event. Split out of history_graph.py (order
0031), verbatim; history_graph.py re-exports every name.

Wired: WIRED (the history graph).
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

__wiring__ = "WIRED"


FORWARD = ("caused", "contributed to", "precursor of")      # A before B, A bears on B


RELATIONS = FORWARD + ("response to", "part of", "ended")


SLACK = 1                                                    # years a source may round by


_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}


_BC = r"(?:\s*(?:b\.?\s?c\.?(?:e\.?)?|bce|av\.?\s*j\.?-?c\.?))"


_AD = r"(?:\s*(?:a\.?\s?d\.?|c\.?e\.?|ap\.?\s*j\.?-?c\.?))"


def _year(num: str, bc: bool) -> int:
    y = int(num)
    return -y if bc else y


def _day(y: int, month: int, day: int) -> str | None:
    """The ISO day when it is one the calendar has, else None."""
    try:
        return datetime.date(y, month, day).isoformat()
    except ValueError:
        return None


def parse_time(text: str) -> dict | None:
    """'April 12, 1861' / '12 avril 1861' / '1861-04-12' -> day; '44 BC'; '1861-1865'; 'the 1790s';
    '18th century (BC)'; 'c. 1500' -> {'y0', 'y1', 'day' (ISO or None), 'approx'}. None when no year is stated."""
    if not text:
        return None
    t = " ".join(str(text).lower().replace("–", "-").replace("—", "-").split())
    approx = bool(re.search(r"\b(c\.|ca\.|circa|about|around|approximately|vers|environ)\b", t))
    m = re.search(r"\b(\d{3,4})-(\d{2})-(\d{2})\b", t)
    if m:                                           # '1813-02-29', '1581-82-83': the year, no day
        y = int(m.group(1))
        return {"y0": y, "y1": y, "day": _day(y, int(m.group(2)), int(m.group(3))), "approx": approx}
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+centur(?:y|ies)" + f"({_BC})?", t) or \
        re.search(r"\b(\d{1,2})(?:e|ème)\s+siècle" + f"({_BC})?", t)
    if m:
        n, bc = int(m.group(1)), bool(m.group(2))
        return ({"y0": -(n * 100), "y1": -((n - 1) * 100 + 1)} if bc else {"y0": (n - 1) * 100 + 1, "y1": n * 100}) | \
            {"day": None, "approx": True}
    m = re.search(r"\b(\d{2,3})0s\b", t)
    if m:
        y = int(m.group(1)) * 10
        return {"y0": y, "y1": y + 9, "day": None, "approx": True}
    m = re.search(r"\b(\d{1,4})" + f"({_BC})?" + r"\s*(?:-|to|until|à|au)\s*(\d{1,4})" + f"({_BC})?" + r"\b", t)
    if m and (m.group(2) or m.group(4) or (len(m.group(1)) >= 3 and len(m.group(3)) >= 2)):
        a, b, bc_a, bc_b = m.group(1), m.group(3), bool(m.group(2)), bool(m.group(4))
        if not (bc_a or bc_b) and len(b) < len(a):               # '1861-65'
            b = a[:len(a) - len(b)] + b
        if bc_b and not bc_a:                                    # '500-400 BC': both BC
            y0, y1 = -int(a), -int(b)
        elif bc_a and not bc_b:                                  # '44 BC to 14' (AD); '500 BC to 400' (BC)
            y0, y1 = -int(a), (-int(b) if int(b) < int(a) else int(b))
        else:
            y0, y1 = _year(a, bc_a), _year(b, bc_b)
        return {"y0": min(y0, y1), "y1": max(y0, y1), "day": None, "approx": approx}
    names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    m = re.search(rf"\b({names})\.?\s+(\d{{1,2}}),?\s+(\d{{3,4}})\b", t) or \
        re.search(rf"\b(\d{{1,2}})(?:er)?\s+({names})\.?,?\s+(\d{{3,4}})\b", t)
    if m:
        g = m.groups()
        month, day = (_MONTHS[g[0]], int(g[1])) if g[0] in _MONTHS else (_MONTHS[g[1]], int(g[0]))
        y = int(g[2])
        if iso := _day(y, month, day):                          # 'June 31, 1934' falls through to its year
            return {"y0": y, "y1": y, "day": iso, "approx": approx}
    m = re.search(r"\b(\d{1,4})" + f"({_BC})" + r"(?!\w)", t)
    if m:
        y = -int(m.group(1))
        return {"y0": y, "y1": y, "day": None, "approx": approx}
    m = re.search(r"(?<![\d,.])(\d{3,4})(?![\d,])" + f"({_AD})?", t) or re.search(r"\b(\d{1,2})" + f"({_AD})", t)
    if m:
        y = int(m.group(1))
        return {"y0": y, "y1": y, "day": None, "approx": approx}
    return None


def key(name: str) -> str:
    """The name an event is merged by: case, punctuation and a leading article aside."""
    k = " ".join(re.sub(r"[^\w\s]", " ", str(name).lower()).split())
    return re.sub(r"^(the|la|le|les|l)\s+", "", k)


# how an event's date was found, strongest first: the passage dates it; the passage's context gives the year; the
# year the book is telling about around that passage; the span of the event it was part of
BASIS = {"stated": 3, "context": 2, "book": 1, "part of": 0}


CONTEXT_SLACK = 3                                            # a context year is the passage's year, not the event's


def rank(w: dict | None) -> int:
    return -1 if not w else BASIS.get(w.get("basis", "stated"), 3)


def slack_for(a: dict | None, b: dict | None) -> int:
    return CONTEXT_SLACK if any(w and w.get("basis") not in (None, "stated") for w in (a, b)) else SLACK


def overlap(a: dict | None, b: dict | None, slack: int | None = None) -> bool:
    if not a or not b:
        return True
    slack = slack_for(a, b) if slack is None else slack
    return a["y0"] - slack <= b["y1"] and b["y0"] - slack <= a["y1"]


def year_text(y: int) -> str:
    return f"{-y} BC" if y < 0 else str(y)


@dataclass
class Event:
    id: str
    name: str
    when: dict | None = None
    where: list = field(default_factory=list)
    who: list = field(default_factory=list)
    kind: str = ""
    names: list = field(default_factory=list)
    sources: list = field(default_factory=list)

    def span(self) -> str:
        if not self.when:
            return ""
        if self.when.get("day"):
            return self.when["day"]
        y0, y1 = self.when["y0"], self.when["y1"]
        return year_text(y0) if y0 == y1 else f"{year_text(y0)} to {year_text(y1)}"
