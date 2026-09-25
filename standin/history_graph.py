"""history_graph -- events in time, linked by cause and precursor: the base of history the worlds branch from.

Wired: STANDALONE (built by standin/data/build_history_graph.py; read by the talk data and, next, the VM's
history world).

An event is a node with a time span (years, BC negative; a day when the source gives one), places, the
people and bodies that took part, and every source it was read from (book, passage). Links run between
events and say how one bears on another:

  caused          A caused B                        A no later than B
  contributed to  A was one of B's causes           A no later than B
  precursor of    A came first and set B up         A no later than B
  response to     A was a response to B             B no later than A
  part of         A was part of B (a battle, a war) A within B's span
  ended           A ended B                         A no earlier than B's start

A link whose dates contradict its meaning is refused (a cause dated after its effect), with a year of slack
for sources that round; `prune_links()` checks every link again once the dates are final. `downstream(x)` is
what a branch that changes x has to re-evaluate: every event x caused, contributed to or set up, every response
to it, and its parts, transitively. `chain(a, b)` is the
shortest causal path from a to b. The same event read from two books is one node when the names match and
the dates agree; the same name with dates years apart is two events ("Battle of Newbury", 1643 and 1644).
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass, field

__wiring__ = "STANDALONE"

FORWARD = ("caused", "contributed to", "precursor of")      # A before B, A bears on B
RELATIONS = FORWARD + ("response to", "part of", "ended")
SLACK = 1                                                    # years a source may round by

_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
_MONTHS.update({m: i + 1 for i, m in enumerate(
    "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split())})
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
                "oct": 10, "nov": 11, "dec": 12})
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


class HistoryGraph:
    def __init__(self):
        self.events: dict[str, Event] = {}
        self.by_key: dict[str, list[str]] = collections.defaultdict(list)
        self.links: dict[tuple, list] = {}                  # (a id, rel, b id) -> sources
        self.facts: dict[tuple, list] = {}                  # (entity, relation, value) -> sources
        self.refused: collections.Counter = collections.Counter()
        self._out: dict[str, list] = collections.defaultdict(list)   # a id -> [(rel, b id)], in link order
        self._in: dict[str, list] = collections.defaultdict(list)    # b id -> [(a id, rel)]

    def _index(self, a: str, rel: str, b: str) -> None:
        self._out[a].append((rel, b))
        self._in[b].append((a, rel))

    # ---------------------------------------------------------------- building
    def add_event(self, name: str, when: dict | None = None, where=(), who=(), kind: str = "", source=None) -> str:
        k = key(name)
        for eid in self.by_key.get(k, []):
            ev = self.events[eid]
            if overlap(ev.when, when):
                if when and (rank(when) > rank(ev.when) or
                             (rank(when) == rank(ev.when) and when.get("day") and not ev.when.get("day"))):
                    ev.when = when                                   # a stated date over a context year; a day over a year
                for lst, new in ((ev.where, where), (ev.who, who)):
                    for x in new:
                        if x and x not in lst:
                            lst.append(x)
                if name not in ev.names:
                    ev.names.append(name)
                if source and source not in ev.sources:
                    ev.sources.append(source)
                ev.kind = ev.kind or kind
                return eid
        eid = "ev:" + hashlib.sha1(f"{k}|{when['y0'] if when else ''}".encode()).hexdigest()[:12]
        self.events[eid] = Event(eid, name, when, [x for x in where if x], [x for x in who if x], kind, [name],
                                 [source] if source else [])
        self.by_key[k].append(eid)
        return eid

    def find(self, name: str, when: dict | None = None) -> str | None:
        ids = [i for i in self.by_key.get(key(name), []) if overlap(self.events[i].when, when)]
        return ids[0] if ids else None

    def time_ok(self, a: str, rel: str, b: str) -> bool:
        ta, tb = self.events[a].when, self.events[b].when
        if not ta or not tb:
            return True
        s = slack_for(ta, tb)
        if rel in FORWARD:
            return ta["y0"] <= tb["y1"] + s
        if rel == "response to":
            return tb["y0"] <= ta["y1"] + s
        if rel == "part of":
            return ta["y0"] >= tb["y0"] - s and ta["y1"] <= tb["y1"] + s
        if rel == "ended":         # the end comes after what it ends began: not a reign "ending" the next one, a war
            if ta.get("day") and tb.get("day"):             # "ending" its treaty (readers turn this one round often,
                return (ta["y0"], ta["day"]) >= (tb["y0"], tb["day"])   # so two stated dates get no slack)
            return ta["y0"] >= tb["y0"] - (s if s > SLACK else 0)
        return True

    def _inherit_parts(self, stats) -> None:
        changed = True
        while changed:
            changed = False
            for (a, r, b) in self.links:
                ea, eb = self.events[a], self.events[b]
                if r == "part of" and not ea.when and eb.when:
                    ea.when = {"y0": eb.when["y0"], "y1": eb.when["y1"], "day": None, "approx": True, "basis": "part of"}
                    stats["part of"] += 1
                    changed = True

    def infer_dates(self, passage_year: dict, hints: dict | None = None, agree: int = 2) -> collections.Counter:
        """Every event the books did not date gets one, marked by `basis` and approx -- a date to order and branch
        by, not one to quote: first the span of the dated event it was part of; then the year its book is telling
        about where it is told (the passage's year, else the nearest before it) -- or, where the reader offered a
        year the passage does not hold (`hints`) and it agrees with the book's year within `agree`, that year."""
        stats = collections.Counter(ev.when.get("basis", "stated") for ev in self.events.values() if ev.when)
        hints = hints or {}
        self._inherit_parts(stats)
        for eid, ev in self.events.items():
            if ev.when:
                continue
            for src in ev.sources:
                y = passage_year.get(tuple(src))
                if y is None:
                    continue
                h = hints.get(eid)
                if h and abs(h["y0"] - y) <= agree:
                    ev.when = {"y0": h["y0"], "y1": h["y1"], "day": None, "approx": True, "basis": "book"}
                    stats["book+reader"] += 1
                else:
                    ev.when = {"y0": y, "y1": y, "day": None, "approx": True, "basis": "book"}
                    stats["book"] += 1
                break
        self._inherit_parts(stats)
        stats["undated"] = sum(1 for ev in self.events.values() if not ev.when)
        return stats

    def add_link(self, a: str, rel: str, b: str, source=None) -> str | None:
        """None when kept, else why it was refused."""
        why = None
        if rel not in RELATIONS:
            why = "relation"
        elif a == b:
            why = "self"
        elif not self.time_ok(a, rel, b):
            why = "time_contradicts"
        if why:
            self.refused[why] += 1
            return why
        if (a, rel, b) not in self.links:
            self._index(a, rel, b)
        srcs = self.links.setdefault((a, rel, b), [])
        if source and source not in srcs:
            srcs.append(source)
        return None

    def prune_links(self, why: str = "time_contradicts_dated") -> int:
        """Drop the links whose dates contradict them now -- a link checked while an end was undated, or before a
        merge gave an event a stronger date, is checked again once the dates are final."""
        bad = [k for k in self.links if not self.time_ok(*k)]
        for k in bad:
            del self.links[k]
        if bad:
            self.refused[why] += len(bad)
            self._out.clear(); self._in.clear()
            for k in self.links:
                self._index(*k)
        return len(bad)

    def add_fact(self, e: str, r: str, v: str, source=None):
        srcs = self.facts.setdefault((e, r, v), [])
        if source and source not in srcs:
            srcs.append(source)

    # ---------------------------------------------------------------- reading
    def out(self, x: str, rels=RELATIONS) -> list[tuple[str, str]]:
        return [(r, b) for (r, b) in self._out.get(x, ()) if r in rels]

    def into(self, x: str, rels=RELATIONS) -> list[tuple[str, str]]:
        return [(a, r) for (a, r) in self._in.get(x, ()) if r in rels]

    def causes(self, x: str) -> list[str]:
        return [a for a, r in self.into(x, FORWARD)] + [b for r, b in self.out(x, ("response to",))]

    def effects(self, x: str) -> list[str]:
        return [b for r, b in self.out(x, FORWARD)] + [a for a, r in self.into(x, ("response to",))]

    def downstream(self, x: str, limit: int | None = None) -> list[str]:
        """Every event a change to x reaches: what it caused, contributed to or set up, the responses to it,
        its parts -- transitively, in time order. `limit` stops the walk once that many are found (a caller that
        only uses small reaches need not walk a whole war)."""
        seen, todo = {x}, [x]
        while todo and (limit is None or len(seen) <= limit):
            cur = todo.pop()
            nxt = self.effects(cur) + [a for a, r in self.into(cur, ("part of",))]
            for n in nxt:
                if n not in seen:
                    seen.add(n); todo.append(n)
        return self.timeline(seen - {x})

    def chain(self, a: str, b: str, max_hops: int = 6) -> list[str] | None:
        """The shortest path of cause links from a to b ([a, ..., b]), or None."""
        prev, frontier = {a: None}, [a]
        for _ in range(max_hops):
            nxt = []
            for cur in frontier:
                for n in self.effects(cur):
                    if n not in prev:
                        prev[n] = cur; nxt.append(n)
                        if n == b:
                            path = [b]
                            while prev[path[-1]] is not None:
                                path.append(prev[path[-1]])
                            return path[::-1]
            frontier = nxt
        return None

    def timeline(self, ids) -> list[str]:
        def order(i):
            w = self.events[i].when or {"y0": 10 ** 6, "y1": 10 ** 6, "day": None}
            return w["y0"], w.get("day") or "~", w["y1"], self.events[i].name        # a dated day first, then the shorter span
        return sorted(ids, key=order)

    # ---------------------------------------------------------------- disk
    def save(self, path: str | pathlib.Path):
        with open(path, "w", encoding="utf-8") as f:
            for ev in self.events.values():
                f.write(json.dumps({"type": "event", **ev.__dict__}, ensure_ascii=False) + "\n")
            for (a, r, b), s in self.links.items():
                f.write(json.dumps({"type": "link", "a": a, "rel": r, "b": b, "sources": s}, ensure_ascii=False) + "\n")
            for (e, r, v), s in self.facts.items():
                f.write(json.dumps({"type": "fact", "e": e, "r": r, "v": v, "sources": s}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "HistoryGraph":
        g = cls()
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            t = d.pop("type")
            if t == "event":
                ev = Event(**d)
                ev.sources = [tuple(s) if isinstance(s, list) else s for s in ev.sources]
                g.events[ev.id] = ev
                g.by_key[key(ev.name)].append(ev.id)
            elif t == "link":
                if (d["a"], d["rel"], d["b"]) not in g.links:
                    g._index(d["a"], d["rel"], d["b"])
                g.links[(d["a"], d["rel"], d["b"])] = [tuple(s) for s in d["sources"]]
            else:
                g.facts[(d["e"], d["r"], d["v"])] = [tuple(s) for s in d["sources"]]
        return g
