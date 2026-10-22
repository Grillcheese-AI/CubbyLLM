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
import itertools
import json
import pathlib
import re
from dataclasses import dataclass, field

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .history_time import (BASIS, CONTEXT_SLACK, Event, FORWARD, RELATIONS, SLACK, _AD, _BC, _MONTHS,  # noqa: F401
                           _day, _year, key, overlap, parse_time, rank, slack_for, year_text)
from .history_repair import GraphRepair

__wiring__ = "STANDALONE"


_MONTHS.update({m: i + 1 for i, m in enumerate(
    "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split())})
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
                "oct": 10, "nov": 11, "dec": 12})


class HistoryGraph(GraphRepair):
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
