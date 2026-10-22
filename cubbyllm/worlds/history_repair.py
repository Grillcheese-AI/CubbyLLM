"""cubbyllm/worlds/history_repair.py -- the history graph's repairs: dates inferred from parts and links, links
pruned, events merged, mirrored dates repaired. Split out of history_graph.py (order 0031) as a mixin of HistoryGraph,
verbatim.

Wired: WIRED (HistoryGraph).
"""
from __future__ import annotations

import collections
import itertools
from .history_time import key

__wiring__ = "WIRED"


class GraphRepair:
    """HistoryGraph's repairs (a mixin; order 0031)."""

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

    def merge_events(self, keep: str, drop: str) -> None:
        """Fold event `drop` into `keep`: names, places, parties and sources joined, links moved over (a link that
        would join the event to itself is dropped), `drop` gone."""
        k, d = self.events[keep], self.events.pop(drop)
        for lst, new in ((k.where, d.where), (k.who, d.who), (k.names, d.names), (k.sources, d.sources)):
            for x in new:
                if x and x not in lst:
                    lst.append(x)
        ids = self.by_key.get(key(d.name), [])
        if drop in ids:
            ids.remove(drop)
        old = [(drop, r, b) for r, b in self._out.pop(drop, [])] + [(a, r, drop) for a, r in self._in.pop(drop, [])]
        for a, r, b in old:
            srcs = self.links.pop((a, r, b), None)
            if srcs is None:
                continue
            if a == drop:
                self._in[b] = [(x, y) for x, y in self._in[b] if (x, y) != (drop, r)]
            else:
                self._out[a] = [(x, y) for x, y in self._out[a] if (x, y) != (r, drop)]
            a2, b2 = (keep if a == drop else a), (keep if b == drop else b)
            if a2 == b2:
                continue
            if (a2, r, b2) not in self.links:
                self._index(a2, r, b2)
            have = self.links.setdefault((a2, r, b2), [])
            have += [x for x in srcs if x not in have]

    def repair_mirrored_dates(self, window: int = 100, min_votes: int = 2, ratio: float = 2.0,
                              max_held: int = 2000, passes: int = 3) -> list[dict]:
        """`_repair_mirrored_once` until a pass decides nothing more (each pass's merges clean the neighbours the
        next pass counts). Ratio 2: on the v2.1 graph a hand check of 30 pairs decided at 2-3 found all 30 right; at
        1.5-2, 28 of 30 (Perpetua's martyrdom, the Arab conquest of Mesopotamia put BC); at 1.2-1.5, 22 of 30."""
        out: list[dict] = []
        for _ in range(passes):
            got = self._repair_mirrored_once(window, min_votes, ratio, max_held)
            done = {(d["bc"], d["ad"]) for d in out if d["era"]}
            out = [d for d in out if d["era"]] + got
            if not any(d["era"] and (d["bc"], d["ad"]) not in done for d in got):
                break
        return out

    def _repair_mirrored_once(self, window: int, min_votes: int, ratio: float, max_held: int) -> list[dict]:
        """One event read twice under one name with mirrored dates (359 and 359 BC): a reader lost the era mark
        (1,568 such pairs in the v2.1 graph, and they go both ways -- 'Accession of Philip II' lost its BC,
        'Construction of Hadrian's Wall' gained one). The era is decided by the events the two readings share a
        party or a place with, and the events they link to: how many are dated near the BC reading and how many
        near the AD one (parties and places held by more than `max_held` events -- Rome, Greeks -- say nothing).
        A side with `min_votes` and `ratio` times the other's wins; the other reading takes its dates, mirrored
        back, and is merged into it. Undecided pairs are left as they are. Returns the decisions, both kinds."""
        held = collections.defaultdict(list)
        for eid, ev in self.events.items():
            if ev.when:
                for x in {key(v) for v in (*ev.where, *ev.who) if v}:
                    held[x].append(eid)
        out = []
        for k, ids in list(self.by_key.items()):
            dated = [i for i in ids if self.events[i].when]
            for a, b in itertools.combinations(dated, 2):
                if a not in self.events or b not in self.events:
                    continue
                wa, wb = self.events[a].when, self.events[b].when
                if not (wa["y0"] == -wb["y1"] and wa["y1"] == -wb["y0"] and wa["y0"] != 0):
                    continue
                bc, ad = (a, b) if wa["y0"] < 0 else (b, a)
                wbc, wad = self.events[bc].when, self.events[ad].when
                near = set()
                for i in (a, b):
                    ev = self.events[i]
                    for x in {key(v) for v in (*ev.where, *ev.who) if v}:
                        if len(held[x]) <= max_held:
                            near.update(held[x])
                    near.update(o for _, o in self._out[i])
                    near.update(o for o, _ in self._in[i])
                near -= {a, b}
                votes = collections.Counter()
                for n in near:
                    w = self.events[n].when if n in self.events else None
                    if not w:
                        continue
                    for side, s in (("bc", wbc), ("ad", wad)):
                        if w["y0"] <= s["y1"] + window and w["y1"] >= s["y0"] - window:
                            votes[side] += 1
                win = next((s for s, o in (("bc", "ad"), ("ad", "bc"))
                            if votes[s] >= min_votes and votes[s] >= ratio * votes[o]), None)
                d = {"name": self.events[a].name, "bc": bc, "ad": ad, "year": abs(wa["y0"]),
                     "votes_bc": votes["bc"], "votes_ad": votes["ad"], "era": win}
                out.append(d)
                if win:
                    keep, drop = (bc, ad) if win == "bc" else (ad, bc)
                    if win == "bc" and self.events[keep].when.get("day"):
                        self.events[keep].when = dict(self.events[keep].when, day=None)   # no ISO day before year 1
                    self.merge_events(keep, drop)
        return out
