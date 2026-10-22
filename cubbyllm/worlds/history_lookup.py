"""The host's lookup over the history graph (standin/history_graph.py): a question in plain words -> the kind
of ask, the event it is about, and the facts block the graph gives for it -- the lines the talk adapter reads
and the values the host returns, which `ask.talk_reply` reads the draft against.

Wired: STANDALONE (read by validation/exp_e8_history_retrieval.py; next, the ask loop's history world).

No model. The kind is read off the question (KINDS: what led to / what came of / when / where / who / what would
change if ... had not happened). The event is found by its names: every name it was read under is indexed by
its words, and an event scores by the IDF-weighted share of its name the question covers, times the root of the
share of the question's words the name covers -- so a question that holds a name word for word scores its full
name coverage, and a reworded one scores by its rarest shared words. A year in the question agreeing with the
event's dates lifts it, a year against them sinks it. The graph holds some events twice, read from two books
under two names ("Sack of Rome by Gauls", "Sack of Rome by Brennus' Gauls"); candidates that are readings of the
top one (`same_event`: overlapping dates, most words shared, no ordinal telling them apart) are served together as
one event. The best event is used only when the match is clear: a score over `min_score` and ahead of the next
distinct event by `margin` -- or, when the events that close would all return the same answer, any of them.
Otherwise the host asks which one was meant (the candidates) instead of guessing: the host's checks read the
draft against the block it retrieved, and a block for the wrong event passes them. (The kind patterns, the word
normalisation and the same-event rule were revised on H-E8's dev split, before its test split was run.)

    lk = HistoryLookup(HistoryGraph.load(path))
    hit = lk.ask("What led to the Battle of Vienna in 1683?")
    hit.status ('clear' | 'ask' | 'none' | 'unparsed'), hit.kind, hit.event, hit.lines, hit.returned, hit.others
"""
from __future__ import annotations

import collections
import math
import re
import time
import unicodedata
from array import array
from dataclasses import dataclass, field

from .history_graph import FORWARD, HistoryGraph, key, overlap, year_text

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .history_words import (CANON, FACT_MARGIN, FACT_MAX_DF, FACT_MIN, Hit, KINDS, MARGIN, MAX_DOWN,  # noqa: F401
                            MAX_LINKS, MIN_SCORE, ORDINAL, STOP, _END, _NOT, _PRON, _SUFFIX, _THE, _X, _YEAR, _fold,
                            _k, _markers, _quotable, _stem, canonical, own_lines, parse_kind, serve, the, words,
                            years)

__wiring__ = "STANDALONE"


# 'Athenian'/'Athens', 'Macedonian'/'Macedonia', 'Persian'/'Persia' (H-E8 dev, second round)


class HistoryLookup:
    def __init__(self, g: HistoryGraph, min_score: float = MIN_SCORE, margin: float = MARGIN, max_df: int = 100_000,
                 max_lines: int = 14):
        self.g, self.min_score, self.margin, self.max_df = g, float(min_score), float(margin), int(max_df)
        self.max_lines = int(max_lines)
        self._name_years: dict[str, frozenset] = {}
        self.ids = list(g.events)
        self.name_event: list[int] = []            # name index -> event index
        self.name_words: list[tuple[str, ...]] = []
        self.name_key: list[str] = []
        post = collections.defaultdict(list)
        for ei, eid in enumerate(self.ids):
            for nm in dict.fromkeys(g.events[eid].names or [g.events[eid].name]):
                ws = tuple(dict.fromkeys(words(nm)))
                if not ws:
                    continue
                ni = len(self.name_event)
                self.name_event.append(ei)
                self.name_words.append(ws)
                self.name_key.append(key(nm))
                for w in ws:
                    post[w].append(ni)
        n = len(self.name_event)
        self.idf = {w: math.log(1 + n / (1 + len(p))) for w, p in post.items()}
        self.post = dict(post)
        self.name_mass = [sum(self.idf[w] for w in ws) for ws in self.name_words]

    def _year_factor(self, eid: str, qyears: list[int]) -> float:
        if not qyears:
            return 1.0
        ev = self.g.events[eid]
        named = self._name_years.get(eid)
        if named is None:
            named = self._name_years[eid] = frozenset(y for n in (ev.name, *ev.names) for y in years(n))
        if named.intersection(qyears):
            return 1.2                  # the year is in the event's name ('... Largest Since 1933', dated 1938): it agrees
        w = ev.when
        if not w:
            return 0.9
        return 1.2 if any(w["y0"] - 1 <= y <= w["y1"] + 1 for y in qyears) else 0.5

    def _facts_index(self):
        """Every event as the words of its names, places and parties (built on first use): the index a description
        is read against when no name fits -- "the 1601 clash near Cork involving Spain, the English and Hugh
        O'Neill" is the Battle of Kinsale by its facts, not by its name."""
        if getattr(self, "_fpost", None) is not None:
            return
        post = collections.defaultdict(lambda: array("i"))
        for ei, eid in enumerate(self.ids):
            ev = self.g.events[eid]
            ws = set()
            for text in (*(ev.names or [ev.name]), *ev.where, *ev.who):
                ws.update(words(text))
            for w in ws:
                post[w].append(ei)
        n = len(self.ids)
        self._fpost = {w: p for w, p in post.items() if len(p) <= FACT_MAX_DF}
        self._fidf = {w: math.log(1 + n / (1 + len(p))) for w, p in self._fpost.items()}

    def find_facts(self, phrase: str, question: str = "", k: int = 5) -> list[dict]:
        """Events ranked by how much of the phrase their names, places and parties cover (IDF-weighted, words too
        common to tell events apart left out), times the year factor: [{event, name, score, shared}]."""
        self._facts_index()
        q = list(dict.fromkeys(w for w in words(phrase) if w in self._fidf))
        if len(q) < 2:
            return []
        qmass = sum(self._fidf[w] for w in q)
        hit, shared = collections.defaultdict(float), collections.defaultdict(int)
        for w in q:
            for ei in self._fpost[w]:
                hit[ei] += self._fidf[w]
                shared[ei] += 1
        qyears = years(question or phrase)
        scored = sorted(((m / qmass * self._year_factor(self.ids[ei], qyears), ei) for ei, m in hit.items()
                         if shared[ei] >= 2), key=lambda t: (-t[0], self.ids[t[1]]))[:k]
        return [{"event": self.ids[ei], "name": self.g.events[self.ids[ei]].name, "score": round(s, 4),
                 "shared": shared[ei]} for s, ei in scored]

    def find(self, phrase: str, question: str = "", k: int = 5) -> list[dict]:
        """Events ranked for `phrase` (the question's years read from `question`): [{event, name, score}]. A name
        counts only when the question covers at least 40% of it (IDF-weighted)."""
        q = list(dict.fromkeys(w for w in words(phrase) if w in self.idf))
        if not q:
            return []
        qmass = sum(self.idf[w] for w in q)
        rare = [w for w in q if len(self.post[w]) <= self.max_df] or q
        hit = collections.defaultdict(float)
        for w in rare:
            for ni in self.post[w]:
                hit[ni] += self.idf[w]
        common = [w for w in q if w not in rare]
        best: dict[int, float] = {}
        qkey = f" {key(question or phrase)} "
        qyears = years(question or phrase)
        for ni, m in hit.items():
            m += sum(self.idf[w] for w in common if w in self.name_words[ni])
            cov = m / self.name_mass[ni]
            if cov < 0.4:
                continue
            s = cov * math.sqrt(m / qmass)
            if len(self.name_key[ni].split()) >= 2 and f" {self.name_key[ni]} " in qkey:
                s *= 1.15                                  # the name as a phrase, not its words scattered
            ei = self.name_event[ni]
            if s > best.get(ei, 0.0):
                best[ei] = s
        scored = []
        for ei, s in best.items():
            eid = self.ids[ei]
            ev = self.g.events[eid]
            s *= self._year_factor(eid, qyears) * (1 + 0.03 * min(math.log1p(len(ev.sources)), 3.0))
            scored.append((s, eid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [{"event": eid, "name": self.g.events[eid].name, "score": round(s, 4)} for s, eid in scored[:k]]

    def same_event(self, a: str, b: str) -> bool:
        """Two readings of one event the graph did not merge (two books naming it differently): dates that overlap,
        names sharing two thirds of the shorter one's words (one of them not a common word), no ordinal or numeral
        telling them apart. 'Sack of Rome by Gauls' / "Sack of Rome by Brennus' Gauls"; not 'Reign of Darius' /
        'Revolts against Darius I', not the First and the Second Battle of Bull Run."""
        ea, eb = self.g.events[a], self.g.events[b]
        if not ea.when or not eb.when or not overlap(ea.when, eb.when):
            return False
        ma, mb = _markers(ea.name), _markers(eb.name)
        if ma and mb and ma != mb:
            return False
        wa, wb = set(words(ea.name)), set(words(eb.name))
        shared = wa & wb
        if not shared or len(shared) < 0.66 * min(len(wa), len(wb)):
            return False
        return any(len(self.post.get(w, ())) <= 5000 for w in shared)

    def serve_all(self, members: list[str], kind: str) -> tuple[list, list, list]:
        """One block for the readings of one event: their lines and returned values together, up to `max_lines`."""
        lines, returned, names = [], [], {self.g.events[m].name for m in members}
        for m in members:
            ls, rs, _ = serve(self.g, m, kind)
            for l in ls:
                if l not in lines and len(lines) < self.max_lines:
                    lines.append(l)
            shown = {x for l in lines for x in (l[0], l[2])}
            returned += [r for r in rs if r in shown and r not in returned]
        keep = names | set(returned)
        # a place or a party inside the event's own name ('Fire at Long Branch' / Long Branch) is the event being
        # named, not another entity the draft must not mention
        in_name = " ".join(f" {key(n)} " for n in names)
        others = list(dict.fromkeys(x for e, _, v in lines for x in (e, v)
                                    if x not in keep and not (key(x) and f" {key(x)} " in in_name)))
        return lines, returned, others

    def ask(self, question: str, k: int = 5) -> Hit:
        kind, phrase = parse_kind(question)
        return self.resolve(question, kind, phrase, k)

    def resolve(self, question: str, kind: str | None, phrase: str, k: int = 5) -> Hit:
        """The event for an ask whose kind is already read (`ask`, or the context graph with its own reading)."""
        t0 = time.perf_counter()
        cands = self.find(phrase, question, 10)
        hit = Hit(question, kind, phrase, "none", candidates=cands[:k])
        if cands:
            top = cands[0]
            hit.members = [top["event"]] + [c["event"] for c in cands[1:] if self.same_event(top["event"], c["event"])]
        if kind is None:
            hit.status = "unparsed"
        elif cands and cands[0]["score"] >= self.min_score:
            top = cands[0]
            rest = [c for c in cands[1:] if c["event"] not in hit.members]
            close = [c for c in rest if top["score"] - c["score"] < self.margin]
            if close:
                mine = tuple(sorted(self.serve_all(hit.members, kind)[1]))
                hit.status = "clear" if all(tuple(sorted(serve(self.g, c["event"], kind)[1])) == mine for c in close) else "ask"
                if hit.status == "ask":                    # the top one's readings, the close ones and theirs -- not a
                    near = [top] + close                   # far fifth candidate a year in the reply could also fit
                    hit.offered = [c for c in cands if c["event"] in hit.members or any(
                        c["event"] == x["event"] or self.same_event(x["event"], c["event"]) for x in near)]
            else:
                hit.status = "clear"
            if hit.status == "clear":
                self.fill(hit, top["event"], kind, hit.members)
                hit.score = top["score"]
        if hit.status == "none" and kind is not None:
            # no name fits: read the description against the events' places and parties, and use the best one only
            # when it stands well clear of the next (H-E8 dev, second round: the right tops scored 0.68-1.01 with the
            # next distinct event 0.28-0.61 below; every wrong top was within 0.1 of its next, and under 0.55)
            fc = self.find_facts(phrase, question, 10)
            if fc and fc[0]["score"] >= FACT_MIN:
                top = fc[0]
                mem = [top["event"]] + [c["event"] for c in fc[1:] if self.same_event(top["event"], c["event"])]
                nxt = next((c["score"] for c in fc[1:] if c["event"] not in mem), 0.0)
                if top["score"] - nxt >= FACT_MARGIN:
                    hit.status, hit.via, hit.score = "clear", "facts", top["score"]
                    hit.candidates = fc[:k]
                    self.fill(hit, top["event"], kind, mem)
        hit.ms = (time.perf_counter() - t0) * 1000
        return hit

    def fill(self, hit: Hit, event: str, kind: str, members: list[str] | None = None) -> Hit:
        """Serve `kind` about `event` (and its other readings) into the hit: block, returned, others, the host's ask."""
        hit.event, hit.name, hit.members = event, self.g.events[event].name, list(members or [event])
        hit.lines, hit.returned, hit.others = self.serve_all(hit.members, kind)
        hit.canonical = canonical(kind, hit.name)
        return hit
