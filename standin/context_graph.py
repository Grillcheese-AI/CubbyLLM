"""context_graph -- the host's memory of one conversation, joined by ids to what it knows and to why it said what it said.

Wired: STANDALONE (read by validation/exp_e9_context_followups.py; next, the ask loop's session).

Three layers, kept apart and linked (Nick, 2026-09-25: "we need a context graph too" -- long-term, short-term and
reasoning memory around the task):

  long-term   what the host knows before the conversation starts: the history graph (standin/history_graph.py),
              held by reference -- an event id, never a copy
  short-term  the conversation: its turns, the ask each turn made (the kind, the event it resolved to and how), the
              events each ask brought into focus, the clarification still open, the branch a "what if" opened
  reasoning   per ask, the trace from question to decision: how the kind was read, the candidates and their scores,
              where the event came from, the lines served and their source, the draft, the host's verdict and why

What it is for. A follow-up -- "why did it happen?", "who else was there?", "what did the battle lead to?" -- is
resolved against the events the conversation is about before any search of the whole graph; the answer to "which
one did you mean?" binds to one of the offered candidates; "why did you say that?" is a lookup of the trace. An
event's salience is the hippocampal-formation scoring (a match, times a decay with the turns since it was last in
focus, times a strength: 1.0 for the event asked about, 0.6 for an event the answer brought in).

Resolution, in order: an open clarification the reply picks from (by ordinal, year or name words) -> the kind read
off the question (history_lookup.KINDS, or a bare follow-up: "why?", "and then what?", "who else?") -> if the
question names no event (only "it", "that", "the battle") the most salient event whose name holds every word it
does use -> else the whole graph (HistoryLookup.resolve, its floor, margin and same-event rule). A follow-up with
nothing in focus is not guessed: the host says it needs the event.

User text is held in memory only; `save` writes the session through the vault (standin/vault.py) -- user chats are
never stored unencrypted.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from history_graph import FORWARD, key
from history_lookup import Hit, HistoryLookup, canonical, parse_kind, words, years

__wiring__ = "STANDALONE"

TAU = 3.0                   # turns for an event's salience to fall to 1/e
FOCUS, BROUGHT = 1.0, 0.6   # strength of the event asked about / of an event its answer brought in
FLOOR = 0.05                # salience under which an event has left the conversation
TIE = 0.1                   # two referents this close in salience are ambiguous: ask
PRONOUNS = frozenset("he she him her hers his they them their theirs it its this that these those one there then "
                     "same said event thing".split())
_B = r"^\s*(?:and\s+|so\s+|ok(?:ay)?,?\s+)?"
FOLLOW = [   # bare follow-ups KINDS does not read: no phrase at all
    ("effect", re.compile(_B + r"(?:then\s+)?what\s+(?:happened|came|followed)\s*(?:next|after(?:wards)?|after\s+that|then|of\s+(?:it|that|this))?\s*[?.!]*$", re.I)),
    ("effect", re.compile(_B + r"then\s+what\s*[?.!]*$", re.I)),
    ("effect", re.compile(_B + r"what\s+(?:were\s+)?the\s+(?:consequences|results|effects)\s*[?.!]*$", re.I)),
    ("cause", re.compile(_B + r"why\s*[?.!]*$", re.I)),
    ("cause", re.compile(_B + r"what\s+(?:caused|led\s+to)\s+(?:it|that|this)\s*[?.!]*$", re.I)),
    ("when", re.compile(_B + r"when\s*[?.!]*$", re.I)),
    ("where", re.compile(_B + r"where\s*[?.!]*$", re.I)),
    ("who", re.compile(_B + r"who\s*(?:else)?\s*[?.!]*$", re.I)),
    ("who", re.compile(_B + r"who\s+(?:else\s+)?(?:was|were|took\s+part|fought|participated)(?:\s+(?:involved|there|present))?"
                       r"\s*[?.!]*$", re.I)),
]
ORDINALS = {"first": 0, "1st": 0, "former": 0, "second": 1, "2nd": 1, "latter": -1, "third": 2, "3rd": 2, "last": -1}
EARLIER = frozenset("earlier earliest older oldest".split())
LATER = frozenset("later latest newer newest recent".split())


@dataclass
class Turn:
    i: int
    text: str
    ask: int | None = None


@dataclass
class Mention:
    event: str
    turn: int
    strength: float
    ask: int


@dataclass
class Ask:
    id: int
    turn: int
    question: str
    kind: str | None
    phrase: str
    via: str = ""                  # question | context | clarified | none
    status: str = "none"           # clear | ask | none | unparsed
    event: str | None = None
    name: str = ""
    members: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    returned: list = field(default_factory=list)
    others: list = field(default_factory=list)
    canonical: str = ""
    draft: str | None = None
    reply: str | None = None
    reason: str = ""
    trace: list = field(default_factory=list)


class ContextGraph:
    def __init__(self, lookup: HistoryLookup, tau: float = TAU):
        self.lk, self.g, self.tau = lookup, lookup.g, float(tau)
        self.turns: list[Turn] = []
        self.asks: list[Ask] = []
        self.mentions: list[Mention] = []
        self.pending: Ask | None = None       # the clarification the host asked and the asker has not answered
        self.branch: dict | None = None       # the "what if" open: {"root", "events", "ask"}
        self.gap = -1                         # the last turn that named something the host could not resolve

    # ---------------------------------------------------------------- short-term memory
    def salience(self, now: int | None = None) -> dict[str, float]:
        """Events in focus. Nothing before `gap`: once the asker has named something the host could not find, the
        conversation has moved on to it, and "it" no longer means the event before (H-E9 dev: a pronoun after an
        unresolved new topic had fallen back to the old one and been answered about the wrong event)."""
        now = len(self.turns) if now is None else now
        out: dict[str, float] = {}
        for m in self.mentions:
            if m.turn <= self.gap:
                continue
            s = m.strength * math.exp(-(now - m.turn) / self.tau)
            if s > out.get(m.event, 0.0):
                out[m.event] = s
        return {e: s for e, s in out.items() if s >= FLOOR}

    def focus(self) -> str | None:
        sal = self.salience()
        return max(sal, key=sal.get) if sal else None

    def _content(self, phrase: str) -> list[str]:
        """The words a phrase names something with -- every one, known to the graph or not: "Zzyzx" is a name the
        host does not have, not a pronoun."""
        return [w for w in words(phrase) if w not in PRONOUNS]

    def _from_context(self, content: list[str]) -> tuple[list[tuple[str, float]], str]:
        """Events in the conversation the phrase can mean: every content word it uses is in the event's name (none:
        a pronoun, so any event in focus), best salience first."""
        sal = self.salience()
        fits = [(e, s) for e, s in sal.items()
                if all(w in set(words(self.g.events[e].name)) or any(w in set(words(n)) for n in self.g.events[e].names)
                       for w in content)]
        return sorted(fits, key=lambda t: -t[1]), ("pronoun" if not content else "description")

    def _one(self, hits: list[dict]) -> dict | None:
        """The candidate a cue picks: the only one, or the first of several that are readings of one event (H-E9 dev:
        'the one in 1550 BC' matched 'Defeat of the Hyksos by Kamose' and 'Defeat of the Hyksos', both 1550 BC, and
        the pick was dropped as a tie). Its `members` are those readings, served together."""
        if hits and all(self.lk.same_event(hits[0]["event"], h["event"]) for h in hits[1:]):
            return dict(hits[0], members=[h["event"] for h in hits])
        return None

    def _pick(self, text: str, cands: list[dict]) -> dict | None:
        """The candidate a reply to "which one did you mean?" picks, most specific first: a year only one of them
        has; name words only one of them holds ("the Second one"); earlier / later by date; an ordinal in the order
        the host offered them. None: the reply picks nothing, and the clarification is dropped."""
        if not cands:
            return None
        ys = years(text)
        if ys:
            for slack in (0, 1):                        # the year inside its dates first, then a year off
                hit = [c for c in cands if (w := self.g.events[c["event"]].when)
                       and any(w["y0"] - slack <= y <= w["y1"] + slack for y in ys)]
                if hit:
                    one = self._one(hit)
                    if one is not None:
                        return one
                    break
        content = [w for w in self._content(text) if w not in ORDINALS]
        if content:
            scored = [(sum(w in set(words(c["name"])) for w in content), c) for c in cands]
            best = max(s for s, _ in scored)
            top = [c for s, c in scored if s == best and s > 0]
            one = self._one(top)
            if one is not None:
                return one
        low = set(key(text).split())
        dated = [c for c in cands if self.g.events[c["event"]].when]
        if dated and low & (EARLIER | LATER):
            order = sorted(dated, key=lambda c: self.g.events[c["event"]].when["y0"])
            return order[0] if low & EARLIER else order[-1]
        for w in key(text).split():
            if w in ORDINALS:
                i = ORDINALS[w]
                return cands[i] if -len(cands) <= i < len(cands) else None
        return None

    # ---------------------------------------------------------------- one turn
    def ask(self, question: str) -> Ask:
        t = len(self.turns)
        self.turns.append(Turn(t, question))
        a = Ask(len(self.asks), t, question, None, "")
        self.asks.append(a)
        self.turns[-1].ask = a.id
        p = None
        if self.pending is not None:
            p, self.pending = self.pending, None
            chosen = self._pick(question, p.candidates)
            if chosen is not None:
                a.kind, a.phrase, a.via = p.kind, p.phrase, "clarified"
                a.trace.append({"step": "clarified", "answers": p.id, "picked": chosen["name"]})
                return self._serve(a, chosen["event"], chosen.get("members"))
            a.trace.append({"step": "clarification_dropped", "was": p.id})
        a.kind, a.phrase = parse_kind(question)
        how = "question"
        if a.kind is None:
            for kind, pat in FOLLOW:
                if pat.match(question):
                    a.kind, a.phrase, how = kind, "", "follow-up"
                    break
        a.trace.append({"step": "kind", "kind": a.kind, "how": how, "phrase": a.phrase})
        if a.kind is None:
            a.status, a.via = "unparsed", "none"
            if self._content(question):                 # it named something the host cannot read: a new topic
                self.gap = a.turn
            return a
        content = self._content(a.phrase)
        if p is not None and p.candidates and all(any(w in set(words(c["name"])) for c in p.candidates) for w in content):
            # "where was that?" while "which one did you mean?" stands unanswered: "that" is still the thing the host
            # could not tell apart, so the host asks again rather than finding nothing (H-E9 dev: "Sack of Rome by
            # Gauls", asked, then two pronoun follow-ups answered "none")
            a.status, a.via, a.candidates = "ask", "context", p.candidates
            a.trace.append({"step": "ask_again", "was": p.id})
            self.pending = a
            return a
        ctx, form = self._from_context(content)
        if ctx and (not content or form == "description"):
            a.trace.append({"step": "context", "form": form, "candidates": [(self.g.events[e].name, round(s, 3)) for e, s in ctx[:5]]})
            top, rest = ctx[0], [c for c in ctx[1:] if not self.lk.same_event(ctx[0][0], c[0])]
            if rest and top[1] - rest[0][1] < TIE and \
                    sorted(self.lk.serve_all([top[0]], a.kind)[1]) != sorted(self.lk.serve_all([rest[0][0]], a.kind)[1]):
                a.status, a.via = "ask", "context"
                a.candidates = [{"event": e, "name": self.g.events[e].name, "score": round(s, 3)} for e, s in [top, rest[0]]]
                self.pending = a
                return a
            a.via = "context"
            return self._serve(a, top[0])
        if not content:                                  # "why did it happen?" with nothing in focus: not guessed
            a.status, a.via = "none", "none"
            a.trace.append({"step": "no_referent"})
            return a
        hit: Hit = self.lk.resolve(question, a.kind, a.phrase)
        a.candidates = hit.offered or hit.candidates       # an ask offers the close ones only (H-E9 test: a far fifth
        # candidate of the same year, "Domestication of Horse" beside "... of Cats", had left "the one in 8000 BC" unbound)
        a.trace.append({"step": "lookup", "candidates": [(c["name"], c["score"]) for c in hit.candidates]})
        a.via = "question"
        if hit.status == "clear":
            return self._serve(a, hit.event, hit.members)
        a.status = hit.status
        self.gap = a.turn                               # a new topic the host could not find: the old focus is gone
        if hit.status == "ask":
            self.pending = a
        return a

    def _serve(self, a: Ask, event: str, members: list[str] | None = None) -> Ask:
        hit = self.lk.fill(Hit(a.question, a.kind, a.phrase, "clear"), event, a.kind, members)
        a.status, a.event, a.name, a.members = "clear", hit.event, hit.name, hit.members
        a.lines, a.returned, a.others, a.canonical = hit.lines, hit.returned, hit.others, hit.canonical
        a.trace.append({"step": "served", "event": a.name, "members": len(a.members), "lines": len(a.lines),
                        "returned": a.returned, "source": "history_graph", "ask": a.canonical})
        for m in a.members:
            self.mentions.append(Mention(m, a.turn, FOCUS, a.id))
        for e in self._brought(a):
            self.mentions.append(Mention(e, a.turn, BROUGHT, a.id))
        if a.kind == "downstream":
            self.branch = {"root": event, "events": [e for e in self.g.downstream(event, limit=64)], "ask": a.id}
            a.trace.append({"step": "branch", "root": a.name, "events": len(self.branch["events"])})
        return a

    def _brought(self, a: Ask) -> list[str]:
        """The events the answer names, by id: what a cause, effect or downstream ask returned."""
        out = []
        for m in a.members:
            if a.kind == "cause":
                out += [x for x, _ in self.g.into(m, FORWARD)] + [b for _, b in self.g.out(m, ("response to",))]
            elif a.kind == "effect":
                out += [b for _, b in self.g.out(m, FORWARD)] + [x for x, _ in self.g.into(m, ("response to",))]
            elif a.kind == "downstream":
                out += self.g.downstream(m, limit=64)
        names = set(a.returned)
        return [e for e in dict.fromkeys(out) if self.g.events[e].name in names]

    def record(self, a: Ask, draft: str | None, reply: str | None, reason: str) -> None:
        """The decision for an ask: the adapter's draft, what the host said, why."""
        a.draft, a.reply, a.reason = draft, reply, reason
        a.trace += [{"step": "draft", "text": draft}, {"step": "verdict", "reply": reply, "reason": reason}]

    # ---------------------------------------------------------------- reasoning memory
    def why(self, ask_id: int | None = None) -> list[dict]:
        """The trace behind an ask (the last one by default): the answer to "why did you say that?"."""
        a = self.asks[ask_id if ask_id is not None else -1]
        return a.trace

    def decided(self, event: str, kind: str) -> list[Ask]:
        """Earlier asks in this conversation about the same event and kind -- for answering it the same way."""
        return [a for a in self.asks if a.status == "clear" and event in a.members and a.kind == kind]

    # ---------------------------------------------------------------- disk (through the vault)
    def to_json(self) -> dict:
        return {"turns": [asdict(t) for t in self.turns], "asks": [asdict(a) for a in self.asks],
                "mentions": [asdict(m) for m in self.mentions], "pending": None if self.pending is None else self.pending.id,
                "branch": self.branch}

    def save(self, path, key: bytes | None = None) -> None:
        import vault
        vault.write_json(path, self.to_json(), key)

    @classmethod
    def load(cls, path, lookup: HistoryLookup, key: bytes | None = None) -> "ContextGraph":
        import vault
        d = vault.read_json(path, key)
        cg = cls(lookup)
        cg.turns = [Turn(**t) for t in d["turns"]]
        cg.asks = [Ask(**a) for a in d["asks"]]
        cg.mentions = [Mention(**m) for m in d["mentions"]]
        cg.pending = cg.asks[d["pending"]] if d["pending"] is not None else None
        cg.branch = d["branch"]
        return cg
