"""cubbyllm/reasoning/context_turns.py -- the context graph's parts: its thresholds, the words that follow up or
point back, a turn, a mention, an ask. Split out of context_graph.py (order 0031), verbatim; context_graph.py
re-exports every name.

Wired: WIRED (the context graph).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from ..worlds.history_lookup import Hit, HistoryLookup, canonical, parse_kind, words, years

__wiring__ = "WIRED"


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
