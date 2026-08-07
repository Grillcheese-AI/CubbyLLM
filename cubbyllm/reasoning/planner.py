"""Question/fact grammar for the multi-hop corpus (spec section 4.1).

The .pq questions follow `What is the R_n of the R_{n-1} of ... of <tail>?`
where the tail mixes the hop-1 relation and the seed entity (relations
contain "of" themselves, e.g. "country of citizenship"), so the tail is
resolved by the hop-1 FACT parse, never guessed here. Facts follow two
observed templates; anything else parses to None — parse failures are
counted by callers, not raised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED

_Q = re.compile(r"^\s*what\s+is\s+the\s+(?P<body>.+?)\s*\?\s*$", re.I)
# "O is the R of S"  (non-greedy obj, greedy rel, greedy subj)
_F_OF = re.compile(r"^(?P<obj>.+?) is the (?P<rel>.+) of (?P<subj>.+)$")
# "O is the R S is in"  (the corpus's second template)
_F_IN = re.compile(r"^(?P<obj>.+?) is the (?P<rel>\S+) (?P<subj>.+) is in$")
_ARTICLES = ("the ", "a ", "an ")


@dataclass(frozen=True)
class QuestionPlan:
    relations: list[str | None]     # walk order; [0] is None (tail hop)
    tail: str
    n_hop: int


@dataclass(frozen=True)
class Triple:
    obj: str
    rel: str
    subj: str


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = " ".join(s.split())
    for a in _ARTICLES:
        if s.startswith(a):
            s = s[len(a):]
    return s


def parse_question(q: str) -> QuestionPlan | None:
    m = _Q.match(q)
    if not m:
        return None
    segments = m.group("body").split(" of the ")
    # segments run answer-side first: [R_n, R_{n-1}, ..., tail]
    tail = segments[-1].strip()
    rels = [s.strip() for s in segments[:-1]]
    relations: list[str | None] = [None] + list(reversed(rels))
    return QuestionPlan(relations=relations, tail=tail, n_hop=len(segments))


def parse_fact(f: str) -> Triple | None:
    f = " ".join(f.split())
    m = _F_IN.match(f) or _F_OF.match(f)
    if not m:
        return None
    return Triple(obj=m.group("obj").strip(), rel=m.group("rel").strip(),
                  subj=m.group("subj").strip())


def relation_matches(expected: str, got: str) -> bool:
    a = set(normalize(expected).split())
    b = set(normalize(got).split())
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.6
