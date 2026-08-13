"""Question/fact grammar for the multi-hop corpus (spec section 4.1).

The .pq questions follow `What is the R_n of the R_{n-1} of ... of <tail>?`
where the tail mixes the hop-1 relation and the seed entity (relations
contain "of" themselves, e.g. "country of citizenship"), so the tail is
resolved by the hop-1 FACT parse, never guessed here. Facts follow two
observed templates; anything else parses to None — parse failures are
counted by callers, not raised.

Grammar v2 (question-grammar coverage follow-up, TODO.md) adds five more
WH-frames on top of the original "What is the ... ?" chain. Every frame
normalizes into the SAME QuestionPlan shape (canonical tail "R of E",
chain split on " of the " unchanged) so `pipeline.py` needs zero changes:

  - "What is/Where is/Who is/Who was/Which is the R of the R of ... <tail>?"
    the original chain grammar, now also matching alternate WH-leads, e.g.
    "Where is the administrative territorial entity of inster?"
  - "Which <class> is the R of the R of ... <tail>?" the same chain grammar
    behind a leading class noun, e.g. "Which country is the country of the
    currency of the country of citizenship of Henry Allcock?" — the class
    duplicates the final relation's answer type, so it is captured into
    `QuestionPlan.answer_class` rather than added as a hop.
  - "Which <class> is <entity> in?" a class-headed 1-hop with no "of"
    chain at all, e.g. "Which country is arcadia, maryland in?"
  - "What R does <entity> have?" an inversion (relation before entity,
    entity before relation) synthesized into canonical "R of E", e.g.
    "What given name does margit koloczy have?" -> tail "given name of
    margit koloczy".
  - "To which R does <entity> belong?" the same inversion, "belong"
    phrasing, e.g. "To which parent entity does Hum Along and Dance
    belong?" -> tail "parent entity of Hum Along and Dance".
  - "What R is <entity>?" a bare inversion with no "the"/"does"/"have",
    e.g. "What genre is Tombo (album)?" -> tail "genre of Tombo (album)".

Excluded by design: nested relative-clause forms ("Which list includes
the list that includes ...") stay unparseable — depth>=2 recursion is
the future LM tier's job, not this regex grammar's (model-panel-validated
boundary).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED

# "What is/Where is/Who is/Who was/Which is the ... ?" — the chain grammar,
# now matching any of the WH-leads the corpus actually uses.
_Q = re.compile(
    r"^\s*(?:what|where|who|which)\s+(?:is|was)\s+the\s+(?P<body>.+?)\s*\?\s*$",
    re.I)
# "Which <class> is the R of the R of ... <tail>?" / "Which <class> is <entity> in?"
_Q_WHICH_CLASS = re.compile(
    r"^\s*which\s+(?P<cls>[A-Za-z][\w\s]*?)\s+is\s+(?P<rest>.+?)\s*\?\s*$",
    re.I)
# "Which <class> is <entity> in?" — rest with no leading "the" chain.
_ENTITY_IN = re.compile(r"^(?P<entity>.+?)\s+in$", re.I)
# "What R does <entity> have?"  (inversion)
_Q_INV_HAVE = re.compile(
    r"^\s*what\s+(?P<rel>.+?)\s+does\s+(?P<entity>.+?)\s+have\s*\?\s*$", re.I)
# "To which R does <entity> belong?"  (inversion, "belong" phrasing)
_Q_INV_BELONG = re.compile(
    r"^\s*to\s+which\s+(?P<rel>.+?)\s+does\s+(?P<entity>.+?)\s+belong\s*\?\s*$",
    re.I)
# "What R is <entity>?"  (bare inversion, no "the"/"does"/"have")
_Q_WHAT_IS = re.compile(
    r"^\s*what\s+(?P<rel>[A-Za-z][\w\s]*?)\s+is\s+(?P<entity>.+?)\s*\?\s*$",
    re.I)
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
    # "Which <class> is ..." captures the leading class noun here — it
    # duplicates the final relation's answer type, so it is never a hop.
    answer_class: str | None = None


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


def _split_chain(body: str) -> tuple[list[str], str]:
    """Split a chain body on ' of the ' -> (relations outer-to-inner, tail).
    segments run answer-side first: [R_n, R_{n-1}, ..., tail]."""
    segments = body.split(" of the ")
    tail = segments[-1].strip()
    rels = [s.strip() for s in segments[:-1]]
    return rels, tail


def parse_question(q: str) -> QuestionPlan | None:
    m = _Q.match(q)
    if m:
        rels, tail = _split_chain(m.group("body"))
        relations: list[str | None] = [None] + list(reversed(rels))
        return QuestionPlan(relations=relations, tail=tail, n_hop=len(rels) + 1)

    m = _Q_WHICH_CLASS.match(q)
    if m:
        cls = m.group("cls").strip()
        rest = m.group("rest").strip()
        if rest[:4].lower() == "the ":
            rels, tail = _split_chain(rest[4:])
            relations = [None] + list(reversed(rels))
            return QuestionPlan(relations=relations, tail=tail,
                                n_hop=len(rels) + 1, answer_class=cls)
        m_in = _ENTITY_IN.match(rest)
        if m_in:
            entity = m_in.group("entity").strip()
            return QuestionPlan(relations=[None], tail=f"{cls} of {entity}",
                                n_hop=1, answer_class=cls)
        return None

    m = _Q_INV_HAVE.match(q)
    if m:
        rel, entity = m.group("rel").strip(), m.group("entity").strip()
        return QuestionPlan(relations=[None], tail=f"{rel} of {entity}", n_hop=1)

    m = _Q_INV_BELONG.match(q)
    if m:
        rel, entity = m.group("rel").strip(), m.group("entity").strip()
        return QuestionPlan(relations=[None], tail=f"{rel} of {entity}", n_hop=1)

    m = _Q_WHAT_IS.match(q)
    if m:
        rel, entity = m.group("rel").strip(), m.group("entity").strip()
        return QuestionPlan(relations=[None], tail=f"{rel} of {entity}", n_hop=1)

    return None


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
