r"""Question/fact grammar for the multi-hop corpus (spec section 4.1).

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

Backtracking guard: four v2 frames pair a non-greedy capture with a
literal the regex engine has to search for (e.g. `\s+is\s+`), which is
quadratic on adversarial input with many near-miss occurrences of that
literal and no closing "?" (measured ~10.7s at 80k chars). `parse_question`
rejects oversized/unterminated input up front (`MAX_QUESTION_LEN`, the
"?" pre-check) so the worst case stays trivial regardless of engine
behavior; the capture groups are also bounded as defense in depth.
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
# cls is bounded (class nouns run 1-3 words in the corpus) to cap the
# backtracking multiplier when the "\s+is\s+" literal isn't found.
_Q_WHICH_CLASS = re.compile(
    r"^\s*which\s+(?P<cls>[A-Za-z][\w\s]{0,63}?)\s+is\s+(?P<rest>.+?)\s*\?\s*$",
    re.I)
# "Which <class> is <entity> in?" — rest with no leading "the" chain.
_ENTITY_IN = re.compile(r"^(?P<entity>.+?)\s+in$", re.I)
# "What R does <entity> have?"  (inversion); rel bounded similarly.
_Q_INV_HAVE = re.compile(
    r"^\s*what\s+(?P<rel>.{1,80}?)\s+does\s+(?P<entity>.+?)\s+have\s*\?\s*$",
    re.I)
# "To which R does <entity> belong?"  (inversion, "belong" phrasing)
_Q_INV_BELONG = re.compile(
    r"^\s*to\s+which\s+(?P<rel>.{1,80}?)\s+does\s+(?P<entity>.+?)\s+belong\s*\?\s*$",
    re.I)
# "What R is <entity>?"  (bare inversion, no "the"/"does"/"have")
_Q_WHAT_IS = re.compile(
    r"^\s*what\s+(?P<rel>[A-Za-z][\w\s]{0,63}?)\s+is\s+(?P<entity>.+?)\s*\?\s*$",
    re.I)
# "O is the R of S"  (non-greedy obj, greedy rel, greedy subj)
_F_OF = re.compile(r"^(?P<obj>.+?) is the (?P<rel>.+) of (?P<subj>.+)$")
# "O is the R S is in"  (the corpus's second template)
_F_IN = re.compile(r"^(?P<obj>.+?) is the (?P<rel>\S+) (?P<subj>.+) is in$")
_ARTICLES = ("the ", "a ", "an ")
# Corpus questions run <200 chars; longer input is pathological, not a
# real question, so it's rejected before any regex work is attempted.
MAX_QUESTION_LEN = 512


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


# Causal phrasings -> the relation-of-entity forms the grammar already parses
# (2026-09-03, GoT challenge pre-check (k): "What is the cause of X?" and
# "What is the impact of X?" parsed, four common phrasings did not — a
# lexical gap, not a structural one). Rewritten before the regexes run:
#   what led to X / what caused X / what brought about X / why did X happen
#       -> what is the cause of X
#   what happened because of X / what did X lead to / what did X cause /
#   what resulted from X / what were the consequences of X
#       -> what is the impact of X
# "Did A cause B?" is a yes/no question, not a chain: left unparsed on purpose.
_CAUSAL_REWRITES = [
    (re.compile(r"^\s*what\s+(?:led|leads)\s+to\s+(?P<x>.+?)\s*\?\s*$", re.I), "what is the cause of {x}?"),
    (re.compile(r"^\s*what\s+(?:caused|causes|brought\s+about)\s+(?P<x>.+?)\s*\?\s*$", re.I), "what is the cause of {x}?"),
    (re.compile(r"^\s*what\s+(?:was|were|is|are)\s+the\s+causes?\s+of\s+(?P<x>.+?)\s*\?\s*$", re.I), "what is the cause of {x}?"),
    (re.compile(r"^\s*why\s+did\s+(?P<x>.+?)\s+(?:happen|occur)\s*\?\s*$", re.I), "what is the cause of {x}?"),
    (re.compile(r"^\s*what\s+happened\s+(?:because|as\s+a\s+result)\s+of\s+(?P<x>.+?)\s*\?\s*$", re.I), "what is the impact of {x}?"),
    (re.compile(r"^\s*what\s+did\s+(?P<x>.+?)\s+(?:lead\s+to|cause|bring\s+about)\s*\?\s*$", re.I), "what is the impact of {x}?"),
    (re.compile(r"^\s*what\s+(?:resulted|followed)\s+from\s+(?P<x>.+?)\s*\?\s*$", re.I), "what is the impact of {x}?"),
    (re.compile(r"^\s*what\s+(?:was|were|is|are)\s+the\s+(?:impacts?|consequences?|effects?|results?)\s+of\s+(?P<x>.+?)\s*\?\s*$", re.I), "what is the impact of {x}?"),
]


def normalize_causal(q: str) -> str:
    """Rewrite a causal phrasing onto 'what is the cause/impact of X?'; any other text is returned unchanged."""
    for pat, tmpl in _CAUSAL_REWRITES:
        m = pat.match(q)
        if m:
            # a leading article would read as a chain boundary ("cause of the X" splits on " of the ")
            x = re.sub(r"^(?:the|a|an)\s+", "", m.group("x").strip(), flags=re.I)
            return tmpl.format(x=x)
    return q


def parse_question(q: str) -> QuestionPlan | None:
    # Backtracking guard: bound worst-case input size, and reject anything
    # without a "?" up front (every frame requires one) — cheap and O(n),
    # so pathological "no closing ?" input never reaches the regexes below.
    if len(q) > MAX_QUESTION_LEN or "?" not in q:
        return None
    q = normalize_causal(q)

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
