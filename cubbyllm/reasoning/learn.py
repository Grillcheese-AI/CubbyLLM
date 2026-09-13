"""learn -- search-and-learn: a refusal becomes a fetch, a gate, a store write and a second walk.

Wired: WIRED (2026-09-11, coverage lever 3; the host side of the 2026-09-04 design note
"I don't know yet -- but I can look it up for you", standin/README.md).

The loop, and what it may and may not do:

    answer(question)                         -> refused / failed with a reason
      -> the entity the walk stalled on      (the seed at hop 0, or the last object reached)
      -> source.facts(entity)                -> candidate template facts, DATA, never a judgement
      -> gate(fact, store): parse, duplicate, sibling recorded  (multi-valued relations are admitted; a
                                             conflict is refused at ANSWER time as an ambiguous hop)
      -> store.add(fact) + known.add(fact)   with PROVENANCE: source, entity queried, time, the
                                             store's snapshot hash before the write, git-free
      -> answer(question) again, once        -> verified by the VM, or refused with a reason

Invariants kept: the VM is the only truth gate (a fetched fact is looked up and verified like any
other; nothing is spoken because a source said so); the model proposes, the host disposes (the
source is chosen and called by the host, its output is gated by the host); the don't-know
contract (a refusal stays a refusal until a verified chain exists); retire, never delete (every
accepted fact carries a provenance record; a later retirement is a record, not a deletion). At
most `max_entities` rounds per question, each one entity, each followed by one walk, and a round
that admits nothing ends it: the loop is bounded by construction.

`Source` is a protocol: `facts(entity) -> list[str]` of template facts ('OBJ is the REL of SUBJ').
A held-out store (validation/exp_r11_search_learn.py) measures the mechanism; a Wikidata-backed
source is the same call with the network behind it.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..core.protocols import Wiring
from .pipeline import CoTResult, answer
from .plan_verify import _FRAME_AND_JOINT, KIND_OF_ASK, ask_type, split_tail
from .planner import QuestionPlan, Triple, normalize, parse_fact, parse_question

__wiring__ = Wiring.WIRED

# the refusals a fetch can address: the store lacks a fact, not the plan a shape
LEARNABLE = ("retrieval_exhausted", "unknown_relation")


@runtime_checkable
class Source(Protocol):
    name: str
    def facts(self, entity: str) -> list[str]: ...
    # optional (lever 4): the canonical relation labels a wording names in this source --
    # 'born' -> ['date of birth'], 'citizenship' -> ['country of citizenship']; [] if none.
    # def relations(self, text: str) -> list[str]: ...


@dataclass
class Provenance:
    fact: str
    source: str
    entity: str
    fetched_at: float
    snapshot_before: str          # sha256 over the store's fact set before this write
    status: str                   # accepted | duplicate | contradiction | unparseable
    clash: str | None = None      # the stored fact a contradiction hit


@dataclass
class LearnResult:
    result: CoTResult             # the final answer() result (second walk when one ran)
    first: CoTResult              # the first walk's result
    learned: list[Provenance] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)     # what was asked of the source
    fetched: int = 0
    # lever 4: (the plan's relation words, the store relation the host rewrote them to)
    aliased: list[tuple[str, str]] = field(default_factory=list)
    plan: QuestionPlan | None = None                      # the plan that was walked last (rewritten or not)
    @property
    def accepted(self) -> list[Provenance]:
        return [p for p in self.learned if p.status == "accepted"]


def snapshot(store) -> str:
    h = hashlib.sha256()
    for f in sorted(getattr(store, "texts", [])):
        h.update(f.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()[:16]


def contradiction(fact: str, index) -> str | None:
    """A stored fact with the same (subject, relation) and a different object, else
    None -- the memory cortex's rule (standin/serve.py MemoryCortex.contradiction),
    answered from the index instead of a scan."""
    t = parse_fact(fact)
    if t is None:
        return None
    for known, k in index._by_subj.get(normalize(t.subj), []):
        if normalize(k.rel) == normalize(t.rel) and normalize(k.obj) != normalize(t.obj):
            return known
    return None


def gate(fact: str, store, source: str, entity: str, functional: bool = False) -> Provenance:
    """Parse, duplicate, sibling -- in that order; the verdict is the record.

    A SIBLING is a stored fact with the same subject and relation and a different
    object. With `functional=True` it is a contradiction and the fact is refused
    (the memory cortex's rule for a user-taught fact). By default it is admitted
    and recorded (`clash` names the sibling): a source states multi-valued
    relations as sets -- three citizenships, a population per census -- and
    exp_r11 (2026-09-11) showed the functional rule refusing 1,069 of 1,919
    fetched facts as 'contradictions', keeping ONE population value and letting
    the walk speak it for 'as of 2022'. The honest place to resolve several
    values is answer time: `pipeline._walk` refuses an ambiguous hop and names
    the candidates. A poisoned value therefore becomes a refusal with both
    facts and their provenance on record, never an answer."""
    snap = snapshot(store)
    now = time.time()
    if parse_fact(fact) is None:
        return Provenance(fact, source, entity, now, snap, "unparseable")
    if fact in store:
        return Provenance(fact, source, entity, now, snap, "duplicate")
    clash = contradiction(fact, store.index)
    if clash is not None and functional:
        return Provenance(fact, source, entity, now, snap, "contradiction", clash=clash)
    return Provenance(fact, source, entity, now, snap, "accepted", clash=clash)


def stalled_entities(question: str, plan: QuestionPlan | None, first: CoTResult, known) -> list[str]:
    """Where the walk stopped: the seed entity when nothing was found at hop 0 (or the
    plan was refused for an unknown relation), else the last object the walk reached."""
    if plan is None:
        plan = parse_question(question)
    if plan is None:
        return []
    out: list[str] = []
    if first.trace and first.reason == "retrieval_exhausted":
        last = first.trace[-1].triple
        if last is not None:
            out.append(last.obj)
    _rel, ent = split_tail(plan.tail, known, question)
    if ent and normalize(ent) not in {normalize(e) for e in out}:
        out.append(ent)
    return out


def narrow_by_ask(question: str | None, candidates: list[str], resolvers) -> list[str]:
    """The typed answer class applied where a wording names several held labels
    (2026-09-13: the local property table says 'born' is an alias of BOTH `date of
    birth` and `place of birth`, where the API's search had ranked one). A *when*
    question keeps the date-valued labels, a *how many* the number-valued, a *who* the
    entity-valued; the value kind is the property's DATATYPE, read from the table by
    `resolver.kind(label)`, never the model's pick or the API's rank. Exactly one must
    remain, and every candidate must have a known kind -- a label the table cannot type
    might be date-valued too, and choosing among those would be a pick. Otherwise the
    candidates come back as they were and the ambiguity stands."""
    if not question or len(candidates) < 2:
        return candidates
    asked = ask_type(question)
    if asked is None:
        return candidates
    typed = [x for x in resolvers if hasattr(x, "kind")]
    if not typed:
        return candidates
    kinds: dict[str, str | None] = {}
    for label in candidates:
        kinds[label] = next((k for k in (x.kind(label) for x in typed) if k is not None), None)
    if any(k is None for k in kinds.values()):
        return candidates
    keep = [label for label in candidates if kinds[label] == KIND_OF_ASK[asked]]
    return keep if len(keep) == 1 else candidates


def resolve_siblings(plan: QuestionPlan, source, known, aliases: dict[str, list[str]],
                     question: str | None = None) -> tuple[QuestionPlan, list[tuple[str, str]]]:
    """Lever 4 for a HELD relation whose walk found nothing (2026-09-13, the gen-3 builder):
    the store holds one property under two of its wordings -- the wiki world's `birthplace`
    and the encyclopedia's `place of birth` -- and the entity's fact sits under the other.
    The table names the property's wordings; when exactly one OTHER wording naming that
    property alone is held (the ask type narrowing two), the plan is translated to it and
    walked once more. None or several: the plan is left as it was, and the loop fetches."""
    resolvers = [x for x in (source if isinstance(source, (list, tuple)) else [source])
                 if hasattr(x, "relations") and hasattr(x, "wordings")]
    if not resolvers:
        return plan, []
    labels, ent = _plan_labels(plan, known, question)
    rels = list(plan.relations); tail = plan.tail; rewrote: list[tuple[str, str]] = []
    for hop, r in enumerate(labels):
        if r not in known:
            continue
        props = [normalize(l) for x in resolvers for l in x.relations(r)]
        sibs = sorted({normalize(w) for x in resolvers for l in props for w in x.wordings(l)
                       if normalize(w) in known and normalize(w) != r and len(x.relations(w)) == 1})
        if len(sibs) > 1:
            sibs = narrow_by_ask(question, sibs, resolvers)
        if len(sibs) != 1:
            continue
        sib = sibs[0]
        aliases.setdefault(sib, []).extend([r] + list(aliases.get(r, [])))   # the question's own words travel with the relation
        rewrote.append((r, sib))
        if hop == 0:
            tail = f"{sib} of {ent}"                               # the tail is 'relation of seed'; the seed stays
        else:
            for i, x in enumerate(rels):
                if x and normalize(x) == r:
                    rels[i] = sib
    if not rewrote:
        return plan, []
    return QuestionPlan(relations=rels, tail=tail, n_hop=plan.n_hop, answer_class=plan.answer_class), rewrote


def resolve_relations(plan: QuestionPlan, unknown: list[str], source, known,
                      aliases: dict[str, list[str]], question: str | None = None) -> tuple[QuestionPlan, list[tuple[str, str]], dict | None]:
    """Lever 4 (2026-09-11, exp_r11): the emitter names a relation in the QUESTION's
    words ('born'); the source states it under its own label ('date of birth'), and the
    two share no word, so neither paraphrase tier can bridge them. The source resolves
    wording to canonical labels the way it resolves an entity label to an item
    (`source.relations`), the host keeps the labels the store HOLDS, and rewrites the
    plan into that wording -- the model proposed, the host translated, and the
    translation is on record (`aliases`, so `covers()` still sees the original words).
    Exactly one label must survive: several ('position' -> location / ranking) is an
    ambiguity the host refuses, never resolves -- unless the question's ask type and
    the labels' datatypes leave exactly one (`narrow_by_ask`); none leaves the plan as
    it was."""
    resolvers = [x for x in (source if isinstance(source, (list, tuple)) else [source]) if hasattr(x, "relations")]
    if not resolvers:
        return plan, [], None
    rels = list(plan.relations); tail = plan.tail; rewrote: list[tuple[str, str]] = []
    for r in unknown:
        labels = [normalize(l) for x in resolvers for l in x.relations(r)]
        held = sorted({l for l in labels if l in known})
        if not held:
            # the store may hold the property under another of its wordings: the wiki world
            # says 'birth date' where Wikidata's label is 'date of birth' (2026-09-13). The
            # table names every wording of the label's property; a held one that names THAT
            # property alone is the target (a wording two properties share is no evidence of
            # either), and two held ('birth date' / 'birthplace' for 'born') is the same ambiguity
            held = sorted({normalize(w) for x in resolvers if hasattr(x, "wordings")
                           for l in labels for w in x.wordings(l)
                           if normalize(w) in known and len(x.relations(w)) == 1})
        if not held:
            continue
        held = narrow_by_ask(question, held, resolvers)
        if len(held) > 1:
            return plan, rewrote, {"relation": r, "candidates": held}
        label = held[0]
        aliases.setdefault(label, []).extend([r] + list(aliases.get(normalize(r), [])))
        rewrote.append((r, label))
        for i, x in enumerate(rels):
            if x and normalize(x) == normalize(r):
                rels[i] = label
        if tail.lower().startswith(r.lower() + " of "):          # the tail's relation prefix
            tail = label + tail[len(r):]
    if not rewrote:
        return plan, [], None
    return QuestionPlan(relations=rels, tail=tail, n_hop=plan.n_hop, answer_class=plan.answer_class), rewrote, None


def _plan_labels(plan: QuestionPlan, known, question: str | None = None) -> tuple[list[str], str]:
    """The plan's relations (hop 0 first) and its seed entity, normalized."""
    tr, ent = split_tail(plan.tail, known, question)
    return [normalize(tr)] + [normalize(r) for r in plan.relations[1:] if r], normalize(ent)


def resolve_wordings(question: str, plan: QuestionPlan, source, known, aliases: dict[str, list[str]],
                     max_n: int = 3, max_candidates: int = 24) -> tuple[list[tuple[str, str]], dict | None]:
    """Lever 6 (2026-09-12, exp_r11 Gemini run 1): lever 4 in reverse. A proposer that
    names a relation by the source's CANONICAL label ('date of birth' for 'was born')
    is refused for coverage: the question carries none of the label's words, and the
    alias step never fires (the label is held; or it is not, and coverage is judged
    before the relation is, so the fetch that would bring it never runs) -- 114 of a
    frontier model's 121 plans died there, most of them right. The host asks the resolvers whether a wording IN
    THE QUESTION names the plan's label: the question's content n-grams (frame words,
    the seed entity and numbers excluded), shortest first, and the first that resolves
    to the label is recorded as its alias, so `covers()` sees it, and the plan is
    walked again. The wording must resolve to exactly ONE held label -- one naming two
    ('position') is the ambiguity lever 4 refuses, and the model's pick between the two
    is not evidence. The residual rule is untouched: a dropped hop still fails."""
    resolvers = [x for x in (source if isinstance(source, (list, tuple)) else [source]) if hasattr(x, "relations")]
    if not resolvers:
        return [], None
    labels, ent = _plan_labels(plan, known, question)
    q = normalize(question)
    text = q.replace(ent, " ") if ent else q
    toks = text.split()
    grams: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if all(w in _FRAME_AND_JOINT for w in g) or any(w.isdigit() for w in g):
                continue
            s = " ".join(g)
            if s not in grams:
                grams.append(s)
    grams = grams[:max_candidates]
    found: list[tuple[str, str]] = []
    for label in labels:
        if label in q or any(normalize(w) in q for w in aliases.get(label, ())):
            continue                                   # the question already says it
        # a label the store does not hold yet is asked about too: coverage is judged
        # before the relation is, so a plan naming 'inception' for 'founded' would
        # otherwise never reach the unknown_relation refusal that fetches the fact
        # that brings the label (exp_r11 Gemini run 3: 180 coverage refusals, most of
        # them this deadlock)
        for g in grams:
            named = {normalize(l) for x in resolvers for l in x.relations(g)}
            if label not in named:
                continue
            candidates = narrow_by_ask(question, sorted({l for l in named if l in known} | {label}), resolvers)
            if len(candidates) > 1:
                return found, {"wording": g, "candidates": candidates}
            if candidates != [label]:
                continue                                   # the ask type kept the OTHER label: this wording is not the plan's
            aliases.setdefault(label, []).append(g)
            found.append((g, label))
            break
    return found, None


def learn_and_answer(question: str, retrieve, run_fn, *, store, known, source: Source,
                     tau_vm: float, tau_ret: float = 0.0, top_k: int = 3, max_repairs: int = 1,
                     plan: QuestionPlan | None = None, max_entities: int = 2,
                     resolvers: list | None = None) -> LearnResult:
    """One question through the loop. `store` needs `add(fact)`, `__contains__`,
    `texts`, `index` (a TripleIndex) and `lookup` (its `index.hop`); `known` is a
    `StoreRelations` (gets `add`). A walk follows every round that admitted a fact;
    a round that admits nothing ends the loop; `max_entities` rounds at most. An
    `unknown_relation` refusal that the source can resolve to ONE relation the store
    holds is walked again with the plan rewritten (lever 4); it costs no round."""
    aliases: dict[str, list[str]] = {}
    if plan is None:
        plan = parse_question(question)
    def walk():
        return answer(question, retrieve, run_fn, tau_vm=tau_vm, tau_ret=tau_ret, top_k=top_k,
                      max_repairs=max_repairs, lookup=store.lookup, known=known, plan=plan, aliases=aliases)
    first = walk()
    out = LearnResult(result=first, first=first, plan=plan)
    # lever 6: a coverage refusal of a plan whose relations the store HOLDS may be the
    # proposer naming them by their canonical label; the question's own wording is
    # asked of the resolvers, recorded as an alias, and the plan walked once more
    if first.reason == "plan_does_not_cover_question" and plan is not None:
        worded, amb = resolve_wordings(question, plan, [source] + list(resolvers or []), known, aliases)
        if amb is not None:
            out.result = CoTResult(answer=None, verified=False, reason="ambiguous_relation", refused=amb)
            return out
        if worded:
            out.aliased.extend(worded)
            out.result = walk()
    asked: set[str] = set(); resolved: set[str] = set(); siblings_tried = False
    # bounded: each round asks the source about ONE entity the walk stalled on, stores what
    # the gate admits, and walks again; at most `max_entities` rounds, and a round that
    # admits nothing ends the loop (there is nothing new to walk on)
    while not out.result.verified and out.result.reason in LEARNABLE:
        if out.result.reason == "unknown_relation" and plan is not None:
            unknown = [r for r in (out.result.refused or {}).get("unknown_relations", []) if r not in resolved]
            resolved.update(unknown)
            plan2, rewrote, amb = resolve_relations(plan, unknown, [source] + list(resolvers or []), known, aliases,
                                                    question=question)
            if amb is not None:
                out.result = CoTResult(answer=None, verified=False, reason="ambiguous_relation", refused=amb)
                break
            if rewrote:
                out.aliased.extend(rewrote); plan = plan2; out.plan = plan
                out.result = walk()
                continue
        if out.result.reason == "retrieval_exhausted" and plan is not None and not siblings_tried:
            # a held relation that found nothing: the store may hold the property under its
            # other wording ('birthplace' / 'place of birth'); one translation, one more walk
            siblings_tried = True
            plan2, rewrote = resolve_siblings(plan, [source] + list(resolvers or []), known, aliases, question=question)
            if rewrote:
                out.aliased.extend(rewrote); plan = plan2; out.plan = plan
                out.result = walk()
                continue
        if len(asked) >= max_entities:
            break
        ents = [e for e in stalled_entities(question, plan, out.result, known) if normalize(e) not in asked]
        if not ents:
            break
        ent = ents[0]
        asked.add(normalize(ent)); out.entities.append(ent)
        admitted = 0
        for item in source.facts(ent):
            out.fetched += 1
            # a source that hands over a Triple knows where its relation ends; the store
            # is told before the string is split ('date of birth', not 'date' | 'birth of X')
            if isinstance(item, Triple):
                fact = f"{item.obj} is the {item.rel} of {item.subj}"
                if hasattr(store, "index") and hasattr(store.index, "declare_relation"):
                    store.index.declare_relation(item.rel)
                if hasattr(known, "declare"):
                    known.declare(item.rel)
            else:
                fact = item
            p = gate(fact, store, source.name, ent)
            out.learned.append(p)
            if p.status == "accepted":
                store.add(fact)
                if hasattr(known, "add"):
                    known.add(fact)
                admitted += 1
        if not admitted:
            break
        resolved.clear()          # new facts may have brought the relation the source names
        out.result = walk()
    return out
