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
from . import events as ev
from .pipeline import CoTResult, answer
from .plan_verify import _FRAME_AND_JOINT, KIND_OF_ASK, ask_type, split_tail
from .planner import QuestionPlan, Triple, normalize, parse_fact, parse_question

__wiring__ = Wiring.WIRED

# the refusals a fetch can address: the store lacks a fact, not the plan a shape
LEARNABLE = ("retrieval_exhausted", "unknown_relation", "latent_only")   # latent_only: an attesting source may lift the hold


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
    snapped: tuple[str, str] | None = None                # lever 7: (the emitted seed, the question's spelling it was snapped to)
    @property
    def accepted(self) -> list[Provenance]:
        return [p for p in self.learned if p.status == "accepted"]


def snapshot(store) -> str:
    """A fingerprint of the store's fact set before a write: the set hash of every text, kept
    INCREMENTALLY on the store (2026-09-13, the ask loop: hashing 552k sorted texts per gated
    fact cost 0.35 s a fact -- 172 facts, a minute). Order-independent (a sum of per-fact
    digests mod 2^128), so it equals the full recomputation and does not depend on which
    facts arrived when; a store that shrank or was replaced is rehashed from scratch."""
    texts = getattr(store, "texts", [])
    n = len(texts)
    cached = getattr(store, "_snap", None)          # (n_texts, acc, texts identity)
    if cached is not None and cached[2] is texts and cached[0] <= n:
        acc, start = cached[1], cached[0]
    else:
        acc, start = 0, 0
    for f in texts[start:] if isinstance(texts, list) else list(texts)[start:]:
        acc = (acc + int.from_bytes(hashlib.sha256(f.encode("utf-8")).digest()[:16], "big")) % (1 << 128)
    try:
        store._snap = (n, acc, texts)
    except AttributeError:
        pass
    return f"{acc:032x}"[:16]


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
    if first.reason == "latent_only":                       # the subjects of the held facts: what an attester is asked about
        for t in (h.triple for h in first.trace if h.fact in (first.refused or {}).get("latent_facts", ())):
            if t is not None and normalize(t.subj) not in {normalize(e) for e in out}:
                out.append(t.subj)
    _rel, ent = split_tail(plan.tail, known, question)
    if ent and normalize(ent) not in {normalize(e) for e in out}:
        out.append(ent)
    return out


def wanted_relations(question: str, plan: QuestionPlan | None, first: CoTResult, known, entity: str) -> list[str]:
    """The plan's wording for the hop the walk needs from `entity`: the relation after the
    last hop reached when the entity is the object the walk stopped at; the tail's relation
    when it is the seed. What a source may use to tell items of one name apart; [] when the
    plan does not say."""
    if plan is None:
        return []
    e = normalize(entity)
    if first.trace and first.reason == "retrieval_exhausted":
        last = first.trace[-1].triple
        if last is not None and normalize(last.obj) == e:
            k = len(first.trace)                            # hops walked; the next relation is plan.relations[k]
            if k < len(plan.relations) and plan.relations[k]:
                return [plan.relations[k]]
            return []
    rel, ent = split_tail(plan.tail, known, question)
    if ent and normalize(ent) == e and rel:
        return [rel]
    return []


def reached_through(first: CoTResult, entity: str) -> str | None:
    """The walked fact whose OBJECT is `entity` -- the last hop's, when the walk stopped at it --
    else None (the seed came from the question, not from a fact)."""
    if first.trace and first.reason == "retrieval_exhausted":
        last = first.trace[-1]
        if last.triple is not None and normalize(last.triple.obj) == normalize(entity):
            return last.fact
    return None


_TAKES_RELATIONS: dict[int, bool] = {}


def _takes_relations(source) -> bool:
    """Whether `source.facts` accepts `relations=` and `via=` (the Source protocol only promises `facts(entity)`)."""
    key = id(type(source))
    if key not in _TAKES_RELATIONS:
        import inspect
        try:
            params = inspect.signature(source.facts).parameters
            _TAKES_RELATIONS[key] = "relations" in params and "via" in params
        except (TypeError, ValueError):
            _TAKES_RELATIONS[key] = False
    return _TAKES_RELATIONS[key]


def snap_seed(question: str, plan: QuestionPlan, known, min_ratio: float = 0.85) -> tuple[QuestionPlan, tuple[str, str] | None]:
    """Lever 7 (2026-09-13, exp_r17): the emitter re-types the entity and garbles it -- 'karol
    burgmann' for Karl Brugmann, 'jean louis bastu' for Jean Louis Barthou -- and the plan dies
    for coverage with the right relation. The question is the only source of the entity's
    spelling: when the emitted seed is not in the question, the question's n-gram closest to it
    (difflib, >= `min_ratio`, a unique best, at least two words or six characters) replaces it,
    and the translation is on record. Everything after -- coverage, the walk, the VM -- runs on
    the snapped plan as on any other; a wrong snap is a plan the gate refuses."""
    import difflib
    rel, ent = split_tail(plan.tail, known, question)
    q = normalize(question)
    e = normalize(ent)
    if not e or f" {e} " in f" {q} ":
        return plan, None
    toks = q.split()
    n_e = len(e.split())
    best: list[tuple[float, str]] = []
    for n in range(max(1, n_e - 2), n_e + 3):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i + n])
            if g in _FRAME_AND_JOINT or (len(g) < 6 and n < 2):
                continue
            best.append((difflib.SequenceMatcher(None, e, g).ratio(), g))
    if not best:
        return plan, None
    best.sort(key=lambda x: -x[0])
    if best[0][0] < min_ratio or (len(best) > 1 and best[1][0] == best[0][0] and best[1][1] != best[0][1]):
        return plan, None
    snapped = best[0][1]
    tail = f"{rel} of {snapped}" if plan.tail.lower().startswith(rel.lower()) else plan.tail.replace(ent, snapped)
    return QuestionPlan(relations=plan.relations, tail=tail, n_hop=plan.n_hop, answer_class=plan.answer_class), (ent, snapped)


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
        # the resolvers in order, and the FIRST that names the wording decides (2026-09-13, the ask
        # loop): the property table said 'mother' IS a property; the store lacked it; the synonym
        # oracle behind it offered 'father' (WordNet's verb sense) and the plan was rewritten to a
        # relation the question never asked -- Pierre Trudeau, verified. A wording the first
        # resolver knows and the store lacks is a fact to FETCH, not a word to paraphrase.
        labels: list[str] = []; decided = None
        for x in resolvers:
            labels = [normalize(l) for l in x.relations(r)]
            if labels:
                decided = x
                break
        held = sorted({l for l in labels if l in known})
        if not held and decided is not None and hasattr(decided, "wordings"):
            # the store may hold the property under another of its wordings: the wiki world
            # says 'birth date' where Wikidata's label is 'date of birth' (2026-09-13). The
            # table names every wording of the label's property; a held one that names THAT
            # property alone is the target (a wording two properties share is no evidence of
            # either), and two held ('birth date' / 'birthplace' for 'born') is the same ambiguity
            held = sorted({normalize(w) for l in labels for w in decided.wordings(l)
                           if normalize(w) in known and len(decided.relations(w)) == 1})
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
    snapped = None
    qid = ev.emit("question", text=question, source=getattr(source, "name", None))     # the run's root event
    if plan is None:
        plan = parse_question(question)
    elif known is not None:
        plan, snapped = snap_seed(question, plan, known)         # lever 7: the question spells the entity
    cur = {"plan": ev.emit("plan", qid, **_plan_fields(plan), snapped=snapped)}
    latent = bool(getattr(source, "latent", False))       # a proposer of facts, not an attester (the LFM Source, 2026-09-13)
    prov = getattr(store, "provenance", None)
    fkey = getattr(store, "_key", None) or (lambda s: " ".join(s.split()))

    def replan(new_plan, how: str, rewrote) -> None:
        """A rewritten plan is a new plan event, child of the one it replaces."""
        cur["plan"] = ev.emit("plan", cur["plan"], **_plan_fields(new_plan), how=how, rewrote=[list(x) for x in rewrote])

    def walk():
        res = answer(question, retrieve, run_fn, tau_vm=tau_vm, tau_ret=tau_ret, top_k=top_k,
                     max_repairs=max_repairs, lookup=store.lookup, known=known, plan=plan, aliases=aliases)
        # the latent tier: a verified chain that rests on a fact only a latent source stated is not spoken;
        # the would-be answer and the facts are on record, and a second source's agreement lifts the hold
        if res.verified and prov is not None:
            held = [h.fact for h in res.trace if h.fact and prov.get(fkey(h.fact), "").endswith("(latent)")]
            if held:
                res = CoTResult(answer=None, verified=False, trace=res.trace, reason="latent_only",
                                refused={"answer": res.answer, "latent_facts": held}, source=res.source)
        ev.emit_walk(cur["plan"], res, provenance=prov, key=fkey)
        return res
    pre: list[tuple[str, str]] = []; amb0 = None
    if plan is not None and known is not None and hasattr(source, "relations"):
        # lever 4 AHEAD of the walk (2026-09-13, hdc): a plan wording the store does not hold
        # exactly, but the table resolves to exactly one held label, is translated before any
        # paraphrase tier can read it. 'administrative territorial entity' is P131's own alias;
        # the walk's overlap tier (Jaccard >= 0.6) had matched it to P150, 'contains
        # administrative territorial entity' -- the INVERSE -- and a store holding both would
        # have verified the wrong direction. Exact evidence outranks overlap; and a wording the
        # table says names two held labels is the ambiguity it always was, never the overlap's pick.
        labels, _ent = _plan_labels(plan, known, question)
        unheld = [r for r in labels if r not in known]
        if unheld:
            plan2, pre, amb0 = resolve_relations(plan, unheld, [source] + list(resolvers or []), known, aliases,
                                                 question=question)
            if amb0 is None and pre:
                plan = plan2; replan(plan, "lever 4 (exact alias, ahead of the walk)", pre)
    if amb0 is not None:
        first = CoTResult(answer=None, verified=False, reason="ambiguous_relation", refused=amb0)
        ev.emit("answer", qid, verified=False, answer=None, reason="ambiguous_relation", refused=amb0)
        return LearnResult(result=first, first=first, plan=plan, snapped=snapped)
    first = walk()
    out = LearnResult(result=first, first=first, plan=plan, snapped=snapped)
    out.aliased.extend(pre)
    # lever 6: a coverage refusal of a plan whose relations the store HOLDS may be the
    # proposer naming them by their canonical label; the question's own wording is
    # asked of the resolvers, recorded as an alias, and the plan walked once more
    if first.reason == "plan_does_not_cover_question" and plan is not None:
        worded, amb = resolve_wordings(question, plan, [source] + list(resolvers or []), known, aliases)
        if amb is not None:
            out.result = CoTResult(answer=None, verified=False, reason="ambiguous_relation", refused=amb)
            ev.emit("answer", qid, verified=False, answer=None, reason="ambiguous_relation", refused=amb)
            return out
        if worded:
            out.aliased.extend(worded)
            ev.emit("alias", cur["plan"], how="lever 6 (the question's own wording)", pairs=[list(x) for x in worded])
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
                replan(plan, "lever 4 (the source names the relation)", rewrote)
                out.result = walk()
                continue
        if out.result.reason == "retrieval_exhausted" and plan is not None and not siblings_tried:
            # a held relation that found nothing: the store may hold the property under its
            # other wording ('birthplace' / 'place of birth'); one translation, one more walk
            siblings_tried = True
            plan2, rewrote = resolve_siblings(plan, [source] + list(resolvers or []), known, aliases, question=question)
            if rewrote:
                out.aliased.extend(rewrote); plan = plan2; out.plan = plan
                replan(plan, "siblings (the property's other held wording)", rewrote)
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
        # the relation the walk needs from this entity (the stalled hop's wording): a source that
        # can use it decides among several ITEMS sharing the name by which of them carries it --
        # the question decides, never a rank (2026-09-13, the ask loop: 'Marie Curie' is a
        # physicist, a book edition, a metro station and a ferry; one has a date of birth)
        need = wanted_relations(question, plan, out.result, known, ent)
        via = reached_through(out.result, ent)                 # the walked fact whose object this entity is, when it is one
        if _takes_relations(source):
            items = list(source.facts(ent, relations=need or None, via=via))
        else:
            items = list(source.facts(ent))
        last = getattr(source, "last", None) if isinstance(getattr(source, "last", None), dict) else {}
        fid = ev.emit("fetch", qid, source=getattr(source, "name", None), entity=ent, n=len(items), latent=latent,
                      needs=need, via=via, how=last.get("how"), item=last.get("qid"),
                      ambiguous=[list(a) for a in last.get("ambiguous", [])] or None)
        for item in items:
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
            lifted = False
            if p.status == "accepted":
                store.add(fact)
                if hasattr(known, "add"):
                    known.add(fact)
                if prov is not None:
                    prov[fkey(fact)] = source.name + (" (latent)" if latent else "")
                admitted += 1
            elif p.status == "duplicate" and prov is not None and not latent and prov.get(fkey(fact), "").endswith("(latent)"):
                # a second, attesting source states the latent fact: the hold is lifted, both names on record
                prov[fkey(fact)] = prov[fkey(fact)][:-len(" (latent)")] + "+" + source.name
                admitted += 1; lifted = True                    # new knowledge about the fact, so the walk runs again
            ev.emit("gate", fid, fact=fact, status=p.status, clash=p.clash, lifted=lifted,
                    provenance=(prov or {}).get(fkey(fact)))
        if not admitted:
            break
        resolved.clear()          # new facts may have brought the relation the source names
        out.result = walk()
    ev.emit("answer", qid, verified=out.result.verified, answer=out.result.answer, reason=out.result.reason,
            refused=out.result.refused if isinstance(out.result.refused, dict) else None,
            aliased=[list(x) for x in out.aliased], entities=list(out.entities), fetched=out.fetched)
    return out


def _plan_fields(plan: QuestionPlan | None) -> dict:
    if plan is None:
        return {"seed": None, "relations": [], "tail": None, "n_hop": 0}
    return {"seed": plan.tail.rsplit(" of ", 1)[1] if " of " in plan.tail else None,
            "relations": [plan.tail.rsplit(" of ", 1)[0] if " of " in plan.tail else plan.tail] + [r for r in plan.relations[1:] if r],
            "tail": plan.tail, "n_hop": plan.n_hop}
