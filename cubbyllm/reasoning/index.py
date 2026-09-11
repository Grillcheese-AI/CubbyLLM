"""The triple index: retrieval as LOOKUP for template facts.

Wired: WIRED — `pipeline._walk` looks up here first and searches (cosine +
tau_ret) only for the hops the index misses; the stand-in `FactStore`
maintains one at every `add()`.

Why (validation/exp_m4_triple_lookup.py, 2026-09-04): the walk's acceptance
test is EXACT on the parsed triple — the tail string at hop 0, relation +
subject afterwards — but its candidates came from cosine top-3 above tau_ret.
On the 800-question harvest 95.7% of the store parses as triples; the same
acceptance run as an index lookup serves 83.8% of hops (= the search ceiling,
cosine@50) where the live cosine@3>=tau served 78.7%; end to end 581/800 vs
517 verified, ambiguity (>1 candidate) on 0.8% of hops. The remaining misses
are the planner rejecting the gold fact, not retrieval.

Cosine keeps its jobs: paraphrase (questions the grammar does not parse),
world routing, the flat fallback, and the tie-break when a lookup is
ambiguous. Facts that do not parse as triples are counted, not indexed —
they stay reachable by search.
"""
from __future__ import annotations

from collections import defaultdict

from ..core.protocols import Wiring
from .planner import QuestionPlan, Triple, accepts, normalize, parse_fact, reused_relations, tail_splits

__wiring__ = Wiring.WIRED


class TripleIndex:
    """(relation, subject) -> facts, over the facts that parse as triples.

    `hop(plan, hop, entity)` returns every indexed fact that `planner.accepts`
    for that hop, in insertion order — the same test `pipeline._walk` applies
    to search candidates, so a lookup hit is accepted by construction.
    `by_object(entity)` is the backward entry (the object side) for a
    bidirectional walk; unused by the forward walk today.
    """

    def __init__(self, facts=()) -> None:
        self._by_tail: dict[str, list[tuple[str, Triple]]] = defaultdict(list)   # normalize("rel of subj")
        self._by_subj: dict[str, list[tuple[str, Triple]]] = defaultdict(list)   # normalize(subj)
        self._by_obj: dict[str, list[tuple[str, Triple]]] = defaultdict(list)    # normalize(obj)
        self._rel_n: dict[str, int] = defaultdict(int)                            # facts per normalize(rel)
        # the relations the store reuses (>= 2 facts): `parse_fact` disambiguates the
        # ' of ' split with them (lever 1b, 2026-09-11). Seeded by a greedy first pass
        # over the facts given here, then kept current by `add`; a fact that arrives
        # live before its relation is reused takes the greedy split (order-dependent
        # for the first two facts of a relation, never afterwards).
        self._reused: set[str] = set()
        self._seen: set[str] = set()
        self.n_facts = 0          # every fact offered (deduped)
        self.n_parsed = 0         # the indexed subset
        facts = list(facts)
        self._reused = reused_relations(facts)
        for f in facts:
            self.add(f)

    def __len__(self) -> int:
        return self.n_parsed

    def declare_relation(self, rel: str) -> None:
        """A relation a SOURCE states structurally (search-and-learn hands over
        triples, not strings): the split prefers it from the first fact on. exp_r11
        alias run (2026-09-11): 'date of birth of Masaki Tsuji' split as 'date' |
        'birth of Masaki Tsuji' because the wiki world reuses a relation 'date'
        (2 facts) and had never seen 'date of birth'; declared, the longest known
        prefix is the right one."""
        self._reused.add(normalize(rel))

    def add(self, fact: str) -> Triple | None:
        """Index one fact; the parsed triple, or None when it is a duplicate or not a template fact."""
        key = " ".join(fact.split())
        if not key or key in self._seen:
            return None
        self._seen.add(key)
        self.n_facts += 1
        t = parse_fact(key, known=self._reused)
        if t is None:
            return None
        self.n_parsed += 1
        entry = (key, t)
        self._by_tail[normalize(f"{t.rel} of {t.subj}")].append(entry)
        self._by_subj[normalize(t.subj)].append(entry)
        self._by_obj[normalize(t.obj)].append(entry)
        r = normalize(t.rel)
        self._rel_n[r] += 1
        if self._rel_n[r] >= 2:
            self._reused.add(r)
        return t

    def __contains__(self, fact: str) -> bool:
        return " ".join(fact.split()) in self._seen

    def hop(self, plan: QuestionPlan, hop: int, entity: str | None) -> list[tuple[str, Triple]]:
        """Every indexed fact that serves hop `hop` of `plan` from `entity` (None at hop 0)."""
        if hop == 0:
            # exact tier first (the tail as one key), then the paraphrase tier: the
            # facts whose SUBJECT is the entity at some ' of ' split of the tail,
            # filtered by `accepts` (relation_matches on the relation part).
            # Exact hits lead so the walk's tie-break never prefers a paraphrase
            # over an exact match; `_seen`-order within a tier, deduped. The
            # reuse guard: a paraphrase is only served for a relation the store
            # states in >= 2 facts (plan_verify.StoreRelations.reused says why:
            # a one-off 'relation' is usually an entity fragment mis-split at an
            # ' of ', and matching it verified a wrong answer).
            exact = self._by_tail.get(normalize(plan.tail), [])
            cands, seen = list(exact), {f for f, _t in exact}
            for _rel, ent in tail_splits(plan.tail):
                for f, t in self._by_subj.get(ent, []):
                    if f not in seen and self._rel_n[normalize(t.rel)] >= 2:
                        cands.append((f, t)); seen.add(f)
        else:
            cands = self._by_subj.get(normalize(entity or ""), [])
        return [(f, t) for f, t in cands if accepts(plan, hop, entity, t)]

    def by_object(self, entity: str) -> list[tuple[str, Triple]]:
        return list(self._by_obj.get(normalize(entity), []))
