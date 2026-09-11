"""Retrieval for the chain walk: lookup first, search second, acceptance always.

Wired: WIRED — `pipeline._walk`'s retrieval seam. The scorer is INJECTED, so
this module has no dependency on any encoder, table or corpus: the validation
scripts pass a fastword cosine closure, the tests pass a table of fixed scores,
and the component under test is the same object in both cases.

Why this exists (it did not, until now)
---------------------------------------
`TripleIndex` was the only retrieval component in the package. Retrieval-as-
search lived inside `validation/exp_m3_cot_pipeline.py` as a per-run closure
(`make_retriever`), and `validation/exp_m4_triple_lookup.py` carried a SECOND
copy of the index — so the numbers everything rests on were measured by code
the shipped walk does not call. (The two index copies were checked equivalent
on 1,041 hops over a 78% store, 0 divergences — `docs/research/
2026-09-11-index-duplication.md` — but equivalence today is not a guarantee.)

What the measurements say this has to do
----------------------------------------
  ENTRY IS THE PRODUCT.  61% of the failures that survive lookup-first never
  produce a hop-0 seed at all (`exp_r4_index_walk`). Later-hop traversal is
  comparatively healthy and the index already serves 83.8% of hops. So hop 0
  gets the fallback ladder; later hops mostly do not need it.

  NO THRESHOLD AT THE ENTRY.  Lookup-first beat cosine 0.710 vs 0.646, +51
  verified and 0 lost, *because* the index applies no threshold — it absorbed
  34 of the 36 `threshold_bound` failures (`exp_r1_gate_diagnostics` Q5). So
  `tau` defaults to None and a caller has to opt in to losing those again.

  SCORE NEVER GATES.  On the threshold_bound cases where a margin was
  computable, BOTH had a distractor outranking the gold — 0.2876 against
  0.9445 (`exp_r1_gate_diagnostics` Q2). Acceptance decides WHAT is eligible;
  score only orders what acceptance already allowed.

  A MISS MUST EXPLAIN ITSELF.  `candidates_topk` was empty for 152 of 246
  failures, so the diagnostic for why retrieval failed was missing for 62% of
  the cases where it failed (`exp_r1_gate_diagnostics` Q1). Every `HopResult`
  carries its candidates and a `reason`, hit or miss.

Contract: `validation/test_retriever_contract.py` (16 pins, mutation-checked).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..core.protocols import Wiring
from .index import TripleIndex
from .planner import QuestionPlan, Triple, accepts, normalize, parse_fact

__wiring__ = Wiring.WIRED


class Scorer(Protocol):
    """query, fact -> similarity. The only thing an encoder is needed for."""

    def __call__(self, query: str, fact: str) -> float: ...


def batch_scorer(retrieve: Callable[[str, int], list[tuple[float, str]]],
                 top_k: int = 50) -> Scorer:
    """Adapt a `make_retriever`-style top-k closure to a pairwise Scorer.
    Memoized per query, so a hop costs one encode, not one per fact."""
    cache: dict[str, dict[str, float]] = {}

    def score(query: str, fact: str) -> float:
        row = cache.get(query)
        if row is None:
            row = cache[query] = {f: float(s) for s, f in retrieve(query, top_k)}
        return row.get(fact, 0.0)

    return score


@dataclass(frozen=True)
class HopResult:
    """The outcome of one hop. `fact is None` is a miss, never an exception."""
    fact: str | None
    triple: Triple | None
    source: str                       # lookup | lookup_backward | search | miss
    reason: str | None                # why a miss happened; None on a hit
    candidates: list[tuple[float, str]] = field(default_factory=list)
    ambiguous: bool = False           # >1 accepted candidate (0.8% of hops)

    def __bool__(self) -> bool:
        return self.fact is not None


class Retriever:
    """Lookup first, backward edge, then scored search. Acceptance gates every
    path; score only orders within what acceptance allowed.

    `tau` is None by default ON PURPOSE — see the module docstring. Passing a
    float reinstates a hard floor on the SEARCH path only; lookup never has one.
    """

    def __init__(self, store, scorer: Scorer, index: TripleIndex | None = None,
                 top_k: int = 50, tau: float | None = None) -> None:
        self.store: list[str] = list(store)
        self.index = index if index is not None else TripleIndex(self.store)
        self.scorer = scorer
        self.top_k = int(top_k)
        self.tau = tau

    # -- internals ---------------------------------------------------------
    def _query(self, plan: QuestionPlan, hop: int, entity: str | None) -> str:
        """The query `pipeline._walk` issues. Kept here so there is one copy."""
        return f"what is the {plan.tail}" if hop == 0 else f"{entity} {plan.relations[hop]}"

    def _ranked(self, query: str) -> list[tuple[float, str]]:
        """Score desc, then fact text: the tie-break must not depend on store order."""
        return sorted(((float(self.scorer(query, f)), f) for f in self.store),
                      key=lambda sf: (-sf[0], sf[1]))[:self.top_k]

    @staticmethod
    def _best(cands, rank: dict[str, float]):
        """Highest-scoring accepted candidate; ties broken by fact text."""
        return max(cands, key=lambda ft: (rank.get(ft[0], 0.0), ft[0]))

    # -- the seam ----------------------------------------------------------
    def hop(self, plan: QuestionPlan, hop: int, entity: str | None,
            bidirectional: bool = True) -> HopResult:
        query = self._query(plan, hop, entity)
        ranked = self._ranked(query)          # ALWAYS computed: it is the diagnostic
        rank = {f: s for s, f in ranked}

        hits = self.index.hop(plan, hop, entity)                    # exact, no threshold
        if hits:
            f, t = self._best(hits, rank)
            return HopResult(f, t, "lookup", None, ranked, ambiguous=len(hits) > 1)

        if bidirectional and entity is not None:                    # the edge that already existed
            back = self.index.by_object(entity)
            if back:
                f, t = self._best(back, rank)
                return HopResult(f, t, "lookup_backward", None, ranked, ambiguous=len(back) > 1)

        eligible = [(f, t) for f, t in ((f, parse_fact(f)) for _s, f in ranked)
                    if t is not None and accepts(plan, hop, entity, t)]
        if self.tau is not None:
            eligible = [(f, t) for f, t in eligible if rank.get(f, 0.0) >= self.tau]
        if eligible:
            f, t = self._best(eligible, rank)
            return HopResult(f, t, "search", None, ranked, ambiguous=len(eligible) > 1)

        return HopResult(None, None, "miss", self._why(plan, hop, entity, ranked), ranked)

    def _why(self, plan, hop, entity, ranked) -> str:
        """A miss must say which stage failed — this is the field that was empty
        for 62% of the failures the decomposition had to replay walks to explain."""
        if not self.store:
            return "empty_store"
        if hop == 0:
            return "no_seed:tail_not_indexed" if normalize(plan.tail) not in self.index._by_tail \
                else "no_seed:tail_indexed_but_rejected"
        if normalize(entity or "") not in self.index._by_subj:
            return "no_forward_edge:entity_not_a_subject"
        return "no_accepted_candidate"

    # -- convenience -------------------------------------------------------
    def seedable(self, plan: QuestionPlan) -> bool:
        """Can hop 0 start at all? 61% of surviving failures cannot."""
        return bool(self.index.hop(plan, 0, None))
