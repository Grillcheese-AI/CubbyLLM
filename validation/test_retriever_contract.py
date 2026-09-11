"""Executable contract for the retriever, written BEFORE it exists.

Fake scorer only -- no table, no corpus, no VM, no GPU. Same pattern as
`test_cot_counterfactuals.py`: inject a fake for the expensive part and pin the
decisions. Run: python -m pytest validation/test_retriever_contract.py -q

WHAT PASSING MEANS
------------------
These pins run against the REAL component, `cubbyllm.reasoning.retriever.Retriever`,
with the encoder replaced by a table of fixed scores. The scorer is injected, so no
table, corpus, VM or GPU is needed to run the contract -- only to judge retrieval
QUALITY, which is an end-to-end question and deliberately not pinned here.

WHY THESE PINS AND NOT OTHERS
-----------------------------
Every one is a measured failure mode, not a guess:

  seed_*        61% of the failures surviving lookup-first never produce a
                hop-0 seed at all (exp_r4_index_walk). Entry is the product.
  no_threshold  lookup-first beat cosine 0.710 vs 0.646, +51 verified 0 lost,
                precisely because the index has NO threshold; 34 of the 36
                `threshold_bound` failures were absorbed by it
                (exp_r1_gate_diagnostics Q5). A hard tau at hop 0 rebuilds the
                class that just disappeared.
  distractor    on the threshold_bound cases where a margin was computable,
                BOTH had a distractor outranking gold -- 0.2876 vs 0.9445
                (exp_r1_gate_diagnostics Q2). Score must never be the gate.
  candidates    `candidates_topk` is empty for 152 of 246 failures, so the
                diagnostic for why retrieval failed is missing for 62% of the
                cases where it failed (exp_r1_gate_diagnostics Q1).
  backward      `TripleIndex.by_object` exists and is unused; worth ~2 cases on
                exact match but the edge is free (exp_r4_index_walk).
  lookup_first  an exact index hit must beat a higher-scoring search candidate.
  determinism   ambiguity is 0.8% of hops and must break the same way twice.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(rel, name, subs):
    src = (ROOT / rel).read_text(encoding="utf-8")
    for a, b in subs:
        src = src.replace(a, b)
    m = types.ModuleType(name); m.__dict__["__name__"] = name
    sys.modules[name] = m
    exec(compile(src, name + ".py", "exec"), m.__dict__)
    return m


_W = ("from ..core.protocols import Wiring", 'class Wiring:\n    WIRED = "WIRED"')
_PLN = ("from .planner import QuestionPlan, Triple, accepts, normalize, parse_fact",
        "from planner_real import QuestionPlan, Triple, accepts, normalize, parse_fact")
try:
    from cubbyllm.reasoning.planner import parse_question
    from cubbyllm.reasoning.index import TripleIndex
    from cubbyllm.reasoning.retriever import HopResult, Retriever
except Exception:                                # standalone: no package tree needed
    _pl = _load("cubbyllm/reasoning/planner.py", "planner_real", [_W])
    _ix = _load("cubbyllm/reasoning/index.py", "index_real", [_W, _PLN])
    _rt = _load("cubbyllm/reasoning/retriever.py", "retriever_real", [_W, _PLN,
        ("from .index import TripleIndex", "from index_real import TripleIndex")])
    parse_question, TripleIndex = _pl.parse_question, _ix.TripleIndex
    HopResult, Retriever = _rt.HopResult, _rt.Retriever


STORE = [
    "Kanton Genf is the administrative territorial entity of Grand Saconnex",
    "Great Council of Geneva is the legislative body of Kanton Genf",
    "Clarion (PA) is the capital of county of clarion, pennsylvania",
    "Paris is the capital of france",
    "Berlin is the capital of germany",
]
QUESTION = "What is the legislative body of the administrative territorial entity of Grand Saconnex?"
GOLD0 = "Kanton Genf is the administrative territorial entity of Grand Saconnex"

# A genuine ambiguity -- two facts serving the SAME hop-0 tail. The index
# docstring puts ambiguity at 0.8% of hops; this is the only shape in which a
# tie-break is observable at all, so the determinism pins use it.
AMBIG = [
    "Kanton Genf is the administrative territorial entity of Grand Saconnex",
    "Canton of Geneva is the administrative territorial entity of Grand Saconnex",
]
AMBIG_STORE = AMBIG + STORE[1:]


class FakeScorer:
    """Stands in for FastWordEncoder + cosine. Scores come from a table, so a
    test can make a distractor outrank the gold on purpose."""

    def __init__(self, scores=None, default: float = 0.1) -> None:
        self.scores, self.default, self.calls = dict(scores or {}), default, []

    def __call__(self, query: str, fact: str) -> float:
        self.calls.append((query, fact))
        return self.scores.get((query, fact), self.default)


@pytest.fixture
def plan():
    p = parse_question(QUESTION)
    assert p is not None and p.n_hop == 2, p
    return p


def make(scores=None, store=None, **kw):
    """The REAL component, with the encoder replaced by a table of scores.

    `store is None` not `store or STORE`: an empty store is a legitimate case
    and the `or` idiom silently substitutes the default for it — which is
    exactly how `test_empty_store_is_reported_not_crashed` passed against a
    retriever that never saw an empty store."""
    return Retriever(STORE if store is None else store, FakeScorer(scores), **kw)


# --- 1. ENTRY: the product. 61% of surviving failures never get a seed. -----
def test_seed_hop0_from_exact_tail(plan):
    got = make().hop(plan, 0, None)
    assert got, "hop-0 must seed from an exact tail match"
    assert got.triple.obj == "Kanton Genf" and got.source.startswith("lookup")
    assert got.reason is None


def test_seed_failure_is_a_value_not_an_exception(plan):
    got = make(store=["Berlin is the capital of germany"]).hop(plan, 0, None)
    assert isinstance(got, HopResult) and not got, "a miss is a value, never an exception"
    assert got.reason, "a miss must say which stage failed"


def test_hop0_seed_does_not_depend_on_score(plan):
    hostile = {(f"what is the {plan.tail}", f): 0.0 for f in STORE}
    assert make(hostile).hop(plan, 0, None), "a zero-scoring exact match must still seed"


# --- 2. SCORE IS NEVER THE GATE: the measured distractor trap. --------------
def test_high_scoring_distractor_never_beats_an_accepted_fact(plan):
    q = f"what is the {plan.tail}"
    scores = {(q, "Paris is the capital of france"): 0.9445, (q, GOLD0): 0.2876}
    got = make(scores).hop(plan, 0, None)
    assert got.triple.obj == "Kanton Genf", "a 0.94 distractor outranked the 0.29 gold"


def test_no_hard_threshold_reintroduced(plan):
    q = f"what is the {plan.tail}"
    for s in (0.0, 0.01, 0.4999, 0.5028, 0.5959):
        assert make({(q, GOLD0): s}).hop(plan, 0, None), f"dropped an accepted fact at {s}"


# --- 3. INSTRUMENTATION: candidates empty for 152 of 246 failures. ----------
def test_candidates_logged_on_success(plan):
    assert make().hop(plan, 0, None).candidates


def test_candidates_logged_on_FAILURE(plan):
    got = make(store=["Berlin is the capital of germany"]).hop(plan, 0, None)
    assert not got
    assert got.candidates, "a FAILED hop must log its candidates -- that is the diagnostic"


def test_candidates_ranked_desc(plan):
    q = f"what is the {plan.tail}"
    r = make({(q, "Paris is the capital of france"): 0.9, (q, "Berlin is the capital of germany"): 0.5})
    got = [s for s, _f in r.hop(plan, 0, None).candidates]
    assert got == sorted(got, reverse=True)


# --- 4. BIDIRECTIONAL: the edge exists and is unused. -----------------------
def test_forward_hop_resolves(plan):
    got = make().hop(plan, 1, "Kanton Genf")
    assert got.triple.obj == "Great Council of Geneva"
    assert got.source in ("lookup", "lookup_backward", "search")


def test_backward_edge_is_consulted_when_forward_misses(plan):
    """No fact has 'Great Council of Geneva' as a subject, so the forward index
    misses; by_object must find the edge pointing at it."""
    got = make().hop(plan, 1, "Great Council of Geneva")
    assert got and got.source == "lookup_backward"


def test_backward_can_be_disabled(plan):
    assert not make().hop(plan, 1, "Great Council of Geneva", bidirectional=False)


# --- 5. PRECEDENCE + DETERMINISM -------------------------------------------
def test_lookup_beats_search(plan):
    q = f"what is the {plan.tail}"
    got = make({(q, "Paris is the capital of france"): 0.99}).hop(plan, 0, None)
    assert got.source.startswith("lookup"), "an exact index hit must not fall through to search"


def test_same_input_same_answer(plan):
    assert make().hop(plan, 0, None).fact == make().hop(plan, 0, None).fact


def test_ambiguity_is_real_in_the_fixture(plan):
    """Guard the guard: if both facts no longer serve the hop, the two pins
    below are vacuous and would pass against a nondeterministic retriever."""
    r = make(store=AMBIG_STORE)
    assert len(r.index.hop(plan, 0, None)) == 2, "the ambiguity fixture stopped being ambiguous"
    assert r.hop(plan, 0, None).ambiguous, "an ambiguous hop must report itself as one"


def test_store_order_does_not_decide_a_TIE(plan):
    """Two equally-scoring accepted facts: store order must not pick the winner."""
    fwd = make(store=AMBIG_STORE).hop(plan, 0, None)
    rev = make(store=list(reversed(AMBIG_STORE))).hop(plan, 0, None)
    assert fwd.fact == rev.fact, "store order decided an ambiguous hop"


def test_score_still_breaks_a_tie_when_it_can(plan):
    """Deterministic does not mean score-blind: where scores differ among
    ACCEPTED candidates, the higher one wins."""
    q = f"what is the {plan.tail}"
    r = make({(q, AMBIG[1]): 0.9, (q, AMBIG[0]): 0.2}, store=AMBIG_STORE)
    assert r.hop(plan, 0, None).fact == AMBIG[1]


# --- 6. A MISS MUST EXPLAIN ITSELF -----------------------------------------
def test_miss_reason_distinguishes_no_seed_from_no_edge(plan):
    """The decomposition had to REPLAY 246 walks to learn why they failed,
    because the harvest recorded no candidates for them. A miss now says."""
    no_seed = make(store=["Berlin is the capital of germany"]).hop(plan, 0, None)
    assert no_seed.reason.startswith("no_seed"), no_seed.reason
    no_edge = make().hop(plan, 1, "Atlantis", bidirectional=False)
    assert not no_edge and no_edge.reason.startswith("no_forward_edge"), no_edge.reason


def test_empty_store_is_reported_not_crashed(plan):
    got = make(store=[]).hop(plan, 0, None)
    assert not got and got.reason == "empty_store"


def test_seedable_matches_hop0(plan):
    assert make().seedable(plan) is True
    assert make(store=["Berlin is the capital of germany"]).seedable(plan) is False
