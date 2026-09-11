"""Pins for hop 0's paraphrase tier (2026-09-11, lever 1 of the coverage work).

Before: the walk's acceptance at hop 0 was string equality on the whole tail
("rel of subj" == plan.tail) and the disposer mirrored it (exact known prefix).
exp_r5 (--real) measured 58% of the surviving entry failures as the store
holding the seed entity under a differently-worded relation. After: hop 0 has
the SAME two tiers hops >= 1 always had -- exact, then `relation_matches`
(Jaccard >= 0.6 on relation words) with the subject exact -- in the acceptance
test, the index, the walk's ranking and the disposer, all four.

What is pinned:
  * the index serves a hop-0 paraphrase (subject exact, relation Jaccard >= 0.6)
    and does NOT serve a different subject or a relation below the floor
    ('place of birth' vs 'place of death' is 0.5 -- the kill line);
  * an exact hit outranks a paraphrase hit whatever the retriever's ranking says;
  * a paraphrase-only walk verifies, its trace says `lookup_paraphrase`, and the
    disposer lets it through with the pair recorded in `paraphrased`;
  * the disposer still refuses a hop-0 relation below the floor, before any walk.
Run: python -m pytest validation/test_hop0_paraphrase.py -q
"""
from __future__ import annotations

import pathlib
import sys

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.index import TripleIndex  # noqa: E402
from cubbyllm.reasoning.pipeline import answer  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations, verify_plan  # noqa: E402
from cubbyllm.reasoning.planner import QuestionPlan, normalize, parse_question  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402

STORE = [
    "france is the country of citizenship of jean",
    "lyon is the place of death of jean",
    "canada is the country of citizenship of marie",
]


def retrieve_prefers(preferred):
    """A retriever that ranks `preferred` first -- to show the exact tier ignores it."""
    def retrieve(query, k):
        return [(1.0 if f == preferred else 0.1, f) for f in STORE][:k]
    return retrieve


def plan1(tail):
    return QuestionPlan(relations=[None], tail=tail, n_hop=1)


def test_index_serves_a_hop0_paraphrase_with_the_subject_exact():
    ix = TripleIndex(STORE)
    hits = [f for f, _t in ix.hop(plan1("citizenship country of jean"), 0, None)]
    assert hits == ["france is the country of citizenship of jean"]
    # a different subject is never served, however the relation matches
    assert ix.hop(plan1("citizenship country of pierre"), 0, None) == []
    # the kill line: a relation below the floor is not a paraphrase
    assert ix.hop(plan1("place of birth of jean"), 0, None) == []     # 'place of death' is 0.5


def test_exact_tier_outranks_the_paraphrase_tier_whatever_the_retriever_says():
    store = STORE + ["quebec is the citizenship country of jean"]    # a paraphrase-tier competitor
    ix = TripleIndex(store)
    q = "What is the country of citizenship of Jean?"
    r = answer(q, retrieve_prefers("quebec is the citizenship country of jean"), faithful_vm([]),
               tau_vm=0.5, tau_ret=0.5, top_k=3, max_repairs=1, lookup=ix.hop)
    assert r.verified and normalize(r.answer) == "france"
    assert r.trace[0].source == "lookup"


def test_a_paraphrase_only_walk_verifies_and_says_so():
    ix = TripleIndex(STORE)
    known = StoreRelations(STORE)
    q = "What is the citizenship country of Jean?"
    plan = parse_question(q)
    assert plan is not None and plan.n_hop == 1
    v = verify_plan(q, plan, known)
    assert v.ok and v.paraphrased == [("citizenship country", "country of citizenship")]
    vm_calls = []
    r = answer(q, retrieve_prefers(None), faithful_vm(vm_calls), tau_vm=0.5, tau_ret=0.5,
               top_k=3, max_repairs=1, lookup=ix.hop, known=known)
    assert r.refused is None and r.verified and normalize(r.answer) == "france"
    assert r.trace[0].source == "lookup_paraphrase" and vm_calls


def test_the_disposer_still_refuses_below_the_floor_before_any_walk():
    known = StoreRelations(STORE)
    q = "What is the place of birth of Jean?"
    v = verify_plan(q, parse_question(q), known)
    assert not v.ok and v.unknown_relations == ["place of birth"] and v.paraphrased == []
    calls = []
    r = answer(q, retrieve_prefers(None), faithful_vm(calls), tau_vm=0.5, tau_ret=0.5,
               top_k=3, max_repairs=1, lookup=TripleIndex(STORE).hop, known=known)
    assert r.reason == "unknown_relation" and calls == [] and r.answer is None


def test_the_reuse_guard_the_first_harvest_paid_for():
    """exp_m3 hop0, first run (2026-09-11): the greedy fact parse reads 'Documentary
    series is the genre of Joan Rivers: A Piece of Work' as rel 'genre of Joan
    Rivers: A Piece' | subj 'Work'; the tail 'award received by the genre of Joan
    Rivers: A Piece of Work' mis-splits at the same ' of ', the two 'relations'
    overlap at Jaccard 0.625, and the VM verified 'Documentary series' for an
    award question (gold: Genesis Awards). A one-off relation is a fragment until
    the store reuses it: neither the index nor the disposer may paraphrase it."""
    fact = "Documentary series is the genre of Joan Rivers: A Piece of Work"
    store = STORE + [fact]
    q = "What is the award received by the genre of Joan Rivers: A Piece of Work?"
    plan = parse_question(q)
    assert plan.n_hop == 1 and plan.tail.startswith("award received by the genre of")
    assert TripleIndex(store).hop(plan, 0, None) == []
    known = StoreRelations(store)
    v = verify_plan(q, plan, known)
    assert not v.ok and v.reason == "unknown_relation" and v.paraphrased == []
    # the same fragment stated twice is a relation as far as the store can tell --
    # the guard is about reuse, not about the words; that is its honest limit
    twice = store + ["Talk show is the genre of Joan Rivers: A Piece of Work"]
    assert len(TripleIndex(twice).hop(plan, 0, None)) == 2
    assert "genre of joan rivers a piece" in StoreRelations(twice).reused()


# ---- lever 1b (2026-09-11): the relation-aware fact split ----

def test_reused_relation_disambiguates_the_fact_split():
    """'Documentary series is the genre of Joan Rivers: A Piece of Work' -- greedy
    reads rel 'genre of Joan Rivers: A Piece' | subj 'Work'. When the store reuses
    'genre', the longest known prefix wins and the fact is indexed under its real
    subject, reachable at hops >= 1; the vocabulary carries no fragment. A relation
    that itself contains ' of ' ('country of citizenship') still wins over its
    shorter prefix 'country' when the store reuses it -- longest known first."""
    from cubbyllm.reasoning.planner import parse_fact, reused_relations
    store = STORE + ["Documentary series is the genre of Joan Rivers: A Piece of Work",
                     "Drama is the genre of Some Film", "Comedy is the genre of Other Film",   # 'genre' reused twice, cleanly
                     "spain is the country of pierre", "italy is the country of anna"]
    known = reused_relations(store)
    assert {"genre", "country of citizenship", "country"} <= known and "genre of joan rivers a piece" not in known
    t = parse_fact("Documentary series is the genre of Joan Rivers: A Piece of Work", known=known)
    assert (t.rel, t.subj) == ("genre", "Joan Rivers: A Piece of Work")
    t = parse_fact("france is the country of citizenship of jean", known=known)
    assert (t.rel, t.subj) == ("country of citizenship", "jean")                  # longest known, not 'country'
    ix = TripleIndex(store)
    assert [f for f, _t in ix._by_subj["joan rivers a piece of work"]] == ["Documentary series is the genre of Joan Rivers: A Piece of Work"]
    assert "genre of joan rivers a piece" not in StoreRelations(store)
    # reachable at hop 1 now: 'What is the genre of the <rel> of X?' walks through the entity
    plan = QuestionPlan(relations=[None, "genre"], tail="country of citizenship of jean", n_hop=2)
    assert ix.hop(plan, 1, "Joan Rivers: A Piece of Work")            # served by subject, relation exact
    # a live-added fact whose relation the store already reuses splits the same way
    ix.add("Comedy is the genre of Best of Both Worlds")
    assert ix._by_subj["best of both worlds"]
