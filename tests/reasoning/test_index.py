"""TripleIndex: retrieval as lookup, over the same acceptance test the walk applies to search."""
from cubbyllm.reasoning.index import TripleIndex
from cubbyllm.reasoning.planner import accepts, parse_question

Q3 = ("What is the continent of the country of the country of "
      "citizenship of cynthia basinet?")
F1 = "united stated is the country of citizenship of cynthia basinet"
F2 = "united stated is the country united stated is in"
F3 = "oceania portal is the continent of united stated"
F1_ALT = "canada is the country of citizenship of cynthia basinet"
NOT_A_TEMPLATE = "S. Koreans is the country samsinbong is in and more words"


def test_index_counts_every_fact_and_indexes_the_template_ones():
    ix = TripleIndex([F1, F2, F3, F1, "  " + F1 + "  ", NOT_A_TEMPLATE])
    assert ix.n_facts == 4 and len(ix) == 3          # duplicates (whitespace-normalized) dropped; the non-template counted, not indexed
    assert F1 in ix and NOT_A_TEMPLATE in ix and "x is the y of z" not in ix
    assert ix.add(F3) is None and ix.add("paris is the capital of france").obj == "paris"


def test_hop_lookup_answers_the_walks_acceptance_test_exactly():
    ix = TripleIndex([F1, F2, F3, "berlin is the capital of germany"])
    plan = parse_question(Q3)
    h0 = ix.hop(plan, 0, None)
    assert [f for f, _ in h0] == [F1]
    h1 = ix.hop(plan, 1, "united stated")
    assert [f for f, _ in h1] == [F2]
    h2 = ix.hop(plan, 2, "united stated")
    assert [f for f, _ in h2] == [F3]
    assert ix.hop(plan, 1, "canada") == [] and ix.hop(plan, 0, None)[0][1].subj == "cynthia basinet"
    for hop, entity in ((0, None), (1, "united stated"), (2, "united stated")):
        for _, t in ix.hop(plan, hop, entity):
            assert accepts(plan, hop, entity, t), "a lookup hit is accepted by construction"


def test_ambiguity_returns_every_candidate_in_insertion_order_and_the_object_side_is_indexed():
    ix = TripleIndex([F1_ALT, F1, F2, F3])
    plan = parse_question(Q3)
    assert [f for f, _ in ix.hop(plan, 0, None)] == [F1_ALT, F1]
    assert [f for f, _ in ix.by_object("United Stated")] == [F1, F2]
    assert ix.by_object("nowhere") == []
