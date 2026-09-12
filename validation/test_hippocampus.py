"""test_hippocampus -- the episodic side cortex proposes; the gate disposes.

What is pinned:
  * DG: the same chain in four surface forms lands at Hamming distance 0; a
    different entity lands apart;
  * CA3: a certified canonical chain is recalled from the have / relative /
    possessive forms, proposed as the plan, and the walk + VM verify it;
  * rebinding: the relation shape of one entity's chain, rebound to the entity a
    new question names, verifies for the new entity;
  * a recalled plan for a question it does not cover (a dropped hop, an extra hop)
    is refused by the disposer -- memory never speaks on its own;
  * consolidation retires beyond `keep` by utility and never deletes; retired
    episodes are not recalled;
  * save / load round-trips codes and recall.
Run: python -m pytest validation/test_hippocampus.py -q
"""
from __future__ import annotations

import pathlib
import sys

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.hippocampus import Hippocampus, content_words, encode, hamming, residual_entity  # noqa: E402
from cubbyllm.reasoning.pipeline import answer  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import STORE, LookupStore  # noqa: E402

NO_SEARCH = lambda q, k: []
FORMS = {"canonical": "What is the capital of the country of citizenship of Jean?",
         "have": "What capital does the country of citizenship of Jean have?",
         "relative": "What is the capital of the thing that is the country of citizenship of Jean?",
         "possessive": "Jean's country of citizenship -- what is its capital?"}


def walk(q, plan, store, known):
    return answer(q, NO_SEARCH, faithful_vm([]), tau_vm=0.4, tau_ret=0.0, top_k=3, max_repairs=1,
                  lookup=store.lookup, known=known, plan=plan)


def certified() -> Hippocampus:
    h = Hippocampus()
    h.write(FORMS["canonical"], ["country of citizenship", "capital"], "jean",
            ["france is the country of citizenship of jean", "paris is the capital of france"], "paris",
            {"source": "test", "store": "STORE"})
    return h


def test_dg_same_chain_any_frame_lands_together_and_another_entity_apart():
    codes = {f: encode(content_words(q)) for f, q in FORMS.items()}
    assert all(hamming(codes["canonical"], c) == 0 for c in codes.values())
    other = encode(content_words("What is the capital of the country of citizenship of Hans?"))
    assert hamming(codes["canonical"], other) > 0


def test_ca3_recalls_the_certified_chain_from_every_form_and_the_gate_verifies_it():
    h = certified(); store, known = LookupStore(STORE), StoreRelations(STORE)
    for form, q in FORMS.items():
        cands = h.propose(q)
        assert cands and cands[0][2] == "recalled" and cands[0][1].seed == "jean", form
        r = walk(q, cands[0][0], store, known)
        assert r.verified and normalize(r.answer) == "paris", (form, r.reason)
        h.reinforce(cands[0][1])
    assert h.episodes[0].utility == 4


def test_the_shape_of_one_chain_rebinds_to_the_entity_a_new_question_names():
    h = certified(); store, known = LookupStore(STORE), StoreRelations(STORE)
    q = "What is the capital of the country of citizenship of Hans?"
    cands = h.propose(q)
    kinds = [k for _p, _e, k in cands]
    assert "rebound" in kinds
    plan = next(p for p, _e, k in cands if k == "rebound")
    assert plan.tail == "country of citizenship of hans"
    recalled = next(p for p, _e, k in cands if k == "recalled")
    assert walk(q, recalled, store, known).refused is not None               # jean's plan does not cover hans's question
    r = walk(q, plan, store, known)
    assert r.verified and normalize(r.answer) == "berlin"


def test_a_recalled_plan_that_does_not_cover_the_question_is_refused():
    h = certified(); store, known = LookupStore(STORE), StoreRelations(STORE)
    for q in ("What is the capital of Jean?",                                            # a hop the question does not ask
              "What is the population of the capital of the country of citizenship of Jean?"):   # a hop the plan lacks
        store2 = LookupStore(STORE + ["2000000 is the population of paris"]); known2 = StoreRelations(store2.texts)
        for plan, _ep, _kind in h.propose(q):
            r = walk(q, plan, store2, known2)
            assert not r.verified and r.reason == "plan_does_not_cover_question", q


def test_consolidation_retires_by_utility_and_never_deletes():
    h = certified()
    h.write("What is the capital of the country of citizenship of Hans?", ["country of citizenship", "capital"], "hans",
            ["germany is the country of citizenship of hans", "berlin is the capital of germany"], "berlin")
    h.reinforce(h.episodes[1])
    retired = h.consolidate(keep=1)
    assert [e.seed for e in retired] == ["jean"] and len(h) == 1 and len(h.episodes) == 2
    assert [e.seed for e, _d in h.recall(FORMS["have"])] == ["hans"]        # the retired one is not recalled


def test_save_and_load_round_trip(tmp_path):
    h = certified()
    p = tmp_path / "episodes.jsonl"
    assert h.save(p) == 1
    h2 = Hippocampus.load(p)
    assert h2.episodes[0].code == h.episodes[0].code and h2.episodes[0].chain == h.episodes[0].chain
    assert h2.recall(FORMS["possessive"])[0][1] == 0


def test_residual_entity_is_the_whole_leftover_span_never_a_fragment():
    assert residual_entity("What is the capital of the country of citizenship of Jean Valjean?",
                           ["country of citizenship", "capital"]) == "jean valjean"
