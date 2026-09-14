"""
The loop asked about itself -- Nick, 2026-09-14: "the model needs to be able to use the vm on
its own so it can reason at all time and be concious of its surroundings by analysing it via its
own 'power vm'."
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for q in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "tests")):
    if q not in sys.path:
        sys.path.insert(0, q)

from ask import AskLoop, cot_self, profile_program, program_world, self_ask  # noqa: E402
from self_source import LOOP, SelfSource  # noqa: E402
from test_ask import FakeEmitter  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402


def _loop():
    world = LookupStore(["Canada is the country of Quebec City"])
    world.times = {}
    return AskLoop(FakeEmitter({}), world=world, source=DictSource({}), lexicon=False,
                   run_fn=faithful_vm([]), tau_profile=0.5)


def test_the_program_says_which_store_it_is_asking_about():
    """A question about what the loop KNOWS and a question about what IS are different
    questions. The program says which one, and absent means the world -- so every program
    written before this role existed still asks what it always asked."""
    assert program_world('bind frame, SEED, "x"; bind frame, ASK, "what";') == "world"
    assert program_world(cot_self("quebec city", "what")) == "self"
    assert profile_program(cot_self("quebec city", "what")) == ("what", "quebec city")


def test_the_host_reads_about_you_off_the_question_until_the_model_can_write_it():
    """gen 2 cannot emit WORLD yet, so the host recognises the shape -- exactly as it does for
    who / what / where. When gen 3 can, this becomes the fallback: a loop that can only be
    introspected by its operator is not aware of its surroundings, it is merely logged."""
    assert self_ask("what do you know about Quebec City?") == ("what", "Quebec City")
    assert self_ask("why did you refuse Quebec City?") == ("what", "Quebec City")
    assert self_ask("what do you know?") == ("what", LOOP)
    assert self_ask("what do you refuse most?") == ("what", LOOP)
    assert self_ask("what is Quebec City?") is None            # about the WORLD, not about you
    assert self_ask("who is bill haslam?") is None


def test_a_subject_with_no_record_is_not_a_claim_about_the_world():
    """"I have no record of that" and "that does not exist" are different sentences, and only
    the first one is the loop's to say."""
    loop = _loop()
    src = SelfSource(loop)
    assert src.facts("some entity nobody asked about") == []
    assert src.last["unresolved"] is True and "no record" in src.last["how"]


def test_the_loop_reports_what_it_did_and_never_what_it_thinks_of_it():
    """"refused 3 times" is a fact the loop witnessed. "is unreliable" is an opinion, and there
    is no relation on this path that can carry one."""
    loop = _loop()
    loop.history = [
        {"question": "q1", "seed": "quebec city", "entities": ["quebec city"], "answer": "Canada",
         "reason": None, "plan": ["country"], "learned": [{"status": "accepted", "source": "wikidata"}]},
        {"question": "q2", "seed": "quebec city", "entities": ["quebec city"], "answer": None,
         "reason": "ambiguous_hop", "plan": ["population"], "learned": []},
        {"question": "q3", "seed": "quebec city", "entities": ["quebec city"], "answer": None,
         "reason": "ambiguous_hop", "plan": ["population"], "learned": []},
    ]
    facts = {(t.rel, t.obj) for t in SelfSource(loop).facts("quebec city")}
    assert ("asked", "3") in facts and ("answered", "1") in facts and ("refused", "2") in facts
    assert ("last verdict", "ambiguous_hop") in facts
    assert ("fact learned", "1") in facts
    assert any(r == "source" and o.startswith("wikidata") for r, o in facts)
    rels = {r for r, _o in facts}
    assert not rels & {"reliable", "unreliable", "good", "bad", "quality", "trust"}


def test_the_loop_can_say_why_it_refuses_which_is_the_thing_nobody_can_read_off_an_answer():
    """The commonest refusal reason, counted. It is the one number about the loop that cannot be
    recovered by reading the answers, because refusals are exactly what the answers are not."""
    loop = _loop()
    loop.history = [{"question": f"q{i}", "seed": "x", "entities": ["x"], "answer": None,
                     "reason": "unknown_relation", "plan": [], "learned": []} for i in range(4)]
    loop.history += [{"question": "q9", "seed": "y", "entities": ["y"], "answer": None,
                      "reason": "ambiguous_hop", "plan": [], "learned": []}]
    facts = {(t.rel, t.obj) for t in SelfSource(loop).facts(LOOP)}
    assert ("asked", "5") in facts and ("refused", "5") in facts
    assert ("refusal", "unknown_relation (4)") in facts
    assert ("refusal", "ambiguous_hop (1)") in facts
    assert any(r == "source" for r, _o in facts)               # it knows what it is reading from


def test_a_self_fact_never_enters_the_world_store():
    """The fence. The loop's notes about the world are not the world, and the day they share a
    store is the day it starts answering questions about Quebec City with its own paperwork."""
    loop = _loop()
    before = set(loop.world.index._seen)
    loop.history = [{"question": "q1", "seed": "quebec city", "entities": ["quebec city"],
                     "answer": "Canada", "reason": None, "plan": ["country"], "learned": []}]
    rec = loop.ask("what do you know about quebec city?")
    assert rec["world"] == "self" and rec["profile"]
    assert set(loop.world.index._seen) == before


def test_the_loop_does_not_narrate_itself():
    """2026-09-14, live and wrong. Given `fact learned: 152; source: wikidata (152)` the talk
    adapter wrote "referenced in 152 sources" -- it is 152 facts from ONE source -- and given
    `last answer: 574482` it wrote "the last answer was given 574482 units ago".

    Both passed the grounding guard, because every name and number in them occurs in the facts.
    That is the guard working as designed and the design having a hole: it checks VOCABULARY,
    not semantics, and it survives on world facts only because those relations are ordinary
    English. "fact learned", "call" and "provenance" are not, and the adapter invents what they
    mean. The counts are already the answer; fluency here buys nothing and costs the kill line."""
    loop = _loop()
    loop.history = [{"question": "q1", "seed": "quebec city", "entities": ["quebec city"],
                     "answer": "574482", "reason": None, "plan": ["population"],
                     "learned": [{"status": "accepted", "source": "wikidata"}] * 152}]
    rec = loop.ask("what do you know about quebec city?")
    assert rec["prose"] is None
    assert "fact learned: 152" in rec["answer"] and "source: wikidata (152)" in rec["answer"]
