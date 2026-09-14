"""standin/ask.py -- the verified-program loop on one natural question, as the panel sees it.

Pinned: a question whose fact the store lacks is fetched from the source, gated, walked and
VM-verified, and the loop's record says so; the events it emits form ONE tree under the
question (plan, walk, fetch, gate, answer) -- what /loop/stream carries to the panel; a
proposal the emitter cannot make is an honest `no_plan` with its own root and answer event;
an answer is never spoken unverified.
Run: python -m pytest standin/tests/test_ask.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning import events as ev  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402

from ask import AskLoop  # noqa: E402


class FakeEmitter:
    """Proposes a CotPlan for the one question it knows; nothing for the rest."""
    def __init__(self, plans): self.plans = plans; self.calls = 0; self.programs = {}
    def emit(self, prompt, max_new_tokens=300, **kw):
        self.calls += 1
        if prompt in self.programs:
            return self.programs[prompt]
        p = self.plans.get(prompt)
        if p is None:
            return "no plan\n"
        seed, rels = p
        lines = [f'    bind frame, SEED, "{seed}";'] + [f'    bind frame, HOP{i + 1}, "{r}";' for i, r in enumerate(rels)]
        return "function plan(): str {\n" + "\n".join(lines) + "\n    return recover(frame, SEED);\n}\n"


STORE = ["paris is the capital of france", "france is the country of citizenship of jean",
         "berlin is the capital of germany", "germany is the country of citizenship of hans"]


def make_loop(plans, source):
    return AskLoop(FakeEmitter(plans), world=LookupStore(STORE), source=source, lexicon=False, run_fn=faithful_vm([]))


def test_a_natural_question_is_fetched_gated_walked_verified_and_the_events_form_one_tree():
    sink = ev.MemorySink(); ev.add_sink(sink)
    try:
        loop = make_loop({"What is the capital of the country of citizenship of Marie?": ("marie", ["country of citizenship", "capital"])},
                         DictSource({"marie": ["canada is the country of citizenship of marie"], "canada": ["ottawa is the capital of canada"]}))
        rec = loop.ask("What is the capital of the country of citizenship of  Marie?")
    finally:
        ev.remove_sink(sink)
    assert rec["verified"] and normalize(rec["answer"]) == "ottawa" and rec["reason"] is None
    assert rec["plan"] == ["country of citizenship", "capital"] and rec["seed"] == "marie"
    assert [normalize(e) for e in rec["entities"]] == ["marie", "canada"]
    assert [l["status"] for l in rec["learned"]] == ["accepted", "accepted"] and all(l["source"] == "test" for l in rec["learned"])
    assert "ottawa is the capital of canada" in loop.world
    assert loop.calls["asked"] == 1 and loop.calls["vm"] >= 1 and rec["wall_s"] >= 0
    # the events: one root (the question), and every other event hangs under it -- the panel's tree
    evs = list(sink)
    roots = [e for e in evs if e.get("parent") is None]
    assert len(roots) == 1 and roots[0]["kind"] == "question" and roots[0]["text"] == rec["question"]
    ids = {e["id"] for e in evs}
    assert all(e["parent"] in ids for e in evs if e.get("parent") is not None)
    kinds = {e["kind"] for e in evs}
    assert {"question", "plan", "walk", "fetch", "gate", "answer"} <= kinds
    ans = [e for e in evs if e["kind"] == "answer"]
    assert ans[-1]["parent"] == roots[0]["id"] and ans[-1]["verified"] and normalize(ans[-1]["answer"]) == "ottawa"


def test_no_proposal_is_an_honest_refusal_with_its_own_root_and_nothing_spoken():
    sink = ev.MemorySink(); ev.add_sink(sink)
    try:
        loop = make_loop({}, DictSource({}))
        rec = loop.ask("Tell me something about Marie")
    finally:
        ev.remove_sink(sink)
    assert not rec["verified"] and rec["answer"] is None and rec["reason"] == "no_plan"
    roots = [e for e in sink if e.get("parent") is None]
    assert len(roots) == 1 and roots[0]["text"] == "Tell me something about Marie"
    ans = [e for e in sink if e["kind"] == "answer"]
    assert len(ans) == 1 and ans[0]["parent"] == roots[0]["id"] and not ans[0]["verified"] and ans[0]["reason"] == "no_plan"


def test_a_question_the_store_already_answers_never_calls_the_source():
    src = DictSource({"jean": ["NOT-paris is the capital of france"]})
    loop = make_loop({"What is the capital of the country of citizenship of Jean?": ("jean", ["country of citizenship", "capital"])}, src)
    rec = loop.ask("What is the capital of the country of citizenship of Jean?")
    assert rec["verified"] and normalize(rec["answer"]) == "paris" and src.calls == [] and rec["learned"] == []


def test_who_is_x_is_a_retrieval_program_the_vm_runs_and_the_reply_is_the_facts_it_recovered():
    """2026-09-14 (Nick: "seems like it does not get who is"; "the model's job is to write the right program so
    the VM can retrieve it, then it will reply with the facts"; "who, what, where"). The plan is SEED + ASK --
    the program gen 3 learns to emit, read off the question until then -- the store is filled from the source
    through the gate, every profile fact is bound into a program and the VM recovers it; only recovered lines
    are spoken; `reason: profile`, never `verified`."""
    from ask import cot_profile, profile_ask, profile_program
    assert profile_ask("who is marie curie?") == ("who", "marie curie") and profile_ask("Where is Lévis") == ("where", "Lévis")
    assert profile_ask("What was the Roman Empire?") == ("what", "Roman Empire")
    assert profile_ask("What is the capital of France?") is None and profile_ask("Who is Canada's prime minister?") is None
    prog = cot_profile("marie curie", "who")
    assert profile_program(prog) == ("who", "marie curie") and 'bind frame, ASK, "who";' in prog
    assert profile_program(prog.replace('bind frame, ASK, "who";', 'bind frame, HOP1, "father";')) is None   # a hop: a plan, not a profile
    sink = ev.MemorySink(); ev.add_sink(sink)
    try:
        src = DictSource({"marie curie": ["physicist is the occupation of marie curie", "chemist is the occupation of marie curie",
                                         "warsaw is the place of birth of marie curie", "1867 is the date of birth of marie curie",
                                         "Q7186 is the freebase id of marie curie"]})
        loop = make_loop({}, src); loop.tau_profile = 0.5                     # the stand-in VM answers at 0.9
        rec = loop.ask("who is marie curie?")
    finally:
        ev.remove_sink(sink)
    assert rec["reason"] == "profile" and not rec["verified"] and loop.emitter.calls == 0 and rec["ask"] == "who"
    assert rec["program"] == prog
    assert [(l["relation"], l["value"]) for l in rec["profile"]] == [("occupation", "physicist"), ("occupation", "chemist"),
                                                                     ("date of birth", "1867"), ("place of birth", "warsaw")]
    assert all(l["similarity"] == 0.9 for l in rec["profile"]) and loop.calls["vm"] == 4      # one recover per line
    assert rec["answer"].startswith("marie curie — occupation: physicist, chemist; date of birth: 1867") and rec["prose"] is None
    assert src.calls == ["marie curie"] and "warsaw is the place of birth of marie curie" in loop.world
    kinds = [e["kind"] for e in sink]
    assert kinds.count("question") == 1 and kinds.count("hop") == 4 and kinds.count("fact") == 4 and "vm" in kinds and kinds[-1] == "answer"
    plan = [e for e in sink if e["kind"] == "plan"][0]
    assert plan["ask"] == "who" and 'bind frame, ASK, "who";' in plan["program"]
    ans = [e for e in sink if e["kind"] == "answer"][0]
    assert ans["reason"] == "profile" and not ans["verified"] and len(ans["profile"]) == 4
    # the VM did not recover a line: it is not spoken
    loop.tau_profile = 0.95
    rec_b = loop.ask("who is marie curie?")
    assert rec_b["reason"] == "profile_unrecovered" and rec_b["answer"] is None and rec_b["profile"] == []
    # the emitter's own retrieval program is read the same way
    loop.tau_profile = 0.5
    loop.emitter.plans["Tell me about Marie Curie"] = None
    loop.emitter.programs = {"Tell me about Marie Curie": prog}
    rec_c = loop.ask("Tell me about Marie Curie")
    assert rec_c["reason"] == "profile" and rec_c["seed"] == "marie curie" and src.calls == ["marie curie"]   # the store answers
    # an entity nobody knows: an honest empty profile
    rec3 = loop.ask("who is zzyzx qqq?")
    assert rec3["answer"] is None and rec3["reason"] == "entity_unresolved" and rec3["profile"] == []


def test_a_paraphrase_is_spoken_only_when_every_name_and_number_in_it_is_in_the_facts():
    from ask import grounded_prose
    facts = ["physicist is the occupation of marie curie", "warsaw is the place of birth of marie curie", "1867 is the date of birth of marie curie"]
    assert grounded_prose("Marie Curie was a physicist, born in Warsaw in 1867.", facts, "marie curie") == (True, [])
    ok, bad = grounded_prose("Marie Curie was a physicist born in Warsaw in 1867 who won the Nobel Prize in 1903.", facts, "marie curie")
    assert not ok and bad == ["Nobel", "Prize", "1903"]


def test_a_date_the_store_holds_may_be_written_out_but_a_date_it_does_not_hold_may_not():
    """2026-09-14, live ('who is openai?'): the store holds `2015-12-11 is the inception of OpenAI` and the
    paraphrase said 'founded on December 11, 2015' -- refused over the word 'December'. A date the facts
    hold, spelled the way people spell it, is not an addition; a date they do not hold still is."""
    from ask import date_words, grounded_prose
    facts = ["2015-12-11 is the inception of OpenAI", "ChatGPT is the notable work of OpenAI"]
    assert date_words(facts) == ["december", "2015", "11"]
    assert grounded_prose("OpenAI was founded on December 11, 2015 and is known for ChatGPT.", facts, "openai") == (True, [])
    ok, bad = grounded_prose("OpenAI was founded in March 2015 by Sam Altman.", facts, "openai")
    assert not ok and bad == ["March", "Sam", "Altman"]
