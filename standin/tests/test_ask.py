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
    def __init__(self, plans): self.plans = plans; self.calls = 0
    def emit(self, prompt, max_new_tokens=300, **kw):
        self.calls += 1
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
