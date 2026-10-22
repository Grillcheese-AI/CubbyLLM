"""Graph-of-thought in the ask loop (2026-09-24): an ambiguous hop is explored, a mid-chain split
comes back as choices the asker can pick from, and the pick is walked and certified like any chain.
Run: python -m pytest standin/tests/test_ask_branching.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning import events as ev  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from test_ask import STORE, FakeEmitter  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402

from ask import AskLoop  # noqa: E402

Q = "What is the continent of the country of citizenship of Marie?"
PLAN = {Q: ("marie", ["country of citizenship", "continent"])}
TWO = ["france is the country of citizenship of marie", "belgium is the country of citizenship of marie"]


def loop_over(extra, source=None):
    return AskLoop(FakeEmitter(PLAN), world=LookupStore(STORE + TWO + extra), source=source or DictSource({}),
                   lexicon=False, run_fn=faithful_vm([]))


def test_agreeing_branches_are_spoken_with_the_branches_on_record():
    rec = loop_over(["europe is the continent of france", "europe is the continent of belgium"]).ask(Q)
    assert rec["verified"] and normalize(rec["answer"]) == "europe" and rec["clarify"] is None
    assert sorted(b["via"][0]["object"] for b in rec["branches"]) == ["belgium", "france"]


def test_a_mid_chain_split_is_a_choice_and_the_pick_is_walked_and_certified():
    loop = loop_over(["europe is the continent of france", "atlantis is the continent of belgium"])
    sink = ev.MemorySink(); ev.add_sink(sink)
    try:
        rec = loop.ask(Q)
    finally:
        ev.remove_sink(sink)
    assert not rec["verified"] and rec["answer"] is None and rec["reason"] == "ambiguous_hop"
    c = rec["clarify"]
    assert c["hop"] == 0 and "marie" in c["question"].lower()
    by = {ch["label"]: ch for ch in c["choices"]}
    assert set(by) == {"france", "belgium"} and by["belgium"]["detail"] == "leads to atlantis"
    # the walk's branches are events: one per branch, under the walk
    walks = [e for e in sink if e["kind"] == "walk"]
    branches = [e for e in sink if e["kind"] == "branch"]
    assert len(branches) == 2 and {b["parent"] for b in branches} <= {w["id"] for w in walks}
    # the asker picks; the host walks that path and the VM certifies it
    again = loop.ask(Q, choose=by["france"]["choose"])
    assert again["verified"] and normalize(again["answer"]) == "europe" and again["branches"] is None
    assert again["chose"] == [{"hop": 0, "object": "france"}]


def test_a_last_hop_set_is_named_never_offered_as_a_pick():
    q = "What is the capital of the country of citizenship of Marie?"
    loop = AskLoop(FakeEmitter({q: ("marie", ["country of citizenship", "capital"])}),
                   world=LookupStore(STORE + ["france is the country of citizenship of marie",
                                              "lyon is the capital of france"]),
                   source=DictSource({}), lexicon=False, run_fn=faithful_vm([]))
    rec = loop.ask(q)
    assert rec["reason"] == "ambiguous_hop" and rec["clarify"] is None
    assert rec["refused"]["verified_answers"] == ["lyon", "paris"]


def test_the_branch_the_store_cannot_finish_is_what_the_loop_fetches():
    src = DictSource({"belgium": ["europe is the continent of belgium"]})
    rec = loop_over(["europe is the continent of france"], src).ask(Q)
    assert rec["verified"] and normalize(rec["answer"]) == "europe"
    assert [normalize(e) for e in rec["entities"]] == ["belgium"]
    assert [l["status"] for l in rec["learned"]] == ["accepted"]


def test_every_kept_record_carries_its_thought_graph():
    loop = loop_over(["europe is the continent of france", "atlantis is the continent of belgium"])
    rec = loop.ask(Q)
    g = rec["graph"]
    assert g["schema"] == 1 and g["summary"]["reason"] == "ambiguous_hop"
    assert any(e["type"] == "contradicts" for e in g["edges"])
    assert loop.history[-1] is rec
