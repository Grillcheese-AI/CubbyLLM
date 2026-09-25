"""The day -> night -> day loop: what the ask loop learned on day 1 is known on day 2.

Pinned: with a history path every record is one JSON line carrying the gate's provenance and each
accepted fact's relation; the sleep cycle makes the accepted facts durable and routes the refusal;
a fresh loop on a fresh store loads them at boot and answers the same question WITHOUT the source --
through the same walk and the same VM, nothing spoken because the log said so.
Run: python -m pytest standin/tests/test_sleep_loop.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning import sleep as S  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402

from ask import AskLoop  # noqa: E402
from test_ask import STORE, FakeEmitter  # noqa: E402

Q = "What is the capital of the country of citizenship of Marie?"
PLANS = {Q: ("marie", ["country of citizenship", "capital"])}


def test_what_day_one_learned_day_two_knows(tmp_path):
    hist = tmp_path / "ask_history.jsonl"
    src = DictSource({"marie": ["canada is the country of citizenship of marie"],
                      "canada": ["ottawa is the capital of canada"]})
    day1 = AskLoop(FakeEmitter(PLANS), world=LookupStore(STORE), source=src, lexicon=False,
                   run_fn=faithful_vm([]), history_path=str(hist))
    assert day1.ask(Q)["verified"]
    assert day1.ask("Tell me something about Marie")["reason"] == "no_plan"

    lines = [json.loads(l) for l in hist.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["id"] and lines[0]["store_snapshot"]
    acc = [l for l in lines[0]["learned"] if l["status"] == "accepted"]
    assert {l["rel"] for l in acc} == {"country of citizenship", "capital"} and all(l["snapshot_before"] for l in acc)

    rep = S.sleep([hist], tmp_path / "sleep", night="2026-09-24", trusted={"test"})
    assert rep["consolidate"]["facts_added"] == 2 and rep["replay"]["episodes_written"] == 1
    assert rep["triage"]["emitter_curriculum"]["tonight"] == 1

    empty = DictSource({})
    day2 = AskLoop(FakeEmitter(PLANS), world=LookupStore(STORE), source=empty, lexicon=False, run_fn=faithful_vm([]))
    assert day2.load_learned(tmp_path / "sleep" / "learned_facts.jsonl") == 2
    rec = day2.ask(Q)
    assert rec["verified"] and normalize(rec["answer"]) == "ottawa" and empty.calls == [] and rec["learned"] == []


def test_without_a_history_path_nothing_is_written(tmp_path):
    loop = AskLoop(FakeEmitter(PLANS), world=LookupStore(STORE), source=DictSource({}), lexicon=False,
                   run_fn=faithful_vm([]))
    loop.ask("Tell me something about Marie")
    assert loop.history and not list(tmp_path.iterdir())
