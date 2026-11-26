"""Order 0083, memory first: a checked plan remembered before the emitter writes one.

Pinned: off (both flags) the ask loop's records are what they were, a mounted memory or not; on, the second asking of
a certified question is answered by a remembered plan without the emitter, re-walked and re-verified over the store;
a remembered plan about another entity is refused by covers() and the emitter then runs; a remembered plan fetches
nothing; the striatum's order and its rewards on a scripted run; the live write is the episode sleep.replay writes for
the same record; the folder sleep writes is what mount loads.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning import ask_recall as R  # noqa: E402
from cubbyllm.reasoning import sleep as S  # noqa: E402
from cubbyllm.reasoning.hippocampus import Hippocampus  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from cubbyllm.reasoning.striatum import question_shape  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402

from cubbyllm.reasoning.ask import AskLoop  # noqa: E402
from test_ask import STORE, FakeEmitter  # noqa: E402

QJ = "What is the capital of the country of citizenship of Jean?"
QH = "What is the capital of the country of citizenship of Hans?"
PLANS = {QJ: ("jean", ["country of citizenship", "capital"]), QH: ("hans", ["country of citizenship", "capital"])}
JEAN_ONLY = STORE[:2]                                       # hans's facts only from the source
VOLATILE = ("wall_s", "ts", "id", "hash", "graph")


def loop(store=STORE, source=None, history=None):
    return AskLoop(FakeEmitter(PLANS), world=LookupStore(list(store)), source=source or DictSource({}), lexicon=False,
                   run_fn=faithful_vm([]), history_path=history)


def steady(rec):
    return {k: v for k, v in rec.items() if k not in VOLATILE}


@pytest.fixture
def memory_on(monkeypatch):
    monkeypatch.setenv("CB_ASK_MEMORY", "1")
    monkeypatch.setenv("CB_ASK_MEMORY_WRITE", "1")


def test_off_the_records_are_as_before_with_or_without_a_mounted_memory(monkeypatch):
    monkeypatch.delenv("CB_ASK_MEMORY", raising=False)
    monkeypatch.delenv("CB_ASK_MEMORY_WRITE", raising=False)
    plain, mounted = loop(), loop()
    R.mount(mounted)
    mounted.hippocampus.write(QJ, ["country of citizenship", "capital"], "jean", list(STORE[:2][::-1]), "paris")
    for q in (QJ, QJ, QH, "Tell me something about Marie"):
        assert steady(plain.ask(q)) == steady(mounted.ask(q))
    assert plain.emitter.calls == mounted.emitter.calls == 4
    assert len(mounted.hippocampus) == 1 and not mounted.striatum.trace      # nothing learned, nothing rewarded


def test_on_without_a_mounted_memory_is_as_before(monkeypatch, memory_on):
    plain = loop()
    monkeypatch.delenv("CB_ASK_MEMORY")
    want = steady(loop().ask(QJ))
    monkeypatch.setenv("CB_ASK_MEMORY", "1")
    assert steady(plain.ask(QJ)) == want and plain.emitter.calls == 1


def test_the_second_asking_is_a_remembered_plan_re_verified_without_the_emitter(memory_on):
    lp = loop()
    R.mount(lp)
    first = lp.ask(QJ)
    assert first["verified"] and normalize(first["answer"]) == "paris" and lp.emitter.calls == 1
    assert first["memory"]["tried"] == [] and len(lp.hippocampus) == 1             # empty before; the live write after
    vm_before = lp.calls["vm"]
    second = lp.ask(QJ)
    assert second["verified"] and second["answer"] == first["answer"] and second["trace"] == first["trace"]
    assert second["how"] == "a remembered plan" and lp.emitter.calls == 1          # the emitter was not asked
    assert second["episode"] == R.episode_id(lp.hippocampus.episodes[0]) and lp.calls["vm"] > vm_before
    assert second["memory"]["tried"][0]["kind"] == "recalled" and lp.hippocampus.episodes[0].utility == 1
    assert len(lp.hippocampus) == 1                                                 # the same question and chain: once


def test_a_remembered_plan_about_another_entity_is_refused_and_the_emitter_runs(memory_on):
    src = DictSource({"hans": [STORE[3]], "germany": [STORE[2]]})
    lp = loop(store=JEAN_ONLY, source=src)
    R.mount(lp)
    assert lp.ask(QJ)["verified"] and src.calls == []
    rec = lp.ask(QH)
    tried = {t["kind"]: t["reason"] for t in rec["memory"]["tried"]}
    assert tried["recalled"] == "plan_does_not_cover_question"                      # jean's own plan: covers() refuses
    assert tried["rebound"] and src.calls[0] == "hans"                              # rebound to hans: nothing fetched
    assert lp.emitter.calls == 2 and rec["verified"] and normalize(rec["answer"]) == "berlin" and "how" not in rec
    assert rec["plan"] == ["country of citizenship", "capital"] and rec["seed"] == "hans"


def test_a_shape_rebound_to_a_held_entity_answers_without_the_emitter(memory_on):
    lp = loop()
    R.mount(lp)
    lp.ask(QJ)
    rec = lp.ask(QH)
    assert rec["verified"] and normalize(rec["answer"]) == "berlin" and rec["how"] == "a remembered plan"
    assert rec["memory"]["tried"][-1]["kind"] == "rebound" and lp.emitter.calls == 1


def test_a_remembered_plan_fetches_nothing(memory_on):
    lp = loop(store=JEAN_ONLY)
    R.mount(lp)
    lp.hippocampus.write(QH, ["country of citizenship", "capital"], "hans", [STORE[3], STORE[2]], "berlin")
    lp.source = src = DictSource({"hans": [STORE[3]], "germany": [STORE[2]]})
    rec = lp.ask(QH)                                         # the store lost hans's facts: the memory refuses, the emitter fetches
    assert rec["memory"]["tried"][0]["reason"] == "retrieval_exhausted" and lp.emitter.calls == 1
    assert rec["verified"] and "how" not in rec and src.calls[0] == "hans"


def test_the_striatum_orders_and_is_rewarded_on_a_scripted_run(memory_on):
    lp = loop()
    R.mount(lp)
    shape = question_shape(QJ)
    lp.ask(QJ)                                               # memory empty: refused; the emitter certifies
    assert lp.striatum.value("hippocampus", shape) == 0.0 and lp.striatum.value("emitter", shape) == pytest.approx(0.2)
    lp.ask(QJ)                                               # remembered: only the hippocampus is rewarded
    assert lp.striatum.value("hippocampus", shape) == pytest.approx(0.2)
    assert lp.striatum.value("emitter", shape) == pytest.approx(0.2)
    assert [(t["proposer"], t["outcome"]) for t in lp.striatum.trace] == [
        ("hippocampus", "refused"), ("emitter", "certified"), ("hippocampus", "certified")]
    lp.striatum.expected[lp.striatum.key("hippocampus", shape)] = -5.0               # an audit's defect, at night
    rec = lp.ask(QJ)
    assert rec["memory"] == {"order": ["emitter", "hippocampus"], "tried": []} and lp.emitter.calls == 2
    assert lp.striatum.trace[-1]["proposer"] == "emitter"


def test_the_live_write_is_the_episode_sleep_replay_writes(memory_on, tmp_path):
    hist = tmp_path / "ask_history.jsonl"
    lp = loop(history=str(hist))
    R.mount(lp)
    rec = lp.ask(QJ)
    live = lp.hippocampus.episodes
    night = Hippocampus()
    S.replay(S.load_day(hist), night, "2026-11-25", S.Ledger(tmp_path / "ledger.jsonl"))
    fields = ("question", "relations", "seed", "chain", "answer", "code", "shape", "utility", "retired")
    assert len(live) == len(night.episodes) == 1
    assert {f: getattr(live[0], f) for f in fields} == {f: getattr(night.episodes[0], f) for f in fields}
    assert live[0].provenance["record"] == rec["id"] and live[0].provenance["live"]


def test_no_live_write_without_its_flag(memory_on, monkeypatch):
    monkeypatch.delenv("CB_ASK_MEMORY_WRITE")
    lp = loop()
    R.mount(lp)
    lp.ask(QJ)
    assert len(lp.hippocampus) == 0 and lp.striatum.trace                           # rewarded, nothing written


def test_mount_reads_the_folder_sleep_writes(memory_on, tmp_path):
    hist = tmp_path / "ask_history.jsonl"
    day = loop(history=str(hist))
    day.ask(QJ)
    S.sleep([hist], tmp_path / "sleep", night="2026-11-25", trusted={"test"})
    lp = loop()
    line = R.mount(lp, tmp_path / "sleep")
    assert len(lp.hippocampus) == 1 and lp.striatum.expected and "1 live episodes" in line
    assert lp.ask(QJ)["how"] == "a remembered plan" and lp.emitter.calls == 0
    empty = loop()
    assert "0 live episodes" in R.mount(empty, tmp_path / "nothing") and len(empty.hippocampus) == 0
    assert json.loads((tmp_path / "sleep" / "striatum.json").read_text(encoding="utf-8"))["expected"]


def test_a_host_shape_answer_is_written_too_and_never_asks_memory(memory_on):
    lp = loop(store=STORE + ["shakespeare is the author of hamlet"])
    vm = faithful_vm([])
    def one_hop(source, fn):                                 # one hop's tau is 1.0; the control still recovers nothing
        out = vm(source, fn)
        return {**out, "similarity": 1.0} if out.get("similarity") else out
    lp._vm = one_hop
    R.mount(lp)
    rec = lp.ask("Who wrote Hamlet?")                       # the everyday table restates it: no memory step, no emitter
    assert rec["verified"] and rec.get("restated") and "memory" not in rec and lp.emitter.calls == 0
    assert [e.seed for e in lp.hippocampus.episodes] == ["hamlet"] and not lp.striatum.trace
