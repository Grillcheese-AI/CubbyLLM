"""validation/system_eval/ -- order 0054's harness, pinned on fixtures.

Pinned (the order's acceptance tests): the grading rule (a right alias, a wrong value, a refusal, a stale WebQuestions
gold read rather than counted wrong); C-VM read from C's records equals a second pass whose VM certifies what it
recovers; the harness makes no request to a model API. And the split seat: the batched programs are
``Cubby450mEmitter.emit``'s, a replayed prompt with no program stops the run past the loop's own handler, and the
recorder keeps each prompt once.
"""
from __future__ import annotations

import pathlib
import re
import socket
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation", ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from system_eval import grade as G  # noqa: E402
from system_eval import seats as SE  # noqa: E402
from system_eval.run_c import kind, proposer, timed  # noqa: E402


def _rows(verdicts):
    return [{"id": f"q{i}", "verdict": v} for i, v in enumerate(verdicts)]


def test_the_grading_rule_alias_wrong_refusal_and_a_stale_gold():
    assert G.verdict(True, "USA", ["United States of America", "USA"], "simpleqa") == "right"       # an alias
    assert G.verdict(True, "Paris", ["London"], "simpleqa") == "unmatched"                         # read, then wrong
    assert G.verdict(False, None, ["London"], "simpleqa") == "refused"
    assert G.verdict(True, "Miami Heat", ["Chicago Bulls"], "webq") == "unmatched"                 # 2013's gold
    rows = _rows([G.verdict(True, "Paris", ["London"], "simpleqa"),
                  G.verdict(True, "Miami Heat", ["Chicago Bulls"], "webq"),
                  G.verdict(True, "USA", ["USA"], "simpleqa"), "refused"])
    r = G.rates(rows, {"q0": "wrong", "q1": "right"})                                             # the read decides
    assert (r["right"], r["wrong"], r["refused"], r["spoken"]) == (2, 1, 1, 3) and not r["estimated"]
    assert r["wrong_of_spoken"]["rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert G.verdict(True, "Lewis is Jason's grandson.", ["grandson"], "clutrr") == "right"
    assert G.verdict(True, "granddaughter", ["grandson"], "clutrr") == "unmatched"


def test_an_unread_rest_is_estimated_from_the_sample_and_c_is_read_whole():
    rows = _rows(["unmatched"] * 400 + ["right"] * 100)
    picked = G.to_read(rows, "M")
    assert len(picked) == G.READ_SAMPLE and picked == G.to_read(rows, "M")                       # seeded
    assert len(G.to_read(rows, "C")) == 400
    reads = {i: ("right" if k % 4 == 0 else "wrong") for k, i in enumerate(picked)}             # a quarter right
    r = G.rates(rows, reads)
    assert r["estimated"] and r["right"] == pytest.approx(100 + 100, abs=1) and r["wrong"] == pytest.approx(300, abs=1)
    lo, hi = r["wrong_of_spoken"]["ci95"]
    assert lo < r["wrong_of_spoken"]["rate"] < hi


def test_cvm_from_records_equals_a_second_pass_that_certifies_what_it_recovers(monkeypatch):
    from test_ask import FakeEmitter, STORE
    from test_search_learn import DictSource, LookupStore
    from cubbyllm.reasoning.ask import AskLoop
    from test_pipeline_plan_refusal import faithful_vm
    monkeypatch.setenv("CB_RECORD_UNCERTIFIED", "1")
    q = "What is the capital of the country of citizenship of Marie?"
    plans = {q: ("marie", ["country of citizenship", "capital"])}
    src = {"marie": ["canada is the country of citizenship of marie"], "canada": ["ottawa is the capital of canada"]}
    vm = faithful_vm([])

    def doubting(source, fn):                         # the VM recovers the value but does not certify it
        r = vm(source, fn)
        return {**r, "similarity": 0.01} if r.get("result") is not None else r

    def loop(run_fn):
        return AskLoop(FakeEmitter(plans), world=LookupStore(STORE), source=DictSource(src), lexicon=False, run_fn=run_fn)
    first = loop(doubting).ask(q)
    assert not first["verified"] and first["uncertified"]
    second = loop(timed(doubting, {}, permissive=True)).ask(q)
    assert second["verified"] and second["answer"] == first["uncertified"]


def test_the_harness_calls_no_model_api():
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "validation" / "system_eval").glob("*.py"))
    for pat in (r"\bimport (openai|anthropic|litellm)\b", r"openrouter\.ai", r"api\.openai\.com", r"api\.anthropic\.com",
                r"OPENROUTER|OPENAI_API_KEY|ANTHROPIC_API_KEY", r"\brequests\.post\b", r"urllib\.request"):
        assert not re.search(pat, src), pat


class FakeTrunk:
    """``Cubby450mTrunk``'s interface: a row's generation is its own last three ids, then </s>."""
    max_ctx, eos = 64, 0

    class tokenizer:                                   # noqa: N801 -- the tokenizers.Tokenizer surface used
        @staticmethod
        def encode(text):
            return type("E", (), {"ids": [ord(c) % 50 + 1 for c in text]})()

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            return "".join(chr(i + 64) for i in ids)

        @staticmethod
        def token_to_id(t):
            return None

    adapter_dirs = {"programs": "x", "talk": "y"}

    def __init__(self):
        self.batches = []

    def generate(self, role, rows, max_new_tokens, temperature=0.0, top_k=40, seed=None):
        self.batches.append(len(rows))
        return [list(r[-3:])[:max_new_tokens] + [self.eos] for r in rows]


def test_batched_programs_are_the_emitters_and_a_missing_one_stops_the_run():
    from cubbyllm.emit.seats_450m import Cubby450mEmitter
    tr = FakeTrunk()
    ps = ["[E1: Who] wrote it?", "When did [E1: Rome] fall?", "x" * 100]
    got = SE.programs(tr, ps, [300, 2, 300])
    assert got == [Cubby450mEmitter(FakeTrunk()).emit(p, max_new_tokens=n) for p, n in zip(ps, [300, 2, 300])]
    assert sorted(tr.batches) == [1, 2]                                    # one batch per budget
    rp = SE.Replay(dict(zip(ps, got)))
    assert rp.emit(ps[1]) == got[1]
    with pytest.raises(SE.MissingProgram):
        try:
            rp.emit("never recorded")
        except Exception:                                                 # noqa: BLE001 -- the loop's own handler
            pytest.fail("a missing program must not be turned into a refusal")


def test_the_recorder_keeps_each_prompt_once_and_proposes_nothing(tmp_path):
    rc = SE.Recorder()
    assert rc.emit("a", max_new_tokens=300) == SE.NO_PLAN
    rc.emit("b"); rc.emit("a")
    f = tmp_path / "prompts.jsonl"
    assert rc.save(str(f)) == 2 and rc.save(str(f)) == 0 and rc.calls == 3


def test_proposer_and_refusal_kinds():
    assert proposer({"restated": "the author of X", "verified": True}, False, None) == "everyday"
    assert proposer({"plan": ["capital"], "verified": True}, True, "function plan() {}") == "450m"
    assert proposer({"plan": None}, True, SE.NO_PLAN) is None
    assert proposer({"kind": "relation", "verified": True}, False, None) == "rules"
    assert kind("emit_error: x") == "no_plan" and kind("retrieval_exhausted") == "fact_missing"
    assert kind("vm_verify_failed") == "not_certified" and kind("weird") == "other:weird"


def test_the_m_arm_makes_no_network_call(monkeypatch):
    from system_eval.gpu import m_answers

    def no_net(*a, **k):
        raise AssertionError("a network call")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    rows = [{"id": "1", "bench": "simpleqa", "question": "Who wrote it?", "golds": ["x"], "meta": {}},
            {"id": "2", "bench": "clutrr", "question": "How is B related to A?", "golds": ["son"],
             "meta": {"facts": ["b is the son of a"]}}]
    assert len(m_answers(FakeTrunk(), rows)) == 2
