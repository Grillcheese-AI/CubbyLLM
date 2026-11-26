"""Order 0085 (cubbyllm/records/arith_trace.py, validation/replay_arith_cert.py, arith_freeze.py): the record of an
arithmetic answer is written from what the loop did, and its replay from the record and the question alone comes out
"same"; every planted change is caught: a span moved by one character, a step's program edited, a step's value
changed, a try's value changed so the vote no longer reaches m, a $K that is not in the closed list. A refusal says
why. Recording changes nothing about the answer. The freeze names what moved."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "validation", ROOT / "tests" / "reasoning", ROOT / "tests" / "validation"):
    sys.path.insert(0, str(p))
import arith_freeze as F  # noqa: E402
import exp_0078_dev_read as R  # noqa: E402
import replay_arith_cert as P  # noqa: E402
from test_exp_0078_dev_read import Sampler  # noqa: E402
from test_step_loop import QUESTION, py_vm  # noqa: E402

from cubbyllm.reasoning import step_loop as sl  # noqa: E402
from cubbyllm.reasoning.answer_kind import RULES  # noqa: E402
from cubbyllm.records.arith_trace import read_span, trace  # noqa: E402

HOURS = "A film runs 90 minutes and another runs 2 hours. How many minutes do both run?"


def _chains(question=QUESTION, k=4, emitter=None):
    e = emitter or Sampler()
    loop = sl.StepLoop(e, vm=py_vm)
    keep: list = []
    rows = R.tries(e, loop, question, qi=3, k=k, keep=keep)
    return keep, rows


def test_recording_changes_nothing_about_the_answer():
    keep, rows = _chains()
    e = Sampler()
    plain = R.tries(e, sl.StepLoop(e, vm=py_vm), QUESTION, qi=3, k=4)
    assert rows == plain and keep[0].log and not R.Chain(QUESTION, None, py_vm).record


def test_a_spoken_answer_gets_a_certificate_that_replays_same():
    keep, _ = _chains()                                  # tries 0, 2 reach 1109; 1, 3 go off the rows
    rec = trace(QUESTION, keep, k=4, m=2, rules=RULES, freeze={"x": 1})
    assert rec["kind"] == "certificate" and rec["answer"] == "1109" and rec["chosen"] == 0
    assert [s["value"] for s in rec["steps"]] == ["154", "924", "1109"] and rec["stop"] == "$S3"
    assert rec["vote"] == {"answer": "1109", "votes": 2, "why": None} and rec["freeze"] == {"x": 1}
    n = [s for s in rec["slots"] if s["kind"] == "N"]
    assert [(s["id"], s["text"], QUESTION[s["start"]:s["end"]]) for s in n][:2] == [("$N1", "308", "308"), ("$N2", "2", "2")]
    assert {c["rule"]: c["outcome"] for c in rec["checks"]} == {"negative": "satisfied", "whole": "does not apply",
                                                               "money": "satisfied"}
    assert all(len(s["ran_sha256"]) == 64 for s in rec["steps"])
    assert P.replay(json.loads(json.dumps(rec)), QUESTION, py_vm) == "same"


def test_every_planted_change_is_caught():
    keep, _ = _chains()
    rec = trace(QUESTION, keep, k=4, m=2, rules=RULES)
    def planted(change):
        r = copy.deepcopy(rec)
        change(r)
        return P.replay(r, QUESTION, py_vm)
    def move(r):
        r["slots"][0]["start"] += 1
        r["slots"][0]["end"] += 1
    assert planted(move) == "slots.$N1.span"                                      # a span moved by one character
    def edit(r):
        r["steps"][0]["program"] = r["steps"][0]["program"].replace("div s0, $N2", "mul s0, $N2")
    assert planted(edit) == "steps.0.program"                                     # a step's program edited
    assert planted(lambda r: r["steps"][1].update(value="925")) == "steps.1.value"    # a value changed
    def one_less(r):
        r["tries"][2]["value"] = "1108"
    assert planted(one_less) == "vote"                                            # the vote no longer reaches m
    assert planted(lambda r: r["slots"].append({"id": "$K1", "kind": "K", "fact": "61 minutes per hour",
                                                 "value": "61"})) == "slots.$K1.fact"   # a $K not in the list
    assert planted(lambda r: r.update(answer="1110")) == "answer"
    assert P.replay(rec, QUESTION.replace("308", "309"), py_vm) == "question_sha256"


def test_a_constant_and_a_number_word_replay_from_the_list_and_the_span():
    class Conv:
        def emit_samples(self, prompt, n, **kw):
            if "So far" not in prompt:
                return ["        create s0 : quantity;   # step 0: $N2 * $K1\n        assign s0 = $N2;\n"
                        "        mul s0, $K1;\n"] * n
            if "S2" not in prompt:
                return ["        create s1 : quantity;   # step 1: $N1 + $S1\n        assign s1 = $N1;\n"
                        "        add s1, $S1;\n"] * n
            return ["        return $S2;\n"] * n
    loop = sl.StepLoop(Conv(), vm=py_vm, world=__import__("cubbyllm.reasoning.arith_world", fromlist=["x"]).ArithmeticWorld())
    keep: list = []
    R.tries(Conv(), loop, HOURS, qi=0, k=2, keep=keep)
    rec = trace(HOURS, keep, k=2, m=2, rules=RULES)
    assert rec["kind"] == "certificate" and rec["answer"] == "210"
    assert {"id": "$K1", "kind": "K", "fact": "60 minutes per hour", "value": "60"} in rec["slots"]
    assert P.replay(rec, HOURS, py_vm) == "same"
    assert read_span("four") == "4" and read_span("1,200") == "1200" and read_span("3.50") == "3.50"
    assert read_span("four apples") is None


def test_a_refusal_says_why_and_replays():
    class Off:
        def emit_samples(self, prompt, n, **kw):
            return ["        create s0 : quantity;   # step 0: 308 / $N2\n        assign s0 = 308;\n"
                    "        div s0, $N2;\n"] * n                                 # a copied literal: refused
    keep, _ = _chains(emitter=Off())
    rec = trace(QUESTION, keep, k=4, m=2, rules=RULES)
    assert rec["kind"] == "refusal" and rec["why"] == "no try reached a value"
    assert rec["try reasons"] == {"copied instead of a slot": 4} and "steps" not in rec
    assert keep[0].log[0]["refused"].startswith("copied") and P.replay(rec, QUESTION, py_vm) == "same"
    keep, _ = _chains()
    rec = trace(QUESTION, keep, k=4, m=3, rules=RULES)
    assert rec["kind"] == "refusal" and rec["why"] == "too few agree" and P.replay(rec, QUESTION, py_vm) == "same"


def test_a_rule_that_takes_out_the_value_is_the_reason():
    class Neg:
        def emit_samples(self, prompt, n, **kw):
            if "So far" not in prompt:
                return ["        create s0 : quantity;   # step 0: $N2 - $N1\n        assign s0 = $N2;\n"
                        "        sub s0, $N1;\n"] * n
            return ["        return $S1;\n"] * n
    keep, _ = _chains(emitter=Neg())
    assert keep[0].answer == "-306"
    assert trace(QUESTION, keep, k=4, m=2, rules=RULES)["why"] == "rule: negative"
    assert trace(QUESTION, keep, k=4, m=2)["kind"] == "certificate"                 # rules off: it would be said


def test_the_freeze_names_what_moved(tmp_path):
    ad = tmp_path / "adapter"
    ad.mkdir()
    (ad / F.WEIGHTS).write_bytes(b"weights")
    s = {"k": 8, "m": 7, "temperature": 0.7, "top-k": 40, "rules": list(RULES)}
    a = F.freeze(str(ad), s, None)
    assert a == F.freeze(str(ad), dict(s), None) and F.differences(a, a) == []
    assert a["adapter"]["name"] == "adapter" and len(a["speak rule"]["sha256"]) == 64
    (ad / F.WEIGHTS).write_bytes(b"weights2")
    b = F.freeze(str(ad), {**s, "m": 6}, None)
    assert F.differences(a, b) == ["adapter.sha256", "settings.m", "settings.sha256"]


def test_the_writer_writes_a_record_a_question_and_the_freeze_at_both_ends(tmp_path):
    import arith_records as W
    ad = tmp_path / "s1"
    ad.mkdir()
    (ad / F.WEIGHTS).write_bytes(b"weights")
    rows = tmp_path / "dev_rows.jsonl"
    rows.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "records.jsonl"
    w = W.Writer(str(out), str(ad), {"k": 4, "m": 2, "temperature": 0.7, "top-k": 40, "rules": list(RULES)}, RULES,
                 exe="", name="t", calibration=str(rows), calibration_at="2026-11-24T00:00:00Z")
    keep, _ = _chains()
    assert w.write("q1", 85, QUESTION, keep) >= 0
    fz = w.close()
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["id"] == "q1" and rec["i"] == 85 and rec["kind"] == "certificate" and rec["cheez why"] is None
    assert rec["freeze"] == fz["start"] and not fz["void"] and fz["records"] == 1
    assert json.loads((tmp_path / "records.jsonl.freeze.json").read_text(encoding="utf-8"))["differences"] == []
    assert P.replay(rec, QUESTION, py_vm) == "same"
