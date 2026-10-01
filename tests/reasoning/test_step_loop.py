"""Pins for cubbyllm/reasoning/step_loop.py (H-E19, the state written back): the re-cut of a gold program into
step rows, the chaining idiom collapsed, the stop row, the round trip of those rows through the loop with a
Python VM, and the refusals (a copied value, an unknown slot, a stop on a slot the host lacks).
Run: python -m pytest tests/reasoning/test_step_loop.py -q -p no:hypothesispytest"""
from __future__ import annotations

from fractions import Fraction

import pytest

from cubbyllm.reasoning import step_loop as sl
from cubbyllm.reasoning.slots import SlotTable, Span, state_block

QUESTION = "You split 308 dollars equally between 2 jars, then save 6 times the amount in one jar and add 185 dollars. How much money?"
PROGRAM = """program GSM2270 implements ISolver {
    type Input = str;
    type Output = quantity;
    @external
    public function solve(input: Input): Output {
        create s0 : quantity;   # step 0: $N1 / $N2
        assign s0 = $N1;
        div s0, $N2;
        create s1 : quantity;   # step 1: 154 * $N3
        assign s1 = 0;
        add s1, s0;
        mul s1, $N3;
        create s2 : quantity;   # step 2: 924 + $N5
        assign s2 = 0;
        add s2, s1;
        add s2, $N5;
        sum s2;
        query s2;
        return s2;
    }
}
"""
SPANS = [{"id": "$N1", "text": "308", "start": 10, "end": 13, "kind": "N"},
         {"id": "$N2", "text": "2", "start": 38, "end": 39, "kind": "N"},
         {"id": "$N3", "text": "6", "start": 56, "end": 57, "kind": "N"},
         {"id": "$N4", "text": "one", "start": 82, "end": 85, "kind": "N", "value": "1"},
         {"id": "$N5", "text": "185", "start": 98, "end": 101, "kind": "N"}]
PROMPT = "Question:\nYou split [N1: 308] dollars equally between [N2: 2] jars, then save [N3: 6] times the amount in [N4: one] jar and add [N5: 185] dollars. How much money?\nProgram:\n"
REC = {"id": "r1", "task": "arithmetic", "subtype": "pf:steps=3", "split": "val", "gold": "1109", "question": QUESTION,
       "prompt": PROMPT, "program": PROGRAM, "spans": SPANS}


def py_vm(program: str):
    """A VM for the tests: the wrapped step's block, literals only, its value."""
    blocks = sl.blocks_of(program)
    if len(blocks) != 1:
        return False, None
    v = sl.simulate(blocks[0].ops, {})
    return (v is not None), (None if v is None else str(int(v)) if v == int(v) else str(float(v)))


def test_recut_rows():
    rows = sl.recut(REC)
    assert [r["gold"] for r in rows] == ["154", "924", "1109", "1109"]
    assert rows[0]["prompt"] == PROMPT                                   # no state yet: the record's own prompt
    assert "So far:\n[S1: 154 = N1 / N2]\n[S2: 924 = S1 * N3]\nProgram:" in rows[2]["prompt"]
    assert rows[1]["program"] == "        create s1 : quantity;   # step 1: $S1 * $N3\n        assign s1 = $S1;\n        mul s1, $N3;\n"
    assert rows[3]["program"] == "        return $S3;\n"
    assert [s["id"] for s in rows[3]["spans"] if s["kind"] == "S"] == ["$S1", "$S2", "$S3"]
    bad = dict(REC, gold="1110")
    assert sl.recut(bad) == []                                            # the gold program must land on its gold


def test_round_trip_through_the_loop():
    rows = sl.recut(REC)

    class Replay:
        def __init__(self):
            self.i = 0
        def emit(self, prompt, max_new_tokens=64, **kw):
            assert prompt == rows[self.i]["prompt"]                      # the loop shows the same prompts the re-cut wrote
            out = rows[self.i]["program"]
            self.i += 1
            return out
    loop = sl.StepLoop(Replay(), vm=py_vm)
    res = loop.solve(QUESTION)
    assert res["refused"] is None and res["answer"] == "1109"
    assert [s["value"] for s in res["steps"]] == ["154", "924", "1109"]


def test_refusals():
    def fixed(text):
        class E:
            def emit(self, prompt, max_new_tokens=64, **kw):
                return text
        return E()
    copied = "        create s0 : quantity;   # step 0: 308 / $N2\n        assign s0 = 308;\n        div s0, $N2;\n"
    assert "copied" in sl.StepLoop(fixed(copied), vm=py_vm).solve(QUESTION)["refused"]
    unknown = "        create s0 : quantity;   # step 0: $S1 / $N2\n        assign s0 = $S1;\n        div s0, $N2;\n"
    assert "unknown slot $S1" in sl.StepLoop(fixed(unknown), vm=py_vm).solve(QUESTION)["refused"]
    assert "unknown slot" in sl.StepLoop(fixed("        return $S1;\n"), vm=py_vm).solve(QUESTION)["refused"]
    assert sl.StepLoop(fixed("hello"), vm=py_vm).solve(QUESTION)["refused"] == "malformed step"
    step = "        create s0 : quantity;   # step 0: $N1 / $N2\n        assign s0 = $N1;\n        div s0, $N2;\n"
    res = sl.StepLoop(fixed(step), vm=py_vm, max_steps=3).solve(QUESTION)   # never stops: the cap refuses
    assert res["answer"] is None and "no stop" in res["refused"] and len(res["steps"]) == 4


def test_wrap_and_state_block():
    prog = sl.wrap_step("        create s1 : quantity;   # step 1: 154 * 6\n        assign s1 = 154;\n        mul s1, 6;\n")
    assert prog.startswith("program STEP implements ISolver") and "return s1;" in prog and "sum s1;" in prog
    assert py_vm(prog) == (True, "924")
    t = SlotTable("a", [Span("$S1", "N1 / N2", -1, -1, "S", "154")])
    assert state_block(t.spans) == "So far:\n[S1: 154 = N1 / N2]"
    assert t.annotate().endswith("So far:\n[S1: 154 = N1 / N2]")
    assert t.check("assign s1 = 154;").copied == ["154"]                # a VM value written out is a copy
