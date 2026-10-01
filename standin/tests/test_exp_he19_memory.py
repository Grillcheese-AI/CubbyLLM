"""Pins for H-E19's two credit-free reads: validation/exp_he19_next_step.py (teacher-forced next block: the
block parser, the step simulator, the stop score) and validation/exp_he19_units.py (unit tags from the words
around a number, the unit algebra, the typed enumeration). No model, no cubelang.
Run: python -m pytest standin/tests/test_exp_he19_memory.py -q -p no:hypothesispytest"""
from __future__ import annotations

import pathlib
import sys
from fractions import Fraction as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp_he19_next_step as ns  # noqa: E402
import exp_he19_units as un  # noqa: E402

PROG = """program X implements ISolver {
    public function solve(input: Input): Output {
        create s0 : quantity;   # step 0: $N1 / $N2
        assign s0 = $N1;
        div s0, $N2;
        create s1 : quantity;   # step 1: 154 * $N3
        assign s1 = 0;
        add s1, s0;
        mul s1, $N3;
        sum s1;
        query s1;
        return s1;
    }
}
"""
SLOTS = {"$N1": F(308), "$N2": F(2), "$N3": F(6)}


def test_blocks_and_first_block():
    lines, blocks, first = ns.blocks_of(PROG)
    assert len(blocks) == 2 and [o for o, *_ in blocks[1]] == ["assign", "add", "mul"]
    kind, ops, complete = ns.first_block("        create s1 : quantity;   # step 1\n        assign s1 = $N3;\n        mul s1, s0;\n        create s2 : quantity;\n")
    assert kind == "step" and complete and ns.canon_ops(ops) == [("assign", "$N3"), ("mul", "s0")]
    kind, _, complete = ns.first_block("        create s1 : quantity;\n        assign s1 = $N3;\n        mul s1")   # cut by the budget
    assert kind == "step" and not complete
    assert ns.first_block("        sum s1;\n        query s1;\n")[0] == ns.CLOSE


def test_value_scores_a_step_however_written():
    _, blocks, _ = ns.blocks_of(PROG)
    regs = {"s0": F(154)}
    other = ns.ops_of(["        assign s1 = $N3;", "        mul s1, s0;"])        # 6 * 154: the same number, another spelling
    sc = ns.score_block(blocks[1], "step", other, True, regs, SLOTS)
    assert sc["value"] and not sc["exact"] and not sc["op"]
    wrong = ns.ops_of(["        assign s1 = $N3;", "        add s1, s0;"])
    assert not ns.score_block(blocks[1], "step", wrong, True, regs, SLOTS)["value"]
    assert ns.score_block(ns.CLOSE, "step", other, True)["ran_on"]
    assert ns.score_block(blocks[1], ns.CLOSE, [], True, regs, SLOTS)["stopped_early"]


def test_unit_tags():
    q = "At the fair, 70 students sign up and 26 more students join; each student receives 79 stickers. How many stickers?"
    assert un.unit_after(q, q.index("70") + 2)[0] == "student"
    assert un.unit_after(q, q.index("26") + 2) == ("student", None, True)          # `more` is skipped, not a scalar
    assert un.unit_after("She has 3 more than him.", 8) == (None, None, True)      # `3 more than` is
    assert un.unit_after("5 miles per hour", 1) == ("mile", "hour", True)
    assert un.rate_before(q, q.index("79")) == "student"
    assert un.unit_after("sells 36, then bakes 15 loaves", 8)[2] is False          # elided
    assert un.asked_unit(q) == "sticker" and un.asked_unit("How much money is left?") == "dollar"


def test_typed_enumeration_prunes_but_does_not_pick():
    leaves = [(F(70), {"student": 1}), (F(26), {"student": 1}), (F(2), {"group": 1}), (F(79), {"sticker": 1, "student": -1})]
    untyped = un.enumerate_values(leaves, "sticker", False, True)
    typed = un.enumerate_values(leaves, "sticker", True, True)
    gold = F(70 + 26) / 2 * 79
    assert gold in typed and gold in untyped and len(typed) < len(untyped)
    assert len(typed) > 1                                                          # + vs - survive typing
    assert un.matches({"sticker": 1, "group": -1}, "sticker")                      # a per-group answer to `how many stickers`
    assert not un.matches({"student": 1}, "sticker")
    elided = [(F(60), {"loave": 1}), (F(36), {"?": 1}), (F(15), {"loave": 1})]
    assert F(60 - 36 + 15) in un.enumerate_values(elided, "loave", True, True)


def test_step_format_teacher_forced_read(tmp_path, monkeypatch):
    """exp_he19_step_loop --teacher-forced: each gold step row's emission is scored by the VALUE it computes over
    the row's slots (N and the written-back S), and the stop row by `return $S<K>;`."""
    import json
    import types
    import exp_he19_step_loop as esl
    import standin.emitter as se
    from cubbyllm.reasoning import step_loop as sl
    program = ("program X implements ISolver {\n    public function solve(input: Input): Output {\n"
               "        create s0 : quantity;   # step 0: $N1 / $N2\n        assign s0 = $N1;\n        div s0, $N2;\n"
               "        create s1 : quantity;   # step 1\n        assign s1 = 0;\n        add s1, s0;\n        mul s1, $N3;\n"
               "        create s2 : quantity;   # step 2\n        assign s2 = 0;\n        add s2, s1;\n        add s2, $N4;\n"
               "        sum s2;\n        query s2;\n        return s2;\n    }\n}\n")
    spans = [{"id": "$N1", "text": "308", "start": 0, "end": 3, "kind": "N"},
             {"id": "$N2", "text": "2", "start": 4, "end": 5, "kind": "N"},
             {"id": "$N3", "text": "6", "start": 6, "end": 7, "kind": "N"},
             {"id": "$N4", "text": "185", "start": 8, "end": 11, "kind": "N"}]
    rec = {"id": "r1", "task": "arithmetic", "subtype": "steps=3", "split": "val", "gold": "1109", "question": "q",
           "prompt": "Question:\nq\nProgram:\n", "program": program, "spans": spans}
    rows = sl.recut(rec)
    assert [r["gold"] for r in rows] == ["154", "924", "1109", "1109"]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    by_prompt = {r["prompt"]: r["program"] for r in rows}
    wrong_k1 = "        create s1 : quantity;   # step 1: $S1 + $N3\n        assign s1 = $S1;\n        add s1, $N3;\n"

    class Fake:
        name = "fake"
        def __init__(self, *a, **k): pass
        def emit(self, prompt, max_new_tokens=64, **kw):
            return wrong_k1 if prompt == rows[1]["prompt"] else by_prompt[prompt]

    monkeypatch.setattr(se, "Cubby450mEmitter", Fake)
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(esl, "HERE", str(tmp_path))
    esl.teacher_forced(types.SimpleNamespace(teacher_forced=str(path), limit=0, gguf="", export="", adapter="",
                                             tokenizer="", tag="t", max_new=64))
    out = json.load(open(tmp_path / "logs" / "exp_he19_step_loop_t.json", encoding="utf-8"))["summary"]
    assert out["step0"]["ok"] == 1.0 and out["stop"]["ok"] == 1.0
    assert out["step1+"]["n"] == 2 and abs(out["step1+"]["ok"] - 0.5) < 1e-9     # k=1 wrong (+ for *), k=2 right
