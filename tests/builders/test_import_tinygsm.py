"""Pins for standin/data/import_tinygsm.py: TinyGSM's Python becomes v12e-dialect steps without ever running the
dataset's code; anything but numbers, names and + - * / is a reject; unused steps are pruned; a program that
invents a number the question never states is dropped. No VM (the VM re-check is the CLI's `--vm-sample`).
Run: python -m pytest standin/tests/test_import_tinygsm.py -q -p no:hypothesispytest"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin" / "data"), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import import_tinygsm as tg  # noqa: E402

MARK = '''def simple_math_problem() -> int:
    """
    Mark has 10 crayons. He gives 2 crayons to his younger sister and loses another 4. How many are left?
    """
    crayonsTotal = 10
    crayonsGiven = 2
    crayonsLost = 4
    unused = crayonsTotal * 3
    crayonsLeft = crayonsTotal - crayonsGiven - crayonsLost
    result = crayonsLeft

    return result
'''
Q_MARK = "Mark has 10 crayons. He gives 2 crayons to his younger sister and loses another 4. How many are left?"


def test_straight_line_python_becomes_steps_and_dead_code_is_pruned():
    steps, vals = tg.compile_steps(MARK)
    assert [(s.a, s.op, s.b) for s in steps] == [(tg.Operand("lit", 10), "-", tg.Operand("lit", 2)),
                                                 (tg.Operand("reg", 0), "-", tg.Operand("lit", 4))]
    assert vals == [8.0, 4.0], "the `unused = total * 3` step is pruned"
    prog = tg.render(steps, vals, Q_MARK)
    assert "create s0 : quantity;   # step 0: 10 - 2 = 8" in prog and "assign s1 = 0;\n        add s1, s0;" in prog
    assert prog.rstrip().endswith("sum s1;\n        query s1;\n        return s1;\n    }\n}")


def test_augmented_assignment_and_float_values():
    code = "def f():\n    cost = 3.50\n    cost *= 4\n    cost += 1.25\n    return cost\n"
    steps, vals = tg.compile_steps(code)
    assert vals == [14.0, 15.25] and len(steps) == 2
    assert "assign s0 = 3.5;" in tg.render(steps, vals)


@pytest.mark.parametrize("code, reason", [
    ("def f():\n    x = round(3.3)\n    return x * 2\n", "node Call"),
    ("def f():\n    t = 0\n    for i in range(3):\n        t += i\n    return t\n", "statement For"),
    ("def f():\n    x = 5\n    return x / 0\n", "zero division"),
    ("def f():\n    return 7\n", "no computation"),
    ("import os\ndef f():\n    return 1 + 1\n", "not one function"),
    ("def f():\n    return y + 1\n", "unbound name"),
])
def test_anything_but_arithmetic_is_a_reject_never_run(code, reason):
    with pytest.raises(tg.Reject, match=reason):
        tg.compile_steps(code)


def test_the_literal_filter_drops_an_invented_number_and_keeps_words_and_unit_constants():
    rec = tg.to_record(Q_MARK, MARK, 1)
    assert rec["gold"] == "4" and rec["subtype"] == "steps=2" and rec["difficulty"]["distractors"] == 0
    hike = ("def f():\n    total_time = 5.5\n    distance = 10\n    return distance / total_time\n")
    with pytest.raises(tg.Reject, match="invented number"):
        tg.to_record("They hiked 10 miles for 5 hours and took a 30 minute break. Speed?", hike, 2)
    words = "def f():\n    friends = 4\n    each = 3\n    minutes = friends * each * 60\n    return minutes\n"
    rec = tg.to_record("Four friends each walk 3 hours. How many minutes in all?", words, 3)
    assert rec["gold"] == "720" and rec["difficulty"]["word_numbers"] == 1 and rec["difficulty"]["unit_constants"] == 1


def test_slotted_by_emitter_data_unchanged():
    from emitter_data import fill, slot_record
    rec = tg.to_record("Four friends each walk 3 hours. How many minutes in all?",
                       "def f():\n    return 4 * 3 * 60\n", 4)
    s = slot_record(rec, words=True, step_values=False)
    assert "assign s0 = $N1;" in s["program"] and "mul s0, $N2;" in s["program"] and "mul s1, 60;" in s["program"]
    assert "= 12" not in s["program"] and fill(s["program"], s["spans"]).count("assign s0 = 4;") == 1


def test_the_schools_step_values_read_the_rendered_program_back():
    from cubbyllm.reasoning.school import arith_step_values
    for code in (MARK, "def f():\n    cost = 3.50\n    cost *= 4\n    cost += 1.25\n    return cost / 5\n"):
        steps, vals = tg.compile_steps(code)
        assert arith_step_values(tg.render(steps, vals)) == pytest.approx(vals)
