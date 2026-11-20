"""Order 0078 step 1: the converter's P6 forms (// and %), and the eval overlap filter on planted overlaps."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from validation.sft import import_tinygsm as tg  # noqa: E402

FLOOR = "def f():\n    eggs = 17\n    box = 4\n    boxes = eggs // box\n    left = eggs % box\n    return boxes * 10 + left\n"


def test_floor_division_and_remainder_are_steps_with_pythons_values():
    steps, vals = tg.compile_steps(FLOOR)
    assert [s.op for s in steps] == ["//", "%", "*", "+"] and vals == [4, 1, 40, 41]
    prog = tg.render(steps, vals)
    assert "assign s0 = 17 // 4;" in prog and "assign s1 = 17 % 4;" in prog
    neg = "def f():\n    a = 0 - 7\n    b = a // 2\n    return b % 3\n"
    assert tg.compile_steps(neg)[1][-2:] == [-4, 2]               # floor and remainder as Python (and the VM) give them


@pytest.mark.parametrize("code", ["def f():\n    a = 5\n    return a // 0\n", "def f():\n    a = 5\n    return a % 0\n"])
def test_a_zero_divisor_is_a_reject(code):
    with pytest.raises(tg.Reject, match="zero division"):
        tg.compile_steps(code)


def test_a_power_is_still_a_reject():
    with pytest.raises(tg.Reject, match="node BinOp"):
        tg.compile_steps("def f():\n    a = 3\n    return a ** 2\n")


def test_the_overlap_filter_drops_a_planted_eval_question(tmp_path):
    q = "A baker sells 12 loaves each morning and 9 loaves each evening for 3 dollars. How much does she make a day?"
    banned = tg.shingles("Planted: A baker sells 12 loaves each morning and 9 loaves each evening for 3 dollars.")
    rows = tmp_path / "rows.jsonl"
    code = "def f():\n    a = 12 + 9\n    return a * 3\n"
    rows.write_text("\n".join(json.dumps({"question": x, "code": code}) for x in
                              (q, "Tom has 12 apples and 9 pears and sells each for 3 dollars. How much?")), encoding="utf-8")
    m = tg.convert(str(rows), str(tmp_path / "out.jsonl"), banned=banned, log=lambda s: None)
    assert m["rejects"].get("overlaps an eval question") == 1 and m["kept"] == 1


def test_floor_or_remainder_of_a_decimal_is_a_reject():
    # Python's 5 // 0.1 is 49.0 (0.1 has no exact binary form); the VM's, and the arithmetic's, is 50
    with pytest.raises(tg.Reject, match="floor or remainder of a decimal"):
        tg.compile_steps("def f():\n    a = 5\n    return a // 0.1\n")


def test_a_marker_print_gives_the_answer_as_a_return_does():
    printed = 'def f():\n    a = 12 + 9\n    result = a * 3\n    print(f"<<<{result}>>>")\n'
    returned = 'def f():\n    a = 12 + 9\n    result = a * 3\n    return result\n'
    assert tg.compile_steps(printed) == tg.compile_steps(returned)
    for other in ('def f():\n    a = 1 + 2\n    print(a)\n', 'def f():\n    a = 1 + 2\n    print(f"<<<{a:.2f}>>>")\n',
                  'def f():\n    a = 1 + 2\n    print(f"{a}")\n'):
        with pytest.raises(tg.Reject):
            tg.compile_steps(other)
