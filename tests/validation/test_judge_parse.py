"""Order 0075 step 6: the judge parser on NeMo Gym issue #5430's own controls, and on the ways a parser fools itself."""
import pytest

from validation.system_eval.judge_parse import parse


@pytest.mark.parametrize("text,letter,grade", [
    # the issue's four controls (SimpleQA's labels: A correct, B incorrect, C not attempted)
    ("NOT ATTEMPTED", None, "NOT_ATTEMPTED"),
    ("C: PARTIAL_ANSWER", "C", "NOT_ATTEMPTED"),
    ("Answer: B", "B", "INCORRECT"),
    ("B: INCORRECT", "B", "INCORRECT"),
    # what else must not read as A
    ("NOT_ATTEMPTED", None, "NOT_ATTEMPTED"),
    ("A correct answer would name the city; this one does not.", None, "NOT_ATTEMPTED"),
    ("A", "A", "CORRECT"),
    ("A: CORRECT", "A", "CORRECT"),
    ("Grade: (A)", "A", "CORRECT"),
    ("A: INCORRECT", "A", "NOT_ATTEMPTED"),
    ("A\nB", None, "NOT_ATTEMPTED"),
    ("", None, "NOT_ATTEMPTED"),
    ("The grade is somewhere between A and B.", None, "NOT_ATTEMPTED"),
])
def test_simpleqa(text, letter, grade):
    assert parse(text, "simpleqa") == {"letter": letter, "grade": grade}


def test_aa_omniscience_labels():
    assert parse("C: PARTIAL_ANSWER", "aa") == {"letter": "C", "grade": "PARTIAL_ANSWER"}
    assert parse("D", "aa")["grade"] == "NOT_ATTEMPTED"
    assert parse("NOT_ATTEMPTED", "aa")["grade"] == "NOT_ATTEMPTED"
    assert parse("Answer: B", "aa")["grade"] == "INCORRECT"


def test_only_a_correct_grade_is_right():
    for t in ("NOT ATTEMPTED", "C: PARTIAL_ANSWER", "Answer: B", "B: INCORRECT", "A and B"):
        for b in ("simpleqa", "aa"):
            assert parse(t, b)["grade"] != "CORRECT"
