"""Order 0075, step 2: the grader's controls, each with the class it must get (written before the matcher changed).

- Non-answers (empty, "I don't know", "The facts don't say.", NOT_ATTEMPTED, N/A) are never graded right, by either
  matcher (the kill); under the strict matcher they are refused or declined for every arm.
- Partials and lists against a gold "Paris", and years against a gold "1990": the lenient matcher (today's, 0054's)
  takes most of them as right -- recorded here as measured, the leniency the order counts; the strict matcher
  (`--strict`) takes none of them but "Paris?" and leaves them to a person's read (``unmatched``).
- Each "right" names its rule (step 1); a rule exists exactly when the lenient matcher matches.
"""
import pytest

from validation.system_eval import grade
from validation.system_eval.match_rules import non_answer, rule

NON_ANSWERS = ["", "I don't know", "The facts don't say.", "NOT_ATTEMPTED", "N/A"]

# against the gold "Paris": (answer, lenient class, strict class)
PARTIALS = [
    ("Paris", "right", "right"),
    ("Paris?", "right", "right"),
    ("Paris, Texas", "right", "unmatched"),
    ("Paris or Lyon", "right", "unmatched"),
    ("Paris, Lyon, Nice", "right", "unmatched"),
    ("not Paris", "right", "unmatched"),
]
# against the gold "1990"
YEARS = [
    ("1990", "right", "right"),
    ("in 1990", "right", "right"),
    ("1990s", "right", "unmatched"),
    ("between 1990 and 2000", "right", "unmatched"),
    ("born 1950, died 1990", "right", "unmatched"),
]


@pytest.mark.parametrize("answer", NON_ANSWERS)
@pytest.mark.parametrize("declines", [False, True])
def test_a_non_answer_is_never_right(answer, declines):
    for gold in (["Paris"], ["1990"], ["Michio Sugeno"]):
        assert grade.verdict(True, answer, gold, "simpleqa", declines=declines) != "right"
        assert grade.verdict(True, answer, gold, "simpleqa", declines=declines, strict=True) in ("refused", "declined")
    assert non_answer(answer)


@pytest.mark.parametrize("answer,lenient,strict", PARTIALS)
def test_partials_and_lists(answer, lenient, strict):
    assert grade.verdict(True, answer, ["Paris"], "webq") == lenient
    assert grade.verdict(True, answer, ["Paris"], "webq", strict=True) == strict


@pytest.mark.parametrize("answer,lenient,strict", YEARS)
def test_years(answer, lenient, strict):
    assert grade.verdict(True, answer, ["1990"], "nq") == lenient
    assert grade.verdict(True, answer, ["1990"], "nq", strict=True) == strict


def test_the_year_rule_needs_a_date_gold_and_one_year():
    assert rule("May 3 1990", ["3 May 1990"], "nq", strict=True) == "year"
    assert rule("May 3 1990", ["1990 elections"], "nq", strict=True) is None        # the gold is not a date
    assert rule("1990 to 1995", ["1990"], "nq", strict=True) is None
    assert rule("Lyon 1990", ["Lyon Olympique"], "nq", strict=True) is None


@pytest.mark.parametrize("answer,golds,want", [
    ("Paris", ["Paris"], "equal"),
    ("the Paris", ["Paris"], "equal"),
    ("Pari", ["Paris"], "answer in gold"),
    ("Paris France", ["Paris"], "gold in answer"),
    ("in 1990", ["1990"], "gold in answer"),
    ("May 3 1990", ["1990 elections"], "year"),
    ("Lyon", ["Paris"], None),
    ("abc", ["abcd"], None),              # an answer of 3 characters never matches by containment
])
def test_each_match_names_its_rule(answer, golds, want):
    assert rule(answer, golds, "webq") == want
    assert (rule(answer, golds, "webq") is not None) == grade.matches(answer, golds, "webq")


def test_clutrr_rules():
    assert rule("aunt", ["aunt"], "clutrr") == "relation"
    assert rule("She is his aunt.", ["aunt"], "clutrr") == "relation word"
    assert rule("uncle", ["aunt"], "clutrr") is None


def test_the_strict_matcher_never_takes_more_than_the_lenient_one():
    for answer, golds in [(a, ["Paris"]) for a, _, _ in PARTIALS] + [(a, ["1990"]) for a, _, _ in YEARS]:
        if rule(answer, golds, "webq", strict=True) is not None:
            assert rule(answer, golds, "webq") is not None
