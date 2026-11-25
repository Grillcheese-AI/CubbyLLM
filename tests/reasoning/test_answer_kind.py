"""Order 0078's addendum, part 1 (cubbyllm/reasoning/answer_kind.py): a try whose value cannot answer the question does
not vote. A negative value is out; a "how many" of a thing that is not a measure must be whole; a money "how much" gets
at most 2 decimals. Values that break no rule are untouched, and a value taken out is a try, never a vote."""
from __future__ import annotations

from cubbyllm.reasoning.answer_kind import asked, broken, is_measure, kept
from cubbyllm.reasoning.speak_vote import vote

APPLES = "Ann has 5 apples and gives away 2. How many apples does she have left?"
HOURS = "A trip takes 150 minutes. How many hours does it take?"
MORE_MINUTES = "Bo reads for 20 minutes. How many more minutes must he read to reach 45?"
BARE = "Cy bought 12 pens and gave half away. How many did he keep?"
SQUARE = "A room is 3 by 4 feet. How many square feet is the room?"
PAID = "Di paid $7 for 3 pens. How much did one pen cost?"
FLOUR = "A cake needs 3 cups of flour for 8 people. How much flour is needed for 1 person?"
NEITHER = "What is 5 minus 9?"


def test_the_ask_and_its_noun():
    assert asked(APPLES) == ("many", "apples")
    assert asked(MORE_MINUTES) == ("many", "minutes")           # a modifier is skipped
    assert asked(BARE) == ("many", None)                        # a verb first: no noun asked
    assert asked(SQUARE) == ("many", "feet")
    assert asked(PAID) == ("much", True) and asked(FLOUR) == ("much", False)
    assert asked(NEITHER) is None
    two = "How much does a pen cost? Ann buys 3 pens. How many pens has she?"
    assert asked(two) == ("many", "pens")                       # the last ask is the one read


def test_measures_with_their_singular_and_british_spelling():
    for w in ("hours", "hour", "feet", "foot", "inch", "percent", "metres", "litre", "dollar"):
        assert is_measure(w), w
    for w in ("apples", "pens", None, "times", "people"):
        assert not is_measure(w), w


def test_negative_is_out_everywhere():
    assert broken(-3, APPLES) == "negative" and broken(-3, NEITHER) == "negative" and broken(-0.0, NEITHER) is None


def test_a_count_of_things_is_whole_and_a_measure_need_not_be():
    assert broken(2.5, APPLES) == "whole" and broken(3, APPLES) is None and broken(3.0000000001, APPLES) is None
    assert broken(2.5, BARE) == "whole"
    assert broken(2.5, HOURS) is None and broken(1.25, MORE_MINUTES) is None and broken(12.5, SQUARE) is None


def test_money_gets_at_most_two_decimals():
    assert broken(2.333333, PAID) == "money" and broken(2.33, PAID) is None and broken(2.5, PAID) is None
    assert broken(0.375, FLOUR) is None                         # no money word: not a money question
    assert broken(2.333333, NEITHER) is None


def test_the_rules_named_are_the_only_ones_applied():
    assert broken(2.5, APPLES, ("negative",)) is None and broken(-1, APPLES, ("whole", "money")) is None


def test_a_value_taken_out_is_a_try_and_never_a_vote():
    tries = [3.0, 3.0, 2.5, 2.5, 2.5, None, -1.0, 3.0]
    assert kept(tries, APPLES) == [3.0, 3.0, None, None, None, None, None, 3.0]
    assert vote(tries, 3)["why"] == "tie"                       # 3 and 2.5 tie without the rules
    v = vote(kept(tries, APPLES), 3)
    assert v["answer"] == 3.0 and v["k"] == 8                   # with them, 3 is said; still 8 tries
