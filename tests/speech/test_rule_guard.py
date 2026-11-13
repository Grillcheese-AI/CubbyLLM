"""Order 0067: the host guard on a written rule plan. A number, a word or a unit the instruction never says makes the
plan "asked"; a plan whose every number, word and unit is in the instruction passes, English and French."""
from __future__ import annotations

from cubbyllm.speech import rule_guard as G

WORDS_LT_50 = {"m": "word count", "op": "<", "v": 50}


def test_a_rule_said_in_full_passes():
    assert G.guard("Answer in fewer than 50 words.", WORDS_LT_50) is None
    assert G.guard("Réponds en moins de cinquante mots.", WORDS_LT_50) is None
    assert G.guard("Use the word river at least twice.", {"m": "count of the word", "op": ">=", "v": 2, "arg": "river"}) is None
    assert G.guard('Wrap your whole answer in double quotation marks.',
                   {"and": [{"m": "starts with", "op": "==", "v": "yes", "arg": '"'},
                            {"m": "ends with", "op": "==", "v": "yes", "arg": '"'}]}) is None


def test_an_invented_number_is_part_one():
    got = G.guard("Explain how tides work. Mention the moon.", {"m": "sentence count", "op": ">=", "v": 22})
    assert got["part"] == 1


def test_zero_and_one_need_no_digit():
    assert G.guard("Include a title.", {"m": "title count", "op": ">=", "v": 1}) is None
    assert G.guard("Write no markdown at all.", {"m": "markdown", "op": "==", "v": "no"}) is None


def test_a_word_the_instruction_never_says_is_part_two():
    got = G.guard("Use keyword paradox seven times.", {"m": "count of the word", "op": ">=", "v": 1, "arg": "riddle"})
    assert got["part"] == 2


def test_a_unit_the_instruction_never_names_is_part_three():
    got = G.guard("Every word must start with the next letter of the alphabet; write 3 lines.",
                  {"m": "sentence count", "op": "==", "v": 3})
    assert got["part"] == 3
    assert [p["part"] for p in G.parts("Give 3 hashtags.", {"m": "sentence count", "op": "==", "v": 3})] == [3]


def test_no_plan_no_guard():
    assert G.guard("anything", None) is None and G.parts("anything", None) == []
