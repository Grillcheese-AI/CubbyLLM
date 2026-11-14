"""Rule 9 of the second reader's grammar (order 0067; `cubbyllm/speech/view_words.py`): a part's numbers, positions
and views are in its own check line's text; one part per check. Pinned on the dev read's own failures ("fewer than 48
words" read as paragraphs, "exactly one paragraph" read as words, a 3 for a 5); all 12,923 generator parts kept it
when it was built."""
from __future__ import annotations

from cubbyllm.speech import view_words as VW

HEAD = "\ncheck 1 | Your reply should have fewer than 48 words.\ncheck 2 | The first list item has a number of words other than 5.\n"
P1 = ("program Instruction implements ISolve {\n    function part1(n_word: number): number {\n"
      "        create c1: number; assign c1 = 0;\n        if (n_word >= 48) { assign c1 = 1; }\n"
      "        create c2: number; assign c2 = 0;\n        if (c1 == 0) { assign c2 = 1; }\n        return c2;\n    }\n")
P2 = ("    function part2(item_n_word: number, n_item: number): number {\n        create c1: number; assign c1 = 0;\n"
      "        create p2: number; assign p2 = 0;\n"
      "        if (p2 >= 0) { if (p2 < n_item) { let x3 = item_n_word[p2]; if (x3 == 5) { assign c1 = 1; } } }\n"
      "        create c4: number; assign c4 = 0;\n        if (c1 == 0) { assign c4 = 1; }\n        return c4;\n    }\n")
T = VW.texts(HEAD)


def test_the_generators_parts_pass():
    assert VW.words_ok(P1 + P2, T)
    assert all(VW.words_ok((P1 + P2)[:k], T) for k in range(len(P1 + P2)))


def test_a_view_its_words_do_not_name_is_refused():
    assert not VW.words_ok(P1.replace("n_word", "n_paragraph"), T)


def test_a_number_or_a_position_its_words_do_not_give_is_refused():
    assert not VW.words_ok(P1 + P2.replace("x3 == 5", "x3 == 3"), T)
    assert not VW.words_ok(P1.replace(">= 48", ">= 47"), T)
    assert VW.words_ok(P1.replace(">= 48", ">= 4"), T) is False and VW.words_ok(P1[:P1.index("48") + 1], T)
    assert not VW.words_ok(P1 + P2.replace("assign p2 = 0", "assign p2 = 2"), T)     # "first" is position 0
    assert VW.allowed("the third list item")[1] >= {0, 2}


def test_one_part_per_check():
    assert not VW.words_ok(P1 + P2 + P2.replace("part2", "part3"), T)
    solve = "    public function solve(mention: str): number {\n        let p1 = part1(n_word);\n        create g"
    assert not VW.words_ok(P1 + solve, T)                                            # check 2 has no part yet


def test_never_a_dead_end():
    """The first rules 1-9 read stopped mid-name (`part3(n_paragraph: numb`) and after a last part (`function part`):
    both are refused where another way on still exists."""
    assert not VW.words_ok("program Instruction implements ISolve {\n    function part1(n_parag", T)   # check 1: words
    assert VW.words_ok("program Instruction implements ISolve {\n    function part1(n_wo", T)
    assert not VW.words_ok(P1 + P2 + "    f", T)                                     # both checks have their part
    assert VW.words_ok(P1 + P2 + "    public function solve(mention: str): number {", T)
    odd = VW.texts("\ncheck 1 | Write fewer than 48 words.\ncheck 3 | The first list item has a number of words other than 5.\n")
    assert VW.words_ok(P1 + P2, odd)                                                 # checks numbered 1, 3: by order
    assert VW.words_ok(P1, {})                                                       # no check line: rule 9 is off


def test_strings_are_held_to_the_column_and_the_text():
    q = VW.quotes('\ncheck 1 | Do not begin a sentence with the word "calme".\n')
    line = '        while (i2 < n_sentence) { let x3 = sentence_first_word[i2]; if (x3 == "'
    assert VW._strings_ok(line + 'calme") { add k1, 1; } add i2, 1; }', q["1"])
    assert not VW._strings_ok(line + 'nereaine', q["1"])                               # the endless string
    assert VW._strings_ok('        while (i2 < n_item) { let x3 = item_marker[i2]; if (x3 == "*") {', set())   # its own
    assert VW._strings_ok('        if (p2 < n_json) { let x3 = json_key[p2]; if (x3 == "anything', set())    # nothing to hold
