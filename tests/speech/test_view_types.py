"""The second reader's typed grammar (order 0067, step 5; `cubbyllm/speech/view_types.py`). Pinned: a comparison's
literal has its column's type, strings only by == / !=; a part's names are real view names and its own; a call passes
the part's parameters in order; a name or a literal still being written is not judged early; the generator's own
programs pass (here, one of each shape; all 3,900 rows and the SFT set's 4,465 programs were checked when it was built)."""
from __future__ import annotations

import pytest

from cubbyllm.speech import view_types as VT

HEAD = "program Instruction implements ISolve {\n"
PART = ("    function part1(n_sentence: number, sentence_n_word: number, sentence_first_word: number): number {\n"
        "        create k1: number; assign k1 = 0;\n"
        "        create i2: number; assign i2 = 0;\n"
        "        while (i2 < n_sentence) { let x3 = sentence_n_word[i2]; let x4 = sentence_first_word[i2]; "
        'if (x3 != 2) { if (x4 == "-") { add k1, 1; } } add i2, 1; }\n'
        "        create c5: number; assign c5 = 0;\n"
        "        if (k1 == 0) { assign c5 = 1; }\n"
        "        return c5;\n    }\n")
SOLVE = ("    public function solve(mention: str): number {\n"
         "        let p1 = part1(n_sentence, sentence_n_word, sentence_first_word);\n"
         "        return p1;\n    }\n}")
GOLD = HEAD + PART + SOLVE


def test_the_generators_program_passes_whole_and_at_every_prefix():
    assert VT.typed_ok(GOLD)
    assert all(VT.typed_ok(GOLD[:k]) for k in range(len(GOLD)))


@pytest.mark.parametrize("bad", [
    PART.replace('x4 == "-"', "x4 == 9"),                        # a string column with a number
    PART.replace("x3 != 2", 'x3 != "none"'),                     # a number column with a string
    PART.replace('x4 == "-"', 'x4 > "-"'),                       # a string ordered
    PART.replace("sentence_first_word: number", "sentence_first: number"),        # no such column
    PART.replace("while (i2 < n_sentence)", "while (i2 < n_word)"),             # a name the part does not take
])
def test_a_part_that_breaks_a_rule_is_refused(bad):
    assert VT.typed_ok(HEAD + PART + SOLVE)
    assert not VT.typed_ok(HEAD + bad + SOLVE)


def test_a_call_passes_the_parts_parameters_in_order():
    assert not VT.typed_ok(GOLD.replace("part1(n_sentence, sentence_n_word, sentence_first_word)",
                                        "part1(sentence_n_word, n_sentence, sentence_first_word)"))
    assert not VT.typed_ok(GOLD.replace("part1(n_sentence, sentence_n_word, sentence_first_word)",
                                        "part1(n_sentence, sentence_n_word)"))
    assert not VT.typed_ok(GOLD.replace("let p1 = part1(", "let p1 = part2("))      # no such part
    partial = HEAD + PART + "    public function solve(mention: str): number {\n        let p1 = part1(n_sentence, sent"
    assert VT.typed_ok(partial) and not VT.typed_ok(partial.replace("sent", "word"))


def test_the_kinds_follow_the_views():
    assert VT.kind("n_word") == "number" and VT.kind("word_len") == "number" and VT.kind("word_case") == "string"
    assert VT.kind("line_item") == "string" and VT.kind("item_level") == "number"
    assert VT.kind("word_nope") is None and VT.kind("n_thing") is None


def test_the_typed_grammar_refuses_what_the_line_shapes_let_through():
    pytest.importorskip("regex")
    instruction = 'Do not include any sentence that has a number of words other than 2 and that begins with the word "-".'
    answer = f'\ncheck 1 | {instruction}\n---\n'
    g = VT.TypedGrammar(instruction)
    assert g.viable(answer + HEAD + PART)
    assert not g.viable(answer + HEAD + PART.replace('x4 == "-"', "x4 == 9"))


TWO = HEAD + PART + PART.replace("part1", "part2") + "    public function solve(mention: str): number {\n"
CALLS = ("        let p1 = part1(n_sentence, sentence_n_word, sentence_first_word);\n"
         "        let p2 = part2(n_sentence, sentence_n_word, sentence_first_word);\n")
AND = "        if (p1 == 1) { if (p2 == 1) { assign g1 = 1; } }\n"
OR = "        if (p1 == 1) { assign g1 = 1; } if (p2 == 1) { assign g1 = 1; }\n"
IF = "        if (p1 == 0) { assign g1 = 1; } if (p2 == 1) { assign g1 = 1; }\n"


def test_and_is_the_default_and_or_only_when_the_instruction_says_it():
    plain, either, cond = "Use two sentences. Keep it short.", "Either use two sentences, or keep it short.", \
        "If you use two sentences, keep it short."
    G = TWO + CALLS + "        create g1: number; assign g1 = 0;\n"
    assert VT.joins_ok(G + AND, plain)
    assert not VT.joins_ok(G + OR, plain) and VT.joins_ok(G + OR, either)
    assert not VT.joins_ok(G + IF, plain) and VT.joins_ok(G + IF, cond)
    assert not VT.joins_ok(G + "        if (p1 == 0", plain)                      # refused as it opens
    assert VT.joins_ok(G + OR, "Soit deux phrases, soit une réponse courte.")
    assert not VT.joins_ok(G + OR, "Keep the colors bright.")                     # "or" inside a word is no cue


def test_return_comes_only_once_every_part_is_joined():
    instr = "Use two sentences. Keep it short."
    whole = TWO + CALLS + "        create g1: number; assign g1 = 0;\n" + AND + "        return g1;\n    }\n}"
    assert VT.joins_ok(whole, instr)
    dropped = TWO + CALLS + "        create g1: number; assign g1 = 0;\n        if (p1 == 1) { assign g1 = 1; }\n"
    assert not VT.joins_ok(dropped + "        r", instr)                     # part 2 never joined: no return yet
    assert not VT.joins_ok(TWO + CALLS + "        return p1;", instr)
    assert VT.joins_ok(TWO + CALLS + "        create g1: number; assign g1 = 0;\n" + AND + "        return g", instr)
    assert not VT.joins_ok(TWO + CALLS + "        create g1: number; assign g1 = 0;\n" + AND + "        return p1;", instr)


def test_no_value_is_read_twice_and_no_group_is_left_unwritten():
    instr = "Use two sentences. Keep it short."
    top = TWO + CALLS + "        create g1: number; assign g1 = 0;\n" + AND
    assert not VT.joins_ok(top + "        create g2: number; assign g2 = 0;\n        if (g1 == 1) { if (g1", instr)
    assert not VT.joins_ok(top + "        create g2: number; assign g2 = 0;\n        if (p1", instr)     # p1 joined
    assert not VT.joins_ok(TWO + CALLS + "        create g1: number; assign g1 = 0;\n        create g", instr)
    assert not VT.joins_ok(TWO + CALLS + "        create g1: number; assign g2", instr)
    assert not VT.joins_ok(TWO + CALLS + "        create g1: number; assign g1 = 0;\n        if (p1 == 1) { a", instr)
    assert VT.joins_ok(TWO + CALLS + "        create g1: number; assign g1 = 0;\n        if (p1 == 1) { a",
                       "Use two sentences or keep it short.")             # may still become an "or" group


def test_parameters_views_and_part_order():
    p4 = "    function part1(n_sentence: number, sentence_last_word: number, sentence_last_word: numbe"
    assert not VT.typed_ok(HEAD + p4)                                            # a name declared twice
    assert not VT.typed_ok(HEAD + "    function part1(n_sentence: number, sentence_las_wo")  # no real name ahead
    assert VT.typed_ok(HEAD + "    function part1(n_sentence: number, sentence_last_wo")
    assert not VT.typed_ok(HEAD + PART.replace("let x3 = sentence_n_word[i2]", "let x3 = word_len[i2]")
                           .replace("n_sentence: number, sentence_n_word", "n_sentence: number, word_len"))  # another view
    assert not VT.typed_ok(HEAD + PART + PART)                                  # part1 defined twice
    assert not VT.typed_ok(HEAD + PART + "    public function solve(mention: str): number {\n"
                           "        let p2 = part1(")                           # p1 = part1 first
    assert not VT.typed_ok(HEAD + PART + PART.replace("part1", "part2") + "    public function solve(mention: str): number {\n"
                           "        let p1 = part1(n_sentence, sentence_n_word, sentence_first_word);\n        create g")
