"""Order 0073, step 1: one spec language, one evaluator, one compiler and one join, in the core. The builders' old
names are the core's objects (not copies); the core's three modules name nothing under ``validation/``; and what they
write is pinned on a small spec, so a change to the compiler's text shows here before it shows in a stored row."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders")]
import view_compile as OLD_VC  # noqa: E402
import view_join as OLD_VJ  # noqa: E402
import view_programs as OLD_VP  # noqa: E402

from cubbyllm.speech import view_compile as VC  # noqa: E402
from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

SHORT = {"q": "n", "view": "sentence", "op": "<=", "v": 3}
FILTER = {"q": "none", "view": "word", "where": [{"col": "first_lower", "op": "==", "v": "z"}]}


def test_the_builders_names_are_the_cores_objects():
    assert OLD_VP.evaluate is VS.evaluate and OLD_VP.atoms is VS.atoms and OLD_VP.cmp is VS.cmp
    assert OLD_VP.FILTERS is VS.FILTERS and OLD_VP._match is VS.match
    assert OLD_VC.body is VC.body and OLD_VC.program is VC.program and OLD_VC.views_of is VC.views_of
    assert OLD_VJ.join is VJ.join and OLD_VJ.spec is VJ.spec and OLD_VJ.program is VJ.program and OLD_VJ.HEAD is VJ.HEAD


@pytest.mark.parametrize("name", ["view_spec", "view_compile", "view_join"])
def test_the_core_names_nothing_under_validation(name):
    src = (ROOT / "cubbyllm" / "speech" / f"{name}.py").read_text(encoding="utf-8")
    code = [ln for ln in src.splitlines() if ln.startswith(("import ", "from ")) or "sys.path" in ln]
    assert code and not any("validation" in ln for ln in code)


def test_the_evaluator_reads_a_spec_as_its_words_say():
    views = {"sentence": [{}, {}, {}, {}], "word": [{"first_lower": "a"}, {"first_lower": "z"}]}
    assert VS.evaluate(SHORT, views) is False                               # four sentences, at most three asked
    assert VS.evaluate({"not": SHORT}, views) is True
    assert VS.evaluate(FILTER, views) is False                              # a word begins with z
    assert VS.evaluate({"unless": [SHORT, {"not": FILTER}]}, views) is True  # a unless b: a or b
    assert VS.evaluate({"if": FILTER, "then": SHORT}, views) is True        # b whenever a, and a does not hold
    assert [a["q"] for a in VS.atoms({"and": [SHORT, {"not": FILTER}]})] == ["n", "none"]


def test_the_compilers_text_is_pinned():
    assert VC.body(SHORT) == ("create c1: number; assign c1 = 0;\n"
                              "if (n_sentence <= 3) { assign c1 = 1; }\n"
                              "return c1;")
    assert VC.body({"not": FILTER}) == (
        "create k1: number; assign k1 = 0;\n"
        "create i2: number; assign i2 = 0;\n"
        'while (i2 < n_word) { let x3 = word_first_lower[i2]; if (x3 == "z") { add k1, 1; } add i2, 1; }\n'
        "create c4: number; assign c4 = 0;\n"
        "if (k1 == 0) { assign c4 = 1; }\n"
        "create c5: number; assign c5 = 0;\n"
        "if (c4 == 0) { assign c5 = 1; }\n"
        "return c5;")
    assert VC.views_of({"and": [SHORT, FILTER]}) == ["sentence", "word"]


def test_the_joins_text_is_pinned():
    out = VJ.join([SHORT, FILTER], [("either", 0, 1)])
    assert out == (
        "program Instruction implements ISolve {\n"
        "    function part1(n_sentence: number): number {\n"
        "        create c1: number; assign c1 = 0;\n"
        "        if (n_sentence <= 3) { assign c1 = 1; }\n"
        "        return c1;\n"
        "    }\n"
        "    function part2(n_word: number, word_first_lower: number): number {\n"
        "        create k1: number; assign k1 = 0;\n"
        "        create i2: number; assign i2 = 0;\n"
        '        while (i2 < n_word) { let x3 = word_first_lower[i2]; if (x3 == "z") { add k1, 1; } add i2, 1; }\n'
        "        create c4: number; assign c4 = 0;\n"
        "        if (k1 == 0) { assign c4 = 1; }\n"
        "        return c4;\n"
        "    }\n"
        "    public function solve(mention: str): number {\n"
        "        let p1 = part1(n_sentence);\n"
        "        let p2 = part2(n_word, word_first_lower);\n"
        "        create g1: number; assign g1 = 0;\n"
        "        if (p1 == 1) { assign g1 = 1; } if (p2 == 1) { assign g1 = 1; }\n"
        "        create g2: number; assign g2 = 0;\n"
        "        if (g1 == 1) { assign g2 = 1; }\n"
        "        return g2;\n"
        "    }\n"
        "}\n")
    assert VJ.spec([SHORT, FILTER], [("either", 0, 1)]) == {"or": [SHORT, FILTER]}
    assert VJ.spec([SHORT, FILTER], [("if", 1, 0)]) == {"if": FILTER, "then": SHORT}
    assert VJ.spec([SHORT, FILTER], [("one", 0), ("one", 1)]) == {"and": [SHORT, FILTER]}
