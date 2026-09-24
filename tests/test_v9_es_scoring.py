"""The H-A9 fitness ladder, and the guard that stops it rewarding a cheat.

The ES run is only as meaningful as this scoring. A member is ranked
against its twin on these numbers, so a ladder that pays out for a program
which ignores its input would have ES optimize toward exactly that — and
the run would look like it was working.

So the case that matters most here is the **hardcoder**: a program that
returns the gold constant and computes nothing. It executes, it matches the
gold, and on the bare ladder it scores a perfect 1.6. It must cap at 0.6.

These run the real cubelang VM (a subprocess), not a mock: a mock of the
VM would be a mock of the only thing being measured.
"""
from __future__ import annotations

import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# appended, not inserted: standin/hypothesis.py would otherwise shadow the
# installed `hypothesis` package and break pytest's plugin
sys.path.insert(0, str(ROOT / "validation"))
sys.path.append(str(ROOT / "standin"))

exp = pytest.importorskip("exp_v9_es")


def vm_available() -> bool:
    try:
        from cubbyllm.bridges.cubelang_client import find_cubelang_exe

        find_cubelang_exe(None)
        return True
    except Exception:
        return False


needs_vm = pytest.mark.skipif(not vm_available(), reason="cubelang.exe not built")


def program(body: str, name: str = "P0") -> str:
    return f"""program {name} implements ISolver {{
    type Input = str;
    type Output = quantity;
    public function parse(raw: str): Input {{ return raw; }}
    public pure function verify(input: Input, output: Output): bool {{ return true; }}
    public function solve(input: Input): Output {{
{body}
    }}
}}"""


#: "48 clips, half as many again" -> 72, the shape of a real record
HONEST = program("""        create s0 : quantity;
        assign s0 = 48;
        div s0, 2;
        create s1 : quantity;
        assign s1 = 48;
        add s1, s0;
        sum s1;
        query s1;
        return s1;""")

#: computes nothing; returns the answer it was told
HARDCODED = program("""        create s0 : quantity;
        assign s0 = 72;
        sum s0;
        query s0;
        return s0;""")

RECORD = {
    "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold "
              "half as many clips in May. How many clips did Natalia sell altogether?",
    "program": HONEST,
    "gold": 72,
}


# ── the substitution the guard rests on ──────────────────────────────────


def test_substitution_does_not_corrupt_a_longer_number():
    """Replacing 4 must not turn 48 into something else — the reason the
    rewrite goes longest-first through placeholders."""
    got = exp.substitute("assign a = 4; assign b = 48; assign c = 448;", {4: 9, 48: 50})
    assert got == "assign a = 9; assign b = 50; assign c = 448;"


def test_substitution_does_not_re_substitute():
    """With 2 -> 3 and 3 -> 7, the 2 must become 3 and stop, not 7."""
    assert exp.substitute("assign a = 2; assign b = 3;", {2: 3, 3: 7}) == "assign a = 3; assign b = 7;"


def test_the_numbers_read_are_the_prompts():
    assert exp.numbers_in(RECORD["prompt"]) == [48]
    assert exp.numbers_in("no digits here") == []


def test_a_perturbation_never_leaves_a_value_alone():
    rng = random.Random(0)
    mapping = exp.perturb([48, 12, 7], rng)
    assert set(mapping) == {48, 12, 7}
    assert all(new != old for old, new in mapping.items())


def test_a_perturbation_skips_values_it_cannot_move_meaningfully():
    assert exp.perturb([0, 1], random.Random(0)) == {}


# ── the ladder ───────────────────────────────────────────────────────────


def test_an_empty_generation_scores_zero():
    assert exp.score("", RECORD, random.Random(0)) == 0.0


def test_prose_gets_only_the_parse_rung():
    assert exp.score("Here is the answer: 72", RECORD, random.Random(0)) == pytest.approx(0.1)


def test_a_structurally_right_but_broken_program_stops_below_execution():
    """Shaped like a program, so it takes the compile rung, but the VM
    refuses it — the rungs above must not pay out."""
    broken = program("        this is not cubelang;")
    assert exp.score(broken, RECORD, random.Random(0)) == pytest.approx(0.3)


@needs_vm
def test_an_honest_program_scores_the_top_of_the_ladder():
    assert exp.score(HONEST, RECORD, random.Random(0)) == pytest.approx(1.6)


@needs_vm
def test_a_hardcoder_caps_at_six_tenths():
    """The point of the whole guard. It executes and it matches the gold,
    so the bare ladder would pay it 1.6; it must get 0.6, because when the
    numbers move it does not."""
    assert exp.score(HARDCODED, RECORD, random.Random(0)) == pytest.approx(0.6)


@needs_vm
def test_the_counterfactual_separates_the_two_directly():
    rng = random.Random(1)
    assert exp.counterfactual_holds(HONEST, RECORD, rng) is True
    assert exp.counterfactual_holds(HARDCODED, RECORD, random.Random(1)) is False


@needs_vm
def test_the_guard_abstains_rather_than_failing_a_program_it_cannot_judge():
    """With no oracle — a prompt with no numbers to move — the guard must
    not silently mark an honest program a cheat."""
    record = dict(RECORD, prompt="How many clips altogether?")
    assert exp.counterfactual_holds(HONEST, record, random.Random(0)) is True


# ── the bar the run is judged against ────────────────────────────────────


def test_the_bar_is_the_two_standard_error_one_from_step_zero():
    """0.775 is v14e's measured arithmetic verify-to-gold and 329 is the
    whole held-out split, so the bar is 0.821. A 40-record read would put
    it at 0.907 and make the criterion unreachable by construction."""
    import math

    n = 329
    se = math.sqrt(exp.BASELINE * (1 - exp.BASELINE) / n)
    assert exp.BAR == pytest.approx(exp.BASELINE + 2 * se, abs=0.002)
