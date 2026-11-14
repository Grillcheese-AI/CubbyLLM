"""The view reader's second chance (order 0067; Nick: "train rows + pattern table"): a part whose comparison is
reversed is found by one reversal that gives the generator's part on every reply; a right part is left as it is; the
test splits are refused. Needs the VM (skipped without one)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cubbyllm.bridges import cubelang_client as cc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation")]
import exp_view_second_chance as SC  # noqa: E402

GOLD = ("program Instruction implements ISolve {\n"
        "    function part1(n_word: number): number {\n"
        "        create c1: number; assign c1 = 0;\n"
        "        if (n_word < 5) { assign c1 = 1; }\n"
        "        return c1;\n    }\n"
        "    public function solve(mention: str): number {\n"
        "        let p1 = part1(n_word);\n"
        "        create g1: number; assign g1 = 0;\n"
        "        if (p1 == 1) { assign g1 = 1; }\n"
        "        return g1;\n    }\n}\n")
INSTR = "Write fewer than 5 words."
REPLIES = [{"text": "Yes.", "held": True}, {"text": "One two three.", "held": True},
           {"text": "one two three four five six", "held": False}, {"text": "a b c d e f g h", "held": False}]


@pytest.fixture(scope="module")
def vm():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    return cc.CubelangSession()


def test_a_reversed_comparison_is_found_and_a_right_part_left(vm):
    truth = SC.results(vm, GOLD, "1", "n_word", REPLIES)
    assert truth == [True, True, False, False]
    wrong = GOLD.replace("n_word < 5", "n_word >= 5")
    assert SC.results(vm, wrong, "1", "n_word", REPLIES) != truth
    fix = next(v for v in SC.variants(wrong, "1") if SC.results(vm, v[4], "1", "n_word", REPLIES) == truth)
    assert fix[:4] == ("negated", "n", ">=", "<") and fix[4] == GOLD


def test_the_test_splits_are_refused(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "t1", "split": "test", "instruction": INSTR, "join": GOLD, "segments": [],
                                "replies": REPLIES, "n_parts": 1, "lang": "en"}) + "\n", encoding="utf-8")
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"id": "t1", "answer": ""}) + "\n", encoding="utf-8")
    p = subprocess.run([sys.executable, str(ROOT / "validation" / "exp_view_second_chance.py"), "--answers", str(answers),
                        "--rows", str(rows)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0 and "test row" in (p.stderr + p.stdout)
