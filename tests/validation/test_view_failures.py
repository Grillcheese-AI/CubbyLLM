"""validation/exp_view_failures.py: a program's or / if groups read from its solve, in the gold's part numbers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation")]
import exp_view_failures as F  # noqa: E402

GOLD = ("    public function solve(mention: str): number {\n"
        "        create g1: number; assign g1 = 0;\n"
        "        if (p2 == 0) { assign g1 = 1; } if (p3 == 1) { assign g1 = 1; }\n"
        "        create g2: number; assign g2 = 0;\n"
        "        if (p1 == 1) { assign g2 = 1; } if (g1 == 1) { assign g2 = 1; }\n")


def test_groups_name_their_members_and_nest():
    assert F.groups(GOLD, {"1": "1", "2": "2", "3": "3"}) == {("2", "if", "3"), ("1", "or", ("2", "if", "3"))}


def test_my_numbers_are_mapped_to_the_golds():
    mine = GOLD.replace("p2 ==", "pX ==").replace("p3 ==", "p2 ==").replace("pX ==", "p3 ==")
    assert F.groups(mine, {"1": "1", "2": "3", "3": "2"}) == F.groups(GOLD, {"1": "1", "2": "2", "3": "3"})
    assert F.groups(mine, {"1": "1", "2": "2", "3": "3"}) != F.groups(GOLD, {"1": "1", "2": "2", "3": "3"})
