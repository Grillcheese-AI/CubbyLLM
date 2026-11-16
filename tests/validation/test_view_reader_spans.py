"""Round 2's scorer on order 0073's rows (validation/exp_view_reader.py, `spans_right`): the two row sets put the
space between two clauses on different sides, so a reader trained on 0067's rows copies a clause of 0073's a space
off. Compared as round 2 did, such a part is not counted; with ``trim`` (arm P on 0073's rows) it is, and a wrong
kind or a wrong number is still wrong."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import exp_view_reader as E  # noqa: E402

TEXT = "Write a note. Unless it is short, use ten words."
ROW = {"instruction": TEXT, "segments": [{"kind": "task", "start": 0, "end": 13},
                                         {"kind": "check", "part": 0, "role": "condition", "start": 14, "end": 33},
                                         {"kind": "check", "part": 1, "start": 33, "end": 48}]}       # the space leads the clause


def read(*segs) -> list:
    return [{"kind": k, "part": p, "start": a, "end": b} for k, p, a, b in segs]


def test_a_clause_copied_a_space_off_counts_only_with_trim():
    assert TEXT[14:33] == "Unless it is short," and TEXT[33:48] == " use ten words." and len(TEXT) == 48
    old = read(("task", None, 0, 13), ("check", 0, 14, 34), ("check", 1, 34, 48))                      # the space trails the clause
    assert E.spans_right(ROW, old) == (0, 1) and E.spans_right(ROW, old, trim=True) == (2, 3)
    same = read(("task", None, 0, 13), ("check", 0, 14, 33), ("check", 1, 33, 48))
    assert E.spans_right(ROW, same) == (2, 3) == E.spans_right(ROW, same, trim=True)                   # off: as round 2 counted


def test_a_wrong_kind_or_number_is_wrong_with_trim_too():
    assert E.spans_right(ROW, read(("task", None, 0, 13), ("check", 1, 14, 34), ("check", 0, 34, 48)), trim=True) == (0, 3)
    assert E.spans_right(ROW, read(("task", None, 0, 13), ("none", None, 14, 34), ("check", 1, 34, 48)), trim=True) == (1, 2)
    assert E.spans_right(ROW, read(("task", None, 0, 13), ("check", 0, 14, 48)), trim=True) == (0, 1)   # two clauses as one
