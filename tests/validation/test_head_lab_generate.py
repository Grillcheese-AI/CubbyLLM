"""Order 0068, step 2f: a step is the time between two tokens reaching the host, after the first one (the prompt's pass
is not a step), the warm steps left out; and the first difference of two token lists. No card."""
from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("head_lab_generate", ROOT / "validation" / "head_lab_generate.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)


def test_a_step_runs_from_one_token_to_the_next_after_the_first():
    stamps = [0.0, 0.500, 0.520, 0.541, 0.563, 0.586]  # the prompt's put, the first token (after the prompt), 4 steps
    got = lab.step_times(stamps, warm=0)
    assert [round(t, 6) for t in got] == [20.0, 21.0, 22.0, 23.0]
    assert [round(t, 6) for t in lab.step_times(stamps, warm=2)] == [22.0, 23.0]


def test_the_clock_stamps_every_put():
    c = lab.Clock()
    for _ in range(3):
        c.put(None)
    c.end()
    assert len(c.stamps) == 3 and c.stamps == sorted(c.stamps)


def test_first_difference():
    assert lab.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert lab.first_difference([1, 2, 3], [1, 5, 3]) == 1
    assert lab.first_difference([1, 2], [1, 2, 3]) == 2
