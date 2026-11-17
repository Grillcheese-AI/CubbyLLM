"""Order 0068, step 2: the timing harness builds route 5's context list as ``ScreenIndex.shortlist`` does. No card."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

from cubbyllm.model.vocab.shortlist import ScreenIndex, successors

ROOT = pathlib.Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("head_lab_time", ROOT / "validation" / "head_lab_time.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)


def test_context_ids_match_the_screen_index():
    rng = np.random.default_rng(0)
    v, d, n_states = 300, 16, 400
    hidden = rng.normal(size=(n_states, d)).astype(np.float32)
    top = rng.integers(0, v, size=(n_states, 8))
    weight = rng.random(size=(n_states, 8)).astype(np.float32)
    screen = ScreenIndex.build(hidden, top, weight, v, n=12, keep=30)
    succ = successors(rng.integers(0, v, size=2000), v, top=4)
    for k in (10, 40, 90):
        for i in range(20):
            h = rng.normal(size=(1, d)).astype(np.float32)
            prev = np.array([int(rng.integers(0, v))])
            ref = screen.shortlist(h, k, succ, prev)[0]
            order = np.argsort(-(h @ screen.centroids.T).reshape(-1), kind="stable")
            assert np.array_equal(lab.context_ids(screen, order, k, succ[int(prev[0])]), ref)
