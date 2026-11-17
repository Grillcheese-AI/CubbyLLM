"""Order 0068, step 2e: the synthetic 256k head's lists and rows are built as the order says. No card."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("head_lab_decode", ROOT / "validation" / "head_lab_decode.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)


def test_each_token_is_followed_by_its_copy_in_its_own_list():
    ids = np.array([5, 1, 9, 2, 7])
    off = np.array([0, 2, 2, 5])                       # an empty list in the middle
    got, go = lab.doubled_lists(ids, off, 10)
    assert got.tolist() == [5, 15, 1, 11, 9, 19, 2, 12, 7, 17]
    assert go.tolist() == [0, 4, 4, 10]
    for c in range(3):
        assert set(got[go[c]:go[c + 1]]) == set(ids[off[c]:off[c + 1]]) | set(ids[off[c]:off[c + 1]] + 10)


def test_the_copy_is_the_rows_plus_noise_at_a_tenth_of_their_spread():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((300, 64)).astype(np.float32) * np.arange(1, 301, dtype=np.float32)[:, None]
    c = lab.noisy_copy(w, 0.1, 68, chunk=128)
    assert c.dtype == np.float32 and c.shape == w.shape
    ratio = (c - w).std(axis=1) / w.std(axis=1)
    assert np.all(np.abs(ratio - 0.1) < 0.04)
    assert np.array_equal(c, lab.noisy_copy(w, 0.1, 68, chunk=128))      # the seed fixes it
