"""Order 0068, the wiring check: on a made-up head, index and states the wired "+ham" arm reads what the lab reads,
and the check refuses the held-out positions. No model, no dump."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest

from cubbyllm.model.vocab.shortlist import HAM, ScreenIndex, ShortlistIndex, successors

_spec = importlib.util.spec_from_file_location("head_lab_wired",
                                               pathlib.Path(__file__).parents[2] / "validation" / "head_lab_wired.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)

V, D, K, N, ARM = 400, 64, 40, 24, "screen+succ" + HAM


def quiet(*_a):
    return None


def world(seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(V, D)).astype(np.float32)
    hid = rng.normal(size=(300, D)).astype(np.float32)
    full = hid[:200] @ w.T
    top = np.argsort(-full, axis=1)[:, :4]
    wt = np.take_along_axis(full, top, 1)
    idx = ShortlistIndex.build(w, 16, seed=0)
    idx.screen = ScreenIndex.build(hid[:200], top, np.exp(wt - wt.max(1, keepdims=True)), vocab=V, n=8, seed=0)
    idx.succ = successors(rng.integers(0, V, 4000), V, top=4)
    d = {"hidden": hid[200:].astype(np.float16), "prev": rng.integers(0, V, 100).astype(np.int32),
         "target": np.zeros(100, dtype=np.int32)}
    return w, idx.attach_hamming(w, N), d


def test_the_wired_list_reads_what_the_lab_reads():
    w, idx, d = world()
    r = lab.lists(idx, w, d, K, N, ARM, chunk=32, log=quiet)
    assert r["positions"] == 100 and r["positions_where_they_differ"] == 0
    assert r["wired_agree"] == r["lab_agree"] > r["context_list_alone"]      # the pass catches some the list misses
    assert K <= r["rows_scored_exactly"]["p50"] <= r["rows_scored_exactly"]["max"] <= K + 4 + N
    assert r["rows_by_bytes"]["total_mean"] == round(8 + V / 32 + r["rows_scored_exactly"]["mean"], 1)
    assert set(r["seconds_per_position_cpu"]) == {"full_head", "context_list", "hamming_pass"}


def test_distance_alone_is_read_with_its_ties_for_and_against():
    w, idx, d = world(1)
    r = lab.lists(idx, w, d, K, N, ARM, chunk=32, log=quiet)["distance_alone"]
    assert list(r) == [str(N), str(2 * N), str(4 * N)]
    assert all(v["ties_against"] <= v["ties_for"] for v in r.values())
    assert r[str(N)]["ties_for"] <= r[str(4 * N)]["ties_for"]


def test_the_torch_head_keeps_the_list_and_scores_it_exactly(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("CB_HEAD_SHORTLIST", raising=False)
    w, idx, d = world()
    r = lab.head_check(idx, w, d, K, N, ARM, every=3, chunk=16, log=quiet)
    assert r["positions"] == 34 and r["kept_rows_are_the_list"] == 34 and r["bits_are_the_heads_rows"]
    assert r["best_token_is_the_full_heads"] == r["best_token_in_the_list"]
    assert r["max_abs_logit_difference_on_kept_rows"] < 1e-4
    assert "CB_HEAD_SHORTLIST" not in os.environ                             # the flag is left as it was found


def test_the_check_refuses_the_held_out_positions(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["head_lab_wired.py", "--export", "e", "--dumps", "d", "--dump", "heldout"])
    with pytest.raises(SystemExit, match="held-in dumps only"):
        lab.main()
