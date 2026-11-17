"""Order 0068, route 5 by distance alone: the nearest rows and their ties, the hit counted both ways, the interval over
windows, and the fresh held-out windows (none shares a token with the ones read before). No model, no dump."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "validation"))
_spec = importlib.util.spec_from_file_location("head_lab_distance", ROOT / "validation" / "head_lab_distance.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)

import head_dump  # noqa: E402


def test_nearest_breaks_ties_to_the_lower_id():
    dist = np.array([3, 1, 1, 2, 1], dtype=np.int32)
    assert lab.nearest(dist, 2).tolist() == [1, 2]
    assert lab.nearest(dist, 4).tolist() == [1, 2, 3, 4]
    assert lab.nearest(dist, 9).tolist() == [0, 1, 2, 3, 4]


def test_a_tie_at_the_nth_place_counts_against_and_for():
    dist = np.array([3, 1, 1, 2, 1], dtype=np.int32)
    assert lab.hit_counts(dist, 2, 2) == (False, True)          # three rows at distance 1: against misses, for hits
    assert lab.hit_counts(dist, 2, 3) == (True, True)
    assert lab.hit_counts(dist, 0, 4) == (False, False)          # four rows nearer than row 0


def test_the_interval_is_over_windows():
    hits = np.ones(4 * lab.WINDOW, dtype=bool)
    assert lab.interval(hits) == [1.0, 1.0]
    hits[: lab.WINDOW] = False                                    # one window of four wrong
    lo, hi = lab.interval(hits)
    assert 0.0 <= lo <= 0.75 <= hi <= 1.0


def test_fresh_starts_share_no_token_and_old_starts_are_unchanged():
    total, hold, length = 2_000_000, 600_000, 1024
    old = head_dump.starts(total, hold, "heldout", 4, length, 0.0)
    span = hold - length - 1
    assert old == [total - hold + span // 3 * i for i in range(4)]
    new = head_dump.starts(total, hold, "heldout", 4, length, 0.5, avoid_phase=0.0)
    assert len(new) == 3 and all(abs(s - o) > length for s in new for o in old)


def test_fresh_reads_the_cache_and_refuses_a_shared_window(tmp_path):
    total, hold = 200_000, 60_000
    (tmp_path / "_holdout.json").write_text(json.dumps({"arxiv": hold}), encoding="utf-8")
    np.zeros(total, dtype=np.uint32).tofile(tmp_path / "arxiv.u32")
    new = head_dump.starts(total, hold, "heldout", 4, 1024, 0.5, avoid_phase=0.0)
    d = {"win_shard": np.zeros(len(new), dtype=np.int8), "win_start": np.array(new, dtype=np.int64)}
    assert lab.fresh(d, str(tmp_path))["shares_no_token"] is True
    old = head_dump.starts(total, hold, "heldout", 4, 1024, 0.0)
    d = {"win_shard": np.zeros(1, dtype=np.int8), "win_start": np.array([old[1] + 500], dtype=np.int64)}
    assert lab.fresh(d, str(tmp_path))["shares_no_token"] is False
