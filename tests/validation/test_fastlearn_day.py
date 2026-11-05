"""Order 0053's day replay (``validation/selfimprove/fastlearn_day.py``) on a synthetic cache: the arms and days run,
held problems are read greedily (N reads the trunk's top candidate), and a three-factor learner learns a pattern the
trunk gets wrong: in one cluster of states the right candidate is the trunk's second, and S's held read finds it."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from selfimprove import fastlearn_day as FD  # noqa: E402


def _cache(tmp: pathlib.Path, d=16, n=120, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(2, d)) * 3
    cache, marks, states = [], [], []
    for fam in ("wd/en", "wd/fr"):
        for i in range(n):
            held = i >= n - 30
            c = i % 2                                          # cluster 1: the trunk's top is wrong, its second right
            v = ["right", "wrong"] if c == 0 else ["wrong", "right"]
            cache.append({"id": f"{fam}-{i}", "family": fam, "held": held, "order": i, "slot": True,
                          "scores": [0.0, -0.5]})
            marks.append({"id": f"{fam}-{i}", "verdicts": v, "props": [["PA"], ["PB"]] if c == 0 else [["PB"], ["PA"]]})
            states.append(centers[c] + rng.normal(size=d) * 0.3)
    (tmp / "fastlearn_cache.jsonl").write_text("\n".join(json.dumps(x) for x in cache) + "\n")
    (tmp / "fastlearn_marks.jsonl").write_text("\n".join(json.dumps(x) for x in marks) + "\n")
    np.save(tmp / "fastlearn_states.npy", np.stack(states).astype(np.float32))


def test_the_days_run_and_the_three_factor_learner_finds_what_the_trunk_misses(tmp_path, monkeypatch):
    _cache(tmp_path)
    monkeypatch.setenv("CB_FASTLEARN", "1")
    monkeypatch.setattr(FD, "GRID", {"gain": (4.0,), "eta": (0.05,), "tau": (0.25,), "norm": (4.0,)})
    FD._OFFSETS.clear()
    FD.main(["--dir", str(tmp_path), "--seeds", "2", "--neurons", "256", "--T", "4"])
    res = json.loads((tmp_path / "fastlearn_day.json").read_text())
    n, s = res["arms"]["N"][0], res["arms"]["S"][0]
    assert n["en_held_1"]["right_rate"] == 0.5                          # the trunk's top: right in one cluster only
    assert s["en_held_1"]["right_rate"] > 0.9                           # learned the other cluster from the verdicts
    assert set(s) >= {"day1", "en_held_1", "day2", "fr_held", "en_held_2", "bootstrap_day", "bootstrap_fr_held"}
    assert all(len(res["arms"][a]) == 2 for a in FD.ARMS)
