"""Order 0045, part A -- ``validation/pod/prepare_inputs.py --dry`` on a toy cache: the slice takes no block the first run
(``--exclude-frac``) or the continuation (``--exclude-dir``, its ``_subset.json``) took, and the epoch table flags a shard
drawn more than twice. The block size is shrunk so the toy fits in memory."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation", ROOT / "validation" / "pod"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_base_cache as mbc  # noqa: E402
import prepare_inputs as pi  # noqa: E402


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setattr(mbc, "BLOCK", 1000)
    src = tmp_path / "full"
    src.mkdir()
    for name, n in (("fineweb_edu", 60_000), ("books", 45_000), ("tiny", 3_000)):
        np.arange(n, dtype=np.uint32).tofile(src / f"{name}.u32")
    cont = tmp_path / "cont"
    cont.mkdir()
    # the continuation took these blocks (as make_base_cache records them)
    rec = {"fineweb_edu": {"starts": [0, 10_000, 20_000], "block": 1000}, "books": {"starts": [5_000], "block": 1000},
           "tiny": {"starts": [0], "block": 2_850}}
    (cont / "_subset.json").write_text(json.dumps({"frac": 0.2, "shards": rec}))
    return src, cont


def test_the_dry_plan_takes_no_block_an_earlier_subset_took(toy):
    src, cont = toy
    plan = pi.plan_slice(str(src), frac=0.3, exclude_frac=[0.2], exclude_dirs=[str(cont)], val_tokens=2000, keep_small=4000)
    assert pi.overlaps(plan) == []
    fw = plan["fineweb_edu"]
    first = set(mbc.spread(fw["train"], 0.2))
    for s in fw["starts"]:
        assert all(abs(s - u) >= 1000 for u in first | {0, 10_000, 20_000}), (s, first)
    assert len(fw["starts"]) == fw["wanted"] and fw["unseen"] is not None
    assert plan["tiny"]["whole"]                                           # under --keep-small: copied whole
    # a shard that has run out of unseen blocks takes what is left, and says so
    full = {"books": {"starts": list(range(0, plan["books"]["train"] - 999, 1000)), "block": 1000}}
    (cont / "_subset.json").write_text(json.dumps({"frac": 0.2, "shards": full}))
    plan2 = pi.plan_slice(str(src), frac=0.3, exclude_frac=[], exclude_dirs=[str(cont)], val_tokens=2000, keep_small=4000)
    assert plan2["books"]["unseen"] == 0 and plan2["books"]["starts"] == [] and pi.overlaps(plan2) == []


def test_the_epoch_table_flags_a_shard_drawn_more_than_twice():
    sizes = {"a": 1_000_000, "b": 1_000_000}
    rows = pi.epoch_table(sizes, {"a": 300_000, "b": 300_000}, run_tokens=1_400_000, decay_json={})
    by = {r["shard"]: r for r in rows}
    assert abs(by["a"]["drawn"] + by["b"]["drawn"] - 1_400_000) <= 2
    text = pi.table_text(rows, 1_400_000)
    assert "raise --frac" in text and by["a"]["epochs"] > 2
    rows = pi.epoch_table(sizes, {"a": 800_000, "b": 800_000}, run_tokens=1_400_000, decay_json={})
    assert "raise --frac" not in pi.table_text(rows, 1_400_000)
    assert 13e9 < pi.tokens_for("b200", 14, 0.35) < 14e9 and pi.tokens_for("h200", 7, 0.35) < pi.tokens_for("b200", 7, 0.35)


def test_dry_run_from_the_command_line_writes_nothing(toy, tmp_path):
    src, cont = toy
    dst = tmp_path / "r3"
    out = subprocess.run([sys.executable, str(ROOT / "validation" / "pod" / "prepare_inputs.py"), "--src", str(src), "--dst", str(dst),
                          "--exclude-dir", str(cont), "--keep-small", "4000", "--val-tokens", "2000", "--dry", "--tokens", "5e6"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr[-800:]
    assert "epochs in a" in out.stdout and "slice:" in out.stdout and not dst.exists()
