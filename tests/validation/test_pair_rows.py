"""Order 0067 round 2: the polarity pairs (validation/builders/build_pair_rows.py) and the goal-first / passive
wordings (view_wording_inv.py). Pinned: a goal-first wording stands for the very atom the part is (same truth on every
reply); a flipped part has the opposite truth on every reply; a pair's twin keeps every other part and the groups; the
pair rows are train only. The VM run before keeping is the builder's own (needs cubelang; skipped without)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation" / "builders")]
import build_pair_rows as BP  # noqa: E402
import view_programs as VP  # noqa: E402
import view_replies as VRep  # noqa: E402
import view_wording_inv as WI  # noqa: E402

from cubbyllm.speech import reply_views as RV  # noqa: E402


@pytest.fixture(scope="module")
def views():
    rng = random.Random(3)
    return [RV.views(t) for t in VRep.pool(rng, 60)]


def test_a_goal_first_wording_is_the_same_atom(views):
    rng, said = random.Random(5), 0
    for _ in range(300):
        a = VP.sample_atom(rng, views)
        if a is None:
            continue
        for part in (a, {"not": a}):
            pos = WI.positive(part)
            if pos is None:
                continue
            assert all(VP.evaluate(part, v) == VP.evaluate(pos, v) for v in views)
            for lang in ("en", "fr"):
                s = WI.say_inverted(part, lang, rng)
                said += s is not None
                assert s is None or (s[0].isupper() and s.rstrip().endswith("."))
    assert said > 100


def test_a_flipped_part_has_the_opposite_truth_everywhere(views):
    rng = random.Random(9)
    for _ in range(100):
        a = VP.sample_atom(rng, views)
        if a is None:
            continue
        for part in (a, {"not": a}):
            assert all(VP.evaluate(BP.flip(part), v) != VP.evaluate(part, v) for v in views)
            assert BP.flip(BP.flip(part)) == part


def test_a_pair_differs_in_one_part_only(views):
    from cubbyllm.bridges import cubelang_client as cc
    try:
        cc.find_cubelang_exe()
    except Exception as e:                                # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    rng, vm = random.Random(11), cc.CubelangSession()
    replies = VRep.pool(rng, 80)
    vs = [RV.views(t) for t in replies]
    for _ in range(40):
        pair, _ = BP.build_pair(rng, vm, vs, replies, {"en": [("k", "Be vivid.")], "fr": [("k", "Sois vivant.")]},
                                {}, Path("."), 0.3)
        if pair is None:
            continue
        a, b = pair
        assert a["split"] == b["split"] == "train" and a["groups"] == b["groups"]
        diff = [i for i, (p, q) in enumerate(zip(a["parts"], b["parts"])) if p != q]
        assert len(diff) == 1 and BP.flip(a["parts"][diff[0]]) == b["parts"][diff[0]]
        return
    pytest.fail("no pair built in 40 tries")
