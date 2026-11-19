"""Order 0077's comparison (validation/exp_view_real_compare.py): line 5 keeps arm R only with no kill, line 4 held,
and one of its two conditions; a kill on arm R keeps arm L; line 4 fails when R's false passes are above L's over the
whole interval."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import exp_view_real_compare as X  # noqa: E402


def _counts(agree, checked, fp, fail, right=95, gold=100, pairs=(97, 100), replies=None, drop=0):
    return {"gold checks with their line": gold, "gold checks": gold, "agree": agree, "replies checked": checked,
            "false passes": fp, "checked replies built to fail": fail, "parts right": right, "parts fixed by one reversal": 0,
            "gold pairs right": pairs[0], "gold pairs": pairs[1], "replies": replies or checked,
            "characters in no line and not flagged": drop}


def _rep(sets, **kw):
    return {"answers": "x", "splits": {s: {"counts": _counts(**kw)} for s in sets}}


def _keep(split, n, fp_at=()):
    return [{"id": f"{split}{i}", "split": split, "false passes": int(i in fp_at), "built to fail": 5} for i in range(n)]


def _run(r_kw, l_kw, r74_fp=(), l74_fp=()):
    rep = {"r": _rep(X.REAL, **r_kw), "l": _rep(X.REAL, **l_kw), "r74": _rep(X.OLD, agree=1, checked=1, fp=0, fail=1),
           "l74": _rep(X.OLD, agree=1, checked=1, fp=0, fail=1)}
    keep = {"r": _keep("test_real", 40), "l": _keep("test_real", 40),
            "r74": _keep("test", 40, r74_fp) + _keep("test_long", 40, r74_fp),
            "l74": _keep("test", 40, l74_fp) + _keep("test_long", 40, l74_fp)}
    return _decide_without_breakdowns(rep, keep)


def _decide_without_breakdowns(rep, keep):
    saved = X.breakdowns
    X.breakdowns = lambda k, rows: {}
    try:
        return X.decide(rep, keep, {}, None)
    finally:
        X.breakdowns = saved


GOOD = dict(agree=99, checked=100, fp=0, fail=50)
WEAK = dict(agree=90, checked=100, fp=0, fail=50, replies=120)


def test_arm_r_is_kept_when_arm_l_misses_line_2_and_line_4_holds():
    got = _run(GOOD, WEAK)
    assert got["line 5"]["kept"] == "r" and got["line 4"]["passes"]


def test_a_kill_on_arm_r_keeps_arm_l():
    got = _run(dict(GOOD, fp=6), WEAK)                     # 6 / 100 false passes: past the 0.05 kill
    assert got["line 5"]["kept"] == "l" and got["line 5"]["kills"]["r"]


def test_more_false_passes_on_0074s_sets_fail_line_4():
    got = _run(GOOD, WEAK, r74_fp=tuple(range(40)))
    assert not got["line 4"]["passes"] and got["line 5"]["kept"] == "l"


def test_with_both_arms_passing_line_2_and_no_lower_false_passes_arm_l_stays():
    got = _run(GOOD, GOOD)
    assert got["line 5"]["kept"] == "l"
