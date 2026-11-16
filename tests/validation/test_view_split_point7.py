"""Order 0073, point 7 (validation/exp_view_split_point7.py): the variant an arm keeps, from its two dev reads. The
rule is the order's: fewer false passes over checked replies, then the higher agreement, then the rule on; the
rates are exact fractions because the variants do not check the same number of replies. The file holds dev's
counts, and a report of another arm, another variant, another set of rows or a dev not read whole is refused."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import exp_view_split_point7 as P  # noqa: E402


def counts(checked: int, false_passes: int, agree: int) -> dict:
    return {"rows": 300, "replies": 3300, "replies checked": checked, "false passes": false_passes, "agree": agree}


def test_fewer_false_passes_then_higher_agreement_then_the_rule_on():
    assert P.choose(counts(3000, 30, 2900), counts(3200, 31, 3000)) == {"own_text": True, "why": "fewer false passes over checked replies"}
    assert P.choose(counts(3000, 30, 2900), counts(2000, 21, 1990))["own_text"] is False      # 21/2000 is more than 30/3000
    assert P.choose(counts(3000, 30, 2900), counts(1500, 15, 1460)) == {"own_text": True, "why": "the same false passes; the higher agreement"}
    assert P.choose(counts(3000, 30, 2900), counts(1500, 15, 1440))["own_text"] is False
    assert P.choose(counts(3000, 30, 2900), counts(1500, 15, 1450))["why"] == "the same false passes and the same agreement: the rule on"
    assert P.choose(counts(3000, 30, 2900), counts(1500, 15, 1450))["own_text"] is True
    assert P.choose(counts(3000, 0, 3000), counts(3000, 1, 3000))["own_text"] is False


def test_a_variant_that_checked_no_reply_loses_to_one_that_checked_some():
    assert P.choose(counts(0, 0, 0), counts(10, 9, 1)) == {"own_text": True, "why": "the other variant checked no reply"}
    assert P.choose(counts(10, 9, 1), counts(0, 0, 0))["own_text"] is False
    assert P.choose(counts(0, 0, 0), counts(0, 0, 0)) == {"own_text": True, "why": "neither variant checked a reply: the rule on"}
    assert P.rates({"replies checked": 3, "false passes": 1, "agree": 2}) == (P.Fraction(1, 3), P.Fraction(2, 3)) and P.rates({}) is None


def report(tmp_path, arm: str, own: bool, c: dict, rows: str = "r" * 64, splits=("dev",)) -> str:
    p = tmp_path / f"{arm}_{'own' if own else 'free'}_{len(list(tmp_path.iterdir()))}.json"
    p.write_text(json.dumps({"order": "0073", "arm": arm, "own_text": own, "answers_sha256": f"{arm}{int(own)}" * 32,
                             "rows_sha256": rows, "splits": {s: {"counts": c} for s in splits}}), encoding="utf-8")
    return str(p)


def test_the_file_holds_each_arms_choice_with_devs_counts(tmp_path):
    files = {"m": {"free": report(tmp_path, "m", False, counts(3000, 30, 2900)), "own": report(tmp_path, "m", True, counts(3200, 31, 3000))},
             "w": {"free": report(tmp_path, "w", False, counts(3100, 12, 3050)), "own": report(tmp_path, "w", True, counts(3100, 40, 3000))}}
    out = tmp_path / "point7.json"
    sys.argv = ["exp_view_split_point7.py", "--m-free", files["m"]["free"], "--m-own", files["m"]["own"],
                "--w-free", files["w"]["free"], "--w-own", files["w"]["own"], "--out", str(out)]
    P.main()
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["order"] == "0073" and set(got["kept"]) == {"m", "w"} and got["rows_sha256"] == "r" * 64 and got["rule"] == P.RULE
    assert got["kept"]["m"]["own_text"] is True and got["kept"]["w"]["own_text"] is False        # what the test notebook reads
    dev = got["kept"]["w"]["dev"]
    assert dev["free"]["counts"]["false passes"] == 12 and dev["own"]["counts"]["replies checked"] == 3100
    assert dev["free"]["answers_sha256"] == "w0" * 32 and set(dev["own"]["counts"]) == set(P.COUNTS)


def test_a_report_that_is_not_that_arms_whole_dev_is_refused(tmp_path):
    ok = {"m": {"free": report(tmp_path, "m", False, counts(9, 1, 8)), "own": report(tmp_path, "m", True, counts(9, 1, 8))},
          "w": {"free": report(tmp_path, "w", False, counts(9, 1, 8)), "own": report(tmp_path, "w", True, counts(9, 1, 8))}}
    assert P.decide(ok)["kept"]["m"]["own_text"] is True
    for arm, variant, bad in (("m", "own", report(tmp_path, "m", False, counts(9, 1, 8))),                # the other variant
                              ("w", "free", report(tmp_path, "m", False, counts(9, 1, 8))),               # the other arm
                              ("w", "own", report(tmp_path, "w", True, dict(counts(9, 1, 8), **{"rows not read": 2}))),
                              ("m", "free", report(tmp_path, "m", False, counts(9, 1, 8), splits=("dev", "test"))),
                              ("m", "free", report(tmp_path, "m", False, counts(9, 1, 8), rows="q" * 64))):   # other rows
        with pytest.raises(ValueError):
            P.decide({a: dict(v, **({variant: bad} if a == arm else {})) for a, v in ok.items()})
