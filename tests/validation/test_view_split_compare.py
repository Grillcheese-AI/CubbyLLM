"""Order 0073 (validation/exp_view_split_compare.py): the two arms against "Acceptance", and line 9. Each line is
judged from its two counts at the bound the order writes (a pass at the bound, a kill only past it), on the set
the order's report names; line 9 puts an arm that hit a kill out, then takes the arm with fewer false passes over
replies built to fail when the difference is outside a 95% interval over rows, then the arm with more parts right,
then arm M; the same seeded resamples each time."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import exp_view_split_compare as C  # noqa: E402

POSITIONS = {"1": [95, 100], "4": [90, 100], "5": [88, 100], "8": [85, 100]}


def counts(**over) -> dict:
    base = {"characters in no line and not flagged": 0, "gold checks": 1000, "gold checks with their line": 950, "replies": 3000,
            "replies checked": 2970, "agree": 2900, "false passes": 30, "checked replies built to fail": 1400, "parts right": 960,
            "parts fixed by one reversal": 20, "gold pairs": 200, "gold pairs right": 196, "gold checks asked by the reader": 10,
            "gold checks asked by the host": 5}
    return dict(base, **over)


def report(arm="m", own=False, test=None, long=None, position=None) -> dict:
    return {"order": "0073", "arm": arm, "own_text": own, "rows_sha256": "r" * 64,
            "splits": {"test": {"counts": test or counts(), "by": {"position": {}}},
                       "test_long": {"counts": long or counts(), "by": {"position": position or POSITIONS}}}}


def judged(name: str, **kw) -> tuple:
    got = C.lines(report(**kw))[name]
    return got["passes"], got["kill"]


def test_a_read_inside_every_bound_passes_every_line_and_hits_no_kill():
    got = C.lines(report())
    assert all(v["passes"] and not v["kill"] for v in got.values()), {k: v for k, v in got.items() if not v["passes"]}
    assert got["line 3: agreement, test"]["counts"] == [2900, 2970] and got["line 3: agreement, test"]["rate"] == 0.9764
    assert got["line 4: parts right at positions 5 to 8 over 1 to 4, test_long"]["counts"] == [[173, 200], [185, 200]]
    assert got["line 4: agreement on test_long over test's"]["rate"] == 1.0 and got["line 4: agreement on test_long over test's"]["counted"]
    assert sum(1 for k in got if "(reported)" in k) == 5 and len(got) == 19


def test_each_line_passes_at_its_bound_and_is_a_kill_only_past_it():
    fp, fx, pr = "line 3: false passes, test", "line 5: parts fixed by one reversal, test", "line 6: pairs right, test"
    c = lambda **o: counts(**{"replies checked": 1000, "agree": 990, **o})     # noqa: E731
    assert judged(fp, test=c(**{"false passes": 20})) == (True, False) and judged(fp, test=c(**{"false passes": 21})) == (False, False)
    assert judged(fp, test=c(**{"false passes": 50})) == (False, False) and judged(fp, test=c(**{"false passes": 51})) == (False, True)
    assert judged("line 4: false passes, test_long", long=c(**{"false passes": 51})) == (False, True)
    assert judged(fp, long=c(**{"false passes": 51})) == (True, False)                      # test_long's does not judge test's line
    assert judged("line 3: agreement, test", test=c(agree=950)) == (True, False) and judged("line 3: agreement, test", test=c(agree=949)) == (False, False)
    assert judged("line 2: gold checks with their line, test", test=counts(**{"gold checks with their line": 899})) == (False, False)
    assert judged("line 5: parts right, test", test=counts(**{"parts right": 950}))[0] and not judged("line 5: parts right, test", test=counts(**{"parts right": 949}))[0]
    assert judged(fx, test=counts(**{"parts fixed by one reversal": 30})) == (True, False)
    assert judged(fx, test=counts(**{"parts fixed by one reversal": 80})) == (False, False) and judged(fx, test=counts(**{"parts fixed by one reversal": 81})) == (False, True)
    assert judged(pr, test=counts(**{"gold pairs right": 194})) == (True, False) and judged(pr, test=counts(**{"gold pairs right": 193})) == (False, False)
    assert judged(pr, test=counts(**{"gold pairs right": 180})) == (False, False) and judged(pr, test=counts(**{"gold pairs right": 179})) == (False, True)
    assert judged("line 7: replies checked, test", test=counts(**{"replies checked": 2939}))[0] is False       # under 0.98
    assert judged("line 7: replies checked, test_long", long=counts(**{"replies checked": 2850, "agree": 2800}))[0] is True   # 0.95 is enough there
    assert judged("line 7: gold checks asked, test", test=counts(**{"gold checks asked by the host": 91}))[0] is False
    assert judged("line 1: characters dropped silently, test and test_long", long=counts(**{"characters in no line and not flagged": 1})) == (False, True)
    assert judged(fp, test=counts(**{"replies checked": 0, "agree": 0, "false passes": 0})) == (False, False)    # nothing to count over


def test_line_4_is_counted_only_if_line_3_passes_and_compares_test_long_with_test():
    name = "line 4: agreement on test_long over test's"
    long = counts(**{"replies checked": 1000, "agree": 878})                               # 0.878 against 0.9764 * 0.9 = 0.8788
    assert judged(name, long=long) == (False, False) and judged(name, long=dict(long, agree=879)) == (True, False)
    got = C.lines(report(test=counts(**{"false passes": 90})))[name]                       # line 3 fails on test
    assert got["counted"] is False and got["passes"] is False
    late = "line 4: parts right at positions 5 to 8 over 1 to 4, test_long"
    assert judged(late, position={"1": [90, 100], "6": [81, 100]})[0] and not judged(late, position={"1": [90, 100], "6": [80, 100]})[0]


def rows(n=60, fp=0, fail=4, right=3, checks=3) -> list:
    return [{"id": f"t{k}", "split": "test", "false passes": fp, "built to fail": fail, "parts right": right, "gold checks": checks} for k in range(n)]


def test_line_9_takes_out_a_killed_arm_then_false_passes_then_parts_right_then_arm_m():
    none = {"m": [], "w": []}
    assert C.line9(rows(), rows(), {"m": ["x"], "w": ["y"]})["kept"] is None
    assert C.line9(rows(), rows(fp=4), {"m": ["line 3: false passes, test"], "w": []})["kept"] == "w"     # out, whatever its counts
    assert C.line9(rows(), rows(), {"m": [], "w": ["x"]})["kept"] == "m"
    got = C.line9(rows(fp=2), rows(fp=0), none)
    assert got["kept"] == "w" and "fewer false passes" in got["why"] and "parts right" not in got
    assert got["false passes over replies built to fail"]["rates"] == [0.5, 0.0] and not got["false passes over replies built to fail"]["within"]
    got = C.line9(rows(right=1), rows(right=3), none)                                       # the same false passes
    assert got["kept"] == "w" and "more parts right" in got["why"] and got["false passes over replies built to fail"]["within"]
    assert got["parts right"]["m"] == [60, 180] and got["parts right"]["interval"] == [-0.6667, -0.6667]
    got = C.line9(rows(), rows(), none)
    assert got["kept"] == "m" and got["why"].endswith("arm M") and got["parts right"]["within"]


def test_the_interval_is_over_rows_seeded_and_holds_zero_when_the_arms_differ_on_a_few_rows():
    m, w = rows(), rows()
    for k in range(0, 60, 12):
        m[k] = dict(m[k], **{"false passes": 1})                                            # five rows of sixty
    w[5] = dict(w[5], **{"false passes": 1})
    a, b = C.interval(m, w, "false passes", "built to fail"), C.interval(m, w, "false passes", "built to fail")
    assert a == b and a["resamples"] == 2000 and a["resamples with no rate"] == 0
    assert a["m"] == [5, 240] and a["w"] == [1, 240] and a["interval"][0] <= 0 < a["interval"][1] and a["within"]
    assert C.line9(m, w, {"m": [], "w": []})["kept"] == "m"                                 # within on both: arm M
    none = C.interval(rows(fail=0), rows(), "false passes", "built to fail")
    assert none["resamples with no rate"] == 2000 and none["interval"] == [None, None] and not none["within"]


def test_the_script_reads_the_reports_in_the_variant_point_7_kept(tmp_path):
    def put(name, obj):
        (tmp_path / name).write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return str(tmp_path / name)
    point7 = put("p7.json", {"order": "0073", "rows_sha256": "r" * 64, "kept": {"m": {"own_text": True}, "w": {"own_text": False}}})
    keep = lambda rs: "".join(json.dumps(r) + "\n" for r in rs + [dict(rs[0], split="test_long", id="x")])   # noqa: E731
    args = ["--m", put("m.json", report("m", True)), "--w", put("w.json", report("w", False, test=counts(**{"false passes": 200}))),
            "--keep-m", put("km.jsonl", keep(rows())), "--keep-w", put("kw.jsonl", keep(rows(fp=1))), "--point7", point7]
    p = {"splits": {s: {"counts": {"parts": 900, "parts with span and kind": 800, "agree": 2000, "checked": 2500, "false passes": 250, "replies": 3000},
                        "span_and_kind": 0.8889, "agreement": 0.8, "false_passes": 0.1, "checked_share": 0.8333} for s in ("test", "test_long")},
         "long_over_in_length": 1.0}
    sys.argv = ["exp_view_split_compare.py", *args, "--p", put("p.json", p), "--out", str(tmp_path / "out.json")]
    C.main()
    got = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert got["variant"] == {"m": True, "w": False} and got["line 9"]["kept"] == "m" and "arm w hit a kill" in got["line 9"]["why"]
    assert got["lines not passed"]["m"] == [] and "line 3: false passes, test" in got["lines not passed"]["w"]
    assert got["arm p"]["test"]["line 3: false passes"] == [250, 2500, 0.1] and got["arm p"]["line 4: agreement on test_long over test's"] == 1.0
    sys.argv[sys.argv.index("--m") + 1] = put("m_free.json", report("m", False))            # not the variant point 7 kept
    with pytest.raises(ValueError):
        C.main()
