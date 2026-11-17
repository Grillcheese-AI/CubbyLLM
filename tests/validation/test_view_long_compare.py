"""Order 0074 (validation/exp_view_long_compare.py): arm L against "Acceptance" with arm K beside it, and line 9.
Each line is judged from its two counts at the bound the order writes (a pass at the bound, a kill only past it);
test_long is held to its own bounds; line 3 is not passed only when the whole interval of L - K on test is above 0;
line 5 needs the whole interval on test_long below 0 for false passes and above 0 for parts right; arm L is kept
only with no kill and lines 3 and 5; test_longer is reported and decides nothing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import exp_view_long_compare as L  # noqa: E402

POSITIONS = {"1": [95, 100], "4": [90, 100], "5": [88, 100], "8": [85, 100]}


def counts(**over) -> dict:
    base = {"characters in no line and not flagged": 0, "gold checks": 1000, "gold checks with their line": 950, "replies": 3000,
            "replies checked": 2970, "agree": 2900, "false passes": 30, "checked replies built to fail": 1400, "parts right": 960,
            "parts fixed by one reversal": 20, "gold pairs": 200, "gold pairs right": 196, "gold checks asked by the reader": 10,
            "gold checks asked by the host": 5}
    return dict(base, **over)


def report(test=None, long=None, longer=None, position=None, own=False) -> dict:
    return {"order": "0073", "arm": "m", "own_text": own, "rows_sha256": "r" * 64, "answers": "a.jsonl",
            "splits": {"test": {"counts": test or counts(), "by": {"position": {}}},
                       "test_long": {"counts": long or counts(), "by": {"position": position or POSITIONS}},
                       "test_longer": {"counts": longer or counts(**{"false passes": 900}), "by": {"position": {"9": [1, 9]}}}}}


def rows(split, n=60, fp=0, fail=4, right=3, checks=3) -> list:
    return [{"id": f"{split}{k}", "split": split, "false passes": fp, "built to fail": fail, "parts right": right,
             "gold checks": checks} for k in range(n)]


def keep(test=None, long=None) -> list:
    return (test or rows("test")) + (long or rows("test_long"))


def test_test_long_is_held_to_its_own_bounds_and_test_longer_decides_nothing():
    got = L.lines(report())
    judged = {k: v for k, v in got.items() if "(reported)" not in k}
    assert all(v["passes"] and not v["kill"] for v in judged.values()), {k: v for k, v in judged.items() if not v["passes"]}
    assert got["line 8: false passes, test_longer (reported)"]["counts"] == [900, 2970]
    assert got["line 8: false passes, test_longer (reported)"]["passes"] is None and not got["line 8: false passes, test_longer (reported)"]["kill"]
    c = lambda **o: counts(**{"replies checked": 1000, "agree": 990, **o})                  # noqa: E731
    assert L.lines(report(long=c(**{"false passes": 21})))["line 4: false passes, test_long"]["passes"] is False
    assert L.lines(report(long=c(**{"false passes": 51})))["line 4: false passes, test_long"]["kill"] is True
    assert L.lines(report(long=counts(**{"parts right": 949})))["line 4: parts right, test_long"]["passes"] is False
    assert L.lines(report(long=counts(**{"replies checked": 2850, "agree": 2800})))["line 4: replies checked, test_long"]["passes"]
    assert not L.lines(report(test=counts(**{"replies checked": 2939})))["line 2: replies checked, test"]["passes"]
    late = "line 4: parts right at positions 5 to 8 over 1 to 4, test_long"
    assert L.lines(report(position={"1": [90, 100], "6": [81, 100]}))[late]["passes"]
    assert not L.lines(report(position={"1": [90, 100], "6": [80, 100]}))[late]["passes"]


def test_line_9_keeps_arm_l_only_with_no_kill_and_lines_3_and_5():
    better = keep(long=rows("test_long", fp=0, right=3))
    worse = keep(long=rows("test_long", fp=2, right=1))
    got = L.decide(report(), report(), better, worse)
    assert got["line 5"]["passes"] and got["line 3"]["passes"] and got["line 9"]["kept"] == "l"
    assert got["line 5"]["false passes over replies built to fail"]["l"] == [0, 240]
    got = L.decide(report(), report(), worse, better)                                       # worse on test_long
    assert not got["line 5"]["passes"] and got["line 9"]["kept"] == "k" and "negative" in got["line 9"]["why"]
    same = L.decide(report(), report(), keep(), keep())
    assert same["line 3"]["passes"] and not same["line 5"]["passes"] and same["line 9"]["kept"] == "k"
    lost = L.decide(report(), report(), keep(test=rows("test", fp=2), long=better[60:]), keep(long=worse[60:]))
    assert lost["line 3"]["false passes over replies built to fail"]["whole interval above 0"] and not lost["line 3"]["passes"]
    assert lost["line 9"]["kept"] == "k"
    killed = L.decide(report(long=counts(**{"replies checked": 1000, "agree": 990, "false passes": 51})), report(), better, worse)
    assert killed["line 9"]["kept"] == "k" and "line 4: false passes, test_long" in killed["line 9"]["kills"]["l"]


def test_line_3_holds_when_the_interval_on_test_holds_zero():
    m, k = rows("test"), rows("test")
    for j in range(0, 60, 12):
        m[j] = dict(m[j], **{"false passes": 1})
    k[5] = dict(k[5], **{"false passes": 1})
    got = L.compared(m, k, "test", "false passes", "built to fail")
    assert got["l"] == [5, 240] and got["k"] == [1, 240] and got["within"] and not got["whole interval above 0"]
    assert got["resamples"] == 2000 and got == L.compared(m, k, "test", "false passes", "built to fail")   # seeded
    with pytest.raises(ValueError):
        L.compared(m, k[:-1], "test", "false passes", "built to fail")


def test_the_time_line_and_the_script(tmp_path):
    assert L.time_line({"2 to 4 parts": {"median seconds": 8.0}, "5 to 8 parts": {"median seconds": 12.0}})["passes"]
    assert not L.time_line({"2 to 4 parts": {"median seconds": 8.0}, "5 to 8 parts": {"median seconds": 12.01}})["passes"]
    assert L.time_line(None)["passes"] is None

    def put(name, obj):
        (tmp_path / name).write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return str(tmp_path / name)
    lines = lambda rs: "".join(json.dumps(r) + "\n" for r in rs)                            # noqa: E731
    sys.argv = ["exp_view_long_compare.py", "--l", put("l.json", report()), "--k", put("k.json", report()),
                "--keep-l", put("kl.jsonl", lines(keep(long=rows("test_long", right=3)))),
                "--keep-k", put("kk.jsonl", lines(keep(long=rows("test_long", fp=2, right=1)))), "--out", str(tmp_path / "o.json")]
    L.main()
    got = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert got["line 9"]["kept"] == "l" and got["lines not passed"]["l"] == [] and got["line 7"]["passes"] is None
    sys.argv[sys.argv.index("--k") + 1] = put("k_own.json", report(own=True))              # not free values
    with pytest.raises(ValueError):
        L.main()
