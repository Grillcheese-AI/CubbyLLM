"""Order 0074: arm L (trained on 2 to 8 parts) and arm K (0073's kept adapter) set against "Acceptance", and line 9
(which reader is kept).

Each arm's report is `exp_view_split_score.py`'s on test, test_long and test_longer (arm M's target, free values),
both on the same rows; each arm's ``--keep`` file holds its rows' counts. Every line is judged from its two counts as
an exact fraction (`exp_view_split_compare.line`): a rate at its bound passes, a kill is a rate past its kill bound,
and a rate with nothing to count over neither passes nor kills. Both arms get every line; arm L is the one judged.

**The interval** (lines 3 and 5): the rows of the set, the same rows in the same order for both arms, 2,000
resamples drawn with seed 74, on the difference of the two rates (arm L's less arm K's), from the 51st to the
1,950th of the differences in order (`exp_view_split_compare.interval`).

**Line 9**, as written: arm L replaces arm K if it hits no kill and lines 3 and 5 hold; otherwise arm K stays and
the order ends as a negative result. Line 7 (time at home) is judged from `exp_view_split_time.py`'s report when it
is given; line 8 (test_longer) is reported and decides nothing.

  python validation/exp_view_long_compare.py --l <report> --k <report> --keep-l <counts> --keep-k <counts>
      [--time-l <exp_view_split_time.py's report>] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_view_split_compare as C  # noqa: E402

__wiring__ = "STANDALONE"

SETS, SEED = ("test", "test_long", "test_longer"), 74
SHORT, LONG = 8.0, 12.0                                   # line 7: median seconds at home, 2 to 4 and 5 to 8 parts


def _get(c: dict, k: str) -> int:
    return c.get(k, 0)


def positions(split: dict, lo: int, hi: int) -> tuple[int, int]:
    cells = [v for k, v in split["by"]["position"].items() if lo <= int(k) <= hi]
    return sum(x[0] for x in cells), sum(x[1] for x in cells)


def set_lines(c: dict, name: str, checked: F) -> dict:
    """Lines 2 and 4's counts on one set (`checked`: the bound on replies checked)."""
    g, line = (lambda k: _get(c, k)), C.line
    return {f"gold checks with their line, {name}": line(g("gold checks with their line"), g("gold checks"), "at least 0.90", lambda r: r >= F(9, 10)),
            f"agreement, {name}": line(g("agree"), g("replies checked"), "at least 0.95", lambda r: r >= F(95, 100)),
            f"false passes, {name}": line(g("false passes"), g("replies checked"), "at most 0.02 (kill above 0.05)",
                                          lambda r: r <= F(2, 100), lambda r: r > F(5, 100)),
            f"false passes over replies built to fail, {name} (reported)": line(
                g("false passes"), g("checked replies built to fail"), "reported: the rate that counts for a user", lambda r: True),
            f"parts right, {name}": line(g("parts right"), g("gold checks"), "at least 0.95", lambda r: r >= F(95, 100)),
            f"parts fixed by one reversal, {name}": line(g("parts fixed by one reversal"), g("gold checks"),
                                                         "at most 0.03 (kill above 0.08)", lambda r: r <= F(3, 100), lambda r: r > F(8, 100)),
            f"groups right, {name}": line(g("gold pairs right"), g("gold pairs"), "at least 0.97 (kill below 0.90)",
                                          lambda r: r >= F(97, 100), lambda r: r < F(9, 10)),
            f"replies checked, {name}": line(g("replies checked"), g("replies"), f"at least {float(checked):.2f}", lambda r: r >= checked)}


def lines(report: dict) -> dict:
    """An arm's lines 1, 2, 4, 6 and 8 from its report."""
    sp = report["splits"]
    drops = sum(_get(sp[s]["counts"], "characters in no line and not flagged") for s in SETS if s in sp)
    out = {"line 1: characters dropped silently, every set": {"counts": [drops], "needs": "0 (kill above)", "passes": drops == 0, "kill": drops > 0}}
    out.update({f"line 2: {k}": v for k, v in set_lines(sp["test"]["counts"], "test", F(98, 100)).items()})
    out.update({f"line 4: {k}": v for k, v in set_lines(sp["test_long"]["counts"], "test_long", F(95, 100)).items()})
    early, late = positions(sp["test_long"], 1, 4), positions(sp["test_long"], 5, 8)
    r_early, r_late = (F(*x) if x[1] else None for x in (early, late))
    out["line 4: parts right at positions 5 to 8 over 1 to 4, test_long"] = {
        "counts": [list(late), list(early)], "rate": round(float(r_late / r_early), 4) if r_early and r_late is not None else None,
        "needs": "at least 0.90", "passes": bool(r_early and r_late is not None and r_late >= F(9, 10) * r_early), "kill": False}
    for name in ("test", "test_long"):
        c = sp[name]["counts"]
        asked = _get(c, "gold checks asked by the reader") + _get(c, "gold checks asked by the host")
        out[f"line 6: gold checks asked, {name}"] = C.line(asked, _get(c, "gold checks"), "at most 0.10", lambda r: r <= F(1, 10))
    if "test_longer" in sp:                                 # reported, deciding nothing
        got = set_lines(sp["test_longer"]["counts"], "test_longer", F(95, 100))
        out.update({f"line 8: {k} (reported)": dict(v, passes=None, kill=False) for k, v in got.items()})
        out["line 8: parts right by position, test_longer (reported)"] = sp["test_longer"]["by"]["position"]
    return out


def compared(keep_l: list, keep_k: list, split: str, top: str, bottom: str) -> dict:
    """The interval of arm L's rate less arm K's on one set's rows."""
    rows_l, rows_k = ([r for r in keep if r["split"] == split] for keep in (keep_l, keep_k))
    if [r["id"] for r in rows_l] != [r["id"] for r in rows_k] or not rows_l:
        raise ValueError(f"the two arms' {split} rows are not the same rows in the same order")
    raw = C.interval(rows_l, rows_k, top, bottom, seed=SEED)
    lo, hi = raw["interval"]
    got = C.shown(raw)
    return {"l": got["m"], "k": got["w"], "rates": got["rates"], "interval": got["interval"], "resamples": got["resamples"],
            "resamples with no rate": got["resamples with no rate"], "within": got["within"],
            "whole interval below 0": hi is not None and hi < 0, "whole interval above 0": lo is not None and lo > 0}


def time_line(report: dict | None) -> dict:
    if report is None:
        return {"needs": f"medians of at most {SHORT:g} s (2 to 4 parts) and {LONG:g} s (5 to 8)", "passes": None, "measured": "not given"}
    med = [(report.get(k) or {}).get("median seconds") for k in ("2 to 4 parts", "5 to 8 parts")]
    return {"needs": f"medians of at most {SHORT:g} s (2 to 4 parts) and {LONG:g} s (5 to 8); above 25 s not served",
            "median seconds": med, "passes": None not in med and med[0] <= SHORT and med[1] <= LONG,
            "served": None not in med and max(med) <= 25.0}


def decide(rep_l: dict, rep_k: dict, keep_l: list, keep_k: list, time_l: dict | None = None) -> dict:
    if rep_l["rows_sha256"] != rep_k["rows_sha256"] or rep_l.get("own_text") or rep_k.get("own_text"):
        raise ValueError("the two reports are not on the same rows, or not both free values")
    arms = {"l": lines(rep_l), "k": lines(rep_k)}
    kills = {arm: [k for k, v in got.items() if v.get("kill")] for arm, got in arms.items()}
    fp_test = compared(keep_l, keep_k, "test", "false passes", "built to fail")
    fp_long = compared(keep_l, keep_k, "test_long", "false passes", "built to fail")
    right_long = compared(keep_l, keep_k, "test_long", "parts right", "gold checks")
    line3 = {"what": "no loss on test: not passed if the whole interval of L - K is above 0", "false passes over replies built to fail": fp_test,
             "passes": fp_test["interval"][0] is not None and not fp_test["whole interval above 0"]}
    line5 = {"what": "on test_long, L - K below 0 for false passes over replies built to fail and above 0 for parts right, "
             "the whole interval", "false passes over replies built to fail": fp_long, "parts right": right_long,
             "passes": fp_long["whole interval below 0"] and right_long["whole interval above 0"]}
    kept = "l" if not kills["l"] and line3["passes"] and line5["passes"] else "k"
    why = ("arm L hits no kill and lines 3 and 5 hold" if kept == "l" else
           f"arm L {'hits a kill: ' + ', '.join(kills['l']) if kills['l'] else 'does not hold line 3 or 5'}; arm K stays, a negative result")
    return {"order": "0074", "what": "arm L against Acceptance, arm K beside it, and line 9",
            "reports": {"l": rep_l.get("answers"), "k": rep_k.get("answers")}, "arms": arms,
            "lines not passed": {arm: [k for k, v in got.items() if v.get("passes") is False] for arm, got in arms.items()},
            "line 3": line3, "line 5": line5, "line 7": time_line(time_l), "line 9": {"kept": kept, "why": why, "kills": kills}}


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("--l", "--k", "--keep-l", "--keep-k", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--time-l", default=None)
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))                       # noqa: E731
    rows = lambda p: [json.loads(ln) for ln in open(p, encoding="utf-8")]                  # noqa: E731
    out = decide(load(a.l), load(a.k), rows(a.keep_l), rows(a.keep_k), load(a.time_l) if a.time_l else None)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("lines not passed", "line 3", "line 5", "line 7", "line 9")}, indent=1))
    print("compared:", a.out)


if __name__ == "__main__":
    main()
