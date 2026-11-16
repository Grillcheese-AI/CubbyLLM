"""Order 0073: the two arms' counted reads set against "Acceptance", and line 9 (which arm is kept).

Each arm's report is `exp_view_split_score.py`'s on test and test_long, in the variant point 7 kept. Every line is
judged from its two counts as an exact fraction, on the set the order's report names: lines 2, 3, 5 and 6 on test,
line 4 on test_long (against test), lines 1 and 7 on both. Line 8 (time at home) is not counted here.

**Line 9**, as written: an arm that hits a kill is out; both out, the order ends as a negative result. Otherwise the
arm with fewer false passes over the replies built to fail, on test; if the two are within a 95% interval taken
over rows, the arm with more parts right; if still within, arm M. The interval: 2,000 resamples of the test rows
with replacement, the same resamples for both arms, on the difference of the two rates (arm M's less arm W's), from
the 2.5th to the 97.5th percentile; "within" is an interval that holds 0.

  python validation/exp_view_split_compare.py --m <report> --w <report> --keep-m <rows' counts> --keep-w <rows' counts>
      --point7 validation/logs/view_split_point7.json [--p <arm P's report, exp_view_reader.py --score>] --out <json>
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction as F
from pathlib import Path

__wiring__ = "STANDALONE"

RESAMPLES, SEED = 2000, 73


def line(a: int, b: int, needs: str, ok, kill=None) -> dict:
    """One rate from its two counts, with what it needs; `ok` and `kill` judge the exact fraction. A rate with
    nothing to count over does not pass and is not a kill (a kill is a rate past its bound, as written)."""
    r = F(a, b) if b else None
    return {"counts": [a, b], "rate": round(float(r), 4) if r is not None else None, "needs": needs,
            "passes": bool(r is not None and ok(r)), "kill": bool(kill is not None and r is not None and kill(r))}


def lines(report: dict) -> dict:
    """An arm's lines 1 to 7 from its report on test and test_long."""
    t, lg = (report["splits"][s]["counts"] for s in ("test", "test_long"))

    def get(c, k):
        return c.get(k, 0)

    def positions(lo, hi):
        cells = [v for k, v in report["splits"]["test_long"]["by"]["position"].items() if lo <= int(k) <= hi]
        return sum(x[0] for x in cells), sum(x[1] for x in cells)
    out = {}
    drops = get(t, "characters in no line and not flagged") + get(lg, "characters in no line and not flagged")
    out["line 1: characters dropped silently, test and test_long"] = {"counts": [drops], "needs": "0 (kill above)", "passes": drops == 0, "kill": drops > 0}
    out["line 2: gold checks with their line, test"] = line(get(t, "gold checks with their line"), get(t, "gold checks"), "at least 0.90", lambda r: r >= F(9, 10))
    out["line 3: agreement, test"] = line(get(t, "agree"), get(t, "replies checked"), "at least 0.95", lambda r: r >= F(95, 100))
    out["line 3: false passes, test"] = line(get(t, "false passes"), get(t, "replies checked"), "at most 0.02 (kill above 0.05)",
                                             lambda r: r <= F(2, 100), lambda r: r > F(5, 100))
    out["line 3: false passes over replies built to fail, test (reported)"] = line(
        get(t, "false passes"), get(t, "checked replies built to fail"), "reported: the rate that counts for a user", lambda r: True)
    a_in, a_lg = (F(get(c, "agree"), get(c, "replies checked")) if get(c, "replies checked") else None for c in (t, lg))
    line3 = out["line 3: agreement, test"]["passes"] and out["line 3: false passes, test"]["passes"]
    out["line 4: agreement on test_long over test's"] = {
        "counts": [get(lg, "agree"), get(lg, "replies checked")], "rate": round(float(a_lg / a_in), 4) if a_in and a_lg is not None else None,
        "needs": "at least 0.90, counted only if line 3 passes on test", "counted": line3,
        "passes": bool(line3 and a_in and a_lg is not None and a_lg >= F(9, 10) * a_in), "kill": False}
    out["line 4: false passes, test_long"] = line(get(lg, "false passes"), get(lg, "replies checked"), "at most 0.02 (kill above 0.05)",
                                                  lambda r: r <= F(2, 100), lambda r: r > F(5, 100))
    early, late = positions(1, 4), positions(5, 8)
    r_early, r_late = (F(*x) if x[1] else None for x in (early, late))
    out["line 4: parts right at positions 5 to 8 over 1 to 4, test_long"] = {
        "counts": [list(late), list(early)], "rate": round(float(r_late / r_early), 4) if r_early and r_late is not None else None,
        "needs": "at least 0.90", "passes": bool(r_early and r_late is not None and r_late >= F(9, 10) * r_early), "kill": False}
    out["line 5: parts right, test"] = line(get(t, "parts right"), get(t, "gold checks"), "at least 0.95", lambda r: r >= F(95, 100))
    out["line 5: parts fixed by one reversal, test"] = line(get(t, "parts fixed by one reversal"), get(t, "gold checks"),
                                                            "at most 0.03 (kill above 0.08)", lambda r: r <= F(3, 100), lambda r: r > F(8, 100))
    out["line 6: pairs right, test"] = line(get(t, "gold pairs right"), get(t, "gold pairs"), "at least 0.97 (kill below 0.90)",
                                            lambda r: r >= F(97, 100), lambda r: r < F(9, 10))
    for name, c, need in (("test", t, F(98, 100)), ("test_long", lg, F(95, 100))):
        out[f"line 7: replies checked, {name}"] = line(get(c, "replies checked"), get(c, "replies"), f"at least {float(need):.2f}", lambda r, need=need: r >= need)
        asked = get(c, "gold checks asked by the reader") + get(c, "gold checks asked by the host")
        out[f"line 7: gold checks asked, {name}"] = line(asked, get(c, "gold checks"), "at most 0.10", lambda r: r <= F(1, 10))
    for name, c in (("test_long", lg),):                   # beside test's, deciding nothing
        for label, a, b in (("line 2: gold checks with their line", "gold checks with their line", "gold checks"),
                            ("line 5: parts right", "parts right", "gold checks"),
                            ("line 5: parts fixed by one reversal", "parts fixed by one reversal", "gold checks"),
                            ("line 6: pairs right", "gold pairs right", "gold pairs")):
            out[f"{label}, {name} (reported)"] = line(get(c, a), get(c, b), "reported beside test's", lambda r: True)
    return out


def interval(m: list, w: list, top: str, bottom: str, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """The two arms' rate ``top / bottom`` over the same rows and the 95% interval of their difference (M less W)."""
    n = len(m)
    rng, diffs, skipped = random.Random(seed), [], 0

    def rate(rows, idx):
        b = sum(rows[i][bottom] for i in idx)
        return F(sum(rows[i][top] for i in idx), b) if b else None
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a, b = rate(m, idx), rate(w, idx)
        if a is None or b is None:
            skipped += 1
        else:
            diffs.append(a - b)
    diffs.sort()
    whole = range(n)
    lo, hi = (diffs[int(0.025 * len(diffs))], diffs[max(0, -(-975 * len(diffs) // 1000) - 1)]) if diffs else (None, None)
    return {"m": [sum(x[top] for x in m), sum(x[bottom] for x in m)], "w": [sum(x[top] for x in w), sum(x[bottom] for x in w)],
            "rates": [rate(m, whole), rate(w, whole)], "interval": [lo, hi], "resamples": resamples, "resamples with no rate": skipped,
            "within": bool(diffs) and lo <= 0 <= hi}


def shown(got: dict) -> dict:
    """An interval's fractions as rounded numbers, for the report."""
    def f(x):
        return round(float(x), 4) if x is not None else None
    return dict(got, rates=[f(x) for x in got["rates"]], interval=[f(x) for x in got["interval"]])


def line9(keep_m: list, keep_w: list, kills: dict) -> dict:
    """Which arm is kept. `keep_*`: the test rows' counts, the same rows in the same order; `kills`: {arm: [names]}."""
    out = {arm: bool(kills[arm]) for arm in ("m", "w")}
    if out["m"] and out["w"]:
        return {"kept": None, "why": "both arms hit a kill: a negative result, nothing is served", "kills": kills}
    if out["m"] or out["w"]:
        kept = "w" if out["m"] else "m"
        return {"kept": kept, "why": f"arm {'m' if out['m'] else 'w'} hit a kill and is out", "kills": kills}
    fp = interval(keep_m, keep_w, "false passes", "built to fail")
    got = {"kills": kills, "false passes over replies built to fail": shown(fp)}
    if not fp["within"] and None not in fp["rates"]:
        return dict(got, kept="m" if fp["rates"][0] < fp["rates"][1] else "w", why="fewer false passes over replies built to fail, outside the interval")
    right = interval(keep_m, keep_w, "parts right", "gold checks")
    got["parts right"] = shown(right)
    if not right["within"] and None not in right["rates"]:
        return dict(got, kept="m" if right["rates"][0] > right["rates"][1] else "w", why="false passes within the interval; more parts right, outside the interval")
    return dict(got, kept="m", why="false passes and parts right both within the interval: arm M")


def arm_p(report: dict) -> dict:
    """Round 2's adapter beside the two arms, as round 2's scorer counted lines 2 to 4."""
    out = {}
    for s in ("test", "test_long"):
        got = report["splits"][s]
        c = got["counts"]
        out[s] = {"line 2: parts with span and kind": [c.get("parts with span and kind", 0), c.get("parts", 0), got["span_and_kind"]],
                  "line 3: agreement": [c.get("agree", 0), c.get("checked", 0), got["agreement"]],
                  "line 3: false passes": [c.get("false passes", 0), c.get("checked", 0), got["false_passes"]],
                  "replies checked": [c.get("checked", 0), c.get("replies", 0), got["checked_share"]],
                  "answers not parsed": c.get("answers not parsed (all asked)", 0), "rows not read": c.get("rows not read", 0)}
    out["line 4: agreement on test_long over test's"] = report.get("long_over_in_length")
    return out


def load(report_p: str, keep_p: str, arm: str, point7: dict) -> tuple[dict, list]:
    rep = json.loads(Path(report_p).read_text(encoding="utf-8"))
    kept = point7["kept"][arm]["own_text"]
    if (rep.get("order"), rep.get("arm"), bool(rep.get("own_text"))) != ("0073", arm, kept) or not {"test", "test_long"} <= set(rep["splits"]):
        raise ValueError(f"{report_p} is not arm {arm}'s report on test and test_long in the variant point 7 kept")
    if rep["rows_sha256"] != point7["rows_sha256"] or any(rep["splits"][s]["counts"].get("rows not read", 0) for s in ("test", "test_long")):
        raise ValueError(f"{report_p}: not the rows point 7 was decided on, or a set that was not read whole")
    rows = [r for r in map(json.loads, open(keep_p, encoding="utf-8")) if r["split"] == "test"]
    return rep, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("--m", "--w", "--keep-m", "--keep-w", "--point7", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--p", default=None)
    a = ap.parse_args()
    point7 = json.loads(Path(a.point7).read_text(encoding="utf-8"))
    (rep_m, keep_m), (rep_w, keep_w) = load(a.m, a.keep_m, "m", point7), load(a.w, a.keep_w, "w", point7)
    if [r["id"] for r in keep_m] != [r["id"] for r in keep_w] or not keep_m:
        raise ValueError("the two arms' test rows are not the same rows in the same order")
    arms = {arm: lines(rep) for arm, rep in (("m", rep_m), ("w", rep_w))}
    kills = {arm: [k for k, v in got.items() if v["kill"]] for arm, got in arms.items()}
    out = {"order": "0073", "what": "the two arms against Acceptance, and line 9", "variant": {arm: point7["kept"][arm]["own_text"] for arm in arms},
           "reports": {"m": Path(a.m).name, "w": Path(a.w).name}, "arms": arms,
           "lines not passed": {arm: [k for k, v in got.items() if not v["passes"]] for arm, got in arms.items()},
           "line 9": line9(keep_m, keep_w, kills), "line 8": "time at home: measured apart"}
    if a.p:
        out["arm p"] = arm_p(json.loads(Path(a.p).read_text(encoding="utf-8")))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"lines not passed": out["lines not passed"], "line 9": out["line 9"]}, indent=1))
    print("compared:", a.out)


if __name__ == "__main__":
    main()
