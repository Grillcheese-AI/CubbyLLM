"""Order 0077: arm R (trained with real prompts) and arm L (0074's kept reader) set against "Acceptance", and line 5
(which reader is kept). Counts only; no prompt text is printed or written.

Inputs, each `exp_view_split_score.py`'s report and its ``--keep`` counts:
- both arms on test_real and test_dolly (`long_rows_0077`);
- both arms on 0074's test and test_long (`long_rows_0074`): arm L's answers there are 0074's own read.

Lines, as the order states them (each rate an exact fraction, `exp_view_split_compare.line`):
1. silent drops 0 on every set (kill);
2. on test_real and test_dolly each, for arm R: gold checks with their line >= 0.90, agreement >= 0.95, false passes
   <= 0.02 (kill > 0.05), parts right >= 0.95, groups right >= 0.97 (kill < 0.90), replies checked >= 0.95;
3. the prompt's task lines written as `check`, by set and arm (from the answers and rows: a prompt line any answer
   line of kind `check` overlaps), reported;
4. no loss: on 0074's test and test_long, arm R's false passes over replies built to fail not above arm L's: not
   passed if the whole interval of R - L is above 0;
5. arm R replaces arm L if it hits no kill, line 4 holds, and either arm L misses line 2 on a real set where arm R
   passes it, or on test_real R - L for false passes over replies built to fail is below 0 over the whole interval.
6. the time at home is its own script's report (`exp_view_split_time.py`), given with ``--time-r``;
7. breakdowns, both arms: by source and language, by where the prompt sits, by prompt length (100 characters).

The interval: the same rows in the same order for both arms, 2,000 resamples with seed 77, 2.5th to 97.5th
(`exp_view_split_compare.interval`).

  python validation/exp_view_real_compare.py --r <report> --l <report> --keep-r <counts> --keep-l <counts>
      --r74 <report> --l74 <report> --keep-r74 <counts> --keep-l74 <counts> --answers-r <jsonl> --answers-l <jsonl>
      --rows <long_rows_0077 rows.jsonl> [--time-r <report>] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import exp_view_long_compare as LC  # noqa: E402
import exp_view_split_compare as C  # noqa: E402

__wiring__ = "STANDALONE"

REAL, OLD, SEED = ("test_real", "test_dolly"), ("test", "test_long"), 77
LINE2 = ("gold checks with their line", "agreement", "false passes, ", "parts right", "groups right", "replies checked")


def line2(report: dict, name: str) -> dict:
    got = LC.set_lines(report["splits"][name]["counts"], name, F(95, 100))
    return {k: v for k, v in got.items() if k.startswith(LINE2) and "one reversal" not in k}


def interval(keep_a: list, keep_b: list, split: str) -> dict:
    a, b = ([r for r in keep if r["split"] == split] for keep in (keep_a, keep_b))
    if [r["id"] for r in a] != [r["id"] for r in b] or not a:
        raise ValueError(f"the two arms' {split} rows are not the same rows in the same order")
    raw = C.interval(a, b, "false passes", "built to fail", seed=SEED)
    lo, hi = raw["interval"]
    return dict(C.shown(raw), **{"whole interval below 0": hi is not None and hi < 0,
                                 "whole interval above 0": lo is not None and lo > 0})


def prompt_lines(answers: list, rows: dict) -> dict:
    """Line 3: per set, the prompt's task lines, those an answer line of kind `check` overlaps, and those copied whole."""
    from cubbyllm.speech import view_split as V
    out = defaultdict(Counter)
    for k in answers:
        r = rows[k["id"]]
        got = V.parse(k["answer"], r["instruction"], "m", strict=False).get("lines", [])
        for seg in r["segments"]:
            if seg.get("task") != "real":
                continue
            a, b = V.span(r["instruction"], seg["start"], seg["end"])
            over = [x for x in got if x["start"] < b and x["end"] > a]
            c = out[r["split"]]
            c["prompt lines"] += 1
            c["written, in part or whole, as check"] += any(x["kind"] == "check" for x in over)
            c["copied whole as one task line"] += any(x["kind"] == "task" and (x["start"], x["end"]) == (a, b) for x in over)
    return {s: dict(c) for s, c in sorted(out.items())}


def breakdowns(keep: list, rows: dict) -> dict:
    """Line 7: agreement and false passes over replies built to fail, by source and language, place and length."""
    keys = {"by source and language": lambda r: f"{r['source']} {r['lang']}", "by where the prompt sits": lambda r: r["prompt_at"],
            "by prompt length (100 characters)": lambda r: r["prompt_chars"] // 100 * 100}
    out = {}
    for name, key in keys.items():
        t = defaultdict(Counter)
        for k in keep:
            c = t[f"{k['split']} {key(rows[k['id']])}"]
            for f in ("checked", "agree", "false passes", "built to fail"):
                c[f] += k[f]
            c["rows"] += 1
        out[name] = {g: {"rows": c["rows"], "agreement": [c["agree"], c["checked"]],
                         "false passes over replies built to fail": [c["false passes"], c["built to fail"]]}
                     for g, c in sorted(t.items())}
    return out


def decide(rep: dict, keep: dict, lines3: dict, time_r: dict | None) -> dict:
    drops = {arm: sum(r["splits"][s]["counts"].get("characters in no line and not flagged", 0)
                      for r in (rep[arm], rep[arm + "74"]) for s in r["splits"]) for arm in "rl"}
    l2 = {arm: {s: line2(rep[arm], s) for s in REAL} for arm in "rl"}
    kills = {arm: ([f"line 1: {drops[arm]} characters dropped"] if drops[arm] else [])
             + [k for s in REAL for k, v in l2[arm][s].items() if v["kill"]] for arm in "rl"}
    passes2 = {arm: {s: all(v["passes"] for v in l2[arm][s].values()) for s in REAL} for arm in "rl"}
    l4 = {s: interval(keep["r74"], keep["l74"], s) for s in OLD}
    line4 = not any(v["whole interval above 0"] or v["interval"][0] is None for v in l4.values())
    fp_real = interval(keep["r"], keep["l"], "test_real")
    a = any(passes2["r"][s] and not passes2["l"][s] for s in REAL)
    kept = "r" if not kills["r"] and line4 and (a or fp_real["whole interval below 0"]) else "l"
    why = ("arm R hits no kill, line 4 holds, and " + ("arm L misses line 2 on a real set where arm R passes it"
           if a else "arm R's false passes on test_real are lower over the whole interval")) if kept == "r" else \
        ("arm R hits a kill: " + ", ".join(kills["r"]) if kills["r"] else "line 4 does not hold" if not line4 else
         "neither condition of line 5 holds") + "; arm L stays, and the real-prompt read is reported as arm L's"
    return {"order": "0077", "what": "arms R and L against Acceptance, and line 5",
            "answers": {k: v.get("answers") for k, v in rep.items()},
            "line 1": {arm: {"characters dropped silently, every set": drops[arm], "kill": drops[arm] > 0} for arm in "rl"},
            "line 2": l2, "line 2 passed, by set": passes2, "line 3": lines3,
            "line 4": {"what": "on 0074's test and test_long, R - L false passes over replies built to fail; not passed "
                               "if the whole interval is above 0", "by set": l4, "passes": line4},
            "line 5": {"kept": kept, "why": why, "kills": kills,
                       "test_real: R - L false passes over replies built to fail": fp_real,
                       "arm L misses line 2 on a real set where arm R passes it": a},
            "line 6": time_r or {"measured": "not given (exp_view_split_time.py at home)"},
            "line 7": {arm: breakdowns(keep[arm], ROWS) for arm in "rl"}}


ROWS: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    for n in ("--r", "--l", "--keep-r", "--keep-l", "--r74", "--l74", "--keep-r74", "--keep-l74", "--answers-r",
              "--answers-l", "--rows", "--out"):
        ap.add_argument(n, required=True)
    ap.add_argument("--time-r", default=None)
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))                          # noqa: E731
    jl = lambda p: [json.loads(ln) for ln in open(p, encoding="utf-8")]                       # noqa: E731
    ROWS.update({r["id"]: r for r in jl(a.rows)})
    rep = {"r": load(a.r), "l": load(a.l), "r74": load(a.r74), "l74": load(a.l74)}
    keep = {"r": jl(a.keep_r), "l": jl(a.keep_l), "r74": jl(a.keep_r74), "l74": jl(a.keep_l74)}
    lines3 = {arm: prompt_lines(jl(p), ROWS) for arm, p in (("r", a.answers_r), ("l", a.answers_l))}
    out = decide(rep, keep, lines3, load(a.time_r) if a.time_r else None)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("line 1", "line 2 passed, by set", "line 3", "line 4", "line 5")}, indent=1))
    print("compared:", a.out)


if __name__ == "__main__":
    main()
