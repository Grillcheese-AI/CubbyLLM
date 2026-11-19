"""Order 0075, steps 3, 4 and 7: right, wrong and refused under the lenient matcher (0054's) and the strict one, by arm
and benchmark, with every read applied.

Reads: 0075's read of the lenient matches (``grade_reads_0075_<date>.json``) and 0054's read of C's and C-VM's unmatched
answers (``system_eval_after_2026-11-10.json``, matched to the records by benchmark and question; Cowork's change of
2026-11-11 applied: ``webq:1504`` is debatable). A matched answer the read finds wrong counts wrong under both
matchers; an answer neither read reached stays ``unread`` (M's unmatched answers were never read: their wrong count is
not estimated here). Wrong among spoken carries its Wilson 95% interval when nothing is unread.

  python validation/exp_grade_rates.py [--reads validation/logs/grade_reads_0075_2026-11-19.json] [--out ...]

Wired: STANDALONE (order 0075).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

AFTER = os.path.join(HERE, "logs", "system_eval_after_2026-11-10.json")
COWORK_DEBATABLE = {"webq:1504"}                # Cowork's read of the "after" (2026-11-11): debatable


def after_reads(recs) -> dict[tuple[str, str], tuple[str, str | None]]:
    """{(arm, id): (verdict, reason)} from 0054's read, by question."""
    log = json.load(open(AFTER, encoding="utf-8"))
    by_q = {}
    for x in log["unmatched_read"]:
        by_q[(x["bench"], x["question"], x["arm"])] = x["read"]
    out = {}
    for (arm, b), rows in recs.items():
        if arm == "M":
            continue
        for r in rows:
            v = by_q.get((b, r["question"], "c")) if arm == "C" else (
                by_q.get((b, r["question"], "cvm only")) or by_q.get((b, r["question"], "c")))
            if v:
                v = "debatable" if r["id"] in COWORK_DEBATABLE else v
                out[(arm, r["id"])] = (v, None if v == "right" else "wrong")
    return out


def tally(rows, arm, b, strict, reads) -> dict:
    from system_eval import grade
    from system_eval.grade import wilson
    c = Counter()
    for r in rows:
        spoken = bool(r.get("spoken")) and r.get("verdict") != "refused"
        v = grade.verdict(spoken, r.get("answer"), r["golds"], b, declines=(arm == "M"), strict=strict)
        if v in ("refused", "declined"):
            c["refused"] += 1
            continue
        c["spoken"] += 1
        read = reads.get((arm, r["id"]))
        side = "matched" if v == "right" else "unmatched"
        c[side] += 1
        c[f"{side}_{read[0] if read else ('unread' if side == 'unmatched' else 'not read')}"] += 1
    wrong = c["matched_wrong"] + c["unmatched_wrong"]
    deb = c["matched_debatable"] + c["unmatched_debatable"]
    right = c["matched"] - c["matched_wrong"] - c["matched_debatable"] + c["unmatched_right"]
    out = {"n": len(rows), **dict(c), "right": right, "wrong": wrong, "debatable": deb,
           "unread": c["unmatched_unread"], "right_pct": round(100 * right / max(1, len(rows)), 2),
           "wrong_pct": round(100 * wrong / max(1, len(rows)), 2), "refused_pct": round(100 * c["refused"] / max(1, len(rows)), 2)}
    if not c["unmatched_unread"] and c["spoken"]:
        lo, hi = wilson(wrong, c["spoken"])
        out["wrong_of_spoken"] = {"rate": round(wrong / c["spoken"], 4), "ci95": [round(lo, 4), round(hi, 4)],
                                  "with_debatable": round((wrong + deb) / c["spoken"], 4)}
    return out


def main() -> None:
    from cubbyllm import local_paths
    from exp_grade_rules import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(local_paths.value("CUBBY_DATA") or ".", "system_eval"))
    ap.add_argument("--reads", default=os.path.join(HERE, "logs", "grade_reads_0075_2026-11-19.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = load(a.data)
    reads = after_reads(recs)
    r75 = json.load(open(a.reads, encoding="utf-8"))["reads"]
    for arm, rows in r75.items():
        for i, (v, reason) in rows.items():
            reads[(arm, i)] = (v, reason)
    tab = {f"{arm}/{b}": {"lenient": tally(rows, arm, b, False, reads), "strict": tally(rows, arm, b, True, reads)}
           for (arm, b), rows in sorted(recs.items())}
    reasons = Counter((arm, v[1]) for (arm, _), v in reads.items() if v[0] in ("wrong", "debatable"))
    rep = {"what": "order 0075: right, wrong, refused by arm and benchmark, lenient (0054's) and strict matchers, the reads applied",
           "date": _dt.date.today().isoformat(), "reads": [os.path.basename(a.reads), os.path.basename(AFTER) + " (C, C-VM unmatched)"],
           "table": tab, "wrong_and_debatable_by_reason": {f"{k[0]}/{k[1]}": n for k, n in sorted(reasons.items())}}
    out = a.out or os.path.join(HERE, "logs", f"grade_rates_{rep['date']}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, v in tab.items():
        for m in ("lenient", "strict"):
            x = v[m]
            print(f"{k:13s} {m:7s} n {x['n']:5d} right {x['right']:5d} wrong {x['wrong']:4d} deb {x['debatable']:3d} "
                  f"refused {x['refused']:5d} unread {x['unread']:5d} {x.get('wrong_of_spoken', '')}")
    print(rep["wrong_and_debatable_by_reason"], "->", out)


if __name__ == "__main__":
    main()
