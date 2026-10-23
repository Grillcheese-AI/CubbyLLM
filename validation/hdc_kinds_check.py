"""hdc_kinds_check -- order 0036, item 3: the kind-to-relations table against the scan it replaces, on 0033's hops
(``namesake_probe.HOPS``): the same decisions (the objects spoken, the items withheld), and each arm's latency.

The line: every hop decided the same way by both arms.

    python validation/hdc_kinds_check.py
-> validation/logs/hdc_kinds_check_<date>.json

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"


def arm(scan: bool) -> tuple[list[dict], float]:
    from cubbyllm.reasoning.planner import QuestionPlan
    from cubbyllm.worlds.hdc_store import HdcWorld
    from namesake_probe import HOPS
    hdc = HdcWorld()
    if scan:
        hdc._kinds = None
    rows, t0 = [], time.perf_counter()
    for rel, name in HOPS:
        hdc.clear_withheld()
        t = time.perf_counter()
        got = hdc.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)
        rows.append({"hop": f"{rel} of {name}", "spoken": [x.obj for _f, x in got], "withheld": hdc.withheld(name),
                     "ms": round(1000 * (time.perf_counter() - t), 1)})
    return rows, round(1000 * (time.perf_counter() - t0), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    from cubbyllm.worlds.hdc_kinds import path_for
    from cubbyllm.worlds.hdc_store import HdcWorld
    stamp = dict(sqlite3.connect(f"file:{path_for(HdcWorld().path)}?mode=ro", uri=True).execute("SELECT k, v FROM meta"))
    scan, scan_ms = arm(scan=True)
    table, table_ms = arm(scan=False)
    same = [s["spoken"] == t["spoken"] and s["withheld"] == t["withheld"] for s, t in zip(scan, table)]
    out = {"what": "order 0036 item 3: the kinds table against the scan, on 0033's hops", "table": stamp,
           "hops": len(same), "same_decisions": sum(same), "scan_ms": scan_ms, "table_ms": table_ms,
           "rows": [{"hop": s["hop"], "spoken": t["spoken"], "withheld": t["withheld"], "same": ok,
                     "scan_ms": s["ms"], "table_ms": t["ms"]} for s, t, ok in zip(scan, table, same)]}
    path = os.path.join(HERE, "logs", f"hdc_kinds_check_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), "->", path)


if __name__ == "__main__":
    main()
