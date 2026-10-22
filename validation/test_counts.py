"""test_counts -- a pytest JUnit report summed per test file (order 0031: "every test that passed before passes after,
by count, per test file"), and two such summaries compared.

    python -m pytest tests standin/tests --junitxml=<report.xml>
    python validation/test_counts.py <report.xml> --out validation/logs/<name>.json
    python validation/test_counts.py --compare <before.json> <after.json> [--moved validation/logs/move_map_0031_2026-10-22.json]

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

__wiring__ = "STANDALONE"
KINDS = ("passed", "failed", "skipped", "error")


def per_file(report: str) -> dict:
    per = defaultdict(lambda: dict.fromkeys(KINDS, 0))
    for c in ET.parse(report).iter("testcase"):
        cls = c.get("classname") or ""
        f = (c.get("file") or (cls.replace(".", "/") + ".py" if cls else "(collection)"))
        if cls and f.endswith(".py") and not Path(f).suffix == ".py":
            f += ".py"
        k = ("failed" if c.find("failure") is not None else "error" if c.find("error") is not None
             else "skipped" if c.find("skipped") is not None else "passed")
        per[f][k] += 1
    return dict(sorted(per.items()))


def compare(before: dict, after: dict, moved: dict) -> dict:
    """Per test file (a moved file under its new name): the counts before and after, and every difference."""
    rename = dict(moved)                                  # old test path -> new test path, as the summaries key them
    diff = {}
    after_files = dict(after["files"])
    for f, c in before["files"].items():
        g = rename.get(f, f)
        a = after_files.pop(g, None)
        if a != c:
            diff[f] = {"before": c, "after": a, "now": g}
    for g, a in after_files.items():
        diff[g] = {"before": None, "after": a}
    return {"same": not diff, "differences": diff, "totals": {"before": before["totals"], "after": after["totals"]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--moved", help="a JSON of old test path -> new test path")
    a = ap.parse_args()
    if a.compare:
        b, c = (json.load(open(p, encoding="utf-8")) for p in a.compare)
        moved = json.load(open(a.moved, encoding="utf-8")) if a.moved else {}
        r = compare(b, c, moved)
        print(json.dumps(r, indent=1))
        return 0 if r["same"] else 1
    files = per_file(a.report)
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"git": rev, "totals": {k: sum(v[k] for v in files.values()) for k in KINDS}, "files": files}
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(rev, out["totals"], len(files), "files;", "red:", [f for f, v in files.items() if v["failed"] or v["error"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
