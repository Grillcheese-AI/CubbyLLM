"""build_sets -- every set of order 0046 built into ``$CUBBY_DATA_ROOT/retrieval/sets/<set>.jsonl`` (the schema in
``common``), with a count table (rows and mentions per split and language) written to
``validation/logs/retrieval_sets_<date>.json``: ids and counts only, no question text.

    python validation/retrieval_sets/build_sets.py [--only mintaka qald9plus ...]

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets import build_el, build_qa  # noqa: E402
from retrieval_sets.common import read_rows, source, write_rows  # noqa: E402
from retrieval_sets.download import SETS  # noqa: E402

__wiring__ = "STANDALONE"

BUILDERS = {"mintaka": lambda s: build_qa.mintaka(), "qald9plus": lambda s: build_qa.qald9plus(),
            "webqsp_el": lambda s: build_el.el4qa("webqsp_el"), "graphq_el": lambda s: build_el.el4qa("graphq_el"),
            "simplequestions_wd": lambda s: build_qa.simplequestions_wd(), "shadowlink": build_el.shadowlink,
            "amber": lambda s: build_el.amber(), "cronquestions": lambda s: build_qa.cronquestions()}
RAW_OF = {"webqsp_el": "el4qa", "graphq_el": "el4qa"}


def counts(rows: list[dict]) -> dict:
    c = collections.defaultdict(lambda: {"rows": 0, "mentions": 0, "mentions_with_text": 0})
    for r in rows:
        k = f"{r['split']}/{r['lang']}"
        c[k]["rows"] += 1
        c[k]["mentions"] += len(r["mentions"])
        c[k]["mentions_with_text"] += sum(1 for m in r["mentions"] if m.get("text"))
    return dict(c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    src = source()
    out = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_sets_{_dt.date.today().isoformat()}.json")
    report = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {"sets": {}}
    for name in a.only:
        rows = BUILDERS[name](src)
        write_rows(name, rows)
        back = read_rows(name)
        assert len(back) == len(rows)
        report["sets"][name] = {"counts": counts(rows), "licence": SETS[RAW_OF.get(name, name)]["licence"]}
        if name == "shadowlink":
            report["sets"][name]["unmapped_titles"] = sum(1 for r in rows if not r["extra"]["mapped"])
        print(f"{name:20s} {json.dumps(report['sets'][name]['counts'])}", flush=True)
    report["built"] = _dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
