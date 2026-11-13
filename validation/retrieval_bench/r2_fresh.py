"""r2_fresh -- order 0069, D's fresh read: the entity-linking rows no arm has read, drawn and frozen before any arm runs.

Mintaka EN, WebQSP_EL EN and ShadowLink EN, test split: the rows R2's test read did not take
(``retrieval_R2_test_2026-11-05_test.json``), 300 a set drawn by seed 2 (``common.sample``: sorted ids, then the seed).
The ids go to ``validation/retrieval_sets/r2_fresh_el_seed2.json``, its SHA-256 in ``frozen.json`` (``lexicon.freeze``);
``r2_choosing --el-fresh`` reads exactly these rows and refuses a file that moved since it was frozen.

    python validation/retrieval_bench/r2_fresh.py

Wired: STANDALONE (order 0069's bench).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

SEED, N, KEY = 2, 300, "r2_fresh_el_seed2"
SETS = ("mintaka", "webqsp_el", "shadowlink")
READ_LOG = "retrieval_R2_test_2026-11-05_test.json"
PATH = os.path.join(os.path.dirname(HERE), "retrieval_sets", f"{KEY}.json")


def ids() -> dict[str, list[str]]:
    """The frozen fresh ids, {set: [row id]}; refuses a file that moved since it was frozen."""
    from retrieval_sets.lexicon import frozen, sha256
    if sha256(PATH) != frozen()[KEY]["sha256"]:
        raise SystemExit(f"{KEY}.json moved since it was frozen")
    return json.load(open(PATH, encoding="utf-8"))["sets"]


def main() -> None:
    from retrieval_sets.common import read_rows, sample
    from retrieval_sets.lexicon import freeze
    if os.path.exists(PATH):
        raise SystemExit(f"{PATH} exists: a fresh sample is drawn once")
    log = json.load(open(os.path.join(os.path.dirname(HERE), "logs", READ_LOG), encoding="utf-8"))
    read = {r["id"].split("#")[0] for r in log["rows"]}
    out, counts = {}, {}
    for name in SETS:
        rows = read_rows(name, split="test", lang="en")
        left = [r for r in rows if r["id"] not in read]
        out[name] = sorted(r["id"] for r in sample(left, N, SEED))
        counts[name] = {"test_rows": len(rows), "read_by_r2_test": len(rows) - len(left), "drawn": len(out[name])}
    rep = {"what": "order 0069 D: the fresh entity-linking rows (EN, test), none read by R2's test", "made": _dt.date.today().isoformat(),
           "seed": SEED, "per_set": N, "excluded": READ_LOG, "counts": counts, "sets": out}
    with open(PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    h = freeze(KEY, PATH, f"order 0069's fresh EL rows: {N} a set (EN, test) by seed {SEED}, none of R2's test read; {counts}")
    print(json.dumps(counts), h)


if __name__ == "__main__":
    main()
