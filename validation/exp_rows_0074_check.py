"""Order 0074, step 1: `long_rows_0074` read back from its files and checked, before anything is trained on it.

0073's check (`exp_rows_0073_check.check`: the evaluator only, every count but the counted ones must be 0), with
0067's and 0073's rows as the earlier sets no instruction may come from, and one more count: a row whose number of
parts is not its set's (train and dev 2 to 8, test 2 to 4, test_long 5 to 8, test_longer 9 to 12). The counts of
parts by set are reported beside it.

  python validation/exp_rows_0074_check.py [--set $CUBBY_DATASETS/long_rows_0074] [--out validation/logs/<name>.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation"), str(ROOT / "validation" / "builders")]
import exp_rows_0073_check as CK  # noqa: E402

from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"

PARTS = {"train": (2, 8), "dev": (2, 8), "test": (2, 4), "test_long": (5, 8), "test_longer": (9, 12)}
OLD = ("long_rows_0067", "long_rows_0073")


def check(ds: Path, olds: list[Path]) -> dict:
    got = CK.check(ds, olds, "an earlier set")
    rows = CK._read(ds / "rows.jsonl") + CK._read(ds / "pair_rows.jsonl")
    lo_hi = {r["id"]: PARTS.get(r["split"], (0, -1)) for r in rows}
    got["must be 0"]["a row whose number of parts is not its set's"] = sum(
        not lo_hi[r["id"]][0] <= r["n_parts"] <= lo_hi[r["id"]][1] for r in rows)
    got["parts by set"] = {s: dict(sorted(Counter(str(r["n_parts"]) for r in rows if r["split"] == s).items(),
                                          key=lambda kv: int(kv[0]))) for s in PARTS}
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    got = dict({"order": "0074", "what": "long_rows_0074 read back and checked with the evaluator (no VM)"},
               **check(Path(a.set or ds / "long_rows_0074"), [ds / name for name in OLD]))
    print(json.dumps(got, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if any(got["must be 0"].values()):
        raise SystemExit("the set fails its check and is not to be used")


if __name__ == "__main__":
    main()
