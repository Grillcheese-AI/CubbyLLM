"""Order 0073, step 6: the file the Colab read is given. The held-out rows' ids, splits and instructions, and
nothing else: no target, no description, no reply and no verdict leaves the PC, so an answer cannot be scored (or
looked at against its gold) anywhere but at home.

  python validation/builders/read_rows_0073.py [--rows <rows.jsonl>] --out <folder>

Writes ``read_rows.jsonl`` (dev, test and test_long, in the rows' own order: what `exp_view_split_torch.py` reads)
and ``read_rows.manifest.json`` (counts and SHA-256; the notebook checks the file against the committed copy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

__wiring__ = "STANDALONE"

SPLITS = ("dev", "test", "test_long")
KEYS = ("id", "split", "instruction")


def build(rows_p: Path, out: Path) -> dict:
    raw = rows_p.read_bytes()
    rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines()]
    kept = [{k: r[k] for k in KEYS} for s in SPLITS for r in rows if r["split"] == s]
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "read_rows.jsonl").write_bytes(body)
    man = {"order": "0073", "what": "the held-out rows' ids, splits and instructions, for the read on Colab",
           "keys": list(KEYS), "rows": len(kept), "by_split": dict(Counter(r["split"] for r in kept)),
           "sha256": hashlib.sha256(body).hexdigest(), "rows_sha256": hashlib.sha256(raw).hexdigest()}
    (out / "read_rows.manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from cubbyllm import local_paths
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0073" / "rows.jsonl")
    print(json.dumps(build(rows_p, Path(a.out)), indent=1))


if __name__ == "__main__":
    main()
