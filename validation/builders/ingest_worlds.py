"""ingest_worlds -- the command line of the worlds ingestion (order 0015's tool; split out of standin/worlds_ingest.py
by order 0031, so the library hands nothing to validation/): everything in the datasets into worlds, by the manifest.

    python validation/builders/ingest_worlds.py [--root <worlds dir>] [--only <dataset substring>] [--frames] [--release]

``--frames`` hands order 0013's bulk reader (``validation/books_bulk_read.run``) to ``worlds_ingest.run``. Writes
``validation/logs/worlds_ingest_<date>.json`` (``worlds_ingest.public``: no paths, titles or text).

Wired: STANDALONE (a build tool).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import worlds_ingest as wi  # noqa: E402

__wiring__ = "STANDALONE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.environ.get("CUBBY_WORLDS"), required=not os.environ.get("CUBBY_WORLDS"),
                    help="the worlds folder (CUBBY_WORLDS)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--log", default=str(wi.LOG))
    a = ap.parse_args()
    bulk = None
    if a.frames:
        import books_bulk_read
        bulk = books_bulk_read.run
    rows = wi.run(a.root, a.only, a.frames, a.release, lambda m: print(m, flush=True), bulk=bulk)
    pathlib.Path(a.log).write_text(json.dumps(wi.public(rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
