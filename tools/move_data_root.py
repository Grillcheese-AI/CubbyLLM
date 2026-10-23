"""tools/move_data_root.py -- move the packages' data root to a new folder (order 0033): the caches, the worlds' files,
the keys and tokens go as they are, so nothing is fetched or built again.

Each entry of the current root (``cubbyllm.local_paths.data_dir()``: ``$CUBBY_DATA``, else ``standin/data/out``) is
moved into the new folder. An entry that already exists there is left where it is and listed: nothing is overwritten,
nothing is deleted. A dry run unless ``--write``. With ``--remember``, ``CUBBY_DATA`` is written into the git-ignored
``paths.local.json``, so the next run reads the new root; otherwise set ``CUBBY_DATA`` (or pass the server
``--data-root``).

    python tools/move_data_root.py --to <folder> [--write] [--remember]

Wired: STANDALONE (a maintenance command).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cubbyllm.local_paths import FILE, data_dir  # noqa: E402

__wiring__ = "STANDALONE"


def plan(src: Path, dst: Path) -> tuple[list[Path], list[Path]]:
    """(entries to move, entries already at the destination: left in place)."""
    entries = sorted(src.iterdir()) if src.is_dir() else []
    return [e for e in entries if not (dst / e.name).exists()], [e for e in entries if (dst / e.name).exists()]


def move(src: Path, dst: Path) -> tuple[list[Path], list[Path]]:
    dst.mkdir(parents=True, exist_ok=True)
    go, kept = plan(src, dst)
    for e in go:
        shutil.move(str(e), str(dst / e.name))
    return go, kept


def remember(dst: Path) -> None:
    cur = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {}
    cur["CUBBY_DATA"] = str(dst)
    FILE.write_text(json.dumps(cur, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--to", required=True, help="the new data root")
    ap.add_argument("--write", action="store_true", help="move (default: list what would move)")
    ap.add_argument("--remember", action="store_true", help="also set CUBBY_DATA in paths.local.json (git-ignored)")
    a = ap.parse_args()
    src, dst = data_dir(), Path(a.to).resolve()
    if src.resolve() == dst:
        print(f"the data root is already {dst}")
        return 0
    go, kept = move(src, dst) if a.write else plan(src, dst)
    for e in go:
        print(("moved " if a.write else "would move ") + e.name)
    for e in kept:
        print(f"left in place (already at the destination): {e.name}")
    if a.write and a.remember:
        remember(dst)
        print(f"CUBBY_DATA set in {FILE.name}")
    elif a.write:
        print(f"set CUBBY_DATA={dst} (or pass the server --data-root) so the packages read the new root")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
