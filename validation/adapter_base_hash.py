"""adapter_base_hash -- order 0048: write an adapter's base into its json, once the pairing is shown to work.

The serving guard (``cubbyllm.emit.seats_450m.check_base``) refuses an adapter on any export other than the one its json
names (``base_sha256``: the export's ``model.safetensors``). This tool records it for a known pairing: it loads the
export and the adapter on grilly2, emits one program for a fixed slotted question, and only when that program binds the
slot as the seed and a relation as ``HOP1`` writes ``base_sha256`` and ``base_name`` into the adapter's json (the json
as it was kept beside it, ``<kind>_lora.json.pre_hash``). A pairing that writes no program is refused, nothing written.

    uv run --project <grilly2> python validation/adapter_base_hash.py --export <base export dir> --adapter <adapter dir>

Wired: STANDALONE (a one-off tool).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__wiring__ = "STANDALONE"

QUESTION = "What is the capital of [E1: France]?"
WORKS = re.compile(r'bind\s+frame\s*,\s*SEED\s*,\s*"\$E1"\s*;.*?bind\s+frame\s*,\s*HOP1\s*,\s*"[^"]+"\s*;', re.S)


def works(program: str) -> bool:
    return bool(WORKS.search(program or ""))


def record(meta_path: str, base_sha256: str, base_name: str) -> dict:
    meta = json.load(open(meta_path, encoding="utf-8"))
    if not os.path.exists(meta_path + ".pre_hash"):
        shutil.copyfile(meta_path, meta_path + ".pre_hash")
    meta.update(base_sha256=base_sha256, base_name=base_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def main() -> None:
    from cubbyllm.emit.seats_450m import Cubby450mEmitter, Cubby450mTrunk
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default=None, help="the base's name to record (default: the export folder's parent)")
    a = ap.parse_args()
    trunk = Cubby450mTrunk(a.export, os.path.join(a.export, "tokenizer.json"), {"programs": a.adapter})
    meta_path, _ = trunk._adapter_files("programs")
    prog = Cubby450mEmitter(trunk).emit(QUESTION, max_new_tokens=120)
    print(prog)
    if not works(prog):
        raise SystemExit(f"{os.path.basename(a.adapter)} on {a.export}: no program for {QUESTION!r}; nothing recorded")
    name = a.name or os.path.basename(os.path.dirname(os.path.normpath(a.export)))
    meta = record(meta_path, trunk.base_hash(), name)
    print(f"recorded base {name} ({meta['base_sha256'][:12]}) in {meta_path}")


if __name__ == "__main__":
    main()
