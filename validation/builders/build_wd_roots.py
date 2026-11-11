"""Order 0066: the class-roots table of the hot Wikidata store, from `wdstore subclasses`' edges.

Resolves each root of `cubbyllm.worlds.wd_store_roots.ROOTS` by its label in the store (an exact English or `mul`
label of an item that has a subclass-of edge), writes them beside the raw edges (`roots.json`, for the report), and
builds `<store dir>.roots.sqlite` (or --out). The store and its binary come from `CUBBY_WD_STORE`, `CUBBY_WDSTORE_EXE`.

    python validation/builders/build_wd_roots.py --raw <subclasses' out dir> [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__wiring__ = "STANDALONE"


def main() -> int:
    from cubby_serve.board.store_client import StoreClient
    from cubbyllm.local_paths import value
    from cubbyllm.worlds import wd_store_roots as R
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    store = value("CUBBY_WD_STORE")
    ask = StoreClient(value("CUBBY_WDSTORE_EXE"), store).ask
    classes = set()
    for line in open(os.path.join(a.raw, "subclass.tsv"), encoding="utf-8"):
        c, p = line.rstrip("\n").split("\t")
        classes.add(c), classes.add(p)
    roots = R.resolve(ask, classes)
    brief = ask({"op": "brief", "ids": sorted({q for ids in roots.values() for q in ids})}) or {}
    shown = {n: [(q, (brief.get(q) or {}).get("label"), (brief.get(q) or {}).get("description")) for q in ids]
             for n, ids in roots.items()}
    json.dump({"roots": roots, "shown": shown}, open(os.path.join(a.raw, "roots.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    missing = [n for n, ids in roots.items() if not ids]
    if missing:
        print("no class found for:", missing)
    out = a.out or os.path.normpath(store) + ".roots.sqlite"
    print(json.dumps(shown, indent=1, ensure_ascii=False))
    print(json.dumps(R.build(a.raw, roots, out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
