"""cubbyllm/worlds/wd_store_kinds.py -- the kinds table for the hot Wikidata store's world (order 0055, step 3, option
(a), Nick 2026-11-10): which relations each kind's members carry, over every hot item, so the namesake rule of order
0037 (`hdc_relation.decide`) decides a shared name in the store as it does in Wikidata5M.

`wdstore kinds <store> <raw dir>` counts, per (class, property), the hot items of that class (best-rank `instance of`)
that hold the property (best rank). This turns its ids into the words the world's facts use: a kind is its class's
label, normalized (`planner.normalize`), a relation its property's label, normalized ("instance of" for P31, so a kind's
size is its row on "instance of"). Two classes with one label count together, as Wikidata5M's labels did. The
result is the same table as `hdc_kinds` builds (``kind_rels(kind, rel, n)``, ``meta``), read by the same
`hdc_kinds.Table` (``carries``, ``read_how``, ``applies``), stamped with the store's snapshot:

    python -m cubbyllm.worlds.wd_store_kinds --raw <wdstore kinds' out dir> --out <store dir>.kinds.sqlite

`for_store` gives the table only when it was built from the state of the store being served (its snapshot); else
None, and the world holds every shared name as before.

Wired: WIRED (`cubbyllm/worlds/wd_store_world.py`, read with ``CB_WD_KINDS=1``).
"""
from __future__ import annotations

import argparse
import collections
import os
import sqlite3
import sys
import time

from cubbyllm.reasoning.planner import normalize

from .hdc_kinds import Table

__wiring__ = "WIRED"

INSTANCE = "instance of"


def _tsv(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def build(raw: str, out: str) -> dict:
    t0 = time.time()
    labels = {k: v for k, v in _tsv(os.path.join(raw, "labels.tsv"))}
    meta = {k: v for k, v in _tsv(os.path.join(raw, "meta.tsv"))}
    pairs, dropped = collections.Counter(), 0
    for kind, pid, n in _tsv(os.path.join(raw, "kind_rels.tsv")):
        k, p = labels.get(kind), labels.get(pid)
        if not k or not p:
            dropped += 1                                 # a class or property with no English or mul label: not a word
            continue
        pairs[(normalize(k), INSTANCE if pid == "P31" else normalize(p))] += int(n)
    if os.path.exists(out):
        os.replace(out, out + ".old")                    # moved aside, never deleted
    db = sqlite3.connect(out)
    db.executescript("CREATE TABLE kind_rels(kind TEXT, rel TEXT, n INTEGER, PRIMARY KEY (kind, rel)); "
                     "CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);")
    db.executemany("INSERT INTO kind_rels VALUES (?,?,?)", ((k, r, n) for (k, r), n in pairs.items()))
    stamp = {"snapshot": meta.get("snapshot", ""), "items": meta.get("items"), "typed": meta.get("typed"),
             "pairs": len(pairs), "pairs_unlabelled": dropped, "built_s": round(time.time() - t0, 1)}
    db.executemany("INSERT INTO meta VALUES (?,?)", ((k, str(v)) for k, v in stamp.items()))
    db.commit()
    db.close()
    return {**stamp, "out": out}


def for_store(path: str | None, snapshot: str | None) -> Table | None:
    """The table, when it is there and was built from the store's own state; else None."""
    if not path or not os.path.exists(path):
        return None
    table = Table(path)
    stamp = dict(table.db.execute("SELECT k, v FROM meta"))
    if not snapshot or stamp.get("snapshot") != snapshot:
        print(f"[wd_store_kinds] {path} was built from another state of the store: not read (rebuild it)", flush=True)
        return None
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="`wdstore kinds`' out dir (kind_rels.tsv, labels.tsv, meta.tsv)")
    ap.add_argument("--out", required=True, help="the table to write: <store dir>.kinds.sqlite")
    a = ap.parse_args()
    print(build(a.raw, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
