"""r3g_classes -- order 0070, G2's statistics: per class (``instance of``) of R3o's items, how many of the class's items
in the local Wikidata store carry each property. Statistics, never facts in a store.

The counts are the local store's kinds table (``$CUBBY_WD_STORE.kinds.sqlite``, order 0055: ``wdstore kinds`` over every
hot item of the 2025-12-29 dump, best-rank ``instance of`` and best-rank statements, stamped with the store's snapshot).
That table is keyed by words: a class by its label, a property by its label, both normalized (``planner.normalize``),
two classes with one label counted together. Here the words go back to ids:
  a row's classes   its items' best-rank ``instance of`` values (the items' claims, through the source's cache, as R3o
                    read them), each by its English label (else ``mul``), fetched through the source's cache, dated
  a property        by its English label in the frozen lexicon; a label two properties share is left out (counted)
A class's size is its row on "instance of". Kept: each property carried by at least ``MIN_SHARE`` of a class's items.
Written to ``r3g_classes.json`` beside this file (ids and counts only).

    python validation/retrieval_bench/r3g_classes.py

Wired: STANDALONE (order 0070's bench).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

PATH = os.path.join(HERE, "r3g_classes.json")
MIN_SHARE = 0.05                 # the smallest share on G2's grid
INSTANCE = "instance of"


def load(path: str = PATH) -> dict:
    return json.load(open(path, encoding="utf-8"))


def expected(classes: list[str], table: dict, share: float, n_min: int) -> set | None:
    """The properties at least `share` of the items of any of `classes` carry; None when no class has `n_min` items."""
    out, seen = set(), False
    for c in classes:
        d = table.get(c)
        if not d or (d.get("size") or 0) < n_min:
            continue
        seen = True
        out |= {p for p, n in d["counts"].items() if n / d["size"] >= share}
    return out if seen else None


def build(items: list[str], src, lx: dict, kinds_path: str) -> dict:
    from cubbyllm.reasoning.planner import normalize
    from retrieval_bench.r3g_set import truthy
    from retrieval_sets.common import entities
    claims = entities(src, items, props="claims", langs="en")
    cls_of = {q: truthy((claims.get(q) or {}).get("claims"), "P31") for q in items}
    classes = sorted({c for v in cls_of.values() for c in v}, key=lambda q: int(q[1:]))
    labs = entities(src, classes, props="labels", langs="en|mul")
    by_label: dict[str, list[str]] = {}
    for p, d in lx["label"].items():
        if d.get("en"):
            by_label.setdefault(normalize(d["en"]), []).append(p)
    shared = {k for k, v in by_label.items() if len(v) > 1}
    db = sqlite3.connect(f"file:{kinds_path}?mode=ro", uri=True)
    stamp = dict(db.execute("SELECT k, v FROM meta"))
    table, no_label, no_kind = {}, 0, 0
    for c in classes:
        ls = (labs.get(c) or {}).get("labels") or {}
        lab = (ls.get("en") or ls.get("mul") or {}).get("value")
        if not lab:
            no_label += 1
            continue
        kind = normalize(lab)
        rows = dict(db.execute("SELECT rel, n FROM kind_rels WHERE kind = ?", (kind,)))
        size = rows.get(INSTANCE)
        if not size:
            no_kind += 1
            table[c] = {"label": lab, "size": 0, "counts": {}}
            continue
        counts = {}
        for rel, n in rows.items():
            if rel != INSTANCE and rel in by_label and rel not in shared and n / size >= MIN_SHARE:
                counts[by_label[rel][0]] = n
        table[c] = {"label": lab, "size": size, "counts": dict(sorted(counts.items(), key=lambda x: int(x[0][1:])))}
    return {"what": "order 0070, G2: per class, how many of its items in the local store carry each property (statistics)",
            "made": _dt.date.today().isoformat(), "store_snapshot": stamp.get("snapshot"), "store_items": stamp.get("items"),
            "min_share": MIN_SHARE, "items": len(items), "classes": len(classes),
            "classes_without_label": no_label, "classes_not_in_store": no_kind, "property_labels_shared": len(shared),
            "class_of": {q: cls_of[q] for q in items}, "table": table}


def main() -> None:
    from cubbyllm.local_paths import value
    from retrieval_bench.r3_gold import prepare, rows as load_rows
    from retrieval_sets.common import source
    from retrieval_sets.lexicon import check
    src = source()
    rows = load_rows("dev", 300)
    prepare(rows, src)
    items = sorted({q for r in rows for q in ([m["qid"] for m in r["mentions"] if m.get("span")]
                                               or [m["qid"] for m in r["mentions"]])}, key=lambda q: int(q[1:]))
    store = value("CUBBY_WD_STORE")
    if not store:
        raise SystemExit("CUBBY_WD_STORE is not set")
    rep = build(items, src, check(), store.rstrip("\\/") + ".kinds.sqlite")
    json.dump(rep, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print({k: v for k, v in rep.items() if k not in ("class_of", "table")}, "->", PATH)


if __name__ == "__main__":
    main()
