"""cubbyllm/worlds/wd_store_roots.py -- the class-roots table of the hot Wikidata store (order 0066): which roots a class
reaches by "subclass of", for the four answer checks (a place, a person or a group or an organisation, a work, a role).

`wdstore subclasses <store> <out dir>` writes every best-rank "subclass of" (P279) edge of the hot entities. This
builds the closure under each root downward from the root, as `wdstore cues` builds "fictional entity"'s:

    python -m cubbyllm.worlds.wd_store_roots --raw <subclasses' out dir> --roots <resolved roots .json>
        --out <store dir>.roots.sqlite

The roots are allow-lists of (id, label) (ROOTS: Cowork's picks from the store's own candidates, 0066's addendum);
`resolve` checks each id against the store -- a class (an edge in subclass.tsv) carrying that label -- and stops the
build on any that fails, so nothing is taken from the order's text on trust. The ids go in the table's meta and the
order's report. A class reaches a root
when it is the root or any chain of subclass-of edges leads to it; a class may reach several. Written: ``roots(qid,
mask)`` for every class that reaches one, ``meta`` (the store's snapshot, the root ids, counts). `for_store` gives the
table only when it was built from the state of the store being served.

Wired: WIRED (`wd_store_world.WdStoreWorld`, read when one of 0066's flags is on; `cubby_serve/wd_world.py` finds it).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import time

__wiring__ = "WIRED"

ROOTS = {   # each root: its allow-list of (class id, a label the class carries), Cowork's picks in 0066's addendum
    "place": (("Q27096213", "geographic entity"),),
    "person": (("Q5", "human"),),
    "group": (("Q16334295", "group of humans"),),
    "organisation": (("Q43229", "organization"),),
    "character": (("Q95074", "fictional character"), ("Q21070598", "figure that may or may not be fictional")),
    "work": (("Q15621286", "intellectual work"), ("Q732577", "publication")),
    "role": (("Q4164871", "position"), ("Q216353", "title"), ("Q355567", "noble title"), ("Q56019", "military rank"),
             ("Q16138157", "title")),
}
BIT = {name: 1 << i for i, name in enumerate(ROOTS)}
WHO = {"person", "group", "organisation", "character"}   # what "who" asks for (0066, check 1a)


def _tsv(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def resolve(ask, classes: set, roots: dict = ROOTS) -> dict:
    """{root name: [ids]}: each allow-listed id, checked to be a class of this store (it has a subclass-of edge) that
    carries its label (an English or `mul` label or alias, normalized). Any id that fails stops the build: nothing is
    taken on trust from the order's text."""
    from cubbyllm.reasoning.planner import normalize
    out, bad = {}, []
    for name, pairs in roots.items():
        for q, label in pairs:
            e = ask({"op": "get", "id": q}) or {}
            names = {normalize(n[2]) for n in e.get("names", []) if n[0] in ("en", "mul") and n[1] in (0, 1)}
            if q not in classes or normalize(label) not in names:
                bad.append((name, q, label, q in classes, sorted(names)[:4]))
            out.setdefault(name, []).append(q)
    if bad:
        raise SystemExit(f"roots not verified in this store: {bad}")
    return out


def closure(edges, roots: dict) -> dict:
    """{class: mask} for every class under a root (the root itself included), by subclass-of edges (child, parent)."""
    sub = collections.defaultdict(list)
    for child, parent in edges:
        sub[parent].append(child)
    mask: dict = {}
    for name, ids in roots.items():
        seen, todo = set(ids), collections.deque(ids)
        while todo:
            c = todo.popleft()
            for d in sub.get(c, ()):
                if d not in seen:
                    seen.add(d)
                    todo.append(d)
        for c in seen:
            mask[c] = mask.get(c, 0) | BIT[name]
    return mask


def build(raw: str, roots: dict, out: str) -> dict:
    t0 = time.time()
    meta = dict(_tsv(os.path.join(raw, "meta.tsv")))
    mask = closure(((c, p) for c, p in _tsv(os.path.join(raw, "subclass.tsv"))), roots)
    if os.path.exists(out):
        os.replace(out, out + ".old")                    # moved aside, never deleted
    db = sqlite3.connect(out)
    db.executescript("CREATE TABLE roots(qid TEXT PRIMARY KEY, mask INTEGER); CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);")
    db.executemany("INSERT INTO roots VALUES (?,?)", sorted(mask.items()))
    counts = {name: sum(1 for m in mask.values() if m & BIT[name]) for name in ROOTS}
    info = {"snapshot": meta.get("snapshot", ""), "roots": json.dumps(roots), "classes": json.dumps(counts),
            "edges": meta.get("edges", ""), "seconds": str(round(time.time() - t0))}
    db.executemany("INSERT INTO meta VALUES (?,?)", info.items())
    db.commit()
    db.close()
    return info


class Roots:
    def __init__(self, path: str) -> None:
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.meta = dict(self.db.execute("SELECT k, v FROM meta"))
        self.ids = json.loads(self.meta.get("roots", "{}"))
        self._cache: dict = {}

    def of(self, qid: str) -> set:
        """The roots a class reaches ({"place", "work", ...}; empty for none)."""
        got = self._cache.get(qid)
        if got is None:
            row = self.db.execute("SELECT mask FROM roots WHERE qid = ?", (qid,)).fetchone()
            got = self._cache[qid] = {n for n, b in BIT.items() if row and row[0] & b}
        return got


def for_store(path: str | None, snapshot: str | None) -> Roots | None:
    if not path or not os.path.exists(path):
        return None
    r = Roots(path)
    return r if snapshot and r.meta.get("snapshot") == snapshot else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--roots", required=True, help="the resolved roots, a JSON {name: [ids]} (validation/builders)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.raw, json.load(open(a.roots, encoding="utf-8")), a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
