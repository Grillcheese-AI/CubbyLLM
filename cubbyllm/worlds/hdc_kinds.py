"""cubbyllm/worlds/hdc_kinds.py -- which relations each kind's members carry in the hdc world (order 0036, item 3), built
once from the world's own facts, in place of the scan the relation tier ran per question.

The namesake rule's third guard (``hdc_namesakes.hop_pick``) asks: does any item of this kind carry this relation
anywhere in the world? Order 0033 answered it by reading up to 2,000 of the kind's members per question (cached),
which can miss a relation only a few of a large kind carry, and cost up to a second the first time. This table answers
it completely and at once: ``kind_rels(kind, rel, n)``, the number of items of that kind (``instance of``) that carry
that relation, over every fact.

It lives next to the world's file (``<world>.kinds.sqlite``), built from it:

    python -m cubbyllm.worlds.hdc_kinds --db <hdc_world.sqlite>        (once; seconds: 7 s for Wikidata5M's world)

``HdcWorld`` reads it when it is there and was built from this state of the world (``for_world``); else it scans as
before.

Order 0037 reads more from the same counts (``Table.carries``, ``Table.read``): whether a kind CARRIES a relation, as a
share against the base of the world's typed items (LIFT, NMIN: "any member" let a town carry a capital), and an item's
kinds read from its own relations when it records none (SPREAD keeps generic relations, "country", out of it). The
base is the typed items', not all items': most people in this snapshot record no kind, so against all items even
"human" did not hold a citizenship twice as often as the average.

Wired: WIRED (``hdc_store.HdcWorld`` reads it through ``Namesakes.applies``).
"""
from __future__ import annotations

import argparse
import collections
import os
import sqlite3
import sys
import time

__wiring__ = "WIRED"

INSTANCE = "instance of"


def path_for(db_path: str) -> str:
    root, _ext = os.path.splitext(str(db_path))
    return root + ".kinds.sqlite"


def build(db_path: str, out: str | None = None) -> dict:
    """The table from every fact, items read in id order (a sort, never every item in memory at once)."""
    t0 = time.time()
    out = out or path_for(db_path)
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    pairs, last, kinds, rels, items = collections.Counter(), None, set(), set(), 0

    def flush():
        for k in kinds:
            for r in rels:
                pairs[(k, r)] += 1

    for q, rel, okey in src.execute("SELECT sq, rel, okey FROM facts ORDER BY sq"):
        if q != last:
            flush()
            last, kinds, rels, items = q, set(), set(), items + 1
        rels.add(rel)
        if rel == INSTANCE:
            kinds.add(okey)
    flush()
    if os.path.exists(out):
        os.replace(out, out + ".old")                    # moved aside, never deleted
    db = sqlite3.connect(out)
    db.executescript("CREATE TABLE kind_rels(kind TEXT, rel TEXT, n INTEGER, PRIMARY KEY (kind, rel)); "
                     "CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);")
    db.executemany("INSERT INTO kind_rels VALUES (?,?,?)", ((k, r, n) for (k, r), n in pairs.items()))
    stamp = {"items": items, "pairs": len(pairs), "world_bytes": os.path.getsize(db_path), "built_s": round(time.time() - t0, 1)}
    db.executemany("INSERT INTO meta VALUES (?,?)", ((k, str(v)) for k, v in stamp.items()))
    db.commit()
    db.close()
    return {**stamp, "out": out}


LIFT = 2.0               # order 0037: a kind carries a relation when its members hold it this many times as often as
NMIN = 3                 #   the world's typed items do, from at least NMIN members (a capital: most sovereign states,
SPREAD = 50              #   few towns). A relation more than SPREAD kinds carry says only that an item is somewhere.


class Table:
    """``applies(kind, rel)`` from the built table (``kind`` normalized as the world's okey is); and for order 0037,
    ``carries`` (a share against the typed items' base, not "any member") and ``read`` (an item's kinds read from its
    own relations when it records none). The table is small (some 20,000 pairs): it is read once into memory."""

    def __init__(self, path: str) -> None:
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self._n: dict | None = None

    def applies(self, kind: str, rel: str) -> bool:
        return self.db.execute("SELECT 1 FROM kind_rels WHERE kind = ? AND rel = ? LIMIT 1", (kind, rel)).fetchone() is not None

    def _load(self) -> None:
        self._n = {(k, r): n for k, r, n in self.db.execute("SELECT kind, rel, n FROM kind_rels")}
        self._size = {k: n for (k, r), n in self._n.items() if r == INSTANCE}
        typed, held = sum(self._size.values()) or 1, collections.Counter()
        for (k, r), n in self._n.items():
            held[r] += n
        self._base = {r: n / typed for r, n in held.items()}        # (an item of two kinds counts twice: near enough)
        self._by_rel = collections.defaultdict(set)
        for (k, r) in self._n:
            if r != INSTANCE and self._carries(k, r):
                self._by_rel[r].add(k)

    def _carries(self, kind: str, rel: str) -> bool:
        n, size = self._n.get((kind, rel), 0), self._size.get(kind, 0)
        return n >= NMIN and size > 0 and (n / size) >= LIFT * self._base.get(rel, 1.0)

    def carries(self, kind: str, rel: str) -> bool:
        """Do this kind's members hold `rel` well over the typed items' base (LIFT times, NMIN members at least)?"""
        if self._n is None:
            self._load()
        return self._carries(kind, rel)

    def kinds_carrying(self, rel: str) -> set:
        if self._n is None:
            self._load()
        return set(self._by_rel.get(rel, ()))

    def read(self, rels) -> set:
        """The kinds an item that records none is of, read from its own relations: those its specific relations (each
        carried by SPREAD kinds or fewer) all go with. A citizenship and an occupation read as a person; "country" and
        "located in" alone read as nothing (they say only that it is somewhere)."""
        return self.read_how(rels)[0]

    def read_how(self, rels) -> tuple[set, str]:
        """(kinds, how) for `read`: "read" when the specific relations go with kinds together; "read widely" when they
        go with none together, and the kinds any of them goes with are taken, so the item still competes (an ask back
        costs less than a wrong pick), but nothing says which of them it is (order 0037's audit: a band read so
        was named "a film"); "none" without a specific relation."""
        if self._n is None:
            self._load()
        spec = sorted(r for r in set(rels) if r != INSTANCE and 0 < len(self._by_rel.get(r, ())) <= SPREAD)
        if not spec:
            return set(), "none"
        out = set(self._by_rel[spec[0]])
        for r in spec[1:]:
            out &= self._by_rel[r]
        return (out, "read") if out else (set().union(*(self._by_rel[r] for r in spec)), "read widely")


def for_world(db_path: str) -> Table | None:
    """The table built beside this world, or None: not built, or built from another state of the world (its stamp's
    size is not the world's), so a rebuilt world never reads a stale table; the scan answers until it is rebuilt."""
    path = path_for(db_path)
    if not os.path.exists(path):
        return None
    table = Table(path)
    stamp = dict(table.db.execute("SELECT k, v FROM meta"))
    if stamp.get("world_bytes") != str(os.path.getsize(db_path)):
        print(f"[hdc_kinds] {path} was built from another state of the world: scanning (rebuild it)", flush=True)
        return None
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=os.environ.get("CUBBY_HDC_WORLD"), help="the hdc world's file")
    a = ap.parse_args()
    if not a.db:
        from .hdc_store import DEFAULT
        a.db = str(DEFAULT)
    print(build(a.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
