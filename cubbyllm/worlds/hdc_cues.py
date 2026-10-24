"""cubbyllm/worlds/hdc_cues.py -- which kinds of things each word of the hdc world's values goes with (order 0037, item 3),
built once from the world, beside the kinds table.

Nick: "a building cannot have a Nobel Prize, only a person". A question's own words narrow a shared name past what its
relation can: "award received" fits films and buildings too, "Nobel" goes only with people (and a few organisations).
This table records, for every content word in the world's values, the kinds of the subjects that carry a value with
that word: their recorded kind, or the kind read from their own facts (``hdc_kinds.Table.read``; the snapshot's Nobel
physicists record none). No word list is kept by hand.

A cue word only rules out: a namesake none of whose kinds ever goes with the word does not compete
(``hdc_relation.decide``). It never ranks, and a word the world never uses rules out nothing.

    python -m cubbyllm.worlds.hdc_cues --db <hdc_world.sqlite>     (once, after the kinds table; about a minute)

Wired: WIRED (``hdc_store.HdcWorld`` reads it for ``hdc_relation.decide``, through ``Namesakes.hop_rows``).
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time

__wiring__ = "WIRED"

INSTANCE = "instance of"
STOP = frozenset("""the a an of and or in on at to for from by with into over under about as is are was were be been his
her its their this that these those what which who whom whose where when why how did does do has have had not no de la le
les du des del della di da von van der den das el los las et und""".split())
BATCH = 200_000


def words(text: str) -> list[str]:
    """The content words of a text, as the world's keys are written (lower case, letters and digits)."""
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3 and not w.isdigit() and w not in STOP]


def path_for(db_path: str) -> str:
    root, _ext = os.path.splitext(str(db_path))
    return root + ".cues.sqlite"


def build(db_path: str, table, out: str | None = None) -> dict:
    """The table from every fact: per item (in id order), its kinds (recorded, else read) against its values' words."""
    t0 = time.time()
    out = out or path_for(db_path)
    if os.path.exists(out):
        os.replace(out, out + ".old")                    # moved aside, never deleted
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db = sqlite3.connect(out)
    db.executescript("CREATE TABLE pairs(word TEXT, kind TEXT); CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);")
    buf, last, kinds, rels, ws, items, read = [], None, set(), set(), set(), 0, 0

    def flush():
        nonlocal read
        ks = kinds or table.read(rels)
        read += bool(ks and not kinds)
        buf.extend((w, k) for w in ws for k in ks)
        if len(buf) >= BATCH:
            db.executemany("INSERT INTO pairs VALUES (?,?)", buf)
            buf.clear()

    for q, rel, okey in src.execute("SELECT sq, rel, okey FROM facts ORDER BY sq"):
        if q != last:
            if last is not None:
                flush()
            last, kinds, rels, ws, items = q, set(), set(), set(), items + 1
        rels.add(rel)
        if rel == INSTANCE:
            kinds.add(okey)
        else:
            ws.update(words(okey))
    if last is not None:
        flush()
    db.executemany("INSERT INTO pairs VALUES (?,?)", buf)
    db.executescript("CREATE TABLE word_kinds AS SELECT word, kind, COUNT(*) AS n FROM pairs GROUP BY word, kind; "
                     "DROP TABLE pairs; CREATE INDEX w_k ON word_kinds(word);")
    n = db.execute("SELECT COUNT(*), COUNT(DISTINCT word) FROM word_kinds").fetchone()
    stamp = {"items": items, "read": read, "pairs": n[0], "words": n[1], "world_bytes": os.path.getsize(db_path),
             "built_s": round(time.time() - t0, 1)}
    db.executemany("INSERT INTO meta VALUES (?,?)", ((k, str(v)) for k, v in stamp.items()))
    db.commit()
    db.execute("VACUUM")
    db.close()
    return {**stamp, "out": out}


class Cues:
    """``kinds(word)``: the kinds a word of the world's values goes with, or None for a word the world never uses."""

    def __init__(self, path: str) -> None:
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self._memo: dict = {}

    def kinds(self, word: str) -> set | None:
        if word not in self._memo:
            got = {k for (k,) in self.db.execute("SELECT kind FROM word_kinds WHERE word = ?", (word,))}
            self._memo[word] = got or None
        return self._memo[word]


def for_world(db_path: str) -> Cues | None:
    """The cue table built beside this world, or None (not built, or from another state of the world)."""
    path = path_for(db_path)
    if not os.path.exists(path):
        return None
    cues = Cues(path)
    stamp = dict(cues.db.execute("SELECT k, v FROM meta"))
    if stamp.get("world_bytes") != str(os.path.getsize(db_path)):
        print(f"[hdc_cues] {path} was built from another state of the world: no cue words (rebuild it)", flush=True)
        return None
    return cues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=os.environ.get("CUBBY_HDC_WORLD"), help="the hdc world's file")
    a = ap.parse_args()
    if not a.db:
        from .hdc_store import DEFAULT
        a.db = str(DEFAULT)
    from .hdc_kinds import for_world as kinds_for
    table = kinds_for(a.db)
    if table is None:
        print("build the kinds table first: python -m cubbyllm.worlds.hdc_kinds", file=sys.stderr)
        return 1
    print(build(a.db, table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
