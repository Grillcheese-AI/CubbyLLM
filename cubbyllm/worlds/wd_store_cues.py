"""cubbyllm/worlds/wd_store_cues.py -- the cue table and the fictional classes for the hot Wikidata store's world (order
0055, step 3, option (a), Nick 2026-11-10: "build the cue table and set aside fictional items").

`wdstore cues <store> <raw dir>` counts, per (word, class), the hot items of that class carrying a value with that
word (an item value's label, English or `mul` text, a plain string; words cut as `hdc_cues.words`), and lists every
class under "fictional entity" (Q14897293) by subclass of. This turns the classes into the words the world's kinds
use (their labels, normalized, from `wdstore kinds`' labels.tsv) and writes one file beside the store:
- ``word_kinds(word, kind, n)``, read by `hdc_cues.Cues` as Wikidata5M's is: a cue word rules out a namesake none of
  whose kinds ever goes with it ("Nobel" goes with people, not ships), when CUE_MIN items or more use the word;
- ``fictional(qid)``: the fictional classes, by id; an item of one is set aside among a name's items when a real item
  shares the name (`wd_store_world`), so "the capital of France" is not answered for a Star Trek location;
- ``meta``: the store's snapshot (a table from another state is not read).

    python -m cubbyllm.worlds.wd_store_cues --raw <wdstore cues' out dir> --labels <wdstore kinds' labels.tsv>
        --out <store dir>.cues.sqlite

Wired: WIRED (`cubbyllm/worlds/wd_store_world.py`, read with ``CB_WD_KINDS=1``).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

from cubbyllm.reasoning.planner import normalize

from .hdc_cues import Cues

__wiring__ = "WIRED"


def _tsv(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def build(raw: str, labels_path: str, out: str) -> dict:
    """Streamed: rows go into the table as read, and two classes with one label are summed by SQL at the end (the
    full store has some 36 million pairs)."""
    t0 = time.time()
    labels = {k: normalize(v) for k, v in _tsv(labels_path)}
    meta = {k: v for k, v in _tsv(os.path.join(raw, "meta.tsv"))}
    if os.path.exists(out):
        os.replace(out, out + ".old")                    # moved aside, never deleted
    db = sqlite3.connect(out)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE pairs(word TEXT, kind TEXT, n INTEGER); "
                     "CREATE TABLE fictional(qid TEXT PRIMARY KEY); CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);")
    buf, dropped = [], 0
    for word, kind, n in _tsv(os.path.join(raw, "word_kinds.tsv")):
        lab = labels.get(kind)
        if not lab:
            dropped += 1                                 # a class with no English or mul label: no kind word
            continue
        buf.append((word, lab, int(n)))
        if len(buf) >= 500_000:
            db.executemany("INSERT INTO pairs VALUES (?,?,?)", buf)
            buf.clear()
    db.executemany("INSERT INTO pairs VALUES (?,?,?)", buf)
    db.executescript("CREATE TABLE word_kinds AS SELECT word, kind, SUM(n) AS n FROM pairs GROUP BY word, kind; "
                     "DROP TABLE pairs; CREATE INDEX w_k ON word_kinds(word);")
    fictional = [q for (q,) in _tsv(os.path.join(raw, "fictional.tsv"))]
    db.executemany("INSERT INTO fictional VALUES (?)", ((q,) for q in fictional))
    n_pairs, n_words = db.execute("SELECT COUNT(*), COUNT(DISTINCT word) FROM word_kinds").fetchone()
    stamp = {"snapshot": meta.get("snapshot", ""), "items": meta.get("items"), "typed": meta.get("typed"),
             "words": n_words, "pairs": n_pairs, "pairs_unlabelled": dropped, "fictional_classes": len(fictional),
             "built_s": round(time.time() - t0, 1)}
    db.executemany("INSERT INTO meta VALUES (?,?)", ((k, str(v)) for k, v in stamp.items()))
    db.commit()
    db.execute("VACUUM")
    db.close()
    return {**stamp, "out": out}


def for_store(path: str | None, snapshot: str | None) -> tuple[Cues | None, frozenset]:
    """(the cue table, the fictional classes' ids) when built from the store's own state; else (None, frozenset())."""
    if not path or not os.path.exists(path):
        return None, frozenset()
    cues = Cues(path)
    stamp = dict(cues.db.execute("SELECT k, v FROM meta"))
    if not snapshot or stamp.get("snapshot") != snapshot:
        print(f"[wd_store_cues] {path} was built from another state of the store: not read (rebuild it)", flush=True)
        return None, frozenset()
    return cues, frozenset(q for (q,) in cues.db.execute("SELECT qid FROM fictional"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="`wdstore cues`' out dir (word_kinds.tsv, fictional.tsv, meta.tsv)")
    ap.add_argument("--labels", required=True, help="`wdstore kinds`' labels.tsv (the classes' labels)")
    ap.add_argument("--out", required=True, help="the table to write: <store dir>.cues.sqlite")
    a = ap.parse_args()
    print(build(a.raw, a.labels, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
