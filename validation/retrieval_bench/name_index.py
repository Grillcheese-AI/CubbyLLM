"""name_index -- the local name index R1 measures (order 0046; ``docs/RETRIEVAL.md``, "Find"): every name a local store
gives an item, as rows (name, its lowercase, its folded form, item, store), in one SQLite file off-repo
(``$CUBBY_DATA_ROOT/retrieval/name_index.sqlite``):

- **Wikidata5M:** each item's names (``wikidata5m_entity.txt``: the label and aliases, 4.8M items, mostly English);
- **the hdc world's labels:** the English and French labels of the items it holds (2.46M; 1.2M with a French one).

Lookups (``NameIndex``):
- ``exact``: the mention's lowercase equals a name's (case only: accents and punctuation kept);
- ``normalized``: the folded forms are equal (lowercase, accents and punctuation off: ``common.fold``);
- ``near``: names sharing the mention's rarest letter trigrams (an FTS5 trigram table), then ordered by the capsules'
  binary code (``cubbyllm.worlds.capsules.encode``: words and letter trigrams, Hamming distance);
- ``bm25``: FTS5's BM25 over the names' words, the supplement that only fills a short set.
Each returns items with a score where one exists (a distance, a BM25); exact and normalized matches are a set, unordered.

    python validation/retrieval_bench/name_index.py --build [--w5m <wikidata5m_entity.txt>]

Wired: STANDALONE (an experiment's index; the serving path has its own stores).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets.common import data_root, fold, raw_dir  # noqa: E402

__wiring__ = "STANDALONE"

DIM = 256
RARE = 4                                                 # the rarest trigrams asked for when every one is common
MAX_DF = 400_000                                         # a trigram in more names than this is too common to block on
BLOCK = 300                                              # the most names a near lookup reorders by code


def low(s: str) -> str:
    return " ".join((s or "").lower().split())


def path() -> str:
    return os.path.join(data_root(), "name_index.sqlite")


def _rows_w5m(w5m: str):
    with open(w5m, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            q = parts[0]
            for name in dict.fromkeys(parts[1:]):
                if name.strip():
                    yield name.strip(), q, "w5m", ""


def _rows_hdc():
    from cubbyllm.worlds.hdc_store import DEFAULT
    db = sqlite3.connect(f"file:{DEFAULT}?mode=ro", uri=True)
    for q, en, fr in db.execute("SELECT qid, en, fr FROM labels"):
        if en:
            yield en, q, "hdc", "en"
        if fr:
            yield fr, q, "hdc", "fr"


def build(w5m: str, out: str) -> dict:
    t0 = time.time()
    if os.path.exists(out):
        os.replace(out, out + f".old{int(t0)}")                  # moved aside, never deleted
    db = sqlite3.connect(out)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-2000000;"
                     "CREATE TABLE raw(name TEXT, low TEXT, norm TEXT, qid TEXT, src TEXT, lang TEXT);")
    n, batch = 0, []
    for gen in (_rows_w5m(w5m), _rows_hdc()):
        for name, q, src, lang in gen:
            batch.append((name, low(name), fold(name), q, src, lang))
            if len(batch) >= 200_000:
                db.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?)", batch)
                n += len(batch)
                batch = []
    if batch:
        db.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?)", batch)
        n += len(batch)
    db.executescript("""
        CREATE TABLE names(id INTEGER PRIMARY KEY, name TEXT, low TEXT, norm TEXT, qid TEXT, src TEXT);
        INSERT INTO names(name, low, norm, qid, src)
            SELECT MIN(name), low, MIN(norm), qid, GROUP_CONCAT(DISTINCT src) FROM raw WHERE low != '' GROUP BY low, qid;
        DROP TABLE raw;
        CREATE INDEX n_low ON names(low);
        CREATE INDEX n_norm ON names(norm);
        CREATE INDEX n_qid ON names(qid);
        CREATE VIRTUAL TABLE w USING fts5(low, content='names', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
        INSERT INTO w(w) VALUES('rebuild');
        CREATE VIRTUAL TABLE tri USING fts5(norm, content='names', content_rowid='id', tokenize='trigram');
        INSERT INTO tri(tri) VALUES('rebuild');
        CREATE VIRTUAL TABLE tri_v USING fts5vocab('tri', 'row');
    """)
    rows = db.execute("SELECT COUNT(*), COUNT(DISTINCT qid) FROM names").fetchone()
    db.execute("CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT)")
    db.executemany("INSERT INTO meta VALUES (?,?)", [("built", time.strftime("%F %T")), ("raw_rows", str(n)),
                                                     ("names", str(rows[0])), ("items", str(rows[1])), ("w5m", os.path.basename(w5m))])
    db.commit()
    db.close()
    return {"raw_rows": n, "names": rows[0], "items": rows[1], "bytes": os.path.getsize(out), "seconds": round(time.time() - t0)}


class NameIndex:
    """Read-only lookups over the built index."""

    def __init__(self, p: str | None = None) -> None:
        self.db = sqlite3.connect(f"file:{p or path()}?mode=ro", uri=True, check_same_thread=False)
        self._df: dict = {}

    def exact(self, mention: str) -> set:
        return {q for (q,) in self.db.execute("SELECT qid FROM names WHERE low = ?", (low(mention),))}

    def normalized(self, mention: str) -> set:
        k = fold(mention)
        return {q for (q,) in self.db.execute("SELECT qid FROM names WHERE norm = ?", (k,))} if k else set()

    def has(self, qid: str) -> bool:
        return self.db.execute("SELECT 1 FROM names WHERE qid = ? LIMIT 1", (qid,)).fetchone() is not None

    def names(self, qid: str) -> list[str]:
        return [n for (n,) in self.db.execute("SELECT name FROM names WHERE qid = ?", (qid,))]

    def _doc_freq(self, tri: str) -> int:
        if tri not in self._df:
            r = self.db.execute("SELECT doc FROM tri_v WHERE term = ?", (tri,)).fetchone()
            self._df[tri] = r[0] if r else 0
        return self._df[tri]

    def near(self, mention: str, k: int = 100) -> list[tuple[int, str]]:
        """[(distance, qid)] by the capsules' code, nearest first, over the names that share the mention's rarest
        trigrams (each item at its nearest name)."""
        from cubbyllm.worlds.capsules import encode, hamming_numpy
        import numpy as np
        key = fold(mention)
        tris = sorted({key[i:i + 3] for i in range(len(key) - 2)})
        tris = [t for t in tris if self._doc_freq(t) > 0]
        if not tris:
            return []
        # every trigram under the frequency cap (a misspelling's own trigrams are rare, so "the rarest few" would pick
        # exactly the wrong ones); BM25 over the OR ranks names by how many they share
        use = [t for t in tris if self._doc_freq(t) <= MAX_DF] or sorted(tris, key=self._doc_freq)[:RARE]
        q = " OR ".join('"' + t.replace('"', '""') + '"' for t in use)
        rows = self.db.execute("SELECT names.norm, names.qid FROM tri JOIN names ON names.id = tri.rowid "
                               "WHERE tri MATCH ? ORDER BY rank LIMIT ?", (q, BLOCK)).fetchall()
        if not rows:
            return []
        code = encode(key, DIM)
        book = np.stack([encode(n, DIM) for n, _q in rows])
        dist = hamming_numpy(code, book)
        best: dict = {}
        for d, (_n, qq) in zip(dist.tolist(), rows):
            best[qq] = min(best.get(qq, d), d)
        return sorted(((d, qq) for qq, d in best.items()), key=lambda x: (x[0], x[1]))[:k]

    def bm25(self, mention: str, k: int = 100) -> list[tuple[float, str]]:
        """[(score, qid)] by FTS5's BM25 over the names' words (lower is better in SQLite's rank), best first."""
        words = [w for w in fold(mention).split() if w]
        if not words:
            return []
        q = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
        best: dict = {}
        for score, qq in self.db.execute("SELECT bm25(w), names.qid FROM w JOIN names ON names.id = w.rowid "
                                         "WHERE w MATCH ? ORDER BY rank LIMIT ?", (q, 4 * k)):
            best[qq] = min(best.get(qq, score), score)
        return sorted(((s, qq) for qq, s in best.items()), key=lambda x: (x[0], x[1]))[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--w5m", default=os.path.join(raw_dir("wikidata5m"), "wikidata5m_entity.txt"))
    a = ap.parse_args()
    if a.build:
        print(build(a.w5m, path()), flush=True)


if __name__ == "__main__":
    main()
