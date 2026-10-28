"""stores -- which local store holds an item (order 0046, R0: "item absent from every local store"):

- **Wikidata5M:** the item has a name in the local name index (Wikidata5M's 4.8M items and the hdc world's labels);
- **DBpedia's titles:** its English Wikipedia title is one of the 4.6M entities of the local DBpedia corpus
  (BEIR's dbpedia-entity, ``<dbpedia:Title>`` ids);
- **EN/FR Wikipedia's titles:** its English or French title is an article of the local routed Wikipedia
  (``wikipedia_dedup``: 74,567 articles; the 14 GB ``wiki_full`` text carries no titles and no links, so it cannot
  answer this).

The titles are read once into ``stores.sqlite`` off-repo. An item's titles come from Wikidata's sitelinks
(``mentions.Gold``).

Wired: STANDALONE (order 0046's harness).
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3

from retrieval_sets.common import data_root

__wiring__ = "STANDALONE"

DBPEDIA = r"$CUBBY_DATASETS/full-dbpedia/corpus/corpus-00000-of-00001.parquet"
WIKI_DEDUP = r"$CUBBY_DATASETS/_pipeline/wikipedia_dedup"


def _expand(p: str) -> str:
    from cubbyllm.local_paths import expand
    return expand(p)


def build(out: str | None = None) -> dict:
    import pyarrow.parquet as pq
    out = out or os.path.join(data_root(), "stores.sqlite")
    db = sqlite3.connect(out)
    db.executescript("DROP TABLE IF EXISTS dbpedia; DROP TABLE IF EXISTS wiki;"
                     "CREATE TABLE dbpedia(title TEXT PRIMARY KEY) WITHOUT ROWID;"
                     "CREATE TABLE wiki(title TEXT, lang TEXT, PRIMARY KEY(title, lang)) WITHOUT ROWID;")
    pf = pq.ParquetFile(_expand(DBPEDIA))
    n_db = 0
    for g in range(pf.num_row_groups):
        ids = pf.read_row_group(g, columns=["_id"]).column(0).to_pylist()
        rows = [(i[len("<dbpedia:"):-1].replace("_", " "),) for i in ids if i.startswith("<dbpedia:")]
        db.executemany("INSERT OR IGNORE INTO dbpedia VALUES (?)", rows)
        n_db += len(rows)
    n_wiki = 0
    for f in sorted(glob.glob(os.path.join(_expand(WIKI_DEDUP), "*.jsonl"))):
        rows = []
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append((r.get("title") or "", (r.get("lang") or "").split(".")[-1]))
        db.executemany("INSERT OR IGNORE INTO wiki VALUES (?,?)", rows)
        n_wiki += len(rows)
    db.commit()
    counts = {"dbpedia": db.execute("SELECT COUNT(*) FROM dbpedia").fetchone()[0],
              "wiki_en": db.execute("SELECT COUNT(*) FROM wiki WHERE lang='en'").fetchone()[0],
              "wiki_fr": db.execute("SELECT COUNT(*) FROM wiki WHERE lang='fr'").fetchone()[0]}
    db.close()
    return counts


class Stores:
    def __init__(self, index, gold, path: str | None = None) -> None:
        self.index, self.gold = index, gold
        p = path or os.path.join(data_root(), "stores.sqlite")
        if not os.path.exists(p):
            build(p)
        self.db = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)

    def held(self, qid: str) -> dict:
        """{"wikidata5m", "dbpedia", "wikipedia"}: which local store holds the item."""
        en, fr = self.gold.title(qid, "enwiki"), self.gold.title(qid, "frwiki")
        dbp = bool(en) and self.db.execute("SELECT 1 FROM dbpedia WHERE title = ?", (en,)).fetchone() is not None
        wk = any(t and self.db.execute("SELECT 1 FROM wiki WHERE title = ? AND lang = ?", (t, lg)).fetchone()
                 for t, lg in ((en, "en"), (fr, "fr")))
        return {"wikidata5m": self.index.has(qid), "dbpedia": dbp, "wikipedia": bool(wk)}

    def absent(self, qid: str) -> bool:
        return not any(self.held(qid).values())


__all__ = ["Stores", "build"]
