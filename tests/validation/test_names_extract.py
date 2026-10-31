"""Order 0046, R1b's builders on canned inputs (no network): ``validation/retrieval_sets/names_extract.py`` reads only
the id and name columns and writes id / name / kind / lang; ``wiki_redirects.py`` parses MySQL dump tuples (escapes,
NULL, other namespaces, interwiki targets) into article-space (from, to, fragment)."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from retrieval_sets import names_extract as N  # noqa: E402
from retrieval_sets import wiki_redirects as R  # noqa: E402

PAGE = (b"INSERT INTO `page` VALUES (10,0,'JFK',1,0,0.1,'20260101','20260101',5,40,'wikitext',NULL),"
        b"(11,0,'John_F._Kennedy',0,0,0.2,'20260101','20260101',6,90000,'wikitext',NULL),"
        b"(12,1,'JFK',1,0,0.3,'20260101','20260101',7,30,'wikitext',NULL),"
        b"(13,0,'L\\'\xc3\x89tranger',1,0,0.4,'20260101','20260101',8,30,'wikitext',NULL),"
        b"(14,0,'Wiktionary_link',1,0,0.5,'20260101','20260101',9,30,'wikitext',NULL);\n")
REDIRECT = (b"INSERT INTO `redirect` VALUES (10,0,'John_F._Kennedy','',''),(12,0,'John_F._Kennedy','',''),"
            b"(13,0,'L\\'\xc3\x89tranger_(roman)',NULL,'Intrigue'),(14,0,'jfk','wikt','');\n")


def test_redirect_tuples_become_article_space_pairs():
    pages = R.redirect_pages([b"-- comment\n", PAGE])
    assert pages == {10: "JFK", 13: "L'Étranger", 14: "Wiktionary link"}     # ns 1 and non-redirects left out
    got = list(R.pairs([REDIRECT], pages))
    assert got == [("JFK", "John F. Kennedy", ""), ("L'Étranger", "L'Étranger (roman)", "Intrigue")]


def test_unescape_reads_mysql_escapes():
    assert R.unescape(b"a\\\\b\\nc_d") == "a\\b\nc d"


def _write(tmp, name, cols):
    p = tmp / name
    pq.write_table(pa.table(cols), p, row_group_size=2)
    return {"path": str(p), "size": p.stat().st_size, "repo": "permutans/x", "folder": "en"}


def test_the_extract_reads_two_columns_and_labels_each_name(tmp_path):
    lab = _write(tmp_path, "lab.parquet", {"id": ["Q1", "Q2", "Q3"], "language": ["en"] * 3, "value": ["Earth", "Mars", "Moon"]})
    lnk = _write(tmp_path, "lnk.parquet", {"id": ["Q1"], "site": ["frwiki"], "title": ["Terre"], "badges": [["b"]]})
    fl = [dict(lab, col="value", kind="label", lang="en"), dict(lnk, col="title", kind="title", lang="fr")]
    opener = lambda p: open(p, "rb")                                       # noqa: E731
    p = N.plan(opener, fl)
    assert p["names"] == 4 and p["files"][0]["row_groups"] == 2 and p["projected_extract_bytes"] > 0
    assert 0 < p["files"][0]["bytes_read"] < p["files"][0]["size"]          # the language column is never read
    m = N.extract(opener, fl, str(tmp_path / "out"))
    t = pq.read_table(tmp_path / "out" / f"names_{N.REV}" / m["files"][1]["out"])
    assert t.to_pylist() == [{"id": "Q1", "name": "Terre", "kind": "title", "lang": "fr"}]
    again = N.extract(lambda p: (_ for _ in ()).throw(AssertionError("re-read")), fl, str(tmp_path / "out"))
    assert again["files"] == m["files"]                                     # resumable: done files are not read again
