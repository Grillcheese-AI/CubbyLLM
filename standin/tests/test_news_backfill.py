"""Order 0024: the poller waits a host's Crawl-delay between its requests; Global Voices' history pages enter the
archive under the live feed's id, once (a repeated item, from a neighbouring page or the live poll, is not kept
twice), by published year, a page read once and only when it has settled; the pairs file lists a translation and its
source when both are archived.
Run: python -m pytest standin/tests/test_news_backfill.py
"""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import news_archive as na  # noqa: E402
import news_backfill as nb  # noqa: E402
from robots_rules import crawl_delay  # noqa: E402


def _page(items):
    body = "".join(f"""<item><guid>{g}</guid><title>Story {g}</title><link>{link}</link>
<pubDate>Wed, 21 Oct {year} 10:00:00 +0000</pubDate><description>S.</description>
<content:encoded><![CDATA[<p>Text {g}.</p><div class='gv-rss-footer'>{src}</div>]]></content:encoded></item>"""
                   for g, link, year, src in items)
    return ('<?xml version="1.0"?><rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
            f"<channel>{body}</channel></rss>").encode()


def test_crawl_delay_is_read_and_honoured(tmp_path):
    assert crawl_delay("User-agent: *\nCrawl-delay: 10\n", "CubbyLLM/0.1") == 10.0
    assert crawl_delay("User-agent: *\nDisallow: /x\n", "CubbyLLM/0.1") == 0.0
    waits = []
    pages = {"https://gv.example/robots.txt": b"User-agent: *\nCrawl-delay: 10\n", "https://gv.example/feed": _page([])}
    a = na.Archive(tmp_path, fetch=lambda u, timeout=20: (200, pages[u]) if u in pages else (404, b""),
                   sleep=waits.append)
    a.polite("https://gv.example/feed")
    a.polite("https://gv.example/feed")
    assert len(waits) == 2 and all(9.0 < w <= 10.0 for w in waits)
    a.polite("https://other.example/feed")                     # another host, no delay asked: no wait
    assert len(waits) == 2


def test_backfill_pages_enter_once_by_year_and_pair(tmp_path):
    root, pages = tmp_path / "arch", tmp_path / "pages"
    (pages / "en").mkdir(parents=True)
    (pages / "fr").mkdir()
    src = "<span class='source-link'><a href='https://gv.example/en/1'>original</a></span>"
    (pages / "en" / "page_00001.xml.gz").write_bytes(gzip.compress(_page([("e1", "https://gv.example/en/1", 2026, ""),
                                                                         ("e2", "https://gv.example/en/2", 2025, "")])))
    (pages / "en" / "page_00002.xml.gz").write_bytes(gzip.compress(_page([("e2", "https://gv.example/en/2", 2025, ""),
                                                                         ("e3", "https://gv.example/en/3", 2009, "")])))
    (pages / "fr" / "page_00001.xml.gz").write_bytes(gzip.compress(_page([("f1", "https://gv.example/fr/1", 2026, src)])))
    fresh = pages / "en" / "page_00003.xml.gz"                 # still being written: left for the next poll
    fresh.write_bytes(gzip.compress(_page([("e4", "https://gv.example/en/4", 2026, "")])))
    old = time.time() - 600
    for p in list((pages / "en").glob("*.gz"))[:2] + list((pages / "fr").glob("*.gz")):
        os.utime(p, (old, old))
    os.utime(fresh, None)
    m = tmp_path / "feeds.json"
    m.write_text(json.dumps({"feeds": [{"id": "globalvoices-main", "url": "u", "lang": "en", "kind": "news"},
                                       {"id": "globalvoices-fr", "url": "v", "lang": "fr", "kind": "news"}]}), encoding="utf-8")
    row = nb.ingest_all(na.Archive(root), pages, m)
    en, fr = row["detail"]
    assert (en["pages"], en["new"], en["repeated"]) == (2, 3, 1) and (fr["pages"], fr["new"]) == (1, 1)
    years = sorted(p.name for p in (root / "backfill" / "globalvoices-main").glob("*.jsonl"))
    assert years == ["2009.jsonl", "2025.jsonl", "2026.jsonl"]
    again = nb.ingest_all(na.Archive(root), pages, m)          # a fresh process: keys and pages read from disk
    assert again["new"] == 0 and again["detail"][0]["pages"] == 0
    assert nb.pairs(root) == 1
    (line,) = (root / "globalvoices_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {"translation": "https://gv.example/fr/1", "source": "https://gv.example/en/1"}
