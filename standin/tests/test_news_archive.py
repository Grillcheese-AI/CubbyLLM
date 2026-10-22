"""Order 0024's archive on synthetic feeds (no network): robots read by group, a refusal addressed to us skips the
feed, a new AI-agent block holds it, an item is kept once (a re-poll adds 0), summaries are cut, RSS and Atom both
read, held feeds are never fetched, and the stats count duplicates.
Run: python -m pytest standin/tests/test_news_archive.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import news_archive as na  # noqa: E402
from robots_rules import ai_named, disallowed_all, refuses  # noqa: E402

GROUPED = """User-agent: GPTBot
User-agent: ClaudeBot
User-agent: CCBot
Disallow: /

User-agent: *
Disallow: /recherche
"""

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><guid>a-1</guid><title>Le conseil adopte le budget</title><link>https://x.example/a1</link>
<pubDate>Wed, 21 Oct 2026 15:59:00 -0400</pubDate><description>&lt;p&gt;Court &amp; clair&lt;/p&gt;</description></item>
<item><title>Sans guid</title><link>https://x.example/a2</link><description>%s</description></item>
</channel></rss>""" % (b"mot " * 300)

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>gc</title>
<entry><id>urn:gc:1</id><title>Minister announces funding</title><link href="https://gc.example/1"/>
<published>2026-10-22T10:00:00-04:00</published><summary>Funding for roads.</summary></entry></feed>"""


def test_robots_group_shares_the_rule():
    assert {"gptbot", "claudebot", "ccbot"} <= disallowed_all(GROUPED)
    assert "*" not in disallowed_all(GROUPED)
    assert ai_named(GROUPED) == ["ccbot", "claudebot", "gptbot"]
    assert not refuses(GROUPED, "CubbyLLM/0.1", "https://x.example/rss")
    assert refuses(GROUPED, "CubbyLLM/0.1", "https://x.example/recherche?q=1")
    assert refuses("User-agent: CubbyLLM\nDisallow: /\n", "CubbyLLM/0.1", "https://x.example/rss")


def test_parse_rss_and_atom():
    rss = na.parse(RSS)
    assert [o["title"] for o in rss] == ["Le conseil adopte le budget", "Sans guid"]
    assert rss[0]["id"] == "a-1" and rss[0]["published"] == "2026-10-21" and rss[0]["summary"] == "Court & clair"
    assert rss[1]["id"] == "https://x.example/a2" and rss[1]["published"] is None
    assert len(rss[1]["summary"]) <= na.SUMMARY_MAX + 4 and rss[1]["summary"].endswith("...")
    atom = na.parse(ATOM)
    assert atom == [{"id": "urn:gc:1", "link": "https://gc.example/1", "title": "Minister announces funding",
                     "summary": "Funding for roads.", "published": "2026-10-22"}]


def _fetcher(pages: dict, calls: list):
    def fetch(url, timeout=20):
        calls.append(url)
        return (200, pages[url]) if url in pages else (404, b"")
    return fetch


def _manifest(tmp_path, feeds):
    p = tmp_path / "feeds.json"
    p.write_text(json.dumps({"feeds": feeds}), encoding="utf-8")
    return p


def test_poll_keeps_each_item_once_and_never_fetches_held(tmp_path):
    calls = []
    pages = {"https://x.example/rss": RSS, "https://gc.example/atom": ATOM,
             "https://x.example/robots.txt": GROUPED.encode()}
    m = _manifest(tmp_path, [
        {"id": "x", "url": "https://x.example/rss", "lang": "fr", "status": "active", "every_h": 0,
         "ai_agents_named": ["ccbot", "claudebot", "gptbot"]},
        {"id": "gc", "url": "https://gc.example/atom", "lang": "en", "status": "active", "every_h": 0},
        {"id": "held", "url": "https://held.example/rss", "status": "held"}])
    root = tmp_path / "arch"
    rows = na.Archive(root, fetch=_fetcher(pages, calls)).poll_all(m)
    assert [r["new"] for r in rows] == [2, 1]
    assert not any("held.example" in c for c in calls)
    again = na.Archive(root, fetch=_fetcher(pages, calls)).poll_all(m)     # a fresh process: keys read from disk
    assert [r["new"] for r in again] == [0, 0]
    s = na.stats(root)
    assert s["items"] == 3 and s["duplicates"] == 0 and s["bytes"] > 0
    assert sum(sum(d.values()) for d in s["per_feed_per_day"].values()) == 3
    polls = (root / "_polls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(polls) == 4 and not (root / "_poll.lock").exists()


def test_a_slow_feed_waits_its_interval(tmp_path):
    calls = []
    pages = {"https://gc.example/atom": ATOM}
    m = _manifest(tmp_path, [{"id": "gc", "url": "https://gc.example/atom", "status": "active", "every_h": 6}])
    a = na.Archive(tmp_path / "a", fetch=_fetcher(pages, calls))
    assert [r["feed"] for r in a.poll_all(m)] == ["gc"]
    assert a.poll_all(m) == [] and calls.count("https://gc.example/atom") == 1


def test_dates_in_the_publishers_languages():
    assert na.date_of("Friday, October 16, 2026 - 19:18") == "2026-10-16"
    assert na.date_of("Viernes, Agosto 7, 2026 - 14:45") == "2026-08-07"
    assert na.date_of("Quarta, Maio 13, 2026 - 10:10") == "2026-05-13"
    assert na.date_of("jeudi, août 6, 2026") == "2026-08-06"
    assert na.date_of("Someday, Foo 6, 2026") is None and na.date_of("") is None


def test_refusal_and_new_block_skip_the_feed(tmp_path):
    calls = []
    feed = {"id": "x", "url": "https://x.example/rss", "status": "active", "ai_agents_named": ["gptbot"]}
    pages = {"https://x.example/rss": RSS, "https://x.example/robots.txt": GROUPED.encode()}
    a = na.Archive(tmp_path / "a", fetch=_fetcher(pages, calls))
    row = a.poll(feed)
    assert row["skipped"] == "new_ai_block: ccbot,claudebot" and "https://x.example/rss" not in calls
    pages["https://x.example/robots.txt"] = b"User-agent: *\nDisallow: /\n"
    b = na.Archive(tmp_path / "b", fetch=_fetcher(pages, calls))
    assert b.poll(feed)["skipped"] == "refused" and "https://x.example/rss" not in calls


def test_day_file_is_the_day_first_seen(tmp_path):
    pages = {"https://gc.example/atom": ATOM}
    a = na.Archive(tmp_path, fetch=_fetcher(pages, []))
    a.poll({"id": "gc", "url": "https://gc.example/atom", "status": "active"},
           now=dt.datetime(2026, 10, 23, 1, 0, tzinfo=dt.timezone.utc))
    (line,) = (tmp_path / "2026-10-23.jsonl").read_text(encoding="utf-8").splitlines()
    o = json.loads(line)
    assert o["published"] == "2026-10-22" and o["first_seen"].startswith("2026-10-23") and o["feed"] == "gc"


def test_manifest_feeds_are_well_formed():
    fs = na.feeds()
    assert len({f["id"] for f in fs}) == len(fs)
    for f in fs:
        assert f["status"] in ("active", "held") and f["reason"] and f["licence"] in ("open-government-canada", "publisher")
    assert {f["id"] for f in fs if f["status"] == "active"} >= {"gc-news-releases", "gc-communiques"}


def test_gzip_body_is_read(monkeypatch):
    import gzip

    class R:
        status, headers = 200, {"Content-Encoding": "gzip"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return gzip.compress(ATOM)
    monkeypatch.setattr(na.urllib.request, "urlopen", lambda rq, timeout=20: R())
    code, body = na._get("https://un.example/rss")
    assert code == 200 and na.parse(body)[0]["title"] == "Minister announces funding"


def test_an_item_carries_the_topics_of_its_feeds(tmp_path):
    pages = {"https://gi.example/t/1": ATOM, "https://gi.example/t/2": ATOM}
    m = _manifest(tmp_path, [{"id": f"t{i}", "url": f"https://gi.example/t/{i}", "status": "active", "topic": name}
                             for i, name in ((1, "Aid"), (2, "Food"))])
    a = na.Archive(tmp_path / "a", fetch=_fetcher(pages, []))
    a.poll_all(m)
    rows = [json.loads(l) for f in (tmp_path / "a").glob("????-??-??.jsonl") for l in f.read_text(encoding="utf-8").splitlines()]
    assert na.topics(rows) == {"https://gc.example/1": ["Aid", "Food"]}
