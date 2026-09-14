"""
The news source, tested offline against a fixture feed -- a test that reaches the network is a
test of the network. The live measurement (which feeds answer, which refuse, how far back they
reach) is in `docs/research/2026-09-14-news-sources.md`, not here.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.planner import normalize  # noqa: E402
from news_source import NewsSource  # noqa: E402

FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
 <item><title><![CDATA[Russia hits Ukrainian train]]></title>
       <pubDate>Sun, 13 Sep 2026 18:04:13 -0400</pubDate><link>http://x/1</link></item>
 <item><title>Deep-fried food banned in England</title>
       <pubDate>Mon, 14 Sep 2026 06:00:00 GMT</pubDate><link>http://x/2</link></item>
 <item><title>An item with no date at all</title><link>http://x/3</link></item>
 <item><title>Something from last year</title>
       <pubDate>Sun, 14 Sep 2025 09:00:00 GMT</pubDate><link>http://x/4</link></item>
</channel></rss>"""

OTHER = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
 <item><title>Russia hits Ukrainian train</title>
       <pubDate>Sun, 13 Sep 2026 19:30:00 -0400</pubDate><link>http://y/1</link></item>
</channel></rss>"""


def _src(tmp_path, bodies=None):
    bodies = bodies or {"fixture": FEED}
    src = NewsSource(feeds={k: f"http://test/{k}" for k in bodies}, cache_dir=tmp_path, ttl_s=10_000)
    for k, b in bodies.items():
        src._cache_path(k).parent.mkdir(parents=True, exist_ok=True)
        src._cache_path(k).write_bytes(b)
    src.offline = True                      # the cache IS the feed: no network in a test
    return src


def test_a_headline_is_a_fact_about_a_date_not_about_the_world(tmp_path):
    """The one design decision. A headline enters the store as something a publisher PUBLISHED
    on a day -- the date is the subject, the publisher is the provenance -- so nothing here
    asserts that the headline is true. A source that cannot lie about the world cannot make
    the loop lie about the world."""
    src = _src(tmp_path)
    out = src.facts("2026-09-13")
    assert [t.rel for t in out] == ["headline"] and [t.subj for t in out] == ["2026-09-13"]
    assert out[0].obj == '"Russia hits Ukrainian train"'
    fact = normalize(f"{out[0].obj} is the {out[0].rel} of {out[0].subj}")
    assert src.times[fact] == {"point": "2026-09-13"}
    assert src.provenance[fact] == "fixture"


def test_an_undated_item_is_dropped_and_a_year_gathers_its_days(tmp_path):
    """An item whose date this cannot read is an item this does not date by guess. And a year
    ask collects every day inside it, each keeping its OWN point -- that is what lets the date
    index answer both "what happened in 2026" and "on 2026-09-14" from one fetch."""
    src = _src(tmp_path)
    assert len(src.facts("2026-09-13")) == 1
    year = src.facts("2026")
    assert len(year) == 2                                   # the two 2026 items, not the 2025 one
    assert {t.subj for t in year} == {"2026"}
    points = {w["point"] for w in src.times.values()}
    assert points == {"2026-09-13", "2026-09-14"}
    assert not [t for t in src.facts("2026") if "no date at all" in t.obj]


def test_two_feeds_printing_the_same_words_are_one_fact_not_a_corroboration(tmp_path):
    """Agreement among newsrooms is syndication, not evidence. The same headline from a second
    feed must not become a second fact that makes the first look confirmed."""
    src = _src(tmp_path, {"fixture": FEED, "other": OTHER})
    out = src.facts("2026-09-13")
    assert len(out) == 1
    fact = normalize(f"{out[0].obj} is the {out[0].rel} of {out[0].subj}")
    assert src.provenance[fact] == "fixture"                # the first to say it keeps the attribution


def test_this_source_answers_when_and_refuses_who(tmp_path):
    """It knows dates. A headline never enters the store as a fact ABOUT an entity -- that needs
    a source that states it structurally, which is Wikidata's job, not a newsroom's."""
    src = _src(tmp_path)
    assert src.facts("bill haslam") == []
    assert src.last["unresolved"] is True and "not a date" in src.last["how"]


def test_the_topic_menu_counts_what_was_written_about_never_what_mattered(tmp_path):
    """The clarify menu has to offer choices without taking a position on importance. A count
    over fetched headlines does that; the stop list keeps the first word of a headline from
    becoming the news ("How" led the first live run)."""
    src = _src(tmp_path)
    topics = dict(src.topics("2026", min_n=1))
    assert topics.get("Russia") == 1 and topics.get("Ukrainian") == 1
    assert "An" not in topics and "Deep" not in topics or topics.get("An") is None


JSONFEED = b"""{"version":"https://jsonfeed.org/version/1.1","title":"Iran war","items":[
 {"id":"1","url":"http://z/1","title":"An Iranian commercial ship is struck near Strait of Hormuz",
  "date_published":"2026-09-13T05:53:31.000Z","authors":[{"name":"AP News"}]},
 {"id":"2","url":"http://z/2","title":"Texas stakes its claim for No. 1 in AP Top 25",
  "date_published":"2026-09-13T10:10:08.000Z","authors":[{"name":"Eric Olson"}]},
 {"id":"3","url":"http://z/3","title":"An item with no date"}]}"""


def test_a_json_feed_reads_the_same_as_an_xml_one(tmp_path):
    """rss.app serves JSON Feed 1.1, not RSS XML (2026-09-14, Nick's 'Iran war' feed). Which
    wire format a publisher chose says nothing about the facts, so neither does this -- same
    Triples, same times, same provenance, and the undated item is still dropped."""
    src = _src(tmp_path, {"rssapp": JSONFEED})
    out = src.facts("2026-09-13")
    assert len(out) == 2                                     # the undated third item is not dated by guess
    assert out[0].obj == '"An Iranian commercial ship is struck near Strait of Hormuz"'
    fact = normalize(f"{out[0].obj} is the {out[0].rel} of {out[0].subj}")
    assert src.times[fact] == {"point": "2026-09-13"} and src.provenance[fact] == "rssapp"


def test_a_topic_feed_is_a_curated_selection_and_the_label_is_not_a_fact(tmp_path):
    """The feed is titled "Iran war" and carries "Texas stakes its claim for No. 1 in AP Top 25"
    -- live, unedited. The topic is the FEED OWNER'S claim about what belongs together, so it
    stays provenance and never becomes a fact that the item is ABOUT that topic. Believing the
    label would be adopting someone else's editorial judgement as truth."""
    src = _src(tmp_path, {"rssapp": JSONFEED})
    out = src.facts("2026-09-13")
    for t in out:
        assert t.subj == "2026-09-13"                        # the DATE is the subject. Never "Iran war".
        assert "iran" not in t.rel.lower()
