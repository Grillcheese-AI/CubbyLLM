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
    """The clarify menu offers choices without taking a position on importance -- a count over
    the headlines fetched, which is what was written about, never what mattered.

    A headline's FIRST word is dropped, because it is capitalised for starting the sentence and
    not for being a name: "Have", "Calls" and "Watch" all reached the live menu that way
    (2026-09-14). The trade is deliberate and it does cost something -- "Russia hits Ukrainian
    train" no longer offers Russia -- but a name that matters recurs mid-headline, and a stop
    list is an arms race against English that the sentence's own shape wins outright."""
    src = _src(tmp_path)
    topics = dict(src.topics("2026", min_n=1))
    assert topics.get("Ukrainian") == 1 and topics.get("England") == 1
    assert "Russia" not in topics                             # first word of its headline
    assert "An" not in topics and "Deep" not in topics


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


ROBOTS_BLOCKS = """User-agent: *
Disallow: /search/

User-agent: GPTBot
Disallow: /

User-agent: anthropic-ai
Disallow: /

User-agent: Bingbot
Disallow:
"""
ROBOTS_OPEN = """User-agent: *
Disallow: /private/
"""


DISTINCT = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
 <item><title>A headline no other feed printed</title>
       <pubDate>Sun, 13 Sep 2026 20:00:00 -0400</pubDate><link>http://w/1</link></item>
</channel></rss>"""


def _policy_src(tmp_path, feeds, robots):
    """A NewsSource whose robots.txt lookups are answered from a dict instead of the network."""
    src = NewsSource(feeds=feeds, cache_dir=tmp_path)
    src._robots_for = lambda host: robots.get(host, None)    # type: ignore[assignment]
    return src


def test_the_publishers_stated_position_is_read_from_their_own_front_door(tmp_path):
    """A feed subdomain is delivery infrastructure; the position is stated at the front door.
    Measured 2026-09-14: feeds.bbci.co.uk names no AI crawler while www.bbc.co.uk names fifteen,
    and rss.nytimes.com serves no robots.txt at all while www.nytimes.com names fourteen.
    Reading only the side door is a way of not hearing the answer."""
    src = _policy_src(tmp_path, {"nyt": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"},
                      {"rss.nytimes.com": None, "www.nytimes.com": ["anthropic-ai", "gptbot"]})
    assert src.ai_optout("nyt") == ["anthropic-ai", "gptbot"]
    # the BBC's delivery domain is not a subdomain of its front door at all -- written down,
    # never guessed, and a host not in the table is simply not claimed
    src2 = _policy_src(tmp_path, {"bbc": "https://feeds.bbci.co.uk/news/rss.xml"},
                       {"feeds.bbci.co.uk": [], "www.bbc.co.uk": ["anthropic-ai", "claude-web"]})
    assert src2.ai_optout("bbc") == ["anthropic-ai", "claude-web"]


def test_a_host_that_names_no_ai_crawler_is_distinguished_from_one_with_no_robots_at_all(tmp_path):
    """Three different answers, and collapsing them would be the lie: [] is "they published a
    policy and it names none of these", None is "there is nothing to read". No robots.txt is
    not a yes and it is not a no."""
    src = _policy_src(tmp_path, {"open": "https://globalnews.ca/feed/"}, {"globalnews.ca": []})
    assert src.ai_optout("open") == []
    src2 = _policy_src(tmp_path, {"quiet": "https://rss.politico.com/x.xml"}, {})
    assert src2.ai_optout("quiet") is None


def test_reading_the_policy_never_silently_changes_what_the_source_returns(tmp_path):
    """The position is RECORDED and surfaced; what to do about it is a decision a person makes,
    not one this file makes quietly. A source that dropped feeds on its own would be deciding a
    publishing-rights question by side effect."""
    src = _policy_src(tmp_path, {"fixture": "https://blocked.example/feed"},
                      {"blocked.example": ["anthropic-ai"]})
    src._cache_path("fixture").parent.mkdir(parents=True, exist_ok=True)
    src._cache_path("fixture").write_bytes(FEED)
    src.offline = True
    assert src.ai_optout("fixture") == ["anthropic-ai"]
    assert len(src.facts("2026-09-13")) == 1                 # unchanged: the caller decides


def test_serving_is_allowed_for_everyone_and_training_is_not(tmp_path):
    """Nick's decision, 2026-09-14. Fetching a headline and speaking it once with attribution is
    what a feed reader does and RSS exists for; baking it into model weights is what the block
    lists are about, and attribution does not undo that. So `facts()` serves every feed and
    `training_facts()` serves only publishers who named no AI crawler."""
    feeds = {"open": "https://openhost.example/feed", "blocked": "https://blockedhost.example/feed"}
    src = _policy_src(tmp_path, feeds, {"openhost.example": [], "blockedhost.example": ["anthropic-ai"]})
    for fid in feeds:
        src._cache_path(fid).parent.mkdir(parents=True, exist_ok=True)
    src._cache_path("open").write_bytes(FEED)
    src._cache_path("blocked").write_bytes(DISTINCT)         # NOT the same words as FEED: the
    #                                                          dedupe would collapse them, and
    #                                                          that is a different test
    src.offline = True

    assert src.licence_of("open") == NewsSource.OPEN and src.trainable("open")
    assert src.licence_of("blocked") == NewsSource.SERVE_ONLY and not src.trainable("blocked")

    served = src.facts("2026-09-13")
    assert len(served) == 2                                  # serving sees both publishers
    licences = set(src.licence.values())
    assert licences == {NewsSource.OPEN, NewsSource.SERVE_ONLY}   # ... and every fact carries which

    trained = src.training_facts("2026-09-13")
    assert len(trained) == 1
    fact = normalize(f"{trained[0].obj} is the {trained[0].rel} of {trained[0].subj}")
    assert src.provenance[fact] == "open"


def test_silence_is_not_consent(tmp_path):
    """A host with no robots.txt at all is UNKNOWN, and UNKNOWN does not train. The cost of
    being wrong is asymmetric: a headline wrongly left out of a dataset costs a headline, and
    one wrongly put in cannot be taken back out of the weights."""
    src = _policy_src(tmp_path, {"quiet": "https://nopolicy.example/feed"}, {})
    assert src.licence_of("quiet") == NewsSource.UNKNOWN
    assert not src.trainable("quiet")
