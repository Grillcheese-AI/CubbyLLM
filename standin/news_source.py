"""
Wired: WIRED (stand-in serve stack; implements `cubbyllm.reasoning.learn.Source`).

`NewsSource`: RSS feeds -> dated facts, for the one question the wiki world genuinely cannot
answer. Measured 2026-09-14: the wiki world holds 0 facts whose subject is a year, and its
76,897 `timeline event` facts are dangling leaves with no dates -- so "what happened in 2026?"
was unanswerable and the refusal was correct. A news feed is the missing source.

THE ONE DESIGN DECISION, and it is the whole point:

    A headline is not a fact about the world. It is a fact about what a publisher reported.

So a feed item becomes `"<headline>" is the headline of <date>` with the publisher in the
fact's PROVENANCE, where this architecture already carries it, and the date in `times`, where
the store already reads it. Nothing here asserts that a headline is TRUE -- only that it was
published, which is the part this source can actually witness. The gate admits it on that
basis, the walk can reach it, and the loop can say "on 14 September 2026, the BBC's headlines
included ..." without ever having claimed the content. A source that cannot lie about the world
is a source that cannot make the loop lie about the world.

That also means a headline never becomes a fact ABOUT an entity here. "Russia hits Ukrainian
train" does not enter the store as a fact about Russia; it enters as a fact about a date. An
entity fact needs the gate's provenance and a source that states it structurally -- that is
Wikidata's job, not a newsroom's.

Feeds are cached on disk with a TTL so a rerun inside the window is offline and byte-identical,
the way `WikidataSource` caches its API calls.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

__wiring__ = "WIRED"

from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402

CACHE = pathlib.Path(__file__).resolve().parent / "data" / "out" / "news_cache"
UA = "cubbyllm-standin/0.1 (research; licensing@grillcheese.ai)"
TTL_S = 900                    # a feed is re-fetched at most every 15 minutes
HEADLINE = "headline"          # the relation. The publisher is provenance, not vocabulary.
MAX_PER_FEED = 60

# Nick's list, 2026-09-14, as measured from a real machine. Each is a publisher, not an
# authority: `provenance` says which one, and two feeds reporting the same thing are two facts,
# never one corroborated one -- agreement among newsrooms is not evidence, it is syndication.
#
# CBC is deliberately absent: all three of its feed URLs refused us (timeout on
# www.cbc.ca/webfeed, connection closed on rss.cbc.ca), which is a publisher declining to be
# read by a robot. That is their call and it is not worked around.
FEEDS = {
    "bbc-news":     "https://feeds.bbci.co.uk/news/rss.xml",
    "ap-frontpage": "https://rss.app/feeds/v1.1/8M3w8yDxG7rKvuyn.json",
    "nyt-home":     "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "politico":     "https://rss.politico.com/politics-news.xml",
}
# The AP feed above is titled "Iran war" by whoever built it and is in fact AP's front page
# (Nick, 2026-09-14; the contents agree -- it carries the AP Top 25 and a Georgia election
# story). It is keyed by what it CONTAINS, not by what it is called, and see the note on topic
# labels below. It replaces feedx.net/rss/ap.xml, which was nine days stale when measured.

# AP publishes one feed per desk. This is the menu for Nick's "ask which topic" -- a list the
# PUBLISHER wrote, not a ranking this loop invented, which is exactly the property that lets it
# offer choices without taking a position on what matters.
# ... and it is empty: every file in that bucket is `<items/>`, 55 bytes (measured 2026-09-14).
# Kept here because the LISTING is the right idea and the right shape; only the content is gone.
AP_BUCKET = "http://associated-press.s3-website-us-east-1.amazonaws.com/"
TOPIC_FEEDS = {t: f"{AP_BUCKET}{t}.xml" for t in (
    "world-news", "us-news", "politics", "business", "technology", "science",
    "health", "climate-and-environment", "entertainment", "sports", "religion",
    "lifestyle", "travel", "oddities",
)}

# A TOPIC FEED IS A CURATED SELECTION, NOT A FILTER. Nick's rss.app "Iran war" feed carries
# "A flaw in Georgia's election systems" and "Texas stakes its claim for No. 1 in AP Top 25"
# (2026-09-14, live). So the topic is the FEED OWNER'S claim about what belongs together, and it
# is recorded as provenance -- never as a fact that the item IS about that topic. Taking the
# label at face value would be adopting somebody else's editorial judgement as a fact, which is
# the same error the neutral-prior competition refused for sitelinks and PageRank.
_DATE_ASK = re.compile(r"^\s*(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?\s*$")
_TAG = re.compile(r"<[^>]+>")
# a headline's first word is capitalised because it STARTS the headline, and English capitalises
# these anyway: counting them would make "How" the news of the day (2026-09-14, first run).
_TOPIC_STOP = frozenset("the a an this that and but for new how why what when where who from with "
                        "after before inside amid over under here there his her its their".split())


def _iso(raw: str) -> str | None:
    """An RSS date as YYYY-MM-DD. RFC-822 (`Sat, 13 Sep 2026 18:04:13 -0400`) is what most
    feeds emit; a few write ISO already. A date this cannot read is a date this does not
    invent -- the item is dropped rather than dated by guess."""
    raw = (raw or "").strip()
    if not raw:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return m.group(0)
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except Exception:                                        # noqa: BLE001
        return None


def _text(el, *names) -> str:
    for n in names:
        found = el.find(n)
        if found is not None and (found.text or "").strip():
            return " ".join(_TAG.sub(" ", found.text).split())
    return ""


class NewsSource:
    """`facts("2026-09-14")` or `facts("2026")` -> that day's / that year's headlines as Triples.
    Anything that is not a date gets nothing: this source knows WHEN, not WHO."""

    name = "news-rss"

    def __init__(self, feeds: dict[str, str] | None = None, cache_dir: pathlib.Path | None = CACHE,
                 ttl_s: int = TTL_S, timeout_s: int = 20, offline: bool = False,
                 topic: str | None = None) -> None:
        self.topic = topic if topic in TOPIC_FEEDS else None
        self.feeds = dict(feeds) if feeds else ({f"ap-{self.topic}": TOPIC_FEEDS[self.topic]}
                                                if self.topic else dict(FEEDS))
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.ttl_s, self.timeout_s, self.offline = int(ttl_s), int(timeout_s), bool(offline)
        self.calls = 0
        self.last: dict = {}
        self.times: dict[str, dict] = {}
        self.provenance: dict[str, str] = {}       # fact -> the feed that published it
        self.errors: dict[str, str] = {}

    # -- fetching ------------------------------------------------------------------------
    def _cache_path(self, feed_id: str) -> pathlib.Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{feed_id}-{hashlib.sha256(self.feeds[feed_id].encode()).hexdigest()[:12]}.xml"

    def _raw(self, feed_id: str) -> bytes | None:
        p = self._cache_path(feed_id)
        if p and p.exists() and (self.offline or time.time() - p.stat().st_mtime < self.ttl_s):
            return p.read_bytes()
        if self.offline:
            return None
        try:
            req = urllib.request.Request(self.feeds[feed_id], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                body = r.read()
            self.calls += 1
        except Exception as e:                               # noqa: BLE001 -- a feed that is down is not an answer
            self.errors[feed_id] = f"{type(e).__name__}: {str(e)[:100]}"
            return p.read_bytes() if p and p.exists() else None   # a stale cache beats nothing; it is still dated
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
        return body

    def items(self, feed_id: str) -> list[tuple[str, str, str]]:
        """(headline, iso date, link) for one feed. Undated items are dropped: a headline whose
        date this cannot read cannot answer a question about a date. RSS/Atom XML and JSON Feed
        both arrive here -- rss.app serves the latter (2026-09-14, Nick's 'Iran war' feed), and
        which wire format a publisher chose says nothing about the facts, so neither does this."""
        body = self._raw(feed_id)
        if not body:
            return []
        if body.lstrip()[:1] in (b"{", b"["):
            return self._json_items(feed_id, body)
        try:
            root = ET.fromstring(body)
        except ET.ParseError as e:
            self.errors[feed_id] = f"ParseError: {str(e)[:100]}"
            return []
        out = []
        for it in root.iter("item"):
            title = _text(it, "title")
            when = _iso(_text(it, "pubDate", "{http://purl.org/dc/terms/}modified",
                              "{http://purl.org/dc/elements/1.1/}date", "date"))
            if title and when:
                out.append((title, when, _text(it, "link")))
            if len(out) >= MAX_PER_FEED:
                break
        return out

    def _json_items(self, feed_id: str, body: bytes) -> list[tuple[str, str, str]]:
        """JSON Feed 1.1: `items[]` with `title` and `date_published`."""
        try:
            o = json.loads(body)
        except Exception as e:                               # noqa: BLE001
            self.errors[feed_id] = f"JSONError: {str(e)[:100]}"
            return []
        out = []
        for it in (o.get("items") or [])[:MAX_PER_FEED]:
            title = " ".join(_TAG.sub(" ", str(it.get("title") or "")).split())
            when = _iso(str(it.get("date_published") or it.get("date_modified") or ""))
            if title and when:
                out.append((title, when, str(it.get("url") or "")))
        return out

    # -- the Source contract -------------------------------------------------------------
    def facts(self, entity: str, relations: list[str] | None = None, **_kw) -> list[Triple]:
        """`entity` must be a date -- a year, a month or a day. The headlines of every feed
        that covers it become `"<headline>" is the headline of <entity>`, dated in `times` and
        attributed in `provenance`. A non-date gets [] and says so in `last`."""
        self.last = {"entity": entity, "feeds": [], "n": 0, "how": None, "unresolved": False}
        self.times, self.provenance = {}, {}
        m = _DATE_ASK.match(str(entity or ""))
        if not m:
            self.last["unresolved"] = True
            self.last["how"] = "not a date: this source answers WHEN, not WHO"
            return []
        prefix = "-".join(g for g in m.groups() if g)         # 2026 | 2026-09 | 2026-09-14
        out: list[Triple] = []
        for feed_id in self.feeds:
            n_before = len(out)
            for title, when, _link in self.items(feed_id):
                if not when.startswith(prefix):
                    continue
                t = Triple(obj=f'"{title}"', rel=HEADLINE, subj=entity)
                fact = f"{t.obj} is the {t.rel} of {t.subj}"
                if _normalize(fact) in self.times:            # two feeds, the same words: still one fact
                    continue
                self.times[_normalize(fact)] = {"point": when}
                self.provenance[_normalize(fact)] = feed_id
                out.append(t)
            if len(out) > n_before:
                self.last["feeds"].append(feed_id)
        self.last["n"] = len(out)
        self.last["how"] = f"headlines dated {prefix} from {len(self.last['feeds'])} feed(s)"
        return out

    def topics(self, entity: str, min_n: int = 2) -> list[tuple[str, int]]:
        """The recurring capitalised subjects in the headlines for a date, commonest first --
        the menu for Nick's clarify bar when a date ask is too broad to answer in one breath.
        This is a COUNT OVER THE FETCHED HEADLINES, not a ranking of importance: it says what
        was written about, never what mattered."""
        counts: dict[str, int] = {}
        for t in self.facts(entity):
            words = re.findall(r"\b[A-Z][A-Za-z'\-]{2,}(?:\s+[A-Z][A-Za-z'\-]{2,})?", t.obj)
            for w in {w.strip() for w in words}:
                if w.lower() in _TOPIC_STOP:
                    continue
                counts[w] = counts.get(w, 0) + 1
        return sorted(((w, n) for w, n in counts.items() if n >= min_n), key=lambda q: (-q[1], q[0]))
