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
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .feeds import (AP_BUCKET, CACHE, FEEDS, HEADLINE, MAX_PER_FEED, TOPIC_FEEDS, TTL_S, UA, _DATE_ASK,  # noqa: F401
                    _TAG, _TOPIC_STOP, _iso, _no_fetch, _text)
from .policy import NewsPolicy

__wiring__ = "WIRED"

from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402


# The AP feed above is titled "Iran war" by whoever built it and is in fact AP's front page
# (Nick, 2026-09-14; the contents agree -- it carries the AP Top 25 and a Georgia election
# story). It is keyed by what it CONTAINS, not by what it is called, and see the note on topic
# labels below. It replaces feedx.net/rss/ap.xml, which was nine days stale when measured.


# A TOPIC FEED IS A CURATED SELECTION, NOT A FILTER. Nick's rss.app "Iran war" feed carries
# "A flaw in Georgia's election systems" and "Texas stakes its claim for No. 1 in AP Top 25"
# (2026-09-14, live). So the topic is the FEED OWNER'S claim about what belongs together, and it
# is recorded as provenance -- never as a fact that the item IS about that topic. Taking the
# label at face value would be adopting somebody else's editorial judgement as a fact, which is
# the same error the neutral-prior competition refused for sitelinks and PageRank.
# Crawler products that publishers name when they mean "not for building AI systems". A
# directive naming one of these is not addressed to CubbyLLM -- we are none of them, and
# `User-agent: *` permits the feed path at every host measured. But the INTENT is generic and
# unmistakable, and a loop whose whole kill line is "0 wrong answers spoken" should not be
# coy about reading a "no" it understands. So the policy is READ, RECORDED, and surfaced;
# what to do about it is a decision a person makes, not one this file makes quietly.
# (the list and the group-aware reader live in robots_rules.py since 2026-10-22)
from ..robots import AI_AGENTS, ai_named  # noqa: E402


class NewsSource(NewsPolicy):
    """`facts("2026-09-14")` or `facts("2026")` -> that day's / that year's headlines as Triples.
    Anything that is not a date gets nothing: this source knows WHEN, not WHO."""

    name = "news-rss"

    def __init__(self, feeds: dict[str, str] | None = None, cache_dir: pathlib.Path | None = CACHE,
                 ttl_s: int = TTL_S, timeout_s: int = 20, offline: bool = False,
                 topic: str | None = None, fetch=None) -> None:
        self.topic = topic if topic in TOPIC_FEEDS else None
        self.feeds = dict(feeds) if feeds else ({f"ap-{self.topic}": TOPIC_FEEDS[self.topic]}
                                                if self.topic else dict(FEEDS))
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.ttl_s, self.timeout_s, self.offline = int(ttl_s), int(timeout_s), bool(offline)
        # url -> the body (raising when the feed is down): the server hands it in (cubby_serve.sources.http.news()); this
        # package opens no socket (order 0031). Without one a feed is down, and `errors` says why
        self.fetch = fetch if fetch is not None else _no_fetch
        self.calls = 0
        self.last: dict = {}
        self.times: dict[str, dict] = {}
        self.provenance: dict[str, str] = {}       # fact -> the feed that published it
        self.licence: dict[str, str] = {}          # fact -> serve-only | open | unknown
        self.errors: dict[str, str] = {}
        self._robots: dict[str, list[str] | None] = {}

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
            body = self.fetch(self.feeds[feed_id])
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
        self.times, self.provenance, self.licence = {}, {}, {}
        m = _DATE_ASK.match(str(entity or ""))
        if not m:
            self.last["unresolved"] = True
            self.last["how"] = "not a date: this source answers WHEN, not WHO"
            return []
        prefix = "-".join(g for g in m.groups() if g)         # 2026 | 2026-09 | 2026-09-14
        out: list[Triple] = []
        for feed_id in self.feeds:
            n_before = len(out)
            licence = self.licence_of(feed_id)               # one robots lookup per feed, not per headline
            for title, when, _link in self.items(feed_id):
                if not when.startswith(prefix):
                    continue
                t = Triple(obj=f'"{title}"', rel=HEADLINE, subj=entity)
                fact = f"{t.obj} is the {t.rel} of {t.subj}"
                if _normalize(fact) in self.times:            # two feeds, the same words: still one fact
                    continue
                self.times[_normalize(fact)] = {"point": when}
                self.provenance[_normalize(fact)] = feed_id
                self.licence[_normalize(fact)] = licence
                out.append(t)
            if len(out) > n_before:
                self.last["feeds"].append(feed_id)
        self.last["n"] = len(out)
        self.last["how"] = f"headlines dated {prefix} from {len(self.last['feeds'])} feed(s)"
        return out


    # A publisher whose delivery domain is not their front door. `bbci.co.uk` is the BBC's
    # asset domain and names no AI crawler; `bbc.co.uk` names fifteen, Anthropic's among them.
    # Stripping subdomains never finds that, so the few known cases are written down rather
    # than guessed at -- and a case that is not in this table simply is not claimed.
    FRONT_DOOR = {"bbci.co.uk": "www.bbc.co.uk"}


    # -- serve vs train ------------------------------------------------------------------
    # Nick's decision, 2026-09-14, and it turns on the two uses being genuinely different:
    #
    #   SERVING  -- fetch a headline, speak it once with attribution, retain nothing. That is
    #               what a feed reader does, and RSS exists to be read by software. Allowed
    #               for every feed, including the opted-out ones: `User-agent: *` permits the
    #               path and nothing about it is a claim on the publisher's words.
    #   TRAINING -- bake headlines into a dataset that becomes model weights. That is what the
    #               block lists are about, and attribution does not undo it. Refused for any
    #               publisher who named an AI crawler.
    #
    # The point of putting it HERE rather than in a README is that it is enforceable per fact:
    # every fact carries its licence, so a dataset builder cannot include one by forgetting.
    SERVE_ONLY, OPEN, UNKNOWN = "serve-only", "open", "unknown"


    def topics(self, entity: str, min_n: int = 2) -> list[tuple[str, int]]:
        """The recurring capitalised subjects in the headlines for a date, commonest first --
        the menu for Nick's clarify bar when a date ask is too broad to answer in one breath.
        This is a COUNT OVER THE FETCHED HEADLINES, not a ranking of importance: it says what
        was written about, never what mattered."""
        counts: dict[str, int] = {}
        for t in self.facts(entity):
            # DROP THE FIRST WORD. It is capitalised because it starts the headline, not because
            # it is a name -- "Have", "Calls", "Watch" all reached the live menu that way
            # (2026-09-14). A stop list is an arms race against English; the sentence's own
            # shape is the actual signal, and a real name recurs mid-headline anyway.
            body = t.obj.lstrip('"').split(" ", 1)
            text = body[1] if len(body) > 1 else ""
            # A TITLE-CASED headline capitalises almost every word, so its capitals carry no
            # information about which words are names -- that is how "Run" and "Calls" reached
            # the live menu (2026-09-14). Count only sentence-cased headlines, where a capital
            # mid-sentence actually means something. Skipping a publisher's whole house style
            # costs coverage; inventing names out of their typography costs the truth.
            words_all = [w for w in text.split() if w[:1].isalpha()]
            if len(words_all) >= 4 and sum(w[:1].isupper() for w in words_all) / len(words_all) > 0.6:
                continue
            words = re.findall(r"\b[A-Z][A-Za-z'\-]{2,}(?:\s+[A-Z][A-Za-z'\-]{2,})?", text)
            for w in {w.strip() for w in words}:
                if w.lower() in _TOPIC_STOP:
                    continue
                counts[w] = counts.get(w, 0) + 1
        return sorted(((w, n) for w, n in counts.items() if n >= min_n), key=lambda q: (-q[1], q[0]))
