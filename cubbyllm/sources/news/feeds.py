"""cubbyllm/sources/news/feeds.py -- the RSS date source's feeds and settings: the cache, the poller's name, the TTL,
the default and topic feeds, the date ask, an item's date and text, the absent fetch. Split out of source.py (order
0031), verbatim; source.py re-exports every name.

Wired: WIRED (the news source).
"""
from __future__ import annotations

import re
from email.utils import parsedate_to_datetime

from cubbyllm.local_paths import data_dir

__wiring__ = "WIRED"


CACHE = data_dir("news_cache")


# The BARE PRODUCT TOKEN, and it matters. CBC's edge closes the connection on any UA string
# carrying a URL or parentheses -- "CubbyLLM/0.1 (+https://...)" is refused, "CubbyLLM/0.1" is
# served in 0.1s (measured 2026-09-14, all three CBC feeds). So the feed is reachable while
# saying truthfully who we are; a browser UA also works and is NOT used, because getting past a
# publisher's door by claiming to be Chrome is lying to them about who is asking.
UA = "CubbyLLM/0.1"


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


def _no_fetch(url: str) -> bytes:
    raise RuntimeError("the news source needs fetch= (cubby_serve.sources.http.news()): cubbyllm/ opens no socket")
