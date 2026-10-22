"""Order 0024's Global Voices reader on synthetic HTML (no source text): the story's own text and its translated
quotes are kept; images, captions, embedded posts, iframes, donation boxes, the "originally published" line and the
credits footer are not; the writer, the translator and the source story are read from the credits; categories lose
the language tag; a translation pairs with its source only when both are archived; the archive keeps the text for a
full-text feed and for no other.
Run: python -m pytest standin/tests/test_news_globalvoices.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import news_archive as na  # noqa: E402
import news_globalvoices as gv  # noqa: E402

HTML = """<p><big class='tagline'><em>A short tagline.</em></big></p>
<p class='originally-published'><small>Originally published on <a href='https://x.example/it'>GV</a></small></p>
<div class="wp-caption"><img src="a.webp"/><p class="wp-caption-text">Photo by someone.</p></div>
<p>First paragraph &amp; more.</p>
<blockquote class="translation"><p>A quote the site translated.</p></blockquote>
<blockquote class="twitter-tweet"><p>An embedded post.</p></blockquote>
<iframe src="https://video.example/embed"></iframe>
<h3>A heading</h3><ul><li>One item</li></ul>
<p style="text-align: center;"><a class='donatebutton' href='https://x.example/donate'>Give</a></p>
<div class='gv-rss-footer'><strong><div class='text-credits-container'>
<div class='text-credits-section'><span class='credit-label'>Written (English) by</span> <a class='user-link'>Ana Writer</a></div>
<div class='text-credits-section'><span class='credit-label'>Traduit (Français) par</span> <a class='user-link'>Tom Translator</a></div>
</div><span class='source-link'><a href='https://x.example/en/original'>Voir l'article original</a></span></strong></div>"""


def test_story_keeps_the_text_and_reads_the_credits():
    s = gv.story(HTML)
    assert s["text"].split("\n") == ["A short tagline.", "First paragraph & more.", "A quote the site translated.",
                                     "A heading", "One item"]
    assert (s["writer"], s["translator"], s["source_link"], s["licence"]) == \
        ("Ana Writer", "Tom Translator", "https://x.example/en/original", "CC BY 3.0")
    assert s["writer_label"] == "Written (English) by" and s["translator_label"].startswith("Traduit")


def test_an_original_has_no_translator_and_no_source():
    s = gv.story("<p>Text.</p><div class='gv-rss-footer'><span class='credit-label'>Written by</span> "
                 "<a class='user-link'>Ana</a></div>")
    assert s["text"] == "Text." and s["writer"] == "Ana" and s["translator"] is None and s["source_link"] is None


def test_categories_and_pairs():
    assert gv.categories(["Brazil", "English", "Digital Activism"]) == ["Brazil", "Digital Activism"]
    rows = [{"link": "fr/1", "source_link": "en/1"}, {"link": "en/1"}, {"link": "fr/2", "source_link": "en/9"}]
    assert gv.pairs(rows) == [("fr/1", "en/1")]


FEED = ("""<?xml version="1.0"?><rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><guid>g1</guid><title>A story</title><link>https://x.example/s</link><pubDate>Wed, 21 Oct 2026 10:00:00 +0000</pubDate>
<category>Brazil</category><category>English</category><description>Short.</description>
<content:encoded><![CDATA[""" + HTML + """]]></content:encoded></item></channel></rss>""").encode()


def test_the_archive_keeps_the_text_for_a_full_text_feed_only(tmp_path):
    fetch = lambda url, timeout=20: (200, FEED) if url.endswith("/feed") else (404, b"")    # noqa: E731
    a = na.Archive(tmp_path / "a", fetch=fetch)
    a.poll({"id": "gv", "url": "https://x.example/feed", "status": "active", "full_text": "globalvoices"})
    a.poll({"id": "other", "url": "https://y.example/feed", "status": "active"})
    import json
    rows = {json.loads(l)["feed"]: json.loads(l) for f in (tmp_path / "a").glob("????-??-??.jsonl")
            for l in f.read_text(encoding="utf-8").splitlines()}
    assert rows["gv"]["text"].startswith("A short tagline.") and rows["gv"]["categories"] == ["Brazil"]
    assert rows["gv"]["source_link"] == "https://x.example/en/original" and "content" not in rows["gv"]
    assert "text" not in rows["other"] and "content" not in rows["other"] and rows["other"]["summary"] == "Short."
