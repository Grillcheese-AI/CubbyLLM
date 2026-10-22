"""standin/news_globalvoices.py -- Global Voices' stories as the archive keeps them (order 0024; Nick, 2026-10-22).

Global Voices publishes under CC BY 3.0 and Nick said keep the full text, for this publisher only. From an item's
``content:encoded``, ``story(html)`` keeps:
- **the text**, plain: paragraphs, headings, list items and the site's own translated quotes (``blockquote
  class="translation"``), one block a line. Dropped, as the licence does not cover them or they are not the story:
  images, figures and captions, iframes, scripts, styles, svg, video and audio; embedded posts (a ``blockquote`` with
  any other class: ``twitter-tweet``, ``instagram-media``, ...); the donation and campaign boxes; the "originally
  published" line (it links the item itself) and the credits footer (read below, not kept as text);
- **the credits**: the writer (the first credits section, "Written (English) by", "Ecrit par") and the translator (a
  second one, "Traduit (Français) par"), each with its label;
- **the source story**: the link in ``span.source-link`` -- a translation's original. An item without one was written
  in its own language.

``categories(item)`` reads the item's categories as its places and topics, minus the language tags ("English").
``pairs(rows)``: (translation link, source link) where both stories are in the archive -- parallel text.

Wired: WIRED (``news_archive.Archive.poll``, for a feed whose manifest entry says ``"full_text": "globalvoices"``).
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

__wiring__ = "WIRED"

LICENCE = "CC BY 3.0"
SKIP_TAGS = {"img", "figure", "figcaption", "iframe", "script", "style", "svg", "video", "audio", "embed", "object",
             "picture", "source", "noscript", "form"}
SKIP_CLASS = re.compile(r"caption|tweet|instagram|tiktok|embed|donat|campaign|originally-published|gv-rss-footer")
BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div", "br", "tr"}
VOID = {"br", "img", "hr", "meta", "link", "input", "source", "embed", "wbr", "area", "col", "param", "track"}
LANGUAGE_TAGS = {"English", "Français", "Español", "Português", "Deutsch", "Italiano", "Русский", "العربية"}


class _Story(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.credits, self.label, self.source, self.in_source = [], [], [], None, None, False
        self.in_label = self.in_user = False
        self.user = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if "source-link" in cls:
            self.in_source = True
        if self.in_source and tag == "a" and self.source is None:
            self.source = a.get("href")
        if "credit-label" in cls:
            self.in_label, self.label = True, ""
        if "user-link" in cls:
            self.in_user, self.user = True, []
        skip = tag in SKIP_TAGS or bool(SKIP_CLASS.search(cls)) or (tag == "blockquote" and cls and cls != "translation")
        if tag in VOID:
            if tag == "br" and not self.skip:
                self.out.append("\n")
            return
        self.skip.append(skip or bool(self.skip and self.skip[-1]))
        if tag in BLOCKS and not self.skip[-1]:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if tag == "span" and self.in_label:
            self.in_label = False
        if tag == "a" and self.in_user:
            self.in_user = False
            self.credits.append({"label": (self.label or "").strip(), "name": " ".join("".join(self.user).split())})
        if tag == "span" and self.in_source:
            self.in_source = False
        if self.skip:
            was = self.skip.pop()
            if tag in BLOCKS and not was:
                self.out.append("\n")

    def handle_data(self, data):
        if self.in_label:
            self.label += data
        if self.in_user:
            self.user.append(data)
        if not (self.skip and self.skip[-1]):
            self.out.append(data)


def story(html: str) -> dict:
    """``{text, writer, writer_label, translator, translator_label, source_link, licence}`` of one item."""
    p = _Story()
    p.feed(html or "")
    p.close()
    lines = [" ".join(l.split()) for l in "".join(p.out).split("\n")]
    text = "\n".join(l for l in lines if l)
    w = p.credits[0] if p.credits else {}
    t = p.credits[1] if len(p.credits) > 1 else {}
    return {"text": text, "writer": w.get("name"), "writer_label": w.get("label"), "translator": t.get("name"),
            "translator_label": t.get("label"), "source_link": p.source, "licence": LICENCE}


def categories(names) -> list[str]:
    return [c for c in names if c and c not in LANGUAGE_TAGS]


def pairs(rows) -> list[tuple[str, str]]:
    """(translation link, source link) for every translation whose source story is also in ``rows``."""
    have = {r.get("link") for r in rows}
    return sorted({(r["link"], r["source_link"]) for r in rows if r.get("source_link") and r["source_link"] in have
                   and r.get("link")})
