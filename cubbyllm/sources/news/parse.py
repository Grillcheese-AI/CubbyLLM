"""standin/news_parse.py -- a feed's items as the news archive keeps them (order 0024): RSS `item`s and Atom `entry`s,
headline and summary only (the summary cut at `SUMMARY_MAX` characters), each dated when its date can be read.

Wired: WIRED (`news_archive`).
"""
from __future__ import annotations

import pathlib
import sys
import xml.etree.ElementTree as ET

__wiring__ = "WIRED"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from news_source import _iso  # noqa: E402

SUMMARY_MAX = 400
ATOM = "{http://www.w3.org/2005/Atom}"
DC = "{http://purl.org/dc/elements/1.1/}"


def _clean(text: str | None) -> str:
    import html
    import re
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", html.unescape(text or ""))).split())


def _first(el, *names) -> str:
    for n in names:
        f = el.find(n)
        if f is not None and (f.text or "").strip():
            return f.text.strip()
    return ""


CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"


def parse(body: bytes, full: bool = False) -> list[dict]:
    """RSS `item`s and Atom `entry`s -> [{id, link, published, title, summary}]. An item with no title is dropped; an
    undated one keeps `published: None` (it is still dated by when it was first seen). ``full`` (a publisher whose
    full text is kept, Nick's decision per publisher): also the raw ``content:encoded`` and the categories."""
    root = ET.fromstring(body)
    out = []
    for it in root.iter("item"):
        link = _first(it, "link")
        raw = _first(it, "pubDate", DC + "date")
        out.append({"id": _first(it, "guid") or link, "link": link, "published_raw": raw,
                    "title": _clean(_first(it, "title")), "summary": _clean(_first(it, "description"))})
        if full:
            out[-1].update(content=_first(it, CONTENT), categories=[_clean(c.text) for c in it.findall("category")])
    for it in root.iter(ATOM + "entry"):
        link = next((l.get("href") for l in it.findall(ATOM + "link") if l.get("rel") in (None, "alternate")), "")
        raw = _first(it, ATOM + "published", ATOM + "updated")
        out.append({"id": _first(it, ATOM + "id") or link, "link": link or "", "published_raw": raw,
                    "title": _clean(_first(it, ATOM + "title")), "summary": _clean(_first(it, ATOM + "summary"))})
    for o in out:
        o["published"] = date_of(o.pop("published_raw"))
        if len(o["summary"]) > SUMMARY_MAX:
            o["summary"] = o["summary"][:SUMMARY_MAX].rsplit(" ", 1)[0] + " ..."
    return [o for o in out if o["title"]]


# Month names in the languages the feeds write: Crisis Group dates its Spanish and Portuguese items in theirs
# ("Viernes, Agosto 7, 2026 - 14:45", "Quarta, Maio 13, 2026 - 10:10"; 2026-10-22).
_MONTHS = {}
for _names in ("january february march april may june july august september october november december",
               "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre",
               "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro",
               "janvier février mars avril mai juin juillet août septembre octobre novembre décembre"):
    _MONTHS.update({m: i + 1 for i, m in enumerate(_names.split())})
_MONTHS["setiembre"] = 9


def date_of(raw: str) -> str | None:
    """RFC-822 and ISO (`news_source._iso`), and the long form one publisher writes ("Friday, October 16, 2026 - 19:18",
    Crisis Group), in any language of `_MONTHS`: a date this cannot read stays None (the item is still dated by when it
    was first seen)."""
    import re
    got = _iso(raw)
    if got or not raw:
        return got
    m = re.search(r"([^\W\d_]+) (\d{1,2}), (\d{4})", raw)
    if m and m.group(1).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return None
