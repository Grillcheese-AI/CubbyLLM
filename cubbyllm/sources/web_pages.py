"""cubbyllm/sources/web_pages.py -- a page read: which site a URL is (the witnesses' independence), the page's text and
its JSON-LD, and the facts its schema.org blocks state. Split out of web.py (order 0031), verbatim; web.py re-exports
every name.

Wired: WIRED (the web source).
"""
from __future__ import annotations

import json
import re
import urllib.robotparser
from html.parser import HTMLParser
from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"


# ── sites: who is a separate witness ───────────────────────────────────────────────────────────────
# the Wikipedia family and the sites that republish it: one witness, not several
FOLD = {"wikipedia.org": "wikipedia.org", "wikimedia.org": "wikipedia.org", "wikiwand.com": "wikipedia.org",
        "wikizero.com": "wikipedia.org", "dbpedia.org": "wikipedia.org", "infogalactic.com": "wikipedia.org",
        "wiki2.org": "wikipedia.org", "wikiless.org": "wikipedia.org", "wikimili.com": "wikipedia.org",
        "everybodywiki.com": "wikipedia.org", "dewiki.de": "wikipedia.org", "wikibrief.org": "wikipedia.org",
        # the Wikipedia/Wikidata derivatives the live battery met (2026-09-24): Alchetron and Kiddle republish
        # articles, PeoplePill / FamousFix / Born Glorious are built from Wikidata -- one witness each family
        "alchetron.com": "wikipedia.org", "kiddle.co": "wikipedia.org", "academic.ru": "wikipedia.org",
        "en-academic.com": "wikipedia.org", "justapedia.org": "wikipedia.org", "peoplepill.com": "wikidata.org",
        "famousfix.com": "wikidata.org", "bornglorious.com": "wikidata.org", "wikidata.org": "wikidata.org"}


# not witnesses: they repeat another site's page, or a model wrote them (the web's own LLM encyclopedias are
# exactly the external-LLM content this loop does not take its facts from)
NOT_WITNESSES = {"archive.org", "archive.ph", "archive.today", "archive.is", "webcache.googleusercontent.com",
                 "grokipedia.com"}


_TWO_LEVEL = {"co", "com", "net", "org", "gov", "ac", "edu", "gouv", "gc", "qc"}


def site_of(url: str) -> str | None:
    """The registrable domain a URL belongs to (eTLD+1, approximated: 'co.uk'-style second levels keep
    three labels), folded for the Wikipedia family; None for a non-witness or a malformed URL."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")
    if not host or "." not in host:
        return None
    parts = host.split(".")
    keep = 3 if (len(parts) >= 3 and parts[-2] in _TWO_LEVEL and len(parts[-1]) == 2) else 2
    site = ".".join(parts[-keep:])
    if site in NOT_WITNESSES or host in NOT_WITNESSES:
        return None
    return FOLD.get(site, site)


# ── pages: text and JSON-LD, nothing else ──────────────────────────────────────────────────────────
class _Page(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "nav", "footer", "form", "button"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.jsonld: list[str] = []
        self.lang = ""
        self._skip = 0
        self._ld = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html" and a.get("lang"):
            self.lang = a["lang"].lower()
        if tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._ld = True; self._buf = []
            return
        if tag in self.SKIP:
            self._skip += 1
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div", "section", "article", "td", "dd"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag == "script" and self._ld:
            self._ld = False; self.jsonld.append("".join(self._buf))
            return
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._ld:
            self._buf.append(data)
        elif not self._skip:
            self.text.append(data)


def parse_page(body: bytes, content_type: str = "") -> dict:
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    enc = m.group(1) if m else "utf-8"
    try:
        doc = body.decode(enc, "replace")
    except LookupError:
        doc = body.decode("utf-8", "replace")
    p = _Page()
    try:
        p.feed(doc); p.close()
    except Exception:                                        # noqa: BLE001 -- broken markup: keep what parsed
        pass
    lines = [" ".join(line.split()) for line in "".join(p.text).split("\n")]
    return {"lang": p.lang, "text": "\n".join(l for l in lines if l)[:200_000], "jsonld": p.jsonld[:20]}


# ── readers ──────────────────────────────────────────────────────────────────────────────────────────
# schema.org property -> the store's relation label (Wikidata's), for the ones that mean the same thing
SCHEMA = {"birthDate": "date of birth", "deathDate": "date of death", "birthPlace": "place of birth",
          "deathPlace": "place of death", "nationality": "country of citizenship", "spouse": "spouse",
          "alumniOf": "educated at", "worksFor": "employer", "award": "award received", "memberOf": "member of",
          "foundingDate": "inception", "founder": "founded by", "author": "author", "director": "director",
          "datePublished": "publication date", "containedInPlace": "located in the administrative territorial entity",
          "genre": "genre", "parentOrganization": "parent organization"}


_ISO_DATE = re.compile(r"^(\d{4})(-\d{2}(-\d{2})?)?")


def _nodes(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from _nodes(x)
    elif isinstance(obj, dict):
        yield obj
        for k in ("@graph", "mainEntity", "about"):
            if k in obj:
                yield from _nodes(obj[k])


def _values(v) -> list[str]:
    if isinstance(v, list):
        return [x for item in v for x in _values(item)]
    if isinstance(v, dict):
        n = v.get("name")
        return [n] if isinstance(n, str) else []
    return [v] if isinstance(v, str) else []


# a node of these types is the PAGE, not what the page is about: Wikipedia's JSON-LD is an Article named
# "Ada Lovelace" whose author is "Contributors to Wikimedia projects" (2026-09-24, the live smoke run)
PAGE_TYPES = {"article", "newsarticle", "blogposting", "scholarlyarticle", "report", "webpage", "website", "itempage",
              "profilepage", "aboutpage", "collectionpage", "faqpage", "qapage", "medicalwebpage", "breadcrumblist",
              "searchresultspage", "imageobject", "videoobject", "socialmediaposting", "discussionforumposting"}


def jsonld_facts(entity: str, blocks: list[str]) -> list[Triple]:
    """The schema.org claims a page makes about a node NAMED as the entity (name or alternateName) --
    never about the page itself (`PAGE_TYPES`)."""
    ent = normalize(entity)
    out: list[Triple] = []
    for raw in blocks:
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        for node in _nodes(data):
            types = node.get("@type")
            if {str(x).lower() for x in (types if isinstance(types, list) else [types])} & PAGE_TYPES:
                continue
            names = _values(node.get("name")) + _values(node.get("alternateName"))
            if ent not in {normalize(n) for n in names}:
                continue
            for prop, rel in SCHEMA.items():
                for val in _values(node.get(prop)):
                    val = " ".join(val.split())
                    if not val or len(val) > 80:
                        continue
                    if rel in ("date of birth", "date of death", "inception", "publication date"):
                        m = _ISO_DATE.match(val)
                        if not m:
                            continue
                        val = m.group(0)
                    if normalize(val) != ent:
                        out.append(Triple(obj=val, rel=rel, subj=entity))
    return out
