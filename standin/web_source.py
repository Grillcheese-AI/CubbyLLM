"""standin/web_source.py -- the web as the Source of last resort: search, read, corroborate.

Wired: WIRED (stand-in serve stack; `standin/ask.py` hands it to `learn_and_answer` as a fallback
behind Wikidata; `serve_api --web`). Implements `cubbyllm.reasoning.learn.Source`.

WHAT IT IS FOR. The ask loop fills the store from Wikidata. When Wikidata has no claim the walk
needs -- a recent event, a niche entity, a property nobody entered -- the loop refused, correctly,
and stopped there. This is where it goes next, for the one entity and relation the walk stalled on.

THE RULE: a page is a witness, not a fact. A triple read off the web is HELD -- the latent tier,
exactly as an LFM fact is: a certified chain resting on it is refused `latent_only`, the would-be
answer on record -- until TWO INDEPENDENT SITES state it (Nick, 2026-09-24). Agreement lifts the
hold and both sites stay in the provenance. One page, however confident, never makes the loop speak.

The readers, most structured first:
  wikidata  a Wikipedia hit names its Wikidata item; the item's claims come through `WikidataSource`,
            attested by Wikidata itself (not held). Taken only when ONE item among the hits carries
            the relation the walk needs -- the question decides which item, never the ranking.
  jsonld    schema.org JSON-LD a page publishes about the entity (birthDate, foundingDate, ...).
  frames    `wikitext.py`'s sentence frames (+ day-first dates) over the snippets and the page
            text, on sentences that OPEN with the entity -- 'X's friend was born in Ulm' is not
            about X.
  lfm       the local model (`LfmReader`) reads a passage for the relation the walk needs; a value
            that does not occur in the passage is dropped before it is ever counted.
Each read is credited to its site. Sites are registrable domains (eTLD+1, approximated), with the
Wikipedia family and its mirrors folded into one -- en.wikipedia.org and wikiwand.com are one witness.
Archives are not witnesses at all (they repeat another site).

Politeness: page fetches honour robots.txt (snippets come from the search API); pages are capped in
count and size; search results and pages are cached on disk with a TTL, so a rerun inside the window
is offline. The tally of which sites stated which fact persists beside the cache, so a second site
found tomorrow completes a fact first seen today.

Backends: Brave (`BRAVE_SEARCH_API_KEY`, metered) and SearXNG (a self-hosted instance with JSON output
enabled, `CUBBY_SEARXNG_URL`), one interface. No search engine's answer box or generated summary is
ever read -- only its links and snippets.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import pathlib
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from html.parser import HTMLParser

from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"

CACHE = pathlib.Path(__file__).resolve().parent / "data" / "out" / "web_cache"
UA = "cubbyllm-standin/0.1 (research; licensing@grillcheese.ai)"
TTL_S = 24 * 3600                  # a search or a page is re-fetched at most daily
MAX_PAGE_BYTES = 1_500_000
MIN_SITES = 2                      # independent sites before a web fact may carry an answer


ENV_FILES = (pathlib.Path(__file__).resolve().parent / ".env", pathlib.Path(__file__).resolve().parents[1] / ".env")


def env_key(*names: str) -> str | None:
    """A secret from the environment, else from a gitignored `.env` (standin/ first, then the repo root), as
    `NAME=value` lines. The owner writes the file; the code only reads it, and never logs what it read."""
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    for path in ENV_FILES:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            k, sep, v = line.strip().partition("=")
            if sep and k.strip().removeprefix("export ").strip() in names and v.strip():
                return v.strip().strip('"').strip("'")
    return None


# ── search backends ─────────────────────────────────────────────────────────────────────────────────
@dataclass
class Hit:
    url: str
    title: str
    snippet: str
    extra: list[str] = field(default_factory=list)


def _http(url: str, headers: dict | None = None, timeout_s: float = 10, limit: int = MAX_PAGE_BYTES):
    """(content_type, bytes) or None. Never raises: a site that is down is not an answer."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.headers.get("Content-Type", ""), r.read(limit)
    except Exception:                                        # noqa: BLE001
        return None


_TAG = re.compile(r"<[^>]+>")


def _plain(s: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", s or "")).split())


class BraveBackend:
    """Brave Search API (web). The owner's plan (2026-09-24) licenses the results for AI use AND for
    storage, at 50 requests a second: results are archived (`WebSource` keeps every live search in
    `searches.jsonl`) and a triple's evidence keeps its snippet. The limiter keeps under the ceiling."""
    name = "brave"
    URL = "https://api.search.brave.com/res/v1/web/search"
    license = "brave: data for AI, storage permitted"

    def __init__(self, key: str | None = None, http=_http, rate_per_s: float = 50.0) -> None:
        import threading
        self.key = key or env_key("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY")
        if not self.key:
            raise ValueError("no Brave key: set BRAVE_SEARCH_API_KEY in the environment or in standin/.env")
        self.http = http
        self.min_gap = 1.0 / float(rate_per_s) if rate_per_s else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def _wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            at = max(now, self._next)
            self._next = at + self.min_gap
        if at > now:
            time.sleep(at - now)

    def search(self, q: str, n: int = 20) -> list[Hit] | None:
        self._wait()
        url = f"{self.URL}?{urllib.parse.urlencode({'q': q, 'count': min(int(n), 20), 'extra_snippets': 'true'})}"
        got = self.http(url, {"Accept": "application/json", "X-Subscription-Token": self.key})
        if got is None:
            return None
        try:
            body = json.loads(got[1].decode("utf-8", "replace"))
        except ValueError:
            return None
        return [Hit(r.get("url", ""), _plain(r.get("title", "")), _plain(r.get("description", "")),
                    [_plain(x) for x in (r.get("extra_snippets") or [])])
                for r in ((body.get("web") or {}).get("results") or [])[:n] if r.get("url")]


class SearxngBackend:
    """A SearXNG instance (self-hosted; `search.formats` must include json in its settings.yml)."""
    name = "searxng"
    license = None                     # the engines behind it set their own terms: results are not archived

    def __init__(self, base_url: str | None = None, http=_http) -> None:
        self.base = (base_url or env_key("CUBBY_SEARXNG_URL") or "").rstrip("/")
        if not self.base:
            raise ValueError("no SearXNG instance: pass its URL or set CUBBY_SEARXNG_URL")
        self.http = http

    def search(self, q: str, n: int = 8) -> list[Hit] | None:
        got = self.http(f"{self.base}/search?{urllib.parse.urlencode({'q': q, 'format': 'json'})}",
                        {"Accept": "application/json"})
        if got is None:
            return None
        try:
            body = json.loads(got[1].decode("utf-8", "replace"))
        except ValueError:
            return None
        return [Hit(r.get("url", ""), _plain(r.get("title", "")), _plain(r.get("content", "")))
                for r in (body.get("results") or [])[:n] if r.get("url")]


def backend_from_env(kind: str = "auto", searxng_url: str | None = None):
    """'brave' | 'searxng' | 'auto' (Brave when a key is set, else SearXNG when an instance is named,
    else None: the loop runs without the web, as before)."""
    if kind in ("brave", "auto") and env_key("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"):
        return BraveBackend()
    if kind in ("searxng", "auto") and (searxng_url or env_key("CUBBY_SEARXNG_URL")):
        return SearxngBackend(searxng_url)
    if kind in ("brave", "searxng"):
        raise ValueError(f"--web {kind}: " + ("set BRAVE_SEARCH_API_KEY" if kind == "brave" else "set CUBBY_SEARXNG_URL or --searxng"))
    return None


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


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MNUM = {m.lower(): i for i, m in enumerate(_MONTHS.split("|"), 1)}
# day-first English dates, which the web writes and DBpedia's abstracts did not ('7 November 1867')
WEB_FRAMES_EN = [
    ("date of birth", re.compile(rf"\((?:born )?(?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})\s*(?:[–-]|\)|;|,)")),
    ("date of death", re.compile(rf"\((?:\d{{1,2}} )?(?:{_MONTHS}) ?\d{{0,2}},? \d{{4}}\s*[–-]\s*(?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})\)")),
    ("date of death", re.compile(rf"\bdied (?:on )?(?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})")),
    ("date of death", re.compile(rf"\bdied (?:in [A-Z][\w'-]+(?:,? [A-Z][\w'-]+){{0,3}},? )?(?:on )?(?P<m>{_MONTHS}) (?P<d>\d{{1,2}}),? (?P<y>\d{{4}})")),
    # 'was born in Warsaw on November 7, 1867' (nobelprize.org, live 2026-09-24)
    ("date of birth", re.compile(rf"\bborn (?:in [A-Z][\w'-]+(?:,? [A-Z][\w'-]+){{0,3}},? )?on (?P<m>{_MONTHS}) (?P<d>\d{{1,2}}),? (?P<y>\d{{4}})")),
    ("date of birth", re.compile(rf"\bborn (?:in [A-Z][\w'-]+(?:,? [A-Z][\w'-]+){{0,3}},? )?on (?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})")),
]
# 'died in Desert Hot Springs, California' -- the born-in frame's twin (wikitext.py has none for death)
WEB_PLACE_FRAMES = [
    ("place of death", re.compile(r"\bdied (?:(?:on )?[A-Z][a-z]+ \d{1,2},? \d{4},? |(?:on )?\d{1,2} [A-Z][a-z]+ \d{4},? )?(?:at (?:his|her|their) home )?in (?P<o>[A-Z][\w'-]+(?:,? [A-Z][\w'-]+){0,3})")),
]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(À-Ý])")


def _opens_with(sentence: str, entity: str) -> bool:
    """The sentence is ABOUT the entity: it opens with its name (or 'The' + name), and the name is not a
    possessive or the head of a longer name -- "Einstein's friend was born in Ulm" is not about Einstein."""
    s = sentence.lstrip(" \"“(").lower()
    e = " ".join((entity or "").lower().split())
    if not e:
        return False
    for pre in ("", "the "):
        if s.startswith(pre + e):
            rest = s[len(pre + e):]
            return not rest or not (rest[0].isalnum() or rest[:2] in ("'s", "’s") or rest[0] in "-’'")
    return False


def frame_facts(entity: str, text: str, lang: str = "en") -> list[tuple[Triple, str]]:
    """wikitext's frames, and the day-first date frames, over the sentences that OPEN with the entity."""
    from wikitext import read_frames
    raw: list[tuple[Triple, str]] = []
    out = raw
    for sentence in _SENT.split(" ".join((text or "").split())):
        if not _opens_with(sentence, entity):
            continue
        langs = ("fr",) if lang.startswith("fr") else ("en", "fr")
        for lg in langs:
            out.extend(read_frames(entity, sentence, lg))
        for rel, rx in WEB_FRAMES_EN:
            m = rx.search(sentence)
            if m:
                out.append((Triple(obj=f"{int(m['y']):04d}-{_MNUM[m['m'].lower()]:02d}-{int(m['d']):02d}",
                                   rel=rel, subj=entity), sentence))
        for rel, rx in WEB_PLACE_FRAMES:
            m = rx.search(sentence)
            if m and normalize(m["o"]) != normalize(entity):
                out.append((Triple(obj=m["o"], rel=rel, subj=entity), sentence))
    # a place read up to the next sentence's first word: 'born in Ashtabula County, Ohio Josephine ...' (live)
    names = set(normalize(entity).split())
    trimmed = []
    for t, sentence in raw:
        if t.rel.startswith("place of") or t.rel.startswith("located in"):
            words = t.obj.split()
            while len(words) > 1 and normalize(words[-1]) in names:
                words.pop()
            t = Triple(obj=" ".join(words).rstrip(","), rel=t.rel, subj=t.subj)
        trimmed.append((t, sentence))
    return trimmed


def mentions(entity: str, text: str) -> bool:
    """The name, or every one of its words of three letters or more ('Augusta Ada King, Countess of
    Lovelace' mentions Ada Lovelace)."""
    low = " " + " ".join(re.findall(r"\w+", (text or "").lower())) + " "
    full = " ".join(re.findall(r"\w+", (entity or "").lower()))
    words = [w for w in full.split() if len(w) >= 3]
    return bool(words) and (f" {full} " in low or all(f" {w} " in low for w in words))


_HATNOTE = re.compile(r"^(?:This article is about|For other uses|Not to be confused with)|\(disambiguation\)", re.I)


def passages(entity: str, text: str, limit: int = 2, width: int = 900, min_words: int = 12) -> list[str]:
    """The paragraphs that mention the entity and read as prose (>= `min_words` words): what the reader is
    shown. Navigation and menus are short lines -- live, 2026-09-24, the first MENTION of 'Ada Lovelace'
    on her Wikipedia page was the header beside '112 languages', and the reader read that as a birthplace."""
    out = []
    for line in (text or "").split("\n"):
        if _HATNOTE.search(line):
            continue                           # 'This article is about ... For other uses, see ...'
        if len(line.split()) >= min_words and mentions(entity, line):
            out.append(" ".join(line.split())[:width])
            if len(out) >= limit:
                break
    return out


_VALUE = {"date": re.compile(r"^(?:\d{4}(?:-\d{2}(?:-\d{2})?)?|\d{1,4} BC)$"),
          "number": re.compile(r"^[+-]?\d[\d,.\s]*$")}


_FR_MONTHS = "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"
_MNUM_FR = {m: i for i, m in enumerate(_FR_MONTHS.split("|"), 1)}
_DMY = re.compile(rf"^(?P<d>\d{{1,2}})(?:er)? (?P<m>{_MONTHS}|{_FR_MONTHS}) (?P<y>\d{{4}})$", re.I)
_MDY = re.compile(rf"^(?P<m>{_MONTHS}) (?P<d>\d{{1,2}}),? (?P<y>\d{{4}})$", re.I)


def iso_date(value: str) -> str | None:
    """'10 December 1815' / 'December 10, 1815' / '10 décembre 1815' -> '1815-12-10'; an ISO date or a
    bare year stays as it is; anything else (a range like '1815-1852') is None -- not a date."""
    v = " ".join((value or "").split())
    if _VALUE["date"].match(v):
        return v
    m = _DMY.match(v) or _MDY.match(v)
    if not m:
        return None
    mo = m["m"].lower()
    n = _MNUM.get(mo) or _MNUM_FR.get(mo)
    return f"{int(m['y']):04d}-{n:02d}-{int(m['d']):02d}" if n else None


def plausible(kind: str | None, value: str) -> bool:
    """A value of the kind the relation holds: a date for 'date of birth', a number for 'height', a name
    (letters, not opening on a digit) for 'place of birth'. Live, 2026-09-24: the reader wrote '1815-1852'
    as a date of birth and '112 languages' as a place of birth; both occur in the page, neither is one."""
    v = (value or "").strip()
    if not v or len(v) > 80:
        return False
    if kind in _VALUE:
        return bool(_VALUE[kind].match(v))
    if kind == "name":
        return bool(re.search(r"[^\W\d_]", v)) and not v[0].isdigit() and len(v.split()) <= 8
    return True


# ── the source ───────────────────────────────────────────────────────────────────────────────────────
# how a person types the relation into a search box (the property table's first wording is often 'DOB')
QUERY_WORDS = {"date of birth": "born", "place of birth": "born in", "date of death": "died", "place of death": "died in",
               "inception": "founded", "country of citizenship": "nationality", "educated at": "education",
               "spouse": "married", "author": "written by", "director": "directed by", "publication date": "published",
               "located in the administrative territorial entity": "located in", "headquarters location": "headquarters"}


def _fold(s: str) -> str:
    """normalize, accents folded: 'Móstoles' and 'Mostoles' are one spelling."""
    import unicodedata
    return normalize("".join(ch for ch in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(ch)))


def untruncate(s: str) -> str:
    """A snippet the engine cut off ('... born in Syracuse, New Yor…'), cut back to where it is whole: the
    last sentence end, else the last comma, else without its last word. Live, 2026-09-24: 'New Yor' made
    Tom Cruise's birthplace two claims and the walk refused."""
    s = (s or "").rstrip()
    if not (s.endswith("…") or s.endswith("...")):
        return s
    s = s.rstrip("….").rstrip()
    for sep in (". ", "! ", "? ", "; "):
        i = s.rfind(sep)
        if i > 20:
            return s[: i + 1]
    i = s.rfind(",")
    if i > 20:
        return s[:i]
    return s.rsplit(" ", 1)[0] if " " in s else ""


def _key(t: Triple) -> str:
    return normalize(f"{t.obj} is the {t.rel} of {t.subj}")


class WebSource:
    """`facts(entity, relations=, via=)` -> Triples about the entity from the web. Every triple's
    provenance names the sites that stated it (`provenance[normalize(fact)]`); `held` holds the ones
    fewer than `min_sites` independent sites stated -- `learn_and_answer` stores those in the latent
    tier. `last` carries the run: queries, urls, readers, and per fact which sites said it and how."""

    name = "web"
    latent = False                     # per fact: see `held`

    def __init__(self, backend, reader=None, wikidata=None, aliases=None, cache_dir: pathlib.Path | str | None = CACHE,
                 ttl_s: int = TTL_S, offline: bool = False, results: int = 20, pages: int = 6, reads: int = 4,
                 min_sites: int = MIN_SITES, http=_http, robots: bool = True, max_facts: int = 60,
                 second_query: bool = True, workers: int = 6, containment=None, dissent_blocks: bool = False) -> None:
        self.backend, self.reader, self.wikidata = backend, reader, wikidata
        self._aliases = aliases
        self.cache = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.ttl_s, self.offline = int(ttl_s), bool(offline)
        self.results, self.pages, self.reads = int(results), int(pages), int(reads)
        self.min_sites, self.max_facts = int(min_sites), int(max_facts)
        self.http, self.robots = http, bool(robots)
        self.second_query, self.workers = bool(second_query), int(workers)
        self.containment = containment      # a WikidataSource: which places contain which (None = the value's own commas only)
        self._up: dict[str, set[str]] = {}
        # one site's different value: ignored (it is not a claim, two sites are) or a veto (the web must agree)
        self.dissent_blocks = bool(dissent_blocks)
        self._dissent: dict[str, list[str]] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.provenance: dict[str, str] = {}
        self.held: set[str] = set()
        self.times: dict[str, dict] = {}
        self.last: dict = {}
        self.calls = {"search": 0, "page": 0, "read": 0, "wikipedia": 0}
        self.tally: dict[str, dict[str, list[str]]] = self._load_tally()   # fact -> site -> readers

    # -- the wording questions go to the property table, like every Source -------------------------
    @property
    def aliases(self):
        if self._aliases is None:
            from sources import PropertyAliases
            self._aliases = PropertyAliases()
        return self._aliases

    def relations(self, text: str) -> list[str]:
        return self.aliases.relations(text)

    def kind(self, label: str) -> str | None:
        return self.aliases.kind(label)

    def wordings(self, label: str) -> list[str]:
        return self.aliases.wordings(label)

    # -- cache ---------------------------------------------------------------------------------------
    def _path(self, kind: str, key: str) -> pathlib.Path | None:
        if not self.cache:
            return None
        return self.cache / f"{kind}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}.json"

    def _cached(self, kind: str, key: str, make):
        path = self._path(kind, key)
        if path and path.is_file():
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                if self.offline or time.time() - rec.get("t", 0) < self.ttl_s:
                    return rec["v"]
            except (OSError, ValueError, KeyError):
                pass
        if self.offline:
            return None
        v = make()
        if path and v is not None:
            path.write_text(json.dumps({"t": time.time(), "key": key[:300], "v": v}, ensure_ascii=False), encoding="utf-8")
        return v

    def _load_tally(self) -> dict:
        p = self.cache / "tally.json" if self.cache else None
        if p and p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        return {}

    def _save_tally(self) -> None:
        if self.cache:
            (self.cache / "tally.json").write_text(json.dumps(self.tally, ensure_ascii=False), encoding="utf-8")

    def _archive(self, q: str, hits: list[Hit]) -> None:
        """Every live search, kept -- only when the backend's licence permits storing its results."""
        if self.cache and getattr(self.backend, "license", None):
            with open(self.cache / "searches.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(time.time(), 1), "backend": self.backend.name, "license": self.backend.license,
                                     "q": q, "hits": [h.__dict__ for h in hits]}, ensure_ascii=False) + "\n")

    # -- transport -----------------------------------------------------------------------------------
    def _search(self, q: str) -> list[Hit]:
        def make():
            self.calls["search"] += 1
            hits = self.backend.search(q, self.results)
            if hits is not None:
                self._archive(q, hits)
            return None if hits is None else [h.__dict__ for h in hits]
        got = self._cached(f"search_{self.backend.name}", f"{q}|{self.results}", make)
        return [Hit(**h) for h in (got or [])]

    def _allowed(self, url: str) -> bool:
        if not self.robots:
            return True
        sp = urllib.parse.urlsplit(url)
        host = f"{sp.scheme}://{sp.netloc}"
        if host not in self._robots:
            got = self.http(f"{host}/robots.txt", timeout_s=8, limit=300_000)
            rp = None
            if got is not None:
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(got[1].decode("utf-8", "replace").splitlines())
            self._robots[host] = rp
        rp = self._robots[host]
        return rp is None or rp.can_fetch(UA, url)

    def _page(self, url: str) -> dict | None:
        def make():
            if not self._allowed(url):
                return {"robots": "disallowed"}
            self.calls["page"] += 1
            got = self.http(url)
            if got is None or "html" not in (got[0] or "").lower():
                return None
            return parse_page(got[1], got[0])
        return self._cached("page", url, make)

    def _qid(self, url: str) -> str | None:
        sp = urllib.parse.urlsplit(url)
        if not sp.path.startswith("/wiki/"):
            return None
        title = urllib.parse.unquote(sp.path[6:]).replace("_", " ")
        api = (f"https://{sp.netloc}/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item", "redirects": 1,
             "titles": title, "format": "json"}))

        def make():
            self.calls["wikipedia"] += 1
            got = self.http(api, {"Accept": "application/json"})
            if got is None:
                return None
            try:
                pages = json.loads(got[1].decode("utf-8", "replace"))["query"]["pages"]
            except (ValueError, KeyError):
                return None
            for pg in pages.values():
                q = (pg.get("pageprops") or {}).get("wikibase_item")
                if q:
                    return q
            return ""
        return self._cached("qid", api, make) or None

    # -- corroboration -------------------------------------------------------------------------------
    CONTAINED_BY = ("P131", "P17")        # located in the administrative entity; country

    def ancestors(self, place: str) -> set[str]:
        """The places that contain `place` per Wikidata (P131 / P17, six levels), for EVERY item whose label is
        exactly the name, intersected: a Paris in Texas and one in France vouch for nothing in common."""
        sets = self.items_up(place)
        return set.intersection(*sets) if sets else set()

    def items_up(self, place: str) -> list[set[str]]:
        """Per item labelled exactly `place`: the places that contain it."""
        key = _fold(place)
        if key in self._up:
            return self._up[key]
        wd, got = self.containment, []
        if wd is not None and key:
            try:
                s = wd._get({"action": "wbsearchentities", "search": place, "language": "en", "limit": 7}) or {}
                items = [h["id"] for h in s.get("search") or [] if _fold(h.get("label", "")) == key][:4]
                sets = []
                for qid in items:
                    seen, frontier = set(), {qid}
                    for _ in range(6):
                        nxt = {v["id"] for q in frontier for p in self.CONTAINED_BY for st in wd._claims(q).get(p, [])
                               for v in [((st.get("mainsnak") or {}).get("datavalue") or {}).get("value")]
                               if isinstance(v, dict) and v.get("id") and v["id"] not in seen}
                        seen |= nxt; frontier = nxt
                        if not frontier or len(seen) > 80:
                            break
                    sets.append({_fold(l) for l in (wd._labels_for(sorted(seen)) if seen else {}).values() if l})
                got = sets
            except Exception:                                # noqa: BLE001 -- no oracle is no containment, never a guess
                got = []
        self._up[key] = got
        return got

    def _finer(self, kind: str | None, a: str, b: str) -> bool:
        """Does value `a` say everything `b` says, and more? A date to the day says its year; 'Syracuse, New
        York' says New York; Warsaw says Poland when Wikidata places Warsaw in Poland."""
        if kind == "date":
            return a != b and a.startswith(b + "-")
        if kind != "name":
            return False
        pa = [_fold(x) for x in a.split(",") if _fold(x)]
        pb = [_fold(x) for x in b.split(",") if _fold(x)]
        if not pa or not pb or pa == pb:
            return False
        return pb[0] in pa[1:] or pb[0] in self.ancestors(a.split(",")[0])

    def _claims(self, evidence: dict) -> list[tuple[Triple, set[str], list[str]]]:
        """(the claim, the sites that vouch for it, the tally keys it merges), per relation today's evidence
        touched -- ONE value per relation when the web agrees, all of them when it does not.

        1. One spelling: values whose comma parts are prefixes of one another ('London, England, UK' /
           'London, England') are one claim, stated as the part all of them said; accents fold. Values that
           disagree anywhere stay apart.
        2. Entailment: a finer value vouches for a coarser one -- a site saying 'Syracuse, New York' says
           New York, a date to the day says its year. Never the other way: 'Poland' does not say Warsaw.
        3. Attested = vouched for by `min_sites` sites. If the attested values form one chain (each finer
           or coarser than the next), the finest is THE claim and nothing else is returned: the store gets
           one value and the walk can speak it. If they do not (two dates; London, England / London,
           Ontario), every attested value is returned and the walk refuses on the split. What is not
           attested stays in the tally, unreturned, when something else is -- one site's dissent is not a
           claim -- and is returned held when nothing is."""
        wanted = {(normalize(t.rel), normalize(t.subj)) for t, _s in evidence.values()}
        cands: dict[tuple[str, str], dict[str, tuple[str, str, str, set[str]]]] = {}
        for k, by_site in self.tally.items():
            e0 = next(iter(by_site.values()), None) if by_site else None
            if not isinstance(e0, dict) or "rel" not in e0:
                continue
            g = (normalize(e0["rel"]), normalize(e0["subj"]))
            if g in wanted:
                cands.setdefault(g, {})[k] = (e0["obj"], e0["rel"], e0["subj"], set(by_site))
        out = []
        for _g, members in cands.items():
            keys = list(members)
            kind = self.aliases.kind(members[keys[0]][1])
            parts = {k: [_fold(x) for x in members[k][0].split(",") if _fold(x)] for k in keys}

            def consistent(a, b):
                x, y = (parts[a], parts[b]) if len(parts[a]) <= len(parts[b]) else (parts[b], parts[a])
                return bool(x) and y[: len(x)] == x
            # 1. spellings
            claims: list[tuple[Triple, set[str], list[str]]] = []
            done: set[str] = set()
            for k in keys:
                if k in done:
                    continue
                comp = [k]
                if kind == "name":
                    stack = [k]
                    while stack:
                        c = stack.pop()
                        for o in keys:
                            if o not in comp and consistent(c, o):
                                comp.append(o); stack.append(o)
                    if len(comp) > 1 and not all(consistent(x, y) for x in comp for y in comp):
                        comp = [k]
                done.update(comp)
                shortest = min(comp, key=lambda c: len(parts[c]))
                n = len(parts[shortest])
                obj = ", ".join([x.strip() for x in members[shortest][0].split(",") if x.strip()][:n])
                _o, rel, subj, _s = members[shortest]
                claims.append((Triple(obj=obj, rel=rel, subj=subj), set().union(*(members[c][3] for c in comp)), sorted(comp)))
            # 1b. one place, two qualifiers: 'Matale, Sri Lanka' and 'Matale, Central Province' are one claim
            #     when ONE item named Matale lies in both (Wikidata); the claim is then the name alone
            if kind == "name" and self.containment is not None:
                claims = self._same_head(claims)
            # 2. entailment
            sup = [set(c[1]) for c in claims]
            supk = [set(c[2]) for c in claims]
            for i, ci in enumerate(claims):
                for j, cj in enumerate(claims):
                    if i != j and self._finer(kind, cj[0].obj, ci[0].obj):
                        sup[i] |= cj[1]; supk[i] |= set(cj[2])
            wit = [self.witnesses(sup[i], supk[i], claims[i][0]) for i in range(len(claims))]
            att = [i for i in range(len(claims)) if wit[i] >= self.min_sites]
            if not att:
                out.extend((c[0], c[1], c[2], wit[i]) for i, c in enumerate(claims))   # nothing attested: every value, held
                continue
            # 3. one chain -> the finest; otherwise the split, for the walk to refuse
            chain = all(self._finer(kind, claims[x][0].obj, claims[y][0].obj) or self._finer(kind, claims[y][0].obj, claims[x][0].obj)
                        for x in att for y in att if x != y)
            if chain:
                i = next(i for i in att if not any(j != i and self._finer(kind, claims[j][0].obj, claims[i][0].obj) for j in att))
                others = [c[0].obj for j, c in enumerate(claims) if j != i and not self._finer(kind, claims[i][0].obj, c[0].obj)
                          and not self._finer(kind, c[0].obj, claims[i][0].obj)]
                self._dissent[_key(claims[i][0])] = others
                if others and self.dissent_blocks:
                    out.extend((c[0], sup[j], c[2], wit[j]) for j, c in enumerate(claims))   # a veto: the walk meets them all and refuses
                else:
                    out.append((claims[i][0], sup[i], claims[i][2], wit[i]))
            else:
                out.extend((claims[i][0], sup[i], claims[i][2], wit[i]) for i in att)
        return out

    def _same_head(self, claims: list) -> list:
        by_head: dict[str, list[int]] = {}
        for i, (t, _s, _m) in enumerate(claims):
            by_head.setdefault(_fold(t.obj.split(",")[0]), []).append(i)
        merged, gone = [], set()
        for head, idx in by_head.items():
            if len(idx) < 2:
                continue
            quals = {_fold(q) for i in idx for q in claims[i][0].obj.split(",")[1:] if _fold(q)}
            if quals and any(quals <= anc for anc in self.items_up(claims[idx[0]][0].obj.split(",")[0])):
                t0 = claims[idx[0]][0]
                merged.append((Triple(obj=t0.obj.split(",")[0].strip(), rel=t0.rel, subj=t0.subj),
                               set().union(*(claims[i][1] for i in idx)), sorted({k for i in idx for k in claims[i][2]})))
                gone.update(idx)
        return [c for i, c in enumerate(claims) if i not in gone] + merged

    def witnesses(self, sites: set[str], keys: set[str], t: Triple, n: int = 5) -> int:
        """How many INDEPENDENT witnesses `sites` are: sites whose evidence sentences share an n-word run
        (the fact's own words aside) copied one text, and count once. Live, 2026-09-24: four sites gave Agha
        Hashar Kashmiri's death as 28 April 1935 in the same words -- an old Wikipedia lead, since corrected."""
        fact_words = set(normalize(t.subj).split()) | set(normalize(t.obj).split()) | set(normalize(t.rel).split())
        fact_words |= {w for k in keys for e in self.tally.get(k, {}).values() for w in normalize(e.get("obj", "")).split()}
        shingles: dict[str, set[tuple]] = {}
        for site in sites:
            sh = set()
            for k in keys:
                sent = (self.tally.get(k, {}).get(site) or {}).get("sentence") or ""
                words = [w for w in normalize(sent).split() if w not in fact_words and not w.isdigit()]
                sh |= {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
            shingles[site] = sh
        parent = {x: x for x in sites}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        ss = sorted(sites)
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                if shingles[a] & shingles[b]:
                    parent[find(a)] = find(b)
        return len({find(x) for x in sites})

    def _wikipedia_lead(self, entity: str) -> dict | None:
        """The live English Wikipedia lead for the entity (the REST summary, redirects followed), or None for a
        missing page or a disambiguation page -- then the family's mirrors are witnesses as before."""
        title = urllib.parse.quote(entity.strip().replace(" ", "_"), safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

        def make():
            self.calls["wikipedia"] += 1
            got = self.http(url, {"Accept": "application/json"})
            if got is None:
                return {}
            try:
                d = json.loads(got[1].decode("utf-8", "replace"))
            except ValueError:
                return {}
            if d.get("type") != "standard" or not d.get("extract"):
                return {}
            return {"extract": d["extract"], "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page") or url}
        got = self._cached("wplead", url, make)
        return got or None

    # -- the facts -----------------------------------------------------------------------------------
    def queries(self, entity: str, rels: list[str]) -> list[str]:
        """The entity, quoted, with the relation's label; then (with `second_query`) the way a person would
        type it -- 'Marie Curie born'. Two phrasings bring more sites, which is what a second witness needs."""
        qs = [f'"{entity}" {rels[0]}' if rels else f'"{entity}"']
        if rels and self.second_query:
            w = QUERY_WORDS.get(normalize(rels[0]))
            if w is None:
                w = next((x for x in self.aliases.wordings(rels[0]) if x.islower() and len(x) > 3
                          and normalize(x) != normalize(rels[0])), None)
            if w:
                qs.append(f"{entity} {w}")
        return qs

    def facts(self, entity: str, relations: list[str] | None = None, via: str | None = None, **_kw) -> list[Triple]:
        rels = [r for r in (relations or []) if r]
        qs = self.queries(entity, rels)
        hits, seen_urls = [], set()
        for q in qs:
            for h in self._search(q):
                if h.url not in seen_urls:
                    seen_urls.add(h.url); hits.append(h)
        self.times = {}
        self.last = {"entity": entity, "queries": qs, "backend": self.backend.name, "urls": [h.url for h in hits],
                     "license": getattr(self.backend, "license", None),
                     "wikidata_item": None, "ambiguous_items": None, "pages": 0, "robots_skipped": 0, "reads": 0,
                     "facts": {}}
        out: list[Triple] = []

        # 1. Wikipedia -> Wikidata: structured, attested by Wikidata itself
        if self.wikidata is not None:
            items = []
            for h in hits:
                host = (urllib.parse.urlsplit(h.url).hostname or "")
                if host.endswith("wikipedia.org"):
                    qid = self._qid(h.url)
                    if qid and qid not in items:
                        items.append(qid)
            carrying = []
            for qid in items[:3]:
                got = list(self.wikidata.facts(entity, relations=rels or None, via=via, qid=qid))
                if not rels or any(normalize(t.rel) in {normalize(r) for r in rels} for t in got):
                    carrying.append((qid, got, dict(getattr(self.wikidata, "times", {}) or {})))
            if len(carrying) == 1:
                qid, got, times = carrying[0]
                self.last["wikidata_item"] = qid
                for t in got:
                    k = _key(t)
                    self.provenance[k] = "wikidata (via web)"
                    self.held.discard(k)
                    self.last["facts"][k] = {"sites": ["wikidata.org"], "readers": ["wikidata"], "held": False}
                    out.append(t)
                self.times.update(times)
            elif len(carrying) > 1:
                self.last["ambiguous_items"] = [c[0] for c in carrying]   # the question does not decide: no pick

        # 2. every other hit: its snippets, then (for the first few) its page
        evidence: dict[str, tuple[Triple, dict[str, set[str]]]] = {}
        live = self._wikipedia_lead(entity) if self.wikidata is None else None

        def credit(t: Triple, site: str, reader: str, url: str, sentence: str = "") -> None:
            kind = self.aliases.kind(t.rel)
            if kind == "date":                        # the store's form: every reader's date written one way
                iso = iso_date(t.obj)
                t = Triple(obj=iso, rel=t.rel, subj=t.subj) if iso else t
            if not plausible(kind, t.obj):
                self.last["implausible"] = self.last.get("implausible", 0) + 1
                return
            k = _key(t)
            if k not in evidence:
                evidence[k] = (t, {})
            ev_site = evidence[k][1].setdefault(site, {"readers": set(), "url": url, "sentence": (sentence or "")[:300]})
            ev_site["readers"].add(reader)
            f = self.last["facts"].setdefault(k, {})
            f.setdefault("urls", []); f.setdefault("sentences", [])
            if url not in f["urls"]:
                f["urls"].append(url)
            if sentence and len(f["sentences"]) < 3:
                f["sentences"].append(sentence[:300])

        reads_left = self.reads if (self.reader is not None and rels) else 0
        # the Wikipedia family speaks once: through Wikidata (the handoff), else through its LIVE article --
        # a mirror keeps an old revision (Alchetron's Agha Hashar Kashmiri died on a date Wikipedia has since
        # corrected), so when the live lead was read no mirror is a witness
        witnesses = [h for h in hits if site_of(h.url) is not None
                     and not (site_of(h.url) == "wikipedia.org" and (self.wikidata is not None or live is not None))]
        if live is not None:
            for t, snt in frame_facts(entity, live["extract"]):
                credit(t, "wikipedia.org", "frames", live["url"], snt)
        # the pages, fetched side by side (the first few witnesses, a couple spare for the ones that fail)
        want = [h.url for h in witnesses[: self.pages + 2]]
        pages_by_url: dict[str, dict | None] = {}
        if want:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(want)))) as pool:
                pages_by_url = dict(zip(want, pool.map(self._page, want)))
        fetched = 0
        for h in witnesses:
            site = site_of(h.url)             # a witness; Wikipedia spoke through Wikidata above
            snippet = " ".join(untruncate(x) for x in [h.snippet] + list(h.extra) if x)
            for t, s in frame_facts(entity, snippet):
                credit(t, site, "frames", h.url, s)
            text, lang, jsonld = snippet, "en", []
            page = pages_by_url.get(h.url) if fetched < self.pages else None
            if page and page.get("robots"):
                self.last["robots_skipped"] += 1
            elif page:
                fetched += 1
                text, lang, jsonld = page.get("text", ""), page.get("lang", "") or "en", page.get("jsonld", [])
                for t in jsonld_facts(entity, jsonld):
                    credit(t, site, "jsonld", h.url)
                for t, s in frame_facts(entity, text, lang):
                    credit(t, site, "frames", h.url, s)
            if reads_left > 0:
                for passage in (passages(entity, text, limit=1) or ([snippet] if mentions(entity, snippet) else [])):
                    if reads_left <= 0:
                        break
                    reads_left -= 1
                    self.calls["read"] += 1; self.last["reads"] += 1
                    for rel in rels[:1]:
                        for t in self.reader.read(entity, rel, passage):
                            credit(t, site, "lfm", h.url, passage)
        self.last["pages"] = fetched

        # 3. the tally: every site that stated each fact, today and on earlier days, with its evidence
        for k, (t, sites) in evidence.items():
            seen = self.tally.setdefault(k, {})
            for site, got in sites.items():
                old = seen.get(site) or {}
                seen[site] = {"readers": sorted(set(old.get("readers", [])) | got["readers"]),
                              "url": old.get("url") or got["url"], "sentence": old.get("sentence") or got["sentence"],
                              "t": old.get("t") or round(time.time(), 1), "obj": t.obj, "rel": t.rel, "subj": t.subj}
        # 4. corroborate, one value per claim: 'London, England, UK' on one site and 'London, England' on
        # another are one claim (live, 2026-09-24: Ada Lovelace's birthplace, two sites, never agreeing)
        for t, sites, members, witnesses in self._claims(evidence):
            k = _key(t)
            if k not in self.held and self.provenance.get(k, "").startswith("wikidata"):
                continue                      # already attested by Wikidata above
            self.provenance[k] = "+".join(f"web:{x}" for x in sorted(sites))
            attested = witnesses >= self.min_sites
            if attested:
                self.held.discard(k)
            else:
                self.held.add(k)
            f = self.last["facts"].setdefault(k, {})
            f.update(sites=sorted(sites), witnesses=witnesses, held=not attested, members=members if len(members) > 1 else None,
                     dissent=self._dissent.get(k) or None,
                     readers=sorted({r for m in members for e in self.tally.get(m, {}).values() for r in e.get("readers", [])}))
            if all(normalize(o.obj) != normalize(t.obj) or normalize(o.rel) != normalize(t.rel) for o in out):
                out.append(t)
        self._save_tally()
        n_att = sum(1 for t in out if _key(t) not in self.held)
        self.last["how"] = (f"{self.backend.name}: {len(hits)} results, {fetched} pages read"
                            + (f", {self.last['reads']} passages read by the local model" if self.last["reads"] else "")
                            + (f", Wikidata item {self.last['wikidata_item']}" if self.last["wikidata_item"] else "")
                            + f"; {len(out)} facts, {n_att} attested, {len(out) - n_att} held for a second site")
        return out[: self.max_facts]
