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
import urllib.robotparser
from dataclasses import dataclass, field
from html.parser import HTMLParser

from cubbyllm.reasoning.planner import Triple, normalize

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .web_search import (BraveBackend, ENV_FILES, Hit, MAX_PAGE_BYTES, SearxngBackend, TTL_S, UA, _TAG,  # noqa: F401
                         _no_http, _plain, backend_from_env, env_key)
from .web_pages import (FOLD, NOT_WITNESSES, PAGE_TYPES, SCHEMA, _ISO_DATE, _Page, _TWO_LEVEL, _nodes,  # noqa: F401
                        _values, jsonld_facts, parse_page, site_of)
from .web_frames import (QUERY_WORDS, WEB_FRAMES_EN, WEB_PLACE_FRAMES, _DMY, _FR_MONTHS, _HATNOTE, _MDY,  # noqa: F401
                         _MNUM, _MNUM_FR, _MONTHS, _SENT, _VALUE, _fold, _key, _opens_with, frame_facts, iso_date,
                         mentions, passages, plausible, untruncate)
from .web_cache import WebCache
from .web_wikidata import WebWikidata
from .web_witness import WebWitness

__wiring__ = "WIRED"

CACHE = pathlib.Path(__file__).resolve().parents[2] / "standin" / "data" / "out" / "web_cache"
MIN_SITES = 2                      # independent sites before a web fact may carry an answer


class WebSource(WebWitness, WebWikidata, WebCache):
    """`facts(entity, relations=, via=)` -> Triples about the entity from the web. Every triple's
    provenance names the sites that stated it (`provenance[normalize(fact)]`); `held` holds the ones
    fewer than `min_sites` independent sites stated -- `learn_and_answer` stores those in the latent
    tier. `last` carries the run: queries, urls, readers, and per fact which sites said it and how."""

    name = "web"
    latent = False                     # per fact: see `held`

    def __init__(self, backend, reader=None, wikidata=None, aliases=None, cache_dir: pathlib.Path | str | None = CACHE,
                 ttl_s: int = TTL_S, offline: bool = False, results: int = 20, pages: int = 6, reads: int = 4,
                 min_sites: int = MIN_SITES, http=_no_http, robots: bool = True, max_facts: int = 60,
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
            from .wikidata import PropertyAliases
            self._aliases = PropertyAliases()
        return self._aliases

    def relations(self, text: str) -> list[str]:
        return self.aliases.relations(text)

    def kind(self, label: str) -> str | None:
        return self.aliases.kind(label)

    def wordings(self, label: str) -> list[str]:
        return self.aliases.wordings(label)


    # -- corroboration -------------------------------------------------------------------------------
    CONTAINED_BY = ("P131", "P17")        # located in the administrative entity; country


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
