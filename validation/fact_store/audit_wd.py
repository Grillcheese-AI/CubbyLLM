"""validation/fact_store/audit_wd.py -- order 0055 step 0: live Wikidata, read for the audit.

Its own cache (``$CUBBY_DATA/audit_0055_wd``), never the stack's: 0054's arms read the source cache and are
path-dependent. A generic client name, one request at a time, a pause after each live call (``SLEEP_S``), 503s retried
with backoff (``retrieval_sets.common.get``).
  search_exact(name, lang)  every item whose label or alias, in that language (or ``mul``), is the name once folded;
                            up to two pages of 50 (``truncated`` when more remain): the full set, never a top few
  entities(qids, props)     wbgetentities, 20 to a call (50 for names), redirects followed

Wired: STANDALONE (order 0055's audit).
"""
from __future__ import annotations

import os

__wiring__ = "STANDALONE"

UA = "CubbyLLM-fact-audit/0.1 (order 0055; research measurement)"
SLEEP_S = 0.1


class Wd:
    def __init__(self, cache_dir: str | None = None, offline: bool = False) -> None:
        from cubbyllm.local_paths import data_dir
        from cubbyllm.sources.wikidata import WikidataSource
        from cubby_serve.sources.http import fetcher
        self.dir = cache_dir or str(data_dir("audit_0055_wd"))
        os.makedirs(self.dir, exist_ok=True)
        self.src = WikidataSource(cache_dir=self.dir, sleep_s=SLEEP_S, offline=offline, use_choices=False,
                                  fetch=fetcher(UA, 60))
        self.offline = offline

    def _get(self, params: dict):
        from retrieval_sets.common import get
        return get(self.src, params)

    def search_exact(self, name: str, lang: str) -> tuple[list[str], bool]:
        from fact_store.audit_values import fold
        key, out, cont = fold(name), [], 0
        for _ in range(2):
            d = self._get({"action": "wbsearchentities", "search": name, "language": lang, "uselang": lang,
                           "type": "item", "limit": 50, "continue": cont}) or {}
            for r in d.get("search") or []:
                if fold((r.get("match") or {}).get("text", "")) == key and r["id"] not in out:
                    out.append(r["id"])
            cont = d.get("search-continue")
            if not cont:
                return out, False
        return out, True

    def entities(self, qids, props: str = "claims", langs: str = "en|fr|mul") -> dict:
        out, ids = {}, [q for q in dict.fromkeys(qids) if q]
        size = 20 if "claims" in props else 50                # claims are heavy; names fit 50 to a call
        for i in range(0, len(ids), size):
            chunk = ids[i:i + size]
            d = self._get({"action": "wbgetentities", "ids": "|".join(chunk), "props": props, "languages": langs}) or {}
            ents = d.get("entities") or {}
            for k, e in ents.items():
                if "missing" not in e:
                    out[k] = e
            for r in (d.get("redirects") or []) if isinstance(d.get("redirects"), list) else []:
                if r.get("to") in out:
                    out[r["from"]] = out[r["to"]]
            for q in chunk:                                   # a redirect answers under its target's id
                if q not in out:
                    hit = next((e for e in ents.values() if e.get("id") and q in (e.get("redirects") or {}).values()), None)
                    if hit:
                        out[q] = hit
        return out


__all__ = ["SLEEP_S", "UA", "Wd"]
