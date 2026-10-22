"""cubbyllm/sources/web_cache.py -- the web source's cache and politeness: searches and pages cached a day, the
sites' tally, robots.txt honoured before a page is read. Split out of web.py (order 0031) as a mixin of WebSource,
verbatim.

Wired: WIRED (WebSource).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.robotparser
from .web_pages import parse_page
from .web_search import Hit, UA

__wiring__ = "WIRED"


class WebCache:
    """WebSource's cache, tally and robots (a mixin; order 0031)."""

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
