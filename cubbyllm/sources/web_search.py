"""cubbyllm/sources/web_search.py -- the web source's search backends (Brave, SearXNG) and their keys, the hit they
return, and the transport the package does not have (``_no_http``: the server hands one in). Split out of web.py (order
0031), verbatim; web.py re-exports every name.

Wired: WIRED (the web source).
"""
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field

__wiring__ = "WIRED"


UA = "cubbyllm-standin/0.1 (research; licensing@grillcheese.ai)"


TTL_S = 24 * 3600                  # a search or a page is re-fetched at most daily


MAX_PAGE_BYTES = 1_500_000


ENV_FILES = (pathlib.Path(__file__).resolve().parents[2] / "standin" / ".env", pathlib.Path(__file__).resolve().parents[2] / ".env")


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


def _no_http(url: str, headers: dict | None = None, timeout_s: float = 10, limit: int = MAX_PAGE_BYTES):
    """The default transport: none. A transport, ``(url, headers, timeout_s, limit) -> (content_type, bytes) or None``
    (never raising: a site that is down is not an answer), is handed in: the server's is
    ``cubby_serve.sources.http.web``. This package opens no socket (order 0031)."""
    raise RuntimeError("the web source needs http= (cubby_serve.sources.http.web): cubbyllm/ opens no socket")


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

    def __init__(self, key: str | None = None, http=_no_http, rate_per_s: float = 50.0) -> None:
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

    def __init__(self, base_url: str | None = None, http=_no_http) -> None:
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


def backend_from_env(kind: str = "auto", searxng_url: str | None = None, http=_no_http):
    """'brave' | 'searxng' | 'auto' (Brave when a key is set, else SearXNG when an instance is named,
    else None: the loop runs without the web, as before)."""
    if kind in ("brave", "auto") and env_key("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"):
        return BraveBackend(http=http)
    if kind in ("searxng", "auto") and (searxng_url or env_key("CUBBY_SEARXNG_URL")):
        return SearxngBackend(searxng_url, http=http)
    if kind in ("brave", "searxng"):
        raise ValueError(f"--web {kind}: " + ("set BRAVE_SEARCH_API_KEY" if kind == "brave" else "set CUBBY_SEARXNG_URL or --searxng"))
    return None
