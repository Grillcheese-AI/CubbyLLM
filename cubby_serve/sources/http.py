"""cubby_serve/sources/http.py -- a GET over HTTP, for the package's sources (order 0031: ``cubbyllm/`` opens no socket;
each source takes a ``fetch`` and the server hands it one of these). Each fetcher sends the User-Agent and uses the
timeout its source sent and used before the split.

Wired: WIRED (the server builds the online Wikidata source with ``wikidata()``, the web source with ``web``, the news source with ``news()``).
"""
from __future__ import annotations

import urllib.request

__wiring__ = "WIRED"


def get(url: str, ua: str, timeout: float = 30, headers: dict | None = None) -> bytes:
    """The body of a GET of ``url``; raises on failure (the source decides what a failure means). The fetcher's name
    is ``ua``: a header that would name it otherwise is refused."""
    if any(k.lower() == "user-agent" for k in (headers or {})):
        raise ValueError("the fetcher sends its own name; a User-Agent header is not taken")
    req = urllib.request.Request(url, headers={"User-Agent": ua, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(url: str, dest: str, ua: str, timeout: float = 60, chunk: int = 1 << 20) -> int:
    """``url`` streamed to the file ``dest`` (written as ``dest.part``, renamed when whole); returns the bytes written.
    For the measurement sets order 0046 reads (``validation/retrieval_sets/download.py``); a file already there is kept."""
    import os
    if os.path.exists(dest):
        return os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            n += len(b)
    os.replace(dest + ".part", dest)
    return n


def fetcher(ua: str, timeout: float = 30):
    """``url -> body`` with this User-Agent and timeout."""
    return lambda url: get(url, ua, timeout)


def wikidata():
    """The Wikidata source's fetch: its own User-Agent, 30 s."""
    from cubbyllm.sources.wikidata import UA
    return fetcher(UA, 30)


def web(url: str, headers: dict | None = None, timeout_s: float = 10, limit: int | None = None):
    """The web source's transport (moved from ``cubbyllm/sources/web._http``, verbatim): ``(content_type, bytes)`` or
    None, never raising (a site that is down is not an answer); its User-Agent; at most ``limit`` bytes (the source's
    page cap when None)."""
    from cubbyllm.sources.web import MAX_PAGE_BYTES, UA
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.headers.get("Content-Type", ""), r.read(MAX_PAGE_BYTES if limit is None else limit)
    except Exception:                                        # noqa: BLE001
        return None


def news(timeout_s: float = 20):
    """The news source's fetch: the poller's one name (``news.NAME``, ``CubbyLLM/0.1``, set by the server) and its
    timeout (20 s by default). It takes no name."""
    from .news import NAME
    return fetcher(NAME, timeout_s)
