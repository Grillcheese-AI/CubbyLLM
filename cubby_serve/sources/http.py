"""cubby_serve/sources/http.py -- a GET over HTTP, for the package's sources (order 0031: ``cubbyllm/`` opens no socket;
each source takes a ``fetch`` and the server hands it one of these). Each fetcher sends the User-Agent and uses the
timeout its source sent and used before the split.

Wired: WIRED (the server builds the online Wikidata source with ``wikidata()``).
"""
from __future__ import annotations

import urllib.request

__wiring__ = "WIRED"


def get(url: str, ua: str, timeout: float = 30, headers: dict | None = None) -> bytes:
    """The body of a GET of ``url``; raises on failure (the source decides what a failure means)."""
    req = urllib.request.Request(url, headers={"User-Agent": ua, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetcher(ua: str, timeout: float = 30):
    """``url -> body`` with this User-Agent and timeout."""
    return lambda url: get(url, ua, timeout)


def wikidata():
    """The Wikidata source's fetch: its own User-Agent, 30 s."""
    from cubbyllm.sources.wikidata import UA
    return fetcher(UA, 30)
