"""standin/board_api.py -- the control board's routes (order 0027): one static page and read-only JSON.

``serve_api.py --board`` (default off: every ``/board`` path answers 404) mounts it:

    GET /board                     the page (``dashboard/board/index.html``)
    GET /board/static/<file>       its CSS and JS (``dashboard/board/``)
    GET /board/assets/<file>       the section renders (``dashboard/assets/sections/``: stills and animated WebPs)
    GET /board/home                per section: built (three cards) or waiting (the order it waits for)
    GET /board/journey/<id>        one answer's steps and the question's spans (``board_data.journey``)
    GET /board/system              orders, the news poller, copy-or-refuse rates, training runs
    GET /board/watch | worlds | forecast | mood | agents   "not built yet (order NNNN)" until their orders are done

The audience is 0016's, decided on the server from the request's token: the public one (``user``) never gets a
source, a site or a dataset's name. Each response is built for its audience, and before a public one leaves, every
string in it is checked against the internal names the server knows (the feeds and their sites, the store's source
names, the mounted worlds): a hit is replaced by ``[internal]``. The page's Public switch only changes which audience
it asks as; it can never show what the server did not send.

Wired: WIRED (``serve_api.py``, behind ``--board``).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

__wiring__ = "WIRED"

ROOT = Path(__file__).resolve().parents[1]
PAGE, ASSETS = ROOT / "dashboard" / "board", ROOT / "dashboard" / "assets" / "sections"
STATIC_EXT = {".css": "text/css", ".js": "text/javascript", ".html": "text/html", ".svg": "image/svg+xml"}
ASSET_EXT = {".webp": "image/webp", ".png": "image/png"}
_SAFE = re.compile(r"^[\w.-]+$")
_cache: dict[str, tuple[float, object]] = {}


def cached(key: str, ttl: float, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def internal_names(brain) -> set[str]:
    """Every name 0016 keeps internal that a board response could carry, lowercased."""
    from audience import SOURCE_NAMES
    names = set(SOURCE_NAMES)
    try:
        from news_archive import feeds
        for f in feeds():
            names.add(f["id"])
            names |= {h.replace("www.", "") for h in f.get("publisher_hosts") or []}
            host = re.sub(r"^https?://", "", f.get("url", "")).split("/")[0]
            names.add(host.replace("www.", ""))
    except Exception:                                    # noqa: BLE001 -- no manifest: the other names still count
        pass
    names |= {str(w) for w in (getattr(brain, "worlds", {}) or {})}
    return {n.lower() for n in names if n and len(n) > 3}


def scrub(o, names: set[str]):
    """A public response's every string checked against the internal names: a hit becomes ``[internal]``."""
    if isinstance(o, dict):
        return {k: scrub(v, names) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(v, names) for v in o]
    if isinstance(o, str):
        low = o.lower()
        return "[internal]" if any(n in low for n in names) else o
    return o


def _file(base: Path, name: str, types: dict):
    if not _SAFE.match(name or "") or Path(name).suffix.lower() not in types:
        return None
    p = base / name
    return (p.read_bytes(), types[p.suffix.lower()]) if p.is_file() else None


def handle(brain, path: str, who: str, ring_times=None):
    """``(code, payload)`` for a JSON route, ``(code, (bytes, type))`` for a file; None when ``path`` is not the board's."""
    import board_data as bd
    route = path.split("?", 1)[0].rstrip("/")
    if route == "/board":
        got = _file(PAGE, "index.html", STATIC_EXT)
        return (200, got) if got else (404, {"error": "dashboard/board/index.html not found"})
    if route.startswith("/board/static/"):
        got = _file(PAGE, route[len("/board/static/"):], STATIC_EXT)
        return (200, got) if got else (404, {"error": "not found"})
    if route.startswith("/board/assets/"):
        got = _file(ASSETS, route[len("/board/assets/"):], ASSET_EXT)
        return (200, got) if got else (404, {"error": "not found"})
    if route == "/board/home":
        out = cached(f"home:{who}", 5, lambda: bd.home(brain, who))
    elif route == "/board/system":
        out = cached(f"system:{who}", 5, lambda: bd.system(brain, who))
    elif route.startswith("/board/journey/"):
        rid = route[len("/board/journey/"):]
        loop = getattr(brain, "ask_loop", None)
        rec = next((r for r in reversed(getattr(loop, "history", []) or []) if r.get("id") == rid), None)
        if rec is None:
            return 404, {"error": f"no record {rid}"}
        out = bd.journey(rec, (ring_times or (lambda: {}))(), getattr(loop, "_forms", None), who)
    elif route in ("/board/watch", "/board/worlds", "/board/forecast", "/board/mood", "/board/agents"):
        sid = route.rsplit("/", 1)[1]
        out = {"section": sid, **bd.empty(next(o for s, _n, _d, o in bd.SECTIONS if s == sid))}
    elif route.startswith("/board"):
        return 404, {"error": "unknown board path"}
    else:
        return None
    out = json.loads(json.dumps(out, default=str))
    if who != "audit":
        out = scrub(out, cached("names", 60, lambda: internal_names(brain)))
    return 200, {**out, "audience": "audit" if who == "audit" else "public"}

