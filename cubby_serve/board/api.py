"""standin/board_api.py -- the control board's routes (order 0027): one static page and read-only JSON.

``serve_api.py --board`` (default off: every ``/board`` path answers 404) mounts it:

    GET /board                     the page (``dashboard/board/index.html``)
    GET /board/static/<file>       its CSS and JS (``dashboard/board/``)
    GET /board/assets/<file>       the section renders (``dashboard/assets/sections/``: stills and animated WebPs)
    GET /board/home                per section: built (three cards) or waiting (the order it waits for)
    GET /board/journey/<id>        one answer's steps and the question's spans (``board_data.journey``)
    GET /board/system              orders, the news poller, copy-or-refuse rates, training runs
    GET /board/mood                today's chemistry (``/state``), a display heartbeat and the cube point; arcs wait for 0020
    GET /board/watch | worlds | forecast | agents   "not built yet (order NNNN)" until their orders are done

The audience is 0016's, decided on the server from the request's token: the public one (``user``) never gets a
source, a site or a dataset's name. Each response is built for its audience, and before a public one leaves, every
string in it is checked, word for word, against the internal names the server knows (the store's source names, the
feeds and their sites, the worlds registry's dataset names, internal plugins; not the generic keys worlds are mounted
under, "facts" or "self"): a hit is replaced by ``[internal]``. The page's Public switch only changes which audience
it asks as; it can never show what the server did not send.

``start_refresh(brain)`` (the server's ``--board``) rebuilds Home, System and the name list in a background thread, so
no request pays for reading run logs on a synced drive; a request then reads the last build.

Wired: WIRED (``serve_api.py``, behind ``--board``).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

__wiring__ = "WIRED"

ROOT = Path(__file__).resolve().parents[2]
PAGE, ASSETS = ROOT / "dashboard" / "board", ROOT / "dashboard" / "assets" / "sections"
STATIC_EXT = {".css": "text/css", ".js": "text/javascript", ".html": "text/html", ".svg": "image/svg+xml"}
ASSET_EXT = {".webp": "image/webp", ".png": "image/png"}
_SAFE = re.compile(r"^[\w.-]+$")
GENERIC = {"misc", "data", "other", "facts", "wiki", "news", "books", "self", "general", "reality", "fiction"}
_cache: dict[str, tuple[float, object]] = {}
_refreshing: set[str] = set()                           # keys the background thread keeps fresh


def cached(key: str, ttl: float, fn):
    hit = _cache.get(key)
    if hit and (key in _refreshing or time.time() - hit[0] < ttl):
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def _datasets() -> set[str]:
    """The worlds registry's dataset names (``$CUBBY_WORLDS/registry.sqlite``, order 0015), and its world ids."""
    root = os.environ.get("CUBBY_WORLDS")
    reg = Path(root or "") / "registry.sqlite"
    if not root or not reg.is_file():
        return set()
    con = sqlite3.connect(f"file:{reg.as_posix()}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute("SELECT name FROM datasets")}
    finally:
        con.close()


def internal_names(brain) -> set[str]:
    """Every name 0016 keeps internal that a board response could carry, lowercased: source names, feeds and their
    sites, dataset names, internal plugins. A world's generic mount key ("facts", "self", "reality:history") is a label,
    not a source, and is left out, with ``GENERIC``."""
    from cubbyllm.speech.audience import SOURCE_NAMES
    names = set(SOURCE_NAMES) | _datasets()
    try:
        from cubbyllm.sources.news.archive import feeds
        for f in feeds():
            names.add(f["id"])
            names |= {h.replace("www.", "") for h in f.get("publisher_hosts") or []}
            host = re.sub(r"^https?://", "", f.get("url", "")).split("/")[0]
            names.add(host.replace("www.", ""))
    except Exception:                                    # noqa: BLE001 -- no manifest: the other names still count
        pass
    from cubby_serve.plugins import internal_names as plugin_internal
    names |= plugin_internal(brain) | {f"plugin:{n}" for n in plugin_internal(brain)}
    keys = {str(w).lower() for w in (getattr(brain, "worlds", {}) or {})}
    generic = GENERIC | keys | {p for k in keys for p in k.split(":")}
    return {n.lower() for n in names if n and len(n) > 3 and n.lower() not in generic}


_rx: dict[frozenset, re.Pattern] = {}


def _pattern(names) -> re.Pattern | None:
    key = frozenset(names)
    if key not in _rx:
        alts = "|".join(re.escape(n) for n in sorted(key, key=len, reverse=True))
        _rx.clear()
        _rx[key] = re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.I) if alts else None
    return _rx[key]


def scrub(o, names: set[str]):
    """A public response's every string checked against the internal names, whole words only (a name inside a longer
    word is another word): a hit becomes ``[internal]``."""
    rx = _pattern(names)
    if rx is None:
        return o

    def go(x):
        if isinstance(x, dict):
            return {k: go(v) for k, v in x.items()}
        if isinstance(x, list):
            return [go(v) for v in x]
        if isinstance(x, str):
            return "[internal]" if rx.search(x) else x
        return x
    return go(o)


def start_refresh(brain, every: float = 5.0, stop: threading.Event | None = None) -> threading.Thread:
    """Home and System for both audiences, and the name list, rebuilt every ``every`` seconds in a daemon thread."""
    from cubby_serve.board import data as bd
    jobs = [(f"{k}:{who}", (lambda f, w: lambda: f(brain, w))(f, who)) for k, f in (("home", bd.home), ("system", bd.system))
            for who in ("user", "audit")] + [("names", lambda: internal_names(brain))]

    def run():
        while not (stop is not None and stop.is_set()):
            for key, fn in jobs:
                try:
                    _cache[key] = (time.time(), fn())
                    _refreshing.add(key)
                except Exception as e:                   # noqa: BLE001 -- a failed build keeps the last; the request rebuilds
                    _refreshing.discard(key)
                    print(f"[board] refresh {key}: {type(e).__name__}: {e}", flush=True)
            time.sleep(every)
    t = threading.Thread(target=run, name="board-refresh", daemon=True)
    t.start()
    return t


def _file(base: Path, name: str, types: dict):
    if not _SAFE.match(name or "") or Path(name).suffix.lower() not in types:
        return None
    p = base / name
    return (p.read_bytes(), types[p.suffix.lower()]) if p.is_file() else None


def handle(brain, path: str, who: str, ring_times=None):
    """``(code, payload)`` for a JSON route, ``(code, (bytes, type))`` for a file; None when ``path`` is not the board's."""
    from cubby_serve.board import data as bd
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
    elif route.startswith("/board/plugins"):                 # order 0028: the Plugins view
        from cubby_serve.plugins import board_get
        out = board_get(brain, route, path.split("?", 1)[1] if "?" in path else "", who)
        if out is None:
            return 404, {"error": "unknown board path"}
    elif route == "/board/mood":
        out = bd.mood(brain, who)
    elif route in ("/board/watch", "/board/worlds", "/board/forecast", "/board/agents"):
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

