"""serve_get -- every GET route of the stand-in server (split out of serve_api.py, order 0030): the board, the
audit chain, the pac maze, /health /state /worlds, the dashboards, the loop's events and stream.
`handle_get(h, brain)` runs on the request handler `h` (its `_send`, `_raw`, `_page`, `_who`).

Wired: WIRED (cubby_serve.api's handler).
"""
from __future__ import annotations

import json
import os
import queue as _queue
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cubby_serve.events import _LOOP_CLIENTS, _LOOP_LOCK, _LOOP_RING, _json_safe, ring_times  # noqa: E402

__wiring__ = "WIRED"


def handle_get(h, brain) -> None:
    from cubbyllm.speech.audience import audit_chain, audit_only
    live = getattr(brain, "pac_live", None)
    who = h._who()
    if h.path.startswith("/board"):                # the control board (order 0027): off = 404
        from cubby_serve.board.api import handle
        got = handle(brain, h.path, who, ring_times) if getattr(brain, "board", False) else (404, {"error": "unknown path"})
        if got is not None:
            code, out = got
            h._raw(code, *out) if isinstance(out, tuple) else h._send(code, out)
            return
    if audit_only(h.path) and who != "audit":      # the panel, the event streams, the audit view
        h._send(403, {"error": "audit only"})
        return
    if h.path.startswith("/audit/"):              # the whole chain behind one answer (order 0016)
        rid = h.path.split("?", 1)[0][len("/audit/"):]
        loop, harness = getattr(brain, "ask_loop", None), getattr(brain, "harness", None)
        chain = audit_chain(loop, rid) if loop is not None else None
        if chain is None and harness is not None and rid in getattr(harness, "turns", {}):
            chain = {"id": rid, "turn": harness.turns[rid]}
        h._send(200 if chain is not None else 404, chain or {"error": f"no record {rid}"})
    elif h.path in ("/", "/index.html"):
        h._page("demo.html")
    elif h.path.startswith("/pac"):           # pacman_live's exact frontend + its protocol
        if live is None:
            h._send(404, {"error": "no pac maze mounted (start with --pacman)"})
        elif h.path in ("/pac", "/pac/"):
            html = live.frontend()
            if html is None:
                h._send(404, {"error": "pacman_live.py frontend not found (cubbyverse checkout)"})
            else:
                h._raw(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif h.path.startswith("/pac/init"):
            h._send(200, live.init_payload())
        elif h.path.startswith("/pac/state"):
            h._send(200, live.poll())
        elif h.path.startswith("/pac/next"):
            h._send(200, live.next_level())
        elif h.path.startswith("/pac/programs"):   # his notebook: the programs + reasoning
            lib = live.man.library
            if h.path.endswith(".md"):
                h._raw(200, lib.notebook().encode("utf-8"), "text/markdown; charset=utf-8")
            else:
                h._send(200, {"path": str(lib.path) if lib.path else None,
                                 "entries": lib.entries})
        elif h.path.startswith("/pac/music"):     # the soundtrack under the 8-bit effects
            from pacman import load_music
            data = load_music()
            if data is None:
                h._raw(404, b"no soundtrack (CB_PAC_MUSIC)", "text/plain")
            else:
                h._raw(200, data, "audio/mpeg")
        elif h.path.startswith("/pac/sfx/"):        # sampled effects (the level-start jingle)
            from pacman import load_sfx
            data = load_sfx(h.path[len("/pac/sfx/"):].split("?")[0])
            if data is None:
                h._raw(404, b"no such effect", "text/plain")
            else:
                h._raw(200, data, "audio/mpeg")
        elif h.path.startswith("/pac/assets/"):
            from pacman import load_asset
            data = load_asset(h.path[len("/pac/assets/"):])
            if data is None:
                h._raw(404, b"not found", "text/plain")
            else:
                h._raw(200, data, "image/png")
        else:
            h._send(404, {"error": "unknown path"})
    elif h.path == "/health":
        loop = getattr(brain, "ask_loop", None)
        h._send(200, {"ok": True, "emitter": getattr(brain.emitter, "name", "?"),
                         "ask": loop is not None,                                   # POST /ask mounted (the panel shows its box)
                         "ask_source": getattr(getattr(loop, "source", None), "name", None) if loop and who == "audit" else None,
                         "news": getattr(getattr(loop, "news", None), "name", None) if loop and who == "audit" else None,
                         "asked": loop.calls["asked"] if loop else 0,
                         "two_adapters": bool(getattr(brain.emitter, "is_split", False)),
                         "adapters": (brain.emitter.usage() if hasattr(brain.emitter, "usage") else None)})
    elif h.path == "/state":
        h._send(200, brain.chat.chem.to_dict())
    elif h.path == "/worlds":
        worlds = {n: len(getattr(w, "texts", [])) for n, w in brain.worlds.items()}
        h._send(200, {"worlds": worlds if who == "audit" else {"all": sum(worlds.values())},   # no dataset names
                         "cortices": sorted(brain.cortices)})
    elif h.path.split("?", 1)[0] in ("/panel", "/panel/"):     # the three.js control panel over the loop's events
        try:
            body = open(os.path.join(ROOT, "dashboard", "control_panel.html"), "rb").read()
            h._raw(200, body, "text/html; charset=utf-8")        # no-store: the panel is edited live
        except OSError:
            h._send(404, {"error": "dashboard/control_panel.html not found"})
    elif h.path.split("?", 1)[0].rstrip("/") == "/brain":   # the 3-D brain: a prompt's journey, zone to zone
        try:
            body = open(os.path.join(ROOT, "dashboard", "brain3d.html"), "rb").read()
            h._raw(200, body, "text/html; charset=utf-8")
        except OSError:
            h._send(404, {"error": "dashboard/brain3d.html not found"})
    elif h.path.rstrip("/") == "/map":            # the component map: every part, lit when it fires
        try:
            body = open(os.path.join(ROOT, "dashboard", "component_map.html"), "rb").read()
            h._raw(200, body, "text/html; charset=utf-8")
        except OSError:
            h._send(404, {"error": "dashboard/component_map.html not found"})
    elif h.path.startswith("/loop/events"):       # the ring: every loop event with id > since
        try:
            since = int(h.path.split("since=", 1)[1].split("&")[0]) if "since=" in h.path else 0
        except ValueError:
            since = 0
        with _LOOP_LOCK:
            evs = [e for e in _LOOP_RING if e["id"] > since]
        h._send(200, {"next": evs[-1]["id"] if evs else since, "events": evs})
    elif h.path.startswith("/loop/stream"):       # server-sent events: the backlog, then live
        q: "_queue.Queue[dict]" = _queue.Queue(maxsize=10000)
        with _LOOP_LOCK:
            backlog = list(_LOOP_RING)
            _LOOP_CLIENTS.append(q)
        try:
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream; charset=utf-8")
            h.send_header("Cache-Control", "no-store")
            h.send_header("Access-Control-Allow-Origin", "*")
            h.end_headers()
            for e in backlog:
                h.wfile.write(f"data: {json.dumps(_json_safe(e), ensure_ascii=False)}\n\n".encode("utf-8"))
            h.wfile.flush()
            while True:
                try:
                    e = q.get(timeout=15)
                    h.wfile.write(f"data: {json.dumps(_json_safe(e), ensure_ascii=False)}\n\n".encode("utf-8"))
                except _queue.Empty:
                    h.wfile.write(b": ping\n\n")
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with _LOOP_LOCK:
                if q in _LOOP_CLIENTS:
                    _LOOP_CLIENTS.remove(q)
    elif h.path.startswith("/events"):
        try:
            since = int(h.path.split("since=", 1)[1].split("&")[0]) if "since=" in h.path else 0
        except ValueError:
            since = 0
        evs = [e for e in list(brain.events) if e["i"] > since]
        h._send(200, {"next": evs[-1]["i"] if evs else since, "events": evs})
    else:
        h._send(404, {"error": "unknown path"})
