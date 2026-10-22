"""serve_post -- every POST route of the stand-in server (split out of serve_api.py, order 0030): the board's
plugin actions, /ask/feedback, /turn, /ask, /pac/say. `handle_post(h, brain)` runs on the request handler `h`.

Wired: STANDALONE (stand-in; serve_api.make_handler).
"""
from __future__ import annotations

import json

from serve_events import _json_safe

__wiring__ = "STANDALONE"


def handle_post(h, brain) -> None:
    if h.path.startswith("/board/plugins/") and getattr(brain, "board", False):   # order 0028: approve / pin
        from plugins_serve import board_post
        n = int(h.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(h.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            body = {}
        h._send(*board_post(brain, h.path.split("?", 1)[0], h._who(), body))
        return
    if h.path == "/ask/feedback":             # the asker on one of the loop's records (AskLoop.feedback)
        loop = getattr(brain, "ask_loop", None)
        if loop is None:
            h._send(503, {"error": "no ask loop mounted: start serve_api with --ask"})
            return
        try:
            n = int(h.headers.get("Content-Length", 0))
            req = json.loads(h.rfile.read(n) or b"{}")
            h._send(200, loop.feedback(str(req.get("id") or ""), verdict=req.get("verdict") or None,
                                          relation=req.get("relation") or None,
                                          asker=(str(req.get("asker")) if req.get("asker") else None)))
        except ValueError as e:
            h._send(400, {"error": str(e)[:300]})
        except Exception as e:
            h._send(500, {"error": str(e)[:300]})
        return
    if h.path not in ("/turn", "/ask", "/pac/say"):
        h._send(404, {"error": "unknown path"})
        return
    from audience import for_user
    try:
        n = int(h.headers.get("Content-Length", 0))
        req = json.loads(h.rfile.read(n) or b"{}")
        text = str(req.get("text", "")).strip()
        if not text:
            h._send(400, {"error": "empty text"})
            return
        if h.path == "/pac/say":              # somebody at the side of the maze (standin/pacman.py
            live = getattr(brain, "pac_live", None)   # LivePac.say): straight to the coach, no routing
            if live is None:
                h._send(404, {"error": "no pac maze mounted (start with --pacman)"})
                return
            h._send(200, _json_safe(live.say(text)))
            return
        if h.path == "/ask":                  # the verified-program loop on one natural question (standin/ask.py);
            loop = getattr(brain, "ask_loop", None)   # every step streams to /loop/stream -> the panel
            if loop is None:
                h._send(503, {"error": "no ask loop mounted: start serve_api with --ask"})
                return
            if getattr(brain, "plugins", None) is not None:      # order 0028: wake what the question needs
                from plugins_serve import on_question
                on_question(brain, text)
            rec = loop.ask(text, item=(str(req.get("item")) if req.get("item") else None),
                           asker=(str(req.get("asker")) if req.get("asker") else None), choose=req.get("choose") or None)
            if getattr(brain, "plugins", None) is not None:
                from plugins_serve import on_question
                on_question(brain, text, rec)
            h._send(200, _json_safe(rec if h._who() == "audit" else for_user(rec)))   # sources stay internal
            return
        harness = getattr(brain, "harness", None)   # --harness: the turn is a frame, a budget and a ledger row
        woke = []
        if getattr(brain, "plugins", None) is not None:  # order 0028: a turn's plugins (the talk role) first
            from plugins_serve import on_question
            woke = on_question(brain, text)
        rec = (harness.turn(text, feedback=req.get("feedback")) if harness is not None
               else brain.turn(text, feedback=req.get("feedback")))
        rec.pop("raw", None)
        if woke:
            rec["plugins"] = [f"plugin:{n}" for n in woke]
        h._send(200, _json_safe(rec if h._who() == "audit" else for_user(rec)))
    except Exception as e:
        h._send(500, {"error": str(e)[:300]})
