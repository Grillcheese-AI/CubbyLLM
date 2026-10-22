"""serve_api — the HTTP surface a mounted front-end (cubbyverse) talks to.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this).

Transport only — the brain stays `serve.CubbyBrain`; this is the thin shell
the plugin rule allows (cubbyverse's web brain reads /state for its emotion
compass and posts /turn for chat; it implements `worlds.CubbyPlugin` in its
own repo for anything deeper). stdlib http.server, JSON, CORS open for the
local demo. NOT hardened for the public internet — bind localhost.

  POST /turn   {"text": "...", "feedback": "..."?}   -> the turn record
  POST /ask/feedback {"id": record id, "verdict": "right"|"wrong"?, "relation": "..."?, "asker": "..."?}
               -> the asker on an ask record; a relation stated for a relation ask is an episode for the
               skill library at night (cubbyllm/reasoning/skills.py)
               (reply, kind, register, emotion, state, route, task/learn)
  GET  /state  -> the full neurochemistry read (hormones, valence, arousal,
               emotion, receptor sensitivity) — the compass feed
  GET  /worlds -> mounted worlds and their sizes; mounted cortices
  GET  /health -> {"ok": true, "emitter": name}
  GET  /brain  -> dashboard/brain3d.html: a prompt's journey through the brain's zones, live, in 3-D

  python standin/serve_api.py --gguf standin/models/emitter_v3.Q4_K_M.gguf --port 8765
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

__wiring__ = "STANDALONE"


from cubby_serve.events import (_LOOP_CLIENTS, _LOOP_LOCK, _LOOP_RING, _json_safe, _loop_sink,  # noqa: E402,F401
                                ring_times, wire_loop_events)


def make_handler(brain):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CubbyServe/0.1"

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(_json_safe(payload)).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Audit-Token")
            self.end_headers()

        def _who(self) -> str:
            """'user' (the default) or 'audit' (the token in CUBBY_AUDIT_TOKEN; order 0016)."""
            from cubbyllm.speech.audience import audience, token_of
            return audience(token_of(self.headers, self.path))

        def _page(self, name: str) -> None:
            page = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standin")), name)
            try:
                body = open(page, "rb").read()
            except OSError:
                self._send(404, {"error": f"{name} not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _raw(self, code: int, body: bytes, ctype: str) -> None:
            if ctype.startswith("text/html") and self.path.split("?", 1)[0].rstrip("/") in ("/panel", "/brain", "/map"):
                from cubbyllm.speech.audience import AUDIT_SHIM                # an audit page's own requests carry its token
                body = body.replace(b"<head>", b"<head>" + AUDIT_SHIM.encode("utf-8"), 1)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from cubby_serve.get import handle_get
            handle_get(self, brain)

        def do_POST(self):
            from cubby_serve.post import handle_post
            handle_post(self, brain)

        def log_message(self, fmt, *args):
            from cubbyllm.speech.audience import redact
            print(f"  api: {redact(fmt % args)}", flush=True)   # a token in a query string never reaches the log

    return Handler


def serve_http(brain, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    wire_loop_events()
    httpd = ThreadingHTTPServer((host, port), make_handler(brain))
    print(f"CubbyServe API on http://{host}:{port}  (POST /turn, GET /state /worlds /health; the control panel at /panel, "
          f"its stream at /loop/stream)")
    return httpd


def main():
    try:                                                 # say which code is running (a stale process is the usual "still broken")
        import subprocess
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        print(f"CubbyServe at git {rev or '?'}")
    except Exception:
        pass
    from cubby_serve.cli import boot, parser
    ap = parser()
    args = ap.parse_args()
    brain = boot(args, ap)
    serve_http(brain, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
