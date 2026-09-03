"""serve_api — the HTTP surface a mounted front-end (cubbyverse) talks to.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this).

Transport only — the brain stays `serve.CubbyBrain`; this is the thin shell
the plugin rule allows (cubbyverse's web brain reads /state for its emotion
compass and posts /turn for chat; it implements `worlds.CubbyPlugin` in its
own repo for anything deeper). stdlib http.server, JSON, CORS open for the
local demo. NOT hardened for the public internet — bind localhost.

  POST /turn   {"text": "...", "feedback": "..."?}   -> the turn record
               (reply, kind, register, emotion, state, route, task/learn)
  GET  /state  -> the full neurochemistry read (hormones, valence, arousal,
               emotion, receptor sensitivity) — the compass feed
  GET  /worlds -> mounted worlds and their sizes; mounted cortices
  GET  /health -> {"ok": true, "emitter": name}

  python standin/serve_api.py --gguf standin/models/emitter_v3.Q4_K_M.gguf --port 8765
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


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
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _page(self, name: str) -> None:
            page = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
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
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            live = getattr(brain, "pac_live", None)
            if self.path in ("/", "/index.html"):
                self._page("demo.html")
            elif self.path.startswith("/pac"):           # pacman_live's exact frontend + its protocol
                if live is None:
                    self._send(404, {"error": "no pac maze mounted (start with --pacman)"})
                elif self.path in ("/pac", "/pac/"):
                    html = live.frontend()
                    if html is None:
                        self._send(404, {"error": "pacman_live.py frontend not found (cubbyverse checkout)"})
                    else:
                        self._raw(200, html.encode("utf-8"), "text/html; charset=utf-8")
                elif self.path.startswith("/pac/init"):
                    self._send(200, live.init_payload())
                elif self.path.startswith("/pac/state"):
                    self._send(200, live.poll())
                elif self.path.startswith("/pac/next"):
                    self._send(200, live.next_level())
                elif self.path.startswith("/pac/programs"):   # his notebook: the programs + reasoning
                    lib = live.man.library
                    if self.path.endswith(".md"):
                        self._raw(200, lib.notebook().encode("utf-8"), "text/markdown; charset=utf-8")
                    else:
                        self._send(200, {"path": str(lib.path) if lib.path else None,
                                         "entries": lib.entries})
                elif self.path.startswith("/pac/assets/"):
                    from pacman import load_asset
                    data = load_asset(self.path[len("/pac/assets/"):])
                    if data is None:
                        self._raw(404, b"not found", "text/plain")
                    else:
                        self._raw(200, data, "image/png")
                else:
                    self._send(404, {"error": "unknown path"})
            elif self.path == "/health":
                self._send(200, {"ok": True, "emitter": getattr(brain.emitter, "name", "?")})
            elif self.path == "/state":
                self._send(200, brain.chat.chem.to_dict())
            elif self.path == "/worlds":
                self._send(200, {"worlds": {n: len(getattr(w, "texts", []))
                                            for n, w in brain.worlds.items()},
                                 "cortices": sorted(brain.cortices)})
            elif self.path.startswith("/events"):
                try:
                    since = int(self.path.split("since=", 1)[1].split("&")[0]) if "since=" in self.path else 0
                except ValueError:
                    since = 0
                evs = [e for e in list(brain.events) if e["i"] > since]
                self._send(200, {"next": evs[-1]["i"] if evs else since, "events": evs})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/turn":
                self._send(404, {"error": "unknown path"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                text = str(req.get("text", "")).strip()
                if not text:
                    self._send(400, {"error": "empty text"})
                    return
                rec = brain.turn(text, feedback=req.get("feedback"))
                rec.pop("raw", None)
                self._send(200, rec)
            except Exception as e:
                self._send(500, {"error": str(e)[:300]})

        def log_message(self, fmt, *args):
            print(f"  api: {fmt % args}", flush=True)

    return Handler


def serve_http(brain, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(brain))
    print(f"CubbyServe API on http://{host}:{port}  (POST /turn, GET /state /worlds /health)")
    return httpd


def main():
    from serve import build_serve
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--table", default=None)
    ap.add_argument("--n-store", type=int, default=2000)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--route-tau", type=float, default=0.30)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pacman", action="store_true",
                    help="mount cubby-man in the cubbyverse pac maze (standin/pacman.py)")
    ap.add_argument("--model-appraisal", action="store_true",
                    help="the trunk reads each turn's emotion (the v5 task); its Plutchik petal drives the hormones")
    args = ap.parse_args()
    brain = build_serve(args.gguf, args.table, args.n_store, None, args.route_tau, args.n_gpu_layers)
    if args.model_appraisal:
        from perception import ModelAppraiser
        brain.chat.appraiser = ModelAppraiser(brain.emitter, brain.facts)
        print("model appraisal ON: the trunk reads the emotion of every turn (sense events carry the label)")
    if args.pacman:
        from pacman import PROGRAMS_PATH, CubbyGhost, LivePac
        man = CubbyGhost(memory=PROGRAMS_PATH)           # the big game; his programs persist on disk
        brain.mount(man)
        brain.pac_live = LivePac(man)
        print(f"mounted: cubby-man in the pac maze (ghosts, hazards, power stars, levels) — "
              f"LIVE view at http://{args.host}:{args.port}/pac (he plays while it is open)\n"
              f"  his programs + reasoning: {PROGRAMS_PATH} (and .md) | "
              f"http://{args.host}:{args.port}/pac/programs.md"
              + (f" | {len(man.library.entries)} programs remembered from earlier runs"
                 if man.library.entries else ""))
    serve_http(brain, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
