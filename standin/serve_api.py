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


# ---- the loop's events, listened to (2026-09-13): a ring of the last events and a queue per live client ----
import collections as _collections
import queue as _queue
import threading as _threading

_LOOP_RING: "_collections.deque[dict]" = _collections.deque(maxlen=20000)
_LOOP_CLIENTS: list = []
_LOOP_LOCK = _threading.Lock()
_LOOP_WIRED = False


def _loop_sink(ev: dict) -> None:
    with _LOOP_LOCK:
        _LOOP_RING.append(ev)
        clients = list(_LOOP_CLIENTS)
    for q in clients:
        try:
            q.put_nowait(ev)
        except _queue.Full:
            pass


def ring_times() -> dict:
    """Each loop event's time by its id, from the ring (the board's journey timings, order 0027)."""
    with _LOOP_LOCK:
        return {e["id"]: e.get("t") for e in _LOOP_RING if "id" in e}


def wire_loop_events() -> None:
    """Register the server as a listener of `cubbyllm.reasoning.events` (idempotent)."""
    global _LOOP_WIRED
    if _LOOP_WIRED:
        return
    from cubbyllm.reasoning import events as loop_events
    loop_events.add_sink(_loop_sink)
    _LOOP_WIRED = True


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
            from audience import audience, token_of
            return audience(token_of(self.headers, self.path))

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
            if ctype.startswith("text/html") and self.path.split("?", 1)[0].rstrip("/") in ("/panel", "/brain", "/map"):
                from audience import AUDIT_SHIM                # an audit page's own requests carry its token
                body = body.replace(b"<head>", b"<head>" + AUDIT_SHIM.encode("utf-8"), 1)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from audience import audit_chain, audit_only
            live = getattr(brain, "pac_live", None)
            who = self._who()
            if self.path.startswith("/board"):                # the control board (order 0027): off = 404
                from board_api import handle
                got = handle(brain, self.path, who, ring_times) if getattr(brain, "board", False) else (404, {"error": "unknown path"})
                if got is not None:
                    code, out = got
                    self._raw(code, *out) if isinstance(out, tuple) else self._send(code, out)
                    return
            if audit_only(self.path) and who != "audit":      # the panel, the event streams, the audit view
                self._send(403, {"error": "audit only"})
                return
            if self.path.startswith("/audit/"):              # the whole chain behind one answer (order 0016)
                rid = self.path.split("?", 1)[0][len("/audit/"):]
                loop, harness = getattr(brain, "ask_loop", None), getattr(brain, "harness", None)
                chain = audit_chain(loop, rid) if loop is not None else None
                if chain is None and harness is not None and rid in getattr(harness, "turns", {}):
                    chain = {"id": rid, "turn": harness.turns[rid]}
                self._send(200 if chain is not None else 404, chain or {"error": f"no record {rid}"})
            elif self.path in ("/", "/index.html"):
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
                elif self.path.startswith("/pac/music"):     # the soundtrack under the 8-bit effects
                    from pacman import load_music
                    data = load_music()
                    if data is None:
                        self._raw(404, b"no soundtrack (CB_PAC_MUSIC)", "text/plain")
                    else:
                        self._raw(200, data, "audio/mpeg")
                elif self.path.startswith("/pac/sfx/"):        # sampled effects (the level-start jingle)
                    from pacman import load_sfx
                    data = load_sfx(self.path[len("/pac/sfx/"):].split("?")[0])
                    if data is None:
                        self._raw(404, b"no such effect", "text/plain")
                    else:
                        self._raw(200, data, "audio/mpeg")
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
                loop = getattr(brain, "ask_loop", None)
                self._send(200, {"ok": True, "emitter": getattr(brain.emitter, "name", "?"),
                                 "ask": loop is not None,                                   # POST /ask mounted (the panel shows its box)
                                 "ask_source": getattr(getattr(loop, "source", None), "name", None) if loop and who == "audit" else None,
                                 "news": getattr(getattr(loop, "news", None), "name", None) if loop and who == "audit" else None,
                                 "asked": loop.calls["asked"] if loop else 0,
                                 "two_adapters": bool(getattr(brain.emitter, "is_split", False)),
                                 "adapters": (brain.emitter.usage() if hasattr(brain.emitter, "usage") else None)})
            elif self.path == "/state":
                self._send(200, brain.chat.chem.to_dict())
            elif self.path == "/worlds":
                worlds = {n: len(getattr(w, "texts", [])) for n, w in brain.worlds.items()}
                self._send(200, {"worlds": worlds if who == "audit" else {"all": sum(worlds.values())},   # no dataset names
                                 "cortices": sorted(brain.cortices)})
            elif self.path.split("?", 1)[0] in ("/panel", "/panel/"):     # the three.js control panel over the loop's events
                try:
                    body = open(os.path.join(ROOT, "dashboard", "control_panel.html"), "rb").read()
                    self._raw(200, body, "text/html; charset=utf-8")        # no-store: the panel is edited live
                except OSError:
                    self._send(404, {"error": "dashboard/control_panel.html not found"})
            elif self.path.split("?", 1)[0].rstrip("/") == "/brain":   # the 3-D brain: a prompt's journey, zone to zone
                try:
                    body = open(os.path.join(ROOT, "dashboard", "brain3d.html"), "rb").read()
                    self._raw(200, body, "text/html; charset=utf-8")
                except OSError:
                    self._send(404, {"error": "dashboard/brain3d.html not found"})
            elif self.path.rstrip("/") == "/map":            # the component map: every part, lit when it fires
                try:
                    body = open(os.path.join(ROOT, "dashboard", "component_map.html"), "rb").read()
                    self._raw(200, body, "text/html; charset=utf-8")
                except OSError:
                    self._send(404, {"error": "dashboard/component_map.html not found"})
            elif self.path.startswith("/loop/events"):       # the ring: every loop event with id > since
                try:
                    since = int(self.path.split("since=", 1)[1].split("&")[0]) if "since=" in self.path else 0
                except ValueError:
                    since = 0
                with _LOOP_LOCK:
                    evs = [e for e in _LOOP_RING if e["id"] > since]
                self._send(200, {"next": evs[-1]["id"] if evs else since, "events": evs})
            elif self.path.startswith("/loop/stream"):       # server-sent events: the backlog, then live
                q: "_queue.Queue[dict]" = _queue.Queue(maxsize=10000)
                with _LOOP_LOCK:
                    backlog = list(_LOOP_RING)
                    _LOOP_CLIENTS.append(q)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    for e in backlog:
                        self.wfile.write(f"data: {json.dumps(_json_safe(e), ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    while True:
                        try:
                            e = q.get(timeout=15)
                            self.wfile.write(f"data: {json.dumps(_json_safe(e), ensure_ascii=False)}\n\n".encode("utf-8"))
                        except _queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    with _LOOP_LOCK:
                        if q in _LOOP_CLIENTS:
                            _LOOP_CLIENTS.remove(q)
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
            if self.path == "/ask/feedback":             # the asker on one of the loop's records (AskLoop.feedback)
                loop = getattr(brain, "ask_loop", None)
                if loop is None:
                    self._send(503, {"error": "no ask loop mounted: start serve_api with --ask"})
                    return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(n) or b"{}")
                    self._send(200, loop.feedback(str(req.get("id") or ""), verdict=req.get("verdict") or None,
                                                  relation=req.get("relation") or None,
                                                  asker=(str(req.get("asker")) if req.get("asker") else None)))
                except ValueError as e:
                    self._send(400, {"error": str(e)[:300]})
                except Exception as e:
                    self._send(500, {"error": str(e)[:300]})
                return
            if self.path not in ("/turn", "/ask", "/pac/say"):
                self._send(404, {"error": "unknown path"})
                return
            from audience import for_user
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                text = str(req.get("text", "")).strip()
                if not text:
                    self._send(400, {"error": "empty text"})
                    return
                if self.path == "/pac/say":              # somebody at the side of the maze (standin/pacman.py
                    live = getattr(brain, "pac_live", None)   # LivePac.say): straight to the coach, no routing
                    if live is None:
                        self._send(404, {"error": "no pac maze mounted (start with --pacman)"})
                        return
                    self._send(200, _json_safe(live.say(text)))
                    return
                if self.path == "/ask":                  # the verified-program loop on one natural question (standin/ask.py);
                    loop = getattr(brain, "ask_loop", None)   # every step streams to /loop/stream -> the panel
                    if loop is None:
                        self._send(503, {"error": "no ask loop mounted: start serve_api with --ask"})
                        return
                    rec = loop.ask(text, item=(str(req.get("item")) if req.get("item") else None),
                                   asker=(str(req.get("asker")) if req.get("asker") else None), choose=req.get("choose") or None)
                    self._send(200, _json_safe(rec if self._who() == "audit" else for_user(rec)))   # sources stay internal
                    return
                harness = getattr(brain, "harness", None)   # --harness: the turn is a frame, a budget and a ledger row
                rec = (harness.turn(text, feedback=req.get("feedback")) if harness is not None
                       else brain.turn(text, feedback=req.get("feedback")))
                rec.pop("raw", None)
                self._send(200, _json_safe(rec if self._who() == "audit" else for_user(rec)))
            except Exception as e:
                self._send(500, {"error": str(e)[:300]})

        def log_message(self, fmt, *args):
            from audience import redact
            print(f"  api: {redact(fmt % args)}", flush=True)   # a token in a query string never reaches the log

    return Handler


def serve_http(brain, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    wire_loop_events()
    httpd = ThreadingHTTPServer((host, port), make_handler(brain))
    print(f"CubbyServe API on http://{host}:{port}  (POST /turn, GET /state /worlds /health; the control panel at /panel, "
          f"its stream at /loop/stream)")
    return httpd


def build_web(args):
    """The web source behind the ask loop's source, or None when no search backend is configured
    (standin/web_source.py). The reader is the local LFM base, on the CPU: the GPU is the emitter's."""
    from web_source import WebSource, backend_from_env
    backend = backend_from_env(args.web, args.searxng)
    if backend is None:
        return None
    path = args.web_reader
    if path == "auto":
        cand = os.path.join(ROOT, "standin", "models", "LFM2.5-2.6B.Q4_K_M.gguf")
        path = cand if os.path.exists(cand) else "off"
    reader = None
    if path and path != "off":
        try:
            import llama_cpp  # noqa: F401 -- the reader needs it; without it the web runs on the deterministic readers
            from lfm_source import LfmReader
            reader = LfmReader(path)
        except ImportError:
            print("  web reader off: llama_cpp is not installed", flush=True)
    return WebSource(backend, reader=reader)


def main():
    from serve import build_serve
    try:                                                 # say which code is running (a stale process is the usual "still broken")
        import subprocess
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        print(f"CubbyServe at git {rev or '?'}")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=None, help="the PROGRAM adapter (emitter) -- the stand-in; with --cubby450m, the A/B arm in shadow")
    ap.add_argument("--talk-gguf", default=None, help="the TALK adapter (chat, perception, thoughts); defaults to --gguf")
    ap.add_argument("--cubby450m", default=None,
                    help="GRL's own base takes the seats: the base EXPORT dir (config.json + model.safetensors + tokenizer.json), served on grilly2")
    ap.add_argument("--emitter-adapter", default=None, help="with --cubby450m: the emitter LoRA dir (the programs seat, slotted)")
    ap.add_argument("--talk-adapter", default=None, help="with --cubby450m: the talk LoRA dir (the talk seat); absent -> --talk-gguf, else programs only")
    ap.add_argument("--tokenizer", default=None, help="with --cubby450m: the bbpe128k tokenizer.json (default: the export's)")
    ap.add_argument("--talk-export", default=None,
                    help="with --talk-adapter: the base export the talk adapter was trained on, when it is not --cubby450m's (a second trunk)")
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
    ap.add_argument("--wiki", nargs="?", const="auto", default=None,
                    help="mount the wikikg world (standin/data/wikikg.py): bare = the cached Hub export, or a path to triplets.parquet")
    ap.add_argument("--ask", nargs="?", const="wikidata", default=None,
                    help="mount the verified-program loop on natural questions (POST /ask, live in /panel): the emitter "
                         "proposes, the host disposes, the VM verifies, the source fills the store through the gate -- "
                         "'wikidata' (default), 'wikidata-offline' (cache only) or 'lfm' (the local base model, latent tier)")
    ap.add_argument("--ask-max-new", type=int, default=300)
    ap.add_argument("--news", action="store_true",
                    help="mount the RSS date source for 'what happened in <date>' asks: headlines enter the "
                         "store as facts about the DATE, attributed to the publisher, through the same gate. "
                         "Only a `when` ask ever consults it.")
    ap.add_argument("--board", action="store_true",
                    help="serve the control board at /board (dashboard/board/, standin/board_api.py; order 0027); "
                         "off, every /board path answers 404")
    ap.add_argument("--news-archive", default=os.environ.get("CUBBY_NEWS_ARCHIVE") or None,
                    help="poll the feed manifest's active feeds hourly into this archive folder (standin/news_archive.py, "
                         "order 0024; default CUBBY_NEWS_ARCHIVE; off when unset)")
    ap.add_argument("--web", choices=["auto", "brave", "searxng", "off"], default="auto",
                    help="the web behind Wikidata for the ask loop: 'auto' = Brave when BRAVE_SEARCH_API_KEY is set, "
                         "else SearXNG when --searxng / CUBBY_SEARXNG_URL names an instance, else off. A web fact is "
                         "held until two independent sites state it")
    ap.add_argument("--searxng", default=None, help="a SearXNG instance's base URL (JSON output enabled)")
    ap.add_argument("--books", default=os.environ.get("CUBBY_BOOKS_INDEX") or None,
                    help="the books' index (standin/books.py; default CUBBY_BOOKS_INDEX): asked before the source, "
                         "a fact one author states spoken hedged until a second source states it")
    ap.add_argument("--world", action="append", default=[], choices=["hdc"],
                    help="a read-only world behind the wiki world for the ask loop: 'hdc' = the hdc set's Wikidata5M facts "
                         "(standin/data/hdc_world.py; CUBBY_HDC_WORLD), a changing value said 'as of' its snapshot")
    ap.add_argument("--loci", default=os.environ.get("CUBBY_LOCI_CKPT") or None,
                    help="the 450M checkpoint whose loci path reads a fact before the lookup (order 0007; CUBBY_LOCI_CKPT)")
    ap.add_argument("--tense", action="store_true",
                    help="check each answer's dates against the question's tense and year without the books too "
                         "('who rules X?' on a reign that ended is not answered); on whenever --books is")
    ap.add_argument("--web-reader", default="auto",
                    help="the local model that reads pages for the relation the walk needs: a GGUF path, 'auto' "
                         "(the LFM base in standin/models when present) or 'off'")
    _out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "standin", "data", "out")
    ap.add_argument("--history", default=os.path.join(_out, "ask_history.jsonl"),
                    help="append every ask record here (one JSON line each) for the nightly sleep cycle; '' = off")
    ap.add_argument("--learned", default=os.path.join(_out, "sleep", "learned_facts.jsonl"),
                    help="the sleep cycle's durable facts, loaded into the store at boot; '' = off")
    ap.add_argument("--skills", default=os.path.join(_out, "sleep", "skills.jsonl"),
                    help="the skill library's ledger (composition rules the sleep cycle adopted), read at boot for "
                         "'how is B related to A?'; '' = none")
    ap.add_argument("--science", default=None,
                    help="mount the science worlds built by standin/data/import_science.py (a folder of saved "
                         "worlds): their ATTESTED facts only, one world per subject area; latent facts stay out")
    ap.add_argument("--harness", nargs="?", const="", default=None,
                    help="boot through the harness (H-E14 v0.1): the owner-signed config's path, or bare for a dev "
                         "config signed here from the brain's own thresholds and the shipped gate pack; a failed "
                         "boot check refuses to serve")
    args = ap.parse_args()
    if args.ask and not args.wiki:
        args.wiki = "auto"                               # the loop walks the wiki world
    if not args.gguf and not args.cubby450m:
        ap.error("give --gguf (the stand-in) or --cubby450m <export> --emitter-adapter <dir> [--talk-adapter <dir>]")
    brain = build_serve(args.gguf, args.table, args.n_store, None, args.route_tau, args.n_gpu_layers,
                        talk_gguf=args.talk_gguf, wiki=args.wiki, cubby450m=args.cubby450m,
                        emitter_adapter=args.emitter_adapter, talk_adapter=args.talk_adapter, tokenizer=args.tokenizer,
                        talk_export=args.talk_export)
    if args.ask:
        from ask import AskLoop
        print(f"mounting the ask loop (source {args.ask}, lexicon on, news {'on' if args.news else 'off'}) ...", flush=True)
        web = build_web(args) if args.web != "off" else None
        world = brain.worlds["wiki"]
        if "hdc" in args.world:                          # the stores first: the wiki world, then the hdc world (order 0014)
            from hdc_store import HdcWorld, Stores
            world = Stores(world, HdcWorld())
        brain.ask_loop = AskLoop(brain.emitter, world=world, source=args.ask, lexicon=True,
                                 max_new=args.ask_max_new, news="rss" if args.news else None,
                                 history_path=args.history or None, web=web, skills=args.skills or None,
                                 books=args.books, tense=True if args.tense else None, loci=args.loci)
        print("  web: " + (f"{web.backend.name} behind {args.ask}, reader {'the local model' if web.reader else 'off'}; "
                           "a web fact is held until two independent sites state it" if web else
                           "off (set BRAVE_SEARCH_API_KEY or --searxng to turn it on)"), flush=True)
        if args.learned:
            n = brain.ask_loop.load_learned(args.learned)
            print(f"  {n} facts learned on earlier days, loaded from the sleep cycle's log", flush=True)
        print(f"  skills: {len(brain.ask_loop.skills)} composition rules for 'how is B related to A?'", flush=True)
        if args.history:
            print(f"  every record appended to {args.history} (the sleep cycle's input)", flush=True)
        print(f"  POST /ask {{\"text\": ...}} -> the loop's record; watch it at http://{args.host}:{args.port}/panel", flush=True)
    brain.board = bool(args.board)
    if args.board:
        print(f"control board: http://{args.host}:{args.port}/board (add ?token=<audit token> for the audit view)", flush=True)
    if args.news_archive:
        from news_archive import start_poller
        start_poller(args.news_archive, log=lambda m: print(m, flush=True))
        print(f"news archive: polling the active feeds hourly into {args.news_archive}", flush=True)
    if args.science:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
        from import_science import mount_science
        mounted = mount_science(brain, args.science)
        print(f"science worlds (attested facts only): {mounted or 'none attested yet'}", flush=True)
    if args.model_appraisal:
        from perception import ModelAppraiser
        brain.chat.appraiser = ModelAppraiser(brain.emitter, brain.facts)   # asks with context="talk"
        print("model appraisal ON: the trunk reads the emotion of every turn (sense events carry the label)")
    if args.pacman:
        from pacman import PROGRAMS_PATH, CubbyGhost, LivePac
        from ledger import Ledger
        man = CubbyGhost(memory=PROGRAMS_PATH, ledger=Ledger())   # the big game; his programs persist on disk, every VM decision in the ledger
        brain.mount(man)
        brain.pac_live = LivePac(man)
        gfl = man.env.ghost_free_levels
        print(f"mounted: cubby-man in the pac maze (ghosts, hazards, power stars, levels) — "
              f"LIVE view at http://{args.host}:{args.port}/pac (he plays while it is open)\n"
              + (f"  levels 1-{gfl} run with NO ghosts: he learns the maze first "
                 f"(CUBBYMAN_GHOST_FREE_LEVELS=0 turns the threat on at level 1)\n" if gfl else "")
              + "  he is offered what his BODY can do, not the maze's legal moves: "
                "a refused move is how he finds a wall\n"
              + f"  every component, lit when it fires: http://{args.host}:{args.port}/map\n"
              f"  his programs + reasoning: {PROGRAMS_PATH} (and .md) | "
              f"http://{args.host}:{args.port}/pac/programs.md"
              + (f" | {len(man.library.entries)} programs remembered from earlier runs"
                 if man.library.entries else ""))
    if args.harness is not None:                     # last: every door the plugins mounted is behind it too
        from harness import RefuseToServe
        from harness_wiring import boot_serve
        try:
            seats = ({"programs": args.emitter_adapter, "talk": args.talk_adapter or args.talk_gguf or args.emitter_adapter}
                     if args.cubby450m else {"programs": args.gguf, "talk": args.talk_gguf or args.gguf})
            h = boot_serve(brain, args.harness or None, adapters=seats,
                           on_event=lambda ev: brain.trace("harness", **{("event" if k == "kind" else k): v for k, v in ev.items()}))
        except RefuseToServe as e:
            print(f"REFUSED TO SERVE: {e} (the ledger row names the check)", flush=True)
            sys.exit(3)
        gen = f"generation {h.generation}" if h.generation is not None else "stores built in memory (no generation manifest)"
        print(f"harness: serving, {gen}; every turn a frame, a budget and a ledger row "
              f"(POST /turn replies carry turn_id, ledger, status, spoken)", flush=True)
    serve_http(brain, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
