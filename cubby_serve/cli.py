"""serve_cli -- the stand-in server's command line (split out of serve_api.py, order 0030): `parser()` its
arguments, `boot(args)` the brain they describe with everything mounted, `build_web(args)` the web source.

Wired: WIRED (cubby_serve.api.main).
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

__wiring__ = "WIRED"


def build_web(args):
    """The web source behind the ask loop's source, or None when no search backend is configured
    (standin/web_source.py). The reader is the local LFM base, on the CPU: the GPU is the emitter's."""
    from cubbyllm.sources.web import WebSource, backend_from_env
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


def parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--plugins", default=os.environ.get("CUBBY_PLUGINS") or None,
                    help="plugin folders, comma-separated (order 0028; standin/plugins_*.py): their front matter is read at "
                         "start, a plugin loads when a question needs it; off when not given")
    ap.add_argument("--plugin-idle", type=float, default=30.0, help="minutes before an unused plugin adapter unloads")
    ap.add_argument("--plugin-router", choices=["minilm", "nearest"], default="minilm",
                    help="which router wakes plugins (standin/plugins_route.py): 'minilm', the question encoder and a "
                         "logistic head per plugin (order 0028 review 2), or 'nearest', the nearest example line")
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
    _out = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "standin", "data", "out")
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
    return ap


def boot(args, ap: argparse.ArgumentParser):
    """The brain the arguments describe, everything mounted (or `ap.error` when no emitter is named)."""
    from cubby_serve.build import build_serve
    from cubby_serve.api import serve_http  # noqa: F401 -- the shell serves it
    if args.ask and not args.wiki:
        args.wiki = "auto"                               # the loop walks the wiki world
    if not args.gguf and not args.cubby450m:
        ap.error("give --gguf (the stand-in) or --cubby450m <export> --emitter-adapter <dir> [--talk-adapter <dir>]")
    brain = build_serve(args.gguf, args.table, args.n_store, None, args.route_tau, args.n_gpu_layers,
                        talk_gguf=args.talk_gguf, wiki=args.wiki, cubby450m=args.cubby450m,
                        emitter_adapter=args.emitter_adapter, talk_adapter=args.talk_adapter, tokenizer=args.tokenizer,
                        talk_export=args.talk_export)
    if args.ask:
        from cubbyllm.reasoning.ask import AskLoop
        print(f"mounting the ask loop (source {args.ask}, lexicon on, news {'on' if args.news else 'off'}) ...", flush=True)
        web = build_web(args) if args.web != "off" else None
        world = brain.worlds["wiki"]
        if "hdc" in args.world:                          # the stores first: the wiki world, then the hdc world (order 0014)
            from cubbyllm.worlds.hdc_store import HdcWorld, Stores
            world = Stores(world, HdcWorld())
        source = args.ask
        if source == "lfm":                             # the stand-in's LFM as the source (latent tier): built here,
            from lfm_source import LfmSource             # so the loop never imports the stand-in (order 0031)
            source = LfmSource(os.path.join(ROOT, "standin", "models", "LFM2.5-2.6B.Q4_K_M.gguf"))
        brain.ask_loop = AskLoop(brain.emitter, world=world, source=source, lexicon=True,
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
    brain.news_archive = args.news_archive              # the board's poller tile reads the archive this server polls
    if args.plugins:
        from cubby_serve.plugins import mount as mount_plugins
        lp = mount_plugins(brain, args.plugins, args.plugin_idle, log=lambda m: print(m, flush=True),
                           router=args.plugin_router)
        print(f"plugins: {len(lp.index)} indexed from their front matter, {len(lp.loaded)} pinned and loaded, "
              f"tau {lp.router.tau}", flush=True)
    if args.board:
        from cubby_serve.board.api import start_refresh
        start_refresh(brain)                            # Home / System built in the background: no request pays for them
        print(f"control board: http://{args.host}:{args.port}/board (add ?token=<audit token> for the audit view)", flush=True)
    if args.news_archive:
        from cubbyllm.sources.news.archive import start_poller
        start_poller(args.news_archive, log=lambda m: print(m, flush=True))
        print(f"news archive: polling the active feeds hourly into {args.news_archive}", flush=True)
    if args.science:
        from cubbyllm.worlds.science import mount_science
        mounted = mount_science(brain, args.science)
        print(f"science worlds (attested facts only): {mounted or 'none attested yet'}", flush=True)
    if args.model_appraisal:
        from cubbyllm.affect.perception import ModelAppraiser
        brain.chat.appraiser = ModelAppraiser(brain.emitter, brain.facts)   # asks with context="talk"
        print("model appraisal ON: the trunk reads the emotion of every turn (sense events carry the label)")
    if args.pacman:
        from pacman import PROGRAMS_PATH, CubbyGhost, LivePac
        from cubby_serve.ledger import Ledger
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
        from cubby_serve.harness import RefuseToServe
        from cubby_serve.harness_wiring import boot_serve
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
    return brain
