"""cubby_serve/build.py -- the server's brain, built (moved from standin/serve.py in order 0031; the brain itself is
cubbyllm/brain.py): the fact pool and the fast-word table into the first world, the seats (the 450M's program and
talk seats on grilly2, or the stand-in's GGUF), the wiki world and the slotted emitter's name index; and the
command line.

`--selftest N`: N val chain questions with their Facts blocks STRIPPED,
answered through the brain's own retrieval — the serve stack's own number.

  python -m cubby_serve.build --gguf standin/models/emitter_v3.Q4_K_M.gguf   # REPL
  python -m cubby_serve.build --gguf ... --selftest 25 --tag _v3
  (python standin/serve.py ... runs this too, for one cycle)

The fast-word encoder is loaded by file path from the mowm checkout (``exp_m3_domain_routing._load_semantic_words``),
as before the move: whether to port it is the 0031 report's question 8.

Wired: WIRED (serve_cli builds the server's brain with ``build_serve``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "standin"),
          os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.brain import CubbyBrain, FactStore  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402

__wiring__ = "WIRED"


# ── assembly from the real artifacts ────────────────────────────────────────
def slot_index(*worlds):
    """The names the host knows, for the slotted emitter: every subject and object the worlds' triple
    indexes hold, plus a question's own capitalised runs (`NameIndex`)."""
    from cubbyllm.reasoning.slots import NameIndex
    names: set = set()
    for w in worlds:
        idx = getattr(w, "index", None)
        for attr in ("_by_subj", "_by_obj"):
            names.update(getattr(idx, attr, {}) or {})
    return NameIndex(names)


def build_serve(gguf: str | None, table: str | None, n_store: int, exe: str | None,
                route_tau: float, n_gpu_layers: int,
                extra_facts: list[str] | None = None, talk_gguf: str | None = None,
                wiki: str | None = None, cubby450m: str | None = None, emitter_adapter: str | None = None,
                talk_adapter: str | None = None, tokenizer: str | None = None,
                talk_export: str | None = None) -> CubbyBrain:
    """`cubby450m` (2026-10-17): the export of GRL's own base takes BOTH seats -- the programs seat as
    `SlotEmitter(Cubby450mEmitter)` over the worlds' names, the talk seat as `Cubby450mTalk` when a talk
    adapter is given (else the talk GGUF, else programs-only) -- one base on grilly2, the LoRAs swapped per
    role. `gguf` then stays the A/B arm: given beside it, it is loaded into the bank's shadow as `programs`
    candidate for `emit_candidate`, never routed. `talk_export`: the talk adapter's own base when it was
    trained on another checkpoint than the emitter's (2026-10-17: the emitter on base450m_cont, talk_v2_2 on
    base450m) -- a second trunk then, 2.3 GB more, until the talk adapter is retrained on the same base."""
    from cubbyllm.worlds.fact_pool import V4_TABLE, load_sample
    from exp_m3_domain_routing import _load_semantic_words

    from cubbyllm.emit.talk import ContextualEmitter
    from emitter_llama import LlamaCppEmitter                       # the stand-in's GGUF seat (--gguf), by the name
                                                                    # the facade imports it by: one module object
    print(f"loading fact store ({n_store} sampled questions' fact pool) ...", flush=True)
    _q, _a, _h, _chains, store = load_sample(n_store, seed=0)
    if extra_facts:                                      # selftest: the walked facts must BE in the store
        seen = {" ".join(f.split()) for f in store}
        store = store + [f for f in extra_facts if " ".join(f.split()) not in seen]
    print(f"  {len(store)} facts | loading fastword table ...", flush=True)
    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(str(table or V4_TABLE))
    world = FactStore(store, enc=enc, name="facts")
    slotted = None
    if cubby450m:
        from cubbyllm.reasoning.slots import SlotEmitter
        from cubbyllm.emit.seats_450m import Cubby450mEmitter, Cubby450mTrunk
        from cubbyllm.emit.talk import Cubby450mTalk
        if not emitter_adapter:
            raise ValueError("--cubby450m needs --emitter-adapter (the programs seat)")
        split = bool(talk_adapter and talk_export and os.path.abspath(talk_export) != os.path.abspath(cubby450m))
        roles = {"programs": emitter_adapter, **({"talk": talk_adapter} if talk_adapter and not split else {})}
        print(f"loading base450m {cubby450m} on grilly2 with {sorted(roles)} ...", flush=True)
        trunk = Cubby450mTrunk(cubby450m, tokenizer or os.path.join(cubby450m, "tokenizer.json"), roles)
        slotted = SlotEmitter(Cubby450mEmitter(trunk), index=None)      # the names come once the worlds are up
        if split:
            print(f"loading the talk adapter's own base {talk_export} (a second trunk) ...", flush=True)
            talk = Cubby450mTalk(Cubby450mTrunk(talk_export, tokenizer or os.path.join(talk_export, "tokenizer.json"),
                                                {"talk": talk_adapter}))
        elif talk_adapter:
            talk = Cubby450mTalk(trunk)
        elif talk_gguf:
            print(f"loading talk stand-in {talk_gguf} (n_gpu_layers={n_gpu_layers}) ...", flush=True)
            talk = LlamaCppEmitter(talk_gguf, n_ctx=4096, n_gpu_layers=n_gpu_layers)
        else:
            talk = None
        emitter = ContextualEmitter({"programs": slotted, **({"talk": talk} if talk else {})}, default="programs")
        if gguf:                                         # the stand-in beside it, for the A/B: in shadow, never routed
            emitter.shadow("programs", LlamaCppEmitter(gguf, n_ctx=4096, n_gpu_layers=n_gpu_layers))
    else:
        print(f"loading emitter {gguf} (n_gpu_layers={n_gpu_layers}) ...", flush=True)
        emitter = LlamaCppEmitter(gguf, n_ctx=4096, n_gpu_layers=n_gpu_layers)   # v6 trained at 4096
        if talk_gguf and os.path.abspath(talk_gguf) != os.path.abspath(gguf):
            print(f"loading talk adapter {talk_gguf} (n_gpu_layers={n_gpu_layers}) ...", flush=True)
            talk = LlamaCppEmitter(talk_gguf, n_ctx=4096, n_gpu_layers=n_gpu_layers)
            emitter = ContextualEmitter({"programs": emitter, "talk": talk}, default="programs")
    brain = CubbyBrain(emitter, world, exe=exe, route_tau=route_tau)
    if wiki:                                             # the wikikg world (H-G6): lookup-first, lexical fallback, no cosine rows
        from cubbyllm.worlds.wikikg import wiki_world
        print("loading wiki world (wikikg triples -> template facts -> triple index) ...", flush=True)
        t0 = time.perf_counter()
        ww = wiki_world(path=None if wiki == "auto" else wiki)
        brain.worlds["wiki"] = ww
        print(f"  wiki world: {len(ww):,} facts, {len(ww.index):,} indexed, {time.perf_counter() - t0:.0f}s", flush=True)
    if slotted is not None:
        slotted.index = slot_index(*brain.worlds.values())
        print(f"  slot index: {len(slotted.index):,} names the host knows, plus a question's own capitalised runs", flush=True)
    return brain


def load_val_chains(n: int) -> list[dict]:
    val_path = os.path.join(ROOT, "standin", "data", "out", "emitter_sft.jsonl")
    chains = [json.loads(l) for l in open(val_path, encoding="utf-8")]
    return [r for r in chains if r["task"] == "chain" and r["split"] == "val"][:n]


def given_facts(r: dict) -> list[str]:
    if "\nFacts:" not in r["prompt"]:
        return []
    return [l[2:] for l in r["prompt"].split("\nFacts:\n", 1)[1].splitlines() if l.startswith("- ")]


def selftest(serve: CubbyBrain, chains: list[dict], tag: str) -> dict:
    """Re-answer val chain questions WITHOUT their given facts: retrieval's
    own read. Reports gold-match, what the turn would SAY, and whether
    retrieval recovered the walked facts."""
    rows, hit, ret_ok = [], 0, 0
    spoken = {"correct": 0, "dont_know": 0, "wrong": 0}
    for i, r in enumerate(chains, 1):
        question = r["prompt"].split("\nFacts:")[0].strip()
        given = given_facts(r)
        out = serve.task_answer(question)
        gold = r.get("gold")
        ok = out["vm_answer"] is not None and gold is not None and normalize(out["vm_answer"]) == normalize(str(gold))
        recovered = all(any(" ".join(g.split()) == " ".join(f.split()) for f in out["facts"]) for g in given) if given else None
        say = "correct" if (out["speak_ok"] and ok) else ("wrong" if out["speak_ok"] else "dont_know")
        spoken[say] += 1
        hit += int(ok)
        ret_ok += int(bool(recovered))
        rows.append({"question": question, "gold": gold, "vm_answer": out["vm_answer"], "gold_match": ok,
                     "grounded": out["grounded"], "speak_ok": out["speak_ok"], "spoken": say,
                     "retrieval_recovered_walked_facts": recovered,
                     "facts": out["facts"], "vm_error": out["vm_error"], "walk": out["walk"],
                     "program": out["program"], "raw": out["raw"]})
        print(f"  {i}/{len(chains)} gold_match={ok} spoken={say} retrieval_recovered={recovered} ({out['wall_s']}s)", flush=True)
    n = max(1, len(chains))
    summary = {"n": len(chains), "gold_match": hit / n,
               "retrieval_recovered_walked_facts": ret_ok / n,
               "spoken": {k: v / n for k, v in spoken.items()},
               # which weights answered (the v8 self-test record had no way to tell one adapter from two)
               "emitter": getattr(serve.emitter, "name", None),
               "adapters": serve.emitter.usage() if hasattr(serve.emitter, "usage") else None}
    print(f"\n[stand-in] serve selftest: {summary}")
    out_path = os.path.join(ROOT, "standin", "data", "out", f"serve_selftest{tag}.json")
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the PROGRAM adapter (emitter): reasoning, memory writes, forge, the game")
    ap.add_argument("--talk-gguf", default=None, help="the TALK adapter (chat, perception, thoughts); defaults to --gguf")
    ap.add_argument("--table", default=None)
    ap.add_argument("--n-store", type=int, default=2000, help="questions sampled to build the fact store")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--route-tau", type=float, default=0.30)
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pacman", action="store_true",
                    help="mount cubby-man in the cubbyverse pac maze (standin/pacman.py)")
    ap.add_argument("--wiki", nargs="?", const="auto", default=None,
                    help="mount the wikikg world (standin/data/wikikg.py): bare = the cached Hub export, or a path to triplets.parquet")
    args = ap.parse_args()
    chains = load_val_chains(args.selftest) if args.selftest else []
    extra = [f for r in chains for f in given_facts(r)]
    serve = build_serve(args.gguf, args.table, args.n_store, None, args.route_tau, args.n_gpu_layers,
                        extra_facts=extra, talk_gguf=args.talk_gguf, wiki=args.wiki)
    if args.pacman:
        from pacman import CubbyPac
        serve.mount(CubbyPac())
        print("mounted: cubby-man in the pac maze (say 'play pacman for 30')")
    if args.selftest:
        selftest(serve, chains, args.tag)
        return
    print("\nCubby [stand-in] — Ctrl+C to quit. Every reply is VM-mediated.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        rec = serve.turn(text)
        if rec["kind"] == "task":
            extra_s = f"  [task, grounded={rec['task']['grounded']}]"
        elif rec["kind"] == "learn":
            extra_s = f"  [learn, accepted={rec['learn']['accepted']}]"
        else:
            extra_s = f"  [{rec['kind']}, {rec['register']}, {rec['emotion']}]"
        print(f"cubby> {rec['reply']}{extra_s}")


if __name__ == "__main__":
    main()
