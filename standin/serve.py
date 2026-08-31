"""serve — the stand-in brain, wired end to end.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this).

The turn is a brain pipeline (owner's shape, 2026-08-31):

    input ──► sense (appraisal → the cubemind neurochemistry ODE; hormones
              modulate ROUTING CAUTION via modulate_threshold, never facts)
          ──► fast route (the SNN slot: today a lexical + retrieval read —
              the interface is shaped for the spikeybrain SNN port, and this
              implementation does NOT claim to be one)
          ──► cortex router ──► the right cortex:
                MemoryCortex     "remember that …" / a bare template fact —
                                 gate (parse, contradiction, dedup) → world
                                 .add() → the trained Evt write through the
                                 VM → it is retrievable NEXT turn (live
                                 learning, host-store durable)
                ReasoningCortex  plan+walk the routed world (measured CoT
                                 pipeline, tau_vm=0.2202/tau_ret=0.5959) →
                                 Facts block → v3 emitter (CotChain opening
                                 prefilled) → VM → ground check + consistency
                                 gate (a verified walk that disagrees vetoes)
                plugin cortices  mounted MindForge-style (cubbyverse mounts
                                 its world models / game here; it imports us,
                                 never the reverse)
                TalkCortex       CubbyChat — the DEFAULT, and the only EXIT:
    every spoken reply, from ANY cortex, leaves through CubbyTalk's ASK
    (`CubbyChat.mediate`) — the VM rejects a reply it was not offered, and
    the voice rules + the verbatim don't-know line hold whatever is mounted.

`--selftest N`: N val chain questions with their Facts blocks STRIPPED,
answered through the brain's own retrieval — the serve stack's own number.

  python standin/serve.py --gguf standin/models/emitter_v3.Q4_K_M.gguf   # REPL
  python standin/serve.py --gguf ... --selftest 25 --tag _v3
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "standin"),
          os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_emitter_sft import answer_fn, shim_isolver, wrap_role_prompt  # noqa: E402
from chat import CubbyChat  # noqa: E402
from identity import EMITTER_SYSTEM, T, guess_lang, load_facts, voice_ok  # noqa: E402
from worlds import FactStore, route_world  # noqa: E402

from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact  # noqa: E402

_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)

# every v3 chain target opens exactly like this; prefilling it pins the task
# style (without it, distractor-laden Facts blocks flip the emitter into the
# event/kernel program shapes — measured 0/25 on the first selftest pass) and
# must run through `create frame: number;` — cutting it at `{` leaves the model
# mid-line and it degenerates into flattened non-syntax (second 0/25 pass)
PROGRAM_PREFIX = ("use vsa;\n\nprogram CotChain implements ISolve {\n"
                  "    public function solve(mention: str): str {\n"
                  "        create frame: number;\n")


def _clean(gen: str) -> str:
    """Strip think blocks and fences, then cut from the first real program
    line — v3 sometimes echoes the prompt as a header, and an echoed Facts
    bullet is a parse error at top level. No program line -> ""."""
    s = _THINK_RE.sub("", gen, count=1) if "</think>" in gen else gen
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    lines = s.strip().splitlines()
    for i, l in enumerate(lines):
        if l.startswith(("use ", "program ")):
            return "\n".join(lines[i:]).strip() + "\n"
    return ""


# ── the reasoning cortex ────────────────────────────────────────────────────
class ReasoningCortex:
    """plan+walk → Facts block → emitter → VM → ground + consistency gates."""

    # the measured verify/walk operating point (cot_harvest_v2:
    # validation/logs/exp_m3_cot_pipeline_v2.log); tau_vm=0.0 is degenerate —
    # the control role (sim ~0.05) then always reads as a violation and
    # every good walk gets banned
    TAU_VM = 0.22021484375
    TAU_RET = 0.595863088965416

    def __init__(self, emitter, exe: str | None = None, k_facts: int = 3,
                 tau_vm: float = TAU_VM, tau_ret: float = TAU_RET) -> None:
        self.emitter = emitter
        self.exe = exe
        self.k_facts = int(k_facts)
        self.tau_vm = float(tau_vm)
        self.tau_ret = float(tau_ret)

    def walk_facts(self, question: str, retriever) -> tuple[list[str], dict]:
        """Plan+walk through the measured CoT pipeline: the walked facts, in
        hop order, are what v3's chain prompts were trained on. Empty when
        the planner can't parse the question or the walk finds no path."""
        from cubbyllm.bridges import cubelang_client as cc

        def run_fn(source: str, fn: str) -> dict:
            return cc.run_program_proto(source, fn=fn, exe=self.exe)

        res = pipeline_answer(question, retriever, run_fn, tau_vm=self.tau_vm,
                              tau_ret=self.tau_ret, top_k=self.k_facts, max_repairs=3)
        facts = [h.fact for h in res.trace if h.fact]
        meta = {"walk_answer": res.answer, "walk_verified": res.verified,
                "walk_reason": res.reason, "repairs": res.repairs_used}
        return facts, meta

    def gather_facts(self, question: str, retriever) -> list[tuple[float, str]]:
        """Flat fallback: top-k for the question plus ONE expansion hop
        through the best facts' objects. Deduped, discovery order."""
        seen, out = set(), []

        def take(hits):
            for s, f in hits:
                key = " ".join(f.split())
                if key not in seen:
                    seen.add(key)
                    out.append((float(s), f))

        first = retriever(question, self.k_facts)
        take(first)
        for _, f in first[:2]:
            t = parse_fact(f)
            if t is not None:
                take(retriever(f"{t.obj} ", 2))
        # v3 trained on 1-3 walked facts; past ~4 the block is out of
        # distribution and the emitter derails (echoes the list, wrong style)
        return out[: self.k_facts + 1]

    def task_answer(self, question: str, retriever, trace=None) -> dict:
        t0 = time.perf_counter()
        trace = trace or (lambda kind, **d: None)
        facts, walk = self.walk_facts(question, retriever)
        trace("walk", facts=list(facts), verified=walk["walk_verified"],
              answer=walk["walk_answer"], reason=walk["walk_reason"])
        scores: list[float] = []
        if not facts:                                    # planner miss -> flat retrieval fallback
            hits = self.gather_facts(question, retriever)
            facts, scores = [f for _, f in hits], [s for s, _ in hits]
            trace("retrieve_fallback", facts=list(facts))
        prompt = question + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts) if facts else question
        gen = self.emitter.emit(prompt, max_new_tokens=450, system=EMITTER_SYSTEM,
                                prefix=PROGRAM_PREFIX)
        raw = gen
        cleaned = _clean(gen)
        program = shim_isolver(cleaned) if cleaned else ""
        trace("emit", chars=len(program), ok=bool(program))
        from cubbyllm.bridges import cubelang_client as cc
        if not program:
            vm_answer, vm_error = None, "no program emitted"
        else:
            try:
                out = cc.run_program_proto(program, fn=answer_fn(program), exe=self.exe)
                vm_answer = None if out.get("result") is None else str(out["result"])
                vm_error = None
            except cc.CubelangRunError as e:
                vm_answer, vm_error = None, str(e)[:200]
        trace("vm", answer=vm_answer, error=vm_error)
        # ground check: the VM's answer must be the object of an offered fact —
        # the emitter may only ever bind what retrieval put on the table.
        grounded = False
        if vm_answer is not None:
            objs = {normalize(t.obj) for t in (parse_fact(f) for f in facts) if t is not None}
            grounded = normalize(vm_answer) in objs
        # consistency gate: a VERIFIED walk that disagrees with the emitter is a
        # caught inconsistency — never speak the suspect answer (selftest: 5 of
        # the 6 emitter misses had a verified walk holding the gold answer)
        agrees = (walk["walk_answer"] is not None and vm_answer is not None
                  and normalize(vm_answer) == normalize(walk["walk_answer"]))
        speak_ok = grounded and (agrees or not walk["walk_verified"])
        trace("gate", grounded=grounded, walk_agrees=agrees, speak_ok=speak_ok)
        return {"kind": "task", "question": question, "facts": facts, "scores": scores,
                "walk": walk, "program": program, "raw": raw, "vm_answer": vm_answer,
                "vm_error": vm_error, "grounded": grounded, "walk_agrees": agrees,
                "speak_ok": speak_ok, "wall_s": round(time.perf_counter() - t0, 3)}


# ── the memory cortex: live learning ("remember forever" v0) ────────────────
_LEARN_EN = re.compile(r"^\s*(?:please\s+)?(?:remember|note)(?:\s+that)?\s*[:,]?\s+(.+?)\s*$", re.I)
_LEARN_FR = re.compile(r"^\s*(?:retiens|retenez|souviens-toi|rappelle-toi|note)"
                       r"(?:\s+que)?\s*[:,]?\s+(.+?)\s*$", re.I)


class MemoryCortex:
    """Gate → store → VM event write. The gate is the anti-poisoning shape's
    v0: only template facts (`X is the R of Y` — what parse_fact reads), no
    duplicates, and no contradiction of a stored fact with the same (subject,
    relation). Accepted facts go into the world's store (durable, retrievable
    next turn) and through the trained Evt path ("Record this as an event: …")
    so the write executes on the VM."""

    def __init__(self, emitter, facts: dict, exe: str | None = None) -> None:
        self.emitter = emitter
        self.facts = facts
        self.exe = exe

    @staticmethod
    def detect(text: str) -> str | None:
        """The fact to learn, or None. Explicit remember/retiens prefixes, or
        a bare statement that already parses as a template fact."""
        for rx in (_LEARN_EN, _LEARN_FR):
            m = rx.match(text)
            if m:
                return m.group(1).rstrip(".").strip()
        if "?" not in text and parse_fact(text.strip().rstrip(".")) is not None:
            return text.strip().rstrip(".")
        return None

    @staticmethod
    def contradiction(fact: str, world) -> str | None:
        """A stored fact with the same (subject, relation) but a different
        object, else None. Static so any writer (the explorer's discovery
        loop included) runs the same anti-poisoning gate."""
        t = parse_fact(fact)
        if t is None:
            return None
        for known in getattr(world, "texts", []):
            k = parse_fact(known)
            if (k is not None and normalize(k.subj) == normalize(t.subj)
                    and normalize(k.rel) == normalize(t.rel)
                    and normalize(k.obj) != normalize(t.obj)):
                return known
        return None

    def learn(self, user_text: str, fact: str, world) -> dict:
        t0 = time.perf_counter()
        lang = guess_lang(user_text)
        rec = {"kind": "learn", "fact": fact, "accepted": False, "vm_written": False,
               "reason": None, "line": None}
        if parse_fact(fact) is None:
            rec["reason"] = "unparseable"
            rec["line"] = ("For now I can only remember simple facts, like "
                           "'Quuxville is the capital of Fnordovia'." if lang == "en" else
                           "Pour l'instant je ne peux retenir que des faits simples, comme "
                           "« Quuxville is the capital of Fnordovia ».")
        elif not hasattr(world, "add"):
            rec["reason"] = "store_is_read_only"
            rec["line"] = ("I can't save new facts right now." if lang == "en"
                           else "Je ne peux pas enregistrer de nouveaux faits pour le moment.")
        elif fact in world:
            rec["reason"] = "duplicate"
            rec["line"] = ("I already know that." if lang == "en" else "Je le sais déjà.")
        else:
            clash = self.contradiction(fact, world)
            if clash is not None:
                rec["reason"] = "contradiction"
                rec["clash"] = clash
                rec["line"] = (f"That clashes with what I know: {clash}." if lang == "en"
                               else f"Cela contredit ce que je sais : {clash}.")
            else:
                world.add(fact)
                rec["accepted"] = True
                rec["vm_written"] = self._vm_write(fact)
                rec["line"] = (f"Got it — I'll remember: {fact}." if lang == "en"
                               else f"C'est noté — je retiendrai : {fact}.")
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def _vm_write(self, fact: str) -> bool:
        """The trained Evt path: "Record this as an event: {fact}" -> program
        -> VM executes (`remember`). run-proto is per-process, so the durable
        store is the host world; this proves the write EXECUTES on the VM —
        VM-side persistence is the memory-service cycle, not claimed here."""
        from cubbyllm.bridges import cubelang_client as cc
        try:
            gen = self.emitter.emit(wrap_role_prompt(fact), max_new_tokens=450,
                                    system=EMITTER_SYSTEM)
            program = shim_isolver(_clean(gen))
            if not program:
                return False
            cc.run_program_proto(program, fn=answer_fn(program), exe=self.exe)
            return True
        except Exception:
            return False


# ── the brain: sense → fast route → cortex router → cortex → CubbyTalk ─────
class CubbyBrain:
    """One Cubby: one hormonal state, mounted worlds and cortices, and one
    speech exit (CubbyTalk's ASK). `CubbyServe` is the back-compat alias."""

    def __init__(self, emitter, retriever, store_texts: list[str] | None = None,
                 facts: dict | None = None, exe: str | None = None,
                 route_tau: float = 0.30, k_facts: int = 3,
                 tau_vm: float = ReasoningCortex.TAU_VM,
                 tau_ret: float = ReasoningCortex.TAU_RET) -> None:
        self.emitter = emitter
        self.facts = facts or load_facts()
        self.exe = exe
        self.route_tau = float(route_tau)
        self.worlds: dict[str, object] = {"facts": retriever}   # FactStore or bare callable
        self.store_texts = store_texts if store_texts is not None else getattr(retriever, "texts", [])
        self.chat = CubbyChat(emitter, self.facts, exe=exe)     # TalkCortex + the speech exit
        self.reason = ReasoningCortex(emitter, exe=exe, k_facts=k_facts,
                                      tau_vm=tau_vm, tau_ret=tau_ret)
        self.memory = MemoryCortex(emitter, self.facts, exe=exe)
        self.cortices: dict[str, object] = {}                   # plugin cortices: match/handle
        self._observers: list = []
        self.events: collections.deque = collections.deque(maxlen=500)   # the console feed
        self._event_i = 0

    def trace(self, kind: str, **data) -> None:
        """One console-visible event. Bounded; `serve_api` streams these so a
        demo window can SHOW the reasoning and actions as they happen."""
        self._event_i += 1
        self.events.append({"i": self._event_i, "t": round(time.time(), 3),
                            "kind": kind, **data})

    # ── plugin mount (MindForge style: plugins sit on top, never inside) ────
    def mount(self, plugin) -> None:
        if hasattr(plugin, "bind"):                      # hand the plugin the brain it sits on
            plugin.bind(self)
        for name, world in plugin.worlds().items():
            if name in self.worlds:
                raise ValueError(f"world name already mounted: {name}")
            self.worlds[name] = world
        for name, cortex in (plugin.cortices() if hasattr(plugin, "cortices") else {}).items():
            if name in self.cortices:
                raise ValueError(f"cortex name already mounted: {name}")
            self.cortices[name] = cortex
        if hasattr(plugin, "on_turn"):
            self._observers.append(plugin)

    # ── fast route (the SNN slot) + cortex router ───────────────────────────
    def route(self, text: str) -> dict:
        """Today: learn-detect + per-world retrieval scores + the hormone-
        modulated engage threshold. The interface (text + state in, cortex
        name out) is the slot the spikeybrain SNN port fills later — this
        lexical implementation makes no SNN claim."""
        fact = MemoryCortex.detect(text)
        if fact is not None:
            return {"cortex": "memory", "fact": fact, "score": 1.0}
        best_c, best_m = None, 0.0
        for name, cortex in self.cortices.items():
            m = float(cortex.match(text)) if hasattr(cortex, "match") else 0.0
            if m > best_m:
                best_c, best_m = name, m
        if best_c is not None and best_m >= 0.5:
            return {"cortex": best_c, "score": best_m}
        world, score = route_world(self.worlds, text)
        tau_eff = self.chat.chem.modulate_threshold(self.route_tau)
        if score >= tau_eff:
            return {"cortex": "reasoning", "world": world, "score": score, "tau_eff": round(tau_eff, 3)}
        return {"cortex": "talk", "score": score, "tau_eff": round(tau_eff, 3)}

    # ── one turn ────────────────────────────────────────────────────────────
    def turn(self, user_text: str, feedback: str | None = None) -> dict:
        self.trace("user", text=user_text)
        self.chat.nudge(user_text)                       # sense: appraisal -> ODE
        self.trace("sense", signals=getattr(self.chat, "signals", None),
                   state=dict(self.chat.state), emotion=self.chat.emotion)
        lang = guess_lang(user_text)
        dont_know = T(self.facts, "dont_know_line", lang)
        route = self.route(user_text)
        self.trace("route", **route)
        cortex = route["cortex"]

        if cortex == "talk":
            rec = self.chat.turn(user_text, feedback)
            rec["kind"] = "chat"
        elif cortex == "memory":
            world = self.worlds[route.get("world", "facts")]
            learn = self.memory.learn(user_text, route["fact"], world)
            self.trace("learn", fact=learn["fact"], accepted=learn["accepted"],
                       reason=learn["reason"], vm_written=learn["vm_written"])
            offered = ([learn["line"], dont_know] if learn["line"] and voice_ok(learn["line"], self.facts)
                       else [dont_know])
            rec = self.chat.mediate(user_text, offered, rejected=[], feedback=feedback)
            rec.update({"kind": "learn", "learn": learn})
        elif cortex == "reasoning":
            task = self.reason.task_answer(user_text, self.worlds[route["world"]], trace=self.trace)
            offered = ([task["vm_answer"], dont_know] if task["speak_ok"] else [dont_know])
            rec = self.chat.mediate(user_text, offered, rejected=[], feedback=feedback)
            rec.update({"kind": "task", "task": task})
        else:                                            # a mounted plugin cortex
            res = self.cortices[cortex].handle(user_text) or {}
            raw = [c for c in res.get("offered", []) if c and voice_ok(c, self.facts)]
            rec = self.chat.mediate(user_text, raw + [dont_know], rejected=[], feedback=feedback)
            rec.update({"kind": f"plugin:{cortex}", "plugin": res.get("meta")})

        rec["route"] = route
        rec["signals"] = getattr(self.chat, "signals", None)
        rec["emotion"] = self.chat.emotion
        self.trace("speak", turn_kind=rec["kind"], reply=rec["reply"],
                   offered=len(rec.get("offered", [])), register=rec.get("register"))
        for obs in self._observers:
            try:
                obs.on_turn(dict(rec))
            except Exception as e:                       # a plugin must not kill the turn
                rec.setdefault("plugin_errors", []).append(f"{getattr(obs, 'name', obs)}: {e}")
        return rec

    # back-compat for the selftest and tests
    def task_answer(self, question: str) -> dict:
        return self.reason.task_answer(question, self.worlds["facts"])


CubbyServe = CubbyBrain


# ── assembly from the real artifacts ────────────────────────────────────────
def build_serve(gguf: str, table: str | None, n_store: int, exe: str | None,
                route_tau: float, n_gpu_layers: int,
                extra_facts: list[str] | None = None) -> CubbyBrain:
    from exp_m3_cot_pipeline import V4_TABLE, load_sample
    from exp_m3_domain_routing import _load_semantic_words

    from standin.emitter import LlamaCppEmitter
    print(f"loading fact store ({n_store} sampled questions' fact pool) ...", flush=True)
    _q, _a, _h, _chains, store = load_sample(n_store, seed=0)
    if extra_facts:                                      # selftest: the walked facts must BE in the store
        seen = {" ".join(f.split()) for f in store}
        store = store + [f for f in extra_facts if " ".join(f.split()) not in seen]
    print(f"  {len(store)} facts | loading fastword table ...", flush=True)
    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(str(table or V4_TABLE))
    world = FactStore(store, enc=enc, name="facts")
    print(f"loading emitter {gguf} (n_gpu_layers={n_gpu_layers}) ...", flush=True)
    emitter = LlamaCppEmitter(gguf, n_ctx=2048, n_gpu_layers=n_gpu_layers)
    return CubbyBrain(emitter, world, exe=exe, route_tau=route_tau)


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
               "spoken": {k: v / n for k, v in spoken.items()}}
    print(f"\n[stand-in] serve selftest: {summary}")
    out_path = os.path.join(ROOT, "standin", "data", "out", f"serve_selftest{tag}.json")
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--table", default=None)
    ap.add_argument("--n-store", type=int, default=2000, help="questions sampled to build the fact store")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--route-tau", type=float, default=0.30)
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pacman", action="store_true",
                    help="mount cubby-man in the cubbyverse pac maze (standin/pacman.py)")
    args = ap.parse_args()
    chains = load_val_chains(args.selftest) if args.selftest else []
    extra = [f for r in chains for f in given_facts(r)]
    serve = build_serve(args.gguf, args.table, args.n_store, None, args.route_tau, args.n_gpu_layers,
                        extra_facts=extra)
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
