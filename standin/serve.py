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
from emitter import ContextualEmitter  # noqa: E402
from identity import (EMITTER_SYSTEM, T, guess_lang, is_identity_question,  # noqa: E402
                      load_facts, voice_ok)
from worlds import FactStore, route_world  # noqa: E402

from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question, relation_matches  # noqa: E402

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

        from cubbyllm.reasoning import events as ev                                     # every step, listened to (the control panel)
        qid = ev.emit("question", text=question, source="serve")
        res = pipeline_answer(question, retriever, run_fn, tau_vm=self.tau_vm,
                              tau_ret=self.tau_ret, top_k=self.k_facts, max_repairs=1,   # 3 -> 1: lossless on the 800-question harvest, 2.6x faster walks (rb1 run, 2026-09-03)
                              lookup=getattr(retriever, "lookup", None))                 # a FactStore looks up first (exp_m4, 2026-09-04)
        ev.emit_walk(qid, res, provenance=getattr(retriever, "provenance", None), key=getattr(retriever, "_key", None))
        ev.emit_answer(qid, res)
        facts = [h.fact for h in res.trace if h.fact]
        meta = {"walk_answer": res.answer, "walk_verified": res.verified,
                "walk_reason": res.reason, "repairs": res.repairs_used,
                "walk_sources": [h.source for h in res.trace if h.fact]}
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
        for ps, f in first[:2]:
            t = parse_fact(f)
            if t is not None:                            # an expansion fact's own score is relative to the OBJECT query,
                take([(min(float(ps), float(es)), ef) for es, ef in retriever(f"{t.obj} ", 2)])   # not the question: it inherits its parent's at most
        # v3 trained on 1-3 walked facts; past ~4 the block is out of
        # distribution and the emitter derails (echoes the list, wrong style)
        return out[: self.k_facts + 1]

    def task_answer(self, question: str, retriever, trace=None, allow_flat: bool = True) -> dict:
        t0 = time.perf_counter()
        trace = trace or (lambda kind, **d: None)
        facts, walk = self.walk_facts(question, retriever)
        trace("walk", facts=list(facts), verified=walk["walk_verified"],
              answer=walk["walk_answer"], reason=walk["walk_reason"])
        scores: list[float] = []
        if not facts and allow_flat:                     # planner miss -> flat retrieval fallback
            hits = self.gather_facts(question, retriever)
            facts, scores = [f for _, f in hits], [s for s, _ in hits]
            trace("retrieve_fallback", facts=list(facts))
        elif not facts:
            # a question retrieval was NOT confident about: no walk -> no answer.
            # Flat facts here would let the emitter ground something irrelevant.
            trace("no_walk", reason="retrieval below threshold; flat fallback disabled")
        prompt = question + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts) if facts else question
        gen = self.emitter.emit(prompt, context="programs", max_new_tokens=450, system=EMITTER_SYSTEM,
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
        # For a PARSED multi-hop question the answer must also come from a fact carrying the question's
        # FINAL relation: an exhausted walk offers hop 1's fact, and "the parent entity of the instance of
        # X" must not be answered with the instance (v6 self-test, 2026-09-03: the one wrong chain).
        plan = parse_question(question)
        final_rel = plan.relations[-1] if (plan is not None and plan.n_hop >= 2) else None
        grounded = False
        if vm_answer is not None:
            hits = [t for t in (parse_fact(f) for f in facts) if t is not None and normalize(t.obj) == normalize(vm_answer)]
            grounded = bool(hits) and (final_rel is None or any(relation_matches(final_rel, t.rel) for t in hits))
            if hits and not grounded:
                trace("gate_relation", answer=vm_answer, final_relation=final_rel, offered=[t.rel for t in hits])
            # a question the grammar did NOT parse has no relation to hold the answer to: the only evidence the offered
            # fact is about the question is retrieval's own confidence — require the grounding fact to be a hit for the
            # question at >= tau_ret (live misroute 2026-09-03: 'what are the things you learned?' was answered 'lists'
            # from a 0.137 flat hit; real fact questions the grammar misses score 0.82-0.97 on their top fact)
            if grounded and plan is None and scores:
                best = max((sc for sc, f in zip(scores, facts)
                            if (t := parse_fact(f)) is not None and normalize(t.obj) == normalize(vm_answer)), default=0.0)
                if best < self.tau_ret:
                    grounded = False
                    trace("gate_retrieval", answer=vm_answer, score=round(best, 3), tau_ret=round(self.tau_ret, 3))
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


_WH_OPEN = re.compile(r"^\W*(what|who|whom|whose|where|which|when|why|how|is|are|was|were|does|do|did|can|could|"
                      r"quel(le)?s?|qui|o[ùu]|quand|pourquoi|comment|combien|est[- ]ce|quesse|c'est quoi)\b", re.I)


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
        if "?" not in text and parse_fact(text.strip().rstrip(".")) is not None and not _WH_OPEN.match(text):
            return text.strip().rstrip(".")   # "what is the capital of france" (no '?') is a question, not a fact (2026-09-04)
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
            gen = self.emitter.emit(wrap_role_prompt(fact), context="programs", max_new_tokens=450,
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
                 tau_ret: float = ReasoningCortex.TAU_RET, appraiser=None, talk_emitter=None) -> None:
        # ONE trunk interface, theta = f(c) shaped (2026-09-03): every cortex calls `self.emitter.emit(...,
        # context=<role>)` — "programs" for the VM path (reasoning, memory writes, forge, the game),
        # "talk" for the words (chat, the perception reads, the thought verbalizer). A `ContextualEmitter`
        # resolves the role to an adapter (v8: two fine-tunes on one base — one model for both was measured
        # to interfere: identity fell in v5, forge decision in v6, arithmetic 0.825 -> 0.75 -> 0.675 as the
        # chat volume grew); a single model ignores the context; the 2B trunk will condition on it. Callers
        # never pick weights (guardrail 2: the trunk drops in behind the same interface).
        if hasattr(emitter, "resolve") and hasattr(emitter, "adapters"):   # already contextual (duck-typed: `emitter` vs `standin.emitter` import paths)
            self.emitter = emitter
        elif talk_emitter is not None and talk_emitter is not emitter:
            self.emitter = ContextualEmitter({"programs": emitter, "talk": talk_emitter}, default="programs")
        else:
            self.emitter = emitter                        # one model, both roles (pre-v8 behaviour)
        self.facts = facts or load_facts()
        self.exe = exe
        self.route_tau = float(route_tau)
        self.worlds: dict[str, object] = {"facts": retriever}   # FactStore or bare callable
        self.store_texts = store_texts if store_texts is not None else getattr(retriever, "texts", [])
        self.chat = CubbyChat(self.emitter, self.facts, exe=exe, appraiser=appraiser)   # TalkCortex + the speech exit
        self.mediate_chat = False                        # no-facts turns skip the VM (task/learn/help answers still go through the ASK)
        self.reason = ReasoningCortex(self.emitter, exe=exe, k_facts=k_facts,
                                      tau_vm=tau_vm, tau_ret=tau_ret)
        self.memory = MemoryCortex(self.emitter, self.facts, exe=exe)
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

    # ── the thalamus: what needs FACTS, and what does not ───────────────────
    # Owner's shape (2026-09-02): input neurons (sense) -> THALAMUS (this) ->
    # the routed cortex -> CubbyTalk (the LLM speaks) -> routed again -> the
    # VM, or not. A turn that needs facts goes to the VM-verified reasoning
    # path; one that does not ("how are you", a joke, an opinion) is answered
    # by the model itself, modulated by the hormones only — our host guards
    # (voice rules, no base-model guard, no bio) still apply to the words.
    _HELP = re.compile(r"^\s*(help|aide|what can you do|que sais[- ]tu faire|que peux[- ]tu faire|qu'?est[- ]ce que tu (peux|sais) faire|"
                       r"quesse que tu (peux|sais) faire|tu peux faire quoi|c'est quoi que tu (peux|sais) faire)\b", re.I)   # Quebec phrasings (live, 2026-09-04)
    _QUESTION = re.compile(r"\?|^\s*(what|who|where|which|when|how many|how much|is|are|was|were|does|did|"
                           r"quel(le)?s?|qui|o[ùu]|combien|quand|est[- ]ce que)\b|\bwhich\b|\b(is|are|was|were) called\b", re.I)   # real queries: 'perth is the capital of which australian state', 'phase change from gas to solid is called'
    # a wh-FACT opening that does not ask Cubby to produce something wins over the creative words below:
    # 'who sang the theme song to that 70s show' is a fact question, not a request for a song (natural_questions slice, 2026-09-03)
    _WH_FACT = re.compile(r"^\s*(who|whom|whose|what|which|when|where|is|are|was|were|does|did|"   # subject-first yes/no too: 'is draft day the movie based on a true story'
                          r"qui|quel(le)?s?|quand|o[ùu]|est[- ]ce que)\b", re.I)          # not do/can/could/would: those carry the opinion and creative asks
    _ASK_TO_PRODUCE = re.compile(r"\b(write|sing|tell|make|compose|give|invent|create|draw) (me|us)\b|\b(can|could|would|will) you (write|sing|tell|make|compose|invent|create|draw)\b|"
                                 r"\b(what|which) (would|do) you (think|say|prefer|like)\b|"
                                 r"\b([ée]cris|chante|raconte|invente|compose)[- ](moi|nous)\b|\bpeux[- ]tu ([ée]crire|chanter|raconter|inventer)\b", re.I)
    _NO_FACTS = re.compile(r"\b(how are you|how do you feel|how('s| is) it going|what do you think|your opinion|"
                           r"(how |what )?would you (like|want|prefer|say)|do you want|would you|"
                           r"(tu )?(voudrais|aimerais|veux)[- ]?(tu)?|"
                           r"do you (like|love|enjoy|prefer)|favou?rite|tell me a joke|joke|riddle|poem|story|"
                           r"song|rhyme|write|imagine|pretend|sing|advice|should i|what would you|cheer me up|"
                           r"comment (vas[- ]tu|[çc]a va)|que penses[- ]tu|ton avis|tu (aimes|pr[ée]f[èe]res)|"
                           r"raconte|blague|devinette|po[èe]me|histoire|chanson|[ée]cris|imagine|conseil)\b", re.I)

    _CAPABILITY = re.compile(r"\b(do you know how to|can you|could you|are you able to|do you know (how|what|about)|"
                             r"sais[- ]tu|peux[- ]tu|pourrais[- ]tu|es[- ]tu capable)\b", re.I)
    # a procedural ask ("how can we implement svd?", "how do I add this to the training data?") is not a fact lookup:
    # the VM path has nothing to ground it on and answers the don't-know line — 5 of 40 real user turns went there
    # in the v9 talk probe (2026-09-04). Talk answers it under the guards.
    _HOWTO = re.compile(r"^\W*(how|what)('s| is| are)? (the best way to\b|(do|does|can|could|should|would|might) (i|we|you|one|someone|it)\b)|"
                        r"^\W*how to\b|^\W*comment (est[- ]ce qu'on|est[- ]ce que je|on|je|peut[- ]on|faire pour|puis[- ]je|fait[- ]on)\b", re.I)

    def needs_facts(self, text: str) -> tuple[bool, str]:
        """Does answering need facts about the world? Facts: a question the
        fact grammar parses, or a wh-question that is not about Cubby himself
        or a matter of taste. No facts: identity turns, feelings, opinions,
        creative asks, small talk."""
        from cubbyllm.reasoning import parse_question
        if is_identity_question(text):
            return False, "about Cubby himself"
        if self._HOWTO.search(text):                     # "how can we implement svd?": procedural, no fact to ground -> talk
            return False, "a how-to, not a fact"       # before the grammar: "what is the best way to learn piano?" parses as a fact shape
        if parse_question(text) is not None:
            return True, "the fact grammar parses it"
        if self._CAPABILITY.search(text):                # "do you know how to write code?" (live misroute 2026-09-03): about Cubby, not the world
            return False, "about Cubby himself"
        if self._NO_FACTS.search(text) and not (self._WH_FACT.search(text) and not self._ASK_TO_PRODUCE.search(text)):
            return False, "a feeling, opinion or creative ask"
        if self._QUESTION.search(text):
            return True, "a question about the world"
        return False, "small talk"

    def route(self, text: str) -> dict:
        """Learn-detect, commands, plugin cortices, then the thalamus:
        needs facts -> reasoning (the VM path; an unknown answer is the
        don't-know line, never improvised); no facts -> talk (the model,
        hormones only). Confident retrieval also engages reasoning. The
        interface (text + state in, cortex out) is the slot the spikeybrain
        SNN port fills later — this lexical implementation makes no SNN claim."""
        fact = MemoryCortex.detect(text)
        if fact is not None:
            return {"cortex": "memory", "fact": fact, "score": 1.0, "needs_facts": True}
        if self._HELP.search(text):
            return {"cortex": "help", "score": 1.0, "needs_facts": False}
        if is_identity_question(text):                   # about Cubby himself (his name, his world, a greeting): he answers, no VM, no plugin
            return {"cortex": "talk", "score": 1.0, "needs_facts": False, "why": "about Cubby himself"}
        best_c, best_m = None, 0.0
        for name, cortex in self.cortices.items():
            m = float(cortex.match(text)) if hasattr(cortex, "match") else 0.0
            if m > best_m:
                best_c, best_m = name, m
        # a plugin takes a statement at >= 0.5 but a QUESTION only at 1.0 (its own
        # words): "how would you like a plugin to explore the web?" is a question
        # to Cubby, not a game command (live misroute, 2026-09-02)
        question = bool(self._QUESTION.search(text)) or is_identity_question(text) or bool(self._NO_FACTS.search(text))
        if best_c is not None and (best_m >= 1.0 or (best_m >= 0.5 and not question)):
            return {"cortex": best_c, "score": best_m, "needs_facts": False}
        world, score = route_world(self.worlds, text)
        tau_eff = self.chat.chem.modulate_threshold(self.route_tau)
        facts, why = self.needs_facts(text)
        if facts:
            hit = self.lookup_world(text)                # an exact hop-0 hit in a world's triple index beats any similarity (exp_m4, 2026-09-04)
            if hit is not None:
                return {"cortex": "reasoning", "world": hit, "score": 1.0, "tau_eff": round(tau_eff, 3),
                        "needs_facts": True, "why": "lookup"}
        # confident retrieval engages reasoning only for residual small talk: a definite no-facts read (about Cubby,
        # a feeling, a creative ask) stands whatever the store resembles ('do you know how to write code?' hit an
        # 'audio album ... write you a song' fact at 0.377 > tau 0.319 and was answered 'audio album', 2026-09-03)
        if facts or (score >= tau_eff and why == "small talk"):
            return {"cortex": "reasoning", "world": world, "score": score, "tau_eff": round(tau_eff, 3),
                    "needs_facts": True, "why": why if facts else "retrieval"}
        return {"cortex": "talk", "score": score, "tau_eff": round(tau_eff, 3), "needs_facts": False, "why": why}

    def lookup_world(self, text: str) -> str | None:
        """The first mounted world whose triple index holds a fact serving the question's hop 0 — exact, so it
        wins over retrieval similarity; None when the question does not parse or no world has such a fact."""
        plan = parse_question(text)
        if plan is None:
            return None
        for name, w in self.worlds.items():
            lookup = getattr(w, "lookup", None)
            if lookup is not None and lookup(plan, 0, None):
                return name
        return None

    def help_line(self, lang: str) -> str:
        games = [n for n in self.cortices]
        worlds = ", ".join(f"{n} ({len(getattr(w, 'texts', []))} facts)" for n, w in self.worlds.items())
        if lang == "fr":
            return (f"Je peux répondre à des questions sur ce que je sais ({worlds}), retenir un fait "
                    f"(« retiens que X est la capitale de Y »)"
                    + (f", et jouer : dis « explore 30 » ou « status » ({', '.join(games)})." if games else "."))
        return (f"I can answer questions about what I know ({worlds}), remember a fact "
                f"('remember that X is the capital of Y')"
                + (f", and play: say 'explore for 30' or 'status' ({', '.join(games)})." if games else "."))

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

        if cortex == "talk":                             # no facts at stake: the model speaks, hormones only
            rec = self.chat.turn(user_text, feedback, mediate=self.mediate_chat)
            rec["kind"] = "chat"
        elif cortex == "help":
            line = self.help_line("fr" if re.search(r"\b(aide|que sais)", user_text, re.I) else lang)
            rec = self.chat.mediate(user_text, [line, dont_know] if voice_ok(line, self.facts) else [dont_know],
                                    rejected=[], feedback=feedback)
            rec["kind"] = "help"
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
            task = self.reason.task_answer(user_text, self.worlds[route["world"]], trace=self.trace,
                                           allow_flat=(route.get("score", 0.0) >= route.get("tau_eff", 1.0)))
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
                extra_facts: list[str] | None = None, talk_gguf: str | None = None,
                wiki: str | None = None) -> CubbyBrain:
    from exp_m3_cot_pipeline import V4_TABLE, load_sample
    from exp_m3_domain_routing import _load_semantic_words

    from standin.emitter import ContextualEmitter, LlamaCppEmitter
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
    emitter = LlamaCppEmitter(gguf, n_ctx=4096, n_gpu_layers=n_gpu_layers)   # v6 trained at 4096
    if talk_gguf and os.path.abspath(talk_gguf) != os.path.abspath(gguf):
        print(f"loading talk adapter {talk_gguf} (n_gpu_layers={n_gpu_layers}) ...", flush=True)
        talk = LlamaCppEmitter(talk_gguf, n_ctx=4096, n_gpu_layers=n_gpu_layers)
        emitter = ContextualEmitter({"programs": emitter, "talk": talk}, default="programs")
    brain = CubbyBrain(emitter, world, exe=exe, route_tau=route_tau)
    if wiki:                                             # the wikikg world (H-G6): lookup-first, lexical fallback, no cosine rows
        from wikikg import wiki_world
        print("loading wiki world (wikikg triples -> template facts -> triple index) ...", flush=True)
        t0 = time.perf_counter()
        ww = wiki_world(path=None if wiki == "auto" else wiki)
        brain.worlds["wiki"] = ww
        print(f"  wiki world: {len(ww):,} facts, {len(ww.index):,} indexed, {time.perf_counter() - t0:.0f}s", flush=True)
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
