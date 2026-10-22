"""cubbyllm/brain_reasoning.py -- the brain's reasoning cortex: plan and walk the routed world, the Facts block, the
emitter's program (the CotChain opening prefilled), the VM, the ground check and the consistency gate; a generation
cleaned, a tail matched. Split out of brain.py (order 0031), verbatim; brain.py re-exports every name.

Wired: WIRED (CubbyBrain).
"""
from __future__ import annotations

import re
import time
from cubbyllm.emit.format import answer_fn, shim_isolver, wrap_role_prompt  # noqa: E402
from cubbyllm.speech.identity import (EMITTER_SYSTEM, T, guess_lang, is_identity_question,  # noqa: E402
                                      load_facts, voice_ok)
from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question, relation_matches  # noqa: E402

__wiring__ = "WIRED"


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
def tail_matches(tail: str, t) -> bool:
    """Does fact `t` answer a one-hop tail "<rel> of <subj>"? Its subject must end the tail and its relation match
    the rest (relation_matches: word Jaccard >= 0.6)."""
    nt, ns = normalize(tail), normalize(t.subj)
    if not ns or not (nt == ns or nt.endswith(" " + ns)):
        return False
    rel = re.sub(r"\s+of$", "", nt[: len(nt) - len(ns)].strip())
    return relation_matches(rel, t.rel)


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
            # a ONE-hop parsed question names its relation and subject in the tail ("anthem of kozjaty"): the grounding
            # fact must be that relation OF that subject. Harness preflight 2026-09-29: "What is the anthem of kozjaty?"
            # was spoken "Track (song)" from a flat hit on another relation, and "What is the administrative
            # territorial entity of Atlantis?" was spoken "taipei" from another subject's fact.
            if grounded and plan is not None and plan.n_hop == 1 and not any(tail_matches(plan.tail, t) for t in hits):
                grounded = False
                trace("gate_tail", answer=vm_answer, tail=plan.tail, offered=[f"{t.rel} of {t.subj}" for t in hits])
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
