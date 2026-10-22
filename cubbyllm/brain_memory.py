"""cubbyllm/brain_memory.py -- the brain's memory cortex: "remember that ..." (or a bare template fact) gated (parse,
contradiction, dedup), added to the world and written through the VM, retrievable next turn. Split out of brain.py
(order 0031), verbatim; brain.py re-exports every name.

Wired: WIRED (CubbyBrain).
"""
from __future__ import annotations

import re
import time
from cubbyllm.emit.format import answer_fn, shim_isolver, wrap_role_prompt  # noqa: E402
from cubbyllm.speech.identity import (EMITTER_SYSTEM, T, guess_lang, is_identity_question,  # noqa: E402
                                      load_facts, voice_ok)
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question, relation_matches  # noqa: E402
from .brain_reasoning import _clean

__wiring__ = "WIRED"


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
            elif world.add(fact) is False and getattr(world, "last_refusal", None):
                # the worlds layer refused it (a CapsuleStore with an oracle): which world, on what member
                ref = world.last_refusal
                rec["reason"] = "impossible"
                rec["possibility"] = ref
                clash = (ref.get("support") or [[None]])[0][0]
                rec["line"] = (f"That can't be, as far as the {ref['world']} world knows"
                               + (f": {clash}." if clash else ".") if lang == "en"
                               else f"Ce n'est pas possible, d'après le monde {ref['world']}"
                               + (f" : {clash}." if clash else "."))
            else:
                rec["accepted"] = True
                m = getattr(world, "meta", {}).get(" ".join(fact.split()))   # a CapsuleStore says how it was judged
                if m and m.get("latent"):
                    rec["latent"] = m["latent"]
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
