"""chat — Cubby's chat turn, mediated by a CubeLang `IAgent` program.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this).

The division of labour, which is the whole point:
  * the MODEL supplies the words — candidate replies under the current
    hormonal state (system prompt carries `affect_block(state)`);
  * the HOST filters them (voice rules: `identity.voice_ok`) and always adds
    the verbatim don't-know line as the safe candidate;
  * the VM PROGRAM (`CubbyTalk implements IAgent`, rendered per turn with the
    surviving candidates embedded as string literals) offers them through
    `ask`, and `VM::resume` REJECTS any selection that is not one of the
    offered candidates — the host may choose, never supply. `act` remembers
    the reply, `observe` records feedback.

Why the filter is host-side and not in the program (2026-08-30 probe): under
the current VM, `str.contains(...)` evaluates false and a comparison against
a function parameter is true for any input, while `check --strict` and
`validate` both report "all executing". Until that is fixed (see TODO.md,
cubelang), the program is trusted only with what it demonstrably does:
candidate offering, the selection guard, counters, `remember`.

Every number that comes out of this is [stand-in].
"""
from __future__ import annotations

import os
import re
import sys
import time

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .chat_talk import RESTING, _NOT_GROUND, _THINK_RE, _escape, _self_grounding, render_talk_program  # noqa: F401
from .chat_state import ChatState

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from .identity import (T, affect_block, derived, identity_system, load_facts,  # noqa: E402
                      guess_lang, is_identity_question, is_identity_reply, is_model_guard, voice_ok)
from ..affect.afferent import Afferent  # noqa: E402
from ..affect.neurochem import Neurochemistry, appraise  # noqa: E402
from ..affect import choosing  # noqa: E402
from ..affect import saying  # noqa: E402

__wiring__ = "STANDALONE"


class CubbyChat(ChatState):
    """One Cubby, one hormonal state, one VM-mediated turn at a time."""

    def __init__(self, emitter, facts: dict | None = None, state: dict | None = None,
                 exe: str | None = None, max_new_tokens: int = 200, appraiser=None,
                 n_candidates: int = 1) -> None:
        self.emitter = emitter
        # How many replies to draw before the filter picks. ONE by default, so
        # no turn gets slower than it was: at n=1 the filter still does the
        # thing that matters — deciding whether that one reply is worth
        # speaking against the sanctioned fallback. Above 1 it also gets to
        # choose between ways of answering, at n decode passes per turn.
        self.n_candidates = max(1, int(n_candidates))
        self.last_choice: dict = {}
        self.appraiser = appraiser                       # perception.ModelAppraiser: the trunk reads emotions
        self.facts = facts or load_facts()
        self.grounded = _self_grounding(self.facts)      # what he may assert without reaching past a source
        self.chem = Neurochemistry()                     # the real ODE (cubemind port)
        self.afferent = Afferent()                       # the user's state, and what it may do to his
        self._seen_vocab: set[str] = set()
        self.state = dict(state) if state else self._hormones()
        self.exe = exe
        self.max_new_tokens = max_new_tokens
        self.history: list[dict] = []
        self.history_turns = 1                           # v9: the previous exchange rides inline (the multi-turn chat records' shape)


    MULTI_PROMPT = {"en": "Conversation so far:\nUser: {u1}\nCubby: {a1}\nUser: {u2}",       # = build_chat_sft.MULTI_PROMPT
                    "fr": "Conversation jusqu'ici :\nUtilisateur : {u1}\nCubby : {a1}\nUtilisateur : {u2}"}


    # ── the turn ────────────────────────────────────────────────────────────
    def _guard(self, reply: str, user_text: str) -> str | None:
        """The host guards, unchanged. -> the reason it may not be spoken, or
        None if it survives. Pulled out of `candidates` so the same cascade
        runs over every sample when more than one is drawn."""
        from .identity import has_non_latin
        if has_non_latin(reply):                         # EN/FR only: a base that drifts into another script is not spoken
            return "non-latin script"
        if is_model_guard(reply):
            # the base model's own guard (refusal / "as an AI" / its maker):
            # not Cubby's rule, never spoken — only OUR guards are enforced
            return "base-model guard leaked"
        if not voice_ok(reply, self.facts):
            return "voice rule"
        if is_identity_reply(reply, self.facts) and not is_identity_question(user_text):
            # the identity SFT is the model's only chat training: it answers
            # who-it-is to anything. Off topic -> not offered; the don't-know
            # line is the sanctioned answer for what it can't do yet.
            return "identity reply to a non-identity turn"
        return None

    def candidates(self, user_text: str, grounded: set[str] | None = None) -> tuple[list[str], list[str]]:
        """-> (offered, rejected): the model's replies (think-stripped) that
        pass the host guards, plus the verbatim don't-know line in the user's
        language, which is always offered — ordered by the filter.

        THE GUARDS DECIDE WHAT MAY BE SAID; THE BODY DECIDES WHAT IS SAID.
        Those were one step until now: a reply that cleared the cascade was
        spoken, and the only alternative was the don't-know line reached by
        failing a check. A cascade is the same cascade in a quiet room and in
        a fire, so nothing that happened to Cubby could make him more careful
        about a claim — the whole affect system reached his phrasing and never
        his willingness to commit.

        Now the survivors and the fallback go through `saying.rank`, which is
        `choosing` with the same five terms the maze uses. `caution` scales
        the claim surface `saying` measures, so an alarmed body needs a better
        reply before it will speak one at all and takes the sanctioned line
        when it does not have one. Nothing here can make a guard-failing reply
        speakable: rejected candidates are never scored, and risk only ever
        subtracts (`choosing`, rule one).

        `grounded` is the hook for the verified path: specifics the caller can
        back do not count as claim surface. Free chat passes nothing and every
        specific counts, which is the correct reading of an unverified turn.
        """
        system = identity_system(self.facts, self.state)
        lang = guess_lang(user_text)
        dont_know = T(self.facts, "dont_know_line", lang)
        prompt = self.with_history(user_text)
        mod = self._modulation()

        # URGENCY NARROWS THE FIELD, and this is where breadth belongs for
        # speech: a pressed body generates fewer things to say rather than
        # weighing the same set differently. It is also the cheap direction —
        # fewer decode passes exactly when the turn is time-critical.
        draws = choosing.breadth(max(1, int(getattr(self, "n_candidates", 1))), mod)

        offered, rejected, seen = [], [], set()
        for k in range(draws):
            self._turns = getattr(self, "_turns", 0) + 1
            raw = self.emitter.emit(prompt, context={"role": "talk", "state": dict(self.state),   # the talk adapter; the state rides in c
                                                     **self.talk_context(user_text, lang)},        # the turns / the identity facts, for an adapter that renders its own format
                                    max_new_tokens=self.max_new_tokens, system=system,
                                    temperature=getattr(self, "temperature", 0.7), seed=self._turns)   # words, not programs: sample
            from ..emit.base import clean_reply
            reply = clean_reply(raw)                     # closed think block out, stray tags out, complete tool calls kept
            if not reply or reply in seen:
                continue
            seen.add(reply)
            why = self._guard(reply, user_text)
            if why is None:
                offered.append(reply)
            else:
                rejected.append(reply)
                self.last_rejection = why

        spoken = [r["reply"] for r in self.history[-8:]]
        backed = set(self.grounded) | set(grounded or ())
        offered, self.last_choice = saying.rank(offered, mod, safe=dont_know,
                                                history=spoken, grounded=backed)
        return offered, rejected

    def mediate(self, user_text: str, offered: list[str], rejected: list[str] | None = None,
                feedback: str | None = None) -> dict:
        """The VM-mediated half of a turn, shared by chat AND task replies:
        render CubbyTalk with `offered` embedded, think() ASKs, the host
        selects offered[0], resume returns it (the VM rejects anything not
        offered), act() remembers, observe() records feedback."""
        from cubbyllm.bridges import cubelang_client as cc
        t0 = time.perf_counter()
        register = derived(self.state)["register"]
        src = render_talk_program(offered)
        asked = cc.run_program_proto(src, fn="think", args=[user_text, register], exe=self.exe)
        if not asked.get("suspended"):
            raise RuntimeError(f"CubbyTalk.think did not ASK: {asked}")
        if asked["candidates"] != offered:
            raise RuntimeError("the VM offered different candidates than the host embedded")
        chosen = offered[0]                              # policy: the first surviving candidate
        res = cc.resume_program_proto(src, fn="think", args=[user_text, register], answers=[chosen], exe=self.exe)
        reply = res["result"]
        acted = cc.run_program_proto(src, fn="act", args=[reply, register], exe=self.exe)
        if feedback is not None:
            cc.run_program_proto(src, fn="observe", args=[feedback, register], exe=self.exe)
        rec = {"user": user_text, "reply": reply, "register": register, "state": dict(self.state),
               "offered": offered, "rejected": list(rejected or []), "question": asked["question"],
               "acted": acted["result"], "wall_s": round(time.perf_counter() - t0, 3)}
        self.history.append(rec)
        return rec

    def turn(self, user_text: str, feedback: str | None = None, mediate: bool = True) -> dict:
        """A chat turn. `mediate=False` (no facts at stake): the model's
        reply is spoken directly under the host guards, no VM in the loop —
        the hormones modulate it, nothing verifies it, because there is
        nothing to verify."""
        offered, rejected = self.candidates(user_text)
        if mediate:
            rec = self.mediate(user_text, offered, rejected, feedback)
        else:
            rec = {"user": user_text, "reply": offered[0], "register": derived(self.state)["register"],
                   "state": dict(self.state), "offered": offered, "rejected": rejected, "question": None,
                   "acted": None, "vm_mediated": False, "wall_s": 0.0}
            self.history.append(rec)
        # The filter's working, in the record. A threshold that moves with the
        # body is only worth having if you can see it move — `spoke_safe` is
        # the one bit that matters and the gains say why.
        rec["choice"] = dict(self.last_choice)
        rec["spoke_safe"] = bool(rec["reply"] == T(self.facts, "dont_know_line", guess_lang(user_text)))
        return rec
