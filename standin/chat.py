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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from identity import (T, affect_block, derived, identity_system, load_facts,  # noqa: E402
                      guess_lang, is_identity_question, is_identity_reply, is_model_guard, voice_ok)
from afferent import Afferent  # noqa: E402
from neurochem import Neurochemistry, appraise  # noqa: E402

__wiring__ = "STANDALONE"

RESTING = {"dopamine": 0.25, "serotonin": 0.20, "cortisol": 0.15, "oxytocin": 0.08, "noradrenaline": 0.07}
_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)


def _escape(s: str) -> str:
    """A CubeLang string literal: no raw newlines, escaped backslashes/quotes."""
    return " ".join(s.split()).replace("\\", "\\\\").replace('"', '\\"')


def render_talk_program(candidates: list[str], question: str = "which reply") -> str:
    """`CubbyTalk implements IAgent` with the candidates embedded as literals.
    think() offers them via ASK; act() remembers the chosen reply; observe()
    counts feedback. Compiles under --strict; every op executes (validate)."""
    if not candidates:
        raise ValueError("at least one candidate (the don't-know line) is required")
    regs = [f"c{i}" for i in range(len(candidates))]
    lits = "\n".join(f'        create {r} : str;\n        assign {r} = "{_escape(c)}";' for r, c in zip(regs, candidates))
    return f'''# CubbyTalk -- one chat turn, mediated by the VM (stand-in, 2026-08-30)
program CubbyTalk implements IAgent {{
    storage {{
        turns: mutable u64 = 0;
        replies: mutable u64 = 0;
        good: mutable u64 = 0;
        bad: mutable u64 = 0;
    }}

    @system @once
    public function constructor() {{
        assign turns = 0;
        assign replies = 0;
        assign good = 0;
        assign bad = 0;
    }}

    @external
    public function think(input: str, ctx: str): str {{
        add turns, 1;
{lits}
        ask "{_escape(question)}", {", ".join(regs)};
        create chosen : str;
        pop chosen;
        return chosen;
    }}

    @external
    public function act(decision: str, ctx: str): str {{
        add replies, 1;
        create said : str;
        assign said = decision;
        remember said;
        return said;
    }}

    @external
    public function observe(result: str, ctx: str): str {{
        create seen : str;
        assign seen = result;
        remember seen;
        return seen;
    }}
}}
'''


class CubbyChat:
    """One Cubby, one hormonal state, one VM-mediated turn at a time."""

    def __init__(self, emitter, facts: dict | None = None, state: dict | None = None,
                 exe: str | None = None, max_new_tokens: int = 200, appraiser=None) -> None:
        self.emitter = emitter
        self.appraiser = appraiser                       # perception.ModelAppraiser: the trunk reads emotions
        self.facts = facts or load_facts()
        self.chem = Neurochemistry()                     # the real ODE (cubemind port)
        self.afferent = Afferent()                       # the user's state, and what it may do to his
        self._seen_vocab: set[str] = set()
        self.state = dict(state) if state else self._hormones()
        self.exe = exe
        self.max_new_tokens = max_new_tokens
        self.history: list[dict] = []
        self.history_turns = 1                           # v9: the previous exchange rides inline (the multi-turn chat records' shape)

    # ── hormones: the cubemind neurochemistry ODE is the state source ───────
    def _hormones(self) -> dict:
        d = self.chem.to_dict()
        return {h: round(d[h], 3) for h in ("dopamine", "serotonin", "cortisol",
                                            "oxytocin", "noradrenaline")}

    def set_state(self, state: dict) -> None:
        """Pin an explicit state (tests / replaying a recorded trajectory);
        the next nudge() resumes from the ODE, not from this override."""
        self.state = dict(state)

    def nudge(self, user_text: str, *, gap_s: float | None = None, burst: int = 0,
              hour: int | None = None) -> dict:
        """One message through appraisal -> the ODE (a couple of perception
        frames), then the 5-hormone slice becomes the serving state.

        THE READ AND THE TRANSFER ARE TWO STEPS, and keeping them apart is the
        point. The appraiser reports how the USER is; `afferent` decides what
        that may do to CUBBY. They used to be one step, which is how a sad user
        made Cubby sad and an angry one made him frightened.

        The split of the five drives is by what they are about:

          novelty, threat   the TASK. A message can be urgent with nobody upset,
                            and `appraise`'s threat lexicon reads exactly that.
          focus             both: the task asks something, and concern attends.
                            Whichever is larger.
          valence, social   the PERSON, so they go through the transfer and
                            come out as concern and warmth rather than as a copy.
          surge             arousal contagion, which only exists on this path.

        `gap_s`/`burst`/`hour` are the behavioural channel and do nothing at
        all until the user grants it (`self.afferent.grant()`).
        """
        if self.appraiser is not None:                   # v5: the trunk reads the emotion, its petal drives
            self.signals = self.appraiser.signals(user_text, self._seen_vocab, guess_lang(user_text))
        else:
            self.signals = appraise(user_text, self._seen_vocab)
        self._seen_vocab.update(re.findall(r"[\w']+", user_text.lower()))

        self.afferent.read(user_text, lexical_valence=self.signals.get("valence", 0.0),
                           warmth=self.signals.get("social", 0.0), gap_s=gap_s, burst=burst,
                           hour=hour, words=len(user_text.split()))
        tr = self.afferent.drives()
        self.chem.step_message({
            "novelty": self.signals.get("novelty", 0.0),
            "threat": self.signals.get("threat", 0.0),          # the task's danger, never the user's mood
            "focus": max(self.signals.get("focus", 0.0), tr["focus"]),
            "valence": tr["valence"],                           # 0.0 when they are having a bad time
            "social": tr["social"],                             # ... which is where it went instead
            "surge": tr["surge"]})
        self.state = self._hormones()
        return self.state

    def user_state(self) -> dict:
        """What Cubby holds about the person, in full. Small enough to show
        them, which is the only honest way to keep it."""
        return self.afferent.state()

    @property
    def emotion(self) -> str:
        return self.chem.dominant_emotion

    MULTI_PROMPT = {"en": "Conversation so far:\nUser: {u1}\nCubby: {a1}\nUser: {u2}",       # = build_chat_sft.MULTI_PROMPT
                    "fr": "Conversation jusqu'ici :\nUtilisateur : {u1}\nCubby : {a1}\nUtilisateur : {u2}"}

    def with_history(self, user_text: str) -> str:
        """The prompt for this turn: the previous spoken exchange inline (v9 multi-turn shape), or the bare
        turn when there is none / history is off. An identity question stands alone: it is answered from
        who Cubby is, not from the last exchange."""
        if not self.history_turns or not self.history or is_identity_question(user_text):
            return user_text
        last = self.history[-1]
        u1, a1 = str(last.get("user") or "").strip(), str(last.get("reply") or "").strip()
        if not u1 or not a1:
            return user_text
        lang = "fr" if guess_lang(user_text) == "fr" else "en"
        return self.MULTI_PROMPT[lang].format(u1=u1, a1=a1, u2=user_text)

    # ── the turn ────────────────────────────────────────────────────────────
    def candidates(self, user_text: str) -> tuple[list[str], list[str]]:
        """-> (offered, rejected): the model's reply (think-stripped) if it
        passes the voice rules, plus the verbatim don't-know line in the
        user's language, which is always offered."""
        system = identity_system(self.facts, self.state)
        self._turns = getattr(self, "_turns", 0) + 1
        prompt = self.with_history(user_text)
        raw = self.emitter.emit(prompt, context={"role": "talk", "state": dict(self.state)},   # the talk adapter; the state rides in c
                                max_new_tokens=self.max_new_tokens, system=system,
                                temperature=getattr(self, "temperature", 0.7), seed=self._turns)   # words, not programs: sample
        from emitter import clean_reply
        reply = clean_reply(raw)                         # closed think block out, stray tags out, complete tool calls kept
        lang = guess_lang(user_text)
        dont_know = T(self.facts, "dont_know_line", lang)
        offered, rejected = [], []
        from identity import has_non_latin
        if reply and has_non_latin(reply):               # EN/FR only: a base that drifts into another script is not spoken
            rejected.append(reply)
            self.last_rejection = "non-latin script"
        elif reply and is_model_guard(reply):
            # the base model's own guard (refusal / "as an AI" / its maker):
            # not Cubby's rule, never spoken — only OUR guards are enforced
            rejected.append(reply)
            self.last_rejection = "base-model guard leaked"
        elif reply and not voice_ok(reply, self.facts):
            rejected.append(reply)
            self.last_rejection = "voice rule"
        elif reply and is_identity_reply(reply, self.facts) and not is_identity_question(user_text):
            # the identity SFT is the model's only chat training: it answers
            # who-it-is to anything. Off topic -> not offered; the don't-know
            # line is the sanctioned answer for what it can't do yet.
            rejected.append(reply)
            self.last_rejection = "identity reply to a non-identity turn"
        elif reply:
            offered.append(reply)
        offered.append(dont_know)
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
            return self.mediate(user_text, offered, rejected, feedback)
        rec = {"user": user_text, "reply": offered[0], "register": derived(self.state)["register"],
               "state": dict(self.state), "offered": offered, "rejected": rejected, "question": None,
               "acted": None, "vm_mediated": False, "wall_s": 0.0}
        self.history.append(rec)
        return rec
