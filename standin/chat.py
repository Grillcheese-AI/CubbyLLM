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
                      guess_lang, voice_ok)

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
                 exe: str | None = None, max_new_tokens: int = 200) -> None:
        self.emitter = emitter
        self.facts = facts or load_facts()
        self.state = dict(state or RESTING)
        self.exe = exe
        self.max_new_tokens = max_new_tokens
        self.history: list[dict] = []

    # ── hormones (host-owned; a real host passes neurochemistry.to_dict()) ──
    def set_state(self, state: dict) -> None:
        self.state = dict(state)

    def nudge(self, user_text: str) -> dict:
        """A tiny, transparent stand-in for the host's neurochemistry: decay
        toward resting, bump noradrenaline on shouting/urgency, oxytocin on
        thanks/greetings. Replace with the real ODE at serve time."""
        st = {h: round(v + 0.5 * (RESTING[h] - v), 3) for h, v in self.state.items()}
        t = user_text
        if t.isupper() and len(t) > 3 or "!!" in t or re.search(r"\b(urgent|now|hurry|asap)\b", t, re.I):
            st["noradrenaline"] = min(0.9, st["noradrenaline"] + 0.4)
        if re.search(r"\b(thanks|thank you|merci|please|s'il te plaît|hi|hello|bonjour|salut)\b", t, re.I):
            st["oxytocin"] = min(0.85, st["oxytocin"] + 0.3)
        self.state = st
        return st

    # ── the turn ────────────────────────────────────────────────────────────
    def candidates(self, user_text: str) -> tuple[list[str], list[str]]:
        """-> (offered, rejected): the model's reply (think-stripped) if it
        passes the voice rules, plus the verbatim don't-know line in the
        user's language, which is always offered."""
        system = identity_system(self.facts, self.state)
        raw = self.emitter.emit(user_text, max_new_tokens=self.max_new_tokens, system=system)
        reply = _THINK_RE.sub("", raw, count=1).strip() if "</think>" in raw else raw.strip()
        lang = guess_lang(user_text)
        dont_know = T(self.facts, "dont_know_line", lang)
        offered, rejected = [], []
        if reply and voice_ok(reply, self.facts):
            offered.append(reply)
        elif reply:
            rejected.append(reply)
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

    def turn(self, user_text: str, feedback: str | None = None) -> dict:
        offered, rejected = self.candidates(user_text)
        return self.mediate(user_text, offered, rejected, feedback)
