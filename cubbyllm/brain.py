"""cubbyllm/brain.py -- the brain, wired end to end (moved from standin/serve.py in order 0031; the server builds it
in cubby_serve/build.py).

Wired: WIRED (the server builds it: cubby_serve.build.build_serve).

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
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import time

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .brain_reasoning import PROGRAM_PREFIX, ReasoningCortex, _THINK_RE, _clean, tail_matches  # noqa: F401
from .brain_memory import MemoryCortex, _LEARN_EN, _LEARN_FR, _WH_OPEN  # noqa: F401
from .brain_route import BrainRoute

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "WIRED"

from cubbyllm.emit.format import answer_fn, shim_isolver, wrap_role_prompt  # noqa: E402
from .speech.chat import CubbyChat  # noqa: E402
from cubbyllm.emit.talk import ContextualEmitter  # noqa: E402
from cubbyllm.speech.identity import (EMITTER_SYSTEM, T, guess_lang, is_identity_question,  # noqa: E402
                                      load_facts, voice_ok)
from .worlds.base import FactStore, route_world  # noqa: E402

from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question, relation_matches  # noqa: E402


# ── the brain: sense → fast route → cortex router → cortex → CubbyTalk ─────
class CubbyBrain(BrainRoute):
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
        self.oracle = None                               # the worlds layer's gate (`mount_oracle`); None = MoWM v0 routing, no possibility check
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

    def mount_oracle(self, oracle) -> None:
        """The world model behind the worlds (`cubbyllm.bridges.possibility.PossibilityOracle`): MoWM's
        adapter, or `StorePossibility(self.worlds)` when MoWM is not installed. From then on `route` asks
        it first, and every store that has an `oracle` slot judges its adds through it (impossible ->
        refused and ledgered, uncovered -> latent). One oracle per brain; mounting again replaces it."""
        from cubbyllm.bridges.possibility import PossibilityOracle
        if not isinstance(oracle, PossibilityOracle):
            raise TypeError(f"{oracle!r} is not a PossibilityOracle: it needs possible(), route(), worlds()")
        self.oracle = oracle
        for w in self.worlds.values():
            if hasattr(w, "oracle"):
                w.oracle = oracle
        self.trace("oracle", worlds=list(oracle.worlds())[:16], kind=type(oracle).__name__)


    # ── one turn ────────────────────────────────────────────────────────────
    def sense(self, user_text: str) -> None:
        """The turn's first step: appraisal -> the hormone ODE. The harness calls it before it routes and
        frames the turn (the row records the state at turn start); a plain `turn` calls it itself."""
        self.trace("user", text=user_text)
        self.chat.nudge(user_text)                       # sense: appraisal -> ODE
        self.trace("sense", signals=getattr(self.chat, "signals", None),
                   state=dict(self.chat.state), emotion=self.chat.emotion)

    def turn(self, user_text: str, feedback: str | None = None, route: dict | None = None,
             sensed: bool = False) -> dict:
        if not sensed:
            self.sense(user_text)
        lang = guess_lang(user_text)
        dont_know = T(self.facts, "dont_know_line", lang)
        route = route or self.route(user_text)           # the harness senses and routes once, outside the frame, then hands the route in
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
        # the brain's own "I don't know" marker, on every cortex: the harness counts a turn as SPOKEN unless this is
        # set (preflight 2026-09-29 counted the dont-know line as speech and refused to serve on 5 correct refusals)
        rec["dont_know"] = bool(rec.get("reply") == dont_know or rec.get("spoke_safe"))
        rec["signals"] = getattr(self.chat, "signals", None)
        rec["emotion"] = self.chat.emotion
        from .speech.audience import LeakGuard                   # sources are internal (order 0016): the last check at the mouth
        loop = getattr(self, "ask_loop", None)
        guard = getattr(loop, "guard", None) or LeakGuard()
        said, hits = guard.check(rec.get("reply"), {"answer": (rec.get("task") or {}).get("vm_answer"), "learned": []})
        if hits:                                         # cut where it only named a source; refused (I don't know) otherwise
            rec.update(guard=hits, reply=said if said is not None else dont_know, dont_know=said is None or rec["dont_know"])
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
