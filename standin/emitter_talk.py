"""emitter_talk -- the talk seat on base450m (its prompt and answer formats) and the `ContextualEmitter` that
routes a call to a role by its context (split out of emitter.py, order 0030).

Wired: STANDALONE (stand-in; re-exported by emitter.py).
"""
from __future__ import annotations

import os

from emitter_450m import Cubby450mTrunk

__wiring__ = "STANDALONE"


# the talk adapter's own formats (standin/data/build_ground_sft.py, build_chat_sft.py): a facts block, a
# conversation, or a bare question; the answer follows "Answer:".
TALK_DASH = " \u2014 "
TALK_UNLOCK = "Explicit: allowed"
_TALK_STOPS = ("\nQuestion:", "\nFacts:", "\nConversation:", "\nUser:", "\nSystem:")


def talk_prompt(prompt: str, context=None) -> str:
    """What the 450M talk adapter is shown, rendered from the host's structured context when the caller gave
    one -- `facts` ([(entity, relation, value)] or ready lines) and `question`; `turns` ([(role, text)], the
    last one the user's) -- else the prompt as the question. The stand-in reads instructions; this adapter
    was trained on these shapes and nothing else, so the host renders, never the model."""
    ctx = context if isinstance(context, dict) else {}
    head = f"{TALK_UNLOCK}\n" if ctx.get("explicit") else ""
    facts = ctx.get("facts")
    if facts:
        lines = []
        for f in facts:
            if isinstance(f, (list, tuple)) and len(f) == 3:
                e, r, v = f
                lines.append(f"- {e}{TALK_DASH}{r}: {v}")
            else:
                lines.append(f"- {str(f).strip()}")
        q = (ctx.get("question") or prompt).strip()
        return f"{head}Facts:\n" + "\n".join(lines) + f"\nQuestion: {q}\nAnswer:"
    turns = ctx.get("turns")
    if turns:
        *before, (_, last) = turns
        if before:
            lines = "\n".join(f"{str(role).capitalize()}: {str(text).strip()}" for role, text in before)
            return f"{head}Conversation:\n{lines}\nQuestion: {str(last).strip()}\nAnswer:"
        return f"{head}Question: {str(last).strip()}\nAnswer:"
    return f"{head}Question: {prompt.strip()}\nAnswer:"


def talk_answer(text: str) -> str:
    """The adapter's answer out of its generation: cut where it would start the next block, one line."""
    for stop in _TALK_STOPS:
        text = text.split(stop)[0]
    return " ".join(text.strip().split("\n")[0].split())


class Cubby450mTalk:
    """The talk seat on base450m: the talk LoRA (`validation/train_talk_torch.py`, gates E6-E9) behind the
    `Emitter` protocol, sharing the base with the programs seat through a `Cubby450mTrunk`. The host's talk
    calls (`ask.paraphrase`, the chat cortex, the perception reads) are rendered into the adapter's own formats
    by `talk_prompt`; `system` is ignored (the adapter has no system turn), `prefix` is prepended to the
    answer, and a temperature above 0 samples (the chat cortex's draws)."""

    def __init__(self, trunk: Cubby450mTrunk, role: str = "talk", max_ctx: int = 2048) -> None:
        if role not in trunk.adapter_dirs:
            raise KeyError(f"the trunk has no {role!r} adapter")
        self.trunk, self.role = trunk, role
        self.max_ctx = min(int(max_ctx), trunk.max_ctx)
        self.name = f"cubby450m:{os.path.basename(os.path.normpath(trunk.adapter_dirs[role]))}"
        self.last: dict = {}

    def emit(self, prompt: str, max_new_tokens: int = 160, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:
        text = talk_prompt(prompt, context) + ((" " + prefix) if prefix else "")
        ids = self.trunk.tokenizer.encode(text).ids[-self.max_ctx:]
        out = self.trunk.generate(self.role, [ids], max_new_tokens, temperature=temperature, seed=seed)[0]
        if self.trunk.eos in out:
            out = out[:out.index(self.trunk.eos)]
        raw = self.trunk.tokenizer.decode(out)
        self.last = {"prompt": text, "raw": raw}
        return (prefix + " " if prefix else "") + talk_answer(raw)


def context_role(context) -> str | None:
    """The role tag inside a context: the tag itself, or a dict's "role"."""
    if context is None:
        return None
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        return context.get("role")
    return None


class ContextualEmitter:
    """One trunk interface over several adapters, selected by the context — the
    stand-in's stand-in for theta = f(c): a K-entry, tag-indexed adapter bank (the
    "cheapest MoE" of the panel agenda: no learned router; the thalamus's rules are
    the frozen router H-C4 asked for). Today the entries are whole fine-tunes on one
    base ("programs": the emitter, "talk": the talk cortex); the same object later
    holds one base + LoRA deltas, and the 2B trunk replaces the lookup with generated
    parameters — callers never change. An unknown or missing role uses `default` —
    the exact-match version of MindForge's SDLS purification (a context below the
    similarity threshold of every registered context gets the default adapter)."""

    def __init__(self, adapters: dict, default: str = "programs") -> None:
        if not adapters:
            raise ValueError("ContextualEmitter needs at least one adapter")
        if default not in adapters:
            raise ValueError(f"default role {default!r} not among adapters {sorted(adapters)}")
        self.adapters = dict(adapters)
        self.default = default
        self.calls: dict = {k: 0 for k in self.adapters}   # per-role usage, for /health and tests
        self.fallbacks: dict = {}                            # role asked for but absent -> count: the "need" monitor
        self.history: list = []                              # (role, adapter name, "registered"|"retired"|"shadowed"|"promoted"|"rejected")
        self.candidates: dict = {}                           # role -> adapter in SHADOW: evaluated, never routed (GrillCheese's PROGENITOR stage)

    @property
    def name(self) -> str:
        return "ctx[" + ",".join(f"{k}={getattr(v, 'name', '?')}" for k, v in self.adapters.items()) + "]"

    @property
    def is_split(self) -> bool:
        """True when at least two roles resolve to different adapters."""
        return len({id(v) for v in self.adapters.values()}) > 1

    # ── the bank grows and shrinks at runtime (spawned specialists arrive here after promotion) ──
    def register(self, role: str, adapter) -> None:
        """Add (or replace) the adapter for a role; the retired one is kept in `history`, never lost."""
        if not role or not hasattr(adapter, "emit"):
            raise ValueError("register(role, adapter) needs a role and an object with .emit")
        old = self.adapters.get(role)
        if old is not None:
            self.history.append((role, getattr(old, "name", "?"), "retired"))
        self.adapters[role] = adapter
        self.calls.setdefault(role, 0)
        self.history.append((role, getattr(adapter, "name", "?"), "registered"))

    def unregister(self, role: str) -> None:
        if role == self.default:
            raise ValueError("the default adapter cannot be unregistered")
        if role in self.adapters:
            self.history.append((role, getattr(self.adapters.pop(role), "name", "?"), "retired"))

    # ── the maturation ladder: candidate (shadow) -> routed -> stable ─────────────────────────
    def shadow(self, role: str, adapter) -> None:
        """Hold a spawned adapter as a CANDIDATE for a role: it can be emitted from explicitly
        (`emit_candidate`) so the promote step can compare it against the incumbent on the same
        contexts, but `resolve` never routes live traffic to it."""
        if not role or not hasattr(adapter, "emit"):
            raise ValueError("shadow(role, adapter) needs a role and an object with .emit")
        self.candidates[role] = adapter
        self.history.append((role, getattr(adapter, "name", "?"), "shadowed"))

    def emit_candidate(self, role: str, prompt: str, **kw) -> str:
        return self.candidates[role].emit(prompt, **kw)

    def promote(self, role: str) -> None:
        """The candidate becomes the routed adapter for its role (the incumbent, if any, is retired
        into history). Called ONLY by the promotion rule after it passed."""
        if role not in self.candidates:
            raise KeyError(f"no candidate in shadow for role {role!r}")
        self.register(role, self.candidates.pop(role))
        self.history.append((role, getattr(self.adapters[role], "name", "?"), "promoted"))

    def reject(self, role: str) -> None:
        """The candidate failed the promotion rule: dropped from shadow, kept in history."""
        if role in self.candidates:
            self.history.append((role, getattr(self.candidates.pop(role), "name", "?"), "rejected"))

    def stage(self, role: str) -> str:
        """'routed' (in the bank), 'candidate' (in shadow), or 'absent'."""
        if role in self.adapters:
            return "routed"
        if role in self.candidates:
            return "candidate"
        return "absent"

    def idle_roles(self, min_calls: int = 1) -> list:
        """Routed specialists below `min_calls` — the prune candidates (neurogenesis prunes what never
        fires); the default is never listed."""
        return sorted(r for r, n in self.calls.items() if r != self.default and r in self.adapters and n < min_calls)

    def resolve(self, context) -> str:
        role = context_role(context)
        if role in self.adapters:
            return role
        if role is not None and role != self.default:          # a context asked for a specialist that does not exist
            self.fallbacks[role] = self.fallbacks.get(role, 0) + 1
        return self.default

    def usage(self) -> dict:
        """What /health shows and the need-detector reads: calls per role, fallbacks per missing role,
        and the fallback rate (the share of contexts no specialist claimed)."""
        total = sum(self.calls.values())
        fb = sum(self.fallbacks.values())
        return {"roles": sorted(self.adapters), "candidates": sorted(self.candidates), "calls": dict(self.calls),
                "fallbacks": dict(self.fallbacks), "fallback_rate": (fb / total) if total else 0.0,
                "idle": self.idle_roles(), "history": list(self.history)}

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:
        role = self.resolve(context)
        self.calls[role] += 1
        return self.adapters[role].emit(prompt, max_new_tokens=max_new_tokens, system=system, prefix=prefix,
                                        temperature=temperature, seed=seed, context=context)
