"""emitter — the trunk-facing interface the stand-in sits behind.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this — guard-pinned).

`Emitter` is the contract: text in, CubeLang program out. Three implementations:

  LlamaCppEmitter     in-process GGUF via llama-cpp-python (installed here with
                      the Vulkan backend bundled — it finds the RX 6750 XT);
                      the default for the local VM eval:
                        python -c "from standin.emitter import LlamaCppEmitter as E; print(E('emitter-q4_k_m.gguf').emit('What is 3 plus 4?'))"
  LlamaServerEmitter  HTTP against an OpenAI-compatible server — either a real
                      llama.cpp `llama-server` (not installed on this machine) or
                      the bindings' own:  python -m llama_cpp.server --model emitter-q4_k_m.gguf --n_gpu_layers -1 --port 8080
  ReplayEmitter       replays recorded generations (the Colab notebook's val_generations.json)

The 2B CubbyLLM trunk replaces the stand-in by implementing the same protocol
(a `CubbyEmitter` over `cubbyllm.core.generation`) — nothing above this
interface changes. No third-party dependency: urllib only.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Protocol, runtime_checkable

__wiring__ = "STANDALONE"

SYSTEM = ("You are the CubeLang emitter. Given a question or instruction, output ONLY a complete "
          "CubeLang program that solves it. No prose, no explanation.")


@runtime_checkable
class Emitter(Protocol):
    """Text -> CubeLang program source. Deterministic at the default temperature 0
    (programs, the VM-verified eval); `temperature`/`seed` exist for the words —
    thought verbalization and free chat — where sameness is the defect."""

    name: str

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:
        """`system` overrides the default system prompt — this is how the host
        injects the hormonal-state block (standin/data/identity.py) for chat
        turns; emitter turns leave it None and get the strict prompt.
        `prefix` is forced assistant-prefill text (style steering: the serve
        loop pins task turns to the CotChain opening); implementations return
        prefix + continuation so callers always see the full program.
        `context` is the trunk's c in theta = f(c): today a role tag ("programs" |
        "talk") or a dict with a "role" (and, when the host has them, "world" and
        "state"); a single-model emitter ignores it, `ContextualEmitter` resolves it
        to an adapter, the 2B trunk will condition its parameters on it."""
        ...


class LlamaServerEmitter:
    """OpenAI-compatible chat completion against a local llama-server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str = "emitter",
                 system: str = SYSTEM, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system = system
        self.timeout = timeout
        self.name = f"llama-server:{model}"

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:   # a single model: the context is the caller's business
        if prefix:
            raise NotImplementedError("assistant prefill is not supported over the chat endpoint")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system or self.system},
                         {"role": "user", "content": prompt}],
            "temperature": float(temperature), "max_tokens": int(max_new_tokens), "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        return out["choices"][0]["message"]["content"]


NO_THINK_PREFILL = "<think>\n</think>\n"


def render_chatml(system: str, user: str, prefill: str = NO_THINK_PREFILL) -> str:
    """LFM2.5's chat template, rendered by hand (verified against the GGUF's
    tokenizer.chat_template 2026-08-30): ChatML, system optional, generation
    prompt `<|im_start|>assistant\\n` with NO think opener — the model opens
    <think> on its own, so an empty `<think>\\n</think>\\n` prefill is how the
    answer is made to start immediately. The BOS token is added by the
    tokenizer, not here."""
    out = ""
    if system:
        out += f"<|im_start|>system\n{system}<|im_end|>\n"
    out += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefill}"
    return out


class LlamaCppEmitter:
    """In-process GGUF inference through llama-cpp-python (the installed
    0.3.30 bundles ggml-vulkan.dll and finds the RX 6750 XT — verified
    2026-08-30; there is no standalone llama-server on this machine).
    Loads lazily on first emit; n_gpu_layers=-1 offloads everything.
    `prefill` (default: an empty think block) is appended to the rendered
    assistant turn so the model does not spend tokens reasoning — the first
    SFT run showed LFM2.5 opening <think> on its own and confabulating
    context; set prefill="" to let it think."""

    def __init__(self, gguf_path: str, system: str = SYSTEM, n_ctx: int = 4096,
                 n_gpu_layers: int = -1, verbose: bool = False, prefill: str = NO_THINK_PREFILL) -> None:
        self.gguf_path = gguf_path
        self.system = system
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.verbose = verbose
        self.prefill = prefill
        self.name = f"llama-cpp:{gguf_path.replace(chr(92), '/').rsplit('/', 1)[-1]}"
        self._llm = None

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self.gguf_path, n_ctx=self.n_ctx, n_gpu_layers=self.n_gpu_layers,
                              verbose=self.verbose, seed=0)
        return self._llm

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:   # a single model: the context is the caller's business
        text = render_chatml(system or self.system, prompt, self.prefill + prefix)
        sampling = {"temperature": float(temperature), "top_p": 0.9} if temperature > 0 else {"temperature": 0.0}
        out = self._load().create_completion(text, max_tokens=int(max_new_tokens), seed=(0 if seed is None else int(seed)),
                                             **sampling,
                                             stop=["<|im_end|>"])
        return prefix + out["choices"][0]["text"]


class ReplayEmitter:
    """Replays generations recorded elsewhere (the Colab notebook's
    val_generations.json) keyed by prompt — lets the VM eval run without a
    model on the machine, and pins the eval to an exact set of outputs."""

    def __init__(self, generations: list[dict], name: str = "replay") -> None:
        self._by_prompt = {g["prompt"]: g["generated"] for g in generations}
        self.name = name

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:   # a single model: the context is the caller's business
        try:
            return self._by_prompt[prompt]                # recordings are complete programs
        except KeyError:
            raise KeyError(f"no recorded generation for prompt: {prompt[:80]!r}") from None


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
