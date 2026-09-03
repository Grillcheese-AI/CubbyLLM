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
    parameters — callers never change. An unknown or missing role uses `default`."""

    def __init__(self, adapters: dict, default: str = "programs") -> None:
        if not adapters:
            raise ValueError("ContextualEmitter needs at least one adapter")
        if default not in adapters:
            raise ValueError(f"default role {default!r} not among adapters {sorted(adapters)}")
        self.adapters = dict(adapters)
        self.default = default
        self.name = "ctx[" + ",".join(f"{k}={getattr(v, 'name', '?')}" for k, v in self.adapters.items()) + "]"
        self.calls: dict = {k: 0 for k in self.adapters}   # per-role usage, for /health and tests

    @property
    def is_split(self) -> bool:
        """True when at least two roles resolve to different adapters."""
        return len({id(v) for v in self.adapters.values()}) > 1

    def resolve(self, context) -> str:
        role = context_role(context)
        return role if role in self.adapters else self.default

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:
        role = self.resolve(context)
        self.calls[role] += 1
        return self.adapters[role].emit(prompt, max_new_tokens=max_new_tokens, system=system, prefix=prefix,
                                        temperature=temperature, seed=seed, context=context)
