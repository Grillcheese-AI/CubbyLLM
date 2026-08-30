"""emitter — the trunk-facing interface the stand-in sits behind.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this — guard-pinned).

`Emitter` is the contract: text in, CubeLang program out. `LlamaServerEmitter`
implements it over HTTP against a local llama.cpp `llama-server` (the Vulkan
build runs the exported GGUF on the AMD card):

    llama-server -m emitter-q4_k_m.gguf -c 4096 --port 8080
    python -c "from standin.emitter import LlamaServerEmitter as E; print(E().emit('What is 3 plus 4?'))"

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
    """Text -> CubeLang program source. Implementations must be deterministic
    at temperature 0 so the VM-verified eval is reproducible."""

    name: str

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None) -> str:
        """`system` overrides the default system prompt — this is how the host
        injects the hormonal-state block (standin/data/identity.py) for chat
        turns; emitter turns leave it None and get the strict prompt."""
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

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system or self.system},
                         {"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": int(max_new_tokens), "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        return out["choices"][0]["message"]["content"]


class ReplayEmitter:
    """Replays generations recorded elsewhere (the Colab notebook's
    val_generations.json) keyed by prompt — lets the VM eval run without a
    model on the machine, and pins the eval to an exact set of outputs."""

    def __init__(self, generations: list[dict], name: str = "replay") -> None:
        self._by_prompt = {g["prompt"]: g["generated"] for g in generations}
        self.name = name

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None) -> str:
        try:
            return self._by_prompt[prompt]
        except KeyError:
            raise KeyError(f"no recorded generation for prompt: {prompt[:80]!r}") from None
