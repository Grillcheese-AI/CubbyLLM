"""emitter_base -- the emitter contract and what every emitter shares (split out of emitter.py, order 0030).

`Emitter` (text in, a CubeLang program or a reply out), the HTTP `LlamaServerEmitter`, reply cleaning
(`clean_reply`), the chat renderers and families, the non-Latin script ban, the card's decode lock, and the
`ReplayEmitter`. `emitter.py` re-exports every name here.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this).
"""
from __future__ import annotations

import json
import os
import re
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

_THINK_BLOCK = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)
_TOOL_CALL = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.S)
_STRAY_TAG = re.compile(r"^\s*</?(?:think|tool_call|tool_response|tools)>\s*$", re.M)
_STRAY_INLINE = re.compile(r"</?(?:think|tool_call|tool_response|tools)>")


def clean_reply(raw: str) -> str:
    """A generation as a reply: a closed <think> block dropped, then every stray special tag that is not part of a
    complete <tool_call>{json}</tool_call> removed (Qwen3's base leaks '<tool_response>', '</tool_call>' and an
    unclosed '<think>' at the edges of plain replies — live session 2026-09-04). Complete tool calls are kept for the
    host to parse."""
    text = _THINK_BLOCK.sub("", raw, count=1) if "</think>" in raw else raw
    keep = {}
    def _hold(m):
        key = f"\x00{len(keep)}\x00"
        keep[key] = m.group(0)
        return key
    text = _TOOL_CALL.sub(_hold, text)
    text = _STRAY_TAG.sub("", text)
    text = _STRAY_INLINE.sub("", text)
    for key, val in keep.items():
        text = text.replace(key, val)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

import threading

GPU_LOCK = threading.Lock()   # one decode at a time on the card: Llama is not thread-safe, and two decodes cannot overlap usefully


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


def render_gemma(system: str, user: str, prefill: str = "") -> str:
    """Gemma's chat template (Gemma 3/4, `google/gemma-4-*-it`): turns are
    `<start_of_turn>{role}\n…<end_of_turn>\n`, the system text rides in the
    first user turn, the generation prompt is `<start_of_turn>model\n`. No
    think block, so the default prefill is empty. BOS is the tokenizer's."""
    body = f"{system}\n\n{user}" if system else user
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n{prefill}"


CHAT_FAMILIES = {   # family -> (renderer, stop strings, default prefill)
    "lfm": (render_chatml, ["<|im_end|>"], NO_THINK_PREFILL),         # LFM2.5 (2.6B, 8B-A1B): ChatML, opens <think> on its own
    "chatml": (render_chatml, ["<|im_end|>"], ""),                     # Qwen3-*-Instruct-2507 and other plain ChatML bases
    "gemma": (render_gemma, ["<end_of_turn>"], ""),                    # Gemma 3 / 4
}


def chat_family(template: str | None, arch: str | None = None) -> str:
    """The template family of a GGUF from its `tokenizer.chat_template` and
    `general.architecture` (the sniff a talk adapter on a non-LFM base needs).
    Gemma from the template; among ChatML bases the LFM architectures
    (lfm2, lfm2moe) get the empty think-block prefill and the rest do not.
    No architecture in the metadata -> lfm, the family every GGUF served so
    far belongs to. The talk notebook applies the same rule from the model id."""
    t = template or ""
    if "<start_of_turn>" in t:
        return "gemma"
    if arch:
        return "lfm" if arch.lower().startswith("lfm2") else "chatml"
    return "lfm"


# the same class as identity._NON_LATIN (pinned equal by a test): Cyrillic, Hebrew/Arabic, Indic, Thai, kana, CJK,
# Hangul, full-width forms — none of them Cubby's languages
NON_LATIN = re.compile(r"[\u0400-\u04FF\u0590-\u06FF\u0900-\u0DFF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uFF00-\uFFEF]")


class ScriptBan:
    """A llama-cpp logits processor: every token whose piece carries a non-Latin script is set to -inf before
    sampling, so a base that drifts into Chinese mid-reply ("建筑师: 8/15, nice.", the Qwen3 talk arm, 2026-09-04)
    cannot — the guards (`rephrase_ok`, the chat guard) stay as the second line for byte-fallback sequences.
    Built from the vocab alone (`vocab_only=True` loads no weights and touches no GPU); cached beside the GGUF as
    `<gguf>.scriptban.npy`. CB_SCRIPT_BAN=0 disables it."""

    def __init__(self, ids) -> None:
        import numpy as np
        self.ids = np.asarray(ids, dtype=np.intp)

    @classmethod
    def from_vocab(cls, llm) -> "ScriptBan":
        """`llm` needs `n_vocab()` and `detokenize([id]) -> bytes` (a llama_cpp.Llama, vocab_only or not)."""
        n = int(llm.n_vocab())
        ids = [t for t in range(n) if NON_LATIN.search(llm.detokenize([t]).decode("utf-8", "ignore"))]
        return cls(ids)

    @classmethod
    def for_gguf(cls, gguf_path: str, llm=None) -> "ScriptBan":
        import numpy as np
        cache = gguf_path + ".scriptban.npy"
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(gguf_path):
            return cls(np.load(cache))
        if llm is None:
            from llama_cpp import Llama
            llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)
        ban = cls.from_vocab(llm)
        try:
            np.save(cache, ban.ids)
        except OSError:
            pass
        return ban

    def __call__(self, input_ids, scores):
        scores[self.ids] = -float("inf")
        return scores




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
