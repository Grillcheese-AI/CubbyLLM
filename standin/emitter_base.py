"""emitter_base -- the emitter contract and what every emitter shares (split out of emitter.py, order 0030; the
stand-in's parts went to emitter_llama in order 0031).

`Emitter` (text in, a CubeLang program or a reply out), reply cleaning (`clean_reply`; `strip_think` and
`strip_fences` for programs), the card's decode lock, and the `ReplayEmitter`. `emitter.py` re-exports every name
here.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this).
"""
from __future__ import annotations

import re
import threading
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


GPU_LOCK = threading.Lock()   # one decode at a time on the card: Llama is not thread-safe, and two decodes cannot overlap usefully


_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)


def strip_think(s: str) -> str:
    """LFM2.5's chat template opens a <think> block in the generation prompt
    and the model fills it before the answer (seen on the first SFT run,
    2026-08-30). Everything up to and including </think> is reasoning, not
    the answer: drop it before scoring programs OR identity turns (the
    reasoning text talks about AGI/other models and would trip the check)."""
    return _THINK_RE.sub("", s, count=1) if "</think>" in s else s


def strip_fences(s: str) -> str:
    """Models sometimes wrap programs in ``` fences; the VM does not want them."""
    s = strip_think(s).strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip() + "\n"


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
