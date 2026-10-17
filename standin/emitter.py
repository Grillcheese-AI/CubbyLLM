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
                 n_gpu_layers: int = -1, verbose: bool = False, prefill: str | None = None,
                 family: str | None = None, script_ban: bool | None = None) -> None:
        # Fail here, not on the first emit. The load is lazy, so a path that
        # does not resolve from the process CWD used to surface minutes later
        # as a SPEECH failure ("Model path does not exist" out of `_speak`),
        # which reads like a bug in the talking rather than in the launch —
        # found 2026-09-15 by the component map, which lit `thought_error`
        # while the game itself ran perfectly.
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(
                f"no GGUF at {gguf_path!r} (from {os.getcwd()!r}). "
                "Paths like standin/models/... are relative to the REPO ROOT — "
                "run serve_api.py from there, or pass an absolute path.")
        self.gguf_path = gguf_path
        self.script_ban = (os.environ.get("CB_SCRIPT_BAN", "1") != "0") if script_ban is None else bool(script_ban)
        self._logits_processor = None
        self.system = system
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.verbose = verbose
        self._prefill = prefill          # None: the family's default (an empty think block for lfm, nothing for chatml/gemma)
        self._family = family            # None: sniffed from the GGUF's tokenizer.chat_template at load
        self.name = f"llama-cpp:{gguf_path.replace(chr(92), '/').rsplit('/', 1)[-1]}"
        self._llm = None

    @property
    def family(self) -> str:
        if self._family is None:
            self._load()
        return self._family

    @property
    def prefill(self) -> str:
        return CHAT_FAMILIES[self.family][2] if self._prefill is None else self._prefill

    def _load(self):
        if self._llm is None:
            with GPU_LOCK:                                # a load beside a running decode is the two-model reset
                if self._llm is None:
                    from llama_cpp import Llama
                    self._llm = Llama(model_path=self.gguf_path, n_ctx=self.n_ctx, n_gpu_layers=self.n_gpu_layers,
                                      verbose=self.verbose, seed=0)
                    if self._family is None:
                        md = getattr(self._llm, "metadata", None) or {}
                        self._family = chat_family(md.get("tokenizer.chat_template"), md.get("general.architecture"))
                    if self.script_ban:
                        from llama_cpp import LogitsProcessorList
                        self._logits_processor = LogitsProcessorList([ScriptBan.for_gguf(self.gguf_path, self._llm)])
        return self._llm

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:   # a single model: the context is the caller's business
        llm = self._load()
        render, stop, _ = CHAT_FAMILIES[self.family]
        text = render(system or self.system, prompt, self.prefill + prefix)
        with GPU_LOCK:                                    # the HTTP server is threaded; the game and a chat turn take turns here
            out = self._decode(llm, text, int(max_new_tokens), float(temperature), seed, stop)
        return prefix + out

    def complete(self, text: str, max_new_tokens: int = 256, temperature: float = 0.0,
                 stop: list[str] | None = None, seed: int | None = None) -> str:
        """A raw completion: no chat render, no prefill -- a BASE model prompted few-shot (the LFM
        fact Source, 2026-09-13). Same decoder, same script ban, same lock."""
        llm = self._load()
        with GPU_LOCK:
            return self._decode(llm, text, int(max_new_tokens), float(temperature), seed, list(stop or []))

    def _decode(self, llm, text: str, max_new: int, temperature: float, seed: int | None, stop: list[str]) -> str:
        """Token-level decode with special tokens RENDERED. `create_completion` detokenizes with special=False, so a
        base's control tokens never reach the host — LFM2.5's <|tool_call_start|>/<|tool_call_end|> vanished and the
        v10 tool probe read a bare call body, 0/3, where the merged model read 3/3 (2026-09-04). Same sampler
        defaults as create_completion (min_p 0.05, top_k 40; temperature 0 = greedy), the script ban applied, stop
        at EOS or a stop string (single-token stops by id, others by text)."""
        ids = llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)
        if temperature > 0:
            llm.set_seed(0 if seed is None else int(seed))
        sampling = {"temp": temperature, "top_p": 0.9} if temperature > 0 else {"temp": 0.0}
        end = {int(llm.token_eos())}
        text_stops: list[str] = []
        for st in stop or []:
            st_ids = llm.tokenize(st.encode("utf-8"), add_bos=False, special=True)
            if len(st_ids) == 1:
                end.add(int(st_ids[0]))
            else:
                text_stops.append(st)
        out: list[int] = []
        for tok in llm.generate(ids, logits_processor=self._logits_processor, **sampling):
            if int(tok) in end:
                break
            out.append(int(tok))
            if len(out) >= max_new:
                break
            if text_stops and len(out) % 8 == 0:
                partial = llm.detokenize(out, special=True).decode("utf-8", "replace")
                if any(st in partial for st in text_stops):
                    break
        decoded = llm.detokenize(out, special=True).decode("utf-8", "replace")
        for st in text_stops:
            decoded = decoded.split(st, 1)[0]
        return decoded


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


class Cubby450mTrunk:
    """ONE base450m on grilly2 (the local card) and several LoRA adapters, one per role, swapped in place
    (2026-10-17, Nick: "I want to be able to replace the stand-in"). The base is loaded once (2.3 GB fp32);
    each adapter is its A/B tensors kept on the host (30 MB), copied into the wrapped projections when a call
    asks for another role than the last -- the delta moves, the base never does. Every adapter must share the
    base's LoRA layout (rank, alpha, the same targets: `talk_data.lora_targets`, every projection of every
    layer), which both the emitter and the talk trainers use. One decode at a time (the HTTP server is
    threaded; a swap under a running decode would corrupt it).

    Needs the base EXPORT (`validation/export_base.py`: config.json + model.safetensors), the tokenizer, and per
    role an adapter dir holding `<kind>_lora.safetensors` + `<kind>_lora.json` (kind: emitter | talk)."""

    KINDS = {"programs": "emitter", "talk": "talk"}

    def __init__(self, export_dir: str, tokenizer: str, adapters: dict, max_ctx: int = 2048) -> None:
        if not adapters:
            raise ValueError("Cubby450mTrunk needs at least one adapter: {role: adapter dir}")
        for p, what in [(export_dir, "export"), (tokenizer, "tokenizer")] + [(d, f"{r} adapter") for r, d in adapters.items()]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{what} not found: {p}")
        self.export_dir, self.tokenizer_path, self.max_ctx = export_dir, tokenizer, int(max_ctx)
        self.adapter_dirs = dict(adapters)
        self.name = "cubby450m[" + ",".join(f"{r}={os.path.basename(os.path.normpath(d))}" for r, d in adapters.items()) + "]"
        self._model = self._tk = self._eos = self._grilly = None
        self._wrapped: dict = {}
        self._weights: dict = {}                             # role -> {path: (A numpy, B numpy)}
        self._role = None
        self.swaps = 0
        self._lock = threading.Lock()

    def _adapter_files(self, role: str) -> tuple:
        d = self.adapter_dirs[role]
        kind = self.KINDS.get(role, role)
        for k in (kind, "emitter", "talk"):
            meta = os.path.join(d, f"{k}_lora.json")
            if os.path.exists(meta):
                return meta, os.path.join(d, f"{k}_lora.safetensors")
        raise FileNotFoundError(f"no <kind>_lora.json in {d}")

    def _load(self) -> None:
        if self._model is not None:
            return
        import sys
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in (here, os.path.join(here, "validation")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import grilly  # grilly2
        from grilly.infer.lora import apply_lora_
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer
        from exp_e5_base450m_probes import load_grilly
        model, _cfg = load_grilly(self.export_dir)
        layout = None
        for role in self.adapter_dirs:
            meta_p, tensors_p = self._adapter_files(role)
            meta = json.load(open(meta_p, encoding="utf-8"))
            this = (int(meta["rank"]), float(meta["alpha"]), tuple(meta["targets"]))
            if layout is None:
                layout = this
                self._wrapped = apply_lora_(model, meta["rank"], meta["alpha"], meta["targets"])
            elif this != layout:
                raise ValueError(f"adapter {role!r} has another LoRA layout than the first (rank/alpha/targets differ): "
                                 "one base can only swap adapters of one layout")
            tensors = load_file(tensors_p)
            self._weights[role] = {path: (tensors[f"{path}.lora_A.weight"], tensors[f"{path}.lora_B.weight"])
                                   for path in self._wrapped}
        self._tk = Tokenizer.from_file(self.tokenizer_path)
        self._eos = self._tk.token_to_id("</s>")
        self._model, self._grilly = model, grilly

    def use(self, role: str) -> None:
        """The role's adapter into the wrapped projections (a no-op when it is already there)."""
        if role not in self.adapter_dirs:
            raise KeyError(f"no adapter for role {role!r}; have {sorted(self.adapter_dirs)}")
        self._load()
        if role == self._role:
            return
        with self._grilly.no_grad():
            for path, mod in self._wrapped.items():
                a, b = self._weights[role][path]
                mod.lora_A.weight.copy_(self._grilly.from_numpy(a))
                mod.lora_B.weight.copy_(self._grilly.from_numpy(b))
        self._role = role
        self.swaps += 1

    @property
    def tokenizer(self):
        self._load()
        return self._tk

    @property
    def eos(self) -> int:
        self._load()
        return self._eos

    def generate(self, role: str, rows: list, max_new_tokens: int, temperature: float = 0.0,
                 top_k: int = 40, seed: "int | None" = None) -> list:
        """`rows` of token ids -> the generated ids per row (the prompt cut off), under the role's adapter."""
        with self._lock:
            self.use(role)
            g = self._grilly
            if seed is not None and temperature > 0:
                g.manual_seed(int(seed))
            with g.no_grad():
                if temperature > 0:
                    out = self._model.generate(g.tensor(rows), max_new_tokens=max_new_tokens, eos_token_id=self._eos,
                                               do_sample=True, temperature=float(temperature), top_k=int(top_k), top_p=1.0)
                else:
                    out = self._model.generate(g.tensor(rows), max_new_tokens=max_new_tokens, eos_token_id=self._eos)
            return [r[len(p):] for r, p in zip(out.tolist(), rows)]


class Cubby450mEmitter:
    """The 450M program emitter (H-E15, 2026-09-29): CubbyLLM's own base on grilly2 (the local card),
    plus the emitter LoRA `validation/train_emitter_torch.py` trained, behind the same `Emitter`
    protocol as the stand-in -- `ContextualEmitter({"programs": Cubby450mEmitter(...), "talk": <stand-in>})`
    is the seat decision of 2026-09-29 in one line.

    It emits SLOTTED programs: the prompt must be annotated by the host (`emitter_data.frame_prompt`
    over a `SlotTable.annotate()`), and the program comes back with `$E1` / `$N1` in it for the host to
    check and fill (`cubbyllm.reasoning.slots.SlotEmitter` does exactly that around this class; its
    `index` is the names the host knows). `emit` on an un-annotated prompt still runs -- the program
    then names slots the host does not have, which the check refuses. Greedy, no repetition guards
    (a program repeats `create frame: number;` on purpose), stops at </s>.

    Built either on its own base (`Cubby450mEmitter(export_dir, adapter_dir, tokenizer)`, as before) or on a
    shared `Cubby450mTrunk` (`Cubby450mEmitter(trunk)`, 2026-10-17), where the talk seat shares the base."""

    def __init__(self, export_dir, adapter_dir: "str | None" = None, tokenizer: "str | None" = None,
                 max_ctx: int = 2048, role: str = "programs") -> None:
        if hasattr(export_dir, "generate") and hasattr(export_dir, "adapter_dirs"):     # a trunk (or a test's stand-in for one)
            self.trunk = export_dir
        else:
            if adapter_dir is None or tokenizer is None:
                raise ValueError("Cubby450mEmitter(export_dir, adapter_dir, tokenizer) or Cubby450mEmitter(trunk)")
            self.trunk = Cubby450mTrunk(export_dir, tokenizer, {role: adapter_dir}, max_ctx=max_ctx)
        self.role = role
        self.max_ctx = self.trunk.max_ctx
        self.name = f"cubby450m:{os.path.basename(os.path.normpath(self.trunk.adapter_dirs[role]))}"

    def _ids(self, text: str) -> list:
        return self.trunk.tokenizer.encode(text).ids[-self.max_ctx:]

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None,
             prefix: str = "", temperature: float = 0.0, seed: int | None = None,
             context: "str | dict | None" = None) -> str:
        from emitter_data import PROMPT_HEAD, PROMPT_TAIL, decode_program
        text = prompt if prompt.startswith(PROMPT_HEAD) else PROMPT_HEAD + prompt.strip() + PROMPT_TAIL
        out = self.trunk.generate(self.role, [self._ids(text + prefix)], max_new_tokens)[0]
        return prefix + decode_program(self.trunk.tokenizer, out, self.trunk.eos)   # keeps ACTION/AGENT/OBJECT (special tokens)

    def emit_samples(self, prompt: str, n: int, max_new_tokens: int = 96, temperature: float = 0.7,
                     top_k: int = 40, seed: int | None = None) -> list[str]:
        """`n` sampled continuations in ONE batched decode (the prompt repeated n times; rows stop at </s>):
        the candidates a step vote runs in the VM (`step_loop.StepLoop.solve_vote`)."""
        from emitter_data import PROMPT_HEAD, PROMPT_TAIL, decode_program
        text = prompt if prompt.startswith(PROMPT_HEAD) else PROMPT_HEAD + prompt.strip() + PROMPT_TAIL
        ids = self._ids(text)
        rows = self.trunk.generate(self.role, [ids] * n, max_new_tokens, temperature=temperature, top_k=top_k, seed=seed)
        return [decode_program(self.trunk.tokenizer, r, self.trunk.eos) for r in rows]


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
