"""emitter_llama -- the in-process GGUF emitter, llama-cpp-python (split out of emitter.py, order 0030).

Wired: STANDALONE (stand-in; re-exported by emitter.py).
"""
from __future__ import annotations

import os

from emitter_base import CHAT_FAMILIES, GPU_LOCK, SYSTEM, ScriptBan, chat_family

__wiring__ = "STANDALONE"


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
