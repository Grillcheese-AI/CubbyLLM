"""emitter_450m -- CubbyLLM's own base450m on grilly2 and its LoRA roles: the trunk that swaps adapters in place,
and the program emitter on it (split out of emitter.py, order 0030).

Wired: STANDALONE (stand-in; re-exported by emitter.py).
"""
from __future__ import annotations

import json
import os
import threading

__wiring__ = "STANDALONE"


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
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer
        from ..ops import grilly_seats as ops            # grilly2, behind the ops seam (order 0031)
        model, _cfg = ops.load_export(self.export_dir)
        layout = None
        for role in self.adapter_dirs:
            meta_p, tensors_p = self._adapter_files(role)
            meta = json.load(open(meta_p, encoding="utf-8"))
            this = (int(meta["rank"]), float(meta["alpha"]), tuple(meta["targets"]))
            if layout is None:
                layout = this
                self._wrapped = ops.wrap_lora(model, meta["rank"], meta["alpha"], meta["targets"])
            elif this != layout:
                raise ValueError(f"adapter {role!r} has another LoRA layout than the first (rank/alpha/targets differ): "
                                 "one base can only swap adapters of one layout")
            tensors = load_file(tensors_p)
            self._weights[role] = {path: (tensors[f"{path}.lora_A.weight"], tensors[f"{path}.lora_B.weight"])
                                   for path in self._wrapped}
        self._tk = Tokenizer.from_file(self.tokenizer_path)
        self._eos = self._tk.token_to_id("</s>")
        self._model, self._grilly = model, ops.grilly()

    def use(self, role: str) -> None:
        """The role's adapter into the wrapped projections (a no-op when it is already there)."""
        if role not in self.adapter_dirs:
            raise KeyError(f"no adapter for role {role!r}; have {sorted(self.adapter_dirs)}")
        self._load()
        if role == self._role:
            return
        from ..ops import grilly_seats as ops
        ops.copy_pair(self._wrapped, self._weights[role], self._grilly)
        self._role = role
        self.swaps += 1

    # -- a role from a plugin (order 0028): added on demand, refused when its layout or base differs ----------------
    def layout(self) -> tuple:
        """(rank, alpha, targets) every adapter on this base shares: the first role's."""
        meta = json.load(open(self._adapter_files(next(iter(self.adapter_dirs)))[0], encoding="utf-8"))
        return int(meta["rank"]), float(meta["alpha"]), tuple(meta["targets"])

    def base_hash(self) -> str:
        """The export's weights' SHA-256, computed once and kept beside them (keyed by size and time)."""
        import hashlib
        w = os.path.join(self.export_dir, "model.safetensors")
        st, side = os.stat(w), w + ".sha256.json"
        try:
            got = json.load(open(side, encoding="utf-8"))
            if got.get("size") == st.st_size and got.get("mtime") == int(st.st_mtime):
                return got["sha256"]
        except (OSError, ValueError, KeyError):
            pass
        h = hashlib.sha256()
        with open(w, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        json.dump({"size": st.st_size, "mtime": int(st.st_mtime), "sha256": h.hexdigest()}, open(side, "w", encoding="utf-8"))
        return h.hexdigest()

    def add_role(self, role: str, adapter_dir: str, base: str | None = None) -> None:
        """A plugin's adapter as a new role: refused (ValueError, with the reason) when its LoRA layout is not the
        base's or its ``base`` hash names another export. The base and the other roles' weights do not move."""
        if role in self.adapter_dirs:
            raise ValueError(f"role {role!r} is already taken")
        if base and base != self.base_hash():
            raise ValueError(f"adapter {role!r} was trained on another base ({base[:12]} vs {self.base_hash()[:12]})")
        self.adapter_dirs[role] = adapter_dir
        try:
            meta_p, tensors_p = self._adapter_files(role)
            meta = json.load(open(meta_p, encoding="utf-8"))
            if (int(meta["rank"]), float(meta["alpha"]), tuple(meta["targets"])) != self.layout():
                raise ValueError(f"adapter {role!r} has another LoRA layout than the base's (rank/alpha/targets)")
            if self._model is not None:                      # loaded: its weights now; else at the first load
                from safetensors.numpy import load_file
                tensors = load_file(tensors_p)
                self._weights[role] = {path: (tensors[f"{path}.lora_A.weight"], tensors[f"{path}.lora_B.weight"])
                                       for path in self._wrapped}
        except Exception:
            del self.adapter_dirs[role]
            raise

    def remove_role(self, role: str) -> None:
        """A plugin's role unloaded; the next call that asks another role copies that role's weights back in."""
        if role not in self.adapter_dirs or len(self.adapter_dirs) == 1:
            raise ValueError(f"cannot remove role {role!r}")
        del self.adapter_dirs[role]
        self._weights.pop(role, None)
        if self._role == role:
            self._role = None                                 # the wrapped weights are stale: the next use() rewrites

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
            from ..ops import grilly_seats as ops
            out = ops.generate(self._model, rows, max_new_tokens, self._eos, temperature=temperature,
                               top_k=top_k, seed=seed, g=self._grilly)
            return [r[len(p):] for r, p in zip(out, rows)]


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
        from .format import PROMPT_HEAD, PROMPT_TAIL, decode_program
        text = prompt if prompt.startswith(PROMPT_HEAD) else PROMPT_HEAD + prompt.strip() + PROMPT_TAIL
        out = self.trunk.generate(self.role, [self._ids(text + prefix)], max_new_tokens)[0]
        return prefix + decode_program(self.trunk.tokenizer, out, self.trunk.eos)   # keeps ACTION/AGENT/OBJECT (special tokens)

    def emit_samples(self, prompt: str, n: int, max_new_tokens: int = 96, temperature: float = 0.7,
                     top_k: int = 40, seed: int | None = None) -> list[str]:
        """`n` sampled continuations in ONE batched decode (the prompt repeated n times; rows stop at </s>):
        the candidates a step vote runs in the VM (`step_loop.StepLoop.solve_vote`)."""
        from .format import PROMPT_HEAD, PROMPT_TAIL, decode_program
        text = prompt if prompt.startswith(PROMPT_HEAD) else PROMPT_HEAD + prompt.strip() + PROMPT_TAIL
        ids = self._ids(text)
        rows = self.trunk.generate(self.role, [ids] * n, max_new_tokens, temperature=temperature, top_k=top_k, seed=seed)
        return [decode_program(self.trunk.tokenizer, r, self.trunk.eos) for r in rows]
