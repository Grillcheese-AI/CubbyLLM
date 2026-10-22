"""standin/question_encoder.py -- a question encoder for plugin routing (order 0028, review addition 4).

paraphrase-MiniLM-L6-v2 (a 22M-parameter BERT sentence encoder, Apache-2.0) read from the Hugging Face cache with
torch and ``tokenizers`` alone: no ``transformers``, nothing downloaded (``find`` returns None when the cache lacks it).
``encode(text)`` is the unit mean of the last layer's token states, the model's own pooling -- the shape
``worlds.FactStore(enc=...)`` and ``plugins_route.LogisticRouter`` take. An encoder only: it never writes text, so it
is not a language model on the serving path; the fastword tables the serve stack routes worlds with were distilled
from its sibling all-MiniLM-L6-v2.

Wired: WIRED, behind ``CUBBY_PLUGINS_ROUTER=minilm`` (``plugins_route.make_router``); off, nothing imports it.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

__wiring__ = "WIRED"

REPO = "sentence-transformers/paraphrase-MiniLM-L6-v2"


def _tensors(path: Path) -> dict:
    import torch
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    out = {}
    for k, t in head.items():
        if k == "__metadata__":
            continue
        s, e = t["data_offsets"]
        dt = {"F32": np.float32, "F16": np.float16, "I64": np.int64}[t["dtype"]]
        a = np.frombuffer(raw[8 + n + s:8 + n + e], dt).astype(np.float32).reshape(t["shape"])
        out[k.replace("bert.", "")] = torch.from_numpy(a.copy())
    return out


class QuestionEncoder:
    def __init__(self, folder) -> None:
        from tokenizers import Tokenizer
        folder = Path(folder)
        self.cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.W = _tensors(folder / "model.safetensors")
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(128)
        self.heads, self.dim = self.cfg["num_attention_heads"], self.cfg["hidden_size"]
        self._memo: dict[str, np.ndarray] = {}

    @classmethod
    def find(cls, repo: str = REPO) -> "QuestionEncoder | None":
        """$CUBBY_QUESTION_ENCODER (a folder), else the Hugging Face cache's copy of ``repo``; None if absent."""
        cands = [Path(os.environ["CUBBY_QUESTION_ENCODER"])] if os.environ.get("CUBBY_QUESTION_ENCODER") else []
        snap = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / (
            "models--" + repo.replace("/", "--")) / "snapshots"
        cands += sorted(snap.iterdir()) if snap.is_dir() else []
        for c in cands:
            if (c / "model.safetensors").exists() and (c / "tokenizer.json").exists():
                return cls(c)
        return None

    def _ln(self, x, p):
        import torch
        return torch.nn.functional.layer_norm(x, x.shape[-1:], self.W[p + ".weight"], self.W[p + ".bias"], 1e-12)

    def _lin(self, x, p):
        return x @ self.W[p + ".weight"].T + self.W[p + ".bias"]

    def encode(self, text: str) -> np.ndarray:
        if text not in self._memo:
            self._memo[text] = self._encode(text)
        return self._memo[text]

    def _encode(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            ids = torch.tensor(self.tok.encode(text).ids)
            T, H = len(ids), self.heads
            d = self.dim // H
            x = (self.W["embeddings.word_embeddings.weight"][ids] + self.W["embeddings.position_embeddings.weight"][:T]
                 + self.W["embeddings.token_type_embeddings.weight"][0])
            x = self._ln(x, "embeddings.LayerNorm")
            for i in range(self.cfg["num_hidden_layers"]):
                p = f"encoder.layer.{i}"
                q, k, v = (self._lin(x, f"{p}.attention.self.{n}").view(T, H, d).transpose(0, 1)
                           for n in ("query", "key", "value"))
                a = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(d), -1) @ v
                x = self._ln(x + self._lin(a.transpose(0, 1).reshape(T, -1), f"{p}.attention.output.dense"),
                             f"{p}.attention.output.LayerNorm")
                x = self._ln(x + self._lin(torch.nn.functional.gelu(self._lin(x, f"{p}.intermediate.dense")),
                                           f"{p}.output.dense"), f"{p}.output.LayerNorm")
            v = x.mean(0)
            return (v / (v.norm() + 1e-9)).numpy().astype(np.float32)


__all__ = ["QuestionEncoder", "REPO"]
