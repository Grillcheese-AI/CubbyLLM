"""ear_lexicon — word meanings for the ear (standin/ear.py), from a static embedding table.

Wired: optional, under `ear.Ear(lexicon=...)`. Nothing in cubbyllm/ imports this.

A model2vec-style folder (vocab.txt + model.safetensors: one vector per WordPiece token, the
potion tables fastword was distilled from) read with numpy alone: no model runs, a word is a
lookup, a word the vocabulary lacks is the mean of its WordPiece pieces. What it brings to the
ear: "hopeless" lands near "useless" and "brilliant" near "great" without either being taught, so
a contrast taught on a few words generalises to their neighbours. Hashed codes still carry
spelling, negation and French; this carries meaning where the table has it.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

DEFAULT = "minishlab/potion-retrieval-32M"


def _strip(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", w.lower()) if unicodedata.category(c) != "Mn")


def _safetensors(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    name = next(k for k in head if k != "__metadata__")
    t = head[name]
    dt = {"F32": np.float32, "F16": np.float16, "BF16": None}[t["dtype"]]
    s, e = t["data_offsets"]
    if dt is None:                                      # bfloat16: the high half of a float32
        u = np.frombuffer(raw[8 + n + s:8 + n + e], np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(t["shape"])
    return np.frombuffer(raw[8 + n + s:8 + n + e], dt).astype(np.float32).reshape(t["shape"])


class Lexicon:
    def __init__(self, folder, name: str | None = None) -> None:
        folder = Path(folder)
        self.name = name or folder.name
        toks = (folder / "vocab.txt").read_text(encoding="utf-8").splitlines()
        self.ids = {t: i for i, t in enumerate(toks)}
        self.E = _safetensors(folder / "model.safetensors")   # (row norms kept: they weigh a token's say)
        self.dim = self.E.shape[1]
        self._memo: dict[str, np.ndarray | None] = {}

    @classmethod
    def find(cls, repo: str = DEFAULT) -> "Lexicon | None":
        """$CUBBY_EAR_LEXICON (a folder), else the Hugging Face cache's copy of `repo`; None if absent."""
        cands = [Path(os.environ["CUBBY_EAR_LEXICON"])] if os.environ.get("CUBBY_EAR_LEXICON") else []
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
        if snap.is_dir():
            cands += sorted(snap.iterdir())
        for c in cands:
            if (c / "vocab.txt").is_file() and (c / "model.safetensors").exists():
                return cls(c, None if os.environ.get("CUBBY_EAR_LEXICON") else repo)
        return None

    def pieces(self, word: str) -> list[int]:
        w, out, i = _strip(word), [], 0
        if w in self.ids:
            return [self.ids[w]]
        while i < len(w):
            j = len(w)
            while j > i and ((w[i:j] if i == 0 else "##" + w[i:j]) not in self.ids):
                j -= 1
            if j == i:
                return out
            out.append(self.ids[w[i:j] if i == 0 else "##" + w[i:j]])
            i = j
        return out

    def word(self, w: str) -> np.ndarray | None:
        if w not in self._memo:
            p = self.pieces(w)
            self._memo[w] = self.E[p].mean(axis=0) if p else None
        return self._memo[w]

    def encode(self, text: str) -> np.ndarray:
        """A sentence as model2vec reads it: the unit mean of its WordPiece rows (zeros if none is known). The shape
        ``worlds.FactStore(enc=...)`` takes; order 0028's routing bench scores plugins with it (``--encoder potion``)."""
        ids = [i for w in re.findall(r"[\w']+", text) for i in self.pieces(w)]
        if not ids:
            return np.zeros(self.dim, np.float32)
        v = self.E[ids].mean(axis=0)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


__all__ = ["Lexicon", "DEFAULT"]
