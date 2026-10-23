"""cubbyllm/worlds/fast_words.py -- the fast word-table encoder, ported from MoWM (order 0033; Cowork's answer 3 in
0031's review). Before the port the server loaded it by file path from a mowm checkout beside this one
(``validation/exp_m3_domain_routing._load_semantic_words``); now it is the package's own.

The source is ``mowm/encoding/semantic_words.py`` (Nick's MoWM repo), and the code is that file's, unchanged except
for one thing. The offline teacher's default (a sentence-transformer that ``build`` loaded when no ``embed_fn`` was
given) is left out: ``build`` takes the teacher as an argument, so this module imports only numpy. The tables in
``$CUBBY_MODELS`` load as before (``from_npz``), and ``tests/worlds/test_fast_words.py`` pins the port against mowm's
module while mowm is there; ``validation/pin_fast_words.py`` does so on the production table.

Serving is model-free: text -> words -> table lookup -> IDF-weighted bundle of (k, l) block codes. Recipe (built
offline): unit teacher vectors per WORD, minus the anisotropy direction (mean + top principal component), projected
through an orthonormal map into (k, l) block space (cosine-exact), then 0.5/0.5-blended with each word's seeded
one-hot SIGNAL code, so that exact surface overlap keeps contributing. An unseen word falls back to its signal code,
so out-of-table text still routes on surface identity.

Wired: WIRED (``cubby_serve.build`` loads the server's table with it; ``worlds.base.FactStore`` and
``reasoning.hippocampus`` read its codes).
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

__wiring__ = "WIRED"

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def split_words(text: str) -> set[str]:
    """Lowercased word tokens of length >= 2 (the validated serving unit)."""
    return {w for w in _WORD_SPLIT.split(text.lower()) if len(w) >= 2}


def signal_code(word: str, k: int, l: int) -> np.ndarray:  # noqa: E741
    """Seeded one-hot-per-block code for a word: unit norm, flat (k*l,). BLAKE2b over the namespaced gram ``"w:<word>"``,
    so tables and screens reproduce each other bit-for-bit."""
    digest = hashlib.blake2b(f"w:{word}".encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") % (2**63)
    idx = np.random.default_rng(seed).integers(0, l, size=k)
    v = np.zeros(k * l, dtype=np.float32)
    v[np.arange(k) * l + idx] = 1.0 / np.sqrt(k)
    return v


def orthonormal_projection(latent_dim: int, k: int, l: int, seed: int = 0) -> np.ndarray:  # noqa: E741
    """(k*l, latent_dim) with orthonormal columns: a cosine-EXACT embedding of the teacher space into block space
    ((Px)·(Py) = x·y)."""
    if k * l < latent_dim:
        raise ValueError(f"need k*l={k * l} >= latent_dim={latent_dim}")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((k * l, latent_dim)))
    return Q.astype(np.float32)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def fit_anisotropy(E: np.ndarray, n_pc: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The shared-direction statistics (mean + top PCs) of an embedding table. Fit them on broad external vocabulary
    when there is one: in-domain statistics inflated routing by about 0.03 macro."""
    mu = E.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(E - mu, full_matrices=False)
    return mu, Vt[:n_pc]


def remove_anisotropy(E: np.ndarray, mu: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """All-but-the-top: subtract the mean and the top PC(s), re-unit rows."""
    Ec = E - mu
    for v in pcs:
        Ec = Ec - np.outer(Ec @ v, v)
    return _unit_rows(Ec)


class FastWordEncoder:
    """Serve-time encoder over a distilled word table. numpy-only. ``encode(text)`` -> (k, l) float32 block code: the
    IDF-weighted bundle of each word's table vector, with the seeded signal code for a word the table lacks."""

    def __init__(self, words: list[str], vectors: np.ndarray, idf: dict[str, float], default_idf: float, k: int,
                 l: int, meta: dict | None = None) -> None:  # noqa: E741 -- `vectors`: (n, k*l) float32, unit rows
        self.k, self.l = int(k), int(l)
        self._vec = {w: np.asarray(vectors[i], dtype=np.float32) for i, w in enumerate(words)}
        self._idf = dict(idf)
        self.default_idf = float(default_idf)
        self.meta = dict(meta or {})
        self._sig_cache: dict[str, np.ndarray] = {}

    @classmethod
    def build(cls, words: list[str], embed_fn, idf: dict[str, float], default_idf: float, *, k: int = 80,
              l: int = 128, proj_seed: int = 0, anisotropy: tuple[np.ndarray, np.ndarray] | None = None,  # noqa: E741
              signal_blend: float = 0.5, meta: dict | None = None) -> "FastWordEncoder":
        """Distill ``embed_fn`` (the teacher, called here once, offline) into a table. ``anisotropy`` is (mu, pcs) fit
        elsewhere with ``fit_anisotropy``; None fits it on this table's own vectors (the in-domain form)."""
        words = sorted(set(words))
        E = _unit_rows(np.asarray(embed_fn(list(words)), dtype=np.float32))
        if anisotropy is None:
            anisotropy = fit_anisotropy(E)
        E_pc = remove_anisotropy(E, *anisotropy)
        P = orthonormal_projection(E.shape[1], k, l, seed=proj_seed)
        proj = _unit_rows(E_pc @ P.T)
        vectors = np.stack([(1.0 - signal_blend) * proj[i] + signal_blend * signal_code(w, k, l)
                            for i, w in enumerate(words)]).astype(np.float32)
        info = {"teacher_dim": int(E.shape[1]), "proj_seed": proj_seed, "signal_blend": signal_blend, "n_words": len(words)}
        info.update(meta or {})
        return cls(words, vectors, idf, default_idf, k, l, meta=info)

    def encode(self, text: str) -> np.ndarray:
        """(k, l) float32 block code for ``text``. Lookup and adds; no model."""
        acc = np.zeros(self.k * self.l, dtype=np.float64)
        for w in split_words(text):
            v = self._vec.get(w)
            if v is None:
                v = self._sig_cache.get(w)
                if v is None:
                    v = signal_code(w, self.k, self.l)
                    self._sig_cache[w] = v
            acc += self._idf.get(w, self.default_idf) * v.astype(np.float64)
        return acc.reshape(self.k, self.l).astype(np.float32)

    def save_npz(self, path, dtype=np.float32) -> None:
        """``dtype=np.float16`` halves a production-size table on disk; ``from_npz`` casts back to float32 (the cosine
        moves by about 1e-3)."""
        words = sorted(self._vec)
        np.savez_compressed(path, words=np.array(words), vectors=np.stack([self._vec[w] for w in words]).astype(dtype),
                            idf_words=np.array(sorted(self._idf)),
                            idf_values=np.array([self._idf[w] for w in sorted(self._idf)], dtype=np.float32),
                            default_idf=np.array(self.default_idf, dtype=np.float32), k=np.array(self.k), l=np.array(self.l),
                            meta=np.array(json.dumps(self.meta)))

    @classmethod
    def from_npz(cls, path) -> "FastWordEncoder":
        d = np.load(path, allow_pickle=True)
        idf = {str(w): float(v) for w, v in zip(d["idf_words"], d["idf_values"])}
        return cls([str(w) for w in d["words"]], d["vectors"], idf, float(d["default_idf"]), int(d["k"]), int(d["l"]),
                   meta=json.loads(str(d["meta"])))
