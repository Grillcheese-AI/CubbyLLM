"""BlockCodeVSA — the single seam onto grilly's VSA algebra.

Wired: WIRED — everything that binds/unbinds goes through this facade.

Binding algebra is the DECIDED choice (H-B5, user 2026-07-23):
grilly ``BlockCodeOps`` (IBM-NVSA sparse block codes, k x l). This facade
mirrors cubemind's proven fallback:

  1. grilly.experimental.vsa.block_ops.BlockCodeOps  (grilly, incl. Vulkan tier)
  2. numpy reference (the exact same per-block circular-convolution math)

**This is the only module in the whole package that names ``grilly``.** The
enforcement test ``test_ops_is_sole_grilly_importer`` depends on that.

All hypervectors are ``float32`` block codes of shape ``(..., k, l)`` where
``d = k * l``. Discrete codes are one-hot per block; continuous codes are
per-block distributions. Binding of one-hot codes recovers exactly under
``unbind`` (verified in the tests and by the H-B5 3-way prototype).
"""
from __future__ import annotations

import numpy as np

from ..core.config import SEED
from ..core.protocols import Wiring

_EPS = 1e-20


def _detect_backend() -> str:
    """Return the highest available grilly tier, or 'numpy' if grilly absent."""
    try:  # tier 1: C++/Vulkan bridge (perf path; algebra still via BlockCodeOps)
        from grilly.backend import _bridge  # noqa: F401

        if hasattr(_bridge, "blockcode_bind"):
            return "grilly_bridge"
    except Exception:
        pass
    try:  # tier 2: grilly python BlockCodeOps
        from grilly.experimental.vsa.block_ops import BlockCodeOps  # noqa: F401

        return "grilly_python"
    except Exception:
        pass
    return "numpy"  # tier 3: reference


def _load_grilly_ops():
    try:
        from grilly.experimental.vsa.block_ops import BlockCodeOps

        return BlockCodeOps
    except Exception:
        return None


class BlockCodeVSA:
    """Facade over grilly BlockCodeOps for (k, l) block-code hypervectors.

    ``k`` and ``l`` set the block shape (D = k * l). The backend is detected once
    at construction; if grilly is unavailable, a numpy reference implementing the
    identical math is used so the algebra always works.
    """

    def __init__(self, k: int = 80, l: int = 128) -> None:
        self.k = int(k)
        self.l = int(l)
        self._backend = _detect_backend()
        self._ops = _load_grilly_ops()  # None -> use numpy fallback

    def backend(self) -> str:
        """Which fallback tier is active: 'grilly_bridge' | 'grilly_python' | 'numpy'."""
        return self._backend

    # ── algebra ───────────────────────────────────────────────────────────
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-block circular convolution (association)."""
        if self._ops is not None:
            return self._ops.bind(a, b)
        fa = np.fft.fft(a, axis=-1)
        fb = np.fft.fft(b, axis=-1)
        return np.real(np.fft.ifft(fa * fb, axis=-1)).astype(np.float32)

    def unbind(self, composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        """Per-block circular correlation — recover the other factor."""
        if self._ops is not None:
            return self._ops.unbind(composite, known)
        fc = np.fft.fft(composite, axis=-1)
        fk = np.fft.fft(known, axis=-1)
        return np.real(np.fft.ifft(fc * np.conj(fk), axis=-1)).astype(np.float32)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Superposition — normalized per-block sum."""
        if not vectors:
            raise ValueError("cannot bundle an empty list")
        if self._ops is not None:
            return self._ops.bundle(list(vectors), normalize=True)
        s = np.sum(vectors, axis=0).astype(np.float32)
        block_sums = s.sum(axis=-1, keepdims=True)
        block_sums = np.where(block_sums == 0, 1.0, block_sums)
        return (s / block_sums).astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """IBM-NVSA similarity (1/k) * sum(a*b)."""
        if self._ops is not None:
            return float(self._ops.similarity(a, b))
        k = a.shape[-2]
        return float(np.sum(a * b) / k)

    def similarity_batch(self, query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Similarity of ``query`` (k, l) against every entry of ``codebook`` (n, k, l)."""
        if self._ops is not None:
            return np.asarray(self._ops.similarity_batch(query, codebook))
        k = query.shape[-2]
        return (codebook.reshape(codebook.shape[0], -1) @ query.reshape(-1) / k).astype(
            np.float32
        )

    def codebook(self, n: int, *, orthogonal: bool = True) -> np.ndarray:
        """Generate an (n, k, l) discrete codebook (deterministic under SEED).

        ``orthogonal=True`` (grilly default) builds entries by successive binding
        (cb[i] = bind(cb[i-1], cb[1])) — an algebraically structured set for
        RESONATOR FACTORIZATION slots. For ROLE/FILLER binding use
        ``orthogonal=False`` (independent random atoms): the structured set
        produces systematic unbind crosstalk onto a neighbor, whereas
        independent atoms recover cleanly (verified in the binding tests).
        """
        if self._ops is not None:
            return self._ops.codebook_discrete(
                self.k, self.l, n, seed=SEED, orthogonal=orthogonal
            )
        rng = np.random.default_rng(SEED)
        cb = np.zeros((n, self.k, self.l), dtype=np.float32)
        idx = rng.integers(0, self.l, size=(n, self.k))
        np.put_along_axis(cb, idx[:, :, None], 1.0, axis=2)
        return cb

    def zero(self) -> np.ndarray:
        """Identity element (position-0 hot per block): bind(x, zero) == x."""
        v = np.zeros((self.k, self.l), dtype=np.float32)
        v[:, 0] = 1.0
        return v


__wiring__ = Wiring.WIRED
