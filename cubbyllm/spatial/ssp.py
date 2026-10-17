"""SSP -- a place as a unit vector in a Hilbert space, by fractional power encoding.

Wired: STANDALONE (2026-10-17, Nick: "implement spatial understanding provably at the input
level"; "Hilbert spaces could be useful for the spatial features").

A coordinate ``p`` (3-D by default) becomes ``S(p) = IFFT(exp(i * Theta p / scale))`` with
``Theta`` a fixed table of random phases, one per frequency per axis (Plate's fractional
power of a unitary vector; Komer & Eliasmith's spatial semantic pointers; Frady, Kleyko &
Sommer's kernel reading). Everything the host and the trunk rely on is a theorem of that
construction, not a property the model has to learn, and ``tests/spatial/test_ssp.py``
checks each one to float precision:

    S(p + d) = S(p) (*) S(d)             moving is binding (circular convolution)
    S(0)     = e_0, the identity          a token with no place is bound to nothing
    <S(p), S(q)> = k(p - q)               a kernel of the displacement only, 1 at 0, falling
                                          with distance at the rate ``scale`` sets
    sum_{q in box} S(q)                   a region, in closed form (a geometric series per axis)
    argmax_q <v, S(q)> = p for v = S(p)   a place is read back exactly on the lattice

What it brings CubbyLLM: the trunk's RoPE is this very encoding in one dimension (a rotation
per position); ``S(p)`` is its 3-D sibling at the input, so the same algebra that already
places tokens in a sequence places blocks in a world. A conventional model learns "north of"
from text and still gets coordinates wrong; here the geometry is algebra with provable
equivariance, and the model only has to choose the ops.

numpy only; the torch twin that binds the trunk's embeddings is ``cubbyllm.model.spatial``
and shares this file's phase table, so host and trunk agree on what ``S(p)`` is.
"""
from __future__ import annotations

import numpy as np

from ..core.protocols import Wiring

__all__ = ["SSP"]


def _phases(d: int, dims: int, seed: int) -> np.ndarray:
    """``(d // 2 + 1, dims)`` phases in ``(-pi, pi)``: zero at DC and at Nyquist (both must be
    real for ``S(p)`` to be a real vector), uniform elsewhere."""
    if d < 4 or d % 2:
        raise ValueError(f"d must be even and at least 4, got {d}")
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, size=(d // 2 + 1, dims))
    th[0] = 0.0
    th[-1] = 0.0
    return th


class SSP:
    """Fractional power encoding of ``dims``-D coordinates into ``d``-D real unit vectors.

    ``scale`` is the length unit: two places ``scale`` apart have similarity about 0.6, a
    lattice cell apart (``scale=1``) they are nearly orthogonal. ``seed`` fixes the phases;
    the same ``(d, dims, scale, seed)`` gives the same space everywhere."""

    def __init__(self, d: int = 512, dims: int = 3, scale: float = 2.0, seed: int = 0) -> None:
        self.d, self.dims, self.scale, self.seed = int(d), int(dims), float(scale), int(seed)
        self.phases = _phases(self.d, self.dims, self.seed)            # (d/2+1, dims)

    # -- the encoding ----------------------------------------------------------------------
    def spectrum(self, p) -> np.ndarray:
        """``exp(i Theta p / scale)``: ``S(p)``'s rfft, ``(..., d/2+1)`` complex."""
        p = np.asarray(p, dtype=np.float64)
        if p.shape[-1] != self.dims:
            raise ValueError(f"expected coordinates with {self.dims} components, got shape {p.shape}")
        return np.exp(1j * (p @ self.phases.T) / self.scale)

    def encode(self, p) -> np.ndarray:
        """``S(p)`` for coordinates ``p`` ``(..., dims)`` -> ``(..., d)`` real, unit norm."""
        return np.fft.irfft(self.spectrum(p), n=self.d, axis=-1)

    def identity(self) -> np.ndarray:
        """``S(0)``: the delta vector, the unit of binding."""
        return self.encode(np.zeros(self.dims))

    # -- the algebra --------------------------------------------------------------------
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution ``a (*) b``; ``S(p) (*) S(q) = S(p + q)``."""
        return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1), n=self.d, axis=-1)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        """The exact inverse of a unitary vector (``S(-p)`` for ``S(p)``): the conjugate
        spectrum. For a vector that is not unitary it is the approximate inverse."""
        return np.fft.irfft(np.conj(np.fft.rfft(a, axis=-1)), n=self.d, axis=-1)

    def unbind(self, composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        """``composite (*) known^-1``: what ``known`` was bound with."""
        return self.bind(composite, self.inverse(known))

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The inner product over the last axis (cosine, for unit vectors)."""
        return np.sum(a * b, axis=-1)

    @staticmethod
    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n > 0, n, 1.0)

    # -- regions and reading back -------------------------------------------------------
    def box(self, lo, hi) -> np.ndarray:
        """The unit vector of every lattice cell in the box ``lo <= q <= hi`` (inclusive, ints):
        ``sum_q S(q)`` in closed form -- a geometric series per axis, so a 1000-cell box costs
        the same as one cell."""
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        if lo.shape != (self.dims,) or hi.shape != (self.dims,) or np.any(hi < lo):
            raise ValueError("box wants lo <= hi, one integer per axis")
        n = hi - lo + 1.0
        spec = np.ones(self.d // 2 + 1, dtype=np.complex128)
        for a in range(self.dims):
            th = self.phases[:, a] / self.scale
            z = np.exp(1j * th)
            with np.errstate(divide="ignore", invalid="ignore"):
                series = np.exp(1j * th * lo[a]) * (1.0 - z ** n[a]) / (1.0 - z)
            series = np.where(np.isclose(z, 1.0), n[a], series)        # theta = 0: n cells, no rotation
            spec = spec * series
        return self.unit(np.fft.irfft(spec, n=self.d, axis=-1))

    def region(self, cells) -> np.ndarray:
        """The unit vector of an explicit set of cells (any shape of region)."""
        cells = np.asarray(cells, dtype=np.float64).reshape(-1, self.dims)
        return self.unit(self.encode(cells).sum(axis=0))

    def decode(self, v: np.ndarray, candidates) -> tuple[np.ndarray, float]:
        """The candidate coordinate whose ``S`` is most similar to ``v``, and that similarity."""
        cands = np.asarray(candidates, dtype=np.float64).reshape(-1, self.dims)
        s = self.encode(cands) @ np.asarray(v, dtype=np.float64)
        i = int(np.argmax(s))
        return cands[i], float(s[i])

    def kernel(self, delta) -> np.ndarray:
        """``k(delta) = <S(p), S(p + delta)>`` for displacements ``delta`` ``(..., dims)``."""
        return self.similarity(self.identity(), self.encode(delta))


__wiring__ = Wiring.STANDALONE
