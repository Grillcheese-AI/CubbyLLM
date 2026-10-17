"""The Hilbert curve -- a line through a box of cells that keeps neighbours near.

Wired: STANDALONE (2026-10-17). The world is 3-D and the trunk reads a sequence; a window of
512 tokens and an episodic store that ages by sequence distance both ask that what the trunk
sees at once be a compact piece of the world. Row-major order makes 512 consecutive cells a
2 x 16 x 16 slab; on the Hilbert curve every two consecutive cells share a face and every
``8**j`` consecutive cells fill an aligned cube of side ``2**j`` (theorems of the recursive
construction: 512 cells are an 8 x 8 x 8 block), so cells near on the curve are near in space
(the reverse holds for no curve: a cube has seams). ``order`` sorts any set of cells along it;
``curve`` enumerates a box along it; both compose with ``ssp.SSP``: the curve says in which
order the trunk reads the blocks, ``S(cell)`` says where each one is.

Skilling's algorithm (``Programming the Hilbert curve``, 2004), vectorised over numpy arrays
of points, any number of dimensions, ``bits`` per axis.
"""
from __future__ import annotations

import numpy as np

from ..core.protocols import Wiring

__all__ = ["index", "point", "order", "curve", "bits_for"]


def bits_for(extent: int) -> int:
    """Bits per axis for coordinates in ``[0, extent)``."""
    return max(1, int(np.ceil(np.log2(max(int(extent), 2)))))


def _gray_to_axes(x: np.ndarray, bits: int) -> np.ndarray:
    """Transposed Hilbert index ``(N, dims)`` -> axes, in place on a copy."""
    x = x.copy()
    n = x.shape[1]
    top = 1 << bits
    t = x[:, n - 1] >> 1
    for i in range(n - 1, 0, -1):
        x[:, i] ^= x[:, i - 1]
    x[:, 0] ^= t
    q = 2
    while q != top:
        p = q - 1
        for i in range(n - 1, -1, -1):
            inv = (x[:, i] & q) != 0
            x[inv, 0] ^= p
            ex = ~inv
            t = (x[ex, 0] ^ x[ex, i]) & p
            x[ex, 0] ^= t
            x[ex, i] ^= t
        q <<= 1
    return x


def _axes_to_gray(x: np.ndarray, bits: int) -> np.ndarray:
    """Axes ``(N, dims)`` -> transposed Hilbert index, on a copy."""
    x = x.copy()
    n = x.shape[1]
    m = 1 << (bits - 1)
    q = m
    while q > 1:
        p = q - 1
        for i in range(n):
            inv = (x[:, i] & q) != 0
            x[inv, 0] ^= p
            ex = ~inv
            t = (x[ex, 0] ^ x[ex, i]) & p
            x[ex, 0] ^= t
            x[ex, i] ^= t
        q >>= 1
    for i in range(1, n):
        x[:, i] ^= x[:, i - 1]
    t = np.zeros(x.shape[0], dtype=x.dtype)
    q = m
    while q > 1:
        t ^= np.where((x[:, n - 1] & q) != 0, q - 1, 0)
        q >>= 1
    for i in range(n):
        x[:, i] ^= t
    return x


def index(points, bits: int) -> np.ndarray:
    """Hilbert index of each point ``(N, dims)`` with coordinates in ``[0, 2**bits)``."""
    x = np.asarray(points, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("points must be (N, dims)")
    if np.any(x < 0) or np.any(x >= (1 << bits)):
        raise ValueError(f"coordinates must lie in [0, {1 << bits}) for {bits} bits")
    g = _axes_to_gray(x, bits)
    n = x.shape[1]
    h = np.zeros(x.shape[0], dtype=np.int64)
    for b in range(bits - 1, -1, -1):                 # interleave: high bit of each axis first
        for i in range(n):
            h = (h << 1) | ((g[:, i] >> b) & 1)
    return h


def point(indices, bits: int, dims: int = 3) -> np.ndarray:
    """The inverse of ``index``: ``(N,)`` indices -> ``(N, dims)`` points."""
    h = np.asarray(indices, dtype=np.int64).reshape(-1)
    g = np.zeros((h.shape[0], dims), dtype=np.int64)
    k = bits * dims
    for b in range(bits - 1, -1, -1):
        for i in range(dims):
            k -= 1
            g[:, i] |= ((h >> k) & 1) << b
    return _gray_to_axes(g, bits)


def order(cells) -> np.ndarray:
    """The cells ``(N, dims)`` sorted along the curve through their bounding box
    (coordinates may be negative; the box is shifted to the origin to index it)."""
    c = np.asarray(cells, dtype=np.int64)
    if c.ndim != 2 or c.shape[0] == 0:
        return c.reshape(0, c.shape[-1] if c.ndim == 2 else 3)
    lo = c.min(axis=0)
    shifted = c - lo
    bits = bits_for(int(shifted.max()) + 1)
    return c[np.argsort(index(shifted, bits), kind="stable")]


def curve(lo, hi) -> np.ndarray:
    """Every cell of the box ``lo <= q <= hi`` (inclusive) in curve order, ``(N, dims)``."""
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    if lo.shape != hi.shape or lo.ndim != 1 or np.any(hi < lo):
        raise ValueError("curve wants lo <= hi, one integer per axis")
    dims = lo.shape[0]
    bits = bits_for(int((hi - lo).max()) + 1)
    pts = point(np.arange(1 << (bits * dims)), bits, dims) + lo
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[keep]


__wiring__ = Wiring.STANDALONE
