"""r9b_arms -- order 0046, R9b: block codes computed four ways on the CPU (arm 5, grilly2's Vulkan path, is timed on the
card in its own slot), and MAP-bipolar beside them. Bench code: nothing here changes cubemind, cubelang or grilly2.

A block code is k blocks of length l. Arms 1-3 hold it dense (k x l float32), arm 4 as one index a block:
1. ``numpy``: cubemind's ``BlockCodes`` (R9's path), the reference.
2. ``scipy``: the same per-block circular convolution with ``scipy.fft``'s real FFT, in float32.
3. ``spectral``: a frame held as its per-block real FFT (k x (l/2 + 1) complex64); bind = element product, bundle =
   sum with each block renormalized to sum 1 (its DC term), similarity through Parseval; an FFT only when a vector
   enters (``to_spec``) and when a frame leaves (``from_spec``).
4. ``index`` (numba): a key is one u8 a block; bind of two keys = index sum mod l; a frame against a key = the k values
   gathered; a frame bound with a key = a per-block shift; bundle to a key = the per-block vote (the most frequent index,
   ties to the lowest), which drops information: its accuracy is read again (R9's tasks).
MAP-bipolar: numpy int8 (R9's numpy pair) and packed uint64 with popcount for the index job; cubelang's Rust is timed by
``r9_bench`` (``examples/r9_bench.rs``).

Wired: STANDALONE (order 0046's bench).
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from numba import njit, prange

__wiring__ = "STANDALONE"


# ── arm 2: scipy.fft, float32 ──────────────────────────────────────────────────────────────────────────────

def scipy_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    l = a.shape[-1]
    return sfft.irfft(sfft.rfft(a, axis=-1) * sfft.rfft(b, axis=-1), n=l, axis=-1).astype(np.float32)


def scipy_unbind(c: np.ndarray, key: np.ndarray) -> np.ndarray:
    l = c.shape[-1]
    return sfft.irfft(sfft.rfft(c, axis=-1) * np.conj(sfft.rfft(key, axis=-1)), n=l, axis=-1).astype(np.float32)


def bundle_dense(vs: np.ndarray) -> np.ndarray:
    """BlockCodes.bundle(normalize=True): the sum, each block divided by its own sum."""
    s = vs.sum(axis=0).astype(np.float32)
    t = s.sum(axis=-1, keepdims=True)
    return (s / np.where(t == 0, 1.0, t)).astype(np.float32)


def similarity_dense(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b) / a.shape[-2])


# ── arm 3: spectral frames ─────────────────────────────────────────────────────────────────────────────────

def to_spec(a: np.ndarray) -> np.ndarray:
    return np.fft.rfft(a, axis=-1).astype(np.complex64)


def from_spec(x: np.ndarray, l: int) -> np.ndarray:
    return np.fft.irfft(x, n=l, axis=-1).astype(np.float32)


def spec_bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y


def spec_unbind(x: np.ndarray, key: np.ndarray) -> np.ndarray:
    return x * np.conj(key)


def spec_bundle(xs: np.ndarray) -> np.ndarray:
    s = xs.sum(axis=0)
    dc = s[..., :1].real
    return (s / np.where(dc == 0, 1.0, dc)).astype(np.complex64)


def _parseval_weights(l: int) -> np.ndarray:
    w = np.full(l // 2 + 1, 2.0, dtype=np.float32)
    w[0] = 1.0
    if l % 2 == 0:
        w[-1] = 1.0
    return w / l


def spec_similarity(x: np.ndarray, y: np.ndarray, l: int) -> float:
    """(1/k) sum(a * b), from the spectra: sum_t a_t b_t = (1/l) sum_f w_f Re(X_f conj(Y_f))."""
    k = x.shape[-2]
    return float(np.sum(_parseval_weights(l) * (x * np.conj(y)).real) / k)


def spec_similarity_batch(x: np.ndarray, ys: np.ndarray, l: int) -> np.ndarray:
    k = x.shape[-2]
    w = _parseval_weights(l)
    return np.einsum("kf,nkf->n", (np.conj(x) * w).astype(np.complex64), ys).real / k


# ── arm 4: the index form (numba) ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def idx_bind(a, b, l):
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = (int(a[i]) + int(b[i])) % l
    return out


@njit(cache=True)
def idx_unbind(a, key, l):
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = (int(a[i]) - int(key[i])) % l
    return out


@njit(cache=True)
def idx_bundle(keys, l):
    """The per-block vote: the most frequent index of each block, ties to the lowest."""
    m, k = keys.shape
    out = np.empty(k, dtype=keys.dtype)
    counts = np.zeros(l, dtype=np.int32)
    for i in range(k):
        counts[:] = 0
        for j in range(m):
            counts[keys[j, i]] += 1
        out[i] = np.argmax(counts)
    return out


@njit(cache=True)
def idx_similarity(a, b):
    s = 0
    for i in range(a.shape[0]):
        s += a[i] == b[i]
    return s / a.shape[0]


@njit(cache=True)
def frame_against_key(frame, key):
    """A dense frame scored against a one-hot key: the k values gathered, over k."""
    s = 0.0
    for i in range(key.shape[0]):
        s += frame[i, key[i]]
    return s / key.shape[0]


@njit(cache=True)
def frame_shift(frame, key):
    """A dense frame bound with a one-hot key: each block rotated by the key's index (convolution with a one-hot)."""
    k, l = frame.shape
    out = np.empty_like(frame)
    for i in range(k):
        r = key[i]
        for t in range(l):
            out[i, (t + r) % l] = frame[i, t]
    return out


@njit(parallel=True, cache=True)
def idx_scores(q, keys):
    """One key query against N keys: the share of equal blocks, for each."""
    n, k = keys.shape
    out = np.empty(n, dtype=np.float32)
    for j in prange(n):
        s = 0
        for i in range(k):
            s += keys[j, i] == q[i]
        out[j] = s / k
    return out


# ── MAP-bipolar ────────────────────────────────────────────────────────────────────────────────────────────

def map_pack(v: np.ndarray) -> np.ndarray:
    """int8 {-1, +1} rows -> packed uint64 (+1 -> 1)."""
    bits = np.packbits((v > 0).astype(np.uint8), axis=-1, bitorder="little")
    return bits.view(np.uint64)


def map_scores_packed(q: np.ndarray, keys: np.ndarray, dim: int) -> np.ndarray:
    """Cosine of one packed query against N packed keys: (D - 2 * hamming) / D."""
    ham = np.bitwise_count(np.bitwise_xor(keys, q)).sum(axis=-1)
    return (dim - 2.0 * ham) / dim


def top(scores: np.ndarray, t: int) -> np.ndarray:
    t = min(t, scores.shape[-1])
    part = np.argpartition(-scores, t - 1, axis=-1)[..., :t]
    return np.take_along_axis(part, np.argsort(-np.take_along_axis(scores, part, axis=-1), axis=-1), axis=-1)


__all__ = ["bundle_dense", "frame_against_key", "frame_shift", "from_spec", "idx_bind", "idx_bundle", "idx_scores",
           "idx_similarity", "idx_unbind", "map_pack", "map_scores_packed", "scipy_bind", "scipy_unbind",
           "similarity_dense", "spec_bind", "spec_bundle", "spec_similarity", "spec_similarity_batch", "spec_unbind",
           "to_spec", "top"]
