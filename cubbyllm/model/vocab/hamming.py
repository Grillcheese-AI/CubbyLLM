"""cubbyllm/model/vocab/hamming.py -- a Hamming pass over a trained output head's rows (order 0068, route 5).

The pass that looks at every row of the 128k head, cheaply. A row is kept as the signs of its weights, one bit each,
and its norm; the trunk's state is read as its signs. A row's score is its norm times (the signs that agree minus
those that do not): norm * (d - 2 * Hamming distance), which is sign(h) . sign(w) * |w|. The pass reads 1/32 of the
head's bytes and nothing is trained. Its best N rows join a context shortlist (``shortlist.py``, an arm ending in
"+ham") and the union is scored exactly with the trained rows.

The bits are packed the way grilly2's bit codes are (``ops/packed.py``: 32 bits a word, little bit order), so a card
can run the pass with a Hamming kernel. grilly2's own top-k ranks by distance alone; this pass weights each row by
its norm, and ``HammingIndex.scores`` is the reference such a kernel must match. A zero counts as negative on both
sides. Ties at the Nth place go to the lower token id, so the list is the same on every machine.

numpy only (``cubbyllm`` stays torch-free).

Wired: WIRED (``ShortlistIndex.shortlist`` reads it for an arm ending in "+ham"; ``TopKRetrievalHead.attach_shortlist``
builds it from the head's own rows).
"""
from __future__ import annotations

import numpy as np

__wiring__ = "WIRED"

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(x: np.ndarray) -> np.ndarray:
    """The set bits of each uint32 word (numpy 2's ``bitwise_count``; a byte table before it)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP8[np.ascontiguousarray(x).view(np.uint8)].reshape(*x.shape, 4).sum(-1, dtype=np.uint8)


def pack(x: np.ndarray) -> np.ndarray:
    """(n, d) values -> (n, ceil(d / 32)) uint32 words: bit i of a row is ``x[i] > 0``; the padding bits are zero."""
    b = np.packbits(np.atleast_2d(np.asarray(x)) > 0, axis=1, bitorder="little")
    b = np.pad(b, ((0, 0), (0, (-b.shape[1]) % 4)))
    return np.ascontiguousarray(b).view(np.uint32)


class HammingIndex:
    """A head's rows as bits and norms. ``top`` is the pass: every row scored, the best N returned."""

    def __init__(self, bits: np.ndarray, norms: np.ndarray, dim: int) -> None:
        self.bits = np.ascontiguousarray(bits, dtype=np.uint32)         # (V, ceil(d / 32))
        self.norms = np.asarray(norms, dtype=np.float32)                # (V,)
        self.dim, self.vocab = int(dim), len(self.norms)

    @classmethod
    def build(cls, rows: np.ndarray) -> "HammingIndex":
        w = np.asarray(rows, dtype=np.float32)
        return cls(pack(w), np.linalg.norm(w, axis=1), w.shape[1])

    def _each(self, h: np.ndarray):
        for q in pack(np.asarray(h).reshape(-1, self.dim)):
            dist = popcount(self.bits ^ q).sum(1, dtype=np.int32)
            yield self.norms * (self.dim - 2 * dist).astype(np.float32)

    def scores(self, h: np.ndarray) -> np.ndarray:
        """(n, V): each row's norm * (d - 2 * the Hamming distance between its signs and the state's)."""
        return np.stack(list(self._each(h))) if len(np.asarray(h)) else np.zeros((0, self.vocab), dtype=np.float32)

    def top(self, h: np.ndarray, n: int) -> list:
        """[token ids (sorted)] per state: the N best-scored rows; ties at the Nth place go to the lower id."""
        n = max(0, min(int(n), self.vocab))
        out = []
        for s in self._each(h):
            if n in (0, self.vocab):
                out.append(np.arange(n, dtype=np.int64))
                continue
            cut = np.partition(s, self.vocab - n)[self.vocab - n]       # the Nth best score
            above = np.flatnonzero(s > cut)
            out.append(np.sort(np.concatenate([above, np.flatnonzero(s == cut)[:n - len(above)]])).astype(np.int64))
        return out


__all__ = ["HammingIndex", "pack", "popcount"]
