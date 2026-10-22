"""SpatialKey -- the spatial link's text-keyed addresses, and the pointer that asks them (order 0006,
H-K10: free-text questions).

Why. A loci token is answered when its question carries the address. A free-text question carries
none, and today a fact's address is a hash of the subject's name (``make_fact_shard.cell``), which
the question's words cannot recompute: the learned route can only memorize it for a closed set (run
3: villages 0.748 by the route alone, far facts 0.014 over 160k subjects). So the address becomes a
function of the words, and the writer and the asker compute the same one:

    key(s, r) = u(e(s)) (*) u(e(r)),   u(x) = unitary(W x),   piece i of the object at key (*) P_i

with ``e`` the trunk's token embedding (read without gradient), ``W`` a fixed random orthogonal
projection and ``P_i`` fixed random unitaries (buffers from ``seed``; ``unitary`` is scale-invariant,
so no scale of W changes a key: different tokens' keys sit at the random-phase floor, |cos| ~
1/sqrt(d)). A write
carries ``(s, r, i)`` as side data in the slots a place's coordinates use (``positions`` (..., 3)):
the host writes the facts, so it knows them. Everything is kept as unit-modulus spectra, where
binding is a product.

**The pointer** (``query="pointer"``). At an asking token, two soft pointers -- subject, relation --
over the previous ``w`` tokens, from the stream at the read's layer: a query from the asker, a key
from each earlier token (the stream there knows its context: "the token after *of*"), a learned bias
per offset. The key is computed exactly from what they point at,
``u(sum_k a_k e_{t-k}) (*) u(sum_k b_k e_{t-k}) (*) P_j``, ``j`` the piece the asker asks for (its
``positions[..., 2]``: the answer tokens so far, which the host counts); a one-hot pointer gives the
write's key. Fast-weight memories key on (subject, relation) the same way (Schlag et al. 2021); the
pointer is a pointer network's (Vinyals et al. 2015). Decode keeps a ring of the last ``w`` ids and
their pointer keys -- fixed-size, like the attention window's.

Wired: STANDALONE until a base run trains with ``meta["spatial"]["key"] = "text"`` (``SpatialLink``
owns a ``TextKeys`` and, with ``query="pointer"``, a ``Pointer``; ``CubbyModel`` reaches them through
the link's ``bind_input`` / ``hook`` / ``add_copy``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..core.protocols import Wiring

__all__ = ["TextKeys", "Pointer", "unit_spectrum", "STATE"]

#: the pointer's decode state: the last ``w`` ids, their keys, and how many of them are this document's
STATE = ("ptr_ids", "ptr_keys", "ptr_n")


def unit_spectrum(x: torch.Tensor) -> torch.Tensor:
    """``rfft(x)`` brought to unit modulus: the spectrum of ``SpatialLink.unitary(x)``."""
    s = torch.fft.rfft(x.float(), dim=-1)
    return s / (s.abs() + 1e-6)


class TextKeys(nn.Module):
    """``key(s, r) (*) P_i`` as a unit spectrum (..., d/2+1), from the embedding table."""

    def __init__(self, d_model: int, pieces: int = 16, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(int(seed) + 7121)
        self.d, self.pieces = int(d_model), int(pieces)
        # orthogonal, not Gaussian: a Gaussian W's W W^T is anisotropic and lifts the keys' crosstalk above
        # the random floor (mean |cos| 0.064 against 0.050 at d = 256; tests/model/test_spatial_key.py)
        self.register_buffer("W", torch.linalg.qr(torch.randn(self.d, self.d, generator=g))[0])
        self.register_buffer("P", torch.angle(unit_spectrum(torch.randn(self.pieces, self.d, generator=g))))

    def u(self, e: torch.Tensor) -> torch.Tensor:
        """``u(e)``'s spectrum for embeddings ``e`` (..., d). The callers read the table detached: no gradient
        reaches it, while a pointer's mixture of rows keeps its own."""
        return unit_spectrum(e.float() @ self.W.T)

    def piece(self, i: torch.Tensor) -> torch.Tensor:
        p = self.P[i.long().clamp(0, self.pieces - 1)]
        return torch.polar(torch.ones_like(p), p)

    def address(self, table: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """``p`` (..., 3) = (subject id, relation id, piece) -> the address's spectrum (..., d/2+1)."""
        s, r, i = p.long().unbind(-1)
        t = table.detach()
        return self.u(t[s]) * self.u(t[r]) * self.piece(i)

    def extra_repr(self) -> str:
        return f"d={self.d}, pieces={self.pieces}"


class Pointer(nn.Module):
    """The asker's two soft pointers (subject, relation) over the previous ``w`` tokens, and the key
    they make. ``n`` is the link's normed stream at the read's layer."""

    def __init__(self, d_model: int, w: int = 16, dk: int = 64) -> None:
        super().__init__()
        self.w, self.dk = int(w), int(dk)
        self.q = nn.Linear(d_model, 2 * self.dk, bias=False)
        self.k = nn.Linear(d_model, 2 * self.dk, bias=False)
        self.bias = nn.Parameter(torch.zeros(2, self.w))                 # per offset 1..w, per pointer

    def logits(self, n_ask: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``n_ask`` (n, d), ``keys`` (n, w, 2, dk) at offsets 1..w, ``valid`` (n, w) -> (n, 2, w), -inf where
        nothing is there (before the document's start)."""
        q = self.q(n_ask).view(-1, 2, self.dk)
        s = torch.einsum("nhk,nwhk->nhw", q, keys) / math.sqrt(self.dk) + self.bias
        return s.masked_fill(~valid[:, None], float("-inf"))

    def key(self, keys_of: TextKeys, a: torch.Tensor, e_win: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """The asked address, (n, d): ``a`` (n, 2, w) the pointers, ``e_win`` (n, w, d) the pointed-at
        tokens' embeddings, ``j`` (n,) the piece asked."""
        subj, rel = torch.einsum("nhw,nwd->nhd", a, e_win.detach().float()).unbind(1)
        spec = keys_of.u(subj) * keys_of.u(rel) * keys_of.piece(j)
        return torch.fft.irfft(spec, n=e_win.shape[-1], dim=-1)

    def query(self, keys_of: TextKeys, n: torch.Tensor, ids: torch.Tensor, table: torch.Tensor,
              positions: torch.Tensor, asks: torch.Tensor, starts=None, state: dict | None = None):
        """``(q, mask)`` for ``SpatialLink.read``: the pointer's key at every asking token that has
        something before it in its document, (B, S, d) / (B, S) in a forward, (B, d) / (B,) at a decode
        step (``state`` given: its ring is read, then this token joins it). Also ``self.last`` =
        ``(flat asking index, pointer logits)`` for a pointer loss."""
        if state is not None:
            return self._step(keys_of, n, ids, table, positions, asks, starts, state)
        B, S, d = n.shape
        K = self.k(n).view(B, S, 2, self.dk)
        t = torch.arange(S, device=n.device)
        seg = torch.zeros(B, S, dtype=torch.long, device=n.device)
        if starts is not None and bool(starts.any()):
            seg = torch.where(starts, t.expand(B, S), torch.zeros_like(seg)).cummax(1).values
        idx = asks.reshape(-1).nonzero().squeeze(1)
        b, tt = idx // S, idx % S
        src = tt[:, None] - torch.arange(1, self.w + 1, device=n.device)          # (n, w): offsets 1..w
        valid = src >= seg[b, tt][:, None]
        keep = valid.any(1)
        idx, b, src, valid = idx[keep], b[keep], src[keep], valid[keep]
        src = src.clamp_min(0)
        lg = self.logits(n.reshape(B * S, d)[idx], K[b[:, None], src], valid)
        self.last = (idx, lg)
        q = n.new_zeros(B * S, d, dtype=torch.float32)
        q = q.index_copy(0, idx, self.key(keys_of, lg.softmax(-1), table[ids[b[:, None], src]], positions.reshape(-1, 3)[idx, 2]))
        mask = torch.zeros(B * S, dtype=torch.bool, device=n.device).index_fill(0, idx, True)
        return q.reshape(B, S, d), mask.reshape(B, S)

    def _step(self, keys_of, n, ids, table, positions, asks, starts, state):
        ring, rk, cnt = state["ptr_ids"], state["ptr_keys"], state["ptr_n"]
        if starts is not None:                                               # a document begins: nothing before it
            cnt.mul_((~starts).long())
        offs = torch.arange(1, self.w + 1, device=n.device)
        valid = offs[None] <= cnt[:, None]                                   # (B, w): offset k is ring slot w - k
        slot = self.w - offs
        q = n.new_zeros(n.shape, dtype=torch.float32)
        mask = asks & valid.any(1)
        if bool(mask.any()):
            lg = self.logits(n[mask], rk[mask][:, slot], valid[mask])
            self.last = (mask.nonzero().squeeze(1), lg)
            q[mask] = self.key(keys_of, lg.softmax(-1), table[ring[mask][:, slot]], positions[mask][:, 2])
        ring.copy_(torch.cat([ring[:, 1:], ids[:, None]], 1))               # this token joins the ring
        rk.copy_(torch.cat([rk[:, 1:], self.k(n).view(-1, 1, 2, self.dk).to(rk.dtype)], 1))
        cnt.copy_((cnt + 1).clamp_max(self.w))
        return q, mask

    def init_state(self, batch: int, device=None) -> dict:
        return {"ptr_ids": torch.zeros(batch, self.w, dtype=torch.long, device=device),
                "ptr_keys": torch.zeros(batch, self.w, 2, self.dk, device=device),
                "ptr_n": torch.zeros(batch, dtype=torch.long, device=device)}

    def extra_repr(self) -> str:
        return f"w={self.w}, dk={self.dk}"


__wiring__ = Wiring.STANDALONE
