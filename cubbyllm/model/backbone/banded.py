"""Sliding-window causal attention in bands: the reference path's exact twin at half the work.

Wired: by ``HybridBackbone``'s windowed attention with ``CB_BANDED=1`` when flex attention is not used
(``CB_NO_FLEX=1``, or a GPU whose shared memory per block is under flex's tile -- the RTX PRO 6000
Blackwell's 99 KB) and the sequence is longer than two windows. Opt-in until a GPU measures it: the
scores are halved at S = 2048, but the blocks' key/value copies are extra traffic (on the CPU the two
paths time the same, 2026-10-20) -- ``CB_MODE=bench`` with and without it decides.

The dense reference builds an S x S boolean mask (causal, at most ``window - 1`` back) and lets
``scaled_dot_product_attention`` compute every score and discard all but a band of width ``window``:
at S = 2048, window 512, three quarters of the work is thrown away. Here the queries go in blocks of
``window``; a query's keys can reach at most ``window - 1`` back, so they all lie in its own block or
the one before -- each block attends over those ``2 * window`` keys under one fixed (window, 2 window)
mask (the first block's missing predecessor masked out). The same keys, the same softmax, the same
output (``tests/model/backbone/test_banded.py`` holds it to the dense path); S / (2 * window) times
the reference's score work -- half at S = 2048, a quarter at S = 4096.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core.protocols import Wiring

__all__ = ["banded_attention", "dense_attention"]


def dense_attention(q, k, v, window: int):
    """The reference: (B, h, S, dh) each -> (B, h, S, dh) under the S x S causal-window mask."""
    S = q.shape[-2]
    i = torch.arange(S, device=q.device)
    keep = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=keep)


def banded_attention(q, k, v, window: int):
    """The same attention in blocks of ``window`` queries over ``2 * window`` keys each."""
    B, h, S, dh = q.shape
    W = int(window)
    n = -(-S // W)
    pad = n * W - S
    if pad:                                   # padded keys sit after every real query: causally masked
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    qb, kb, vb = (t.reshape(B, h, n, W, dh) for t in (q, k, v))
    prev = lambda t: torch.cat([torch.zeros_like(t[:, :, :1]), t[:, :, :-1]], 2)
    k2 = torch.cat([prev(kb), kb], 3)         # (B, h, n, 2W, dh): the block before, then its own
    v2 = torch.cat([prev(vb), vb], 3)
    i = torch.arange(W, device=q.device)[:, None]
    j = torch.arange(2 * W, device=q.device)[None, :]
    back = W + i - j                          # query position minus key position
    keep = (back >= 0) & (back < W)           # (W, 2W)
    mask = keep.expand(n, W, 2 * W).clone()
    mask[0] &= j >= W                         # block 0 has no predecessor
    o = F.scaled_dot_product_attention(qb.reshape(B, h * n, W, dh), k2.reshape(B, h * n, 2 * W, dh),
                                       v2.reshape(B, h * n, 2 * W, dh), attn_mask=mask.repeat(h, 1, 1))
    return o.reshape(B, h, n * W, dh)[:, :, :S]


__wiring__ = Wiring.STANDALONE
