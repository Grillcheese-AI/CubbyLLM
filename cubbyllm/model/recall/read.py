"""MemoryRead — the differentiable episodic read (soft attention over top-K).

Wired: STANDALONE — attaches into HybridBackbone once the toy gate (exp_d5) passes.

The read is a modern-Hopfield / kNN-attention lookup: a per-position query attends
over the top-K most similar candidate keys and pools their values. It is
per-position, per-sequence by construction (each query gets its own set) — the
structural fix for MEMORY_PROBE.md's DC-offset failure. The training path masks
candidates to the BEYOND-window causal past (j < i - window) so the memory learns
long-range recall instead of duplicating the window.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.protocols import Wiring


class MemoryRead(nn.Module):
    def __init__(self, d_model: int, d_key: int = 64, topk: int = 8):
        super().__init__()
        self.topk = int(topk)
        self.scale = float(d_key) ** -0.5
        self.q = nn.Linear(d_model, d_key, bias=False)
        self.k = nn.Linear(d_model, d_key, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def qkv(self, h):
        return self.q(h), self.k(h), self.v(h)

    def read(self, q, k_set, v_set, valid=None):
        """q (B,S,dk); k_set (B,S,K,dk); v_set (B,S,K,d); valid (B,S,K) bool or
        None -> r (B,S,d). Rows with no valid candidate read zero."""
        sim = (q.unsqueeze(2) * k_set).sum(-1) * self.scale      # (B,S,K)
        if valid is not None:
            sim = sim.masked_fill(~valid, -1e9)                  # finite -> backward-safe
        w = torch.softmax(sim, dim=-1)
        r = self.o(torch.einsum("bsk,bskd->bsd", w, v_set))     # a batched matvec, no (B,S,K,d) product
        if valid is not None:
            no_mem = ~valid.any(-1, keepdim=True)                # (B,S,1)
            r = torch.where(no_mem, torch.zeros_like(r), r)
        return r

    def forward(self, h, window):
        B, S, d = h.shape
        q, k, v = self.qkv(h)
        # Selection metric is COSINE, not scaled dot-product: it must match
        # EpisodicStore.retrieve_cosine (the decode-time selector) and the
        # Hamming index (Task 4), which both approximate cosine, not raw dot
        # products. Only the SELECTION uses normalized q/k; the gathered
        # k_set/v_set below are the ORIGINAL (un-normalized) projections, and
        # self.read()'s softmax weighting stays dot-product, unchanged.
        #
        # The selection carries no gradient (top-k indices), so it runs under no_grad: nothing of the
        # (B, S, S) similarity is saved for backward. The gather is FLAT -- an index_select of rows of
        # (B*S, d) -- because gathering from ``v.unsqueeze(1).expand(B, S, S, d)`` (the version before
        # 2026-10-20) is cheap forward but its backward scatters into a DENSE (B, S, S, d) gradient:
        # 17 GB per memory layer at B=2, S=2048, d=1024 in bf16 -- why the grown 1.3B fit micro 2 only and
        # ran at 7-9% MFU against the 450M's 34% on the same card. Same values, same gradients.
        dk = k.shape[-1]
        kk = min(self.topk, S)
        with torch.no_grad():
            qn = F.normalize(q, dim=-1)
            kn = F.normalize(k, dim=-1)
            sim = torch.einsum("bik,bjk->bij", qn, kn)
            i = torch.arange(S, device=h.device)
            allowed = (i[:, None] - i[None, :]) > window         # j < i - window
            sim = sim.masked_fill(~allowed[None], float("-inf"))
            topv, topi = sim.topk(kk, dim=-1)                    # (B,S,kk)
            valid = ~torch.isinf(topv)                           # real vs masked-pad slots
            flat = (topi + (torch.arange(B, device=h.device) * S).view(B, 1, 1)).reshape(-1)
        k_set = k.reshape(B * S, dk).index_select(0, flat).view(B, S, kk, dk)
        v_set = v.reshape(B * S, d).index_select(0, flat).view(B, S, kk, d)
        return self.read(q, k_set, v_set, valid)


__wiring__ = Wiring.STANDALONE
