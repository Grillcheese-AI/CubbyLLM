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
        r = self.o((w.unsqueeze(-1) * v_set).sum(2))
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
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)
        sim = torch.einsum("bik,bjk->bij", qn, kn)
        i = torch.arange(S, device=h.device)
        allowed = (i[:, None] - i[None, :]) > window             # j < i - window
        sim = sim.masked_fill(~allowed[None], float("-inf"))
        kk = min(self.topk, S)
        topv, topi = sim.topk(kk, dim=-1)                        # (B,S,kk)
        dk = k.shape[-1]
        k_set = torch.gather(k.unsqueeze(1).expand(B, S, S, dk), 2,
                             topi.unsqueeze(-1).expand(B, S, kk, dk))
        v_set = torch.gather(v.unsqueeze(1).expand(B, S, S, d), 2,
                             topi.unsqueeze(-1).expand(B, S, kk, d))
        valid = ~torch.isinf(topv)                               # real vs masked-pad slots
        return self.read(q, k_set, v_set, valid)


__wiring__ = Wiring.STANDALONE
