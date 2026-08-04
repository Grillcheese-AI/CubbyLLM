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

    def read(self, q, k_set, v_set):
        """q (B,S,dk); k_set (B,S,K,dk); v_set (B,S,K,d) -> r (B,S,d)."""
        sim = (q.unsqueeze(2) * k_set).sum(-1) * self.scale     # (B,S,K)
        w = torch.softmax(sim, dim=-1)
        w = torch.nan_to_num(w)                                 # all-(-inf) row -> 0
        r = (w.unsqueeze(-1) * v_set).sum(2)                    # (B,S,d)
        return self.o(r)

    def forward(self, h, window):
        B, S, d = h.shape
        q, k, v = self.qkv(h)
        sim = torch.einsum("bik,bjk->bij", q, k) * self.scale   # (B,S,S)
        i = torch.arange(S, device=h.device)
        allowed = (i[:, None] - i[None, :]) > window            # j < i - window
        sim = sim.masked_fill(~allowed[None], float("-inf"))
        kk = min(self.topk, S)
        topv, topi = sim.topk(kk, dim=-1)                       # (B,S,kk)
        # gather the selected keys/values per query
        dk = k.shape[-1]
        k_set = torch.gather(k.unsqueeze(1).expand(B, S, S, dk), 2,
                             topi.unsqueeze(-1).expand(B, S, kk, dk))
        v_set = torch.gather(v.unsqueeze(1).expand(B, S, S, d), 2,
                             topi.unsqueeze(-1).expand(B, S, kk, d))
        # rows whose every candidate was masked (-inf): read zero
        no_mem = torch.isinf(topv).all(-1, keepdim=True)        # (B,S,1)
        sim_set = torch.where(torch.isinf(topv), torch.full_like(topv, -1e9), topv)
        w = torch.nan_to_num(torch.softmax(sim_set, dim=-1))
        r = self.o((w.unsqueeze(-1) * v_set).sum(2))
        return torch.where(no_mem, torch.zeros_like(r), r)


__wiring__ = Wiring.STANDALONE
