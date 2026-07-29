"""MinGRUBackbone — the chosen default backbone (H-D1 RESOLVED 2026-07-23).

Wired: WIRED — the trunk in the default forward path.

Resolved by the in-architecture bake-off (``validation/exp_d1b_backbone_bakeoff``):
in the real CubbyModel on subword TinyStories, gated recurrence beat the
attention-free Q=K recurrence and softmax attention by ~10% bpc; MinGRU matched
GRU's quality (1.308 vs 1.300 bpc — within single-run noise) at 25% fewer
backbone params, plus a log-domain PARALLEL SCAN (throughput at scale, where
GRU's kernel is sequential) and continuity with cubby-lm. User decision: MinGRU.

Trunk: N layers of [RMSNorm -> MinGRU recurrence -> +res, RMSNorm -> SwiGLU ->
+res]. The recurrence is an elementwise gate + Heinsen-2023 log-domain parallel
scan (cubby-lm form). Satisfies ``Backbone``: forward((B,S,d)) -> (B,S,d).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.protocols import Wiring

if TYPE_CHECKING:
    pass


class _RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class _SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int = 2):
        super().__init__()
        h = d * mult
        self.g = nn.Linear(d, h, bias=False)
        self.u = nn.Linear(d, h, bias=False)
        self.o = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.o(F.silu(self.g(x)) * self.u(x))


def _sequential_scan(x_scan: "torch.Tensor", a: "torch.Tensor") -> "torch.Tensor":
    """Exact recurrence h_t = a_t*h_{t-1} + x_t as an O(S) loop of mul/add only.

    Used on DirectML, whose op set does not cover ``logcumsumexp`` (the parallel
    scan below). Same result, GPU-resident; slower per step but correct."""
    B, S, d = x_scan.shape
    h = torch.zeros(B, d, device=x_scan.device, dtype=x_scan.dtype)
    out = []
    for t in range(S):
        h = a[:, t] * h + x_scan[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def _log_domain_scan(x_scan: "torch.Tensor", a: "torch.Tensor") -> "torch.Tensor":
    """Causal linear-RNN scan h_t = a_t*h_{t-1} + x_t via Heinsen 2023 log-domain
    parallel scan (O(log T) depth, sign-split for negative x).

    DirectML (device type 'privateuseone') lacks ``logcumsumexp``, so the exact
    sequential recurrence is used there instead — the only device-specific branch
    in the backbone, and it changes nothing on CPU/CUDA."""
    if x_scan.device.type == "privateuseone":
        return _sequential_scan(x_scan, a)
    log_a = a.clamp(min=1e-8).log()
    a_star = log_a.cumsum(dim=1)
    eps = 1e-30
    hp = (a_star + ((x_scan.clamp(min=0) + eps).log() - a_star).logcumsumexp(dim=1)).exp()
    hn = (a_star + (((-x_scan).clamp(min=0) + eps).log() - a_star).logcumsumexp(dim=1)).exp()
    return hp - hn


class _MinGRUMixer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj_g = nn.Linear(d, d)
        self.proj_v = nn.Linear(d, d)
        self.proj_d = nn.Linear(d, d)
        nn.init.constant_(self.proj_d.bias, 1.0)   # ~0.73 retention at init

    def forward(self, x):
        x_scan = torch.sigmoid(self.proj_g(x)) * torch.tanh(self.proj_v(x))
        a = 0.001 + 0.998 * torch.sigmoid(self.proj_d(x))
        return _log_domain_scan(x_scan, a)


class MinGRUBackbone(nn.Module):
    """The default CubbyLLM backbone: a MinGRU + SwiGLU trunk.

    forward: (B, S, d_model) embeddings -> (B, S, d_model) hidden states.
    Conforms to ``cubbyllm.model.backbone.Backbone``.
    """

    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        self.d_model = int(d_model)
        self.n1 = nn.ModuleList([_RMSNorm(d_model) for _ in range(n_layers)])
        self.mix = nn.ModuleList([_MinGRUMixer(d_model) for _ in range(n_layers)])
        self.n2 = nn.ModuleList([_RMSNorm(d_model) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([_SwiGLU(d_model) for _ in range(n_layers)])

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        for n1, m, n2, f in zip(self.n1, self.mix, self.n2, self.ffn):
            x = x + m(n1(x))
            x = x + f(n2(x))
        return x


__wiring__ = Wiring.WIRED
