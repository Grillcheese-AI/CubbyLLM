"""Differentiable (torch) VSA bind/unbind + the auxiliary binding loss.

Wired: WIRED — used by TrainLoop when the binding loss is on.

Uses the **MAP scheme from `cubby-concepts/vsa`** (the tested reference network):
bind is the bipolar **element-wise product**, which is its own inverse
(``bind(a, bind(a, b)) == b``) — the bipolar XOR analogue. That's already
differentiable, so the torch version is a Hadamard product; no FFT/block codes
needed. With fixed **bipolar** role keys, single-pair recovery is *exact*
(``role * role == 1``), so the loss is a clean capacity signal.

``binding_aux_loss`` BUNDLES N hidden states, each bound to a fixed bipolar role
key, into one hypervector, then unbinds each. Recovery is noisy from bundling
crosstalk, so minimizing the reconstruction error pushes the trunk's h toward the
quasi-orthogonal, high-capacity regime the VSA network needs (H-B5) — this is what
makes the trunk "know about" binding, rather than training the parameter-free algebra.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core.protocols import Wiring


def bind(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """MAP association: element-wise product (cubby-concepts ``vsa.bind``)."""
    return a * b


def unbind(composite: "torch.Tensor", key: "torch.Tensor") -> "torch.Tensor":
    """Recover a bound factor: bind again with the key (self-inverse for bipolar keys)."""
    return composite * key


def make_roles(n: int, d: int, device, seed: int = 0) -> "torch.Tensor":
    """N fixed **bipolar** {-1,+1} role keys (VSA keys are arbitrary; NOT learned —
    the trunk's h is what adapts). Bipolar => exact single-pair recovery."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(0, 2, (n, d), generator=g).float() * 2.0 - 1.0).to(device)


def binding_aux_loss(h: "torch.Tensor", roles: "torch.Tensor") -> "torch.Tensor":
    """Bundle N (role*filler) pairs, unbind each, cosine-reconstruct.

    h: (B, S, d) trunk hidden states (fp32). roles: (N, d) bipolar keys.
    Returns a scalar in [0, 2] (0 = perfect recovery of every bundled filler).
    """
    B, S, d = h.shape
    n = roles.shape[0]
    idx = torch.randint(0, S, (B, n), device=h.device)               # sample N positions
    fillers = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, d))  # (B, N, d)
    r = roles.unsqueeze(0)                                            # (1, N, d)
    composite = (r * fillers).sum(dim=1, keepdim=True)               # (B, 1, d) bundle
    recovered = composite * r                                        # (B, N, d) unbind
    cos = (F.normalize(recovered, dim=-1) * F.normalize(fillers, dim=-1)).sum(-1)
    return (1.0 - cos).mean()


__wiring__ = Wiring.WIRED
