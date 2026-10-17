"""SpatialLink -- the trunk's route to space: places bound into its input, and a gated read
of a scene's Hilbert-space vector from its state, so the trunk itself decides when to use it.

Wired: STANDALONE until a base run trains with ``CB_SPATIAL=1`` (2026-10-17, Nick: "spatial
understanding provably at the input level"; "it also needs to have a link route from the trunk
so it knows when to use it").

Two halves, one phase table (``cubbyllm.spatial.ssp._phases``, so the host's ``Scene`` and the
trunk agree on what ``S(p)`` is):

**bind** (the input level). A token that has a place ``p`` enters the trunk as
``e (*) S(p)`` -- its embedding circularly convolved with the place's unit vector. Provable,
and ``tests/model/test_spatial.py`` holds each to float precision: a token with no place is
unchanged (``S(0)`` is the identity; a batch with no places at all is returned as is, so
every text golden is bit-identical with the link on); moving a token is binding
(``bind(e, p + d) = bind(e, p) (*) S(d)``); and the place can be read back from the bound
embedding by unbinding with ``e``. RoPE is this same encoding in one dimension applied inside
attention; this is its 3-D sibling at the input, applied to every mixer the trunk has.

**read** (the route from the trunk). After the trunk, from its state ``h``: a gate
``g = sigmoid(w . rms(h) + b)`` says *whether* to consult the scene, a query ``q = W rms(h)``
made unitary (its spectrum reduced to its phases, so it *is* a place, ``S(p)`` for the ``p``
the trunk means) says *what to ask*, and the answer ``V (*) q^-1`` -- the scene vector
unbound by that place, the code of what is there -- comes back through ``o``. ``o`` starts at
zero and the gate's bias at -4, so a model with the link added computes what it computed
before (growth's rule: exact at step 0); with no scene the read is exactly zero. What the
gate and the query learn is the route; the lookup they route to is algebra.

What it brings CubbyLLM: a trunk that can be told where every block is without a word of
text, and ask the scene "what is here" at a place it chose, with the answer's correctness
guaranteed by the encoding rather than hoped for from training. A conventional model has no
place to put a coordinate but its tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.protocols import Wiring
from ..spatial.ssp import SSP, _phases

__all__ = ["SpatialLink"]


class SpatialLink(nn.Module):
    def __init__(self, d_model: int, scale: float = 2.0, seed: int = 0, dims: int = 3,
                 gate_bias: float = -4.0) -> None:
        super().__init__()
        self.d, self.dims, self.scale, self.seed = int(d_model), int(dims), float(scale), int(seed)
        self.register_buffer("phases", torch.tensor(_phases(self.d, self.dims, self.seed), dtype=torch.float32))
        self.norm_w = nn.Parameter(torch.ones(self.d))
        self.q = nn.Linear(self.d, self.d, bias=False)
        self.gate = nn.Linear(self.d, 1)
        self.o = nn.Linear(self.d, self.d, bias=False)
        with torch.no_grad():
            self.o.weight.zero_()
            self.gate.bias.fill_(float(gate_bias))

    def space(self) -> SSP:
        """The host's ``SSP`` in this link's space: same phases, so ``Scene`` vectors read here."""
        return SSP(self.d, self.dims, self.scale, self.seed)

    # -- the encoding, in torch ------------------------------------------------------------
    def spectrum(self, p: torch.Tensor) -> torch.Tensor:
        """``exp(i Theta p / scale)`` for places ``p`` (..., dims) -> (..., d/2+1) complex."""
        ang = p.float() @ self.phases.T / self.scale
        return torch.polar(torch.ones_like(ang), ang)

    def encode(self, p: torch.Tensor) -> torch.Tensor:
        """``S(p)``, (..., d) float32 unit vectors."""
        return torch.fft.irfft(self.spectrum(p), n=self.d, dim=-1)

    # -- bind: the input level ------------------------------------------------------------
    def bind(self, e: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        """``e (*) S(positions)``; ``positions`` (..., dims) matching ``e``'s leading shape,
        or None for a batch without places, returned untouched."""
        if positions is None:
            return e
        spec = torch.fft.rfft(e.float(), dim=-1) * self.spectrum(positions)
        return torch.fft.irfft(spec, n=self.d, dim=-1).to(e.dtype)

    def unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a (*) b^-1`` with ``b``'s exact inverse (its conjugate spectrum; exact for a
        unitary ``b`` such as ``S(p)``)."""
        spec = torch.fft.rfft(a.float(), dim=-1) * torch.conj(torch.fft.rfft(b.float(), dim=-1))
        return torch.fft.irfft(spec, n=self.d, dim=-1).to(a.dtype)

    # -- read: the route from the trunk ---------------------------------------------------
    @staticmethod
    def unitary(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """``q`` with every spectral component brought to unit modulus: a place, whatever
        the trunk's magnitudes were."""
        spec = torch.fft.rfft(q.float(), dim=-1)
        return torch.fft.irfft(spec / (spec.abs() + eps), n=q.shape[-1], dim=-1)

    def lookup(self, scene: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """The algebra alone: ``scene (*) unitary(q)^-1`` -- what the scene holds at the place
        ``q`` names, (..., d)."""
        return self.unbind(scene, self.unitary(q))

    def read(self, h: torch.Tensor, scene: torch.Tensor | None) -> torch.Tensor:
        """The gated read of ``scene`` (B, d), one vector per batch row, from the trunk state
        ``h`` (B, S, d) or (B, d) -> the same shape as ``h``; zero when ``scene`` is None."""
        if scene is None:
            return torch.zeros_like(h)
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm_w
        g = torch.sigmoid(self.gate(n))                                  # (..., 1): whether
        q = self.q(n)                                                    # (..., d): what place
        v = scene.reshape(scene.shape[0], *([1] * (h.dim() - 2)), self.d).expand_as(h)
        return g * self.o(self.lookup(v, q).to(h.dtype))

    def extra_repr(self) -> str:
        return f"d={self.d}, dims={self.dims}, scale={self.scale}, seed={self.seed}"


__wiring__ = Wiring.STANDALONE
