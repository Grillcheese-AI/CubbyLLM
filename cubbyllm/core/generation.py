"""Parameter generation (the hypernetwork) and the anti-drift Hardener.

Wired: STANDALONE — protocols + concrete torch implementations.

H0 (the central bet) validated at toy scale that theta=f(c) removes catastrophic
forgetting ONLY when hardened — a naive hypernetwork ends up *worse* than a
fixed-weight baseline because its own shared weights are themselves a C_ctx=0
learner. So ``Hardener`` is first-class here, never buried in a training loop.

The concrete ``HyperGenerator`` + ``SnapshotHardener`` are direct ports of the
validated ``validation/exp_h0_gce.py`` mechanism (which showed near-zero
forgetting hardened vs. worse-than-baseline naive).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor

    from .context import Context


@dataclass(frozen=True)
class GeneratedParams:
    """Opaque carrier for weights emitted by a ParameterGenerator.

    weights — the generated tensor (flat); a consumer reshapes it.
    meta    — optional bookkeeping (e.g. which block they target).
    """

    weights: "Tensor"
    meta: "dict | None" = None


@runtime_checkable
class ParameterGenerator(Protocol):
    """Maps a context to a block of active weights: theta = generate(c)."""

    def generate(self, ctx: "Context") -> GeneratedParams:
        ...


@runtime_checkable
class Hardener(Protocol):
    """The anti-drift countermeasure H0 proved essential.

    ``snapshot`` records a generator's output for prior contexts before a new
    task; ``penalty`` returns a regularization term keeping the generator's
    output for those contexts close to the snapshot during subsequent training.
    """

    def snapshot(self, gen: ParameterGenerator, contexts: "list[Context]") -> None:
        ...

    def penalty(self, gen: ParameterGenerator) -> "Tensor":
        ...


# ── Concrete implementations (torch; imported lazily by callers) ────────────

class HyperGenerator:
    """A small MLP hypernetwork mapping a context vector to a flat weight block.

    Port of exp_h0_gce.py's ``GCE.hyper``: ``ctx_dim -> hidden -> n_out`` with a
    small final-layer init so generated weights start near zero. ``n_out`` is the
    flattened size of the target block (e.g. DH*DOUT + DOUT for a linear head).
    """

    def __init__(self, ctx_dim: int, n_out: int, hidden: int = 64) -> None:
        import torch
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_out)
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def parameters(self):
        return self.net.parameters()

    def generate(self, ctx: "Context") -> GeneratedParams:
        theta = self.net(ctx.vector)
        return GeneratedParams(weights=theta, meta={"source_id": ctx.source_id})


class BasisHyperGenerator:
    """theta=f(c) as a mix over learned LOW-RANK BASES — the scalable form.

    Ported from cubemind's ``execution/mindforge.py`` (surveyed 2026-08-02: real,
    runs, full analytic backward, wired into its own ``VSALayer``). That design is
    adopted here because ``HyperGenerator``'s shape does not survive contact with
    a 2B model: emitting a flat ``d*d`` block means its final layer alone is
    ``hidden x d^2`` — measured at **272.6M parameters** for d=2048, hidden=64,
    i.e. **13.6% of the whole model spent on one generator**. This form is ~400x
    smaller and conditions on the layer as well as the context::

        h       = GELU(LayerNorm(W_proj @ c) ++ layer_emb[layer_id])
        coeffs  = W_coeff @ MLP(h)                       # (B, n_basis)
        A       = sum_i coeffs[i] * A_basis[i]           # (B, rank, d)
        B       = sum_i coeffs[i] * B_basis[i]           # (B, d, rank)
        W       = B @ A                                  # never materialised

    ``apply`` uses the factored path (``(x @ A^T) @ B^T``), so the d x d transform
    is never built. ``generate`` materialises it only for drop-in compatibility
    with the existing ``ParameterGenerator`` protocol and ``MemoryLayer``.

    ``B_basis`` is zero-initialised: standard LoRA init, so the generated delta is
    exactly identity at step 0. Verified 2026-08-02 that this does NOT block
    training — the delta is linear in B so ``B_basis`` receives gradient
    immediately, and ``A_basis`` (zero-grad only at step 0) unblocks as soon as B
    moves. A cubemind comment claiming zero-init "kills gradients" was checked and
    is wrong for this parameterisation.
    """

    def __init__(self, ctx_dim: int, d_model: int, n_layers: int = 1,
                 n_basis: int = 16, rank: int = 8, hidden: int = 256) -> None:
        import math

        import torch
        import torch.nn as nn

        self.d_model, self.rank, self.n_basis = int(d_model), int(rank), int(n_basis)
        self.n_layers = int(n_layers)
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        self.ctx_norm = nn.LayerNorm(hidden)
        self.layer_emb = nn.Parameter(torch.randn(self.n_layers, hidden) * 0.02)
        self.mix = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                 nn.Linear(hidden, self.n_basis))
        std = math.sqrt(2.0 / (d_model + rank))
        self.A_basis = nn.Parameter(torch.randn(self.n_basis, rank, d_model) * std)
        # zero-init => identity adapter at step 0 (standard LoRA; see docstring)
        self.B_basis = nn.Parameter(torch.zeros(self.n_basis, d_model, rank))

    def parameters(self):
        yield from self.ctx_proj.parameters()
        yield from self.ctx_norm.parameters()
        yield from self.mix.parameters()
        yield self.layer_emb
        yield self.A_basis
        yield self.B_basis

    def _coeffs(self, ctx: "Context", layer_id: int):
        import torch
        import torch.nn.functional as F

        # ctx.vector is (B, ctx_dim) at decode or (B, S, ctx_dim) in a causal
        # training forward; every op below works on any leading dims.
        h = F.gelu(self.ctx_norm(self.ctx_proj(ctx.vector)))      # (..., hidden)
        emb = self.layer_emb[int(layer_id) % self.n_layers]        # (hidden,)
        h = torch.cat([h, emb.expand(*h.shape[:-1], -1)], dim=-1)  # (..., 2*hidden)
        return self.mix(h)                                         # (..., n_basis)

    def factors(self, ctx: "Context", layer_id: int = 0):
        """The (A, B) low-rank factors — the form to actually compute with."""
        import torch

        c = self._coeffs(ctx, layer_id)
        A = torch.einsum("...n,nrd->...rd", c, self.A_basis)       # (..., rank, d)
        B = torch.einsum("...n,ndr->...dr", c, self.B_basis)       # (..., d, rank)
        return A, B

    def apply(self, x: "Tensor", ctx: "Context", layer_id: int = 0) -> "Tensor":
        """x @ W(c)^T with W = B(c) @ A(c), one transform PER CONTEXT ROW.

        ``ctx.vector`` is (B, ctx_dim) — one context per sample, the decode
        shape — or (B, S, ctx_dim) — one per position, the causal training
        shape (``CubbyModel.infer_context``). Its leading dims broadcast
        against ``x``'s, so a (1, ctx_dim) context is shared by the batch.

        Neither the d x d matrix nor the per-row factors are built. A(c) and
        B(c) are both linear in the coefficients, so

            x @ A(c)^T  = sum_n c_n (x @ A_n^T)
            xa @ B(c)^T = sum_n c_n (xa @ B_n^T)

        and both contractions run against the stacked bases — two matmuls of
        (..., d) against (d, n_basis*rank) — with the coefficients applied on
        the small (..., n_basis, rank) side. Per-position factors would cost
        B*S*rank*d floats each; this costs B*S*n_basis*rank.

        Until 2026-09-23 this averaged A and B over the batch ("shared
        context"): in training every sample was transformed by the batch's mean
        adapter, while decode (batch 1) used its own. Per row is what decode
        always did; training now matches it.
        """
        import torch

        c = self._coeffs(ctx, layer_id)                            # (..., n)
        while c.dim() < x.dim():                                   # (B, n) vs (B, S, d)
            c = c.unsqueeze(-2)
        xa = torch.einsum("...d,nrd->...nr", x, self.A_basis)      # (..., n, r)
        xa = (xa * c.unsqueeze(-1)).sum(dim=-2)                    # (..., r) = x A(c)^T
        y = c.unsqueeze(-1) * xa.unsqueeze(-2)                     # (..., n, r)
        return torch.einsum("...nr,ndr->...d", y, self.B_basis)    # (..., d)

    def generate(self, ctx: "Context", layer_id: int = 0) -> GeneratedParams:
        """Materialise the flat d*d block. Drop-in for ``ParameterGenerator``;
        prefer ``apply``/``factors``, which skip the d x d entirely."""
        import torch

        A, B = self.factors(ctx, layer_id)
        W = torch.einsum("...dr,...rk->...dk", B, A)               # (..., d, d)
        return GeneratedParams(weights=W.reshape(*W.shape[:-2], -1),
                               meta={"source_id": ctx.source_id, "layer_id": layer_id})


@dataclass
class SnapshotHardener:
    """von-Oswald-style output regularization (exp_h0_gce.py's ``harden`` path).

    Before training on a new context, ``snapshot`` records the generator's output
    for every previously-seen context. ``penalty`` then returns the mean squared
    drift of the generator's current output from those snapshots — added to the
    loss so prior contexts' generated weights stay put.
    """

    beta: float = 50.0
    _snaps: dict = field(default_factory=dict)
    _contexts: list = field(default_factory=list)

    def snapshot(self, gen: ParameterGenerator, contexts: "list[Context]") -> None:
        import torch

        self._contexts = list(contexts)
        with torch.no_grad():
            self._snaps = {
                c.source_id: gen.generate(c).weights.clone() for c in contexts
            }

    def penalty(self, gen: ParameterGenerator) -> "Tensor":
        import torch

        if not self._snaps:
            return torch.zeros(())
        terms = [
            ((gen.generate(c).weights - self._snaps[c.source_id]) ** 2).mean()
            for c in self._contexts
        ]
        return self.beta * (sum(terms) / len(terms))


__wiring__ = Wiring.STANDALONE
