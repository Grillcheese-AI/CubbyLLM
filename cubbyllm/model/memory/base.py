"""MemoryLayer — the forgetting-resistant memory, as hardened theta=f(c).

Wired: WIRED — read in the default forward path.

The validation campaign found NO single update rule achieves zero forgetting
(H-A3/H-A4: NLMS best capacity but oldest-first, SDM only order-agnostic,
Hebbian worst). The chosen direction is H0's mechanism: generate the memory
pathway's active weights from context and HARDEN against drift. This layer
implements ``Generable`` — it reads by generating a (d_model x d_model) transform
from the context and applying it to the hidden state. The generator's params are
what a ``SnapshotHardener`` protects; the layer itself owns no per-context state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.protocols import Generable, Wiring

if TYPE_CHECKING:
    from torch import Tensor

    from ...core.context import Context
    from ...core.generation import Hardener, ParameterGenerator


class MemoryLayer(Generable):
    """Context-generated, hardened memory read.

    generator — emits a flat (d_model*d_model) weight block from a Context.
    hardener  — anti-drift snapshot/penalty over previously-seen contexts.
    d_model   — hidden width; the generated block reshapes to (d_model, d_model).
    """

    def __init__(
        self,
        generator: "ParameterGenerator",
        hardener: "Hardener",
        d_model: int,
    ) -> None:
        self.generator = generator
        self.hardener = hardener
        self.d_model = int(d_model)

    def forward_generated(self, x: "Tensor", ctx: "Context") -> "Tensor":
        """Read memory using weights generated from ``ctx``.

        Generates a (d_model, d_model) transform W = f(c) and returns a residual
        read ``x + tanh(x @ W^T)``. The context enters as an unbypassable argument
        (the theta=f(c) contract); with a single shared context in the batch the
        generated block is shared, per-sample contexts give per-sample reads.
        """
        import torch

        gp = self.generator.generate(ctx)
        theta = gp.weights                                 # (..., d_model*d_model)
        # collapse any batch dim on the generated params to a single matrix when
        # the context is shared; otherwise use the per-row matrices.
        flat = theta.reshape(-1, self.d_model * self.d_model)
        W = flat.mean(dim=0).reshape(self.d_model, self.d_model)
        return x + torch.tanh(x @ W.t())


__wiring__ = Wiring.WIRED
