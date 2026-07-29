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
