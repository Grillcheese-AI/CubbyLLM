"""Context: the ``c`` in theta=f(c), and the source that infers it.

Wired: STANDALONE — a type + a seam + a concrete router.

H0b (the learned-context experiment) showed that context *inference* is the
bottleneck, not generation, and that an online-learned router catastrophically
forgets its own routing (its slots collapsed 6->3, task-0 confidence -> 37%).
So the concrete ``FrozenSlotRouter`` is designed to be pretrained OFFLINE over
all contexts and then FROZEN at inference (the H-C4 ``domain_head.pt`` pattern),
never learned online alongside the head. ``is_frozen`` enforces the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class Context:
    """The inferred context vector ``c`` plus provenance.

    vector      — (B, ctx_dim) the c that parameter generation conditions on.
    source_id   — which ContextSource produced it (for P5 probing / audit).
    confidence  — optional (B,) routing confidence, when the source reports it.
    """

    vector: "Tensor"
    source_id: str
    confidence: "Tensor | None" = None


@runtime_checkable
class ContextSource(Protocol):
    """Infers a ``Context`` from model input.

    Intended to be an offline-pretrained, frozen-at-inference classifier
    (see module docstring). ``is_frozen`` must be True during eval.
    """

    def infer(self, x: "Tensor") -> Context:
        ...

    @property
    def is_frozen(self) -> bool:
        ...


class FrozenSlotRouter:
    """Offline-pretrained soft router: input -> slot distribution -> context vector.

    A small MLP over an input feature maps to ``n_slots`` logits; a softmax over
    those weights a learned slot-embedding table into the context vector ``c``.
    Train this OFFLINE with slot/domain supervision (H-C4), then call ``freeze()``
    so it stops learning — H0b showed an online router forgets its own routing.
    """

    def __init__(self, input_dim: int, n_slots: int, ctx_dim: int,
                 source_id: str = "frozen_slot_router") -> None:
        import torch.nn as nn

        self.source_id = source_id
        self.router = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, n_slots)
        )
        self.slots = nn.Embedding(n_slots, ctx_dim)
        self._frozen = False

    def parameters(self):
        import itertools

        return itertools.chain(self.router.parameters(), self.slots.parameters())

    def route_logits(self, x: "Tensor") -> "Tensor":
        return self.router(x)

    def freeze(self) -> "FrozenSlotRouter":
        """Stop learning: detach parameters and mark frozen (call after offline
        pretraining, before inference/sequential use)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        return self

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def infer(self, x: "Tensor") -> Context:
        import torch.nn.functional as F

        p = F.softmax(self.router(x), dim=-1)      # (B, n_slots)
        c = p @ self.slots.weight                   # (B, ctx_dim)
        conf = p.max(dim=-1).values                 # (B,)
        return Context(vector=c, source_id=self.source_id, confidence=conf)


__wiring__ = Wiring.STANDALONE
