"""Model configuration and canonical seed.

Wired: STANDALONE — configuration, no model logic.

Vocab strategy is the validated HYBRID (H-C6): a fixed core (``vocab_core``,
next step 128k-256k BPE per H-G1) plus an optional dynamic tail
(``vocab_dynamic``). Block dims default to cubemind's world-arena shape
(k=80, l=128 -> D=10240), matching the chosen BlockCodeOps binding algebra.
Backbone shape is deliberately NOT here — that decision is still open (H-D1).
"""
from __future__ import annotations

from dataclasses import dataclass

from .protocols import Wiring

#: Canonical deterministic seed shared across the stack (matches the sibling
#: repos' 0xC0DEB00C convention for cross-repo codebook reproducibility).
SEED = 0xC0DEB00C


@dataclass(frozen=True)
class CubbyConfig:
    """Static architecture config for an assembled CubbyModel.

    d_model       — hidden width.
    n_layers      — backbone depth (backbone *choice* is separate, still open).
    ctx_dim       — dimensionality of the context vector c.
    vocab_core    — fixed core vocabulary size (hybrid decision, H-C6).
    vocab_dynamic — whether a dynamic/open tail is enabled (gated, later).
    block_k, block_l — BlockCodeOps block dims; k*l is the VSA dimension D.
    """

    d_model: int
    n_layers: int
    ctx_dim: int
    vocab_core: int
    vocab_dynamic: bool = False
    block_k: int = 80
    block_l: int = 128

    @property
    def d_vsa(self) -> int:
        """The flat VSA dimension D = k * l."""
        return self.block_k * self.block_l


__wiring__ = Wiring.STANDALONE
