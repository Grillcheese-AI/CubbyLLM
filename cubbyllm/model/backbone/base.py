"""Backbone protocol — interface only (implementations live elsewhere).

Wired: WIRED — the trunk sits in the default forward path.

The chosen default is ``MinGRUBackbone`` (``mingru.py``), resolved 2026-07-23 by
the in-architecture bake-off (``validation/exp_d1b_backbone_bakeoff``): gated
recurrence beat attention-free Q=K and softmax attention by ~10% bpc in the real
CubbyModel, and MinGRU matched GRU at fewer params + a parallel scan. **This file
stays protocol-only** so alternate backbones remain pluggable and no single
choice is hard-coded into the interface; a guard test enforces it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...core.protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor


@runtime_checkable
class Backbone(Protocol):
    """Maps a token sequence to per-position hidden states."""

    def forward(self, tokens: "Tensor") -> "Tensor":
        ...


__wiring__ = Wiring.WIRED
