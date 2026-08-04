"""Backbone protocol — interface only (implementations live elsewhere).

Wired: WIRED — the trunk sits in the default forward path.

The chosen default is ``HybridBackbone`` (``hybrid.py``) — MinGRU recurrence with a
sliding-window attention layer every 3rd, which won the H-D4 needle A/B
(2026-08-04). ``MinGRUBackbone`` (``mingru.py``) is the pure-recurrence baseline it
beat, kept selectable. **This file stays protocol-only** so both remain pluggable
and no single choice is hard-coded into the interface; a guard test enforces it.
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
