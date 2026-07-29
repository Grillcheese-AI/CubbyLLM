"""cubbyllm.model.memory — the hardened theta=f(c) memory layer.

Wired: WIRED — package marker.
"""
from __future__ import annotations

from ...core.protocols import Wiring
from .base import MemoryLayer

__all__ = ["MemoryLayer"]
__wiring__ = Wiring.WIRED
