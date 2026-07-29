"""cubbyllm.bridges — first-class cross-repo interfaces.

Wired: STANDALONE — package marker for the bridge layer.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .world_model import WorldModelBridge

__all__ = ["WorldModelBridge"]
__wiring__ = Wiring.STANDALONE
