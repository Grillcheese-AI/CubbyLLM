"""cubbyllm.bridges — first-class cross-repo interfaces.

Wired: STANDALONE — package marker for the bridge layer.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .possibility import Possibility, PossibilityOracle, Verdict
from .world_model import WorldModelBridge

__all__ = ["WorldModelBridge", "PossibilityOracle", "Possibility", "Verdict"]
__wiring__ = Wiring.STANDALONE
