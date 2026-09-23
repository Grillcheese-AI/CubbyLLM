"""cubbyllm.ops — the ONLY package permitted to import grilly.

Wired: WIRED — the VSA algebra seam every generated/binding path depends on.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .graph import graph_stats, graph_step
from .quant import int8_available, int8_weight_only
from .vsa import BlockCodeVSA

__all__ = ["BlockCodeVSA", "graph_stats", "graph_step",
           "int8_available", "int8_weight_only"]
__wiring__ = Wiring.WIRED
