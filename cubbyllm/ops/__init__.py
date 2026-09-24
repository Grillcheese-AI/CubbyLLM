"""cubbyllm.ops — the ONLY package permitted to import grilly.

Wired: WIRED — the VSA algebra seam every generated/binding path depends on.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .es import evolution_strategy, greedy, load_emitter
from .graph import graph_stats, graph_step
from .quant import int8_available, int8_weight_only
from .vsa import BlockCodeVSA

__all__ = ["BlockCodeVSA", "evolution_strategy", "graph_stats", "graph_step",
           "greedy", "int8_available", "int8_weight_only", "load_emitter"]
__wiring__ = Wiring.WIRED
