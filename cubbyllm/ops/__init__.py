"""cubbyllm.ops — the ONLY package permitted to import grilly.

Wired: WIRED — the VSA algebra seam every generated/binding path depends on.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .es import (evolution_strategy, greedy, greedy_many, load_emitter, optimizer,
                 population_greedy, population_greedy_many, update)
from .graph import graph_stats, graph_step
from .quant import int8_available, int8_weight_only
from .vsa import BlockCodeVSA

__all__ = ["BlockCodeVSA", "evolution_strategy", "graph_stats", "graph_step",
           "greedy", "greedy_many", "int8_available", "int8_weight_only", "load_emitter",
           "optimizer", "population_greedy", "population_greedy_many", "update"]
__wiring__ = Wiring.WIRED
