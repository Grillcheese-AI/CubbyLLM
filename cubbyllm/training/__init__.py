"""cubbyllm.training — data pipeline, train loop, P5 gate (all stubs).

Wired: STANDALONE — package marker for the training layer.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .data import DataPipeline, InMemoryDataPipeline, WeightedCorpusPipeline
from .loop import TrainLoop
from .probing import P5WrongContextProbe

__all__ = [
    "DataPipeline",
    "InMemoryDataPipeline",
    "WeightedCorpusPipeline",
    "TrainLoop",
    "P5WrongContextProbe",
]
__wiring__ = Wiring.STANDALONE
