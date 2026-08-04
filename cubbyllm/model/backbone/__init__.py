"""cubbyllm.model.backbone — the sequence backbone.

Wired: WIRED — package marker. ``base.py`` holds the ``Backbone`` protocol only,
so alternate backbones stay pluggable; ``HybridBackbone`` (hybrid.py) is the
chosen default after the H-D4 needle A/B (2026-08-04: ~98% vs ~32% recall inside
its window, bounded state), with ``MinGRUBackbone`` the pure-recurrence baseline.
"""
from __future__ import annotations

from ...core.protocols import Wiring
from .base import Backbone
from .hybrid import HybridBackbone
from .mingru import MinGRUBackbone

__all__ = ["Backbone", "MinGRUBackbone", "HybridBackbone"]
__wiring__ = Wiring.WIRED
