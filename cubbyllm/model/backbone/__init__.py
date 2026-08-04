"""cubbyllm.model.backbone — the sequence backbone.

Wired: WIRED — package marker. ``base.py`` holds the ``Backbone`` protocol only,
so alternate backbones stay pluggable; ``MinGRUBackbone`` (mingru.py) is the
chosen default (H-D1 resolved 2026-07-23 via the in-architecture bake-off).
"""
from __future__ import annotations

from ...core.protocols import Wiring
from .base import Backbone
from .hybrid import HybridBackbone
from .mingru import MinGRUBackbone

__all__ = ["Backbone", "MinGRUBackbone", "HybridBackbone"]
__wiring__ = Wiring.WIRED
