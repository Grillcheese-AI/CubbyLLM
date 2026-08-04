"""cubbyllm.model.recall — episodic (per-token) recall memory past the window.

Wired: WIRED — package marker. Distinct from cubbyllm/model/memory (the
parametric theta=f(c) memory). See docs/superpowers/specs/2026-08-04-episodic-
memory-design.md and H-D5.
"""
from __future__ import annotations

from ...core.protocols import Wiring
from .read import MemoryRead

__all__ = ["MemoryRead"]
__wiring__ = Wiring.WIRED
