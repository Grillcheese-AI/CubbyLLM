"""cubbyllm.core — thesis abstractions (theta=f(c)).

Wired: STANDALONE — package marker for the core type layer.
"""
from __future__ import annotations

from .protocols import Generable, Wiring, declare_wiring

__all__ = ["Generable", "Wiring", "declare_wiring"]
__wiring__ = Wiring.STANDALONE
