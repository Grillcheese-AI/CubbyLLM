"""CubbyLLM — context-conditioned LM (theta=f(c)). Package skeleton.

Wired: STANDALONE — top-level package marker.

This package is currently a THESIS-FIRST SKELETON: interfaces only, every
component body raises ``NotImplementedError``. See
``docs/superpowers/specs/2026-07-23-cubbyllm-package-skeleton-design.md``.
Nothing here trains anything yet — by design.
"""
from __future__ import annotations

from .core.context import Context
from .core.protocols import Generable, Wiring, declare_wiring

__version__ = "0.0.0"

__all__ = ["Context", "Generable", "Wiring", "declare_wiring", "__version__"]
__wiring__ = Wiring.STANDALONE
