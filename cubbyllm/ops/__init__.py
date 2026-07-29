"""cubbyllm.ops — the ONLY package permitted to import grilly.

Wired: WIRED — the VSA algebra seam every generated/binding path depends on.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .vsa import BlockCodeVSA

__all__ = ["BlockCodeVSA"]
__wiring__ = Wiring.WIRED
