"""cubbyllm.model.binding — the VSA binding head + real unbind.

Wired: WIRED — package marker.
"""
from __future__ import annotations

from ...core.protocols import Wiring
from .base import BindingHead, DirectUnbinder, Unbinder

__all__ = ["BindingHead", "DirectUnbinder", "Unbinder"]
__wiring__ = Wiring.WIRED
