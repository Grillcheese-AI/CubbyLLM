"""Core protocols for CubbyLLM.

Wired: STANDALONE — pure type definitions, not part of any forward path.

This module is the root of the thesis-first design: ``Generable`` makes
"active weights are a function of context" (theta=f(c)) a type-level constraint,
and ``Wiring`` lets every module declare whether it is *meant* to sit in the
default forward path — an intent declaration kept deliberately separate from
build status (a stub body raising ``NotImplementedError``).
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

    from .context import Context


class Wiring(Enum):
    """Declares a module's *intent*, never its build status.

    WIRED      — designed to sit in the default forward path once implemented.
    STANDALONE — a standalone attachment / type module / bridge.

    A component is only ever "done" when its intent is WIRED, its body is real,
    ``assembly.py`` actually calls it, AND its test is un-skipped and green —
    never on the strength of ``__wiring__`` alone.
    """

    WIRED = "wired"
    STANDALONE = "standalone"


def declare_wiring(status: Wiring) -> Wiring:
    """Identity helper so ``__wiring__ = declare_wiring(Wiring.WIRED)`` reads clearly."""
    return status


@runtime_checkable
class Generable(Protocol):
    """Anything whose active weights are generated from context (theta=f(c)).

    Memory, binding, and vocab heads all implement this, so a forward path that
    ignores context does not type-check.
    """

    def forward_generated(self, x: "Tensor", ctx: "Context") -> "Tensor":
        ...


__wiring__ = Wiring.STANDALONE
