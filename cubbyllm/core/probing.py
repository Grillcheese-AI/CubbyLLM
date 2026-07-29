"""Wrong-Context Probing (P5) — the validation seam, in-package.

Wired: STANDALONE — a protocol; the concrete training-gate wiring lives in
``cubbyllm.training.probing``.

The GCE "Structural Bypass Defect" is detected by feeding a ``Generable`` the
RIGHT context vs deliberately WRONG contexts and checking that wrong contexts
measurably hurt. If they don't, the context signal was optimized away and
theta=f(c) is not actually doing the work. Keeping this protocol in-package (not
only in ``validation/``) means the discipline can gate real training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor

    from .context import Context
    from .protocols import Generable


@runtime_checkable
class WrongContextProbe(Protocol):
    """Runs the P5 protocol against a Generable.

    Returns a metrics dict, at minimum ``{"right": float, "wrong_mean": float,
    "bypass": bool}`` where ``bypass`` is True when wrong contexts fail to hurt.
    """

    def probe(
        self,
        target: "Generable",
        x: "Tensor",
        right: "Context",
        wrong: "list[Context]",
    ) -> "dict[str, float]":
        ...


__wiring__ = Wiring.STANDALONE
