"""P5WrongContextProbe — concrete P5 gate wired for training.

Wired: STANDALONE — a validation gate around training, not the forward path.

Implements ``core.probing.WrongContextProbe``. Direct port of the validated
``validation/exp_h0_gce.py`` / ``exp_h0b`` P5 logic: feed a ``Generable`` the
RIGHT context vs deliberately WRONG contexts and check that wrong contexts
measurably HURT. If they don't, the GCE "bypass" defect is present — the context
signal was optimized away and theta=f(c) is not doing the work. Catching this
before a full run is the whole point (H0's naive variant failed exactly here).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.probing import WrongContextProbe
from ..core.protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor

    from ..core.context import Context
    from ..core.protocols import Generable


class P5WrongContextProbe(WrongContextProbe):
    """Concrete Wrong-Context Probe used as a training gate.

    ``bypass_ratio`` — wrong-context error must exceed right-context error by at
    least this factor to count as load-bearing; below it, ``bypass`` is True.
    """

    def __init__(self, bypass_ratio: float = 1.15) -> None:
        self.bypass_ratio = float(bypass_ratio)

    def probe(
        self,
        target: "Generable",
        x: "Tensor",
        right: "Context",
        wrong: "list[Context]",
    ) -> "dict[str, float]":
        """Return {"right", "wrong_mean", "bypass"}.

        Error is the mean-squared deviation of the target's generated output
        under a wrong context from its output under the right context — i.e.
        how much forcing the wrong context changes the read. If wrong contexts
        barely change the output, the context is being bypassed.
        """
        import torch

        with torch.no_grad():
            out_right = target.forward_generated(x, right)
            wrong_devs = []
            for w in wrong:
                out_w = target.forward_generated(x, w)
                wrong_devs.append(float(((out_w - out_right) ** 2).mean()))
        right_scale = float((out_right ** 2).mean()) + 1e-12
        wrong_mean = sum(wrong_devs) / max(len(wrong_devs), 1)
        # bypass when wrong contexts fail to move the output relative to its scale
        bypass = wrong_mean < (self.bypass_ratio - 1.0) * right_scale
        return {
            "right": right_scale,
            "wrong_mean": wrong_mean,
            "bypass": float(bool(bypass)),
        }


__wiring__ = Wiring.STANDALONE
