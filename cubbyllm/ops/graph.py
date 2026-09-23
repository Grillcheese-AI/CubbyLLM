"""graph_step — record the decode step once, replay it every token.

Wired: WIRED — ``CubbyModel.generate`` wraps its step with this.

A decode step issues the same kernels over the same buffers every token and
changes only the bytes inside them. Running the Python again to rediscover
that costs more than the kernels do: measured on an RX 6750 XT at the hd5
shape, the full step is 8.1 ms eager against 5.1 ms replayed, and the
difference is entirely host time.

Both backends that matter expose the same mechanism under different names —
grilly2's ``grilly.graphed`` (Vulkan command buffers) and torch's CUDA
graphs. They also impose the same contract, which is why this seam is one
function and not two implementations: the step has to be a pure function of
the bytes in fixed buffers, with no host read, no host write, no shape
change, and state updated in place. ``CubbyModel.step`` and everything
under it was made to satisfy that; this is the switch that cashes it in.

Only ``cubbyllm.ops`` may import grilly, so the capability test lives here
and nowhere else. It is a test for the *capability*, not for the name: a
module that offers ``graphed`` gets used, anything else is passed through
unchanged. ``torch.cuda.graphs`` is a later branch and is deliberately not
guessed at here — a CUDA graph needs warmup on a side stream and static
input tensors, which is a different wrapper, not the same one.
"""
from __future__ import annotations

from typing import Callable

from ..core.protocols import Wiring


def graph_step(fn: Callable) -> Callable:
    """``fn`` with capture-and-replay if the backend has it, else ``fn``.

    Returns the argument untouched on a backend without capture, so a
    caller never needs to ask which backend it is on.
    """
    import torch

    graphed = getattr(torch, "graphed", None)
    return fn if graphed is None else graphed(fn)


def graph_stats(step: Callable) -> "dict | None":
    """``{"graphs", "captures", "replays", "misses"}`` for a wrapped step.

    ``None`` when ``step`` is not wrapped. The diagnostic that matters is
    ``replays``: a step that captures repeatedly and never replays is one
    whose shapes are not settling, and wrapping it is pure overhead. That
    was the state of this model's step until the decode state became
    fixed-size buffers with device cursors.
    """
    names = ("graphs", "captures", "replays", "misses")
    if not all(hasattr(step, n) for n in names):
        return None
    return {n: getattr(step, n) for n in names}


__wiring__ = Wiring.WIRED
