"""int8 weight-only quantization, through whichever backend is loaded.

Wired: WIRED — ``TopKRetrievalHead.quantize_`` calls it; opt-in, never on
by default.

The retrieval head is one matmul against a (V, d) codebook and at the real
vocabulary it is the largest single tensor in the model — at V=127996,
d=512 it is 262 MB of float32, and decode reads all of it for every token.
That makes it memory-bound, which is exactly the case weight-only int8
pays for: a quarter of the bytes at the same arithmetic. Measured on an
RX 6750 XT with random weights at that shape, **1.35 ms -> 0.42 ms**, with
argmax unchanged.

It stays opt-in because "argmax unchanged on random weights" is not
evidence about a trained model. What decides it is the teacher-forced
cross-entropy delta and the greedy-token agreement on a real checkpoint,
which is a measurement on the owner's side of the fence.

**Weight-only, not activations.** The activation stays float32 and the
weight is dequantized inside the matmul. W8A8 was measured on this card
and does not pay: the int8 decode linear already runs at 377 GB/s of the
card's 432 GB/s, so it is memory-bound at 87% and there is no arithmetic
headroom to buy.

The seam reaches the quantizer through the **active tensor module**, not
through an ``import grilly``. Under ``-p grilly.torch_alias`` that finds
grilly2; under real torch it finds nothing and the caller is told so
rather than silently getting float32 back. A quantization that quietly
did not happen is the worst of the three outcomes, because the number it
produces looks like a result.
"""
from __future__ import annotations

from ..core.protocols import Wiring


def int8_available() -> bool:
    """Whether the loaded backend can quantize a linear weight to int8."""
    return _int8_tensor() is not None


def _int8_tensor():
    import torch

    infer = getattr(torch, "infer", None)
    quantization = getattr(infer, "quantization", None)
    return getattr(quantization, "Int8Tensor", None)


def int8_weight_only(weight):
    """``weight`` as int8 values with one float32 scale per output row.

    The result is a tensor the backend's ``F.linear`` consumes directly —
    torchao's ``Int8WeightOnlyConfig`` shape, which is what grilly2
    mirrors. Every other operation on it sees the dequantized weight, so
    nothing downstream has to know.

    Raises if the backend has no int8 path, because the caller asked for
    quantization and getting float32 back would misreport the run.
    """
    int8 = _int8_tensor()
    if int8 is None:
        import torch

        raise RuntimeError(
            f"{getattr(torch, '__name__', 'the tensor backend')} has no weight-only int8 "
            "path (cubbyllm.ops.quant looks for torch.infer.quantization.Int8Tensor, "
            "which grilly2 provides). Check ops.quant.int8_available() first."
        )
    return int8.from_hp(weight)


__wiring__ = Wiring.WIRED
