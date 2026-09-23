"""RetrievalHead — the mandatory full-softmax bypass (H-C3 decision).

Wired: WIRED — output head in the default forward path.

H-C3 measured that a full-vocabulary softmax is linear in V and that at large V
the codebook itself is the wall (V=1M fp32 = 10-41GB), and that an exact top-K
shortlist read is ~32x faster at the real head shape while staying accurate.
So the head does NOT full-softmax: it scores a query against the codebook,
keeps the top-K candidates, and softmaxes only over those.

``TopKRetrievalHead`` uses an EXACT brute-force top-K — the correctness
reference. A real ANN index (the gcheese-faiss existence proof, H-C3) is a
drop-in swap for the scoring step later; the top-K-then-local-softmax contract
is unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...core.protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor


@runtime_checkable
class RetrievalHead(Protocol):
    """ANN top-K retrieval + local softmax over the candidates.

    ``logits`` returns a (B, V) tensor where non-top-K entries are ``-inf`` so a
    downstream softmax/cross-entropy is a local softmax over the retrieved set.
    """

    def logits(self, query: "Tensor", k: int) -> "Tensor":
        ...


class TopKRetrievalHead:
    """Top-K retrieval + local softmax (H-C3 reference).

    Two code modes, both returning top-K-masked logits (rest ``-inf``):
      - ``learnable=False`` (default): FIXED row-normalized codebook, cosine
        readout scaled by sqrt(d) — the VSA-style head cubby-lm used. Needs
        large d for large V (H-C7 crosstalk), so use it at the real block dim.
      - ``learnable=True``: the codebook is an ``nn.Parameter`` and scoring is a
        plain linear projection (temperature 1) — a standard learnable LM head,
        which separates tokens by training rather than by dimensionality. Use
        this for small-d subword runs where a fixed codebook would drown in
        crosstalk.
    """

    def __init__(
        self,
        codebook: "Tensor",
        temperature: float | None = None,
        learnable: bool = False,
    ) -> None:
        import math

        import torch.nn as nn
        import torch.nn.functional as F

        self.learnable = bool(learnable)
        self.vocab, d = codebook.shape[0], codebook.shape[1]
        if self.learnable:
            self._param = nn.Parameter(codebook.clone())
            self.temperature = float(temperature) if temperature is not None else 1.0
        else:
            self.codebook = F.normalize(codebook, dim=-1)      # (V, d), fixed
            self.temperature = (
                float(temperature) if temperature is not None else math.sqrt(d)
            )

    def parameters(self):
        return iter([self._param]) if self.learnable else iter(())

    def _codes(self) -> "Tensor":
        # learnable head is a plain linear projection; fixed head is cosine.
        return self._param if self.learnable else self.codebook

    def quantize_(self) -> "TopKRetrievalHead":
        """Replace the codebook with a weight-only int8 one, in place.

        Opt-in, inference-only, and irreversible for this object — the
        float32 codebook is dropped. At the real head shape it is the
        largest tensor in the model and decode reads all of it per token,
        so it is the one place weight-only int8 clearly pays; see
        ``cubbyllm.ops.quant`` for the measurement and for why the
        activations stay float32.

        Raises on a backend with no int8 path rather than leaving float32
        in place, so a benchmark cannot report an unquantized run as a
        quantized one. ``ops.int8_available()`` asks first.
        """
        from ...ops import int8_weight_only

        if self.learnable:
            self._param = int8_weight_only(self._param)
        else:
            self.codebook = int8_weight_only(self.codebook)
        return self

    def logits(self, query: "Tensor", k: int) -> "Tensor":
        import torch
        import torch.nn.functional as F

        q = query if self.learnable else F.normalize(query, dim=-1)
        # F.linear, not `q @ codes.t()`: identical arithmetic — the same
        # kernel with the weight read transposed — but it is the call a
        # quantized weight can answer with its own kernel, and it does not
        # ask the backend to materialize a (V, d) transpose first.
        sims = F.linear(q, self._codes()) * self.temperature   # (..., V)
        k = min(int(k), self.vocab)
        if k >= self.vocab:
            return sims                                        # full softmax
        topk = sims.topk(k, dim=-1)
        out = torch.full_like(sims, float("-inf"))
        out.scatter_(-1, topk.indices, topk.values)            # keep only top-K
        return out


__wiring__ = Wiring.WIRED
