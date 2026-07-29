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
        import torch.nn.functional as F

        # learnable head is a plain linear projection; fixed head is cosine.
        return self._param if self.learnable else self.codebook

    def logits(self, query: "Tensor", k: int) -> "Tensor":
        import torch
        import torch.nn.functional as F

        q = query if self.learnable else F.normalize(query, dim=-1)
        sims = q @ self._codes().t() * self.temperature        # (..., V)
        k = min(int(k), self.vocab)
        if k >= self.vocab:
            return sims                                        # full softmax
        topk = sims.topk(k, dim=-1)
        out = torch.full_like(sims, float("-inf"))
        out.scatter_(-1, topk.indices, topk.values)            # keep only top-K
        return out


__wiring__ = Wiring.WIRED
