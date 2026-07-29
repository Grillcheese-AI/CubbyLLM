"""EmbeddingSource — the HYBRID vocabulary embedding (H-C6 decision).

Wired: WIRED — input-side embedding in the default forward path.

The decided strategy (H-C6): a fixed CORE embedding table plus an optional
generated/composed TAIL. H-C1 measured that pure surface-form generation loses
quality at toy scale while its cost is negligible/cacheable — hence hybrid, not
pure generation. ``HybridEmbedding`` always does the core lookup; when a
``ParameterGenerator`` and a ``Context`` are supplied it adds a context-generated
shift (theta_embedding = f(c)), the seam for the dynamic tail.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...core.protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor

    from ...core.context import Context
    from ...core.generation import ParameterGenerator


@runtime_checkable
class EmbeddingSource(Protocol):
    """Embeds token ids via a fixed core, optionally extended by a generated tail.

    ``ctx=None`` -> core-only lookup. A non-None ``Context`` may drive the
    generated/dynamic tail once implemented.
    """

    def embed(self, token_ids: "Tensor", ctx: "Context | None" = None) -> "Tensor":
        ...


class HybridEmbedding:
    """Fixed core table + optional context-generated additive tail.

    vocab_core     — size of the fixed core embedding table.
    d_model        — embedding width.
    tail_generator — optional ParameterGenerator emitting a (d_model,) shift from
                     a Context; ``None`` -> core-only (a pure static embedding).
    """

    def __init__(
        self,
        vocab_core: int,
        d_model: int,
        tail_generator: "ParameterGenerator | None" = None,
    ) -> None:
        import torch.nn as nn

        self.core = nn.Embedding(vocab_core, d_model)
        self.d_model = int(d_model)
        self.tail_generator = tail_generator

    def parameters(self):
        return self.core.parameters()

    def embed(self, token_ids: "Tensor", ctx: "Context | None" = None) -> "Tensor":
        e = self.core(token_ids)                                # (..., d_model)
        if ctx is not None and self.tail_generator is not None:
            shift = self.tail_generator.generate(ctx).weights   # (..., d_model)
            shift = shift.reshape(-1, self.d_model).mean(dim=0)
            e = e + shift
        return e


__wiring__ = Wiring.WIRED
