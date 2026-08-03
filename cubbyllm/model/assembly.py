"""CubbyModel — the single place the forward path is assembled.

Wired: WIRED — this IS the default forward path.

The whole thesis is visible here: a ``Context`` is inferred once (by an
offline-pretrained, frozen ``ContextSource`` — H0b), then threaded explicitly
through every generated stage. Omitting ``ctx`` from a generated call does not
type-check, which is the structural guard against the GCE "bypass" defect.

The runnable next-token path is: infer context -> embed -> backbone ->
hardened memory read -> retrieval output head. The ``BindingHead`` is a held,
WIRED component serving the relational / world-model channel (role-filler
structure, H-B5), not the vanilla token-logit path — the output-side readout is
the retrieval head (project + cosine top-K), which is what cubby-lm's clean
VSABindingHead actually did.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.protocols import Wiring

if TYPE_CHECKING:
    from torch import Tensor

    from ..core.config import CubbyConfig
    from ..core.context import ContextSource
    from .backbone.base import Backbone
    from .binding.base import BindingHead
    from .memory.base import MemoryLayer
    from .vocab.embedding import EmbeddingSource
    from .vocab.output_head import RetrievalHead


class CubbyModel:
    """Assembles backbone + hardened memory + binding head + hybrid vocab head,
    threading an inferred Context through the generated stages."""

    def __init__(
        self,
        config: "CubbyConfig",
        context_source: "ContextSource",
        backbone: "Backbone",
        memory: "MemoryLayer",
        binding: "BindingHead",
        embedding: "EmbeddingSource",
        head: "RetrievalHead",
        retrieval_k: int = 64,
    ) -> None:
        self.config = config
        self.context_source = context_source
        self.backbone = backbone
        self.memory = memory
        self.binding = binding
        self.embedding = embedding
        self.head = head
        self.retrieval_k = int(retrieval_k)

    def parameters(self):
        """Trainable parameters: embedding + backbone + memory's generator +
        (a learnable retrieval head, if used).

        The context source is EXCLUDED (offline-pretrained + frozen, per H0b). A
        FIXED VSA codebook head contributes nothing (no grads); a learnable head
        contributes its codebook. Only ``requires_grad`` params are yielded.
        """
        comps = (
            self.embedding,
            self.backbone,
            getattr(self.memory, "generator", None),
            self.head,
        )
        for comp in comps:
            if comp is None or not hasattr(comp, "parameters"):
                continue
            for p in comp.parameters():
                if getattr(p, "requires_grad", False):
                    yield p

    def infer_context(self, tokens: "Tensor"):
        """Pool the core embedding (tail off) into a router feature, infer c.

        Uses ``ctx=None`` so the router feature does not depend on the very
        context it is about to produce (no chicken-and-egg).
        """
        core_feat = self.embedding.embed(tokens, ctx=None).mean(dim=1)  # (B, d)
        return self.context_source.infer(core_feat)

    def features(self, tokens: "Tensor") -> "Tensor":
        """The trunk representation h (B, S, d) — pre-head hidden states. Exposed
        so an auxiliary objective (e.g. the VSA binding loss) can shape h without
        re-running the forward."""
        ctx = self.infer_context(tokens)                     # frozen at inference
        x = self.embedding.embed(tokens, ctx)                # (B, S, d), hybrid
        h = self.backbone.forward(x)                         # (B, S, d)
        return self.memory.forward_generated(h, ctx)         # (B, S, d), hardened

    def logits_from(self, h: "Tensor") -> "Tensor":
        return self.head.logits(h, self.retrieval_k)         # (B, S, V), top-K

    def step(self, token: "Tensor", state: "dict | None" = None):
        """Decode ONE token, carrying state. token: (B,) ids -> (logits, state).

        The inference path. Cost per token is independent of how many tokens came
        before, and the carried state is n_layers x (B, d) plus two context
        accumulators — it does not grow with context, unlike a KV cache. Without
        this, generation re-runs the whole prefix per token (O(S) work per token,
        worse than attention-with-cache), which is what ``sample_text`` does today.

        ONE DOCUMENTED DIFFERENCE FROM ``forward``: ``infer_context`` mean-pools
        the embedding over the WHOLE sequence, so in training the context at
        position t sees tokens after t. That is not reproducible causally, so
        decode uses a running mean over tokens seen so far. Contexts therefore
        differ slightly from a full forward on the same prefix; everything else
        is exact (see tests/model/test_mingru_decode.py).
        """
        import torch

        if state is None:
            state = {"bb": None, "ctx_sum": None, "ctx_n": 0}
        e = self.embedding.embed(token.unsqueeze(1), ctx=None)[:, 0]   # (B, d)
        s = e if state["ctx_sum"] is None else state["ctx_sum"] + e
        n = state["ctx_n"] + 1
        ctx = self.context_source.infer(s / n)                          # running mean
        x = self.embedding.embed(token.unsqueeze(1), ctx)[:, 0]
        y, bb = self.backbone.step(x, state["bb"])
        y = self.memory.forward_generated(y.unsqueeze(1), ctx)[:, 0]
        logits = self.head.logits(y.unsqueeze(1), self.retrieval_k)[:, 0]
        return logits, {"bb": bb, "ctx_sum": s, "ctx_n": n}

    def forward(self, tokens: "Tensor") -> "Tensor":
        """Context-threaded forward pass -> next-token logits (B, S, V).

        ``ctx`` is a required argument to every generated call below — a
        context-ignoring path cannot be written here without a type error.
        """
        return self.logits_from(self.features(tokens))


__wiring__ = Wiring.WIRED
