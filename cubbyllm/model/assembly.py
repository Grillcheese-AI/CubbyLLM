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
        """Infer c at EVERY position, from the tokens up to and including it.

        The router feature at position t is the running mean of the core
        embedding (tail off) over tokens 0..t: exactly what :meth:`step`
        accumulates one token at a time. So a training forward and a decode
        see the same context at every position, and no position's context
        depends on a later token. ``ctx=None`` so the feature does not depend
        on the very context it is about to produce (no chicken-and-egg).
        Returns a Context whose ``vector`` is (B, S, ctx_dim).

        Until 2026-09-23 this mean-pooled the WHOLE sequence into one (B, d)
        feature: in training, position t's context saw the tokens after t,
        and decode, which cannot, ran under a different context from the one
        the weights were trained with.
        """
        import torch

        e = self.embedding.embed(tokens, ctx=None).float()           # (B, S, d)
        n = torch.arange(1, e.shape[1] + 1, device=e.device, dtype=e.dtype)
        core_feat = e.cumsum(dim=1) / n.unsqueeze(-1)                # causal running mean
        return self.context_source.infer(core_feat)                  # vector (B, S, ctx)

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
        worse than attention-with-cache), which is what ``train_colab.sample_text``
        still does; ``generate`` below is built on this method instead.

        The context is the running mean of the core embedding over the tokens
        seen so far, which is what ``infer_context`` computes at every position
        of a training forward. Decode and forward therefore see the same
        context (tests/model/test_causal_context.py); until 2026-09-23 the
        forward pooled the whole sequence and the two differed.

        **Both accumulators are device tensors updated in place**, and the
        token count is one of them — a float32 scalar, not a Python int.
        That looks like a pointless change until you try to record the step
        and replay it (grilly2's ``graphed``, ``docs/capture.md``; CUDA
        graphs, identically). A Python ``ctx_n`` is part of the call's
        signature, so every token is a *new* signature, a recording is
        taken and thrown away every time, and nothing ever replays — the
        whole mechanism silently does nothing while looking like it works.
        Counting on the device instead costs one 4-byte buffer and is
        exact to 2**24 tokens.

        In place, and returned, for the second half of the same contract:
        a replay re-issues commands over the buffers the recording named,
        so the state a replay produces has to *be* the state the next
        replay reads. Returning fresh tensors would leave the wrapper
        copying them back every token.

        ``state=None`` builds one through :meth:`init_state`, which is the
        only place this path allocates.
        """
        import torch

        if state is None:
            state = self.init_state(token.shape[0], token.device)
        e = self.embedding.embed(token.unsqueeze(1), ctx=None)[:, 0]   # (B, d)
        s, n = state["ctx_sum"], state["ctx_n"]
        s.add_(e)
        n.add_(1.0)
        ctx = self.context_source.infer(s / n)                          # running mean
        x = self.embedding.embed(token.unsqueeze(1), ctx)[:, 0]
        y, bb = self.backbone.step(x, state["bb"])
        y = self.memory.forward_generated(y.unsqueeze(1), ctx)[:, 0]
        logits = self.head.logits(y.unsqueeze(1), self.retrieval_k)[:, 0]
        return logits, {"bb": bb, "ctx_sum": s, "ctx_n": n}

    def init_state(self, batch: int = 1, device=None) -> dict:
        """Every buffer a decode sequence needs, allocated once.

        ``step`` allocates nothing once it has one of these, and that is
        what makes a recorded step reusable across ``generate`` calls: the
        first step used to build the backbone's episodic store, SimHash
        projection and all, and ``torch.randn`` inside a capture is a host
        write the replay cannot reproduce. A graphed step kept from an
        earlier sequence therefore failed at its next capture rather than
        at the call that caused it.

        The context accumulators are seeded with zeros rather than with the
        first embedding, so token 1 computes ``0 + e`` — exact, and the
        same shapes as every later token.
        """
        import torch

        d = self.config.d_model
        bb = self.backbone.init_state(batch, device) if hasattr(self.backbone, "init_state") else None
        return {"bb": bb,
                "ctx_sum": torch.zeros(batch, d, device=device),
                "ctx_n": torch.zeros((), device=device)}

    def graphed_step(self):
        """The step, recorded once and replayed, kept for this model.

        One per model rather than one per ``generate``: a recording is
        keyed by shape signature, and a decode step's signature is the same
        for every sequence, so recapturing per call throws away a graph
        that was about to be reused. It is built lazily because
        ``ops.graph_step`` is the identity on a backend without capture and
        there is then nothing to keep."""
        from ..ops import graph_step

        if getattr(self, "_graphed_step", None) is None:
            self._graphed_step = graph_step(self.step)
        return self._graphed_step

    def generate(self, prompt_ids, state: "dict | None" = None, on_token=None, **cfg):
        """Continue ``prompt_ids`` on the O(1) ``step`` path, with the
        anti-repetition guards. ``cfg`` fields are ``DecodeConfig``'s (default:
        greedy + repetition penalty 1.3 + 3-gram block). Returns (new ids, state);
        pass ``state`` back to continue a conversation. See ``core/decoding.py``.

        The step goes through :meth:`graphed_step`. On a backend with graph
        capture that records the step once and replays it for every later
        token, and the recording is kept on the model so a second
        ``generate`` replays rather than recaptures; on a backend without,
        it is the identity and this reads exactly as it did."""
        from ..core.decoding import DecodeConfig, generate

        return generate(self, prompt_ids, DecodeConfig(**cfg), state=state,
                        on_token=on_token, step=self.graphed_step())

    def forward(self, tokens: "Tensor") -> "Tensor":
        """Context-threaded forward pass -> next-token logits (B, S, V).

        ``ctx`` is a required argument to every generated call below — a
        context-ignoring path cannot be written here without a type error.
        """
        return self.logits_from(self.features(tokens))


__wiring__ = Wiring.WIRED
