"""TrainLoop — the training driver.

Wired: STANDALONE — orchestration, not the forward path.

Holds the model, a manifest-pinned ``DataPipeline``, and the ``Hardener`` that
H0 proved essential — so anti-drift regularization is a structural part of the
loop, not an afterthought. ``step`` does one gradient step: forward -> next-token
cross-entropy (over the retrieval head's local candidate set) + the hardener's
drift penalty -> backward -> optimizer step. The hardener contribution is what
makes this theta=f(c) loop retain prior contexts rather than overwrite them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.protocols import Wiring

if TYPE_CHECKING:
    from ..core.generation import Hardener, ParameterGenerator
    from ..model.assembly import CubbyModel
    from .data import DataPipeline


class TrainLoop:
    """Drives training with hardened context-conditioned generation."""

    def __init__(
        self,
        model: "CubbyModel",
        data: "DataPipeline",
        hardener: "Hardener",
        lr: float = 1e-3,
        batch_size: int = 8,
        seq_len: int = 16,
        device=None,
    ) -> None:
        import torch

        self.model = model
        self.data = data
        self.hardener = hardener
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        # device: None keeps CPU behavior; pass a resolved device (e.g. cuda) to
        # train on GPU. The model must already be moved there (see core.device).
        self.device = device
        self.opt = torch.optim.Adam(list(model.parameters()), lr=lr)
        self._batches = data.batches(self.batch_size, self.seq_len)

    def _generator(self) -> "ParameterGenerator | None":
        return getattr(self.model.memory, "generator", None)

    def step(self) -> float:
        """One training step. Returns the scalar loss (CE + hardener penalty)."""
        import torch
        import torch.nn.functional as F

        x, y = next(self._batches)
        if self.device is not None:                          # GPU: batches are CPU
            x, y = x.to(self.device), y.to(self.device)
        logits = self.model.forward(x)                       # (B, S, V)
        vocab = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))

        gen = self._generator()
        if gen is not None:
            pen = self.hardener.penalty(gen)
            if torch.is_tensor(pen):
                loss = loss + pen

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


__wiring__ = Wiring.STANDALONE
