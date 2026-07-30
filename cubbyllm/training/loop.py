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
        amp: bool = True,
        grad_clip: float = 1.0,
        warmup: int = 0,
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
        # amp: bf16 autocast, applied ONLY on CUDA (halves memory, ~2-3x faster at
        # 2B scale). bf16 needs no GradScaler; weights stay fp32. No-op off CUDA.
        self.amp = bool(amp)
        # grad_clip: global-norm clip before opt.step — REQUIRED for stability with
        # the generated θ=f(c) memory (a single bad batch otherwise detonates the
        # loss). warmup: linear LR ramp over the first N steps (early-divergence guard).
        self.grad_clip = float(grad_clip)
        self.warmup = int(warmup)
        self._base_lr = float(lr)
        self._nstep = 0
        self.opt = torch.optim.Adam(list(model.parameters()), lr=lr)
        self._batches = data.batches(self.batch_size, self.seq_len)

    def _generator(self) -> "ParameterGenerator | None":
        return getattr(self.model.memory, "generator", None)

    def step(self) -> float:
        """One training step. Returns the scalar loss (CE + hardener penalty)."""
        import contextlib

        import torch
        import torch.nn.functional as F

        x, y = next(self._batches)
        if self.device is not None:                          # GPU: batches are CPU
            x, y = x.to(self.device), y.to(self.device)
        # bf16 autocast on CUDA only; forward + loss run mixed-precision, backward
        # and the fp32 master weights are untouched (bf16 range needs no scaler).
        use_amp = self.amp and self.device is not None and self.device.type == "cuda"
        actx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp else contextlib.nullcontext())
        with actx:
            logits = self.model.forward(x)                   # (B, S, V)
            vocab = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            gen = self._generator()
            if gen is not None:
                pen = self.hardener.penalty(gen)
                if torch.is_tensor(pen):
                    loss = loss + pen

        self._nstep += 1
        if self.warmup and self._nstep <= self.warmup:           # linear LR warmup
            for g in self.opt.param_groups:
                g["lr"] = self._base_lr * self._nstep / self.warmup

        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:                # kill explosion spikes
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.detach())


__wiring__ = Wiring.STANDALONE
