"""cubbyllm/model/vocab/fused_ce.py -- the exact cross-entropy without the logits (order 0050, stage 2a).

The training loss over a 128k vocabulary stores the logits of every position (8 x 1,024 x 128k: 13.3-13.8 GB a micro
batch with their gradient and the casts; 0050's A100 read). Here the loss and its gradients are computed chunk by
chunk from the hidden state and the head's rows, and only the hidden state and one log-sum-exp a position are kept:
  forward   per chunk of positions: logits = h W^T (fp32), the log-sum-exp, the target's logit -> the loss
  backward  per chunk again: p = softmax(h W^T), p[target] -= 1, scaled by the upstream gradient over the count;
            dh = p W, dW += p^T h -- the logits are recomputed, never stored
The loss is the same function as ``F.cross_entropy(h @ W.T, y)`` (mean over the positions whose target is not
``ignore_index``); only the memory changes. Pure torch (no kernel to build), so every card runs it.

Wired: WIRED (``validation/train_base.py`` with ``CB_FUSED_CE=1``; off by default, the full path unchanged).
"""
from __future__ import annotations

import os

__wiring__ = "WIRED"

CHUNK = 1024


def on() -> bool:
    """``CB_FUSED_CE=1``: the training loss without the logits. Off by default."""
    return os.environ.get("CB_FUSED_CE") == "1"


def _function():
    import torch

    class ChunkedCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h, w, y, chunk, ignore_index):
            with torch.autocast(device_type=h.device.type, enabled=False):
                return ChunkedCE._fwd(ctx, h, w, y, chunk, ignore_index)

        @staticmethod
        def _fwd(ctx, h, w, y, chunk, ignore_index):
            n, wd = h.shape[0], w.to(h.dtype)               # the matmul in h's dtype, as autocast would run it
            lse = torch.empty(n, dtype=torch.float32, device=h.device)
            total = torch.zeros((), dtype=torch.float32, device=h.device)
            keep = y != ignore_index
            count = keep.sum().clamp(min=1)
            for s in range(0, n, chunk):
                logits = (h[s:s + chunk] @ wd.t()).float()
                lse[s:s + chunk] = torch.logsumexp(logits, -1)
                yc, kc = y[s:s + chunk], keep[s:s + chunk]
                tl = logits.gather(1, yc.clamp(min=0).unsqueeze(1))[:, 0]
                total = total + ((lse[s:s + chunk] - tl) * kc).sum()
            ctx.save_for_backward(h, w, y, lse, count)
            ctx.chunk, ctx.ignore = chunk, ignore_index
            return total / count

        @staticmethod
        def backward(ctx, g):
            h, w, y, lse, count = ctx.saved_tensors
            chunk, wd = ctx.chunk, w.to(h.dtype)
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dw = torch.zeros(w.shape, dtype=torch.float32, device=w.device) if ctx.needs_input_grad[1] else None
            scale = (g / count).float()
            for s in range(0, h.shape[0], chunk):
                hc = h[s:s + chunk]
                p = torch.exp((hc @ wd.t()).float() - lse[s:s + chunk, None])
                yc = y[s:s + chunk]
                kc = (yc != ctx.ignore)
                p.scatter_add_(1, yc.clamp(min=0)[:, None], -kc.float()[:, None])   # p - onehot (no host sync)
                p = (p * kc[:, None]) * scale
                if dh is not None:
                    dh[s:s + chunk] = p.to(h.dtype) @ wd
                if dw is not None:
                    dw += p.t() @ hc.float()
            return dh, (None if dw is None else dw.to(w.dtype)), None, None, None

    return ChunkedCE


_FN: list = []


def chunked_cross_entropy(h, w, y, chunk: int = CHUNK, ignore_index: int = -100):
    """Mean cross-entropy of ``h @ w.T`` against ``y`` (h (N, d) or (B, S, d); y matching), the logits never stored."""
    if not _FN:
        _FN.append(_function())
    return _FN[0].apply(h.reshape(-1, h.shape[-1]), w, y.reshape(-1), int(chunk), int(ignore_index))


__all__ = ["CHUNK", "chunked_cross_entropy", "on"]
