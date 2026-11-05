"""cubbyllm/model/vocab/fused_ce.py -- the exact cross-entropy without the logits (order 0050, stage 2a).

The training loss over a 128k vocabulary stores the logits of every position (8 x 1,024 x 128k: 13.3-13.8 GB a micro
batch with their gradient and the casts; 0050's A100 read). Here the loss and both gradients are computed chunk by
chunk from the hidden state and the head's rows, in the forward pass, and only the gradients are kept:
  per chunk of positions   logits = h W^T (the matmul in ``dtype``, the softmax in fp32), the log-sum-exp, the
                           target's logit -> the loss; p = softmax - onehot over the count -> dh = p W, dW += p^T h
  backward                 the upstream gradient times dh and dW (a scalar: the loss is the last op)
Three matmuls, as the full path runs (forward, and the two of its backward), never a (positions x vocabulary) tensor
beyond one chunk. Without grad (eval) only the loss is computed.
``dtype``: the matmuls' dtype. The callers compute ``h`` under autocast and call this outside it, where ``h`` (after the
trunk's last norm) is fp32: the matmuls then ran in fp32 -- 4x the full path's bf16 head on the A100's first read
(+460 ms a step on both shapes, ``head_share_train_cuda_2026-11-05``). None: autocast's dtype if it is on at the call,
else h's.
The loss is the same function as ``F.cross_entropy(h @ W.T, y)`` (mean over the positions whose target is not
``ignore_index``); only the memory changes. Pure torch (no kernel to build), so every card runs it.

Wired: WIRED (``validation/train_base.py`` with ``CB_FUSED_CE=1``; off by default, the full path unchanged).
"""
from __future__ import annotations

import os

__wiring__ = "WIRED"

CHUNK = int(os.environ.get("CB_FUSED_CE_CHUNK") or 1024)    # positions a chunk (0050: one try at larger ones)


def on() -> bool:
    """``CB_FUSED_CE=1``: the training loss without the logits. Off by default."""
    return os.environ.get("CB_FUSED_CE") == "1"


def _function():
    import torch

    class ChunkedCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h, w, y, chunk, ignore_index, dt, grad_h, grad_w):
            n = h.shape[0]
            hd, wd = h.to(dt), w.to(dt)
            keep = y != ignore_index
            count = keep.sum().clamp(min=1)
            total = torch.zeros((), dtype=torch.float32, device=h.device)
            dh = torch.empty_like(h) if grad_h else None
            dw = torch.zeros(w.shape, dtype=torch.float32, device=w.device) if grad_w else None
            for s in range(0, n, chunk):
                hc, yc, kc = hd[s:s + chunk], y[s:s + chunk], keep[s:s + chunk]
                logits = (hc @ wd.t()).float()
                lse = torch.logsumexp(logits, -1)
                tl = logits.gather(1, yc.clamp(min=0).unsqueeze(1))[:, 0]
                total = total + ((lse - tl) * kc).sum()
                if dh is None and dw is None:
                    continue
                p = torch.exp(logits.sub_(lse[:, None]))                       # the chunk's logits, reused in place
                p.scatter_add_(1, yc.clamp(min=0)[:, None], -kc.float()[:, None])   # p - onehot (no host sync)
                p = (p * kc[:, None]) / count
                pd = p.to(dt)
                if dh is not None:
                    dh[s:s + chunk] = (pd @ wd).to(h.dtype)
                if dw is not None:
                    dw += (pd.t() @ hc).float()
            ctx.save_for_backward(*(t for t in (dh, dw) if t is not None))
            ctx.has = (dh is not None, dw is not None)
            ctx.w_dtype = w.dtype
            return total / count

        @staticmethod
        def backward(ctx, g):
            saved = list(ctx.saved_tensors)
            dh = saved.pop(0) * g if ctx.has[0] else None                       # g is 0-d: dh keeps h's dtype
            dw = (saved.pop(0) * g.float()).to(ctx.w_dtype) if ctx.has[1] else None
            return dh, dw, None, None, None, None, None, None

    return ChunkedCE


_FN: list = []


def _dtype(h, dtype):
    import torch
    if dtype is not None:
        return dtype
    dev = h.device.type
    try:
        if torch.is_autocast_enabled(dev):
            return torch.get_autocast_dtype(dev)
    except (TypeError, AttributeError):                                     # torch < 2.4: the CUDA-only API
        if dev == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return h.dtype


def chunked_cross_entropy(h, w, y, chunk: int = CHUNK, ignore_index: int = -100, dtype=None):
    """Mean cross-entropy of ``h @ w.T`` against ``y`` (h (N, d) or (B, S, d); y matching), the logits never stored.
    ``dtype``: the matmuls' (None: autocast's when it is on, else h's)."""
    import torch
    if not _FN:
        _FN.append(_function())
    dt = _dtype(h, dtype)
    grad = torch.is_grad_enabled()
    with torch.autocast(device_type=h.device.type, enabled=False):
        return _FN[0].apply(h.reshape(-1, h.shape[-1]), w, y.reshape(-1), int(chunk), int(ignore_index), dt,
                            grad and h.requires_grad, grad and w.requires_grad)


__all__ = ["CHUNK", "chunked_cross_entropy", "on"]
