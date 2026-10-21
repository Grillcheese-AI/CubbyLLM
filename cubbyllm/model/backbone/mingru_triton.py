"""The MinGRU recurrence as one fused Triton kernel each way (opt-in: ``CB_MINGRU_TRITON=1``).

The torch path (``mingru._MinGRUMixer``) computes ``u = sigmoid(g) * tanh(v)``,
``a = 0.001 + 0.998 * sigmoid(d)`` and the scan ``h_t = a_t h_{t-1} + u_t`` as a dozen
separate kernels -- the gates, a sign-split log-domain scan (two ``logcumsumexp``), the
casts in and out of float32, and their reversed (``flip``) backward. On the 450M + link
(run 2 of the spatial notebook, 2026-10-20) that was 13% of a training step's device time,
plus part of the 22% of loose elementwise work, all of it memory-bound.

Here the three projections' outputs ``g, v, d`` go in (bf16 under autocast), ``h`` comes
out in float32 (as the torch path's), and one kernel does gates + scan; the backward is one
kernel too: it recomputes the gates from ``g, v, d``, runs the adjoint recurrence
``G_t = dh_t + a_{t+1} G_{t+1}`` in reverse, and writes the three input gradients. The scan
is direct, not log-domain: ``0.001 <= a < 1``, so a float32 product never overflows, and a
sequential carry has no underflowing prefix product to protect against (the log domain is
what the parallel *torch* scan needs, not the recurrence). Time goes in chunks of ``T``
steps -- a ``(T, C)`` tile, scanned with ``tl.associative_scan`` and carried across chunks
-- with ``C`` channels a program and one program per (batch row, channel block).

It is a ``torch.library.custom_op``: ``torch.compile`` sees one opaque op with a fake
(shape-only) implementation, autograd sees its backward. Exactness against the torch path
(values and gradients) is ``tests/model/backbone/test_mingru_triton.py``, which runs under
``TRITON_INTERPRET=1`` on a CPU and natively on a GPU.

What it brings CubbyLLM: the recurrence that makes its decode state constant-size costs a
fused memory pass in training instead of a dozen -- the same kind of fused kernel a
transformer gets for attention from FlashAttention, here for the part of the trunk that has
no off-the-shelf kernel.
"""
import os

import torch

from ...core.protocols import Wiring

try:
    import triton
    import triton.language as tl
    _jit = triton.jit
except ImportError:                    # no triton here (Windows, CPU-only wheels): the torch path is the only one
    triton = None

    class tl:                          # noqa: N801 -- just enough for the kernels' signatures to be defined
        constexpr = None

    def _jit(fn):
        return fn

AVAILABLE = triton is not None

T_CHUNK = int(os.environ.get("CB_MINGRU_T", "64"))     # time steps a tile
C_BLOCK = int(os.environ.get("CB_MINGRU_C", "32"))     # channels a program
WARPS = int(os.environ.get("CB_MINGRU_WARPS", "4"))


@_jit
def _combine(a1, x1, a2, x2):
    """(a1, x1) then (a2, x2): x -> a2 (a1 x + x1) + x2."""
    return a1 * a2, a2 * x1 + x2


@_jit
def _gates(g, v, d):
    sg = tl.sigmoid(g)
    tv = 2.0 * tl.sigmoid(2.0 * v) - 1.0                     # tanh
    sd = tl.sigmoid(d)
    return sg, tv, sd


@_jit
def _last_row(x, T: tl.constexpr):
    r = tl.arange(0, T)
    return tl.sum(tl.where(r[:, None] == T - 1, x, 0.0), axis=0)


@_jit
def _mingru_scan_fwd(zg, zv, zd, out, S, D, T: tl.constexpr, C: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    cm = cols < D
    base = b * S * D
    carry = tl.zeros([C], dtype=tl.float32)
    for t0 in range(0, S, T):
        rows = t0 + tl.arange(0, T)
        m = (rows[:, None] < S) & cm[None, :]
        off = base + rows[:, None].to(tl.int64) * D + cols[None, :]
        g = tl.load(zg + off, mask=m, other=0.0).to(tl.float32)
        v = tl.load(zv + off, mask=m, other=0.0).to(tl.float32)
        d = tl.load(zd + off, mask=m, other=0.0).to(tl.float32)
        sg, tv, sd = _gates(g, v, d)
        a = 0.001 + 0.998 * sd
        A, H = tl.associative_scan((a, sg * tv), 0, _combine)
        H = A * carry[None, :] + H
        tl.store(out + off, H, mask=m)
        carry = _last_row(H, T)


@_jit
def _mingru_scan_bwd(zg, zv, zd, out, dout, dg, dv, dd, S, D, T: tl.constexpr, C: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    cm = cols < D
    base = b * S * D
    carry = tl.zeros([C], dtype=tl.float32)                  # G of the chunk after this one
    for k in range(0, tl.cdiv(S, T)):
        rows = S - 1 - k * T - tl.arange(0, T)                # newest first: a forward scan over rows
        ok = rows >= 0
        m = ok[:, None] & cm[None, :]
        r64 = rows[:, None].to(tl.int64)
        off = base + r64 * D + cols[None, :]
        g = tl.load(zg + off, mask=m, other=0.0).to(tl.float32)
        v = tl.load(zv + off, mask=m, other=0.0).to(tl.float32)
        d = tl.load(zd + off, mask=m, other=0.0).to(tl.float32)
        go = tl.load(dout + off, mask=m, other=0.0).to(tl.float32)
        mn = m & (rows[:, None] + 1 < S)                      # a_{t+1}: the next step's retention
        dn = tl.load(zd + off + D, mask=mn, other=0.0).to(tl.float32)
        c = tl.where(mn, 0.001 + 0.998 * tl.sigmoid(dn), 0.0)
        Cc, G = tl.associative_scan((c, go), 0, _combine)
        G = Cc * carry[None, :] + G
        mp = m & (rows[:, None] >= 1)                         # h_{t-1}, zero before the first step
        hp = tl.load(out + off - D, mask=mp, other=0.0)
        sg, tv, sd = _gates(g, v, d)
        tl.store(dg + off, (G * sg * (1.0 - sg) * tv).to(dg.dtype.element_ty), mask=m)
        tl.store(dv + off, (G * sg * (1.0 - tv * tv)).to(dv.dtype.element_ty), mask=m)
        tl.store(dd + off, (G * hp * 0.998 * sd * (1.0 - sd)).to(dd.dtype.element_ty), mask=m)
        carry = _last_row(G, T)


def _grid(B: int, D: int):
    return (B, triton.cdiv(D, C_BLOCK))


@torch.library.custom_op("cubby::mingru_scan", mutates_args=())
def mingru_scan(zg: torch.Tensor, zv: torch.Tensor, zd: torch.Tensor) -> torch.Tensor:
    """``h`` (B, S, D) float32 for the pre-activations ``g, v, d`` (B, S, D) of a MinGRU layer."""
    zg, zv, zd = zg.contiguous(), zv.contiguous(), zd.contiguous()
    B, S, D = zg.shape
    out = torch.empty(B, S, D, device=zg.device, dtype=torch.float32)
    _mingru_scan_fwd[_grid(B, D)](zg, zv, zd, out, S, D, T=T_CHUNK, C=C_BLOCK, num_warps=WARPS)
    return out


@mingru_scan.register_fake
def _(zg, zv, zd):
    return zg.new_empty(zg.shape, dtype=torch.float32)


@torch.library.custom_op("cubby::mingru_scan_bwd", mutates_args=())
def mingru_scan_bwd(zg: torch.Tensor, zv: torch.Tensor, zd: torch.Tensor, out: torch.Tensor,
                    dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zg, zv, zd, out, dout = (t.contiguous() for t in (zg, zv, zd, out, dout))
    B, S, D = zg.shape
    dg, dv, dd = torch.empty_like(zg), torch.empty_like(zv), torch.empty_like(zd)
    _mingru_scan_bwd[_grid(B, D)](zg, zv, zd, out, dout, dg, dv, dd, S, D, T=T_CHUNK, C=C_BLOCK, num_warps=WARPS)
    return dg, dv, dd


@mingru_scan_bwd.register_fake
def _(zg, zv, zd, out, dout):
    return torch.empty_like(zg), torch.empty_like(zv), torch.empty_like(zd)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output)


def _backward(ctx, dout):
    zg, zv, zd, out = ctx.saved_tensors
    return mingru_scan_bwd(zg, zv, zd, out, dout)


mingru_scan.register_autograd(_backward, setup_context=_setup)


__wiring__ = Wiring.WIRED
