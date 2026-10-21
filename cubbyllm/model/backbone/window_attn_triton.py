"""Sliding-window causal attention as Triton flash kernels: only the band (opt-in: ``CB_ATTN_TRITON=1``).

Where flex attention is not used -- a card whose shared memory per block (99 KB on the RTX
PRO 6000 Blackwell and the L4) is under flex's tile for a head of 256 -- the windowed layers
fall back to ``dense_attention``: an S x S boolean mask through SDPA's memory-efficient kernel,
every score computed and most discarded. On the 450M + link (run 2 of the spatial notebook,
2026-10-20) that was 15% of a step's device time for 11 of 32 layers.

Here, flash attention restricted to the band: a block of ``BM`` queries visits only the key
blocks that hold its window (``q - W < key <= q``), with the online softmax in registers;
the backward is FlashAttention-2's split -- one kernel per key block for dK, dV (looping over
the query blocks that see it), one per query block for dQ -- from the forward's log-sum-exp
and ``delta = rowsum(O * dO)``. At S = 1024, W = 512 that is ~0.6 of the causal half's work
and ~0.3 of the dense mask's, with no mask in memory. Tiles are autotuned among configs that
fit 99 KB at a head of 256 (a config that does not fit is skipped by the tuner).

A ``torch.library.custom_op`` (``cubby::window_attention``; torch.compile sees one opaque
op), exact against ``dense_attention`` in values and gradients:
``tests/model/backbone/test_window_attn_triton.py``, under ``TRITON_INTERPRET=1`` on a CPU and
natively on a GPU.

What it brings CubbyLLM: its attention is windowed by design -- the decode state stays a
fixed ring of ``window`` keys -- and now training pays for the window only, on any card,
instead of for the full square on cards flex does not fit.
"""
import math
import os

import torch

from ...core.protocols import Wiring

try:
    import triton
    import triton.language as tl
    _jit = triton.jit
except ImportError:                    # no triton here (Windows, CPU-only wheels): dense_attention stays
    triton = None

    class tl:                          # noqa: N801 -- just enough for the kernels' signatures to be defined
        constexpr = None

    def _jit(fn):
        return fn

AVAILABLE = triton is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _configs(pairs):
    """(BM, BN, warps, stages) -> triton Configs; one small config under the interpreter."""
    if not AVAILABLE:
        return []
    if INTERPRET:
        pairs = pairs[:1]
    return [triton.Config({"BM": m, "BN": n}, num_warps=w, num_stages=s) for m, n, w, s in pairs]


def _tune(pairs):
    return (lambda f: f) if not AVAILABLE else triton.autotune(configs=_configs(pairs), key=["S", "D", "W"])


FWD = [(16, 16, 4, 1), (64, 32, 4, 1), (64, 32, 8, 2), (128, 32, 8, 1), (64, 64, 8, 1)]
DKDV = [(16, 16, 4, 1), (32, 32, 8, 1), (64, 32, 8, 1), (32, 64, 8, 1)]
DQ = [(16, 16, 4, 1), (32, 32, 8, 1), (64, 32, 8, 1), (32, 64, 8, 1)]
if not INTERPRET:                      # the 16 x 16 tile is the interpreter's (small, fast tests)
    FWD, DKDV, DQ = FWD[1:], DKDV[1:], DQ[1:]


@_jit
def _dot(a, b, F32: tl.constexpr):
    """``tl.dot``, IEEE float32 when the inputs are float32 (else tf32 would round them)."""
    if F32:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@_tune(FWD)
@_jit
def _win_attn_fwd(Q, K, V, O, LSE, S, W, scale,
                  D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1).to(tl.int64)
    base = bh * S * D
    rm = m0 + tl.arange(0, BM)
    rd = tl.arange(0, D)
    mq = rm < S
    q = tl.load(Q + base + rm[:, None] * D + rd[None, :], mask=mq[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    lo = (tl.maximum(m0 - W + 1, 0) // BN) * BN
    hi = tl.minimum(m0 + BM, S)
    for n0 in range(lo, hi, BN):
        rn = n0 + tl.arange(0, BN)
        mk = rn < S
        k = tl.load(K + base + rn[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
        v = tl.load(V + base + rn[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
        s = _dot(q, tl.trans(k), F32) * (scale * 1.4426950408889634)
        back = rm[:, None] - rn[None, :]
        s = tl.where((back >= 0) & (back < W) & mk[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_ok = tl.where(m_new == float("-inf"), 0.0, m_new)    # a row with no key yet: nothing to rescale
        p = tl.exp2(s - m_ok[:, None])
        alpha = tl.exp2(m_i - m_ok)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + _dot(p.to(v.dtype), v, F32)
        m_i = m_new
    l_ok = tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(O + base + rm[:, None] * D + rd[None, :], (acc / l_ok[:, None]).to(O.dtype.element_ty), mask=mq[:, None])
    tl.store(LSE + bh * S + rm, m_i + tl.log2(l_ok), mask=mq)            # log2 units


@_tune(DKDV)
@_jit
def _win_attn_bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, S, W, scale,
                       D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr):
    n0 = tl.program_id(0) * BN
    bh = tl.program_id(1).to(tl.int64)
    base = bh * S * D
    rn = n0 + tl.arange(0, BN)
    rd = tl.arange(0, D)
    mk = rn < S
    k = tl.load(K + base + rn[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
    v = tl.load(V + base + rn[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    lo = (n0 // BM) * BM
    hi = tl.minimum(n0 + BN + W - 1, S)                 # the last query that sees this block's last key
    for m0 in range(lo, hi, BM):
        rm = m0 + tl.arange(0, BM)
        mq = rm < S
        q = tl.load(Q + base + rm[:, None] * D + rd[None, :], mask=mq[:, None], other=0.0)
        do = tl.load(DO + base + rm[:, None] * D + rd[None, :], mask=mq[:, None], other=0.0)
        lse = tl.load(LSE + bh * S + rm, mask=mq, other=0.0)
        delta = tl.load(DELTA + bh * S + rm, mask=mq, other=0.0)
        sT = _dot(k, tl.trans(q), F32) * (scale * 1.4426950408889634)       # (BN, BM)
        back = rm[None, :] - rn[:, None]
        keep = (back >= 0) & (back < W) & mq[None, :] & mk[:, None]
        pT = tl.where(keep, tl.exp2(sT - lse[None, :]), 0.0)
        dv += _dot(pT.to(do.dtype), do, F32)
        dpT = _dot(v, tl.trans(do), F32)
        dsT = pT * (dpT - delta[None, :])
        dk += _dot(dsT.to(q.dtype), q, F32)
    tl.store(DK + base + rn[:, None] * D + rd[None, :], (dk * scale).to(DK.dtype.element_ty), mask=mk[:, None])
    tl.store(DV + base + rn[:, None] * D + rd[None, :], dv.to(DV.dtype.element_ty), mask=mk[:, None])


@_tune(DQ)
@_jit
def _win_attn_bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, S, W, scale,
                     D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1).to(tl.int64)
    base = bh * S * D
    rm = m0 + tl.arange(0, BM)
    rd = tl.arange(0, D)
    mq = rm < S
    q = tl.load(Q + base + rm[:, None] * D + rd[None, :], mask=mq[:, None], other=0.0)
    do = tl.load(DO + base + rm[:, None] * D + rd[None, :], mask=mq[:, None], other=0.0)
    lse = tl.load(LSE + bh * S + rm, mask=mq, other=0.0)
    delta = tl.load(DELTA + bh * S + rm, mask=mq, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    lo = (tl.maximum(m0 - W + 1, 0) // BN) * BN
    hi = tl.minimum(m0 + BM, S)
    for n0 in range(lo, hi, BN):
        rn = n0 + tl.arange(0, BN)
        mk = rn < S
        k = tl.load(K + base + rn[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
        v = tl.load(V + base + rn[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
        s = _dot(q, tl.trans(k), F32) * (scale * 1.4426950408889634)
        back = rm[:, None] - rn[None, :]
        keep = (back >= 0) & (back < W) & mk[None, :] & mq[:, None]
        p = tl.where(keep, tl.exp2(s - lse[:, None]), 0.0)
        dp = _dot(do, tl.trans(v), F32)
        ds = p * (dp - delta[:, None])
        dq += _dot(ds.to(k.dtype), k, F32)
    tl.store(DQ + base + rm[:, None] * D + rd[None, :], (dq * scale).to(DQ.dtype.element_ty), mask=mq[:, None])


def _flat(t):
    B, h, S, D = t.shape
    return t.contiguous().view(B * h, S, D)


@torch.library.custom_op("cubby::window_attention_fwd", mutates_args=())
def window_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(O, LSE)``: (B, h, S, dh) each -> the windowed causal attention and its log2-sum-exp (B*h, S)."""
    B, h, S, D = q.shape
    qf, kf, vf = _flat(q), _flat(k), _flat(v)
    o = torch.empty_like(qf)
    lse = torch.empty(B * h, S, device=q.device, dtype=torch.float32)
    grid = lambda m: (triton.cdiv(S, m["BM"]), B * h)  # noqa: E731
    _win_attn_fwd[grid](qf, kf, vf, o, lse, S, int(window), 1.0 / math.sqrt(D), D=D, F32=q.dtype == torch.float32)
    return o.view(B, h, S, D), lse


@window_attention_fwd.register_fake
def _(q, k, v, window):
    B, h, S, D = q.shape
    return torch.empty_like(q), q.new_empty(B * h, S, dtype=torch.float32)


@torch.library.custom_op("cubby::window_attention_bwd", mutates_args=())
def window_attention_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor,
                         do: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, h, S, D = q.shape
    qf, kf, vf, of, dof = _flat(q), _flat(k), _flat(v), _flat(o), _flat(do)
    delta = (of.float() * dof.float()).sum(-1)                           # rowsum(O * dO): (B*h, S)
    dq, dk, dv = torch.empty_like(qf), torch.empty_like(kf), torch.empty_like(vf)
    scale = 1.0 / math.sqrt(D)
    _win_attn_bwd_dkdv[lambda m: (triton.cdiv(S, m["BN"]), B * h)](qf, kf, vf, dof, lse, delta, dk, dv, S, int(window),
                                                                  scale, D=D, F32=q.dtype == torch.float32)
    _win_attn_bwd_dq[lambda m: (triton.cdiv(S, m["BM"]), B * h)](qf, kf, vf, dof, lse, delta, dq, S, int(window),
                                                                scale, D=D, F32=q.dtype == torch.float32)
    return dq.view(B, h, S, D), dk.view(B, h, S, D), dv.view(B, h, S, D)


@window_attention_bwd.register_fake
def _(q, k, v, o, lse, do, window):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, window = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.window = window


def _backward(ctx, do, _dlse):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = window_attention_bwd(q, k, v, o, lse, do, ctx.window)
    return dq, dk, dv, None


window_attention_fwd.register_autograd(_backward, setup_context=_setup)


def window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    """(B, h, S, dh) each -> (B, h, S, dh): ``dense_attention(q, k, v, window)``, the band only."""
    return window_attention_fwd(q, k, v, int(window))[0]


__wiring__ = Wiring.WIRED
