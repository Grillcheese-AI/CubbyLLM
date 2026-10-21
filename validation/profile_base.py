"""profile_base -- where a training step's time goes, by kernel and by kind, on the card at hand.

The bench (``train_base.py`` ``CB_MODE=bench``) says how fast; this says why. It builds the model
``train_base`` would build from the same env (shape, memory reads, spatial link, recompute, micro,
S, compile), runs a few warm steps on random tokens -- with random places on a third of them when
the link is on, so its FFTs and the route loss are in the step -- then profiles ``CB_PROF_STEPS``
forward+backward+optimizer steps and prints:

  the step: tok/s and MFU as train_base computes them;
  by kind: the share of device time in matmuls, attention, the vocabulary's softmax / cross
           entropy, the MinGRU scan, the FFTs, the optimizer, copies and casts, other fused work;
  the top kernels by device time, names shortened.

That ranking is the order the speed work should go in (2026-10-20: the 450M + link ran at 54.6k
tok/s, 29.5% MFU on the RTX PRO 6000; the question is which of micro, the 128k-vocab loss, the
attention without flex, the scan or the GEMM size holds it there).

Only the device's own events are summed (2026-10-20, fixed): the first version summed every row of
``key_averages``, so each kernel counted twice -- as itself and inside the op that launched it
(``aten::mm`` = its cutlass GEMMs) -- the optimizer three times, and "Command Buffer Full" (the
CPU waiting on a full launch queue: the GPU is the bottleneck, which is what we want) as GPU work;
the attention kernels, named ``fmha_cutlass*``, were filed under matmul. The GEMM line now also
says how fast the GEMMs themselves run: their FLOPs (``flops_per_token`` less attention's) over
their device time, against the card's peak.

    <train_base env: CB_D, CB_L, CB_S, CB_MICRO, CB_SPATIAL, ...> [CB_PROF_STEPS=4] [CB_PROF_TRACE=<dir>]
    python validation/profile_base.py
"""
from __future__ import annotations

import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import train_base as tb  # noqa: E402

STEPS = int(os.environ.get("CB_PROF_STEPS", "4"))
TRACE = os.environ.get("CB_PROF_TRACE", "")
KINDS = [                                                      # first match wins: attention before matmul
    ("attention", r"fmha|flash|attention|sdpa"),                 # (fmha_cutlass* would match "cutlass")
    ("MinGRU scan", r"logcumsumexp|cumsum|scan|flip"),           # the scan and its reversed backward
    ("vocab softmax / CE", r"log_softmax|nll_loss|cross_entropy"),
    ("FFT (spatial link)", r"fft|polar"),
    ("optimizer", r"adam|multi_tensor"),
    ("matmul", r"gemm|cutlass|cublas|nvjet|xmma|_mm_|_mm$|addmm|bmm|matmul|_scaled_mm"),
    ("copy / cast", r"copy_kernel|direct_copy|memcpy|memset|catarraybatched|fused__to_copy_\d+$|fused_clone_\d+$"),
]
NOT_WORK = ("Command Buffer Full",)                              # CUPTI markers, not kernels


def batch(meta, dev, n: int, vocab: int):
    x = torch.randint(0, vocab, (n, tb.SEQ), device=dev)
    y = torch.randint(0, vocab, (n, tb.SEQ), device=dev)
    if not meta.get("spatial"):
        return x, y
    from cubbyllm.model.spatial import NO_PLACE
    pos = torch.randint(4096, 5120, (n, tb.SEQ, 3), device=dev)
    keep = torch.rand(n, tb.SEQ, device=dev) < 0.33
    pos = torch.where(keep.unsqueeze(-1), pos, torch.full_like(pos, NO_PLACE))
    return tb.Tokens(x, pos), y


def kind_of(name: str) -> str:
    low = name.lower()
    for k, rx in KINDS:
        if re.search(rx, low):
            return k
    return "other fused / elementwise"


def main() -> None:
    dev = tb.device()
    _, _, v, vp = tb.vocab_sizes()
    meta = tb.arch_meta(v, vp)
    model = tb.build_model(meta, dev, tb.SEED)
    opt = tb.make_opt(model, 1e-4, dev)
    loss_fn, _ = tb.make_fns(model, vp, dev)
    fpt, pk = tb.flops_per_token(model, meta, tb.SEQ), tb.peak_flops(dev)
    x, y = batch(meta, dev, tb.MICRO, v)
    print(f"profile: D{tb.D}/L{tb.L} S{tb.SEQ} micro {tb.MICRO} mem_every {meta.get('mem_every', 0)} "
          f"spatial {bool(meta.get('spatial'))} route {tb.ROUTE_LOSS} recompute {tb.GRAD_CKPT} compile {tb.COMPILE} "
          f"no_flex {os.environ.get('CB_NO_FLEX') == '1'} banded {os.environ.get('CB_BANDED') == '1'} | {dev}", flush=True)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for _ in range(3):                                            # compile and warm
        loss_fn(x, y).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    sync()
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    t0 = time.perf_counter()
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(STEPS):
            loss_fn(x, y).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        sync()
    dt = (time.perf_counter() - t0) / STEPS
    tps = tb.MICRO * tb.SEQ / dt
    print(f"step (micro {tb.MICRO} x S {tb.SEQ}, with the optimizer each micro): {1e3 * dt:.1f} ms, {tps:,.0f} tok/s, "
          f"MFU {100 * tps * fpt / pk:.1f}% (under the profiler: a few % slower than a run)")
    use_dev = dev.type == "cuda"
    rows, waits = [], 0.0
    for e in prof.key_averages():
        on_dev = "CPU" not in str(getattr(e, "device_type", "CPU"))
        t = (getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)) if use_dev else e.self_cpu_time_total
        if e.key in NOT_WORK:
            waits += t
        elif t > 0 and (on_dev if use_dev else not e.key.startswith(("ProfilerStep", "aten::empty"))):
            rows.append((t, e.key, e.count))
    total = sum(t for t, _, _ in rows) or 1.0
    by_kind: dict = {}
    for t, k, _ in rows:
        by_kind[kind_of(k)] = by_kind.get(kind_of(k), 0.0) + t
    print(f"\nby kind ({'device kernels' if use_dev else 'CPU'} time, {total / 1e3 / STEPS:.1f} ms a step"
          f"{f'; the CPU waited {waits / 1e3 / STEPS:.0f} ms a step on a full launch queue: GPU-bound' if waits else ''}):")
    for k, t in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {100 * t / total:5.1f}%   {t / 1e3 / STEPS:8.1f} ms")
    n_attn = sum(1 for i in range(meta["L"]) if i % meta["attn_every"] == 0)
    w_avg = sum(min(t + 1, meta["window"]) for t in range(tb.SEQ)) / tb.SEQ
    gemm_fpt = fpt - 12.0 * meta["D"] * w_avg * n_attn
    t_mm = by_kind.get("matmul", 0.0) / 1e6 / STEPS
    if use_dev and t_mm > 0:
        tf = gemm_fpt * tb.MICRO * tb.SEQ / t_mm / 1e12
        print(f"\nGEMMs: {gemm_fpt * tb.MICRO * tb.SEQ / 1e12:.1f} TFLOP a step in {1e3 * t_mm:.0f} ms = {tf:.0f} TFLOPS, "
              f"{100 * tf * 1e12 / pk:.0f}% of peak; the vocabulary head is {100 * 6 * meta['vocab'] * meta['D'] / gemm_fpt:.0f}% "
              f"of the GEMM FLOPs. MFU = GEMM speed x GEMM share of the step.")
    print("\ntop kernels:")
    for t, k, n in sorted(rows, reverse=True)[:25]:
        print(f"  {100 * t / total:5.1f}%  {t / 1e3 / STEPS:7.2f} ms  x{n // STEPS:<4d} {kind_of(k)[:10]:10s}  {k[:110]}")
    if TRACE:
        os.makedirs(TRACE, exist_ok=True)
        path = os.path.join(TRACE, f"trace_D{tb.D}_L{tb.L}_S{tb.SEQ}_m{tb.MICRO}.json")
        prof.export_chrome_trace(path)
        print(f"\nchrome trace: {path}")


if __name__ == "__main__":
    main()
