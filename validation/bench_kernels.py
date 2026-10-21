"""bench_kernels -- the Triton kernels against what they replace, one op at a time, forward + backward.

At the 450M's shapes on the card at hand (bf16, as under autocast; ``CB_B`` rows of ``CB_S``
tokens): the MinGRU recurrence -- the torch path (gates + log-domain scan) against the fused
kernel -- and the windowed attention of one layer -- the dense-mask SDPA the run falls back to,
the banded flash kernels, and flex attention compiled with its default tiles and with tiles for
a 99 KB card. Each: the time of a forward + backward (CUDA events, after warm-up), its error
against the reference in the output and in one input gradient, or the error that stopped it.
``profile_base`` / ``speed_check`` measure the whole step; this says which op moved and by how much.

    [CB_B=32] [CB_S=1024] [CB_D=1024] [CB_HEADS=4] [CB_WINDOW=512] python validation/bench_kernels.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

B, S = int(os.environ.get("CB_B", "32")), int(os.environ.get("CB_S", "1024"))
D, H, W = int(os.environ.get("CB_D", "1024")), int(os.environ.get("CB_HEADS", "4")), int(os.environ.get("CB_WINDOW", "512"))
FLEX_SMALL = {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 1, "num_warps": 8,
              "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32}


def timed(fn, inputs, reps: int = 10) -> float:
    for _ in range(3):
        out = fn(*inputs)
        torch.autograd.grad(out.float().sum(), inputs)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        out = fn(*inputs)
        torch.autograd.grad(out.float().sum(), inputs)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def compare(name: str, fn, ref, inputs) -> dict:
    try:
        t0 = time.time()
        ms = timed(fn, inputs)
        o, r = fn(*inputs).float(), ref(*inputs).float()
        g = torch.autograd.grad((fn(*inputs).float() * 1.0).sum(), inputs[0])[0].float()
        gr = torch.autograd.grad(ref(*inputs).float().sum(), inputs[0])[0].float()
        res = {"ms": round(ms, 3), "out_err": float((o - r).abs().max() / r.abs().max()),
               "grad_err": float((g - gr).abs().max() / gr.abs().max().clamp_min(1e-12)), "wall_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 -- a variant that fails is a result, not a crash
        res = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    print(f"  {name:26s} " + (f"{res['ms']:8.2f} ms   out err {res['out_err']:.1e}   grad err {res['grad_err']:.1e}"
                              if "ms" in res else res["error"]), flush=True)
    return res


def main() -> None:
    assert torch.cuda.is_available(), "a GPU benchmark"
    dev, bf = "cuda", torch.bfloat16
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} (sm_{props.major}{props.minor}, {getattr(props, 'shared_memory_per_block_optin', 0) // 1024} KB "
          f"shared a block) | torch {torch.__version__} | B {B} S {S} D {D} heads {H} window {W}", flush=True)
    rep = {"device": props.name, "shape": dict(B=B, S=S, D=D, heads=H, window=W), "mingru": {}, "attention": {}}

    from cubbyllm.model.backbone.mingru import _log_domain_scan
    from cubbyllm.model.backbone.mingru_triton import mingru_scan
    g = [torch.randn(B, S, D, device=dev, dtype=bf).requires_grad_() for _ in range(3)]

    def torch_scan(zg, zv, zd):
        with torch.autocast("cuda", dtype=bf):
            return _log_domain_scan(torch.sigmoid(zg) * torch.tanh(zv), 0.001 + 0.998 * torch.sigmoid(zd))
    print("MinGRU, one layer, forward + backward:")
    rep["mingru"]["torch"] = compare("torch (log-domain scan)", torch_scan, torch_scan, g)
    rep["mingru"]["triton"] = compare("fused Triton kernel", mingru_scan, torch_scan, g)
    rep["mingru"]["torch_compiled"] = compare("torch, compiled", torch.compile(torch_scan), torch_scan, g)

    from cubbyllm.model.backbone.banded import dense_attention
    from cubbyllm.model.backbone.window_attn_triton import window_attention
    qkv = [torch.randn(B, H, S, D // H, device=dev, dtype=bf).requires_grad_() for _ in range(3)]
    dense = lambda q, k, v: dense_attention(q, k, v, W)          # noqa: E731
    print("windowed attention, one layer, forward + backward:")
    rep["attention"]["dense"] = compare("dense mask (SDPA)", dense, dense, qkv)
    rep["attention"]["band"] = compare("banded flash (Triton)", lambda q, k, v: window_attention(q, k, v, W), dense, qkv)
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        bm = create_block_mask(lambda b, h, qi, ki: (qi >= ki) & (qi - ki < W), None, None, S, S, device=dev)
        for name, opts in (("flex, default tiles", None), ("flex, 99 KB tiles", FLEX_SMALL)):
            f = torch.compile(lambda q, k, v, o=opts: flex_attention(q, k, v, block_mask=bm, kernel_options=o))
            rep["attention"][name] = compare(name, f, dense, qkv)
    except ImportError as e:
        rep["attention"]["flex"] = {"error": str(e)}
    out = os.environ.get("CB_OUT", "")
    if out:
        json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
