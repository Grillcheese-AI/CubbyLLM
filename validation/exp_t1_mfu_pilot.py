"""exp_t1_mfu_pilot — measured training MFU for the trunk, before any 2B rental.

The panel-settled decision chain (docs/research/2026-08-model-panel-agenda.md,
3/3 convergent) starts here: "the 2-hour MFU pilot at 150M = single highest-EV
action before any GPU rental". Recurrence without fused kernels was quoted at
10-35% MFU vs 35-50% for a transformer — a penalty that can eat the hybrid's
FLOP advantage. This script replaces those quotes with measurements and prints
the projected A100-hours + $ for the real 14.37B-token cycle-one run.

Arms (each x {eager, torch.compile}):
  hybrid  -- HybridBackbone at attn_every=3, window=MFU_W  (the cycle-one plan)
  mingru  -- pure MinGRUBackbone                (isolates the recurrence penalty)
  attn    -- HybridBackbone at attn_every=1, window=S  (= full-causal transformer
             comparator; decides transformer-vs-hybrid before renting)

A step is a REAL training step: forward + backward + AdamW on random data.
MFU_V=0 (default) benches the trunk alone on (B,S,D) inputs — the trunk is what
the kernel/ratio decision is about. MFU_V>0 adds embedding + tied linear head +
cross-entropy (the O(V*D) matmul shape of the full-vocab cosine-CE head).

MFU convention: model FLOPs (3x forward — 6*N_matmul per token + the attention
quadratic term) / device peak. Grad-checkpointing recomputes a forward but adds
no MODEL FLOPs, so it lowers reported MFU — that is the standard convention and
the honest one for cost projection. Peaks are dense bf16/fp16 TFLOPS from a
small name-matched table; override with MFU_PEAK_TFLOPS for unlisted devices.
On CPU (or unknown GPUs without an override) tokens/sec is reported and MFU is
marked n/a — CPU numbers smoke-test the script, they do not inform the decision.

Usage:
  # local smoke (CPU, ~1 min):
  MFU_D=256 MFU_L=6 MFU_B=4 MFU_S=256 MFU_STEPS=6 python validation/exp_t1_mfu_pilot.py --tag _smoke
  # Colab / RunPod decision run (150M-class defaults):
  python validation/exp_t1_mfu_pilot.py --tag _a100
  # 2B-shape probe (needs grad-ckpt on most cards):
  MFU_D=2048 MFU_L=32 MFU_B=4 MFU_S=1024 MFU_CKPT=1 python validation/exp_t1_mfu_pilot.py --tag _a100_2b

Standalone; never imported by cubbyllm/. Log: tee to validation/logs/; JSON is
written by the script itself.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.model.backbone.hybrid import HybridBackbone  # noqa: E402
from cubbyllm.model.backbone.mingru import MinGRUBackbone  # noqa: E402


def _env(k: str, d: int) -> int:
    return int(os.environ.get(k, d))


D = _env("MFU_D", 1024)
L = _env("MFU_L", 16)
B = _env("MFU_B", 16)
S = _env("MFU_S", 1024)
W = _env("MFU_W", 512)
HEADS = _env("MFU_HEADS", 8)
V = _env("MFU_V", 0)             # 0 = trunk-only; >0 adds embed + head + CE
STEPS = _env("MFU_STEPS", 30)
WARMUP = _env("MFU_WARMUP", 5)
CKPT = bool(_env("MFU_CKPT", 0))
COMPILE = bool(_env("MFU_COMPILE", 1))   # compile arms attempted (skipped on failure)
TOKENS_2B = 14.37e9              # the built pretraining cache (memory: uint32 shards, D:)

# dense bf16/fp16 tensor-core peaks, TFLOPS (substring-matched on device name)
PEAKS = {
    "A100": 312.0, "H100": 989.0, "H200": 989.0, "B200": 2250.0,
    "L40S": 181.0, "L4": 121.0,
    "A40": 74.8, "A10": 62.5, "T4": 65.0, "V100": 125.0,
    "PRO 6000": 252.0,           # RTX PRO 6000 Blackwell (docs/COLAB.md card)
    "4090": 165.2, "3090": 71.0, "5090": 209.5,
}


def device_peak_tflops(name: str) -> float | None:
    if os.environ.get("MFU_PEAK_TFLOPS"):
        return float(os.environ["MFU_PEAK_TFLOPS"])
    for key, tf in PEAKS.items():
        if key in name:
            return tf
    return None


class TrunkLM(nn.Module):
    """Optional full-stack wrapper: embed -> trunk -> tied linear head."""

    def __init__(self, trunk: nn.Module, vocab: int, d: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.trunk = trunk
        # tied head: the V x D matmul is the same shape/cost as the full-vocab
        # cosine-CE readout the runbook plans, which is what matters for MFU
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x):
        return self.head(self.trunk(self.emb(x)))


def build_arm(arm: str) -> nn.Module:
    if arm == "hybrid":
        trunk = HybridBackbone(D, L, attn_every=3, window=W, heads=HEADS,
                               grad_checkpoint=CKPT)
    elif arm == "mingru":
        trunk = MinGRUBackbone(D, L)
    elif arm == "attn":
        trunk = HybridBackbone(D, L, attn_every=1, window=S, heads=HEADS,
                               grad_checkpoint=CKPT)
    else:
        raise ValueError(arm)
    return TrunkLM(trunk, V, D) if V else trunk


def attn_flops_per_token_fwd(model: nn.Module) -> float:
    """Attention-matmul FLOPs per token, forward (QK^T + AV = 4*D per attended
    key), averaged over positions with the causal+window mask."""
    trunk = model.trunk if V else model
    if not isinstance(trunk, HybridBackbone):
        return 0.0
    win = trunk.window
    avg_keys = sum(min(i + 1, win) for i in range(S)) / S
    return trunk.n_attn_layers * 4.0 * D * avg_keys


def model_flops_per_token_train(model: nn.Module) -> tuple[float, int]:
    """(FLOPs/token for fwd+bwd, matmul param count). 6*N + 3*attention-term;
    the embedding table is a lookup, not a matmul, so it is excluded from N
    (the tied head matmul is counted via the same tensor once)."""
    # parameters() deduplicates the tied embed/head tensor, so the V x D weight
    # is counted once — as the head matmul; the embedding lookup is free
    n_matmul = sum(p.numel() for p in model.parameters())
    return 6.0 * n_matmul + 3.0 * attn_flops_per_token_fwd(model), n_matmul


def make_batch(device: torch.device):
    if V:
        x = torch.randint(0, V, (B, S), device=device)
        y = torch.randint(0, V, (B, S), device=device)
        return x, y
    return torch.randn(B, S, D, device=device), None


def loss_of(model: nn.Module, x, y):
    out = model(x)
    if V:
        return F.cross_entropy(out.view(-1, V), y.view(-1))
    return (out ** 2).mean()   # trivial target, real backward through the trunk


def bench(model: nn.Module, device: torch.device, use_amp: bool) -> dict:
    model = model.to(device).train()
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device.type == "cuda"))
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y = make_batch(device)
    times = []
    for step in range(WARMUP + STEPS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = loss_of(model, x, y)
        else:
            loss = loss_of(model, x, y)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= WARMUP:
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"ms_per_step": med * 1e3,
            "ms_p10": sorted(times)[max(0, int(0.1 * len(times)) - 1)] * 1e3,
            "ms_p90": sorted(times)[int(0.9 * len(times))] * 1e3,
            "tok_per_s": B * S / med,
            "final_loss": float(loss.detach().float().cpu()),
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                            if device.type == "cuda" else None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="suffix for the json log")
    ap.add_argument("--arms", default="hybrid,mingru,attn")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    peak = device_peak_tflops(dev_name)   # name table is CUDA-only; the env
    # override works on any device (e.g. a DirectML bench with a known peak)
    use_amp = device.type == "cuda"
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT, text=True).strip()
    except Exception:
        rev = "unknown"

    print(f"python {platform.python_version()} | torch {torch.__version__} | git {rev}")
    print(f"device {dev_name} | peak {'%.0f TF' % peak if peak else 'UNKNOWN (set MFU_PEAK_TFLOPS)'}"
          f" | amp={'bf16' if use_amp else 'off'} | ckpt={int(CKPT)}")
    print(f"shapes D={D} L={L} B={B} S={S} W={W} heads={HEADS} V={V or 'trunk-only'}"
          f" | steps {STEPS} (+{WARMUP} warmup)\n")

    out: dict = {"config": {"D": D, "L": L, "B": B, "S": S, "W": W, "heads": HEADS,
                            "V": V, "steps": STEPS, "warmup": WARMUP, "ckpt": CKPT},
                 "env": {"python": platform.python_version(),
                         "torch": torch.__version__, "git": rev,
                         "device": dev_name, "peak_tflops": peak,
                         "amp": "bf16" if use_amp else "off"},
                 "results": {}}

    rows = []
    for arm in args.arms.split(","):
        for mode in (["eager", "compile"] if COMPILE else ["eager"]):
            torch.manual_seed(0)
            model = build_arm(arm)
            fpt, n_params = model_flops_per_token_train(model)
            if mode == "compile":
                try:
                    model = torch.compile(copy.deepcopy(model))
                except Exception as e:  # no triton / no cl.exe etc.
                    print(f"  {arm}/{mode}: torch.compile unavailable ({type(e).__name__}) — skipped")
                    continue
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                r = bench(model, device, use_amp)
            except Exception as e:
                print(f"  {arm}/{mode}: FAILED ({type(e).__name__}: {e})")
                out["results"][f"{arm}/{mode}"] = {"error": f"{type(e).__name__}: {e}"}
                continue
            achieved_tf = r["tok_per_s"] * fpt / 1e12
            r.update({"params": n_params, "flops_per_token_train": fpt,
                      "achieved_tflops": achieved_tf,
                      "mfu": (achieved_tf / peak) if peak else None})
            out["results"][f"{arm}/{mode}"] = r
            rows.append((arm, mode, r))
            mfu_s = f"{100 * r['mfu']:.1f}%" if r["mfu"] is not None else "n/a"
            mem_s = f" | mem {r['peak_mem_gb']:.1f}G" if r["peak_mem_gb"] else ""
            print(f"  {arm:7s}/{mode:7s} {r['ms_per_step']:9.1f} ms/step "
                  f"{r['tok_per_s']:10.0f} tok/s | {achieved_tf:6.1f} TF | MFU {mfu_s}"
                  f" | {n_params / 1e6:.1f}M params{mem_s}")

    # projection: cycle-one cost at the 2B shape, using each arm's best measured
    # MFU applied to a menu of rental cards. MFU is a fraction of each card's own
    # peak, so it transfers only APPROXIMATELY across architectures — the row
    # matching the pilot card is the measured one, the others are estimates.
    RENTALS = [  # (name, dense-bf16 TFLOPS, $/hr lo-hi) — re-verify prices at rental time
        ("A100-80G", 312.0, 1.2, 1.9),           # cheapest $/hr, 80G
        ("RTX PRO 6000 Blackwell 96G", 252.0, 1.8, 2.3),  # faster in practice, 96G, pricier
        ("H100-80G", 989.0, 2.0, 3.0),
        ("H200-141G", 989.0, 2.3, 3.5),          # H100 compute + 141G HBM3e / 4.8TB/s
        ("B200-180G", 2250.0, 4.5, 6.5),         # Blackwell DC: ~2.3x H100 peak, 180G
    ]
    if peak and rows:
        print("\n=== projection -- 14.37B tokens at the 2B shape (D2048/L32/V131k)")
        print("    (pilot-card row = measured; other cards = MFU-transfer estimate)")
        d2, l2, v2 = 2048, 32, 131072
        out["projection"] = {}
        # rough 2B analytic: params scale ~ (d2/D)^2 * (l2/L) on the trunk + head
        for arm in dict.fromkeys(r[0] for r in rows):
            best = max((r for r in rows if r[0] == arm), key=lambda r: r[2]["tok_per_s"])
            n2 = best[2]["params"] * (d2 / D) ** 2 * (l2 / L) + (v2 * d2 if not V else 0)
            fpt2 = 6.0 * n2 + (3.0 * attn_flops_per_token_fwd(build_arm(arm))
                               * (d2 / D) * (l2 / L) if arm != "mingru" else 0.0)
            mfu = best[2]["mfu"]
            print(f"  {arm} (best={best[1]}, MFU {100 * mfu:.1f}%):")
            out["projection"][arm] = {"mfu": mfu, "flops_per_token_2b": fpt2, "cards": {}}
            for name, ptf, lo, hi in RENTALS:
                hours = TOKENS_2B * fpt2 / (mfu * ptf * 1e12) / 3600
                out["projection"][arm]["cards"][name] = {
                    "hours": hours, "usd_lo": lo * hours, "usd_hi": hi * hours}
                print(f"    {name:28s} ~{hours:7,.0f} hrs ({hours / 24:5.1f} d) "
                      f"-> ${lo * hours:,.0f}-{hi * hours:,.0f}")

    log = ROOT / "validation" / "logs" / f"exp_t1_mfu_pilot{args.tag}.json"
    log.parent.mkdir(parents=True, exist_ok=True)   # fresh Colab unzip has no logs/
    log.write_text(json.dumps(out, indent=2))
    print(f"\njson -> {log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
