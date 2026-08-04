# Running CubbyLLM on the local GPU via DirectML

Target machine: Windows 11 + **AMD Radeon RX 6750 XT** (RDNA2, 12 GB). There is
no CUDA on AMD, and ROCm is Linux-only, so torch reaches this GPU through
**DirectML** (`torch-directml`), which exposes a device of torch type
`privateuseone`.

## The one hard constraint

`torch-directml` (latest `0.2.5.dev240914`, Sep 2024) **hard-pins
`torch==2.4.1`** (and `torchvision==0.19.1`). The repo's main env is torch
**2.10** CPU. So DirectML lives in a **separate venv**, and the main test env is
left alone. The `cubbyllm` package is `dependencies = []` and imports torch
lazily, so it runs unchanged in either env.

## Install (one-time)

```powershell
# from the repo root
py -3.10 -m venv .venv-dml          # torch-directml wheels: cp39–cp312
.\.venv-dml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch-directml==0.2.5.dev240914   # pulls torch==2.4.1
python -m pip install numpy sentencepiece pytest
```

`torch-directml` depends on a recent AMD Adrenalin / DirectX 12 driver (already
present on this box: driver 32.0.21043.x).

## Use it in code

```python
from cubbyllm.core.device import resolve_device, move_model, describe

device = resolve_device("auto")     # DirectML > CUDA > CPU
model  = build_model(...)            # assembled CubbyModel
move_model(model, device)            # moves embedding/backbone/memory/head/router
print(describe(device))              # -> "DirectML (AMD Radeon RX 6750 XT)"
# then create input tensors with .to(device) as usual
```

- `resolve_device("dml")` forces DirectML and raises if it's missing.
- `is_directml(device)` is true for a DirectML device.
- The **grilly/numpy VSA** (binding) stays on CPU by design — it is fixed,
  non-differentiable algebra, and grilly has its own Vulkan tier for the GPU
  side. Only the torch parts (embedding, backbone, memory, head) move to DML.

## The backbone scan on DirectML

The MinGRU backbone's default recurrence is a log-domain **parallel** scan using
`logcumsumexp`, which DirectML's op set does **not** cover. `mingru.py` detects a
`privateuseone` device and uses an **exact sequential recurrence** instead
(mul/add only, verified identical to the parallel scan within fp32 noise). It is
the only device-specific branch in the model and changes nothing on CPU/CUDA.
Trade-off: O(S) sequential steps instead of O(log S) parallel depth — fine at
training sequence lengths, revisit if long-context throughput matters.

## Verify

```powershell
.\.venv-dml\Scripts\Activate.ps1
python validation\bench_dml.py
```

Builds a small real `CubbyModel`, runs train steps on CPU and on the GPU, and
prints per-step ms + the DirectML/CPU speedup. Each stage is guarded, so if some
op is unsupported on DirectML it reports exactly where.

## Verdict (measured 2026-07-25, RX 6750 XT)

DirectML **works but is not a viable training accelerator on this card**:

- Correctness is fine — the full `CubbyModel` forward/backward runs on the GPU
  and gives identical loss to CPU (the `move_model` recursive mover + the
  sequential-scan fallback were both needed to get there).
- At the toy shape (d=128, L=3, B=16, S=64) it was **~2x slower than CPU**
  (66 ms CPU vs 137 ms DML) — kernel-launch overhead + the O(S) sequential scan
  + Adam's `lerp` falling back to CPU.
- At a **larger shape it TDR-crashed the whole GPU** (driver hang + display
  reset; recovered, no damage). Consumer RDNA2 + DirectML is unstable under
  sustained/large compute.

So: keep DirectML only for small correctness checks, and gate large shapes
(`bench_dml.py` refuses them without `DML_ALLOW_LARGE=1`). For real larger-model
training on this hardware the realistic paths are **native-Linux dual-boot +
ROCm** (`HSA_OVERRIDE_GFX_VERSION=10.3.0`, gfx1031 unofficial) or **cloud GPU**.
ROCm-on-WSL2 is blocked at the driver level (no WSL ROCm runtime for gfx1031).

## Known DirectML limitations to watch

- **Incomplete op coverage.** Unsupported ops either error or silently fall back
  to CPU (a hidden slowdown). `bench_dml.py` surfaces hard failures; watch for
  ops that "work" but are secretly on CPU.
- **No float64.** Keep everything float32 (the package already does).
- **Slower than native CUDA/ROCm** for the same silicon — DirectML is a
  compatibility layer, not a peak-performance path. For maximum throughput on
  this card the alternative is WSL2 + ROCm with
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` (RX 6750 XT is gfx1031, unofficial), which is
  more setup but faster; DirectML is the low-friction Windows option.
