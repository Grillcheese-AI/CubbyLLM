"""DirectML smoke test + micro-benchmark for the assembled CubbyModel.

Run inside the .venv-dml environment (torch 2.4.1 + torch-directml). Builds a
small REAL ``CubbyModel``, moves it to the GPU via ``resolve_device('dml')`` +
``move_model``, and runs a few train steps on random tokens — proving every op
in the forward/backward path (embedding, MinGRU scan via the sequential DML
fallback, hardened memory generator, retrieval head) actually runs on the
DirectML device. Times DML vs CPU so we can see the speedup on real shapes.

Each stage is guarded so an unsupported op reports exactly where it broke,
rather than a bare stack trace.

  (.venv-dml) python validation/bench_dml.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\CubbyLLM")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.device import describe, is_directml, move_model, resolve_device  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402

# Shapes overridable via env, e.g.  DML_D=512 DML_L=6 DML_B=32 DML_S=128
def _env(k, d):
    return int(os.environ.get(k, d))


D, N_LAYERS, CTX, N_SLOTS, VOCAB = _env("DML_D", 128), _env("DML_L", 3), 32, 8, _env("DML_V", 4096)
SEQ, BATCH, STEPS = _env("DML_S", 64), _env("DML_B", 16), _env("DML_STEPS", 40)


def build(vocab):
    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=D, n_layers=N_LAYERS, ctx_dim=CTX, vocab_core=vocab)
    router = FrozenSlotRouter(input_dim=D, n_slots=N_SLOTS, ctx_dim=CTX).freeze()
    from cubbyllm.model.backbone import MinGRUBackbone
    return CubbyModel(
        config=cfg, context_source=router,
        backbone=MinGRUBackbone(D, N_LAYERS),
        memory=MemoryLayer(HyperGenerator(ctx_dim=CTX, n_out=D * D), SnapshotHardener(), d_model=D),
        binding=BindingHead(), embedding=HybridEmbedding(vocab, D),
        head=TopKRetrievalHead(torch.randn(vocab, D) * 0.02, learnable=True),
        retrieval_k=vocab,
    )


def bench(device_str: str):
    dev = resolve_device(device_str)
    label = describe(dev)
    print(f"\n=== {device_str} -> {label} (directml={is_directml(dev)}) ===")
    try:
        model = build(VOCAB)
        move_model(model, dev)
        params = [p for p in model.parameters()]
        opt = torch.optim.Adam(params, lr=3e-3)
        g = torch.Generator().manual_seed(1)
        step_ms, last = [], None
        for s in range(STEPS):
            toks = torch.randint(0, VOCAB, (BATCH, SEQ + 1), generator=g).to(dev)
            x, y = toks[:, :-1], toks[:, 1:]
            t = time.perf_counter()
            logits = model.forward(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if hasattr(torch, "directml"):
                pass
            step_ms.append((time.perf_counter() - t) * 1000)
            last = float(loss)
        warm = step_ms[5:]                       # drop warmup
        med = sorted(warm)[len(warm) // 2]
        print(f"  OK — {STEPS} steps, final loss {last:.3f}, median {med:.0f} ms/step")
        return med
    except Exception:
        print("  FAILED on this device:")
        traceback.print_exc()
        return None


def _guard_shape():
    """Refuse GPU-crashing shapes. DirectML on consumer RDNA2 TDR-hangs the whole
    GPU under large sustained allocations (the memory generator emits ~D*D per
    step). Require an explicit opt-in for anything big."""
    gen_mb = (D * D * 4) / 1e6                    # memory generator output tensor
    act_mb = (BATCH * SEQ * D * 4) / 1e6
    big = D > 256 or BATCH * SEQ > 8192 or gen_mb > 512
    if big and os.environ.get("DML_ALLOW_LARGE") != "1":
        print(f"\n[GUARD] shape d={D} L={N_LAYERS} B={BATCH} S={SEQ} is large "
              f"(gen ~{gen_mb:.0f}MB, act ~{act_mb:.0f}MB).")
        print("  DirectML on this card can TDR-crash the whole GPU at this size.")
        print("  Re-run with DML_ALLOW_LARGE=1 only if you accept that risk;")
        print("  prefer native-Linux ROCm or cloud for real large-model runs.")
        raise SystemExit(1)


def main():
    print(f"torch {torch.__version__}")
    _guard_shape()
    try:
        import torch_directml
        print(f"torch_directml devices={torch_directml.device_count()} "
              f"({torch_directml.device_name(0)})")
    except Exception:
        print("torch_directml NOT installed (CPU-only run)")
    cpu = bench("cpu")
    dml = bench("dml")
    if cpu and dml:
        print(f"\n[result] DirectML {cpu/dml:.2f}x vs CPU "
              f"(CPU {cpu:.0f} ms/step, DML {dml:.0f} ms/step) at "
              f"d={D}, L={N_LAYERS}, B={BATCH}, S={SEQ}")
        print("  Note: the win grows with d_model/batch — this is a tiny shape.")


if __name__ == "__main__":
    main()
