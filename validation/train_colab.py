"""GPU training run for the assembled CubbyModel — built for Colab + CUDA.

Runs the REAL stack (frozen router -> hybrid embedding -> MinGRU backbone ->
hardened theta=f(c) memory -> retrieval head) through the REAL device-aware
``TrainLoop`` (hardener penalty included) on the GPU. On CUDA the backbone uses
the fast O(log T) parallel scan automatically. Shapes and data paths are env-
configurable so the same file scales from a laptop CPU to a 96 GB card.

  # Colab (after uploading the package + data — see docs/COLAB.md):
  CUBBY_SPM=/content/grillcheese_spm32k_v2.model \
  CUBBY_STORIES=/content/tinystory_50k.json \
  CB_D=512 CB_L=8 CB_B=64 CB_S=256 CB_STEPS=1000 \
  python validation/train_colab.py

Env knobs (all optional): CB_D CB_L CB_CTX CB_SLOTS CB_B CB_S CB_STEPS CB_LR
CB_EVAL CB_STORIES_N ; data: CUBBY_SPM CUBBY_STORIES ; force device: CB_DEVICE.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.device import describe, move_model, resolve_device  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402
from cubbyllm.training import InMemoryDataPipeline, TrainLoop  # noqa: E402


def _env(k, d, cast=int):
    return cast(os.environ.get(k, d))


D = _env("CB_D", 512)
N_LAYERS = _env("CB_L", 8)
CTX = _env("CB_CTX", 32)
N_SLOTS = _env("CB_SLOTS", 8)
BATCH = _env("CB_B", 64)
SEQ = _env("CB_S", 256)
STEPS = _env("CB_STEPS", 1000)
LR = _env("CB_LR", 3e-3, float)
EVAL_EVERY = _env("CB_EVAL", 100)
STORIES_N = _env("CB_STORIES_N", 20000)
SPM = os.environ.get("CUBBY_SPM", r"C:\Users\grill\Documents\GitHub\cubby-lm"
                     r"\cubby\tokenizers\spm32k_1p7b\grillcheese_spm32k_v2.model")
STORIES = os.environ.get("CUBBY_STORIES",
                         r"C:\Users\grill\Documents\GitHub\cubby-lm\tinystory_50k.json")


def check_gpu(dev):
    print(f"device: {describe(dev)}")
    if dev.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"  {torch.cuda.get_device_name()} | compute capability {cap[0]}.{cap[1]}"
              f" | {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
        if cap[0] >= 12:  # Blackwell sm_120
            print("  (Blackwell sm_120 — needs a cu128 torch build; if you see "
                  "'no kernel image is available', install torch for CUDA 12.8.)")
        print(f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}")


def build(vocab, dev):
    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=D, n_layers=N_LAYERS, ctx_dim=CTX, vocab_core=vocab)
    model = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=D, n_slots=N_SLOTS, ctx_dim=CTX).freeze(),
        backbone=MinGRUBackbone(D, N_LAYERS),
        memory=MemoryLayer(HyperGenerator(ctx_dim=CTX, n_out=D * D), SnapshotHardener(), d_model=D),
        binding=BindingHead(), embedding=HybridEmbedding(vocab, D),
        head=TopKRetrievalHead(torch.randn(vocab, D) * 0.02, learnable=True),
        retrieval_k=vocab,
    )
    move_model(model, dev)                       # BEFORE the optimizer is built
    return model


@torch.no_grad()
def eval_ce(model, val, vocab, dev, n_batches=12):
    losses, rng = [], np.random.default_rng(123)
    for _ in range(n_batches):
        ix = rng.integers(0, len(val) - SEQ - 1, size=min(BATCH, 32))
        x = torch.from_numpy(np.stack([val[i:i + SEQ] for i in ix])).to(dev)
        y = torch.from_numpy(np.stack([val[i + 1:i + SEQ + 1] for i in ix])).to(dev)
        losses.append(float(F.cross_entropy(model.forward(x).reshape(-1, vocab), y.reshape(-1))))
    return float(np.mean(losses))


def main():
    import sentencepiece as spm

    dev = resolve_device(os.environ.get("CB_DEVICE", "auto"))
    check_gpu(dev)
    sp = spm.SentencePieceProcessor(); sp.Load(SPM)
    vocab = sp.GetPieceSize()
    text = "\n".join(json.load(open(STORIES, encoding="utf-8"))[:STORIES_N])
    data = np.array(sp.EncodeAsIds(text), dtype=np.int64)
    cpt = len(text) / len(data)
    split = int(len(data) * 0.95)
    train, val = data[:split], data[split:]
    print(f"config: d={D} L={N_LAYERS} B={BATCH} S={SEQ} steps={STEPS} | "
          f"{len(data):,} tokens, {cpt:.2f} chars/tok, vocab {vocab}")

    model = build(vocab, dev)
    n_params = sum(p.numel() for p in model.parameters())
    pipe = InMemoryDataPipeline(train, manifest=f"tinystory/{STORIES_N}/spm32k", seed=0)
    loop = TrainLoop(model, pipe, SnapshotHardener(), lr=LR,
                     batch_size=BATCH, seq_len=SEQ, device=dev)
    print(f"trainable params: {n_params:,} | manifest {pipe.manifest_hash()[:16]}\n")

    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); toks = 0
    for s in range(1, STEPS + 1):
        loss = loop.step(); toks += BATCH * SEQ
        if s % EVAL_EVERY == 0 or s == 1:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ce = eval_ce(model, val, vocab, dev)
            print(f"  step {s:>5}  loss {loss:6.3f}  ppl {math.exp(ce):7.1f}  "
                  f"bpc {ce/math.log(2)/cpt:5.3f}  {toks/dt:>9,.0f} tok/s")
    if dev.type == "cuda":
        torch.cuda.synchronize()
        print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"wall {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
