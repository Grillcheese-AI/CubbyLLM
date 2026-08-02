"""H-B6 transfer probe: does MAP-trained bindability carry over to BlockCodeOps?

The 2026-07-30 live 2B run trained the trunk with a **MAP** binding aux loss
(bipolar elementwise product) because MAP is differentiable. But the DECIDED
production algebra for the VM/world-model channel is **BlockCodeOps** (H-B5) —
a different VSA family (sparse block codes, per-block circular convolution).
Nothing so far shows the two transfer. This script answers that, offline, with
no training: load the trained checkpoint, pull real trunk features h, and test
role/filler bundle-then-unbind recovery under BOTH algebras.

The load-bearing comparison is against an UNTRAINED control. "BlockCode recovery
= 0.8" means nothing on its own; what matters is trained-vs-random-init on
identical inputs. Four cells:

              |  MAP (what was trained)  |  BlockCode (what production uses)
  trained h   |  should ~reproduce the   |  THE QUESTION
              |  run's cos ~0.84         |
  random h    |  crosstalk baseline      |  crosstalk baseline

Read the RIGHT column against the BOTTOM row. If trained ~= random under
BlockCode, the aux loss bought nothing for the production algebra and H-B6's
open item (2) resolves negative — the aux loss would need reformulating in the
block-code basis (or the VM channel needs a projection trained for it).

DO NOT COMPARE ABSOLUTE COSINES ACROSS THE TWO ALGEBRAS. Measured on random
h (dry run, 2026-08-01): MAP floor at n=16 is 0.237 (~= the theoretical
1/sqrt(16) = 0.25 crosstalk floor), but BlockCode's floor is **0.648** — the
per-block softmax makes every filler a non-negative distribution, so any two
are positively correlated by construction and cosine cannot approach 0. The
BlockCode column is inflated by that alone. Only the trained-minus-untrained
DELTA and the retrieval accuracy carry information there.

Usage (env knobs mirror train_colab.py; vocab/D/L are read from the ckpt meta,
so no tokenizer is needed):

  CB_CKPT=/content/drive/MyDrive/cubbyllm/ckpt.pt \
  CB_CORPUS=/content/token_cache CB_SOURCES=.../corpus_sources.json \
  python validation/exp_b6_blockcode_transfer.py

Without CB_CORPUS it falls back to RANDOM token ids and says so loudly — h from
random tokens is off-distribution, so treat that mode as a smoke test only.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.device import move_model  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402
from cubbyllm.ops import BlockCodeVSA  # noqa: E402


def _env(k, d, cast=int):
    return cast(os.environ.get(k, d))


CKPT = os.environ.get("CB_CKPT", "")
CORPUS = os.environ.get("CB_CORPUS", "")
SOURCES = os.environ.get("CB_SOURCES", "")
SPM = os.environ.get("CUBBY_SPM", "")
CTX = _env("CB_CTX", 32); N_SLOTS = _env("CB_SLOTS", 8)
BATCH = _env("CB_B", 4); SEQ = _env("CB_S", 256)
N_BATCH = _env("CB_NBATCH", 8)               # how many batches of h to pool
BLOCK_L = _env("CB_BLOCK_L", 128)            # block length; k = D // l
SEED = _env("CB_SEED", 0)
N_PAIRS = [int(x) for x in os.environ.get("CB_NPAIRS", "4,8,16,32").split(",")]


# ── model (mirrors train_colab.build; standalone by validation/ convention) ──
def build(vocab, d, layers, dev):
    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=d, n_layers=layers, ctx_dim=CTX, vocab_core=vocab)
    model = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=d, n_slots=N_SLOTS, ctx_dim=CTX).freeze(),
        backbone=MinGRUBackbone(d, layers, grad_checkpoint=False),
        memory=MemoryLayer(HyperGenerator(ctx_dim=CTX, n_out=d * d), SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(vocab, d),
        head=TopKRetrievalHead(torch.randn(vocab, d) * 0.02, learnable=True),
        retrieval_k=vocab)
    move_model(model, dev)
    return model


def load_params(path, model, dev):
    ck = torch.load(path, map_location=dev)
    ps = list(model.parameters())
    if len(ck["params"]) != len(ps):
        raise SystemExit(f"param count mismatch: ckpt {len(ck['params'])} vs model {len(ps)}")
    with torch.no_grad():
        for p, s in zip(ps, ck["params"]):
            p.copy_(s.to(p.device))
    return ck


# ── feature extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def collect_h(model, batches, dev):
    """Pool trunk features h over several batches -> (N, d) float32 numpy."""
    out = []
    for x in batches:
        h = model.features(x.to(dev))          # (B, S, d)
        out.append(h.reshape(-1, h.shape[-1]).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def token_batches(vocab, dev):
    """Real corpus windows if CB_CORPUS is set, else random ids (smoke only)."""
    if CORPUS:
        from cubbyllm.training import WeightedCorpusPipeline
        if SOURCES:
            srcs = json.load(open(SOURCES, encoding="utf-8"))["sources"]
        else:
            srcs = [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                    for p in sorted(glob.glob(os.path.join(CORPUS, "*.u32")))]
        pipe = WeightedCorpusPipeline(srcs, SPM, CORPUS, seed=SEED, cache_only=True).prepare()
        it = pipe.batches(BATCH, SEQ)
        return [next(it)[0] for _ in range(N_BATCH)], "real corpus"
    g = torch.Generator().manual_seed(SEED)
    return ([torch.randint(0, vocab, (BATCH, SEQ), generator=g) for _ in range(N_BATCH)],
            "RANDOM ids (off-distribution — smoke test only)")


# ── the two algebras ────────────────────────────────────────────────────────
def map_probe(h, n, rng):
    """MAP (what the aux loss trained): bipolar roles, elementwise product.

    Bundle n (role * filler) pairs, unbind each, report cosine recovery and
    whether the recovered vector retrieves its own filler among the n bundled.
    """
    d = h.shape[-1]
    roles = (rng.integers(0, 2, size=(n, d)).astype(np.float32) * 2.0 - 1.0)
    idx = rng.integers(0, h.shape[0], size=n)
    fillers = h[idx]                                    # (n, d)
    composite = (roles * fillers).sum(axis=0)           # bundle
    recovered = composite[None, :] * roles              # unbind (self-inverse)
    return _score(recovered, fillers)


def blockcode_probe(h, n, vsa, rng):
    """BlockCodeOps (production algebra): per-block circular convolution.

    h is dense/real, block codes are per-block distributions — so the trunk
    features are reshaped to (k, l) and softmaxed per block (the continuous
    relaxation the facade documents, and the form H-B5's 3-way prototype found
    trainable). Roles are DISCRETE atoms with orthogonal=False: the ops
    docstring is explicit that the structured/orthogonal set is for resonator
    factorization and produces systematic crosstalk on role/filler duty.
    """
    k, l = vsa.k, vsa.l
    idx = rng.integers(0, h.shape[0], size=n)
    f = h[idx].reshape(n, k, l)
    f = f - f.max(axis=-1, keepdims=True)
    f = np.exp(f); f /= f.sum(axis=-1, keepdims=True)   # per-block distribution
    roles = vsa.codebook(n, orthogonal=False)           # (n, k, l) discrete atoms
    composite = vsa.bundle([vsa.bind(roles[i], f[i]) for i in range(n)])
    recovered = np.stack([vsa.unbind(composite, roles[i]) for i in range(n)])
    return _score(recovered.reshape(n, -1), f.reshape(n, -1))


def _score(recovered, fillers):
    """Cosine recovery + retrieval accuracy against the bundled candidate set."""
    r = recovered / (np.linalg.norm(recovered, axis=-1, keepdims=True) + 1e-12)
    f = fillers / (np.linalg.norm(fillers, axis=-1, keepdims=True) + 1e-12)
    cos = float((r * f).sum(axis=-1).mean())
    sim = r @ f.T                                       # (n, n) cross-similarity
    acc = float((sim.argmax(axis=1) == np.arange(len(r))).mean())
    return cos, acc


def main():
    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("set CB_CKPT to the trained checkpoint (see module docstring)")

    meta = torch.load(CKPT, map_location="cpu")["meta"]  # D / L / vocab live here
    d, layers, vocab = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    k = d // BLOCK_L
    if k * BLOCK_L != d:
        raise SystemExit(f"CB_BLOCK_L={BLOCK_L} does not divide d_model={d}")

    print(f"H-B6 transfer probe — MAP (trained) vs BlockCodeOps (production)")
    print(f"  ckpt   {CKPT}")
    print(f"  model  d={d} L={layers} vocab={vocab} | device {dev} | torch {torch.__version__}")
    vsa = BlockCodeVSA(k=k, l=BLOCK_L)
    print(f"  blocks k={k} x l={BLOCK_L} (=d) | VSA backend: {vsa.backend()}")

    batches, src = token_batches(vocab, dev)
    print(f"  tokens {BATCH}x{SEQ} x{N_BATCH} batches from {src}")

    print("\n  building trained model ...", flush=True)
    trained = build(vocab, d, layers, dev)
    load_params(CKPT, trained, dev)
    h_tr = collect_h(trained, batches, dev)
    del trained
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # CONTROL: identical architecture, never trained. build() seeds torch to 0,
    # so this is the exact init the run started from — the honest "did training
    # do anything" reference for both algebras.
    print("  building untrained control ...", flush=True)
    control = build(vocab, d, layers, dev)
    h_ct = collect_h(control, batches, dev)
    del control
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  h pooled: trained {h_tr.shape}, control {h_ct.shape}")
    print("\n  n_pairs | algebra    | trained cos / acc | untrained cos / acc | delta cos")
    print("  --------+------------+-------------------+---------------------+----------")
    results = {}
    for n in N_PAIRS:
        for name, fn in (("MAP", map_probe), ("BlockCode", blockcode_probe)):
            rng_a = np.random.default_rng(SEED)          # same roles/positions
            rng_b = np.random.default_rng(SEED)          # for both h sets
            if name == "MAP":
                c_tr, a_tr = fn(h_tr, n, rng_a)
                c_ct, a_ct = fn(h_ct, n, rng_b)
            else:
                c_tr, a_tr = fn(h_tr, n, vsa, rng_a)
                c_ct, a_ct = fn(h_ct, n, vsa, rng_b)
            results[(n, name)] = (c_tr, a_tr, c_ct, a_ct)
            print(f"  {n:>7} | {name:<10} |     {c_tr:.3f} / {a_tr:5.1%} |"
                  f"      {c_ct:.3f} / {a_ct:5.1%} |   {c_tr - c_ct:+.3f}")

    # ── verdict ─────────────────────────────────────────────────────────────
    n16 = 16 if 16 in N_PAIRS else N_PAIRS[len(N_PAIRS) // 2]
    m_tr, _, m_ct, _ = results[(n16, "MAP")]
    b_tr, b_acc, b_ct, _ = results[(n16, "BlockCode")]
    print(f"\n  NOTE: absolute cosines are NOT comparable across algebras — on random h"
          f" the MAP floor is ~0.24 but BlockCode's is ~0.65 (per-block softmax makes"
          f" all fillers non-negative, hence positively correlated). Read the DELTA.")
    print(f"  SANITY (n={n16}): MAP on trained h = cos {m_tr:.3f}. The run's final"
          f" bind loss was ~0.158 => cos ~0.842; a probe far from that means this"
          f" script is not measuring what the loss measured.")
    print(f"  MAP gain from training:       {m_tr - m_ct:+.3f} cos")
    print(f"  BlockCode gain from training: {b_tr - b_ct:+.3f} cos (retrieval {b_acc:.1%})")
    if b_tr - b_ct > 0.05:
        print("  => TRANSFERS: MAP-trained h is measurably more BlockCode-bindable"
              " than untrained. H-B6 open item (2) resolves POSITIVE.")
    elif b_tr - b_ct < 0.01:
        print("  => DOES NOT TRANSFER: BlockCode recovery is no better than an"
              " untrained trunk. H-B6 open item (2) resolves NEGATIVE — the aux"
              " loss shapes h for MAP only; the production channel needs its own"
              " objective (or a trained projection into the block-code basis).")
    else:
        print("  => WEAK/AMBIGUOUS transfer. Report the number, do not round it up.")
    print(f"\n  wall {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
