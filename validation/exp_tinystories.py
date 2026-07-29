"""Integration test: the assembled CubbyModel on real TinyStories text.

Not a hypothesis check — a full-stack run of the IMPLEMENTED package
(``cubbyllm``) on real English: context inference (frozen router) -> hybrid
embedding -> MinGRU reference backbone -> hardened theta=f(c) memory ->
retrieval output head, trained through the real ``TrainLoop`` over a
manifest-pinned ``InMemoryDataPipeline``.

SUBWORD variant: uses the real grillcheese_spm32k_v2 SentencePiece tokenizer
(V=32000) with a LEARNABLE output head (a fixed random codebook at d=128 would
drown in crosstalk — H-C7 — so the head learns to separate tokens instead) and
temperature+top-k SAMPLING for generation (greedy fell into repetition loops).
Reports token perplexity + a char-normalized bits-per-char for comparison to
the char-level exp_d1 bake-off, then samples a story.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\CubbyLLM")
import sentencepiece as spm  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402
from cubbyllm.training import InMemoryDataPipeline, TrainLoop  # noqa: E402

STORIES = r"C:\Users\grill\Documents\GitHub\cubby-lm\tinystory_50k.json"
SPM = (r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\tokenizers"
       r"\spm32k_1p7b\grillcheese_spm32k_v2.model")
D_MODEL = 128
N_LAYERS = 3
CTX_DIM = 32
N_SLOTS = 8
SEQ = 64
BATCH = 16
STEPS = 800
LR = 3e-3
N_STORIES = 8000
EVAL_EVERY = 150


def load_token_stream(sp):
    stories = json.load(open(STORIES, encoding="utf-8"))[:N_STORIES]
    text = "\n".join(stories)
    ids = np.array(sp.EncodeAsIds(text), dtype=np.int64)
    chars_per_tok = len(text) / len(ids)
    return ids, chars_per_tok


def build_model(vocab):
    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=D_MODEL, n_layers=N_LAYERS, ctx_dim=CTX_DIM,
                      vocab_core=vocab)
    # single-domain corpus -> the offline router is frozen at init (one context
    # regime); the memory GENERATOR + LEARNABLE HEAD train.
    router = FrozenSlotRouter(input_dim=D_MODEL, n_slots=N_SLOTS,
                              ctx_dim=CTX_DIM).freeze()
    gen = HyperGenerator(ctx_dim=CTX_DIM, n_out=D_MODEL * D_MODEL)
    memory = MemoryLayer(gen, SnapshotHardener(), d_model=D_MODEL)
    head = TopKRetrievalHead(torch.randn(vocab, D_MODEL) * 0.02, learnable=True)
    return CubbyModel(
        config=cfg, context_source=router,
        backbone=MinGRUBackbone(D_MODEL, N_LAYERS),
        memory=memory, binding=BindingHead(),
        embedding=HybridEmbedding(vocab, D_MODEL),
        head=head, retrieval_k=vocab,          # full softmax while training
    )


@torch.no_grad()
def eval_ce(model, val, vocab, n_batches=12):
    losses = []
    rng = np.random.default_rng(123)
    for _ in range(n_batches):
        ix = rng.integers(0, len(val) - SEQ - 1, size=BATCH)
        x = torch.from_numpy(np.stack([val[i:i + SEQ] for i in ix]))
        y = torch.from_numpy(np.stack([val[i + 1:i + SEQ + 1] for i in ix]))
        logits = model.forward(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))))
    return float(np.mean(losses))


@torch.no_grad()
def sample(model, sp, prompt, vocab, n=60, temperature=0.8, top_k=40):
    ids = sp.EncodeAsIds(prompt)
    for _ in range(n):
        x = torch.tensor(ids[-SEQ:], dtype=torch.long).unsqueeze(0)
        logits = model.forward(x)[0, -1] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, vocab))
            logits = torch.where(logits < v[-1], torch.full_like(logits, -1e30), logits)
        probs = F.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(probs, 1)))
    return sp.DecodeIds(ids)


def main():
    t0 = time.time()
    print("=" * 74)
    print("CubbyLLM integration test — TinyStories (SUBWORD spm32k, full stack)")
    print("=" * 74)
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM)
    vocab = sp.GetPieceSize()
    data, cpt = load_token_stream(sp)
    split = int(len(data) * 0.95)
    train, val = data[:split], data[split:]
    print(f"  {N_STORIES} stories -> {len(data):,} tokens (vocab {vocab},"
          f" {cpt:.2f} chars/token)")

    model = build_model(vocab)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_params:,}  (router frozen; head learnable)")

    pipe = InMemoryDataPipeline(train, manifest="tinystory_50k/first8k/spm32k", seed=0)
    loop = TrainLoop(model, pipe, SnapshotHardener(), lr=LR, batch_size=BATCH, seq_len=SEQ)
    print(f"  manifest: {pipe.manifest_hash()[:16]}...\n")

    print(f"  {'step':>6} {'loss':>8} {'ppl':>8} {'bpc':>7}")
    for s in range(1, STEPS + 1):
        loss = loop.step()
        if s % EVAL_EVERY == 0 or s == 1:
            ce = eval_ce(model, val, vocab)
            bpc = ce / math.log(2) / cpt          # bits/token -> bits/char
            print(f"  {s:>6} {loss:>8.3f} {math.exp(ce):>8.1f} {bpc:>7.3f}")

    ce = eval_ce(model, val, vocab, n_batches=30)
    print(f"\n  held-out: perplexity {math.exp(ce):.1f}/token,"
          f" {ce/math.log(2)/cpt:.3f} bits/char"
          f"  (exp_d1 char MinGRU was ~1.45 bpc)")

    print("\n  --- sampled story (prompt: 'Once upon a time', T=0.8 top-k=40) ---")
    for _ in range(1):
        print("  " + sample(model, sp, "Once upon a time", vocab, n=70).replace("\n", " "))
    print(f"\n(wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
