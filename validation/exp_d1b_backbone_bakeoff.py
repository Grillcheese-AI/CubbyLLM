"""H-D1 (resolved edition) — backbone bake-off INSIDE the real CubbyModel.

exp_d1_backbone.py was a standalone char micro-benchmark with a stylized BDH.
This is the real thing: each candidate backbone is plugged into the ACTUAL
assembled ``cubbyllm.CubbyModel`` (frozen router -> hybrid embedding -> backbone
-> hardened theta=f(c) memory -> learnable retrieval head) and trained on the
real subword TinyStories setup that produced 1.32 bpc, under a matched budget.
So the numbers reflect each backbone in the architecture it would actually sit
in, not in isolation.

Four candidates, identical trunk scaffold (RMSNorm -> mixer -> +res, RMSNorm ->
SwiGLU -> +res), only the mixer differs; every non-backbone component is built
from the SAME per-component seed so the backbone is the only variable:

  mingru  cubby-lm's MinGRU recurrence (log-domain parallel scan)
  bdh     attention-free linear recurrence, Q=K + per-channel decay (BDH class)
  gru     classic gated recurrence (torch nn.GRU) — the strong old baseline
  attn    causal multi-head softmax attention — the reference

Reports held-out perplexity + char-normalized bpc + params + ms/step. The winner
is a candidate to PROMOTE into cubbyllm/model/backbone/ (replacing the reference
stand-in) — a decision surfaced to the user after the run, not taken here.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\CubbyLLM")
import sentencepiece as spm  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402
from cubbyllm.training import InMemoryDataPipeline, TrainLoop  # noqa: E402

STORIES = r"C:\Users\grill\Documents\GitHub\cubby-lm\tinystory_50k.json"
SPM = (r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\tokenizers"
       r"\spm32k_1p7b\grillcheese_spm32k_v2.model")
D, N_LAYERS, CTX, N_SLOTS = 128, 2, 32, 8
SEQ, BATCH, STEPS, LR = 64, 16, 600, 3e-3
N_STORIES, EVAL_EVERY = 8000, 200


# ── trunk scaffold + mixers (nn.Modules; the winner promotes cleanly) ───────
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        h = d * mult
        self.g, self.u, self.o = nn.Linear(d, h, False), nn.Linear(d, h, False), nn.Linear(h, d, False)

    def forward(self, x):
        return self.o(F.silu(self.g(x)) * self.u(x))


def _scan(x_scan, a):
    log_a = a.clamp(min=1e-8).log()
    a_star = log_a.cumsum(1)
    eps = 1e-30
    hp = (a_star + ((x_scan.clamp(min=0) + eps).log() - a_star).logcumsumexp(1)).exp()
    hn = (a_star + (((-x_scan).clamp(min=0) + eps).log() - a_star).logcumsumexp(1)).exp()
    return hp - hn


class MinGRUMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g, self.v, self.d = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return _scan(torch.sigmoid(self.g(x)) * torch.tanh(self.v(x)),
                     0.001 + 0.998 * torch.sigmoid(self.d(x)))


class BDHMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.kq, self.v, self.o = nn.Linear(d, d, False), nn.Linear(d, d, False), nn.Linear(d, d, False)
        self.log_decay = nn.Parameter(torch.zeros(1) - 2.0)

    def forward(self, x):
        B, S, d = x.shape
        kq = F.elu(self.kq(x)) + 1.0                       # nonneg, Q=K
        v = self.v(x)
        scores = torch.einsum("bsd,btd->bst", kq, kq) / math.sqrt(d)
        pos = torch.arange(S)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
        mask = (pos[:, None] >= pos[None, :]).float()
        weight = mask * (torch.sigmoid(self.log_decay) ** dist)
        return self.o(torch.einsum("bst,btd->bsd", scores * weight, v) / (S ** 0.5))


class GRUMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x):
        return self.gru(x)[0]


class AttnMixer(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.qkv, self.o = nn.Linear(d, 3 * d, False), nn.Linear(d, d, False)

    def forward(self, x):
        B, S, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, S, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, S, d))


MIXERS = {"mingru": MinGRUMixer, "bdh": BDHMixer, "gru": GRUMixer, "attn": AttnMixer}


class Trunk(nn.Module):
    """A Backbone: N layers of norm->mixer->res, norm->ffn->res over (B,S,d)."""

    def __init__(self, d, n_layers, mixer_cls):
        super().__init__()
        self.n1 = nn.ModuleList([RMSNorm(d) for _ in range(n_layers)])
        self.mix = nn.ModuleList([mixer_cls(d) for _ in range(n_layers)])
        self.n2 = nn.ModuleList([RMSNorm(d) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([SwiGLU(d) for _ in range(n_layers)])

    def forward(self, x):
        for n1, m, n2, f in zip(self.n1, self.mix, self.n2, self.ffn):
            x = x + m(n1(x))
            x = x + f(n2(x))
        return x


def build_model(vocab, mixer_name):
    # per-component seeds so ONLY the backbone differs across candidates
    torch.manual_seed(5); router = FrozenSlotRouter(D, N_SLOTS, CTX).freeze()
    torch.manual_seed(3); gen = HyperGenerator(CTX, D * D)
    memory = MemoryLayer(gen, SnapshotHardener(), d_model=D)
    torch.manual_seed(1); emb = HybridEmbedding(vocab, D)
    torch.manual_seed(4); head = TopKRetrievalHead(torch.randn(vocab, D) * 0.02, learnable=True)
    torch.manual_seed(2); backbone = Trunk(D, N_LAYERS, MIXERS[mixer_name])
    cfg = CubbyConfig(d_model=D, n_layers=N_LAYERS, ctx_dim=CTX, vocab_core=vocab)
    return CubbyModel(config=cfg, context_source=router, backbone=backbone,
                      memory=memory, binding=BindingHead(), embedding=emb,
                      head=head, retrieval_k=vocab)


@torch.no_grad()
def eval_ce(model, val, vocab, n_batches=20):
    losses, rng = [], np.random.default_rng(123)
    for _ in range(n_batches):
        ix = rng.integers(0, len(val) - SEQ - 1, size=BATCH)
        x = torch.from_numpy(np.stack([val[i:i + SEQ] for i in ix]))
        y = torch.from_numpy(np.stack([val[i + 1:i + SEQ + 1] for i in ix]))
        losses.append(float(F.cross_entropy(model.forward(x).reshape(-1, vocab), y.reshape(-1))))
    return float(np.mean(losses))


def run(name, train, val, vocab, cpt):
    model = build_model(vocab, name)
    params = sum(p.numel() for p in model.parameters())
    bb_params = sum(p.numel() for p in model.backbone.parameters())
    pipe = InMemoryDataPipeline(train, manifest=f"tinystory/spm32k/{name}", seed=0)
    loop = TrainLoop(model, pipe, SnapshotHardener(), lr=LR, batch_size=BATCH, seq_len=SEQ)
    step_ms = []
    for s in range(1, STEPS + 1):
        t = time.perf_counter(); loop.step(); step_ms.append(time.perf_counter() - t)
    ce = eval_ce(model, val, vocab, n_batches=30)
    ppl, bpc = math.exp(ce), ce / math.log(2) / cpt
    ms = float(np.median(step_ms) * 1000)
    print(f"  {name:<7} ppl {ppl:>7.1f}  bpc {bpc:>6.3f}  backbone-params {bb_params:>8,}"
          f"  total {params:>10,}  {ms:>6.0f} ms/step")
    return {"name": name, "ppl": ppl, "bpc": bpc, "ms": ms, "bb_params": bb_params}


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"H-D1  Backbone bake-off in the real CubbyModel  (subword spm32k, d={D},"
          f" L={N_LAYERS}, {STEPS} steps)")
    print("=" * 78)
    sp = spm.SentencePieceProcessor(); sp.Load(SPM)
    vocab = sp.GetPieceSize()
    stories = json.load(open(STORIES, encoding="utf-8"))[:N_STORIES]
    text = "\n".join(stories)
    data = np.array(sp.EncodeAsIds(text), dtype=np.int64)
    cpt = len(text) / len(data)
    split = int(len(data) * 0.95)
    train, val = data[:split], data[split:]
    print(f"  {N_STORIES} stories -> {len(data):,} tokens, {cpt:.2f} chars/token\n")

    results = [run(name, train, val, vocab, cpt) for name in MIXERS]

    best = min(results, key=lambda r: r["ppl"])
    fastest = min(results, key=lambda r: r["ms"])
    print(f"\n[verdict]")
    print(f"  best quality: {best['name']} (ppl {best['ppl']:.1f}, bpc {best['bpc']:.3f})")
    print(f"  fastest:      {fastest['name']} ({fastest['ms']:.0f} ms/step)")
    print("  -> promote the quality winner into cubbyllm/model/backbone/ (replacing")
    print("     the reference stand-in) unless the speed gap changes the call.")
    print(f"\n(wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
