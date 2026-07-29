"""H-D1 — does an attention-free Q=K recurrence learn TEXT competitively?

BDH's headline efficiency numbers come from CIFAR-10/Sudoku, not language.
H-D1's validate clause: "Validate at small scale on a real token-modeling task
before assuming these numbers transfer." This does exactly and only that — a
matched-size, matched-budget char-LM bake-off answering the VIABILITY question
("does it learn language at all, competitively?"), NOT the efficiency claims
(those need scale; H-D2 already handled the asymptotic KV-vs-recurrence compute
trade separately).

Three mixers, identical block scaffold (RMSNorm -> mixer -> +res, RMSNorm ->
SwiGLU -> +res), identical d_model/depth/budget, real English char stream:

  mingru   cubby-lm's ACTUAL MinGRULayer (imported) — the predecessor backbone
  bdh      attention-free linear recurrence with the Q=K constraint (shared
           key/query projection), nonneg feature map + learned per-channel
           decay — a STYLIZED stand-in for the BDH family, labeled as such,
           not a faithful BDH reimplementation
  attn     small causal softmax attention — the reference point everyone knows

Reports val bits-per-char, params, and per-step wall (with the caveat that
T=128 is too short to measure asymptotic cost — wall here is implementation
latency, not the scaling claim). CPU, minutes.
"""
from __future__ import annotations

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-lm")
from cubby.trunk_torch.layers import MinGRULayer  # noqa: E402  (predecessor code)
from cubby.trunk_torch.nn_primitives import RMSNorm  # noqa: E402

CORPUS = r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\trunk_torch\data\valid.txt"
D_MODEL = 128
N_LAYERS = 4
SEQ = 128
BATCH = 32
STEPS = 2500
LR = 3e-3


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        h = d * mult
        self.g = nn.Linear(d, h, bias=False)
        self.u = nn.Linear(d, h, bias=False)
        self.o = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.o(F.silu(self.g(x)) * self.u(x))


class BDHMixer(nn.Module):
    """Attention-free linear recurrence, Q=K (shared projection), per-channel
    learned decay, nonneg feature map. Stylized BDH-family stand-in."""
    def __init__(self, d):
        super().__init__()
        self.kq = nn.Linear(d, d, bias=False)     # shared key AND query (Q=K)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.log_decay = nn.Parameter(torch.zeros(d) - 2.0)  # sigmoid-ish decay

    def forward(self, x):
        B, S, D = x.shape
        kq = F.elu(self.kq(x)) + 1.0              # nonneg feature map (>0)
        v = self.v(x)
        # causal scores with Q=K, decayed by distance; parallel form (T small)
        scores = torch.einsum("bsd,btd->bst", kq, kq) / math.sqrt(D)
        decay = torch.sigmoid(self.log_decay).mean()   # scalar effective decay
        pos = torch.arange(S)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
        mask = (pos[:, None] >= pos[None, :]).float()
        weight = mask * (decay ** dist)
        out = torch.einsum("bst,btd->bsd", scores * weight, v)
        return self.o(out / (S ** 0.5))


class AttnMixer(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, S, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, S, self.h, D // self.h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, S, D))


class MinGRUWrap(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.m = MinGRULayer(d)

    def forward(self, x):
        return self.m(x)


MIXERS = {"mingru": MinGRUWrap, "bdh": BDHMixer, "attn": AttnMixer}


class Block(nn.Module):
    def __init__(self, d, mixer_cls):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.mix = mixer_cls(d)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))


class LM(nn.Module):
    def __init__(self, vocab, mixer_name):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, MIXERS[mixer_name]) for _ in range(N_LAYERS)])
        self.nf = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.emb.weight       # tied

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x))


def load_chars(n_bytes=3_000_000):
    with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(n_bytes)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, len(chars)


def get_batch(data, rng):
    ix = rng.integers(0, len(data) - SEQ - 1, size=BATCH)
    x = np.stack([data[i:i + SEQ] for i in ix])
    y = np.stack([data[i + 1:i + SEQ + 1] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


def run(name, data, vocab, rng):
    torch.manual_seed(0)
    model = LM(vocab, name)
    nparam = sum(p.numel() for p in model.parameters()) - model.emb.weight.numel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(data)
    split = int(n * 0.9)
    train, val = data[:split], data[split:]
    step_times = []
    for s in range(STEPS):
        x, y = get_batch(train, rng)
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        step_times.append(time.perf_counter() - t0)
    # eval
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(40):
            x, y = get_batch(val, rng)
            losses.append(float(F.cross_entropy(
                model(x).reshape(-1, vocab), y.reshape(-1))))
    bpc = float(np.mean(losses)) / math.log(2)
    ms = float(np.median(step_times) * 1000)
    print(f"  {name:<8} val bpc {bpc:.3f}   non-emb params {nparam:>8,}"
          f"   {ms:6.1f} ms/step")
    return bpc, nparam, ms


def main():
    print("=" * 74)
    print(f"H-D1  Backbone viability on real char-LM  (d={D_MODEL}, L={N_LAYERS},"
          f" seq={SEQ}, {STEPS} steps)")
    print("=" * 74)
    data, vocab = load_chars()
    rng = np.random.default_rng(0)
    print(f"  corpus: valid.txt char stream, {len(data):,} chars, vocab {vocab}")
    print(f"  (lower bits-per-char = better; per-step wall is implementation"
          f" latency at T={SEQ}, NOT the asymptotic scaling claim)\n")
    res = {name: run(name, data, vocab, rng) for name in MIXERS}
    print("\n[verdict]")
    best = min(res, key=lambda k: res[k][0])
    mg = res["mingru"][0]
    bd = res["bdh"][0]
    gap = bd - mg
    print(f"  best val bpc: {best} ({res[best][0]:.3f})")
    print(f"  attention-free Q=K (bdh-style) vs MinGRU: {gap:+.3f} bpc"
          f"  ({'competitive' if abs(gap) < 0.1 else 'MinGRU ahead' if gap > 0 else 'bdh ahead'}).")
    print("  VIABILITY only: this says an attention-free Q=K recurrence"
          " does/doesn't learn text at small scale — it does NOT test BDH's"
          " 35x/10x efficiency claims (those need scale + a faithful BDH impl;"
          " the asymptotic compute trade is H-D2). Backbone choice stays the"
          " last, most expensive decision per the doc's own ordering.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n(wall {time.time()-t0:.1f}s)")
