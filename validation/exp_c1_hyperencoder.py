"""H-C1 — generated token embeddings vs a static lookup table, head to head.

The claim (ZeTT/zip2zip-shaped): a hyper-encoder that GENERATES a token's
embedding from its surface form can replace the static (V x d) embedding table
— the enabling mechanism for a much larger vocab, since its parameter count is
independent of V. H-C1's validation clause asks for exactly two checks at a
scale where both are cheap:

  (a) quality — are generated embeddings competitive with a learned static
      table at MATCHED vocab, matched training budget, on a real LM-ish task?
  (b) cost    — what does generating cost per token vs a table lookup?

Task: next-token prediction over the REAL cubby-lm tokenizer's output on the
REAL valid.txt corpus (trigram context -> next token), so token identity and
frequency statistics are real, not synthetic. Model: tiny MLP over the
concatenated context embeddings, weight-tied output over the same embeddings
(so the embedding mechanism is exercised on BOTH sides, like a real LM head).

  static  — nn.Embedding(V, d): V*d learned params, the thing that scales as
            O(V) and motivates the whole group.
  hyper   — embedding = f(token's SentencePiece surface string): a char-CNN
            (char embed 16 -> conv d -> max-pool -> MLP d). Parameter count is
            INDEPENDENT of V. All-token embedding matrix is re-generated every
            step during training (the honest cost), cached once for eval.

Both models see identical batches, steps, optimizer, lr. CPU, minutes.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sentencepiece as spm  # noqa: E402

SPM_PATH = (r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\tokenizers"
            r"\spm32k_1p7b\grillcheese_spm32k_v2.model")
CORPUS = r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\trunk_torch\data\valid.txt"

D_EMB = 64
CTX = 3
STEPS = 4000
BATCH = 256
LR = 3e-3
V_KEEP = 8000          # most-frequent tokens kept; rest -> UNK (keeps CPU cheap)
MAX_CHARS = 12


def load_stream(n_bytes=4_000_000):
    sp = spm.SentencePieceProcessor(); sp.Load(SPM_PATH)
    with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(n_bytes)
    ids = sp.EncodeAsIds(text)
    # keep the top tokens (up to V_KEEP-1), remap, everything else -> 0 (UNK)
    uniq, counts = np.unique(ids, return_counts=True)
    keep = uniq[np.argsort(-counts)][: V_KEEP - 1]     # may be fewer than asked
    remap = np.zeros(sp.GetPieceSize(), dtype=np.int64)
    remap[keep] = np.arange(1, len(keep) + 1)
    stream = remap[np.asarray(ids)]
    # surface forms for the hyper-encoder, index-aligned with the remapped ids
    pieces = ["<unk>"] + [sp.IdToPiece(int(t)) for t in keep]
    return stream, pieces


def char_tensorize(pieces):
    """(V, MAX_CHARS) int64 of char codes (byte values of the piece string)."""
    out = np.zeros((len(pieces), MAX_CHARS), dtype=np.int64)
    for i, p in enumerate(pieces):
        bs = p.encode("utf-8")[:MAX_CHARS]
        out[i, : len(bs)] = np.frombuffer(bs, dtype=np.uint8)
    return torch.from_numpy(out)


class HyperEncoder(nn.Module):
    """Surface-form -> embedding. Params independent of V (the whole point)."""
    def __init__(self, d=D_EMB):
        super().__init__()
        self.char_emb = nn.Embedding(256, 16)
        self.conv = nn.Conv1d(16, d, kernel_size=3, padding=1)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(d, d))

    def forward(self, chars):                      # (V, MAX_CHARS)
        h = self.char_emb(chars).transpose(1, 2)   # (V, 16, L)
        h = self.conv(h).max(dim=2).values         # (V, d)
        return self.out(h)


class TinyLM(nn.Module):
    """Trigram MLP LM with weight-tied output over the embedding matrix."""
    def __init__(self, mode, vocab, chars=None):
        super().__init__()
        self.mode = mode
        if mode == "static":
            self.emb = nn.Embedding(vocab, D_EMB)
        else:
            self.hyper = HyperEncoder()
            self.register_buffer("chars", chars)
        self.mlp = nn.Sequential(
            nn.Linear(CTX * D_EMB, 256), nn.Tanh(), nn.Linear(256, D_EMB))

    def emb_matrix(self):
        if self.mode == "static":
            return self.emb.weight
        return self.hyper(self.chars)              # regenerated — honest cost

    def forward(self, x):                          # x: (B, CTX)
        E = self.emb_matrix()                      # (V, d)
        h = E[x].reshape(x.shape[0], -1)           # (B, CTX*d)
        q = self.mlp(h)                            # (B, d)
        return q @ E.t()                           # tied logits (B, V)


def run(model, stream, steps, rng, tag):
    n = len(stream) - CTX - 1
    split = int(n * 0.95)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for s in range(steps):
        idx = rng.integers(0, split, size=BATCH)
        x = np.stack([stream[i:i + CTX] for i in idx])
        y = stream[idx + CTX]
        logits = model(torch.from_numpy(x))
        loss = F.cross_entropy(logits, torch.from_numpy(y))
        opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0
    # eval on held-out tail
    model.eval()
    with torch.no_grad():
        E = model.emb_matrix()                     # cache once (deploy mode)
        losses = []
        for s in range(split, n - BATCH, BATCH * 7):
            idx = np.arange(s, s + BATCH)
            x = np.stack([stream[i:i + CTX] for i in idx])
            y = stream[idx + CTX]
            h = E[torch.from_numpy(x)].reshape(BATCH, -1)
            logits = model.mlp(h) @ E.t()
            losses.append(float(F.cross_entropy(logits, torch.from_numpy(y))))
    model.train()
    val = float(np.mean(losses))
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  {tag:<8} val loss {val:.3f}  ppl {np.exp(val):7.1f}"
          f"  params {nparam:>9,}  train {train_time:5.1f}s")
    return val, nparam, train_time


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("=" * 78)
    print(f"H-C1  Generated vs static token embeddings"
          f"  (real spm32k stream, V={V_KEEP}, d={D_EMB}, {STEPS} steps)")
    print("=" * 78)
    stream, pieces = load_stream()
    chars = char_tensorize(pieces)
    print(f"  corpus stream: {len(stream):,} tokens; UNK rate"
          f" {(stream == 0).mean()*100:.1f}%\n")

    v_s, p_s, _ = run(TinyLM("static", V_KEEP), stream, STEPS, rng, "static")
    v_h, p_h, _ = run(TinyLM("hyper", V_KEEP, chars), stream, STEPS, rng, "hyper")

    # deploy-time generation cost: embedding table regen vs lookup
    lm = TinyLM("hyper", V_KEEP, chars)
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            lm.hyper(chars)
    t_gen = (time.perf_counter() - t0) / 10
    print(f"\n  hyper-encoder full-table regen: {t_gen*1000:.1f} ms for"
          f" {V_KEEP:,} tokens ({t_gen/V_KEEP*1e6:.2f} us/token, one-off/"
          f"cacheable at inference)")

    gap = v_h - v_s
    print("\n[verdict]")
    print(f"  quality gap (hyper - static val loss): {gap:+.3f} nats"
          f"  ({'competitive' if gap < 0.15 else 'NOT competitive'} at matched"
          f" budget)")
    print(f"  param scaling: static grows {V_KEEP*D_EMB:,}+ with V;"
          f" hyper's embedding params are V-independent"
          f" ({p_h - (p_s - V_KEEP*D_EMB):+,} vs the table's {V_KEEP*D_EMB:,}).")
    print("  caveat: toy scale (V=8k, d=64, trigram LM). The H-C1 kill"
          " criterion needs re-checking at the real target vocab/model size"
          " before betting the vocab strategy on it.")


if __name__ == "__main__":
    main()
