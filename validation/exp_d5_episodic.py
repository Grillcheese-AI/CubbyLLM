"""Falsifiable gate: does a trained per-token MemoryRead recall PAST the window?

Success: with the memory read on, second-half induction accuracy at distance >
window climbs off chance; with it off, it stays at chance. Kill criterion: if
memory-on does not beat memory-off beyond the window, the approach is wrong and we
stop before touching the real trunk. Same spirit as exp_d3_induction.

  CB_STEPS=4000 CB_S=256 CB_WINDOW=32 python validation/exp_d5_episodic.py
"""
from __future__ import annotations
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn as nn, torch.nn.functional as F
from exp_d1b_backbone_bakeoff import RMSNorm, SwiGLU
from exp_d3_induction import _rope_tables, _apply_rope  # reuse RoPE helpers
from cubbyllm.model.recall import MemoryRead

D = int(os.environ.get("CB_D", "128"))
S = int(os.environ.get("CB_S", "256")); S += S % 2
T = S // 2
B = int(os.environ.get("CB_B", "32"))
STEPS = int(os.environ.get("CB_STEPS", "4000"))
EVERY = int(os.environ.get("CB_EVAL", "500"))
LR = float(os.environ.get("CB_LR", "3e-3"))
V = int(os.environ.get("CB_V", "128"))
WINDOW = int(os.environ.get("CB_WINDOW", "32"))   # deliberately << T so the twin is beyond it


def make_seq(n, g):
    base = torch.randint(0, V, (n, T), generator=g)
    return torch.cat([base, base], dim=1)


class WinAttn(nn.Module):
    def __init__(self, d, heads=4, window=WINDOW):
        super().__init__(); self.h = heads; self.dh = d // heads; self.window = window
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B_, Sx, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B_, Sx, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        cos, sin = _rope_tables(self.dh, torch.arange(Sx, device=x.device), x.device)
        cos, sin = cos.view(1, 1, Sx, self.dh), sin.view(1, 1, Sx, self.dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        i = torch.arange(Sx, device=x.device)
        keep = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return self.o(o.transpose(1, 2).reshape(B_, Sx, d))


class Model(nn.Module):
    def __init__(self, use_mem):
        super().__init__()
        self.emb = nn.Embedding(V, D); self.pos = nn.Embedding(S, D)
        self.n1 = nn.ModuleList([RMSNorm(D) for _ in range(4)])
        self.mix = nn.ModuleList([WinAttn(D) for _ in range(4)])
        self.n2 = nn.ModuleList([RMSNorm(D) for _ in range(4)])
        self.ffn = nn.ModuleList([SwiGLU(D) for _ in range(4)])
        self.use_mem = use_mem
        if use_mem:
            self.mem_norm = RMSNorm(D); self.mem = MemoryRead(D, d_key=64, topk=8)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for a, m, b, f in zip(self.n1, self.mix, self.n2, self.ffn):
            h = h + m(a(h))
            if self.use_mem:
                h = h + self.mem(self.mem_norm(h), WINDOW)
            h = h + f(b(h))
        return self.head(h)


def run(use_mem, dev):
    torch.manual_seed(0)
    model = Model(use_mem).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    for s in range(1, STEPS + 1):
        x = make_seq(B, g).to(dev)
        logits = model(x)
        loss = F.cross_entropy(logits[:, T - 1:-1].reshape(-1, V), x[:, T:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    # Eval: second-half induction accuracy. The twin of every second-half token sits
    # exactly T positions back, and T > WINDOW by construction — so EVERY second-half
    # prediction requires beyond-window recall. Overall 2nd-half acc IS the
    # beyond-window recall number; no depth split needed.
    with torch.no_grad():
        ge = torch.Generator().manual_seed(999); x = make_seq(512, ge).to(dev)
        pred = model(x)[:, T - 1:-1].argmax(-1).cpu(); tgt = x[:, T:].cpu()
        acc = (pred == tgt).float().mean().item()
    return acc


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"gate: window={WINDOW}, induction distance T={T} (>{WINDOW}) | {dev}")
    a_off = run(False, dev)
    a_on = run(True, dev)
    print(f"  memory OFF: 2nd-half acc {a_off:.1%}")
    print(f"  memory ON : 2nd-half acc {a_on:.1%}")
    if a_on > 0.5 and a_on > a_off + 0.3:
        print("  PASS: the memory read recalls past the window. Proceed to integration.")
    else:
        print("  FAIL/INCONCLUSIVE: memory did not clear beyond-window recall. STOP and")
        print("  revisit the design before touching the real trunk.")


if __name__ == "__main__":
    main()
