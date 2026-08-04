# Episodic Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give CubbyLLM recall past the attention window — a trained, differentiable per-token episodic memory whose read is soft attention over the top-K most similar past tokens, retrieved from the beyond-window causal past.

**Architecture:** A shared differentiable read (`MemoryRead`: soft attention over a (key, value) set) drives two paths — a masked dense top-K over the sequence's own beyond-window past during training, and an `EpisodicStore` (a persisted, growing per-sequence store, retrieved by exact Hamming over binary key-codes) at inference. Prove the capability on a toy needle-beyond-window gate BEFORE integrating into `HybridBackbone` as a third interleaved mixer.

**Tech Stack:** Python, PyTorch (lazy-imported), numpy. No new deps. Torch-free `import cubbyllm` preserved.

## Global Constraints

- `import cubbyllm` MUST stay torch-free — torch imported lazily inside methods, never at module top level (mirror `mingru.py`).
- Every new module declares `__wiring__` (`Wiring.STANDALONE` until the toy gate passes and it is interleaved into the default `HybridBackbone`, then `WIRED`).
- No `model.py` file or `models/` dir anywhere (guard test).
- Only `cubbyllm/ops/**` may import grilly — the binary-code path uses plain torch `sign`/comparison, no grilly import.
- Reads are **per-position, per-sequence** — never a batch-broadcast DC offset (trap 1, `MEMORY_PROBE.md`).
- The store is an explicit `state_dict`/`load_state_dict` object — not held in `__dict__` (trap 2).
- Tests: `python -m pytest tests -q` from repo root (needs `torch`+`numpy` installed).
- Commit messages end with the two trailer lines used across this repo (`Co-Authored-By:` and `Claude-Session:`).

## File Structure

- Create `cubbyllm/model/recall/__init__.py` — package exports + `__wiring__`.
- Create `cubbyllm/model/recall/read.py` — `MemoryRead` (the differentiable read + masked-dense training path).
- Create `cubbyllm/model/recall/store.py` — `EpisodicStore` (inference store: write, cosine + Hamming retrieve, persist).
- Create `validation/exp_d5_episodic.py` — the falsifiable toy gate (needle beyond window).
- Modify `cubbyllm/model/backbone/hybrid.py` — add `mem_every` and a `MemoryRead` branch on those layers (forward masked path + step store path).
- Create `tests/model/recall/test_read.py`, `tests/model/recall/test_store.py`.
- Create `tests/model/backbone/test_hybrid_memory.py` — integration + decode equivalence.

---

### Task 1: `MemoryRead` — the differentiable read

**Files:**
- Create: `cubbyllm/model/recall/read.py`
- Test: `tests/model/recall/test_read.py`

**Interfaces:**
- Produces:
  - `MemoryRead(d_model: int, d_key: int = 64, topk: int = 8)` — `nn.Module`.
  - `.qkv(h: Tensor) -> tuple[Tensor, Tensor, Tensor]` — `h (B,S,d)` → `q (B,S,d_key)`, `k (B,S,d_key)`, `v (B,S,d)`.
  - `.read(q: Tensor, k_set: Tensor, v_set: Tensor) -> Tensor` — soft attention of `q (B,S,d_key)` over a per-query candidate set `k_set (B,S,K,d_key)`, `v_set (B,S,K,d)` → `r (B,S,d)` (already output-projected). All-masked rows (any `k_set` row all `-inf` in similarity) read zero.
  - `.forward(h: Tensor, window: int) -> Tensor` — training path: builds `q,k,v`, masks to the beyond-window causal past (`j < i - window`), selects top-K per query, calls `.read`, returns `r (B,S,d)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/model/recall/test_read.py
import pytest
torch = pytest.importorskip("torch")
from cubbyllm.model.recall import MemoryRead

D, DK, B, S = 32, 16, 2, 20


def test_read_is_per_position_not_a_dc_offset():
    """Two query positions with different content must get different reads —
    the exact failure MEMORY_PROBE.md diagnosed (one vector broadcast to all)."""
    torch.manual_seed(0)
    m = MemoryRead(D, d_key=DK, topk=4)
    h = torch.randn(B, S, D)
    r = m.forward(h, window=4)
    assert r.shape == (B, S, D)
    # positions far apart, both with beyond-window memory, differ
    assert not torch.allclose(r[:, 10], r[:, 18], atol=1e-5)


def test_read_only_sees_beyond_window_past():
    """Perturbing a token INSIDE the window of position i must NOT change i's read;
    perturbing one BEYOND the window must. That is the whole point of the mask."""
    torch.manual_seed(1)
    m = MemoryRead(D, d_key=DK, topk=8)
    W = 4
    h = torch.randn(1, S, D)
    with torch.no_grad():
        r0 = m.forward(h, window=W)[0, S - 1]
        h_in = h.clone(); h_in[0, S - 2] = torch.randn(D)        # inside window
        r_in = m.forward(h_in, window=W)[0, S - 1]
        h_far = h.clone(); h_far[0, 0] = torch.randn(D)          # beyond window
        r_far = m.forward(h_far, window=W)[0, S - 1]
    assert torch.allclose(r0, r_in, atol=1e-6), "read used a within-window token"
    assert not torch.allclose(r0, r_far, atol=1e-6), "read ignored the beyond-window past"


def test_early_positions_with_no_memory_read_zero():
    """Positions i <= window have no beyond-window past; their read must be a
    clean zero, not a NaN."""
    m = MemoryRead(D, d_key=DK, topk=4)
    h = torch.randn(1, S, D)
    r = m.forward(h, window=6)
    assert torch.isfinite(r).all()
    assert torch.allclose(r[0, 0], torch.zeros(D), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/model/recall/test_read.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cubbyllm.model.recall'`.

- [ ] **Step 3: Write minimal implementation**

```python
# cubbyllm/model/recall/read.py
"""MemoryRead — the differentiable episodic read (soft attention over top-K).

Wired: STANDALONE — attaches into HybridBackbone once the toy gate (exp_d5) passes.

The read is a modern-Hopfield / kNN-attention lookup: a per-position query attends
over the top-K most similar candidate keys and pools their values. It is
per-position, per-sequence by construction (each query gets its own set) — the
structural fix for MEMORY_PROBE.md's DC-offset failure. The training path masks
candidates to the BEYOND-window causal past (j < i - window) so the memory learns
long-range recall instead of duplicating the window.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.protocols import Wiring


class MemoryRead(nn.Module):
    def __init__(self, d_model: int, d_key: int = 64, topk: int = 8):
        super().__init__()
        self.topk = int(topk)
        self.scale = float(d_key) ** -0.5
        self.q = nn.Linear(d_model, d_key, bias=False)
        self.k = nn.Linear(d_model, d_key, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def qkv(self, h):
        return self.q(h), self.k(h), self.v(h)

    def read(self, q, k_set, v_set):
        """q (B,S,dk); k_set (B,S,K,dk); v_set (B,S,K,d) -> r (B,S,d)."""
        sim = (q.unsqueeze(2) * k_set).sum(-1) * self.scale     # (B,S,K)
        w = torch.softmax(sim, dim=-1)
        w = torch.nan_to_num(w)                                 # all-(-inf) row -> 0
        r = (w.unsqueeze(-1) * v_set).sum(2)                    # (B,S,d)
        return self.o(r)

    def forward(self, h, window):
        B, S, d = h.shape
        q, k, v = self.qkv(h)
        sim = torch.einsum("bik,bjk->bij", q, k) * self.scale   # (B,S,S)
        i = torch.arange(S, device=h.device)
        allowed = (i[:, None] - i[None, :]) > window            # j < i - window
        sim = sim.masked_fill(~allowed[None], float("-inf"))
        kk = min(self.topk, S)
        topv, topi = sim.topk(kk, dim=-1)                       # (B,S,kk)
        # gather the selected keys/values per query
        dk = k.shape[-1]
        k_set = torch.gather(k.unsqueeze(1).expand(B, S, S, dk), 2,
                             topi.unsqueeze(-1).expand(B, S, kk, dk))
        v_set = torch.gather(v.unsqueeze(1).expand(B, S, S, d), 2,
                             topi.unsqueeze(-1).expand(B, S, kk, d))
        # rows whose every candidate was masked (-inf): read zero
        no_mem = torch.isinf(topv).all(-1, keepdim=True)        # (B,S,1)
        sim_set = torch.where(torch.isinf(topv), torch.full_like(topv, -1e9), topv)
        w = torch.nan_to_num(torch.softmax(sim_set, dim=-1))
        r = self.o((w.unsqueeze(-1) * v_set).sum(2))
        return torch.where(no_mem, torch.zeros_like(r), r)


__wiring__ = Wiring.STANDALONE
```

Also create `cubbyllm/model/recall/__init__.py`:

```python
"""cubbyllm.model.recall — episodic (per-token) recall memory past the window.

Wired: WIRED — package marker. Distinct from cubbyllm/model/memory (the
parametric theta=f(c) memory). See docs/superpowers/specs/2026-08-04-episodic-
memory-design.md and H-D5.
"""
from __future__ import annotations

from ...core.protocols import Wiring
from .read import MemoryRead

__all__ = ["MemoryRead"]
__wiring__ = Wiring.WIRED
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/model/recall/test_read.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cubbyllm/model/recall/__init__.py cubbyllm/model/recall/read.py tests/model/recall/test_read.py
git commit -m "feat(recall): MemoryRead — per-position differentiable top-K read

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

### Task 2: The falsifiable toy gate — needle beyond the window

**Files:**
- Create: `validation/exp_d5_episodic.py`

**Interfaces:**
- Consumes: `MemoryRead` from Task 1.
- Produces: a runnable script printing beyond-window vs within-window recall for a small model **with** and **without** the memory read. This is GO/NO-GO before any package integration.

The gate reuses `exp_d3_induction`'s dense repeated-sequence induction task but plants the repeat so the answer sits BEYOND a small window. A tiny model (windowed attention, `window` small) with a `MemoryRead` branch should recall beyond the window; the same model without the memory read should sit at chance beyond it.

- [ ] **Step 1: Write the gate script**

```python
# validation/exp_d5_episodic.py
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
```

- [ ] **Step 2: Smoke-run locally (tiny) to confirm it executes**

Run: `CB_STEPS=40 CB_EVAL=40 CB_S=48 CB_WINDOW=8 CB_DEVICE=cpu python validation/exp_d5_episodic.py`
Expected: prints both accuracies and a PASS/FAIL line without error (numbers meaningless at 40 steps — this only checks it runs).

- [ ] **Step 3: Commit**

```bash
git add validation/exp_d5_episodic.py
git commit -m "feat(recall): exp_d5 gate — needle-beyond-window recall (GO/NO-GO)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

- [ ] **Step 4: GATE — run at real scale on GPU (Colab) and record**

Run: `CB_STEPS=6000 CB_S=256 CB_WINDOW=32 python validation/exp_d5_episodic.py`, tee to `validation/logs/exp_d5_episodic.log`.
Decision: **PASS → continue to Task 3. FAIL → STOP**, bring the log back, and revise the spec before any further tasks.

---

### Task 3: `EpisodicStore` — the inference store (cosine path)

**Files:**
- Create: `cubbyllm/model/recall/store.py`
- Test: `tests/model/recall/test_store.py`
- Modify: `cubbyllm/model/recall/__init__.py` (export `EpisodicStore`)

**Interfaces:**
- Consumes: nothing from prior tasks (pure store).
- Produces:
  - `EpisodicStore(d_key: int, d_model: int)` — plain object (not `nn.Module`), one per sequence.
  - `.write(k: Tensor, v: Tensor) -> None` — append `k (M,d_key)`, `v (M,d_model)`; also stores `sign(k)` as `codes`.
  - `.__len__() -> int`.
  - `.retrieve_cosine(q: Tensor, topk: int) -> tuple[Tensor, Tensor]` — `q (d_key,)` → `(k_top (K,d_key), v_top (K,d_model))`, exact cosine.
  - `.retrieve_hamming(q: Tensor, topk: int) -> tuple[Tensor, Tensor]` — same shape, ranked by Hamming distance of `sign(q)` to `codes`. (Added in Task 4.)
  - `.state_dict() -> dict`, `.load_state_dict(sd: dict) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/model/recall/test_store.py
import pytest
torch = pytest.importorskip("torch")
from cubbyllm.model.recall import EpisodicStore

DK, D = 16, 32


def test_write_grows_and_retrieves_the_nearest_by_cosine():
    torch.manual_seed(0)
    s = EpisodicStore(DK, D)
    keys = torch.randn(10, DK); vals = torch.randn(10, D)
    s.write(keys, vals)
    assert len(s) == 10
    # query = one of the stored keys -> its own value comes back on top
    q = keys[4]
    k_top, v_top = s.retrieve_cosine(q, topk=3)
    assert k_top.shape == (3, DK) and v_top.shape == (3, D)
    assert torch.allclose(v_top[0], vals[4], atol=1e-5)


def test_state_dict_round_trip_preserves_the_store():
    s = EpisodicStore(DK, D)
    s.write(torch.randn(5, DK), torch.randn(5, D))
    sd = s.state_dict()
    s2 = EpisodicStore(DK, D); s2.load_state_dict(sd)
    assert len(s2) == 5
    assert torch.equal(s2.keys, s.keys) and torch.equal(s2.values, s.values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/model/recall/test_store.py -q`
Expected: FAIL — `ImportError: cannot import name 'EpisodicStore'`.

- [ ] **Step 3: Write minimal implementation**

```python
# cubbyllm/model/recall/store.py
"""EpisodicStore — the inference-time, persisted per-token memory store.

Wired: STANDALONE — the decode-side counterpart to MemoryRead's training path.

Holds one sequence's (key, value) pairs plus binary key-codes (sign(key)) for
O(N) Hamming retrieval. An explicit state_dict object so it can be checkpointed —
the fix for MEMORY_PROBE.md trap 2 (the last store lived in __dict__ and every
resume wiped it). retrieve_* return the top-K set; the read over that set is
MemoryRead.read, identical to training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core.protocols import Wiring


class EpisodicStore:
    def __init__(self, d_key: int, d_model: int):
        self.d_key, self.d_model = int(d_key), int(d_model)
        self.keys = torch.empty(0, d_key)
        self.values = torch.empty(0, d_model)
        self.codes = torch.empty(0, d_key)

    def __len__(self):
        return self.keys.shape[0]

    def write(self, k, v):
        self.keys = torch.cat([self.keys.to(k.device), k], 0)
        self.values = torch.cat([self.values.to(v.device), v], 0)
        self.codes = torch.cat([self.codes.to(k.device), torch.sign(k)], 0)

    def retrieve_cosine(self, q, topk):
        n = len(self)
        kk = min(topk, n)
        sims = F.cosine_similarity(q.unsqueeze(0), self.keys, dim=-1)   # (N,)
        idx = sims.topk(kk).indices
        return self.keys[idx], self.values[idx]

    def state_dict(self):
        return {"keys": self.keys, "values": self.values, "codes": self.codes,
                "d_key": self.d_key, "d_model": self.d_model}

    def load_state_dict(self, sd):
        self.keys, self.values, self.codes = sd["keys"], sd["values"], sd["codes"]
        self.d_key, self.d_model = sd["d_key"], sd["d_model"]


__wiring__ = Wiring.STANDALONE
```

Add to `cubbyllm/model/recall/__init__.py`:

```python
from .store import EpisodicStore
```
and add `"EpisodicStore"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/model/recall/test_store.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add cubbyllm/model/recall/store.py cubbyllm/model/recall/__init__.py tests/model/recall/test_store.py
git commit -m "feat(recall): EpisodicStore — persisted per-sequence store, cosine retrieve

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

### Task 4: Hamming retrieval + the training/serving-skew guard

**Files:**
- Modify: `cubbyllm/model/recall/store.py` (add `retrieve_hamming`)
- Test: `tests/model/recall/test_store.py` (add the guard)

**Interfaces:**
- Produces: `EpisodicStore.retrieve_hamming(q, topk)` (signature as in Task 3).

This task also adds a `return_idx: bool = False` kwarg to `retrieve_cosine` and the
new `retrieve_hamming`; when true they return `(idx, k, v)` so the guard can compare
*index sets*.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/model/recall/test_store.py
def test_hamming_topk_recovers_the_cosine_neighbour_set():
    """The O(N)-Hamming path must return essentially the same top-K as exact cosine
    — otherwise inference silently recalls a different set than training shaped.
    High overlap on well-separated keys (sign hashing preserves order when keys are
    far apart)."""
    torch.manual_seed(0)
    s = EpisodicStore(DK, D)
    keys = torch.randn(200, DK) * 2.0          # well-separated
    s.write(keys, torch.randn(200, D))
    overlap = 0
    for _ in range(20):
        q = keys[torch.randint(0, 200, (1,)).item()] + torch.randn(DK) * 0.1
        ci = set(s.retrieve_cosine(q, 8, return_idx=True)[0].tolist())
        hi = set(s.retrieve_hamming(q, 8, return_idx=True)[0].tolist())
        overlap += len(ci & hi)
    assert overlap / (20 * 8) > 0.6, "Hamming index diverges from cosine ranking"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/model/recall/test_store.py::test_hamming_topk_recovers_the_cosine_neighbour_set -q`
Expected: FAIL — `retrieve_hamming` not defined.

- [ ] **Step 3: Write minimal implementation**

Add `return_idx` to both retrieves and implement Hamming:

```python
    def retrieve_cosine(self, q, topk, return_idx=False):
        kk = min(topk, len(self))
        sims = F.cosine_similarity(q.unsqueeze(0), self.keys, dim=-1)
        idx = sims.topk(kk).indices
        return (idx, self.keys[idx], self.values[idx]) if return_idx else (self.keys[idx], self.values[idx])

    def retrieve_hamming(self, q, topk, return_idx=False):
        kk = min(topk, len(self))
        qc = torch.sign(q)
        ham = (qc.unsqueeze(0) != self.codes).sum(-1)          # (N,) Hamming distance
        idx = (-ham).topk(kk).indices
        return (idx, self.keys[idx], self.values[idx]) if return_idx else (self.keys[idx], self.values[idx])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/model/recall/test_store.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cubbyllm/model/recall/store.py tests/model/recall/test_store.py
git commit -m "feat(recall): O(N) Hamming retrieval + train/serve-skew guard

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

### Task 5: Interleave the memory read into `HybridBackbone`

**Files:**
- Modify: `cubbyllm/model/backbone/hybrid.py`
- Test: `tests/model/backbone/test_hybrid_memory.py`

**Interfaces:**
- Consumes: `MemoryRead` (Task 1), `EpisodicStore` (Tasks 3-4).
- Produces:
  - `HybridBackbone(..., mem_every: int = 0, mem_topk: int = 8, mem_key: int = 64)` — `mem_every=0` disables memory (default stays behaviour-identical to today until a run turns it on). When `> 0`, layers where `i % mem_every == 0` get a `MemoryRead` branch.
  - `forward` unchanged in signature; on memory layers adds `h = h + mem(norm(h), window)` (masked beyond-window training path).
  - `step` gains a per-layer `EpisodicStore` in its carried state on memory layers: each step writes the layer's (k,v) and reads top-K by Hamming. `state` stays bounded in *compute* (top-K), store grows off the recurrent path.

- [ ] **Step 1: Write the failing test**

```python
# tests/model/backbone/test_hybrid_memory.py
import pytest
torch = pytest.importorskip("torch")
from cubbyllm.model.backbone import HybridBackbone

D, L, B, S, W = 32, 6, 2, 24, 6


def test_memory_off_is_identical_to_a_plain_hybrid():
    torch.manual_seed(0)
    bb = HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=4, mem_every=0)
    x = torch.randn(B, S, D)
    assert bb.forward(x).shape == x.shape
    # no MemoryRead modules created when mem_every=0
    from cubbyllm.model.recall import MemoryRead
    assert not any(isinstance(m, MemoryRead) for m in bb.modules())


def test_memory_layer_reads_the_beyond_window_past_in_forward():
    torch.manual_seed(1)
    bb = HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=4, mem_every=2)
    from cubbyllm.model.recall import MemoryRead
    assert any(isinstance(m, MemoryRead) for m in bb.modules())
    x = torch.randn(1, S, D)
    with torch.no_grad():
        y0 = bb.forward(x)[0, -1]
        xf = x.clone(); xf[0, 0] = torch.randn(D)      # beyond every window
        yf = bb.forward(xf)[0, -1]
    assert not torch.allclose(y0, yf, atol=1e-6), "memory did not carry the far token"


def test_step_with_memory_matches_forward():
    """Incremental decode with the store must equal the parallel forward, so a
    benchmark measures the trained model. Single sequence (B=1): the v1 store is
    per-sequence, and eval decode (exp_needle_recall) is B=1. `step` retrieves by
    COSINE (matching forward's dense selection); Hamming is the O(1) option the
    Task-4 guard covers, not the decode-equivalence contract."""
    torch.manual_seed(0)
    bb = HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=4,
                        mem_every=2, mem_topk=8)
    x = torch.randn(1, S, D)
    with torch.no_grad():
        parallel = bb.forward(x)
        states, seq = None, []
        for t in range(S):
            y, states = bb.step(x[:, t], states)
            seq.append(y)
        inc = torch.stack(seq, dim=1)
    assert torch.allclose(parallel, inc, atol=1e-3), \
        f"decode diverged; max |diff| {(parallel-inc).abs().max():.2e}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/model/backbone/test_hybrid_memory.py -q`
Expected: FAIL — `HybridBackbone.__init__() got an unexpected keyword argument 'mem_every'`.

- [ ] **Step 3: Write minimal implementation**

In `hybrid.py`, extend `HybridBackbone.__init__` and the loops:

```python
# add imports at top of hybrid.py
from ..recall import MemoryRead, EpisodicStore

# __init__ signature adds: mem_every: int = 0, mem_topk: int = 8, mem_key: int = 64
        self.mem_every = int(mem_every)
        self.mem_topk = int(mem_topk)
        self.is_mem = [mem_every and (i % mem_every == 0) for i in range(n_layers)]
        self.mem_n = nn.ModuleList([_RMSNorm(d_model) if a else nn.Identity()
                                    for a in self.is_mem])
        self.mem = nn.ModuleList([MemoryRead(d_model, d_key=mem_key, topk=mem_topk)
                                  if a else nn.Identity() for a in self.is_mem])

# forward: inside the layer loop, after the mixer residual and before ffn:
        for idx, (n1, m, n2, f) in enumerate(zip(self.n1, self.mix, self.n2, self.ffn)):
            x = self._layer_mix(x, n1, m)               # x + m(n1(x))
            if self.is_mem[idx]:
                x = x + self.mem[idx](self.mem_n[idx](x), self.window)
            x = x + f(n2(x))
```

For `step`, carry the mixer states and the per-memory-layer stores together. Change
the step state from a bare list to `{"mix": [...per-layer mixer state...], "mem":
{layer_idx: EpisodicStore}}`. On a memory layer, after the mixer residual: normalise,
project to (q, k, v) via `self.mem[i].qkv`, write (k, v) to that layer's store,
retrieve top-K by cosine, and read — reusing `MemoryRead.read` so decode and forward
share the exact read. Refactor the shared mixer+residual into `_layer_mix(x, n1, m)`
so `forward` and `step` call the same code. Concrete step (B=1 decode):

```python
    def step(self, x_t, states=None):
        n = len(self.mix)
        if states is None:
            states = {"mix": [None] * n,
                      "mem": {i: EpisodicStore(self.mem[i].q.out_features, self.d_model)
                              for i in range(n) if self.is_mem[i]}}
        mixs = states["mix"]; stores = states["mem"]
        new_mix = []
        for i in range(n):
            h, s = self.mix[i].step(self.n1[i](x_t), mixs[i])   # existing mixer step
            x_t = x_t + h
            if self.is_mem[i]:
                store = stores[i]
                q, k, v = self.mem[i].qkv(self.mem_n[i](x_t))    # (B,dk),(B,dk),(B,d)
                # write happens on the CURRENT token but the read must not see it
                # (causal): retrieve from what is already stored, THEN write.
                if len(store) >= 1:
                    kk = min(self.mem_topk, len(store))
                    k_top, v_top = store.retrieve_cosine(q[0], kk)   # (kk,dk),(kk,d)
                    r = self.mem[i].read(q.unsqueeze(1),             # (B,1,dk)
                                         k_top.unsqueeze(0).unsqueeze(0),  # (1,1,kk,dk)
                                         v_top.unsqueeze(0).unsqueeze(0))  # (1,1,kk,d)
                    x_t = x_t + r[:, 0]
                store.write(k, v)
            x_t = x_t + self.ffn[i](self.n2[i](x_t))
            new_mix.append(s)
        return x_t, {"mix": new_mix, "mem": stores}
```

Match `forward`'s causal contract: `forward` masks candidates to `j < i - window`
(strictly beyond-window past). To make `step` agree, the store read must exclude the
last `window` written tokens — keep a small rolling buffer of the most recent
`window` (k, v) pairs *outside* the store, and only `write` a token to the store
once it has fallen `window` steps behind. (Simplest: buffer the last `window` (k,v),
and on each step write the one leaving the buffer.) The decode-equivalence test is
what verifies this matches `forward`; iterate against it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/model/backbone/test_hybrid_memory.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full backbone suite (nothing else broke)**

Run: `python -m pytest tests/model/backbone tests/test_guards.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cubbyllm/model/backbone/hybrid.py tests/model/backbone/test_hybrid_memory.py
git commit -m "feat(recall): interleave MemoryRead into HybridBackbone (mem_every)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

### Task 6: Wire the training flag + full-suite green + hypothesis entry

**Files:**
- Modify: `validation/train_colab.py` (add `CB_MEM_EVERY`, thread into `_make_backbone`, record in meta)
- Modify: `validation/exp_needle_recall.py` and `validation/exp_decode_throughput.py` (`build` reads `mem_every`/`mem_topk`/`mem_key` from meta)
- Modify: `CUBBYLLM_HYPOTHESES.md` (add H-D5), `CLAUDE.md` (note the memory rung is now built, gated)

**Interfaces:**
- Consumes: the `HybridBackbone(mem_every=...)` from Task 5.
- Produces: `CB_MEM_EVERY` (default `0` = off until a run enables it), recorded in checkpoint meta so eval reconstructs the memory-enabled model.

- [ ] **Step 1: Add the flag and meta (no test — config plumbing, covered by the run)**

In `train_colab.py`, near the other backbone env vars:
```python
MEM_EVERY = _env("CB_MEM_EVERY", 0)       # 0 = off; >0 = MemoryRead every Nth layer
MEM_TOPK = _env("CB_MEM_TOPK", 8)
MEM_KEY = _env("CB_MEM_KEY", 64)
```
In `_make_backbone`, pass `mem_every=MEM_EVERY, mem_topk=MEM_TOPK, mem_key=MEM_KEY`
to `HybridBackbone`. In the `meta` dict, add `"mem_every": MEM_EVERY, "mem_topk":
MEM_TOPK, "mem_key": MEM_KEY`.

In `exp_needle_recall.py` and `exp_decode_throughput.py` `build`, when constructing
`HybridBackbone`, pass `mem_every=int(meta.get("mem_every", 0)), mem_topk=int(meta.get("mem_topk", 8)), mem_key=int(meta.get("mem_key", 64))`.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS (all).

- [ ] **Step 3: Add H-D5 to `CUBBYLLM_HYPOTHESES.md`**

Add under Group D, after H-D4:
```markdown
**H-D5 — a trained per-token episodic memory (differentiable top-K read over the
beyond-window causal past, HDC as the O(1) index) extends recall past the window
that H-D4 bounded.** Status: **design + toy-gate stage.** Built:
`cubbyllm/model/recall/{read,store}.py`, interleaved into `HybridBackbone`
(`mem_every`). Success metric = the beyond-window depths of `exp_needle_recall`
(currently chance) rising off the floor. **Kill criterion:** the `exp_d5` toy gate
must show memory-on beating memory-off on needle-beyond-window recall before the
real-trunk run; if not, the design is wrong. Design:
`docs/superpowers/specs/2026-08-04-episodic-memory-design.md`.
```

- [ ] **Step 4: Note it in `CLAUDE.md`**

Update the "Next cycles" line to mark the memory rung (0.0.5) as built-and-gated
(pending the exp_d5 gate + a real-trunk run), pointing at the spec.

- [ ] **Step 5: Commit**

```bash
git add validation/train_colab.py validation/exp_needle_recall.py validation/exp_decode_throughput.py CUBBYLLM_HYPOTHESES.md CLAUDE.md
git commit -m "feat(recall): CB_MEM_EVERY training flag + H-D5 + docs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Post-plan: the real validation (not a task — a run)

After Task 6, the memory-enabled hybrid trains with `CB_MEM_EVERY=2` at the 413M
config, and `exp_needle_recall` is re-run. **The claim is proven only if the
beyond-window depths (e.g. length 4096, depths 0.0–0.75) rise materially above the
~0% shuffled floor** — read against the same controls as H-D4. That run flips H-D5
to scale-VERIFIED or kills it.
