# Architecture — the CubbyLLM backbone

Status: **RESOLVED (H-D1, 2026-07-23) — MinGRU.** `__wiring__ = Wiring.WIRED`.
This document describes the backbone model as it is actually implemented in
`cubbyllm/model/backbone/mingru.py`, how it sits in the assembled forward path
(`cubbyllm/model/assembly.py`), and the empirical bake-off that chose it. Every
number below links to the `validation/logs/` file that produced it (per the
`PACKAGE_LAYOUT.md` rule); nothing here describes a mechanism the code does not
have.

Scope note: the backbone only means something in context, so this doc situates
it in the whole next-token path. But its own responsibility is narrow — see
"What the backbone is (and is not)".

---

## 1. Where the backbone sits

The single place the forward path is assembled is `CubbyModel.forward`
(`model/assembly.py:89`). One inferred `Context` `c` is threaded explicitly
through the *generated* stages; the backbone is the one stage between embedding
and memory:

```text
tokens
  │
  ▼  infer_context(tokens)               FrozenSlotRouter — offline-pretrained,
  │                                        frozen at inference (H0b). Produces c.
  ├───────────────────────────── c ──────────────┐
  ▼                                               │
embedding.embed(tokens, c)      HybridEmbedding   │  c-conditioned
  │  (B, S, d_model)                              │
  ▼                                               │
backbone.forward(x)             MinGRUBackbone    │  ← THIS DOC. NOT c-conditioned.
  │  (B, S, d_model)                              │
  ▼                                               │
memory.forward_generated(h, c)  MemoryLayer       │  c-conditioned  θ = f(c)
  │  (B, S, d_model)              (hardened)       │
  ▼                                               │
head.logits(h, k)               TopKRetrievalHead ◄┘  project + cosine top-K
  │  (B, S, V)
  ▼
next-token logits
```

The architectural point worth internalizing: **the backbone is the fixed,
shared trunk.** It is deliberately *not* a function of context. The whole
central bet (H0, θ = f(c)) — the specialization/anti-forgetting mechanism —
lives in the **memory layer** downstream, which generates its active weights
from `c`. The backbone contributes shared sequence-mixing capacity that every
context reuses. Keeping specialization out of the trunk is what lets the trunk
stay a plain, well-understood recurrent stack.

`c` is a *required* argument to every generated call in `forward`; a
context-ignoring path cannot be written there without a type error — the
structural guard against the GCE "bypass" defect (H0). The backbone sits inside
that guarded path but does not itself take `c`.

---

## 2. What the backbone is (and is not)

**Is:** a map from a token-position embedding sequence to per-position hidden
states — `forward: (B, S, d_model) → (B, S, d_model)`. It conforms to the
`Backbone` protocol (`model/backbone/base.py`), which is deliberately
**interface-only** (a guard test keeps it so) so alternate backbones stay
pluggable and no single choice is welded into the interface.

**Is not:** the output head (that's the retrieval head), the memory (that's
where θ=f(c) lives), the binding head (relational / world-model channel, H-B5),
or the router. It does no context conditioning, no unbind, no retrieval.

---

## 3. The MinGRU backbone in detail

`MinGRUBackbone(d_model, n_layers=2)` (`model/backbone/mingru.py:76`). The trunk
is `n_layers` identical blocks, each a pre-norm mixer + pre-norm FFN with
residuals — the same scaffold every bake-off candidate shared, so the *mixer* is
the only thing MinGRU-specific:

```text
for each layer:
    x = x + MinGRUMixer(RMSNorm(x))     # sequence mixing (the recurrence)
    x = x + SwiGLU(RMSNorm(x))          # per-position channel mixing
```

### 3.1 The MinGRU mixer (`_MinGRUMixer`, mingru.py:62)

A minimal gated linear recurrence — cubby-lm's form. Per position it computes a
candidate `x_scan` and a per-channel retention `a`, then runs the linear
recurrence `h_t = a_t · h_{t-1} + x_t` over the sequence:

```text
x_scan = sigmoid(proj_g(x)) · tanh(proj_v(x))     # gated candidate,  (B,S,d)
a      = 0.001 + 0.998 · sigmoid(proj_d(x))       # retention in (0.001, 0.999)
h      = log_domain_scan(x_scan, a)               # h_t = a_t·h_{t-1} + x_t
```

- Three `Linear(d, d)` projections: `proj_g` (gate), `proj_v` (value), `proj_d`
  (decay/retention). `proj_d.bias` is initialised to `1.0`, giving ~0.73
  retention at init (`sigmoid(1) ≈ 0.73`) so early training starts with a
  usefully long memory rather than near-zero.
- `a` is clamped into `(0.001, 0.999)` so the recurrence is always contractive
  and never a pure copy or pure reset.

### 3.2 The log-domain parallel scan (`_log_domain_scan`, mingru.py:51)

The recurrence is evaluated by the **Heinsen (2023) log-domain parallel scan**,
not a sequential loop. It computes the cumulative products/sums in log space so
the whole sequence resolves in `O(log T)` parallel depth:

```text
a_star = cumsum(log a, dim=seq)                    # cumulative log-retention
h      = exp(a_star + logcumsumexp(log(x⁺) − a_star))       positive part
       − exp(a_star + logcumsumexp(log((−x)⁺) − a_star))    negative part
```

The sign-split (positive/negative parts computed separately, then subtracted) is
what lets a log-domain scan handle the signed `x_scan`. This is the property
that motivated the choice over a plain GRU: same recurrence, but a parallel scan
instead of a sequential CUDA kernel — the throughput lever at scale.

### 3.3 SwiGLU FFN (`_SwiGLU`, mingru.py:39)

`o(silu(g(x)) · u(x))` with hidden width `d_model · 2` (`mult=2`), all
bias-free `Linear`s. Standard gated-linear-unit channel mixer.

### 3.4 RMSNorm (`_RMSNorm`, mingru.py:30)

`x · rsqrt(mean(x²) + 1e-6) · w`, a learnable per-channel scale `w`. Pre-norm on
both the mixer and the FFN sub-blocks.

### 3.5 Shapes and parameters

- Input/output: `(B, S, d_model)` → `(B, S, d_model)`; the trunk preserves
  width and length.
- Per layer, backbone params ≈ 3·d² (mixer projections) + 3·(2d²) (SwiGLU) +
  norms. At the bake-off's `d=128, L=2` the backbone was **296,192** params
  (see log below).
- `d_model` and `n_layers` come from `CubbyConfig` (`core/config.py`); the class
  default is `n_layers=2`. The subword integration run uses `L=3`.

---

## 4. Why MinGRU — the bake-off (H-D1)

The choice was **not** inherited from cubby-lm and **not** taken in isolation.
`validation/exp_d1b_backbone_bakeoff.py` plugs four candidate mixers into the
**real assembled `CubbyModel`** (frozen router → hybrid embedding → backbone →
hardened θ=f(c) memory → learnable retrieval head), trains each on the same
subword TinyStories setup under a matched budget, with per-component seeds so the
**backbone is the only variable**. Numbers reflect each backbone in the
architecture it would actually sit in.

Setup: subword `grillcheese_spm32k_v2` (V=32000), `d=128, L=2, 600 steps`, 8000
stories → 1,739,764 tokens. Source log:
[`validation/logs/exp_d1b_backbone_bakeoff.log`](validation/logs/exp_d1b_backbone_bakeoff.log).

| candidate | mixer | ppl | bpc | backbone params | ms/step |
| --- | --- | --- | --- | --- | --- |
| **mingru** | MinGRU log-domain scan (chosen) | 34.8 | **1.308** | 296,192 | 155 |
| gru | classic `nn.GRU` | 34.0 | 1.300 | 395,264 | 157 |
| bdh | attention-free Q=K + per-channel decay | 49.1 | 1.435 | 295,426 | 137 |
| attn | causal multi-head softmax attention | 48.0 | 1.427 | 328,192 | 135 |

Reading the table:

- **Gated recurrence beats both attention-free Q=K and softmax attention by
  ~10% bpc** (1.30–1.31 vs 1.43) at this scale — the clearest signal in the run.
- **MinGRU ties GRU** (1.308 vs 1.300 bpc — within single-run noise) while using
  **25% fewer backbone params** (296k vs 395k).
- On top of the tie, MinGRU brings the **log-domain parallel scan** (GRU's kernel
  is sequential) and **continuity with cubby-lm**. Those two tie-breakers, plus
  the param saving, are why the user decision was **MinGRU**.
- `attn`/`bdh` were marginally fastest at this tiny `d=128, S=64` (135 ms), but
  that ordering is not expected to hold as `d`/`S` grow and is not the operating
  regime; quality and scan-parallelism drove the call.

Caveat (kept honest per repo norms): these are **toy-scale** numbers (d=128,
L=2, 600 steps, CPU). The parallel-scan throughput advantage is argued by
construction, not yet measured at scale — see `VALIDATION_REPORT.md`'s "what
would change these conclusions" and `TODO.md` ("Scale H0 past toy scale").

`exp_d1_backbone.py` is the earlier standalone char-level micro-benchmark with a
stylized BDH; `exp_d1b` (above) supersedes it by running inside the real model.

---

## 5. Pluggability

`model/backbone/base.py` defines `Backbone` as a `runtime_checkable` `Protocol`
with a single `forward(tokens) -> Tensor` method and nothing else — a guard test
enforces that the base file stays interface-only. Any module satisfying that
signature can be swapped in at `CubbyModel(..., backbone=...)`. The bake-off's
`Trunk` (with a `mixer_cls` slot) is the canonical example of how an alternate
mixer drops in without touching the rest of the model.

---

## 6. Running / verifying it

- **Full-stack integration on real text:** `python validation/exp_tinystories.py`
  — assembles the real `CubbyModel` with `MinGRUBackbone(d=128, L=3)` on subword
  TinyStories, trains through the real `TrainLoop` over a manifest-pinned
  pipeline, reports token perplexity + char-normalized bpc (~1.32 bpc), and
  samples a story. Log: `validation/logs/exp_tinystories.log`.
- **The decision experiment:** `python validation/exp_d1b_backbone_bakeoff.py`
  (reproduces §4).
- **Package tests:** `python -m pytest tests -q` (needs `torch`+`numpy`
  installed). The `tests/model/` guards check the backbone conforms to the
  protocol and that `base.py` stays interface-only.

---

## 7. Open items (backbone-specific)

- **Scale.** All numbers are toy-scale/CPU. Real `d_model`/`n_layers` and the
  parallel-scan throughput win are untested at size (`TODO.md`).
- **Depth/width.** `n_layers` default is 2 (bake-off) / 3 (integration); the
  production shape is not chosen. `d_model` interacts with the H-C7 codebook
  sizing and the retrieval head — a fixed random head at d=128 drowns in
  crosstalk (H-C7), which is why the integration run uses a *learnable* head.
- **Backbone stays context-free by design.** If a future change wants the trunk
  itself to specialize, that is a departure from H0's "specialization lives in
  the memory layer" split and should be added to `CUBBYLLM_HYPOTHESES.md` as its
  own claim before implementing — not slipped into the trunk.

---

Sources: `cubbyllm/model/backbone/mingru.py`, `cubbyllm/model/backbone/base.py`,
`cubbyllm/model/assembly.py`, `cubbyllm/core/config.py`,
`validation/exp_d1b_backbone_bakeoff.py` + its log, `validation/exp_tinystories.py`.
Decision record: `CUBBYLLM_HYPOTHESES.md` (H-D1) and `VALIDATION_REPORT.md`
(scorecard row D1).
