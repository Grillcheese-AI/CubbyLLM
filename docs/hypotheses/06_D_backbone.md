<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 6. Group D — Backbone architecture hypotheses

This group comes from a single "trillion-parameter architecture blueprint" document the user supplied "to validate and
see how it fits." It bundles dynamic tokenization (already covered in Group C), a BDH+TTT backbone, a GCE controller
(already the central bet, H0), and AIMC hardware (Group E) into one end-to-end pitch, plus a multiplicative
energy-savings calculation. It should not be adopted as a package — each piece is at a different maturity level, and the
blueprint itself is a synthesis document, not an implementation.

**H-D1 — BDH's Q=K constrained attention-free recurrence is a candidate backbone for CubbyLLM, one option among others
including cubby-lm's MinGRU approach.** Status: PLAUSIBLE — BDH and its `bdh-cl` continual-learning extension are real
published architecture work, but the efficiency numbers attached to them (35x faster training, 10x lower inference) come
from CIFAR-10/Sudoku-scale experiments, not LLM-scale text modeling. Since CubbyLLM's backbone is being chosen fresh
rather than inherited, this is a genuine head-to-head decision, not a replace-in-place question. **Validate** at small
scale on a real token-modeling task before assuming these numbers transfer — image/puzzle-benchmark speedups are not
evidence about a multi-billion-parameter text model.

**Validation result (2026-07-23 campaign): VIABILITY checked, real decision still deferred.** Matched-size char-LM
bake-off (`validation/exp_d1_backbone.py`, d=128/L=4/2500 steps, real valid.txt, cubby-lm's ACTUAL MinGRULayer
imported): an attention-free Q=K recurrence learns text competitively (1.584 bpc, far ahead of small softmax attention's
2.470) at lower per-step latency — but **MinGRU is the best learner here** (1.454 bpc, +0.13 over the Q=K mixer).

**RESOLVED (2026-07-23): the in-architecture bake-off picked MinGRU.** `validation/exp_d1b_backbone_bakeoff.py` plugged
four candidates into the *real* assembled `CubbyModel` (frozen router → hybrid embedding → backbone → hardened memory →
learnable head) on the real subword-TinyStories setup, matched budget, every non-backbone component built from the same
per-component seed so the backbone was the only variable. Held-out perplexity / bpc: **gru 34.0 / 1.300, mingru 34.8 /
1.308, bdh(Q=K) 49.1 / 1.435, attn 48.0 / 1.427.** Two clean clusters: gated recurrence (GRU, MinGRU) beats the
attention-free Q=K recurrence and softmax attention by ~10% bpc at this scale. GRU vs MinGRU is within single-run noise,
but MinGRU matches GRU's quality at **25% fewer backbone params**, with the log-domain parallel scan (throughput at
scale, where GRU's kernel is sequential) and continuity with cubby-lm. **User decision: MinGRU is the package default
backbone** (`cubbyllm/model/backbone/mingru.py`, `MinGRUBackbone` — a RMSNorm + MinGRU-recurrence + SwiGLU trunk; the
interface-only `base.py` keeps alternates pluggable). The 35×/10× BDH efficiency claims remain untested (need scale + a
faithful BDH impl); at this scale BDH-Q=K is a viable-but-not-winning contender.

**H-D4 — the H-D1 bake-off chose the backbone on bpc, which is blind to content-based retrieval; a windowed-attention
hybrid beats pure MinGRU on long-context recall, at bounded state.** Status: **scale-VERIFIED (2026-08-04) —
decisively.** *(Renumbered from a duplicate H-D3; the energy-claim H-D3 below predates this entry.)* The H-D1 bake-off
scored on held-out **bpc** — a next-token-prediction metric. Attention's mechanistic edge is *content-based lookup*
("where did this token appear before, and what followed it"), the induction-head circuit, which barely shows up in
average perplexity and is exactly what a long-document/needle capability needs. So H-D1 resolved the backbone on the one
axis that cannot see the thing at issue. The dense-induction probe (`validation/exp_d3_induction.py`, canonical
repeated-sequence task, per-position loss on the second half, d=128/L=6, multi-seed) measures it directly. Result,
sweeping distance across the window (`window=96`): at **T=64 (within window)** pure MinGRU **0%**, full-attention hybrid
**100%**, windowed hybrid **100%** — bounded state; at **T=256 (beyond window)** MinGRU **0%**, windowed hybrid **0%**
(collapses to MinGRU, as it must — a 96-token window cannot reach 256 back), full-attention hybrid still **100%**
(unbounded). So the windowed arm buys distance-robust recall out to ~window (×depth for stacked layers) at *fixed,
context-independent* state, and no further; full attention solves at any distance but only by paying a KV cache that
grows with context. The mechanism is controlled, not inferred (verified: within-window 100% by step 100; beyond-window
`W=16,T=32` stuck at chance). This is cubby-lm's own production shape (attention every 3rd layer, 8 of 22,
`trunk_torch/blocks.py`), which CubbyLLM had dropped without a decision. **Why windowed, not full:** full attention
keeps a KV cache that grows with context, breaking the O(1)-decode guarantee
(`tests/model/test_mingru_decode.py::test_state_size_is_independent_of_context_length`) that the inference-cost thesis
rests on; a sliding window keeps per-attention-layer state at O(window) = constant. **Built:**
`cubbyllm/model/backbone/hybrid.py::HybridBackbone` (MinGRU + windowed attention every Nth, **RoPE** on the attention so
it is position-aware — the package has no other positional signal and MinGRU-implicit position would leave layer-0
attention blind; RoPE also extrapolates train-S to the 4096 needle length; bounded-KV `step()` that reproduces
`forward()` even across window slides), `__wiring__ = STANDALONE` (pluggable alternate, **not** the default), drops into
`CubbyModel` with no assembly change; `tests/model/backbone/test_hybrid.py` (8) pins decode-equivalence, bounded state,
window reach + the ~depth×window compounding of stacked windows, and RoPE position-awareness. Wired through
`train_colab.py` (`CB_BACKBONE=hybrid`, meta-recorded) and `exp_needle_recall`/`exp_decode_throughput` (reconstruct
backbone from meta) so the A/B is a single flag. **Kill criterion / open:** (1) toy-scale only — re-run at the 413M
shape on real **needle recall across lengths**, pure-MinGRU vs windowed-hybrid, before making the hybrid the default;
(2) the honest limit for the 1M-context pitch — *no* bounded-state model does arbitrary-distance recall; windowed
attention extends distance-robust recall to ~depth×window tokens at fixed state, and beyond that recall falls back to
what the recurrent state retains. `validation/exp_d3_induction.py`, `validation/logs/exp_d3_*`.

**Scale result (2026-08-04): DECISIVE — kill-criterion (1) retired.** Two matched runs, identical except `CB_BACKBONE` —
pure MinGRU vs windowed-hybrid (MinGRU + windowed attention every 3rd of 8 layers, `window=512`, RoPE), d=512 / L=8 /
~0.98B tokens each — scored on `exp_needle_recall` (PMI, shuffled + untrained controls, argmax-spread check) across
lengths **256 / 512 / 1024 / 4096**. Floor: MinGRU's shuffled control ~15%, untrained ~33% (a PMI base-rate residual),
chance 12.5%. **Pure MinGRU (trained): ~28–32% at every length** — barely above its own shuffled floor, flat, weak.
**Windowed hybrid (trained): ~98% at 256, ~100% at 512** (every depth except the window-edge depth 0.0 = 41.7%), and at
**1024 the depth pattern IS the window, drawn to a single step:** `16.7 / 16.7 / 8.3 / 91.7 / 100` — chance while the
needle sits beyond the 512 window (depths 0.0–0.5), near-perfect once it is inside it (depths 0.75–1.0), the transition
landing exactly at the window boundary. Recall is a pure function of one variable — **needle-to-query distance vs
window**: inside → ~100%, outside → chance. That fingerprint rules out memorisation or artifact; it is the
sliding-window mechanism itself, at scale. So a windowed-attention hybrid gives near-perfect long-context recall within
its window at bounded, context-independent state, and pure recurrence does not (**98% vs 32%** at 256). The honest bound
is in the same data: beyond the window the hybrid falls to the floor like MinGRU — windowed attention buys perfect
recall to ~window (×depth for stacked layers), no further; recall *past* the window is the memory layer's job (rung
0.0.5, still unbuilt). **Decision: the backbone is the hybrid** — H-D1's "RESOLVED: MinGRU" is superseded; it was
resolved on the one metric (bpc) that cannot see retrieval. `validation/logs/needle_mingru.log`,
`validation/logs/needle_hybrid.log`.

**H-D5 — a trained per-token episodic memory (differentiable top-K read over the
beyond-window causal past, HDC as the O(1) index) extends recall past the window
that H-D4 bounded.** Status: **scale-VERIFIED (2026-08-05) — decisively.** Built:
`cubbyllm/model/recall/{read,store}.py`, interleaved into `HybridBackbone`
(`mem_every`). Success metric = the beyond-window depths of `exp_needle_recall`
(currently chance) rising off the floor. **Kill criterion:** the `exp_d5` toy gate
must show memory-on beating memory-off on needle-beyond-window recall before the
real-trunk run; if not, the design is wrong. Design:
`docs/superpowers/specs/2026-08-04-episodic-memory-design.md`. The store's
Hamming index is a **256-bit SimHash** (`sign(k · R)`, `R` a fixed random
projection), not single-bit `sign(k)` — Task 4 measured single-bit binarization
recovering only ~34% of the cosine top-K under small perturbation (too few bits),
and the 256-bit SimHash restores ~70% overlap, the honest-negative-result-first
finding this repo values. **Toy-gate result (2026-08-05, `exp_d5_episodic.py`,
D=128/L=4, window=32, induction distance T=128 > window, 6000 steps, RTX PRO 6000):
memory-ON 100.0% vs memory-OFF 0.8%** second-half (beyond-window) recall — a
decisive PASS of the kill criterion. This is also the re-run that resolves the
prior caveat: the pre-cosine green (99.9%) is reconfirmed at 100% now that
`MemoryRead.forward`'s top-K selection is **cosine** (changed from dot-product in
Task 5 to match the decode-time `EpisodicStore.retrieve_cosine` / Hamming path).
`validation/logs/exp_d5_episodic.log`.

**Scale result (2026-08-05): VERIFIED — decisively.** The memory-enabled hybrid
(D512 / L8 / attn_every=3 / window=512 / **mem_every=2**, ~0.98B tokens, matched to
H-D4's shape) scored on `exp_needle_recall` (PMI; shuffled + untrained + argmax-spread
controls) across lengths 256 / 512 / 1024 / 4096. **The beyond-window depths that were
at chance for the mem-OFF hybrid lifted to 33–58%, while in-window stayed 100%:**

```
              d=0%   d=25%  d=50%  d=75%  d=100%      (mem-OFF hybrid, same depths)
   len 1024 | 58.3   58.3   41.7   100    100         16.7 / 16.7 / 8.3 / 91.7 / 100
   len 4096 | 41.7   33.3   41.7   58.3   100         at the floor, depths 0.0–0.75
```

Real recall, isolated by three independent controls: the **shuffled-needle** control
(same model, true signal scrambled out) sits at the **~0–8% floor** — *below* chance,
i.e. the trained model actively rejects the wrong continuation; the **untrained model**
is uniform **16.7%** (chance on this setup); **argmax-spread** is diverse (8 winners /
240, top 17%) — not a degenerate single-token collapse. So beyond-window recall is
**2–3.5× the untrained chance and 4–7× the shuffled floor**, holds even at **8× the
window** (4096), and rises with proximity to the query (a real distance gradient — an
artifact would not respect distance). **The episodic memory carries recall past the
window at real scale — chance → ~50% beyond-window — while attention keeps in-window
recall perfect. Rung 0.0.5 complete.** Honest bound: it is a *learned, lossy* top-K read
on an undertrained (~0.98B-token) model, so ~40–58% (not attention's in-window 100%) is
the expected shape and should sharpen with training; the claim was "materially off the
floor," and 4–7× the floor across every beyond-window depth is emphatically that.
`validation/logs/needle_mem2.log`. (Running it needed one probe fix: `exp_needle_recall`'s
`_clone_bb` had to learn the memory decode-state shape — `9e07375`.)

**H-D2 — In-place Test-Time Training eliminates the KV-cache but is a compute-for-memory trade, not a free win.** The
blueprint proposes repurposing FFN projection layers as dynamically-updated fast weights during inference, updated via
local next-token-prediction gradients, to handle long context (128k tokens) without KV-cache memory-bandwidth cost.
Status: SPECULATIVE as a net win — TTT's per-token update is itself a gradient step, so inference compute per token goes
up even as cache memory goes down. **Validate** by measuring wall-clock/FLOPs per generated token with TTT-in-the-loop
versus standard KV-cache attention at matched context length — the blueprint only argues the memory side, not the
compute side, of this trade.

**Validation result (2026-07-23 campaign): compute side now measured (d=512, CPU, rank-1/NLMS-style update — the cheap
TTT variant).** TTT per-token cost is constant 0.253ms; KV-attention grows 0.545→210.4ms from L=1k→131k; measured
crossover **L≈1,024 (≈d)** — so on this shape the compute side *also* favors TTT for essentially all long-context
regimes, rather than opposing the memory win (1.0MB vs 536.9MB state at 128k). Two honest limits: full-backprop TTT
variants cost more than the rank-1 form measured, and — stated plainly — nothing here tests whether local-gradient fast
weights actually *retain* 128k tokens of usable context. The quality half still needs a trained model.
`validation/exp_d2_ttt.py`.

**H-D3 — HIGH SCRUTINY: the blueprint's ~2,250x (reduced to "~1,900x" after a 15% overhead haircut) total
energy-reduction claim should not be treated as a real number.** It's constructed by multiplying four
independently-sourced factors: ~90x from eliminating weight-loading via analog in-memory compute, 2.5x from
zip2zip-style sequence compaction, 5x from spike-based activation sparsity, 2x from softmax/KV-cache bypass. Status:
SPECULATIVE, and specifically flagged here because multiplying independently-sourced efficiency claims from four
different papers/contexts is not evidence they compose in one real system — savings mechanisms frequently target
overlapping bottlenecks (if weight-loading is already ~90x cheaper via analog compute, the marginal value of also
cutting softmax cost is not obviously still a clean independent 2x on top), and none of these four numbers were measured
together in a single working pipeline anywhere in the source material. **Validate only by building an integrated
small-scale prototype and measuring the composed number directly.** Treat 1,900–2,250x as a theoretical ceiling under
best-case independence assumptions, never as a planning number.
