# CubbyLLM Hypothesis Validation Report — 2026-07-23

Full-campaign pass over `CUBBYLLM_HYPOTHESES.md` (all groups, §10 order).
Every number below traces to a script in `validation/` and its captured run in
`validation/logs/` (environment: Python 3.12.10, torch 2.10.0+cpu, numpy 2.4.6,
CPU only — per H-E3 no special hardware was used or needed). Claim-checking
against sibling repos was done by direct file reads with quoted evidence.

Statuses use the doc's own scale. **VERIFIED** = reproduced/measured here.
Verdict shorthand: ✅ supported · ⚠️ partially/with caveats · ❌ refuted/killed
· ⏸ not validatable cheaply (stated why).

---

## Scorecard (16 experiments, all CPU, all logged in `validation/logs/`)

| # | Hyp | Verdict | One-line result |
|---|---|---|---|
| H0 | central bet | ✅⚠️ | θ=f(c) removes forgetting — **only** hardened; naive is worse than baseline |
| H0b | H0 + H-C4 | ⚠️ | context *inference* is the bottleneck, not generation; online routers forget too |
| A1 | Hebbian C_ctx | ✅ | C_ctx=0 by construction; 94% empirical forgetting on the real layer |
| A3/A4 | SDM vs NLMS | ⚠️ | benchmark built; NLMS best capacity, SDM only order-agnostic, no zero-forget winner |
| A5 | DG sparse | ✅ | size-matched control proves sparsity (not params) buys 2× capacity |
| A6 | no sibling fix | ✅ | every sub-claim confirmed verbatim |
| B1/B2/B5 | binding algebra | ✅ | grilly IS the shared substrate; 3-way run → **BlockCodeOps chosen** |
| B3 | unbind placeholder | ❌→✅ | claim misattributed (VM code, not the head); conclusion survives |
| C1 | hyper-encoder | ⚠️ | quality clause fails at toy scale (surface-form); cost clause passes |
| C2 | LZW compression | ✅ | 22% free / 53–67% with persistent dict, on the real tokenizer |
| C3 | retrieval head | ✅ | 32× win at real shape; V=1M = 10–41GB → head *requires* the bypass |
| C4 | domain router | ✅ | 569,944 params, ran offline; role upgraded by H0b |
| C5 | economics | ✅ | unvalidated narrative confirmed + citation fix |
| C7 | codebook capacity | ✅ | direct readout fine to 1M; flat factored breaks; memory is the wall |
| D1 | backbone | ⚠️ | Q=K recurrence viable for text; MinGRU leads at small scale |
| D2 | TTT compute | ⚠️ | TTT constant-cost, wins past L≈d; retention quality untested |
| E3 | CPU-only | ✅ | whole campaign ran without special hardware |
| F1 | package layout | ✅ | `PACKAGE_LAYOUT.md` written, rules mapped to failures |
| G1 | vocab cost | ✅ | 65k→131k BPE cost ratio 1.00× — next step is ~5 min, not the lever |
| G2 | dated forgetting | ✅ | real correlated NYT keys cost 2–3× recall; ranking survives |
| G3 | corpus overlap | ✅ | factual/↔gbooks/ effectively disjoint (9 IDs); other issues stand |

**Three structural decisions, all now made:** vocab = **hybrid** (fixed core +
retrieval head + gated dynamic tail); binding algebra = **BlockCodeOps**;
memory = **hardened θ=f(c) + offline-pretrained router** (an online router
recreates the forgetting).

---

## Headline results

1. **H0 (central bet) — ✅ SUPPORTED at toy scale, with a sharp caveat.**
   A hardened θ=f(c) head shows **near-zero forgetting** across 6 sequential
   tasks (task-0 MSE 0.062→0.060) where the fixed-weight baseline degrades
   0.060→1.778 (worse than chance 0.99), and P5 probing confirms the context
   signal is load-bearing (wrong-context MSE 1.87 vs right-context 0.06).
   The caveat is load-bearing too: the **naive** hypernetwork WITHOUT the
   anti-drift countermeasure ends *worse than the baseline* (2.734) — the
   hypernetwork's own shared weights are a C_ctx=0 learner and interfere
   internally. H0 works **only as "generation + hardening,"** never as naive
   generation. (`exp_h0_gce.py`)

2. **H-B3 — ❌ the hypothesis doc's premise was MISATTRIBUTED (and the fix is
   good news).** cubby-lm's `vsa_binding_head.py` contains **no `UNBIND_ROLE`
   and no unbind mechanism at all** — it is a clean, well-engineered cosine
   readout head (continuous query → cosine vs fixed bipolar codebook). The
   dict-lookup `UNBIND_ROLE` lives in **cubemind's VM**
   (`reasoning/vm.py:602-606`, confirmed: `self._role_bindings[reg].get(role)`
   — never calls `bc.unbind`). "Build real unbind from scratch" stands, but it
   is VM/reasoning-side work, not a defect of the LM head.

3. **H-B5 — ✅ CONFIRMED: grilly is the shared production VSA substrate.**
   cubemind's `BlockCodes` is a thin fallback wrapper around
   `grilly.backend._bridge.blockcode_*` (Vulkan) →
   `grilly.experimental.vsa.block_ops.BlockCodeOps` (`block_codes.py:35-45,
   127-187`). grilly ships **all three candidate algebra families** —
   `BinaryOps` (bipolar, BLAKE-hashed vectors), `HolographicOps` (HRR/FFT),
   `BlockCodeOps` (IBM-NVSA sparse block codes) — plus a resonator. Group B's
   question collapses from "which of two prototypes do we build" to "which
   grilly family do we standardize on."

4. **Zero-Forgetting Stability Benchmark — built and run** (it existed nowhere
   in any sibling repo; cubby-concepts only *names* it). Results in H-A3/H-A4
   below. (`exp_a_forgetting.py`)

---

## Group A — memory & forgetting

### H-A1 — Hebbian layer runs at C_ctx≈0 → **VERIFIED** (both halves)
`exp_a1_drift.py` · `logs/exp_a1_drift.log`
- **Structural:** cubby-lm's real `HebbianGrowthLayer.forward(x)` takes no
  context input of any kind; `_update_basis` is dW = f(x, xWᵀ, W). C_ctx = 0
  **by construction** — no estimation needed, the context channel does not
  exist.
- **Empirical:** trained to EV(A)=0.97 on task A (sanger mode, calibrated),
  then B→C→D sequentially: growth OFF (shared basis) retains **EV 0.06 —
  forgot 94%** (catastrophic drift confirmed). Growth ON (neurogenesis K
  16→40) forgets 72% — mitigates, does not rescue.
- Two incidental findings: (a) the layer's **default `nonlinear` (y³) mode
  never tracks subspace structure** under a reconstruction metric (stays at
  chance ~0.06 for every lr/epoch tried — only `sanger` mode learns); (b) with
  grown non-orthogonal bases the layer's reconstruction can exceed 1.0
  (EV=1.62), i.e. its own residual metric becomes unreliable after growth.

### H-A2 — GCE-style generation removes forgetting → **✅ toy-scale VERIFIED (hardened form only)**
See H0 headline. Kill criterion checked: bypass did NOT survive hardening
(P5 separation 0.06 vs 1.87); compute cost of hypernetwork+hardening measured
at 4.1x baseline per step at toy scale (0.41→1.70 ms) and grows with number of
protected contexts — the "fast" half needs watching at scale, exactly as H0's
kill criterion anticipates.

### H-A3 — SDM more forgetting-resistant than dense Hebbian → **⚠️ CONFIRMED in kind, REFUTED as a capacity claim at parameter parity**
`exp_a_forgetting.py` · `logs/exp_a_forgetting.log` — the benchmark
cubby-concepts names but never implemented, now real: D=2048, parameter-matched
(each method holds ~D² numbers; SDM M=D), online single-pass sequential
key→value writes, recall@cos>0.9, wall 1197.5s.

Final recall vs N associations:

| N | Hebbian λ=0 (Hopfield) | Hebbian λ=0.02 (flat decay) | NLMS | SDM (M=2048) |
|---:|---:|---:|---:|---:|
| 128 | 100% | 65.6% | 100% | 99.2% |
| 256 | 100% | 33.6% | 100% | 96.1% |
| 512 | 100% | 16.4% | 100% | 76.6% |
| 1024 | **0%** | 8.2% | 50.7% | 15.4% |
| 2048 | 0% | 4.1% | 17.3% | 0% |

Recency profile at N=1024 (recall by write-order quartile, oldest→newest):
Hopfield **0/0/0/0** (overload destroys *everything*, old and new); flat-decay
0/0/0/32 (only newest survive); NLMS 0/11/93/100 (strong recency bias);
SDM **15/16/15/16 — the only order-agnostic method**.

Reading: SDM's *graceful, fair* degradation — the actual "by construction"
claim — is real. Its *capacity* at matched state is the lowest of the viable
three (cubby-concepts' own 100%@1000 result used M=D=10000, ~5× the state per
stored pair used here). And the flat-decay Hebbian — the exact form running in
GrillCheese production (H-A6) — is the worst curve measured. **No candidate
achieves zero forgetting — independent support for H0 being necessary, not
just sufficient.**

### H-A4 — NLMS a cheaper drop-in → **✅ SUPPORTED (in the never-tested shared-weight setting), with its failure mode now named**
Same benchmark, matrix NLMS (`w ← (1−l2)w + μ·e·xᵀ/(‖x‖²+ε)`, exactly
AURA's `nlms.py:196-198` rule generalized to the shared-weight setting AURA
never tested). Best capacity curve of all candidates, no catastrophic cliff —
but the strongest recency bias past capacity (oldest quartile 0%). As a cheap
upgrade over Oja/Hebbian: yes. As a zero-forgetting fix: no — it forgets
*oldest-first*, which for a memory layer may be acceptable (LRU-like) or not
(depends on design intent); that choice now has data behind it.

### H-A5 — episodic/sparse side-channel complements the fix → **✅ mechanism VERIFIED (at toy scale)**
`exp_a5_sparse.py` · `logs/exp_a5_sparse.log` — DG-style sparse expansion
(8× up-projection, 2% WTA) vs a **size-matched dense control** (isolates
sparsity from parameter count) over a Hebbian store, D=1024:

| N assoc | dense | dense-wide (control) | dg-sparse 2% |
|---:|---:|---:|---:|
| 256 | 100% | 98.8% | 100% |
| 512 | **0%** | **0%** | **100%** |
| 1024 | 0% | 0% | 0% |

The control is decisive: at N=512 the same-size dense W fails exactly like the
small one while the sparse code holds 100% — **pattern separation, not
parameter count, buys the ~2× capacity**. Note dense Hebbian doesn't degrade
gracefully — it cliffs (100%→0% between 256 and 512). Confirms H-A5's framing:
a real complement, not a substitute, for fixing the update rule.

### H-A6 — no sibling has a validated fix → **VERIFIED, every sub-claim**
Fact-checked by direct reads (agent report, quoted evidence):
- GrillCheese `gpu_brain.py:376`: `delta_w = learning_rate * correlation -
  weight_decay * weights` — **flat decay, no Oja `post²·W` term** ✓. The
  documented shape-mismatch bug is real but filed against the *caller*
  (`unified_brain.py:_update_hebbian_weights`, per
  `gcheese-faiss/CRITICAL_ISSUES.md:104-110`) — minor attribution fix.
- GrillCheese `learning/` package: **zero hits** for
  EWC/elastic/orthogonal/replay/consolidat/fisher ✓.
  `forreview/capsules/DECISION.md:241`: "Add EWC for continual learning" ✓.
- Novel_GNN_Arch `apply_outcome_update()`: duplicated near-verbatim in
  `capsule_cat_nuclear_test.py:102` and `capsule_extended_demo.py:108` (~38
  lines each, not ~250 — the doc's line count was high), demo data 2 and 10
  hardcoded capsules, **no forgetting benchmark anywhere** ✓ (nuance: the
  nuclear test uses FIXED ±0.8 outcomes; only the extended demo randomizes).

## Group B — binding head

### H-B1 — cubby-concepts algebra usable, conditionally → **VERIFIED + condition resolved**
- Test suite independently re-reproduced **again** this session: **61/61
  passed, 55.1s** (`logs` in task output; previously 81.6s — both pass).
- The bipolar condition, measured (`exp_b_algebra.py`): sign() keeps
  cos 0.798 with the continuous query (= theoretical √(2/π)); in the
  **operating regime** (query cos=0.55 to target) binarized readout keeps
  **100.0%** top-1 target retention (worst case, zero-signal regime: 17.6% —
  reported for honesty, not the operating point). **Binarization is ~free
  where it matters** → the condition on H-B1 is satisfied.

### H-B2 — HRR is a second viable algebra → **VERIFIED, and the decision data exists now**
Head-to-head at D=10240 (`exp_b_algebra.py`): role-filler unbind from bundled
superpositions — both algebras ≥97% recall out to **n=320 bound pairs** (far
beyond any realistic head load); first separation at n=320 (HRR 100%, BSC 97%).
Deciding factor is not capacity: HRR handles the head's continuous query
natively; BSC needs the (measured-free) sign(). Note grilly ships both (H-B5)
**plus** block codes, which specifically target the continuous/bipolar tension
("prevents… zero-gradient problem of bipolar sign()", `block_ops.py` docstring).

### H-B3 — unbind is a placeholder → **❌ claim as written / ✅ conclusion**
Misattribution found (headline 2). The *conclusion* — real unbind must be
new-build — survives, relocated to the VM/bridge side.

### H-B4 — chained F=2, never flat F≥5 → **VERIFIED (re-run)**
`test_resonator_quadrillion_is_beyond_capacity` passes in the 61/61 re-run;
capacity law re-applied quantitatively in H-C7 below.

### H-B5 — grilly is the production substrate → **✅ CONFIRMED (was SPECULATIVE)**
Headline 3. Remaining open question shifts as the hypothesis predicted: not
"do the primitives work" but whether the SVC role-filler scheme generalizes to
CubbyLLM's token/concept binding — and now additionally **which of grilly's
three families** the binding head standardizes on.

### H0 part 2 — θ=f(c) with INFERRED context (the "automatic" half) → **⚠️ generation isn't the bottleneck; context INFERENCE is**
`exp_h0b_learned_context.py` · `logs/exp_h0b_learned_context.log` — the toy H0
win used the *true* task id. This replaces it with a learned router (context
inferred from x), tasks given separable input statistics so inference is
possible. Final task-0 MSE (chance 1.50, just-learned 0.10):

| model | final task-0 MSE | vs oracle | router slots used |
|---|---:|---:|---|
| baseline (fixed head) | 2.26 | 23× | — |
| **oracle-ctx** (true id) | **0.097** | 1× | — |
| router-naive (unsupervised) | 2.20 | 23× | **1/6 (collapsed)** |
| router-super (task-label routing, online) | 1.75 | 18× | 3/6, task-0 conf ↓ to 37% |

Two findings, both load-bearing for the roadmap:
1. **The unsupervised router collapses to a single slot** (slot 3 for all 6
   tasks) — confident routing ≠ *distinct* routing, so it degenerates into the
   fixed baseline and forgets identically. **Unsupervised context discovery is
   the hard part, not the θ=f(c) generation** the oracle already nailed.
2. **A router trained online is itself a C_ctx=0 sequential learner that
   forgets its own routing** — even with task-label supervision, its slots
   drift from distinct to 3/6 and task-0 routing confidence decays to 37% as
   later tasks are learned. It halves the forgetting but stays 18× the oracle.

**Actionable:** pair H0's hardened head with an **offline-pretrained** router
over all contexts — which is *exactly* what H-C4's `domain_head.pt` is (trained
once on 88 domains, not online). An online-learned router recreates the very
forgetting H0 solves for the head. This sharpens where the H-C4 baseline must
sit in the architecture: upstream, pretrained, frozen-at-inference.

## Group C — vocabulary / softmax bottleneck

### H-C1 — hyper-encoder embeddings enable large vocab → **⚠️ kill criterion HALF-triggered at toy scale**
`exp_c1_hyperencoder.py` · real spm32k stream from cubby-lm's valid.txt,
V=8k, d=64, identical budgets:
- **Quality clause: FAILS** for the pure surface-form instantiation — static
  table val loss 3.886 (ppl 48.7) vs char-CNN hyper-encoder 4.389 (ppl 80.6),
  **+0.50 nats**. Surface form alone cannot recover what a table memorizes.
- **Cost clause: PASSES decisively** — 1.26 µs/token full-table regen,
  cacheable at inference; embedding params V-independent (77k vs 578k total).
- Read with zip2zip's actual design (static base vocab + composed hypertokens),
  this points to a **hybrid**: static core + generated/composed tail, not
  pure generation. Re-check at real scale before betting the vocab strategy.

### H-C2 — LZW cuts steps 20–60% → **VERIFIED (measured on the real tokenizer)**
`exp_c2_lzw.py` · grillcheese_spm32k_v2 over 8MB of cubby-lm valid.txt
(2.07M base tokens): **per-document (streaming-honest) 22.2%**; persistent
global dictionary **53.4→66.6%** (32k→992k hypertokens) — the claimed range is
reproduced at its low end honestly and exceeded with a persistent dictionary.
The persistent-dictionary variant IS the H-C6 dynamic-vocab question in
disguise: 22% is free; the rest is bought with exactly the machinery H-C6
would have to decide to build.

### H-C3 — retrieval output head makes large vocab affordable → **✅ mechanism REQUIRED, bounds measured**
`exp_c3_latency.py` (CPU, batch=1): full-head cost is linear in V (measured
7.8× for 8× vocab at d=2560; 47.9ms/token at the real 32k×10240 shape). The
retrieval floor (`ideal-K`, K=1024) is **~flat at 0.2–1.5ms — a 32× win** at
the real shape. Naive IVF over *random* codewords is poor (37–80% top-1
agreement — the pessimistic bound; real embeddings cluster). The decisive
finding is memory: **V=1M fp32 codebook = 10–41GB** (d=2560/10240) — at large
V the softmax bypass isn't an optimization, it's the only way the head exists.
(1M numbers reported as explicit linear extrapolation, not measurement.)

### H-C4 — trained 88-domain router is the lowest-risk baseline → **VERIFIED, ran end-to-end**
Agent-verified + executed offline: `domain_head.pt` loads clean
(`weights_only=True`), exactly **569,944 params**, 1024→512→88 matching the
code; eval harness ran on CPU (metrics: exact-match, Hamming, micro/macro F1).
Two precision notes: outputs are **logits** (sigmoid lives downstream,
`BCEWithLogitsLoss` training), and the shipped dataset is a 3-record fixture
(`file_capsules.jsonl` is 0 bytes) — **no headline accuracy is reproducible
from the repo as shipped**; embedding new text needs `BAAI/bge-m3` (not in
local HF cache). Baseline: usable. Claimed metrics: unverifiable as shipped.

### H-C5 — economics are unvalidated narrative → **VERIFIED, plus a citation fix**
Agent-verified: the "$956M" Year-5 valuation is real
(`revenues-simulation.txt:57`; = 12× self-asserted Y5 revenue; every driver
unsourced; internal inconsistencies: narrative says $58.9M EBITDA, table says
74.93; malformed table). **Correction to the doc:** the "100–1000×" figure is
NOT in `vs-Kimmy-K3.txt` (which only says "a fraction of the cost") — it's in
`simulatorandprofitability.txt:138`, resting on unsourced round numbers
($15,000 vs $15.00 per 1M queries, 700W vs 20W). Verdict unchanged:
directional motivation only.

### H-C6 — fixed-large vs open/dynamic vocab → **decision package assembled (user call)**
Evidence gathered this session, all pointing the same direction:
- H-G1 (verified): trained history 19,947→32,000→65,536 — fixed-vocab
  pipeline is proven one order of magnitude up.
- H-C1: pure generation loses quality at toy scale; composition/caching is
  cheap.
- H-C2: 22% steps free without any vocab change; persistent hypertokens buy
  2–3× more but require dynamic-vocab machinery.
- H-C3/H-C7: direct readout survives to V=1M; memory is the wall; retrieval
  head required either way.
→ **DECIDED (user, end of this session): HYBRID** — fixed core (next step
128k–256k BPE on the existing pipeline) + retrieval output head from day one +
hypertoken/dynamic tail as a later, separately-gated addition.
Also decided the same session (Group B): **prototype all three grilly algebra
families** (BinaryOps / HolographicOps / BlockCodeOps) against a representative
codebook before standardizing.

## Continuation (same day): the 3-way prototype ran — duties matrix

`exp_b5_grilly_3way.py` · `logs/exp_b5_grilly_3way.log` (grilly's actual code,
D=10240 / k=80×l=128, V=32k; wall 377s):

| Duty | binary | hrr | block |
|---|---|---|---|
| Unbind roundtrip | exact 1.000 | approx 0.711 | exact 1.000 |
| Readout @32k, 5% signal | 84% | 86% | **100%** |
| Superposition n=320 | 96% | 100% | 100% |
| Trainability (2k steps) | 19.8% hard-STE / 99.6% tanh | 12.8%* | **96.3%** |
| Bind latency | **3µs** | 309µs | 186µs |
| Readout @32k | 36.7ms | 428.5ms | 37.2ms |

*probe caveat: shared untuned lr/temperature; HRR's number is likely partly
artifact — its approximate unbind and 11× readout cost are not.
Also: grilly BinaryOps.bind ≡ cubby-concepts bind (parity assert passed), and
the sign() zero-gradient claim is now MEASURED (19.8% vs 99.6% relaxed).
Data points at **BlockCodeOps** — also cubemind's existing substrate, with the
Vulkan path observed initializing on this machine's RX 6750 XT.

**DECIDED (user): the binding head standardizes on `BlockCodeOps`.**
Binary/HRR stay as reference implementations. Same continuation also ran the
H-G2 dated benchmark (ranking survives real correlated keys) and the H-G3
Gutenberg ID-overlap check (factual/↔gbooks/ effectively disjoint: 9 IDs,
~0.00GB — that sub-fear retired; the other confirmed unified/ issues stand).

### H-C7 — big vocab strains the codebook → **✅ RESOLVED with numbers (partly ❌ as feared)**
`exp_c7_capacity.py`: the fear was aimed at the wrong place. **Direct one-
codeword-per-token readout does NOT strain**: expected max crosstalk grows as
√(2lnV/D) — margin at cos=0.55 signal stays +0.50 at V=1M, empirically 100%
top-1. What DOES break is **flat factored identity**: at D=10240 per-slot
capacity is F=3: 65, F=4: 14 (verified law) → 1M vocab at F=3 needs 100/slot
(over), F≥4 hopeless; factored readout at 1M needs D≥24k (F=3). Real pressure
= codebook memory + O(V·D) compute (see H-C3). Chained F=2 (H-B4) remains the
escape hatch if factored identities are ever wanted.

## Group D — backbone

### H-D1 — backbone → **✅ RESOLVED: MinGRU (in-architecture bake-off)**
First a standalone char-LM screen (`exp_d1_backbone.py`, d=128, L=4, 2500 steps,
real valid.txt char stream, cubby-lm's ACTUAL MinGRULayer as the baseline):

| mixer | val bits/char | non-emb params | ms/step |
|---|---:|---:|---:|
| **mingru** (predecessor) | **1.454** | 592,512 | 183 |
| bdh-style Q=K recurrence | 1.584 | 591,488 | 118 |
| attn (softmax) | 2.470 | 656,512 | 78 |

Reading: an **attention-free Q=K recurrence IS viable for text** — it clearly
learns (1.584 bpc, far ahead of small attention's 2.470, which underperforms on
char-LM at this budget as expected) at lower per-step latency than MinGRU. But
**MinGRU (the predecessor's real code) is the best learner here** by 0.13 bpc.
Scope, stated plainly: this answers only H-D1's viability question ("does it
learn language at all, competitively?" — yes). It does NOT test BDH's 35×/10×
efficiency claims (those need scale + a faithful BDH impl; the asymptotic
compute trade is H-D2's job), and the bdh-style mixer here is a stylized
stand-in, not a faithful BDH.

Then the **deciding run** — `exp_d1b_backbone_bakeoff.py` · `logs/…` — four
candidates in the *real* assembled `CubbyModel` on the subword-TinyStories setup,
matched budget, every non-backbone component from the same per-component seed
(backbone = the only variable):

| backbone | perplexity | bpc | backbone params | ms/step |
|---|---:|---:|---:|---:|
| gru | 34.0 | 1.300 | 395,264 | 157 |
| **mingru** | 34.8 | **1.308** | 296,192 | 155 |
| bdh (Q=K) | 49.1 | 1.435 | 295,426 | 137 |
| attn | 48.0 | 1.427 | 328,192 | 135 |

**Two clean clusters:** gated recurrence (GRU, MinGRU) beats attention-free Q=K
and softmax attention by ~10% bpc — large, consistent, and matching the char-LM
screen. GRU vs MinGRU is within single-run noise, but MinGRU matches GRU's
quality at **25% fewer backbone params**, plus the log-domain parallel scan
(throughput at scale; GRU's kernel is sequential) and continuity with cubby-lm.
**Decision (user): MinGRU** — promoted to
`cubbyllm/model/backbone/mingru.py::MinGRUBackbone` (RMSNorm + MinGRU recurrence
+ SwiGLU trunk), replacing the reference stand-in; `base.py` stays interface-only
so alternates stay pluggable. BDH's 35×/10× claims remain untested (need scale +
a faithful BDH); at this scale BDH-Q=K is viable-but-not-winning.

### H-D2 — TTT is compute-for-memory, not free → **⚠️ compute side now measured**
`exp_d2_ttt.py` (d=512, CPU): TTT per-token cost is **constant 0.253ms**;
KV-attention grows 0.545→210.4ms from L=1k→131k; measured crossover **L≈1,024
(≈d)**, i.e. TTT wins on compute for essentially all long-context regimes on
this shape — the blueprint's memory argument (1.0MB vs 536.9MB at 128k) holds
AND the compute side, which it never argued, also favors TTT past L≈d rather
than opposing it (the rank-1/NLMS-style update variant; full-backprop TTT
variants cost more).
What remains open (stated plainly in the log): whether local-gradient fast
weights *retain* long-context information competitively — a quality question
needing a trained model.

### H-D3 — ~1,900–2,250× composed energy claim → **⏸ unvalidatable without an integrated prototype — treat as ceiling, never a plan number**
Unchanged; nothing cheap can validate a product of four independently-sourced
factors. Flag stands.

## Group E — hardware/economics

- **H-E1, H-E2 — ⏸ deferred by design** (custom silicon; projections
  downstream of it). Nothing validated, nothing needed.
- **H-E3 — ✅ VERIFIED by construction and by execution:** every experiment in
  this campaign ran on CPU (torch 2.10.0+cpu; the machine's AMD RX 6750 XT was
  unused by the experiments — no CUDA/ROCm torch — though grilly's Vulkan
  backend was observed initializing on it successfully, a good sign for the
  H-B5 GPU path). No Group A–D validation required special hardware. The
  sequencing decision holds.

## Group F — structure

- **H-F1 — ✅ validation step executed:** target layout written as
  `PACKAGE_LAYOUT.md` (draft/proposal), each rule mapped to a named observed
  failure in the siblings, including two new ones from this session (numbers
  must link their producing log; corpus manifests pinned per H-G3).
- **H-F2 — ⚠️ analysis, not yet closable:** what H0's validated mechanism
  implies for the interface: the bridges must carry (a) a context/task
  embedding (model→world and back), (b) novelty events (exists, one-way),
  (c) generated-parameter or specialist handles (doesn't exist anywhere).
  Neither current bridge carries (a) or (c); the duck-typed one-way
  `NoveltyToWorldBridge` and the subprocess CubeLang bridge both fail the
  "first-class interface" bar. Concrete definition belongs to design, after
  H-C6/backbone decisions — validated only in the sense that the *need* is now
  demonstrable from H0's requirements rather than assumed.

## Group G — training data (surveyed by the parallel session; spot-checked here)

Statuses were assigned VERIFIED by the parallel survey session; this session
independently **spot-checked** (not full re-verification):
- **H-G1 ✓ spot-checked:** all three trained tokenizers + `report.json`s
  present in `D:\grillcheese_training_data\tokenizer\`.
- **H-G2 ✅ DONE (continuation session, same day):** the dated harness ran —
  `exp_g2_nyt_forgetting.py`, 8 eras 1852→2024 × 128 real headlines,
  chronological writes through the same four rules. Real keys are 3.1× more
  correlated than random (|cos| 0.055 vs 0.018) and cost every candidate
  ~2–3× recall — but **the synthetic ranking survives unchanged**: NLMS 18%
  overall with a clean chronological forgetting curve (0%→61% oldest→newest
  era); SDM ~9–12% flat across 150 years (still the only order-agnostic
  method); flat-decay 24% newest-era-only; Hopfield 0%.
  `logs/exp_g2_nyt_forgetting.log`.
- **H-G3 — not re-verified here** (120GB sweep out of scope for spot-checks);
  its action items are encoded in `PACKAGE_LAYOUT.md` rules.
- **H-G4 ✓ spot-checked:** 1,319 lines, first record is canonical GSM8K.

---

## Corrections the hypotheses doc needs (all applied this session)

1. H-B3/§1 Problem 1: `UNBIND_ROLE` is cubemind VM code, not
   `vsa_binding_head.py` — the head has no unbind at all.
2. H-C5: cost-claim citation moved to `simulatorandprofitability.txt:138`.
3. H-A6: `apply_outcome_update` is ~38 lines ×2 copies (not ~250); GrillCheese
   shape-bug lives in `unified_brain.py` caller.
4. H-C4: "88 sigmoid outputs" → logits + downstream sigmoid; shipped repo has
   no real test set (headline metrics unreproducible as shipped).

## What would change these conclusions

- H0/H-A2: scale. The hardening cost grows with protected contexts; a real
  memory-layer integration could reintroduce bypass. Next: H0 on a small LM
  memory pathway with learned (not given) context — H-C4's router is the
  intended context source.
- H-A3/A4: different key statistics (correlated real text keys, not random
  bipolar) can reorder the candidates — the NYT-dated harness (H-G2) is the
  designed follow-up.
- H-C1: a context-conditioned (not surface-only) generator, or zip2zip-style
  composition over a static base, could flip the quality clause.
- H-C3: IVF numbers on *trained* embeddings (not random codewords) needed
  before committing to a specific index.
