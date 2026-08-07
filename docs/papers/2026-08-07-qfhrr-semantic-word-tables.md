# Semantic Block Codes: Distilling a Sentence Encoder into a qFHRR Word Table for Microsecond Retrieval and Routing

**Status: DRAFT — private working paper (CubbyLLM repo).** Publishing this (e.g. in the
public `grilly-next-paper` repo) discloses the method; that is a separate business
decision. Every number below links to a script in `validation/` and its captured log in
`validation/logs/` — nothing here is self-reported.

---

## Abstract

Vector Symbolic Architectures built on Fourier Holographic Reduced Representations
(FHRR) offer a complete compositional algebra — binding, bundling, exact unbinding —
but their hypervectors are random by construction, so two texts that *mean* the same
thing get orthogonal codes. Neural sentence encoders have the opposite profile:
semantic cosine geometry, but millisecond latency and no algebra. We show the two are
not in tension. We distill a sentence encoder (all-MiniLM-L6-v2) into a **static word
table over sparse block codes** — the quantized-phase form of FHRR (qFHRR) — via a
cosine-exact orthonormal projection, blended 0.5/0.5 with deterministic random phasor
identities. The result is a text encoder that is a pure table lookup: **no model in
the serving path**, ~7–25 µs per short text on one CPU core, while keeping the block
structure that FHRR binding and bundling operate on.

On a 15-domain routing screen the distilled table reaches **0.95× the teacher's
macro score** (0.341 vs 0.358; 0.86× under fully out-of-distribution calibration
statistics), against a measured matched null of 0.063. At production scale it routes
42 real corpus domains at **0.75× teacher** (0.595 vs 0.797 macro, 17.5× the
measured noise null). The argmax-discretized form of the same table stores a document
in **80 bytes** (one integer phase per block) and still retrieves needles from a
1M-document haystack at 0.71 hit@1 / 0.92 hit@20, encoding the full million-document
store in 9 minutes where the teacher needs ~30. A bundled-centroid routing rule with
a margin gate — using the similarity-*preserving* half of the VSA algebra for routing
and reserving the similarity-*destroying* half (binding) for composition — routes
open-set challenges with 92% precision while sending 87% of never-seen-domain inputs
to spawn. On multi-hop fact retrieval, iterative query expansion — a protocol that
measurably does nothing for the neural teacher — lifts the table's full-chain
recovery to parity with the teacher (0.733 vs 0.720), because retrieved facts
donate exactly the bridging words a bag-of-words query lacks. We report the
negative results with the same care: distributional (BEAGLE-style) co-occurrence
memory, frequency-weighted phase advancement, complex bundling, and a
product-quantization win that we retracted after finding codebook leakage.

---

## 1. Introduction

Two constraints collided in the CubbyLLM/MoWM architecture:

1. **The challenge path must contain no model.** Routing a challenge to a specialist
   world, deciding route-vs-spawn, and reading episodic memory happen on every
   input; a ~9 ms sentence-encoder call there dominates the entire budget.
2. **The representation must support the VSA algebra.** Worlds, axioms, and episodic
   memory in this architecture are sparse block codes (k=80 blocks × l=128 slots);
   whatever encodes text must land in that space, or nothing composes.

Random hypervectors satisfy (2) and fail semantics; sentence encoders have semantics
and fail both (1) and (2). The standard VSA answer — hand-designed feature bindings,
n-gram encoding, or distributional learning in the hypervector space — was screened
here and measured to fall far short of a neural teacher (§5.1, §6).

Our observation is that the gap closes with a **distillation, not a model**: embed a
*vocabulary* once with the teacher, offline; clean the anisotropy; project
cosine-exactly into the block space; add back a random phasor identity per word.
Serving is then a dictionary lookup plus an IDF-weighted sum — the oldest trick in
information retrieval, running inside a representation that binds and bundles.

Contributions:

- **A recipe** (§3) turning any text embedder into a qFHRR-compatible word table,
  with each design choice validated in an ablation arm (§5.1), including the ones
  that failed (§6).
- **The identification** (§3.2) that argmax-discretized sparse block codes *are*
  qFHRR with integer phases — so the compact form of our table is simultaneously an
  80-byte storage format, an ADC-scorable index, and a legal operand for phasor
  binding (mod-l index addition).
- **A measured quality/latency/scale envelope** (§5): 0.75–0.95× teacher quality
  at 350× single-text speedup (14–30× batched), with a scale curve to 1M documents
  and a two-stage compact-shortlist + dense-rerank design implied by it.
- **A routing rule** (§3.4, §5.3) derived from the algebra itself: bundled centroids
  for similarity, a best-minus-second margin gate for open-set spawn decisions.

---

## 2. Background

**FHRR and sparse block codes.** Plate's Holographic Reduced Representations in the
frequency domain (FHRR) represent symbols as vectors of unit phasors; binding is
element-wise complex multiplication (phase addition), bundling is complex summation,
and unbinding is exact via conjugation. Sparse block codes (NVSA-style) partition a
D-dimensional vector into k blocks of length l with a single active slot per block;
binding is circular convolution per block, which for one-hot blocks reduces to
**modular addition of the active indices**. Our substrate is the k=80, l=128 block
algebra of `grilly.experimental.vsa.block_ops.BlockCodeOps`.

**qFHRR.** Snyder, Poursiami and Parsa's qFHRR quantizes FHRR phases to K discrete
levels, making the whole algebra integer arithmetic. A one-hot block of length
l=128 *is* a phasor with 128 quantized phases: the active index is the phase, and
block-code binding (index addition mod l) is exactly qFHRR phasor multiplication.
Our earlier screen (`exp_f2m2` phase-index arms) measured K=128 as already lossless
at these dimensions — quantization is not where quality is lost.

**The semantic-similarity gap.** In every classical VSA construction the atomic
hypervectors are random, so *semantic* similarity exists only where the modeler
hand-encodes it. Learning it inside the hypervector space (BEAGLE-style
distributional memory) was measured here to smear rather than sharpen the signal
(§6). Meanwhile `model2vec` showed that a sentence encoder distills into a static
per-token table at a modest quality cost — but into an unstructured dense space with
no algebra. We take model2vec's distillation idea and land it in qFHRR space.

---

## 3. Method

### 3.1 Building the table

Given a teacher encoder T (all-MiniLM-L6-v2, 384-d) and a vocabulary V:

1. **Vocabulary.** Top-30k words by document frequency over windows of the
   pretraining corpus (`wiki_full` token cache, decoded through the project's
   BBPE-128k tokenizer), union the domain axiom vocabulary → **30,275 words**.
   Words, not BPE tokens: tokens-as-keys were measured consistently worse (§5.1) —
   BPE fragments do not carry the teacher's lexical semantics the way words do.
2. **Teacher pass (offline, once).** Embed each word: t_w = T(w), unit-normalized.
   ~5 minutes of CPU for the full vocabulary; this cost never recurs.
3. **Anisotropy removal.** Subtract the mean vector and project out the first
   principal component. The statistics are fit on a broad *external* vocabulary,
   not the evaluation corpus — the out-of-distribution check in §5.1 measures
   exactly what this choice costs and why it is the honest default.
4. **Cosine-exact projection.** A single orthonormal matrix Q (QR decomposition of
   a seeded Gaussian, seed 0) maps 384 → k·l = 10,240. Orthonormality preserves
   inner products exactly: no distortion enters here.
5. **Signal-code blend.** Each word also gets a deterministic random block code:
   BLAKE2b("w:&lt;word&gt;", 8-byte digest) seeds a generator that draws one active slot
   per block. The stored vector is the 0.5/0.5 blend of (4) and this identity,
   renormalized. The blend both sharpens exact-word matching (the +0.02 macro over
   pure distillation in §5.1) and gives **unseen words a well-defined code at serve
   time** — the same hash path, no table entry required.
6. **IDF weights.** Per-word inverse document frequencies from the same corpus
   statistics, stored alongside.

The shipped artifact is a 573 MB float16 `.npz` (30,275 × 80 × 128 plus IDF).

### 3.2 Serving

**Encoding** a text is: lowercase word split (len ≥ 2), table lookup per word
(hash fallback for misses), IDF-weighted sum, L2 normalization. Pure NumPy;
**6.8–25 µs** for short texts, ~0.5–1.1 ms for ~700-character passages — 350×
faster than the teacher single-text, 14–30× faster than the teacher *batched*, on
the same CPU (`exp_f2m2_m2v.json`, `exp_m3_haystack.json`, `exp_m3_domain_routing.json`).

**Two store forms** of the same encoding:

- **Dense** (k·l float16 = 20 KB/doc): full cosine geometry, the quality ceiling.
- **Compact** (argmax index per block = **80 bytes/doc**): the qFHRR integer-phase
  form. Scored ADC-style — the dense query gathered at each document's hot
  indices — which is ranking-equivalent to cosine against the one-hot store, needs
  no fitted codebook (hence no leakage risk), and is a legal operand for phasor
  binding. The measured cost of this discretization is the compact-vs-dense gap in
  §5.5, and it is recoverable: compact shortlist to top-20, dense re-rank.

### 3.3 What the blend actually is

The dense stored vector is *not* a one-hot block code — it is a point in the same
80×128 space whose geometry carries the teacher's semantics. Discretizing by
per-block argmax projects it onto the nearest legal qFHRR phasor code. So one table
yields both regimes: dense vectors when the task is similarity, integer phasors when
the task is storage, indexing, or algebraic composition. The two regimes agree to
within the measured gap (hit@1 0.978 dense vs 0.910 compact at N=0; §5.5).

### 3.4 Routing with the algebra: centroids and margins

The VSA algebra contains one similarity-*preserving* superposition (bundling) and
one similarity-*destroying* composition (binding). An earlier milestone measured
that a world's bind-chain signature is quasi-orthogonal to its own constituent
axioms — inert for routing. The routing rule that won the open-set screen (§5.3)
uses each operation for what it preserves:

- **Signature:** each world's *axiom centroid* — normalized mean (bundle) of its
  individual axiom vectors. One comparison per world (~40× fewer than
  max-over-axioms) and measurably better (0.680 vs 0.589 macro).
- **Gate:** route to the best world only if its score clears an absolute floor
  (τ_match) **and** beats the second-best by a margin (τ_margin); otherwise spawn a
  new specialist — the recoverable outcome. Both thresholds are
  deployment-calibrated; the screens are the calibration method, and no τ was
  observed to transfer across encoders or corpora.

---

## 4. Experimental discipline

All experiments are CPU-only (Windows 11, Python 3.12, NumPy 2.4.6), standalone
scripts, logs committed. Three rules held throughout:

1. **Measured nulls, not assumed ones.** Every screen runs its metrics over a
   pure-noise encoder and/or a matched analytic null (for argmax-over-candidates,
   micro Σ n_d(n_d−1)/(n(n−1)); macro mean (n_d−1)/(n−1)). Asymmetric max-based
   comparisons have nulls far below 0.5 — this project caught five separate
   instances of a result compared against the wrong reference before this rule
   became mechanical.
2. **Teacher anchors.** MiniLM runs through the *identical* metric code in every
   screen, so "×teacher" is never cross-paper arithmetic.
3. **Negative results are results** (§6), including one retraction.

---

## 5. Results

### 5.1 The distillation screen (15 domains, matched macro null 0.0633)

Route 280 held-out axiom texts to 15 domains by leave-one-out cosine
(`exp_f2m2_semantic_routing.py --fast-arms / --m2v / --m2v-ood`):

| Arm | Macro | ×null | µs/text |
|---|---|---|---|
| character 3-grams (baseline) | 0.140 | 2.2 | 313 |
| words + IDF (no semantics) | 0.173 | 2.7 | **19** |
| BPE token IDs + IDF | 0.152 | 2.4 | 29 |
| m2v tokens −PC1 + IDF | 0.204 | 3.2 | 28 |
| m2v words −PC1 + IDF | 0.320 | 5.1 | **6.8** |
| **m2v words −PC1 + signal blend (shipped)** | **0.341** | **5.4** | 25 |
| MiniLM teacher | 0.358 | 5.7 | ~9,000 |

The shipped arm is **0.95× teacher**. Every ingredient earns its place: words beat
tokens (+0.12 macro at equal recipe), anisotropy removal is the single largest gain
(+0.07), the signal blend adds +0.02 and the OOV fallback.

**Out-of-distribution honesty** (`--m2v-ood`): refitting IDF and PC1 on an external
20k-document corpus — the deployment condition — gives macro **0.308 (0.86×
teacher)**. The convenient in-domain statistics inflate the headline by ~0.03; we
report both and ship the external fit.

### 5.2 Production-scale routing (42 corpus domains)

`exp_m3_domain_routing.py`, real corpus passages (E:\datasets\domains), 40 exemplars
+ 40 challenges per domain, 28 closed-set domains scored, 5 held out as open set:

| | Macro | AUC | µs/passage |
|---|---|---|---|
| FastWord table | 0.595 | 0.568 | 1,105 |
| MiniLM (batched) | 0.797 | 0.794 | 15,175 |
| noise null (3 seeds) | 0.034 | — | — |

**0.75× teacher at 13.7× the batched speed**, 17.5× the measured null, on passages
far messier than the screen's axioms. The independently built table's own build-time
evaluation (different protocol, `fastword_table_v1.npz.build.log`) lands at 0.84×
teacher — same story from a second angle.

### 5.3 Open-set routing: centroid + margin (7 scorers × 3 rotated splits)

`exp_m3_domain_routing.py --open-screen`: which score function best separates
"routable to a known domain" from "never-seen domain, should spawn"?

| Scorer | routed-acc | spawn-detect | AUC |
|---|---|---|---|
| max-over-exemplars / top-1 | 0.611 | 0.650 | 0.654 |
| top-k mean / margin | 0.886 | 0.842 | 0.657 |
| **centroid / margin** | **0.924** | **0.869** | 0.666 |
| MiniLM (anchor) | — | — | 0.825 |

The winner routes 68% of closed-set challenges with **92.4% precision** and sends
**86.9%** of open-set challenges to spawn. This is the rule now wired into the
production bridge (`mowm/bridges/cubby_bridge.py`), with the margin-tie and
single-world edge cases covered by tests. The AUC gap to the teacher (0.666 vs
0.825) is the honest limitation of a bag-of-words confidence signal — see §7.

### 5.4 Question→answer retrieval (science QA; small-n caveat)

`exp_m3_science_qa.py` on KonstantyM/science_qa_prep. Only 238 of 4.28M rows carry
the tagged format; the usable slice is n=60 pairs over 3 tags — directional only:

table hit@1 **0.667** / MRR 0.754 vs teacher 0.95 / 0.975 (**0.70× teacher** on
retrieval; noise null hit@1 0.017). Domain routing *ties* the teacher at 0.90
macro (null 0.32). Retrieval of full answer texts leans on semantics more than
routing does — consistent with §5.5's dense-form results.

### 5.5 Scale: needle-in-haystack to 1M documents

`exp_m3_haystack.py`: 400 multi-hop questions, their 636 deduped supporting facts
as targets, DBpedia abstracts as distractors. Any-supporting-fact hit@1:

| Store size | dense (20 KB/doc) | compact (80 B/doc) |
|---|---|---|
| 0 | 0.978 | 0.910 |
| 100k | 0.973 | 0.853 |
| 1M | (infeasible RAM) | 0.713 — **hit@20 0.920** |

MiniLM anchor at N=0: 0.993. The dense form is essentially flat to 100k
distractors. The compact form degrades gracefully and still holds 0.92 hit@20 at a
**million** documents from **80 bytes each** — while encoding the 1M store took
9 minutes (543 µs/doc) against ~30 minutes for the batched teacher. The two-stage
design follows directly: compact ADC shortlist (top-20), dense re-rank of 20.

### 5.6 Fact injection and multi-hop chase

`exp_m3_injection.py` on a 10k-question fact-injection dataset (each question ships
its supporting chain, one designated fact-to-inject — always hop 1 or 2 — and an
answer that always lives in the final fact; all verified 10000/10000). Store: 1,241
deduped facts (+100k DBpedia distractors in the second sweep). 800 questions
(385/256/159 across 1/2/3 hops).

Post-injection accessibility at N=0: the injected fact is top-5 for **99.5%** of
1-hop questions but only **28.9% / 22.6%** of 2/3-hop questions — the measured
surface-form cliff (injected intermediate-hop facts share only ~55% of their words
with the question, vs ~75% for 1-hop). The teacher sits on the same cliff (hit@5
by hop 1.00 / 0.34 / 0.30): intermediate hops are hard because they are *lexically
and semantically* disjoint from the question, not because the table is weak — on
overall injected-fact accessibility the table is again **0.95× teacher**
(hit@5 0.616 vs 0.649). Adding 100k distractors moves the table's inject hit@1
from 0.481 to 0.480 (chance hit@20 there: 0.0002) — graceful at scale, consistent
with §5.5.

**Iterative chase** — n_hop rounds of top-1, re-encoding the query with each
retrieved text appended (still pure table lookups) — lifts the table's full-chain
recovery from 0.665 to **0.733** overall (3-hop: 0.195 → **0.371**, a paired ~1.9×
on the same questions), despite the chase budget being only n_hop picks against
single-shot's top-20. The same-budget comparison is starker: chain coverage 0.855
(chase) vs 0.743 (single-shot at n_hop picks). At 100k distractors chase holds
0.706.

The instructive contrast: **the identical chase protocol does nothing for the
teacher** (full-chain 0.719 single-shot → 0.720 chased), leaving the chased table
at parity with the chased teacher (0.733 vs 0.720; per-hop differences are within
noise at n=159 for 3-hop). Query expansion is the natural multi-hop mechanism for
a bag-of-words encoder — each retrieved fact donates exactly the bridging entity
words the question lacks — whereas appending text to a sentence-encoder query
dilutes its embedding as much as it informs it. The load-bearing claims are the
table's own paired improvement and the protocol asymmetry, not per-hop superiority
over the teacher.

### 5.7 Temporal worlds, causal events, and inter-world query

Modeling time as *worlds* — each era and each year a world, routed by bundled
centroid — was screened on two corpora (`exp_m3_temporal_causal.py`,
`exp_m3_nyt_years.py`). On 4,000 curated dated events (six named eras, years
with BC negative), the table routes eras at 0.669 macro (0.94× teacher; null
0.163) and places events in their exact year at **0.735** over 239 year-worlds
(0.98× teacher; noise-null MAE median ~730 years vs the table's 0). A
digit-masked arm is identical (0.735) — placement is content, not
date-reading. Year-world centroids decay monotonically in similarity with
temporal distance (0.342 at Δ<10y → 0.251 at Δ>500y): "the world was not the
same from one year to the next" is measurably in the representation. On
14,700 real NYT abstracts (147 year-worlds, 1852–2024, timeless content
included) **the table beats the teacher on every year metric** — MAE median
19y vs 21y (null 50.7) — a third instance of dating and routing being
surface-vocabulary tasks.

Causal structure: each event's stated consequence (`impact`) is retrieved
from among all 4,000 at 0.337 hit@1 / 0.631 hit@5 (0.75× teacher; chance
0.00025), and consequence texts route to *later* year-worlds than their
events at 55.2% vs a matched null of 44.6% — in both encoders: causality
points forward, measurably.

Two-level routing is a real trade, not a free win: gating year-choice behind
an era gate costs ~19 points of exact-year accuracy (identically for the
teacher) at 4× fewer comparisons. The repair is **inter-world query**: a
consulted world answers only if its best member clears τ, else the query
passes to the next-best world. At τ=0.6 this matches exhaustive flat
search's answer-found rate at **63 vs 3,783 member comparisons** (60×
cheaper, 2% of queries delegated). Two more leakage variants were caught and
excluded en route: paraphrase-duplicate events inflating row-level LOO
(exact-year 0.890 → honest 0.735 under event-level exclusion), and full
centroids letting a challenge's own stored vector nudge world ranking (on
NYT, worth 0.064 → 0.372 exact on its own). As with every threshold in this
paper, τ is corpus-specific: the same τ=0.6 that delegates 2% of curated
queries delegates ~99% of NYT queries.

### 5.8 Released-baseline head-to-head, the teacher ladder, and where the method ends

**model2vec/potion, same protocols, same machine**
(`exp_m3_potion_baseline.py`, `exp_m3_teacher_ladder.py`). The released
potion-base-8M beats our MiniLM-taught table on 42-domain routing (0.722 vs
0.595 macro) and loses NYT year placement (0.044 vs 0.064 exact). Distilling
*potion itself* into block space through the identical recipe answers the
attribution question: routing recovers to 0.646 — the block-code target
space costs nothing (the projection is cosine-exact); the gap was teacher
strength plus vocabulary coverage. A blend sweep confirms the 0.5
signal-code blend optimal (0/0.25 lose on both screens): exact-surface
sharpness is load-bearing even under a strong teacher.

**The teacher ladder.** Every teacher distilled into block space through the
identical recipe; native endpoints for reference:

| teacher → block space | teacher dim | routing macro | NYT exact | NYT MAE med |
|---|---|---|---|---|
| bge-m3 (GGUF Q8, direct) | 1024 | 0.566 | 0.070 | 17 |
| all-MiniLM-L6-v2 (= v1 table) | 384 | 0.595 | 0.064 | 19 |
| potion-multilingual-128M | 256 | 0.622 | 0.069 | 17 |
| potion-base-8M (= v2 table) | 256 | 0.646 | 0.069 | 17 |
| static-retrieval-mrl-en-v1 | 1024 | 0.670 | **0.081** | 15 |
| **potion-retrieval-32M (= v3 table)** | 512 | **0.680** | 0.080 | **15** |
| *potion-base-8M, native (no block space)* | 256 | *0.722* | *0.044* | *22* |
| *all-MiniLM-L6-v2, live (the ceiling)* | 384 | *0.797* | *0.046* | *21* |

Raw **bge-m3 is a measured negative**: a far stronger sentence embedder is a
*worse* word-table teacher than MiniLM, because single words are not its
input regime — while its Tokenlearn-compiled static form
(potion-multilingual-128M, whose teacher IS bge-m3) jumps to 0.622.
Compilation into static token vectors is the load-bearing step; retrieval
tuning stacks on top. Lesson: pick teachers by *static-token-vector*
quality, never sentence-benchmark rank. Note also the NYT column: every
block-space table out-dates both native models — the word+IDF+block recipe
is the better serving form for temporal placement regardless of teacher.

**Vocabulary scaling** (potion-retrieval-32M teacher, same recipe):

| vocab | routing macro | NYT exact / MAE med | axiom screen | table size |
|---|---|---|---|---|
| 30k (v3) | 0.680 | 0.080 / 15 | 0.313 | 573 MB |
| **60k (v4, shipped)** | **0.699** | 0.076 / 15 | 0.309 | 1.1 GB |
| 100k | 0.705 (±0.014 noise) | 0.076 / 15 | 0.310 | 1.9 GB |
| *teacher native ceiling* | *0.722* | — | — | — |

Coverage was most of the residual gap and saturates at 60k: the shipped
table (`fastword_table_v4`, 60,151 words, ~33 µs/challenge) sits at **97% of
its teacher's routing quality with the algebra and the 80-byte form intact**.

**Where the method ends — BEIR dbpedia-entity, full 4.64M corpus**
(`exp_m3_beir_dbpedia.py`; graded nDCG@10, all 43,515 judged docs present).
Open-web *entity* search with short keyword queries is the measured boundary:

| arm | store form | nDCG@10 | recall@100 |
|---|---|---|---|
| table v1, compact only | 80 B/doc (371 MB) | 0.040 | 0.143 |
| table v1, two-stage | compact + dense rerank | 0.081 | 0.143 |
| potion-8M, dense | 256-d f16 (2.4 GB) | 0.224 | 0.352 |
| *BM25 (published, unverified)* | inverted index | *~0.313* | — |

Everything static loses to lexical BM25 here, and the word table loses
worst: entity names are exactly the tail vocabulary a corpus-DF word list
misses, and the compact shortlist's 0.143 recall@100 caps the pipeline.
(The run used the weakest table (v1) — it predates the ladder; a v4 rerun
is in flight and will be recorded here, but a 3× gap will not close on
teacher quality alone.) The table's validated domain is routing, temporal
placement, and known-corpus retrieval — not open-vocabulary web entity
search.

**The cascade, measured — and no longer needed for its original purpose**
(`exp_m3_cascade.py`; v4 table + MiniLM, teacher consulted only inside an
ambiguity band of width δ around the route/spawn decision boundary):

| δ | teacher calls | spawn detect | closed routed | routed precision | est. µs/challenge |
|---|---|---|---|---|---|
| 0 (pure table) | 0% | **0.956** | 0.355 | **0.935** | 1,284 |
| 0.01 | 15% | 0.956 | 0.372 | 0.928 | 3,736 |
| 0.05 | 61% | 0.894 | 0.430 | 0.925 | 11,130 |
| ∞ (pure teacher) | 100% | 0.869 | 0.539 | 0.930 | 17,546 |

The confidence gap that motivated the cascade closed at the source: on the
same split and statistic, v4's closed-vs-open AUC is **0.690 vs the
teacher's 0.707** (the earlier 0.666-vs-0.825 comparison was the v1 table
on a different protocol). The pure-table point is the *most conservative
and most precise* operating profile; escalation trades spawn detection
*down* for routing coverage at up to 13× latency. Verdict: the cascade is
a **coverage lever** for deployments that prefer routing over spawning —
not a confidence fix, because the teacher ladder already delivered that.

---

## 6. What did not work (measured)

- **BEAGLE-style distributional memory.** Accumulating co-occurrence structure into
  hypervectors (raw / log / PPMI-weighted, 200M-token corpus) scored *below* the
  signal-only baseline (macro 0.064–0.083 vs 0.113): superposition smears the
  surface signal faster than co-occurrence adds semantics at this scale.
  (`exp_f2m2_beagle.json`)
- **Frequency-weighted phase advancement** (rotating codes by corpus frequency):
  amplifies common grams; strictly worse.
- **Complex-domain bundling** measured indistinguishable from argmax bundling —
  consistent with the qFHRR losslessness observation (§2).
- **Per-domain z-normalized calibration** of routing scores: no gain over raw
  cosine with a global τ.
- **"Product quantization beats dense" — retracted.** An early result showed the
  PQ store outscoring dense; the fitted codebook had seen the leave-one-out-excluded
  items. With the leak controlled the win vanished. The ADC form we ship uses **no
  fitted codebook** for exactly this reason. (PQ remains viable at n ≫ l·k axioms;
  not our regime.)
- **BPE token IDs as table keys** (0.152 vs 0.173 words even without distillation):
  subword fragments don't carry lexical semantics; the tokenizer's vocabulary is
  the wrong unit for a semantic table.
- **bge-m3 as a direct word-table teacher** (0.566 routing — below MiniLM's
  0.595): sentence-embedder strength is the wrong selection axis; the same
  model's knowledge works once Tokenlearn-compiled to static form (§5.8).
- **Open-vocabulary web entity search** (BEIR dbpedia-entity, §5.8): the
  word table's weakest measured regime — 0.081 nDCG@10 vs published BM25's
  ~0.313; tail entity vocabulary defeats a corpus-DF word list.

---

## 7. Limitations

1. **Confidence was the gap — and the teacher upgrade closed it.** The v1
   table separated known from novel less cleanly than the teacher (open-set
   AUC 0.666 vs 0.825); with v4 the same-split, same-statistic comparison is
   0.690 vs 0.707, and the measured cascade (§5.8) showed escalation no
   longer buys confidence — only routing coverage. The residual limitation
   is that all such τ-based profiles remain deployment-calibrated.
2. **No word order.** The encoder is a weighted bag of words; every n-gram and
   phase-binding variant we measured cost more than it bought at these scales.
   Tasks where order is the signal will need the binding layer above the table.
3. **τ does not travel.** Every deployment (encoder × corpus) needs its own
   threshold calibration; we ship the screens as the calibration method, not the
   numbers.
4. **OOV words get identity, not meaning.** The hash fallback makes unseen words
   *distinct*, not *semantic*; the OOD arm (0.86× teacher) prices the whole
   distribution-shift story, but a domain built from vocabulary outside the 30k
   table degrades toward the words-only arm.
5. **Scale of validation.** 42 domains, 1M documents, 10k questions — all
   single-machine screens. No claim survives past the regimes measured here;
   in particular the science-QA slice is n=60.
6. **English, lowercase, len≥2 word splitting.** Untested beyond it.

---

## 8. Related work

**model2vec / potion** distill sentence encoders into static token tables (our
steps 1–3 follow their recipe); we differ in the target space (structured qFHRR
blocks, not free dense space), the signal-code blend, and the resulting
algebra/storage properties. The head-to-head is now measured (§5.8): the
recipe is teacher-portable, potion-retrieval-32M is the best teacher found,
and the shipped table reaches 97% of its teacher's routing quality while
keeping the algebra. **BEAGLE** (Jones & Mewhort) is the classic distributional
hypervector memory — measured negative here at our scale. **NVSA block codes**
(Hersche et al.) and **qFHRR** (Snyder, Poursiami, Parsa) supply the substrate and
its integer-phase reading; our contribution is putting distilled semantic geometry
*inside* that substrate. **PQ/ADC** (Jégou et al.) inspired the compact scoring
path; our one-hot blocks make the codebook free and leakage-proof.

---

## 9. Conclusion

"Random hypervectors have no semantics" was a modeling choice, not a law. A
sentence encoder distilled through a cosine-exact orthonormal projection puts real
semantic geometry into qFHRR phasor space at 0.75–0.95× the teacher's measured
quality; a 0.5/0.5 blend with hashed phasor identities keeps exact-match sharpness
and OOV coverage; per-block argmax turns any encoding into an 80-byte integer-phase
code that scores ADC-style at millions of documents and binds like any other block
code. The serving path is a dictionary and a matmul. For architectures that route,
spawn, and remember on every input — and cannot afford a model call to do it —
that is the property that matters.

---

## Appendix A: Reproducibility map

| Claim | Script | Log |
|---|---|---|
| Fast-arms screen (§5.1 top half) | `validation/exp_f2m2_semantic_routing.py --fast-arms` | `logs/exp_f2m2_fast_arms.{log,json}` |
| Distillation arms (§5.1) | `… --m2v` | `logs/exp_f2m2_m2v.{log,json}` |
| OOD check (§5.1) | `… --m2v-ood` | `logs/exp_f2m2_m2v_ood.{log,json}` |
| BEAGLE negative (§6) | `… --beagle` | `logs/exp_f2m2_beagle.{log,json}` |
| Teacher anchor (§5.1) | `… --stage1 / --stage2 ref` | `logs/exp_f2m2_stage1.json`, `logs/exp_f2m2_stage2_ref.json` |
| 42-domain routing (§5.2) | `validation/exp_m3_domain_routing.py` | `logs/exp_m3_domain_routing.{log,json}` |
| Open-set scorer screen (§5.3) | `… --open-screen` | `logs/exp_m3_open_set.{log,json}` |
| Science QA (§5.4) | `validation/exp_m3_science_qa.py` | `logs/exp_m3_science_qa.{log,json}` |
| Haystack curve (§5.5) | `validation/exp_m3_haystack.py` | `logs/exp_m3_haystack.{log,json}` |
| Injection / chase (§5.6) | `validation/exp_m3_injection.py` | `logs/exp_m3_injection.{log,json}` |
| Temporal worlds / causal / delegation (§5.7) | `validation/exp_m3_temporal_causal.py` | `logs/exp_m3_temporal_causal.{log,json}` |
| NYT year-worlds (§5.7) | `validation/exp_m3_nyt_years.py` | `logs/exp_m3_nyt_years.{log,json}` |
| potion head-to-head + blend (§5.8) | `validation/exp_m3_potion_baseline.py` | `logs/exp_m3_potion_baseline.{log,json}`, `logs/exp_m3_potion_blend0.log` |
| Teacher ladder incl. bge-m3 negative (§5.8) | `validation/exp_m3_teacher_ladder.py` | `logs/exp_m3_teacher_ladder.{log,json}`, `logs/exp_m3_bgem3_block_arm.log` |
| Vocab scaling 30k/60k/100k (§5.8) | `mowm/scripts/build_fastword_table.py --top-words` | `logs/exp_m3_table_{v2,60k,100k}_validation.log`, build logs beside the npz files on D: |
| BEIR dbpedia-entity (§5.8) | `validation/exp_m3_beir_dbpedia.py` | `logs/exp_m3_beir_dbpedia.{log,json}` (+ `_v4` when the rerun lands) |
| Cascade (§5.8) | `validation/exp_m3_cascade.py` | `logs/exp_m3_cascade.{log,json}` |
| Production table build | `mowm/scripts/build_fastword_table.py` | `D:\CUBBY-TRAINED-MODELS\fastword_table_v1.npz.build.log` |
| Bridge wiring + tests (§3.4) | `mowm/bridges/cubby_bridge.py` | `mowm/tests/test_cubby_bridge.py` (6 green) |

Environment: Windows 11, Python 3.12, NumPy 2.4.6, CPU-only. Teacher:
`sentence-transformers/all-MiniLM-L6-v2`. Substrate: `grilly` `BlockCodeOps`
(k=80, l=128; open source, PyPI).
