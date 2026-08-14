# Model-panel research agenda (2026-08)

**What this is:** the scored record of a 27-question interrogation of five
frontier models (Gemini 3.1 Pro, GPT 5.6, DeepSeek V4, Qwen 3.8 Max, Kimi K3)
about CubbyLLM's open problems, run 2026-08-08. Each answer was scored against
this repo's MEASURED record; convergences were adopted, disagreements became
experiments, and one near-scoop (GAVEL) was caught. Questions Q1-Q14 covered
encoder/VSA/memory/CoT/routing/red-team/novelty; Q15 multi-future prediction;
Q16-Q19 facets/training/session-memory/debate; Q20 question grammar; Q21-Q23
calibration-negatives/partitioned-retrieval/harvest-schema; Q24-Q27 trunk
economics (vocab, MoE, split-compute training, hybrid ratios).

**The six paper-shaped holes** (all on existing infrastructure; #2/#4/#5
cheapest): 1 Bayesian bundle-collapse; 2 Tokenlearn-into-block-codes;
3 verifier-grounded heterogeneous debate (cite GAVEL + Irving/Christiano);
4 absence-vs-misrouting discrimination; 5 frozen-both-vocab-surfaces (cite
CWT/Headless LMs); 6 retrieval-for-attention substitution ablation.

**The experiment queue** (cost-ranked): E1 NYT masked/date-scrubbed dating
rerun; E2 capacity sweep (m x decoder x evidence-strength; panel predictions
on record: 2-3 / 8-16 / 5-15); E3 subword-fallback closed-form ridge;
E4 EVT/margin-normalized tau; E5 fused lexical block BEIR arm; E8 futures
kill-test (aWTA coverage@k); E9 debate headroom (single vs union acceptance);
E6 train-into-block-space; E7 H0 350M 3-arm. Plus the 2-hour local MFU pilot
(unanimous highest-EV action before any GPU rental).

**2B runbook (settled 3/3):** MFU pilot -> cycle-one DENSE hybrid at 1:3
(correct for weak recurrence per Griffin), WSD schedule, full-vocab cosine-CE,
frozen-embedding 3-arm test at 151M first, curriculum-MTP arm at pilot; rent
single-node A100s (~$1.3-1.9k spot), stream shards, checkpoint 15-30 min;
after: offline ULD-KD subset pass, int8 PTQ, MQAR probe battery; cycle two:
upcycle to fine-grained tag-initialized MoE (Nexus router fed by our static
table), Zamba-style shared attention block.

Raw scored notes follow (chronological, per panelist).

---

# Model panel — running notes (4 models on CubbyLLM open questions)

## Gemini 3.1 Pro

### Directly refuted by our data
- Q13 confound #1 "newspaper dating = date-token leakage": the CURATED screen's
  digit-masked arm was identical (0.735 vs 0.735) — already closed there.
  BUT: the NYT screen has NO masked arm → the critique stands for the NYT
  claim specifically. ACTION: run NYT digit-masked arm (cheap).

### Validates our measured choices
- Q11 ratio confidence (1st/2nd distance) == our margin gate; "static tau fails
  as worlds grow" == our tau-non-transfer finding. HNSW-over-centroids for scale.
- Q7 router drift: OOD-spawn + async re-pretrain w/ distillation == our
  spawn-recoverable + offline-frozen router shape.
- Q12 transition model: JEPA/InfoNCE not generative; collapse warning == our
  H-B6 Goodhart collapse, lived and measured.
- Q8 trunk-SFT: constrained decoding mandatory; strip NL; hop curriculum
  (single-hop -> n-hop); small models don't generalize deeper than trained.

### New + actionable
- Q1 OOV: BPEmb-style subword-piece table as FALLBACK-only (not keys) + bloom/
  hash-trick embeddings. Distinct from our measured tokens-as-keys negative.
  ACTION: subword-fallback arm for the 0.699->0.722 gap.
- Q2 BEIR hybrid: append 16-byte IDF-weighted Bloom of rare tokens per doc;
  score = dense + lambda*popcount(AND). Fits the 80B form (would be 96B).
  ACTION: future arm.
- Q3 JPQ (Zhan et al.) / STE distill-into-codebook — literature for related
  work; note our cosine-exact QR means the space costs ~nothing already
  (measured attribution), so the win would be in the BLEND/argmax stage.
- Q4 resonator capacity C ~ D/(K log V); k-WTA cleanup for SBC (H-B4).
- Q5 crosstalk breakdown n~15-25 at l=128 — consistent with I-RAVEN plateau
  datum (9 objects at d=2048).
- Q10 calibration: isotonic step-overfits at n<1000 → our n=200 slice should
  use Platt or plain thresholds (Task 5 already uses Youden threshold — fine).
- Q6: chunked hypernetworks (von Oswald) beat EWC sub-1B; OGD basis O(N·D^2).

## GPT 5.6 sol

### DISAGREEMENT with Gemini (resolvable by our experiment)
- Q5 block-code capacity: Gemini says breakdown at n~15-25 bundled pairs
  (l=128); GPT derives R <~ KL/(4 log M) => 185-370 with whole-codebook
  cleanup, "look closely between R=32 and 256", and notes per-block
  independent WTA decoding is MUCH worse (likely the source of Gemini's low
  number). Our I-RAVEN datum (plateau ~9 at d=2048) is ~8x below GPT's bound
  => real-world constants matter. ACTION: capacity sweep at 80x128 (random /
  learned / correlated codebooks) — cheap, resolves the panel split, feeds
  H-B4 and the paper.

### Corrects Gemini (and matches our measured bottleneck)
- Q2: an in-payload lexical fingerprint can only RERANK candidates dense ANN
  already surfaced — cannot fix lexical RECALL (our compact shortlist
  recall@100 0.135 IS the bottleneck; rerank can't recover missed docs).
  GPT's engineering answer: 4x32-bit high-IDF hashes as keys into a tiny
  compressed postings SIDECAR (a micro inverted index for rare terms only).
  Cites DHR (Lin&Ma 2022), SPAR.

### Richer than Gemini / new
- Q1: Bag of Subwords (Zhao 2018) = exactly our setting (additive substring
  vectors reproducing an existing word table); PBoS; T-FREE hashed trigrams.
  Recipe: artificial-OOV holdouts, boundary symbols, IDF-weighted recon,
  preserve exact vectors for names/dates; eval split by in-vocab/OOV/
  morphology/typos/entities. Honest ceiling: composition can't identify
  opaque entities (consistent with our BEIR boundary).
- Q3: DPQ/RepCONC/Poeem/Contrastive-PQ + curriculum (kmeans init -> soft ->
  anneal -> hard -> fine-tune under serving ops) + code-balance loss.
- Q4: decompose the problem (known-role retrieval != set decomposition !=
  resonator factorization); build MATCHING PURSUIT + softmax cleanup before
  resonators for additive bundles (H-B4 build order).
- Q6 H0: "no established 100M-2B scaling result vs replay/OGD/EWC —
  genuinely open" => our H0-at-scale is a real contribution slot.
  Approximations: protect adapters only, sketch generated params, sample
  contexts, low-rank deltas over frozen base (== MindForge direction).
- Q7: frozen encoder + mutable prototype registry == our FrozenSlotRouter +
  spawn; "routing drift" in incremental MoE; monitor list (fallback rate,
  margins, PSI/MMD, router-vs-member disagreement).
- Q8: verify round-trip of serialized programs; init opcode embeddings from
  related tokens; TRACE prediction as denser supervision; split eval by
  program template/length; rejection sampling + DPO/RLVR with executable
  rewards.
- Q10: do NOT calibrate steps independently and multiply; answer-level
  features (min step score is the key one — Stengel-Eskin & Van Durme 2023);
  ridge logistic at small n; both models agree: avoid isotonic at our n.
- Q12: consequence MULTIMODALITY (point predictors average futures) ->
  mixture/set losses; hard negatives incl. same-event-wrong-time.
- Q13 additions: bootstrap CIs; second-teacher rerun (we already have the
  ladder = multi-teacher); source-held-out splits (NYT is single-source —
  note as limitation); duplicate/near-dup splits (we caught this ourselves,
  event-level exclusion).

### Paper framing gift (adopt)
- Q14 narrower claim, near-verbatim our contribution: "VSA-compatible atomic
  vectors need not trade semantic geometry for algebraic structure:
  distilled semantic vectors can live directly inside a sparse block-coded
  space while retaining binding and cleanup." Novelty = the CONJUNCTION
  (teacher distillation + no model at serve + structured sparse geometry +
  retained VSA algebra + strict byte budget + measured envelope), not
  "non-random hypervectors".

### GPT's experiment priority (mostly matches ours)
1 subword residual gap-closer; 2 constrained-space training vs projection;
3 capacity sweep; 4 matching-pursuit unbind; 5 64B PQ + 16B lexical
fingerprint on DBpedia; 6 H0 hardening curves.

## DeepSeek V4

### Resolves the capacity dispute (decoder-conditioned)
- Q5: THREE-way split is actually agreement conditioned on decoder:
  naive per-block argmax reliable only at R~1-2 (per-block success 0.37 @R=2,
  0.15 @R=3; crosstalk terms are THEMSELVES valid one-hot codes — full-
  correlation distractors, not noise); bounded-codebook AGREEMENT scoring
  (sum of per-block counts for candidate) + SIC/matching-pursuit ->
  R ~ kl/(2 ln m) ~ 10^3 @ m=128 (matches GPT); Gemini's 15-25 = middle
  (k-WTA-ish). OUR DATA FITS: cubelang recover = cosine cleanup vs symbol
  table (= agreement decode) works at R=2 with sim ~0.5 (live smoke);
  I-RAVEN plateau 9 @ d=2048. ACTION: capacity sweep must sweep DECODERS x R
  (argmax / agreement / agreement+SIC), not just R. BCF (arXiv 2303.13957)
  = sparse-block-code factorizer SOTA for H-B4.

### Panel-synthesis insight (only visible combining GPT + DeepSeek)
- Q2: GPT's recall objection (in-payload lexical only reranks) applies to
  ANN candidate generation. OUR compact store is scanned EXHAUSTIVELY (ADC
  over all 4.64M) — no candidate stage — so DeepSeek's fused-vector design
  (48B semantic + 32B lexical hash dims, sqrt-IDF magnitudes, one dot does
  the fusion) DOES fix recall in our architecture. GPT's sidecar becomes
  necessary only if/when we move to ANN. ACTION: fused-vector BEIR arm is
  the right experiment for us; A/B dense-only vs +lex-subspace vs
  +title-minhash.

### New + actionable
- Q11: EVT null — max cosine of N random unit vectors ~ sqrt(2 ln N / d);
  normalize route confidence by (sim1 - mu_N)/sigma or normalized margin ->
  N-invariant threshold. Might make tau PARTIALLY transferable (test on our
  3 rotated open-set splits). Cheapest high-leverage fix on the list.
- Q1: + model2vec's own truncate-from-end OOV recipe (free, do first);
  hashed n-gram head 200-500k buckets int8 = 25-64MB; train composition to
  reproduce TEACHER (not co-occurrence).
- Q3: warm-start codebooks from our existing free-trained table then
  fine-tune through the quantizer (Gumbel-softmax per block, anneal);
  the win = codes closed under the algebra.
- Q6: cites the context-channel-capacity paper (our own H0 lineage!) —
  external validation of the framing; context-conditional low-rank adapters
  theta_k = theta_0 + A diag(c_k) B == MindForge; regime diagnostic =
  condition number of the context matrix (O(log K) vs O(K)).
- Q7: residual router trained only on low-margin traffic; ADWIN/PSI
  triggers; shadow/canary.
- Q8: "atomic opcodes in tokenizer = worth more than any data trick" (we
  already have this); program-first synthesis + execute-filter; train free
  + reject-sample, decode constrained; 150M: no multi-step text CoT, offload
  to interpreter (= our architecture exactly).
- Q10: CONVERGED 3/3: Platt over isotonic at small n (NM&C 2005, <200-1000
  pts); logit-transform cosine first; min-step feature; conformal for shift.
- Q9: DetIE 67.7 F1 CaRB; ceiling 65-70 clean / 45-55 noisy web.

### Red-team additions
- Ratio-of-small-numbers check (report absolute + CIs, n=400 queries);
  structure-circularity ablation (ablate block structure, rerun); swap-
  teacher control (we have: the ladder); hold-out whole years for dating.

### Q14 — CONVERGED 3/3 on the narrow claim
- Bare "learned hypervectors" not novel (TrainableHD, THDC, FLASH, semantic
  hashing 2009); Kanerva purists will argue learned codes break algebraic
  guarantees; the novel package = distilled + structured-block + teacher +
  hard byte budget + retained algebra, framed as bias-variance (random =
  no-data prior; distilled = teacher-informed, budget-dominant).

## Qwen 3.8 Max
- Q5: argmax dies ~l/2k+1 (~2); single-shot agreement m~25-30; SIC m~45-55;
  wall ~l/2=64. Budget <=16-24 pairs/frame + HIERARCHY of frames (tree walk).
- Q6: hypernet scaling-laws paper (2607.19604, LoRA-emitting, steeper OOD
  exponents); CREATIVE: condition f on block codes as context keys -> drift
  IS Q5 crosstalk (m·k/l), unifies memory+algebra stacks, testable; decisive
  350M 3-arm experiment (EWC / finetune-f / slot-only) — slot-only flat
  drift curve = the paper figure.
- Q1: subword RETROFITTING inside distillation loss + subword dropout
  (shared geometry); claims closes 70-90% of gap.
- Q10: synthetic planted-fault calibration data from the VERIFIER
  (unlimited labels) — clever, usable post-Task-5.
- Q12: bind time as one-hot block-code (VSA-native, dogfoods algebra);
  counterfactual delta-t probe; ATOMIC as substrate.
- Q13: per-decade confusion; EQUAL-BYTE teacher control; intersection-of-
  lists protocol.
- Q14: Random Indexing (Sahlgren 2005) = literal empirical version of the
  headline; "substrate sufficiency: geometry need not be learned, only
  lookup content".

## Kimi K3
- Q5: OR-semantics birthday m<~sqrt(2l)~16; sum+cleanup 0.2-0.3·l ~30-40;
  OMP law D >~ c·k·log M; SNR sqrt(D/k) (k=32 comfy, 200 borderline). Pin
  roles to dedicated blocks; chunk frames past 30-40.
- Q1: CLOSED-FORM ridge regression of hashed n-gram table onto teacher
  vectors — cheapest implementation of the consensus recipe (no training).
- Q2: 16B PQ + 64B top-IDF keyterm hash LIST (16x4B, query-side IDF sum) —
  most exact-match-targeted; Count-Sketch block for one-dot fusion; CWS alt.
- Q3: FSQ (no codebook, no collapse); "Tokenlearn into sparse block codes:
  nobody has published — paper-worthy delta" (independent novelty confirm).
- Q6: protect the FUNCTION not params (union probe set + masked KL ->
  ~O(1)/step); honest: no 100M-2B head-to-head exists — design comparison
  AS the contribution (matches GPT).
- Q8: RPN/flat opcode streams (bracket depth kills small LMs); STaR/
  rejection sampling = highest leverage; tiny critic best-of-n.
- Q10: beta calibration; Venn-Abers; split-conformal "only honest answer at
  your scale"; per-version recalibration.
- Q11: per-world Weibull tails (OpenMax) — thresholds in PROBABILITY space,
  N-invariant; separate novelty head; nightly probe suite.
- Q13: equal-byte teacher control; NULL-MODEL ablation (train-free-project
  vs train-structured) "mandatory before headline" — we largely HAVE this
  (potion->block attribution); post-cutoff dating falsification; "ask each
  model for cheapest falsification" meta-protocol (his predicted 3/4
  convergence on dating leakage CAME TRUE — all five named it).
- Q14: Thomas et al. JAIR 2021 — load-bearing property is INCOHERENCE not
  randomness; deterministic Hadamard codebooks exist; framing: "quasi-
  orthogonality as a learned constraint, not a random axiom" + capacity
  analysis under LEARNED code distributions (the theory-camp burden).

## PANEL SYNTHESIS (5 models)
CONVERGED (4-5/5):
1. Dating-leakage = #1 red-team item for ALL FIVE. Curated masked arm
   answers partially; NYT masked + date-scrubbed + post-cutoff run needed.
2. Subword/char-n-gram fallback fit to TEACHER = the gap-closer (Kimi's
   closed-form ridge = cheapest path; DeepSeek's truncate-from-end free).
3. Train-through-quantizer (STE/Gumbel per block; FSQ codebook-free
   option; warm-start from current table). Kimi: unpublished = paper delta.
4. Calibration: Platt/beta + split-conformal, never isotonic at our n,
   never multiply step probs; min-step feature; answer-level logistic.
5. Capacity: spread 2 -> 10^3 is DECODER-conditioned; sweep m x decoder
   (argmax / bounded-codebook agreement / +SIC); design answer regardless:
   hierarchy of frames at <=16-30 pairs.
6. H0 at 100M-2B: no published head-to-head — design it as contribution;
   LoRA-emitting hypernets consensus; Qwen's block-code context keys =
   creative unification (drift == crosstalk law).
7. Threshold N-drift: margin/z features, EVT null sqrt(2lnN/d), Weibull/
   OpenMax probability space, conformal sets.
8. BEIR: fused lexical-in-vector works FOR US (exhaustive scan, no ANN
   candidate stage — GPT's rerank-only objection void in our architecture);
   equal-byte BM25-parity is the honest target.
9. Trunk-SFT: atomic opcodes (have) + program-first synthesis + execute-
   filter + constrained decode + STaR/ReST with our verifier as multiplier.
10. Novelty: claim the conjunction/recipe/numbers + capacity-under-learned-
   codes; must-cite: SPA, Random Indexing, BEAGLE, Harmonic Mind, NVSA,
   THDC/LeHDC, Thomas JAIR 2021, semantic hashing, BPEmb, JPQ/DPQ/FSQ.

EXPERIMENT QUEUE (cost-ranked):
E1 NYT masked/date-scrubbed dating rerun (hours) — red-team convergence #1
E2 capacity sweep m x decoder at 80x128 (afternoon) — panel tiebreaker,
   H-B4 constant, paper figure
E3 subword-fallback closed-form ridge (day) — 0.699->0.722 gap
E4 EVT/margin-normalized tau on rotated splits (hours) — tau stability
E5 fused lexical block BEIR arm (day) — BM25-parity attempt
E6 STE/Gumbel train-into-block-space (week) — the unpublished delta
E7 H0 350M 3-arm slot experiment (biggest) — decides memory architecture

## Q15 (multiple futures + selection) — DeepSeek V4
- (a) MCL/WTA k-heads on latent targets = conditional quantization of the
  future distribution (TimeMCL 2025); anti-idle tricks LOAD-BEARING (1%
  prediction dropout + epsilon-softened assignment — idle head = silently
  dropped mode). Eval = coverage@k, never MSE. DINO-WM = the point-predictor
  baseline (19M works).
- (b) Filter-hard (verifier) THEN rank (calibrated logistic on [v-soft,
  retrieval, likelihood] + v*r interaction). External scorer > self-
  likelihood (self-ranking correlated with own errors — same argument as
  our H-B6 Goodhart). Best-of-n reward-hacking knee (Gao 2022): scorer-noise
  max grows sqrt(2 ln n) — the Q11 EVT effect again.
- (c) Entropy-adaptive k, floor 2. ASYMMETRY: mode dropping unrecoverable,
  scorer-fooling recoverable == our validated routing asymmetry (spawn-
  recoverable vs misroute-not) transplanted to futures. Architectural echo:
  MoWM worlds ARE an adaptive-k discrete mixture (spawn = grow k, prune =
  anti-idle, delegation margin = selection).
- (d) Precedent exists: search-in-superposition / resonator loop; framing
  "bundle = prior, evidence-unbind = unnormalized posterior via agreement,
  collapse = argmax" — publishable. DeepSeek says one-hot bundles break at
  k~2-3 so don't bundle candidates — BUT that's the naive-readout number;
  with bounded-codebook AGREEMENT decoding (what cubelang recover actually
  does, proven live at R=2 sim~0.5) the Q5 consensus (25-55) governs ->
  a 5-15 candidate one-hot bundle may be viable. ADD to E2 sweep: the
  "bundle candidates + evidence-collapse" case. Fallback: separate store +
  dot-product agreement (no algebra needed), or phasor codes (evidence-SNR
  limits instead).
- E8 (new, cheap): the kill-test — 5-head WTA over frozen (event, impact)
  embeddings from train_augmented; coverage@{1,3,5}; if coverage@5 ~=
  coverage@1 the future distribution is unimodal and the diverse-futures
  pipeline is unnecessary. Run BEFORE building anything.

## Q15 — Qwen 3.8 Max
- (a) Upgrades DeepSeek: ANNEALED WTA (aWTA 2024; 64->6 hypotheses needed),
  DETR-style mode-query tokens (not K full heads), Resilient-MCL scoring
  heads (free self-rank baseline), pairwise repulsion. V-JEPA 2-AC: 300M
  latent world model already does generate-then-select (CEM) in production.
- (b) Same filter-then-rank + w3·log p_proposal ANTI-GOODHART penalty
  (2025 BoN theory: restores monotonicity); retrieval caveat: key on
  event+time, score CONSEQUENCE agreement (anti topic-shortcut); planted-
  fault labels for fusion weights; ReST winners back into predictor.
- (c) k~8-16; spread-adaptive knee; conformal audits SIZE not content;
  non-monotone win-rate = the Goodhart signature.
- (d) DISAGREES with DeepSeek the right way: with match-count AGREEMENT
  decoding + partial evidence (k_e~20-35 of 80 blocks), the SELECTION
  SIGNAL breaks first at m~8-16 (before ~25-30 unbind capacity). DeepSeek's
  2-3 was the naive-readout number. Track k_e (evidence quality) as its own
  signal — weak evidence fails even at m=1. Superposition's real payoff:
  CROSS-FRAME selection over shared memory + resonator constraint-
  propagation (clamp evidence blocks); for one event's <=16 candidates a
  flat bank ties it. Elegant close: match-count vs evidence blocks IS (b)'s
  fusion in VSA form, unit weights.
- Creative unification: aWTA annealing == Q3 codebook-balance term
  (implement once, reuse in both stacks).
- E2 EXTENSION sharpened into 2D: bundle m futures, evidence cue with k_e
  matching blocks, measure winner-runnerup separation vs (m, k_e).
  Panel predictions to test: DeepSeek 2-3, Qwen 8-16, our estimate 5-15.
- E8 refinement: step 1 = aWTA + rMCL heads (gen + self-rank in one run);
  step 2 = planted-fault fusion, best-of-8 vs greedy vs self-rank.

## Q15 — Kimi K3
- (a) Convergent (aWTA + rMCL + anchoring at our world centroids — "free
  infrastructure", 3rd panelist to map onto worlds). NEW: VQ-COVER variant —
  quantize consequences into a codebook (our Q3/FSQ machinery), predict
  categorical over codes; regression->classification kills mode-averaging;
  codebook doubles as (d)'s item memory.
- (b) Verifier-gate -> product of experts; retrieval formalized as kNN
  density (hubness-correct via local z-scores); WARN (ii)-(iii) correlated
  (model trained on the store's events) — ablate, drop likelihood at serve
  if redundant. ROC-n-reroll 2025: BoN scaling characterized by verifier
  ROC; deterministic symbolic verifier = near-ideal = our favorable case.
  Amortize: distill selections back (STaR), keep cascade for high-stakes.
- (c) Three nested stops: conformal floor (Conformal LM, w/ dedup-rejection
  doubling as anti-collapse), stability stop, uncertainty-adaptive k_min.
  The two-curve experiment: recall@k vs P(top-scored correct) vs k.
- (d) Best precedent sweep: SDM; modern Hopfield ("attention where the
  value matrix is a bundle"); SPAUN basal-ganglia action selection
  (superposed actions, utility dot-products, production 2.5M-neuron model)
  — nobody else named it; resonators; HDC probing. THE GAP = OUR CLAIM:
  "weighted bundling as amortized Bayesian posterior; evidence cue =
  likelihood; temperature-calibrated collapse = MAP" — unpublished framing.
  Verdict: sides with Qwen (selection first, GRACEFUL; capacity second,
  ABRUPT phase transition). The real capacity-eater: CORRELATED hypotheses
  (futures share event structure) -> dedup before bundling / repulsion-
  trained codebook.
- DESIGN RULES TO ADOPT: (1) role-probe readout (slot-role per hypothesis;
  unbind THEN score — separates capacity noise from evidence noise, both
  independently measurable); (2) IMMUTABLE bundle + external weight vector
  (collapse = softmax reweight of w, never rewrite B -> reversible, exact
  recursive Bayes; temperature via Platt); (3) k<=16/bundle, hierarchical
  beyond; (4) winner's-curse shrinkage on the argmax.
- 3rd convergence on the 2D sweep (k x evidence strength, phase boundary
  at top-2 margin ~ 3 sigma_xt).
- Build order: (a)+(b) standalone week-of-work first; (d) evaluated as an
  ABLATION over a working pipeline, not a bet == our validate-then-wire.

## Q15 PANEL VERDICT (DeepSeek, Qwen, Kimi)
- CONVERGED 3/3: aWTA + scoring heads (k~8-16, anchored); verifier hard
  gate -> calibrated fusion; external scorer > self-rank below Goodhart
  knee; mode dropping = the unrecoverable failure; coverage@k not MSE;
  the 2D (m/k x evidence) sweep as the decisive experiment.
- RESOLVED 2v1: selection signal breaks first (gracefully) under agreement
  decoding; DeepSeek's k~2-3 was the naive-readout regime.
- OUR CLAIMABLE GAP (Kimi): the Bayesian-collapse framing of weighted
  bundles; + Qwen's cross-frame selection; + the capacity analysis under
  learned/correlated codes.
- E8 final shape: aWTA+rMCL heads on (event,impact) w/ world-centroid
  anchors -> coverage@{1,3,5} kill-test -> planted-fault fusion best-of-8
  -> (d) as ablation w/ immutable-bundle reweighting + role-probe readout.

## Q16-Q19 — DeepSeek V4

### Q16 multi-facet
- Per-facet tables, JOINTLY distilled w/ anti-leakage penalty; linear-probe
  leakage audit. VSA role-binding "wrong tool" for QUERYABLE facets
  (crosstalk + unbind latency) — but composite bound records still right for
  WRITES into VSA-native stores (worlds/episodic) — both true, note split.
- SRRL: distill classifier teachers through FROZEN CLASSIFIER (match
  student_feat·W_t to logits), NOT penultimate features; temperature 2-4
  load-bearing (one-hot posteriors carry no graded affect structure).
  -> the recipe for distilling AURA emotion/domain teachers.
- VAD lexicons as init/prior, not target. Negation: bigram features +
  negation-window heuristic; sarcasm: don't fix, attach confidence drop.
- FACET GRAIN TABLE: temporal/register = word-level (cites our dating win);
  emotion = word-with-caveats; INTENT = PHRASE-level (word table weak);
  domain = DOC-level. Don't force all facets through one word table.

### Q17 encoders-in-training — REDIRECTS OUR PLAN
- MeCo (2025): metadata-prepend + mandatory cooldown = 33% less data @1.6B.
- "URLs Help, Topics Guide" (2505.16570): ONLY provenance conditioning
  accelerates; topic/quality tags as conditioning = nil; quality+URL
  combined = WORSE than nothing. "(NOT) Work" (2504.17562): conditioning
  hurts latent-semantics inference at small scale.
- SPEND TAGS ON: quality -> filtering/upsampling (biggest ROI); domain/era
  -> MeCo or theta=f(domain) adapters; last 5-10% annealing by quality/
  domain. NOT conditioning on emotion/intent.
- Frozen tagger as theta=f(c) source: legitimate ONLY for loss-predictive
  facets (domain/era/quality) — "learned context module clusters the LOSS
  landscape; frozen tagger clusters semantics; they coincide only for
  provenance facets." VALIDATES domain_head.pt frozen-router choice with a
  mechanism; bonus: frozen tagger can't drift (Q7 vanishes).

### Q18 session memory
- Single EMA = blurry mean (Q15 mode-averaging pathology transplanted).
- DUAL TIMESCALE: fast EMA (lambda .3-.5) + slow novelty-gated LANDMARK SET
  (online coreset; fixes chatty-filler domination) + frozen session anchor.
  Capacity ~ d·lambda.
- CUSUM on fast-vs-slow divergence: off-topic = transient spike-and-return;
  session change = sustained slow drift -> model-free gate for world-switch
  vs tolerate. Direct fit to our routing.
- VSA variant: bind msgs to turn/speaker roles; decay as READ-TIME weight
  (reversible — same immutable-bundle principle as Kimi's Q15 rule).

### Q19 inner debate — GAP CONFIRMED (1st witness)
- "Heterogeneous-knowledge debaters emitting only machine-checkable
  propositions, judged by deterministic verifier: no direct precedent."
  Honest caveat: collapses to adversarial search over checkable claims —
  a feature at our scale. "Delegation made adversarial" endorsed.
- Debate bounded by strongest reasoner (2511.07784); heterogeneity must be
  KNOWLEDGE-diversity (worlds = strongest form); rhetoric-beats-validity
  removed by construction with VM judge. MORE ROUNDS HURT — stop at 2.
  Voting +13% reasoning / consensus +3% knowledge (ACL 2025); reliability-
  weighted consensus over majority vote (A-HMAD).
- Design: k worlds propose claim+trace -> VM hard-check -> ONE rebuttal
  round vs leader -> calibrated-confidence + reliability weights +
  RELEVANCE check (valid-but-irrelevant must not win).
- Kill-tests: (1) hetero debate vs single-best-world+VM (if single wins,
  debate is overhead); (2) hetero vs homogeneous sampling (if tie, worlds
  aren't knowledge-diverse — framing rework); (3) VM-judge vs LLM-judge
  ablation (if no drop, verifier isn't doing the work).
- Connective: Q16 tables = context vocab for Q17 theta=f(c) = worlds Q18
  tracks = Q19 debaters. Internalize: "tags are for selection, not
  conditioning" + "debate bounded by strongest reasoner".

## Q16-Q19 — Kimi K3

### CONFLICTS with DeepSeek
- Q16 storage: Kimi = ONE subspace-blocked table + barrier-augmented
  triplet objective (MUCARE; naive orthogonalization destroys real facet
  correlations; leakage 10-20% plain block-PCA -> <5% barrier). DeepSeek =
  per-facet tables. AGREE on: (c) role-binding = write-time composition
  format only; correlated facets are real. Kimi's 5x-latency argument is
  weak for us (~125us still fine); resolution = empirical A/B + the linear-
  probe leakage audit both endorse.
- Q17 conditioning: Kimi Tier-1 includes conditioning tokens broadly
  (Peeperkorn + label smoothing) BUT self-flags a key positive result as
  unverifiable; DeepSeek cites 2025 negative results (topics/quality
  conditioning nil-to-harmful). AGREE on: DoReMi-style mixture reweighting/
  selection = highest-evidence lever; aux losses weak (Kimi adds detach-
  and-predict if ever); per-facet adapters over-engineering. RESOLUTION:
  selection/reweighting settled Tier-1; conditioning restricted to
  provenance (MeCo+cooldown) pending our own ablation.
- Q19 rounds: DeepSeek "more rounds hurt, stop at 2" vs Kimi/Du "needs
  rounds >=2". Reconciled: ~2 rounds is the consensus sweet spot.

### Kimi additions to adopt
- Q16: soft-label KD works for classifier teachers (penultimate = mere
  convenience; SRRL still the sharper recipe); NRC-VAD as L1-ball priors
  (small epsilon, data wins on polysemous words); ABSTAIN class for
  negation/sarcasm cues + per-facet confidence ("know what it doesn't
  know") == our honest-fail philosophy.
- Q17: the two cheap legitimacy tests for frozen-encoder-as-theta-f(c):
  (1) router-accuracy probe >= 90% of learned-context baseline;
  (2) sufficiency-invariance split test (train hypernet on facet subset A,
  test B, degrade gracefully). Operationalizes DeepSeek's loss-predictive
  criterion.
- Q18: numbers — beta 0.9/0.99, capacity ~50/~500 items at D=10240;
  periodic slow-EMA reset w/ archival; the affect BLOCK of the fast EMA =
  the emotional trajectory for free (multi-facet payoff). Novelty gating
  +10-19pt in streaming-memory lit. Reservoir-VSA session memory = another
  paper-shaped hole (not the engineering answer).
- Q19: dRAG (NAACL 2026) — heterogeneous-RETRIEVAL debate beats
  homogeneous +8-15pt; "inter-resource diversity dominates intra-resource
  sampling" = direct evidence for worlds-as-debaters. Precise gap boundary:
  SAFE (verifier-eval, no debate), LeanDojo (verifier-gen, no debate),
  Du (LLM judge, free text), PRM (learned verifier, no debate).
  GAP CONFIRMED 2/2 witnesses. Failure modes convergent: verifiability-
  hacking (-> relevance gate) + expressiveness ceiling (-> report verifier
  coverage as a metric). Paper hooks: does deterministic judging remove
  judge bias; can smaller debaters beat larger when right.

### Running tally: paper-shaped holes
1. Tokenlearn-into-block-codes (Kimi, r1)
2. Weighted-bundle Bayesian collapse (Kimi Q15)
3. Verifier-grounded heterogeneous debate (DeepSeek + Kimi, 2 witnesses)
4. (minor) reservoir-VSA session memory

## Q16-Q19 — Qwen 3.8 Max (final panelist)

### THE CATCH: GAVEL kills the "clean gap" claim for Q19
- GAVEL (ACL Findings 2026): debate w/ Evidence Contract (atomic subclaims
  bound to evidence units) + Mechanized Chain of Scrutiny (deterministic
  validation before judging). "Cite GAVEL or get scooped." DeepSeek+Kimi
  both called it a clean gap — the panel's value demonstrated.
- SURVIVING claim: (1) deterministic symbolic EXECUTION as sole acceptance
  (GAVEL checks provenance, nobody executes claims in a VM); (2)
  heterogeneous small specialist WORLDS at <=2B (existing work = general
  LLMs/retrieval); (3) debate as adversarial delegation over shared
  structured memory. Position: "GAVEL showed mechanized scrutiny works for
  provenance; we generalize to symbolic execution with specialist
  proposers + the when-does-debate-pay analysis."
- Also: safety-via-debate lineage (Irving/Christiano 2018; doubly-efficient
  debate w/ Lean 4 formalization) = formal pedigree to cite.
- Debate findings: unstructured homogeneous small-scale debate ACTIVELY
  HARMFUL (never free-text rounds between worlds); structure flips it
  (ColMAD); diversity is the engine (MADKE hetero retrieval +8.1%;
  Diversity-of-Thought 78->91); MORE AGENTS THAN ROUNDS (one wide round +
  <=1 refinement + early stop); verifier gate STRUCTURALLY removes the
  persuasion/sycophancy channel (the mechanistic dodge of the main
  failure mode).
- E9 (new, cheapest — delegation already wired): single-world acceptance
  rate vs union-of-worlds acceptance rate; the gap = debate's headroom.
  Decision rule: if one proposer + verifier accepts most, debate is waste.

### Q16 resolution of the DeepSeek-vs-Kimi conflict
- "(b) implemented as (c)": facet subspaces AS DISJOINT BLOCKS of the
  block code — the record IS a VSA bundle, facet queries are pure block
  slices/popcount; role-binding with disjoint blocks IS subspacing.
  Elegant; adopt as the working design.
- Distillation: MSE on logits/pre-logits, NOT KL (IJCAI 2021: KL distorts
  penultimate geometry — and geometry is our product); Sparse Distillation
  (NAACL 2022): n-gram-table students keep ~97% of RoBERTa classifiers at
  600x speedup = EXISTENCE PROOF for our whole approach, cite in paper;
  context-averaging (Bommasani ACL 2020) for table construction;
  rank-deficiency caveat -> uniformity regularizer if similarity search.
- VAD: initialize dedicated affect dims (8-16), let distillation refine;
  phrase ROWS for negated/intensified bigrams + VADER-style scope rules;
  concede sarcasm.

### Q17
- MeCo + DoReMi/RegMix = the only two hard-evidence levers. RegMix: 1M-
  param proxies, ~10% DoReMi compute, mixture rank-invariance across scale
  — we can afford hundreds of proxy runs. Aux tag loss = freebie (mild
  panel spread). Tag-indexed adapter bank = "cheapest MoE you'll ever
  build" (no learned router). Eval few-shot AND zero-shot (URL gains
  show few-shot only). Frozen tagger probe rule: >=0.97 pairwise context
  discrimination (compatible w/ Kimi's 90%-of-learned). Unification: one
  tagger serves memory keys + routing centroids + session frames.

### Q18 — the creative standout
- FACET-SPECIFIC DECAY: emotion fast (half-life 2-4 turns = trajectory),
  topic slow (12-14), register/era ~frozen; per-subspace lambda (facets
  are subspaces per Q16) = multi-timescale reservoir in block-code
  geometry. Q5 law quantifies: lambda <= ~0.95 (m_eff <= 20 at 80x128);
  long sessions CHUNK into session-frame hierarchy (tree, not deeper EMA).
- Titans surprise-gated writes (weight decay = biggest ablation
  contributor; model-free implementable); delta-rule remove-before-write
  (kills redundancy exactly); Frequent Directions if formal guarantees
  wanted; BOCPD two-threshold confirm; change detection on the TOPIC
  subspace only (emotion legitimately jumps turn-to-turn).

### Cross-cutting (Qwen's closing synthesis)
- Facet subspaces = decay channels; the microsecond tagger = the single
  shared organ (distillation targets, change-detection series, context
  keys); debate = Q15 generate-then-select at world scale; superposition
  collapse = debate's memory-side dual.

## Q20 (question grammar) — DeepSeek V4

- (a) Template induction: DIRT dependency-path clustering + Chambers&
  Jurafsky 3-step; mine FROM questions (Fader 2013 — question syntax is
  skewed); recipe uses OUR fast table for skeleton clustering ("template
  induction at microseconds-per-item"). Parse-coverage = first-class
  metric (Dasigi NAACL 2019 formal defn) + CUSUM drift -> re-induction +
  route-to-parser triggers.
- (b) Ceilings: rules 35-45 F1; <=100M staged query-graph (Yih 2015 +
  BERT-base ranker) 55-72. THE NUMBER: relation errors = 45-50% of ALL
  KBQA errors (entities 81-87%) -> canonicalization is the bottleneck,
  not parsing. Dependency patterns > regexes (survive inversions) for the
  natural-question phase. Nested forms need constraint attachment;
  multi-entity breaks single-seed-chain assumptions.
- (c) Embedding canonicalization = PRE-RANKER only. Fact-level context
  (use the seed entity) >> relation-in-isolation; inverse-direction errors
  caught by our VERIFIER by construction (VM rejects swapped facts);
  OPEN-SET REJECTION non-negotiable (documented blind spot: models always
  return nearest neighbor, never "no such relation") — margin/entropy
  threshold, same Q11 machinery.
- (d) Named precedent: ON-DEMAND PARSING (Thomason RSS 2015) — grammar
  fast-path + learned fallback, coverage as the switch. VERIFIED
  SUPERVISION: filter grammar parses through the VM -> clean (question,
  program, denotation) triples for trunk-SFT — "the grammar is an asset
  generator, not a cost, until coverage saturates." Retirement rule:
  precision-parity ON THE COVERED SUBSET + coverage saturation, never
  overall-F1 crossover.
- OUR ACTIONS: grammar v2 regex-extension fine for the templated .pq;
  natural-question phase = dependency skeletons + coverage monitor;
  invest in relation matching/_accept and canonicalization (the 45-50%
  number); wire the verified-supervision harvest (every parsed+verified
  question = free SFT pair) into grammar v2's eval rerun.

## Q20 — Qwen 3.8 Max

- CONVERGENT w/ DeepSeek: coverage as SLO + drift alarm; grammar-as-
  bootstrap w/ verifier filtering (Zelle&Mooney/AMR pedigree); retirement
  = parser beats grammar ON ITS OWN TURF (covered subset); relation
  linking is where the errors live; OOD/open-set gating mandatory.
- THE VSA FIX (c): relation DIRECTION should not live in the embedding —
  bind relation to subject-role vs object-role vectors; "located in" vs
  "contains" share the embedding, opposite ROLES; match-count separates.
  Embedding proposes identity -> role binding fixes direction -> verifier
  confirms existence. Three cheap layers, all ours. (DeepSeek had verifier-
  catches-direction; Qwen makes direction STRUCTURAL.)
- (a) intent-cluster framing: templates = slot patterns within Q16 intent-
  facet centroids; drift-out = below-margin novelty (Q11/Q18 reuse) —
  "clustering with schema extraction, not a new subsystem." Caveat: dep
  parsers worst exactly on questions (inversion/fronted-WH) — use for
  induction (clusters absorb noise), not serve-time matching.
- (b) break ORDER: inversions kill rules first (fixable w/ templates —
  our probe found exactly these); recursion depth>=2 kills BOTH non-LM
  tiers (do NOT engineer recursion into the grammar — LM tier's job);
  tier(i)+inversion templates covers the bulk of traffic. Tier iii
  (constrained LM + verifier feedback) 80%+.
- (d) extras: teams report grammar saturating at 60-80% traffic; keep the
  retired grammar as REGRESSION HARNESS/test oracle (golden cases for the
  parser) — never delete it. ReST expansion cures template-shaped bias.
- Closing thesis (both models now): "the verifier is the load-bearing
  wall" — the highest-leverage investment is expanding what the verifier
  can CHECK, not improving any single parser tier.

## Q20 — Kimi K3 (3/3, panel closed)

- (a) MSParS-style induction (SRL + dep-path clustering recovers ~85% of
  hand-template coverage); CLOSED LOOP: coverage CUSUM -> cluster the
  UNCOVERED questions by embedding -> induce -> redeploy; merge templates
  by relation-chain equivalence; SRL weak on <5-token questions.
- (b) Rules 55-70 templated / 30-45 natural; <=100M 75-85 WITH constrained
  decoding (60-70 without — hallucinated relations); relation-vocab >500
  breaks <=100M -> two-stage coarse-type-then-specific (STAGG); 3+-hop
  compositional ~50 ceiling.
- (c) Retrieve-then-rerank (recall@10 ~90%); INVERSE RELATIONS worst
  (~30% bare-similarity error) — direction via dep-parse subject/object +
  inventory direction (complements Qwen's role-binding); granularity via
  domain/range type penalties; polysemy via Q16 DOMAIN facet narrowing the
  inventory first; abstain threshold. TinyRelationLinker (2511.10960):
  66M beats 7B LLMs on relation linking — external validation that
  small+features wins this subtask.
- (d) SiPQC = direct precedent (simple/complex router -> pattern vs
  learned parser, shared executor; our verifier strictly better);
  crossover ~85% coverage BUT the real driver = template MAINTENANCE cost
  as the relation inventory grows (STAGG); 3-consecutive-months rule;
  grammar demotes to fast-path + regression harness, never deleted.
- UNIFICATION: parse-fallback cascade IS Q19 verifier-grounded debate
  applied to parsing (worlds propose parses, VM judges) — second instance
  of paper-hole #3.

## Q20 PANEL VERDICT (3/3)
- Converged: relation layer >> parser (invest there); retrieve+rerank w/
  direction handling + open-set abstain; coverage-monitored grammar fast
  path; verified-supervision bootstrap (grammar+VM = free SFT data);
  retire by covered-subset parity (Kimi: ~85%/maintenance-cost driver);
  no recursion in the grammar tier.
- Grammar v2 brief additions: inversion frames from the probe; parse-
  coverage as reported metric; harvest (question, program) pairs from
  parsed+verified questions into a jsonl (the SFT asset); _accept()
  direction check noted for the relation layer.

## Q21-Q23 — Kimi K3

### Q21 — REDESIGNS the rerun calibration (adopted)
- Easy-negative thresholds DON'T transfer (OpenOOD near/far gap: FPR95 <5%
  far can be 40-80% near). Assignment: random distractors = sanity unit
  test ONLY (never in calibration — they pull tau uselessly low); PLANTED
  FAULTS = the calibration set (NINCO unit-test idea; tau = quantile of
  the HARDEST fault class — inverted direction — at FPR<=1%, bootstrap CI,
  deploy the UPPER bound); organic cross-chain confusables = held-out
  TEST set (Beyond AUROC fixed-threshold protocol).
- Cost asymmetry: false accept poisons downstream hops; false reject just
  re-retrieves -> FPR-target thresholding, NOT Youden (replaces our
  current Youden calibration). Report: hardness-ladder table (random ->
  cross-chain -> wrong-entity -> wrong-relation -> inverted), per-type FPR
  at the fixed tau; prevalence-honest precision.

### Q22 — world-backed retriever design + PAPER HOLE #4
- Re-route per hop, top-2 partitions, beam 3-5 chains (MDR compounding);
  delegation = per-hop fallback. Q20 parse = shard-selection PRIOR
  (relation domain/range -> world; unpublished, clean ablation).
- HOLE #4: absence-vs-misrouting discrimination from per-partition score
  DISTRIBUTIONS (Taily tails / our Q11 Weibull fits + delegation probes;
  misrouted -> tail-mass elsewhere, absent -> bulk-mass everywhere; no
  precedent found; false-absence labels feed world-spawning).
- Selective search MEASURED (Kulkarni&Callan): top-shard search matches/
  beats exhaustive at <2% docs touched; 86-90% of queries as-good-or-
  better — SAME SIGN as our own curated delegation result (routed-first
  0.951 beat flat 0.942). Invest in routing RANKING quality; cross-
  partition chains get beam budget.

### Q23 — harvest schema (adopted for the wave)
- Store the TRACE TREE, not the accepted path. Layers: core (q, program,
  answer, verified); trace (per-step opcodes + VERIFIER OUTCOME PER STEP =
  free PRM data, process>outcome per Lightman); failure (codes + failed-
  then-repaired same-question pairs = free West-of-N/BoNBoN DPO pairs;
  near-miss failures most valuable); confidence (per-step margins —
  unreconstructable later); group (all rollouts/question for GRPO
  advantage stats); difficulty (hops, retries, margins); provenance
  (verifier version + FACT-STORE SNAPSHOT HASH for invalidation, grammar
  template id, world ids).
- Three regretted discards: failure taxonomies, per-step margins, the
  counterfactual neighborhood (log which planted faults each verified
  chain rejects — hard negatives for the verifier's successor).
- Running holes tally: #1 Bayesian collapse, #2 Tokenlearn-into-blocks,
  #3 verifier-grounded debate, #4 absence-vs-misrouting. Kimi: #2 and #4
  cheapest to execute on existing infrastructure.

## Q21-Q23 — DeepSeek V4 (2/2, converged with Kimi)

- Q21 CONVERGED: planted near-misses calibrate (hardest stratum, FPR
  budget); random = sanity floor only; per-stratum reporting w/ score-
  distribution OVERLAP diagnostic. NEW: EVT/GPD tail transform — threshold
  on tail-probability not raw score (extends our Q10/Q11 line). THE JOINT
  INSIGHT (both models): failed-then-repaired traces ARE organic near-miss
  negatives — harvest them and the zero-negative problem closes itself.
  Resolution of the K/D split: manufactured faults calibrate NOW;
  harvested failed-trace organics become the deployment-matched test/
  recalibration pool over time.
- Q22 CONVERGED on re-route per hop. NEW: carry DELEGATION STATE not
  routing state (visited-worlds locality prior: try previous world ->
  neighbors -> global). Upfront planning is MORE established than Kimi
  found (FedNGDB decompose-upfront, SPLIT-RAG retrieval plans, RAGRoute
  federated router) — tempers hole #4's neighborhood, though the specific
  absence-vs-misrouting discriminator + parse-as-prior ablation still
  look open ("literature doesn't hand you a drop-in classifier").
  (c) signatures: wrong-partition = sharp peak somewhere; absent = flat-
  low everywhere + entity-existence check; centroid-only probing suffices.
  (d) cross-list bridge facts; measure recall@routing separately (>=95%
  rule of thumb).
- Q23 CONVERGED on the three regrets. NEW: SRFT numbers (61% of discarded
  trajectories informative; <=24% of failed-run steps actually wrong;
  MASK bad steps rather than mix — 32.2 vs 30.9); SDPO (verifier's
  textual rejection reason as dense signal beats scalar); nuance: per-
  step scores load-bearing exactly when TRACE correctness is the
  deliverable (us — Uesato's condition), else outcome+reason suffices.
- CONNECTIVE (both models verbatim): one harvest schema pays 3x — failed-
  repaired traces = Q21 negatives + Q22 partition-vs-absent labels + Q23
  DPO pairs.

## Q21-Q23 — Qwen 3.8 Max (3/3, panel closed)

- Q21 NEW INSTRUMENTS: k-MUTATION FAULTS (flip k blocks of a true fact,
  k=1..6; fault edit distance = free EXACT hardness label — VSA-native
  MoCHi ladder); rejection-rate-vs-k curve replaces the scalar tau as the
  honest operating description; CALIBRATION CARD format (per-bucket
  quantiles, mix disclosure, "untested against X" statements, re-mine-on-
  store-change staleness rule); calibrate PER FAULT CLASS (a global tau
  hides that inverted-direction sits near true hops while wrong-entity
  separates); easy bucket is one-sided (can only argue to LOWER tau).
- Q22 NEW: LogosKG (ACL 2026) = production precedent, billion-triple
  partitioned KG re-maps entities per hop; error-compounding bounds:
  safe chain length n <~ (delta/eps)^(1/(k+1)) with k-fold verification —
  per-hop verification formally EXTENDS safe chain length (justifies our
  verify-per-hop design); ELOQ warning: absent facts still retrieve their
  source doc at rank 1 ~50% — retrieval confidence provably insufficient,
  answerability layer (no candidate passes verifier -> absent) + reroute
  budget cap mandatory; GrailQAbility ceiling caution. (d) CLEF 2025:
  ORACLE shard selection BEATS exhaustive — selector quality is the whole
  game; connected-subgraph tradeoff (alpha 0.3-0.7; 55 vs 48 Hit@1) ->
  budget an explicit cross-world bridge hop.
- Q23 NEW: STaR rationalization for unsolved questions (inject answer,
  derive program backward, strip hint) — harvest booster; RISE step-wise
  DPO needs an NLL ANCHOR on chosen (near-identical pairs suppress it
  otherwise); record the repair DELTA (error-localized pairs beat whole-
  solution pairs); keep candidate sets w/ scores (runners-up = free
  negatives); graded outcomes (hops-verified-before-failure); AuPair:
  repaired pairs double as in-context self-repair examples at inference.
  Curriculum: two-phase (easy->hard early, shuffle later); FILTERING >
  ordering — spend on schema completeness first.
- 3/3 SYNTHESIS: one loop — harvest refills calibration buckets; routing
  features are the soft labels; ONE shared fault-class taxonomy across
  Q21 faults / Q23 failure tags / calibration-card coverage. Build order:
  taxonomy+schema first (zero compute), harvest immediately (retro-
  harvesting expensive, under-harvesting irreversible), k-mutation sweep,
  per-hop rerouting when a cross-world chain first fails.

## Q24-Q27 — Kimi K3

- Q24: two-stage InfoNCE(in-batch)->CE = full-CE quality at ~40% compute;
  shortlist-CE naive = rich-get-richer collapse (never-shortlisted items
  get no gradient) — anneal-phase only or uniform-sampling term.
  Retrofitting (2602.20735): FROZEN INPUT embeddings proven at scale;
  untying confirmed when output = retrieval index; PQ the output index w/
  QAT-through-assignment. PAPER HOLE #5: frozen BOTH vocab surfaces +
  trained-trunk-only, both surfaces distilled from the SAME teacher into
  shared codebook geometry (alignment risk -> same-teacher mitigation =
  literally our fastword pipeline). Sits on our Q3/Q24 intersection.
- Q25: fine-grained MoE 1.5-2x quality-per-FLOP at <=2B (OLMoE);
  TAG-ROUTED experts competitive w/ learned routers + more stable (no
  load-balance collapse; Landscape-of-MoE + Choose-Your-Weapon) —
  validates our free-tag expert bank; upcycling recovers 85-90% (tag-
  routing narrows the specialization gap further); UT-recurrence = money-
  pit; MatFormer only for elastic serving; MoE-pretrain + adapter-serve
  COMPLEMENTARY (freeze shared expert + router, adapt specialists).
- Q26: muP BREAKS on hybrid/recurrent archs + depth -> re-anchor LR at 2B
  w/ short sweep; proxy-safe: RegMix mixtures, tokenizer, schedule SHAPE;
  flips: peak LR, batch size, mixture-capacity interactions. Rental:
  single A100-80G + grad-accum (no FSDP at 2B), ckpt 500-1000 steps,
  bake shards into image. MFU REALITY: hybrid 20-35% vs pure transformer
  35-50% w/o hand kernels — recurrence MFU penalty can eat the FLOP
  advantage; cost = FLOPs/(MFU*peak)*$/hr, MEASURE our MinGRU MFU on a
  cheap short rental BEFORE the big run. Auxiliaries: top-2 = logit-KD
  (tokenizer mismatch SOLVED: Universal Logit Distillation 2402.12030;
  precompute teacher logits once) + MTP; QAT = brief final anneal only;
  activation sparsity = skip at 2B.
- Q27: attention:recurrence 1:6-1:7 suffices (Jamba/Zamba; OURS IS 1:3 —
  we may be over-provisioned on attention); window 2-4k; RETRIEVAL LAYERS
  SUBSTITUTE for attention layers (RETRO lineage) — sweep attention count
  with/without CB_MEM_EVERY at fixed budget (our exact infrastructure);
  state ~1-2x d_model; THE THREE-NEEDLE PROBE: (1) within window ->
  attention, (2) beyond window in KV, (3) beyond all attention reach ->
  recurrent state only; the first needle to fail = the saturated pathway.
  Direct extension of our H-D4 methodology.
- Kimi offers a concrete MFU-vs-FLOP cost model for hybrid-vs-transformer
  at 2B/14B on A100 (user can request).
- Holes tally: 5. (#5 = frozen-both-vocab-surfaces, cheapest-to-execute
  candidate alongside #2 and #4 — all three sit on existing infra.)

## Q24-Q27 — DeepSeek V4 (CONFLICT ROUND vs Kimi)

### Q24 CONFLICT — frozen embeddings
- Kimi: frozen input proven (Retrofitting), frozen-both = hole #5.
- DeepSeek: frozen static input NEGATIVE for open-text LM (2407.12514:
  random init often beats pretrained static; BlackboxNLP 2021: embeddings
  are THE most load-bearing LM component). "Frozen both = almost certainly
  a loss for LM loss." ESCAPE: for closed-schema PROGRAM emission w/
  verifier (our CoT trunk), frozen input + small learned output IS
  defensible. -> HOLE #5 NARROWED to the program-emitter regime (cleaner
  claim, still ours). DECISIVE LOCAL TEST at 151M: trained vs frozen-
  fastword vs frozen+low-rank-delta input embeddings.
- Tie-vs-untie: DeepSeek says tie (halve tax); Kimi/Retrofitting untie
  for retrieval output. Test both; PQ/DPQ-during-training agreed 2/2
  (14-238x compression).
- Loss: DeepSeek "train what you serve" — InfoNCE/continuous-output
  (Kumar&Tsvetkov: regress-to-embedding, 80% params gone); Kimi two-stage
  InfoNCE->CE @40% compute. Both: dark-vocab/rich-get-richer shortlist
  hazard.

### Q25 — mild conflicts
- Shared expert: Kimi pro; DeepSeek cites NVIDIA upcycling = WASH at
  small scale. Tag routing 2/2 competitive; DeepSeek adds SMEAR: soft
  learned router beats both by 2-3% -> TAG-INITIALIZED SOFT ROUTING =
  best-of-both; EMO warning re hard domain routing + novel domains.
- Upcycling 2/2 positive (4.1% lower val loss iso-FLOP; granularity helps
  in our <1T-token regime). MatFormer CONFLICT: Kimi "not worth it";
  DeepSeek "best measured param-sharing win at <=2.6B, small submodels
  BEAT independent baselines" — unresolved, cheap to defer until elastic
  serving matters.

### Q26 — MTP CONFLICT + logistics
- MTP: Kimi #2 buy; DeepSeek MONEY-PIT at 2B (Gloeckle: gain disappears
  >=100M; ACL 2025: static MTP doesn't beat NTP at 1.3B/3B) — keep only
  for self-speculative DECODE. ADOPT DeepSeek (more specific citations);
  revisit only as serving accel.
- KD 2/2 = #1 buy. DeepSeek warning: vocab mismatch = #1 silent KD
  killer, "reconsider the custom 128k" — REJECTED for us (tokenizer is a
  settled asset w/ atomic opcodes); use ULD-style logit alignment (Kimi's
  cite) instead. Precompute teacher logits on a rented GPU once.
- 1xA100+accum (Kimi) vs 8xA100 FSDP (DeepSeek): not a real conflict —
  2B bf16 + Adam FITS one A100-80G; 8x buys calendar not necessity.
  Stream-vs-bake shards: minor; DeepSeek says stream w/ NVMe cache
  (image bloat slows preemptible restarts) — sensible.
- muP 2/2: breaks on recurrent path; tune recurrent-specific scales at
  target; regularization never transfers; WSD-vs-cosine + RegMix + vocab
  ablations settle locally.

### Q27 — convergent, sharper
- Ratio band 2:1..6:1 (Griffin 2:1, Zamba ~1:6, Jamba 1:7; systematic
  sweep 2507.06457: recall degrades monotonically, cliff past ~6:1-12:1).
  OUR 1:3 IS INSIDE THE SAFE BAND (tempers Kimi's over-provisioned read);
  could test toward 1:6 w/ recall monitoring + memory layer.
- Retrieval substitutes for attention 2/2 (RETRO); not free (different
  frontier point).
- State capacity now PROVEN: Impossibility Triangle (2605.05066) — fixed
  state recalls O(poly(d)/log V) pairs regardless of context; linear-attn
  state ~ softmax over ~64 tokens (SSE); expanded-state 454M matches
  1.47B on recall (ACL 2025).
- Probe refinement: single-needle WRONG (saturates trivially) -> MULTI-
  ITEM associative recall (sweep m), + prefix-sensitivity + order-
  permutation; combine w/ Kimi's three-needle placement = full suite.
- DeepSeek offers a Q26 rental runbook incl. the MFU A/B that decides
  transformer-vs-hybrid before renting.

## Q24-Q27 — Qwen 3.8 Max (3/3, batch closed)

### Tie-breaks
- FROZEN EMBEDDINGS 2-1 (Kimi+Qwen vs DeepSeek): Qwen finds the close
  prior art — HEADLESS LMs / Contrastive Weight Tying (ICLR 2024): no
  softmax projection, contrastive reconstruction of INPUT embeddings,
  in-batch negatives = our retrieval-head objective, PUBLISHED (must-cite;
  hole #5 = the fully-frozen-both + stack-only variant, thinner precedent,
  50M-testable). DeepSeek's counter-evidence is GloVe-era static; Qwen
  cites BERT-TINY+fastText 2025 (static init HELPS small transformers).
  DECIDER: the 3-arm 151M experiment (trained / frozen-fastword / frozen+
  low-rank delta).
- MTP 2-1 (Kimi+Qwen vs DeepSeek), with the SAME ACL 2025 paper read both
  ways: static MTP loses (all agree); FORWARD-CURRICULUM MTP + register
  tokens wins per Qwen. ADOPT: curriculum-MTP arm at the 150M pilot,
  decide there. Cheap either way.
- Stream-vs-bake: 2-1 stream w/ prefetch+NVMe cache (survives preemption).

### Qwen's reframes (adopt)
- Q24(b) THE BIG ONE: full cosine-CE over 128k is AFFORDABLE at train
  time — the softmax-free saving is at DECODE; softmax-CE-over-cosine-
  with-temperature IS full-vocab InfoNCE (same run). Don't import
  sampling complexity; if ever sampling: logQ correction is load-bearing
  (k-of-n sampled softmax provably biased, NeurIPS 2025) + uniform floor
  for rare rows. PQ-ADC over the full codebook = effectively full softmax
  = sidesteps sampling bias entirely. Train what you serve, full-vocab.
- Q25: MoE TOKEN TAX ~4.6x more tokens to realize per-FLOP gains — at our
  14B-token budget this is binding -> CYCLE-ONE DENSE settled (matches
  the upcycling threshold formula D* ~ 4B tokens; from-scratch beats
  two-stage without a checkpoint). NEXUS (EMNLP-F 2025) = the tag-
  initialized soft router implementation (domain embeddings from an
  external embedder = OUR STATIC TABLE; +2.1/+1.6%; new domains +18.8%
  w/ limited FT). Zamba's weight-SHARED attention block: steal directly.
- Q26: cost table ~1,350 A100-hrs ~$1.3-1.9k spot; 8xA100 FSDP ~1 week
  or 1xA100 ~8 weeks; torch.compile-only recurrence MFU 10-25% (bleaker
  than others) vs Mamba-with-fused-kernels ~ transformer -> THE 2-HOUR
  LOCAL MFU PILOT AT 150M = single highest-EV action (3/3 convergent);
  then Triton port (days, saves hundreds); ratio shift only as last
  resort. KD: OFFLINE precomputed logits on a 2-4B-token SUBSET; worth it
  because 14B tokens ~ 7 tok/param = data-constrained regime; skip if
  teacher <4x student.
- Q27: RATIO LAW REFINED — weak recurrence (LRU-class) needs 1:3, strong
  (Mamba-class) tolerates 1:6-8. Our MinGRU = weak-class -> OUR 1:3 IS
  EXACTLY RIGHT per Griffin (settles the Kimi-vs-DeepSeek read). Window
  can stay ~1-2k across scales. Retrieval-substitution: NO clean ablation
  exists — attention-out vs recurrent-out with retrieval held fixed is
  PUBLISHABLE (hole #6 candidate); prediction: retrieval absorbs STATE/
  factual capacity, not attention's positional binding. Probe battery:
  MQAR k-item recall (the cliff's k = state capacity), delayed-match,
  parity beyond train length, linear-probe-on-state decay, fact-PPL;
  interpret state as a BUNDLE via our Q5 crosstalk law (SSM state =
  superposition — our algebra unifies with state-capacity theory).

### Decision chain settled (3/3)
1. MFU pilot (2h, local) -> decides kernels/ratio/budget.
   **DONE 2026-08-14 (Colab A100-SXM4-80GB; H-E4; logs
   `validation/logs/exp_t1_mfu_pilot_a100*`): the recurrence penalty is an
   eager artifact — compiled, hybrid=32.8% / mingru=32.6% / attn=33.5% at
   150M (hybrid = 98% of transformer; bar was 2/3), and at the true 2B shape
   hybrid/compile hits 40.3% (no ckpt, B8, 40G/80G) BEATING attn (26.2 vs
   25.4 in the ckpt arm) — then the same-day FlexAttention window fix
   (dense S x S masks were forcing full-quadratic SDPA) lifted it to
   **45.5%** at S1024/B8 and 32.5% at S4096 (was 21.8%). Decisions: Triton
   port SKIPPED; ratio 1:3 stands; torch.compile+flex is the kernel story;
   no grad-ckpt at 2B/80G; long context ~ free (curriculum choice, not
   budget). Budget (token basis corrected by H-G3 2026-08-14: cache is
   17.74B, not 14.37B): one epoch ~101 H100-hrs ($203-304); Chinchilla
   ~40B (~2.25 epochs) ~228 H100-hrs ($457-685) / ~100 B200-hrs
   ($451-651) / ~724 A100-hrs — still under the panel estimate, which
   assumed ~14B tokens at far lower MFU. H100-class wins both axes
   (spot-check the actual rental card ~15 min with the same script first).**
2. Cycle-one dense hybrid @ 1:3, WSD, full-vocab cosine-CE; frozen-
   embedding 3-arm test at 151M first; curriculum-MTP arm at pilot.
3. Rent single-node A100s, stream shards, ckpt 15-30min, spot.
   (Per step 1's measurement, prefer H100-class over A100 — better $ AND
   wall-clock at measured MFU.)
4. After: offline ULD-KD subset pass, int8 PTQ, MQAR probe battery.
5. Cycle two: upcycle to fine-grained tag-initialized MoE (Nexus router
   fed by our table), shared attention block.

### Paper must-do (novelty/prior art)
- Q14: cite SPA / Eliasmith semantic pointers (semantically-derived hypervectors,
  2000s), Plate fractional binding, projection-to-HDC (Rahimi). Nengo/VSA
  community would dispute bare novelty. Position novelty as: qFHRR/block target
  + signal blend + measured envelope + algebra retention (jointly structured
  codebook), NOT "non-random hypervectors" per se.
- Q13 #2 vocabulary-overlap bias: partially addressed by OOD arm; state that
  axiom words in vocab are by-design serving coverage.
- Q13 #3 entity-search positional loss: already in limitations (bag of words).
