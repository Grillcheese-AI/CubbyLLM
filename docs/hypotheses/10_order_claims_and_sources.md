<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 10. Prioritized validation order

**Campaign status (2026-07-23):** steps 1–6 below were executed the same day this order was written — see
`VALIDATION_REPORT.md`. The list is kept with ✅/⏳ marks so the *next* session starts from what's actually left, not from
a stale queue.

1. ✅ **H-B3** — done: `vsa_binding_head.py` reread; `UNBIND_ROLE` was misattributed (it's cubemind VM code; the head has
   no unbind at all). ✅ **H-B5** — done: grilly IS the shared substrate (cubemind's `BlockCodes` wraps it directly); all
   three algebra families live there.
2. ✅ **H-C6** — **DECIDED (user, 2026-07-23): hybrid** — fixed core (next: 128k–256k BPE on the proven pipeline) +
   retrieval head from day one + separately-gated dynamic tail later. ✅ **H-A1** — done: C_ctx=0 structural + 94%
   empirical forgetting measured on the real layer.
3. ✅ **H-A3 vs. H-A4** — done: the Zero-Forgetting Stability Benchmark exists (`validation/exp_a_forgetting.py`) and ran
   all four candidates. Headline: NLMS best capacity but oldest-first forgetting; SDM only order-agnostic method but
   lower capacity at matched state; flat-decay Hebbian worst measured. ✅ **NYT-dated iteration (H-G2) also done same
   day** (`validation/exp_g2_nyt_forgetting.py`): real correlated keys cost everyone ~2–3× recall but the ranking
   survives unchanged — it can be trusted.
4. ✅ **H0 / H-A2** — done at toy scale: supported ONLY in the hardened form (naive hypernetwork is *worse* than
   baseline); P5 clean on the hardened variant. ✅ **Learned-context follow-up (H0 part 2 + H-C4) also done**
   (`validation/exp_h0b_learned_context.py`): with context INFERRED by a router instead of given, an unsupervised router
   collapses to one slot (reverts to baseline) and even a task-supervised router trained *online* forgets its own
   routing (slots drift 6→3 distinct, task-0 confidence →37%). Finding: **context inference, not θ=f(c) generation, is
   the bottleneck** — the router must be an OFFLINE-pretrained classifier (H-C4's `domain_head.pt` is exactly that), not
   learned online. ⏳ Still open: the hardening cost curve vs number of protected contexts, and H0 wired into a real LM
   memory pathway.
5. ✅ **H-B1 vs. H-B2 vs. H-B5** — the user-directed 3-way prototype over grilly's actual families is DONE
   (`validation/exp_b5_grilly_3way.py`, D=10240 / k=80×l=128, V=32k). Duties matrix: **block codes win or tie every duty
   except raw bind speed** — exact unbind (1.000, vs HRR's approximate 0.711), 100% readout at 5% signal (binary 84%,
   HRR 86%), 100% at n=320 superposition, 96.3% trainability through per-block softmax, readout ≈ binary speed (37ms
   @32k). Binary is 60× faster at bind (3µs) and trains at 99.6% when tanh-relaxed — but the hard-sign STE path measures
   **19.8%**, turning block_ops.py's "zero-gradient problem of bipolar sign()" claim from assertion into measurement.
   HRR: approximate unbind, 11× slower readout, 12.8% on the probe (partly probe-artifact — untuned shared
   lr/temperature — but the two independent negatives stand). Cross-implementation parity verified (grilly
   BinaryOps.bind ≡ cubby-concepts bind). **DECIDED (user, 2026-07-23): standardize the binding head on `BlockCodeOps`**
   — NVSA sparse block codes, k=80×l=128, one algebra across model + world-arena (also cubemind's existing substrate;
   Vulkan path confirmed initializing on this machine's RX 6750 XT). BinaryOps/HolographicOps remain reference
   implementations, not the standard. **Two grilly hazards found while surveying it for reusable VSA work
   (2026-08-02).**

(a) **`blockcode_bind` exists twice with different semantics, and one silently corrupts bundles.**
`grilly.experimental.vsa.block_ops.BlockCodeOps.bind` (what `cubbyllm.ops.BlockCodeVSA` actually calls) is true
per-block circular convolution. `grilly.backend._bridge.blockcode_bind` — both the GPU shader
`shaders/blockcode-bind.glsl` and its numpy fallback — is a one-hot **argmax shift**: it scans `if (a[i] > 0.5) hot_a =
i`, so on a continuous block code (every entry < 0.5, since each block sums to 1 across l slots) both indices stay 0 and
it returns the **identity element without raising**. CubbyLLM is safe only because its ops seam never dispatches to
`_bridge` — see the TODO item, which now says *do not* wire that dispatch. grilly's own `tests/test_blockcode.py` cannot
catch this because every test uses strictly one-hot inputs; there is no test anywhere in grilly that unbinds from a
superposition. Ours now has three: `tests/ops/test_vsa.py`.

(b) **`blockCodeCodebook` caps at n <= l — 128 codes at l=128.** IBM's successive-binding construction walks the cyclic
group Z_l per block with the same shift each step, so it wraps after exactly l entries; the C++ raises `n must be <= l`,
and the Python twin *silently* degrades to random sampling "(may have collisions for large n)". Irrelevant for
role/filler duty (n=16), but it means the decided k=80 x l=128 shape cannot supply a large concept codebook by that
construction. Independent random one-hot per block gives ~l^k distinct codes with birthday-bounded collisions, at the
cost of algebraic closure — budget that design choice before the VM needs a vocabulary-scale codebook.

Group B's remaining open item is now only the generalization question from H-B5 (SVC role-filler scheme → token/concept
binding) plus building the real unbind on the VM side (H-B3's relocated conclusion). **Added 2026-07-30 (H-B6):** a
*separate* differentiable path — a parameter-free MAP binding aux loss — now shapes the trunk to be VSA-bindable during
LM pretraining, validated live at 2B (binding loss → ~0.20 plateau, bpc unharmed through full-LR onset out to step
3000). Mechanism VERIFIED; "free" still PLAUSIBLE pending a clean `ckpt_1` A/B and full-run confirmation; and
MAP→`BlockCodeOps` transfer is untested (the aux loss uses MAP for differentiability; the production algebra is a
different family).
6. ✅ **H-C7** — done ahead of H-C6 because it turned out decision-independent: direct readout scales to V=1M at D=10240;
   flat factored readout does not; memory/compute (H-C3) is the real constraint either way.
7. ⏳ **H-G3** — unchanged: dedup/NSFW-filter/reproducibility pass on `unified/` before any real training run.
8. ✅ **Group D backbone (H-D1) — RESOLVED: MinGRU.** The in-architecture bake-off (`exp_d1b_backbone_bakeoff.py`, 4
   candidates in the real CubbyModel on subword TinyStories) found gated recurrence beats attention-free Q=K and softmax
   attention by ~10% bpc; MinGRU ties GRU at 25% fewer params + a parallel scan. Promoted to
   `cubbyllm/model/backbone/mingru.py` as the package default. H-D2's TTT compute side stays
   measured-but-quality-untested; a faithful BDH + TTT backbone at scale is a possible future revisit, not a blocker.
9. ⏸ **Group E (hardware)** — deferred by design (H-E3, now verified by execution: the whole campaign ran CPU-only).
10. ⏳ **H-G5 (later / SFT phase)** — synthetic domain-tagged Q&A distilled via OpenRouter from the domain-sorted corpus,
    once it's pinned (H-G3). Prototype the generator small now to lock format/quality; defer the bulk spend to after
    pretraining. Not part of the pretraining gate.
11. ✅ **Group P (added 2026-08-25)** — H-P1–H-P5 toy-scale VERIFIED and the H-P1/P2/P3/P4 checkpoint arms run the same
    day on `ab_hybrid.pt` / `hd5_mem2.pt` (`validation/prospection/`, logs linked per entry). Scale verdicts: forking
    exact and bounded (P1); the toy sparsity of choice points does NOT transfer — 70–90% of real-text steps above 1–3
    nats, budget rule needs a relative criterion (P2); **the trained recurrence is a 1–2-step mixer, all long-range work
    is in the attention layers** (P3); interpolation-based keyframe compression is dead, recompute wins (P4). ✅ **H-P6
    RUN 2026-08-26 (A100, 65M tokens/arm):** the multi-horizon head is free (Δ held-out CE +0.002 nats) but redundant —
    a 300-step linear probe on the *baseline* trunk already reads the 50- and 500-token future at cosine 0.33 / 0.41 vs
    trivial 0.03 / 0.08, the objective adds ~0.02, and the recurrent gate spectra of the two arms are identical (1.2–2.1
    steps). **Decision: not in the 2B runbook; successor-feature readouts become post-training probes on the frozen
    trunk.** ✅ **H-P7 RAN 2026-08-26 (Colab A100, 65.5M tokens/arm; verdict recorded 2026-08-28 from
    `validation/logs/exp_p7_{baseline,chrono}.*`): KILLED on CE** — chrono init costs +0.444 nats held-out at matched
    tokens (threshold 0.01), the τ spectrum erodes ~4× but survives (5–10% of units ≥ 100 steps vs 0 for the baseline),
    and the beyond-window impulse response is non-zero but ~1e-3. The gate can hold a long time constant; nothing useful
    goes in it at this scale. **Group P's last open route to long *recurrent* time constants is closed: beyond-window
    memory lives in the attention window and the episodic store — H-P3 final.** Chrono is not a cycle-one arm.
    Needle-recall (protocol step 4) was not run on Colab and deliberately not run after — ~13 CPU-hours for a read both
    arms would fail at 65M tokens (see the entry). ⏳ Still open in Group P: the relative-entropy reformulation of the P2
    budget rule (re-test on the 2B checkpoint). Not part of the pretraining gate; the remaining roadmap items named in
    §9b's preamble get entries only once each has a cheap experiment.
12. ✅ **H-A7 — the learning gate (added 2026-08-28; built + RAN 2026-08-30 on Colab,
    `validation/logs/exp_a7_learning_gate*.*`): the gate SEPARATES.** LR ladder on `hd5_mem21.pt`, 3.3M-token updates: a
    replay update is CE-neutral on every source at 3e-5 (mix +0.005) and 1e-5 (−0.002) while the code-only update is
    still rejected there with the mechanism named (3e-5: mix +0.059, 12/12 unseen sources, memorized `pretrain_ext`
    +0.208, trained source −0.019; 1e-5: mix +0.017, `pretrain_ext` +0.053); at 1e-4 replay drifts uniformly (+0.046)
    and restoring the Adam state makes it worse — LR is the lever. Null promotes everywhere. **Nightly LR = 3e-5.**
    Three of the six proposed terms cannot see a trunk delta (CoT precision, routing precision, NYT-vs-NLMS);
    refinements queued: per-source ε from the paired SE, and a `uniform_regression` vs `forgetting` naming term. Gate
    eval 37–50 s/A100. Not part of the pretraining gate. ✅ **Counterfactual-neighborhood harvest (same day, DONE):**
    `exp_m3_cot_pipeline.py` now writes `counterfactuals[]` for every verified chain — see the H-F2/H-B3 CoT entry's
    2026-08-28 addendum for the run.
13. ⏳ **Group K (added 2026-10-20) — the research pause.** H-K8's law and chunk store measured the same day
    (`validation/exp_k8_holo.py`: gate 1 failed as written — the law marks the 0.68 point, 90% holds to half of it; gate
    2 passed at the line, 0.90); H-K1 read the same day (`validation/exp_k1_map.py`: the map 2× TransE on one hop, 4.6×
    on relation discovery, both compose; gate 2 failed as pre-registered); H-K7 gate 1 read (`exp_k7_exposure.py`: 0.00
    / 0.00 / 0.50 / 0.99 at 1 / 10 / 100 / 1,000 sightings); H-K10 gate 1 read (`exp_k10_hypertoken.py`: a fact shown
    once answered at 0.94 beyond the window through the loci token, 0.00 for text; the learned route 0.00). Next,
    cheapest first: H-K7 gate 1 (the exposure curve on our trunk, a CPU afternoon), H-K4's host side (`Scene` with dims
    = 1), H-K8 gate 3 (a `CubbyMemory` sibling on a synthetic recall task), then H-K2 and H-K4's trunk sides on the
    grown model's second round, H-K9 on BlockWorld traces, H-K5's rule generator; H-K10's route read the same evening
    (the learned route 0.00 → 0.82 when taught by the route loss; attention over the whole document 0.03; a mismatched
    address 0.00) and its shards built (`facts`, `spatial_here` + text controls): the 450M run on them (`CB_SPATIAL=1
    CB_SPATIAL_QUERY=placed CB_ROUTE_LOSS=1`) is the next Colab job — H-S12 gates 1–2 and H-K10 gate 4 in one.

## 11. What this document does not claim

It does not claim the imported research (GCE, BDH, Sylph, CMP, ZeTT, zip2zip, Hypencoder, the trillion-parameter
blueprint) is correct — these are external claims supplied by the user, not independently reproduced by anyone on this
project, except where a hypothesis above is explicitly marked VERIFIED (the cubby-concepts `vsa/` test suite and the
SDM/resonator numbers, both independently re-run this session).

It does not claim any of the four surveyed sibling folders (cubby-concepts, AURA_GENESIS, GrillCheese, Novel_GNN_Arch)
contain a ready-to-ship solution. The single strongest finding across all four, repeated because it matters: no
implemented, validated fix for Hebbian catastrophic forgetting exists anywhere in the group (H-A6). That's a real,
useful negative result, not a gap in the survey.

It does not commit to AIMC/analog hardware, BDH, or any specific backbone swap. All of those remain open, explicitly
gated behind the cheaper validations in Section 10.

~~It does not claim `grilly.experimental.vsa.ops` (H-B5) is the same mechanism as cubby-lm's binding head or cubemind's
block-codes.~~ **Resolved 2026-07-23:** grilly IS confirmed as the shared substrate for cubemind's block-codes (direct
wrapper, verified in source) and is the named hash-lineage for the binding head's cross-repo codebook sharing; what this
document still does NOT claim is that grilly's SVC role-filler scheme generalizes to CubbyLLM's token/concept binding
needs — that remains open (see H-B5's validation result).

It also does not claim any toy-scale validation result in this document transfers to CubbyLLM's real scale — every
"toy-scale VERIFIED" tag above (H0, H-A2, H-A5, H-C1's kill-half) is a mechanism demonstration with its scale caveat
stated inline, and `VALIDATION_REPORT.md` ends with a "what would change these conclusions" list that should be treated
as part of each result.

## Appendix: source map

| Group | Hypotheses | Primary sources |
|---|---|---|
| H0 (central bet) | H0 | GCE / Ran Cheng "Context Channel Capacity"; Sylph; BDH `bdh-cl`; AURA_GENESIS `liquid_moe.py` (controller template only); von Oswald et al. 2019 (arXiv:1906.00695 — the hardening term, pinned 2026-08-28) |
| A — Memory/forgetting | H-A1–H-A7 | GCE paper; `cubby-concepts/vsa/sdm.py`; `AURA_GENESIS/aura/core/nlms.py`; GrillCheese `capsulememoryintegration.md`, `STATUS.md`, `gpu_brain.py`; Novel_GNN_Arch `apply_outcome_update`; H-A7: `docs/research/2026-08-28-oracle-competition-scored.md` (KTO 2402.01306, O-LoRA 2310.14152, Online-LoRA 2411.05663, replay 1811.11682, ripple effects 2307.12976, TracIn 2002.08484, conformal LM 2306.10193, sleep-time compute 2504.13171) |
| B — VSA binding head | H-B1–H-B5 | `cubby-concepts/vsa/{core,resonator,graph,sdm}.py`; GrillCheese `experiments/{holographic_relation_test,brain_resonator_test}.py`; Cubby `vsa_binding_head.py`; `grilly.experimental.vsa.ops` (via `$CUBBY_TRAINING_DATA\regenerate_svc_targets.py`, found 2026-07-23) |
| C — Vocabulary/tokenization/softmax bottleneck | H-C1–H-C7 | ZeTT; zip2zip; Hypencoder; GrillCheese `gcheese-faiss/`; Novel_GNN_Arch `capsules/domain_classifier.py` + `domain_head.pt`; cubby-concepts `budget-hardware/`; the much-larger-vocabulary requirement (user, 2026-07-23) |
| D — Backbone | H-D1–H-D3 | Trillion-parameter architecture blueprint (user-supplied, 2026-07-23); BDH; TTT literature |
| E — Hardware/economics | H-E1–H-E3 | CAPEX/OPEX comparison (user-supplied, 2026-07-23); AIMC/memristive-crossbar claims in the same blueprint |
| F — Structural | H-F1–H-F2 | `cubby-lm-restructure` architecture map; `CUBEMIND_REFACTOR_PLAN.md` |
| G — Training data | H-G1–H-G5 | `$CUBBY_TRAINING_DATA\{tokenizer,temporal/nyt_data,unified,svc_targets*}\` (surveyed 2026-07-23); domain-sorted corpus `$CUBBY_DATASETS\domains\` + OpenRouter (H-G5, 2026-07-25) |
| P — Prospection (branching futures / timescales / keyframe compression) | H-P1–H-P5 | design conversation 2026-08-25; `cubbyllm/model/backbone/mingru.py` gate; AURA note `$CUBBY_DATA_ROOT/AURA_MIND\papers\temporal_memory_interpolation.md`; Pearl (ladder of causation); Tallec & Ollivier 2018; Murray et al. 2014; Redish 2016; Delétang et al. 2023; Orvieto et al. 2023 (LRU, unbuilt) |
| K — Knowledge, the map, the state (research pause) | H-K1–H-K10 | Nick's request and questions 2026-10-20; H-S12's algebra; inspiration only: Titans (arXiv:2501.00663), Nested Learning / HOPE (research.google blog), Gated DeltaNet-2 (NVIDIA Research), Kimi Linear (arXiv:2510.26692), Attention as Binding (arXiv:2512.14709), Haga (bioRxiv 2023.05.11.540307), grid-cell path integration (`ruiqigao/grid-cell-path`, NeurIPS 2021), Language Models Need Sleep (arXiv:2605.26099), HeLa-Mem (ACL 2026), Hrrformer (ICML 2023, arXiv:2305.19534), Capacity Analysis of VSA (arXiv:2301.10352), Physics of LMs 3.3 (arXiv:2404.05405), Memory Layers at Scale (arXiv:2412.09764), ARC-AGI-3 leaderboard (arcprize.org), Raschka "Beyond Standard LLMs", `google-research/swirl-dynamics` (DySLIM, flow-map matching / MeanFlow, Evolve Smoothly, debiasing, WENO-NN) |

Companion documents in this folder: `cubby-model-environment-map.md` (verified current-state architecture map of Cubby's
model and CubeMind's world-model) and `README.md` (project front door).
