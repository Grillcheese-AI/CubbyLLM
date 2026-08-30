# Oracle-competition scoring (2026-08-28)

**What this is:** the scored record of a five-round OpenRouter "room"
contest run 2026-08-28 — *"it's 2028, build yourself an always-on,
self-learning, auditable, chatable oracle on $300 → $10k"* — whose raw export
is `docs/model-competition-for-cubby-finalconsensus.json` (2.4 MB, `orpg.3.0`).
Round 1 was blind; from round 2 on the contestants were handed
`CUBBYLLM_HYPOTHESES.md`, then the model-panel agenda + architecture vision,
then the Group P headline numbers, then the whole prior transcript with the
instruction "add your own ideas and reach consensus". Scored here the same way
`2026-08-model-panel-agenda.md` scored the earlier panel: every number a model
quoted was checked against this repo's docs and `validation/logs/`; only what
survived was adopted.

**Bottom line:** three concrete things came out of it and are now in the repo
— the **learning-gate spec + prior art (H-A7)**, the **von Oswald citation on
H0's hardening term**, and the **counterfactual-neighborhood harvest**
(`counterfactuals[]`, built, unit-pinned, and run the same day). Everything
else is either the hypothesis doc reflected back, or fabricated.

## 1. Who actually participated

| Model | Output |
|---|---|
| Gemini Pro Latest, Grok 4.20, GPT Latest, MiniMax M3 | **nothing** — removed from the room before answering (MiniMax's one message is empty) |
| NVIDIA Nemotron 3 Ultra | round 1 only (blind pitch, ~12k chars) |
| Qwen3.7 Flash | rounds 1–4; round 5 = three web searches, no text |
| Qwen3.8 2.4T A95B | round 5 = a 246-char preamble ("one last verification pass…"), then nothing |
| **Z.ai GLM 5.3 Flash** | all five rounds (~100k chars), every web-verification pass, and the "consensus" |

So the "final consensus" is **single-authored by GLM**, which in round 1 had
also invented and judged its own four contestants (ATLAS / MOSAIC / DARWIN /
LEDGER). Its round-5 claim that "nobody saw each other's answers" is false
twice over: the room shares the transcript, and the final prompt pasted the
whole prior export. Treat the file as red-team input from one strong model
plus one blind baseline, not as five independent research runs.

## 2. The epistemic caveat that governs everything below

From round 2 on, every contestant was pitching *from* the hypothesis doc.
Agreement with the architecture (θ=f(c), block-code memory, VM verification,
exact state fork, three answer classes, calibration floors) is therefore the
document reflected back, not validation of it. The only genuinely independent
signal is **round 1** — and there all three respondents landed on the same
2026 default, which is the useful thing (§3).

## 3. What was worth taking

### 3.1 The blind-round baseline = the competitor shape

Before seeing anything of ours, Nemotron, GLM and Qwen all proposed: frozen
3–8B open base + versioned LoRA zoo (TIES/DARE/DoRA merges) + vector DB /
GraphRAG + episodic SQLite + a self-critique loop (Self-RAG/CRAG) + a
SHA-chained audit log + nightly ORPO/SimPO/KTO on thumbs. That is what a
competent team builds by default in 2026, and CubbyLLM's differentiators
fall out by diff:

| Default 2026 oracle | CubbyLLM (measured) |
|---|---|
| adapter zoo, merged at inference | θ=f(c) basis coefficients, hardened (H0) |
| vector DB + knowledge graph | block-code index (80 B/doc, 543 µs/doc) + worlds + SimHash episodic store |
| self-critique loop | verify-before-execute through the Rust VM (precision 0.9923, coverage 0.646) |
| chunk-hash replay of answers | exact state fork (`0.00e+00`) + counterfactual re-entry (H-P5) |
| online-learned router | offline-pretrained, frozen, model-free router (H0 part 2 is *why*) |

### 3.2 Prior art for the learning gate — a real hole, now H-A7

The nightly/learning gate was `[far]` in `docs/ARCHITECTURE_VISION.md` with
no hypothesis entry, no kill criterion and no experiment. GLM's round-1
"MOSAIC" section is a ready-made source list, **none of which was in the repo**:
KTO (2402.01306), O-LoRA (2310.14152), Online-LoRA (2411.05663), experience
replay (1811.11682), the ripple-effect test for edits (2307.12976), TracIn
(2002.08484), conformal LM (2306.10193), sleep-time compute (2504.13171). Its
promotion rule (Δ general-capability ≥ 0 ∧ Δ NYT-forgetting ≥ 0 ∧ needle ≥ 95%
∧ binding-health ≥ 70% ∧ claimed-precision ≥ 0.99 ∧ routing ≥ 0.90) is
composed entirely from harnesses that already exist here. Adopted as
**H-A7** in `CUBBYLLM_HYPOTHESES.md` with a cheap first experiment (three
known deltas on `hd5_mem21.pt`; the gate must reject the forgetting one) and
two kills. The arXiv IDs above are as reported by GLM's own search passes
(it corrected one of its own from memory, Online-LoRA) — **not re-verified in
this repo**; verify before citing in a paper.

**Built and run 2026-08-30** (`validation/exp_a7_learning_gate.py`,
`notebooks/a7_learning_gate.ipynb`, logs `validation/logs/exp_a7_learning_gate.*`).
Two corrections to the composed rule surfaced in the build: three of its six
terms cannot see a trunk delta (CoT precision and routing precision are
model-free surfaces; NYT-vs-NLMS benchmarks memory methods), and thresholds
must be *paired* (delta vs base on identical windows), not seed-to-seed.
Result: the gate rejects a code-only forgetting update and names it (every
unseen source +0.12–0.19 nats, the memorized `pretrain_ext` +1.09, trained
source flat), passes the null, and rejected the replay arm for a *uniform*
+0.04–0.07 on all 13 sources — whose own train loss rose 3.05 → 3.27, i.e. a
fresh-Adam-at-1e-4 perturbation of a converged checkpoint, not forgetting.
Separation inconclusive on the clean arm; the LR ladder decides. Full record
on H-A7.

### 3.3 von Oswald et al. 2019 is H0's hardening term, by name

GLM identified the "von-Oswald-style output regularization" that makes H0
work as literally the hypernetwork output regularizer of *Continual learning
with hypernetworks* (ICLR 2020, arXiv:1906.00695). Correct, and the citation
lived only in the agenda's Q6 note. Pinned on H0's validation line and in the
source map.

### 3.4 The counterfactual-neighborhood harvest (GLM's "N4") — built and run

The one new idea with a cheap experiment. The panel's Q23 had already listed
"the counterfactual neighborhood — log which planted faults each verified
chain rejects" as a regretted discard, and the harvest schema reserved
`counterfactuals[]` for it. GLM's framing added the point that it **closes the
zero-negatives calibration degeneracy as a byproduct of answering**: every
verified chain manufactures its own per-hop hard negatives, and any fault the
verifier fails to reject is a would-be false accept with a name and a hop.

Built 2026-08-28 in `validation/exp_m3_cot_pipeline.py` (`_cf_outcome`,
`_counterfactual_neighborhood`; flags `--no-counterfactuals`,
`--max-cf-vm-calls`), unit-pinned with a fake VM in
`validation/test_cot_counterfactuals.py` (5 tests: faithful-VM catch,
lying-VM escape, budget truncation, mojibake exclusion, the outcome
classifier), schema section added in `docs/schemas/cot-harvest-schema.md`.
Honest scope: GLM's version also *forked the trunk state* — the CoT pipeline
is model-free (word table + VM), so here "fork" is re-verification of a
mutated program; and Qwen's k-mutation ladder (flip k blocks of a true fact)
**cannot cross the symbolic bridge** (no raw hypervector passes), so it needs
a VM-side op before it exists.

**Run (800 eval / 200 calibration, no distractors, `_v3cf`;
`validation/logs/exp_m3_cot_pipeline_v3cf.{log,json}`, harvest
`cot_harvest_v3cf.jsonl`; headline eval unchanged from v2 — coverage 0.646,
precision 0.9923, OVERALL PASS):**

| | attempted | caught | escaped | by hop (caught/attempted) |
|---|---|---|---|---|
| wrong_entity | 598 | 598 | 0 | h0 435/435 · h1 141/141 · h2 22/22 |
| wrong_relation | 233 | 233 | 0 | h0 198/198 · h1 35/35 |
| inverted_direction | 662 | 662 | 0 | h0 472/472 · h1 165/165 · h2 25/25 |
| wrong_hop_order | 406 | 406 | 0 | h0 175/175 · h1 186/186 · h2 45/45 |
| **total** | **1,899** | **1,899** | **0** | frames n1 605/605 · n2 942/942 · n3 352/352 |

517/517 verified chains harvested (0 budget-skipped), 1,899 VM calls, 189
non-ASCII hops excluded, ~262 ms/chain — ~3.7 hard negatives per verified
answer, for free, every run.

Two honest readings. **(1) Every catch was by symbol mismatch; the floor
caught nothing.** The caught recoveries' similarity is median 0.503 / q90
1.00 — a wrong filler binds and recovers *faithfully* — so on this VM the
exact string check is 100% of the defense against content faults, and the
negatives this manufactures carry no τ information. They are hard negatives
for a *successor* verifier (a learned one, or the bundled-superposition
cleanup H-B3 still owes), not for the current floors. Fidelity ≠ truth, now
quantified on 1,899 hops. **(2) The first pass reported 3 "escapes" — all a
generator bug**, kept as `*_v3cf_run1.*`: `wrong_relation` on 3-hop chains
whose hop-0 and hop-1 objects coincide compared a recovery against itself (a
no-op fault the hop-order branch already skipped). Fixed in
`_fault_instances` + a regression test; the rerun's 0 is the number that
matches the shipped code. The harvest found a bug in the calibration
generator on its first run — which is the kind of thing it is for.

### 3.5 Frontier precedent for H-P6's "probe, not head"

GLM's round-4 search confirmed DINO-WM (2411.04983) and V-JEPA 2 / 2-AC
(2506.09985): planning over *frozen* features with a small post-trained
predictor, and Meta's own anticipation SOTA being an attentive probe on frozen
encoder features. Already in the agenda; noted here as the external
precedent for the H-P6 decision (successor readouts are post-training probes
on the frozen trunk).

## 4. GLM's five "new" ideas, graded

| # | Idea | Grade | Why |
|---|---|---|---|
| N4 | Counterfactual hard-negative factory | **adopted, built, run** | see §3.4 |
| N5 | Era-stamped claims (every verified answer carries its world/era; contradictions between worlds surfaced as "as of the 1953 world … the 2024 world holds …") | cheap product metadata; needs an experiment before it is a hypothesis | temporal worlds exist (era 0.669 / year 0.735 are real); the missing piece is a contradiction-surfacing measurement |
| N2 | "The Sibling" — a ≤1B program-proposer distilled from the harvest, compounding with coverage | this is TODO's queued trunk-SFT program generation with a name and a flywheel story | real but not cheap; depends on harvest yield, which K3 already flags (1 DPO pair / 800 records) |
| N3 | Epistemic Friction Index — decompose multi-world disagreement into resolved-by-verifier / by-debate / escalated / silent | a dashboard over trace-tree logs with an honest kill | low priority; nothing to measure until the world-backed retriever lands (Q22) |
| N1 | "Correction Propagation Guarantee" — turn H-P5's washout (KL 0.050 nats, \|Δh\| 4.6) into a regulator-facing SLA | **not adopted — conflates two mechanisms** | H-P5's washout is about in-context interventions on recurrent state; fact corrections for *verified* claims propagate through the store snapshot hash + re-verify, which is deterministic and immediate. The only place a propagation curve is non-trivial is θ=f(c)-generated params and the episodic store, and nothing was measured there. A rebrand of what the fork already gives. |

## 5. Fabrications and filler — do not reuse

**Qwen3.7 Flash** (rounds 1–4) invented or misread the following; if any of
its text is ever quoted, strip these:

- "Kalman-Estimator Verifier", "Distill-Q" — do not exist anywhere.
- "cost-per-query ~350× / 350–500× lower than dense 70B" — 350× is the
  fastword encoder's *single-text speedup vs a sentence-transformer*
  (`docs/papers/2026-08-07-qfhrr-semantic-word-tables.md` §latency), not an
  inference-cost ratio. Exactly the composed-efficiency-factor class H-D3
  forbids.
- "episodic store lifts needle recall 0% → ~95%" — measured 33–58% at 8×
  window (H-D5).
- "`exp_d1b` showed MinGRU beats attention on perplexity" offered as the
  reason for the backbone — that bake-off was *overturned* by H-D4 on
  needle recall.
- "we spent $300 validating on a Colab A100" — the campaign ran CPU-only for
  $0 (H-E3).
- chrono arm "in the 151M pilot"; frozen-surface 3-arm "H-P6 validated";
  "0.680 macro at 40× fewer comparisons" — conflations (the delegation result
  is 60× at flat-search quality; 0.680 vs 0.589 is centroid routing).
- Dopamine / serotonin / cortisol / oxytocin "analogues" — the vision doc has
  a bounded hormonal system; the named hormones are Qwen's embellishment.

**GLM 5.3 Flash** was clean on every headline number checked (0.9923, 0.646,
543 µs/doc, 0.4518, 60×, 33–58%, 2.4 s vs 5.2 s, 517 pairs, 2.734 vs 1.778,
67.15, the six-entry retraction ledger) — all trace to the pasted docs or
logs. But its round-3 **calibration card is padded with invented filler**
that exists nowhere in the repo: `grounded 0.27 · refused 0.084`, the fault
ladder `k=1 2.1% · k=3 9% · k=6 88%`, per-class FPRs `0.9% / 0.4% / 1.2%`,
`CI .986–.996, n=800×2`, `spawn-rate 87%` presented as a card line. Its
"Impossibility Triangle [2605.05066]" and GAVEL claims rest on the agenda's
`[panel-cited]` tags plus its own searches — same trust level as before, not
upgraded.

## 6. Prior-art IDs surfaced (as reported by the contestants' search passes; not re-verified here)

Learning gate: KTO 2402.01306 · O-LoRA 2310.14152 · Online-LoRA 2411.05663 ·
replay 1811.11682 · ripple effects 2307.12976 · TracIn 2002.08484 · conformal
LM 2306.10193 · sleep-time compute 2504.13171 · RouteLLM 2406.18665 · LinUCB
1003.0146 · doubly-robust OPE 1103.4601 · Data Shapley 1904.02868 · MEMIT
2305.14795 · GaLore 2403.03507 · Absolute Zero 2505.03335 (the self-play
route — dead here by H-B6). H0: von Oswald 1906.00695 · HyperNetworks
1609.09106. Prospection precedent: DINO-WM 2411.04983 · V-JEPA 2 2506.09985 ·
chrono init 1804.11188. Audit: SelfCheckGPT 2303.08896 · RARR 2210.08726 ·
CRAG 2401.15884 · ALCE 2305.14627 · indirect prompt injection 2302.12173
(GLM's one design note nobody else made: an always-on ingester is a
prompt-injection magnet — the ingest path needs that threat model day one;
the vision doc's guardian/`net` items already say this, unmeasured).

## 7. Pointers

- Raw export: `docs/model-competition-for-cubby-finalconsensus.json`
- Adopted: `CUBBYLLM_HYPOTHESES.md` H0 (citation), **H-A7** (new), the
  H-F2/H-B3 CoT entry (counterfactual harvest addendum), §10 item 12.
- Code: `validation/exp_m3_cot_pipeline.py`, `validation/test_cot_counterfactuals.py`
- Logs: `validation/logs/exp_m3_cot_pipeline_v3cf.{log,json}`,
  `validation/logs/cot_harvest_v3cf.jsonl`
- Schema: `docs/schemas/cot-harvest-schema.md` → `counterfactuals[]`
