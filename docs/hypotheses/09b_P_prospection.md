<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 9b. Group P — Prospection: branching futures, timescales, keyframe compression (added 2026-08-25)

Origin: a design conversation (2026-08-25) framing a "5th dimension" — the landscape of alternate futures and pasts
indexed by the choices made along the 4D trajectory — as a capability target: counterfactual timelines ("what if
Alexander the Great was never born?") and goal-directed choice selection by predicted short/medium/long-term outcomes.
Operationally the object is path space with a measure over it, and an LM already contains it implicitly (every softmax
is a one-step fan of branches); what it lacks is an explicit handle. The five hypotheses below are the parts that are
*about this architecture* and cheaply falsifiable now. The parts that are not — a causal graph over history with a
stated held-fixed set, multi-γ successor-feature heads, eligibility traces plus episodic write-back for delayed credit,
goal vectors with generalized policy improvement, Dyna-style dreaming behind a grounding filter, provenance-tagging of
imagined episodes in the store — are a roadmap, not claims, and are deliberately NOT entered here until each has a cheap
experiment. Scripts: `validation/prospection/` (its README has the run matrix and the checkpoint env vars);
`validation/prospection/test_prospection.py` runs all five quick. Toy-scale numbers are d=32, L=2, CPU — mechanism
demonstrations; the checkpoint arms ran the same day on `$CUBBY_DRIVE\cubbyllm\ab_hybrid.pt` (d=512/L=8, the H-D4
winner) and `hd5_mem2.pt`, and where the two disagree the checkpoint is the one that counts.

**H-P1 — The decode state is a forkable world snapshot: branching from it is exact, O(1) in context length, and
isolated.** Source: `HybridBackbone.step` / `MinGRUBackbone.step` carry fixed-size state (the H-D4 bounded-decode
guard), and `exp_needle_recall.py` already branches candidates from one prefix state. Validate: k branches from one
cloned state vs k independent full passes; state floats vs prefix length; origin/store isolation. Kill: any fp drift,
any growth with prefix, any leak. **Validation result (2026-08-25): VERIFIED at toy scale** — max |forked − independent|
= 0.00e+00 on mingru, hybrid and hybrid+episodic-memory; state floats 96 / 577 / 1345 constant across prefixes 8/32/64;
no origin mutation, no store leak; 96 vs 240 `step()`s for 4 branches (`validation/logs/exp_p1_fork_equivalence.log`).
**Checkpoint arm (same day): VERIFIED at scale** on `ab_hybrid.pt` (d=512/L=8, the H-D4 winner) and `hd5_mem2.pt` (+
episodic memory every 2nd layer): exact at 0.00e+00 on both; state bounded past the window at 1,575,427 floats (≈6.3 MB
— 1,572,864 of them the three attention KV caches, 2,560 the five recurrent layers) and 2,755,075 with the memory FIFOs;
store isolation holds at scale (`validation/logs/exp_p1_fork_equivalence_ab_hybrid.log`, `_hd5_mem2.log`). **Honest cost
accounting:** the hybrid's fork/keyframe cost is the attention window, not the recurrence — a snapshot re-enterable
without the preceding 512 tokens is ~6 MB, not ~10 KB; with tokens stored, a recurrent-only keyframe at t−window plus a
512-step recompute is exact and small. This is the structural argument for building counterfactual machinery on this
backbone rather than on a KV cache that grows with context, and it is the primitive H-P4/H-P5 rely on.

**H-P2 — Choice points are sparse and the model's own next-token entropy localizes them, so "open branches only at
high-entropy steps" is a budget rule with a real signal behind it (vicarious trial and error happens only at decision
points).** Source: the design conversation; Redish (2016) for the biological shape. Validate: on the branch grammar
(exactly one free choice per 8-token segment, everything else deterministic) AUROC of entropy vs the labeled choice
positions on the parallel and the decode path; flagged fraction vs the true 1/8. Kill: AUROC well under 0.9 or the
fraction far off. **Validation result (2026-08-25): VERIFIED at toy scale** — loss 0.198 nats/token vs the grammar floor
0.173; entropy 1.375 (≈ ln 4) at choice points vs 0.002 elsewhere; AUROC 1.000 on both paths; flagged fraction 0.125
(`validation/logs/exp_p2_choice_points.log`). **Checkpoint arm (same day): the toy result does NOT transfer as-is.** On
`ab_hybrid.pt` over 4,095 positions of real Wikipedia text (80 articles across the 20 domain files next to the
checkpoints, `grillcheese_bbpe128k.json`): entropy median 4.32 nats, p90 6.14; fraction above 1 / 2 / 3 nats = **0.914 /
0.830 / 0.701** (`validation/logs/exp_p2_choice_points_ab_hybrid.log`). At this scale choice points are not sparse — an
absolute threshold would open branches at 70–90% of steps. Status of the *budget rule*: **PLAUSIBLE, needs
reformulation** — a relative criterion (entropy above the model's running median, or the top decile within a window) or
a stronger model; the detector claim (entropy localizes the choices that exist) stands, the sparsity claim is a property
of the model's confidence, not of text.

**H-P3 — The MinGRU retention gate is uniformly short-memory at init, cannot represent time constants above ~1000 steps,
and only lengthens as far as the data forces it; a chrono-style init spreads the spectrum over decades without touching
the scan or the decode path.** Source: `mingru.py` (`a = 0.001 + 0.998·σ(proj_d x)`, bias init 1.0); Tallec & Ollivier
2018 (chrono init); Murray et al. 2014 (cortical hierarchy of intrinsic timescales). Validate: per-unit τ = −1/ln a from
the geometric-mean retention over real inputs at init / after toy training / after chrono init; pin the formula against
`_MinGRUMixer.step`; delayed copy default-vs-chrono at lags 8/32/128. Kill: any of the three substrate facts wrong.
**Validation result (2026-08-25): VERIFIED at toy scale** — init τ p50 ≈ 2.8 steps for every unit; **τ ceiling 999.5
steps** (a ≤ 0.999 by construction); toy training moved no unit past τ ≈ 4 (its longest dependency is 3 steps — gates
lengthen only as far as the data demands); chrono init τ p10/p50/p90 ≈ 2.4 / 15 / 260 with 22% of units slower than 100
steps; delayed copy (300 steps, informational): default init sits at chance beyond lag 8 (2.71 vs ln 16 = 2.77) while
chrono reaches 2.52 at lag 32 and 2.60 at lag 128 (`validation/logs/exp_p3_gate_spectrum.log`). **Checkpoint arm (same
day) — the load-bearing finding of the group:** on `ab_hybrid.pt` over 2,048 real tokens the five recurrent layers have
median τ = 0.5 / 0.7 / 1.1 / 1.5 / 1.8 steps (layers 1/2/4/5/7) — **shorter than init (2.8)**; 0.5% of recurrent units
have τ ≥ 8, none ≥ 100, slowest single unit 111 against a cap of 999; conditional latching (instantaneous τ ≥ 100) does
not hide long memory either — 1.2% of units latch on ≥10% of steps, the p90 unit on 0.4%
(`validation/logs/exp_p3_gate_spectrum_ab_hybrid.log`). **After ~1B tokens the MinGRU layers of the hybrid are 1–2-step
mixers; every long-range dependency lives in the three windowed-attention layers.** That is consistent with H-D4 (pure
MinGRU ~32% recall vs ~98% for the hybrid inside its window) and it reframes the whole timescales thread: there is no
slow subspace in the trained substrate to hang a multi-timescale scheme on — long time constants have to be *made*
(chrono init, an explicit long-horizon objective, or an LRU-style rotation-block recurrence), or the timescale machinery
lives in attention and the episodic store, where the H-D5 memory rung already is. **Decision pending, not made:**
whether the 2B run uses chrono init on its recurrent layers. It is free and decode-neutral, but the long-lag gain is
small at d=32 and unmeasured at scale — one pilot arm, not a default, and the checkpoint spectrum above is the arm's
baseline. The ceiling is a constraint to carry into any "learn across timescales" design: anything slower than ~1000
steps has to live in the episodic store or the attention window, not in a gate.

**H-P4 — Keyframe compression of the state trajectory should be recompute-from-keyframe (exact, cheap by H-P1), not
interpolation; interpolation earns its place only on slow units, by the sampling theorem.** Source: the AURA note
`$CUBBY_DATA_ROOT/AURA_MIND\papers\temporal_memory_interpolation.md` (memory compression as keyframes + regenerated
in-betweens via Hilbert / Hamiltonian / Fourier interpolation); Delétang et al. 2023 for the wider
prediction-is-compression principle. Validate: linear-interpolation error vs hold-last per gap; per-unit error vs τ;
recompute error; bytes/step; numerical pins on the note's operators. Kill: interpolation not under hold-last at gap 4;
recompute not exact. **Validation result (2026-08-25): VERIFIED at toy scale, and the note's operators are FALSIFIED as
written** — with two keyframes and no time axis, the Hilbert and Fourier strategies equal plain linear interpolation to
4e-16 (Re and F are linear maps), and the rank-one Hamiltonian flow conserves the analytic-signal norm (drift 4e-10) and
misses keyframe 1 by 0.715 relative — it rotates one component's phase, it does not travel; on the toy (τ ≈ 3) linear
interpolation rel-RMS 0.59 / 0.83 / 1.16 at gaps 2 / 4 / 8 vs hold-last 0.80 / 1.03 / 1.26 — barely better than doing
nothing, worse than the unit mean from gap 8; Spearman(log τ, error) = −0.38, so slow units do interpolate better and
the multi-resolution rule has signal; recompute exact at 0.0e+00; 256 B/step of states vs 32.5 B/step for keyframes +
tokens at gap 8 (`validation/logs/exp_p4_keyframe_interp.log`). **Checkpoint arm (same day): VERIFIED at scale, same
verdict, stronger** — on `ab_hybrid.pt` over a 1,024-token real trajectory (5 recurrent layers × 512): linear
interpolation rel-RMS 0.74 / 0.87 / 1.00 / 1.10 / 1.17 at gaps 2 / 4 / 8 / 16 / 32 vs hold-last 0.92 / 1.10 / 1.21 /
1.30 / 1.34 — worse than the unit mean from gap 8; Spearman(log τ, per-unit error) = **−0.88**; recompute exact at
0.0e+00; recurrent state 10,240 B/step vs 1,282 B/step for keyframes + tokens at gap 8, with the attention KV cache on
top per H-P1 (`validation/logs/exp_p4_keyframe_interp_ab_hybrid.log`). Interpolation-based compression is dead on this
model. **Design consequence for the store:** keep tokens plus sparse keyframes and recompute — reconstructed states are
generated, so if they are ever written back they carry the same provenance tag an imagined episode must. If a
phase-aware slow subspace is wanted, the rigorous form is a complex-diagonal (rotation-block) recurrence à la LRU
(Orvieto et al. 2023), not interpolation between two real vectors — SPECULATIVE, unbuilt, no experiment yet.

**H-P5 — Counterfactual re-entry works mechanically on a world the model fully knows: snapshot at a choice point,
substitute the choice, re-run, and the continuation switches to the ground-truth alternative while overdetermined
outcomes stay put.** Source: Pearl's ladder (abduction → intervention → prediction) with the recurrent state as the
abduced latent; the design conversation's fragile-vs-overdetermined distinction. Validate: on the branch grammar,
teacher-forced counterfactual accuracy vs template_j; agreement on shared-prefix positions; flip on the first divergent
position; next-choice washout KL; state-distance profile. Kill: counterfactual at chance, overdetermined predictions
changing, no washout. **Validation result (2026-08-25): VERIFIED at toy scale** — factual control 1.000; counterfactual
1.000 (chance 0.125); overdetermined unchanged 1.000; fragile flipped / correct 1.000 / 1.000; washout KL 0.050 nats at
the next SEP while |h_f − h_cf| is still 4.6 — washout is a readout property, not a state one, so an intervention's
residue persists in state past the boundary and its horizon is a number to measure, never assumed zero
(`validation/logs/exp_p5_counterfactual_probe.log`). **What this does NOT show:** rung three on a world the model does
not fully know — history — where the dependency structure has to come from the DBpedia/Wikidata graph and the held-fixed
set has to be stated explicitly ("never born" is entangled with Philip's court before 336 BC; "dies at the Granicus in
334" is clean). That is a bridge/graph item in H-F2's territory, not a backbone item, and no experiment exists for it
yet.

**H-P6 — A multi-horizon (successor-feature) prediction head on the trunk is free at matched tokens and makes the
medium-term future (8–500 tokens) linearly readable from h_t — without any change to the backbone. Added 2026-08-25.**
Source: the design conversation's short/medium/long-horizon mapping plus the H-P3 checkpoint finding that the trained
recurrence carries no long time constants, so the medium horizon must live in the trunk representation and the attention
window, not in a gate; Dayan 1993 / Barreto et al. 2017 (successor features: predict discounted future feature
occupancy, then re-target goals by re-weighting features). Construction: per horizon γ ∈ {0, 0.875, 0.98, 0.998} (mean
horizons 1 / 8 / 50 / 500 tokens) one linear map h_t → the discounted average of the future tokens' centered unit
embeddings, targets computed inside the training window by the package's own log-domain scan run time-reversed, targets
DETACHED (the H-B6 lesson: an aux loss that can reach its own target Goodharts), loss 1−cos, weight 0.1 next to
next-token CE. **Validate** (`validation/prospection/exp_p6_multihorizon_pilot.py`,
`notebooks/multihorizon_pilot.ipynb`, ~15 min/arm on an A100 at d=512/L=8/2000 steps): two matched arms, then held-out
CE; per-horizon cosine skill vs the trivial mean-direction predictor; a linear probe fitted post hoc on frozen features
of BOTH arms (how much future the trunk carries anyway vs how much the objective adds); the H-P3 gate spectrum of both
arms (does an explicit long-horizon objective lengthen any recurrent time constant?); the representation-health and
copy-floor gates. **Kill:** held-out CE worse than baseline beyond run-to-run noise (~0.01 nats at 65M tokens); or skill
only at horizon 1 — then the trunk cannot express the medium-term future at this scale and the multi-γ design is dead.
Status: ~~SPECULATIVE / UNRUN~~ **RUN 2026-08-26 on an A100-80G (`notebooks/multihorizon_pilot.ipynb`, d=512/L=8 hybrid
1:3, B=32/S=1024, 2000 steps = 65.5M tokens per arm, 16.4 / 16.6 min, 62.6 GB peak; the completed head arm ran at weight
0.03 — a 0.1 attempt was stopped at ~500 steps, where its head cosines matched 0.03's within 0.01). Logs:
`validation/logs/exp_p6_{baseline,multihorizon}.{json,log}`.** Three findings:
(1) **"Free" — VERIFIED.** Held-out CE 5.0025 (baseline) vs 5.0044 (+head): Δ = +0.0019 nats, +0.003 bits/token, inside
eval noise; the train curves match to three decimals once the aux term is subtracted (step 2000: 4.864 vs 4.948 − 0.081
= 4.867). Health gates identical in kind: retrieval 95.3% vs 92.2% (both fluctuated 90.6–93.8% during training), ff
0.415 vs 0.418, copy-freq 12% in both (the copying circuit emerged by step 500 in both arms).
(2) **"Medium-term future linearly readable" — VERIFIED, but the head is not what makes it so.** With centered
unit-embedding targets the trivial mean-direction predictor scores 0.004 / 0.014 / 0.031 / 0.079 at horizons 1 / 8 / 50
/ 500. A linear probe fitted post hoc on the frozen features of the *baseline* arm scores **0.188 / 0.232 / 0.331 /
0.414** — the trunk already carries the 50- and 500-token future without ever being asked to; the probe on the head arm
scores 0.191 / 0.240 / 0.348 / 0.437 (+0.003 / +0.008 / +0.017 / +0.023) and the trained head itself 0.194 / 0.237 /
0.340 / 0.426. Longer horizons are *easier* (averaging over more future tokens), the next token is the hardest. So the
successor-feature quantity the design wants is a **post-hoc readout on the trunk, not a pretraining objective**: the
explicit loss buys ~0.02 cosine at 500 tokens over a probe that costs nothing and risks nothing.
(3) **"An explicit long-horizon objective lengthens the recurrence" — FALSIFIED at this scale/weight.** The gate spectra
of the two arms are identical to two decimals: recurrent-layer τ p50 = 1.18 / 1.40 / 1.86 / 2.01 / 2.09 (layers
1/2/4/5/7) in both, no unit ≥ 8 steps in either. The horizon information lives in the attention layers and the trunk's
composition, exactly as H-P3's checkpoint arm predicted. Bonus trajectory point from the same run: τ p50 ≈ 2.0 flat at
step 500 (`_mh_baseline_step500` logs), 1.2–2.1 at step 2000, 0.5–1.8 at ~1B tokens (`ab_hybrid`) — the recurrence
shortens monotonically with training from its init of 2.8.
**Decision:** the head does NOT go into the 2B runbook, not even as an arm — there is nothing for it to buy that a
300-step linear probe on the finished trunk doesn't already deliver. The multi-γ successor readouts become a
post-training component of the prospection design (fit on frozen features, like the FastWord table is model-free), which
is a strictly cheaper and safer place for them. The remaining open route to long *recurrent* time constants is chrono
init (H-P3's pending pilot); the objective route is closed. **Caveat on the checkpoint files, not the numbers:** the
`mh_baseline.pt` that reached Drive is the step-500 periodic save (Drive sync race across five 1.8 GB writes;
`mh_multihorizon.pt` is step 2000), so the optional P2/P3 cell measured a step-500 baseline — logged as
`*_mh_baseline_step500.log` and valid as a step-500 point only; every number above comes from the in-script evaluation
of the in-memory step-2000 models in the JSON. The script now saves once, staged locally, and verifies the step.

**H-P7 — Chrono init on the recurrent gates survives training on the hybrid and gives the recurrence a usable memory
beyond the attention window, at no cost in held-out CE. Added 2026-08-26 — the last open route to long *recurrent* time
constants.** Source: Tallec & Ollivier 2018 ("Can recurrent neural networks warp time?" — gate biases set so the time
constants are log-uniform over the range the task needs); H-P3's measured trajectory (τ shortens 2.8 → 2.0 → 1.2–2.1 →
0.5–1.8 from init through ~1B tokens, cap ~1000) and H-P6's closure of the objective route (a 500-token prediction loss
moved no time constant). The specific doubt this tests: whether gradient descent keeps a long-τ spectrum when windowed
attention is available to do the long-range work more cheaply — the same pressure that eroded the default init.
Construction: `_common.chrono_init_` on every recurrent layer's `proj_d` bias, τ log-uniform in [1, 500] (below the
gate's 999 ceiling), bias-only (weight scale 1.0 — the paper's form; P3's CPU measurement used 0.1 only to make the
init spectrum legible). **Validate** (`validation/prospection/exp_p7_chrono_pilot.py` — a thin wrapper over the P6
pilot runner — and `notebooks/chrono_pilot.ipynb`; ~16 min/arm on an A100 at d=512/L=8/2000 steps plus needle eval):
two matched arms, CB_CHRONO=0/1, then in this order (1) held-out CE at matched tokens; (2) **survival** — the gate
spectrum at init vs after training, per layer (the runner now records both); (3) **beyond-window memory** — the impulse
response of the stacked recurrent state to one substituted token, read at lags past the 512 window, where only the
recurrence can carry it (baseline reads 0.000 there; pinned on a tiny hybrid in `test_prospection.py`: default init <
1e-3 at lag 128 past a 16-window, chrono > 10× that); (4) **capability** — `exp_needle_recall.py` on both checkpoints
at 256/512/1024/4096, TRAINED vs its own shuffled control. **Kill:** CE worse beyond ~0.01 nats; or the trained
spectrum collapsing onto the baseline's (p50 1–2, no unit ≥ 8) — then chrono erodes like the default init and the route
is closed for good, leaving the episodic store as the only beyond-window mechanism (H-P3's conclusion becomes final).
Survival with a non-zero beyond-window impulse response but no needle gain is a recorded fact, not a pass: the gate
*can* hold a long time constant but has nothing useful to put in it at this scale. CPU smoke (2026-08-26): chrono arm
init τ p50 16–23 with 20–30% of units ≥ 100, preserved after 30 steps; impulse response 0.115 at lag 64 past a
32-window vs 0.000 for the default init. **Status: RAN 2026-08-26 (Colab A100, two matched arms, d=512/L=8/window 512,
2000 steps × 32 × 1024 = 65.5M tokens/arm, ~15–16 min/arm; logs `validation/logs/exp_p7_{baseline,chrono}.{log,json}`,
copied from Drive 2026-08-28; checkpoints `$CUBBY_DRIVE\cubbyllm\chrono_{baseline,arm}.pt`) — KILLED on clause (1),
verdict recorded 2026-08-28.** (1) **Held-out CE 5.442 vs 4.998 nats — +0.444, 44× the 0.01-nat kill threshold**, at
matched tokens with the LR already decayed to 3e-5; the gap shrinks monotonically (51 at step 1 — the paper-form
bias-only init with weight scale 1.0 starts the recurrence saturated, train loss 63 vs 12 — 4.1 at step 100, 0.73 at
500, 0.53 at 1000, 0.44 at 2000) but is decelerating, not closing; bits/token 7.85 vs 7.21. (2) **Survival: partial —
eroded ~4×, did NOT collapse onto the baseline**, so the second kill trigger did not fire: trained τ p50 4.4–9.4 per
layer (init 18–28; baseline 1.2–2.2), units ≥ 8 steps 40–54% (init 62–70%; baseline 0), units ≥ 100 steps 5–10% (init
23–27%; baseline 0). (3) **Beyond-window impulse response: non-zero but ~1e-3** — at lags 768 / 1000 past the 512
window, 0.0015 / 0.0010 vs the baseline's 0.00016 / 0.00002 (9× / 45×), while *inside* the window the chrono arm's
response is about half the baseline's (0.25 vs 0.53 at lag 1: slow gates damp the immediate signal). (4) The
needle-recall capability read was not run by the Colab session (no log exists) and was deliberately not run afterwards:
a CPU smoke on 2026-08-28 (`exp_needle_recall.py` on `chrono_baseline.pt`, 256 tokens, 2 trials, local
`$CUBBY_TRAINING_DATA\token_cache`) confirmed the probe runs at ~5.5 s/trial at 256 tokens, which prices the full
pre-registered read (4 lengths × 5 depths × 12 trials × 3 arms × 2 checkpoints) at ~13 CPU-hours — for a read that
cannot inform anything: (1) already kills, and at this pilot's 65.5M-token budget the *baseline* is far below the ~1B
tokens at which H-D4's hybrid first showed in-window needle recall (copy-frequency health 12% / 9% says neither arm
copies from context yet), so both arms would read near chance regardless of the gates. The smoke's own 0/2 is n=2
noise, not a number. Side readings: frozen-trunk probe at 500 tokens 0.447 vs 0.416 (slightly better on the worse model
— not comparable at a 0.44-nat CE gap); binding health retrieval 92.2% vs 95.3%, ff-cos 0.43 vs 0.41; copy-frequency 9%
vs 12%. **Reading: this is the entry's pre-registered "recorded fact, not a pass" case at full strength — the gate
*can* hold a long time constant through 65M tokens of training, and it has nothing useful to put in it at this scale,
at a cost of 0.44 nats.** Decision: chrono is **not** a cycle-one arm at the 2B pilot; H-P3's conclusion becomes final
— *beyond-window memory lives in the attention window and the episodic store, not in the recurrence*. If anyone
re-opens this, the only arm worth running is the un-pre-registered one — weight scale 0.1 (P3's smoke form) and τ_max ≈
64 to avoid the saturated start, at 4× tokens — and it would need to close the whole 0.44 before survival even matters;
not queued.

**H-P8 — A fork stays told apart only as long as the hybrid trunk can carry the substitution: inside the attention
window by attention, beyond it at the recurrence's floor (H-P7: ~1e-3). So a branch's program (order 0046's arm B3) and
a counterfactual re-entry (H-P5) must be written within the window, and every longer "what if" must hand off to the
episodic store. Added 2026-10-31 (Nick: "let's also check how it compares to" LeHami).** Source: LeHami (Yalovetzky,
Balestriero, LeCun, arXiv 2610.15267, 2026). Its symplectic (leapfrog) latent integrator preserves phase-space volume,
so distinct states never collapse over long rollouts, and a rollout runs backward exactly. The gated recurrence here is
contractive by construction (a_t < 1), so forks that differ by one choice converge at the retention time. H-P7's impulse
response measured this for one token on a toy hybrid (0.25-0.53 at lag 1, ~0.0015 past the window). What is not
measured: the same on the trained trunk (base450m, and the grown trunk after 0045), for the substitutions B3 makes (a
candidate item in place of a shared name), read where it matters: the program the branch writes. **Validate**
(`validation/prospection/exp_p8_fork_separation.py`; CPU or any card, no training): take N = 200 questions from 0046's
frozen fame set, two candidates each. Fork the trunk's state at the name (H-P1), substitute each candidate's label
(H-P5), and continue with the same suffix: the rest of the question, then a fixed program prefix, then neutral text out
to 4,096 tokens. Read per lag: (a) the hidden-state distance between the forks, per layer, normalized by the distance
between two unrelated questions; (b) the KL between the forks' next-token distributions; (c) whether the program each
fork writes names its own candidate (the B3 read). Report the lag at which (a) and (b) fall to the H-P7 floor, inside
and beyond the 512 window, for the 450M and, once it exists, the grown trunk. **Pass** (a recorded fact that sets B3's
design, not a build): (c) holds for at least 95% of fork pairs at the lengths 0046's questions and programs reach.
**Kill** (for B3's forked-trunk arm): (c) fails above 5% inside the window. Then B3 writes each branch's program from a
fresh read of the substituted question, not a fork. **Beyond the window:** whatever (a) and (b) read is how far a "what
if" reaches in the state. Past it, only the episodic store carries a consequence: Group P's decision of 2026-08-25, made
measurable on the real trunk. **Recorded, not queued:** LeHami's own remedy is a volume-preserving side state (separable
Hamiltonian, leapfrog) that never forgets and runs backward, so a choice point could be re-entered by running back
instead of from a keyframe (H-P4). It is the wrong cell for language, which must forget. It earns an entry only if this
test shows the episodic store cannot carry what long-horizon prospection needs. Status: **proposed** (2026-10-31).
