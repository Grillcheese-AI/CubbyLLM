<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 3. Group A — Memory & forgetting fix

**H-A1 — Diagnosis: the Hebbian layer fails because it runs at C_ctx≈0.** The Impossibility Triangle (zero forgetting +
online learning + finite parameters can't all hold for a state-based learner) predicts this. Status: **VERIFIED
(2026-07-23 campaign, both halves)** — structurally, the real `HebbianGrowthLayer` update is dW = f(x, xWᵀ, W) with no
context input existing anywhere in its signature or update path, so C_ctx = 0 *by construction* (no estimation needed);
empirically, after learning a task to EV 0.97 the shared basis retains EV 0.06 (**forgot 94%**) after three subsequent
tasks; neurogenesis mitigates to 72% forgotten but doesn't rescue. Two incidental findings: the layer's default
`nonlinear` (y³) mode never tracks subspace structure under a reconstruction metric (only `sanger` mode learns), and its
residual metric becomes unreliable (EV>1) once grown bases decorrelate. See `VALIDATION_REPORT.md` /
`validation/exp_a1_drift.py`.

**H-A2 — GCE-style generation removes catastrophic forgetting in CubbyLLM's memory layer.** Status: **toy-scale VERIFIED
in the hardened form** (2026-07-23 campaign — see H0's validation result above for the numbers; the toy experiment *is*
this hypothesis's specified side-experiment). Kill criterion checked and not triggered: bypass did not survive
hardening. Still open at scale: the hardening cost curve and whether the mechanism stays scoped to the memory pathway in
a real model — the second kill clause is untested until there is a real network to scope it in.
`validation/exp_h0_gce.py`.

**H-A3 — A Sparse Distributed Memory substrate is more forgetting-resistant than dense Hebbian updates, purely by
construction.** Source: `cubby-concepts/vsa/sdm.py` — real code, independently verified this session (staged the files,
ran `pytest` directly: 61/61 tests passed, 81.6s wall time). Verified recall: 100% accuracy (1000/1000, cosine
similarity 0.9957) at M=D=10000 with 1000 stored associations. Each write spreads across roughly 40 of 10,000 hard
locations (calibrated Hamming radius) instead of overwriting one dense weight matrix — the standard SDM
graceful-degradation argument versus Hopfield-style catastrophic collapse. Status: **the mechanism is VERIFIED; the
forgetting-resistance property itself is PLAUSIBLE, not proven** — the cubby-concepts repo has no test that pushes the
SDM toward/past capacity or compares its degradation curve against a Hebbian baseline. The repo's own CLAUDE.md names a
"Zero-Forgetting Stability Benchmark" but it doesn't exist as code anywhere — file, function, or test. Also note: the
repo proves the CLAUDE.md-specified Hamming radius of 4500 bits is actually broken (activates ~0 locations at full
dimension) and ships a regression test for that fixed bug — a good sign for the code's honesty, a reminder to use the
calibrated radius, not the originally-specified one. **Validate** by building that benchmark (it doesn't exist yet) and
running it against SDM and the current Hebbian layer side by side.

**Validation result (2026-07-23 campaign): the benchmark now exists (`validation/exp_a_forgetting.py`) and the answer is
split.** At matched state (~D² numbers each, D=2048, SDM M=D, online single-pass): SDM is the **only order-agnostic
candidate** — at N=1024 its recall is ~15% uniformly across write order, while every dense method is recency-biased or
totally collapsed (pure Hebbian: 0% for old AND new — overload destroys everything). That's the "graceful by
construction" half, CONFIRMED. The capacity half is REFUTED at parameter parity: SDM's curve (99%→77%→15% at
N=128/512/1024) sits below NLMS (100%→100%→51%) — the famous 100%@1000 cubby-concepts result used M=D=10000, ~5× the
state per stored pair. SDM buys *fairness*, not *capacity*. Status: mechanism VERIFIED, blanket forgetting-resistance
claim now has honest boundaries.

**H-A4 — NLMS is a cheaper drop-in replacement than a full hypernetwork rewrite.** Source:
`AURA_GENESIS/aura/core/nlms.py` — real, complete, ~14KB, textbook Normalized LMS (`w ← (1−l2)·w + mu·(e·x)/(‖x‖²+ε)`),
extended with group-sliced per-feature learning rates, a prosody-style gate, and dead-zone guards that skip updates when
the prediction is already close enough. NLMS is error-corrective and self-normalizing by input energy, which classical
adaptive-filter theory uses to bound weight drift — a real structural advantage over raw Hebbian/Oja. Status: PLAUSIBLE,
with an important caveat surfaced by the AURA_GENESIS survey: AURA's actual anti-forgetting strategy is not "a smarter
shared update rule," it's hard parameter isolation — one private linear head per domain/realm (88 domain files + ~150
realm files under `svc_nlms_weights/`), created on demand and never shared across concepts. That sidesteps interference
by construction, not through NLMS's math, and doesn't obviously transfer to a shared dense memory layer where isolation
isn't an option. **Validate** by swapping Oja for NLMS in one memory update path and rerunning the same forgetting
benchmark as H-A3, in a setting where weights are actually shared (not per-concept-isolated) — that's the setting
AURA_GENESIS never tested.

**Validation result (2026-07-23 campaign): SUPPORTED in exactly that never-tested shared-weight setting, with its
failure mode now named.** In the H-A3 benchmark, matrix NLMS posts the best capacity curve of every candidate (100%
recall through N=512, 51% at N=1024, graceful decline, no catastrophic cliff) — clearly better than pure Hebbian (cliff
to 0% at 1024) and *vastly* better than the flat-decay Hebbian production form (65.6% already at N=128, worst curve
measured). The cost: strongest recency bias past capacity (write-order quartiles 0/11/93/100 oldest→newest) — NLMS
forgets oldest-first, LRU-like. As a cheap upgrade over Oja: yes. As zero-forgetting: no.
`validation/exp_a_forgetting.py`.

**H-A5 — An external episodic store can complement, not replace, a fixed weight-based memory layer.** Source:
GrillCheese's `capsulememoryintegration.md` (833 lines, full read) — a hippocampus-inspired design: Dentate-Gyrus-style
sparse expansion (32D→128D, ~2% top-k winner-take-all) feeding CA3-style FAISS pattern completion, with
importance-scored eviction (`score = consolidation_priority · stability · (recency + log(access_count))`, top 80% kept,
protected items never evicted). STATUS.md corroborates a running instance (97 memories stored). This is architecturally
a complementary-learning-systems split (fast episodic store outside the weights + slow weight-based consolidation), the
same philosophy as Memorizing Transformer/MemGPT — it never touches synaptic plasticity at all, so it's orthogonal to,
not competing with, the Hebbian fix. Status: PLAUSIBLE as a complementary mechanism, not a fix for the underlying
problem. One documented weakness: the residual-stream injection hook (tiling a 32D vector across a 3072D residual stream
with no learned up-projection) is methodologically weak; the text-context injection half is more credible. **Validate**
by prototyping the DG-style sparse-expand + pattern-completion step as a side channel and measuring whether it reduces
pressure on the weight-based memory — explicitly not a substitute for fixing that memory.

**Validation result (2026-07-23 campaign): mechanism VERIFIED at toy scale.** DG-style expansion (8× up-projection, 2%
WTA) over a Hebbian store, with a *size-matched dense control*: at N=512 associations both dense variants collapse to 0%
recall while the sparse code holds 100% — pattern separation, not parameter count, buys ~2× capacity (its own cliff
arrives at N=1024). Also observed: dense Hebbian fails as a cliff (100%→0% between N=256 and 512), not a graceful curve.
Complement framing confirmed. `validation/exp_a5_sparse.py`.

**Order 0083 (2026-11-25, Code B): a certified plan remembered before the 450M writes one (`CB_ASK_MEMORY`); killed as
scored, memory stays off.** On the gen-3 held split (seed 7, 600 rows, asked twice) the second asking gave back 241 of
the first asking's 242 answers at a median 0.008 s on those rows (a certified answer through the seat takes 2.46 s), and
a miss cost 0.047 s. The kill (any wrong answer from a remembered plan) fired: 4, 8 and 5 scored wrong in the first,
second and reworded passes. Each is an answer the stack also gives without memory, on that row or for that person in
other words, and one is real: "Ghana" for a person's citizenship, from a seat defect `covers()` lets through. Memory
widens what the gate already says, wrong answers included, so it is only as safe as `covers()`.

**H-A6 — Negative-result flag: no sibling folder has a validated fix for this problem.** All four surveys converge on
the same finding. GrillCheese's own production Hebbian layer (`grillcheese/backend/brain/gpu_brain.py::hebbian_update`)
implements `ΔW = η·⟨pre⊗post⟩ − λ·W` — a flat decay term, not true Oja normalization (no `post²·W` term, no
norm-convergence guarantee) — the same failure class CubbyLLM is trying to fix, with a documented live shape-mismatch
bug in the same function. Its own `forreview/capsules/DECISION.md` lists "Add EWC for continual learning" as still-open
future work. Grepping GrillCheese's entire `learning/` package turns up no EWC, no orthogonal projection, no
replay-based consolidation — only a crude activation-threshold "neuronal growth" heuristic, loosely adjacent to SPHeRe's
structural-projection idea but untested against it. Novel_GNN_Arch's closest analog, `apply_outcome_update()`
(importance-gated plasticity — a per-memory-slot effective learning rate driven by a plasticity trait and a simulated
reward-prediction-error signal), is a well-shaped design sketch conceptually close to Elastic Weight Consolidation
applied at memory-slot granularity, but it's ~250 lines of hand-tuned, duplicated-across-two-files demo code, tested
only on 2–10 hardcoded synthetic records with randomly sampled (not task-derived) outcomes — zero quantitative
forgetting-benchmark evidence either way. **Conclusion, not really a hypothesis: CubbyLLM cannot shortcut this problem
by importing a sibling's solution. Budget real R&D time and benchmark every "reuse" candidate rather than adopting any
of them on faith.**

**Validation result (2026-07-23 campaign): every sub-claim fact-checked by direct file reads and CONFIRMED**, with three
precision fixes: GrillCheese's flat-decay update is at `gpu_brain.py:376` exactly as described, but the documented
shape-mismatch bug is filed against its *caller* (`unified_brain.py::_update_hebbian_weights`, per
`gcheese-faiss/CRITICAL_ISSUES.md:104-110`), not `hebbian_update` itself; `apply_outcome_update()` is ~38 lines
duplicated across two files (not ~250 total); and the nuclear-test demo uses fixed ±0.8 outcomes — only the extended
demo randomizes. Zero grep hits for EWC/elastic/orthogonal/replay/consolidat/fisher across GrillCheese's `learning/`
package, and `DECISION.md:241` lists EWC as future work, both confirmed verbatim. The negative result stands at full
strength.

**H-A7 — The learning gate: weight updates from live data enter the trunk only through a nightly, pre-registered
promotion rule composed from harnesses that already exist — and that rule can tell a forgetting update from a clean one.
Added 2026-08-28.** Source: the oracle-competition scoring (`docs/research/2026-08-28-oracle-competition-scored.md`),
whose blind round produced the industry-default learning loop and a prior-art list none of which was in this repo.
Context: `docs/ARCHITECTURE_VISION.md` lists the learning-gate as `[far]` ("vetted sources, anomaly detection,
rollback") with no hypothesis entry, no kill criterion and no experiment — the one piece of the continual-learning story
with no measurement path. H0 says *how* weights may depend on context; this entry says *when* a candidate update is
allowed in at all. Prior art to build from (all ≤ 2025, IDs verified by the competition's own search passes, not
re-verified here): **KTO** (Ethayarajh et al. 2024, arXiv:2402.01306 — learns from unpaired binary 👍/👎, the feedback a
product actually produces); **O-LoRA** (Wang et al. 2023, 2310.14152) and **Online-LoRA** (Wei et al. 2024, 2411.05663)
— orthogonal-subspace / task-free forgetting control for adapter-shaped deltas; **experience replay** (Rolnick et al.
2019, 1811.11682); the **ripple-effect test** for knowledge edits (Cohen et al. 2023, 2307.12976) as the regression
suite an edit must pass; **TracIn** (Pruthi et al. 2020, 2002.08484) as per-update influence tracing — `git blame` for a
promoted delta; **conformal LM** (Quach et al. 2023, 2306.10193) for the abstention contract; **sleep-time compute**
(Lin et al. 2025, 2504.13171) for the always-on consolidation slot. Claim, concrete — promote candidate a_{t+1} iff
every term holds: Δ general-capability regression ≥ 0 (H-G4's `socratic_test_text.jsonl` harness); Δ dated-key
forgetting ≥ 0 against the NLMS curve (H-G2's NYT harness); in-window needle recall ≥ 95% (H-D4's
`exp_needle_recall.py`); binding-channel retrieval ≥ 70% (H-B6's offline probe); claimed-answer precision ≥ 0.99 (the
CoT pipeline eval); routing precision ≥ 0.90 (the margin gate). Every term is a measured artifact today; every candidate
run writes a manifest (data hashes, seeds, base-checkpoint hash, delta hash) and is rollbackable by construction because
deltas are never merged destructively. Two measured reasons the gate is not paranoia: H-B6 (a self-grading objective
Goodharts while its loss looks like progress) and H0 part 2 (online-trained routers collapse) — so nothing the model
scores itself ever enters, only execution-verified or human-thumbed signal. **Validate** (cheap, 151M checkpoint,
CPU-tolerable): build the gate as one script over the existing harnesses and feed it three known deltas on
`hd5_mem21.pt` — (a) a deliberately-forgetting update (plain SFT on one narrow slice, no replay: the H-A1 failure
shape), (b) a matched update with replay + hardening, (c) the all-zero delta as the null; record which gate fires, at
what margin, and the wall-clock. **Kill:** if the composed gate cannot separate (a) from (b) at any threshold the
individual harnesses support — passes the forgetting arm or fails the clean one — then the harnesses measure the wrong
things for update admission and the learning gate needs its own benchmark before any nightly path is built; second kill:
if a full gate pass exceeds a nightly budget (~1 GPU-hour at 151M) it cannot be nightly and must be tiered (cheap gates
every night, expensive weekly). Status: **RAN 2026-08-30 (Colab A100-40G) — the gate SEPARATES: at the nightly LR where
a 3.3M-token replay update is CE-neutral on every source (3e-5, and 1e-5), the code-only update is still rejected with
the mechanism named. Kill did not fire. Nightly LR = 3e-5. Full ladder below.** `validation/exp_a7_learning_gate.py` (+
`test_a7_gate.py`, 9 pins on the pure decision logic) implements the experiment with two pre-registered corrections
found while building it: (1) **three of the six proposed terms cannot see a trunk delta** — claimed-answer precision and
routing precision are model-free surfaces (word table + VM), and the H-G2 NYT-vs-NLMS curve benchmarks memory *methods*;
the trunk's forgetting signal is per-source held-out CE instead; (2) every delta is scored against the base on
*identical* held-out windows, so thresholds are max(min ε, 3 × **paired** SE) per metric — the unpaired seed-to-seed
noise (0.14 nats on the mix at 16 batches) is the wrong noise for a paired comparison and would hide real forgetting;
retrieval, which has no pairing, uses the seed-to-seed ε with a 0.50 collapse floor. CPU shape: ~9 s/step, ~50 min end
to end — a session teardown killed it twice at step ~100/120 (partial base numbers, for orientation only: ce_mix 3.21 /
3.34 on the two streams, GSM8K text 3.71, retrieval 0.70 / 0.74, code-only fine-tune loss ~1.9 flat). The recorded run
is the Colab one: **`notebooks/a7_learning_gate.ipynb`** (A100, B32 × 200 steps = 3.3M tokens per delta, minutes).
**Result** (`validation/logs/exp_a7_learning_gate.{log,json}`, copied from Drive; `notebooks/a7_learning_gate.ipynb`;
base `hd5_mem21.pt` step 92k, deltas 200 steps × B32 × S512 = 3.28M tokens at LR 1e-4, fresh Adam, 10-step warmup; eval
32 mix / 8 per source / 8 GSM8K batches = 1,024 / 256 / 256 paired windows; base ce_mix 3.2146 (seed 779: 3.1998), GSM8K
text 3.733, retrieval 0.738 / 0.744): **C (null) PROMOTE** — every paired difference exactly 0. **A (code-only, no
replay) REJECT, and the rejection names the mechanism:** mix +0.201 (thr 0.024), GSM8K +0.087 (thr 0.020), **every one
of the 12 unseen sources +0.116 to +0.185 nats** (thr 0.030) and `pretrain_ext` **+1.09** (thr 0.091) — the smallest
source (0.9M tokens, base CE 0.25 = memorized) is the most fragile, exactly the forgetting-theory prediction — while the
trained source itself sat at **−0.003** and its train loss was flat (1.87 → 1.86; code was already learned). Retrieval
*rose* to 0.777. **B (full mixture = replay) REJECT** — but not for forgetting: mix +0.046 (thr 0.010) and **all 13
sources +0.035 to +0.066, *including* the sources it was trained on**, GSM8K −0.005 (passes), retrieval 0.801. A uniform
rise across every source, code included, is something forgetting cannot produce; the train loss confirms it — **B's
training loss went *up*, 3.05 → 3.27, over its 200 steps**: a fresh Adam at 1e-4 on a converged checkpoint (which ended
its cosine near 3e-5) perturbs the whole model. The gate rejected a genuinely bad update. **Reading:** by the letter the
first kill fires (B rejected); by the evidence the *clean arm was not clean* — the failure is in the arm's construction
(optimizer state + LR), not in the gate's ability to discriminate, and the gate lacks only a *naming* term: it labels a
uniform rise "forgetting_source:*" on 13 sources when the honest label is "uniform regression (LR/optimizer)". Two
consequences, both pre-registered here: (1) **decider — the LR ladder**: rerun B alone at LR 3e-5 and 1e-5, and once
with the checkpoint's Adam state restored (`CB_A7_ARMS=B`, `CB_A7_RESUME_OPT=1` in the script; ladder cell in the
notebook); the gate separates iff some nightly LR exists at which a replay update is CE-neutral on every source while
the code-only update at the *same* LR still fires — if no such LR exists at 3.3M tokens, the kill stands and the nightly
path needs a different update rule (KTO on thumbs, not SFT); (2) **add the pattern term**: a uniform rise (all sources
fire with |spread| ≪ mean) is reported as `uniform_regression`, a selective rise (unseen sources fire, trained source
improves) as `forgetting` — same decision, correct name. Budget: one full gate evaluation = **48 s** on an A100 (nightly
cost of *checking*); each 3.3M-token proposal = 72–76 s. **Run 2 (same day, user-run;
`validation/logs/exp_a7_learning_gate_run2_src0.1.{log,json}` — the Drive log is an append log holding the first run,
two aborted starts at 3e-4 and 8e-5, and this one): LR 8e-5, ε_src raised to 0.10.** Same shape: A rejected and named
(mix +0.162, GSM8K +0.065, 10/12 unseen sources above even the 0.10 ε, `pretrain_ext` +0.83, trained source −0.009); C
passes; **B rejected on mix alone** (+0.033 vs 0.010) with the per-source column silenced by construction (+0.026…+0.040
on all 13) — and **B's train loss still rises, 3.051 → 3.255**. Two things this pins: the uniform drift scales with LR
(mix +0.046 → +0.033 for 1e-4 → 8e-5) and so does A's signal (+0.201 → +0.162), so the ladder's real question is whether
an LR exists at which B is CE-neutral *and* A still fires — or whether restoring the optimizer state
(`CB_A7_RESUME_OPT=1`) removes the drift at the same LR, which would leave A's signal intact. Raising ε_src is not a fix
(it is fitting the threshold to the arm); the mix term, which cannot be loosened without changing the claim, still
rejects. **The LR ladder (same day, user-run, B alone, fresh Adam unless noted;
`validation/logs/exp_a7_learning_gate_{lr3e-5,lr1e-5,lr1e-4_resume}.*`): B at 3e-5 → PROMOTE** (mix +0.005 vs thr 0.010;
worst source `kbtxt` +0.009 vs 0.030; GSM8K −0.014; retrieval 0.759); **B at 1e-5 → PROMOTE** (mix −0.002, every source
−0.001…−0.012, GSM8K −0.011); **B at 1e-4 with the checkpoint's Adam state restored → REJECT, *worse* than the fresh
run** (mix +0.055, all sources +0.038…+0.052, `pretrain_ext` +0.19) — the drift is the learning rate, not the optimizer
state. **Correction to the reading above:** the "train loss rose 3.05 → 3.23" was not evidence of perturbation — the
curve has the identical shape at 1e-5, 3e-5, 8e-5 and 1e-4 (same data seed 11 ⇒ same batch order; the first ten batches
are simply easier); the held-out CE is the only evidence and it was sufficient. **State of the kill:** a nightly LR at
which a 3.3M-token replay update is CE-neutral on every source exists (≤ 3e-5); the separation verdict now waits on **A
at 3e-5 and 1e-5** (the notebook's ladder cell runs both): the gate separates iff the code-only update still fires
there. Prediction on record, from the 1e-4/8e-5 scaling (mix +0.20 → +0.16, sources +0.12–0.19 → +0.09–0.16): at 3e-5 A
fires on mix and most sources; at 1e-5 A fires on mix and `pretrain_ext` but most per-source rises fall under 0.03 — if
that is what happens, the per-source term needs its ε derived from the paired SE at the nightly LR rather than the 0.03
floor, and the mix + memorized-source terms carry the decision. **A ran (same day;
`exp_a7_learning_gate_A_{lr3e-5,lr1e-5}.*`) — VERDICT: the gate separates.** At **3e-5**: B PROMOTE (above) vs **A
REJECT** — mix **+0.059** (paired SE 0.0018), **12/12 unseen sources +0.032…+0.074** (all above the 0.03 floor),
`pretrain_ext` **+0.208**, GSM8K +0.012, trained source **−0.019**. At **1e-5**: B PROMOTE vs **A REJECT** — mix
**+0.017** (SE 0.0006), `pretrain_ext` **+0.053**, the other sources +0.010…+0.024 under the floor, trained source
−0.017 — exactly the prediction on record. Null promotes in every run. **Decisions:** (1) the composed gate — paired mix
CE, per-source CE, GSM8K-text CE, retrieval health — *is* a gate for trunk updates at this scale; H-A7's first kill does
not fire, and neither does the second (a gate evaluation is 37–50 s on an A100, a 3.3M-token proposal 65–77 s). (2)
**Nightly LR for the 151M trunk = 3e-5** — the largest A/B margin with B neutral (A's mix margin 0.059 vs B's 0.005).
(3) Two refinements, cheap and queued: the per-source ε should be max(floor, 3 × paired SE) with the floor lowered
toward the SE scale (at 1e-5 the 0.03 floor hid 11/12 sources whose rises were 10–40× their SE); and the naming term
(uniform rise on every source incl. the trained one ⇒ `uniform_regression`; selective rise with the trained source
improving ⇒ `forgetting`) so the reason on a rejection is the true one. (4) Optimizer-state restore is not a lever
(worse at the same LR); LR is. Not part of the pretraining gate.

**H-A8 — Capping the episodic store at 16384 beyond-window tokens costs nothing measurable, and buys a decode step that
can be recorded once and replayed. Added 2026-09-23.** Source: the capture work on `HybridBackbone.step`. The store was
three tensors grown by `torch.cat` and read through a host `len()`. Both are fatal to a recorded step — a growing tensor
is a new graph signature every token, so a recording is taken and thrown away every token and nothing ever replays; a
host `len()` is a value the replay cannot reproduce, because a replay re-issues commands and never runs the Python that
computed it. Fixing both means a fixed-capacity store written in place at a slot chosen on the device, and a fixed
capacity means that past `capacity` aged-out tokens the oldest entry is overwritten. **Claim:** at `capacity = 16384`
(16x the training context, and 32x the 512-token window that gates what reaches the store at all) that eviction is
unreachable in any context this model is evaluated at, so the behaviour change is nominal; and the recording it enables
is worth more than any recall it could cost. **Why we believe it.** Measured on an RX 6750 XT with `hd5_mem21` (d512/L8,
V=127996, `mem_every=2`, window 512), 700 generated tokens: the capture-safe step produces **token-for-token identical**
output to the old growing store on real torch and on grilly2, and graph replay takes the token from 61.6 to 102.7 tok/s
(1.67x); with the head quantized to int8, 118.4. Eviction never fires in that run — the store holds 193 entries of 16384
— which is the point: at any context under 16896 tokens the fixed store and the growing one hold exactly the same set.
**The cost, which is real and is not recall.** The read is sized by capacity rather than by fill, so it scans 16384 rows
to find 8 among 193. Measured per episodic read, 4 memory layers: **3.32 ms/token at 16384 against 1.75 ms at 512** —
and only 2x for a 64x longer scan, because the cost is dispatch count, not bytes. That is why graph replay at capacity
16384 (102.7 tok/s) is still below the old *eager* step (118.8), while at capacity 1024 it is 125.8. **How to
validate.** Run `validation/exp_needle_recall.py` at needle distances past 16384 with `mem_capacity` at 16384 and at a
capacity larger than the context, and compare recall; separately, sweep `mem_capacity` over 512...16384 and record
tok/s, which is the curve that says what the default should be. **Kill criterion.** Either half kills the default: (a)
recall at a distance inside 16384 differs at all between the capped and uncapped stores — that would mean the causal
accounting is wrong, not that the capacity is too small, and the bug is in `_memory_step`, not the number; or (b) recall
past 16384 matters for a real evaluation, in which case the capacity is a config knob (`HybridBackbone(mem_capacity=)`)
and the default moves. A third, softer kill on the *default value* rather than the mechanism: if the scan cost above is
not removed by a real top-k selection (grilly2's `topk` is currently sort-then-take), the honest default is the smallest
capacity that covers the evaluated context — on the measurements above that is 1024, not 16384.

**H-A9 — Evolution strategies on the program adapter's LoRA factors, scored by the cubelang VM, buys held-out
verify-to-gold that supervised fine-tuning has stopped buying — and the gain survives the H-A7 gate. Added 2026-09-23.**
Source: the v9 job, and TODO.md's RLVR rung, which already names ES as the gradient-free alternative to GRPO. Two facts
set it up. First, **SFT has run out of signal on at least one family**: v14e_nochain is 600/600 on the gen-3 held split
(exp_r17, 2026-09-15), so there is nothing there for any optimizer to win. Second, **something did regress and may not
have come back**: arithmetic fell 0.825 -> 0.750 -> 0.675 across the v5-v7 rounds as chat volume grew (the two-adapter
interference recorded under H0), and each repair was a replay, not a fix. ES is the shape that fits what is left: the
reward is the VM's verdict, which is not differentiable, the base is frozen, and only the adapter's low-rank factors
move, so a forward-only optimizer on a 12 GB card is enough. **Claim, in two parts.** (a) *Headroom exists*: at least
one program family scores below 0.95 verify-to-gold on the held-out split under v14e_nochain Q4. (b) *ES closes some of
it*: perturbing only the LoRA factors with rank-1 antithetic noise and scoring each member by the shaped ladder of
cubby-lm's TRUNK_VM_EMISSION_CONTRACT §3a (0.1*parses + 0.2*compiles + 0.3*executes + 1.0*(result == gold)) raises that
family's **held-out** verify-to-gold over v14e by at least two standard errors, within a fixed budget, while holding
every other family at v14e (40/task) and holding the output format. **Why we believe it.** ES needs no gradient through
the VM, which is the whole reason the RLVR rung was blocked; EGGROLL-style low-rank perturbation makes a population of
32 a batched forward rather than 32 weight copies, which is what makes it fit beside a 2.6B int8 base; and the thing
being tuned is 64-rank factors, not 2.6B parameters, so the search space is the size ES is good at. **What would make
the result a lie, and the guards against each.** A program that hardcodes the gold answer scores 1.0 on the ladder and
is worthless, so the correctness term counts **only if the same program also passes the counterfactual planter's
perturbed numbers and entities** — a hardcoder caps at 0.6. Fitness noise can look like learning, so a **null control**
runs the same budget with fitness shuffled across members and must not improve held-out. Train-split leakage can look
like learning, so held-out is read every K iterations and never used to select. **How to validate.** Step 0, before
anything is built: `standin/eval_emitter_vm.py --gguf <v14e_nochain Q4> --per-task 40` per family on the held-out split,
covering arithmetic, kernel, chain, role binding and forge decision/compare. Step 1: ES on the family with the most
headroom — population 32 as 16 antithetic pairs, greedy decoding, a fresh 16-32 prompt minibatch from that family's
TRAIN split each iteration, held-out every K. Step 2: the candidate goes to the H-A7 gate at its measured thresholds
and, if it passes, serves beside v14e for an A/B on the live self-test before anything changes. **Kill criterion, three
ways.** (i) *Step 0 kills (a)*: if every family is already >= 0.95, there is nothing to tune and ES would need a new
family — a different decision, and the job stops there. (ii) *The budget kills (b)*: if held-out verify-to-gold has not
beaten v14e by >= 2 SE within the budget (proposed: 200 iterations), stop and record it; ES at this population and rank
is not the tool. (iii) *A regression kills it outright*: a gain on the target family bought with any loss on another
family or on output format is a fail, not a trade, because the serving stack runs all of them. **Step 0 result
(2026-09-23, RX 6750 XT, v14e_nochain Q4_K_M through
llama.cpp Vulkan, 40 val records per family, 160 total, 3m26s;
`standin/data/out/eval_emitter_vm_v9_step0.json`): claim (a) HOLDS — the
first kill does not fire.** Scored against the val split of
`emitter_sft_v14e_nochain.jsonl`, the file v14e was actually trained on:

| family | n | executes | verify-to-gold | SE | +2 SE |
|---|---:|---:|---:|---:|---:|
| arithmetic | 40 | 1.000 | **0.775** | 0.066 | 0.907 |
| kernel | 40 | 0.950 | **0.850** | 0.056 | 0.963 |
| plan | 40 | 1.000 | no gold | — | — |
| role_binding | 40 | 1.000 | no gold | — | — |

Within kernel the game sub-families are already solved — `game:compare`
1.000 (n=6), `game:count` 1.000 (n=10), `game:decision` 1.000 (n=9) — and
the residual sits in bare `kernel`, 0.583 at n=12, too few to target.
`plan` and `role_binding` carry no gold by construction (role binding
returns a bound symbol; there is nothing to match), so verify-to-gold is
not defined for them and they are held at `executes`, both 1.000.

**Two corrections to this entry's own framing, from running it.** First,
the families are not the six listed above: **`chain` does not exist in this
dataset** — v14e's work order WO-1.3 dropped it — which is also why
exp_r17 reports 600/600 on the gen-3 held split, and `forge
decision/compare` are not top-level families but the `kernel/game:*`
sub-groups, both already at 1.000. Second, **`kernel`'s headline 0.850 is
higher than the 0.583 the eval prints for its bare sub-row**, because that
row excludes the game sub-families; by family, arithmetic is the weakest.

**Target: arithmetic.** It is the weakest family, it has by far the largest
held-out pool (329 val records against kernel's 95), and it is the one with
the documented regression history (0.825 -> 0.750 -> 0.675 over v5-v7).
Its 9 misses in 40 are **wrong answers, not malformed programs**: every one
executed, and the VM returned a value that was not the gold (28 vs 40.5,
100 vs 200, 72 vs 1152, 24 vs 6 — ratios and a non-integer, so wrong
operator or wrong operand order rather than a format failure). That is the
shape ES can move; a parse or compile failure would not have been.

**The kill criterion's bar, fixed here rather than left to step 1.** "Beats
v14e by >= 2 SE" is only meaningful once the SE is pinned to a stated
sample. At n=40 the bar is **0.907** — a 0.13 jump, almost certainly out of
reach, which would make the criterion fail by construction rather than on
the merits. So **the held-out read is the whole arithmetic val split, 329
records**, giving SE 0.023 and a bar of **0.821** — a 0.046 jump over
v14e's 0.775. That is the number ES has to clear; 40-record reads are for
the every-K-iterations progress check only and never for the decision.

**Settings, corrected against the EGGROLL paper (2026-09-23).** The ES
implementation was built from a shader audit plus standard ES, not from
the paper (Sarkar et al., *Evolution Strategies at the Hyperscale*,
arXiv:2511.16652). Reading it changed three things before any iteration
ran:

- **The step size was wrong by ~2000x.** The paper absorbs `1/sigma` into
  its learning rate (Eq. 6), so `alpha = lr/sigma` is what sets the step,
  and it uses `alpha = 0.001` across populations from 256 to 65,536
  (Tables 3, 4). This job carried `sigma=0.01, lr=0.02`, i.e.
  `alpha = 2.0` — a **measured per-element step of 7.4 sigma**, seven
  times the radius at which the fitness was measured. It would have
  diverged on iteration 1 and read as "ES does not help here". Now
  `lr=1e-5`.
- **Fitness shaping is now EGGROLL §6.3**, which is what the paper uses
  for its reasoning fine-tunes: linear in the score rather than in the
  rank, so a member that solved the batch outright counts for more than
  one that edged out a tie. `fitness()` therefore returns one ladder
  score **per prompt** and the optimizer takes the (member, prompt)
  matrix. `shaped_sd` is logged per iteration, because unlike centred
  ranks its spread is not fixed.
- **What ES moves stays the LoRA factors**, which is the paper's Table 4
  / §M setup (persistent rank-4 LoRA on all projections, base frozen,
  sigma 0.01) rather than its headline formulation, where the mean
  parameter is the weight matrix and only the *perturbation* is low-rank.
  The difference does not bind here: at population 32 the per-step update
  is rank <= 32 either way, far under the rank-64 adapter's ceiling.

**sigma, measured rather than copied (2026-09-24).** The paper's 0.01 is
meaningless without the scale of what it perturbs, and this adapter's
factors are not the paper's: `std(lora_A) = 1.33e-2` but
`std(lora_B) = 1.72e-3` across all 122 of each, so 0.01 perturbs every
`lora_B` by **5.8x its own standard deviation**. A member that far out is
not a neighbour of v14e, it is a different model — and a different model
does not emit `<|im_end|>`. Since grilly's decode compaction can only drop
rows that *finish*, one non-terminating member holds the whole batch open
to `--max-new-tokens`. Measured, 32 members on one prompt, RX 6750 XT:

| sigma | sigma/std(B) | terminated | decode steps | seconds | distinct outputs |
|---|---|---|---|---|---|
| 0.010 | 5.83 | **31/32** | **768** | **150.7** | 32 |
| 0.006 | 3.50 | 32/32 | 177 | 34.4 | 28 |
| 0.004 | 2.33 | 32/32 | 160 | 31.4 | 26 |
| **0.003** | 1.75 | 32/32 | 99 | **20.0** | 22 |
| 0.002 | 1.17 | 32/32 | 95 | 19.2 | 16 |
| 0.001 | 0.58 | 32/32 | 93 | 18.8 | 5 |
| 0.0005 | 0.29 | 32/32 | 88 | 17.9 | **1** |

Both ends fail and neither raises. Above the knee one hung member costs
the batch **7.5x**; below it every member decodes identical tokens, the
fitness is the same for all 32, and ES takes no step — which is why the
shaped spread is now logged per iteration. **sigma is 0.003** (lr 3e-6, so
alpha stays 0.001). It is the knee on both axes and also maximises
distinct outputs per second.

Two things this surfaced. First, `_centred_ranks` gave a *tied* population
a full spread — a flat objective produced a full-size step in whatever
direction the noise generator happened to order the members. Ties now
share their average rank, so a flat population takes no step (fixed in
grilly, with tests). Second, grilly inherits `repetition_penalty = 1.1`
from LFM2's `generation_config.json` while llama.cpp's default is 1.0, so
it is **another unaccounted difference in the open 0.625-vs-0.800 baseline
comparison**, alongside the BOS token. It costs no measurable time (46.49
vs 46.53 s) — this is a correctness note, not a speed one.

**The run was executed, and claim (b) is in trouble before the budget is
(2026-09-24).** Two iterations at the real shape — population 32,
minibatch 16, sigma 0.003, prompts-per-batch 1, RX 6750 XT:

| | iter 1 | iter 2 |
|---|---|---|
| train fitness mean | 1.1234 | 1.2426 |
| shaped_sd | 0.216 | 0.211 |
| **member effect** | **0.0000** | **0.0000** |
| split-half correlation of member means | **-0.377** | -0.086 |
| seconds | 1057 | 995 |

The member effect is the fraction of score variance that belongs to the
*member* rather than the prompt, calibrated against simulation: pure
scatter reads 0.01-0.02, a real effect reads 0.3-0.9. It reads **zero**,
and `shaped_sd` sits *below* the 0.246 that m=16 noise alone produces.
A member's score on the first eight prompts does not predict its score on
the other eight. Members do differ — they diverge on 12 of 16 prompts —
but they differ in *which* prompts they happen to get right, not in how
good they are. **ES is ranking scatter.**

The arithmetic says why, and it is not fixable by budget:

- **The ladder is binary in practice.** Only two rungs ever occur, 0.6 and
  1.6. v14e always emits a program that parses, compiles and executes, so
  `0.1*parses + 0.2*compiles + 0.3*executes` is a constant and the whole
  reward is the correctness bit. The ladder was put there precisely
  because "a binary right/wrong reward is flat almost everywhere"; on this
  model it *is* binary.
- **One bit times sixteen prompts cannot resolve the effect.** At p=0.523
  and m=16, sampling noise on a member's mean is sd 0.125, while the
  measured spread of member means is 0.054 — the observed spread is *less*
  than chance. Resolving a true 2% quality difference needs m > 624
  prompts per member, i.e. ~20,000 generations an iteration against the
  current 512.

Caveat on power: with 32 members the statistic cannot see an effect below
roughly 5% of total variance, so this shows the effect is small, not that
it is exactly zero. It is small enough that 200 iterations of this design
would rank noise.

**And the starting point is wrong anyway.** Held-out read 0.250 on 8
prompts (SE 0.153) where v14e measures 0.775. That is the same unresolved
grilly2-vs-llama.cpp gap as the 0.625-vs-0.800 arithmetic read — the BOS
fix is applied but unmeasured, and `repetition_penalty = 1.1` was found
on 2026-09-24. **The bar of 0.821 was set from llama.cpp's baseline; it
cannot be chased from a backend scoring 0.15-0.5 below where that baseline
was measured.** Close the baseline gap before spending any ES budget.

Cost, for whenever it is spent: **~1,000 s an iteration** at minibatch 16
(prompts-per-batch 1, which measured 1.24-2.0x faster and far steadier
than 4: 289/284 s vs 360/572 s at minibatch 4, because a batch costs its
longest row and max-of-32 is shallower in the length tail than
max-of-128). 200 iterations is **~56 hours**.

The one gap left is **population size**. Every LLM run in the paper uses
256 to 8192, and its central pretraining finding is that large populations
are what make ES work; 32 is what fits this card serially. If (b) fails on
the budget, read it as a statement about population 32, not about ES.

Status: **step 0 DONE, claim (a) supported; claim (b) PARKED 2026-09-24**
(Nick) before any real iteration. What stands:

- **The engine is built.** grilly2 has `grilly.learning.es`, LFM2, an
  unmerged LoRA with per-row population noise, and (24 Sep) left-padded
  `generate` that drops rows as they finish. `exp_v9_es.py` batches
  members x prompts (`--prompts-per-batch`) and the held-out read
  (`--held-out-batch`, 16 prompts at once: 40/40 texts identical to the
  serial read, 120 s against 357 s).
- **Why parked: the tuned model is not the served one.** On the same 40
  held-out prompts, the emitter through grilly2 (int8, 768 tokens) reads
  26/40 where llama.cpp step 0 read 31/40. ES would climb the grilly
  version; whether a gain carries to what Cubby serves is unknown, so the
  gap comes first if this is picked up again.
- **Throughput ceiling on 12 GB.** 32 members x 1 prompt: 63 s a prompt
  at 768 tokens (serial 122 s). 2 prompts per batch spilled past the card
  because the padded KV cache sits in grilly's pooled (power-of-two,
  retained) memory; at 128 tokens, where it fit, it was 1.5x faster a
  prompt. Fix before reuse: exact-size persistent buffers for
  `PaddedKVCache`. Realistic gain 2-3x, not 10x.

Logs at `validation/logs/exp_v9_es_*.{log,json}`.

**H-A10 — (added 2026-09-24) A nightly sleep cycle turns the day's refusals into the night's curriculum: over successive
nights on a fixed battery the refusal share falls while spoken defects stay at zero.** Source: the self-evolution
brainstorm (Nick, 24 Sep) under invariants 1 and 6 -- the model proposes, the host disposes, only a gate promotes, every
change on a ledger. **Built (v1):** `cubbyllm/reasoning/sleep.py` (replay certified chains into hippocampal episodes;
make gate-accepted facts from trusted, non-latent sources durable, loaded at boot by `AskLoop.load_learned`; route every
refusal by reason to the queue for the level that must change; audit spoken answers against gold or the asker and
deliver every outcome to the striatum; a ledger with the git revision and the input hash; idempotent per night),
`AskLoop(history_path=)` and `serve_api --history/--learned`. Pinned by `tests/reasoning/test_sleep.py` and
`standin/tests/test_sleep_loop.py` (day 1 fetches, the night consolidates, day 2 answers with the source emptied).
**First run** on the SimpleQA search-and-learn bench replayed as three nights
(`docs/research/2026-09-24-sleep-cycle.md`): the report shows refusals MOVING (lev9: not-covered 464 -> 338,
unknown-relation 56 -> 130, so the next lever is the lexicon), surfaced the one recorded defect (Kafr al-Awamid, a
subdistrict spoken for a district) and made 134 accepted facts durable; the ledger is unchanged by re-running all three
nights. **Kill:** after five nights on a fixed battery the refusal share has not fallen, or any defect is spoken, or the
queues grow while the battery stands still (the loop collecting, not learning). Status: **OPEN -- v1 built and run on
recorded days; the queues are not yet acted on (v2) and the ask loop does not yet read the episodes.**

**H-A11 — (added 2026-09-24) Graph-of-thought at an ambiguous hop: following every candidate to the end of the chain and
certifying each branch in the VM speaks where the branches agree and names the split where they do not, without a single
wrong answer; and in an open world, one pick by the asker recovers what the refusal withheld.** Source: Nick, 24 Sep
("graph of thought with branching also needs some love"), on exp_r31, which branched OUTSIDE the walk. **Built:**
`pipeline._explore` / `_converged` (every candidate object a branch, nested splits branch again, bound `branch=12`, each
branch through the same `_check_chain` as the greedy walk; speak only if every branch finished, verified and names ONE
answer; `closed_world=True` lets a branch the store cannot finish count as refuted, never the default);
`answer(choose=)` narrows a hop to a candidate the asker picked and can never add one; `learn` fetches the entity a
branch stalled on; `standin/ask.branch_clarify` offers a MID-chain split as choices and never a last-hop one (a pick
there would be the answer coming back 'verified' -- the poisoned-capital case); `graph.py` keeps every ask's run as
typed nodes and edges (supports, verifies, alternative_of, agrees_with, contradicts, learned_into, retries) in the ask
history; the panel draws branches, green where they agree and red where they contradict. **Measured**
(`validation/exp_r32_branching_inpipe.py`, the 65 CLUTRR fork records r31 used, real cubelang VM): refuse-on-sight 12/65
spoken; closed world 44/65 (r31 exactly: 12 + 32 recovered), 0 wrong; open world 12/65 (32 held by a branch that could
not finish, 18 divergent last-hop sets, 3 VM rejections), 0 wrong; open + one asker pick 44/65, 0 wrong (34 splits
offered, all resolved in one round, 2 then VM-rejected at 5-6 hops). Cost: 9x the VM calls on an ambiguous question (439
vs 48 over the 65), none on the rest. **Kill:** any wrong answer spoken through a converged branch set or a choice; or
live ambiguous_hop refusals whose splits are mid-chain are rarely resolved by the offered pick (the asker's choice list
is the wrong question). Status: **OPEN -- supported on CLUTRR (0 wrong in every arm); not yet measured on live askers.**

**H-A12 — (added 2026-09-24) The web as the source of last resort turns refusals the store cannot fill into verified
answers without a single wrong one, because a web triple is only a witness: it is held until two independent sites state
it, and the VM still certifies the chain.** Source: Nick, 24 Sep ("need to add websearch to it too to find answers");
the don't-know offer design of 2026-09-04. **Built:** `standin/web_source.py` (Brave / SearXNG; Wikipedia→Wikidata
handoff when one item carries the relation, schema.org JSON-LD minus page metadata, entity-opening sentence frames, the
local LFM as a grounded, kind-checked reader; sites = registrable domains with the Wikipedia family folded and archives
excluded; a persistent tally), `learn_and_answer(fallbacks=)` (the web only after the primary source has nothing more;
per-fact provenance and hold), `serve_api --web`, the sleep cycle keeping only two-site facts. Pinned by
`standin/tests/test_web_source.py` (16). **Live smoke on real pages** (no search key yet): the reader, unguarded, wrote
'112 languages' as a birthplace and '1815-1852' as a birth date; with prose-only passages, grounding and the kind check
every value kept was right, and every fact stayed held at one site each -- the rule working. **Kill:** any wrong answer
spoken on web facts; or, on a live battery of `retrieval_exhausted` questions with a search key, fewer than one in ten
reach two independent sites (the rule holds everything and the web buys nothing). **Measured live**
(`validation/exp_r33_web_heldout.py`, 60 withheld wiki-world facts, Brave, real VM, reader off): v1 spoke 12 with 1
WRONG (Agha Hashar Kashmiri's death date, four sites repeating an old Wikipedia lead); v3 -- spellings merged, finer
values vouching for coarser ones, one value per relation, copied text counted as one witness, the Wikipedia family
speaking through its live article -- spoke 15/60 (11 correct, 4 near, **0 wrong**), held 14 (11 right, 3 wrong held
back), split 1, found nothing for 30 (place of death 14/15). Scoring: a containing place answers a PLACE-of-birth
question (Nick). Status: **OPEN -- survives the kill clause on 60; the reader arm and a bigger battery are next.**

**H-A13 — (added 2026-09-24) A skill library of composition rules, learned overnight from episodes whose relation gold
or an asker states and adopted only with zero counterexamples across the whole record, answers "how is B related to A?"
at depths no episode reaches, without a single wrong answer.** Source: Nick, 24 Sep (self-evolution level 3; choices:
rules first then CubeLang helpers, zero-counterexample gate, every application through the VM). **Built:**
`cubbyllm/reasoning/skills.py` (`Library` replayed from an append-only ledger, retire never delete; `mine` with closure
-- a premise is read off every split of an episode the library already composes -- and a gate that re-derives EVERY past
episode before a rule gets in; `derive` over every bracketing, a split refused; `relate` over every path between the
two, speaking only when the composing paths agree, each certified in the VM with its facts through
`pipeline._check_chain`), `TripleIndex.paths`, the ask loop's `relation_ask` / `AskLoop.relation`, the sleep cycle's
`skills` phase and four new routes, `serve_api --skills`. Pinned by `tests/reasoning/test_skills.py` (17) and
`standin/tests/test_ask_relation.py` (5: day 1 refuses, the night learns, day 2 answers a four-hop question no episode
was about). **Measured on CLUTRR's own question** (`validation/exp_r34_clutrr_relations.py`, gen_train23_test2to10, real
cubelang VM; train as one day of refused asks with gold, one night, test as day 2): 109 rules adopted in 3 rounds;
**827/1,146 correct, 319 refused, 0 wrong; at 4-10 hops, depths no training record reaches, 746/1,003, 0 wrong**; the
null control (edges into B relabelled) 0/1,146 spoken; the shuffled library 29 wrong in 150 (the instrument sees
wrongs). The whole-record gate rejected 8 rules the 2-hop records alone support -- (son, grandfather) is father 156
times and father-in-law 16, a real ambiguity only 3-hop records show; the 2-hop-only arm adopts them and says nothing
wrong on this split by luck, not construction. 206 of the 319 refusals are true bindings recovered at 0.447-0.473
against the served two-binding threshold 0.4736 (exp_r11's lowest observed true binding; the control floor there is
~0.03, exp_r28): a diagnostic at 0.2 reads 1,033/1,146 and 950/1,003 at >=4, still 0 wrong -- recalibrating is Nick's
decision, not taken here. Write-up: `docs/research/2026-09-24-skill-library.md`. **Kill:** any wrong relation spoken; or
a rule adopted by the gate that a later clean episode contradicts more than once in five nights (the gate is admitting
non-functions); or on a live battery the relation asks are refused `no_rule` night after night while their episodes
accumulate (the library collecting, not learning). **Second split** (gen_train234_test2to10, train 2-4 hops;
`exp_r34_clutrr_relations_gen_train234_test2to10.log`): the gated arm 552/1,048 correct, **0 wrong** (469/903 at >=4);
the 2-hop-only ablation 727 correct and **3 WRONG** (*mother* for *mother-in-law*, through a (husband, brother) ->
brother rule the whole-record gate keeps out) -- the harness's verdict line counts both arms and reads KILLED; what it
killed is the ablation, and the criterion is left as written. The cost of zero counterexamples shows there: 16 premises
kept out, some real ambiguities, some apparent label noise ((husband, father): father-in-law 114, father 2), and 395 of
the 496 refusals are `no_rule`; the night now lists contested premises for the host (`skills_contested.jsonl`), most
one-sided first, and `python -m cubbyllm.reasoning.skills adopt` is the host's word on one (the minority discounted on
the ledger; refused if any other episode would derive wrongly). A diagnostic of that word
(`..._gen_train234_host_diag.log`): adopting the four premises >=95% one-sided ((husband|wife, father|mother) ->
father/mother-in-law) reads 560 correct and **10 wrong** against 552 and 0 -- every one CLUTRR's gold calling a wife's
father *father* in a story with a two-hop shortcut beside the proof chain. Scored against gold, the strict gate is what
kept the record clean; the host should not adopt these. The robustness splits cannot be measured from this pull (noise
edges carry no relation types). **Live teaching:** `POST /ask/feedback` -- an asker's stated relation makes a relation
ask an episode; support counts distinct askers, so one asker cannot put a rule in. **Real families
(`validation/exp_r35_wikidata_kinship.py`, Wikidata's own P1038+P1039 statements as gold, the family graph gendered with
Wikidata's declared inverses, the P279 hierarchy as term grain, 2,050 train / 486 test pairs by subject): KILLED.** 255
rules and 109 inverse rules adopted; the gated arm spoke 70 and got **8 wrong**, the reversed questions 34 with 6 wrong,
the inverse arm (store leads only B to A, `relate_back`) 22 with 4 wrong; null 0 breaches, shuffled 18 wrong. Most
wrongs are two TRUE relations against one gold (a son-in-law who is also an adopted son; a brother who is also a twin --
the hierarchy says what entails what, not what excludes what) or gold that contradicts the graph; one is a real
induction failure (the inverse of *niece* for a woman learned as *father's sister* from brothers' daughters only). Fixed
on the way: the null arm had fabricated facts from direct edges (10 false breaches), and `inverse_of` had borrowed
coarser terms' inverses through the hierarchy (*ancestor* for a grand-nephew). **Option for the contradiction model
(added 2026-11-06; Nick: "yes"):** an exact rule engine can state part of it. OWL 2 RL property chains (an uncle is a
parent's brother), asymmetry (if A is B's parent, B is not A's), irreflexivity and disjoint properties (no one is both
another's father and son) can be checked by `reasonable` (gtfierro; Rust, BSD-3, OWL 2 RL on DataFrog) or by DataFrog
alone. A rule the night would adopt is refused if it breaks a declared asymmetry or disjointness. **To report on the
real-families split:** how many of the 8 wrongs, and the learned *niece* inverse, such checks would have refused, and
how many right answers they would cost. Generation, line and sex per term remain the model's other half. Status:
**CLUTRR: survives (0 wrong, both splits). Real families: KILLED as built -- the next step is a kinship model of
CONTRADICTION (generation, line, sex per term), not more rules.**

**H-A14 — (added 2026-09-24) CubeLang helpers mined from the emitter's own verified programs -- two binary steps whose
first feeds only the second, as one VM-verified function -- halve the programs the emitter has to write without changing
a single result, and an emitter taught to call them gets arithmetic right at least as often as v14e.** Source: Nick, 24
Sep (the skill library's second kind, emitter sub-programs first). **Built:** `cubbyllm/reasoning/helpers.py` (parse the
step programs, mine the recurring two-op shapes, bake in a leaf that is mostly one number, rewrite, and gate: every
program on record that a helper would rewrite must run in the VM to exactly its own result, then all helpers jointly;
retire, never delete, on the skill ledger as `kind: helper`); `eval_emitter_vm.run_vm` supplies the definitions a
program calls (`with_helpers`). Pinned by `tests/reasoning/test_helpers.py` (6). **Measured**
(`validation/exp_r36_emitter_helpers.py`, v14e_nochain arithmetic, real VM): 32 helpers adopted on train; 5,551/5,819
train and 315/329 val programs rewritten, **0 results moved**, `solve` statements -54.6% / -55.5%; 60/60 val programs
written with the calls alone ran through `run_vm`; the 16 helpers with swapped sub/div operands all rejected by the same
gate. **Kill:** any rewritten program whose result moves; or, after an adapter round on the rewritten set
(`exp_r36/emitter_sft_arithmetic_helpers.jsonl`), held-out verify-to-gold below v14e's 0.775 on the same split (the
shorter program bought with wrong answers is a fail). Status: **OPEN -- the rewrite half supported (0 moved); the
training half not run.**

**H-A15 - (added 2026-11-04) Learning the way a person learns, as Cubby's own training loop: a failure that keeps coming
back is noticed as a gap; it is practised until the VM says right, its mistakes kept as "never this"; and an adapter is
made for it on demand. An adapter in use is kept and later consolidated; an unused one goes latent, then to a shadow
that relearns faster than starting from nothing. The loop lifts each gap without lowering anything else and without a
spoken wrong answer.** Source: Nick, 2026-11-04, on how he learns: "when I see something new, I first think about what
the hell is that? If I don't know and it's interesting to me (or new or surprise), I will search about it, make sense of
it (for example a new word, I put it in a sentence) then add it to my vocab"; unused, it waits "as latent skill waiting
for a need", and if the need never comes it is forgotten consciously "but it will still be there in a shadow memory";
something learned wrong goes to "a memory in my brain where I remember what I did wrong. Until I reach the right result
it will continue adding to the mistake memory until I'm right, then it switches to the 'I'm right and it's useful'
memory"; "mistakes can be reused in other concepts" (2 + 2 = 10 sits in the error memory, so 2 + 2 is never 10, even
before it is solved). Then: "I don't mean use it in cubby world but as training signal and on demand adapters auto
creation / self-improvement loop". Builds on H-A7 (the learning gate), H-A10 (refusals as the night's curriculum), H-K3
(a frozen trunk, adapters per context), H-E19 (try until the VM says right), H-S5 (learning progress as the interest
signal) and H-S10 (the postmortem). Order 0048 is the loop run once by hand. Related, to cite once checked: STaR (keep
only the reasoning that reaches a right answer), preference learning from (wrong, right) pairs, unlikelihood training,
and the savings effect in relearning. **The loop:** (1) notice: every request's VM verdict is logged, and failures are
grouped by a skill signature (relation, language, program shape, operation); (2) interest: a group that recurs and is
learnable on a short probe becomes a gap, while noise does not; (3) practice: the trunk attempts the gap's problems and
the VM marks each attempt; (4) mistakes: wrong attempts are kept per problem as negatives until a right one is found,
then become (wrong, right) pairs; (5) an adapter for the gap, trained in minutes; (6) the learning gate; (7) a router by
skill signature; (8) the lifecycle: loaded while used, latent (on disk) when unused, a shadow (compressed, archived)
when long unused, from which a returning gap starts; merged into the main adapter when used often. **In plain words:**
Cubby notices what it keeps getting wrong and trains itself a small add-on for it overnight; it keeps what helps,
shelves what it does not use, and remembers its mistakes as "never this". **Over a conventional model:** the model waits
for its maker to retrain it, at a cost that grows with its size, and the retraining can erase what it knew. Here the
trunk never changes, each skill is a small adapter that passes a gate before use, and the VM still checks every answer.
**Validate** (the 450M with adapters, the trunk frozen; the lines are in the orders): T1, the loop on a stream with
planted gaps (Wikidata's relation words in EN and FR, the next arithmetic level) and a shuffled-gold family that must
never be flagged; T2, self-practice against gathered examples at equal tokens; T3, mistakes as training signal (right
answers only, plus its own negatives, plus negatives shared across skills; transfer to a sibling gap; right answers lost
at most 0.5% of rows); T4, a shadow against starting from scratch (steps to relearn); T5, routing (at most 2% of
requests outside a gap sent to its adapter). Order 0049 runs T1 and T3; T2 and T4 follow. **Kill:** a gated adapter
lowers the base battery; the loop flags noise as gaps more often than real ones; or negatives make Cubby refuse right
answers more than right answers alone do. **The paper** (ICDL 2027; Nick: "if it solves postmortem could be the whole
paper"): the same mistake memory, at the level of causes, is the sandbox agent's postmortem keeping its blamed bits as
suspects that later items prune, until one explains everything seen, at which point it becomes a lesson and the cleared
suspects stay as guards (the sandbox's own order). If that moves the postmortem's transfer off chance on public rules,
the paper becomes one learning lifecycle, shown in the agent and in the language model. **T6, adapters generated instead
of trained (added 2026-11-06; Nick: "yes"):** once the loop holds about 20 promoted adapters, each kept beside its gap's
VM-verified examples, a small hypernetwork learns to map a gap's examples to its LoRA. This is H0's θ = f(c) applied to
the loop's own library. **The test:** leave one gap out; generate its adapter from its examples, and compare it with the
trained adapter and with no adapter on the gap's held right rate, with seconds per adapter against minutes. A generated
adapter passes the same gate as a trained one. **Pass:** at least 70% of the trained adapter's lift. **Kill:** under
30%, or the base battery lower. **In plain words:** after Cubby has trained itself enough small add-ons, it learns to
make new ones directly, in seconds. **Over a conventional model:** fine-tuning there is a training job every time; here
the skill-maker improves, and nothing it makes is used before the gate passes it. Facts stay in the store, so a
generated adapter carries only a skill, which asks less of the hypernetwork than general instruction tuning does. **The
known trap:** H0's own toy test found a naive hypernetwork worse than none; only the hardened form worked. Related, to
cite once checked: HyperTuning (Phang et al., ICML 2023), HINT (Ivison et al., ACL 2023), Text-to-LoRA (Sakana AI,
2025). T6 waits for the library. Status: **proposed** (2026-11-04; T6 added 2026-11-06).

**H-A16 — (added 2026-11-05) A spiking fast learner beside the frozen trunk, its STDP gated by the VM's dopamine (a
three-factor rule), lifts a gap within one day of verdicts, before any adapter is trained; its sparse code keeps it from
H-A1's forgetting; and the hormones, driven by the loop, set its neurons' threshold and leak.** Source: Nick,
2026-11-05: the SNN "should be the one involved in the self learning loop like in cubemind", and "I wonder why we dont
train these". **What the code did, read the same day:** in cubemind's trained LM (`cubemind/training/vsa_lm.py`) the GIF
neurons and the hormone ODE are built but not wired (the block's forward is the plain MinGRU block's, by its own
docstring, and `neuro.update()` is called with no inputs); in cubemind's live brain (`cubemind/model.py`) the SNN
changes the state, but its STDP (`brain/synapsis.py`) strengthens what fires together whatever the answer was, the
hormones are computed after the SNN from novelty and the state's norm (threat and valence constant) and go only to the
speaking LLM, `train_step`'s loss is never used, and HybridFFN's "learnable" gate is never updated (`snn_ratio` also
goes through a sigmoid, so 0.3 is a 57% SNN blend); here, the ported ODE's `modulate_tau` has no caller, and the brain's
fast route (`cubbyllm/brain.py`) has an SNN slot that was never filled (a lexical and retrieval read today). So nothing
has ever taught the spiking part what was right, and a correlation rule alone is the failure H-A1 measured (forgot 94%).
**What H-A15's loop adds:** a VM verdict on every attempt, the third factor that turns STDP from drift into learning
(related, to cite once checked: Izhikevich 2007 on STDP gated by dopamine; Frémaux & Gerstner 2016 on three-factor
rules; e-prop, Bellec et al. 2020). 0049's first T1 also found the limit this could lift: the loop learns only from its
own right attempts, so wd/fr (5.7% right) could not start, while this learner learns from wrong verdicts too. Builds on
H-A15 (the loop), H-A1 and H-A5 (dense Hebbian forgets; a sparse code holds 100% at N = 512 where dense codes fall to
0), the striatum (its reward: +1 certified, 0 refused, −5 wrong; δ = reward − expected) and H0 (the hormones as
context). **The learner:** (1) the trunk's state at the decision, through a fixed 8× random up-projection, drives GIF
neurons ported clean from `cubemind/brain/gif_neuron.py`, their threshold set so about 2% fire at rest (H-A5's
sparsity); (2) plastic weights from the spikes give a bias over the decision's candidates (for wd, the property word at
HOP1); (3) each synapse from a firing neuron to the chosen candidate keeps a decaying eligibility trace; (4) the VM's
verdict, through the striatum's reward, gives δ per skill signature, and Δw = η·δ·trace, each neuron's outgoing weights
bounded; (5) the ported ODE, driven by the loop (a right verdict as reward, a run of wrongs as threat, a new key as
novelty), sets the neurons' threshold and leak through its own `modulate_threshold` and `modulate_tau`; (6) by day it
learns from each verdict, and by night 0049's adapter trains on what the day found right, after which the learner's bias
for that key can be cleared (the hippocampus by day, the cortex by night). **In plain words:** Cubby gets a quick reflex
beside its slow memory. Each time the checker says right or wrong, the connections that produced the answer get stronger
or weaker on the spot, and overnight the lesson moves into a permanent add-on. **Over a conventional model:** a
conventional model learns only when it is retrained, from labelled examples, so between retrainings the same mistake
comes back every time. Here the correction starts with the next question, needs no GPU, learns from wrong answers as
well as right ones (so a skill near zero can still start), and the trunk never changes. **Validate** (order 0053; the
lines are there): A, a day replayed on CPU on wd/en and wd/fr, each candidate's verdict cached once, in four arms: none,
the spiking three-factor learner, plain STDP without dopamine (cubemind's rule), and a size-matched dense learner with
the same rule; B, consolidation through one adapter night; C, the hormones against fixed threshold and leak, run only if
A passes. **Kill:** the held right rate up less than 5 points on both keys; dopamine adding under 1 point over plain
STDP; wd/en losing more than 1 point after a wd/fr day (H-A1's failure); or wrong verdicts on the held set above the
base's. A dense learner matching the spiking one kills the spiking part only: the rule survives, without spikes. Status:
**proposed** (2026-11-05).
