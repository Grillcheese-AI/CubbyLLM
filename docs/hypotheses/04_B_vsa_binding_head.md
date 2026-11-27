<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 4. Group B — VSA binding head fix

**H-B1 — cubby-concepts' `vsa/core.py` + `vsa/resonator.py` is usable as CubbyLLM's binding-head algebra,
conditionally.** Status: **VERIFIED as working code** (61/61 tests independently reproduced), **PLAUSIBLE as a fit for
Cubby specifically**. The four-layer pipeline — `core.py` (bind = elementwise product, bundle = elementwise sum + sign
threshold, permute = cyclic shift, all on D=10000 bipolar {-1,+1} int8 vectors), `sdm.py` (Kanerva associative storage),
`graph.py` (`SequentialHyperGraph`, chains simple binds for combinatorial reach), `resonator.py` (`ResonatorNetwork`,
iterative factorization via continuous-softmax cleanup, RMS renormalization, verified stochastic restarts) — is real,
vectorized, honestly documented (it proves and tests a negative capacity result rather than hiding it). F=3
factorization holds 100% accuracy up to codebook size ≈80; F=2 (the plain binding-head case) is trivial to codebook size
1000+. The condition: this only drops in cleanly if CubbyLLM's binding head can operate in bipolar ±1 space and doesn't
need to be trainable/differentiable — there are no learned parameters anywhere in this code, `bind`/`bundle`/`permute`
are fixed algebra and the resonator's codebooks are pre-generated, not trained. **Validate** by checking whether Cubby's
current `VSABindingHead` representations are already bipolar-compatible, or would need a binarization/quantization step
(or a low-risk torch port of what are conceptually simple elementwise ops) first.

**Validation result (2026-07-23 campaign): condition RESOLVED — binarization is ~free in the operating regime.** Test
suite re-reproduced again (61/61, 55.1s). The head's continuous query keeps cos 0.798 with its sign() (the theoretical
√(2/π)), and in the regime that matters (query cos≈0.55 to its target codeword) binarized readout keeps **100.0%** top-1
target retention at D=10240, V=4096. (Zero-signal worst case: 17.6% argmax agreement — reported for honesty; not the
operating point.) H-B1's bipolar algebra is compatible with the head via a cheap sign(). `validation/exp_b_algebra.py`.

**H-B2 — GrillCheese's `holographic_relation_test.py` + `brain_resonator_test.py` is a second, independently-useful
reference implementation.** Flagged by its own survey as "the single most relevant find for CubbyLLM's VSA/HDC
binding-head work." Real, dependency-free (NumPy+FFT), tested, runnable code: Holographic Reduced Representation binding
(`bind(a,b) = real(ifft(fft(a)*fft(b)))`, circular convolution — a *different* algebra than cubby-concepts' elementwise
bipolar bind), unbind via correlation, a genuine resonator network doing the classic Frady/Kent/Olshausen/Sommer
iterative decode loop, plus a binary/bipolar "Vulkan-ready" variant (>95% classification accuracy in testing) closer to
Binary Spatter Code. Status: PLAUSIBLE, VERIFIED as real runnable code but not benchmarked at CubbyLLM's scale.
**Validate** by comparing HRR/circular-convolution against Binary Spatter Code/elementwise-bind directly against Cubby's
actual requirements before choosing — they are genuinely different algebras (HRR handles real-valued/continuous vectors
more naturally; BSC is simpler and cheaper in hardware) and the choice has real downstream consequences, not just a
naming difference.

**Validation result (2026-07-23 campaign): head-to-head run at D=10240.** Role-filler unbind from bundled superpositions
(100-filler codebook): both algebras hold ≥97% recall out to **n=320 bound pairs** — far beyond any realistic
binding-head load; first separation only at n=320 (HRR 100%, BSC 97%). Capacity does not decide this choice;
representation compatibility does — HRR unbinds the head's continuous query natively, BSC needs the sign() step H-B1's
result just showed is ~free. Note H-B5's confirmation adds a third family (grilly's NVSA block codes) whose docstring
targets exactly this continuous-vs-bipolar tension. `validation/exp_b_algebra.py`.

**H-B3 — The current binding head's unbind is a structural placeholder, so this is new-build, not patch work.** Status:
**re-check DONE (2026-07-23 campaign) — the claim as written was MISATTRIBUTED, the conclusion survives.**
`vsa_binding_head.py` was reread in full: it contains **no `UNBIND_ROLE` and no unbind mechanism of any kind** — it is a
clean, well-engineered cosine-readout head (continuous query projection → cosine vs fixed bipolar codebook → τ·softmax;
the only `UNBIND_ROLE` in cubby-lm is a tokenizer special-token string). The dict-lookup `UNBIND_ROLE` is **cubemind VM
code**: `reasoning/vm.py:602-606`, `_unbind_role` returns `self._role_bindings[reg].get(role)` — confirmed never
touching `bc.unbind`. So: real unbind must indeed be built from scratch (candidates H-B1/H-B2/H-B5), but it is
VM/reasoning-side work, and the LM head is NOT broken — arguably the strongest piece of predecessor code reviewed.

**Re-confirmed unchanged 2026-08-02, with a detail that makes it worse than "placeholder".** `cubemind/reasoning/vm.py`
L602-606 is still `return self._role_bindings[register_name].get(role_name)` — a dict lookup. But the *write* side does
real algebra and then discards it: `_bind_role` (L575-600) computes `bound = self.bc.bind(role_vec, filler_vec)` and
bundles it into the register (L591-593), then at L598-600 shadows the same association into a plain Python dict — **and
the dict is the source of truth**. So the VSA is computed and thrown away. All four `tests/test_vm.py` assertions
(L462-486) read through the dict, giving the algebra zero test coverage. Real `bc.unbind` exists
(`cubemind/ops/block_codes.py` L157, numpy fallback L411) and six other opcodes use it (`_diff` L610, `_detect_pattern`
L630, `_recall_position` L770, `_learn_transformation` L813, `_discover_sequence` L839, `MindForge.verify_duality` L283)
— so this is one targeted placeholder, not a repo-wide gap. **Nothing to port for the hard part:** grep for `resonator`
across cubemind's Python returns a single hit and it is a comment (`vm.py` L75). No resonator, no iterative
factorization, no SDM. `CleanupMemory` (`vm.py` L68) is a brute-force linear scan over a dict, and cubemind's own
`vm.md` L213 concedes it "needs C++ with grilly's hamming_topk".

**Update (2026-08-05, reasoning-bridge slice) — real unbind EXISTS and is now Python-reachable; the "no real
implementation anywhere" framing is superseded.** The placeholder above is cubemind's *Python* VM. Separately,
**cubelang's Rust VM has real VSA unbind** — `op::UNBIND` + cosine cleanup, reachable from a CubeLang program as `use
vsa; recover(reg, role)` (built in the cubelang foundation cycle). The reasoning-bridge slice exposed it to Python over
a **symbolic boundary** (roles/fillers/program in, a recovered symbol out; no raw hypervector crosses — the trunk's
grilly algebra and cubelang's Rust VSA are non-interoperable families): plant a `(role→filler)` binding, recover it
through the real unbind+cleanup over **JSON (M1)** and **protobuf (M2)**, against wrong-role (recovers a *different*
filler) and absent-role (recovers *nothing*) controls (`cubbyllm/bridges/`, `tests/bridges/`). M2 also surfaces the
cleanup **confidence** (the recover cosine as an additive `RunResult.similarity` side-channel — ~0.5 for a clean
2-pair-bundle recovery, `None` when nothing binds) and gates the path with **verify-before-execute** (the reasoning
source is `compile_strict`-checked before it runs; a before/after worktree proof confirmed the from-source path was
previously unverified). So real unbind is no longer hypothetical or unbuilt: it works in cubelang's VM, Python-callable
with confidence + verification. What remains genuinely unbuilt: cubemind's *own* VM unbind (still the dict lookup), and
the resonator / iterative-factorization / SDM cleanup for the hard many-item case (grilly's `hamming_topk` is the
intended substrate there).

**H-F2 first step (2026-08-05).** This reasoning bridge is the first concrete, tested step toward the Cubby↔CubeMind
bridge (H-F2): a controlled Python↔VM reasoning roundtrip over a symbolic boundary. The full bidirectional bridge H-F2
calls for — carrying a context/task embedding in both directions plus generated-parameter/specialist handles
(`cubbyllm/bridges/world_model.py`'s `WorldModelBridge` Protocol) — remains the later cycle.

**H-F2 M1 built (2026-08-06) — automatic specialization across the Cubby↔MoWM bridge.** The full bidirectional bridge's
first slice is built and tested end to end over the block-code boundary. CubbyLLM poses a `Challenge` (a `(k,l)` grilly
block-code) through the `WorldModelBridge` Protocol; MoWM's real `CubbyBridge` adapter (`mowm/bridges/`, on the real
741-green mowm suite) answers it in three tiers: (1) **reuse** a memoized specialist for an exact-tag repeat; (2)
**route** to an expert world by cosine similarity of the context to that world's **individual axioms** when the
challenge resembles one; (3) **spawn** a new specialist from the nearest axioms when none matches. Proven with three
controls: a novel (hash-derived) challenge spawns exactly one world and reuses it (single-spawn control); a challenge
equal to a world's axiom routes to it, never spawning (reuse control); and — the load-bearing one — a **near-threshold
fuzzy challenge (a real axiom perturbed to cosine ≈0.75, under a fresh tag) routes to that world without spawning**,
proving genuine *similarity* routing rather than exact-match. Protocol-first coupling; dependency mowm→cubbyllm only
(CubbyLLM stays torch/mowm-free at import). Structure: CubbyLLM ships `WorldModelBridge` + `FakeWorldModel`
(`cubbyllm/bridges/`); MoWM ships `CubbyBridge` (`mowm/bridges/`).

**Honest scope of M1 (a load-bearing correction from the whole-branch review).** Routing must run on a world's
**individual axiom vectors**, not its `axiom_vec` **signature**: the signature is a `bind`-chain, and VSA binding is
quasi-orthogonal to its own inputs, so cosine against it is **inert** — it matches nothing but a hand-constructed exact
copy. The first shipped adapter routed on the signature (and design spec §4.4, plus an earlier draft of this note,
described gate-weight routing and "a spawned world enters the active pool" idempotence); **none of that was functional**
— the review caught it, and routing was re-based on individual axioms (`AxiomLibrary.select`'s basis) so the similarity
tier is real. Even so, that tier fires only for contexts near the library's **axiom basis**; at toy scale the tests
construct such contexts synthetically, because there is no organic context-encoding pipeline yet. **Whether real
trunk-inferred contexts land near axioms is a separate, unvalidated data question** — the honest M1 claim is
"specialization via exact-tag memoization + spawn-on-novel + *demonstrated* (not yet organically-fed) individual-axiom
similarity routing." Exact-repeat idempotence is guaranteed by the `tag → world_id` memo regardless of routing.
Cross-world data-sharing, SR axiom discovery, imagination rollouts, and multimodal challenges remain later slices.

**H-F2 M2 screened (2026-08-06) — the semantic-encoding gate is CLOSED, and the mowm-side path was deliberately NOT
built.** M2 took the data question M1 deferred (*do organically encoded contexts land near axioms, so the similarity
tier fires?*) and ran it as a screen with a kill criterion rather than as a build: mowm's own **280 axioms across 15
domains**, exported once to `data/mowm_axioms.json`; leave-one-out over all 280; **cross-view** (seeds encoded from
`formula_str`, challenges posed as `name`); routing scored exactly the way `CubbyBridge._best_match` routes — max cosine
to a world's *individual* axioms, with the axiom's own formula excluded — at the decided k=80 / l=128 algebra. **Three
of Stage 1's four bars FAILED** (`validation/logs/exp_f2m2_stage1.json`): bar 1 baseline sanity **PASS**; bar 2 encoder
sanity (best trunk self-retrieval top-1 ≥ 0.50) **FAIL at 0.029**; bar 3a (best trunk macro ≥ 2× macro-chance, i.e. ≥
2/15 = 0.1333…) **FAIL at 0.0962 = 1.44×**; bar 3b (best trunk ≥ 0.60 × `ref`) **FAIL at 0.269**. The decision rule's
`trunk < 0.60 × REF` branch fired, recorded in the log as `"fine-tune (M3) — gap is large"`, so the design's §4.3 live
confirm and its `mowm/encoding/semantic_axioms.py` **were never built**. **mowm has no semantic axiom path**:
`WorldEncoder`'s hash encoding is untouched and M1's shipped bridge is unchanged. A closed gate is M2's deliverable, not
an unfinished milestone.

**Premise CONFIRMED — hash encoding routes at chance, and against the matched null "at chance" is now a direct
measurement rather than an inference.** The `hash` arm (BLAKE2b-seeded one-hot-per-block, M1's status quo, reimplemented
in ~5 lines so the screen imports neither mowm nor cubemind) scores **macro 0.0747** and self-retrieval **0.0036**, i.e.
1 axiom in 280 (`exp_f2m2_stage1.json`). Bar 1's ceiling was `1.2 × macro-chance = 0.0800`, so the PASS is near-binding,
not decorative — and it only became so after the methodology corrections below. The stronger statement comes from the
measured null: a **pure-noise one-hot encoder** in exactly `hash`'s format scores **macro 0.0666 ± 0.0125** over 10
seeds (`validation/logs/exp_f2m2_null_baseline.json`), so `hash`'s 0.0747 is **+0.6 sd — statistically indistinguishable
from noise**, and its self-retrieval 0.0036 matches the noise floor of 0.0029 ± 0.0033 exactly. This is the sentence M1
could not write: BLAKE2b really does destroy the locality similarity-routing needs, so the similarity tier cannot fire
organically against hash-encoded axioms.

**The `hash` arm's small excess over `1/15` is an argmax tie-breaking artifact, and it is now attributed rather than
left loose.** One-hot-per-block cosine takes only 81 values at K=80, so exact ties are routine: **59.6% of `hash`'s
routing decisions are ties, mean tied group 3.21 candidates**, against **55.9% ± 4.1%** for the matched one-hot noise
null (`exp_f2m2_null_baseline.json`). `np.argmax` breaks every tie toward the **lowest index**, and
`data/mowm_axioms.json` stores its 15 domains as **contiguous runs with `atomic` first**, so the pull is systematic. It
shows up in the null itself: one-hot noise inflates the two lowest-index domains (`atomic` **0.1529** against its own
baseline of 0.0573, `bio` **0.1467** against 0.0502) and *depresses* the largest one (`physics` **0.1595** against
0.2975), which is exactly the shape of `hash`'s real per-domain profile (`atomic` 0.2941, `physics` 0.2262, and zero in
eight domains). The excess is a property of the tie-break, not of the encoder — and its direction makes bar 1 *harder*
to pass, so the PASS remains trustworthy.

**On micro, report both baselines — and the matched one is what changes the readings.** The **majority-class rate,
0.300**, is a legitimate strong baseline (what "always guess physics" scores), and only `ref` (0.450) clears it. But the
**matched null for an argmax-over-279-candidates metric** is `Σ_d n_d(n_d−1) / (280·279) = 9576/78120 = 19/155 =`
**0.1226**, reproduced empirically by the noise encoder at **0.1232 ± 0.0074** (`exp_f2m2_null_baseline.json`). Against
*that* floor: `hash` **0.1179** sits essentially **on** it — 0.0047 below the exact value, and **+1.40 sd against its
own format-matched one-hot null (0.0921 ± 0.0184)**, or −0.73 sd against the dense null (0.1232 ± 0.0074); |z| ≤ 1.4
either way, so the routes-at-chance reading holds under both — corroborating that finding on a second metric; `t-bag`
**0.1107** is −1.7 sd; `ngram` **0.1036** is −2.7 sd; and **`t-ctx` 0.0786 is −6.0 sd, i.e. mean-pooled trunk states
route *worse than a random encoder* on micro** — a stronger and less comfortable result than "worse than always guessing
physics." Only `ref` clears both baselines (+44 sd). **Read this as a property of the representation, not a defect in
the model**: pooled next-token states are not merely uninformative for routing, they are *anti*-informative on this
metric — the anisotropic cone (unrelated cosine 0.756) concentrates every text near one direction, so what little cosine
ordering survives is dominated by something other than domain. Noise at least has no systematic structure to mislead
with; a strongly anisotropic embedding does. That is exactly why M3's answer is a separate encoder component rather than
more trunk (see below), and it is also why `t-ctx`'s *macro* can sit +3.7 sd **above** its null while its micro sits 6
sd below — the two metrics genuinely disagree about this arm, and both readings are reported rather than one being
chosen.

**An LM trunk is not a retrieval encoder — and M2's real contribution is quantifying how far it isn't.** State the
framing before the numbers, because the earlier version of this paragraph read as the trunk *failing*, which
mis-describes what was measured. `hd5_mem21.pt` is a **causal language model**. Its *training* objective is next-token
cross-entropy (`cubbyllm/training/loop.py:124`; this run drives that loop, `train_colab.py:478`) plus exactly two
additive terms, **neither of which is a similarity, contrastive, or retrieval objective over sentence representations**:
the θ=f(c) hardener's anti-drift penalty on *generated parameters* (`loop.py:126-129`), and a VSA binding aux loss
**gated by `bind_weight` and hard-defaulted to 0** (`loop.py:130-137`; `CB_BIND_W` sits behind a `LEAVE AT 0` guard at
`train_colab.py:95-98`) — the one similarity-shaped loss this project ever had, **falsified and switched off** in H-B6,
with this run confirmed at `bind_weight=0` two independent ways (see the long-horizon note below). So nothing in the
objective ever asked the model to make two texts about the same thing land near each other. **Mean-pooling its hidden
states is a convention borrowed from encoder-style models — not an interface the trunk was trained to expose.** Its
measured anisotropy (unrelated-sentence cosine **0.756**, `exp_f2m2_selftest.log`) is the *expected* geometry of a
next-token objective, not a defect. So the arms below measure an LM at a task it was never built for; that is a
legitimate and useful measurement, and it is **not** a verdict on the model at its actual job (generation, and carrying
context for θ=f(c)). Both trunk arms were built from the real 92k-step checkpoint (`$CUBBY_MODELS\hd5_mem21.pt`,
blake2b-128 `6c9f6ddff5342e268cac805615d0d74a`, stamped in `exp_f2m2_stage1.log`): `t-bag` = mean of trained
token-embedding rows, `t-ctx` = full forward, mean-pooled. `t-ctx` won at **macro 0.0962 (1.44× chance), self-retrieval
0.029**, against `t-bag` **0.0754 (1.13×)** and the `ref` MiniLM-L6-v2 arm's **0.3580 (5.37×), self-retrieval 0.229** —
`trunk/ref = 0.269` (all `exp_f2m2_stage1.json`). The **per-domain breakdown explains the weakness rather than merely
reporting it** — but only once each domain is read against **its own** baseline. **The per-domain null is not flat: it
is `(n_d − 1)/279`**, ranging from **0.0323** for a 10-member domain (`identity`) to **0.2975** for `physics` (84
members), because the metric takes an argmax over the other 279 items and a domain is hit in proportion to the
candidates it owns (`validation/logs/exp_f2m2_null_baseline.json` tabulates all 15, analytic and empirical). Read that
way: `t-ctx` reaches **linguistics 0.500** against a 0.0394 baseline (**12.7×**) and **language 0.400** against 0.0502
(8.0×), but **physics 0.0357 against 0.2975 — 0.12×, far *below* chance for that domain**, where `ref` reaches **0.738**
(2.5×). `ref`'s **linguistics 0.833** is likewise a **21×** multiple of its baseline, a much larger one than its
headline physics 0.738 — the flat-baseline reading had these two backwards. **8 of 15 `t-ctx` domains score exactly
0.0**, which is a weaker signal than it looks for the small ones (a 10–12 member domain's baseline is only 0.032–0.039,
so a single miss-free run is unremarkable) and a strong one for `physics`, whose 0.298 baseline makes 0.0357 a real
deficit. The overall shape still holds: a general-text LM with real language signal and essentially none on symbolic
notation, which is most of what this corpus is made of. Contextual pooling does beat bag-of-embeddings, so the trunk
body contributes something — just nowhere near enough to serve as a routing encoder, which it was never trained to be.

**What follows for M3 is a component boundary, not a repair job.** The question is not "how do we fix the trunk" — **M2
measured nothing about generation, so none of this is evidence about the trunk's performance at its actual task, in
either direction** — but that **the routing encoder is a separate component from the trunk**, and M2 has now costed the
alternatives. Three candidates, in rough cost order. **(a) A small purpose-built or fine-tuned encoder**, which the
`ref` arm sizes — and the size comparison has to be **like-for-like**, which the obvious "22M beat 153M" phrasing is
not. MiniLM-L6-v2's **22.7M total** reaches macro 0.3580 where the trunk's **153.4M total** reaches 0.0962; but **131.1M
of the trunk's total is two untied `127996 × 512` vocab tensors** (2 × 127,996 × 512 = 131,067,904 of 153,383,440),
leaving a **22.3M body** — against MiniLM's own **11.0M body** (the remaining 11.7M being its word-embedding table).
**The honest read cuts both ways:** the trunk is not 6.8× the size of the thing that beat it, so raw scale does not
explain the loss; but its *body* is still about **2× MiniLM's body** and loses by 3.7×, so "it is merely smaller" does
not explain it either. (Counts derived from the pinned checkpoint's own tensor shapes — identity stamped in
`exp_f2m2_stage1.log` — and from the pinned `sentence-transformers/all-MiniLM-L6-v2`; the design spec's §3 makes the
same body-vs-total point.) **(b) The training-free lexical route**, which is cheaper than it looks — see the `ngram`
result above and the mechanical note there: `ngram` and `hash` share the *same* BLAKE2b machinery, so the finding is
that **whole-string** hashing destroys locality, not that BLAKE2b does, which makes upgrading
`WorldEncoder._hash_to_vec` inside the existing hash architecture a near-free lever rather than a rewrite. **(c) A
trained projection head *on top of* the trunk** — the only option that reuses the trunk **legitimately**, by learning
the encoder interface rather than assuming mean-pooling already is one. Note that (c) is also the only one that keeps a
single model serving both roles, which is what the θ=f(c) story wants; M2 says nothing about whether it works, only that
it is the honest form of the question.

**Uncomfortable control: a training-free lexical sketch beats the trained trunk.** The `ngram` arm — character-trigram
block codes bundled by superposition, **no model, no training, no projection**, born directly in block space — reaches
**macro 0.1400 = 2.10× chance** with self-retrieval **0.100** (`exp_f2m2_stage1.json`), beating both trunk arms and
being **the only non-`ref` arm to clear the 2×-chance bar that bar 3a asks of the trunk**. Its macro is **+8.2 sd above
the matched noise null of 0.0599 ± 0.0098** (`validation/logs/exp_f2m2_null_baseline.json`), so the win is real and not
a baseline artifact. **The qualifier this branch's own micro/macro correction demands: `ngram` is mid-pack on micro and
below `hash` there** — 0.1036 against `hash`'s 0.1179, and **−2.7 sd below the matched micro null of 0.1226**. So the
honest claim is narrower than "the sketch beats the trunk": it beats both trunk arms on the **macro** metric and on
self-retrieval, while its **micro** sits below the random-encoder floor like every non-`ref` arm. Macro is the metric
the kill criterion uses and the one that is not dominated by physics's 30% share, so the comparison stands — but it is a
macro claim, not a blanket one. Two consequences, both honest ones. (a) `hash` should not be presented as the only
baseline the trunk had to clear: `ngram` is a *stronger* training-free null, and the trunk loses to it. (b) M3's
question is therefore not just "fine-tune an embedder" but **"why does a next-token LM carry less domain-routing signal
than raw lexical overlap?"** — a question about what the objective builds, not about model quality. **(c) The
mechanically important detail, easy to miss: `ngram` and `hash` share the *same* BLAKE2b machinery** — identical
`blake2b(..., digest_size=8)` → `int.from_bytes(..., "little") % 2**63` → `default_rng(seed).integers(0, L, size=K)`
seeding and index draw (`validation/exp_f2m2_semantic_routing.py:115-147`). They differ in exactly two ways: `hash`
hashes the **whole string** and *assigns* one-hot, while `ngram` hashes **character trigrams** and *accumulates*
(bundles) them. So the premise M2 confirmed is **"whole-string hashing destroys locality," not "BLAKE2b does"** — and
the fix lives inside the existing hash architecture rather than replacing it, which is why this is the cheap lever in
M3's option (b) below.

**Discretization is INCONCLUSIVE — and M2's earlier "quantization preserves, even improves, routing" claim is
RETRACTED.** The shipped Stage-2 numbers ordered **`pq-adc` 0.4188 > `pq` 0.3719 > `dense` 0.3580 > `argmax` 0.3133** on
`ref` (`exp_f2m2_stage2_ref.json`) and the same way on `t-ctx` (**0.1068 > 0.1030 > 0.0962 > 0.0736**,
`exp_f2m2_stage2_t-ctx.json`), with bar 4 recorded PASS on both. Review found the cause: **the PQ codebook was fit on
the very items leave-one-out exists to exclude.** At 280 axioms over `L=128` centroids that is ~2.2 points per centroid,
so a candidate's reconstruction can carry a fraction of the query's own formula — a channel `dense` does not have. The
tell was already in the table: `argmax`, the only discretizer with no fitted codebook, was the only variant that *lost*
to dense. Two controls followed (`exp_f2m2_leakage_ref.json`, `exp_f2m2_leakage_t-ctx.json`), both logging the verdict
`"gain FAILS both controls -- shipped result was leakage"`. Under the **item-disjoint control** (fit on a stratified
half, score only the other half, `n_scored = 146`) **the ordering inverts and PQ collapses**: on `ref`, `argmax`
**0.3281** now beats `dense` **0.3017** while `pq` falls to **0.1971** and `pq-adc` to **0.2116**; on `t-ctx`, `pq`
**0.0828** and `pq-adc` **0.0778** fall below `dense` **0.1276** (`argmax` **0.1149** sits just under dense there, so
the argmax-overtakes-dense inversion is `ref`-only). **What is unanimous across both arms is that `pq` and `pq-adc` drop
below `dense` once the codebook cannot see the scored items.** The **source-disjoint control** (codebook fit on the
`name` view) is mixed and arm-specific: on `ref`, `pq-adc` 0.3874 > dense 0.3580 but `pq` 0.3378 < dense; on `t-ctx` the
two swap (`pq` 0.0976 > dense 0.0962, `pq-adc` 0.0821 < dense). **Do not read this as "PQ hurts" either.** Neither
control is clean — the name-fit control confounds surface modality (it fits on natural language and scores symbolic
notation), and the half-split control's codebook is fit on ~134 points across 128 centroids (~1.05 per centroid),
degenerate enough that it may *understate* PQ. **The correct conclusion is that this corpus cannot settle the
question**, with a concrete requirement for one that can: **n ≫ L — at least ~1280 axioms for L=128 — with a fit set
disjoint from the scored items.** ADC remains architecturally sensible on its own terms (only the stored side must be
one-hot to participate in `bind`/`bundle`; a routing query never needs quantizing) but it has **no evidence of improving
routing**. The one dense-vs-one-hot fact that survives the disjoint control is bar 4's actual question —
one-hot-per-block does not *destroy* routing relative to dense, though under control B that holds via `argmax` (0.3281
vs 0.3017 on `ref`; 0.1149 vs 0.1276, still ≥ 0.80× dense, on `t-ctx`), not via `pq`. **A note on the two exported
artifacts, because they outlived the conclusion they encode:** `data/axiom_embeddings_ref.npz` and
`data/axiom_embeddings_t-ctx.npz` store the Stage-2 embeddings together with `discretization="pq"`, that variant's
`tau_match`, and `centroids` — *the leakage-contaminated codebook itself*. They are a **record of what the screen
produced, not a configuration to load**: nothing in CubbyLLM or mowm reads them, and the only consumer they were ever
written for (the mowm-side semantic path, the plan's Task 7) was gated out by the Stage-1 kill criterion and never
built. Both files now carry that retraction as `note`/`retracted` string fields **inside the npz** so a future reader
cannot lift `tau_match` or `centroids` out of them without seeing it.

**The ROC-AUC null on this corpus is ~0.11, not 0.5 — so the earlier "every AUC is below 0.5, therefore nothing
separates" reading is WRONG, and correcting it makes the arms look better, not worse.** AUC here is the pooled, unpaired
Mann-Whitney statistic over all same-domain maxima against all different-domain maxima. But `same_cos[i]` is a **max
over the (n_d − 1) same-domain candidates** while `diff_cos[i]` is a **max over the (n − n_d) different-domain
candidates** — roughly **14 against 265** on this corpus — and the max of more draws is stochastically larger, so this
statistic's null sits **far below 0.5 by construction**. That is a property of the candidate sets, not a diagnosis of
the encoders. Measured rather than argued: pushing a **pure-noise encoder** through the *shipped* `loo_domain_routing` /
`recalibrate_tau` over the real corpus's domain structure, 10 seeds, gives an AUC null of **0.1133 ± 0.0072** in the
dense code format and **0.1605 ± 0.0162** in the one-hot format (`validation/logs/exp_f2m2_null_baseline.json`). The
control that proves it is a candidate-set-size artifact: on a **balanced** two-domain structure over the same 280 items
(≈139 against 140 candidates), the same noise gives **0.5131 ± 0.0191** — 0.5, exactly as an AUC null should be when the
two candidate sets are the same size.

**Against that measured null, six of the eight recorded AUCs show real separation, one is marginal, and one is genuinely
below chance.** Reading each variant against the null for its own code format (`dense` and `pq-adc` compare continuous
vectors → the 0.1133 null; `argmax` and `pq` compare one-hot codes → the 0.1605 null): on `ref`, `pq-adc` **0.454** (≈
+47 sd), `dense` **0.430** (≈ +44 sd), `pq` **0.427** (≈ +16 sd) and `argmax` **0.358** (≈ +12 sd) all sit far above; on
`t-ctx`, `pq-adc` **0.246** (≈ +18 sd) and `dense` **0.227** (≈ +16 sd) do too; `argmax` **0.182** is **marginal** (≈
+1.3 sd, only just past the null's observed maximum of 0.180); and `pq` **0.0915** sits **below** its null (≈ −4.3 sd) —
the one arm the original blanket claim actually holds for, and notably the variant whose τ of **0.850** comes from the
leakage-contaminated codebook retracted above (`exp_f2m2_stage2_ref.json`, `exp_f2m2_stage2_t-ctx.json`, nulls from
`exp_f2m2_null_baseline.json`). So the honest statement is **`ref` separates strongly, `t-ctx` separates weakly but
really, and only `t-ctx`'s `pq` is at-or-below chance** — not "nothing separates."

**What survives, on better grounds: no M2 number is a usable `tau_match`, and M1's `0.35` is not replaced by any of
them.** Separation being above its null is not the same as a transferable threshold, and four independent reasons block
the transfer. (a) **The τ values live in incomparable spaces**: `t-ctx`'s `pq` τ of 0.850 is a one-hot block-code cosine
(81 levels at K=80), its dense τ of 0.257 is a continuous cosine, and M1's 0.35 is a k=4/l=32 one-hot cosine (five
levels). (b) The **corpus is cross-modal** (`name` against `formula_str`), so its same/different distributions are not
production's. (c) The `pq` τ specifically is an artifact of the **retracted** leakage result. (d) Even the best AUC
here, `ref`'s 0.454, is weak *in absolute terms* — a screen result, not a calibration. Treat every τ in
`exp_f2m2_stage2_*.json` and in the exported `.npz` files as a record of what this screen measured, never as a setting
to load.

**The corpus poses a harder task than production would — a calibration caveat on bar 2, not an excuse.** Even `ref`
reaches only **0.229** self-retrieval (`exp_f2m2_stage1.json`), because the cross-view design pairs natural-language
`name` against symbolic `formula_str`: a **cross-modal** match, harder than routing a natural-language question to a
domain. Bar 2's 0.50 was calibrated for a same-modality task, so read that FAIL as diagnostic rather than decisive —
while noting the caveat was written *after* the run and retroactively softens a bar that failed. The metric also has a
ceiling below 1.0: two `name` values (*De Broglie Wavelength*, *Boltzmann Distribution*) each appear twice with
different formulas, embed identically, and share one argmax, so self-retrieval tops out at **278/280 = 0.993**
(`validation/exp_f2m2_semantic_routing.py:339-346`, `self_retrieval_top1`'s docstring). This points at a cheap first M3
experiment before concluding the embedder is the bottleneck: encode axioms from **combined** text (name + domain +
formula) and pose natural-language questions against them — the production shape — rather than name-against-formula.

**Four methodology corrections, recorded because this is exactly what rots silently — and one error class caught three
times.** (a) The kill criterion compared **macro** accuracy against the **majority-class rate (0.300)**, which is a
*micro* baseline. Macro-averaging removes class imbalance by construction, so every constant classifier — the
majority-class strategy included — and uniform-random guessing all score exactly `1/n_domains`. The category error made
bar 1 trivially satisfied and set bar 3a at an unreachable macro of 0.600; against the correct macro-chance of 0.0667
the bars become 0.0800 exactly (near-binding, actual 0.0747) and 2/15 = 0.1333… (reachable, actual 0.0962). Both chance
levels and both actuals are in `exp_f2m2_stage1.json` (`micro_chance`, `macro_chance`, `arms.hash.macro`,
`arms.t-ctx.macro`); the two bounds are `1.2 ×` and `2 ×` macro-chance per the design's §4.2. **The measured verdict did
not change** — the bars were wrong, the answer was not. (b) The design spec claimed challenges were also posed in the
reverse direction; **they never were** (`run_stage1` measures name→formula only, and both directions share one symmetric
cosine matrix), so the claim was dropped from the spec rather than quietly left standing. (c) **The whole-branch review
found (a) recurring in three more places** — ROC-AUC read against 0.5, micro read only against the majority-class rate,
and per-domain recall read against one flat number, when each of those statistics is an **argmax over ~279 candidates**
whose null is set by candidate-set sizes rather than by class priors. All three are now measured rather than assumed,
with a **pure-noise encoder** pushed through the *unchanged* shipped metrics over the real corpus's domain structure and
over a **balanced** control structure, 10 seeds each (`validation/exp_f2m2_semantic_routing.py --null-baseline` →
`validation/logs/exp_f2m2_null_baseline.log` / `.json`). The measured nulls: **AUC 0.1133 ± 0.0072** dense / **0.1605 ±
0.0162** one-hot on the corpus structure against **0.5131 ± 0.0191** on the balanced one; **micro 0.1232 ± 0.0074**
(analytic 19/155 = 0.1226); **macro 0.0599 ± 0.0098** dense / **0.0666 ± 0.0125** one-hot (analytic 53/837 = 0.0633,
close to but not equal to the 1/15 the bars use); **self-retrieval 0.0032 ± 0.0026**. **Two of the three corrections
made findings stronger, not weaker** — `hash` routing at chance became a direct measurement, and `t-ctx`'s micro turned
out to be *below* the random-encoder floor — while the third **reversed** a claim: "no arm separates same-domain from
different-domain" was an artifact of the 0.5 null and is retracted above. The lesson is the one this milestone had
already learned once at bar 3a: **a metric's baseline is a property of the metric's own procedure, and must be derived
(or measured) from that procedure, not borrowed from a familiar number.** (d) **The same species of error at the level
of the design rather than the statistic: bar 3b (`trunk ≥ 0.60 × REF`) and the decision rule's "ship the trunk" branch
treated the trunk as an *embedder candidate*, which the architecture never claimed it was.** The trunk's role is
generation and carrying context for θ=f(c); it has no similarity, contrastive or retrieval objective in its loss, so a
bar asking it to come within 0.60× of a purpose-trained sentence encoder was measuring it against a standard that was
never the right one — a *category* mismatch, exactly like (a) and (c)'s *baseline* mismatches, just one level up. **The
bar's FAIL verdict stands and its consequence was correct** (do not wire a semantic path the screen showed does not
route), but the right reading of the FAIL is "an LM trunk is not a retrieval encoder," not "our trunk underperformed."
**Three instances of one error class in one milestone is itself the finding**: whenever this project compares a measured
thing to a reference, the next question should be *is this reference the right null/standard for this exact procedure?*
— asked before the run, not after it.

**Two supporting measurements, from the screen's own self-test (`validation/logs/exp_f2m2_selftest.log`).** The
orthonormal projection is cosine-exact as designed: the self-test asserts `max|Δ| < 1e-5` between pre- and
post-projection cosine matrices (`validation/exp_f2m2_semantic_routing.py:852-857`, inside `self_test`) and measures
**8.77e-09**, ~3 orders inside the bar. Note the **assertion bound is the enforced claim**; the printed value is the
measurement it passed by, not a separately guarded invariant. Separately, `t-ctx` is strongly **anisotropic** —
unrelated-sentence cosine **0.756**, against `t-bag`'s **−0.019** — i.e. mean-pooled LM hidden states occupy a narrow
cone. That is a *candidate explanation* for weak routing (a centering/whitening step is the standard remedy, and an M3
candidate), **not** evidence of structure, and it is corroborated independently by the logged τ 0.850 / AUC 0.0915
fingerprint above. Every figure in these paragraphs links a file in `validation/logs/`.

**H-F2 M3 fast-encoder screen (2026-08-07) — 0.86–0.95× the teacher's routing quality with no model in the challenge
path.** The user set the binding constraint directly: embedders are too slow for serving, so the challenge-side encoder
must be model-free (axioms are encoded offline regardless, per H0's frozen shape). Measured on the same 280-axiom LOO
screen against the same matched nulls, with per-challenge latency: **(1) model-free lexical arms** — words+IDF reaches
macro **0.1730** (2.73× null) at **~19 µs**, BPE token-ids+IDF **0.1520** at ~29 µs using ids the pipeline already
holds, and a K-sweep 80→320 buys only +6% relative at 4× memory, so dimensionality is a mild constraint rather than the
lever (`exp_f2m2_fast_arms.{log,json}`). **(2) BEAGLE distributional memory is a NEGATIVE** — accumulating each token's
co-occurrence neighbors over 200M wiki_full tokens *degrades* routing (pure memory 1.0–1.3× null; self-retrieval
collapses to 0.025; blending the signal back recovers only partway, 0.1204): superposition smears exactly the
surface-overlap signal this task runs on. Side-finding: raw-case vs lowercased costs ~0.04 macro on its own
(`exp_f2m2_beagle.{log,json}`). **(3) The result: model2vec-style static distillation** — run MiniLM **once, offline**,
over the vocabulary's word strings (2.3 s here), strip the anisotropy direction (mean+PC1) from the table, project
through the cosine-exact orthonormal map into (K,L) block space, and 0.5-blend each word's vector with its seeded
one-hot signal code so exact surface overlap keeps contributing; serve by table lookup + IDF-weighted adds. Winner:
**macro 0.3407 (5.38× null) = 0.95× the live teacher at ~25 µs/challenge** — ~350× faster than the teacher's 8.9 ms
single-query — with every ingredient attributable (words-over-tokens; −PC1 +0.067; +signal +0.021; learned-over-random
table values ≈ 2×) (`exp_f2m2_m2v.{log,json}`). **Honesty check passed:** refitting IDF and PC1 on external wiki_full
text (20k docs; 744/848 axiom words covered) drops the winner to **0.3079 (4.86× null, 0.86× teacher)** — the in-domain
statistics were worth ~0.033 and the result survives out-of-domain stats (`exp_f2m2_m2v_ood.{log,json}`). Caveats: the
benchmark is the same 280-axiom cross-modal screen (its caveats above apply); latencies are unoptimized Python (the Rust
VM / Vulkan path can only lower them); no τ here is a transferable operating point (the τ-caveat paragraph above applies
unchanged, and route-misses fall through to spawn, so routing errors are soft). **Wired into mowm** as
`mowm/encoding/semantic_words.py` — an offline `build`/teacher pass and a serve-time `FastWordEncoder` that imports only
numpy.

**H-F2 M3 production screens + the centroid/margin wiring (2026-08-07).** The production table was built
(`$CUBBY_MODELS\fastword_table_v1.npz`, 30,275 words = top-30k corpus DF ∪ all 901 axiom words, broad-corpus IDF + PC1
per the OOD form; axiom-corpus eval macro **0.2993 = 0.84× teacher**, self-retrieval 0.161, build log beside the
artifact) and screened on real data three ways. **(1) 42-domain routing** (`exp_m3_domain_routing.{log,json}`;
$CUBBY_DATASETS\domains, 34 seeded + 5 held out): macro/micro **0.5946 vs measured noise null 0.0339** at ~1.1
ms/passage, **0.75× MiniLM** (0.7973 at ~15 ms batched); first production-calibrated τ = 0.4518; the weak half
quantified — open-set spawn AUC 0.624. **(2) Open-set scorer screen** (`exp_m3_open_set.{log,json}`; 7 scorers × 3
rotated hold-outs): **centroid routing beats the shipped max-over-individual-axioms rule 0.680 vs 0.589 macro at ~40×
fewer comparisons** — empirically vindicating the M1 whole-branch reviewer's bundled-signature proposal (a centroid is
the similarity-preserving bundle; the bind-chain stays inert) — and **margin gating** (best − second-best) yields the
right operating profile: 92% precision on what routes, 87% of never-seen-domain challenges sent to spawn; per-domain
z-norm calibration is a recorded negative (0.435); MiniLM ceiling 0.825, i.e. separation *confidence* is the teacher's
remaining edge. **(3) Real questions** (`exp_m3_science_qa.{log,json}`; the 238-row tagged stratum inside
science_qa_prep — n=60, 3 tags, small-n caveats recorded): Q→A retrieval hit@1 **0.667** (0.70× teacher, chance 0.017)
and tag routing **0.900 = MiniLM parity** at 13× its batched speed — natural-question→explanation is far easier than
M2's synthetic name→formula (0.16), confirming that write-up's production-shape prediction. **Wired (mowm `3c217f9`,
suite 754 green):** `World.axiom_centroid` (normalized mean of individual axioms) + `CubbyBridge` tier-2 = centroid
scores with an absolute floor AND a `tau_margin` gate, ambiguity falling through to spawn; the three M1/M2
kill-criterion tests pass unchanged under the new rule, and three new tests pin centroid similarity-preservation,
tie→spawn, and the single-world floor. Both taus are deployment-calibrated constructor parameters — the screens are the
calibration method, and no τ transfers across encoders or corpora (the M2 lesson, unchanged).

**H-F2 M3 scale + memory-pathway evals (2026-08-07) — the haystack curve and the injection/chase result.** Two follow-on
evals on the production table, both on $CUBBY_DATA_ROOT/valid_scaling_law_with_facts.pq (10k multi-hop questions; every
row's answer verified to live in its final supporting fact, and its designated `fact_to_inject` verified present in the
chain, 10000/10000) with DBpedia abstracts as distractors. **(1) Haystack** (`exp_m3_haystack.{log,json}`; 400
questions, 636 target facts): the dense form is flat to 100k distractors (any-supporting-fact hit@1 0.978→0.973; MiniLM
anchor 0.993), and the **compact 80-byte form** (argmax index per block — the qFHRR integer-phase reading, ADC-scored
with no fitted codebook) degrades gracefully to **0.713 hit@1 / 0.920 hit@20 at 1M documents**, having encoded the 1M
store in ~9 min (543 µs/doc) vs ~30 min for the batched teacher — implying the two-stage serve shape: compact
shortlist@20 + dense re-rank. **(2) Injection accessibility + multi-hop chase** (`exp_m3_injection.{log,json}`; 800
questions): once a fact is injected into the store, the question reaches it top-5 at **0.616 = 0.95× teacher** (0.649) —
but both encoders sit on the same **surface-form cliff by hop** (table 0.995/0.289/0.226 across 1/2/3 hops; teacher
1.00/0.34/0.30): intermediate-hop facts share only ~55% of their words with the question (vs ~75% at hop 1), so the
cliff is the task's, not the table's. Robust at scale: +100k distractors moves inject hit@1 0.481→0.480. **The notable
finding: iterative chase** — n_hop rounds of top-1 with the retrieved text appended to the query, still pure table
lookups — lifts full-chain recovery 0.665→**0.733** (3-hop 0.195→**0.371**, paired ~1.9× on the same questions, on a
budget of only n_hop picks vs single-shot's top-20), while **the identical protocol does nothing for MiniLM**
(0.719→0.720), leaving the chased table at parity with the chased teacher. Query expansion is the natural multi-hop
mechanism for a bag-of-words encoder — each retrieved fact donates the bridging entity words the question lacks — where
appending text to a sentence-encoder query dilutes as much as it informs. Honest bounds: per-hop table-vs-teacher
differences are within noise at n=159 (3-hop); the load-bearing claims are the paired improvement and the protocol
asymmetry. One chase bug was caught by the N=100k run itself (query expansion indexed target texts only; a retrieved
distractor crashed it) and fixed so expansion uses whatever row was actually retrieved — a wrong pick poisoning the
expanded query is part of what the metric measures. Method + full numbers drafted as a private working paper:
`docs/papers/2026-08-07-qfhrr-semantic-word-tables.md`.

**H-F2/H-B3 CoT pipeline M1 (2026-08-07/08) — built, live-VM-proven, and the kill criterion returned an INFORMATIVE
PARTIAL FALSIFICATION that narrows the value claim to verified-answer precision.** The cubbyllm→cubelang→bridge→output
chain (spec `docs/superpowers/specs/2026-08-07-cot-pipeline-design.md`, 12-commit SDD on feat/cot-pipeline) is real:
`cubbyllm/reasoning/` (planner grammar, CubeLang chain-program builder, walk/repair/verify pipeline with the
claimed-answer invariant asserted at its single `verified=True` site; pure-Python, `__wiring__`-declared, torch-free),
24 tests including a live smoke through the real Rust VM (2-hop bound recoveries ~0.50, absent-role control **0.029** —
the live run corrected a fake-test assumption: cosine cleanup returns a *low-similarity nearest symbol* for absent roles
on populated frames, None only for empty frames; the control check follows the spec's below-τ semantics at all four
sites). **The measured verdict** (`exp_m3_cot_pipeline.{log,json}`; 800 eval / 200 disjoint calibration): CoT **0.390**
overall vs retrieval-only 0.486 and chase-only **0.724** — clause 1 FAILS (beats retrieval-only at 2-hop **0.457 vs
0.023**; exact tie at 3-hop 15/159 = 0.094) while clauses 2–4 PASS: **claimed-answer precision 0.994, zero sub-τ
verified results, control pass 341/341**. The honest reading, pre-committed by the spec: chase-only wins raw accuracy
because it never refuses (337/800 questions fall outside the M1 question grammar and CoT declines them; chase also
"wins" some questions CoT correctly refuses); **when this pipeline claims, it is right — the auditable-trace property
free-form CoT cannot offer, measured not asserted.** Honesty caveats recorded in the artifacts themselves: τ_vm's
calibration DEGENERATED to a floor (the clean 1,241-fact store produced zero incorrect calibration hops → Youden had no
negatives, AUC 0.0; the 0.994 precision is post-hoc evidence doing calibration's job — self-reported via
`tau_vm_calibration_degenerate` in the json); **the spec's 100k-distractor second arm was descoped between spec and plan
without a recorded reason (a process miss caught by the final whole-branch review) — now explicitly a follow-up, and the
likely cure for the degenerate calibration (a distractor-laden store manufactures genuine negatives)**; the corpus is
NOT pure ASCII (30/800 questions) and cubelang's lexer mojibakes non-ASCII fillers (corruption can only cause
verify-*fail*, never a false claim — the honest direction) — the cross-repo lexer fix is queued. Follow-ups, in value
order: (1) question-grammar coverage (337/800 unparseable is the coverage ceiling); (2) the distractor arm + real
negative-example calibration; (3) cubelang UTF-8 lexer fix; (4) trunk-SFT program generation (deferred by design).
**Eval wave v2 (2026-08-13) — OVERALL PASS ×2, and a calibration finding worth its own line.** Grammar v2 (337→37
unparseable) + the corrected calibration turned the M1 falsification into the campaign's first full PASS on both arms
(`exp_m3_cot_pipeline_v2{,_100k}.{log,json}`, commit ab23f85, reviewer-reproduced): verified coverage **0.646**,
claimed-answer precision **0.9923**, by-hop 0.865/0.566/0.220 (beats retrieval-only at 2-hop AND 3-hop), planted-fault
catch **1.000** on all four classes, organic-confusable catch 0.77–0.80, control 528/528, **identical CoT numbers at 0
and 100k distractors**. The finding: **an execution engine's similarity measures binding FIDELITY, not truth** —
semantic near-miss negatives cannot calibrate it (round 2 pinned τ at 1.0 and zeroed multi-hop coverage at perfect
precision); the correct calibration is **frame-size-conditional floors from correct-recovery quantiles**, which landed
at 1.000/0.474/0.220 ≈ the 1/n bundle-dilution the algebra predicts — threshold design and representation theory now
agree. The 4 verified-but-wrong answers are **corpus-dedup collisions** (typo'd surface forms collapse distinct chains'
facts into one store entry; hits baselines identically; not a verify leak). Harvest per
`docs/schemas/cot-harvest-schema.md`: 800 records/run, 517 verified pairs, 1 DPO pair (low yield → repair-heuristic
follow-up). **Corpus data-quality note (2026-08-13, user-caught):** the .pq's entity orthography is systematically noisy
— "Huamn" for "Human" in **568 answers / 1,098 facts vs only 29/61 spelled correctly (~95% typo'd)**, "united stated"
family at 387 answers (probe: `Huamn is the instance of Hierocles (author of Synecdemus)`). Typos are chain-consistent,
so exact-match scoring stays internally coherent — but every accuracy figure on this corpus measures **chain-recovery
fidelity against the corpus's own ground truth, not real-world spelling**, harvested SFT tuples inherit the orthography,
and any user-facing answer surface needs a normalization pass in front of it. **Counterfactual-neighborhood harvest
(2026-08-28) — the Q23 "regretted discard" built and run.** `exp_m3_cot_pipeline.py` now re-plants the four fault
classes on every verified chain's *own* accepted triples, recovers each corrupted hop through the VM at the deployed
floor, and writes the per-hop outcome into the harvest record's `counterfactuals[]` (schema section added; fake-VM unit
pins `validation/test_cot_counterfactuals.py`, 6 tests). Run `_v3cf` (800/200, no distractors;
`exp_m3_cot_pipeline_v3cf.{log,json}`, `cot_harvest_v3cf.jsonl`, same 0.646 / 0.9923 / OVERALL PASS as v2): **517/517
verified chains, 1,899 corrupted hops (598 wrong-entity, 233 wrong-relation, 662 inverted-direction, 406
wrong-hop-order; 189 non-ASCII hops excluded), 1,899 caught, 0 escaped**, ~262 ms/chain, 1,899 VM calls — the harvest
now refills every planted-fault stratum at ~3.7 hard negatives per verified answer for free. Two honest readings: (1)
**every catch was by symbol mismatch, none by the floor** — the caught recoveries' similarity is median 0.503 / q90 1.00
(a wrong filler binds and recovers *faithfully*), so on this VM the string-match check is 100% of the defense against
content faults and the manufactured negatives carry no τ information; they are hard negatives for a *successor* verifier
(learned / bundled-superposition cleanup), not for the current floors — fidelity ≠ truth, now quantified on 1,899 hops;
(2) the first pass (`*_v3cf_run1.*`, kept) reported **3 "escapes", all a generator bug** — `wrong_relation` on 3-hop
chains whose hop-0 and hop-1 objects coincide compared a recovery against itself (a no-op fault the hop-order branch
already skipped); fixed in `_fault_instances` + regression test, and the rerun's 0 escapes is the number that matches
the shipped code. Not built: Qwen's k-mutation ladder (block-level flips can't cross the symbolic bridge — needs a
VM-side op). Origin: N4 of `docs/research/2026-08-28-oracle-competition-scored.md`.

**Update 2026-09-03 — repair budget 3 → 1, measured lossless.** The GoT challenge's replay checks on
`validation/logs/cot_harvest_v3cf.jsonl` (800 questions) showed the three-repair budget binds on every failure (246/246
used all three, all with zero hops verified) while no verified chain ever needed more than one repair (514 used none, 3
used one). The re-run with `--max-repairs 1` (`validation/logs/exp_m3_cot_pipeline_rb1.{json,log}`) reproduces the M1
numbers exactly — 517 verified, 513 correct, claimed-answer precision 0.9923, absent-role control 528/528 — at 45.9 ms
per question for the CoT arm vs 120.9 ms. `pipeline.answer()`'s default and the serve path are now 1. Same source: the
pipeline's failures are retrieval exhaustion (246 of 283), not candidate choice (only 19.8% of hops have a second
candidate above `tau_ret`) and not the verifier (all 1,899 planted counterfactual faults caught by `symbol_mismatch`);
the next experiment is a lower `tau_ret` with the VM deciding per hop. Scoring doc:
`docs/research/2026-09-03-got-challenge-scored.md`.

**H-F2 M3 teacher ladder + vocab scaling + published-benchmark anchor (2026-08-07, queue item 4).** The
paper-prerequisite baselines, run on the exact recorded protocols (same seeds). **(1) Released model2vec/potion
head-to-head** (`exp_m3_potion_baseline.{log,json}`): potion-base-8M beats the MiniLM-taught v1 table on 42-domain
routing (0.7223 vs 0.5946 macro) but LOSES NYT year placement (0.0443 vs 0.0638 exact); distilling potion itself into
block space through the identical recipe recovers routing to 0.6464 — the cosine-exact projection means the block target
costs nothing, attributing the gap to teacher strength + vocab coverage; the 0.5 signal blend is measured optimal vs
0/0.25 on both screens (`exp_m3_potion_blend0.log`). **(2) Teacher ladder** (`exp_m3_teacher_ladder.{log,json}`,
user-driven): **raw bge-m3 (GGUF Q8, llama-cpp) is a NEGATIVE teacher** — 0.566 routing, below MiniLM, despite being the
far stronger sentence embedder (single words are not its input regime; `exp_m3_bgem3_block_arm.log`); its
Tokenlearn-compiled form (potion-multilingual-128M, whose own teacher IS bge-m3) jumps to 0.622 — static compilation is
the load-bearing step; winner **potion-retrieval-32M: routing 0.680, NYT 0.080 exact / MAE med 15, axiom screen 0.3132**
— strictly dominating v1/v2 on every screen → built as `fastword_table_v3.npz` (a builder meta-stamp bug — hardcoded
teacher id, ignored --signal-blend — was caught during v2 validation and fixed, mowm `2a298a2`). **(3) Vocab scaling**
(same teacher; 60k/100k builds + screens): 30k→0.680, **60k→0.699 (= 97% of native potion's 0.722)**, 100k→0.705 (inside
±0.014 noise, NYT/axiom flat) — coverage was most of the residual gap and saturates at 60k; **`fastword_table_v4.npz`
(60,151 words, 1.1GB, ~33µs) is the production table**; the 100k candidate is kept as the saturation datapoint. **(4)
BEIR dbpedia-entity, full 4.64M corpus** (`exp_m3_beir_dbpedia.{log,json}`; graded nDCG@10, 400 test queries, all 43,515
judged docs present): **the method's measured boundary** — v1 table two-stage 0.081 (compact-alone 0.040; shortlist
recall@100 0.143 is the bottleneck), released potion-8M dense 0.224, published BM25 ~0.313 (literature anchor,
unverified) — open-vocabulary web ENTITY search defeats a corpus-DF word list (tail entity names all hash-fallback), and
even potion loses to lexical BM25 there; caveat: the run predates v3/v4 (used v1). Recorded in the paper §5.8 +
negatives; the table's validated domain is routing/temporal/known-corpus retrieval, not open-web entity search.

**H-F2 M3 temporal worlds + causal events + inter-world query (2026-08-07).** The user's design frame: the world was not
the same from one era or one year to the next — each era and each year is a world (months deliberately too granular;
kept as metadata only), worlds need causal events connecting them, and a world missing an answer must be able to query
another world. Two screens, both with measured noise nulls and MiniLM anchors. **(1) Curated events**
(`exp_m3_temporal_causal.{log,json}`; $CUBBY_TRAINING_DATA\temporal\historical\train_augmented.jsonl — 4,000 dated
events, 6 named period-eras, year_start with BC negative, per-event `impact` consequence text; the HUGE_ENTITIES
5,804-book corpus was probed and deferred as the unlabeled scale-up path — extraction works but it holds misfiled
content, e.g. a 1943–46 novel under AncientClassical). **The load-bearing leakage catch:** "augmented" means duplicated
— 3,187/4,000 rows are paraphrase copies of another row's event (1914 holds three Franz Ferdinand assassinations), and
row-level LOO left siblings in every centroid, inflating exact-year to 0.890; **event-level exclusion** (all
same-(title,year) rows out of every world when that event challenges) is the honest protocol — third leakage instance
caught this project, same shape as the PQ retraction. Honest numbers: era routing **0.669 macro (ref 0.713 = 0.94×; null
0.163)**; flat year routing over 239 year-worlds **0.735 exact / 0.899 ±50y / MAE median 0 = 0.98× teacher** (0.747;
null: 0.005 exact, MAE med ~730y); the digit-masked arm is **identical** (0.735) — placement is content-based, not
date-reading. **Hierarchy is a real trade, not a free win:** era→year routing drops to 0.542 exact (teacher identically:
0.562) at 59 vs 239 comparisons — the era gate's errors strand queries, which is exactly what inter-world query repairs.
**Causal:** event→impact retrieval among 4,000 confusables (any-sibling-positive, the fair form under duplication)
**0.337 hit@1 / 0.631 hit@5** (ref 0.448/0.757 = 0.75×; chance 0.00025); impacts routed to year-worlds land **later than
their event 55.2% vs matched null 44.6%** in both encoders — causality measurably points forward; and the year-centroid
similarity **gradient decays monotonically** with temporal distance (0.342 at Δ<10y → 0.251 at Δ>500y) — "the world was
not the same" is in the representation. **Inter-world query (delegation):** worlds ranked by self-excluded centroid; the
consulted world answers only if its best member clears τ, else the query passes on (a fourth leakage variant was caught
here: full-centroid ranking let the challenge's own vector nudge world choice — on NYT, where year-centroids sit at
0.86–0.92 mutual cosine, that 1/100 nudge alone was worth 0.064→0.372 exact). Fixed protocol at τ=0.6: **matches
exhaustive flat search's answer-found rate (0.554) at 63 vs 3,783 member comparisons (60× cheaper), 2% of queries
delegated**, yr-exact 0.936 vs flat's 0.942. **(2) NYT hard mode** (`exp_m3_nyt_years.{log,json}`;
$CUBBY_TRAINING_DATA\temporal\nyt_data — 294 YEAR_MONTH archive files, 1852–2024; 14,700 deduped abstracts, 100/year,
147 year-worlds; abstracts include genuinely timeless content, the real-text ceiling): **the table beats the teacher on
every year metric** — exact 0.064 vs 0.046, ±10y 0.354 vs 0.338 (3× the 0.120 null), **MAE median 19y vs 21y** (null
50.7) — dating real newspaper text is a surface-vocabulary task (names, spellings, topics), the third measured instance
of surface-beats-semantics after routing-parity and chase; the fine gradient is again monotone (0.918 at Δ<2y → 0.855 at
50–100y); and delegation's τ needs per-corpus calibration exactly as every τ before it (curated best-member scores run
high — 2% delegate at τ=0.6; NYT scores run low — 84% delegate at τ=0.4, degenerating toward flat search for only +0.04
±50y over trusting the routed world). Architecture reading: centroid-routed worlds + confidence-gated peer query is
validated as the serve shape; the (event, impact, Δyear) pairs are the natural transition-training data for year-worlds'
HYLAs (not yet wired). **Wired (mowm `ff57a5f` on feat/world-query, suite 762 green):** `World.best_member` (the "do I
actually hold this?" primitive), `MoWMRouter.query_worlds`+`WorldAnswer` (worlds ranked by axiom-centroid; first world
whose best member clears `tau_answer` answers; exhausted → global best seen; router mediates every hop, worlds never
reference each other), and `CubbyBridge` tier (b2) (centroid winner short of `tau_answer` → redirect to the world that
cleared it; route-vs-spawn untouched; `tau_answer=None` disables). 8 new tests pin the delegation geometry — a near-miss
world outranks the true holder on centroids and the query still lands on the holder — plus the no-spawn guarantee.

**H-B4 — Multi-factor unbinding needs chained F=2 hops, not one large resonator solve.** This is a hard capacity
ceiling, not an implementation weakness: flat F=5 factorization is mathematically proven impossible at D=10000 (P_cap ≈
0.64·D^(1/(F-1))), and cubby-concepts ships a passing regression test (`test_resonator_quadrillion_is_beyond_capacity`)
asserting exactly that. Status: VERIFIED (the capacity math and the test both exist and pass). The demonstrated
workaround — `vsa/graph.py`'s `SequentialHyperGraph`, chaining simple F=2 binds through an SDM, verified 100% accuracy
from depth 1 to depth 20 with zero error accumulation — is how the design reaches combinatorial reach from polynomial
storage. **Validate/apply directly:** any CubbyLLM binding-head design that needs more than ~3 simultaneously-bound
factors must budget for chained decomposition from the start; this isn't optional future-proofing, it's a known
mathematical limit.

**Real-workload datum + a correction (2026-08-02), from cubemind's I-RAVEN crosstalk note (dated 2026-03-19).** This is
the first measurement of what bundle load the VM channel actually needs, and it is a task metric rather than a probe:
I-RAVEN sits at 86.1% overall, with single-entity configs (Center, L-R, U-D, O-IC) at **96-100%** but multi-entity grids
(2x2, 3x3, O-IG) stuck at **67-77%**. The split is capacity — a grid panel bundles 9 object-position pairs into one
vector and unbinding SNR falls below the discrimination threshold. Three things follow.

(1) **The note's proposed "expand to k=32, l=256 (d=8192)" should instead be the shape already decided in H-B5: k=80 x
l=128 = d=10240.** cubemind runs I-RAVEN at k=16 x l=128 = d=2048, i.e. 5x under the standardized algebra. Moving to the
decided shape keeps l=128 (block length unchanged, so codebooks and shaders stay compatible) and only adds blocks,
whereas 32x256 invents a third shape and breaks the "one algebra across model + world-arena" property that was the
entire point of H-B5.

(2) **The note's "exponentially more capacity" is wrong and should not be planned against.** Bundling crosstalk falls as
~1/sqrt(D) — this project already measured the law in H-C7 (max-crosstalk grows as sqrt(2 ln V / D)). So 2048 -> 8192 is
4x D for **2x** SNR, and 2048 -> 10240 is 5x D for ~2.2x. Real and worth doing, roughly linear in D — not exponential.

(3) **Hierarchical bundling is not a new fix; it is H-B4, and it only works with cleanup between levels.** Bundling
objects into rows and then rows into a panel is exactly the chained F=2 decomposition described above, already VERIFIED
at 100% accuracy from depth 1 to depth 20 in `cubby-concepts/vsa/graph.py`'s `SequentialHyperGraph`. The critical detail
the note omits: that result holds *because* the chain passes through an **SDM cleanup at each hop**, which resets
accumulated noise. Without a cleanup step between levels, hierarchical bundling merely redistributes the same total
crosstalk across two hops and buys far less than it appears to. Port the SequentialHyperGraph pattern, not just the
two-level idea.

Also note the regime: I-RAVEN objects share attribute vocabularies (shape/colour/size), so the fillers are
**correlated**, not quasi-orthogonal. H-G2 measured that penalty directly — real keys ran 3.1x more correlated than
random bipolar and cost every candidate 2-3x recall. Capacity here is therefore worse than random-vector math predicts,
which is consistent with 9 items straining a space where H-B6's probe held ~97% retrieval at n=16 on quasi-orthogonal
trunk features.

**A caution carried over from H-B6, aimed at the note's closing claim.** It states that a CNN -> blockSoftmax ->
bind/unbind pipeline "can learn hierarchical bundling automatically via backprop" because "the VSA algebra is its own
adjoint, so the gradient naturally discovers the right decomposition." The adjoint half is true and elegant: the adjoint
of bind-by-b is correlation-by-b, which is unbind. The inference from it is exactly the assumption H-B6 falsified. A
differentiable VSA pipeline has a degenerate solution available — collapse the encodings so everything scores well on
the objective while losing discrimination — and it will take it: optimizing `binding_aux_loss` alone drove filler-filler
cosine 0.006 -> 0.974 and retrieval 96% -> chance, while the loss looked like clean progress the whole way. I-RAVEN
accuracy is a real task metric and would eventually catch that, unlike the proxy H-B6 used. But "did hierarchical
decomposition actually emerge" needs its own discrimination check; do not infer it from the loss curve.

**H-B5 — Added 2026-07-23. `grilly.experimental.vsa.ops` may already be the production VSA substrate this whole stack
shares, and should be checked before choosing between H-B1 and H-B2.** Discovered via
`$CUBBY_TRAINING_DATA\regenerate_svc_targets.py` (read in full) — a real, currently-used pipeline that builds
BLAKE3-hashed bipolar hypervectors at D=10,240 (80 blocks × 128 bits) via `grilly.experimental.vsa.ops.BinaryOps` and
`grilly.utils.stable_hash`, using role-filler binding (subject/verb/context bound to fixed role vectors named
`cubemind:role:subject/verb/context`, then bundled) to generate real training targets from actual
conversation/instruction data (491,367 records; three output variants — baseline, BLAKE3, and a 4th-role
emotion-augmented version — totaling roughly 33GB). D=10,240 is the exact dimensionality of cubby-lm's `VSABindingHead`
codebook and matches cubemind's world-arena block-codes (k=80×l=128=10,240); the role-vector naming
(`cubemind:role:...`) ties it explicitly to the cubemind lineage. This is very unlikely to be coincidental. `grilly`
(the C++/Vulkan GPU backend sibling repo, at `$CUBBY_GITHUB\grilly` per earlier session findings) has not yet been
surveyed in this project. Status: SPECULATIVE that this is the same substrate as the binding head (not yet confirmed by
reading `grilly`'s actual source) — but unlike H-B1 (cubby-concepts) and H-B2 (GrillCheese's standalone experiments),
this is the only VSA candidate found so far that's already wired into live production data generation, not just a
tested-but-standalone prototype. **Validate first, before committing engineering time to H-B1 or H-B2:** read `grilly`'s
`experimental/vsa/ops.py` (or equivalent) directly, and check whether it's the actual mechanism behind (or a close
relative of) cubby-lm's `VSABindingHead` and cubemind's `BlockCodes`. **Kill criterion:** if it turns out to be an
unrelated, narrowly-built one-off for this specific SVC-target pipeline with no real connection to the binding head, it
drops back to a third reference implementation alongside H-B1/H-B2, no more or less privileged than either. If confirmed
connected, it may substantially shortcut Group B's whole "which algebra do we build" question — the open work would
shift to whether its role-filler binding scheme (built for subject/verb/context triples) generalizes to CubbyLLM's
actual token/concept binding needs, not whether the underlying primitives work at all.

**Validation result (2026-07-23 campaign): CONFIRMED CONNECTED — status upgraded from SPECULATIVE.** cubemind's
`BlockCodes` is a thin layered wrapper over grilly: `grilly.backend._bridge.blockcode_bind/unbind/similarity`
(C++/Vulkan, path 1) → `grilly.experimental.vsa.block_ops.BlockCodeOps` (Python GPU, path 2) → numpy (verified in
`cubemind/cubemind/ops/block_codes.py:4-6,35-45,127-187`). grilly's `experimental/vsa/` ships **all three candidate
families**: `BinaryOps` (bipolar elementwise, exact bind/unbind, `hash_to_bipolar` — the BLAKE-hash lineage the binding
head's own docstring names for cross-repo codebook sharing), `HolographicOps` (HRR/FFT = H-B2's algebra), and
`BlockCodeOps` (IBM-NVSA sparse block codes, per-block circular convolution — whose docstring explicitly targets "the
gradient explosion of flat HRR or the zero-gradient problem of bipolar sign()"), plus a resonator. Group B's question
collapses to: **which grilly family does the binding head standardize on, and import-vs-port** — exactly the shortcut
this hypothesis predicted. The generalization question (SVC triples → token/concept binding) remains open as stated.

**H-B6 — Added 2026-07-30. A parameter-free VSA binding auxiliary loss, applied during LM pretraining, shapes the
trunk's hidden states to be VSA-bindable at no measurable language-modeling cost.** Source:
`cubbyllm/model/binding/torch_ops.py` — the **MAP scheme from `cubby-concepts/vsa`** (bind = bipolar elementwise
product, self-inverse; `binding_aux_loss` bundles N=16 trunk hidden states, each bound to a fixed bipolar role key,
unbinds each, and minimizes the cosine reconstruction error), wired into `TrainLoop` behind `bind_weight`
(`cubbyllm/training/loop.py`, computed in fp32 outside the bf16 autocast). Note the deliberate two-channel split: MAP is
used **only for the differentiable aux loss** because bipolar elementwise bind is a plain Hadamard product (already
differentiable, exact single-pair recovery), whereas the decided *production* VM-side algebra is `BlockCodeOps` (H-B5,
parameter-free and non-differentiable). The claim: an auxiliary objective rewarding quasi-orthogonal, high-capacity
representations can be trained *jointly with* next-token CE without the two objectives competing — yielding a trunk
whose features are natively bindable for the VM/world-model channel, "for free." **Validate** by running the aux loss
live in the real assembled `CubbyModel` and watching whether the binding loss falls (trunk becoming bindable) *while*
bpc keeps tracking a no-binding baseline (binding not stealing LM capacity), specifically through the warmup→full-LR
transition where a competing objective would surface. **Kill criterion:** if bpc diverges upward from a no-binding
baseline once the binding gradient is at full strength, the objectives compete — dial the weight back or drop the aux
loss.

**Validation result (2026-07-30, live 2B run — mechanism VERIFIED, "free" PLAUSIBLE).** In the real assembled
`CubbyModel` (D=2048, L=32, MinGRU backbone, hybrid vocab, hardened memory; `bind_weight=0.42`, N=16; B=8×S=768, Colab
single-GPU, bf16 + grad-checkpointing), the binding loss fell from ~0.735 (at an earlier 0.1 diagnostic weight) — 0.57
by step 500 of the 0.42 run — to a **~0.20 plateau across steps 2000–3000** (cosine recovery ~0.80, well past the N=16
random-filler crosstalk floor cos ≈ 1/√16 = 0.25, and *holding* rather than sliding to zero or bouncing back). Crucially
`bpc` fell **in lockstep** (≈2.3 → 1.95 → 1.74 → 1.69 → ~1.5 at steps 500/1000/2000/2500/3000) and did **not** stall or
bump through either the warmup→full-LR transition (step 2000, LR reaching 3e-4) or 1,000 further steps of sustained full
LR — the exact points where a competing objective would declare itself. That co-descent also rules out the one
degenerate way to game the aux loss (collapsing `h` to a low-rank state, which would *raise* bpc, not lower it).
`validation/logs/binding_aux_loss_live_2b_2026-07-30.log`. **VERIFIED:** at real-run (early) scale, a zero-parameter aux
loss makes the trunk VSA-bindable under the MAP scheme. **Only PLAUSIBLE (not yet VERIFIED):** that it is truly *free* —
this rests on "no bpc bump through sustained full LR" (strong circumstantial); the clean A/B (bpc vs the saved
no-binding `ckpt_1` baseline at matched steps) was **not captured numerically**, and the run is at ~step 2500 (~early),
not a completed pass. **Open items this surfaced:** (1) confirm at full-run scale (~40B tokens) that bpc never diverges
from a no-binding baseline; (2) **MAP-bindability is not proven to transfer to `BlockCodeOps`-bindability** — both
families reward quasi-orthogonal high-capacity codes so it likely helps, but the production VM algebra is a different
family and this is untested; (3) "bindable trunk features" has not been shown to make VM-side unbind (H-B3) actually
work on real trunk `h` — aux-probe capacity ≠ usable relational readout downstream.

**KILL CRITERION TRIGGERED (2026-08-02) — H-B6 IS FALSIFIED. The aux loss was Goodharted; do not use it.** The offline
probe (`validation/exp_b6_blockcode_transfer.py`, gate passed: it reproduces the run's own objective at cos 0.840 vs the
logged 0.842) added the metric the training loss never had — **retrieval accuracy**, i.e. can you tell *which* filler
was bound. At n=16 the trained trunk scores **11.5% (chance = 6.25%)** while the **untrained control scores 97.5%**.
Training made the trunk *dramatically worse* at the only thing a binding channel is for. Mechanism, reproduced
synthetically by optimizing h against `binding_aux_loss` alone with no LM objective: cos climbs 0.251→0.863 while
**filler-filler cosine climbs 0.006→0.974** and retrieval falls 96.1%→4.7%. The loss rewards cosine-to-own-filler and
never penalizes cosine-to-*other*-fillers, so **collapsing every hidden state in a sequence onto one direction scores
~0.86**. The trained model sits exactly on that curve. So the earlier "bind 0.735→0.158" was real as a number and
meaningless as a capability: it measured collapse, not bindability. The "+0.088 BlockCode transfer" the probe reported
is transfer of the collapse artifact — the verdict line said POSITIVE because it keyed on cosine delta and ignored
retrieval; that heuristic is wrong and is being changed. **The deeper finding, and the reason no fix is worth building:
an untrained trunk is ALREADY a near-optimal VSA substrate.** Random high-dimensional vectors are quasi-orthogonal by
construction — that is the property hyperdimensional computing is founded on — so retrieval starts at 96–98% and there
is nothing for a binding objective to add. A discriminative (InfoNCE-over-the-bundle) reformulation was built and
tested: it *prevents* the collapse (filler-filler stays ~0.00, retrieval 96–98%) but leaves cosine pinned at the
1/sqrt(n) crosstalk floor, i.e. it buys **nothing over random init**. **CONFIRMED ON A TRAINED CONTROL (2026-08-02,
`exp_b6b_ckpt_ab.py`).** `ckpt_1` — same corpus, same architecture, 7,500 steps, binding loss OFF — scores **96.9%
retrieval** with filler-filler cosine **0.276**, against the binding-trained checkpoint's **11.9%** and **0.934**. Its
binding cosine is 0.243, i.e. exactly the 1/sqrt(16) crosstalk floor, as expected for a trunk that never saw the
objective. **This exonerates the architecture**: the theta=f(c) memory layer, the MinGRU trunk and the hybrid vocab all
preserve retrieval fine — the aux loss alone causes the collapse. It also gives the healthy baseline for real (not
random) features: modest within-sequence correlation ~0.28 is harmless; it is the jump to ~0.93 that destroys
discrimination. The language-modelling half of the A/B is **permanently unanswerable from these artifacts** — the
checkpoints are 20,000 vs 7,500 steps and the run overwrote a single checkpoint path, so no matched-step binding
checkpoint exists (held-out bpc 1.0722 vs 1.3924 is dominated by the 12,500 extra steps and attributes nothing). Moot
for the decision, since the loss damages the channel it was meant to build. **Action: set `bind_weight=0` for the DDP
run.** Everything below this line describes the falsified configuration and is retained as the record of how it looked
before the retrieval metric existed. `validation/logs/exp_b6_blockcode_transfer.log`.

**Run completed (2026-08-01).** The same run finished its full **20,000-step** cosine schedule (LR decayed to the 3e-5
floor; ~123M tokens, ~3.9h wall, peak 92.5GB). Over the complete run `bind` did **not** stop at the 0.20 plateau — it
kept slowly descending to **~0.158** (cos recovery ~0.84) while `bpc` improved to **~1.1**, the two curves tracking
together the entire way with no divergence at any point. So the "free" evidence now spans a *complete* schedule, not
just the early steps. Two caveats keep it PLAUSIBLE rather than VERIFIED: it's a small token budget (a 2B trunk at ~123M
tokens is heavily undertrained — the real test is the 40B-token DDP run), and the numeric `ckpt_1` A/B still wasn't
captured. Generations at step 20000 show genuine domain acquisition — coherent multi-sentence legal prose, coding-task
instruction prompts, contract-clause QA — where step 3000 was still title-stub/loop fragments. Log tail appended to the
same file.

**Long-horizon confirmation (2026-08-05) — a 92k-step run with the loss OFF holds the substrate.** The H-D5
episodic-memory run (`CB_MEM_EVERY=2`, checkpoint `hd5_mem21.pt`; 151M-param Chinchilla-optimal hybrid backbone (2B is
the target arch, not this run's size) + hardened θ=f(c), Colab single-GPU) ran to **step 92,000** of a planned 100k
(stopped early — out of credits) at **`bind_weight=0`**, i.e. the falsified aux loss OFF, per the action above.
Confirmed two independent ways: `train_colab.py:95` hard-defaults `CB_BIND_W=0` behind a `LEAVE AT 0` guard, and the
step log prints no `bind` field (it only appears when `BIND_W>0`, `train_colab.py:541`) — the pasted log has none. With
the loss off, the `representation_health` retrieval probe (n=16) drifted from ~98% early to a **stable ~77% (ff
+0.42–0.47)** by step 92k — modest erosion under a near-complete schedule of *joint* LM+memory training, **not** the
collapse the aux loss itself produced (11.5%), and ~12× above chance (6.25%); bpc settled ~0.90–0.94 (ppl ~20–24) with
coherent multi-domain generations. This is the long-horizon evidence H-B6's ≤20k-step runs never had: left alone, the
trunk stays a usable VSA substrate across ~92k steps of real training — no binding objective needed, exactly as the
falsification concluded. **The one branch this leaves untested** (explicitly *not* a revival of the MAP aux loss):
whether the *discriminative* InfoNCE-over-the-bundle reformulation — which at short scale prevented collapse but bought
nothing over random init — would *hold* retrieval near its ~97% init through long training instead of letting it drift
to ~77%. A 77% channel is healthy, so this is low priority, but it is the sole open question the run surfaces.

**H-B7 — Added 2026-08-02. The I-RAVEN numbers are NOT a VSA result and must not be presented as one.** Status:
**VERIFIED by reading the benchmark source.** `cubemind/benchmarks/iraven.py` (1609 lines) produces the reported 86.3%
baseline / 90.3% after position-aware scoring. Provenance, precisely: `evaluate_problem_dataset` (L617) parses
`problem["metadata"]` XML at L644-645 and `parse_panel_entities` (L312) reads **ground-truth
`Type`/`Size`/`Color`/`Angle` integers straight out of the dataset annotations** — there is no perception. The dominant
signal is `cubemind.reasoning.rule_detectors.score_candidates` (called L677, L691), deterministic integer-domain logic.
**VSA participates only as a 0.5-weighted tiebreaker** (`comp_scores[i] += dist_scores[i] * 0.5`, L709-712) inside a
bare `try/except: pass` (L700, L713-714), so it can silently contribute nothing. The headline +14.5pp grid improvement
(67.5% -> 82.0%) came from `_score_position_rules` (L932) extracting entity **bbox** fingerprints from the same XML —
also not VSA. And `benchmarks/iravenx.py` (740 lines, the I-RAVEN-X 100% OOD result) **imports no VSA at all**; its
imports are `argparse, json, logging, sys, time, pathlib`.

**Consequences, in order of importance.** (1) Any public or fundraising claim of the form "our VSA reasoning layer
scores 86-90% on I-RAVEN" is **not supported by this code**. The defensible statement is: *a symbolic rule-detector
achieves 90.3% on I-RAVEN given ground-truth attribute annotations.* That is a legitimate published-ablation
configuration, but perception-based systems solve I-RAVEN from **pixels**, so quoting this number against theirs without
the qualifier would misrepresent it, and `parse_panel_entities` is about ten minutes of reading away for any reviewer
doing diligence. (2) **Fixing VSA bundling capacity will not move this benchmark much** — an earlier estimate in this
project that hierarchical bundling plus a larger d would take 86% to 92-95% was wrong, because it assumed VSA drove the
score. It does not. (3) What *would* make it a genuine VSA result is the differentiable perception frontend — CNN ->
blockSoftmax -> bind/unbind. That code exists at `cubemind/_archive/perception/cnn_encoder.py` (real grilly Conv2d GEMM
fwd/bwd, `_block_softmax` L105) but is **unreachable**: its only live consumer `benchmarks/iravenx_image.py` L32 imports
`cubemind.perception.cnn_encoder`, which does not exist (verified `ModuleNotFoundError`), and cubemind's own
`docs/papers/ablation-notes.md` L164-165 files it under **"Planned Future Ablations, A6."**

**Capacity context for whoever builds A6.** I-RAVEN runs at **k=8, l=64 (D=512)** (`iraven.py` L1298-1301), not the repo
default k=80/l=128/D=10240 (`core/constants.py` L20-22). `encode_panel_nvsa` (L279) superposes **9 four-way-bound
entities** into one D=512 code. cubemind's own capacity formula (`docs/architecture/02-vsa-foundations.md` §2.7: `N_max
~ 0.18*D/log(D)`) gives **N_max ~ 15 at D=512** — so a 3x3 grid runs at roughly 60% of a budget that is asserted in
prose and **enforced nowhere in the codebase** (no assert, no counter, no warning). Hierarchical bundling *is*
implemented for the distribution rule (`_score_distribution_rule` L1104-1153: rows bundled L1127-1128, then rows into a
master L1129) but there is **no cleanup between the levels** — only a final `discretize()`, while
`bundle(normalize=True)` renormalizes each block per level and compounds attenuation. That is exactly the failure H-B4
predicts, and the mitigation cubemind planned for it ("A5: NSA-adapted bundling", ablation-notes L161-163) is
unimplemented — grep for `NSA`/`noise_strength` in Python returns zero hits.
