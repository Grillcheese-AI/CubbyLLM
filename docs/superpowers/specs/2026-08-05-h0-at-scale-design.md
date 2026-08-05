# H0 at scale — does θ=f(c) resist forgetting on a real trunk?

**Date:** 2026-08-05 · **Status:** design, approved to plan · **Hypothesis:** H0 (central bet), the "Scale H0 past toy scale" item in `TODO.md`

## 1. Why this exists

H0 — context-conditioned parameter generation (θ=f(c)) as the cure for catastrophic
forgetting — is the central bet of the whole project ("specialization, automatic").
It is the one pillar still validated **only at toy scale** (the 2026-07-23 campaign:
`exp_h0_gce` showed a *hardened* θ=f(c) head with near-zero sequential forgetting,
while the *naive* version ended up worse than doing nothing; `exp_a_forgetting` and
its NYT re-run measured the forgetting curves; `exp_h0b_learned_context` found the
context router must be **offline-pretrained and frozen** — an online router recreates
the forgetting).

Two things are untested at scale, and `TODO.md` names them:
- **(a) cost vs #protected-contexts** — how the hardening's compute/memory grows as
  the number of protected contexts rises.
- **(b) scoping inside a real net** — whether the mechanism stays confined to the
  memory pathway once it lives inside a real trunk, or leaks into and corrodes the
  general capability.

This is the **falsifiable gate** before the expensive step (scaling the whole stack
to the 2B trunk). It mirrors the discipline that just worked for the memory rung: a
cheap toy gate (`exp_d5`) *before* the real needle run. If hardened θ=f(c) does not
hold at real-trunk scale, the "specialization, automatic" pillar needs rethinking
**before** any 2B commitment — which is exactly what we want to learn cheaply.

**This is Phase 1 (the gate) only.** Phase 2 — sequential *domain* specialization on
the trained trunk (the product case) — is a separate, later cycle, gated on this
passing (§6).

## 2. Settled design decisions (approved 2026-08-05)

- **Gate-first, then domains.** This spec is the synthetic gate. Domain
  specialization is Phase 2, deferred (§6).
- **Substrate is the real trunk, not a toy.** The experiment runs on the trained
  H-D5 checkpoint via frozen `model.features(x)`, exercising the real
  `MemoryLayer(HyperGenerator, SnapshotHardener)` at real `d_model`, with the
  **offline-frozen** router (H-C4's `domain_head`, `.freeze()`) as the context
  source. Reusing the real trunk is the whole point — it is what makes unknown (b)
  observable. An online/trainable router is explicitly out (H0b).
- **Task: sequential associative recall**, the same family as the toy Zero-Forgetting
  benchmark, but with keys/values drawn from the **real trunk features** rather than
  synthetic random vectors. Store context c₁'s associations, then c₂ … c_N, then
  measure retention of *every earlier* context. Tightest possible forgetting signal.
- **Three arms, reported together (honest testing):** (1) **hardened θ=f(c)** — the
  bet; (2) **naive θ=f(c)** (no `SnapshotHardener`) — the toy's "worse than nothing"
  control; (3) **fixed-state overwrite** — the in-place learner θ=f(c) must beat.
- **Frozen trunk features + θ=f(c) on top** (not joint training) — isolates the
  memory pathway and matches the "specialize a trained base" product shape.
- **Sweep N = {2, 4, 8, 16, 32} contexts** to trace both the forgetting(N) and the
  cost(N) curves in one run.

## 3. Architecture

This is a validation experiment (`validation/exp_h0_scale.py`), not package code —
it scales the toy `exp_h0_gce` / `exp_a_forgetting` machinery onto the real trunk. It
imports the real `cubbyllm` components; it is never imported *by* the package.

### 3.1 Load the substrate
- Load the H-D5 checkpoint; reconstruct `CubbyModel` from its meta (the same path
  `exp_needle_recall` uses). Expose `model.features(x)` (already exists — used by
  `representation_health`) as the frozen `h` the memory reads/writes over.
- Instantiate the context source as `FrozenSlotRouter(...).freeze()` (offline,
  H-C4/H0b). The router maps a context id / task embedding to the slot(s) θ=f(c)
  conditions on.

### 3.2 The three arms (a common harness, one knob)
A single loop `learn_then_measure(arm, N)`:
1. For each context `c` in `1..N`: draw a batch of (key, value) associations from the
   trunk's features on real corpus text; write them via the arm's learner.
   - **hardened**: `MemoryLayer(HyperGenerator, SnapshotHardener())` — θ=f(c) with the
     anti-drift snapshot.
   - **naive**: the same `MemoryLayer` with the hardener disabled (a no-op hardener).
   - **fixed**: one adapted weight overwritten in place per context (the C_ctx=0
     learner — the baseline θ=f(c) must beat).
2. After all N contexts, for each earlier context `i`, re-query its keys and score
   recall@1 of its values → `retention[i]`.

### 3.3 Metrics
- **forgetting(N)** — mean `retention[i]` across `i` after training through all N
  (per arm). Near-1.0 = no forgetting.
- **cost(N)** — wall-clock and peak memory of the hardened arm as N grows (unknown a).
  Recorded per N in the sweep; the shape (linear vs super-linear) is the result.
- **scoping** — held-out general **bpc** and the H-D5 **needle recall** measured
  before vs after the θ=f(c) contexts are written (unknown b). A healthy result:
  both survive; a scoping failure: the general trunk degrades.

## 4. Success / kill criterion

**Pass** (H0 holds at scale): the **hardened** arm keeps `forgetting(N)` near-zero
(retention ≳ toy-scale level) and **clearly above the fixed-state baseline** across
the whole N-sweep; `cost(N)` grows no worse than ~linearly; and **scoping holds**
(general bpc and needle recall survive writing the contexts).

**Kill** (H0 does not hold at scale, stop before 2B): hardened θ=f(c) forgets at
scale, or fails to beat the fixed-state baseline; **or** `cost(N)` grows
super-linearly to impracticality; **or** writing the contexts degrades the general
trunk (scoping fails). Any of these means the "specialization, automatic" pillar
needs rethinking before the 2B run.

The **naive** arm is expected to reproduce the toy's "worse than nothing" result; it
is reported alongside, not hidden — it is the evidence that the *hardening*, not
θ=f(c) alone, is what does the work.

## 5. Testing (this is a validation script)
- **Smoke run** (tiny: N=2, a handful of associations, CPU) confirms the harness runs
  end-to-end and prints all three arms + the three metrics — numbers meaningless,
  only that it executes. Same discipline as `exp_d5`'s Step-2 smoke test.
- **Controls, per the repo standard:** always print the naive and fixed-state arms
  next to the hardened one, and the scoping before/after pair — never the flattering
  number alone.
- Tee the full run to `validation/logs/exp_h0_scale.log` with the env/GPU stamp
  (persist-experiment-logs rule).

## 6. Scope — explicitly deferred (YAGNI)
- **Phase 2: sequential domain specialization** on the trained trunk (legal → code →
  qa → arxiv, real corpus domains from the manifest), measuring retention of earlier
  *domains* + the general base. The product case ("specialization, automatic"). Its
  own spec → plan → build cycle, gated on this Phase-1 gate passing.
- **Any change to the θ=f(c) mechanism.** This gate measures the mechanism as built;
  it does not redesign the hardener or the generator.
- **The 2B-scale run.** Downstream of both phases; not in scope here.

## 7. Placement & naming
`validation/exp_h0_scale.py` (+ `validation/logs/exp_h0_scale.log`). Reuses
`exp_h0_gce` / `exp_a_forgetting` structure; imports real `cubbyllm` components
(`CubbyModel`, `MemoryLayer`, `HyperGenerator`, `SnapshotHardener`,
`FrozenSlotRouter`). Not package code; nothing in `cubbyllm/` imports it. Result
updates the **H0** entry in `CUBBYLLM_HYPOTHESES.md` (toy → scale-tested, pass or
kill) and checks off "Scale H0 past toy scale" in `TODO.md`.

## 8. Open question for the plan
Whether the associative-recall keys are (a) raw trunk features `h` at sampled
positions, or (b) a learned key projection of them. Default (a) — no new learned
parameters, cleanest read of the *stored* θ=f(c) associations. Revisit in the plan if
(a) gives a degenerate (too-easy or too-hard) recall floor.
