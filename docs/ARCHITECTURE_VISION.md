# CubbyLLM — Architecture Vision

**Captured:** 2026-08-05, from an extended design session. **Status: north-star direction, not settled architecture.** This complements — does not replace — `CUBBYLLM_HYPOTHESES.md` (the *validated* decisions) and `docs/superpowers/specs/` (the *concrete* cycles). Read those for what's proven; read this for where it's all pointing. Calibration is explicit throughout: **[exists]** = code is there and works (owner-reported or verified), **[near]** = specced / in flight, **[far]** = on the axis, needs real breakthroughs.

## 0. The one-line thesis

A **safe, self-improving AI system**: a continually-learning trunk (**Cubby**) plus a verified, capability-based reasoning VM (**CubeLang**) — an *OS for AI* — where the system can safely **execute, extend, and repair itself**. The tech shape of a "living EVM/blockchain" (verified-before-execution, sandboxed, deterministic, evolvable), applied to AI. **No financial** — only the safety/verification concept.

## 1. The three parts

- **Cubby (the trunk) — the intelligence.** The base model + memory + specialization. Continual / real-time learning without catastrophic forgetting (H0, θ=f(c)). Efficient by design (bounded state, O(1) decode). *This repo (`cubbyllm/`).*
- **CubeLang (the VM) — the safe execution substrate.** The reasoning engine and the place programs run: verified-before-execution, tamper-proof core, capability-based. *`cubelang` repo (Rust).*
- **cubemind — the environment/substrate.** Hosts the VM, the WorldManager/arena, the brain modules and bridges. *`cubemind` repo.*

Split of concerns: **Cubby learns; CubeLang executes safely; cubemind is the body/world they live in.**

## 2. CubeLang as an OS for AI (the verified capability VM)

- **Verified before executing — deny-by-default.** The verifier *whitelists* known-good and rejects everything else (`--strict`; the audit found and we're fixing a blacklist hole where it silently accepted typos). Crucially: **verify *containment*, not *correctness*** — you can't prove an arbitrary (LM-generated) program correct, but you *can* prove it can't escape the sandbox or touch the protected core. That's the only version of "safe" that's achievable, and it's the target.
- **Capability model — tamper-proof core, extend by permission.** Core helpers and standard interfaces live *in the VM* (Rust), like EVM **precompiles** — versioned, name-addressed, unmodifiable: `use vsa;`, `use isolver;`. Your own external code is `import`ed from files (modularity, still verified). Programs may `override` a core function **only** where it's explicitly marked overridable. Nothing a program does can break the core.
- **The through-line:** deny-by-default at every layer (§7).
- Design/plan in flight: `docs/superpowers/specs/2026-08-05-cubelang-foundation-design.md` **[near]**. VM already does real VSA bind/unbind + cosine cleanup (`cubelang/src/vm/engine.rs`) **[exists]**.

## 3. The security layer — an immune system

- **`net` capability — safe external ingestion.** All egress goes through a trusted, sandboxed gateway: domain allowlist + anti-SSRF (block private/metadata IPs, re-check resolved IPs), sandboxed fetch (isolated, resource-limited), **inert data** (never executed), sanitize, and **provenance + trust tagging** on every byte. **[far]**
- **Guardians / reference monitors at every entry & exit.** Complete mediation: a tamper-proof checkpoint at every boundary (ingress, egress, codegen→execute, memory/learning write). *Innate* (static, fast, hard rules) + *adaptive* (learned, pattern-recognizing) — an immune system. **The one rule:** the intelligent layer may only ever *restrict, never grant* — it can be fooled, so it sits *on top of* hard static containment; a fooled guardian degrades to "containment still holds," never "breach." Tiered by risk (full guardians on dangerous crossings; cheap static checks everywhere). **[far]**
- **The learning-gate (Cubby-side).** Reading untrusted data ≠ learning from it. Model updates from live sources are the highest risk (poisoning) → strictest gate: vetted sources, anomaly detection, rollback. **[far]**

## 4. Self-modification & the codegen specialist

- **Programs come from two sources:** a library of pre-made CubeLang, plus a **separate codegen specialist model** (not trunk0) that generates the rest on demand. The language is designed to be **LM-writable and human-readable** — helpers hide the esoteric runtimes so generated programs read naturally. **[far]** (the model), the language work is **[near]**.
- **Self-repair as a function call:** `override`-with-permission + versioned helpers = patch flaws within safe bounds, core intact. The mechanism exists in the foundation design; the *autonomy* is far.
- **[far]** self-generating virtual processors — on the same axis (a CubeLang program is already a computational unit), needs breakthroughs.

## 5. The Brain SDK — standardized cortex/adapter contracts

The extensibility layer: adding a capability should be "implement the contract," not "reinvent the plumbing." **Parts already exist and work.**

- **Cortex = a pluggable module** (vision, language, memory, reasoning). **The base contract (the "base rules"):** identity + typed I/O (so cortexes compose) + **executor** (`run(input, state) → (output, new_state)`) + **state** (bounded, checkpointable) + **lifecycle** (create → activate → grow/prune → save/load → destroy) + **declared capabilities** (mediated through the guardian boundaries). **Architecture-agnostic** — SNN / dense / hybrid / VSA is the cortex's own internal choice, offered as SDK building blocks, never mandated.
- **Adapter = a lightweight specialization overlay** on a cortex (θ=f(c) / LoRA-style) — the "specialization, automatic" (H0) mechanism, produced by the codegen specialist. Cortex vs adapter are two granularities.
- **The SDK API** (owner's sketch): `b = Brain("cubemind")` → `b.create_cortex("vision", kind=...)` → `cortex.add_layer/set_executor/on_lifecycle` → `cortex.attach_adapter(...)` → `b.run(input)`. Routing/composition = the WorldManager.
- **You already have cortexes** — they each implement an *ad-hoc* version of this contract; the SDK's job is to name the one shared contract:
  - the **backbone** (`cubbyllm/model/backbone/hybrid.py`) already exposes the executor: `forward` / `step(x_t, state) → (out, new_state)`, bounded state — the *language cortex*. **[exists]**
  - the **episodic memory** (`cubbyllm/model/recall/`) already has `state_dict`/`load_state_dict` (persist) + a read/write executor — the *memory cortex*. **[exists]**
  - a **vision/perception cortex with a hormonal system + SNN**, with **multiple demos** — owner-reported **[exists]**; real code at `cubemind/cubemind/modules/live_brain.py`, `cubemind/cubemind/perception/`, benchmarks `cubemind/benchmarks/qc_perception_eval.py` + `qc_end_to_end_smoke.py` + `benchmarks/results/qc_perception.md`, tests `cubemind/tests/perception/qc/`. (Hormonal/SNN specifics to be documented precisely when this pillar is specced.)
  - **Quality-Control (QC)** functionality with its CubeLang program — `cubelang/examples/qc_decision.cube` (+ `.cubebin`) **[exists]**.
  - growth/lifecycle seed: `cubemind/model/cubby/hebbian.py`; routing/composition: `cubemind/cubemind/execution/world_manager.py`. **[exists]**

## 5.5 The affective layer — hormonal modulation, personality, empathy

Cross-cutting and **core to the project's lineage** (AURA → emotional-intelligence integration → here), not a feature. A **hormonal system** — slow-varying neuromodulator states (dopamine/serotonin/cortisol/oxytocin analogues) — **modulates every cortex globally**: gating learning rate, attention, exploration-vs-caution, memory consolidation, risk sensitivity. It integrates with θ=f(c): the hormonal state is a slow, affective part of the context `c` that modulates the generated parameters — so "modulated by hormones" *is* the specialization mechanism reading an emotional-state signal.

- **Personality develops, isn't hardcoded.** Personality = the *learned* set-points + dynamics of the hormonal system, shaped by experience via continual learning (H0). Differentiator: GPT-4 is stateless/flat; this has a persistent, *developing* affective state. **[exists]** in cubemind (hormonal + SNN — `cubemind/modules/live_brain.py`, `cubemind/perception/`); developing-over-time is [near].
- **Empathy — the honest mechanism.** Internal emotional states give the substrate to model and resonate with others' states — you empathize better with states you can also represent. The hormonal system is *necessary machinery* for genuine (functional) empathy vs imitated-from-training empathy. Necessary-not-sufficient: the empathy is emergent and *shaped* through interaction — the mechanism enables it, experience grows it (so it's earned and individual, not a canned persona).
- **Discipline (so it's real, not mood labels):** every hormone must do real computational work (define what it gates/scales); the state must be **bounded, observable, homeostatic** — checkpointed, readable, self-regulating, modulating within limits. That both keeps the personality from drifting into bad states AND fits the verifiable/guardian discipline: a mood-modulated system is powerful, so it gets the same deny-by-default containment (can't be pushed into an unsafe state by manipulating its "stress"). Expressive affect, bounded safely.

## 6. Calibration — existing / near / far

- **[exists] / working:** hybrid backbone + episodic memory (this repo); CubeLang VM with real bind/unbind (`cubelang`); the vision/perception cortex + hormonal + SNN + demos + QC benchmarks (`cubemind`); QC CubeLang program (`cubelang/examples/qc_decision.cube`); θ=f(c) + `FrozenSlotRouter`; the WorldManager + Hebbian growth.
- **[near] / specced or in flight:** H0-at-scale gate (`specs/2026-08-05-h0-at-scale-design.md`); the reasoning-bridge slice (`specs/2026-08-05-reasoning-bridge-slice-design.md`); the CubeLang foundation cycle (verification, `use`/`import`, `implements`, `vsa` helper — `specs/2026-08-05-cubelang-foundation-design.md`, executing on `cubelang:feat/foundation`).
- **[far] / needs breakthroughs:** the full guardian mesh + `net` capability + learning-gate; the codegen specialist model; self-generating virtual processors; real-time learning at scale.

## 7. The through-line

**Deny-by-default. Verify containment, not correctness. Tamper-proof core, extend by permission.** The *same* principle recurs at every layer — the `--strict` verifier, the `use`/`override` capabilities, the guardians, the `net` boundary, the cortex contracts. It keeps reappearing because it is the *only* notion of "safe" that holds against inputs you didn't foresee — which is the entire premise of a system that **learns continuously** and **writes its own code**. Everything else is an implementation detail of that one idea.

## 8. Pointers

| Piece | Where |
|---|---|
| Trunk + episodic memory | `cubbyllm/` (this repo) |
| Validated decisions / hypotheses | `CUBBYLLM_HYPOTHESES.md`, `VALIDATION_REPORT.md` |
| Active design cycles | `docs/superpowers/specs/2026-08-05-*` |
| CubeLang VM (bind/unbind, run, strict) | `cubelang/src/vm/engine.rs`, `src/compiler.rs`, `src/main.rs` |
| CubeLang programs incl. QC | `cubelang/examples/` (`qc_decision.cube`, `ground_min`, `ask_min`, …) |
| Brain / cortexes / perception / QC | `cubemind/cubemind/modules/live_brain.py`, `cubemind/cubemind/perception/`, `cubemind/benchmarks/qc_*`, `cubemind/tests/perception/qc/` |
| Routing / world | `cubemind/cubemind/execution/world_manager.py` |
| VSA algebra (Rust encoder) | `opcode-vsa-rs/` |

_This is a living document. As pillars land, move items from [far]/[near] to [exists] with links to the validation, the same way `CUBBYLLM_HYPOTHESES.md` tracks proof._
