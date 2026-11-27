<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 8. Group F — Structural and organizational hypotheses

**H-F1 — CubbyLLM's layout should mirror cubemind's intended structure, not its actual sprawl.** Both `cubby-lm` and
`cubemind` carry known structural debt (recorded during the earlier architecture-mapping phase): `trunk`/`trunk_torch`
duplication, a ~35GB tracked `sandbox/` directory in cubemind, a `cloned/` vendor directory, an `_archive/` that's still
importable despite being gitignored elsewhere, triplicated routing/kernels/registry code, and a
`model.py`/`models/`/`model` naming collision. "Mirror cubemind's structure" should mean the clean package layout the
docs describe, not a copy of the accumulated cruft. **Validate** by writing CubbyLLM's target package layout as its own
short spec before porting any code, explicitly checking it against each known sprawl pattern above.

**Validation result (2026-07-23 campaign): spec written — `PACKAGE_LAYOUT.md` (status: PROPOSAL).** Every anti-sprawl
rule in it is mapped one-to-one to a named observed failure, including two new rules from this campaign: any number
quoted in docs must link the `validation/logs/` file that produced it, and corpora are referenced by pinned manifest
(H-G3's drift lesson). Remains a draft until the user signs off.

**H-F2 — Both existing Cubby↔CubeMind bridges are too narrow for "automatic specialization" and likely need to become
first-class interfaces.** The `NoveltyToWorldBridge` (Hebbian novelty → `WorldManager.register_specialist`) is a
one-way, duck-typed data bridge. The `CubeLangHead` → `cubelang_bridge.py` path is a subprocess call to a compiled
`cubelang.exe` — a text/file bridge, not a tensor bridge, and `CubeLangHead` is self-labeled "SCAFFOLD." Status: this is
a probable-but-unconfirmed scope item, not yet a validated need. **Validate** by first defining concretely what
"automatic specialization" requires from the environment (what needs to flow from model to world-model and back, and how
often) before committing to redesigning either bridge — it's possible a narrower fix suffices.

**Bridge addendum (2026-08-30): ASK crosses the wire.** The reasoning bridge (`cubbyllm/bridges/cubelang_client.py` ↔
cubelang `run-proto`) now carries the VM's third outcome: a program that grounds several candidates and `ask`s is
reported as `suspended{question, candidates}` (JSON Values) instead of an error, and is resumed **by re-execution**
(`RunRequest.answers`, one per ASK; `VM::resume` still rejects any answer that is not an offered candidate — the trunk
selects, never supplies). cubelang `d969a76` also seeds the SPEC's `IAgent` (think/act/observe) as a checked interface
and fixes the strict verifier, which had classified `ask` as "trace-only" although it executes — the reason no asking
program could pass verify-before-execute. Live tests: `tests/bridges/test_cubelang_ask.py`. This is the mechanism the
stand-in's VM-mediated chat (and the "which one did you mean?" grounding contract, `examples/ground_min.cube`) needs;
the chat program itself is unbuilt.

**Validation result (2026-07-23 campaign): the *need* is now demonstrable rather than assumed, definition still open.**
H0's validated mechanism implies the bridges must carry (a) a context/task embedding in both directions, (b) novelty
events (exists today, one-way only), and (c) generated-parameter or specialist handles (exists nowhere). Neither current
bridge carries (a) or (c). Concrete interface definition belongs after the H-C6 and backbone decisions. **Added
2026-08-02 — DisARM is the estimator this bridge would need.** If the trunk must *emit* discrete hypervectors for the VM
to consume, and that path is to stay end-to-end trainable, something has to carry gradient through the discretization.
H-B5 measured the naive route and it is poor: the hard-sign STE path trains at **19.8%**, which is what turned
block_ops' "zero-gradient problem of bipolar sign()" from assertion into measurement. grilly ships **DisARM**
(`grilly.nn.disarm.disarm_gradient`, antithetic coupling, ~7.5x lower variance than REINFORCE), which targets exactly
that. **Verified 2026-08-02: `nn/disarm.py` is real.** But note `docs/advanced/hdc-ops.md` is stale — the
`disarm-*.glsl` shaders it lists do not exist in grilly's `shaders/`, along with three other `hdc-*.glsl` files it
names. So DisARM is a CPU/Python path today, not GPU-backed; do not plan around shader performance. Scope it carefully
after H-B6: this is NOT for the binding head, which is parameter-free and needs no gradients, and NOT for training the
trunk to be bindable, which H-B6 falsified. It applies only to a differentiable model->VM emission path, if the bridge
turns out to require one.
