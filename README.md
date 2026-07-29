# CubbyLLM

Fast and profitable LLM gen 2 hybrid — general tasks, plus specialization automatic.

## Status

Validated, with a first-pass implementation already running: a full validation campaign (`VALIDATION_REPORT.md`, 2026-07-23) resolved all three structural decisions below, and a first-pass `cubbyllm/` package now exists and trains end to end on a toy corpus. This isn't a finished model — real training data still needs a cleanup pass, and several follow-on questions remain open — but it's well past the planning-only stage. See `CUBBYLLM_HYPOTHESES.md` for the full hypothesis-by-hypothesis record (most entries now carry a dated validation result, not just a claim), `VALIDATION_REPORT.md` for the campaign itself, and `TODO.md` for the working checklist of what's left.

## Where this comes from

CubbyLLM is the redesigned successor to `cubby-lm`, drawing on lessons from `cubby-lm` itself and its environment sibling `cubemind` — but it's a genuinely fresh design, not a fork. Two different axes worth keeping separate: the *package/directory structure* mirrors cubemind's — its intended, documented layout, not the sprawl both sibling repos have actually accumulated (duplicated trunk implementations, an oversized tracked sandbox directory, a stale-but-still-importable archive, and so on — see `CUBBYLLM_HYPOTHESES.md`, Group F, and `PACKAGE_LAYOUT.md` for the concrete spec). The *model architecture* is an entirely different structure than cubby-lm's, built from scratch, including the vocabulary — cubby-lm's frozen trunk shape and 32k tokenizer are reference and lessons-learned, not a constraint CubbyLLM inherits.

## The structural decisions this project is built around — all now resolved

**VSA binding head.** Cubby-lm's binding head turned out to have no unbind mechanism at all — the placeholder dict-lookup everyone (including this project's own earlier architecture map) attributed to it actually lives in cubemind's VM instead. The LM head itself is clean, well-engineered cosine-readout code. Real unbind still needs building from scratch, and the algebra to build it with is decided: `grilly`'s `BlockCodeOps` (NVSA sparse block codes) — `grilly` turned out to be the shared production VSA substrate cubemind's own code already wraps directly, a stronger candidate than either of the two standalone reference implementations the original survey found. See `CUBBYLLM_HYPOTHESES.md` / `VALIDATION_REPORT.md`, Group B.

**Hebbian memory layer.** Cubby-lm's Hebbian/Oja-style memory update drifted and forgot catastrophically under sequential learning — now measured directly (94% forgetting empirically, zero context-sensitivity by construction). The fix is context-conditioned parameter generation, but only in a *hardened* form (a naive version is measurably worse than a frozen baseline) and only when the context comes from a router that's pretrained offline and frozen at inference — a router learned online turns out to forget its own routing, recreating the exact problem it's meant to solve. A Zero-Forgetting Stability Benchmark that had only ever been named, never built, in any sibling repo now exists and has been run against four real candidates. See `CUBBYLLM_HYPOTHESES.md`, Group A, and the central bet, Section 2.

**A much larger vocabulary, unlocked by removing the softmax bottleneck — decided: hybrid.** A fixed core vocabulary (next concrete step: a 128k–256k BPE build on the existing proven pipeline, costed at roughly five minutes of single-machine CPU time) plus a retrieval output head from day one (direct readout scales cleanly to a million-token vocabulary; the softmax bypass isn't an optimization at that scale, it's the only way the head fits in memory) plus a hypertoken/dynamic tail deferred as a later, separately-gated addition. See `CUBBYLLM_HYPOTHESES.md`, Group C.

All three decisions traced back to the same underlying idea (`CUBBYLLM_HYPOTHESES.md`, Section 2 — the central bet): making a model's active weights a function of context rather than a fixed, stored state is the forgetting fix, the mechanism for automatic specialization, and the enabling trick for a much larger vocabulary. The validation campaign confirmed the bet holds — with the important caveat that it only ever works hardened against its own internal drift, never as naive generation, and that context *inference* turned out to be the harder half of the problem, not context-conditioned generation itself.

## Documents in this folder

`CLAUDE.md` — orientation file for Claude Code sessions working in this repo. States the from-scratch/no-frozen-trunk correction up front, walks through the current implemented state, and lists known anti-patterns from the sibling repos to avoid repeating.

`CUBBYLLM_HYPOTHESES.md` — the working document. Every research claim and every sibling-repo reuse candidate gathered so far, organized as falsifiable hypotheses with a validation method and a kill criterion each; most now carry a dated validation result from the 2026-07-23 campaign. This is the one to read first.

`VALIDATION_REPORT.md` — the results of that campaign: 16+ experiments, a scorecard, and every number linked to a runnable script in `validation/` and a captured log in `validation/logs/`.

`PACKAGE_LAYOUT.md` — the target package layout, written as its own short spec before code could accumulate it by accident, the way it did twice already in the sibling repos.

`docs/superpowers/` — the implemented package's design spec and implementation plan.

`TODO.md` — the working checklist for what's left, grouped by what blocks real training, what architecture work is still open, and smaller follow-ups.

`CUBEMIND_CLEANUP_PLAN.md` — a freshly-verified status check on cubemind's own (separate, real) refactor plan, plus exactly what CubbyLLM depends on there and needs to stay stable while it's in progress.

`cubby-model-environment-map.md` — a verified (checked against actual source, not just docs) map of Cubby's current model architecture and CubeMind's world-model environment, as of 2026-07-23. This is the "current state" reference the hypotheses in the other documents are meant to change.

## Training data

`D:\grillcheese_training_data` (~210GB, surveyed 2026-07-23) is real material for two things at once: validating open hypotheses and building CubbyLLM's actual training corpus — see `CUBBYLLM_HYPOTHESES.md`, Group G. Headline finding: a script in this folder revealed `grilly.experimental.vsa.ops`, which the validation campaign has since confirmed is the same production VSA substrate behind cubby-lm's binding head and cubemind's world-arena block-codes (see the structural decisions above). Other findings: a real trained-tokenizer history (19,947 → 32,000 → 65,536 vocab, and now costed one step further to 131,072 at effectively no extra time) grounding the vocabulary decision; a dated NYT archive (1851–2024) that has since actually been used to re-run the Zero-Forgetting Stability Benchmark on real, correlated keys (the synthetic ranking survived unchanged); a 120GB candidate pretraining corpus that still needs a dedup/content-filter/reproducibility pass before use — this is the one substantial piece of Group G not yet done; and a ready-made general-capability regression check already sitting in the folder.

## Source repos

`cubby-lm` and `cubemind` (both at `C:\Users\grill\Documents\GitHub\`) are the direct predecessors. `cubby-concepts` (same location) is a related solo VSA/HDC/SDM prototype; its `vsa/` package is real, tested code that served as one of two benchmarked reference algebras for the binding head. `H:\AURA_GENESIS`, `C:\Users\grill\Desktop\GrillCheese`, and `H:\Novel_GNN_Arch` are independent sibling projects surveyed for reusable concepts; specific findings from each, including which ones the validation campaign confirmed or promoted, are cited throughout `CUBBYLLM_HYPOTHESES.md`. `C:\Users\grill\Documents\GitHub\grilly` — the GPU/Vulkan backend sibling — turned out to be the shared VSA substrate cubemind's own code already wraps; the binding head now standardizes on its `BlockCodeOps` family.
