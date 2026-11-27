# CubbyLLM Hypothesis Document — Things to Validate

Status: draft, 2026-07-23, revised same day. This is a pre-registration of falsifiable hypotheses, not a design doc.
Nothing here is a commitment to build anything — it's the accumulated research and concept-survey material from the
CubbyLLM planning phase, organized so each claim can be checked before engineering time gets spent on it.

**Revision note (2026-07-23):** the first draft of this document implicitly treated cubby-lm's frozen trunk shape as a
constraint CubbyLLM would build on top of. That premise was wrong and has been corrected throughout: CubbyLLM's
architecture is an entirely different structure, built from scratch, including the vocabulary — which will be
significantly larger than cubby-lm's, specifically enabled by removing the full-vocabulary softmax step. See Section 1
and the rewritten Group C.

**Validation-campaign note (2026-07-23, later the same day):** a full validation pass was executed over every group —
see **`VALIDATION_REPORT.md`** for results and `validation/` for the runnable experiments + captured logs behind every
number. Hypotheses below now carry a **Validation result** line where the campaign touched them; several statuses
changed, and one premise (H-B3's `UNBIND_ROLE` location) was corrected. The report also lists what would change each
conclusion.

## 0. How to use this document

Every hypothesis below follows the same shape: a claim, why we currently believe it (which source it traces back to), a
concrete way to check it, and what result would kill it. A hypothesis with no kill criterion isn't really a hypothesis —
it's an opinion — so every entry has one even where it's approximate. Status is one of **VERIFIED** (independently
reproduced this session, not just asserted by a source document), **PLAUSIBLE** (a real source claims it, mechanism
makes sense, but nobody has measured it against Cubby's actual code), or **SPECULATIVE** (asserted in a vision/blueprint
document with no implementation behind it at all).

Hypotheses are grouped by the two structural problems driving the redesign (memory/forgetting, VSA binding head), then
by the two halves of the one-line goal (fast/profitable, automatic specialization), then backbone and hardware, then a
small set of organizational hypotheses, then training-data findings. Section 10 gives a suggested validation order —
cheapest and most falsifiable first, hardware last.

**Held-out discipline (added 2026-11-24, from an outside review of this file; Nick's yes).** A threshold, a rule or a
gate that keeps training rounds is tuned only on train and on dev or calibration slices disjoint from what it is read
on, and each held-out set is read once per decision. Never tuned on: H-E19's dev 80:200 (120 of the emitter's 329 GSM8K
val problems, all unseen templates), the 236 program-first held-out problems, H-E18's 815 unseen-writer records, and
H-E8's and H-E9's test splits. Tuned on: train; dev 0:80 and val 200:329 (order 0078's vote, the school's round gate,
any path verifier's threshold; the beam's tau stays on 0:80); held-in slices of the wiki world and of the far-fact dev
documents (relation rules, copy_refuse's theta and theta_end). A result counts as a win only on unseen templates: (1)
the emitter on H-E18's unseen-writer eval at plan >= 0.55 and chain >= 0.99 (H-E18's program-first arm read 0.495 and
0.993) with 0 copied literals and 0 unknown slots; (2) copy_refuse served on grilly2 at 0% false facts on unseen
questions the scene cannot answer; (3) order 0078's vote, or a path verifier after it, at >= 0.340 on the 236's 98
unseen-template problems (the mean ranker: 0.286) with spoken-wrong <= 1%. Base v2 (grouped-query attention with 8 query
and 2 key-value heads of 128, H-P3's chrono init, a tied vocabulary) stays a hypothesis, gated on a 30-minute A/B at d =
512, L = 8 whose bar is written before it runs; it is not an order.

## Where each part lives (sharded 2026-11-27)

The document was split by its sections into `docs/hypotheses/`, word for word, with long lines wrapped at
120 characters (order 0089, after the audit of 2026-11-27). `docs/hypotheses/INDEX.md` lists every entry, its
status and its file. New entries go into their group's file; this file keeps section 0 and this map.

| Section | File |
|---|---|
| 1. Context; 2. The central bet (H0) | [docs/hypotheses/01_context_and_central_bet.md](docs/hypotheses/01_context_and_central_bet.md) |
| 3. Group A — Memory & forgetting fix | [docs/hypotheses/03_A_memory_and_forgetting.md](docs/hypotheses/03_A_memory_and_forgetting.md) |
| 4. Group B — VSA binding head fix | [docs/hypotheses/04_B_vsa_binding_head.md](docs/hypotheses/04_B_vsa_binding_head.md) |
| 5. Group C — Vocabulary, tokenization, and the softmax bottleneck (a core design requirement, not an optimization) | [docs/hypotheses/05_C_vocabulary_and_softmax.md](docs/hypotheses/05_C_vocabulary_and_softmax.md) |
| 6. Group D — Backbone architecture hypotheses | [docs/hypotheses/06_D_backbone.md](docs/hypotheses/06_D_backbone.md) |
| 7. Group E — Hardware and economics (explicitly gated, long-horizon) | [docs/hypotheses/07_E_hardware_economics_and_the_seats.md](docs/hypotheses/07_E_hardware_economics_and_the_seats.md) |
| 8. Group F — Structural and organizational hypotheses | [docs/hypotheses/08_F_structural.md](docs/hypotheses/08_F_structural.md) |
| 9. Group G — Training data (`$CUBBY_TRAINING_DATA`) | [docs/hypotheses/09_G_training_data.md](docs/hypotheses/09_G_training_data.md) |
| 9b. Group P — Prospection: branching futures, timescales, keyframe compression (added 2026-08-25) | [docs/hypotheses/09b_P_prospection.md](docs/hypotheses/09b_P_prospection.md) |
| 9c. Group S — Senses: a sensory cortex learned by play (added 2026-10-10) | [docs/hypotheses/09c_S_senses.md](docs/hypotheses/09c_S_senses.md) |
| 9d. Group K — Knowledge, the map, the state: the pause of 2026-10-20 (added 2026-10-20) | [docs/hypotheses/09d_K_knowledge_map_state.md](docs/hypotheses/09d_K_knowledge_map_state.md) |
| 10. Prioritized validation order; 11. What this document does not claim; Appendix: source map | [docs/hypotheses/10_order_claims_and_sources.md](docs/hypotheses/10_order_claims_and_sources.md) |
