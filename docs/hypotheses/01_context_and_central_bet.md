<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 1. Context

**Goal (one-liner):** Fast and profitable LLM gen 2 hybrid — general tasks plus specialization, automatic.

CubbyLLM is the redesigned successor to `cubby-lm`, drawing on lessons from `cubby-lm` and its environment sibling
`cubemind`. Two concrete problems motivate the redesign, both already scoped by the user:

**Problem 1 — VSA binding head.** Cubby's current `VSABindingHead` operates over a frozen D=10240 codebook. **Corrected
2026-07-23 (validation campaign):** the dict-lookup `UNBIND_ROLE` this paragraph previously attributed to the binding
head actually lives in **cubemind's VM** (`reasoning/vm.py:602-606`) — `vsa_binding_head.py` contains no unbind
mechanism at all; it is a clean cosine-readout output head. The real-unbind gap is real but sits on the VM/reasoning
side; the binding-head work for CubbyLLM is a from-scratch build either way (see H-B3, H-B5).

**Problem 2 — Hebbian memory layer.** The current Hebbian/Oja-style memory update is detrimental to memory: it drifts
and forgets catastrophically under sequential learning. This needs fixing using the research summarized in Section 3.

**Structure — two different axes, kept deliberately separate.** CubbyLLM's *package/directory layout* will likely mirror
cubemind's — meaning cubemind's intended, documented structure, not its accumulated sprawl (see Group F). CubbyLLM's
*model architecture* is a different question entirely: **CubbyLLM is an entirely different structure than cubby-lm,
built from scratch, including the vocabulary** (correction, 2026-07-23). cubby-lm's trunk is prior art — informative
about what worked, what didn't, and what to avoid repeating — not a frozen substrate CubbyLLM has to sit on top of.
Nothing below should be read as "patch cubby-lm's trunk"; where a hypothesis references cubby-lm's actual code, it's
diagnosing the predecessor to inform a fresh design, not scoping an in-place edit.

**New, explicit requirement (2026-07-23): a much larger vocabulary than cubby-lm's 32k, unlocked specifically by
removing the full-vocabulary softmax step.** The causal order matters and is worth stating precisely: vocab size is
being treated as a variable that gets freed up once the output/embedding mechanism stops paying an O(vocab) cost per
step — not a number chosen first that then needs optimizing around. This promotes what was previously framed as a set of
possible efficiency optimizations (old Group C) into a load-bearing design requirement; see the rewritten Group C below,
including two new hypotheses (H-C6, H-C7) this requirement surfaces.

cubby-lm's own shape, kept here for reference and lessons-learned, not as a CubbyLLM constraint: d_model=2560,
n_layers=22, d_ffn=6848, tokenizer `grillcheese_spm32k_v2` (32k SPM, frozen in cubby-lm), MinGRU recurrence per
`HybridBlock` with optional MoE / sliding-window attention / `EpisodicMemory` / SwiGLU FFN. `SegmentMemory`, `MTP`, and
`CubeLangHead` exist as code but are not wired into the default forward pass — `CubeLangHead` is self-labeled
"SCAFFOLD." Worth carrying forward deliberately rather than repeating by accident: the value of gating each new
capability behind an explicit version flag, and the risk of building attachments (MTP, CubeLangHead, SegmentMemory) that
never get wired into the actual forward path.

**Training data (added 2026-07-23):** a real, substantial corpus (~210GB) exists at `$CUBBY_TRAINING_DATA` — the user's
own framing was that it can be used both for testing and as raw material for "a better dataset." Full findings are in
the new Group G below, but the single most consequential discovery deserves a mention here: a training-data-generation
script in this folder revealed an already-existing, already-in-production VSA/hyperdimensional-computing implementation
(`grilly.experimental.vsa.ops`) running at the exact same D=10,240 dimensionality as both cubby-lm's binding head and
cubemind's world-arena block-codes. That's not yet confirmed to be the same substrate, but it's a strong enough lead
that it now sits at the top of Group B's validation order (H-B5), ahead of the two candidates the earlier survey found.

## 2. The central bet (H0)

**Claim.** Context-conditioned, hypernetwork-style parameter generation — making a model's active weights a function of
context, θ=f(c), rather than a fixed state that gets overwritten in place — is simultaneously the fix for catastrophic
forgetting *and* the mechanism for automatic specialization. If true, the two halves of CubbyLLM's one-line goal
("general tasks" and "specialization automatic") collapse into one underlying mechanism rather than two separate
subsystems.

**Why we believe it.** Three independent sources converge on this same shape, which is a stronger signal than any one of
them alone:

The Gradient Context Encoder (GCE) work (Ran Cheng, "Context Channel Capacity") argues catastrophic forgetting is
mathematically forced whenever a state-based learner runs at C_ctx=0 — i.e., whenever weight updates don't structurally
depend on context — and that the fix is making θ an unbypassable function of context, f_GCE(c), verified by
Wrong-Context Probing (P5) so the context signal can't be optimized away in training (the "Structural Bypass Defect").
This closed a CIFAR-10 oracle gap from 23.3pp to 0.7pp in the source material.

Sylph, independently, replaces global softmax classifiers with sigmoid-decomposed binary classifiers whose parameters
are generated by a few-shot hypernetwork conditioned on task embeddings — explicitly to avoid global gradient
interference and prevent forgetting during sequential task adaptation. Same shape (hypernetwork generates params from
context), different framing (classifier heads, not general weight rectification).

Dragon Hatchling (BDH) arrives at a related idea from the architecture side: its `bdh-cl` continual-learning extension
pairs fast local Hebbian plasticity with mechanisms to keep it from drifting, and the broader "trillion-parameter
blueprint" research (Section 6) explicitly proposes a GCE-style controller as the piece that prevents BDH's in-place
Test-Time-Training fast weights from drifting.

**What's notable:** none of the four surveyed sibling folders (cubby-concepts, AURA_GENESIS, GrillCheese,
Novel_GNN_Arch) implement this pattern anywhere. The closest analog is AURA_GENESIS's Liquid-MoE `tau(x)`-conditioned
continuous-time gating cell (`aura/core/liquid_moe.py` — real, working, NumPy-only), which is *routing* (selecting among
fixed experts) rather than *generating* continuous parameters, but is a plausible template for the small controller
network that would emit the context embedding driving a hypernetwork. This means H0 is genuinely new work for CubbyLLM,
not something to import — everything downstream in Groups A–D is really this one bet applied to different parts of the
model.

**A fourth independent instance, added 2026-07-23:** generating token embeddings from context instead of storing them in
a static lookup table (Group C's H-C1/H-C3, now driven by the much-larger-vocabulary requirement in Section 1) is itself
a specific case of H0's mechanism — θ_embedding = f(token/context) rather than a fixed embedding matrix. The vocab
requirement and the forgetting fix aren't two unrelated bets that happen to share a document; they're two applications
of the same underlying idea, this time one driven by a hard product requirement rather than a research paper. That's a
reason for more confidence in H0, not less.

**How to validate.** Build the smallest possible GCE-style controller (a context embedding feeding a rectification to
one small weight block, not the frozen trunk) and run the P5 Wrong-Context Probing protocol against it, plus a toy
sequential-task retention benchmark, compared against a frozen-weight baseline and the current Hebbian layer.

**Kill criterion.** If P5 probing shows the context signal gets optimized away even after architectural hardening (the
Structural Bypass Defect resists the standard countermeasures), or if the added hypernetwork forward/backward pass
erases the "fast" half of the goal at the scale CubbyLLM actually needs, the central bet needs revision — not
abandonment, but a scaled-down or partial version (e.g., gating only specific memory slots rather than general weight
generation).

**Validation result (2026-07-23 campaign): SUPPORTED at toy scale, with one load-bearing caveat.** The smallest θ=f(c)
prototype (frozen backbone per this section's own protocol, hypernetwork-generated head, 6 sequential tasks, no replay)
shows near-zero forgetting **only with architectural hardening** (von-Oswald-style output regularization — literally the
hypernetwork output regularizer of von Oswald, Henning, Sacramento & Grewe, *Continual learning with hypernetworks*,
ICLR 2020, arXiv:1906.00695; citation pinned here 2026-08-28 after the oracle-competition scoring in
`docs/research/2026-08-28-oracle-competition-scored.md` noticed it lived only in the agenda's Q6 note): task-0 MSE
0.062→0.060 across all tasks, P5 right-ctx 0.06 vs wrong-ctx 1.87 (context load-bearing, no bypass). The **naive**
hypernetwork without hardening ends *worse than the fixed-weight baseline* (2.734 vs 1.778 final MSE) — the
hypernetwork's own shared weights are themselves a C_ctx=0 learner and interfere internally, and its P5 probe flags
bypass. So H0 is only ever "generation + anti-drift countermeasure," never naive generation. Cost: 4.1× baseline per
step at toy scale, growing with number of protected contexts — the "fast"-half kill clause stays open at scale.
`VALIDATION_REPORT.md` / `validation/exp_h0_gce.py`.



**Stand-in pointer (2026-09-03, `standin/README.md` § "Two adapters on one base — v8"; a `[stand-in]` fact, not a trunk
result).** One fine-tuned 2.6B carrying both the program-emission and the talk data was measured to interfere across
rounds (identity fell in v5, forge decision 1.00 → 0.17 in v6, arithmetic 0.825 → 0.750 → 0.675 over v5–v7 as the chat
volume grew; each repaired by replaying the hurt family ×3/×4). The serve stack now speaks the θ=f(c) interface —
`Emitter.emit(…, context=)` with the route role and the hormonal state as c — and resolves it through a K=2 tag-indexed
adapter bank (`ContextualEmitter`, the frozen-router shape of H-C4). The v8 A/B (program-only vs talk-only adapters,
same base, same 40/task VM eval) is the cheapest measurement of whether context-selected parameters remove that
interference; its kill line is recorded there. It says nothing about the trunk's hypernetwork itself.

**A WORKING theta=f(c) IMPLEMENTATION ALREADY EXISTS — found 2026-08-02 in cubemind, port it rather than rebuild.**
`cubemind/execution/mindforge.py` (452 lines) is real, runs, and has full analytic gradients:
`MindForge.forge(context, layer_id) -> (A, B)` at L302, `forge_with_cache` L309, `backward(d_A, d_B, layer_id,
cache)` L358. Its parameterization is the one CubbyLLM should adopt: a block-code hypervector context is projected
`W_proj (d_hidden, k*l)`, LayerNormed, concatenated with a per-layer embedding `layer_embeddings (n_layers,
d_hidden)`, pushed through an MLP to `W_coeff (n_basis, d_hidden)`, and the generated adapter is a **linear
combination of learned low-rank bases**: `A = sum_i coeffs[i]*A_basis[i]`, `B = sum_i coeffs[i]*B_basis[i]`. Verified
by execution — gradients populate all 10 parameters, and it is genuinely conditioned on both inputs (layer
sensitivity |A(ctx,L0)-A(ctx,L1)| = 2.30; context sensitivity |A(c0,L0)-A(c1,L0)| = 67.15). It is correctly wired
into `cubemind/training/vsa_lm.py::VSALayer` (L877 forge, L921 the analytic backward). **This matters because
CubbyLLM's own `HyperGenerator` generates a full d*d matrix (ctx_dim=32 -> 4.19M outputs, ~134M params) which will
not scale**; the basis-coefficient form is roughly 40x smaller and generates *per-layer* adapters. Two caveats: the
GPU path is dead (`_mindforge_gpu_ok()` checks for `mindforge_basis_mix`/`mindforge_bwd_coeff`/`mindforge_bwd_basis`
in grilly_core, which do not exist, so everything runs numpy `tensordot`), and there is a stale duplicate
`mindforge.patch.py` encoding a removed-softmax design that `harrier_pretrain.py` was written against — that trainer
hand-rolls approximate gradients, never calls `forge.backward()`, and has produced no checkpoint. Take
`mindforge.py`, not either of those.

**The torch port that exists is DEGENERATE — this is the live gap between the bet and the code.**
`cubemind/model/cubby/heads.py::MindForgeLoRAHead` (L17) claims to be a "PyTorch port of cubemind.execution.mindforge"
but drops the two things that make it theta=f(c): there is **no layer embedding** (it is per-head, not per-layer) and
**no VSA input** (context is a plain dense vector). Worse, it is self-conditioned in every call site —
`MinGRUMultiTask.forward` L272 calls `head(pooled)` with `context=None`, and L86-87 then assigns `context = x`, with the
default head spec declaring `"task_code_source": "self"`. So in practice it computes **theta = f(h)**, conditioned on
the model's own pooled hidden state; no external task/context signal crosses into it at all. The `context=` argument is
plumbed and nothing supplies it. Also `basis_B` is zero-initialised there (L80) while the numpy version uses
small-random (L243) with the explicit comment "Zero init kills gradients through the LoRA path" — the two
implementations contradict each other on their own core design point, so the torch LoRA path may not be training.
