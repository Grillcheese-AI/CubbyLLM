# Panel: fluent at 450M, the agent harness, and an HDC FFN (2026-09-28)

Asked independently, same prompt, temperature 0.3, reasoning effort `low`. Cached under `standin/data/out/panel_cache/`; `--offline` re-renders without spending.

**These are model opinions, not findings.** Each proposal still has to pass the measurement it names. Where the answers disagree, our framing is probably underspecified.

## The question

<details><summary>full prompt</summary>

```
# The system (CubbyLLM, as of 2026-09-28)

One principle runs through it: the model proposes, the host disposes, a verifier decides. Nothing is
spoken unless a chain verified by a symbolic VM reaches it, and knowledge lives outside the weights.

**Layers**
- Representation: hypervectors. Words map to quantised FHRR block codes (80 blocks x 128 phases,
  D = 10,240) from a fixed 60,151-word table; an 80-byte compact form is searched without a codebook.
  The perception side (an SNN front end over camera and audio) reads out into the same space.
- Stores, outside the weights: a history graph (940,897 events, 45,588 causal and temporal links); a
  wiki world of 552,297 template facts; capsule facts with provenance and verified-use counts; an
  episodic store of VM-certified chains; a skill library of composition rules mined nightly; a signed
  ledger of every VM verdict; an encrypted vault for conversations.
- Verifier: CubeLang, a Rust VM for vector-symbolic programs (bind, unbind, cleanup). Concepts go in and
  (symbol, similarity) comes out; raw vectors never cross. Every hop of a reasoning chain must clear a
  threshold, and an absent-role control must stay silent.
- Host (the "brain", Python): reads the kind of question, resolves "it"/"that" from a context graph
  (long-term events, the conversation, per-ask reasoning traces), looks up the event, runs a plan through
  the walk and the VM, then speaks, refuses, or asks a clarifying question. Every draft passes a
  name-and-number guard, a value check (the reply may state only the value the VM returned) and a claim
  check (everything named from the facts must be tied to what the facts tie it to). A 5-hormone
  neurochemical ODE bends routing caution and tone, never facts.
- Trunk: base450m, our own model.
  - d1024, 32 layers: 21 MinGRU-style recurrent layers and 11 window-512 attention layers (every 3rd,
    4 heads x 256); SwiGLU FFN 2048.
  - 128k BBPE tokenizer with CubeLang opcodes as atomic tokens; untied embedding and head (262M of its
    576M parameters; the trunk is 314M).
  - A context-generated per-layer adapter (theta = f(c): 16 bases x rank 8, mixed by a frozen,
    offline-trained context router).
  - 2.77B training tokens (~9 per trunk parameter), 16.8 A100-hours; held-out loss 2.36. Serves on our
    own Vulkan engine on a 12 GB consumer AMD GPU at ~40 tok/s decode, batch 1.
  - Probes: fluent English and French prose; in-context copying works (copy loss 1.2); but it does not
    ground. It binds a value copied from context to the entity asked about at chance (52%), and answers
    from memory over a contradicting context 22 times in 24.
- Adapters on the base, routed per turn by the host (a frozen tag router, never learned online): a talk
  adapter (LoRA r16 on all 181 projections, 7.9M parameters) trained to say what the VM returned, and a
  program emitter (text -> CubeLang; still served by a stand-in model behind the same interface).
- Learning. Nothing learned changes what may be spoken; it only proposes. The VM verdict is the only
  reward (a striatum-like arbiter reorders plan proposers by reward-prediction error). A refusal becomes
  fetch -> gate -> store -> a second walk. A nightly sleep cycle replays the day into episodes, skills and
  adapter training data. A forge writes new CubeLang tools that the VM must certify. An adapter
  lifecycle (detect -> spawn -> train -> promote -> route/prune) is specified; its detector and promote
  rule are not built yet.
- An embodied agent, Cubby-Man, lives on the same stack in a 3-D maze it learns by exploring: walls
  learned from refused moves, hypotheses kept apart from facts, power-moves it invents and the VM certifies.

**Measured state of the talk adapter** (held-out data, pre-registered gates; "behind the host" = after
the host's checks)
- The mix trade-off. With fact records at 11% of the talk mix (more conversational data), the fact
  families were: relation 96.2%, bind 87.0%, counter-fact 91.2%, "the facts don't say" 70.3%, and 18
  answers that were wrong yet named only grounded things. With facts at 63%: 98.8 / 100 / 93.8 / 89.1%,
  and 6 such answers. More "don't say" records lifted that family but cost right answers elsewhere (384
  vs 389 of 407 behind the host). Behind the host's checks, 0 wrong answers were spoken in all three.
- History questions in plain words: spoken and correct 59% when reworded, 90% when the event is named;
  1% spoken wrong. Follow-ups that name no event: 97% resolved through the context graph, 94.7% spoken
  and correct.
- It answers like a form, by design: the adapter reads only the host's canonical restatement of the
  question, never the person's words or earlier turns; the value check lets it say one value; 63% of its
  data is one-sentence fact answers; the host has three moves (answer, ask which one, "The facts don't say.").

**Other recent results**
- Growth. The base was grown exactly (identical loss at step 0) to 1.29B (depth x2 + FFN x2, zeroed
  exits) and 1.78B (width x2). After one equal hour of training, every grown arm was behind the 450M
  continued on all 9 shared held-out sources (+0.03 to +0.08 nats). The base is short of data before
  parameters, and it is now being continued on 3.2B unseen tokens. The target stays 450M.
- The system's parts exist as separate modules: a serving pipeline (sense -> neurochemistry -> fast
  route -> cortex router -> memory / reasoning / talk / plugin cortices), the verified ask loop with an
  event stream to a live 3-D control panel, the context graph, the sleep cycle, the forge. No single
  process owns boot, state, scheduling and the turn loop.

# Three problems

## A. Fluent without losing capacity, at 450M

We need Cubby to hold a natural conversation in English and French: small talk, follow-ups,
corrections, "tell me more", varied wording, a fuller "I don't know" ("I don't have what caused it - I
know it was 1683"). It must keep every grounded behaviour above: the fact families, zero wrong answers
behind the host, the follow-ups, and the program emitter's VM-verified rates. The data says fluency and
grounding compete, at least in one mix, for the same 7.9M adapter parameters or the same base.

Options we have written down: a mode line; two adapters the host switches per turn; a larger LoRA
(rank 64, ~31M); new identity-initialised blocks inserted into the frozen base, trained alone and skipped
on the facts path; experts in two tiers (the host picks the group by turn kind; a learned mixture of
LoRA experts only within that group); rank-wise gating; per-layer rank. Separately: moving the host's
scaffolding into the weights (train with the scaffold, update as if it had not been there, rewarded only
by the host's checks).

Which is right, and why? Or is the premise wrong? Is the bottleneck the 450M base's own language
ability (so fluency should come from continued pretraining and its data mix, not an adapter), the
adapter's size, or interference between tasks? What would you measure first to tell these apart?

## B. A bootstrap and an orchestrator: the whole system as an agent harness

We want the whole system to act as an agent harness: one process that boots the brain, owns its state,
runs the turn loop, schedules day and night, and exposes the host's capabilities as tools.
1. The loop: what one step of the harness is (observe -> plan -> call tools -> verify -> speak or ask ->
   record), where multi-step tasks and sub-goals live, and what stops a runaway.
2. Tools: VM programs, lookups over the stores, fetch-and-learn, the forge (new tools, VM-certified),
   world plugins (the maze, future worlds), memory reads and writes. How are they declared, permissioned
   (deny by default; the guards above; the explicit-content gate; the vault) and audited (the signed ledger)?
3. Where the 450M model sits: does it propose tool calls (as a policy), only phrase results, or both
   through separate adapters? Since it does not ground on its own, what must the harness never let it decide?
4. Bootstrap, in both senses: (i) cold start of the running system - load stores and adapters, self-test
   against the gates, calibrate thresholds, refuse to serve if a check fails; (ii) a capability or a world
   starting from almost nothing (Cubby-Man starts knowing two facts) and growing from its own verified traces.
5. The smallest first version, and how to show the harness beats today's separate modules beyond "it runs".

## C. An HDC-based FFN whose units react like linked neurons

The idea is the owner's and deliberately rough. Replace or augment the dense FFN with a feed-forward
memory built in the same hypervector space the rest of the system uses. Its units are hypervectors, and
they are linked: activating one also activates the units bound to it, like linked neurons (spreading
activation, cleanup). Further, a multi-layer version in which every unit is anchored at a position
x, y, z and a time t, so that units near each other in space or time respond together and sequences are
laid down along t. The block codes are FHRR phases, so positions can be encoded by fractional powers of
base vectors (X^x * Y^y * Z^z * T^t, with * as binding), and similarity falls off with distance.

Hoped for: more capacity per parameter at 450M; associations that can be written without gradient
descent; a representation the symbolic side can read, without breaking the rule that only symbols cross
into the VM.

Is this sound? Relate it to what is known: FFNs as key-value memories, product-key and memory layers,
modern Hopfield layers, topographic models, spatial semantic pointers and other VSA/FHRR position codes.
How would it be trained, and inserted into a pretrained 450M model? What is its capacity limit
(crosstalk in superposition), and its failure mode? Does it conflict with "knowledge lives outside the
weights", or is it the natural bridge between the trunk and the stores? What is the cheapest experiment
that would tell us it is worth building?

# Constraints that rule out the easy answers

1. No external LLM at serve time. Frontier models may write training data and judge tests, never answer a user.
2. The target is 450M: growth was just measured and lost at our budget. A proposal that needs a bigger
   base must say so and say why.
3. Serving is local: a 12 GB consumer GPU, our own Vulkan engine, batch 1. Training is one rented GPU
   for hours, not a cluster.
4. The host disposes and the VM verifies. Nothing may let the model's output reach speech unchecked, and
   a learned router may not replace the host's frozen routing online (a router that learns online
   forgets its own routing).
5. Every claim must be measurable against a pre-registered gate; "should improve" is not an answer.

# What we want back

For each of A, B and C: your recommendation, the reasoning, the specific failure mode it introduces, and
the measurement that would show it working or not. Then, across all three, the order you would do them
in and why. Disagree with the framing where it is wrong.
```

</details>

## Answers (6 of 6 returned)

### anthropic/claude-opus-4.6

## A. Fluency at 450M

**Diagnosis first.** You need one measurement before choosing a mechanism: train two *separate* LoRA-r16 adapters on the continued base (after the 3.2B tokens)—one fact-only (63% mix), one chat-only—and evaluate each on its own domain. Compare to a single mixed adapter. If each separate adapter matches or beats the mixed one on its own domain, the bottleneck is **interference within shared adapter parameters**. If the chat-only adapter still produces stilted prose, the bottleneck is the **base's language ability** and no adapter trick will fix it.

**My estimate:** the base at 2.36 nats after only ~9 tok/param is undertrained for fluency. The continued pretraining on 3.2B tokens (bringing you to ~13 tok/param) will do more for fluency than any adapter architecture change. Fluency is a base capability; grounding compliance is an adapter capability. They shouldn't need to share parameters.

**Recommendation: two adapters, host-switched deterministically.** The host already classifies turn kind with a frozen router. Fact turns get the current talk adapter (63% fact mix). Chat turns get a chat adapter (same LoRA r16, trained on conversational data with the host scaffold still enforced). The host's value check, claim check, and name guard apply to both—so the chat adapter cannot smuggle an unchecked fact into speech.

**Why not the others:** MoE, rank gating, per-layer rank, and identity-initialised blocks all introduce a learned routing or mixing component that must be validated not to degrade grounding. Two host-switched adapters add zero learned routing. A larger LoRA (r16→r64) quadruples parameters but doesn't resolve interference—it gives both tasks more room to fight.

**Failure mode:** The host misclassifies a grounded-fact turn as chat, and the chat adapter confabulates a value. This is caught: the claim check requires every named entity to tie to a store fact, and the value check requires the stated value to match the VM return. You already measured 0 wrong answers behind the host across all three mixes. The gate that detects this failure: run the full 407-item fact eval through the chat adapter *with host checks*. If any wrong answer passes, the host's guards are insufficient for that adapter.

**Validation:**
- **Works:** After continued pretraining + two-adapter split, fact families stay ≥ current (98.8/100/93.8/89.1), 0 wrong behind host, AND a new conversational eval (estimate: 50 multi-turn dialogues with follow-ups, corrections, "tell me more") scores ≥80% natural by a blind human rater (against current system as baseline).
- **Doesn't work:** Fact families drop by ≥2 points on any family, OR conversational naturalness doesn't improve over current single-adapter system. If the latter, the base is the bottleneck and you need more pretraining data, not adapter work.

---

## B. Agent Harness

**Recommendation: a single-process state machine with a tick loop.**

One tick:

```
observe: parse input + context graph resolve ("it"/"that")
classify: host's frozen tag router → {fact, chat, tool, clarify, refuse}
plan: if tool/fact → program emitter proposes CubeLang program
       if multi-step → push sub-goals onto a bounded stack (max depth 4)
execute: VM runs program, returns (symbol, similarity); timeout at 200ms
gate: name guard, value check, claim check, content gate, hormone modulation
act: speak / ask clarification / refuse
record: append to event stream, signed ledger entry, context graph update
```

**Multi-step tasks** live on a bounded goal stack (not a queue—LIFO, max depth 4). Each sub-goal gets one plan-execute-gate cycle. If any sub-goal fails the VM or exceeds depth, the harness returns "I can't answer that yet" and records why. This is the runaway stop: **hard depth bound + per-step timeout + VM must clear on every hop.**

**Tools** are declared as CubeLang type signatures (input roles → output roles) registered in a manifest. Permissions: deny-by-default allowlist per tool category. The vault requires an explicit unlock token per conversation. The signed ledger gets an entry for every tool invocation, pass or fail.

**Where the 450M sits:** It proposes tool calls (program emitter adapter) and phrases results (talk adapter). It never decides: routing (host decides), whether to serve (gate decides), what's true (VM decides), or store writes (VM certifies). The harness must treat model output as *untrusted proposals* at every point.

**Cold-start bootstrap:** Load stores → load base + adapters → run a self-test suite (estimate: 80 items covering every fact family, every gate, similarity threshold calibration on 20 known-good and 20 known-bad pairs) → compute pass rates → if any family below its pre-registered threshold, disable that capability and log → enter serve loop. Estimate: ~30 seconds on the target GPU.

**Capability bootstrap (Cubby-Man pattern):** Seed facts → explore (propose actions) → VM certifies outcomes → store as capsule facts → nightly sleep consolidates into episodes and skills → forge proposes new tools → VM certifies tools → cycle. The invariant: the verified-fact store grows monotonically; no unverified hypothesis is ever treated as fact.

**Smallest first version:** The tick loop + cold-start self-test + signed ledger. Three files. No nightly cycle, no forge, no multi-step—just single-turn observe-classify-plan-execute-gate-speak-record.

**Failure mode:** The state machine adds latency. Estimate: ~50ms overhead per tick on top of model inference. If total turn latency exceeds 1.5s (current estimate: model ~250ms at 40 tok/s for a 10-token answer + VM + lookups), the system feels sluggish.

**Validation:**
- **Works:** All pre-registered gates pass at cold start and on every turn in a 200-turn test session. Zero regressions on fact families. End-to-end latency ≤1.5s p95. The signed ledger is complete (every tool call auditable).
- **Doesn't work:** Any gate regression, or the harness crashes/hangs on an input the separate modules handled.

---

## C. HDC-based FFN

**The premise is partially wrong.** Here's why:

An FFN with hidden dimension 2048 in a d=1024 model has ~4M parameters per layer. It acts as a key-value memory (Geva et al., 2021) where capacity scales with the number of parameters. A superposition-based FHRR memory at D=10,240 can store approximately **D/(C·log D)** items before crosstalk corrupts retrieval, where C depends on the similarity threshold. For a 0.9 cosine threshold, this is roughly **500–1000 items** (estimate based on Plate's capacity results for HRR). This is orders of magnitude less than what a trained FFN implicitly encodes.

The spreading-activation idea (bind-linked neighbors activate together) worsens this: each hop multiplies crosstalk. At 2 hops with fan-out 5, you're superimposing ~25 vectors; at 3 hops, ~125. Retrieval collapses.

The spatio-temporal encoding (X^x · Y^y · Z^z · T^t) is well-established—Eliasmith & colleagues' spatial semantic pointers do exactly this, and the graceful similarity falloff with distance is real and useful. But this is a **memory indexing** scheme, not a computation scheme. It doesn't replace what FFN nonlinearities do.

**What this actually is:** a read head over external stores, not an FFN replacement. And that's useful—it's the bridge you mention. The model's hidden state, projected into FHRR space, does a similarity lookup against the capsule/wiki stores, and the result is injected back as a residual. This is essentially a product-key memory (Lample et al., 2019) but in your native vector space, which means the symbolic side can inspect the lookup without crossing the raw-vector boundary.

**Recommendation: build it as an external memory attention layer, not an FFN replacement.** Insert one such layer (after layer 16, estimate) that projects the residual stream into FHRR space, does top-k lookup (k=4) against a table of store-derived key-value pairs, and projects back. Freeze the base; train only the projection matrices (estimate: 2 × 1024 × 10240 ≈ 21M parameters, or use a bottleneck to ~2M).

**Does it conflict with "knowledge outside the weights"?** Only if you bake the store contents into trained projections. If the lookup table is populated from the external stores at load time and updated when stores change, knowledge stays outside. The projections learn *how to query*, not *what to know*.

**Training insertion:** Freeze the pretrained base. Initialize the new layer's output projection to zero (identity skip connection). Train only the projection matrices on the existing fact data with the VM-verified labels as signal. Estimate: 1–2 A100-hours.

**Cheapest experiment:** Don't build the layer. Instead: take 200 fact-retrieval queries, encode them as FHRR vectors using your existing pipeline, do cosine lookup against FHRR-encoded wiki facts, measure recall@4. If recall@4 < 70%, the FHRR encoding doesn't separate your queries well enough and no amount of architecture will fix it. If recall@4 > 90%, the lookup works and it's worth building the layer. This experiment costs zero GPU hours—it's numpy on a CPU.

**Failure mode:** Crosstalk at scale. With 552K wiki facts superimposed or even indexed, the top-k lookup returns wrong neighbors. This is measurable as recall@k degradation as the store grows.

**Validation:**
- **Works:** The retrieval layer improves fact-question accuracy (spoken-and-correct) by ≥5 points on the hard cases (reworded history questions, currently 59%), with no regression on other families.
- **Doesn't work:** Recall@4 on the FHRR lookup experiment is below 70%, or held-out loss increases by >0.05 nats after layer insertion.

---

## Order: A → B → C

1. **A first.** The continued pretraining is already running. The two-adapter diagnostic is cheap (one day of adapter training) and answers the fundamental question: is it interference or the base? Everything else depends on this.

2. **B second.** It's engineering, not research. The architecture is already clear from your existing modules. The harness makes A's two-adapter switching clean (the host's routing lives in one place). It also gives you the scaffold to test C—you can't insert a retrieval layer and measure it end-to-end without the harness.

3. **C last.** Highest risk due to the capacity concern. The cheap CPU-only recall experiment should gate whether you build it at all. And you need A and B working to have a system worth augmenting.

---

### openai/gpt-5.5

## Executive recommendation

- **A (fluency vs grounding): use two host-selected adapters, not one bigger mixed adapter, and measure whether the base or adapter is the bottleneck before adding capacity.**  
  My bet: the main problem is **task interference in the adapter and host interface**, not an intrinsic 450M fluency ceiling. Your base is weakly grounded, but it is already fluent enough to phrase results if you stop asking the same small LoRA to be both a fact transducer and a conversational persona. Keep the fact path narrow and verified. Add a separate “conversation/formulation” adapter that never introduces factual claims unless given VM-returned slots.

- **B (agent harness): build the orchestrator first, but make it boring.**  
  One deterministic process should own boot, turn state, tool permissions, ledger writes, scheduling, and refusal. The 450M model should propose plans and language, but must not decide truth, permission, identity, memory writes, or tool safety. The smallest useful harness is not “autonomous agent”; it is a **typed verified turn loop with a task stack and budgets**.

- **C (HDC FFN): do not replace FFNs yet. Build it first as a frozen/side memory layer with explicit read/write probes.**  
  The idea is not crazy: it resembles FFN-as-key-value-memory, product-key memory, modern Hopfield retrieval, and spatial semantic pointer systems. But as a trainable replacement for transformer FFNs it is high-risk. The cheapest test is a **plug-in associative memory after selected layers**, identity-initialized/gated to zero, trained or written on synthetic binding/retrieval tasks and then on your own VM traces. If it cannot beat a small LoRA or kNN memory on controlled recall under the same latency/memory budget, stop.

Order: **B first, A second, C third**. B gives you stable measurement and safety. A improves user-facing quality without violating architecture. C is research; do it only behind gates.

---

# A. Fluent without losing capacity at 450M

## Recommendation

Use **two host-selected adapters**:

1. **Grounded answer adapter**
   - Current talk adapter, or a successor trained mostly on canonical host restatements and VM-returned values.
   - Output contract: answer from slots, ask clarification, or say facts do not say.
   - No access to raw conversation beyond what the host canonicalizes.
   - Keep it deliberately narrow.

2. **Conversational phrasing adapter**
   - Used only when the host has already decided the response type and provided a constrained response frame.
   - It can do small talk, empathy, turn-taking, elaboration style, French/English surface realization.
   - It receives:
     - allowed facts as immutable slots,
     - allowed uncertainty statements,
     - allowed follow-up prompts,
     - prohibited-claim list or “no new claims” flag.
   - Its output still goes through the same name/number/value/claim checks.

Do **not** first do rank-64 LoRA, learned mixtures, rank-wise gates, or scaffold distillation into the base. Those are plausible later, but they obscure the central question: is the loss caused by capacity or by interference? A single adapter trained on both “be conversational” and “be a deterministic fact form” has contradictory gradients.

The premise I disagree with: **“fluency and grounding compete for the same 450M base.”**  
Your own results do not prove that. They show that, with the current adapter/data/interface, fact accuracy and conversational coverage trade off. That is an adapter/control problem until proven otherwise.

## Why two adapters, not one larger adapter?

Because the failure you cannot tolerate is factual drift. A larger mixed adapter gives the model more capacity to invent plausible connective tissue. It may improve surface fluency while making the host checks work harder.

Two host-switched adapters preserve a clean invariant:

- Grounded path: “say this value and only this value.”
- Conversational path: “make this allowed act sound natural, but do not add facts.”

The host already routes per turn. Use that.

Estimated size:
- Keep grounded answer adapter around **r16 to r32**.
- Add conversational adapter **r16** initially.
- Total extra parameters likely under **16–24M**, estimate, depending on exact projection coverage.
- This fits local serving better than a broad r64 everywhere.

## Concrete mechanism

Define response acts:

```text
ANSWER_SLOT
ANSWER_WITH_LIMITATION
ASK_CLARIFYING
SMALLTALK_NO_FACT
ACKNOWLEDGE_CORRECTION
TELL_ME_MORE_PROMPT
REFUSE_UNVERIFIED
SUMMARIZE_VERIFIED_TRACE
```

The host chooses the act. The adapter does not.

Example host packet to conversational adapter:

```json
{
  "act": "ANSWER_WITH_LIMITATION",
  "language": "en",
  "style": "warm_brief",
  "allowed_claims": [
    {"subject": "event_9134", "predicate": "year", "object": "1683"}
  ],
  "unknowns": [
    {"predicate": "cause", "status": "not_in_facts"}
  ],
  "required_semantics": [
    "state_known_year_1683",
    "state_cause_unknown"
  ],
  "forbidden": [
    "do_not_name_cause",
    "do_not_infer_motive",
    "do_not_add_source"
  ]
}
```

Acceptable output:

> I don’t have what caused it. The verified fact I have is that it was in 1683.

Then the ordinary guards check:
- number `1683` matches VM,
- no extra named entity,
- no unsupported cause,
- required unknown statement present.

For French:

> Je n’ai pas la cause dans les faits vérifiés. Ce que j’ai, c’est l’année : 1683.

## What to measure first

Before changing architecture, run three diagnostic experiments.

### 1. Base fluency ceiling test

Freeze everything except prompting/interface. Feed the base or current talk adapter verified slot packets and ask for paraphrases that contain no factual invention.

Dataset:
- 1,000 English packets.
- 1,000 French packets.
- Include answer, limitation, clarification, correction, “tell me more”, refusal.

Metrics:
- Human/blind preference or small rubric score for naturalness.
- Claim violation rate.
- Required semantic inclusion rate.
- Host pass rate.

Interpretation:
- If naturalness is poor while claim violations are low, maybe base/adapters lack fluency.
- If naturalness is good but claim violations rise, control/interface is the bottleneck.
- If both are poor, adapter capacity/data are bottleneck.

Gate estimate:
- **≥ 4.0/5 naturalness** on simple acts.
- **0 spoken wrong behind host**, unchanged.
- **≥ 98% required semantic inclusion** before host.
- **≤ 1% unsupported claim attempts** before host.

### 2. Adapter interference test

Train three adapters with equal token budgets:

A. fact-only adapter  
B. conversation-only adapter  
C. mixed adapter

Same base, same evaluation.

Measure:
- Fact family scores.
- “Facts don’t say.”
- Conversational act completion.
- Unsupported claim attempts.
- Host rejection rate.
- French quality separately.

If A and B each work but C regresses, interference is proven.

Expected result, estimate: C will underperform on either “don’t say” or natural continuation because the output distributions conflict.

### 3. Capacity scaling test

Train mixed adapters at r16, r32, r64.

If r64 closes both fact and fluency gaps without higher violation rate, capacity was the bottleneck. If not, routing/task conflict was.

Gate:
- r64 must improve naturalness by meaningful margin, e.g. **+0.4/5**, while preserving:
  - zero wrong spoken behind host,
  - same or better fact family scores,
  - no increase in host rejection above, say, **+2 absolute points**.

If it fails, do not carry r64 complexity.

## On moving host scaffolding into weights

Do not make this the first fix.

“Train with scaffold, update as if it had not been there” sounds like useful distillation, but it attacks the wrong boundary. You already know the model does not ground and prefers memory over contradictory context. Distilling host behavior into weights may improve style, but it risks creating a model that **sounds like it followed the host while not preserving the symbolic invariant**.

Acceptable use:
- Distill only surface forms:
  - how to phrase uncertainty,
  - how to ask clarification,
  - how to acknowledge correction.
- Do not distill:
  - entity resolution authority,
  - value selection,
  - permission decisions,
  - memory writes,
  - tool safety.

## Failure mode introduced by two adapters

The main failure is **routing error**.

Example:
- Host sends a factual answer to the conversational adapter with too much freedom.
- The adapter adds a plausible extra fact.
- The guard may catch it, but rejection rate rises and conversation becomes brittle.

Mitigation:
- Host acts are typed.
- Conversational adapter receives explicit forbidden fields.
- All outputs pass existing guards.
- If conversational output fails twice, fall back to grounded form adapter.

## Validation

Works if:
- Existing held-out grounded gates remain at or above current best:
  - relation around **98%+**,
  - bind **~100%**,
  - counter-fact **≥ 93–94%**,
  - “facts don’t say” **≥ 89%**,
  - **0 wrong spoken behind host**.
- Reworded history questions improve from **59% spoken/correct** to, estimate, **75–80%** without increasing wrong spoken above current 1%.
- Follow-up resolution remains near **97% resolved**, **94%+ spoken/correct**.
- Conversational act benchmark:
  - **≥ 90% act satisfaction**,
  - **≤ 1% unsupported claim attempts before host**,
  - **0 after host**.
- French responses pass the same semantic gates, not just fluency.

Does not work if:
- Host rejection rate climbs substantially, e.g. +5 absolute points.
- The conversational adapter frequently inserts unsupported claims.
- The grounded adapter regresses when co-served.
- User-visible responses remain form-like despite the extra adapter.

---

# B. Bootstrap and orchestrator: whole system as an agent harness

## Recommendation

Build a single deterministic orchestrator process with:

1. **Boot manager**
2. **State owner**
3. **Typed turn loop**
4. **Tool registry**
5. **Permission system**
6. **Budgeted task stack**
7. **Verifier/ledger integration**
8. **Day/night scheduler**

Do not start with a free-form autonomous agent. Start with a controlled harness where the model proposes but every proposal is interpreted as a typed object.

## One step of the harness

One turn should be:

```text
observe
→ normalize/input policy checks
→ update context graph
→ classify turn kind
→ create task frame
→ retrieve relevant state
→ plan candidate actions
→ validate plan syntax and permissions
→ execute tool calls under budgets
→ verify results with VM/guards
→ decide response act
→ generate phrasing
→ check response
→ speak/ask/refuse
→ record trace and ledger entries
→ schedule after-turn jobs
```

A minimal internal task frame:

```json
{
  "turn_id": "...",
  "user_id": "...",
  "input_refs": ["vault_msg_..."],
  "task_kind": "history_question | fact_question | smalltalk | maze_action | correction | fetch_request",
  "goal": "...",
  "allowed_tools": [...],
  "budgets": {
    "tool_calls": 8,
    "vm_steps": 20000,
    "wall_ms": 3000,
    "fetches": 1,
    "forge_attempts": 0
  },
  "state_refs": {
    "context_graph": "...",
    "conversation": "...",
    "reasoning_trace": "..."
  },
  "status": "open | waiting_clarification | answered | refused | failed"
}
```

## Where multi-step tasks live

Use a **task stack**, not hidden model state.

Each task has:
- goal,
- parent task,
- current verified facts,
- unresolved variables,
- allowed tools,
- budget,
- stop condition,
- ledger pointer.

Subgoals are host objects, not prose in the model context.

Example:

```text
Task: answer "Why did it happen?"
Subgoal 1: resolve "it" from context graph.
Subgoal 2: retrieve event.
Subgoal 3: query cause role.
Subgoal 4: if absent, query known roles worth stating.
Subgoal 5: choose response act ANSWER_WITH_LIMITATION.
```

## What stops a runaway

Hard stops:
- max tool calls per turn,
- max VM steps,
- max wall-clock time,
- max recursion depth,
- max forge attempts,
- max fetch attempts,
- no network/tool expansion inside a turn unless specifically permitted,
- repeated-failure cutoff.

Semantic stops:
- if goal is satisfied, stop.
- if missing variable requires user choice, ask clarification.
- if permissions fail, refuse.
- if verification fails twice, fall back to canonical refusal.

Estimated defaults:
- tool calls: **8** per ordinary ask.
- VM steps: **20k–100k**, depending on current VM cost.
- recursion depth: **3**.
- wall-clock: **3 seconds** ordinary, **10 seconds** explicit longer task.
- forge attempts: **0 during user turn**, only queued for night unless developer mode.

## Tools

Declare every tool with a schema.

Example:

```json
{
  "name": "history.lookup_event",
  "version": "1.2",
  "input_schema": {
    "event_ref": "EventID | Description",
    "roles": ["agent", "object", "time", "cause", "location"]
  },
  "output_schema": {
    "facts": ["FactRef"],
    "missing_roles": ["Role"],
    "provenance": ["ProvenanceRef"]
  },
  "permissions": {
    "default": "deny",
    "requires": ["read_history"],
    "forbidden_if": ["explicit_content_blocked"]
  },
  "audit": {
    "ledger": true,
    "store_inputs": "hash_or_ref",
    "store_outputs": "fact_refs"
  }
}
```

Tool categories:

1. **VM programs**
   - bind/unbind/cleanup.
   - Always ledgered.
   - Raw vectors never leave VM.

2. **Store lookups**
   - history graph,
   - wiki templates,
   - capsule facts,
   - episodic traces,
   - skill library.

3. **Fetch-and-learn**
   - denied by default.
   - requires explicit user permission or configured policy.
   - fetched material enters as untrusted until verified/provenanced.

4. **Forge**
   - no direct speech authority.
   - produces candidate CubeLang tools.
   - VM certification required.
   - promotion requires test suite pass and signed ledger entry.
   - preferably night-only at first.

5. **World plugins**
   - maze, future worlds.
   - Expose observation/action interfaces.
   - Distinguish:
     - observed facts,
     - inferred hypotheses,
     - planned actions,
     - failed/refused moves.

6. **Memory reads/writes**
   - Reads can be broad under policy.
   - Writes must be typed:
     - conversation vault entry,
     - verified episode,
     - hypothesis,
     - user preference,
     - skill candidate.
   - The model must not directly write a fact. It may propose a memory write; host validates.

## Permission model

Deny by default.

Permission decision is made by the host/orchestrator, never by the 450M model.

Policy layers:

1. **Tool availability by turn kind**
   - smalltalk cannot call forge.
   - ordinary fact question cannot write world facts.
   - maze action can call maze plugin and VM movement verifier.

2. **User/session permissions**
   - vault access.
   - explicit content gate.
   - network/fetch permission.
   - persistent memory permission.

3. **Data sensitivity**
   - encrypted conversation vault.
   - provenance restrictions.
   - “do not surface raw trace” policy.

4. **Capability risk**
   - forge and plugins require elevated mode.
   - write operations require stronger checks than reads.

## Auditing

Ledger every consequential operation:

- turn start/end,
- tool call input hash,
- tool output hash or fact refs,
- VM verdict,
- guard verdict,
- permission decision,
- spoken response hash,
- memory write,
- adapter/model version,
- threshold versions,
- failure/refusal reason.

The ledger should allow replay:

> Given the same stores, model versions, thresholds, and tool outputs, can we reproduce why this was spoken?

You do not need to store all private plaintext in the ledger. Store encrypted refs/hashes where needed.

## Where the 450M model sits

Use the 450M model in two roles, through separate adapters:

1. **Policy/proposal adapter**
   - proposes:
     - candidate plan,
     - tool call arguments,
     - clarification questions,
     - candidate response act.
   - Output must be parsed as typed JSON or CubeLang/text-to-program.
   - Invalid syntax is no-op/fail.

2. **Phrasing adapter**
   - phrases approved response acts.

The harness must never let the model decide:

- whether a fact is true,
- whether a VM threshold passed,
- whether a user permission exists,
- whether content is allowed,
- whether a memory should be promoted to fact,
- whether a tool is safe to run,
- whether an entity was correctly resolved if the context graph/host disagrees,
- whether to override a guard,
- whether to speak after a failed check.

The model may rank/propose. The host disposes.

## Bootstrap: cold start

Cold boot sequence:

```text
1. Load config and pinned versions.
2. Open stores read-only.
3. Verify store signatures and schema migrations.
4. Load VM and CubeLang tool registry.
5. Load model weights/adapters/router.
6. Run VM self-tests.
7. Run vector cleanup calibration tests.
8. Run adapter smoke tests.
9. Run host guard test suite.
10. Run permission test suite.
11. Run ledger write/read test.
12. Open stores read-write only after self-tests pass.
13. Serve only if all mandatory gates pass.
```

Refuse to serve if:
- VM tests fail,
- ledger cannot write,
- guard suite fails,
- store signatures fail,
- threshold calibration outside expected band,
- wrong-answer sentinel test produces a spoken answer.

Threshold calibration:
- Use a fixed calibration set of positive/negative bindings and absent-role controls.
- Recompute similarity distributions.
- Require separation margin.

Example estimated gate:
- Positive cleanup top-1 ≥ **99%** on calibration.
- Absent-role false positive ≤ **0.1%**.
- Margin between accepted and rejected distributions ≥ pre-registered value.

## Bootstrap: new capability/world from almost nothing

For Cubby-Man, keep four separate stores:

1. **Observed facts**
   - “Move north from cell A failed.”
   - “At t=12, wall detected ahead.”

2. **Hypotheses**
   - “There may be a wall at x,y.”
   - Not speakable as fact unless marked.

3. **Verified skills**
   - “If obstacle ahead, turn right then move.”
   - Certified by world verifier or repeated success criterion.

4. **Episodes**
   - Full traces of action, observation, result.

The loop:

```text
observe world
→ propose action
→ check action permission/safety
→ execute
→ record result
→ update observed facts
→ update hypotheses separately
→ test hypotheses when possible
→ promote to fact/skill only after criterion
```

Promotion rule example:
- A wall fact is promoted after:
  - direct collision/refused move at coordinate, or
  - sensor observation above threshold plus one confirming failed move.
- A skill is promoted after:
  - ≥ N successes across varied states,
  - no catastrophic failures,
  - VM/certifier can express its preconditions/effects.

Estimated N:
- Start with **20 successes** across at least **5 distinct local map contexts**.

## Smallest first version

Build this:

1. One orchestrator process.
2. One read-only history/fact lookup tool.
3. One VM tool.
4. One memory write: append turn trace only.
5. One response path: answer/clarify/don’t-say.
6. One task stack with budgets.
7. One signed ledger.
8. Cold boot self-test.
9. No forge in live turns.
10. No online learned router.

This is enough to replace “separate modules” with an actual state-owning loop.

## How to show the harness beats today’s separate modules

Not “it runs.” Measure:

### 1. End-to-end reproducibility

Given a turn ID, replay should reconstruct:
- selected tools,
- VM verdicts,
- guard decisions,
- response act,
- final response hash.

Gate:
- **≥ 99% replay agreement** on deterministic tools.
- Remaining cases explained by external fetch/world nondeterminism with recorded outputs.

### 2. Lower integration failure rate

Create 200 scripted multi-turn tests involving:
- pronoun resolution,
- event lookup,
- clarification,
- absent facts,
- vault read denial,
- explicit-content blocked request,
- failed VM check,
- tool timeout.

Compare current modules vs harness.

Gate:
- Harness has **≥ 50% fewer orchestration failures**, estimate.
- Zero unsafe speaks in both; if current has any, harness must eliminate them.

### 3. Better recovery

Inject failures:
- store unavailable,
- VM threshold fail,
- malformed model plan,
- ledger write fail,
- permission denied.

Gate:
- Correct refusal/degradation in **≥ 95%** of injected cases.
- If ledger write fails, no user-facing factual answer is served.

### 4. Latency budget

Gate:
- Ordinary fact turn p95 under, estimate, **2–3 seconds** local.
- No more than **+20% latency** over current pipeline unless it buys measurable reliability.

## Failure mode introduced by harness

The main new failure is **centralized policy brittleness**.

If the orchestrator schema or permission config is wrong, everything consistently fails or consistently allows the wrong thing.

Mitigation:
- policy tests at boot,
- signed config,
- deny by default,
- typed schemas,
- replay,
- staged rollout,
- “safe mode” that only answers from already verified facts.

---

# C. HDC-based FFN with linked hypervector units

## Is the idea sound?

Partly.

The sound part:

- Transformer FFNs can be viewed as **key-value memories**: hidden state matches keys; output injects values.
- Product-key memories and memory layers increase capacity by factorizing lookup.
- Modern Hopfield layers retrieve stored patterns by similarity.
- Vector-symbolic architectures/FHRR support binding, superposition, cleanup, and compositional codes.
- Spatial semantic pointers use fractional powers of base vectors to encode continuous positions.

So an HDC memory layer that does approximate associative retrieval is conceptually aligned with known mechanisms.

The risky part:

- Replacing dense FFNs in a pretrained 450M model will likely destabilize the model.
- Superposition has crosstalk limits.
- “Linked neurons” can become uncontrolled spreading activation.
- Topographic x,y,z,t anchoring sounds elegant, but may impose geometry where the language model does not need it.
- The symbolic side can read symbols, not raw vectors; bridging must still go through cleanup and verified symbol extraction.

## Recommendation

Do **not** replace the FFN yet.

Build a **side HDC associative memory layer** inserted after a few transformer layers with a zero-initialized gate:

```text
h_l
→ normal transformer layer
→ HDC memory read M(h_l)
→ gated addition: h_l + α M(h_l), with α initialized to 0
```

Train only:
- projection from model hidden state to HDC query,
- projection from retrieved HDC value back to model hidden state,
- gate α,
- optionally a small key/value adapter.

Keep the base frozen initially.

This gives you an identity path and a clean ablation:
- α = 0 is the original model.
- α > 0 shows memory contribution.

## Concrete mechanism

### Memory contents

Each memory item:

```text
key_i = HDC(subject, relation, context, maybe position/time)
value_i = HDC(object or continuation feature)
metadata_i = provenance, type, confidence, write_time
```

But the model-facing layer should not directly speak these. It retrieves a vector feature that helps the model propose; the host/VM still verifies facts externally.

### Retrieval

Given hidden state `h`:

```text
q = quantize/project(h) into FHRR space
scores_i = sim(q, key_i)
retrieve top-k or soft aggregate
r = Σ softmax(β scores_i) value_i
out = W_out(r)
```

For linked activation:

```text
r0 = retrieve(q)
r1 = cleanup(r0)
linked = retrieve(bind(r1, LINK_ROLE)) or adjacency expansion
final = r0 + λ linked
```

Keep spreading to **one hop** at first. More hops will hallucinate associations.

### Positional anchoring

Use FHRR position code:

```text
P(x,y,z,t) = X^x * Y^y * Z^z * T^t
unit_key = concept * P(x,y,z,t)
```

Similarity can fall with distance if base vectors and fractional powers are implemented carefully. But this should be tested independently. Do not assume the geometry is useful for text.

For maze/world memory, positional anchoring is natural. For general language FFN replacement, it is less obviously useful.

## How to train

Start with synthetic and internal verified data, not web text.

### Stage 1: synthetic binding/retrieval

Tasks:
- A has color red.
- B has color blue.
- Query A/color → red.
- Include distractors and counterfacts.
- Include absent-role queries.

Train:
- hidden-to-HDC projection,
- HDC-to-hidden projection,
- gate.

Measure:
- top-1 retrieval,
- absent-role silence,
- crosstalk vs number of stored items,
- latency.

### Stage 2: VM trace replay

Use episodic store of VM-certified chains.

Input:
- canonical question,
- partial reasoning state.

Target:
- next verified symbol,
- correct tool/program proposal,
- or correct abstention.

The HDC layer should help propose the next symbol/program, not decide truth.

### Stage 3: controlled language insertion

Evaluate whether it improves:
- binding values to entities,
- resisting memory-over-context,
- program emission accuracy,
- without harming held-out loss too much.

## How to insert into pretrained 450M

Use identity-safe insertion:

1. Choose 2–4 layers, likely middle layers.
2. Insert memory block with zero gate.
3. Freeze trunk.
4. Train memory projections on synthetic and trace data.
5. Unfreeze only layernorm/gates if needed.
6. Compare with equal-parameter LoRA baseline.

Do not insert into all 32 layers initially.

Estimated first experiment:
- 2 layers.
- HDC memory with **10k–100k items**.
- top-k retrieval **k=8 or 16**.
- Additional latency target under **10–15%**.

## Capacity limit

The capacity limit is crosstalk.

For high-dimensional random hypervectors, superposition capacity scales with dimension, but usable capacity depends on:
- dimensionality,
- quantization,
- similarity threshold,
- number of roles,
- number of stored items,
- cleanup vocabulary size,
- noise from model projection,
- whether you aggregate many items.

D = 10,240 is large. That helps. But your compact 80-block quantized representation and cleanup over 60k words mean errors will appear when many similar bindings share roles or entities.

Rough estimate:
- Clean symbolic superposition may tolerate thousands of random items under controlled conditions.
- Model-projected queries will be noisier, so effective capacity may be much lower, perhaps hundreds to low thousands per active memory bank.
- If you add linked spreading activation, false positives grow quickly with degree.

The critical failure mode:

> A query retrieves the right neighborhood but the wrong bound value, and because it is semantically nearby, the language model phrases it confidently.

That is exactly the failure your architecture is designed to prevent. Therefore HDC memory may propose, but the VM/store must still verify before speech.

## Does this conflict with “knowledge lives outside the weights”?

If the HDC memory is writable and inspectable as an external store, no. It is a bridge.

If it becomes opaque trained parameters containing facts, yes.

So define it as one of two things:

1. **Ephemeral/model-assist memory**
   - Helps propose symbols/tools.
   - Not authoritative.
   - Can be rebuilt from verified traces.
   - Does not count as factual knowledge.

2. **External HDC store**
   - Has provenance per item.
   - Writes are ledgered.
   - Reads return candidate symbols with similarity.
   - VM/host verifies before use.

Do not make it an unledgered learned fact cache inside the trunk.

## Cheapest experiment worth doing

Build a standalone HDC memory assist for the known base weakness:

> It binds a value copied from context to the entity asked about at chance, 52%, and answers from memory over contradicting context 22/24.

Experiment:

### Dataset

Create synthetic/contextual examples:

```text
Alice's code is 1937. Bob's code is 2841. Claire's code is 6502.
Question: What is Bob's code?
```

Include:
- English and French.
- 2, 4, 8, 16 entities per context.
- distractor world-knowledge-like facts.
- counterfactual context contradicting memorized associations.
- absent-role queries.

### Systems compared

1. Base 450M.
2. Base + LoRA r16.
3. Base + kNN/context retrieval baseline.
4. Base + HDC side memory.
5. Base + HDC side memory + VM verification.

Equalize added parameters and latency where possible.

### Metrics

- entity-value binding accuracy before host,
- verified spoken accuracy after host,
- abstention on absent role,
- contradiction obedience,
- latency,
- memory footprint,
- degradation on ordinary language held-out set.

Gate to continue:
- Binding accuracy improves from **52% to ≥ 85%** before host on 8-entity contexts.
- Contradicting-context failure drops from **22/24 to ≤ 3/24** before host, or host catches all with lower rejection cost.
- Absent-role false answer rate **≤ 1% before host**, **0 after host**.
- Latency overhead **≤ 15%**.
- No material regression in held-out loss, estimate **≤ +0.02 nats**, or no user-visible fluency regression.

If HDC does not beat kNN/context retrieval or a small LoRA under these gates, do not build the larger “linked neuron” system.

## Failure mode introduced by HDC FFN

The new failure is **associative bleed**:

- Related concepts activate each other.
- Linked activation retrieves a neighbor.
- Cleanup snaps to a valid but wrong symbol.
- The host may see a clean symbol and treat it as a serious proposal.

Mitigation:
- One-hop spreading only.
- Require provenance for external HDC items.
- Use absent-role controls.
- Return similarity and ambiguity.
- If top-1/top-2 margin is low, abstain.
- Never let HDC output bypass VM/store verification.

---

# Cross-cutting order of work

## 1. Build B first: the orchestrator

Reason:

- You currently have modules, not a single state-owning system.
- A and C need reliable evaluation, routing, logging, replay, and boot gates.
- Without B, adapter failures and HDC failures will be hard to attribute.
- B directly reduces operational risk.

First milestone:
- deterministic harness,
- boot self-test,
- typed turn loop,
- tool registry,
- ledger,
- current verified ask path ported unchanged.

Success:
- same answer quality as today,
- fewer integration failures,
- replayable traces,
- safe refusal on injected failures.

## 2. Then do A: separate grounded and conversational adapters

Reason:

- It addresses the user-visible weakness.
- It is low-risk if the orchestrator enforces acts and guards.
- It tests the actual bottleneck: capacity vs interference vs base fluency.

First milestone:
- response-act dataset,
- conversational adapter,
- unchanged grounded adapter,
- fallback to canonical form after failed checks.

Success:
- warmer English/French interaction,
- existing grounded gates preserved,
- zero wrong spoken remains true,
- reworded history questions improve.

## 3. Then do C as a research sidecar

Reason:

- It may help the base’s binding weakness.
- But it is not needed to get the system coherent.
- It can introduce exactly the kind of associative hallucination your VM architecture prevents.

First milestone:
- zero-gated HDC side memory on synthetic binding and VM traces.
- Compare against LoRA and kNN.

Success:
- large measured gain on entity-value binding and contradiction obedience under fixed latency/memory budget.
- no authority granted to HDC outputs.

---

# Final disagreement with the framing

The largest architectural risk is not that the 450M base is too small. The largest risk is **letting learned components absorb responsibilities that currently belong to the host and VM**.

A stronger local model would be useful, but your own measurements say the model is not trustworthy for grounding. Therefore the right strategy is:

- make orchestration deterministic and replayable,
- separate proposal from authority,
- separate factual transduction from conversational phrasing,
- add associative memory only as a proposal mechanism,
- validate every promotion against pre-registered gates.

If a proposed change cannot say what it is allowed to decide, what catches it when wrong, and which gate proves it helped, do not build it yet.

---

### google/gemini-3.1-pro-preview

Here is my assessment of the three problems, adhering to your constraints for concrete mechanisms, specific failure modes, and measurable validation.

### A. Fluent without losing capacity, at 450M

**Premise Check:** The premise that fluency and grounding inherently compete in the base model is likely wrong. The base model is fluent in prose (as shown by your probes); the bottleneck is the 7.9M parameter adapter trying to map strict VM outputs to *both* rigid forms and fluid conversational prose. 

**Recommendation:** **Two separate adapters (LoRA r16, ~7.9M each) switched by the host per turn.** 
Keep the current adapter strictly for fact-reporting. Train a second adapter exclusively for conversational fluency (small talk, "tell me more", clarifications). The host, which already reads the question kind, routes the turn to the appropriate adapter.

**Reasoning:** You already have a frozen tag router and a host that classifies intents. Forcing one small adapter to handle disjoint distributions (strict VM-bound facts vs. fluid ungrounded chat) causes catastrophic interference. Two adapters cleanly separate the tasks without increasing the active parameter count during a forward pass (maintaining your 12GB/40 tok/s budget).

**Failure Mode:** The host misclassifies the turn type. If it routes a fact question to the chat adapter, the model will hallucinate because the chat adapter isn't trained to strictly obey the VM's value check. 

**Validation:** 
*   *Measurement for success:* The fact adapter maintains its current grounding metrics (98.8% relation, 0 wrong answers behind host). The chat adapter achieves a lower perplexity (estimate: -0.4 nats) on held-out conversational data compared to the unified adapter.
*   *Measurement for failure:* Host routing accuracy drops below 99%, or latency spikes beyond your 40 tok/s budget due to adapter swapping in VRAM (though Vulkan should handle two 8MB LoRAs trivially).

### B. A bootstrap and an orchestrator: the whole system as an agent harness

**Premise Check:** You ask if the 450M model should propose tool calls as a policy. It absolutely should not. At 450M, with a 52% bind chance, it cannot be trusted with autonomous, unconstrained planning. 

**Recommendation:** **A strict State-Machine Harness in Python (the Host).** 
The LLM acts *only* as a translation layer. The loop is strictly: `User Input -> Host parses intent -> LLM proposes CubeLang (via program adapter) -> VM executes/verifies -> Host checks constraints -> LLM phrases output (via talk adapter)`. Multi-step tasks live in the Host's memory as a stack of unresolved sub-goals. Runaways are stopped by a hard limit (e.g., max 3 VM hops per turn) enforced by the Host.

**Reasoning:** The system's core principle is "the host disposes, a verifier decides." If the LLM acts as the agentic policy, you violate this by letting weights dictate logic. The Host must own the state machine. Tools are declared as Python functions wrapping VM calls, permissioned by the Host's explicit-content gate, and audited by writing every VM verdict to the signed ledger before the talk adapter is invoked.

**Failure Mode:** The harness becomes too rigid, failing to handle ambiguous user inputs that require multi-step clarification, resulting in infinite loops of "I don't understand" or immediate fallbacks to the user.

**Validation:** 
*   *Measurement for success:* Task completion rate on a benchmark of 500 multi-step queries (requiring 2+ VM hops). Success is >85% completion with exactly 0 unverified claims reaching the user.
*   *Measurement for failure:* >5% of queries hit the hard 3-hop turn limit (timeout/runaway), indicating the LLM's proposed CubeLang programs are thrashing.

### C. An HDC-based FFN whose units react like linked neurons

**Premise Check:** Replacing the dense FFN with an HDC-based memory trained via gradient descent conflicts directly with your rule that "knowledge lives outside the weights." If you bake hypervectors into the FFN weights, you are putting knowledge back into the weights.

**Recommendation:** **Do not replace the FFN; augment the architecture with a read-only Product-Key Memory (PKM) layer tied to the hypervector store.**
Place this right before the final projection. Project the residual stream to the D=10,240 hypervector space, perform a nearest-neighbor lookup against a frozen dictionary of your 60,151 FHRR block codes, and project the retrieved vector back into the residual stream. 

**Reasoning:** This acts as a natural bridge between the trunk and the stores without destroying the base model's learned representations. It allows the symbolic side to read the exact hypervector retrieved, maintaining the strict boundary of what crosses into the VM. Fractional binding for spatial/temporal encoding (X^x * Y^y) happens in the external store, not in the model's weights.

**Failure Mode:** Crosstalk in superposition. At D=10,240, capacity is high, but dense clusters of similar concepts will cause the nearest-neighbor lookup to retrieve blended, noisy vectors, destroying the exactness required by the VM.

**Validation (The cheapest experiment):** 
Freeze the 450M base. Train *only* the projection matrices (down to D=10,240 and back) on a next-token prediction task using a fixed dictionary of 10,000 hypervectors. 
*   *Measurement for success:* The held-out loss drops compared to the base model alone.
*   *Measurement for failure:* Retrieval accuracy of the correct hypervector falls below 90% due to crosstalk, proving the continuous residual stream cannot reliably target discrete FHRR codes.

### Order of Execution

1.  **B (The Harness):** You must build the orchestrator first. Without the state machine and the strict turn loop, you cannot reliably test A or C in an end-to-end environment. It is the foundation of the system.
2.  **A (Fluency via Two Adapters):** This is a low-risk, high-reward engineering task. You already have the data and the routing logic. It immediately solves the user-facing UX problem.
3.  **C (HDC-based FFN/PKM):** Do this last. It is a high-risk research bet that requires modifying the forward pass and could destabilize the 450M base you just spent 16.8 A100-hours training.

---

### moonshotai/kimi-k3

# Review: CubbyLLM, three problems

General note first, because it colors everything below: your single most important measured fact is not in the three problem statements. It is that the base binds a copied value to the asked entity at chance (52%) and prefers memory over a contradicting context 22/24 times. A, B, and C all look different once you take that seriously.

---

## A. Fluency at 450M

**The premise is mostly wrong.** "Fluency and grounding compete for the same 7.9M parameters" is an artifact of putting two different *tasks* into one adapter, not a law about capacity. Look at your own architecture: the fact path is a form-filler — canonical restatement in, one sanctioned value out, three host moves. The conversational path you want is open-ended dialogue over the user's actual words. Different inputs, different outputs, different correctness criteria. When you mixed them at 11% vs 63%, you measured a swing of ~5 net right answers (384 vs 389 of 407) and 18→6 wrong-but-grounded. That is a small, real interference effect — not evidence that 7.9M parameters can't hold both, and certainly not evidence that the base can't.

**Recommendation: two adapters, frozen fact path, and the load-bearing work is a new guard, not an adapter topology.**

1. Freeze the talk adapter exactly as it is at the 63% mix. It passes gates with 0 wrong behind the host. Do not touch it.
2. Train a separate chat adapter on dialogue data (frontier-generated is allowed for training). The critical design constraint: this adapter is trained to *never assert world facts*. It acknowledges, asks, rephrases, expresses partial ignorance, offers to check. If a turn needs a fact, it hands back to the host, which runs the fact path. This is what keeps "the VM decides" intact on a path where there is no VM value to check against.
3. The new guard — this is the piece your option list is missing. The value check works because fact turns have a VM-returned value. Chat turns don't. So define a per-turn **sanctioned entity set**: entities and numbers the user named, plus whatever the context graph licenses for this turn. The claim check extends to: every named entity and number in the reply must be in the sanctioned set or tied to facts as today. Without this, your chat adapter — sitting on a base that answers from memory 22/24 times — will eventually state a confident, fluent, ungrounded fact, and your current claim check (which only covers "things named from the facts") has a hole exactly there.

**Failure mode this introduces:** misrouting. A fact question routed to the chat adapter gets a fluent non-answer or, worse, a fluent memory-based answer that the extended claim check must catch. The frozen tag router becomes a safety component, not a convenience.

**Measurements (pre-registered):**
- Fact gates re-run with the chat adapter present: must be identical to today (zero weight sharing, so any change indicates a serving bug — this is nearly free to check and decisive about interference).
- Routing confusion matrix on a held-out turn-kind set; gate: <1% of fact turns routed to chat (estimate of achievable: 0.2–0.5%).
- Adversarial red-team: ~500 chat turns designed to elicit memory-based assertions (asking opinions about entities in the wiki world, "are you sure X isn't Y?"). Gate: 0 unsanctioned assertions behind the host.
- Dialogue quality: frontier-model-as-judge (allowed for tests) on naturalness, plus French dialogue loss measured separately — if French was thin in the 2.77B tokens, French fluency is a pretraining-mix problem and no adapter will fix it. Check tokenizer fertility for French first; it's a one-hour measurement.

**On the other options:** rank 64 only if you measure underfitting (chat adapter training loss plateauing high); r16 across 181 projections is very likely enough for style — estimate: rank is not the constraint, data is. A mode line reintroduces interference in one adapter and adds a mode-confusion failure; skip. Rank-wise gating and per-layer rank are second-order tuning; skip. **Moving the scaffold into the weights: don't.** That is scaffold distillation onto a base that demonstrably asserts from memory over context. You would be training the model to internalize the behavior of being checked without the checking. The benefit is latency; the cost is a more confident ungrounded model and a higher guard load. Revisit only if the binding probe (see C) is ever fixed.

**The fuller "I don't know" is not a fluency problem at all.** "I don't have what caused it — I know it was 1683" is a partial-slot rendering: the VM returns some roles filled, one absent. Extend the canonical restatement schema to carry partial results and extend the value check to verify each stated slot independently. That is a data-format change plus a small fine-tune of the *existing* fact adapter, measurable on a partial-facts suite. Cheapest win in this whole document.

**What to measure first, to separate base vs adapter vs interference:** (1) base-only dialogue continuation loss (estimate: ~2.6–2.9 given held-out 2.36 — if much worse, fluency belongs in the continued-pretraining mix you're already running); (2) the frozen-fact-adapter + separate-chat-adapter gate re-run, which settles interference for almost no cost. Do those two before any topology experiments.

---

## B. The harness

The framing is right, and I'll add urgency: today you have a sleep cycle writing episodes, skills, and training data while a serving pipeline reads the same stores, with no single owner of state. That is a store-corruption risk *now*, not a future tidiness issue.

**1. The loop.** One step: observe (turn + context graph) → host classifies turn kind (frozen router) → plan proposed as a CubeLang program or typed tool sequence → VM executes/verifies every hop → guards → speak/ask/refuse → append to signed ledger + event stream. Multi-step tasks live as a DAG in the context graph, each node carrying a **VM-checkable success predicate**; the loop pops nodes, and a task is done when its root predicate verifies, not when the model says so. Runaway stops: per-turn and per-task budgets — estimate ~32 tool calls/turn, ~200 VM hops/task, but set them from p99 of your existing chain statistics, not from my numbers — plus deny-by-default tool ACLs. Budget exhaustion is a declared outcome ("I ran out of steps on this"), never a silent truncation.

**2. Tools.** Declared with typed signatures (input roles, output symbol) and an **effect class**: pure-read / proposes-write / writes. Certification (for forge tools) covers the program in isolation; the harness enforces effect classes *at call time*, because the composition gap — two certified tools whose composition does something neither was certified for — is the real hole. Every call logged with inputs, outputs, verdict. Audit = replay: given the ledger and a store snapshot, an auditor re-verifies every spoken claim's chain. Gate: 100% of a sampled day's spoken claims replay to a VM verdict.

**3. Where the 450M sits.** Both proposing and phrasing, through separate adapters, with the host in between — the program emitter proposes tool calls as CubeLang (VM-checked by construction), the talk/chat adapters phrase. The harness must never let it decide: what is true, whether a chain verified, threshold values, routing-table changes, tool certification, vault access, fact-vs-hypothesis promotion, or refusal overrides. This is not a new principle for you; the harness just makes it enumerable and enforceable in one place.

**4. Bootstrap.** (i) Cold start order: verify ledger hash chain → verify store integrity hashes → load adapters (hash-checked) → run the pre-registered gate battery as preflight → serve, or refuse. One correction to your framing: **do not "calibrate thresholds" at boot.** A boot process that can lower thresholds to pass its own self-test is a self-sealing hole. Thresholds are signed config; preflight passes or refuses. Test refusal by fault injection: corrupt each component in turn; gate is 100% refusal, zero silent degraded serves. (ii) Capability-from-nothing (Cubby-Man): the mechanism is verified traces → episodes → mined skills → certified tools, with hypothesis/fact separation you already have. The failure mode is arbiter gaming: RPE-reordered proposers converging on chains that clear the VM threshold while being semantically empty. Measure: audit samples of high-reward chains for content, and track the threshold-margin distribution over time — margins clustering just above threshold is the signature of gaming.

**5. Smallest version and honest proof.** One process that boots with preflight, runs the existing ask loop, owns the sleep scheduler, writes the ledger. Nothing new capability-wise. It beats today's modules on three measurable axes: (a) fault injection — kill any module mid-turn 100 times; gate: 0 lost or double-spoken turns per the ledger; (b) a multi-step task suite (~50 pre-registered tasks like "find X, then tell me what caused it," requiring chained verified walks) — today's modules need a human gluing steps; gate: ≥80% success (estimate of achievable), 0 unchecked utterances; (c) cold-start-to-serve time with full preflight. If it only passes (c), you built plumbing, not a harness — say so in the report.

---

## C. The HDC FFN

**As an FFN replacement: the premise "more capacity per parameter" is very likely wrong, and you can kill it cheaply.** Your FFNs hold ~134M parameters. At int8 that's ~134MB; a full FHRR vector at 10,240 phases × 2 bytes is ~20KB, so the same budget buys ~6,700 hypervectors. Superposition capacity per block code: with 128 phases per block, crosstalk variance grows as N/128, giving roughly ~5 reliably decodable items per block against a large cleanup memory, so a few hundred associations per full vector (estimate, and note your 80-byte compact form is one phase index per block — it encodes a *single item*, not a superposition; the set memory needs the full representation). That yields on the order of 1–3M associations (estimate) — versus a dense FFN whose keys are *learned* to tile the training distribution. Random codes spend their capacity on orthogonality; learned keys spend it on the data manifold. Product-key memory and modern Hopfield layers are the relevant evidence: a similarity-readout memory layer is functionally attention with a fixed codebook, and attention with learned keys won. Your own growth experiment — every grown arm behind on all 9 sources after an equal hour — is the template for how this will go.

**The (x,y,z,t) anchoring is sound and known** — it is exactly fractional binding / spatial semantic pointers (Komer, Frady, Eliasmith's line of work), with a sinc-like similarity falloff. But its natural home is Cubby-Man's episodic *store*, where graded spatial similarity ("have I been near here") is genuinely useful. That requires no trunk changes at all.

**The one version of C worth building is neither of those.** Your base fails at binding (52%). Gradient training has not given it a binding mechanism; an explicit associative memory in the forward pass might. So: augment *one* layer with a small HDC key-value memory that is **written only from the current context and cleared between sequences** — a VSA context cache, not knowledge — inserted behind a zero-initialized gate (your exact-growth trick, so loss is identical at step 0). Train one hour on copy/bind-heavy data.

**Failure modes:** crosstalk ceiling (capacity curve bends fast); two-hop spreading activation amplifies noise — each hop multiplies noise by roughly sqrt(fan-out · N/D) (estimate), so gate any prototype on two-hop precision, not one-hop; loss regression from random-code inefficiency; and the governance hole if anyone is tempted to let gradient writes store world facts in it.

**On "knowledge lives outside the weights":** it conflicts only if the memory holds world knowledge. Draw the line explicitly: the memory may hold (a) ephemeral per-turn context bindings, (b) a cache of VM-certified facts with provenance. Never gradient-written assertions. Under that rule it is a bridge; without it, it's a smuggling route.

**Cheapest decisive experiment (pre-registered):** binding probe 52% → target ≥90%; held-out loss regression ≤ +0.005 nats; copy loss unchanged at ~1.2. One rented GPU-hour scale. If binding doesn't move, kill C entirely and put the (x,y,z,t) codes in the maze store where they belong. If it does move, you have found something more valuable than an FFN: a fix for the trunk's one measured cognitive deficit.

---

## Order

**B-minimal first.** It is small, it fixes a live integrity risk (unowned state, concurrent store writes), and it creates the preflight-and-ledger surface that A's new guard and every future gate must register in. Everything else becomes more measurable the day it lands.

**A second.** The fact path is already safe (0 wrong behind the host), so A is about experience, not safety — but it is the largest user-visible gap, and its real work (the sanctioned-entity claim check) is harness plumbing that B just built. The partial-slot "I don't know" sub-item can ship immediately, before B finishes, since it touches only the existing fact path.

**C last, and only as the killable binding-probe experiment.** Its expected value is speculative, its safe form depends on B's discipline, and its most likely positive outcome (fixing binding) would retroactively change the A conversation — a trunk that binds might eventually earn the right to more in-weights behavior. Not before.

One sentence of disagreement to leave with: the system is better verified than it is capable, and the binding probe is the gap — fund the experiment that attacks it directly rather than the ones that route around it.

---

### x-ai/grok-4.6

**A — Two host-switched adapters; fluency is mostly a base-data problem, not a 7.9M LoRA problem.**

The mix numbers already show **task interference in one adapter**, not “450M cannot talk.” Behind the host you already have **0 spoken wrong**. The form-like adapter (canonical restatement, one value, 63% one-sentence facts) is doing its job. Forcing small-talk through the same 7.9M LoRA is the wrong premise.

**Do this:** keep the current talk adapter as **grounded-talk** (facts / “don’t say” / named history). Add a second LoRA **chat** (same r16, ~7.9M), trained only on English/French conversation, corrections, “tell me more,” fuller refusals. The **frozen host** already classifies turn kind; it picks adapter **per turn**. Never a learned mixture online. Do **not** grow LoRA to r64, insert blocks, or MoE until this is measured — those spend params and Vulkan complexity on an unseparated interference problem.

**Why not “put fluency in continued pretrain only”:** you should **also** continue the 450M on the 3.2B unseen tokens (growth already lost; target stays 450M). That is where **varied wording and French** belong. The adapter still must not be asked to be both a form-filler and a chatter.

**Failure mode:** host mis-routes a fact question to chat → fluent ungrounded speech. Mitigate: chat adapter still goes through **name/number, value, claim** checks; if VM/facts path is empty, only the existing three host moves. Chat never emits a number/name unless the value check passes.

**Validate (pre-register):**
- Switch accuracy on held-out turn labels (estimate: need ≥98% or you will leak; estimate).
- Fact families **in grounded mode** stay ≥ current 63%-mix numbers (98.8 / 100 / 93.8 / 89.1) and **0 wrong behind host**.
- Chat mode: human or scripted diversity (unique trigrams / turn, follow-up without event name) vs today’s form; **0 wrong behind host**.
- Program-emitter VM-pass rate unchanged (same interface).

**Measure first to split causes:** freeze talk adapter; continue base 1 equal hour on language-heavy mix; re-run (i) copy/bind/contradict probes, (ii) adapter-on-frozen-base fact families. If bind stays ~52% and contradict-from-memory stays high but **base** prose improves, fluency is **data/base**. If fact families drop when you later mix chat into **one** adapter, interference is **adapter**. (Estimate: 1 GPU-hour is enough to see the same +0.03–0.08 nat pattern you already saw.)

Scaffold-into-weights is wrong under your rules: it trains the model to skip the host.

---

**B — One process: observe → host plan → permissioned tools → VM → speak/ask → ledger. Model phrases and proposes CubeLang; it never decides truth or permissions.**

**One step:**  
`obs` (user + sensors + hormone state) → host builds plan (graph resolve, event lookup) → **tool calls** → each result through CubeLang/VM or store API → host choose {answer, clarify, refuse} → **value+claim guards** → speak → append signed ledger + history graph.

Multi-step / subgoals live in the **host plan**, not in the 450M. Stop runaway: max hops (estimate **8** VM hops / turn), wall-clock (estimate **2 s** decode budget at 40 tok/s), and **absent-role must stay silent**. Night is a scheduled job in the same process, not a second brain.

**Tools:** declare as `(name, schema, permission class, audit tag)`. Default **deny**. Classes: read-store, VM-exec, fetch-and-learn, forge (output must VM-certify before install), world plugin, vault (never to talk adapter). Every call writes the ledger. Guards you already have are the permission layer; do not replace them with a learned router.

**Where 450M sits:**  
- **Program emitter** proposes CubeLang (stand-in until yours exists).  
- **Talk adapter** phrases **only** host-canonical + VM value.  
It must **never** decide: which fact is true, whether to speak, fetch, forge install, vault open, or plugin side effects.

**Bootstrap (i):** load stores + adapters → run a **fixed gate pack** (estimate **50** items covering fact families + 0-wrong) → calibrate similarity thresholds on a frozen set → **refuse to bind port** if any spoken-wrong or VM/host mismatch.  
**(ii) Capability from nothing:** same as Cubby-Man: two facts, refused moves → graph, hypotheses ≠ facts, forge only if VM certifies. Growth = verified traces only.

**Smallest v1:** one process wrapping today’s turn loop + cold-start gate pack + sleep cron + ledger on every speech.  

**Beats modules:** not “it runs.” Pre-register: (1) gate pack pass rate = today’s modules, (2) **100%** spoken utterances have a ledger hash, (3) kill -9 and restart → same graph hashes and **refuse-if-gates-fail**, (4) time-to-first-token after boot (estimate target **<15 s** on 12 GB). If (2) or (3) fail, harness is theatre.

**Failure mode:** tool API that lets the model skip VM (“just lookup and talk”). That reintroduces the 22/24 memory-over-context failure into speech.

---

**C — Not sound as an FFN replacement at 450M. Cheapest test is an **off-net** FHRR memory, not a trunk edit.**

FFNs-as-key-value, product keys, modern Hopfield, SSP/FHRR position codes are real. **Linked spreading inside a SwiGLU slot** is not a drop-in: your trunk is MinGRU + window-512 attn; Vulkan batch-1; knowledge is **supposed** to live in stores. Writing associations into weights **does** conflict with “knowledge outside weights” unless the block is an **explicit cache of capsule/wiki vectors** with the same cleanup the VM already does — then it is redundant with stores + cleanup.

**Capacity:** superposition crosstalk. For FHRR D=10,240, random item memory, reliable cleanup is on the order of **O(D / log D)** to a small multiple of D depending on density — estimate **~10³–10⁴** clean items before similarity collapses (estimate; measure, don’t believe). Sequences via \(X^x Y^y Z^z T^t\) add more collisions. Failure mode: **wrong cleanup** that looks like a high similarity — exactly what the VM’s absent-role control is for, now inside the net, **unchecked** if you read vectors not symbols.

**Do not insert into pretrained 450M** until the off-net test works. Training-by-gradient on hypervector units will fight quantised block codes and your 2.77B-token trunk.

**Cheapest experiment (hours, one GPU or CPU):**  
Fixed 60,151-word table you already have. Store N random (key,value) FHRR pairs, N = 1e3, 3e3, 10e3, 30e3. Query with noise. **Pass:** cleanup to correct value ≥95% at N=10k with your 80×128 codes. **Fail:** <90% at N=3k, or any case where top-1 is wrong but sim > your VM hop threshold. If fail, stop. If pass, next is **readout to symbols only** (same as VM), still **outside** the trunk.

---

**Order:** **A measurement split (continue base + two adapters)** → **B harness v1 (gates + ledger + boot refuse)** → **C only if A/B are green.**  
A tells you where fluency lives before you spend serve complexity. B makes every later number reproducible and stops the model deciding truth. C is research with a known crosstalk fail; it must not delay 0-wrong speech.

**Framing disagreements:** (1) Fluency vs grounding is not primarily “same 7.9M.” (2) Growing the base is already measured-wrong at your budget. (3) An HDC FFN is not the natural bridge; the bridge is **symbols + VM + stores**. (4) “Experts / rank-wise gating / identity blocks” are premature until two frozen host-switched adapters are measured.

---

### deepseek/deepseek-v3.2

## Recommendations

### **A. Fluent without losing capacity**

**Recommendation:** Use **two separate adapters** (factual and conversational) switched by the host based on turn type, and **continue pretraining the base model** with a mixed curriculum that gradually introduces conversational data while preserving canonical form responses.

**Reasoning:**
- The bottleneck is both the base's language ability *and* adapter interference. The base is undertrained (2.77B tokens for 450M params ≈ 6 tokens/param; typical is 100+). Its loss (2.36) indicates poor language modeling, limiting fluency regardless of adapter.
- Interference is real: the adapter's 7.9M parameters are tasked with both strict factual generation (63% of data) and free-form conversation, causing competition. Separate adapters isolate tasks.
- Continued pretraining on diverse text (not just factual Q/A) improves foundational language skill, which benefits both adapters. However, care must be taken to preserve the base's ability to follow the host's canonical restatements—this can be done by mixing in scaffolded examples during pretraining.
- The host already classifies turns (fact lookup vs. conversation), so routing is natural. The conversational adapter can be trained on data that includes original wording and follow-ups, but its output must still pass the host's checks (e.g., no ungrounded claims). This maintains safety.

**Failure mode:** Host misroutes a conversational turn as factual, leading to stilted replies; or routes a factual turn to the conversational adapter, generating ungrounded but fluent answers that might still pass the host's checks (e.g., naming grounded entities incorrectly). The host's value and claim checks may not catch all such errors if the answer is plausible but wrong.

**Validation measurement:**
- Pre-registered gate: On a test set of 500 mixed turns (250 factual, 250 conversational), measure:
  1. **Factual accuracy** for factual questions (must stay ≥98%).
  2. **Fluency score** (human rating on a 1–5 scale, or use a proxy like token-level entropy of the response under a held-out fluent model—but no external LLM, so human eval is required).
  3. **Wrong answers spoken** (must remain 0).
  4. **Routing accuracy**: percentage of turns where the host correctly chooses the adapter (audit via logs).
- Thresholds: Factual accuracy ≥98%, fluency score ≥4.0 for conversational turns, routing accuracy ≥99%. If any threshold is missed, the dual-adapter approach fails.

**First diagnostic:** Measure the base model's perplexity on a conversational corpus (e.g., a cleaned subset of the training data used for the conversational adapter). If perplexity is high (>3.0), the base is the bottleneck, and continued pretraining is necessary before adapter changes.

---

### **B. Bootstrap and orchestrator**

**Recommendation:** Build a **single orchestrator process in Rust** that owns the turn loop, state, and scheduling. It exposes the host's capabilities as tools with a permission system, and uses the model only for proposal generation, never for decisions.

**Reasoning:**
- The current modular separation leads to coordination overhead and deadlock risks. A unified orchestrator ensures deterministic turn execution and clear failure boundaries.
- The model's role must be restricted: it proposes tool calls (via the program emitter adapter) and phrases results (via talk adapters), but the host decides and the VM verifies. The orchestrator enforces that no tool is executed without VM certification.
- Cold start bootstrap should be self-validating: load stores, run sanity checks (e.g., VM can bind/unbind, host resolves context), and refuse service if any check fails.
- The smallest viable version is a thin wrapper that calls existing modules in the correct sequence, adding tool registration and a signed ledger. This already beats today's system by eliminating inter-process communication and ensuring atomic turns.

**Failure mode:** The orchestrator becomes a single point of failure—if it crashes, the entire system halts. Also, if tool permissions are incorrectly configured, the model could propose harmful actions that get approved (though the VM should catch them). Complexity in scheduling day/night cycles could lead to resource exhaustion.

**Validation measurement:**
- Pre-registered gate: Run the orchestrator for 48 hours on a simulated workload (mixed user queries and maze exploration). Measure:
  1. **Uptime**: must remain responsive (latency <2s per turn) with no crashes.
  2. **Tool safety**: 0 unauthorized tool executions (audit ledger).
  3. **Multi-step task success**: complete a previously impossible task (e.g., "explore the maze, find the red door, and describe what's behind it") by chaining tool calls.
  4. **Night cycle completion**: sleep cycle runs without error, producing new skills/episodes.
- Thresholds: uptime 100%, tool safety 100%, multi-step task success ≥90%, night cycle errors 0. If any fail, the orchestrator is insufficient.

**Key design:**
- One step = observe → host resolves context → model proposes plan → host selects/verifies plan → execute tools with VM → host decides speak/ask/refuse → record.
- Multi-step tasks live as goal stacks in the host's planning module, with timeouts to prevent runaways.
- Tools are declared in a Rust enum, each with a VM-certified implementation and permission flags (e.g., `read_fact`, `write_episode`). The host checks permissions against a policy file.

---

### **C. HDC-based FFN**

**Recommendation:** **Do not pursue this now.** Instead, run a cheap simulation to estimate capacity and interference, then reconsider.

**Reasoning:**
- The idea is theoretically sound—hyperdimensional associative memories have been studied (e.g., sparse distributed memory, modern Hopfield networks). However, integrating it into a pretrained transformer is fraught:
  - The FFN operates on d=1024 hidden vectors, while HDC uses D=10,240. Projecting between spaces adds parameters and may lose information.
  - Training would require fine-tuning the entire base (violating the frozen base constraint) or adding it as a side network, which could disrupt existing representations.
  - The rule "raw vectors never cross" is preserved if the VM decodes the HDC memory's output, but that adds latency.
- Capacity estimate: with D=10k and FHRR, the number of storable uncorrelated patterns is roughly D/(2 log D) ≈ 500–1000. That's likely insufficient for language model needs.
- Failure mode: Crosstalk from superposition degrades recall, and the non-differentiable cleanup operation blocks gradient flow, making training unstable.
- It conflicts with "knowledge outside weights" if it becomes a primary knowledge store; it should only serve as a working memory bridge.

**Cheapest experiment:**
- Simulate a standalone HDC memory (Python) with D=10,240, storing random patterns and testing recall under noise. Measure retrieval accuracy as a function of stored patterns.
- Then, replace **one** FFN layer in the base with an HDC layer (project input to HDC space, query memory, project back). Train only this layer on the binding task (value→entity) using the existing data.
- **Validation measurement:** If binding accuracy improves from 52% to >80% without harming loss (increase <0.1 nats), consider further work. If not, abandon.

**Alternative:** Use HDC as an external working memory module accessed by the host, not inside the model. This aligns with "knowledge outside weights" and can be tested without modifying the base.

---

## **Execution Order**

1. **First, address B (orchestrator).** A reliable harness is foundational—it enables controlled experiments for A and C, and ensures system stability. The orchestrator can be built incrementally without changing the model.
2. **Simultaneously, continue pretraining the base** for A, as this takes GPU time but minimal engineering. Once the base's perplexity drops (<2.0 on conversational data), implement the dual-adapter system.
3. **Postpone C** until after A and B are stable. Run the cheap simulation in the background; if results are promising, consider a small-scale integration later.

**Why this order:** The orchestrator reduces integration debt, enabling faster iteration on model improvements (A). The base pretraining is a slow burn that should start early. The HDC idea is speculative and risks distracting from core goals.

---

## **Disagreements with Framing**

- **Problem A's premise** that fluency and grounding compete is partially wrong: the competition is due to insufficient base capacity and mixed training data, not an inherent trade-off. A larger base (e.g., 1B+ parameters) would likely handle both, but the 450M constraint forces compromises. However, within 450M, continued pretraining and task separation can mitigate.
- **Problem C's hope** that HDC will "bridge the trunk and stores" overlooks the complexity of integrating two fundamentally different representations (dense transformer activations vs. sparse hypervectors). The stores are already symbolic; a simpler bridge is to use the VM to decode transformer outputs (as currently done). HDC inside the weights adds unnecessary indirection.

---

## **Summary**

- **A:** Dual adapters + base continued pretraining. Measure fluency and grounding separately.
- **B:** Rust orchestrator with tool permissioning. Measure reliability and multi-task capability.
- **C:** Simulate first, then maybe test one layer. Measure binding accuracy and loss.
- **Order:** B → A → C, with base pretraining in parallel with B.

---

