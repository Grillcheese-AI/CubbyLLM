# Panel, round 2: the trunk as a compiler of branch programs (2026-09-28)

Round 1: `docs/panel_fluent_harness.md`. Same six models, asked independently, temperature 0.3, reasoning effort `low`. **Model opinions, not findings.**

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

# Round 1, in brief (six reviewers, asked independently)

- A (fluency at 450M): all six said two adapters switched by the host per turn (the grounded talk
  adapter frozen as it is; a separate conversation adapter), before any larger LoRA, mixture or inserted
  block, and a diagnostic first (fact-only vs chat-only vs mixed adapters at equal tokens). Several
  added: continued pretraining is where fluency comes from; the chat path needs its own guard (every
  entity and number it names must be in a per-turn sanctioned set, since it has no VM value to check);
  a partial answer ("I don't have the cause - it was 1683") is a slot format for the fact adapter, not
  fluency. Most said not to distil the host's scaffold into the weights.
- B (harness): one deterministic process; typed tools with effect classes, deny by default, every call
  ledgered; a task stack or DAG with budgets; boot preflight that refuses to serve on a failed gate, with
  thresholds as signed config (not recalibrated at boot); the model proposes (programs) and phrases, never
  decides truth, permissions, memory writes or tool safety. Four of six: build a minimal harness first.
- C (HDC FFN): not as an FFN replacement (crosstalk; learned keys beat random codes); the x, y, z, t
  anchoring is sound (spatial semantic pointers) but belongs in the stores and Cubby-Man's episodic
  memory; the version worth one experiment is a zero-gated associative memory in one layer, written from
  the current context only, aimed at the measured binding deficit (52%).

# D. The trunk as a compiler of branch programs (the owner's goal)

What the owner wants: the model reads a question, lays out the alternatives (choices a, b, c) against the
contexts they are judged in (timelines or eras x, y, z: "what would change if X had not happened", "in
1900" vs "in 2028"), and emits one CubeLang program per branch. The VM executes each branch in isolation,
and what comes back (the verdict, how far it ran, what changed) steers what is emitted next. Today
counterfactual history questions ("what would change if X had not happened") are answered by following the
history graph's links downstream of X: 18% spoken and correct on reworded questions, the weakest kind.

A proposal the owner received from another model (summarised): a CfC-style "liquid" module takes the
question embedding, a [choices] tensor and a [timelines] tensor, broadcasts them into a
[choices x timelines] grid, fuses each cell with an MLP, applies a continuous-time gate
(state = f*g + (1-f)*h, with f = sigmoid(...) described as the time constant tau), and projects each
cell to a FIXED-LENGTH block of opcode logits (max_code_len x vocab) that are argmax-decoded into
bytecode and run in a VM sandbox. The time constant is said to set how deep a branch runs; faulty branches
"adjust the weights via backpropagation" at runtime. It also asserted, wrongly, that in our system "only
matrices cross into the VM" (the rule is the opposite: only symbols cross), and asked whether the liquid
parameters should update during runtime or only in the nightly loop.

The VM as it is (CubeLang, from its source and spec):
- A typed scripting language compiled to a 0x00-0xFF bytecode: CREATE, ASSIGN, arithmetic, PUSH/POP,
  QUERY, STORE/RECALL, BIND_ROLE, UNIFY, COND and jumps to labels, CALL (intra-program, deny-by-default
  resolution of modules), RETURN, arrays; plus VSA ops (bind, unbind, cleanup against a codebook).
- Registers, stack, storage and accumulator are VM-global; there are no call frames. LOOP is currently a
  structural marker executed as a single pass (iterating a collection is a documented follow-up). Every
  jump spends a 1,000,000-jump budget. A debug trace of executed ops exists.
- ASK suspends a program when it has grounded several real candidates and cannot choose; the host answers
  and the program resumes, and resume rejects anything that was not offered (the host may choose, never
  supply). The VM runs as a resident process over protobuf on stdio, verifies a program before executing
  it, and returns (symbol, similarity) per hop, plus verified or refused with a reason.
- The spec describes ctx.snapshot / restore / fork / merge (try a branch in an isolated copy, then
  merge or discard it); fork is specified, not implemented.
- The 128k tokenizer holds CubeLang opcodes as atomic tokens. The program emitter is trained on
  VM-verified programs (517 verified chain programs plus arithmetic, role-binding and kernel families) and
  is today served by a stand-in model behind the same interface.
- The trunk's recurrent layers (MinGRU-style) already update as a gated interpolation of the kind the
  proposal calls liquid; they have no explicit time input.

Questions:
1. Is "the trunk compiles a question into branch programs, the VM runs each branch, the outcome steers
   the next emission" sound for this system? Give the right split of work between the model (what it
   emits), the host (who enumerates branches, sets budgets, compares outcomes) and the VM (what it runs,
   what it returns).
2. Critique the proposal's mechanics specifically: fixed-length opcode logits per grid cell, argmax
   decoding, a gate standing in for execution depth, runtime backpropagation. Keep what is right.
3. The "liquid" part: does a time-conditioned gate in the trunk (CfC-style, time as an input) buy anything
   for reasoning across eras, compared with putting time into the data and the VM (a timeline as a role
   bound in the program, T^t position codes in the stores)?
4. Feedback: what exactly should the VM return per branch, and how should it train the emitter - at
   night through the gated promotion rule, or ever at runtime? Our rule is that serving never changes
   weights.
5. Where this sits in the order you would do A, B, C in, the smallest first version, and its pre-registered
   gate. Object to round 1's consensus if you think it is wrong.

Constraints as before: no external LLM at serve time; 450M target; local serving on a 12 GB GPU, batch 1;
the host disposes and the VM verifies; only symbols cross into the VM; every claim measurable.
```

</details>

## Answers (6 of 6 returned)

### anthropic/claude-opus-4.6

## 1. The split is sound, but the model should not enumerate branches.

The trunk emits a **single CubeLang program that contains the branch structure**—forks, comparisons, ASK suspensions. The host does not enumerate branches; it does what it does now: canonicalise the question, resolve references, set budgets, and refuse or pass through. The VM executes the program, hits a fork (ctx.fork when you build it), runs each branch in isolation, and returns per-branch verdicts. The trunk then receives a structured result (not raw vectors) and emits the next program or a final answer draft.

**Why the host should not enumerate branches:** The host doesn't know the causal structure of the history graph. "What if X hadn't happened" requires following X's downstream links—that's the VM's job via the program. The host can cap the number of forks (budget) and reject programs that request too many, but the branching logic belongs in the emitted code.

**The split:**
- **Model:** reads the host's canonical question + VM return from the previous step; emits one CubeLang program (which may contain forks and ASK points).
- **Host:** canonicalises, resolves references, sets a fork budget (estimate: 4 branches max initially), passes VM returns back as structured text for the next emission, applies all existing guards before speaking.
- **VM:** executes, forks context, runs each branch to its verdict or budget exhaustion, returns per-branch: (verdict, similarity, hops_used, hops_budget, terminal_symbol, reason_if_refused). ctx.fork must be implemented.

**Validation:** On your 18%-correct reworded counterfactual set, measure spoken-and-correct. Gate: ≥ 40% spoken-and-correct with 0 spoken-and-wrong behind the host. Failure: if the emitter can't produce valid forking programs at > 50% parse rate after training on ~200 verified fork examples, the approach stalls at the emitter, not the architecture.

## 2. The proposal is mostly wrong.

**Fixed-length opcode blocks per grid cell:** Wrong. CubeLang programs vary in length and structure. A fixed-length block either truncates real programs or wastes capacity on padding. Autoregressive token-by-token emission (which you already do) is correct; the tokenizer already has CubeLang opcodes as atomic tokens. Keep autoregressive emission.

**The [choices × timelines] grid with MLP fusion:** Wrong framing. The choices and timelines are not known before the program runs—they're discovered by walking the history graph. A question like "what if the printing press hadn't been invented" doesn't come with a pre-enumerated timeline tensor. The program discovers downstream effects. Kill the grid.

**Argmax decoding:** Acceptable for CubeLang (you want deterministic programs, not sampled ones), but only token-by-token with VM syntax checking after each statement, not over a fixed-length block.

**A gate standing in for execution depth:** Wrong. Execution depth is a runtime property of the VM (how many hops clear threshold before one fails or the budget runs out). A learned gate cannot substitute for this; it would be a hallucinated prediction of what the VM would do. The VM already returns hop count and budget—use that.

**Runtime backpropagation:** Violates your serving rule. Kill it.

**What is right:** The intuition that branch programs should be generated from a compact representation of the question, and that branch outcomes should steer subsequent generation. That's just multi-turn program emission with VM feedback, which your architecture already supports.

## 3. The liquid gate buys nothing here.

Your MinGRU layers already have gated interpolation. Adding an explicit time input to the gate (CfC-style) would help if the trunk needed to represent temporal dynamics internally—but it doesn't. The trunk's job is to emit programs; temporal reasoning happens in the VM via bound timeline roles and in the stores via T^t position codes.

**Concrete argument:** The 52% binding deficit is between entity and value, not between entity and time. A CfC gate conditions on continuous time, but your history questions involve discrete eras retrieved from a graph, not continuous time evolution. The graph already has temporal links.

**What would change my mind:** If you measured that the emitter systematically produces wrong temporal orderings in its programs (e.g., checking post-event state before removing the event), and that this correlates with era distance. I estimate this is not your bottleneck—your bottleneck is the emitter producing valid fork programs at all.

## 4. VM returns and training.

**Per branch, the VM should return (all as symbols/numbers, never vectors):**

```
branch_id: int
verdict: verified | refused | suspended | budget_exhausted
terminal_symbol: string (the answer or the point of failure)
similarity: f32 (of the terminal cleanup)
hops_completed: int
hops_budget: int
reason: string (if refused/suspended)
offered_choices: [string] (if suspended via ASK)
```

**Training the emitter:** Nightly only, consistent with your rule. The process:
1. During serving, log (canonical_question, emitted_program, per_branch_VM_return) to the signed ledger.
2. Nightly, the forge generates candidate programs for failed questions, the VM certifies them, certified programs join the training set.
3. The emitter is retrained/fine-tuned nightly on the expanded set of VM-verified programs.
4. Promotion rule: a new emitter checkpoint replaces the serving one only if it matches or exceeds the old one on the pre-registered gate (parse rate, verdict rate, spoken-correct rate on the held-out counterfactual set).

**Never at runtime.** The striatum-like reordering of plan proposers is acceptable (it reorders, doesn't retrain), but even that should not change the emitter's weights.

**Validation:** Track nightly the count of verified fork programs in the training set. Gate: emitter parse rate ≥ 80% and VM-verified rate ≥ 50% on held-out questions before promoting to serve. If after 4 nightly cycles the verified fork corpus hasn't grown past ~50 programs, the forge can't write them and you need hand-written templates first.

## 5. Order, smallest first version, and one objection.

**Order: B → A → (this) D, with C folded into A's diagnostic.**

Round 1 consensus is right that B (harness) comes first—you literally said "no single process owns boot, state, scheduling and the turn loop," and D requires multi-step VM interaction that a harness must manage. A (two adapters) comes next because the talk adapter's form-like answers are the bottleneck users see. D comes after because it depends on both (the harness for multi-turn VM orchestration, the conversation adapter so the final answer isn't robotic).

**One objection to round 1:** The C recommendation (zero-gated associative memory for the 52% binding deficit) is misplaced effort. The binding deficit is a base model problem at 2.77B tokens with held-out loss 2.36—it's undertrained. You're already continuing on 3.2B unseen tokens. Measure binding accuracy after continuation before adding architectural complexity. I estimate continuation alone brings binding from 52% to ~70%, which still isn't enough, but the right next step is the talk adapter's training data (entity-value pairs as explicit supervised signal), not an architectural patch to the base.

**Smallest first version of D:**

1. Implement ctx.fork/merge in the VM (it's already specified).
2. Hand-write 20 fork-program templates for counterfactual history questions (remove event X, walk downstream, compare).
3. Train the emitter on those 20 + the existing 517, nightly.
4. Measure: on your existing counterfactual test set, what fraction of emitted programs parse, what fraction the VM verifies, what fraction yield spoken-correct answers behind the host.

**Pre-registered gate:** Counterfactual questions, reworded (the 18% set). Metric: spoken-and-correct ≥ 35%, spoken-and-wrong = 0 behind the host. If after the emitter is trained on 50+ verified fork programs the spoken-correct rate is still below 25%, the bottleneck is not program structure but history graph coverage (the downstream links don't exist for those events), and you should measure that directly.

---

### openai/gpt-5.5

## Short answer

Yes: **“model emits candidate branch programs; VM runs them; host uses VM outcomes to choose the next step”** is sound for CubbyLLM, but only if the **branching is owned by the host**, not by a learned liquid grid module.

The right architecture is:

- **Model/emitter:** proposes small, typed CubeLang programs or program fragments from a constrained grammar.
- **Host:** enumerates branches, eras, counterfactual assumptions, budgets, and comparison criteria; schedules VM runs; decides whether to answer, ask, refuse, or request another program.
- **VM:** verifies and executes isolated symbolic programs; returns a structured branch report: verified/refused, trace summary, facts touched, symbols returned, similarity margins, causal frontier reached, unresolved choices, budget use, and deltas inside the forked context.

Do **not** do fixed-length argmax opcode blocks, do **not** let a learned gate stand in for execution depth, and do **not** backpropagate at runtime. The “liquid” proposal mostly confuses neural sequence generation with symbolic execution. The only part worth keeping is the idea of a **branch grid** over choices × timelines — but that grid should be a **host-side task matrix**, not a learned tensor whose cells decode bytecode independently.

The smallest useful version is not a new trunk module. It is:

> **Host-side branch runner + VM fork/restore + constrained emitter for branch query programs + pre-registered branch-report metric.**

I would do it after the minimal harness from B, and before any HDC-FFN experiment from C. It also depends on A’s split-adapter plan if you want decent phrasing, but not for the reasoning core.

---

# 1. Is “trunk compiles question into branch programs; VM runs each branch; outcome steers next emission” sound?

## Yes, with a stricter split of responsibilities

The premise is sound if “compiles” means:

> The model proposes candidate CubeLang programs under a schema, and the host/VM reject most forms of freedom.

It is unsound if “compiles” means:

> The model autonomously invents branch structure, execution depth, branch comparison semantics, and final truth conditions.

CubbyLLM’s existing principle is correct: **model proposes, host disposes, verifier decides.** Counterfactual branch reasoning should preserve that exactly.

---

## Correct split of work

### A. Model/emitter responsibilities

The emitter should produce **program skeletons**, not free bytecode blobs.

For a counterfactual question such as:

> “What would change if X had not happened?”

the emitter may propose something like:

```cubelang
PROGRAM counterfactual_downstream
INPUTS event_x, timeline_t, max_depth

ASSUME_NOT(event_x, timeline_t)
frontier = QUERY downstream(event_x, timeline_t)
FOR_EACH_CAUSAL_STEP(frontier, max_depth):
    candidate = QUERY causal_successor(frontier.item, timeline_t)
    VERIFY_EDGE(frontier.item, candidate, timeline_t)
    EMIT_CHANGE(candidate)
RETURN changes
```

But practically, because CubeLang currently lacks real iteration and fork is not implemented, first version should avoid general loops. Use bounded unrolled templates:

```cubelang
BIND_ROLE(event, X)
BIND_ROLE(timeline, T)
QUERY downstream_1(event, timeline)
QUERY downstream_2(symbol_from_previous, timeline)
QUERY downstream_3(symbol_from_previous, timeline)
RETURN collected_symbols
```

The emitter should output:

1. **Program template ID**, from a small host-approved set.
2. **Bindings**, e.g. event symbol, timeline symbol, relation type.
3. **Depth request**, e.g. 1, 2, 3 hops, not arbitrary.
4. **Ask points**, where ambiguity is allowed.
5. **Expected return schema**, e.g. list of changed events with causal links.

It should not emit unconstrained raw opcode sequences in the first version.

Estimated starting library: **10–30 branch program templates** is enough. More than that before evaluation will hide failure modes.

---

### B. Host responsibilities

The host should own all branching.

For a question, the host should construct a **branch job**:

```text
Question: what would change if X had not happened?
Root event: X
Timelines: [actual, counterfactual_without_X]
Branch assumptions:
  actual: X occurred
  counterfactual_without_X: NOT X
Budget:
  max_hops: 3
  max_nodes_per_hop: 20
  max_vm_steps: N
  max_ambiguities: 2
Comparison:
  report symbols downstream of X present in actual but unsupported in counterfactual
Answer policy:
  answer only verified deltas with causal path evidence
  otherwise say facts don't support a counterfactual difference
```

The host decides:

- Which branches exist.
- Which timeline/era symbols are legal.
- Which event is X.
- How many hops to search.
- Whether branch outputs are comparable.
- Whether to ask the user to disambiguate.
- Whether evidence is sufficient to speak.

The host should not ask the model:

> “How deep should this branch run?”

The model can propose a depth, but the host should clamp it.

Concrete estimate: first version should use **max depth 2 or 3**, because history graphs explode combinatorially and your present counterfactual accuracy is only 18% on reworded questions. A depth-5 free brancher will likely look impressive while silently accumulating garbage.

---

### C. VM responsibilities

The VM should run programs inside isolated symbolic contexts.

The VM should provide:

1. **Verification before execution.**
2. **Execution under budget.**
3. **Symbol-only inputs and outputs.**
4. **No raw vector crossing.**
5. **Per-hop similarity and margin reports.**
6. **Trace summary.**
7. **Fork/restore semantics for counterfactual branches.**

For this goal, implementing `ctx.fork / restore / discard` matters more than any new trunk mechanism.

The counterfactual assumption “X did not happen” should not mutate the real store. It should exist only in a branch context:

```text
actual_ctx = snapshot()
cf_ctx = fork(actual_ctx)
cf_ctx.assert_absent(event_X)
run downstream program in cf_ctx
discard cf_ctx
```

Until `fork` exists, simulate branches host-side by passing an explicit `timeline` / `assumption_set` role into every query. But the real fix is VM-level isolated branch contexts.

---

## One concrete mechanism I recommend

Build a **Branch Task Runner** in the host.

### Branch Task Runner interface

Input:

```json
{
  "question_id": "...",
  "root_event": "EVENT_X",
  "branch_type": "counterfactual_absence",
  "branches": [
    {
      "branch_id": "actual",
      "assumptions": ["EVENT_X_OCCURRED"],
      "timeline": "ACTUAL"
    },
    {
      "branch_id": "without_x",
      "assumptions": ["NOT EVENT_X"],
      "timeline": "CF_WITHOUT_X"
    }
  ],
  "max_depth": 3,
  "max_nodes": 50,
  "program_template": "bounded_downstream_causal_walk_v1"
}
```

Emitter output:

```json
{
  "template_id": "bounded_downstream_causal_walk_v1",
  "bindings": {
    "root_event": "EVENT_X",
    "relation": "causes_or_enables",
    "timeline": "$BRANCH_TIMELINE"
  },
  "depth": 2,
  "return_schema": "causal_delta_list"
}
```

VM returns one report per branch.

Host compares reports and generates a canonical answer frame:

```text
Question: What would change if EVENT_X had not happened?
Verified actual downstream effects: A, B, C.
Verified counterfactual unsupported/preserved effects: ...
Safe answer slots:
  - changed: [A, B]
  - unchanged/unknown: [C unknown]
  - confidence reason: verified causal paths of length <= 2
```

The talk adapter then phrases only the canonical frame.

---

## Validation

### Measurement that would show it works

Pre-register a held-out set of counterfactual/history questions.

Separate categories:

1. Event named exactly.
2. Event reworded.
3. Era/timeline contrast: “in 1900 vs 2028”.
4. Absence counterfactual: “if X had not happened”.
5. Ambiguous root event requiring ASK.

Metrics:

- **Spoken-correct rate**.
- **Spoken-wrong rate.**
- **Refusal/ask rate.**
- **Correct root event resolution.**
- **Correct causal path evidence rate:** every spoken delta has a verified path in the branch report.
- **Branch isolation violations:** any counterfactual assumption contaminates the actual store.
- **Trace reproducibility:** same signed config + same stores gives same branch report.

Gate estimate:

Current counterfactual reworded questions: **18% spoken and correct**.

First useful gate, estimated:

- Named-event counterfactuals: **≥60% spoken-correct**, **≤1% spoken-wrong**.
- Reworded counterfactuals: **≥35% spoken-correct**, **≤1% spoken-wrong**.
- Every spoken change must have a VM-returned causal path: **100% guard pass**.
- Branch isolation violation: **0**.

What would show it does not work:

- Spoken-correct rises but spoken-wrong exceeds 1–2%.
- The model emits plausible deltas not present in VM branch reports.
- Reworded failures are mostly root-event resolution errors, meaning the branch runner is not the bottleneck.
- Depth-3 branches produce many low-margin symbolic cleanups and wrong causal attributions.

---

# 2. Critique of the proposed mechanics

The proposal has one good intuition and several bad mechanics.

## What is right

The useful part is:

> Represent the problem as choices × timelines / branches.

For example:

```text
choices:   [X happened, X did not happen]
timelines: [actual, 1900, 2028, hypothetical_without_X]
cells:     branch jobs
```

That is a good **host scheduling abstraction**.

But it should not be a dense neural tensor that directly decodes bytecode per cell. It should be a typed symbolic task table.

---

## Fixed-length opcode logits per grid cell

This is the wrong interface.

CubeLang programs are structured objects with labels, jumps, calls, type constraints, and return schemas. A fixed `max_code_len × vocab` block encourages exactly the failure mode you cannot tolerate: superficially valid bytecode with broken control flow, bad labels, invalid stack discipline, or nonsensical calls.

Problems:

1. **Length is semantic in programs.** Padding and truncation are not harmless.
2. **Labels and jumps require consistency.** Independent token logits do not guarantee this.
3. **VM-global registers and stack make malformed programs dangerous or misleading**, even if sandboxed.
4. **The training set is tiny:** 517 verified chain programs plus families. That is nowhere near enough to learn robust raw bytecode generation.
5. **Argmax fixed blocks remove uncertainty**, so the host cannot distinguish “model is unsure” from “model confidently emitted garbage”.

Better mechanism:

- Emitter outputs a typed AST or template call.
- Compiler lowers typed AST to bytecode.
- VM verifies bytecode.
- Invalid AST/program becomes a training negative.

Concrete first version:

```json
{
  "template": "causal_walk_bounded",
  "args": {
    "root": "EVENT_X",
    "relation": "downstream_causal",
    "timeline": "ACTUAL",
    "max_depth": 2
  }
}
```

Not:

```text
[0x13, 0x44, 0x2a, 0xff, ...]
```

Validation:

- Program-verify rate on held-out program tasks.
- Execution-success rate.
- Correct-return-schema rate.
- Spoken-answer correctness downstream.

Gate estimate:

- Template/AST verify rate should be **>95%** on in-distribution tasks before serving.
- Raw bytecode argmax would likely be far below that; if it is not, require adversarial label/jump tests.

---

## Argmax decoding

Argmax is also wrong.

For natural language, sampling can be useful. For verified programs, you want **constrained decoding plus repair/retry**, not free argmax.

Use:

1. Grammar-constrained decoding.
2. Type-constrained slots.
3. Beam of small size, maybe **3–5** candidates, estimate.
4. VM verification chooses.
5. Host ranks by execution result, not model logprob alone.

Failure mode:

- Beam search may find programs that verify but are irrelevant.
- Therefore include a task relevance check: did the returned symbols match the requested root event, timeline, and relation?

Validation:

- Compare argmax vs constrained beam:
  - VM verification rate.
  - Correct task completion.
  - Wrong spoken rate after host.
- If constrained beam improves verification but not correctness, the issue is semantic relevance, not syntax.

---

## Gate standing in for execution depth

This is conceptually wrong.

A neural gate may control how much hidden state changes. It does not define how many causal hops a branch has executed. In CubbyLLM, execution depth is:

- Number of graph hops.
- Number of VM calls.
- Number of verified symbolic transitions.
- Budget consumed.
- Similarity/margin per cleanup.

Those are symbolic runtime quantities, not hidden-state interpolation constants.

The proposal’s “tau sets how deep a branch runs” is not compatible with your VM design.

Keep depth explicit:

```text
max_hops = 2
max_nodes_per_hop = 20
max_jumps = 1,000,000
min_similarity = threshold
min_margin = threshold
```

Validation:

- Depth-1/2/3 ablation on counterfactual benchmark.
- Measure correctness versus graph explosion and refusal rate.
- If deeper runs increase unsupported deltas or low-margin cleanups, clamp depth.

---

## Runtime backpropagation

No.

This violates the serving rule and is architecturally dangerous.

Runtime backprop would make answers non-reproducible, create ledger/debugging problems, allow user interaction to modify future behavior, and blur the boundary between proposal and verification.

Your current rule is right:

> Serving never changes weights.

At runtime, the system may update:

- Conversation context.
- Ephemeral branch contexts.
- Ledger.
- Provenance records.
- Candidate training examples for sleep.

It must not update:

- Trunk weights.
- Adapter weights.
- Emitter weights.
- Router weights.

Validation:

- Hash all served weights at boot and periodically during serving.
- Gate: **zero hash drift** across serving.
- Any training example produced at runtime must have a signed VM verdict and remain quarantined until nightly training.

---

# 3. Does CfC-style time conditioning buy anything for reasoning across eras?

## Probably not for this system

The premise is mostly wrong: “time-conditioned neural gate” is not the right representation for historical era reasoning in CubbyLLM.

Your trunk already has MinGRU-style gated interpolation. Adding explicit continuous-time gates may help sequence modeling in some domains, but the bottleneck you measured is not temporal dynamics. The bottlenecks are:

- Entity/value binding at chance: **52%**.
- Memory overriding contradictory context: **22/24** failures.
- Counterfactual graph reasoning: **18% spoken-correct** on reworded questions.
- History event resolution: **59% reworded vs 90% named**.

Those are not solved by a continuous-time hidden-state mechanism.

For this architecture, time should live in:

1. **The stores**, as timeline/era roles.
2. **The VM programs**, as explicit bound symbols.
3. **The host branch runner**, as branch metadata.
4. **Spatial/temporal semantic pointers**, if useful for retrieval.
5. **The context graph**, for resolving “then”, “in that era”, “before that”.

Example:

```cubelang
BIND_ROLE(event, EVENT_X)
BIND_ROLE(timeline, YEAR_1900)
BIND_ROLE(location, PLACE_Y)
QUERY fact(event, timeline, location)
```

Or with temporal powers:

```text
T^1900 * EVENT_X
T^2028 * EVENT_X
```

That is inspectable and VM-verifiable. A neural time gate is not.

---

## When would a time-conditioned trunk be justified?

Only if you show a serving-relevant failure that persists after symbolic time roles are implemented.

For example:

- The emitter reliably confuses program templates for “before”, “after”, “during”, “in era X”, despite explicit time roles and training examples.
- A time-conditioned adapter improves program emission correctness without increasing unsupported claims.

Even then, I would add time conditioning to the **program emitter adapter**, not the whole trunk.

Validation:

Run an ablation:

1. Baseline emitter.
2. Baseline + explicit symbolic timeline roles in prompt/schema.
3. Baseline + T^t store encoding.
4. Time-conditioned neural gate.

Metrics:

- Correct timeline binding.
- Correct before/after relation.
- Correct branch separation.
- Spoken-wrong rate.

A CfC gate is justified only if it beats explicit symbolic time by a material margin. Estimate threshold: **≥5 absolute points** on timeline-binding correctness with no increase in wrong spoken answers. I doubt it will.

---

# 4. What should the VM return per branch, and how should it train the emitter?

## VM branch return schema

The VM should return more than “verified/refused”. For branch steering and training, it should return a structured report.

Recommended branch report:

```json
{
  "branch_id": "without_x",
  "program_id": "bounded_downstream_causal_walk_v1",
  "program_hash": "...",
  "input_bindings": {
    "root_event": "EVENT_X",
    "timeline": "CF_WITHOUT_X",
    "assumptions": ["NOT EVENT_X"]
  },
  "verdict": "verified | refused | ask | budget_exhausted | partial",
  "refusal_reason": null,
  "ask": {
    "slot": null,
    "offered_candidates": []
  },
  "returned_symbols": [
    {
      "symbol": "EVENT_A",
      "role": "changed_event",
      "similarity": 0.91,
      "margin": 0.12,
      "path": [
        {
          "from": "EVENT_X",
          "relation": "causes",
          "to": "EVENT_Y",
          "source_fact": "FACT_123",
          "similarity": 0.94,
          "margin": 0.15
        },
        {
          "from": "EVENT_Y",
          "relation": "enables",
          "to": "EVENT_A",
          "source_fact": "FACT_456",
          "similarity": 0.91,
          "margin": 0.12
        }
      ]
    }
  ],
  "touched_facts": ["FACT_123", "FACT_456"],
  "touched_events": ["EVENT_X", "EVENT_Y", "EVENT_A"],
  "deltas": [
    {
      "kind": "unsupported_in_counterfactual | preserved | contradicted | unknown",
      "symbol": "EVENT_A",
      "evidence_path_ids": ["..."]
    }
  ],
  "budgets": {
    "vm_steps_used": 1234,
    "jumps_used": 12,
    "hops_used": 2,
    "nodes_expanded": 9
  },
  "trace_digest": "...",
  "full_trace_ref": "ledger://...",
  "branch_context_delta_hash": "...",
  "store_mutation": "none | ephemeral_only | attempted_forbidden"
}
```

Important fields:

### 1. Verdict

Use a richer verdict set:

- `verified`
- `partial`
- `ask`
- `refused_type_error`
- `refused_unbound_symbol`
- `refused_low_similarity`
- `refused_absent_role_spoke`
- `budget_exhausted`
- `forbidden_effect`
- `branch_isolation_violation`

### 2. Similarity and margin

Similarity alone is not enough. You also want the margin to the next nearest symbol.

A cleanup with similarity 0.82 and margin 0.01 is dangerous. A cleanup with 0.82 and margin 0.25 may be acceptable.

### 3. Evidence path

Every spoken counterfactual delta should cite a path, even if the user does not see it.

### 4. Delta type

Counterfactual outputs should not just be “things downstream”. They should be classified:

- Present in actual and absent/unsupported in counterfactual.
- Present in both.
- Contradicted.
- Unknown due to missing facts.
- Branch refused.

### 5. Budget use

If a branch says “nothing changed” after exhausting depth 1, that is not the same as “nothing changed” after a complete depth-3 search.

The host needs this distinction to phrase safely:

- “The verified graph does not show a change within two causal hops.”
- Not: “Nothing would have changed.”

---

## How branch reports train the emitter

Training should happen at night, never at runtime.

Runtime flow:

1. Emitter proposes candidate program(s).
2. VM verifies/executes.
3. Host logs full branch reports.
4. Host maybe asks or answers.
5. Ledger signs outcomes.
6. Sleep process converts outcomes into training examples.

Nightly training examples:

### Positive examples

Programs that:

- Verified.
- Returned correct schema.
- Produced evidence paths.
- Led to a correct spoken answer or correct refusal.
- Did not require host rescue beyond allowed ASK.

### Negative examples

Programs that:

- Failed verification.
- Used wrong timeline.
- Queried wrong relation.
- Produced low-margin cleanups.
- Exhausted budget due to bad search.
- Returned irrelevant but verified symbols.
- Caused the host claim-check to reject the answer.

Train the emitter with:

- Supervised fine-tuning on successful AST/template outputs.
- Preference/ranking loss: successful program > refused/irrelevant program for same task.
- Hard negative classification: identify why a candidate failed.

Do not train only on “verified” versus “not verified”. A program can verify and still answer the wrong question.

---

## Promotion rule

You said the adapter lifecycle detector and promote rule are not built. For branch-program emission, define a narrow promotion gate.

Candidate emitter is promoted only if, on a frozen benchmark:

- VM verify rate: **≥95%**.
- Correct return schema: **≥95%**.
- Task relevance: **≥90%**, estimated first target.
- Spoken-wrong behind host: **≤1%**.
- Counterfactual named-event spoken-correct improves by **≥15 absolute points** over current.
- Reworded counterfactual spoken-correct improves by **≥10 absolute points** over current.
- No increase in branch isolation violations: **0 allowed**.
- No regression above **2 absolute points** on ordinary fact/history QA.

These numbers are estimates, but the principle matters: promote only if it improves branch tasks without weakening the existing fact safety.

---

# 5. Where this sits relative to A, B, C; smallest first version; gate

## Order I would do

I mostly agree with Round 1.

My order:

1. **B: Minimal deterministic harness.**
2. **D-min: Branch runner + VM fork/restore + constrained branch templates.**
3. **A: Split talk/conversation adapters and chat guard.**
4. **C: Zero-gated associative memory experiment**, only after the above.

Slight nuance: A and D can partly proceed in parallel because D’s core output can be canonical forms, not fluent prose. But if forced to serialize, build B first.

---

## Why B first

No single process owns boot, state, scheduling, and the turn loop. That is a bigger risk than the counterfactual weakness.

Branch reasoning multiplies execution paths. Without a deterministic harness, you will not know whether failures come from:

- Emitter.
- Host branch enumeration.
- VM execution.
- Context graph.
- Store retrieval.
- Adapter phrasing.
- Budget differences.
- Config drift.

B is the substrate that makes D measurable.

Minimum B gate:

- One deterministic process owns boot, config, state handles, scheduling, and turn loop.
- Signed thresholds loaded from config, not recalibrated at boot.
- Typed tools with effect classes.
- Deny by default.
- Every call ledgered.
- Reproducible answer for same stores/config/input.
- Boot refuses to serve if required gates fail.

Gate:

- **100% reproducibility** on a fixed 500-turn replay, except explicitly nondeterministic fields disallowed.
- **0 unledgered tool/VM calls.**
- **0 config threshold mutations during serve.**
- **0 serving with failed preflight.**

---

## Why D before C

The HDC associative memory idea targets the measured binding deficit, which is real. But the counterfactual question here is not primarily an FFN/memory issue. It needs:

- Explicit branch contexts.
- Timeline roles.
- Causal graph traversal.
- Branch comparison.
- VM trace reports.

A neural memory insertion will not give you branch isolation or causal delta semantics.

Do C later, narrowly, as Round 1 said: one zero-gated associative memory in one layer, written from current context only, aimed at binding deficit. But do not confuse it with counterfactual execution.

---

## Relationship to A

A’s consensus is right: split the grounded talk adapter from a conversation adapter. Do not distil host scaffolding into weights.

For D, the important A-related issue is phrasing. Counterfactual answers are dangerous because natural language invites overclaiming:

Bad:

> “Without X, Y would not have happened.”

Safe:

> “In the verified graph, Y is downstream of X. When the branch assumes X absent, the system cannot verify Y through the same path.”

That is a host canonical answer format, not a general fluency problem.

So the first D version can use form-like answers.

---

# Smallest first version of D

## Build this

### 1. Implement or simulate branch isolation

Best: implement VM `ctx.snapshot / fork / restore / discard`.

If too large, simulate first with explicit branch assumption symbols, but still ledger branch contexts separately.

Pre-registered hard requirement:

- A counterfactual branch must not mutate global VM registers/storage in a way visible to the next branch, except through explicit returned report.
- Given your VM currently has global registers, stack, storage, and accumulator with no call frames, branch isolation needs real attention.

Failure mode:

- Branch A leaves accumulator/register state that affects Branch B.
- Counterfactual assumption contaminates actual branch.
- A later answer sees a hypothetical as fact.

Validation:

- Branch contamination tests:
  - Run A then B.
  - Run B then A.
  - Run A alone and B alone.
  - Reports must match their isolated runs.
- Gate: **0 contamination failures** in a fixed suite.

---

### 2. Create 5 branch templates

Start with five, not fifty.

Recommended first templates:

1. `actual_downstream_walk_depth_1`
2. `actual_downstream_walk_depth_2`
3. `counterfactual_absence_downstream_depth_1`
4. `counterfactual_absence_downstream_depth_2`
5. `timeline_fact_compare`

Do not start with open-ended “what would change?” across arbitrary depth.

---

### 3. Host enumerates branch table

For a user question, host does:

```text
resolve root event
if ambiguous: ASK
select branch type
instantiate actual and counterfactual branches
run templates under budget
compare branch reports
produce canonical answer frame
```

The model may help map question type to branch type, but host must verify the resulting type against grammar and context.

---

### 4. Emitter emits template calls only

Input to emitter:

```text
canonical question:
  type: counterfactual_absence
  root_event candidates: [...]
  era/timeline candidates: [...]
allowed templates:
  [...]
```

Output:

```json
{
  "template": "counterfactual_absence_downstream_depth_2",
  "root_event": "EVENT_X",
  "timeline": "CF_WITHOUT_X"
}
```

If emitter gives anything outside allowed set, reject.

---

### 5. VM returns branch reports

Use the schema above, even if some fields are empty in v1.

---

### 6. Host canonical answer

Examples:

If verified delta:

> “The verified graph shows EVENT_A downstream of EVENT_X through EVENT_Y. In the branch where EVENT_X is absent, that path is not verified. So the supported change is: EVENT_A is no longer established by that path.”

If insufficient:

> “The facts do not support a specific change. I found downstream links from EVENT_X, but the counterfactual branch did not verify an alternative outcome.”

If ambiguous:

> “Which event do you mean by X: EVENT_1, EVENT_2, or EVENT_3?”

Avoid:

> “EVENT_A would definitely not have happened.”

Unless your facts explicitly encode necessary causation, which most history graphs will not.

---

# Pre-registered gate for D-min

Create a frozen benchmark before building.

Suggested size estimate:

- **200 named-event counterfactuals**
- **200 reworded-event counterfactuals**
- **100 timeline comparison questions**
- **100 ambiguous questions requiring ASK**
- **100 negative-control questions where facts do not support a counterfactual answer**

Total: **700 questions**, estimated enough to see large effects.

Metrics:

1. **Root event resolution**
   - Named: ≥95%
   - Reworded: ≥75% first target

2. **Branch program verification**
   - ≥95% for named
   - ≥90% for reworded where root resolved

3. **Branch isolation**
   - 0 failures

4. **Spoken-correct**
   - Named counterfactual: ≥60%
   - Reworded counterfactual: ≥35%
   - Timeline comparison: ≥75%

5. **Spoken-wrong**
   - ≤1% overall behind host

6. **Unsupported causal claims**
   - 0 spoken claims without branch path evidence

7. **Ask behavior**
   - Ambiguous questions: ≥90% ask rather than guessing

8. **Negative controls**
   - ≥90% “facts don’t say” or equivalent safe refusal

What would show failure:

- It improves named questions but not reworded ones: root resolution/context graph is the bottleneck.
- It returns many “verified” branches but host refuses: templates are irrelevant or branch reports lack useful deltas.
- Spoken-correct rises with spoken-wrong >1%: answer policy is too permissive.
- Counterfactual answers overstate necessity: causal semantics are too weak.

---

# Specific objections to the owner’s goal

The goal is good, but I would rephrase it.

Bad version:

> “The trunk compiles a question into branch programs.”

Better version:

> “The host turns a question into a typed branch task; the emitter proposes one or more template-constrained programs for each branch; the VM verifies and executes them; the host compares branch reports.”

The trunk should not own branch enumeration. If it does, you will get plausible missing branches, duplicated branches, or branch labels that do not correspond to VM assumptions.

Also, be careful with “what changed” in history. Your graph can usually support:

- “This is downstream of X.”
- “This path is not verified when X is removed.”
- “The facts do not establish whether Y would still occur.”

It often cannot support:

- “Y would not have happened.”
- “Z would replace Y.”
- “The world would be like W.”

Unless the world model encodes necessary/sufficient causal relations, counterfactuals must be phrased as **verified graph deltas**, not metaphysical alternate history.

This is not a wording nit. It is the difference between a verified symbolic system and a fiction generator.

---

# Final recommendations

1. **Reject the fixed-length liquid opcode-grid design.**
   Keep only the host-side branch grid idea.

2. **Implement branch isolation before model changes.**
   `ctx.fork / restore / discard` is more valuable than a CfC gate.

3. **Use constrained template/AST emission, not raw bytecode argmax.**

4. **Have the host enumerate branches and budgets.**
   The model may propose programs; it must not decide truth, depth, branch existence, or final comparison.

5. **Return rich VM branch reports.**
   Include verdict, symbols, similarity, margin, path evidence, touched facts, budget use, trace digest, and deltas.

6. **Train only at night.**
   Runtime produces signed examples and preference pairs. Serving weights never change.

7. **Do not add neural time conditioning yet.**
   Put time into roles, stores, branch contexts, and VM programs. Test that first.

8. **Do B first, then D-min, then A polish/conversation split, then C.**
   If parallel work is possible, A can proceed independently, but D needs B’s deterministic harness to be measurable.

The smallest viable counterfactual system is not a new recurrent module. It is a **deterministic branch runner over verified symbolic programs** with conservative language around causal deltas.

---

### google/gemini-3.1-pro-preview

Here is the review of the branching compiler proposal (D), addressing your questions directly.

### 1. Soundness and the Split of Work
The premise "the trunk compiles a question into branch programs, the VM runs them, the outcome steers the next emission" is **sound**, but the orchestration belongs in the host, not the model. 

**The Split:**
*   **Model (Proposer):** Reads the question and context, and emits *one* CubeLang program representing a single hypothesis or a generator of hypotheses. It does not manage state or grids.
*   **Host (Orchestrator):** Parses the question into discrete choices/timelines. It commands the VM to `fork` its state, feeds the model's program into the VM for each branch with different bound variables (e.g., era = 1900), catches the `ASK` or `RETURN` yields, merges or discards the VM states, and formats the surviving facts into a prompt for the talk adapter.
*   **VM (Verifier):** Executes the bytecode in the sandboxed fork. It enforces the 1,000,000-jump budget and returns the exact VSA bindings `(symbol, similarity)` and a termination status (Success, Refused, Budget Exceeded).

### 2. Critique of the Proposal's Mechanics
The proposal you received is fundamentally flawed and misunderstands both your architecture and how code generation works.
*   **Fixed-length opcode logits per grid cell:** **Wrong.** Programs have variable lengths and branching logic. Emitting a fixed-length block of logits destroys the autoregressive dependencies required for syntactically valid code. The model must emit tokens autoregressively.
*   **Argmax decoding:** **Weak.** While standard for greedy decoding, code generation benefits from constrained decoding (masking invalid CubeLang opcodes at each step based on the VM's grammar).
*   **Gate standing in for execution depth:** **Wrong.** Execution depth is a discrete, deterministic property of the VM's execution path, not a continuous neural activation. A sigmoid gate cannot simulate a Turing-complete VM's loop budget.
*   **Runtime backpropagation:** **Wrong.** This explicitly violates your rule that serving never changes weights. It also introduces catastrophic instability.

### 3. The "Liquid" Part (Time-Conditioned Gates)
A time-conditioned gate (CfC-style) buys you **nothing** for reasoning across historical eras. 

In neurosymbolic systems, "1900" vs "2028" is a relational and symbolic constraint, not a continuous dynamical system time-step. Time belongs in the data and the VM. You should represent timelines as roles bound in the program (`bind(event, time_era)`) and use fractional power encoding ($T^t$) in the hypervector stores. The MinGRU layers already handle sequence routing; do not pollute the trunk with continuous-time ODEs for symbolic logic.

### 4. Feedback and Training the Emitter
**What the VM returns per branch:** A structured payload containing: 1) Exit status (Success, Error, Timeout), 2) The final accumulator value `(symbol, similarity)`, 3) A hash of the execution trace (to detect redundant paths), and 4) Any `ASK` requests.
**Training:** **Nightly only.** During the day, the host logs the (Context, Emitted Program, VM Verdict). At night, programs that yielded a successful VM verdict and led to a correct host resolution are used as positive targets (behavioral cloning/RL with VM verdict as the reward). Serving must remain frozen.

### 5. Ordering, First Version, and Validation
**Order of Execution:** **B -> A -> D -> C.** 
You cannot build D (branching programs) without B (a deterministic harness with typed tools and a task DAG) to orchestrate the forks. A (fluency) is needed to speak the results. C (HDC FFN) is a distraction until the pipeline works.

**Smallest First Version:** 
Implement the `fork` and `discard` ops in the CubeLang VM. The host hardcodes a 2-branch split for a specific counterfactual question. The model emits a standard sequential CubeLang program. The host runs it in Fork A (with Fact X present) and Fork B (with Fact X removed). 

**VALIDATION:**
*   **Metric:** Accuracy on the held-out counterfactual history questions (currently 18% spoken and correct).
*   **Success:** The system reaches >40% spoken and correct on reworded counterfactuals, with the VM trace showing successful `fork` and `discard` operations. (Estimate: 40% is a realistic jump from 18% when moving from downstream-graph-walking to explicit counterfactual simulation).
*   **Failure:** The model emits syntactically invalid CubeLang >20% of the time, or the VM budget (1M jumps) is exhausted on >10% of branches, indicating the model cannot write terminating branch logic.

---

### moonshotai/kimi-k3

# D. The trunk as a compiler of branch programs

## 1. Is the split sound? Yes — but the enumeration belongs to the host, not the model

The sound version of the owner's goal is: **the model emits one program per branch, the host enumerates and schedules branches, the VM executes and returns verdicts.** The unsound version is the model emitting the grid itself (choices × timelines as a tensor it constructs). Reasons:

- The branch set is a *symbolic* object. "Choices a, b, c" and "eras x, y, z" come from the question parse and the stores (which timelines exist, which interventions are representable as graph edits). The host already resolves "it"/"that" through the context graph; enumerating branches is the same kind of work. If the model enumerates, you get hallucinated branches that the VM then has to refuse one by one — wasted budget, and a new failure class (plausible-but-nonexistent era) that your claim check can't catch because the branch never touches facts.
- The model's comparative advantage, given the measured state, is narrow: text → CubeLang for a *specified* branch. The emitter is trained on 517 verified chain programs plus kernel families; that is a small, well-defined distribution. "Given question Q, choice c, era t, emit the counterfactual program" is a templated task. "Given Q, invent the branch structure" is not.

Concrete split:

- **Host**: parses the question into (intervention X, branch set B, timeline set T); for each (b, t) constructs the branch context (a forked store state — see below); sets per-branch budgets (jumps, wall-clock, ASK count); collects verdicts; picks the comparison template ("in branch b, downstream of X: ..."); hands the talk adapter a canonical restatement per branch.
- **Model (emitter)**: emits one CubeLang program per (Q, b, t) triple. Nothing else. It never sees other branches' outputs; cross-branch comparison is the host's, expressed in the canonical restatement.
- **VM**: runs each branch in isolation, returns per-hop (symbol, similarity), verdict, and a *trace digest* (see §4).

The counterfactual mechanism itself: today you follow links downstream of X (18% on reworded). The branch-program version should be: **fork the history graph, remove X and its causal descendants, re-run the query, diff against the unmodified fork.** That requires `ctx.fork` — which is specified but not implemented. **Implementing fork is the actual prerequisite for this entire direction**, and it's a VM engineering task, not a modeling one. Until fork exists, "branch programs" are sequential programs mutating global VM state (registers, stack, storage are VM-global, no call frames) — branches will contaminate each other through storage. That alone invalidates any near-term version that runs branches in one VM session without isolation.

**Validation**: on the counterfactual family, reworded questions, spoken-and-correct should go from 18% to ≥50% with spoken-wrong held at ≤1% (your history-question standard). If fork-based diffing doesn't beat link-following on a hand-built set of 50 counterfactuals with known answers, the premise is wrong and the problem is in the graph's causal links (45,588 links over 940,897 events is sparse — estimate: most events have no causal edge), not in the reasoning machinery.

## 2. Critique of the proposal's mechanics

**Fixed-length opcode logits per grid cell, argmax-decoded.** Wrong on three counts. (a) Programs are sequential and variable-length; a fixed max_code_len block either truncates real programs or pads trivia, and padding tokens in a 128k vocab with atomic opcodes will dominate the loss. (b) Argmax decoding of independent per-position logits produces syntactically invalid bytecode at a rate that grows with length — estimate: even at 99% per-token accuracy, a 40-op program is valid ~67% of the time, and CubeLang's verifier will refuse the rest, so you've built a machine whose primary output is refusals. (c) It bypasses the tokenizer you already paid for: opcodes are atomic tokens, the trunk already does autoregressive generation, and the emitter interface already exists. The correct mechanism is the one you have: autoregressive emission, VM verification, refusal as signal. The grid cell should produce a *conditioning vector* for the emitter, not logits.

**A gate standing in for execution depth.** This conflates two things. Execution depth is a property of the program and the data (how many hops until cleanup fails or the jump budget is spent). A sigmoid gate in a recurrent cell is a property of a forward pass. The proposal's "tau sets how deep a branch runs" is a metaphor, not a mechanism — nothing connects f to the VM's jump counter. What *is* true: the VM already has the real mechanism (1,000,000-jump budget, per-hop thresholds, ASK suspension). Depth control belongs there, set by the host per branch. Keep nothing of the gate-as-depth idea.

**Runtime backpropagation.** Reject absolutely, and note it violates the system's own stated rule ("serving never changes weights"; "the VM verdict is the only reward" — and reward flows to the arbiter's proposer ordering, not to weights). Beyond the rule: backprop through argmax-decoded bytecode is undefined (non-differentiable), so the proposal would require a soft relaxation of program execution, which means the "verifier" is now a differentiable approximation of the verifier — you've rebuilt the exact thing the architecture exists to avoid (unverified claims in weights). The proposal also misread the system's core invariant ("only matrices cross into the VM" — the opposite is true), which suggests its author pattern-matched "neurosymbolic" generically. Discount the whole document accordingly.

**What to keep**: the [choices × timelines] grid as an *organizing structure for the host's scheduler*, and the observation that branch conditioning (choice, era) should modulate emission — which your context-generated adapter (theta = f(c)) already does. The grid is a host-side data structure, not a tensor in the model.

## 3. Does a time-conditioned gate buy anything? No.

The trunk's MinGRU layers are already gated interpolations; adding an explicit time input to the gate (CfC-style) gives the model a continuous knob it must *learn to interpret* as era, from data, at 450M, with 2.77B tokens. Compare the alternative you already have: time as a **role bound in the program** (BIND_ROLE(timeline, T^t)) and T^t position codes in the stores. The symbolic route is (a) verifiable — the VM checks the binding per hop; (b) exact — "in 1900" vs "in 2028" is a different codevector, not a 128-year difference in a continuous input the model must resolve; (c) free at serve time — no architecture change, no retraining of the trunk.

The measured evidence points the same way: the trunk binds a copied value to the asked entity at chance (52%). Its deficit is binding, full stop. Giving it a *second* continuous thing to bind (time into a gate) before fixing the first is inverting the order of difficulty.

Where a time input *would* matter: if the trunk had to generate temporally-extended behavior (motor control, Cubby-Man's continuous navigation). For reasoning across eras, the era is a discrete, enumerable, symbolic quantity. Put it in the data and the VM.

**Validation if you insist on testing**: train two emitter variants on identical verified-program data, one with era as a bound role, one with era as a continuous gate input; compare VM-verification pass rate and downstream counterfactual accuracy on held-out eras. Prediction: the gate variant fails on eras outside the training range (interpolation in a learned continuous representation), the role variant fails only when the timeline codevector is absent from the codebook — a detectable, refusable condition rather than a silent error.

## 4. Feedback: what the VM returns, and when it trains

Per branch, the VM should return a structured record, not a scalar:

- verdict (verified / refused + reason class: threshold miss, absent-role fired, budget exhausted, ASK unresolved, invalid program);
- per-hop (symbol, similarity) — you have this;
- the trace digest: op count, hops taken, deepest role chain, which store facts were touched (provenance IDs), jump budget consumed;
- for counterfactual branches specifically: the **diff** — which downstream facts exist in the unmodified fork and not in this one (this is the answer content, and it feeds the value check directly).

Training: **night only, through the gated promotion rule, never at runtime.** The mechanism: every VM verdict is already ledgered; the sleep cycle already replays the day into adapter training data. Add: verified branch programs (question, branch spec, program, verdict, diff) become emitter training pairs; refused programs become *negative* pairs only if the refusal reason is a program defect (invalid op sequence), not a data defect (threshold miss on a sparse graph) — training the emitter to avoid programs that failed because the graph lacks links teaches it to avoid hard questions, which is the opposite of what you want. This distinction (program defect vs data defect) is the single most important label in the feedback loop, and the refusal reason classes above are how you get it.

Promotion gate: a new emitter checkpoint serves only if, on a frozen held-out set of branch tasks, its VM-verification pass rate and downstream spoken-and-correct are both ≥ the incumbent's, with spoken-wrong ≤1%. This is the adapter lifecycle's missing promote rule — build it here first, since the emitter is the adapter with the cleanest reward signal.

## 5. Ordering, smallest version, gate — and where I object to round 1

**Order: B, then A, then D-smallest, with C folded into D's diagnostic.** I partly object to round 1's implicit "A first" framing. The harness (B) is not a rival priority, it's the substrate: D requires fork, budgets per branch, and ledgered verdicts — all harness work. And "no single process owns boot, state, scheduling and the turn loop" means every other result is currently unreproducible in principle. Four of six said minimal harness first; I'd say it more strongly: nothing else should be pre-registered until the harness can run a gate deterministically, because a gate run on a non-deterministic pipeline is not a gate.

On A I agree with the consensus (two adapters, diagnostic first, sanctioned-set guard for chat) with one objection: the consensus's chat-path guard (every entity/number in a per-turn sanctioned set) is necessary but the sanctioning *source* needs to be specified — it should be the union of facts touched by the VM trace this turn and entities in the context graph's current window, ledgered. Otherwise the sanctioned set itself becomes an unverified channel.

**Smallest first version of D** (estimate: 2–3 weeks of VM work, no trunk changes):

1. Implement `ctx.fork`/`discard` in the CubeLang VM (no merge — discard-only is enough for counterfactuals and avoids merge semantics entirely).
2. Host-side: a counterfactual question type that (a) resolves X via the context graph, (b) forks the store, removes X + causal descendants, (c) emits *one* program via the existing stand-in emitter interface — a hand-written template is fine for v1, do not train anything — that queries the fork, (d) diffs, (e) hands the diff to the talk adapter through the existing canonical restatement.
3. Branch count = 1 for v1. The grid comes later, only if v1 passes.

**Pre-registered gate**: 100 counterfactual history questions (50 reworded, 50 event-named), answers derived from the graph by hand before the experiment. Pass: spoken-and-correct ≥50% on reworded (from 18%), ≥80% named, spoken-wrong ≤1%, and every spoken claim traceable to a fork-diff entry. Fail conditions that kill the direction: accuracy doesn't move (→ the causal graph is too sparse; the fix is link mining in the sleep cycle, not more machinery), or >10% of failures are fork/isolation bugs (→ the VM-global state problem is worse than estimated; stop and fix the VM before any emitter work).

Where C fits: the zero-gated associative memory experiment targets the 52% binding deficit, which is the trunk-side bottleneck that will cap the *emitter* once templates are replaced by learned emission. Run it in parallel with D-v1 — it's one layer, written from current context only, cheap — but its gate should be the binding probe (52% → ≥80% on copied-value binding), not downstream task accuracy. If it fails its gate, the conclusion is that binding stays in the VM and the trunk stays a proposer — which, given this architecture's premise, is a perfectly good outcome, not a consolation.

One number to watch across all of this: the 18 wrong-but-grounded answers at 11% fact mix dropping to 6 at 63% tells you the talk adapter's failure mode is data-mix-sensitive, not capability-bound. Before adding a second adapter (A), re-run the mix diagnostic with the counterfactual family included — if branch answers behave like "the facts don't say" (more data lifts them but costs elsewhere), the two-adapter split will inherit the same trade-off and you should budget for a third slot in the router, not two.

---

### x-ai/grok-4.6

**1. Split of work.** The owner’s loop is sound *only if the model never enumerates branches or sets budgets*. The trunk (or the program emitter) emits **one CubeLang program** whose *body* is a walk/query; the **host** enumerates the (choice, timeline) pairs from the question type + context graph, forks/isolates, sets jump/time budgets, and decides what to emit next from **symbols**. The VM runs one program in one isolated store snapshot and returns (symbol, similarity, hops_used, reason). Putting a [choices × timelines] grid *inside* the net violates “model proposes, host disposes.”

Concrete mechanism: host builds a small DAG of *named* branches (`cf:X_absent@1900`, …), each a CubeLang string the emitter produced from a **canonical restatement** plus a bound role `T^t`. VM `fork` (when implemented) or sequential snapshot/restore. Host compares returned symbols; talk adapter phrases only the winner (or ASK). Failure mode: emitter invents a branch the host did not name → refuse (claim check already exists).

**2. Critique of the liquid grid.** Keep: isolated execution per branch; outcome steers *host* routing. Throw out the rest.

- Fixed-length opcode logits + argmax: CubeLang is variable-length with jumps/CALL. Argmax bytecode will not even parse. Use the existing emitter (token sequence of atomic opcodes) with VM verify-before-run.
- Gate as “depth”: a sigmoid time-constant does not equal jump budget or hop threshold. Depth is a **host integer** (estimate: 64–256 hops; 1e6 jump cap already exists).
- Runtime backprop on faulty branches: forbidden by your rule and by 12 GB batch-1 serving. Weights do not move at serve time.
- “Only matrices cross the VM”: **wrong**; your spec is symbols only. Do not change that.

**3. Liquid / CfC time gate.** It buys nothing for eras. MinGRU already does gated interpolation with no clock. Put time in **data**: bind a timeline role in the program (`BIND_ROLE t, era_1900`) and keep `T^t` codes in the history graph / capsules. A continuous-time gate in the trunk cannot see graph topology and will not fix 18% counterfactuals.

Validation: same reworded “what would change if X had not” set. **Works** if spoken-and-correct ≥ 40% (estimate; 2× current 18%) with 0 spoken-wrong behind host. **Fails** if still ~18% or if wrong answers appear when the gate is ablated but programs still bind `T^t`.

**4. Feedback.** Per branch the VM should return: `(verdict: verified|refused, symbol, similarity, hops, jump_count, reason, ASK-candidates?)`. Train the emitter **only at night** on VM-certified programs (gated promotion: promote if verify rate on held-out chains ≥ 95% estimate, else prune). Never at runtime. Serving never changes weights.

**5. Order.** Round 1 A/B is right; C is a side experiment. Do **B (harness) first** — one process, typed tools, boot preflight — because branch programs without fork/snapshot/ledger will leak state. Then A (second conversation adapter + sanctioned-set guard). Then the smallest D: host-enumerated 2–4 branches, existing emitter, snapshot/restore (fork later), no liquid module.

Pre-registered gate for that D v0: on the existing reworded counterfactual history set, **spoken-and-correct ≥ 35%** (estimate), **spoken-wrong ≤ 1%** behind host, and 0 programs executed that the host did not name. If the 450M continued-pretrain still binds at chance (52%), do not grow the trunk to “compile branches”; keep compilation in the emitter + host.

Object to round 1 only here: do not wait on an HDC FFN for binding; binding for counterfactuals is a **store + program** problem, not a layer.

---

### deepseek/deepseek-v3.2

## 1. Branch Program Compilation: Soundness & Division of Labor

**Yes, the approach is sound** for counterfactual reasoning, but the division of labor must be strict to preserve verification guarantees.

**Model (Trunk + Program Emitter):**
- **Input:** Canonical question restatement from host + context graph pointers (events, timelines).
- **Emit:** One *syntactically valid* CubeLang program *per explicit branch* defined by the host. Each program must:
  - Be a pure function (no side effects on shared VM state).
  - Accept a snapshot/context ID as input.
  - Return a result via `RETURN (symbol, similarity)`.
- **Do NOT:** Enumerate branches, decide which branches to run, compare outcomes, or interpret VM results. Its role is *compilation*, not control flow.

**Host:**
- **Branch Enumeration:** Parse question to identify:
  - **Choices:** Discrete alternatives (e.g., "X did not happen").
  - **Timelines/Eras:** Contextual frames (e.g., "in 1900", "in 2028").
  - Generate a list of `(choice, timeline)` pairs as explicit branches.
- **Orchestration:** For each branch:
  1. Call VM `ctx.snapshot()` to isolate state.
  2. Pass branch parameters + snapshot ID to model to emit program.
  3. Execute program in VM with strict op-count budget (e.g., 10k jumps).
  4. Collect `(symbol, similarity, op_count_used, debug_trace)`.
- **Comparison:** Apply deterministic rules on VM outputs:
  - Highest similarity + valid symbol → "likely outcome".
  - Divergence (e.g., one branch returns `(cat, 0.9)`, another `(dog, 0.2)`) → select best.
  - All similarities below threshold → "inconclusive".
- **Final Answer:** Format result for talk adapter.

**VM:**
- **Execute:** One isolated program per branch.
- **Return:** Per hop: `(symbol, similarity)`. Per program: final `RETURN` value + `op_count_used` + `debug_trace`.
- **Implement:** `fork/merge` as specified (critical for isolation).
- **Do NOT:** Compare branches or steer emission.

**Validation:**
- **Works:** Counterfactual accuracy on reworded questions improves from 18% to >70% (measured on held-out set of 500 questions).
- **Fails:** If any branch program writes to global VM state (corruption), or if host logic selects wrong branch despite higher VM similarity (measure by injecting known-correct branch and checking selection rate >95%).

## 2. Critique of Proposed Mechanics

**Fixed-length opcode logits per grid cell:**
- **Wrong.** CubeLang programs are variable-length. Fixed-length forces padding/truncation, breaking complex branches. Instead, emit a *sequence* of tokens via the existing 128k tokenizer (which includes opcodes). The grid should produce a *program sketch* (key opcodes, branch skeleton), not full logits.

**Argmax decoding:**
- **Acceptable only with constraints.** Autoregressive decoding is necessary for variable length. Argmax (greedy) is fine if:
  - Training uses teacher forcing on verified programs.
  - Validation shows syntax error rate <1% (currently ~5% with stand-in emitter).
  - Alternative: Use a small beam search (beam=2) to reduce errors.

**Gate standing in for execution depth:**
- **Misguided.** Execution depth is controlled by VM jump budget and program logic, not a neural gate. The gate in the proposal is redundant with MinGRU's existing gating. What's needed is a *learned halting* mechanism that predicts when to `RETURN` early, trained on VM op-count traces.

**Runtime backpropagation:**
- **Violates core principle.** Serving never changes weights. Runtime backpropagation would:
  - Break deterministic verification.
  - Introduce latency (~10-100ms per step).
  - Risk adversarial tuning.
  - **Alternative:** Use VM debug trace as supervised loss in *nightly training*.

**Keep what is right:** The `[choices × timelines]` grid is a good *representation structure* for the host to generate branch parameters. The model should consume this grid as a flat list with positional encoding.

## 3. The "Liquid" Time Gate

**No benefit for reasoning across eras.**

- **Why:** Time as a continuous input to a gate adds no symbolic precision. The VM already handles discrete timelines via role binding (e.g., `BIND_ROLE era, 1900`). The stores have `T^t` position codes for temporal relations.
- **Risk:** The gate may blur temporal boundaries, causing era leakage (e.g., mixing 1900 and 2028).
- **Better approach:** 
  1. **Host:** Pass timeline as a symbolic token (e.g., `era_1900`).
  2. **Model:** Treat it as a role to bind in the program.
  3. **VM:** Enforce isolation per era via `fork`.
- **Validation:** Train with/without time gate input; measure accuracy on era-specific queries. If time gate helps, it indicates poor temporal encoding in data—fix the data instead.

## 4. VM Returns & Training

**VM should return per branch:**
- `result_symbol: Symbol`
- `similarity: float` (final hop)
- `op_count_used: int`
- `trace_hash: bytes` (hash of debug trace for provenance)
- `state_changes: list[(key, old_val, new_val)]` (if merge implemented)

**Training the emitter:**
- **Nightly only.** Runtime training violates "serving never changes weights".
- **Gated promotion rule:**
  1. **Detector:** Flag branches where VM returned valid result but emitter syntax error rate >5%.
  2. **Spawn:** Train emitter on corrected programs (VM trace → corrected bytecode).
  3. **Promote:** If validation on held-out branches shows syntax error <1% and semantic accuracy (symbol match) >95%, promote.
- **Pre-registered gate:** Emitter allowed into serve loop only after passing promotion.

## 5. Order & Minimal Version

**Objection to Round 1 consensus:** The consensus prioritizes fluency (two adapters) over reasoning. Fluency is a distraction—the talk adapter already achieves 0 wrong answers behind host. The binding deficit (52%) and counterfactual weakness (18%) are core failures. Fix reasoning first.

**Order:**
1. **C (HDC FFN experiment)** – but narrowed. 
   - **Minimal version:** Add a *single associative memory layer* in the trunk (layer 10). Keys: context-derived hypervectors; values: VM-verified (symbol, role) pairs. Train to predict bindings for "it"/"that" resolution. 
   - **Pre-registered gate:** Binding accuracy improves from 52% to >85% on held-out anaphora set.
   - **Why first:** Binding is foundational for branching (need to resolve references per timeline).

2. **B (Harness)** – but only the deterministic orchestration for branching.
   - **Minimal version:** Host implements branch enumeration, VM `fork/merge`, and outcome comparison as described above. No full task DAG yet.
   - **Pre-registered gate:** 100% isolation (no branch leaks) on 1000 test branches.

3. **D (Branch compilation)** – integrated with above.
   - **Minimal version:** Extend program emitter to generate one program per `(choice, timeline)` pair. Use beam=2 decoding.
   - **Pre-registered gate:** Counterfactual accuracy >70% on reworded questions.

4. **A (Fluency)** – only after reasoning works. Add conversation adapter with *entity guard* (all named entities must be in host-sanctioned set per turn).

**Rationale:** Reasoning failures block all downstream improvements. A fluent model that can't reason correctly is unsafe. The binding deficit must be fixed before branches can be evaluated reliably.

---

