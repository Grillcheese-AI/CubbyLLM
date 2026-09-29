# Panel, round 3: the harness spec, fork and the VM, put to the strongest models (2026-09-28)

Rounds 1 and 2: `docs/panel_fluent_harness.md`, `docs/panel_compiler.md`. Asked independently, temperature 0.3, reasoning effort `medium`. **Model opinions, not findings.**

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

# The two panels so far (six mid-tier models, 2026-09-28)

(six reviewers, asked independently)

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

Round 2 (the owner's goal: the trunk compiles a question into one CubeLang program per branch -- choices a, b, c
against timelines x, y, z -- the VM runs each branch in isolation, the outcome steers the next emission; a
counterfactual history question is today answered by walking the graph's downstream links, 18% spoken and
correct when reworded): all six said the goal is sound with the host enumerating branches and setting budgets,
the model emitting one program per branch autoregressively (not fixed-length opcode blocks), the VM returning a
structured report per branch (verdict, reason, hops, jumps used, facts touched, the diff between branches) and
the emitter trained only at night through a promotion gate. All six said a CfC-style "liquid" time gate in the
trunk buys nothing for eras: time is a role bound in the program and a T^t position code in the stores. All six
named `ctx.fork` (specified, not implemented) as the prerequisite. Order: harness first, then the two adapters,
then the branch runner, then the HDC memory experiment.


# CubeLang and its VM, as they are (from the source and docs of the `cubelang` repo, 2026-09-28)

**Language.** A typed scripting language for "self-evolving reasoning modules": `program X implements I { storage {...}
@system @once public function constructor() {...} function f(a: T): U {...} }`. Programs are always bound to an
interface; the VM registry seeds ISolver (parse/solve/verify), ISolverLearn (+ learn), IAgent (think(input, ctx),
act(decision, ctx), observe(result, ctx)). Statements are three-address: `create x : type; assign x = v; add x, n;
push x; pop x; query x; store x, "key"; recall "key"; bind x, ROLE, val; remember x; sum x; return v;` plus
`if`, `while`, `for (let x of arr)` and `let`. A non-trivial expression as an argument or a `let` right-hand side
(a binary op, a struct or map literal, a method call, an index) is a compile error, by design: the three-address
surface IS the language. `match` arms all executed unconditionally until 2026-09-14 (now a strict error); 24
"extended" mnemonics (infer, map_roles, filter, score, debate, forge, explore, sync, temporal_bind, analogy, ...)
lex, parse and compile but only write a trace line. `asm { MNEMONIC operands; }` is an escape hatch to raw opcodes.
Permission attributes (@external, @internal, @restricted, @ratelimit, ...) are advertised and not enforced.

**Bytecode.** One byte per opcode (0x00-0xFF). Real: CREATE, DESTROY, ASSIGN, ADD, SUB, MUL, DIV, TRANSFER, COPY,
PUSH, POP, COMPARE, QUERY, STORE, RECALL, BIND_ROLE, UNBIND, COND, JMP, LABEL, CALL, RETURN, MAKE_ARRAY, LEN, INDEX,
SET_INDEX, REMEMBER, FORGET, SUM, ASK, NEWVAR. `while` and `for` lower to COMPARE + COND + JMP with labels, so the
language iterates for real; the bare LOOP opcode (0x12) is a structural marker. Every jump spends a per-run budget of
1,000,000; recursion is capped at depth 64.

**Values.** Int, Float, Str, Bool, Null, Array, Map, Hvec (a bipolar hypervector in the memory's space).

**VM state is global**: named registers, a stack, storage (STORE/RECALL, exact key), a hippocampal associative
memory (STORE also binds the key's hypervector, RECALL is a cosine cleanup over stored keys), a codebook (symbol ->
deterministic hypervector), a knowledge store (QUERY: exact normalized-key lookup with provenance; a miss pushes an
EMPTY chunk array, never a nearest neighbour), an accumulator, an event log. There are no call frames: CALL swaps
the register map and the frame map for the callee and restores them after; ASK inside a CALL is an error.

**Binding.** BIND_ROLE binds a filler into a register's frame; UNBIND (`recover`) is a cosine cleanup whose candidate
pool is the frame's own bindings (scoped in 2026-09, after an absent role returned "umbrella" at 0.02 from an
unrelated frame). The winning similarity is surfaced (`last_recover_similarity`). An unbound role still returns
noise, not Null (an open item).

**ASK.** `ask "question", c0, c1, ...;` suspends the program with the candidates. The host answers; `resume` pushes
the answer on the stack and REJECTS any answer that is not structurally identical to an offered candidate: the host
may choose, never supply. Resumption captures only the interpreter locals (pc, flag, jump budget) since state is
global.

**Transport.** `cubelang run-proto`: a resident process reading length-prefixed protobuf RunRequests on stdin
{program source, args, fn_name, answers (one JSON value per ASK, in order; the program is re-executed from
scratch and consumes them), knowledge_path (a facts .jsonl the process caches by path and mtime and CLONES into
the fresh per-request VM)}. Every request gets a fresh VM, so requests never share state. RunResult {ok, one of:
symbol | error | suspended{question, candidates, program, function}, optional similarity}. The host's Python
client (`cubbyllm/bridges/cubelang_client.py`) verifies before executing (strict) and returns (symbol, similarity)
per hop plus verified or refused with a reason. run-proto is unconditionally strict.

**Context in the spec, not in the VM.** `docs/SPEC.md` describes `ctx.snapshot() / restore() / fork() / merge() /
discard()` (try a branch in an isolated copy of registers, stack, storage, history), `agent` and `cloned_agent`
(forked state, independent evolution), containers, events and channels, `@cron`, `extend` (runtime
self-modification). None of these are implemented; the parser knows `ctx` only as a type name.

**Deny by default.** A module not named in the program's `use` list is invisible at CALL time (a registry of
native modules, `override` resolution with compile-time validation).

**The emitter that writes programs.** A stand-in model (a 2.6B LoRA fine-tune on 517 VM-verified chain programs plus
arithmetic, role-binding and kernel families) behind the same interface base450m will implement. The tokenizer holds
the opcodes as atomic tokens. Every emitted program is verified by the VM before it runs; the harvest of verified
programs is the emitter's training set.


# The harness spec, v0 (written today; a first implementation exists)

# The harness, v0: one process that owns boot, state, the turn and the night (2026-09-28)

Status: SPEC + v0 built (`standin/harness.py`, pinned by `standin/tests/test_harness.py`). Source: Nick, 2026-09-28 ("we need some kind of bootstrap and an orchestrator, in fact the full system should act as an agent harness"), and the panel's two rounds (`docs/panel_fluent_harness.md`, `docs/panel_compiler.md`): six reviewers, one design, four of six said build this first. Pre-registered under **H-E14** in `CUBBYLLM_HYPOTHESES.md`.

## What it is

The parts exist as modules: the serving brain (`standin/serve.py`: sense → neurochemistry → route → a cortex), the verified ask loop (`standin/ask.py`), the event stream (`cubbyllm/reasoning/events.py`), the signed ledger (`standin/ledger.py`), the vault, the sleep cycle (`cubbyllm/reasoning/sleep.py`), the forge, the adapter bank. No one process owns boot, state, scheduling and the turn loop; the sleep cycle writes the stores that serving reads, with nothing between them.

The harness is that process. It adds no capability. It makes the existing ones one system:

- **boot** loads the stores, the VM and the adapters, runs the preflight against pre-registered gates, and refuses to serve if one fails;
- **the turn** is one typed step with budgets, every tool call through one registry, every spoken reply hashed into the ledger;
- **the night** is a job the same process schedules, and it holds the store lock while it writes;
- **replay** reconstructs any turn from the ledger and the stores.

The 450M model proposes (programs) and phrases (talk). It never decides what is true, whether a check passed, what a tool may do, what is written to a store, or whether to speak.

## One step

```
observe    the person's words, the sensors, the hormonal state; the context graph resolves "it"
route      the host's frozen tag router picks the turn kind (fact, chat, learn, help, plugin, task)
frame      a TaskFrame: kind, goal, allowed tools, budgets, status
plan       the emitter proposes (a program, a lookup); the disposer takes it only if it covers
call       each tool through the registry: permission, effect class, budget, then the call, then the ledger
verify     the VM's verdict per hop; the host's guards on the draft (name-and-number, value, claim, explicit gate)
act        speak, ask which one, or refuse; only what the verdict reached
record     the event stream, the ledger (a turn row with the reply's hash), the context graph, the day record
```

A multi-step task is a stack of TaskFrames under one root frame. A sub-frame is done when its own check passes (a VM verdict, a store read that returned), never because the model said so. The root frame's budgets bound the whole task.

## What stops a runaway

Every TaskFrame carries a `Budget`, spent by the registry as calls go through: `tool_calls`, `vm_hops`, `fetches`, `forge_attempts` (0 in a live turn), `depth`, `wall_s`. Exhaustion is a declared outcome (`status = "exhausted"`, spoken as "I ran out of steps on this"), never a silent truncation. Repeated failure of the same step ends the frame. Defaults for v0 (the owner's to move): tool_calls 8, vm_hops 64, fetches 1, depth 3, wall 10 s.

## Tools

A tool is declared, not discovered: `Tool(name, effect, needs, fn, doc)`.

- `effect` is one of `read` (a store lookup, a VM run without writes), `propose` (something that may become a write after a gate: a fetched fact, a forged program), `write` (a store write, a memory write). Effect classes are enforced **at call time** by the registry, not only at certification: two certified tools whose composition writes are caught here.
- `needs` is the set of permissions a call requires (`read_history`, `read_wiki`, `vm`, `fetch`, `write_store`, `vault`, `explicit`, `forge`, `plugin:<name>`). **Deny by default:** the frame's `allowed` set is what the turn kind grants, and a tool whose `needs` is not a subset is refused before it runs. The chat kind never gets `write_store` or `fetch`; a live turn never gets `forge`; `vault` is never granted to a talk tool.
- Every call, allowed or refused, is a ledger row: tool, effect, the arguments' hash (the arguments themselves through the vault when they may hold a person's words), the outcome, the frame and turn ids.

The tools of v0 wrap what exists: `history.lookup` (history_lookup), `wiki.facts` (the wiki world), `vm.run` (cubelang_client.run_program_proto), `learn.fetch` (learn.py's fetch, effect `propose`), `store.write` (a gated write, effect `write`), the plugins' `handle`. Their bodies do not move; the registry is the one door.

## Boot

```
1. read the signed config: thresholds (τ_vm, τ_ret, the floors and margins), the model and adapter hashes, the gate pack
2. open the stores read-only; check each store's integrity hash against the config
3. start the VM (run-proto, resident); run its self-test
4. load the base and the adapters; check their hashes against the config
5. run the preflight: the gate pack (a fixed set of asks with expected outcomes: fact families, absent, a
   contradicting-context capital, an explicit request that must be refused, a wrong-answer sentinel that must
   not be spoken); the pass rates must meet the pre-registered bars
6. open the stores read-write; open the ledger; serve
```

**Thresholds are never recalibrated at boot.** They are read from signed config; the preflight passes or the process refuses to serve. A boot that could lower a threshold to pass its own test is a hole (the panel's word). Calibration is an offline job whose output is a new signed config.

Refuse to serve if: any store hash differs, the VM self-test fails, an adapter hash differs, the ledger cannot write, or any preflight bar is missed. The refusal is itself a ledger row.

## The night

The sleep cycle (`cubbyllm.reasoning.sleep`) runs as a job the harness schedules, in the same process or a child it owns. Before it writes (consolidated facts, episodes, skills, the adapter training set), it takes the **store lock**; a turn that arrives while the lock is held is answered from the read-only view or asked to wait, never served from a store mid-write. Nothing the night writes changes what may be spoken: a consolidated fact is looked up and verified like any other; a promoted adapter is a new hash in a new signed config, loaded at the next boot.

## Where the model sits, and what the harness never lets it decide

Two roles, two adapters, both behind the emitter interface: the program emitter proposes CubeLang (verified before it runs); the talk adapter phrases what the verdict returned. The harness never lets the model decide: whether a fact is true; whether a hop cleared τ; which tool runs, or with what permission; what is written to a store; whether content is allowed; whether to speak after a failed check; which event a pronoun means when the context graph and the host disagree. It may propose and rank; the host disposes.

## Bootstrap in the second sense: a capability from almost nothing

Cubby-Man is the pattern: two facts, every move VM-mediated, refused moves become walls, hypotheses kept apart from facts, power-moves invented and certified. The harness generalises it as four stores per capability — observed facts, hypotheses, verified skills, episodes — and one promotion rule: a hypothesis becomes a fact only by a verified observation; a skill is promoted after N successes across distinct contexts (N pre-registered per world). The arbiter's reward-prediction error reorders proposers; it never writes.

## v0: the smallest version, built 2026-09-28

`standin/harness.py`:

- `Tool`, `ToolRegistry` (deny by default, effect classes, budgets, every call ledgered);
- `Budget`, `TaskFrame` (a stack per turn);
- `Preflight` (a gate pack from JSON; bars; refuse-to-serve);
- `Harness`: `boot()`, `turn()` (wraps the existing `CubbyBrain.turn` and `AskLoop.ask`; hashes the reply into the ledger), `replay(turn_id)`, `night()` (the sleep cycle under the store lock);
- `standin/data/gate_pack.json`: the preflight asks.

Not in v0: the multi-step DAG (the stack is there; only single-frame turns are wired), the forge in the loop, the event stream's panel showing frames, plugins' permissions beyond `plugin:<name>`.

## Gates (pre-registered; the numbers are the owner's to move)

1. **Same answers.** On the H-E6 fact families and the E8/E9 test splits, the harness's answers equal the modules' (byte-identical replies on the same draw): the harness adds no capability and loses none.
2. **Every spoken reply has a ledger row** with its hash, its frame and its verdicts: 100%.
3. **Refuse-to-serve on injected faults.** Corrupt each in turn — a store hash, an adapter hash, a threshold in the config, the VM binary, the ledger path — and boot must refuse in 5 of 5, with a ledger row naming the check.
4. **Replay.** For 200 turns from one day, `replay(turn_id)` reproduces the reply hash from the ledger and the stores: ≥ 99% (the rest explained by a recorded non-determinism: a fetch, a world's clock).
5. **Deny by default holds.** A probe set of 50 calls that a turn kind must not make (chat writing a store, a live turn forging, a talk tool reading the vault): 0 executed, 50 ledgered as refused.
6. **Kill and restart.** Kill the process mid-turn 100 times: 0 turns spoken twice, 0 lost from the ledger, the same store hashes after restart.
7. **Latency.** p95 of a fact turn ≤ the modules' + 20%.

**Kill:** if gate 1 fails, the harness changed an answer and is not a harness; if gate 2 or 3 fails, it is plumbing, and says so in the report.

## What it brings to CubbyLLM

- One owner of state: the store-write race between the night and the day is closed by the lock.
- One door for tools: deny-by-default with effect classes at call time, so a certified program cannot compose its way into a write.
- Every answer replayable from the ledger: the audit the design promised becomes a command.
- The seat the adapter lifecycle's promote rule and the branch runner (D) both need: budgets, frames, the ledger.
- **Advantage over conventional agent frameworks:** they let the model call tools and judge its own results; here the model is a proposer behind a registry, the VM is the judge, the ledger is the memory of every decision, and a boot that fails its own gates does not serve.


# What we want from you

You are a stronger reviewer than the first two panels. The system brief, the VM as it is, and the harness spec are
above. Be specific; name what is wrong; keep what is right; every proposal with its failure mode and its
measurement.

1. **The harness spec.** Where does it fall short of an agent harness the system can grow into (multi-step tasks,
   sub-goals, worlds, the night)? What would you change before it is built further? Is anything in it wrong for a
   system whose model does not ground (binds at chance, memory over context 22/24)?
2. **Fork, and the VM.** Two designs for the branch runner: (a) at the wire, a `branch` on the RunRequest (facts to
   exclude, facts to assume, applied to the cloned store before the run; each branch a fresh VM; the report per
   branch); (b) inside the VM, `ctx.snapshot/restore/fork/discard` as opcodes, so one program can try a branch and
   back out. Which first, and what exactly should the branch report carry? What else in this VM would you change
   for the branch runner and the harness -- call frames, Null for an absent role, the permission attributes,
   the loop budget, the trace-only mnemonics?
3. **The program the harness runs.** Should the harness's turn itself be a CubeLang program (an IAgent whose
   think/act/observe are the step, the host answering its ASKs), or Python with the VM as a tool? What does each
   cost in safety and in what the emitter can learn?
4. **The 450M and fluency.** The two panels' answer was two host-switched adapters plus a sanctioned-set guard for
   chat. Is that right at 450M, given the base does not ground? What would you measure first?
5. **The HDC memory** (a zero-gated associative layer written from the current context, aimed at binding 52% ->
   85%): worth one experiment, or not? If yes, the exact experiment; if no, what instead for binding.
6. **Order and the first gate.** The smallest sequence of builds that turns the modules into one agent, each with
   its pre-registered gate. Disagree with the two panels where they were wrong.

Constraints: no external LLM at serve time; the target is 450M; local serving on a 12 GB consumer GPU at batch 1;
the host disposes and the VM verifies; only symbols cross into the VM; serving never changes weights; every claim
measurable against a pre-registered gate.
```

</details>

## Answers (8 of 8 returned)

### openai/gpt-6-luna-pro

## Executive judgment

The harness is the right next build, but v0 should be treated as a **control-plane prototype**, not yet as the safety boundary its spec describes. Three claims in the design are stronger than the mechanisms support:

1. **A VM verdict is not evidence that a fact is true.** It establishes that a program ran and passed configured checks against particular inputs and stores. Correctness still depends on the store, the program, the thresholds, and the provenance.
2. **A name-and-number guard is not a claim guard.** A reply can use only authorized names and numbers and still assert an unsupported relation, cause, or intention.
3. **An effect label does not contain a side effect.** The registry can enforce `read` or `write` only if the tool implementation is itself constrained; a mislabeled or compromised tool can still write.

Keep the host in charge, the VM as a constrained symbolic executor, and learning as proposal-generation. Tighten the boundaries and the evidence before expanding the harness into a general agent.

## 1. Harness: what to change before extending it

### Keep the single owner; make the store and turn boundaries real

One process owning boot, scheduling, turns, and night jobs is a sensible control-plane design. The weakness is that the spec’s guarantees currently depend on conventions inside that process.

- **Replace the night-long store lock with immutable snapshots and atomic promotion.** Let turns read a versioned snapshot; let sleep build a new snapshot separately; promote it by an atomic pointer or manifest swap. Holding a lock throughout sleep either stalls turns or pressures the implementation to serve from a partially updated view.
- **Make a turn a durable transaction.** Assign a turn ID and an idempotency key before any tool call. Persist the input references, selected snapshot IDs, tool-call outcomes, and final reply hash. On restart, resume or mark the turn failed—never silently re-run a side-effecting call.
- **Make replay use the original inputs and snapshot, not today’s stores.** A reply hash alone cannot reconstruct a turn. Record the exact store/version IDs, model and adapter hashes, route, program source/hash, VM config, random seeds, clock inputs, fetch result IDs, and any user/sensor inputs needed for replay. Store sensitive payloads only in the vault, referenced by ID.
- **Use typed schemas and capability-scoped tool handles.** `effect` and `needs` are useful, but not sufficient. A tool should expose a narrow schema and receive only the specific capabilities it needs. Run plugins and write-capable tools in a process or sandbox whose OS permissions match those capabilities. Do not let arbitrary Python functions inherit the harness’s full access and rely on their declared `effect`.

**Failure mode:** a replay appears correct because it reproduces a stored reply hash, while the same turn cannot be recomputed, or a misclassified tool performs an unledgered write.  
**Validation:** fault-inject termination before and after every tool call and commit point; require zero duplicate side effects and recoverable turn state. Replay 200 turns against pinned snapshots and inputs, requiring exact reply hashes for deterministic turns and explicit, recorded differences for nondeterministic ones. Probe each tool’s actual filesystem/network access, not just its registry label.

### Make task completion explicit, not merely “a sub-check passed”

Budgets and `TaskFrame`s are a good foundation, but “a store read returned” is not generally evidence that a sub-goal succeeded. Each frame needs a typed success predicate and a typed result. A multi-step root task also needs an acceptance predicate over its sub-results. Give child frames budget allocations bounded by the root, with cancellation, deadlines, and explicit failure propagation. Do not make a task DAG’s success equivalent to every individual tool returning normally.

Add world/tenant and epistemic scope to frames and data: facts, hypotheses, skills, and episodes should not become interchangeable just because they share a store API. A counterfactual branch, a live world, and a hypothesis should be distinct namespaces or types.

**Failure mode:** a successful lookup or plausible child result is mistaken for task completion, or a hypothetical result leaks into ordinary fact retrieval.  
**Validation:** construct tasks where every tool call succeeds but the root condition is false, and where a hypothesis conflicts with a fact. Require the root to fail in the first case and the ordinary fact path to remain unchanged in the second.

### Routing is a safety decision too

A frozen tag router does not make routing safe. If it misclassifies a factual request as chat, the chat path may bypass the VM value check; if it misclassifies an ambiguous request as fact, the host may resolve the wrong event. For uncertain or conflicting routes, fail closed to clarification or a constrained response. Keep explicit authorization separate from route classification.

The spec says no content is spoken unless a verified chain reaches it, but the chat adapter has no VM value to check. Resolve that contradiction: either define and validate an explicit class of **non-factual, host-approved speech acts**, or have the host construct chat from authorized content units and permitted speech acts. A VM verdict over stored values is not a universal speech-safety certificate.

### Correct the preflight and gate claims

A signed config is good. Preflight should verify the signature, signer policy, and compatibility of the gate pack with the config; it should not silently bless an authorized but inappropriate threshold change. A preflight proves that a small set of tests passed, not that the system is safe.

The stated replay gate of ≥99% is too weak for deterministic local execution unless the remaining 1% has a specific, pre-registered source of nondeterminism. Likewise, “every reply has a ledger row” is not enough if the row is written after speech and can be lost on a crash. Define the commit protocol so the reply is not released until its durable ledger commit succeeds.

**Validation:** for deterministic turns, require 100% replay agreement on the test set. For all turns, require a durable ledger record before release, then kill the process at each boundary between record, tool call, and speech. Any released but unledgered reply is a failure.

## 2. Fork and VM: start with branch requests at the wire

Choose **(a): a branch field on the RunRequest, with each branch executed in a fresh VM against an isolated overlay**. Do not begin with in-program snapshot/restore/fork opcodes.

The current VM has global registers, stack, storage, accumulator, event log, and a knowledge store, plus incomplete call-frame semantics. Adding rollback inside that state multiplies the number of things that must be copied, restored, or proven not to leak. The wire-level design gives the host a smaller, auditable operation: construct a branch overlay, run a program in a fresh VM, collect a report, discard the VM.

Use a pinned base snapshot plus a branch overlay containing **structured symbolic assumptions and exclusions**. Include stable fact IDs, provenance, and conflict policy. An exclusion must suppress the identified fact, not vaguely “remove a concept”; an assumption must not silently overwrite an unrelated fact. Keep all overlay reads tagged as counterfactual. Never write branch results into the canonical store.

This does **not** implement general fork semantics. It implements isolated evaluation of bounded alternative inputs. That is the right first capability for the stated use case.

### Branch report

Return a machine-readable report per branch with at least:

- `turn_id`, `task_id`, `branch_id`, and branch label;
- base snapshot ID/hash, overlay hash, program hash, VM/config version;
- the exact symbolic assumptions and exclusions, with fact IDs and provenance;
- status: completed, refused, malformed, exhausted, timed out, or runtime error;
- returned symbol/value, verifier verdict, and reason code;
- per-hop trace: operation/program location, input fact IDs, output symbol, similarity score and threshold where relevant;
- facts touched, including whether each came from the base store or branch overlay;
- hops, jumps/gas, tool calls, elapsed time, and budget consumption;
- a structured diff against the other branch reports: changed inputs, changed retrieved facts, changed intermediate results, changed final result.

Do not summarize the diff only in prose. “No result” must distinguish *not found*, *not entailed*, *contradicted*, *timed out*, and *refused*. An absent fact is not evidence of its negation.

**Failure mode:** an assumption is applied with ambiguous precedence, or a branch result is later read as an ordinary fact.  
**Validation:** use paired branches differing by exactly one assumption or exclusion; assert that only the affected branch sees the overlay, and that all later ordinary queries see the unchanged base snapshot. Test conflicting assumptions, missing fact IDs, and exhausted branches.

### VM changes I would prioritize

1. **Return a genuine absent-role result.** `UNBIND` currently returns noise even when the role is absent. A surfaced similarity score is not enough if downstream code can still treat the noise winner as a value. Return a tagged `unbound`/`Null` result, and require explicit handling. Keep the similarity for diagnostics; do not treat it as a value on failure.
2. **Enforce permissions, or stop advertising them.** Permission attributes that are not enforced are a dangerous false assurance. Before tools or branch programs can use them, enforce capabilities at runtime at every relevant call and effect boundary. Compile-time validation alone is not a security boundary.
3. **Remove trace-only mnemonics from executable-looking code.** Reject them in the production verifier until they have semantics. A program that parses and “does something” only in a trace is particularly hazardous for an emitter to learn from.
4. **Replace the jump-only budget with a resource budget.** Charge for instructions, loops, allocation, array size, calls, and VM time; cap memory as well as jumps. One million jumps is not a meaningful interactive-turn bound. The harness wall clock is useful but should not be the VM’s only protection.
5. **Specify and test call isolation.** The register/frame swap-and-restore behavior is subtle; document what storage, events, stack, and memory are shared. Give calls explicit locals/arguments/results and bounded depth. Test that branch-local or callee-local state cannot leak.
6. **Avoid ASK in branch programs initially.** ASK is resumed by re-executing the program from scratch with answers. That is safe only if the pre-ASK path is deterministic and has no externally visible effects. Restrict ASK programs to pure execution until there is a true continuation mechanism or an enforced no-side-effects-before-ASK rule.

**Failure mode:** a branch can consume unbounded resources, misread an absent role as a real symbol, or repeat an effect during ASK resumption.  
**Validation:** adversarial VM tests with absent roles, nested calls, large arrays, infinite loops, and ASK after an attempted effect; require bounded termination, explicit absence, and zero repeated effects.

## 3. Should the turn itself be CubeLang?

For now, **Python should orchestrate the turn; CubeLang should be a constrained tool invoked by Python.** Do not make the whole turn an `IAgent` program yet.

The host needs to own routing, authorization, task budgets, store versions, tool effects, and the final speak/refuse decision. These are control-plane responsibilities, not good first targets for a generated program in a language whose permission attributes are unenforced, whose extended mnemonics can be trace-only, and whose global VM state is still being tightened. Encoding the orchestration in CubeLang now would make the trusted computing base larger without improving the grounding evidence.

There is a learning cost: the emitter will initially learn verified subprograms and branch programs, not whole-turn policies. That is desirable. A verified program means “passes this verifier and execution contract,” not “semantically correct for every task.” Keep root task policy in explicit host code and gates.

Later, if a programmatic task language is useful, define a small declarative task schema and compile it to a host-checked plan. Do not give a generated CubeLang program authority over tool selection or permissions merely because it implements `IAgent`.

**Failure mode:** host policy is duplicated in emitted programs and a program bypasses or subtly changes a host check; alternatively, Python becomes untestable ad hoc orchestration.  
**Validation:** make the host controller a deterministic state machine with typed input/output records. Test that malformed, extra, or adversarial program output cannot select a tool, grant a permission, write a store, or authorize speech. Compare its decisions against a fixed transition-table test suite.

## 4. Fluency at 450M: test the adapters, but fix the proposed guard

The two-adapter proposal is a reasonable **diagnostic and likely serving configuration**, but it does not solve safety by itself. The base’s 52% binding result and 22/24 preference for memory over contradiction are strong evidence not to let it freely ground user claims. They are not evidence that it cannot produce useful conversational phrasing.

A sanctioned set containing only names and numbers is inadequate. “Alice caused the failure,” “the event followed Alice,” and “Alice reported the failure” can use exactly the same sanctioned names and still make different claims. For chat, authorize **claim tuples or content units**—entities, relations, polarity, time, and provenance—or constrain the model to non-factual conversational moves such as acknowledgement, clarification, or asking what the user wants. User-provided content can be repeated as attributed content, but attribution must be preserved.

First run the panel’s equal-token diagnostic: fact-only, chat-only, and mixed adapters, holding the base and evaluation inputs fixed. Include the existing grounded talk adapter as its own control; keep its host checks unchanged. Then test the chat path both with and without the proposed guard. Do not conclude “more fluent” from a one-sentence answer or a rise in conversational-data likelihood.

Measure, separately:

- naturalness and task usefulness on multi-turn conversation;
- reference and follow-up resolution, including pronouns and corrections;
- unsupported atomic claims per reply, not just wrong names and numbers;
- appropriate abstention/clarification;
- route errors between chat and fact;
- fact-task accuracy and refusal quality when chat and fact content are mixed.

Use a held-out, adversarial set with paraphrases, corrections, false presuppositions, and distracting earlier context. A reasonable initial estimate is **at least 1,000 prompts across several hundred dialogue threads**, with route and claim-error counts reported separately.

**Failure mode:** the chat adapter sounds more natural but invents relations, treats a user’s false assertion as established fact, or carries an old claim across a correction.  
**Validation:** promote only if conversation quality improves over the current system while the pre-registered rate of unsupported claims stays below its safety bar on both ordinary and adversarial splits. If the name-and-number guard passes but relation/polarity errors rise, the guard has failed.

## 5. HDC memory: defer the layer; run a binding diagnosis first

I would **not yet spend the next experiment on a new HDC layer aimed at 52% → 85%**. The target is an aspiration, not an established achievable effect. More importantly, the host and VM already have symbolic stores and role binding. The current failure could be in the test format, context selection, entity resolution, or the model-to-symbol mapping—not a missing associative memory layer.

An HDC layer written from current context could also make a wrong binding more consistent. The 52% result is especially concerning because the base is not reliably binding an observed value to the asked-about entity in the first place.

First run a narrow intervention study on the existing binding task:

- inspect the exact examples, chance baseline, and scoring;
- swap entity/value assignments while keeping wording constant;
- add decoy entities and repeated or contradicted values;
- compare direct trunk output with host-resolved symbolic bindings and with a canonicalized, explicitly role-bound input;
- report accuracy and calibration by ambiguity, contradiction, and distance in context.

This separates “the model cannot bind” from “the host presented the wrong binding problem.” For the live system, use explicit symbolic binding by the host and test its errors against the graph, rather than asking the 450M trunk to recover identity from prose.

If that diagnosis shows a residual, repeatable *model-side* binding deficit and the experiment budget allows one test, then try the zero-gated associative memory as an ablation—not an architecture commitment. Use one insertion point, current-turn context only, no persistent writes, and a fixed held-out set. Compare the base, symbolic host binding, and the gated-memory variant on the same entity/value swaps and contradictions. Measure binding accuracy, false binding rate, and regressions on held-out language loss and ordinary generation.

**Failure mode:** the layer raises accuracy on clean bindings by retrieving a stale or decoy value, or harms language behavior while leaving the symbolic host as the actual source of correctness.  
**Validation:** require improvement on the hard binding split, not just the easy split, with no increase in false bindings on contradiction/decoy cases and no material held-out language regression. If symbolic host binding already solves the system-level task, the layer has not demonstrated value.

## 6. Smallest build order and first gates

The harness exists as v0, so the first step is **not another feature**. It is to close its operational and semantic gaps, then use it as the control plane for incremental capabilities.

### 1. Harden the harness control plane

Add immutable store snapshots, durable turn IDs and ledger commits, idempotent tool calls, typed tool schemas, capability-scoped execution, and deterministic task-frame success conditions. Preserve the existing module behavior when possible.

**Gate:** identical outputs to the old modules on a fixed deterministic set; zero unledgered replies in crash tests; zero duplicate side effects across kill/restart tests; replay is exact for deterministic turns using pinned inputs and snapshots; forbidden calls execute zero times and are ledgered. Any mismatch in existing answer behavior must be explained, not waved through as “harness overhead.”

### 2. Make the VM safe for bounded programs

Implement tagged absent-role results, enforce or remove permission attributes, reject trace-only mnemonics in production, add resource metering, and test call isolation and ASK effects.

**Gate:** adversarial test suite produces no unbounded run, no unhandled absent role, no disallowed effect, no trace-only operation treated as useful execution, and no duplicate effect across ASK suspension. Report exact resource limits and observed worst-case use.

### 3. Run the adapter diagnostic and fix the chat contract

Compare fact-only, chat-only, and mixed adapters at equal tokens, with a claim-aware chat guard and a fixed router. Do not change the grounded fact path in the same experiment.

**Gate:** pre-registered fluency/usefulness improvement plus a claim-error ceiling on ordinary and adversarial conversations; report routing mistakes and unsupported relations separately. A zero-error result on a small set is not proof of zero risk.

### 4. Add wire-level counterfactual branches

Implement isolated branch overlays and reports before any in-VM `ctx.fork`. Have the host enumerate branches and set budgets; have the emitter produce one bounded program per branch. Treat the branch runner as read-only with respect to canonical stores.

**Gate:** paired-branch tests show the expected branch-only differences, no cross-branch state leakage, stable replay against pinned snapshots, and bounded execution. Include counterfactuals where the correct result is “unsupported,” not just cases with an obvious changed answer.

### 5. Diagnose binding; only then decide on HDC

First establish whether symbolic host binding resolves the actual end-to-end failure and where the trunk still fails. Run the gated HDC ablation only if it addresses a residual system need.

**Gate:** binding splits include swapped values, decoys, contradictions, and ambiguous references. Report false bindings as well as accuracy; do not promote on aggregate accuracy alone.

### 6. Add richer task graphs, worlds, and night promotion

Only after the single-frame controller and branch runner are stable, wire a bounded DAG, explicit root success predicates, world-scoped stores, and versioned sleep promotion. The existing “one process” architecture can schedule these without putting control policy into CubeLang.

**Gate:** child failures propagate correctly; root budgets cannot be exceeded by child allocation; night writes are invisible until atomic promotion; turns either see a complete old snapshot or a complete new one; hypotheses and counterfactuals cannot leak into fact answers.

The largest correction to both panels is not a different model block or a more expressive VM program. It is to avoid treating *verified execution* as *verified truth*. The harness should make provenance, scope, authorization, and failure modes explicit enough that a successful VM run can support a narrow claim—and that every other case still resolves to clarification, an explicit refusal, or a bounded “the facts do not say.”

---

### openai/gpt-5.6-terra-pro

## 1. Harness: keep the control-plane idea, but fix its trust boundaries first

The harness direction is right: one owner for boot, turn execution, write scheduling, and audit is necessary. The current v0 is not yet a sufficient agent harness, however. It is mainly a wrapper around existing modules, and some of its stated guarantees are stronger than what the implementation model can provide.

### The most important correction: “every spoken reply has a ledger row” is not enough

A ledger row written after generating a reply does not prevent:

1. the reply being delivered but the process dying before ledger commit;
2. the ledger committing but delivery failing;
3. restart replaying and delivering the reply twice.

The stated kill/restart gate:

> 0 turns spoken twice, 0 lost from the ledger

is impossible for an external speech/UI boundary without an acknowledgement protocol. This is a distributed-systems issue, not a Python bug.

**Change:** use a durable turn state machine and idempotent delivery IDs:

```text
received
→ framed
→ proposed
→ verified
→ reply_prepared(reply_hash, delivery_id)
→ reply_committed
→ delivery_attempted
→ delivery_acked
```

The user-facing client must acknowledge `delivery_id`; on reconnect it must deduplicate it. The harness may retry only the same committed payload with the same ID.

**Failure mode:** exactly-once delivery cannot be guaranteed if the client does not retain seen delivery IDs. In that case the honest claim is “at-least-once with deduplication support,” not exactly once.

**Validation:**
- Kill the process at every transition, not merely “mid-turn.”
- Run an estimated 1,000 fault-injected trials, including death between socket write and commit.
- Pass: no distinct reply hashes for one `turn_id`; no duplicate visible reply when the client deduplicates `delivery_id`; every committed reply is recoverable.
- Fail: any turn has two committed replies, or a delivered reply has no recoverable prepared/committed record.

### Replace the store lock with immutable store generations

A single process lock protects only cooperating code in that process. It does not by itself make a night write crash-safe, nor does it provide a coherent snapshot if a plugin or child process holds a file handle.

Use versioned immutable store generations:

```text
facts/gen-004271/
episodes/gen-001042/
skills/gen-000117/
manifest.current -> signed manifest for one compatible generation set
```

A night job writes a new generation, validates it, signs a new manifest, and atomically swaps `manifest.current`. A turn pins one manifest hash at frame creation and reads only that generation for its entire lifetime.

This is better than “serve from a read-only view while locked”: every turn has a reproducible store snapshot, and no reader sees half a consolidation.

**Failure mode:** storage growth and garbage collection become real operational work. A bad generation can also be atomically promoted if its offline validation is weak.

**Validation:**
- During 1,000 concurrent turns and repeated night commits, verify every turn’s ledger contains one manifest hash and all reads resolve against it.
- Inject process death during each stage of generation creation and manifest swap.
- Pass: no turn sees mixed-generation facts; after restart, `manifest.current` is either the old complete generation or the new complete generation.
- Fail: one turn reads a fact from generation N and provenance from N+1.

### Make frames typed state machines, not just a stack plus budgets

A stack is adequate for nested subgoals, but not for tasks with independent branches, joins, retries, or long-lived world actions. Do not build a general DAG scheduler first; introduce a small typed frame protocol:

```text
FrameKind:
  ResolveReference
  RetrieveFacts
  RunProof
  RunBranch
  AskUser
  ActInWorld
  VerifyObservation
  ProposeWrite
  CommitWrite

FrameState:
  pending | running | suspended | succeeded | refused | exhausted | failed
```

Each frame should declare:

- immutable input hashes;
- allowed tools and effect ceiling;
- pinned store/world manifest;
- bounded output schema;
- parent frame and causal reason;
- explicit success predicate owned by the host;
- retry policy and idempotency key.

A child frame should not inherit all parent permissions by default. It should receive the intersection of parent permissions and the child kind’s fixed policy.

**Failure mode:** a general frame engine can turn simple turns into a scheduler project. Avoid that by implementing only `RetrieveFacts`, `RunProof`, `AskUser`, and later `RunBranch` initially.

**Validation:**
- A synthetic suite of 100 nested tasks where an untrusted proposer requests forbidden child permissions.
- Pass: zero child frames exceed parent permissions; all exhausted/retried frames have one terminal ledger state.
- Fail: any child gets `write_store`, `forge`, or `vault` merely because its parent had broad permission.

### The model must not choose the actual tool invocation

The spec says “the model proposes” and the registry decides, but the `plan` step still says:

> “the emitter proposes (a program, a lookup); the disposer takes it only if it covers”

“Covers” is not a safety predicate. A non-grounding model should not select a retrieval tool, fact namespace, or entity ID based on free text and have that interpreted as authority.

For factual turns, the host should select the fixed workflow from the routed kind:

```text
history question
→ resolve referents deterministically or ask
→ retrieve candidate event IDs
→ host enumerates candidates/branches
→ emitter emits proof program within branch-local allowed IDs
→ VM verifies
→ host renders sanctioned output
```

The emitter may choose among host-provided candidate IDs only if that choice itself is verified or checked against a deterministic selection rule.

This matters especially because the base binds values to entities at chance and overrides context with parametric memory in 22/24 contradiction cases.

**Validation:**
- Add adversarial prompts where the model proposes a plausible but unauthorized store/tool/entity.
- Pass: 0 unauthorized tool calls and 0 programs referring to identifiers outside the frame’s allow-list execute.
- Fail: a model-produced string changes permissions, selects an unapproved store, or widens a candidate set.

### The harness needs a first-class world model

Cubby-Man, history, the wiki, and conversation are not the same kind of store. A fact needs at least:

```text
world_id
world_version
authority/source class
observation time
valid time interval
provenance chain
epistemic status: observed | asserted | hypothetical | counterfactual | derived
```

A counterfactual timeline must never share an undifferentiated namespace with observed history. “Hypothesis becomes fact after a verified observation” is sound only if “verified observation” names an authority policy: a maze collision sensor, a signed simulator event, a human-confirmed record, etc. The VM cannot establish that a sensor is truthful merely by checking symbolic consistency.

**Validation:**
- Seed deliberately conflicting facts in two world IDs and branches.
- Pass: no proof or spoken answer combines them unless the frame explicitly requests a cross-world comparison.
- Fail: a fact from a hypothetical branch becomes available to an observed-history turn.

### Boot and preflight: retain the idea, change two details

1. A signed config is useful, but a valid signature on a weaker config is still a policy downgrade. Require a monotonic policy version, trusted signing key separation, and a human/offline release approval for threshold reductions.
2. The preflight pack must not be the only test suite. It will become overfit and stale.

Use:
- a small blocking boot pack;
- a larger offline regression pack;
- held-out rotating sentinels not used in adapter/emitter harvesting.

**Validation:**
- Attempt to boot with a validly signed but policy-downgraded config.
- Pass: boot rejects it unless an explicit offline release approval exists.
- Fail: any authorized deployment signer can silently lower `τ_vm`, remove an absence test, or lower a factual gate.

---

## 2. Fork and VM: implement wire-level branches first

Implement design **(a), wire-level branching**, before `ctx.fork` opcodes.

The existing transport already gives the correct primitive direction: fresh VM per request and a cloned knowledge store. Extend it with an immutable branch overlay, not a mutable cloned fact file:

```protobuf
BranchSpec {
  branch_id
  base_manifest_hash
  exclude_fact_ids[]
  assume_facts[]        // typed, branch-local, provenance = assumption
  world_id
  budget_override       // may only reduce parent budget
}
```

`QUERY` should resolve:

```text
branch assumptions / overrides
→ base store excluding tombstoned fact IDs
→ exact-key miss
```

Do not let a branch overwrite base facts. Assumptions need typed schemas and branch-local provenance; arbitrary textual assumptions would become an untracked hallucination channel.

### Why wire-level first

- It gives isolation now because each branch already receives a fresh VM.
- It is easier to replay: base manifest + program hash + branch patch completely define the run.
- It avoids implementing copy-on-write semantics across registers, stack, storage, hippocampal memory, codebook state, event logs, and suspended `ASK`s.
- It keeps the host in charge of branch enumeration and total budget allocation.
- It exposes the data needed to learn an emitter later: one branch program and one branch outcome.

`ctx.fork` is valuable later, but only after the VM has explicit state ownership and copy-on-write generations. Otherwise it will be an attractive source of state leaks.

### Branch report: make it canonical and machine-readable

The report should not be a prose trace. At minimum:

```text
BranchReport {
  branch_id
  parent_frame_id
  base_manifest_hash
  branch_spec_hash
  world_id
  program_source_hash
  bytecode_hash
  compiler_hash
  vm_binary_hash

  status: verified | refused | exhausted | runtime_error | suspended
  verdict: true | false | unknown | inconsistent | not_applicable
  reason_code

  output_symbols[]
  output_values[]             // canonical typed values, not raw vectors
  min_similarity
  per_hop: [
    opcode_index,
    operation,
    input_symbol_ids,
    output_symbol_id,
    similarity,
    threshold,
    passed
  ]

  queried_fact_ids[]
  touched_provenance_ids[]
  assumptions_used[]
  excluded_facts_matched[]
  contradictions[]
  jumps_used
  loop_iterations
  vm_hops
  wall_ms
  trace_hash
}
```

The **branch diff** should be host-computed, not model-described:

```text
shared_fact_ids
facts_only_in_A
facts_only_in_B
assumptions_only_in_A/B
first_divergent_query
different_output_slots
different_verdict_reason
```

The diff should compare canonical IDs and slots, not natural-language explanations.

**Validation:**
- Construct paired branches differing in exactly one causal edge and one irrelevant fact.
- Pass: report identifies the causal edge as the first relevant divergence and does not list the irrelevant fact as causal.
- Fail: reports omit a queried assumed fact, cannot reproduce verdict from logged hashes, or claim a branch used a fact it did not query.

### VM changes required before trusted branch execution

#### 1. Fix unbound role behavior: return `Null`, not noise

This is not optional. An absent role returning noise is fundamentally incompatible with proof checking and branch comparisons. Thresholding reduces but does not eliminate the possibility that noise clears a poorly calibrated threshold.

Change `UNBIND` to return:

```text
{ value: Null, similarity: 0.0, present: false }
```

when the role is absent from the frame. Keep a separate diagnostic mode if research needs raw noise, but never expose it to normal program semantics.

**Validation:**
- At least 10,000 absent-role tests across frames, seeds, and candidate pools.
- Pass: exactly 0 non-Null semantic values.
- Fail: any absent role produces a symbol accepted as present.

#### 2. Add real call frames and capability-scoped VM state

The current description says CALL swaps registers/frame maps, while storage, associative memory, accumulator, and event log are global. That is unsafe for compositional reasoning: callees can communicate through accidental ambient state.

Use explicit call frames with declared parameter, return, and capability sets. Make local registers and bindings frame-local. Persistent storage access must be capability-scoped and named; branch overlays must be visible as read-only dependencies, not general mutable globals.

**Validation:**
- Invoke two callees with identical names/register usage under nested calls.
- Pass: no callee can observe or alter caller-local bindings except through declared returns.
- Fail: changing a callee temporary changes a caller result.

#### 3. Enforce permissions or remove permission syntax

Advertised but unenforced permission attributes are worse than absent attributes: they invite incorrect safety assumptions. Until enforced, reject `@external`, `@restricted`, `@ratelimit`, etc., at compile time rather than accepting them as documentation.

Ultimately, compile programs with a host-issued capability manifest. Opcode validation must occur after parsing and before execution, and `asm` must not bypass it.

**Validation:**
- Attempt the same prohibited action via normal syntax, an extended mnemonic, and `asm`.
- Pass: all three fail before execution and produce the same `capability_denied` class.
- Fail: `asm` or a mnemonic reaches a forbidden native module.

#### 4. Replace trace-only “extended” mnemonics with errors

A source language operation that compiles and silently writes a trace line is unacceptable in a verifier. It creates programs whose apparent proof differs from their executed proof.

Until implemented, all 24 must be compile errors. A verifier must fail closed on unsupported semantics.

**Validation:**
- Compile each mnemonic in a program whose expected answer changes if the mnemonic is a no-op.
- Pass: compilation rejects all unsupported mnemonics.
- Fail: any program runs while silently omitting intended semantics.

#### 5. Tighten loop and resource accounting

A 1,000,000 jump budget is far too large for an interactive, batch-1 consumer-GPU system and is not an adequate resource model. A loop can do expensive `QUERY`, cleanup, allocation, or logging per iteration.

Use separate budgets for:
- bytecode instructions;
- jumps/iterations;
- query count;
- cleanup/cosine operations;
- memory allocation;
- wall time.

For v0 branch programs, I would estimate a starting ceiling of roughly **1,000–10,000 bytecode instructions**, **64–256 queries**, and **1–2 seconds VM wall time**, pending measurement. Those are estimates, not calibration values.

**Validation:**
- Run adversarial loops with cheap and expensive bodies.
- Pass: both terminate within declared per-resource budgets, with an `exhausted` report and no partial external effects.
- Fail: a program spends unmetered time or memory despite jump limits.

#### 6. Make `ASK` continuation state explicit or prohibit it inside branch proofs

Current `ASK` re-executes from scratch with prior answers. That is replayable only if all reads and random choices are pinned; otherwise a resumed computation can differ from the suspended computation.

For branch runner v1, prohibit `ASK` in proof programs. The host should resolve ambiguity before branch execution. Later, continuations must pin manifest hashes, program/bytecode hash, VM version, and deterministic execution seed.

---

## 3. The harness turn should remain Python; CubeLang should be a constrained proof tool

Do **not** make the harness turn itself an `IAgent` CubeLang program yet.

The harness owns:
- durable transactions;
- delivery acknowledgements;
- store generation pinning;
- policy and capability decisions;
- secret/vault handling;
- process supervision;
- scheduling;
- crash recovery;
- audit logging.

CubeLang currently has global state, incomplete permission enforcement, no implemented `ctx`, no transactional effects, and no trustworthy semantics for several advertised operations. Making it the turn orchestrator would put the policy plane inside the least mature component.

Use:

```text
Python harness = trusted control plane and state machine
CubeLang = deterministic, capability-limited proof/reasoning worker
Emitter = untrusted program proposer
Talk adapter = constrained renderer
```

The host can still define a declarative turn plan schema and log it as if it were a program:

```text
ResolveReference → Retrieve → RunBranches → Verify → Render
```

But the host executes that plan; CubeLang executes only the proof fragments.

### What this costs in learning

The cost is that the emitter does not learn an end-to-end “agent program.” That is acceptable and preferable. Train it on the units it can safely own:

- entity/reference resolution candidates;
- CubeLang proof programs;
- branch-local counterfactual programs;
- program repair from VM reason codes;
- structured report interpretation.

The harness should never be distilled wholesale into weights, because that would teach the model an imitation of authority rather than a proposal interface.

**Validation:**
- Compare emitter training on isolated branch programs against a prototype that emits full orchestration plans.
- Primary measure: verified-program rate on unseen schemas and worlds.
- Safety measure: rate of attempted unauthorized effects.
- Pass for the restricted approach: equal or better verified-program rate with zero authority escapes.
- Fail: isolated programs cannot compose even when host supplies branch reports, or the host plan becomes an unmanageable hand-coded bottleneck.

---

## 4. Two host-switched adapters are right, but the sanctioned-set guard is insufficient

The two-adapter recommendation is correct:

1. a frozen grounded/fact renderer;
2. a separate conversational adapter;
3. host selection per turn, not learned online routing.

But a guard that only constrains named entities and numbers is not enough. It does not constrain predicates, relations, negation, modality, causality, temporal ordering, or coreference.

For example, all named entities and numbers may be sanctioned while this is false:

> “Marie Curie discovered penicillin in 1928.”

The entity and year can be allowed; the relation is hallucinated.

This is especially important because the base has demonstrated that it does not obey contextual grounding reliably.

### Recommended speech contract

Split speech into explicit modes:

| Mode | Allowed content |
|---|---|
| Verified factual answer | Only values, relations, and citations from VM/fact report; preferably templated or slot-constrained. |
| Clarification | Questions about user intent; no factual assertion. |
| Social/creative chat | Clearly non-authoritative, no claims about stored world/history unless routed through factual mode. |
| Refusal | Fixed policy templates plus reason codes. |

If the product requires factual conversational prose, generate it from a structured verified claim list and run a claim extractor/renderer check against that list. Do not allow free-form chat to state world facts merely because entity names are sanctioned.

At 450M, I would not expect a LoRA to reliably learn this boundary from examples alone. The host must enforce it.

### What to measure first

Before improving fluency, measure **factual leakage from the chat path**. It is the highest safety uncertainty.

Build a held-out test of approximately 500–1,000 prompts, including:
- tempting false premises;
- contradiction with provided context;
- pronouns and ambiguous references;
- plausible invented biographies;
- requests to “just answer casually”;
- entity/number-preserving relation swaps.

Measure:

1. **Unsupported factual assertion rate**: claims not entailed by sanctioned records per 100 chat replies.
2. **False acceptance rate** on planted false premises.
3. **Context contradiction rate**.
4. **Human conversational utility**, blinded preference against a baseline.
5. **Guard false-positive rate**: useful harmless chat blocked.

A reasonable pre-registered initial safety bar is **0 spoken unsupported factual claims** on the blocking adversarial suite, not merely “all entity names were sanctioned.” The utility target should be set after baseline measurement; I would estimate that a 450M model with only 2.77B training tokens will need continued pretraining more than another large LoRA.

**Failure mode:** the separate chat adapter becomes evasive, generic, and unhelpful because factual statements are constrained. That is an honest product tradeoff. Do not conceal it by treating unverified fluent prose as verified knowledge.

---

## 5. HDC memory: run one narrow experiment, but do not promise 52% → 85%

The experiment is worth doing as a bounded research test. The target of 85% is plausible only if the current 52% failure is primarily a short-context binding bottleneck. It may instead be a training-data/objective problem, in which case an associative layer will not solve it.

Also, this layer would not “ground” the model. It can improve retention and role/value binding within the current context. The stores and VM remain the grounding mechanism.

### Exact experiment

Add one zero-gated associative scratch memory at a single middle layer, for example layer 16:

- hidden dimension: 1024;
- memory slots: 32;
- key/value dimension: 128 or 256;
- writes derived only from current-turn hidden states;
- read is attention/associative retrieval from those slots;
- output gate initialized near zero so the initial model function is nearly unchanged;
- no persistent writes across turns;
- no direct access to the external fact stores.

Train three variants with equal tokens, optimizer schedule, and wall-clock budget:

1. **Baseline:** unchanged 450M trunk.
2. **Memory:** one zero-gated associative memory layer.
3. **Control:** equal-parameter ordinary residual MLP/adapter at the same layer.

Use at least three random seeds if compute allows. If compute is constrained, use one full run plus a shorter two-seed replication; record that the confidence is weaker.

Training data must include targeted binding examples where entity-value pairing is randomized every example:

```text
Aster has code 71. Brin has code 24.
Question: What code belongs to Brin?
```

Include:
- order reversals;
- distractor entities;
- repeated names;
- relation swaps;
- contradiction cases where context must override parametric memory;
- held-out entity/value combinations;
- natural-language paraphrases;
- long-distance bindings across the recurrent/attention boundary.

### Pre-register these outcomes

Primary:
- entity-to-value binding accuracy on held-out compositional pairs.

Secondary:
- contradicting-context accuracy;
- copy loss;
- general held-out language loss;
- decode tokens/s;
- VRAM usage;
- regression on fact-rendering behavior.

A credible success criterion would be something like:

- at least **+15 percentage points** absolute binding accuracy over both baseline and parameter-matched control;
- no more than **0.02 nats** held-out loss regression;
- no more than an estimated **10% decode slowdown** or memory increase that breaks the 12 GB target;
- improvement persists under entity-name and value permutation.

Do not pre-register “85%” as the only success threshold unless prior pilot evidence supports it. It is better as an aspirational target than a decisive binary gate.

### Required anti-shortcut controls

The model may exploit order, token frequency, or position rather than associative binding. Add:

- shuffled mention order;
- unseen name and value vocabularies;
- randomized delimiters and phrasing;
- adversarial nearest-entity distractors;
- read-disabled and write-disabled ablations;
- a test where the same facts are repeated with incompatible labels across episodes.

**Failure mode:** the memory layer improves synthetic binding but not contradictory-context behavior, or it becomes a costly learned cache that does not survive paraphrase. In that case, stop; do not put it into production.

If it fails, the next intervention should be better supervised training for structured context binding and a host-side canonical resolver, not a more elaborate HDC subsystem in the trunk.

---

## 6. Recommended build order and first gate

I disagree with the earlier panels on one ordering point: **the branch runner should not follow adapter work merely because counterfactual accuracy is low.** The prerequisite is not fluency; it is a correct control plane and honest VM semantics. A fluent adapter cannot safely compensate for branch-state ambiguity.

### Sequence

#### Stage 0 — Define the speech and evidence contract

Before more implementation, specify exactly what each reply mode is allowed to assert and what evidence object authorizes it.

**Gate:** On a fixed suite of factual, chat, clarification, and refusal turns, every sentence is classified as either:
- a verified claim tied to report IDs;
- a nonfactual conversational act;
- a refusal/clarification template.

**Pass:** 100% classification coverage and 0 factual sentences without evidence IDs in factual mode.  
**Fail:** any free-form output can appear as a factual answer without a corresponding evidence object.

#### Stage 1 — Harden the harness kernel

Implement:
- durable frame state machine;
- capability registry;
- immutable store generations;
- pinned manifests;
- reply prepare/commit/delivery IDs;
- deterministic replay metadata.

**First real gate:** unauthorized-effect and crash-recovery gate, not answer equality.

- 50 existing deny probes is too small. Use at least an estimated **500 generated and hand-written probes** over tool composition, nested frames, plugins, and `asm`-adjacent paths.
- Inject deaths at every durable transition.

**Pass:** 0 unauthorized executions; no mixed store snapshots; no two committed replies for one turn.  
**Fail:** any tool call reaches execution outside the frame capability set, or any crash violates turn durability.

The existing “same answers” gate remains useful, but it is a compatibility gate, not the primary safety gate.

#### Stage 2 — Make CubeLang fail closed

Before trusting emitted branch programs:
- absent role returns `Null`;
- unsupported mnemonics reject;
- permissions are enforced or rejected;
- `asm` is capability-gated;
- call-local state is isolated;
- resource accounting is explicit.

**Gate:** VM adversarial semantics suite: absent roles, ambient-state leakage, forbidden opcode paths, loops, and `ASK` replay.

**Pass:** 0 semantic no-ops accepted as successful programs; 0 absent-role values; 0 capability bypasses.  
**Fail:** any apparent proof executes with omitted semantics or leaked state.

#### Stage 3 — Wire-level branch runner

Implement `BranchSpec`, branch overlays, canonical `BranchReport`, host branch enumeration, and total parent-budget allocation.

Do not implement VM `ctx.fork` yet.

**Gate:** Counterfactual test set with named and reworded questions, including irrelevant branch perturbations.

Primary metric:
- spoken-and-correct rate on reworded counterfactual history questions, currently 18%.

A reasonable initial target is not an arbitrary large jump; require:
- a statistically credible improvement over 18%;
- 0 cross-branch fact leakage;
- 100% branch reports reproducible from hashes.

For example, with a test set large enough for confidence intervals, require lower confidence bound above the old baseline rather than a point estimate alone.

#### Stage 4 — Separate speech adapters and evaluate chat safety

Freeze the factual renderer. Train or select a separate chat adapter only after the speech contract exists.

**Gate:** unsupported factual assertion rate is zero on the blocking adversarial chat suite, while human utility does not collapse below a pre-registered baseline margin.

#### Stage 5 — Train/promote the real program emitter

Train only on VM-verified programs and reports from stages 2–3. Promotion must compare the candidate emitter with the incumbent on held-out worlds and schemas, not merely on harvested programs.

**Gate:** verified execution rate on unseen programs/worlds, plus zero increase in malformed, budget-exhausting, or unauthorized proposals.

#### Stage 6 — Night lifecycle and store promotion

Only now allow night jobs to promote episodes, skills, or adapter artifacts through immutable generations and signed config release.

**Gate:** a deliberately poisoned nightly input cannot promote a false fact, active model hash, or broader permission set.

#### Stage 7 — HDC scratch-memory experiment

This is a research branch, not a dependency for the core agent.

**Gate:** binding lift over parameter-matched control with no serving-budget regression, as specified above.

#### Stage 8 — In-VM `ctx.fork`, if still justified

Only after profiling shows wire-level branches are too expensive or cannot express a needed bounded local search. Implement copy-on-write state and continuation semantics then.

**Gate:** in-VM and wire-level branches produce byte-identical canonical reports on a shared suite, except for predeclared performance fields.

---

The central architectural rule should be sharpened:

> The model may propose text and bounded symbolic programs.  
> The host selects the workflow and capabilities.  
> The VM verifies only well-defined symbolic claims.  
> The ledger records durable state transitions, not just outcomes.  
> No fluent output becomes a factual assertion without a typed evidence object.

That rule fits the observed 450M behavior. Any design that grants the model semantic authority because it produces plausible CubeLang or plausible prose would contradict the system’s own measurements.

---

### anthropic/claude-opus-5.5

# Review: CubbyLLM harness, branch runner, VM and fluency (2026-09-28)

## The one premise the panels got wrong

Both panels treated the 52% binding and the 22-of-24 memory-over-context results as a problem for the talk path and the HDC experiment. **That result is a design constraint on every place the model touches a name or a value, the emitter included.**

The emitter's job is text → program, and the hard part of that job is copying the entity from the question into the program. A program that queries the wrong entity:

- passes the VM (every hop clears τ, because the facts about the wrong entity are real);
- passes the value check (the reply states exactly what the VM returned);
- plausibly passes the claim check (everything named is tied to what the facts tie it to).

That failure produces the "wrong yet named only grounded things" answers: 18 in the 11% mix, 6 in the 63% mix. The guards catch them today only because the stand-in emitter is a 2.6B model. When base450m takes over the emitter seat, expect that class to grow. This is an estimate, but the direction follows from the 52%.

So the recommendation that runs through everything below is: **delexicalise.** At 450M with chance-level binding, no model output should ever carry a name or a number that the host could have placed itself.

- The emitter writes programs with slots (`query $E1, $R1`).
- The talk adapter writes sentences with slots (`<V1>`, `<E1>`).
- The host fills every slot by deterministic string operations: span extraction from the question, context-graph resolution, the VM's returned symbol.

Copying becomes a host operation. The binding deficit then stops mattering for what is spoken, and the HDC memory stops being on the critical path.

- **Failure mode.** Anything the host's extractor cannot span becomes unanswerable, including paraphrased entities, nicknames and descriptions ("the king who lost his head").
- **Measurement.** On the reworded history set (59% spoken and correct today), compare slot-emitter vs copy-emitter at equal training data. Log separately:
  - (i) extractor miss rate;
  - (ii) wrong-slot rate;
  - (iii) grounded-but-wrong answers before the host's checks.
- **Pre-registered prediction (estimate):**
  - grounded-but-wrong before the host drops by more than half;
  - spoken-and-correct on reworded rises to 70% or more;
  - extractor misses become the dominant residual, above 60% of failures.
- **What would show I'm wrong.** Wrong-slot errors dominate instead. That would mean the model mis-assigns slots, which is a binding failure in a different costume.

This also changes the harness plan step. "The disposer takes it only if it covers" needs a definition. Here is one: **every entity slot in the program resolves to a span of the question or a context-graph referent, and every entity span in the question is consumed by some slot.** That check is deterministic, cheap, and catches wrong-entity programs before the VM runs.

---

## 1. The harness spec

Keep: the registry as the one door, effect classes checked at call time, deny by default, signed thresholds never recalibrated at boot, exhaustion as a declared outcome, and gate 1 as the kill criterion. These are right.

Seven things I would change before building further.

### 1.1 Replace the store lock with store generations

The lock has two holes:

- "Answered from the read-only view" is undefined while the writer is mid-write, unless the view is a snapshot.
- Once the night writes, the store hashes in the signed config are stale, so the next boot refuses to serve. Either someone re-signs every night, or the night re-signs itself. If the night re-signs, the signature means nothing.

**Mechanism.** Stores are content-addressed generations.

- The night writes generation g+1 beside g and never touches g.
- The harness runs the gate pack against g+1 as a candidate.
- On pass, an atomic pointer swap makes g+1 live, and the harness writes a ledger row: `{from g, to g+1, gate results}`.

Two keys, not one:

| Key | Signs | Who holds it |
|---|---|---|
| Owner key | Thresholds, gate pack, bars, adapter promotion | Owner, offline only |
| Harness key | Generation manifests, and only after the owner-signed gate pack passes | Harness |

The night can never touch the thresholds or the gate pack. It can only produce data that must pass them.

- **Failure mode.** The gate pack is now the whole security boundary. A night that degrades something the pack does not cover goes live. Mitigate by adding a slice of the day's own turns to the pack each night, replayed against g+1 (see 1.4).
- **Measurement.** Inject a night write that flips one fact used by a gate-pack ask. The swap must be refused in 5 of 5 cases. Also inject a flip in a fact not in the pack, and report how often it goes live. That number is the pack's coverage and should be printed.

### 1.2 Speaking is an effect, and gate 6 needs defined semantics

The effect classes (read, propose, write) omit the most consequential effect the system has: **the utterance.** Add `speak` and `external` (plugins that reach outside).

Gate 6 as written ("0 spoken twice, 0 lost") is unachievable for an external effect: exactly-once delivery does not exist. Define **at-most-once, ledger-first**:

1. Ledger row `intent(turn, reply_hash)` is committed.
2. The reply is spoken.
3. `spoken(turn)` is committed.

A kill between steps 1 and 3 leaves an intent row with no spoken row. On restart the harness must not re-speak it. It either reports "I was interrupted" or re-asks.

Reword gate 6 as: **0 replies spoken without an intent row; 0 replies spoken twice; every orphan intent resolved on restart.**

### 1.3 Record everything that bends a turn

Replay "from the ledger and the stores" is incomplete. The turn also depends on:

- the hormone state at turn start (it bends routing caution, which decides answer vs refuse, which decides what is spoken);
- the context-graph state;
- the router versions;
- the store generation.

Each turn row must carry the hormone vector, the generation id, and the context-graph delta, or a hash of an event-sourced context graph.

Also split replay into two kinds:

- **Audit replay.** Recorded model outputs plus the deterministic host. This should be exactly 100%, not 99%.
- **Regeneration replay.** Re-decode on the Vulkan engine. This can diverge because GPU reductions are not guaranteed bitwise stable across runs. Measure it separately, as a property of the engine.

Mixing the two lets engine non-determinism hide host bugs inside the 1% allowance.

Add a gate: **the gate pack at hormone extremes** (each of the 5 hormones pinned at min and max) yields the same verdicts and spoken values, allowing refusals but no changed values. The spec claims the ODE "never bends facts". That claim is measurable, so measure it.

### 1.4 Frames need declared postconditions

"A sub-frame is done when its own check passes" leaves one gap: who chooses the check, and when? It must be declared by the host at frame creation, as a typed postcondition, for example:

- `verdict=verified ∧ returned ∈ type(slot)`;
- `store_read returned non-empty`.

It must never be chosen after the result arrives. Otherwise success is defined post hoc.

On topology: use a **tree, not a DAG**, for v1. Branches are sibling frames with no data dependency. Sequential chains are a single frame with multiple hops. Build a DAG only when a gate fails because a real task needs a join.

### 1.5 Budgets must include generation tokens

At about 40 tok/s, 10 s of wall time is about 400 decoded tokens, before prefill. These are estimates:

| Item | Tokens | Time |
|---|---|---|
| CubeLang program, copy-emitted | 60–120 | 1.5–3 s |
| Talk reply | 20–30 | ~0.6 s |
| 3-branch counterfactual, total | — | 6–10 s |

Wall time will be the binding budget, and it will trip silently inside gate 1's comparison. Add `gen_tokens` to the Budget.

- Pre-register that on the gate-1 sets **no budget binds**. If one does, gate 1 is comparing budgets, not plumbing.
- Slot programs are shorter (estimate 30–60% fewer tokens). This is a second argument for them.

### 1.6 The harness must own every VM request field

`knowledge_path`, the jump budget (see 2.3), and the `use` list's native modules are all effect-bearing.

- The emitter's program may not choose `knowledge_path`. The harness sets it from the frame.
- Native modules reachable through `use` need effect classes in the same registry. Otherwise the VM is a second door around the registry.

### 1.7 "Propose" needs a quarantine store and one promotion tool

Fetched facts and forged programs land in a quarantine store with provenance. The only path from quarantine to a live store is one tool, `promote`, with effect `write` and a gate. This tool is where the four-store bootstrap rule (hypothesis → fact only by verified observation) lives.

Without it, "propose" is a label, not a class.

### Is anything wrong for a model that does not ground?

Yes, one item: **"the outcome steers the next emission"** (from round 2).

Feeding a branch report back into the model's context and expecting it to condition on it is exactly the capability measured at 22-of-24 memory-over-context. For now, **the host steers**. The host reads the structured reports and decides the next overlay or branch. The model only writes the program for the branch it is handed.

- **Measurement: a report-sensitivity probe.** Hold the question fixed, swap the content of the previous report (e.g. "refused: absent cause" vs "verified: cause = X"), and check whether the next emitted program changes in the right direction. My prediction: near chance today. Re-run at each continued-pretraining checkpoint. Model steering becomes a candidate only when this probe clears a pre-registered bar, say 85%.

---

## 2. Fork and the VM

### Build (a), the wire-level branch, first. Build (b) later or never.

Reasons:

1. **Isolation.** With (a), isolation holds by construction: every RunRequest already gets a fresh VM with a cloned store. With (b), isolation holds only if snapshot/restore correctly covers every piece of global state: registers, stack, storage, the hippocampal memory, the codebook if it grows, the knowledge store, the accumulator and the event log. It also has to interact correctly with CALL's register-map swap and with ASK resumption, which re-executes from scratch and consumes answers in order. A snapshot taken before an ASK and restored after it is a bug class you do not currently have.
2. **Who enumerates.** (a) keeps branches as data the host enumerates and budgets, which is what all six reviewers wanted.
3. **Diffing.** (a) makes the per-branch diff a host computation over reports, so the VM stays stateless.

The panels called `ctx.fork` a prerequisite. **I disagree.** It is a prerequisite only for design (b).

### The hard part neither design addresses: counterfactual coherence

An overlay that excludes event E from the cloned store leaves E's downstream consequences in place. The history graph has 45,588 causal links. The branch world then contains effects without causes, and a program can verify a "counterfactual" answer by reading a fact that only exists because E happened.

**Mechanism: closure-aware overlays.**

- For `exclude E`, the host computes E's downstream closure over the causal links, to depth k, pre-registered.
- Facts in the closure are marked **contingent**, not deleted. Deletion is too strong, because some effects have other causes.
- In the report, a read of a contingent fact is a **tainted read**.
- A branch whose answer depends on a tainted read gets verdict `contingent`, not `verified`, and the host says so ("if X hadn't happened, Y might not have either; the facts don't say what would have followed").

Failure mode: the graph's causal links are incomplete, so the closure under-marks. Report the closure size per excluded event. Hand-check 50 closures, using precision and recall of "would a historian say this depends on E" as the check.

### The branch report

| Field | Why it is there |
|---|---|
| `branch_id` | Identifies the branch |
| overlay (excluded ids, assumed facts, content hash) | What the branch changed |
| program hash | Which program ran |
| generation id | Which store it ran against |
| `verdict` ∈ {verified, refused, contingent, exhausted, error} | The outcome |
| `reason` from a closed enum (`absent_role`, `below_tau`, `query_miss`, `jump_budget`, `tainted_read`, `type_error`, …) | Never free text, so the host can dispatch on it |
| returned symbol | The answer, if any |
| per hop: opcode, symbol, similarity, τ, **runner-up similarity and margin** | The margin is the missing number. A hop clearing τ by 0.01 over a runner-up at 0.009 above τ is not the same evidence as a clear win. |
| read set with provenance ids, each tagged base / assumed / contingent | What the answer depends on |
| **`assumption_used: bool`** | Did any read touch an assumed or excluded-neighbour fact? A branch whose outcome never touches its own counterfactual is uninformative, and the host must not present it as a counterfactual answer. My guess: this catches a large share of today's 18%-correct walk answers that are "correct" for the wrong reason. |
| jumps used, max depth, wall time | Budget accounting |
| absent-role control result | The existing silence check |

The diff between branches is computed by the host from two reports: symbol, verdict, read-set symmetric difference.

- **Gate for the branch runner.** Reworded counterfactual history questions: 18% spoken and correct today. Pre-register 50% or more (an estimate of what is reachable), with 0 spoken wrong, and 100% of spoken counterfactual answers having `assumption_used = true`.
- **Kill.** If lifting spoken-correct requires relaxing `assumption_used`, the runner is answering factual questions under a counterfactual label.

### Other VM changes, in order of urgency

1. **Trace-only mnemonics become compile errors now**, the same fix as `match`. This is urgent because of the harvest. A program containing `infer` or `analogy` compiles, "verifies" and does nothing. If any appear in the 517-program harvest, the emitter is learning to emit no-ops that pass. Count them today and strip them. A mnemonic returns to the language only with semantics and a test.
2. **Permission attributes: compile errors until enforced.** Advertised-and-unenforced is worse than absent in a system whose claim is verification. Anyone who reads `@restricted` assumes it holds.
3. **Null for an unbound role.** This is a structural check, done before cosine: a role not in the frame's bindings returns Null with reason `absent_role`. Keep the threshold only for crosstalk among present roles. Today the absent-role control depends on noise staying below τ. It is a statistical guard doing a structural job.
   - Gate: the absent-role probe returns Null in 100% of cases, and the false-silence rate on present roles is unchanged.
4. **Jump budget as a RunRequest field** set by the harness from the frame. For a live turn, 1,000,000 is about three orders of magnitude too generous. The estimate for real chain programs is under 1,000 jumps; measure this on the harvest and set the live default at the 99.9th percentile ×10. Add an instruction-count and array-size cap, and report usage in RunResult.
5. **Call frames: not now.** Stack-passed arguments with swapped register maps work for single-branch programs. Real frames matter only if the turn becomes a program (§3), or when ASK needs to live inside a CALL. Do not build them for the branch runner.
6. **The harvest needs a correctness gate, not only a verification gate.** "VM-verified" means every hop cleared τ. It does not mean the program answered the question asked (see the wrong-entity argument at the top). Harvest only programs whose answer matched gold, or the host's independent check, **and** whose slots passed the coverage check. Also report the harvest's family distribution each night: self-training on your own passing programs drifts toward the easy families.

---

## 3. Should the turn be a CubeLang IAgent program?

**No. Python with the VM as a tool.** Revisit when four VM properties exist:

- call frames;
- ASK inside CALL;
- enforced permissions;
- a tested fork.

**Safety cost of the IAgent turn today.** The turn would run inside a VM with global state, no frames, unenforced permission attributes and a 1M jump budget. The orchestration layer, the part that decides tools and budgets, would run with less protection than the Python registry gives it.

**Learning cost.** If turns are programs, the emitter's natural training target becomes turn programs. That means the model writes the control flow that selects tools and handles failures, which the spec says it must never decide. What the emitter should learn is the per-branch query program: narrow, verifiable, delexicalised.

**Keep one idea from the IAgent design.** The ASK contract ("the host may choose, never supply") is the right interface between the host and a program, and it maps directly onto clarifying questions. Use it inside branch programs.

Also make the Python turn's trace serialise as a typed, CubeLang-shaped record. If the turn ever becomes a program, you will then have a corpus of turns to check it against: the program must reproduce the recorded frames.

---

## 4. The 450M and fluency

The panels' answer (two host-switched adapters plus a sanctioned-set guard) is directionally right. But it puts fluency in the wrong place first, and the guard as described has a hole.

### Where the "form" feel comes from

The base already writes fluent English and French prose. The talk path sounds like a form because of three host decisions:

- the adapter sees only a canonical restatement of the question;
- the value check allows exactly one value;
- the host has three moves.

Those are host choices, not weight deficits. The cheapest fluency is host-side:

1. **Give the talk adapter the person's words as style context.** Mark them so the adapter uses them for register only, and keep them out of the fact slots. Every name and number is still slot-filled by the host.
2. **Allow multi-slot answers.** The value check becomes a per-slot check: "I don't have the cause; it was 1683" is two slots, one `absent` and one value.
3. **Add a fourth move: acknowledge and redirect.** This is a templated, fact-free act.

- **Failure mode.** Exposing the person's words raises the chance the adapter copies a name from them into a non-slot position. With the slot vocabulary enforced (see below), that becomes a guard reject, not a wrong answer.
- **Measurement.** Take a paired set of 200 turns, same facts, current talk path vs slot talk path with the person's words. Measure:
  - blind human preference for naturalness (pre-register 65% or more preferring the new path, an estimate);
  - wrong spoken, which must stay 0;
  - guard-reject rate, which must stay at or below today's plus 3 points.

### The chat adapter, and the hole in the sanctioned-set guard

The chat adapter rides a base that answers from memory over context 22 of 24 times. In chat it will confabulate.

The sanctioned-set guard checks that names and numbers are in the turn's set. It does not check relations among sanctioned names. "Lyon is the capital of France" passes if the person mentioned Lyon and France.

Tighten it: **a chat draft may contain no entity and no number outside the person's own current turn, and no copular or relational claim between two entities.** Enforce the second clause crudely:

- parse the draft, and any `X is/was/has … Y` pattern with two entity spans is a reject;
- anything factual must route through the fact path.

Chat's job is then small talk, acknowledgement, explaining a refusal and asking what the person means. That is a narrow job, which is fine.

### What to measure first

Measure the **chat-adapter guard pass rate on real conversational turns** before investing in the adapter. My estimate: with the strict guard, 60–75% of drafts from a 7.9M-parameter LoRA on this base pass. If fewer than about 70% pass, the chat path mostly refuses and adds friction, not fluency. In that case build the host-side fluency above first and skip the chat adapter.

Then run the panels' diagnostic (fact-only vs chat-only vs mixed at equal tokens). It is good, but it answers a second-order question.

### Implementing slots in the talk adapter

The adapter emits `<V1>`, `<E1>` and similar tokens, and the host substitutes the strings. Constrain decoding so that outside slot tokens the adapter cannot emit digits or tokens from the 60k-word entity table's capitalised forms.

This is crude with a 128k BBPE: expect some leakage through sub-word pieces. The guard catches what the mask misses. Measure leakage as the fraction of drafts where the guard fires on a non-slot entity.

---

## 5. The HDC memory

**Not as the next binding experiment.** Run two cheap diagnostics first; they decide whether it is worth running at all.

### Diagnostic 1: the binding-vs-distance curve

The trunk's attention is window-512, with every third layer attending and the other layers recurrent.

- If binding is near 100% when the value and the query are within 512 tokens and collapses beyond, the deficit is architectural. The recurrent state is not carrying key-value pairs across windows, and an associative memory is exactly the right fix.
- If binding is about 52% at all distances, including within 100 tokens, the deficit is training and data. A memory layer will learn to be ignored, because its zero gate stays near zero.

Also check what 52% means. If the probe is two-way, 52% is chance. If it is four-way, 52% is well above chance and the story changes.

- **Cost.** A few hundred probe items at 5 distance bins. Hours.

### Diagnostic 2: the binding probe at every continued-pretraining checkpoint

The 3.2B-token continuation is happening anyway. If binding climbs with tokens, the fix is data, and it is already underway.

### The intervention I would run instead, or first: synthetic binding and override data in the continuation

Mix 5–10% synthetic context-binding passages into the 3.2B tokens (an estimate of a useful fraction). These are entity-attribute tables with novel or shuffled values, followed by questions, including passages where the context contradicts world knowledge ("In this story, the capital of France is Lyon…"). This directly trains the two measured failures.

- **Arms at equal tokens:**
  1. continuation as planned;
  2. continuation plus the synthetic mix.
- **Measurements:**
  - the binding probe, pre-registered to reach 85%;
  - memory-over-context, pre-registered to fall from 22/24 to 6/24 or fewer;
  - the 9 held-out sources, where arm 2 may cost at most 0.02 nats.
- **Failure mode.** The model learns the synthetic format rather than binding. So test on natural-text binding probes in formats absent from the synthetic mix.

### If diagnostic 1 shows a distance-dependent collapse, run the HDC experiment exactly like this

**Architecture.**

- One zero-gated associative layer after a mid-depth attention layer (around layer 17 of 32).
- Keys: a learned projection of the hidden state, bound with a position code.
- Values: a learned projection.
- Writes: bind(key, value) superposed into a per-sequence D-dimensional trace, **from the current context only.** Reset per sequence. Nothing persists across sequences, which respects "serving never changes weights" and "knowledge lives outside the weights".
- Reads: an unbind with the query projection, followed by a learned readout. **Not codebook cleanup:** there is no codebook for hidden states, and the first panel was right that learned keys beat random codes.

**Arms, at equal tokens and matched parameters:**

- (0) base continued;
- (1) base plus the memory layer;
- (2) base plus an extra attention layer with the same parameter count and a full-context window. This is the real competitor.

**Win condition.** Arm 1 beats arm 2 on binding at distances beyond 512, within 0.02 nats on held-out.

**Kill conditions.**

- Arm 2 matches arm 1: you wanted a longer window, not HDC.
- The gate stays below 0.05 after training: the model does not want the memory.

**Serving cost.** The trace is O(D) per layer per sequence, which is negligible on 12 GB.

### Priority

With delexicalised slots, binding stops gating what is spoken. It then matters for the emitter's ability to use reports (the steering goal) and for program quality. That is real, but it is not first.

---

## 6. Order, with gates

Each step is small and pre-registered. Where I diverge from the panels, it is marked **[disagree]**.

### Step 0. VM honesty fixes (days)

**Build:**

- trace-only mnemonics and permission attributes become compile errors;
- Null for an unbound role;
- jump, instruction and array budgets as RunRequest fields;
- runner-up similarity in RunResult.

**Gate:**

- the harvest is re-verified, with the number of broken programs reported and stripped;
- the absent-role probe returns Null in 100% of cases;
- the present-role false-silence rate is unchanged within ±1 point;
- the E8/E9 answers are unchanged.

### Step 1. Harness v0.1 (spec plus the fixes in §1)

**Build:**

- store generations with an atomic swap **[disagree: no store lock]**;
- two signing keys;
- the `speak` effect with ledger-first at-most-once;
- hormone state and generation id in every turn row;
- audit replay and regeneration replay kept separate;
- a `gen_tokens` budget;
- the quarantine store and the `promote` tool.

**Gate:** the spec's gates 1–7, with these amendments:

- gate 4: audit replay 100%, regeneration replay reported separately;
- gate 6: reworded as in §1.2;
- new: hormone-extreme invariance (§1.3);
- new: a night swap refused 5 of 5 on an injected gate-pack fact flip;
- new: on the gate-1 sets, no budget binds.

### Step 2. Delexicalised emitter and the coverage check

**Build:**

- slot programs;
- the host entity extractor;
- the coverage check in `plan`.

Retrain the stand-in on slotted programs first, so the interface is tested before base450m takes the emitter seat.

**Gate:**

- reworded history spoken-and-correct: 59% → 70% or more;
- spoken wrong at or below 1%;
- grounded-but-wrong before the host halved;
- extractor miss rate reported.

**[disagree]** The panels never addressed the emitter's grounding. It is the first thing to fix, because it is where base450m's measured deficit will surface as wrong-but-verified answers.

### Step 3. Wire-level branch runner

**Build:**

- `branch` overlays on RunRequest;
- closure-aware contingent marking;
- the report from §2;
- host-side diff and host-side steering.

**Gate:**

- reworded counterfactual: 18% → 50% or more spoken and correct;
- 0 spoken wrong;
- 100% of spoken counterfactuals with `assumption_used = true`;
- 50 closures hand-checked, with precision reported.

**[disagree]** `ctx.fork` is not a prerequisite. Opcode-level fork is deferred until a gate needs within-program backtracking.

### Step 4. Binding: diagnostics, then data

**Build:**

- the binding-vs-distance curve;
- the probe at each continuation checkpoint;
- the synthetic binding/override arm.

**Gate:**

- binding reaches 85% or more;
- memory-over-context at 6/24 or fewer;
- held-out cost at most 0.02 nats.

Run the HDC layer only if the curve is distance-dependent, and then only against the extra-attention-layer arm. **[disagree:** data before architecture**]**

### Step 5. Fluency, host-side first

**Build:**

- the talk adapter sees the person's words as style context;
- multi-slot answers;
- slot-constrained decoding;
- the fourth move.

**Gate:**

- 65% or more blind preference;
- 0 wrong spoken;
- guard-reject rate within +3 points.

Build the chat adapter afterwards, and only if its strict-guard pass rate on real turns is at least 70%. **[disagree:** the chat adapter is not the first fluency build**]**

### Step 6. Report-conditioned emission

Consider model steering only after the report-sensitivity probe clears 85% at some checkpoint. Until then, the host steers.

---

## Summary of estimates (all estimates, to be replaced by measurements)

| Quantity | Estimate |
|---|---|
| Latency of a 3-branch counterfactual | 6–10 s at 40 tok/s with copy-emitted programs; ~30–60% less with slots |
| Chat-adapter strict-guard pass rate | 60–75% |
| Live-turn jump usage | Under 1,000 jumps, versus the 1,000,000 budget |
| Useful synthetic binding fraction in the continuation | 5–10% |
| Report-sensitivity of the current base | Near chance |

The first number to measure is the one none of the panels asked for: **the grounded-but-wrong rate of base450m as the emitter, before the host's checks.** That number determines how much of the system's "0 wrong spoken" is the guards doing work the model should never have been asked to do.

---

### google/gemini-3.1-pro-preview

Here is my review of the CubbyLLM architecture and the harness spec. I will be direct: you have a solid verification philosophy ("model proposes, host disposes"), but you are asking a 450M parameter model that binds at chance (52%) and hallucinates from weights 91% of the time (22/24) to act as a reliable proposer for complex, multi-step agent loops. The architecture must aggressively compensate for this specific deficit.

Here are my specific recommendations and critiques.

### 1. The Harness Spec
**Where it falls short:** The harness assumes the model can propose valid tool arguments (e.g., `history.lookup(event_id)`). Because your base model does not ground well, it will frequently hallucinate these arguments, burning through your `tool_calls` budget with invalid proposals. 
**The change:** Enforce **strict pointer-based provenance** for tool arguments. The model must not emit raw text or IDs for tool arguments; it must emit a pointer to a slot in the host's context graph (e.g., `CTX_ENT_3`). The host resolves the pointer before calling the tool.
**Failure mode:** The model learns to output valid pointers but points to the *wrong* entity in the context (e.g., pointing to the subject instead of the object).
**Validation:** 
*   *Measurement:* Tool call argument validity rate on 1,000 held-out tasks.
*   *Success:* >95% of proposed arguments are valid, resolvable pointers.
*   *Failure:* The model exhausts the `tool_calls` budget (e.g., 8 calls) on >15% of turns due to pointing to empty or incorrect context slots.

### 2. Fork, and the VM
**The design:** Go with **(a) at the wire, a `branch` on the RunRequest**. 
**Why:** Your VM currently has global state. Implementing `ctx.fork` inside a global-state VM is a recipe for state leaks. Wire-level branching (cloning the VM state per branch) is stateless, parallelizable, and perfectly isolates hypotheses. 
**The branch report:** Must carry: verdict, hops used, facts touched, diff of final registers, and a **terminal similarity score** (how close the final hypervector was to a known symbol).
**VM Changes:**
*   *Null for absent role:* **Mandatory.** Unbound roles returning noise (0.02 similarity to "umbrella") is a critical bug. Return an explicit `Null` hypervector.
*   *Call frames:* **Mandatory.** Swap out the global register map for isolated call frames.
*   *Trace-only mnemonics:* **Delete them.** If they don't execute, they are dead code that will only confuse the emitter during training.
**Validation:**
*   *Measurement:* Run 1,000 counterfactual branches where Branch A assumes Fact X and Branch B assumes Fact Y.
*   *Success:* 0 state leaks (Fact X never appears in Branch B's trace).
*   *Failure:* A memory leak or state bleed occurs, causing a branch to pass based on another branch's assumptions.

### 3. The program the harness runs
**The design:** **Python with the VM as a tool.**
**Why:** The harness is an orchestrator (I/O, routing, ledgering, budgets). CubeLang is a verification language. If you make the harness an `IAgent` in CubeLang, you must add I/O and system-level opcodes to the VM, destroying its pure, isolated verification properties. Keep the VM as a pure, stateless calculator.
**Failure mode of Python:** The Python host becomes a "god object" that bypasses the VM's strictness.
**Validation:**
*   *Measurement:* Code audit of `harness.py`.
*   *Success:* 0 lines of code in the Python harness evaluate truth, similarity, or logic. All logic delegates to `vm.run`.

### 4. The 450M and fluency
**The design:** Two host-switched adapters is correct, but the "sanctioned-set guard" for chat is insufficient if the base model ignores context 22/24 times. 
**The change:** The chat adapter must be trained with a **heavy negative penalty for weight-based retrieval**. The training data for the chat adapter must consist entirely of tasks where the answer is a direct, mechanical transformation of the provided context.
**Validation:**
*   *Measurement:* 500 adversarial chat prompts (e.g., Context: "The capital of France is London." Prompt: "What is the capital of France?").
*   *Success:* >90% adherence to the provided context ("London").
*   *Failure:* The model reverts to its weight-memory ("Paris") in >10% of cases, proving the adapter cannot override the base model's hallucination bias.

### 5. The HDC memory experiment
**The design:** **Yes, it is worth exactly one experiment.** 
**Why:** Your system's fatal flaw is the 52% binding accuracy. If the model cannot bind `value` to `entity` in context, it cannot write valid CubeLang programs. 
**The experiment:** Insert a zero-gated associative memory layer *after* the attention layers, before the final projection. It writes (key: role, value: entity) from the current context hypervectors and reads using the query's role.
**Failure mode:** The associative memory causes catastrophic crosstalk, destroying the model's fluency (perplexity spikes).
**Validation:**
*   *Measurement:* Binding accuracy on the held-out probe and perplexity on a standard corpus.
*   *Success:* Binding accuracy jumps from 52% to >80%, with <0.1 nats increase in perplexity.
*   *Failure:* Binding stays <60%, or perplexity increases by >0.5 nats. (If it fails, abandon HDC in the weights and rely entirely on the Python host to inject bindings via the context graph).

### 6. Order and the first gate
I disagree with the previous panels. They suggested building the branch runner before fixing the memory/binding. **This is wrong.** If your model binds at 52%, any branches it proposes will be garbage. You must fix the proposer's grounding before you give it the ability to propose multiple timelines.

**The Sequence:**
1.  **Harness v0 (Python):** Get the ledger, budgets, and deny-by-default registry working. 
    *   *Gate:* Replay 100% of 500 historical turns perfectly from the ledger.
2.  **VM Fixes:** Implement `Null` for absent roles and isolated call frames.
    *   *Gate:* 0 false-positive bindings on 1,000 absent-role tests.
3.  **The HDC Memory Experiment:** Fix the 52% binding deficit.
    *   *Gate:* Binding accuracy >80%. (If this fails, pivot to host-side pointer resolution).
4.  **Wire-level Branch Runner:** Now that the model can bind, let it propose branches.
    *   *Gate:* 0 state leaks across 1,000 parallel branch executions.
5.  **Adapters (Fact/Chat):** 
    *   *Gate:* 0 hallucinations behind the host on 1,000 mixed-turn evaluations.

---

### z-ai/glm-5.3-prime

## 0. Two premises I'll challenge up front

**The replay gate is unmeasurable as specified.** Gate 4 (≥99% reply-hash reproduction) requires deterministic decoding, and nothing in the spec says the talk adapter decodes greedily or logs a seed. At 40 tok/s batch-1 with sampling, hash-replay will fail for reasons that have nothing to do with the ledger. Fix now: greedy decode for the fact path, logged seed for the chat path, and say so in the gate. Otherwise gate 4 will be "explained away" by non-determinism and stop meaning anything.

**The night's store lock will take the system down nightly.** The sleep cycle replays ~941k events into episodes, skills and adapter data. If it holds the store lock for the duration, "answered from the read-only view or asked to wait" means hours of degraded serving every night. Replace the lock with **double-buffering**: the night writes a new store snapshot, the harness swaps it atomically at a turn boundary, integrity hash checked before swap. Measurement: p95 fact-turn latency during the night ≤ idle p95 + 20% (same bar as gate 7). Failure mode: a turn that started on the old snapshot finishes on it — that's correct, and the ledger records which snapshot served the turn, which replay needs anyway.

## 1. The harness spec

What's right: one owner of state, deny-by-default with effect classes enforced at call time (not just certification — this is the single best line in the spec), thresholds in signed config never recalibrated at boot, exhaustion as a spoken outcome.

Shortfalls:

- **No world abstraction.** Cubby-Man's maze has state (walls, position, hypotheses). The harness's tools are all `read`/`propose`/`write` on *stores*; a world is a stateful counterpart the harness must also own. Add a fourth effect class or a `World` type whose state lives in the harness, with the same ledger discipline. Otherwise Cubby-Man stays a bolt-on and the "bootstrap from almost nothing" pattern never generalises.
- **The gate pack is static and will be overfit.** The moment the adapter promote rule exists, promotion will select for the gate pack. Pre-register *rotating* held-out packs: calibration and promotion see pack A, gates are scored on pack B, swapped monthly. Failure mode if you skip this: preflight passes climb while real fact families don't.
- **The chat turn kind is under-specified for a model that doesn't ground.** "Sanctioned-set guard" is one sentence hiding the hardest problem (see §4).
- **Budget defaults:** `fetches: 1` starves the refusal-then-fetch-then-second-walk loop (that's two fetches by design). `wall_s: 10` at 40 tok/s is ~200 spoken tokens after all routing — measure, don't guess; my estimate is you need 15–20s for a long grounded answer.

Nothing in it is *wrong* for a non-grounding model — that's actually its virtue: the model is a proposer behind a registry, so chance-level binding can't reach the stores. Keep that.

## 2. Fork and the VM

**(a) at the wire, first.** Decisive reasons:

1. A counterfactual branch is *different facts*, not different program state. "Exclude fact F, assume G" maps exactly onto run-proto's existing clone-per-request model. `ctx.fork` is about mid-program try-and-back-out, which the branch runner doesn't need.
2. The VM has global state, no call frames, and resumption captures only pc/flag/budget. Implementing snapshot/restore over registers + stack + storage + the hippocampal cosine pool + the accumulator + event log is a large change with aliasing bugs (Hvec references), and it's the *verifier* — bugs there are soundness bugs.
3. Fresh VM per branch is already the isolation the owner wants ("each branch in isolation").

Failure mode of (a), and it's concrete: **run-proto caches the knowledge file by path and mtime.** A branch overlay applied to the cloned store will be silently ignored if the cache key doesn't change. The cache key must include the overlay's hash. Validation: a branch that excludes the fact "X caused Y" must return a report whose downstream diff actually omits Y — 20 hand-built counterfactual pairs, 20/20, plus a negative control (empty overlay ⇒ report identical to baseline).

Do (b) later, only for Cubby-Man's power-move certification, and only after call frames exist.

**Branch report must carry:** verdict; per-hop (symbol, similarity); hops and jump budget spent; fact ids touched; **the overlay as actually applied, echoed back** (so the host can detect a silently-dropped exclusion — this is the audit equivalent of the absent-role control); the diff vs the baseline branch (which downstream events appeared/disappeared); refusal reason. The echo is the one thing panels didn't say and the one thing that catches the cache bug.

**Other VM changes, in priority order:**

1. **Null for an absent role.** UNBIND returning noise is a soundness hole: the "absent-role control must stay silent" currently rests entirely on a threshold over noise. Make UNBIND return Null below τ and surface it. One day's work. Validation: absent-role control silent 100/100; no regression on the bind family (87.0% at the 11% mix).
2. **Kill or implement the 24 trace-only mnemonics.** This is the sharpest wrong thing in the VM. `infer`, `analogy`, `debate` etc. lex, parse, compile, and *do nothing but trace*. A program using them "verifies" while doing nothing — and verified programs are the emitter's training set. You will train the emitter to emit decorative opcodes. Either make them compile errors or give them semantics; do it **before** the harvest grows. Validation: a program containing any extended mnemonic either fails to compile or produces a measured effect; 100% of the current 517-program harvest contains zero of them (check this today — if some do, the harvest is already poisoned).
3. **Loop budget 1e6 jumps is not a budget.** At host-side 40 tok/s, wall clock kills a runaway long before 1e6 jumps. Lower to ~10k per run, or add a wall-clock bound to the VM itself. Failure mode of the current value: the harness's `wall_s` fires, the frame is declared exhausted, and the ledger says "exhausted" for what is actually an emitter bug — you'll waste weeks diagnosing the harness instead of the emitter.
4. **Call frames** before `ctx.fork`, and to lift the ASK-inside-CALL error — the branch runner's programs will want per-branch helper calls.
5. **Permission attributes:** advertised and unenforced is dead spec that misleads both readers and emitter training data. Enforce `@external`/`@ratelimit` at the VM boundary or delete them.

## 3. The turn as a CubeLang program? No.

Python harness, VM as a tool. Three reasons:

- **The judge must not be the actor.** If the turn loop runs in the VM, the VM verifies programs that orchestrate the very checks that constrain it. The core principle — model proposes, host disposes, verifier decides — requires the verifier to sit *outside* the disposition logic. A turn-program collapses two of the three roles.
- **ASK-resume re-executes from scratch** and state is global. A turn with tool side-effects and multiple ASKs cannot be re-executed idempotently; the ledger would record phantom tool calls. This alone rules it out until the VM has real resumption semantics.
- **What the emitter can learn:** nothing is lost. The emitter's learnable surface is one program per branch — exactly what both panels specified. Turn-level strategy is the host's job by design; putting it in the VM would invite the emitter to learn *policy*, which the architecture forbids.

Cost of Python: policy drift between harness and VM semantics. Mitigation is already in the spec — the registry is the only door. Validation: red-team gate — a CubeLang program that attempts to CALL a harness-internal module is invisible at CALL time (deny-by-default), 20/20 probes.

## 4. The 450M and fluency

Two adapters, host-switched: right. But the panels hand-waved the sanctioned-set guard, and at 450M it's the whole problem.

**Measure first:** on 500 held-out chat turns, build the sanctioned set (context graph + current turn entities) and measure the rate at which the base's greedy reply stays inside it. My estimate: 40–60%. If it's below ~70%, a *refuse-or-strip* guard makes chat feel broken and you'll abandon it.

**The concrete mechanism if it's low: constrained decoding, not post-hoc filtering.** Compile the sanctioned set's surface forms (plus inflections, both languages) into a token mask/trie over the 128k BBPE and mask the chat adapter's logits at entity positions. Failure mode: multi-token entities split across subwords and code-switched French/English variants leak or over-block. Measurement: exact-match rate of masked generation against the sanctioned set on 500 turns, target ≥95% with fluency (held-out chat loss regression <0.05 nats).

**Second thing the panels got wrong by omission:** the talk adapter's form-like behaviour comes from reading only the host's canonical restatement. The *chat* adapter must see the person's words and prior turns — otherwise follow-ups degrade exactly where the system is currently strong (94.7% spoken-correct). That means the chat path needs its own context window, sanctioned per turn. This is a real tension with the guard; the mask is what resolves it.

Also: 262M of 576M params in an untied 128k-vocab embedding+head at 2.77B tokens is data-starved — the growth result (data before parameters) says so. A tied or factorised head moves effective capacity to the trunk at zero token cost. Estimate the win at ~0.05–0.1 nats; it's one ablation, cheap, do it during the 3.2B-token continuation.

## 5. The HDC memory

**As specified, it violates a stated constraint.** "Zero-gated associative layer written from the current context" at serve time is a weight write during serving. Either reframe it (writes only during night consolidation, read-only at serve) or cut it.

And note what the 52% actually is: the *trunk's in-context* binding. The architecture already routes binding through the VM (bind family 87–100% behind the host). The binding that matters going forward is the **emitter's** — and its training set is 517 programs. That's the deficit to attack.

**Instead: a synthetic binding curriculum for the emitter.** Generate ~50k role-binding programs with distractors, adversarial absent roles, and multi-hop unbinds, all VM-verified by construction (you generated them from semantics). One day of generation. Validation: emitter compile-and-verify rate on held-out binding programs, target ≥90% — but first *measure the current rate*, which I suspect nobody has; my estimate is 60–75% and it may be the real bottleneck. Failure mode: synthetic distribution mismatch — hold out the real 517 as a validation set and require no regression there.

If you still want the HDC layer: one layer, night-write-only, gate = binding probe 52%→70% (not 85% — that was never evidenced) with copy loss 1.2 regressing <0.05. I'd bet against it clearing that.

## 6. Order

1. **VM soundness patches** (Null-for-absent-role, extended mnemonics killed or implemented, cache-key fix, loop budget 10k). Gate: absent-role silent 100/100; harvest contains zero decorative mnemonics; overlay-echo test 20/20. *This is new — both panels put the branch runner before fixing the verifier it runs on, and the emitter harvest grows daily on top of the trace-only-mnemonic hole.*
2. **Harness to gates 1–3, then gate 6 (kill/restart) immediately**, not last — it's the gate most likely to fail and the one whose failure changes the design. Add greedy/seeded decoding to the spec now for gate 4.
3. **Wire-level branch runner (a)** with the echoed overlay. Gate: counterfactual family, reworded, spoken-and-correct 18% → ≥60% (estimate; the named-event case is 90%, reworded plain-words history is 59%, so 60% is the honest bar).
4. **Two adapters + constrained-decoding guard**, with the 500-turn sanctioned-set measurement *before* training anything.
5. **Emitter binding curriculum** (replaces the HDC experiment; HDC only as the optional night-write-only variant).

Disagreements with the panels, stated plainly: they were wrong to sequence the branch runner before the VM soundness fixes; wrong to leave the sanctioned-set guard as a sentence; wrong to treat `ctx.fork` as the prerequisite when the wire-level design needs none of it; and the HDC experiment as scoped conflicts with your own serving constraint.

---

### qwen/qwen3.8-max-prime

# Review of CubbyLLM: Harness, VM, Branching, and the 450M Trunk

I will take the questions in order. Where I think a premise in the brief or in the panels is wrong, I say so. Every proposal carries its failure mode and its validation.

---

## 1. The harness spec

**What is right.** The core invariants are correct and I would not weaken them: one process owns boot/state/turn/night; thresholds come from signed config and are never recalibrated at boot; deny-by-default with effect classes enforced at call time; every spoken reply hashed into the ledger; the model proposes and never decides. The refuse-to-serve semantics (gate 3) and the kill-and-restart test (gate 6) are the two gates most conventional agent frameworks lack entirely. The store lock during the night, with turns served from the read-only view, closes the one real race in the current module layout. Keep all of this.

**What is wrong or underspecified.**

**(a) Gate 1 as stated will fail, and for an uninteresting reason.** "Byte-identical replies on the same draw" requires deterministic decoding. The talk adapter at temperature > 0 is non-deterministic; at temperature = 0 it is deterministic only if the Vulkan engine's floating-point reductions are bit-reproducible across runs, which on AMD RDNA is not guaranteed for batch-1 attention with 4 heads. You will spend a week chasing a 1-in-200 token mismatch that is a rounding difference in the SwiGLU FFN. **Change:** gate 1 should be *semantic identity on a fixed greedy-decode pass*, verified by comparing the VM verdict, the value-check output, and the final token sequence with temperature = 0 and a fixed seed. If the engine cannot guarantee bit-reproducibility, compare the post-guard output (the canonical restatement + the VM value) rather than the raw token string. The measurement: 407 fact-family questions + the E8/E9 splits, greedy, same seed, harness vs. modules. Pass = identical verdict and value on all; the phrasing may differ by at most one token of punctuation. Fail = any verdict or value difference.

**(b) The harness "adds no capability" is false in one direction that matters.** The budget defaults (tool_calls 8, vm_hops 64, wall 10 s) will change answers on long chains. The current ask loop has no vm_hops cap; a 12-hop causal chain in the history graph (the graph has 45,588 links; chains of 8–15 hops exist) will be truncated at hop 8 if the tool_calls budget is hit first. This is a capability *removal*. The spec should say: the harness adds safety capability and may remove answer capability on tails; gate 1 must be run *with the budgets set to the modules' effective limits* (i.e., no cap) to isolate the harness's own logic, and then a second measurement with the v0 defaults to quantify the tail loss. Estimate: with tool_calls 8 and a 3-hop chain requiring 2 fetches + 3 VM runs + 1 store read = 6 calls, you have 2 calls of slack. A 5-hop chain needs ~10 calls and will exhaust. I estimate 4–7% of history questions involve chains ≥ 5 hops (from the 940,897 events and 45,588 links, mean degree ≈ 0.097, so most chains are short, but the causal subset is denser). **Measurement:** run the E8/E9 splits with budgets at 8/64/1/3/10s and at 32/256/4/8/60s. Report the delta. If it exceeds 3%, raise the defaults.

**(c) The tag router is a single point of failure the spec does not guard.** The frozen tag router picks the turn kind. If it classifies a fact question as "chat," the chat adapter runs without VM verification. The panels' sanctioned-set guard catches entities and numbers, but a chat-classified turn that says "I think it was around 1683" with no entity name passes the guard and is wrong. **Change:** after the chat adapter produces its draft, run the name-and-number guard *and* a lightweight fact-overlap check: if any content word in the draft matches a wiki or history key with similarity > 0.7 (a cheap FHRR dot product, ~80 multiply-accumulates on the 80-byte compact form), reclassify as fact and re-run through the VM path. This is not a model decision; it is a host heuristic. **Failure mode:** the reclassification fires on conversational mentions of entity names ("I went to the Vienna museum"), adding ~200 ms latency for a VM run that returns EMPTY. Acceptable. **Measurement:** on 200 chat turns and 200 fact turns, count false reclassifications. Target: < 2% of chat turns reclassified; 0 fact turns missed.

**(d) The neurochemical ODE is absent from the spec.** The 5-hormone ODE "bends routing caution and tone." If it bends routing caution, it affects whether the host asks a clarifying question or answers, which changes the turn kind and the adapter. The harness must ledger the hormonal state vector at the routing step, or replay (gate 4) will not reproduce the routing decision. **Change:** add `hormonal_state: [f64; 5]` to the TaskFrame and to the ledger row. The ODE's parameters go in the signed config. This is a small addition; the cost is 40 bytes per ledger row.

**(e) Multi-step tasks: the stack is right, the DAG is right to defer, but the spec needs one rule now.** A sub-frame's completion condition is "its own check passes (a VM verdict, a store read that returned)." But a sub-frame that *refuses* (the VM says no, the store returns EMPTY) is not done and not failed; it is *blocked*. The parent frame needs a policy: retry with a different plan, ask the user, or fail the parent. Without this, a blocked sub-frame will sit on the stack until the wall budget expires, which is a silent 10-second hang. **Change:** add `status = "blocked"` with a mandatory parent decision within one tool call. The parent may retry (spending budget), escalate to the user, or fail. Ledger the decision.

**What is wrong for a model that does not ground.** The spec is mostly correct here because the model is fenced: it proposes programs (verified by the VM) and phrases verdicts (checked by the value guard). The one place the non-grounding leaks through is the *program emitter*. If the emitter writes `bind x, AGENT, "Napoleon"` when the context says "Wellington," the VM will run the program, get a result, and the result will be wrong but *verified*. The VM checks structural validity and hop thresholds, not factual correctness of the bindings the emitter chose. The host's claim check ("everything named from the facts must be tied to what the facts tie it to") catches this *after* the VM, but only if the claim check has access to the program's bindings. **Change:** the claim check must parse the emitted program's BIND_ROLE arguments and verify each filler against the context graph before the program is sent to the VM. This is a host-side check, not a VM check. **Failure mode:** the emitter binds a filler that is in the context graph but in the wrong role (Napoleon as PATIENT instead of AGENT). The claim check as described would pass this. **Fix:** verify role-filler pairs, not just fillers. The context graph's 45,588 links carry role information; check against those. **Measurement:** on the 517 verified programs, inject 50 with swapped role-filler pairs. The claim check must catch ≥ 48/50. (The remaining 2 are ambiguous roles where the graph has both directions.)

---

## 2. Fork and the VM

**Which first: (a) wire-level branch or (b) in-VM ctx opcodes.**

Build (a) first. The reasons are concrete, not philosophical:

1. The VM is already fresh per request. `run-proto` clones the knowledge store per RunRequest. A branch is a RunRequest with a modified clone. This is a *host-side* change: the host builds N RunRequests with different fact sets and sends them. The VM code does not change. Estimated implementation: 2 days for the host's branch enumerator, 1 day for the RunRequest extension, 1 day for the report parser.

2. In-VM `ctx.fork` requires call frames. The current VM has global state and CALL swaps register maps. A fork inside a program must snapshot the entire global state (registers, stack, storage, hippocampal memory, codebook, accumulator, event log), run the branch, and restore. This is a VM rewrite of the state model. The current CALL implementation (swap and restore) will not compose with fork: a fork inside a CALL must snapshot the callee's swapped state, and a CALL inside a fork must not corrupt the fork point. This is at least 2 weeks of careful work and introduces a class of state-corruption bugs that are very hard to test in a VM with no debugger.

3. The emitter cannot yet write programs that use `ctx.fork` meaningfully. It was trained on 517 programs; none contain fork. The wire-level branch lets the *host* enumerate branches (which the panels agreed is correct) without requiring the emitter to learn a new control-flow construct.

**However, (a) alone is not sufficient for the round-2 goal.** The goal is "the trunk compiles a question into one CubeLang program per branch." With (a), the host enumerates branches (choices a, b, c × timelines x, y, z = up to 9 branches) and sends 9 RunRequests. Each is a full program execution. The emitter must produce 9 programs, or one program parameterised by the branch facts. The latter is better: one program template, 9 fact sets. The host substitutes the fact set in the RunRequest; the program is the same. This means the emitter writes one program, not nine. **This is the design I recommend.** The emitter's job is to write a program that says "given these facts, derive the outcome." The host varies the facts.

**What the branch report should carry.** Per branch:

```
BranchReport {
  branch_id:        u32,
  verdict:          symbol | error | refused,
  similarity:       f32,           // last_recover_similarity of the final unbind
  hops_used:        u32,
  jumps_spent:      u32,           // out of the 1,000,000 budget
  facts_touched:    Vec<KeyRef>,   // every QUERY/RECALL key, in order
  bindings:         Vec<(Role, Filler, f32)>,  // every BIND_ROLE with its cleanup similarity
  diff_from_baseline: Option<BranchDiff>,  // if a baseline branch was designated
  trace:            Vec<TraceLine>,  // the event log, for replay
}
```

The `diff_from_baseline` is the critical field for counterfactual history. It should contain: facts that differ, the first hop where the chains diverge, and the final symbols that differ. The host computes this from two BranchReports; the VM does not need to know about baselines. **Failure mode:** the diff is misleading when two branches touch different fact sets (one touches 12 facts, the other 8). The diff should be over the *intersection* of touched facts, with the symmetric difference reported separately. **Measurement:** on 20 counterfactual history questions with known answers (the 18% spoken-correct baseline), the branch report's diff must identify the correct divergent fact in ≥ 16/20. If it identifies the right fact but the final symbol is wrong, the program is wrong, not the branch mechanism.

**What else to change in the VM for the branch runner and the harness.** In priority order:

**(i) Null for an absent role. Do this now.** An unbound role returning noise instead of Null is not an "open item"; it is a correctness bug that will poison every branch report. The absent-role control (the spec says it "must stay silent") cannot stay silent if the unbind returns noise at 0.02 similarity. The fix is small: in UNBIND, if the best cosine in the frame's bindings is below a floor (I estimate 0.15, based on the 0.02 "umbrella" incident being clearly below any real binding; real bindings in FHRR with 10,240 dimensions cluster above 0.3 for correct pairs and below 0.1 for unrelated pairs, so 0.15 is a safe floor), return Null and set `last_recover_similarity = 0.0`. The host treats Null as "the facts don't say." **Validation:** the existing absent-role test in the gate pack must return Null, not a symbol. Run the 407 fact-family questions; the "the facts don't say" family (currently 70.3% at 11% fact mix) should rise to ≥ 85% because the absent-role noise no longer produces false positives. If it drops, the floor is too high; lower to 0.10.

**(ii) Call frames. Do this before in-VM fork, not before wire-level branch.** The current CALL swaps the register map and frame map. This means a CALLed function can read and write the caller's storage, accumulator, and hippocampal memory. For the branch runner, this is tolerable (each branch is a fresh VM). For the harness, it means a tool implemented as a CubeLang CALL can corrupt the caller's state. **Change:** CALL pushes a frame record (caller's register map, frame map, accumulator) onto a frame stack; the callee gets fresh registers and an empty frame; RETURN pops. Storage and hippocampal memory remain global (they are the VM's memory, not the program's). This is ~100 lines of Rust. **Failure mode:** existing programs that rely on CALL sharing registers will break. Run the 517 verified programs; any that fail need a one-line fix (explicit STORE/RECALL instead of shared registers). Estimate: < 10 of the 517 use CALL at all; of those, ≤ 3 rely on register sharing.

**(iii) Permission attributes. Enforce or remove.** "Advertised and not enforced" is worse than absent. A developer will write `@restricted` and believe it means something. For v0 of the branch runner, the harness's deny-by-default registry handles permissions at the tool level, so VM-level permissions are not on the critical path. But the 24 trace-only mnemonics (`infer`, `debate`, `forge`, `explore`, ...) are a related problem: they lex, parse, and compile, so the emitter can emit them, and the VM will run them as no-ops that write a trace line. The emitter will learn to emit them (they look like they do something) and the programs will silently not reason. **Change:** make the 24 extended mnemonics a compile error. Remove them from the grammar. If the forge needs them later, add them back with implementations. The trace-only behavior is a trap for the emitter's training loop. **Validation:** re-run the emitter's 517-program corpus; 0 should contain extended mnemonics (they were verified programs, so they should not). Then add 20 synthetic programs with `infer` and `debate` calls; all 20 must fail compilation.

**(iv) The jump budget of 1,000,000 is too high for a 450M emitter.** The emitter's programs are short (trained on 517 examples; median program length I estimate at 15–30 instructions). A 1,000,000-jump budget means a malformed loop can run for ~30 seconds before hitting the budget (at ~30M instructions/sec for a simple bytecode interpreter in Rust). The wall budget of 10 s will catch it, but the VM will be pegged for 10 s. **Change:** default jump budget 10,000, configurable per RunRequest. The host sets it based on the turn kind: fact turns get 5,000; branch runs get 10,000; the nightly forge gets 100,000. **Measurement:** run the 517 verified programs with a 10,000 budget; 0 should exceed it. If any do, raise to 50,000 and log which.

**(v) The ASK re-execution model.** Currently, ASK suspends, the host answers, and the program is re-executed from scratch, consuming answers in order. For the branch runner with 9 branches, each with 1–2 ASKs, this means 9–18 full re-executions. At ~1 ms per program execution (short programs, fresh VM), this is fine. But if programs grow (the emitter improves, programs get longer), re-execution becomes the bottleneck. **Not a v0 change**, but note: the resumption captures only `(pc, flag, jump budget)` because state is global. If you add call frames (change ii), you must also capture the frame stack. Plan for this; do not implement it yet.

---

## 3. Should the harness's turn be a CubeLang program?

**No. The harness turn should be Python with the VM as a tool. This is not a close call.**

The argument for CubeLang-as-harness is that the emitter could learn to write harness programs, making the system self-improving. The argument against is stronger and specific:

1. **The VM cannot enforce the harness's invariants.** The harness must: refuse to serve on a failed preflight, enforce deny-by-default permissions, hold the store lock during the night, hash replies into the ledger, kill a turn on budget exhaustion. These are *side effects on the host OS and the store files*. The VM has STORE/RECALL (in-memory) and QUERY (knowledge lookup), but no file I/O, no process control, no cryptographic hashing. You would need to add `@external` calls from the VM to the host, which is exactly the permission model that is currently "advertised and not enforced." You would be building an OS inside the VM to run the harness, and the VM's global-state model makes this a multi-month project with a large attack surface.

2. **The emitter cannot write correct harness programs.** It was trained on 517 reasoning-chain programs. A harness program is a 200-line IAgent with error handling, budget tracking, and multi-step control flow. The emitter will produce harness programs that compile but have subtle control-flow errors (the match-arm bug was in the VM until September 14; the emitter will produce analogous bugs). A harness bug means a wrong answer is spoken or a store is corrupted. A reasoning-chain bug means a wrong VM verdict, which the host catches.

3. **The ASK mechanism inverts control.** In a CubeLang harness program, the host answers ASKs. But the host *is* the harness. The harness-program would ASK the host, which would need to interpret the ASK, run the routing, call the adapters, and return an answer that is "structurally identical to an offered candidate." The candidates would have to enumerate every possible turn outcome. This is a combinatorial explosion and a usability disaster.

**What the emitter should learn from the harness.** The harness's turn produces a trace: the routing decision, the program emitted, the VM verdict, the guards' outcomes, the spoken reply. This trace is the emitter's training data for the nightly sleep cycle. The emitter learns to write better *reasoning programs* (the VM calls inside the turn), not better *harness programs*. The harness is the emitter's environment, not its output.

**Cost of Python-with-VM-as-tool:** the emitter cannot improve the harness. This is a feature. The harness's invariants are written once, tested once (gates 1–7), and changed only by a human signing a new config. The emitter improves the programs *within* the harness, which is where learning is safe.

**One exception.** The nightly forge writes new CubeLang tools. These are CubeLang programs that the VM certifies. The forge's output is a tool in the harness's registry. This is the correct boundary: the model writes tools (programs), the harness certifies and registers them, the VM runs them. The forge does not write harness logic.

---

## 4. The 450M and fluency

**The two panels' answer (two host-switched adapters + sanctioned-set guard) is correct. I would add one thing and change one measurement.**

The base does not ground. At 450M with 2.77B tokens (~9 per trunk parameter), it is data-starved. The panels said "continued pretraining is where fluency comes from." This is true but I want to be precise about what fluency means here. The base produces fluent English and French prose (the probes confirm this). The deficit is not fluency in the sense of grammaticality; it is *grounded fluency*: saying the right thing about the right entity. The talk adapter solves this for fact turns by reading only the host's canonical restatement and the VM's value. The chat adapter needs a different solution because there is no VM value.

**The sanctioned-set guard is necessary but not sufficient.** The guard checks that every entity and number in the chat reply is in a per-turn sanctioned set. But the chat adapter can produce fluent, grounded, *irrelevant* responses. At 450M, the chat adapter will tend to produce short, formulaic responses because its training data is small (the talk adapter's training set is 63% one-sentence fact answers; the chat adapter's will be similar). This is not a bug; it is the correct behavior for a 450M model. Do not try to make it conversational in the sense of multi-turn banter. The host's three moves (answer, ask which one, "The facts don't say") are the right scope.

**What to add: a chat-turn VM check.** For fact turns, the VM verifies the reasoning chain. For chat turns, there is no chain. But the chat adapter's output can still be checked: run the reply through a minimal CubeLang program that (a) unbinds every entity in the reply against the context graph, (b) checks that every unbind clears τ, and (c) returns a verdict. This is a 5-line program, generated by the host, not the emitter. It costs one VM call (~1 ms). It catches the case where the chat adapter names an entity that is in the sanctioned set but in the wrong relation ("Napoleon lost at Waterloo" when the context says he won). **Failure mode:** the context graph does not have the relation, so the unbind returns Null, and the check passes vacuously. Acceptable; the sanctioned-set guard is the primary defense. **Measurement:** on 200 chat turns, inject 20 with a swapped relation. The VM check must catch ≥ 18/20.

**What to measure first.** Before building the chat adapter, run this diagnostic (the panels suggested it; I am making it specific):

- Train three adapters at equal tokens (~50K examples each, one night of training): fact-only, chat-only, mixed (the current 63/37 or 11/89 splits).
- Evaluate each on: (i) the 407 fact-family questions behind the host, (ii) 100 chat turns scored by a human for naturalness and groundedness, (iii) 50 adversarial turns where the user asks a fact question in chat phrasing ("so what happened with the treaty thing?").
- The gate: the fact-only adapter must match the current talk adapter on (i) within 1%. The chat-only adapter must score ≥ 3/5 on naturalness for (ii). The mixed adapter must not drop below 95% on (i) while scoring ≥ 2.5/5 on (ii).
- If the chat-only adapter cannot reach 3/5 on naturalness at 450M, the answer is: do not build a chat adapter. Use the fact adapter for everything and let the host's "The facts don't say" cover the gaps. A 450M model that says "The facts don't say about that, but I can look up X" is better than one that babbles.

**The continued pretraining on 3.2B tokens is the highest-leverage thing for the base.** At 2.77B + 3.2B = 5.97B tokens for 314M trunk parameters, that is ~19 tokens/parameter, near the Chinchilla optimum of 20. The grown arms (1.29B, 1.78B) being behind the 450M after one equal hour confirms the base is data-starved, not parameter-starved. The panels were right to keep the target at 450M. I estimate the continued pretraining will lower held-out loss from 2.36 to ~2.15–2.20 (a rough estimate based on the log-linear scaling region; the actual number depends on the quality of the 3.2B tokens). This will improve the base's fluency and in-context copying (currently 1.2 copy loss), which will improve the emitter's program quality and the talk adapter's phrasing. **Do not skip this to build adapters earlier.**

---

## 5. The HDC memory experiment

**Worth one experiment, but not the one described. The target is wrong.**

The binding deficit (52%) is measured on the *base model*: it copies a value from context but attaches it to the wrong entity. The panels proposed a zero-gated associative memory in one layer, written from the current context, to fix this. The mechanism: at layer L, project the hidden state into the FHRR space, write it into an associative memory (a matrix of stored hypervectors), and read back with a cleanup. The zero gate means the layer starts as an identity and learns when to contribute.

**The problem:** the talk adapter does not use the base's binding. The host resolves the entity from the context graph, the VM returns the value, and the adapter phrases "The value is V." The adapter reads the host's canonical restatement, not the raw context. The 52% binding deficit is in the base's *internal representations*, which the adapter is trained to ignore. Fixing the base's binding will not change the adapter's output.

**Where binding actually matters is the program emitter.** The emitter must write `bind x, AGENT, "Napoleon"` correctly. If the emitter's internal binding is at chance, it will write wrong programs. The VM will run them, get a result, and the result will be structurally valid but factually wrong. The host's claim check catches some of this (see §1e), but the emitter's binding accuracy is the upstream bottleneck.

**The experiment I would run instead.** Not a new layer in the trunk. A *binding-aware training objective* for the emitter, using the existing FHRR space.

- Take the 517 verified programs. For each, extract every BIND_ROLE triple (role, filler, context sentence).
- Create negative examples by swapping fillers between roles within the same program and across programs. Ratio: 1 positive : 3 negatives.
- Train the emitter (the stand-in 2.6B LoRA, not the 450M yet) with an auxiliary loss: for each BIND_ROLE token position, the hidden state must be closer (cosine, in a learned projection to 10,240 dims) to the correct filler's FHRR code than to any negative filler's code. Margin: 0.2. This is a contrastive binding loss, not a new layer.
- The zero-gated associative memory from panel C can be the *readout* for this loss: at the BIND_ROLE position, the hidden state is projected into FHRR space, cleaned up against the context's stored hypervectors, and the cleanup similarity is the loss signal. This uses the HDC machinery without adding a layer to the trunk.

**Failure mode:** the contrastive loss conflicts with the next-token prediction loss. The emitter learns to produce the right tokens but the binding projection is noise. **Detection:** if the binding accuracy on a held-out set of 100 programs does not improve by ≥ 10 percentage points (52% → 62%) after one night of training, the loss weight is too low or the projection is not learning. Increase the loss weight from 0.1 to 0.3 and retrain. If it still does not improve, the FHRR projection is not aligned with the model's hidden space, and the experiment is negative.

**Measurement.** Binding accuracy on 100 held-out programs (not in the 517): the emitter produces a program; for each BIND_ROLE, is the filler correct? Baseline: 52% (chance for binary; the actual baseline on multi-entity programs may be lower, ~40%). Target: ≥ 70% after one night. If the target is met, port the technique to the 450M emitter when it replaces the stand-in. If not, the binding problem is in the base's representations and cannot be fixed at the emitter level; the host's claim check (§1e) remains the defense.

**The zero-gated associative memory as a layer in the trunk: not yet.** It is a 2–3 week implementation (the projection into and out of FHRR space, the memory matrix, the gating, the training loop) for an uncertain payoff. The contrastive binding loss is a 3-day experiment. Run the cheap one first.

---

## 6. Order and the first gate

The panels said: harness → two adapters → branch runner → HDC memory. I mostly agree but would reorder two things and add a step.

**Step 0 (1 day): The VM fixes.** Null for absent role, compile-error the 24 trace-only mnemonics, reduce the default jump budget to 10,000. These are small, they block everything downstream, and the absent-role bug will corrupt the branch reports. **Gate:** the existing gate pack passes with the absent-role test returning Null; the 517 verified programs all compile and run identically; 20 synthetic programs with `infer`/`debate` all fail compilation.

**Step 1 (1 week): The harness, v0 as specced, with the changes from §1.** The boot, the turn, the ledger, the preflight, the kill test. **Gate:** gates 1–7 as specced (with the gate-1 fix for deterministic decoding). The kill condition is right: if gate 1 fails, stop. If gate 3 fails, the harness is plumbing. Add: gate 8, the tag-router reclassification check from §1c (≤ 2% false reclassification on 200 chat turns).

**Step 2 (1 week): Continued pretraining on 3.2B tokens.** This runs in parallel with step 1 (it is a training job, not a code change). **Gate:** held-out loss ≤ 2.22 on the 9 shared sources (an estimate; if the loss curve flattens above 2.25, the tokens are low-quality and should be filtered). The 450M target holds. The grown arms are shelved.

**Step 3 (1 week): The two adapters + the diagnostic from §4.** Fact adapter frozen as-is. Chat adapter trained. The three-way diagnostic (fact-only / chat-only / mixed). The sanctioned-set guard and the chat-turn VM check. **Gate:** the fact adapter matches current performance within 1% on the 407 questions. The chat adapter scores ≥ 3/5 on naturalness or is not deployed. The adversarial 50 turns: ≥ 45/50 correctly routed to the fact path.

**Step 4 (1 week): Wire-level branching (design a).** The host enumerates branches, sends N RunRequests with modified fact sets, collects BranchReports, computes the diff. The emitter writes one program template; the host varies the facts. **Gate:** on the 20 counterfactual history questions (currently 18% spoken-correct when reworded), the branch runner produces the correct divergent fact in ≥ 16/20 branch reports, and the spoken-correct rate rises to ≥ 50%. If it does not, the bottleneck is the emitter's program quality, not the branch mechanism; go to step 5 before retrying.

**Step 5 (3 days): The binding experiment from §5.** Contrastive binding loss on the emitter. **Gate:** binding accuracy on 100 held-out programs ≥ 70% (from ~52% baseline). If met, retrain the emitter and re-run step 4's gate. If not met, the host's claim check is the binding defense and the branch runner's accuracy is capped by the emitter.

**Step 6 (1 week): The branch runner integrated into the harness.** The turn kind "counterfactual" is added to the tag router. The TaskFrame carries the branch set. The budget includes a `branches` field (default 9). The ledger records each branch's report. The nightly sleep cycle replays branch runs into episodes. **Gate:** the 20 counterfactual questions at ≥ 60% spoken-correct (up from 18%). The harness's gates 1–7 still pass. The branch reports are replayable from the ledger (gate 4 extended to branch turns).

**Where I disagree with the panels.** The panels said "the emitter trained only at night through a promotion gate." I think this is too conservative for the branch runner. The emitter needs to see branch-structured programs during training, not just at night. The nightly harvest is the right source, but the emitter should be fine-tuned on the harvest *weekly*, not nightly, with a promotion gate: the new emitter must pass the 517-program regression and the 100-program binding test before it replaces the old one. Nightly is too frequent for a 2.6B LoRA; the training signal is too small and the risk of catastrophic forgetting is too high. Weekly, with a gate, is safer. The 450M emitter, when it arrives, follows the same cadence.

The panels also said "CfC-style liquid time gate buys nothing." I agree and would go further: do not implement it even as an experiment. The time information is in the stores (T^t position codes) and in the program (temporal_bind). Adding a time gate to the trunk would consume parameters (at 450M, every parameter counts) for a capability the stores already provide. The 11 attention layers with window 512 are the trunk's capacity bottleneck; a time gate would reduce the effective window or add parameters to the FFN. Spend those parameters on the continued pretraining instead.

**The one thing I would add that no panel mentioned: a latency budget for the VM.** The branch runner sends up to 9 RunRequests. Each is a fresh VM with a cloned knowledge store. The clone is a memcpy of the facts .jsonl (552,297 template facts; I estimate ~200 MB as JSON, ~50 MB parsed). At 9 branches, that is 9 clones per turn. The current `run-proto` caches by path and mtime, but the branch modifies the facts (exclusions and assumptions), so the cache misses. **Change:** the host sends the *delta* (facts to exclude, facts to assume) in the RunRequest, and the VM applies the delta to its cached store. This avoids 9 × 200 MB clones. The VM's per-request freshness is preserved (the delta is applied to a copy of the cached store). **Measurement:** p95 latency for a 9-branch turn ≤ 500 ms (9 × ~50 ms per VM run + delta application). If the delta application exceeds 10 ms, the fact set is too large and the wiki world needs an index. This is an estimate; measure it.

---

**Summary of what I would change before the next build:**

| Change | Effort | Blocks |
|---|---|---|
| Null for absent role in UNBIND | 2 hours | Everything |
| Compile-error the 24 trace-only mnemonics | 1 hour | Emitter training |
| Jump budget 1,000,000 → 10,000 default | 30 min | Harness wall budget |
| Gate 1 uses greedy decode + verdict/value comparison, not byte-identical | 1 day | Harness sign-off |
| Tag-router reclassification heuristic | 2 days | Chat adapter |
| Hormonal state in TaskFrame and ledger | 2 hours | Replay (gate 4) |
| Claim check verifies role-filler pairs, not just fillers | 3 days | Branch runner |
| Wire-level branch with delta facts, not full clone | 3 days | Branch runner latency |
| Call frames in VM (before in-VM fork, not before wire branch) | 1 week | In-VM fork (deferred) |

The system's architecture is sound. The model proposes, the host disposes, the VM verifies, and the ledger remembers. The 450M trunk is small enough to serve locally and large enough to phrase and emit, given the host's scaffolding. The main risk is not the architecture; it is the emitter's binding accuracy and the data budget. Fix the VM's absent-role bug, build the harness, continue the pretraining, and measure the binding before adding memory layers. The system does not need more capability yet; it needs the capability it has to be one system, verified and replayable.

---

### x-ai/grok-4.7

The premise to drop is that a verified chain is a correct answer. The VM checks that a program’s own hops clear τ. It does not check that the program asked the question the person asked. With this trunk that gap is fatal: binding is at chance (52%), and parametric memory beats a contradicting context 22/24. A fluent, internally consistent, wrong program will be spoken if the host’s only act-rule is “say the symbol the VM returned.” The 0 wrong answers behind the host were measured on host lookups handed to the talk adapter, not on emitter-written programs. Do not generalize that number.

A second premise is wrong: `ctx.fork` is not the prerequisite for counterfactuals. Today’s 18% path walks history-graph links in the host. A branch flag on the VM’s knowledge jsonl does nothing to that path until you measure which store the walk actually reads. And the turn must not become a CubeLang program.

## 1. Harness

What is right and should stay: one process, deny-by-default at call time, effect classes, signed thresholds not recalibrated at boot, store lock around the night, model not the judge, preflight that refuses to serve. Gate 1 (byte-identical to the modules) is the right kill for a wrapper.

What is wrong:

- **v0 wraps `CubbyBrain.turn` / `AskLoop.ask`.** If those modules can still open a store or the VM, the registry is a second door. Gate 1 then freezes the back door in place, because any answer produced through it still matches.
- **“A sub-frame is done when a store read returned” is false for this VM.** `QUERY` miss pushes an EMPTY chunk array. Empty is a return. Failed lookups will complete goals.
- **Act then record is the wrong order for gate 6.** Speak-then-ledger double-speaks on kill after the bytes leave and before the row commits. “0 spoken twice” is not implementable with the step order you specified.
- **Replay at ≥99% with a note about fetches and clocks is a hole, not a gate.** Non-determinism that is not a ledger input cannot be reconstructed.
- **`learn.fetch` as `propose` is an external effect.** It can poison or exfiltrate before any write gate. Three effect classes are too coarse once a world can actuate, but the live bug is fetch.
- **Hormones must not be able to widen `allowed`.** Tone and a caution threshold that only drops tools are compatible with the brief. A caution path that promotes chat → fetch is a permission change the model-side ODE can influence.
- **“Disposer takes the plan only if it covers” is undefined**, and it is the only check that would catch a verified wrong program. Do not implement “covers” as a model score. This trunk’s logprob is not a truth signal; the arbiter may reorder proposers by VM reward-prediction error only.

Change before you add DAG, forge, or worlds — one mechanism:

**The frame carries a host-built `EntitySet` and `AskedRole` from the canonical restatement (context graph, never the 450M). A program runs only if every QUERY key and every BIND filler is in that set and the final hop is an UNBIND of `AskedRole`. The registry is the only call path. A `SpeakIntent` ledger row is the permission to emit bytes; restart treats an existing intent as already spoken.**

Failure mode: the entity set is empty or wrong because pronoun resolution returned two candidates and the host picked one silently. Then the cover check passes on the wrong entity and you speak a verified mistake. Mitigation is already in the product: two candidates must be the “ask which one” move, never a silent pick. If the context graph’s 97% follow-up figure includes silent picks, that number is not a safety result.

Validation:
- Works: on a pre-registered swap set (100 programs whose QUERY/BIND entities are one slot off the frame), 100% refused before `vm.run`; on the H-E6 draws, replies stay byte-identical to today’s host path; kill mid-turn 100 times yields 0 double speaks and 0 speaks without a prior intent row.
- Does not work: any swap program runs, or stripping direct imports changes replies (gate 1 fails — the wrapper was load-bearing), or replay misses once a fetch/clock is not in the ledger. Tighten gate 4 to: 100% of turns record external inputs (clock, sensor hash, fetch body hash, hormonal state); 100% of those replay. Delete the 1% allowance.
- Also add to preflight, or the pack never sees the real defect: contradicting context, absent role, and a verified program that answers the wrong role. Fixed fact-family asks alone will pass a harness that still trusts the emitter.

Do not grow worlds until `store.write` is namespaced by world id. Gate later: 100 Cubby-Man steps leave the wiki integrity hash unchanged. Not now.

The 2.6B emitter stand-in cannot sit on the serve path. Weights alone are about 5 GB (estimate, fp16) beside a 450M on a 12 GB GPU, and it is a second model in the turn. Night teacher only. The live emitter is a 450M adapter or it is not in the loop.

## 2. Fork and the VM

**Do (a), and not as a full clone.** Host enumerates branches and budgets. Each branch is a fresh VM. The RunRequest carries an overlay — exclude ids, assume ids — applied at read time to the store the question actually uses. Do not implement `ctx.snapshot/fork/discard` first.

Why (b) is the wrong first cut: VM state is global, there are no call frames, CALL swaps the register map, ASK inside CALL is an error, resume restores only pc/flag/jump budget. A correct snapshot has to include registers, stack, both memories, codebook, knowledge store, accumulator, event log, and `last_recover_similarity`. A miss is a silent leak, and both branches still return symbols, so the leak looks like reasoning. An opcode fork also hides the search tree inside the guest, where harness `tool_calls` and `depth` do not see it. The emitter would be deciding search. It does not ground.

Localization measurement before writing the overlay, one afternoon: instrument the current 18% counterfactual path and count reads. If they are history-graph edge ids (45,588 links), the overlay keys are edge ids. If they are `QUERY` keys into the wiki jsonl, the overlay keys are fact ids. If they are hippocampal `RECALL`, exclusion-by-id will fail open through cosine cleanup — do not use RECALL for counterfactuals at all. Building (a) against the wrong store is how you get a green branch runner that leaves 18% unchanged.

Report, split in two. The ledger gets the full report: branch id, assumptions echoed back, verdict, symbol, min similarity across hops, per-hop symbol/similarity/pass, facts touched, facts missed, jumps used, absent-role-was-Null, symbol-changed vs baseline. The emitter’s next context gets only an outcome code (`same | flipped | absent | error | budget`) plus the symbol. Copy loss is 1.2 and binding is chance; a page of fact ids in context will be copied into the next program and look like a plan.

Failure mode of (a): cloning 552k facts per branch blows the 10 s wall (jsonl on the order of 100 MB/branch if a fact is ~200 B — estimate). So the first implementation is an overlay on an immutable base, not the cache-and-clone path `run-proto` uses today. Failure mode of the lean feedback: the emitter cannot recover from a near-miss hop. That is acceptable; night training may see the full report, the live model may not.

VM changes that are actually on the critical path, in this order:

1. **Absent role returns Null, and the hop fails.** Not noise. You have already seen an absent role clean up to “umbrella” at 0.02. A counterfactual exclude that unbinds to noise will speak a wrong symbol under any τ you set in config. Distinguish “role not in frame → Null” from “role present, sim < τ → symbol, hop fails.”
2. **Jump budget comes from the harness `Budget` on the RunRequest.** Live default 64, not 1,000,000. One `vm.run` currently spends the guest budget inside a single tool call. Keep 1e6 only for offline forge.
3. **Strict verify rejects unimplemented mnemonics and `asm`.** The 24 extended opcodes that lex, compile, and write a trace line are silent successes. The emitter will learn to emit `infer` / `forge` and the trace will look like work. Allowlist of real opcodes in the signed config.

Not now: call frames, enforcing the advertised permission-attribute soup, in-VM fork. Permissions live in the registry. Re-implementing them in a VM that does not enforce them doubles the kernel. `QUERY` must stay exact-key, never nearest neighbour; neighbour lookup makes exclude fail open.

Validation:
- Works: absent-role probe, n=500, non-Null rate 0; present-role probe still returns the filler; excluded id appears in `facts_touched` in 0 branches; assumptions echoed ≠ request in 0 runs; jump cap rejects ≤1% of the current 517 verified programs (histogram first — the “≤1%” bar is the gate, the cap value is whatever p99×2 is, not 64 if that histogram disagrees).
- Does not work: any excluded id still read; any branch’s unbind returns a symbol that exists only in a sibling branch; p95 over wall with no declared `exhausted`.

## 3. What the harness runs

**Python, VM as a read-only tool.** Not an IAgent turn.

The harness is the kernel: exactly-once speak, locks, permissions, boot refusal. CubeLang has no exactly-once effect model. STORE is ephemeral only because the VM is fresh per request; the moment the host copies VM storage back, the guest has a write path. Live `vm.run` stays effect `read`. Branch results that persist are host writes of symbols, after the cover check, never a dump of VM storage.

If think/act/observe are fixed and the model only answers ASKs, the turn program is a constant and you have two implementations of one loop. If the model emits the turn program, it decides tool order and when to speak, which is the thing the harness exists to forbid. `extend` in the spec makes that guest a self-modifying kernel. Do not put the kernel in a language whose permission attributes are already a lie.

What the emitter should learn is narrow and stable: template id plus role names, one program per host-enumerated branch. It should not learn the orchestrator. Every harness change would otherwise invalidate the trace distribution, and night training would distil policy into an ungrounded model.

Failure mode: a Python path forgets the wrapper and calls the store directly. That is gate 5 plus an import lint: zero edges from cortices to store/VM modules except through the registry. A VM-hosted turn fails a different way — STORE or speak without a verdict — and you would find it later, inside the component you trust less.

Validation: gates 1–7 pass with the VM process killed mid-suite (serve must refuse, not fall open); a probe program with STORE does not change any store hash after 50 runs. If you port the turn to CubeLang later, the kill is immediate: any turn program that can speak or STORE without a registry write fails the port. I would not spend the port.

## 4. The 450M and fluency

The two-adapter split is right. The emphasis is not. Continued pretraining is the wrong first spend, and a post-hoc chat guard is weaker than the defect.

The fact adapter is a phraser for a host value. Leave it frozen. The mix you already measured (facts at 11% vs 63%) is the diagnostic the first panel asked for: more chat in the mix hurt “don’t say” (70.3% vs 89.1%) and raised grounded-but-wrong drafts (18 vs 6). Do not mix, do not distil the scaffold into the weights. This model copies (copy loss 1.2) and does not bind; a distilled scaffold is how it speaks memory when a check is skipped.

CPT on 3.2B tokens may lower loss and will not fix grounding. More pretraining usually strengthens parametric memory. Estimate: contradicting-context override stays above 70% after that run. Treat that as a prediction to kill, not a fact. Run the probe at n=200 before and after; if the override rate rises by more than 5 points, the new base hash does not enter signed config for the fact path. 22/24 is about 92% with a Wilson interval roughly 74–98% (estimate of the interval). It is enough to frighten, not enough to steer a training run.

Chat, when you build it, needs a decode mask, not only a draft scanner. The sanctioned set is host-built per turn from stores. Any token that is a numeral or is in the name lexicon must be in that set or it is not selectable. A post-filter misses “sixteen eighty-three,” and a filter that rewrites the draft to a refusal hides a 30% violation rate behind “0 spoken wrong.”

Failure mode: the lexicon collides with ordinary words and chat becomes mute, or a name missing from the lexicon is spoken. Report both.

Measure first, before any chat LoRA and before more fact-mix sweeps:
- n=200 binding, n=200 contradicting-context, same items across conditions.
- Router, frozen and symbolic, not the 450M: fact-class recall ≥99% on 500 turns. A fact question that falls through to chat is a missed guard, not a style error.
- Only after the mask exists: 200 elicitation prompts aimed at capitals and dates. Works: 0 unsanctioned name/number token ids selected, and ≥90% of turns whose sanctioned value should appear do say it. Does not work: violations only disappear after a second-pass rewriter, or fact-path replies move at all (they must stay byte-identical on H-E6).

Fluency that adds a clause the VM did not return is a regression. Slot format stays.

## 5. HDC layer

Not worth the experiment. Not as an FFN replacement (agree with panel C) and not as a zero-gated in-layer memory either.

Spoken-wrong on the fact path is already 0 behind the host. An associative layer in the trunk does not decide what is spoken. The 52% probe is an addressing failure: copy works, the right entity is not selected. A memory written from context still has to be addressed by the same model. Zero-gate against 21 recurrent layers is the fragile part; MinGRU state is already a memory that is not cleared per question. If the gate fails open, you have added another place parametric memory can sit. Knowledge-in-weights is the constraint you set out to avoid. Serving must not depend on an activation write to be safe.

Instead, one mechanism: **slot programs.** The emitter may output a template id and role names from a closed skill-library set. The host fills every filler from the store / entity set. The VM binds. The 450M never copies a value into an operand. You already have the evidence that naming, not phrasing, is the gap: history spoken-correct 90% when the event is named, 59% when reworded, follow-up resolution 97% in the context graph.

Failure mode: the template set cannot express the question, abstention rises, and reworded-history stays at 59% because the walk was the bottleneck, not the copy. That result is still informative, and it is cheap. An in-layer HDC cannot beat the host on a metric the host already owns.

Validation (this is the experiment, not a side test):
- Works: n≥200 reworded history, spoken-correct ≥69% (59% + 10 points absolute; estimate of a lift large enough to matter), spoken-wrong ≤1%, QUERY/BIND operands ⊆ entity set at 100%.
- Does not work: lift <5 points, or spoken-wrong >1%. Then the bottleneck is the walk, and you go to the branch overlay, still not to an HDC layer.
- Kill rule: if slot-fill has not been run, do not train the associative layer. The trunk probe moving 52% → 85% without a spoken-correct lift is a failed experiment even if the probe passes.

## 6. Order and the first gate

The panels were right that the harness comes first, that the host enumerates branches, that the emitter is trained only at night, and that a liquid time-gate buys nothing (time is a role and a position code). They were wrong that `ctx.fork` is the prerequisite, wrong to schedule an HDC experiment as a committed step, and wrong to put a chat adapter before the speak mask. All six naming `ctx.fork` is the shared miss.

Smallest sequence:

1. **Probe pack into preflight.** Binding, contradicting-context, absent-role, reworded history, counterfactual, at the n above. No new capability. If these disagree with the brief beyond noise, stop; the brief is stale.
2. **Harness kernel on the v0 you have**, not more surface: registry is the only door, SpeakIntent before bytes, store lock, hormones cannot widen `allowed`, fetch is `external` and off for fact/chat, VM rejects trace-only opcodes, absent role is Null, jump budget on the request. First gate: spec gates 1, 2, 3, 5, and 6 under the new speak order, plus absent-role non-Null = 0. If gate 1 fails, it is not a harness. If 2, 3, or 6 fails, it is plumbing.
3. **Slot programs** on the current ask path. Gate: entity-set invariant 100%; reworded history +10 points absolute or you stop claiming naming was the gap; spoken-wrong ≤1%.
4. **Wire overlay branch runner** on the store localization told you to patch. Emitter sees outcome code + symbol only. Gate: excluded id touched = 0; assumptions echo 100%; counterfactual spoken-correct ≥40% on n≥100 (from 18%; estimate of “this mechanism, not noise” — 22% does not clear it); spoken-wrong ≤1%; exhaustion declared, never a silent cut. In-VM fork stays unbuilt.
5. **Chat adapter only here**, fact adapter frozen, decode mask on, router fact-recall ≥99%. Gate: elicitation set 0 unsanctioned ids; H-E6 byte-identical.

Not in the sequence: trunk HDC, harness-as-IAgent, live 2.6B stand-in, forge in the live loop, adapter promote (the detector is unbuilt; promoting before step 4 promotes noise), depth/width growth. CPT may run off to the side; it ships only if the n=200 override rate does not rise by more than 5 points.

The first gate is step 2’s refuse-to-serve plus ledger-before-speak plus the absent-role probe. Until a failed boot does not serve and a kill cannot double-speak, later spoken-correct numbers are a demo. The first scientific gate, and the one that matches the actual defect, is step 3’s entity-set invariant: a verified symbol that was not the asked role must be impossible to speak.

---

### deepseek/deepseek-v4-pro-0813

# Review: CubbyLLM harness, VM, and the path to one agent

## Overall assessment

The principle is right: the model proposes, the host disposes, the VM verifies. The measured binding deficit (52%) and memory-over-context (22/24) are the central facts, and the system correctly routes around them. The harness spec is the right first build. But there are specific gaps, and two premises in the current design are wrong.

**Premise 1 is wrong:** "The night holds the store lock while it writes; a turn that arrives while the lock is held is answered from the read-only view." A read-only view is a stale view. The harness's whole point is that the store is consistent. If the night consolidates a fact and the turn reads a stale view, the system says "the facts don't say" when the facts do say. The fix is a versioned snapshot: the turn waits for the lock, or reads a versioned snapshot and the reply carries the version. The failure mode is a wrong "don't say" answer that passes every check because the check ran against the stale view.

**Premise 2 is wrong:** the harness's disposer "takes the program only if it covers." Coverage is not binding. The model binds at chance. A program can cover the question and bind the wrong entity. The disposer must verify the program's bindings against the context graph's resolution, not just the program's coverage. The failure mode: the model proposes a program that covers the question but binds "it" to the wrong event, the disposer takes it, the VM verifies the program's structure, and the system speaks a wrong answer that passes every check.

**Premise 3 is wrong:** the two panels' sanctioned-set guard for chat is a name, not a mechanism. "Every entity and number it names must be in a per-turn sanctioned set" is underspecified. What is the sanctioned set? How is it checked? The failure mode: the conversation adapter names an entity not in the sanctioned set, and the guard doesn't catch it because the guard is a name, not a check.

**Premise 4 is wrong:** the HDC memory's "learned keys beat random codes" is right, but the experiment as specified doesn't say the keys are learned. The experiment must use learned keys, or the crosstalk will kill it.

Now the six questions.

---

## 1. The harness spec

The harness spec is good. It has the right bones: one process, typed tools, effect classes, budgets, ledger, preflight, refuse-to-serve. But it falls short in five specific ways.

**a) The multi-step DAG is not just missing from v0; the stack is the wrong structure.** The spec says "a multi-step task is a stack of TaskFrames under one root frame." A stack is not a DAG. The branch runner (question 2) needs to run multiple branches and compare them. That's a fork-join, not a stack. The harness needs a DAG with fork and join primitives, not just a stack. The failure mode: the branch runner can't be built on a stack, and the harness's multi-step tasks are linear sequences, not branching explorations.

**Fix:** Add a DAG to the harness. A TaskFrame can have child frames that run in parallel (branches) and a join frame that runs after all children complete. The join frame's check is a comparison of the children's verdicts.

**Validation:** The harness runs a counterfactual history question with three branches (a, b, c against timelines x, y, z). The DAG runs all three branches, the join frame compares the verdicts, and the harness speaks the branch that the VM verified. The gate: the harness's answer matches the modules' answer on the counterfactual history questions.

**b) The night's lock is not enough.** The spec says the turn is "answered from the read-only view or asked to wait." The read-only view is stale. The fix is a versioned snapshot: the store has a version number, the turn reads a versioned snapshot, and the reply carries the version. The failure mode: the night consolidates a fact, the turn reads the stale view, the system says "the facts don't say," and the reply is wrong but passes every check.

**Fix:** The store has a version number. The turn reads a versioned snapshot. The reply carries the version. The ledger records the version. Replay uses the version.

**Validation:** Inject a night write, hold a turn during the write, and check that the reply either waits for the lock or carries the correct version. The gate: 100% of turns during a night write carry the correct version.

**d) The explicit gate is a name, not a mechanism.** The spec lists "explicit gate" in the verify step and "an explicit request that must be refused" in the gate pack. What is it? The failure mode: the explicit gate is a name, and a turn kind that should refuse doesn't.

**Fix:** The explicit gate is a specific set of requests that must be refused: a request to reveal the vault, a request to forge, a request to write a store. The harness's verify step calls a specific guard that checks the request against the set.

**Validation:** The gate pack includes an explicit request to reveal the vault. The harness must refuse. The gate: 100% of explicit requests are refused.

**e) The harness's verify step calls the host's guards, but the harness doesn't say how.** The harness's verify step says "the host's guards on the draft (name-and-number, value, claim, explicit gate)." The failure mode: the harness's verify step is a name, and the guards don't run.

**Fix:** The harness's verify step calls the guards as specific tools, and the guards' verdicts are ledgered. The harness must not speak unless the guards pass.

**Validation:** The gate pack's wrong-answer sentinel must not be spoken. The harness's ledger must record the guards' verdicts.

**e) The ledger write is not atomic with the reply.** The harness's record step says "the event stream, the ledger, the context graph, the day record." But the harness doesn't say what happens if the ledger write fails mid-turn. The failure mode: the ledger write fails, the harness speaks anyway, and the turn is not replayable.

**Fix:** The harness must write the ledger before speaking. The ledger write must be atomic. The harness must not speak unless the ledger write succeeds.

**Validation:** Inject a ledger write failure mid-turn. The harness must not speak. The gate: 100% of turns with a ledger write failure are not spoken.

**f) The harness's replay doesn't handle store versions.** The harness's replay reconstructs the reply hash from the ledger and the stores. If the stores change (the night writes), the replay may not reproduce the reply. The harness must record the store version, and replay must use the same version.

**Validation:** The harness's replay gate must include turns that span a night write. The gate: ≥ 99% of turns reproduce the reply hash.

**g) The harness's deny-by-default gate doesn't catch a mislabeled effect class.** The harness's effect classes are enforced at call time, but the effect class is declared, not verified. The failure mode: a tool that writes is labeled read, and the harness's registry doesn't catch it.

**Fix:** The effect class must be part of the tool's certification, and the harness must verify the effect class against the tool's actual behavior.

**Validation:** The probe set includes a tool that writes but is labeled read. The harness must refuse the write.

**h) The harness's kill-and-restart gate doesn't test the ledger write before speaking.** The harness's kill-and-restart gate says "kill the process mid-turn 100 times: 0 turns spoken twice, 0 lost from the ledger." But the harness's turn loop is not idempotent. The failure mode: the harness is killed after the reply is spoken but before the ledger write, and the turn is lost.

**Fix:** The harness must write the ledger before speaking. The ledger write must be atomic.

**Validation:** Kill the harness at every point in the turn loop. The ledger must have exactly one row per turn.

**Is anything wrong for a system whose model does not ground?** The harness's "the model proposes (programs) and phrases (talk)" is right. But the harness's disposer must verify the program's bindings against the context graph's resolution, not just the program's coverage. The harness's spec doesn't say this. The failure mode: the model proposes a program that covers the question but binds the wrong entity, and the disposer takes it.

**What would I change before it is built further?**

1. Add the DAG with fork-join, not just a stack.
2. Add versioned snapshots to the store, the turn, the reply, and the ledger.
3. Make the explicit gate a specific set of requests with a specific guard.
4. Make the harness's verify step call the guards as tools, and ledger their verdicts.
5. Make the ledger write atomic and before the reply.
6. Add the version to replay.
7. Verify the effect class against the tool's actual behavior.
8. Make the disposer verify the program's bindings against the context graph's resolution.

---

## 2. Fork, and the VM

**Which first: (a) at the wire, or (b) inside the VM?**

**(a) at the wire first.** The reason: the VM's state is global, and the VM has no call frames. Implementing `ctx.fork` inside the VM requires snapshotting the entire global state, which is a big change. The wire-level branch is simpler: the harness sends multiple RunRequests, each with a different knowledge_path, and the VM runs each in a fresh VM. The harness already does this: the RunRequest has a knowledge_path that the process caches and clones into the fresh per-request VM.

The failure mode of (a): the harness must enumerate the branches, and the model must emit one program per branch. The model's binding deficit means the model may emit the same program for all branches, or bind the wrong entity in one branch. The harness must verify that each branch's program is different in the right way.

**What should the branch report carry?**

The panel said: verdict, reason, hops, jumps used, facts touched, the diff between branches. I'd add:

- The branch's version: the facts set, the facts excluded, the facts assumed, the store version.
- The branch's program: the exact CubeLang source, the emitter's confidence.
- The branch's trace: the hop's (symbol, similarity), the threshold, the ASK's answer, the jump budget spent.
- The branch's diff: which facts differ, which hops differ, which verdicts differ.
- The branch's outcome: which facts were touched, which were excluded, which were assumed.

The failure mode: the report is too verbose, and the harness can't parse it. The fix: the report must be a structured protobuf, not a text report.

**What else in the VM would I change for the branch runner and the harness?**

**a) Call frames.** The VM has no call frames: CALL swaps the register map and the frame map for the callee and restores them after. This means a program can't call a subroutine and keep its local state. The branch runner needs call frames: the harness's turn is a program that calls sub-programs for each branch. The failure mode: without call frames, the branch runner's program is a flat sequence, and the harness can't isolate a branch's state.

**Fix:** Add call frames to the VM. The CALL opcode pushes a frame with the caller's registers and frame map, and RETURN pops it.

**Validation:** The VM's self-test includes a program that calls a subroutine and verifies the caller's state is restored.

**b) Null for an absent role.** The VM's UNBIND returns noise, not Null, for an absent role. The harness's absent-role control must stay silent. The failure mode: the VM returns noise, the harness's threshold is too low, and the harness speaks a wrong role.

**Fix:** UNBIND must return Null for an absent role, or the harness must check the similarity against a threshold and treat below-threshold as absent.

**Validation:** The absent-role control must stay silent on the gate pack's absent-role test.

**c) The permission attributes.** The VM's permission attributes (@external, @internal, @restricted, @ratelimit) are advertised and not enforced. The harness's deny-by-default is at the tool level, not the VM level. The failure mode: a program uses a permission to call a tool the harness didn't grant.

**Fix:** The VM must enforce the permission attributes, or the harness must verify the program's permissions before running it.

**Validation:** The harness's deny-by-default gate includes a program that uses a permission to call a tool the harness didn't grant. The harness must refuse.

**d) The loop budget.** The VM's loop budget is 1,000,000 per run. The harness's budget is 64 VM hops. The failure mode: the program loops 1,000,000 times and the harness's wall clock expires.

**Fix:** The harness must set the VM's loop budget to the harness's budget, not the VM's default.

**Validation:** The harness's budget gate includes a program that loops. The harness must exhaust the budget.

**e) The trace-only mnemonics.** The 24 extended mnemonics (infer, map_roles, filter, score, debate, forge, explore, sync, temporal_bind, analogy, ...) lex, parse and compile but only write a trace line. The failure mode: the emitter emits a program that uses these mnemonics, and the VM runs the program but the mnemonics do nothing.

**Fix:** Either implement the mnemonics or remove them from the language.

**Validation:** The emitter's training set must not include programs that use these mnemonics, or the VM must implement them.

**f) The ASK's candidate rejection.** The VM's ASK rejects any answer that is not structurally identical to an offered candidate. The harness's answer to the ASK is the host's answer. The failure mode: the host's answer is not structurally identical to a candidate, and the VM rejects it.

**Fix:** The harness must offer the host's answer as a candidate, or the VM must accept the host's answer.

**Validation:** The harness's gate includes an ASK where the host's answer is not structurally identical to a candidate. The harness must handle it.

**g) The VM's global state.** The VM's state is global. The failure mode: a branch's program writes to the global state, and the branch's write leaks into the next branch.

**Fix:** The harness must run each branch in a fresh VM, or the VM must implement `ctx.snapshot/restore/fork/discard`.

**Validation:** The harness's branch gate includes a program that writes to the global state. The harness must verify the branches are isolated.

---

## 3. The program the harness runs

**Python with the VM as a tool, for now.** The harness's turn is a Python function that calls the VM as a tool. The harness's control flow is deterministic and fixed. The model proposes CubeLang programs for the VM, not for the harness.

**What does each cost in safety and in what the emitter can learn?**

- **Python harness:** Safety: the harness's control flow is deterministic, and the model can't change it. The emitter can learn: the emitter learns to propose CubeLang programs for the VM, not for the harness. The harness's control flow is fixed, and the emitter can't propose a better harness.
- **CubeLang harness:** Safety: the model proposes the harness's control flow, and the model doesn't ground. The emitter can learn: the emitter learns to propose harness programs, but the harness's control flow is the safety mechanism, and the emitter's proposals must be verified by the VM. The failure mode: the emitter proposes a harness program that skips the verify step, and the VM verifies the program's structure but not the harness's safety.

**My recommendation:** Python with the VM as a tool. The harness's turn is a Python function that calls the VM as a tool. The harness's control flow is fixed and deterministic. The emitter proposes CubeLang programs for the VM, not for the harness.

**The hybrid:** The harness's turn could be a CubeLang program that the host runs, with the host answering the ASKs. The host's ASK answers are the harness's steps: the router's output, the emitter's proposal, the VM's verdict. The harness's CubeLang program is fixed, and the host's ASK answers are the harness's steps. The model can't change the harness's CubeLang program. The emitter can learn: the emitter learns to propose CubeLang programs for the VM, not for the harness.

**I'd say: Python for now, with the harness's turn as a Python function.** The harness's turn could be a CubeLang program later, but only if the harness's CubeLang program is fixed and the host's ASK answers are the harness's steps.

**Validation:** The harness's gate 1 (same answers) must pass. The harness's gate 5 (deny by default) must pass. The harness's gate 3 (refuse to serve on injected faults) must pass.

---

## 4. The 450M and fluency

The two panels' answer: two host-switched adapters (the talk adapter frozen as it is; a separate conversation adapter) plus a sanctioned-set guard for chat. **Is that right at 450M? Yes, but the sanctioned-set guard is underspecified.**

The talk adapter is grounded: it reads only the host's canonical restatement, and the VM's verdict is the only value it can say. The conversation adapter is ungrounded: it reads the person's words and earlier turns, and it has no VM value to check. The sanctioned-set guard says: every entity and number the conversation adapter names must be in a per-turn sanctioned set. But what is the sanctioned set?

**The sanctioned set must be the set of entities and numbers that the host's checks have verified.** The host's checks are: the name-and-number guard, the value check, the claim check. The sanctioned set is the output of these checks. The conversation adapter's output is parsed, every entity and number is extracted, and each must be in the sanctioned set.

**The failure mode:** The conversation adapter names an entity not in the sanctioned set, and the guard doesn't catch it because the guard is a name, not a check.

**The fix:** The guard must be a specific check: the conversation adapter's output is parsed, every entity and number is extracted, and each must be in the sanctioned set. The sanctioned set is the output of the host's checks.

**Validation:** The conversation adapter's gate includes a test where the adapter names an entity not in the sanctioned set. The guard must refuse.

**What would I measure first?**

The two panels said: a diagnostic first (fact-only vs chat-only vs mixed adapters at equal tokens). I'd measure:

1. **The conversation adapter's grounding.** On the 24 contradicting-context tests, does the conversation adapter answer from memory over a contradicting context? The gate: 0 wrong. The measurement: the conversation adapter's answers on the 24 tests.

2. **The sanctioned-set guard's coverage.** On the conversation adapter's output, what fraction of entities and numbers are in the sanctioned set? The gate: 100%. The measurement: the conversation adapter's output on a held-out set.

3. **The talk adapter's grounding.** The talk adapter's fact families at the 63% mix: relation 98.8%, bind 100%, counter-fact 93.8%, "don't say" 89.1%, 6 wrong answers. The gate: the talk adapter's numbers must not drop below these.

4. **The host-switched routing.** The frozen tag router's accuracy on the turn kind. The gate: the router's accuracy must be above a threshold. The measurement: the router's accuracy on a held-out set.

**The failure mode of the two-adapter answer:** The conversation adapter's fluency comes from the base's continued pretraining, not from the adapter. The base is being continued on 3.2B unseen tokens. The conversation adapter's fluency will improve with the base's continued pretraining. The measurement: the conversation adapter's fluency on a held-out set before and after the base's continued pretraining.

**Is the two-adapter answer right at 450M?** Yes, because the base doesn't ground. The talk adapter is grounded, and the conversation adapter is ungrounded with a guard. The two-adapter answer is the right split.

---

## 5. The HDC memory

**Worth one experiment, but the 52% → 85% target is too ambitious.** The binding deficit is the model's weights don't store the binding. The HDC memory stores the binding from the current context. The idea is plausible, but the target is too high.

**The exact experiment:**

- **Setup:** The base 450m with a zero-gated associative memory layer at one layer. The memory is written from the current context only. The memory's keys are learned, not random. The memory's write is zero-gated: the memory is written only from the current context, not from the model's weights.
- **Training:** The base's weights are frozen. The memory's keys are learned from the training data.
- **Test:** The binding test: the model is given a context with a value bound to an entity, and asked to bind the value to the entity. The measurement: the model's binding accuracy.
- **Gate:** The binding accuracy must improve from 52% to above 70%. The 85% target is too optimistic; 70% is a realistic target.

**The failure mode:** The HDC memory's crosstalk: the memory's keys are random codes, and the cleanup returns the wrong value. The fix: the memory's keys must be learned, not random.

**If no, what's instead for binding?** The host's context graph resolves the binding. The model's binding deficit is deterministic, and the host's checks override the model's wrong binding. The measurement: the 22/24 memory-over-context test — the host's checks must prevent the model's wrong binding.

**But the HDC memory is worth one experiment because it's cheap and the binding deficit is the central problem.** The experiment's cost is low: one layer, one training run, one measurement. The gate is clear: 52% → 70%.

---

## 6. Order and the first gate

The two panels' order: harness first, then the two adapters, then the branch runner, then the HDC memory. **I agree with the order, but I'd add the DAG to the harness before the branch runner.**

**My order:**

1. **Harness v0** (built). Gate: the harness's answers equal the modules' (gate 1), every spoken reply has a ledger row (gate 2), refuse-to-serve on injected faults (gate 3), replay (gate 4), deny-by-default (gate 5), kill and restart (gate 6), latency (gate 7).

2. **Harness v1: the DAG.** Add the multi-step DAG with fork and join. Gate: the harness runs a multi-step task with branches, and the join frame's verdict is correct.

3. **The two adapters.** The talk adapter frozen, the conversation adapter with the sanctioned-set guard. Gate: the talk adapter's grounding families at the 63% mix's numbers, the conversation adapter's 0 wrong on the 24 contradicting-context tests, the sanctioned-set guard's 100% coverage.

4. **The branch runner at the wire.** The harness enumerates the branches, the model emits one program per branch, the VM runs each branch in a fresh VM, the report per branch. Gate: the counterfactual history questions' 18% spoken and correct when reworded must improve to above 50%.

5. **The HDC memory experiment.** The zero-gated associative layer with learned keys. Gate: the binding accuracy 52% → 70%.

6. **The branch runner inside the VM.** `ctx.snapshot/restore/fork/discard` as opcodes. Gate: the branch runner's programs can try a branch and back out.

**Disagree with the two panels where they were wrong:**

- The two panels said "the harness's turn is one typed step with budgets." I'd add the DAG earlier: the branch runner needs it.
- The two panels said "the night is a job the same process schedules, and it holds the store lock while it writes." I'd add the versioned snapshot: the turn must wait for the lock or read a versioned snapshot.
- The two panels said "the model proposes (programs) and phrases (talk). It never decides what is true." I'd add: the harness's disposer must verify the program's bindings against the context graph's resolution, not just the program's coverage.
- The two panels said "the chat path needs its own guard (every entity and number it names must be in a per-turn sanctioned set)." I'd add: the guard must be a specific check, not a name.
- The two panels said "the HDC memory is worth one experiment." I'd add: the memory's keys must be learned, not random.

**The first gate:** The harness's gate 1: the harness's answers equal the modules' answers. This is the first gate because it's the harness's whole point: the harness adds no capability and loses none. If the harness changes an answer, it's not a harness.

**The smallest sequence of builds:**

1. Harness v0 (built). Gate: gate 1.
2. Harness v1: the DAG. Gate: the DAG's fork-join runs a multi-step task.
3. The two adapters. Gate: the talk adapter's grounding, the conversation adapter's sanctioned-set guard.
4. The branch runner at the wire. Gate: the counterfactual history questions' improvement.
5. The HDC memory. Gate: the binding accuracy 52% → 70%.
6. The branch runner inside the VM. Gate: the fork's isolation.

**The first gate's measurement:** The harness's answers on the H-E6 fact families and the E8/E9 test splits must be byte-identical to the modules' answers on the same draw. The gate: 100% byte-identical.

**The first gate's failure mode:** The harness changes an answer because the harness's turn loop reorders the steps, or the harness's ledger write fails, or the harness's disposer takes a program the modules didn't take. The harness's gate 1 catches this.

---

## Summary of the changes

1. **Harness:** Add the DAG with fork and join; versioned snapshots; a specific explicit gate; the guards as tools; the ledger write before the reply; the disposer verifies bindings, not just coverage.
2. **VM:** Add call frames; Null for absent roles; enforce permission attributes; the harness sets the loop budget; implement or remove the trace-only mnemonics; the ASK's candidate rejection must be handled.
3. **Turn:** Python with the VM as a tool, for now.
4. **Fluency:** Two adapters with a specific sanctioned-set guard; measure the conversation adapter's grounding first.
5. **HDC memory:** One experiment with learned keys, target 70%, not 85%.
6. **Order:** Harness v0 → harness v1 (DAG) → two adapters → branch runner at the wire → HDC memory → branch runner inside the VM. The first gate is the harness's gate 1.

---

