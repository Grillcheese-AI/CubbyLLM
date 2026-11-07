# Brief: CheezML, one record vocabulary for CubbyLLM, MoWM and CubeLang (2026-11-07)

Written by Cowork for an OpenRouter panel (Nick: "give it to our ai openrouter panel so they can generate one with
everything we have"). Every fact below comes from the repositories, orders and papers named. Where something is not
stated here, say you don't know; don't fill it in.

## 1. The task
Design **CheezML** (the name is a joke on XML; the format is JSON-LD/RDF, not XML). It is one vocabulary for every
record the system writes:
- **The identifiers:** terms under `https://w3id.org/cheezml/` (redirected later to `ns.grillcheeseai.com`), in
  modules `core#`, `cubby#`, `mowm#` and `cubelang#`, with prefixes `cheez:`, `cubby:`, `mowm:` and `cube:`.
- **Records** get their own ids (`urn:uuid:...`), never in the vocabulary's namespace.
- **No version numbers** in the namespace (`owl:versionIRI` instead).

Deliver, in this order:
1. **The classes and properties.** For each: the module, the domain and range, a one-line definition, and the
   standard term it reuses or specialises (schema.org, PROV-O, the Decision Ontology, OWL-Time, SKOS, RO-Crate,
   EmotionML by link). Reuse wins: mint a term only when no standard one fits.
2. **One JSON-LD 1.1 `@context`.**
3. **Six example records,** with values plainly marked as placeholders where not given here:
   1. a certified answer turn with its certificate;
   2. a refusal with its reason;
   3. a MoWM routing decision that ends in a spawn;
   4. a world split (a sub-world at z+1);
   5. a helper rejected in its sandbox (order 0062);
   6. an invoice field sent to human review and then committed (order 0061).
4. **SHACL shapes** for the three classes you judge most important.
5. **CubeLang programs** for MoWM routing and spawning. They must follow section 4's rules.
6. **What you would cut,** and your open questions.

Keep it under about 7,000 words. Plain, exact prose; no marketing.

## 2. Hard rules (each comes from a written decision; don't trade any away)
- **Records are written by the host from what ran, never declared by hand.** A bridge that never started is not
  "Active"; a check that never ran has no "passed".
- **No prose "rationale" fields** holding model-written reasoning. Any readable sentence is produced by a template
  from codes and values.
- **Scores are not thresholds.** Every threshold carries its calibration: the screen that set it, the data and the
  date. Thresholds do not transfer between encoders or corpora.
- **Vectors, weights and blobs are referenced by hash,** never embedded. A seed alone does not reproduce trained
  weights; record the weights' hash and the training run.
- **Retire, never delete.** Retired, rejected and pruned things stay, with their reason.
- **"Certified" is not "correct".** A certificate says the answer follows from the facts and rules used, under the
  recorded checks. Never record a truth score.
- **Agreement never certifies.** Only the VM certifies; agreement between models or agents may only filter.
- **No neural model on MoWM's routing path.** Text is encoded by a word-table lookup (FastWordEncoder, about 25 µs),
  not a sentence model.
- **No external model writes a training record or a stored fact.**
- **Two clocks per fact** (order 0055): when it held (valid time) and when the store learned it (recorded time).
  Corrections close the old version; they never overwrite it.

## 3. What already exists, and must be reused
- **The certificate (order 0054):**
  - written in PROV-O: the answer as an entity, the program run as an activity, the statements read (with source and
    revision) and the rules applied;
  - plus the decision step in the Decision Ontology: the `Question`, each proposer's program as an `Option`, the
    checks as criteria (`is_satisfied_by` / `is_violated_by`), and the answer or refusal as the `Decision`.
- **The reproducibility package (0054):** a Workflow Run RO-Crate for each evaluation run, and each hypothesis
  verdict as a nanopublication (killed results included; STATO for statistics).
- **Emotion (0020):** EmotionML 1.0 (W3C, 2014) export.
  - Values in [0, 1], with pleasure, arousal and dominance as the "pad-dimensions" set.
  - A user's emotion reading is an inference with a confidence, never a fact.
  - Cubby's own hormone state is a separate, labelled dimension set.
- **The conversation (0059):**
  - the state is kept by the host: entities by id, the last question, program and answer, and any open clarifying
    question;
  - a follow-up rewrite passes an entity guard;
  - the dialogue acts: question, certified answer, refusal (with reason), clarifying question, follow-up (with its
    rewrite), small talk. ISO 24617-2 is the reference set to consider.
- **The refusal reasons in use:**
  - no plan, relation missing, fact missing, walk failed, not certified, ambiguous;
  - the kinds of conflict: value, type, temporal, logical, recorded against derived;
  - dates too coarse;
  - unreadable source (0061).
- **The proposers in use:** the host's grammar, the everyday table, the overnight rule library, an episodic memory of
  certified chains, and the language model's program emitter. Every answer records its proposer.
- **Story worlds (0060):**
  - actions are programs with exact prerequisite checks, and soft (tone) checks are labelled soft;
  - state changes are dated, with old value, new value and turn;
  - invented facts have the source "invented at turn N";
  - a story world is isolated from the real-world store.
- **Documents (0061):**
  - regions with their text and the reader's version;
  - every number in a field must appear in its region's text;
  - exact checks (format, arithmetic, purchase order, supplier, duplicate);
  - derived values are suggestions only;
  - a reviewer is a source (role, id, time, region); a second approval is needed above an amount.
- **Helpers (0062):**
  - proposed by Cubby's own proposers and tested in a locked-down VM session;
  - adopted only if they reproduce every verified answer in scope and change nothing outside it;
  - the host promotes them; a rejected helper is kept with the case that broke it;
  - the proposer checks are the rejection rate over a window, and escape attempts (which must be 0).
- **Split questions (0063):** parts, routes, each part's verdict, the combination program and its verdict; partial
  answers name the missing part.
- **Domain worlds (0058):** an ontology module, open records, rules and a question set. A change to a domain's
  mapping, rules or shapes is published only with no new wrong answer and a gain.

## 4. MoWM as it is (from the `mowm` repository)
- **The pipeline:** Input → HYLA encoder → DSelect-k router → expert worlds (recursive along a Z axis) → bind and
  mixture → cleanup → CVL (a contrastive value estimator) → HYLA decoder → output. HYLA is a hypernetwork that
  generates block-code transition functions. The components come from `cubemind` (BlockCodes, HYLA, DSelectKGate,
  ContrastiveValueEstimator, DisARM, WorldEncoder, WorldManager) and run on `grilly`.
- **The axioms:** a Global World holds about 350 seeded axioms across 14 domains, plus axioms discovered by a
  three-step loop (semantic mapping, symbolic regression, axiom distillation). Axioms are selected by cosine
  similarity and composed by VSA bind.
- **A world** (`world.py`) is a tree node.
  - It holds one or more HYLAs, whose transition deltas are merged by a `MergeStrategy` (for example
    `BindComposeMerge`, weighted bundling) and applied by block-code binding.
  - Sub-worlds at z+1 handle residuals.
  - Each world has `z_max` (maximum depth) and `tau` (a confidence threshold for spawning sub-worlds), and
    `spawn_subworld` refuses at `z_max`.
  - World ids come from one global counter starting at 1000. Oja consolidation applies.
- **The router** (`router.py`) is DSelect-k gating over a dynamic pool of worlds.
  - `maybe_spawn` spawns a world when the residual norm exceeds `tau_spawn` (default 0.5).
  - A world idle for `tau_prune_steps` (default 100) is pruned. Whether pruning deletes or retires is an open
    question.
  - Router-spawned ids come from a counter starting at 5000.
- **The bridge to CubbyLLM** (`bridges/cubby_bridge.py`, measured in CubbyLLM's screens):
  - **Route** only if the best world's centroid score clears an absolute floor AND, with two or more worlds, the
    margin over the runner-up clears `tau_margin`. Otherwise **spawn**: a spawned specialist is recoverable, and a
    mis-route is not.
  - **Measured:** centroid routing beat max-over-axioms, 0.680 against 0.589 macro (these are scores, not
    thresholds), with about 40× fewer comparisons. At its calibrated threshold it routed with 92.4% precision and
    sent 86.9% of never-seen domains to spawn.
  - **Inter-world query:** when the routed world's best member falls short of `tau_answer`, the query passes world
    to world, matching exhaustive search's answer rate with about 60× fewer member comparisons.
  - All thresholds are calibrated per deployment.
- **The possibility oracle** (`bridges/possibility.py`):
  - fact worlds, whose members are facts encoded by the word table;
  - routing by best member (the default) or by centroid. Centroid routing doesn't carry over to fact worlds: a member
    sits about 1/√n from a centroid of n quasi-orthogonal facts;
  - support at or above `tau_answer`;
  - prediction cleaned up to the top-k members;
  - refusal by the host's `contradicts` rule.
- **The vectors:** sparse block codes, k = 80 blocks × l = 128 slots (D = 10,240), the qFHRR integer-phase form
  (the "Semantic Block Codes" preprint).

## 5. CubeLang as it is (from the `cubelang` repository, checked with its compiler on 2026-11-07)
- **The language** is a typed scripting language, `program X implements I { storage {...} functions }`, with
  interfaces such as ISolver and IAgent. The spec is far larger than what compiles.
- **What executes today** (the "safe subset"): create, assign of a literal, add, sub, mul, div, sum, push, pop,
  query, store, recall, remember, forget, compare, if/else and while, with flat operands only.
  - `for`, `match`, intra-program `call` and early `return` are not executed yet.
  - Strict mode (the default) rejects non-executing constructs, `throw` included.
  - A map literal or a method call's result cannot be an operand.
- **Real opcodes** (from `cubelang disasm`; their byte values are synced across three repositories and must never
  change): CREATE 0x00, ASSIGN 0x02, ADD 0x03, COMPARE 0x0b, QUERY 0x0c, STORE 0x0d, COND 0x11, JMP 0x14, LABEL 0x15,
  REMEMBER 0x21, SUM 0x34, RETURN 0x39. Don't invent opcodes or assembly mnemonics.
- **ASK / suspend:** a third execution outcome, next to a return and an error. A program that cannot choose (for
  example, two cited dates) suspends and asks the caller, then resumes by re-execution over `run-proto`.
  Implemented, with 6 tests.
- **Found on 2026-11-07, in Nick's drafts:**
  - a `container` passes `check` but compiles to nothing (no bytecode), so `vm.start_from_file`, `vm.wire` and
    `vm.bridge` don't run;
  - `@hook` is a compile error, and `@hook(*.event)` does not parse;
  - `??`, `revoke ... from` in a function body, `@pure` as an attribute, and destructuring (`let [a, b] = ...`) are
    not parsed;
  - `event` is a reserved word;
  - `implements IOrchestrator` fails unless the interface is declared.
- **The rule for your programs:** each must either use only the executing subset and ASK, or be labelled "spec only:
  does not compile today". They will be compiled by us before anything is adopted.

## 6. Prior drafts (for reference; their known faults are listed so you don't repeat them)
Nick drafted JSON-LD records for chat, roleplay, documents, science, sandboxed helpers, swarms, a container bridge
and MoWM worlds, and CubeLang programs for an orchestrator, a bridge, a MoWM router and a MoWM spawner. Gemini (argon)
proposed a JSON-LD envelope and assembly-style programs. The recurring faults:
- prefixes without a trailing `#` or `/`, and `http://w3.org` used for `xsd:`;
- schema.org claimed for terms it doesn't have;
- prose rationales, and declared statuses ("PASSED", "Active", "Certified" by agent convergence);
- scores used as thresholds (0.589, 0.680);
- a sentence model on the routing path;
- a child id computed as parent id + 1;
- a spawn on every low-confidence step;
- invented opcodes;
- claims nothing supports (a Lean proof, "verified" facts).

On the other hand, Nick's MoWM records name the real concepts: HYLA, merge and cleanup strategies, z depth, residual
spawning and world ids from 1000.

## 7. What we will do with your answer
We compare the panel's answers, write one synthesis, and check it with tools before anything is adopted:
- every `@context` and record is expanded by a JSON-LD processor;
- records are validated against the SHACL shapes;
- every program is compiled with `cubelang check`.

Your answer is an opinion, not a finding.
