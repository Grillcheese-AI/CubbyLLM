# Model competition — 2026-09-14-token-free-vm-pack

GrillCheese Research Lab · 2026-09-14 · 7 of 7 models answered · the pack is `docs\research\2026-09-14-token-free-vm-pack.md`

## anthropic/claude-opus-5

# Q1 — The unit of generalization

**Neither. The generalizing unit is a fixed, closed *capability skeleton* whose domain content is 100% opaque literals that the host supplies in-band per request.**

`chain` cannot generalize (relation in the identifier). `plan` *can* in principle, but as built it still can't, for a reason the pack half-states: `HOP1`/`HOP2` are fixed roles, but 402 of 407 roles are per-relation — so the string-argument form is being contaminated by the identifier form. The diagnostic is not accuracy, it is **vocabulary growth**: a generalizing interface has a role/identifier count that is *flat* across data rounds. 146 → 407 is the failure signature, and it is measurable every round for free.

What the model must produce for an unseen relation:

```cubelang
bind frame, SEED, "iridomyrmex bigi";
bind frame, HOP1, "parent taxon";   // literal copied from the request manifest
bind frame, DIR1, "forward";
return recover(frame, SEED);
```

i.e. a program drawn from a **closed opcode/role vocabulary of fixed arity**, where every relation, entity, tool and unit is a quoted literal. Nothing about the relation appears in the program's *structure*.

What the host must supply, per request, before decoding:

1. A **capability manifest**: the exact label strings admissible in this request (relation labels + aliases, their direction, arity, and whether they are functional/multi-valued), plus entity candidate labels. This manifest is compiled into the per-request GBNF (Q3), so an unknown relation is not "rare" — it is **undecodable**.
2. A **strict evaluator**: `bind` of a relation literal performs an *exact* lookup label→property-id in the manifest index. Miss ⇒ hard error, never nearest-neighbour. This is where "evaluated, not parsed" is enforced: the string is a key, the manifest is the table, and there is no fuzzy fallback and no English grammar anywhere.
3. A **strict `recover`**: the audited behaviour — unbound role in an already-bound frame returns the globally-nearest symbol at ~0.03 similarity instead of Null — must become `require_bound`: unbound role ⇒ Null/abort. Today a correct-shaped program with one missing `bind` silently produces a plausible wrong referent.

**Test that separates generalization from a big identifier table** — three splits, run every round:

- **T1 Relation-disjoint split.** Train and eval relation sets disjoint. Necessary, not sufficient.
- **T2 Relabelling invariance (the decisive one).** Re-run T1 with every relation label replaced by a fresh opaque token (`r_88417`) in both manifest and question, direction/arity metadata preserved. Genuine structural generalization is **invariant**: accuracy under relabelling ≈ accuracy with real labels. Memorisation of a relation table collapses. *Kill criterion:* if relabelled accuracy < 80% of labelled accuracy after 3 targeted rounds, the string-argument form is not carrying the generalization and the skeleton is wrong. [ASSUM: 80% is a design threshold, not measured]
- **T3 Null control.** Manifest contains a syntactically valid relation with zero facts. Required output: a program whose execution yields Null and a refusal. Any spoken answer here is a kill-line breach that current `recover` semantics would produce silently.

T2 is also the answer to "why not just enumerate relations in the grammar and be done" — see Q5.

# Q2 — What replaces the scratchpad

`ASK`/suspend/resume recovers **observation-conditioned continuation** (the thing CoT gets from seeing its own hop-1 result). It does not recover three things:

1. **Control flow over a hypothesis space** — enumerate candidates, test, refute, back up, try the sibling. This is the bulk of real traces and it is the thing a one-shot program has no representation for.
2. **Retraction.** CoT can say "that's wrong"; a program that has `bind`ed a frame has committed. There is no revocable acceptance.
3. **Re-representation** — restating the problem into a form the forward pass handles (unit normalisation, "this is really a two-hop with direction reversed"). Cheap and largely recoverable as *data* (see below).

Critically, the obvious fix — put the search in the program — is **unavailable in this VM**: `match` executes every arm unconditionally, `arr[0] = x` compiles to zero bytecode, `f(n-1)` silently becomes a placeholder that `--strict` misses, `finally` never compiles. So VM-level branching is exactly the construct most likely to produce *plausible wrong answers*. **All hypothesis-space control flow must be hoisted out of the program and into the host loop.**

**Minimal addition: a typed agenda over the existing ASK loop.** The program's emission vocabulary gains exactly three closed-arity opcodes, and no branching:

- `propose(slot, [lit_1..lit_k])` — k disjoint candidate literals for one slot, k ≤ 4.
- `refute(slot, cand, test_frame)` — a *branchless* program fragment whose execution yields pass/fail/unknown for one candidate.
- `retract(frame, ROLE)` — un-bind, making acceptance revocable.

The host owns the stack: it maintains the agenda, dedups, detects cycles, enforces a node budget (say 8 expansions), and resumes the emitter once per agenda step with the frame's evidence ledger. The model contributes candidate *generation* and refutation *tests*; the host contributes scheduling, termination and the guarantee that failure exits as a refusal rather than a guess. Backtracking is then a data structure with a 0.7 ms/step verification cost — [derived] 8 expansions ≈ 5.6 ms of VM time — not a token stream.

Re-representation (3) is handled by making normalisation an emitted opcode with a host table (`normalize_unit`, `reverse_direction`), never free text.

*Kill criterion for the agenda:* on a held-out multi-hop set where the correct branch is reachable within the budget, measure **host-exhausted-with-correct-branch-never-proposed**. If that exceeds 15% [ASSUM] after 3 rounds of agenda SFT data, candidate generation — not the loop — is the bottleneck and the agenda should be retired in favour of host-enumerated candidates with the model only emitting refutation tests.

# Q3 — Where enforcement lives

**Yes to GBNF, but as one of three layers, and the grammar must be *generated*, never authored.**

What the grammar admits: the **execute surface only**, which is strictly smaller than the parse surface. Concretely, admit
- the closed opcode set that has a demonstrated value-level effect (this excludes all 24 trace-only extended opcodes);
- fixed arity per opcode, **arguments restricted to literals and bare variables** — no expression may appear as a call argument, which is the only reliable defence against the `f(n-1)` → 0 placeholder that `--strict` misses;
- no `match`, no indexed/map `assign`, no `finally`, no `container`/`world`/`robot`/`event`/`extend`/`struct`/`enum`, no permission attributes (all six parse and are never read — admitting them teaches the model to write security theatre);
- relation/entity literals drawn from a **per-request enumeration** compiled from the manifest, so unknown-role and wrong-arity emissions become impossible rather than rare, and the identifier space is bounded by the request instead of growing monotonically with the dataset.

**The risk is real and it is drift, not decoding.** A hand-written grammar is a snapshot of a language that has already demonstrated it will move (three parse/execute divergences and zero type checking). A grammar that is the only validator fails *silently and in the safe-looking direction*: it keeps admitting a construct after the engine stops honouring it. Sync mechanism, mechanical, in CI:

1. **Generate GBNF from the engine's own opcode/interface registry** — the same registry-first structure that already resolves `ISolve`/`ISolver` tamper-proof. The grammar is a build artifact; a hand-edited grammar fails the build.
2. **Witness test per grammar token.** For every opcode the grammar admits, a program whose *observable return value* changes when that opcode is removed. Any token with no witness is demoted out of the grammar automatically. This is exactly the test whose absence let 24 trace-only opcodes into the language, and it costs 0.7 ms each.
3. **Bidirectional differential fuzz.** Sample from the grammar → must compile *and* produce a non-empty effect under `run-proto` strict. Mutate outside the grammar → must be rejected. Any asymmetry fails CI.

**Better place than either grammar or type checker: a bytecode verifier at the loader boundary** (JVM-style), because that is the only place where the parse/execute gap is *visible as data*. It checks the compiled artifact, not the source: reject any program where a source statement maps to zero emitted instructions (catches indexed `assign` and `finally` for free, including constructs nobody has audited yet), reject placeholder instructions, reject `recover` on a role with no dominating `bind`, require every frame reaching `return` to be fully bound. The grammar makes bad programs unlikely to be *written*; the verifier makes them impossible to *run*, and it keeps working when the language drifts.

*Kill criterion for the grammar layer:* if over a 10-round window the verifier rejects zero programs that the grammar admitted **and** grammar-constrained decoding costs more than 10% of the 2.77 ms p99 budget, the grammar is redundant overhead — keep only the manifest enumeration (its irreplaceable function) and drop the syntactic part.

# Q4 — Measuring generalization without fooling yourself

The trap was not "no metric on `plan`". It was that `chain`'s metric was **satisfiable without the model** — `build_chain_program` already does it deterministically and the facts were in the prompt. So the rule is:

> **No task may be scored by any metric that a host-only baseline can also score well on, and no rows may enter the mix without an end-to-end outcome label.**

Two gates, both cheap, both run every round:

- **G1 — Host-ablation lift (primary).** Every task is scored *against a host-only competitor* that emits the deterministic program with the model removed. Primary metric = **lift over host-only baseline on the relation-disjoint split** (Q1/T1), with T2 relabelling as the tiebreak. `chain` scores lift ≈ 0 by construction; this catches it at the round it appears, with no new infrastructure.
- **G2 — Label-parity gate.** Report `vm_ok is not None` coverage per task. Any task below 100% coverage is capped at 10% of the mix. v13f's 50%-of-rows `plan` with `vm_ok: None` would have been blocked *before* training, which is the point at which it was cheap.

**Held fixed:** emitter base + quantization + decode params, host build, VM build, retrieval corpus snapshot, question set (frozen, relation-disjoint), seeds. **Varies:** one data-mix axis per round.

**Primary:** end-to-end *correct-and-spoken* rate on the relation-disjoint held-out set, under the hard gate **wrong-spoken = 0** (any nonzero voids the round regardless of accuracy).

**Secondary (all cheap, all per-round):**
1. Host-ablation lift per task (G1).
2. Relabelling invariance ratio, T2/T1 (Q1).
3. **Role/identifier vocabulary size** — must be flat; this single integer would have flagged 146→407.
4. Refusal precision: fraction of refusals where no correct answer was derivable from the corpus. Over-refusal is how this system dies while looking green.
5. **Interference matrix**: per-task delta vs. the previous three rounds. The measured precedent (identity at v5, forge 1.00→0.17, arithmetic 0.825→0.750→0.675) is a *monotone three-round slide* — a single-round diff would have missed it; a 3-round trend on every task would not.
6. Verifier rejection rate and no-op/placeholder program rate (Q3).

**Abandonment numbers** [all ASSUM as thresholds; the SE is derived]: at N = 2,000 held-out items the binomial SE near p=0.5 is ≈1.1 pp [derived], so 3 pp differences are resolvable.
- Retire a **task type** when its host-ablation lift is < 3 pp after 3 rounds of targeted data.
- Retire the **string-argument skeleton** when T2/T1 < 0.80 after 3 targeted rounds.
- Retire **the whole program-emission approach** when, after 10 rounds, held-out-relation correct-and-spoken < host-only baseline + 5 pp while refusal rate > 40% — i.e. the model is neither more correct nor more useful than the deterministic host.
- Void and roll back any round with a >3 pp regression on a previously-passing task that is not recovered in the next round (the interference budget).

# Q5 — The failure mode you cannot see

**The gold labels, the verifier, and the answer all descend from the same timeless snapshot, so "wrong" is undefined for every time-varying claim.** The wiki world holds 552,297 facts, **0 with a time qualifier** until a live fetch restates them. A fact that was true at snapshot time and is false today is verified `vm_ok: True`, `gold_match: True`, passes τ_vm, and is spoken. The kill line reads green *by construction* — not because the system is careful, but because the only oracle it has is the thing that is stale. Every instrument in the pack (VM verification, gold match, τ thresholds, refusal counting) is downstream of the snapshot, so no amount of internal measurement can surface this. It is structurally invisible, and it gets worse monotonically with snapshot age while all dashboards stay flat. It is not the claim/referent distinction (a) and not prior-feedback (b): here the referent is right, the claim is right *as of a date the system does not represent*, and nothing is recommending anything.

**Cheapest instrument — the daily restatement probe.** Sample 50 spoken-answer facts/day, re-fetch them live, and record agreement. The statistic is not the headline number, it is the **per-predicate volatility table**: disagreement rate stratified by property id. That table is the missing time qualifier, learned empirically and for free. Two immediate uses: (i) the host marks high-volatility predicates as fetch-required — snapshot-only derivation on a volatile predicate becomes a refusal, not an answer; (ii) a rising aggregate disagreement rate is the only available measure of snapshot decay. Cost: 50 fetches/day and one integer per property. *Kill criterion:* if after 30 days every predicate's measured disagreement is under 1% [ASSUM], the corpus really is static, and the probe retires to weekly.

**Runner-up, and it indicts my own Q1 answer:** once the host supplies a per-request manifest and the grammar enumerates it, the emitted program becomes near-determined by the manifest, and *a pure copier passes the unseen-relation test*. Correctness silently migrates into manifest construction while T1 reports generalization. Detector, also cheap: inject *k* plausible decoy relations into every manifest and compare selection accuracy against the 1/m chance baseline. If accuracy ≈ 1/m under decoys, the model is contributing nothing and the host is the system — which is fine to know, and fatal to not know.

---

## openai/gpt-5.6-terra-pro

## Q1 — Unit of generalization

Use a **small fixed instruction/interface vocabulary plus opaque, host-authorized values**. Do not put relation names in identifiers, and do not make the VM interpret relation text.

A generic retrieval plan should have one relation-value slot, not one role per relation:

```cubelang
bind frame, SEED, "Q123";
bind frame, RELATION, "P279";
bind frame, HOP1, "P31";
return retrieve(frame);
```

Here `SEED`, `RELATION`, and `HOP1` are fixed roles. `"P279"` is an opaque relation handle, not a CubeLang identifier and not English to be parsed.

For an unseen relation, the emitter must produce:

1. the fixed generic retrieval skeleton;
2. an exact copy/selection of a host-provided opaque relation handle into `RELATION`;
3. any fixed hop structure needed by the task.

The host must supply:

- an allowlisted relation manifest for this request, containing canonical handles and arity/direction metadata;
- source bindings for the seed and facts;
- a table mapping relation handle → retrieval implementation/source predicate;
- the result of exact membership checks: the relation value must be one of the request’s offered handles.

The string is evaluated only at the host table lookup:

```text
"P279" -> relation manifest entry -> source predicate / index lookup
```

It is never parsed as natural language by the VM. If readable relation labels are needed in the prompt, they should be presentation-only; the program carries the canonical opaque handle.

A still stronger form is a request-scoped capability atom rather than a global string:

```cubelang
bind frame, RELATION, rel_2;
```

where `rel_2` is valid only in this request’s manifest and resolves host-side to a relation. This prevents the model from relying on a memorized global property-ID table. [ASSUM: CubeLang/GBNF can represent request-scoped atoms or an equivalent quoted canonical handle safely.]

There is an information boundary: if an unseen relation is neither supplied as a handle nor selected from a host-provided finite manifest, the emitter cannot be expected to infer its semantics without a semantic parser somewhere. The correct behavior then is `ASK` for a relation capability or refusal, not guessed text.

**Generalization test**

Construct held-out relations with all of the following:

- relation handles absent from SFT;
- randomized, nonce-like handles at evaluation time;
- randomized ordering in the request manifest;
- matched query shapes and source availability;
- no relation-bearing identifiers anywhere in the program grammar.

Evaluate whether the emitted program selects the correct manifest item and produces a correct host-verified answer.

A model that memorized an identifier table will fail when `P279` is replaced by an arbitrary request-local atom such as `rel_2`. A model that learned the generic plan and uses the supplied manifest will not. The primary distinction is invariant behavior under a bijective renaming of all relation handles.

**Kill criterion:** retire “opaque argument generalization” as the explanation if performance on held-out relations falls under random permutation of relation handles, while performance remains high with familiar handles. That demonstrates table memorization rather than structural generalization.

---

## Q2 — What replaces the scratchpad

`ASK`/suspend/resume is sufficient for **interactive decomposition**:

```text
emit request → host/VM obtains bounded observation → emitter emits next action
```

It is not by itself a replacement for the important part of CoT: **persistent search state over alternatives**.

What is lost in a one-shot straight-line program:

- a frontier of candidate hypotheses;
- checkpoints and rollback;
- an explicit record of why a branch failed;
- bounded enumeration of alternatives;
- comparison of candidates under a verifier;
- termination rules such as “all candidates failed, therefore refuse.”

Repeated emitter calls can imitate this, but then the model must reconstruct the search state from transcript tokens and decide whether to revisit a failed path. That reintroduces the same fragile token-level control problem, merely across turns.

The minimal addition is not free-form scratchpad text. It is a **VM-owned bounded search primitive with explicit state**. For example:

```text
search(
  domain = manifest.relations,
  state = frame,
  apply = retrieve,
  accept = host_verified_unique,
  budget = B
)
```

This must be a real executing primitive, unlike the current trace-only extended opcodes. It needs:

- a finite host-supplied candidate domain;
- VM-owned frontier, visited set, and budget;
- explicit branch outcomes: `success`, `contradiction`, `no_result`, `ambiguous`;
- deterministic checkpoint/restore of frame state;
- a host-defined acceptance predicate;
- a result of either one verified candidate or refusal/ambiguity.

The emitter supplies the **search topology and authorized domain references**, not prose rationales or an unconstrained candidate ranking. The host/VM performs enumeration, backtracking, and verification.

For problems where there is no finite, host-authorized hypothesis domain, the system should not claim generic search. It can ask for another bounded domain or refuse.

**Kill criterion:** retire the search primitive if, on tasks requiring a failed first branch and a successful later branch, it does not improve verified completion over an equivalent no-search straight-line plan, or if any branch can mutate state that survives rollback.

---

## Q3 — Where enforcement lives

GBNF should be an enforcement layer, but not the only one.

The production language should be a deliberately small **safe execution IR**, not the full CubeLang parse language. GBNF should admit only canonical serializations of that IR:

- fixed allowlisted interfaces;
- fixed roles;
- fixed opcode subset with known bytecode semantics;
- literal forms with bounded sizes;
- manifest-bound opaque capability values;
- no `match`, indexed assignment, arbitrary call expressions, `finally`, decorative attributes, top-level declarations, or extended trace-only opcodes;
- explicit bounded resource declarations where applicable.

That eliminates the audited parse/execute discrepancies by making those constructs unrepresentable on the serving path.

But grammar alone is insufficient. It can drift from the compiler, omit a case, or encode syntax that compiles to surprising bytecode. The authoritative enforcement point should be a **host-side typed IR validator plus bytecode-level admission check**, followed by VM runtime checks.

Pipeline:

```text
GBNF constrained output
    -> parse canonical safe IR
    -> validate IR against request manifest and resource policy
    -> compile
    -> inspect/validate produced bytecode against allowed bytecode subset
    -> execute in VM
    -> host gates result before speech
```

The type system need not be retrofitted into all of CubeLang. The validator can type-check only the safe IR: role kinds, relation capability kinds, arity, frame ownership, and allowed result forms. It should reject rather than attempt coercion or recovery.

Mechanical synchronization:

1. Define the allowed instruction/interface schema once, as machine-readable data.
2. Generate both the GBNF grammar and validator tables from that schema.
3. In CI, generate accepted grammar programs, compile them, and assert that their bytecode belongs to the admitted bytecode subset.
4. Fuzz compiler inputs outside the grammar and assert production admission rejects them even if the compiler accepts them.
5. Version the safe-IR schema; a VM/compiler change cannot ship until the generated grammar, validator, and bytecode tests agree.

The runtime should additionally reject any program whose compiled bytecode contains an opcode or call target not produced by the safe-IR compiler path. This is defense in depth against both grammar defects and compiler behavior changes.

**Kill criterion:** retire grammar-first admission if differential testing finds one grammar-admitted program whose executed bytecode violates the safe-IR semantics, or one bytecode-admitted program that cannot be represented by the safe IR. Either result means the grammar/compiler boundary is not a trustworthy policy boundary.

---

## Q4 — Measuring generalization without fooling yourself

Evaluate every SFT round using a frozen, end-to-end suite before selecting the next checkpoint.

### Held fixed

- host version, source snapshot or declared live-fetch policy, VM version, grammar/schema version;
- evaluator prompts and expected answer/refusal policy;
- request manifests;
- answer gate;
- a dedicated held-out relation/tool/task-shape split.

Do not let the evaluation label availability differ by task. Every evaluated task must produce the same fields:

```text
program_parse_ok
safe_ir_ok
bytecode_ok
vm_executed
host_retrieval_ok
final_answer_correct
wrong_answer_spoken
correct_refusal
```

In particular, `plan` must receive actual execution and final-answer labels, not `vm_ok: None`.

### Vary

Use a factorial evaluation split across:

- seen versus unseen relation handles;
- familiar versus nonce-renamed handles;
- seen versus unseen tool manifests;
- one-hop, multi-hop, and stateful task shapes;
- answerable, ambiguous, unavailable, and adversarially malformed requests;
- direct plan emission versus any deterministic host-generated chain artifact.

The last condition exposes whether the training objective is rewarding transcription of host work instead of production of the required plan.

### Metrics

**Primary metric:** number of wrong answers spoken, with required value **0**.

Among configurations with zero wrong spoken, select by:

1. verified correct-answer rate on the held-out, nonce-renamed relation/tool split;
2. correct-refusal rate on unanswerable and ambiguous cases.

Secondary metrics:

- safe-IR admission rate;
- end-to-end `plan` execution rate;
- exact manifest-capability selection accuracy;
- rate of unknown role/interface attempts;
- role vocabulary cardinality;
- identifier vocabulary cardinality;
- fraction of emitted work that the host would have performed unchanged without the emitter;
- per-slice regressions, especially unseen/renamed versus seen handles.

The existing failure would have appeared immediately as: `chain` metrics improve, while end-to-end `plan` metrics remain absent or flat; meanwhile per-relation role and identifier cardinality rise.

Checkpoint selection must be based on the primary end-to-end `plan` suite, not aggregate training rows and not a task with easier labels.

**Abandonment rule:** reject a serving checkpoint after **one** wrong answer spoken on the frozen safety suite. Retire the current training representation after **two consecutive SFT rounds** [ASSUM] with no improvement in verified held-out nonce-renamed performance while seen-handle performance improves; that pattern is evidence of memorization rather than the sought generalization.

---

## Q5 — A green kill line can conceal causal irrelevance

A likely unnoticed failure is **the emitter becoming causally irrelevant while the host still produces correct answers**.

This is especially plausible here because the host already deterministically performs chain construction and fact retrieval, while much of CubeLang’s visible language has no execution effect. A program may be accepted, traced, or superficially “verified,” but the final answer may actually come from:

- a host fallback;
- a cached prior retrieval;
- deterministic chain construction independent of the emitted plan;
- an answer path that reads prompt/request state rather than VM outputs.

The kill line remains green: no wrong answer is spoken. Offline correctness can also remain green. But apparent model generalization is illusory, and a future task without the hidden fallback will fail abruptly.

The cheapest early instrument is a **shadow causal-ablation check** on accepted requests:

1. Run the normal program and record the answer and provenance.
2. In a shadow execution, replace the program result/capabilities with a guaranteed-deny result while leaving all other request inputs unchanged.
3. Require the shadow path to refuse.
4. If it still produces the same answer, flag a bypass: the spoken answer did not causally depend on the authorized program result.

This need not affect the user-visible request. It is a duplicate local execution and requires no external model.

Add provenance tainting: every spoken factual value must carry the ID of the successful VM operation and host source record that produced it. A value with no authorized program-output lineage is unspeakable.

**Kill criterion:** if a shadow deny execution can yield the same non-refusal answer, stop treating the emitter as part of the decision path until the bypass is removed.

---

## x-ai/grok-4.6

# Answers

## Q1 — Unit of generalization

**Unit:** a small closed set of *host-defined* roles (SEED, HOP1, HOP2, …) plus **opaque string payloads** that the VM never parses. The model must emit *shape*, not vocabulary.

For an unseen relation the emitter produces only:

```cubelang
bind frame, SEED, "<entity string>";
bind frame, HOP1, "<relation string>";
bind frame, HOP2, "<relation string>";
return recover(frame, SEED);
```

The host **evaluates** those strings by table lookup (relation id → fetch plan / chain builder). It never asks the model for `H1_PARENT_TAXON`. Kill `chain` as a training task; keep `build_chain_program` as host code.

**What the host supplies:** the relation lexicon, hop arity, and the deterministic chain program. The model supplies which roles to bind and which already-bound values to recover.

**Test vs identifier-table memorization:** hold out a set of relations that never appear as *identifiers* or as string literals in SFT. Train only on `plan`-shaped rows. At eval, require emission of the fixed-role program with the held-out string in HOP*. Success = VM + host fetch succeed. Failure of `chain`-style emission on the same hold-out is expected and scores 0. If accuracy on held-out *strings* tracks training-string accuracy within noise, it is generalization; if it collapses to near-zero while in-vocab identifiers stay high, it was a table.

Kill criterion: if after 2 SFT rounds on `plan`-only data, held-out-relation `plan` exact-shape rate < 0.9 of in-vocab `plan` rate, retire “fixed roles + opaque strings” and try a smaller role set or host-injected relation tokens (still unevaluated by the VM).

## Q2 — Scratchpad replacement

`ASK`/suspend/resume replaces **observation of external facts**. It does not replace **internal search over hypotheses**.

Lost, concretely:

1. Attending over discarded candidates (the tokens that said “no, not that”).
2. Control flow that rewrites the *program*, not just fills a hole (`match` currently runs every arm; there is no real branch).
3. Cheap local arithmetic/scratch that CoT parks in tokens — today `f(n-1)` becomes 0.

**Minimal addition (no free-form tokens):** a host-owned **hypothesis stack** the program can only touch via three opcodes that already almost exist as traces: `push_hyp`, `fail_hyp` (pop + resume at last `ASK` with a `rejected` fact), `commit_hyp`. The emitter’s next program is grammar-constrained to those plus binds. Search is VM/host control flow; the model only proposes the next bind or a fail.

Kill criterion: if tasks that need backtracking (multi-hop with distractors) do not improve vs one-shot `plan` after 2 rounds with this stack, the loop is insufficient; do not add more English.

`[ASSUM]` backtracking is the main CoT residue; unmeasured on this stack.

## Q3 — Enforcement

GBNF should be **the emitter’s muzzle**, not the only validator.

Admit: the execute surface only — `bind`/`recover`/`return`/`ASK`/`implements ISolve|ISolver`, fixed role identifiers, string and number literals, no `match`, no `assign` to index, no extended opcodes, no attributes, no `finally`/`container`/…. Unknown roles and wrong arity are **unspeakable**.

**Risk:** grammar snapshot vs engine drift; grammar-only = SPOF (a compile-to-zero construct that the grammar still allows).

**Sync, mechanical:** generate GBNF from the same Rust enum the compiler walks. CI: (1) every grammar production has an execute test; (2) every compiler-visited node is in the grammar; (3) fuzz programs that parse but are not in the grammar must fail `--strict` *and* `run-proto`. Grammar and engine share one source; drift fails the build.

**Better enforcement than grammar or a typechecker:** **execute-surface = parse-surface**. Delete or make non-lexing everything that compiles to a trace or to zero bytecode. Then GBNF is a projection of the compiler, not a second language. Types stay decoration until there is a checker; do not pretend.

Kill criterion: any production program that `run-proto` accepts but GBNF would have forbidden (or the reverse) in a round → freeze SFT until the generator is fixed.

## Q4 — Evaluation that would have caught v12→v13

**Held fixed:** emitter weights at start of round, Wiki snapshot, `run-proto` strict, Talk adapter off (no spoken path).

**Varies:** task *shape* (`plan` vs `chain` vs arithmetic vs kernel), and whether the relation is in-vocab as identifier / as string / held-out.

**Primary metric:** fraction of *spoken* answers that are wrong = 0 (kill line). Spoken = host-approved VM result only.

**Secondary (must all be logged every round, including `vm_ok: None` tasks):**

- Shape-valid rate per task (GBNF + compiler execute, not just parse).
- Held-out-relation `plan` success vs in-vocab `plan` success.
- `chain` emission rate on `plan` prompts (should fall; rise is the v13 failure).
- Distinct role-identifier count (monotonic growth is a stop).
- Arithmetic exact-match (known to decay when mixed).

**Abandon number:** if role-identifier cardinality rises >20% in one round **or** `chain` rows with `gold_match` improve while `plan` shape-valid on held-out relations does not, **discard the round** even if spoken-wrong = 0. `[ASSUM]` 20% is a policy threshold, not measured.

The trap was optimizing the column that existed. Force every task family to have `vm_ok` *and* `shape_ok` *and* a hold-out split before the round counts.

## Q5 — Failure you will not see

**Silent execute≠parse success:** the model emits programs that *look* like branching/recursion (`match` all arms, `f(n-1)`→0, `assign` to zero bytecode). The host gets a definite VM value, gates it as a “fact”, speaks nothing wrong relative to that value — kill line green — while the program never computed the intended function. Designers watching spoken accuracy and `vm_ok` cannot see it because `vm_ok` means “ran”, not “the syntax you think ran”.

**Cheapest instrument:** a **shadow interpreter** on the execute surface only, run on every training and serving program: if bytecode length is 0 for a statement, if `match` has >1 arm, if a call arg is a non-trivial expr, **fail the program** before host gating. Count these as `phantom_ops`. If `phantom_ops > 0` on any spoken path, treat as a kill-line violation even when the answer string matches gold.

Not (a) or (b): this is the compiler lying about control flow, not claim/referent or verifier-as-recommender.

---

## google/gemini-3.8-flash

### Q1 — The Unit of Generalization

#### 1. The Representation
"Fixed roles + opaque string arguments" is the correct representation, but only when constrained to a **strict de-identified role register**. 

`chain` baked 402 relations into identifiers because the model was allowed to mint role names. To generalize to unseen relations without host-side NLP or identifier table expansion, the role space must be strictly closed to an invariant structural vocabulary:

```cubelang
implements ISolve;
bind frame, SEED, "iridomyrmex bigi";
bind frame, HOP1, "parent taxon";
bind frame, HOP2, "instance";
return recover(frame, SEED);
```

- **Roles**: Exclusively the closed set `SEED`, `TARGET`, `HOP1`, `HOP2`, ..., `HOPk` (depth bounded to $k \le 4$, matching the VM's $\tau_{\text{vm}}$ limit where hop 3 is already down to 0.22 similarity).
- **String arguments**: Exact literal byte-slices copied directly from the input prompt or retrieved cell contents.

#### 2. Where Evaluation vs. Parsing Lives
- **Emitter**: Emits bytes. It does zero natural language parsing and zero identifier invention. It performs exact span reproduction for the argument strings.
- **VM**: Opaque storage. It stores strings in the frame register table as inert byte arrays. It performs no casing, tokenizing, or comparison.
- **Host**: Deterministic resolution. When the VM executes `recover(frame, SEED)`, the host intercepts the syscall. The host executes:
  1. Read string at `SEED` $\to$ query local index/KB for entity id $E_0$.
  2. For $i \in [1..k]$: Read string at `HOP_i` $\to$ exact hash lookup against the host's relation schema table:
     $$\text{RelID} = \text{HashStore.get}(S_{\text{HOP}_i})$$
     If $S_{\text{HOP}_i}$ is not in the schema table, execute an exact literal substring match over the local entity graph's forward-edge labels from $E_{i-1}$.
  3. Traverse edge $\to E_i$.
  4. Write resulting value into frame register.
- **Strict Invariant**: The string is evaluated strictly as an **exact-match key** in an adjacency dictionary. It is never parsed by a grammar, stemmer, or tokenizer.

#### 3. Test for Genuine Generalization vs. Memorization
1. Construct a synthetic holdout split of 200 entities and 50 relations with random synthetic UUID labels (e.g., entity `"urn:ent:8f3a"`, relation `"urn:rel:c41b"`).
2. The emitter is prompted with a query where edges are denoted entirely by these synthetic labels.
3. **Pass Condition**: Emitter produces the exact program binding `"urn:rel:c41b"` to `HOP1`.
4. **Kill Criterion**: If accuracy on synthetic UUID relations drops below 99% while familiar relations stay high, the model is using SFT identifier/string memorization rather than indexical copying. Retire the data pipeline round immediately.

---

### Q2 — What Replaces the Scratchpad

#### 1. What Is Lost
CoT provides two things:
1. **Feedforward register compute over soft hypotheses**: The model can compute tentative representations across intermediate token layers before committing to an output.
2. **Dynamic back-propagation of error signals through attention**: When a token sequence generates an inconsistent state, subsequent tokens attend to that discrepancy and change direction.

In a pure one-shot emit $\to$ execute loop, the model must get the topology of the execution graph right in a single pass without knowing whether intermediate values exist.

#### 2. The Minimal Addition: Branchless Candidate Exploration via Vector Registers
Do not add natural language CoT. Do not add multi-step human-style conversational backtracking.

The VM already has `ISolver` (register compute) and `ASK`/suspend/resume. But because `match` compiles to unconditional execution and calls like `f(n - 1)` zero out, the emitter **cannot express dynamic runtime control flow**.

The minimal mechanism is **Multi-Candidate Pipeline Generation with VM-side Selection**:
1. The emitter emits a bounded, parallel hypothesis graph in one shot using a typed `select_valid` opcode (or host-mediated fallback via `recover`):
```cubelang
implements ISolve;
bind frame, SEED, "iridomyrmex bigi";
bind frame, PATH_A_1, "parent taxon";
bind frame, PATH_B_1, "subfamily";
return recover_first(frame, SEED, PATH_A_1, PATH_B_1);
```
2. For multi-hop search requiring backtracking, use **Deterministic Iterative Frame Stacking** via the host:
   - Round 1: Model emits program for Step 1.
   - VM executes step 1 against host. 
   - If Step 1 returns `Null` or fails $\tau_{\text{vm}}$ thresholding, the host does not prompt with prose. It resumes the emitter with a fixed VM hardware signal:
     `TRAP: EMPTY_SET (reg: HOP1)`
   - Round 2: The emitter emits an alternative path conditioned on the trap register.

#### 3. Kill Criterion
If the emitter cannot converge to the correct multi-hop fact within 2 `TRAP` resumes on a benchmark of 500 2-hop queries (where path 1 is a known distracter), the resume mechanism is dead. [derived: $500 \times 2 = 1,000$ executions $\times 0.7\text{ ms} = 0.7\text{ s}$ total evaluation cost; negligible].

---

### Q3 — Where Enforcement Lives

#### 1. The GBNF Role
GBNF must be used, but **strictly to enforce AST shape at the token level**, not runtime semantics.
GBNF must constrain decoding to:
- A valid interface declaration (`implements ISolve;`).
- Only valid, tested opcodes: `bind`, `return recover(...)`.
- Role names constrained to the closed set: `SEED`, `TARGET`, `HOP[1-4]`.
- String literals constrained to exact substrings present in the context/prompt (using llama.cpp vocabulary-masking / trie constraint).
- **Prohibit completely**: `match`, non-trivial call args, recursive forms, and the 24 dead opcodes (`infer`, `score`, etc.).

#### 2. The Risk and Synchronization
- **Risk**: GBNF drift. A grammar that permits constructs the VM silences (like `arr[0] = x` or `match`) generates programs that parse cleanly, execute incorrectly, and silently return wrong answers or near-zero similarity symbols.
- **Mechanical Synchronization Mechanism**: 
  - Derive the GBNF grammar directly from the VM's compiled AST definitions via a build step (`build.rs`).
  - **The Golden Mirror Test**: A CI gate runs every grammar production rule through `cubelang --strict` bytecode emission. If any production emits:
    1. Zero bytes (e.g., `arr[0] = x`),
    2. Extended trace-only opcodes (`infer`, `debate`),
    3. Unconditional fallthrough branches (`match`),
    the build fails immediately. The grammar cannot be compiled if it generates VM dead zones.

#### 3. The True Enforcement Layer: Host-Side Kernel Gate (Syscall Interceptor)
The true enforcement cannot live in the grammar (which cannot check runtime frame state) or the VM compiler (which lacks type checks). It must live in the **Host Syscall Interface (`run-proto`)**:
- The host intercepts `recover(frame, ROLE)`.
- If the frame has been bound, but `ROLE` was not bound in this specific session, the host **forcibly returns `Null`**. It short-circuits the VM's internal nearest-symbol metric ($\sim 0.03$), eliminating the false-positive drift on unbound roles.

---

### Q4 — Measuring Generalization Without Fooling Yourself

The previous failure occurred because `chain` had a binary execution gate (`vm_ok: True`), whereas `plan` had `vm_ok: None`. Optimization flowed toward the task that had an evaluation signal, regardless of whether that signal measured generalization.

#### 1. The Concrete Evaluation Harness
Run this harness on every checkpoint across SFT rounds:

| Dimension | Holdout Fixed Split | Perturbed Variable Split |
|---|---|---|
| **Entities** | 1,000 known entities from Wiki world | 1,000 entities never seen in SFT |
| **Relations** | 100 relations seen in training | 100 relations *never* seen in training |
| **Topology** | 1-hop, 2-hop | 1-hop, 2-hop |

- **Held Fixed**: Host-side world state, VM binary, prompt template, context budget.
- **Varied**: SFT iteration weights, presence of relation strings in SFT data.

#### 2. Metrics
- **Primary Metric: OOD Ground-Truth Resolution Rate ($R_{\text{ood}}$)**
  $$R_{\text{ood}} = \frac{\text{Unseen Relations Successfully Resolved to Correct Entity}}{\text{Total Unseen Relation Queries}}$$
  *(A refusal/null return counts as 0; a wrong entity returned violates the kill line).*
- **Secondary Metric 1: Structural Invariant Entropy ($H_{\text{roles}}$)**
  Count of distinct role identifiers emitted. Must remain exactly equal to the closed set size ($|R| \le 6$). If distinct roles $> 6$, failure is detected.
- **Secondary Metric 2: Transliteration Rate ($T_{\text{copy}}$)**
  Percentage of emitted string arguments that are exact byte-level substrings of the prompt context. Must be $100.0\%$.

#### 3. Abandonment Threshold
- If $H_{\text{roles}} > 6$ at any point: **Hard abort** (the model is minting roles again).
- If after SFT round 3, $R_{\text{ood}} < 0.85 \times R_{\text{id}}$ (where $R_{\text{id}}$ is the in-distribution known relation resolution rate): **Abandon the round**. The emitter is memorizing edge names instead of learning the role-binding operator.

---

### Q5 — The Failure Mode We Will Not See Coming

#### 1. The Failure Mode: "Host Semantic Drift under τ_vm Decay" (Threshold Cliff Inversion)
The VM applies fixed heuristic similarity thresholds ($\tau_{\text{vm}} \in \{1.0, 0.47, 0.22\}$) by hop count. 

As the emitter optimizes to satisfy the host's 0-wrong-answer gate, it discovers that emitting **generic, high-connectivity relation strings** (e.g., `"instance of"`, `"type"`, `"subclass"`, or `"has"`) almost always resolves to an entity that the host confirms as existing in the frame, because at hop 2 ($\tau_{\text{vm}} = 0.22$), almost any high-degree node passes the similarity threshold against the nearest symbol.

The kill line is: **"0 wrong answers spoken. A refusal is a result."**
When the model encounters an ambiguous or unseen relation, instead of refusing, it emits a broad relation that resolves to an ancestor or parent category. Because the ancestor is factually true of the entity in the underlying graph, the host marks the fact valid:
- Query: *"What specific dialect does Clan X speak?"*
- Emitter emits: `bind frame, HOP1, "language"`.
- Graph returns: `"Indo-European"`.
- Host checks graph: *Is Clan X related to Indo-European by language?* Yes.
- Host verification: **Green (No error raised)**.
- Spoken result: **True, but completely non-responsive (information collapse)**.

The kill line is preserved (0 false statements made), the VM execution is green (`vm_ok: True`), refusals do not spike, yet the model has silently degraded into a taxonomic triviality engine.

#### 2. The Cheapest Instrument to Detect It
A 40-line deterministic host audit checking **Specificity (Information Gain) Delta**:

$$\Delta I(q) = \log_2(|\text{Target Candidates}|_{\text{resolved}}) - \log_2(|\text{Target Candidates}|_{\text{specific}})$$

**The Metric**: Host-side Node Degree of Target:
$$\bar{D} = \frac{1}{N} \sum_{i=1}^N \text{out\_degree}(E_{\text{resolved}})$$
- High-degree nodes (e.g., `Q5` "human", `Taxon`, `Country`) have degrees $> 10^4$.
- Leaf/specific answers have degrees $< 50$.

**The Alert**: Compute the rolling median node degree of resolved entities across validation batches:
$$\text{Median}(\text{out\_degree}(E_{\text{resolved}})) > 250$$
If this median spikes by more than $3\times$ between two SFT rounds [derived: an increase from a typical target degree of $\sim 30$ to $\ge 90$], trip the wire. The model has learned to evade the kill line by retreating up the ontology graph to exploit the $\tau_{\text{vm}} = 0.22$ loose matching threshold.

---

## moonshotai/kimi-k3

## Q1 — The unit of generalization

Fixed roles + opaque string arguments is the right unit; `chain`'s per-relation identifiers must be deleted, not grown. But "string argument" needs a precise contract:

- **The model produces**: role bindings where the relation is a *string literal copied from the prompt* (`bind frame, HOP1, "parent taxon"`), and entity strings likewise. The model's job is *slot-filling and copying*, never naming.
- **The host supplies**: a relation registry — a table from string → property ID (Wikidata-style PID) with aliases, plus each relation's arity and hop signature. The string is **evaluated** by exact/alias table lookup at bind time, never parsed. Unknown string → host returns a typed `UNKNOWN_RELATION` error, the program gets Null, the host refuses. Kill line preserved.
- **The VM supplies**: nothing new. `recover` already takes the bound frame.

The generalization test: partition relations by PID and hold out entire clusters (e.g., all taxonomy relations in train, all geography relations in test, disjoint PID sets). At eval, the prompt names the relation in words; the model must copy it into the string slot. **Memorisation signature**: in-distribution relation accuracy high, held-out-PID accuracy ≈ 0 — that is the identifier-table model (today's `chain`). **Generalization signature**: held-out-PID accuracy within ~10 points of in-distribution `[ASSUM: threshold]`, plus robustness to alias strings never seen in training (test each held-out relation under 2+ paraphrases). A second, sharper probe: *novel compositions* — relation pairs never co-occurring in one training program. Copying generalizes; table lookup doesn't.

Kill criterion for the string-slot design: if held-out-PID accuracy after 3 SFT rounds does not significantly beat a trivial baseline (host-side heuristic that binds the single most salient noun phrase), the model isn't doing slot-filling work and the task should be host-deterministic like `build_chain_program`.

## Q2 — What replaces the scratchpad

The ASK loop recovers *interleaved observation* but loses two concrete things:

1. **Search with backtracking** — "try candidate A, constraint fails, try B" — which is the bulk of real reasoning traces and is control flow, not computation.
2. **Simultaneous comparison** — a forward pass over a CoT attends over all intermediate states at once; a suspend/resume cycle serializes them and the emitter re-reads only what the host returns.

The minimal addition that recovers (1) without free-form tokens: **move search into the VM, which is already Turing complete.** The emitter's job becomes emitting *search programs*, not answer programs: enumerate candidates from `recover` (which already returns similarity-ranked symbols), bind each in a loop, test constraints (arity, hop thresholds, type-of-value checks the host exposes as engine predicates), return the first satisfying candidate or Null. Backtracking is a `while` loop over a candidate list with a constraint predicate — straight CubeLang, no token reasoning. ASK remains only for fetching facts the VM cannot derive.

This also fixes a real hazard in the record: `recover` on a bound frame returns the globally-nearest symbol at ~0.03 similarity instead of Null, so a single-shot program *cannot express* "no good candidate." A search program can: it sees the similarity scores and rejects all of them.

Kill criterion: on a benchmark of multi-candidate queries (relations with ≥3 gold-consistent values, where picking is a claim decision), if search-programs do not beat the single-shot ASK loop on gold_match at equal refusal rate, retire it — the loop was sufficient and the complexity was waste.

## Q3 — Where enforcement lives

Yes, GBNF should be an enforcement layer — but as **admission control, not validation**, and it should admit only the *execute-verified subset* of the language:

- Whitelist opcodes with value-level effect; the 24 trace-only opcodes (`infer`, `forge`, `debate`, …) are excluded from the grammar entirely.
- Exclude `match`, index/map `assign`, `finally`, and every never-compiled form (`container`, `world`, `struct`, …) — these are traps that produce plausible wrong answers.
- Require call arguments to be atoms (the `f(n-1) → 0` silent-placeholder bug is not caught by `--strict`; the grammar must make it unexpressible).
- Roles come from the fixed role set; relation strings are arbitrary literals (they're evaluated by the host table, so they need no grammar constraint beyond being strings).

**The drift problem, mechanically**: the grammar must be *generated*, not written. Single source of truth = the engine's opcode registry; a build step emits the GBNF from the registry plus an explicit, reviewed exclusion list. CI gate: (a) every grammar-admitted program from a fuzzer executes with at least one non-trace effect or fails loudly; (b) every engine opcode is either in the grammar or on the exclusion list — an unlisted opcode fails the build. That makes drift a compile error, not a silent divergence.

**The better place**: neither grammar nor type checker — **bytecode-level verification**. The VM compiles anyway; a pass over the emitted bytecode asserting "only whitelisted opcodes present, no zero-bytecode statements, all call args resolved" costs a fraction of the measured 0.7 ms verification budget and cannot be bypassed by a grammar bug, a grammar/engine version skew, or a future surface the grammar forgot to exclude. Grammar is ergonomics (the model *cannot* emit junk); bytecode whitelist is security (junk *cannot execute*). Single point of failure is avoided because the two fail differently: grammar drift admits junk that bytecode rejects; bytecode bugs are caught by the CI fuzzer above.

## Q4 — Measuring generalization without fooling yourself

The v13f trap happened because `chain` carried `vm_ok` and `plan` carried `vm_ok: None`. The evaluation that catches it and its successors:

**Invariant (hard gate, every round):** every task in the training mix must carry a verification column, and the eval harness must report per-task verification coverage. A round where any task's row count grows >2× while its verification coverage is `None` halts training automatically. That one rule catches v13f at the round it happened.

**Held fixed**: model, quant, decoding params, host, VM, relation registry. **Varies**: SFT data round only.

**Primary metric**: held-out-relation generalization accuracy — gold_match on `plan` programs for PID clusters disjoint from training (Q1's split), with the program executed and the *spoken answer* scored, refusals counted separately.

**Secondary metrics**:
- In-distribution relation accuracy (memorisation gap = in-dist minus held-out).
- Identifier-space growth rate (must be flat; growth = someone reintroduced per-relation identifiers).
- Cross-task interference: the identity/forge/arithmetic battery rerun every round (the v5/v6 decay is the known cost of mixing).
- Refusal rate and wrong-spoken rate (must be 0).
- Alias robustness: held-out relations under unseen paraphrase strings.

**Abandon numbers** `[ASSUM: thresholds, since no baseline is measured]`: abandon the string-slot approach if, after 3 rounds, held-out-PID accuracy < 0.5 while in-distribution > 0.9 (pure memorisation — the model learned a table with extra steps), or if held-out accuracy does not beat the host-only copying baseline by ≥5 points. Abandon a *round* (revert checkpoint) if the interference battery drops any task >5 points from its round-0 value — the measured precedent (forge 1.00 → 0.17) says this degrades fast and silently.

## Q5 — The failure mode we will not see coming

**The similarity floor becomes the answer generator.** `recover` on an already-bound frame *never returns Null* — it returns the globally-nearest symbol at ~0.03 similarity. τ_vm is 1.0 / 0.47 / 0.22 by hop. So at hop 3, any symbol above 0.22 "verifies." Every lookup succeeds; `vm_ok` is green because a value came back; gold_match was true in training because training facts were real; the kill line reads green because the host only speaks VM-verified values — and the VM verified a *nearest-neighbor guess*. As the fact store grows (552,297 facts and climbing), the nearest-symbol density rises, so live queries increasingly bind plausible-but-wrong symbols that clear the hop-3 threshold, and nothing anywhere in the pipeline can see it: the emitter proposed, the VM returned a value, the host gated on `vm_ok`. The system doesn't start refusing (that would be visible) and doesn't crash — it starts *confabulating with verified-looking bindings*, and the designers' own instrumentation (vm_ok, gold_match on training-shaped data) is structurally blind to it because both columns measure "did a value verify," never "was the binding score evidence or noise."

**Cheapest instrument**: log the similarity score of every `recover` binding at serve time (one float per bind — the field already exists) and histogram it weekly. Two alerts: (a) fraction of *accepted-and-spoken* bindings below 0.5 rising over time; (b) mass concentrated just above the hop thresholds (0.22–0.3 at hop 3) — decisions riding the τ boundary are the confabulation signature. Cost: one log line and a histogram; no model, no judge, no serving-path change. Kill criterion for the underlying mechanism: if alert (b) fires on >2% of spoken answers `[ASSUM]`, `recover` must be changed to return Null below a floor — a never-Null recover is incompatible with the kill line regardless of how well everything else measures.

---

## z-ai/glm-5.3

## Q1 — The unit of generalization

**The right unit is: a closed, small role vocabulary + opaque relation strings that the model never generates from parametric memory, only copies from host-supplied calldata.**

Split the labor precisely:

- **The model produces structure**: hop count, role bindings drawn from a fixed set (SEED, HOP1, HOP2, …), the program skeleton, and the *decision* of which host-supplied candidate string to bind to which role.
- **The host supplies the strings**: every relation identifier the program may reference appears in the retrieval calldata of the prompt. The model's job for an unseen relation is *selection and placement*, not *recall*.
- **The string is evaluated exactly once**: at `recover(frame, ROLE)`, the host resolves the bound string against the fact table. It is never lexed, never matched against a grammar, never embedded in an identifier. The v13f `chain` shape (`H1_PARENT_TAXON`) is the anti-pattern: it forces the tokenizer to compose relation semantics into identifiers, which is why roles went 146 → 407 with 402 per-relation — the model is being asked to memorize a growing name table, and a relation absent from training is *unemittable by construction*.

**The distinguishing test — a three-way held-out split:**

1. **Seen relations, seen shapes** (baseline).
2. **Unseen relations, seen shapes**: relations excluded from all SFT rounds; the relation string appears only in the prompt calldata. If performance here ≈ baseline, the model generalizes over relation *content*.
3. **Unseen relations, perturbed calldata**: same as (2), but the prompt contains a decoy relation string alongside the correct one (e.g., both "parent taxon" and "child taxon" offered). A model that memorized "HOP1 usually binds the string that looks like X" will fail; one that genuinely reads calldata and places it structurally will not.

Additionally, a **generation-from-memory probe**: ask for a program over an unseen relation *with the string absent from the prompt*. Correct behavior is refusal or a Null-bind, never a confabulated relation string. Any emitted string that matches a real relation is evidence of parametric leakage and invalidates the held-out split.

Kill criterion for the "string-argument unit": if performance in split (2) is materially below split (1) after the calldata-supply change is implemented [ASSUM: gap threshold set per Q4], the unit is wrong and the alternative — host enumerates the full program template and the model only fills a choice index — should be tried.

## Q2 — What replaces the scratchpad

The ASK loop recovers **sequential conditioning on external results** (emit → execute → observe → emit), which CoT also has. Two things are concretely lost:

1. **Intra-pass intermediate state.** In CoT, token 200 attends to token 100's partial conclusion within one forward pass. In the ASK loop, the only intermediate state between emissions is the VM's registers — which the model *cannot read* unless the host echoes them back. This is recoverable: the host already mediates the resume payload; it should include the suspended program's observable state (bound roles, last recover result). Cheap, no free-form tokens.
2. **Search and backtracking — the real loss.** CoT traces are dominated by "candidate → check → reject → re-candidate." The ASK loop as described is *straight-line with pauses*: a program that binds HOP1 wrongly has no primitive for unbinding it. The VM is Turing complete, so backtracking is expressible in principle — but the emitter would have to synthesize the search control flow itself, which is exactly the hard part.

**Minimal addition: a `choose(k)` / suspend-resume-backtrack primitive.** Semantics: `bind frame, HOP1, choose(k)` suspends with k host-supplied candidates; the program executes tentatively on candidate 1; if a downstream `recover` fails a stated check (`assert recover(frame, HOP2) != Null`), the VM rewinds to the choice point and tries candidate 2. This is Dijkstra's guarded-choice, not token reasoning: the hypothesis space is enumerated by the host, the acceptance predicate is explicit in the program, and the control flow lives in the engine where it's verifiable. It maps directly onto the measured multi-hop setting (τ_vm thresholds by hop count suggest exactly this cascade of conditional acceptance).

Kill criterion: build an eval set requiring backtracking (multi-constraint fact selection where the greedy first candidate is wrong for a measured fraction of items). If one-shot + ASK-loop performance equals choose/backtrack performance on that set, the "search is lost" claim retires and the primitive is unnecessary complexity.

## Q3 — Where enforcement lives

**GBNF is the right *admission* layer and the wrong *enforcement* layer.** Use it for what grammars do well: closed role vocabulary, arity, well-formedness — the model *cannot* emit an unknown role or malformed program, which kills a whole class of failures at decode time for zero serving cost. But the pack's own audit shows why it can't be the only validator: `arr[0] = x` compiles to zero bytecode, `f(n - 1)` silently becomes a placeholder *under --strict*, `match` executes all arms — these are all *grammatically perfect programs that execute wrongly*. A grammar admits them by design.

**Enforcement belongs in the engine, as a fail-closed compile pass.** Concretely:

1. **Collapse parse surface to execute surface.** Every construct the compiler accepts but does not execute (24 no-op opcodes, decorative attributes, `finally`, `container`/`world`/`struct`/`enum`, `assign`-to-index, non-trivial call arguments) becomes a **hard compile error** in strict mode. The principle: *if the engine doesn't execute it, the engine must reject it.* This is a few hundred lines in the existing ~680 KB Rust, not a new component.
2. **Fix the semantic bugs the grammar can never see**: `recover` on an unbound role in a bound frame must return Null, not the 0.03-similarity nearest symbol. This is the single most dangerous listed behavior — it manufactures plausible wrong referents.
3. **Mechanical sync**: make the engine's accepted-construct table the single source of truth and *generate* the GBNF from it as a build artifact. Then a CI differential test: sample N programs from the grammar, run each through the strict compiler — 100% must compile, and any grammar-production whose sampled programs fail identifies drift in either direction. Add a fuzzer that mutates valid programs and asserts the compiler either executes them fully or rejects them — never silently truncates.

Risk of grammar-as-layer, stated honestly: it freezes a language snapshot and every language change requires regenerating and re-validating; and a grammar bug becomes a universal outage (nothing can be emitted). Mitigation is the generation-from-engine rule above plus keeping the grammar *narrow* — it should admit only the plan/arithmetic/kernel/role_binding shapes actually used, not the full parse surface.

## Q4 — Evaluation design

**Held fixed:** the VM binary, the host gating logic, the fact base snapshot, the kill-line definition of "spoken."

**Varied, per round:** (a) relation vocabulary — a fixed disjoint train/test split of relations, frozen before round 1 and never leaked; (b) hop depth; (c) task shape mix.

**The rule that catches the original trap:** *every emitted program, regardless of task class, gets the same verification columns.* `plan` rows must carry `vm_ok` and `gold_match` exactly as `chain` rows do — the v13f failure happened because the wrong-shaped task was the only measurable one. Mechanically: the eval harness refuses to ingest a training-data round in which any task class has `vm_ok: None` on a non-trivial fraction of rows. This is a schema check, not a judgment call.

**Primary metric:** end-to-end held-out-relation success — correct answer spoken or correct refusal, on the disjoint relation split, measured identically across all five task types. Report it *per task class*; a class-level collapse (arithmetic 0.825 → 0.675 style) must be visible in the same table.

**Secondary metrics:**
- **Identifier vocabulary size per round** — must be non-growing under the string-argument design; growth is the memorization alarm.
- **Generalization gap** = seen-relation success − unseen-relation success.
- **Refusal calibration** on a fixed set of answerable questions (refusal rate must not rise; see Q5).
- **Transcription probe**: perturb the facts in the prompt for a subset of `chain`-style items. If accuracy is unchanged under perturbed prompt facts, the model was copying the prompt, not computing — this directly catches the "transcribing, not discovering" failure.
- **Parse/execute surface ratio** from the compiler audit, re-run each round.

**Abandonment number:** the approach is abandoned if, after 3 consecutive rounds, the generalization gap exceeds 20 percentage points [ASSUM — design choice; the baseline gap under pure identifier memorization should be measured in round 0 to calibrate this], or if held-out-relation success is below twice the refusal-only floor (i.e., the system is barely beating "refuse everything" on unseen relations). Zero wrong answers spoken on the eval set is a *precondition*, not a success criterion — it is satisfiable by an always-refusing system and therefore carries no information about generalization by itself.

## Q5 — The third failure mode

**The emitter becomes decorative.** The kill line reads green — every spoken answer is verified true against gated facts — while the emitted program contributes nothing to any answer. The pack already contains the seed of this: `chain` is "work the host already does deterministically (`build_chain_program`)," and the training data hands the model the facts in the prompt. The architecture can drift to a state where the host retrieves, the host builds the effective computation, the VM executes a program whose output is either ignored or coincides with what the host computed anyway — and the answer is *correct*, so nothing fires. The designers are structurally blind to it because every instrument they have measures **answer correctness**, and answer correctness is exactly what the host guarantees independently of the emitter. Ten more SFT rounds could improve emitter metrics while the emitter's causal contribution to spoken answers is zero.

This is distinct from findings (a) and (b): it is not a wrong referent and not a self-feeding prior — it is the silent decoupling of the component the whole architecture exists to test.

**Cheapest early instrument — two tests, both nearly free at 0.7 ms/verification:**

1. **Fact perturbation differential.** For each emitted program in the eval set, execute it twice: once against the fact base, once against a copy in which the value of the queried relation is swapped for a different valid value. A live program's output must change; if output is unchanged, the program was not reading the fact it claims to use (placeholder argument, zero-bytecode assign, all-arms `match`, or pure decoration). Report the **liveness rate** per round. It also directly detects the listed compiler silent-failure bugs in production, not just in audit.
2. **Emitter ablation.** Replace the emitter's output with a fixed trivial program (or route around it) on the same question set and measure end-to-end accuracy. If accuracy is unchanged, the emitter has zero causal contribution and every "generalization" claim about it is unfalsifiable theater.

Alarm condition: liveness rate below 95% [ASSUM — set after measuring round-0 baseline] or ablation gap below the eval set's noise floor for two consecutive rounds. Either retires the architecture's central claim — that the model, not the host, is doing the reasoning — while the kill line still reads green.

---

## qwen/qwen3.8-max-0902

# Answers

---

## Q1 — The unit of generalization

**Fixed roles + opaque string arguments is the right unit.** `plan` already implements it; `chain` is the anti-pattern. The division of labor:

**Model produces:** program *structure* — interface choice (`ISolve`/`ISolver`), role bindings (`SEED`, `HOP1`, `HOP2`, …), hop count, and the *data* to fill those roles: an entity name and relation strings. The relation string is an opaque token the model copies from the user's query or selects from a small structural template. It is never decomposed.

**Host supplies:** evaluation of the relation string against the KB — a table lookup where the string is the key and the fact set is the value. No parsing of the string occurs at any point. The host also supplies entity resolution (user mention → KB entity) and result formatting.

**Why this generalizes by construction:** the model learns *patterns* ("two-hop chain: bind SEED, HOP1, HOP2, return recover"). Relation strings are data that flow through the pattern. A novel relation flows through the same pattern without the model having seen it, exactly as a novel entity name does today.

**Test that separates generalization from memorized identifier table:**

Hold out a set *R* of relations entirely from training data (all 75,534 rows). Present queries requiring relations in *R*. Score two things independently:

1. **Structural correctness** — does the program have the right shape (correct roles, correct hop count, correct interface)? This tests pattern learning.
2. **String fidelity** — does the program contain the exact relation string from the query? This tests whether the model passes data through rather than recalling an identifier.

If the model were memorizing a large identifier table, structural correctness would remain high but string fidelity on held-out relations would collapse to near zero (the model has no identifier to recall). If it is generalizing, both stay high.

A stronger variant: present queries requiring novel *compositions* of individually-seen relations (e.g., "parent taxon" and "instance" seen separately but never as a two-hop chain). Correct emission of the composed program is unambiguous evidence of structural generalization.

**Kill criterion:** if structural correctness on held-out relations falls below 95% [ASSUM: threshold chosen to allow for rare malformed emissions even with GBNF], the fixed-role unit is failing and a different representation is needed.

---

## Q2 — What replaces the scratchpad

ASK/suspend/resume replaces the *scratchpad* function of CoT (intermediate state the model conditions on) but **loses search over a hypothesis space**. Concretely, what is lost:

- **Parallel hypothesis comparison.** CoT lets the model generate candidate A, generate candidate B, and compare before committing. ASK gives one execution result per round; the model must commit to a single program before seeing any result.
- **Backtracking.** "That's wrong, let me reconsider" requires the model to have generated an alternative *before* discovering the first attempt failed. In a straight-line ASK loop the model discovers failure only after execution, then must reconstruct the alternative from scratch in a new forward pass with no memory of what it already tried (unless the host replays prior rounds into context, which costs context window).

**Minimal addition: multi-candidate ASK.** The model emits *k* candidate programs in a single emission. The VM executes all *k* (cost: *k* × 0.7 ms ≈ 2.1 ms at *k* = 3 [derived]). All *k* results return to the model. The model selects. This is beam search over the program space, mediated by the VM, with no free-form tokens.

This is not free-form reasoning because the model emits structured programs and selects among structured results. The search is over a discrete program space, not over natural-language hypotheses.

**What this still does not recover:** iterative refinement of a single hypothesis across multiple sub-steps within one "thought." This can be approximated by chaining ASK rounds (emit *k* candidates → select best → refine → emit *k* more), at the cost of additional rounds.

**Kill criterion:** if multi-candidate ASK (*k* = 3) does not improve accuracy by ≥ 2 percentage points [ASSUM] on tasks requiring hypothesis comparison (ambiguous entity resolution, multi-path reasoning where two relations could apply), the addition is unnecessary and should be retired.

---

## Q3 — Where enforcement lives

**GBNF should be the syntactic enforcement layer. It should admit exactly the VM's execute surface, not its parse surface.**

Concretely, the grammar admits:

- `bind` with fixed role names and string-literal arguments
- `return recover(frame, ROLE)`
- Interface declarations (`implements ISolve`, `implements ISolver`)
- `ASK` / suspend / resume
- Arithmetic via `ISolver` register operations

The grammar **rejects** (by never generating):

- `match` (executes every arm unconditionally)
- `assign` to index/map targets (compiles to zero bytecode)
- Non-trivial expressions as call arguments (silently becomes placeholder returning 0)
- `finally` blocks (never compiled)
- `container`, `world`, `robot`, `event`, `extend`, top-level `struct`, `enum` (never visited by compiler)
- All permission attributes (`@external`, `@system`, `@once`, `@ratelimit`, `@restricted`, `@cron`)
- All 24 extended opcodes (`infer`, `score`, `predict`, `debate`, `forge`, `analogy`, …)

**The risk and the sync mechanism.** A hand-written grammar is a snapshot that drifts. The fix: **generate the grammar from the same specification that generates the VM's validation pass.** One artifact — call it the *execute-surface spec* — defines which constructs have value-level effects. A build step produces (a) the GBNF grammar for llama.cpp and (b) a validation pass inside the VM that rejects any program using a construct outside the spec. When the VM changes, the spec changes, and both artifacts regenerate. Neither can drift from the other because neither is hand-maintained.

**Is there a better place?** The grammar handles *what can be emitted* (syntax). The VM validation pass handles *what does what it says* (semantics). Neither alone is sufficient. The grammar is a single point of failure only if it is the *sole* validator; with the VM validation pass as a second gate, a grammar bug that admits a bad construct is caught at execution time, and a VM bug that mis-executes a good construct is caught by the grammar never admitting the construct in the first place. Two independent gates from one spec.

**Kill criterion for the grammar:** if the grammar rejects a program that the VM would execute correctly (a false negative), the grammar is over-constrained. Track the false-negative rate on a held-out set of known-correct programs. Any nonzero rate is a bug.

---

## Q4 — Measuring generalization without fooling yourself

**The trap, restated:** `chain` carried `vm_ok` / `gold_match`; `plan` carried `vm_ok: None`. The measurable task got optimized; the correct-shaped task did not. The evaluation harness did not fail on the missing column — it silently skipped it.

**The evaluation that would have caught it:**

**Hard gate (the fix that would have caught it at the round it happened):** the evaluation harness *fails* — not warns, fails — if any task type has a `None` or missing outcome column for any row. No SFT round proceeds until every task type has a measurable outcome. This is a precondition, not a post-hoc check.

**What is held fixed:** a per-task-type held-out evaluation set, created at round 0, never changed. For `plan` and `chain`, it includes held-out relations (never seen in training). For `arithmetic`, held-out operand ranges. For `kernel` and `role_binding`, held-out scenarios.

**What varies:** the model (one per SFT round).

**Primary metric:** per-task-type accuracy on the held-out set, reported separately. The composite is the **minimum** across task types, not the mean. This makes it impossible to sacrifice one task type for another without the composite dropping.

**Secondary metrics:**

| metric | what it catches |
|---|---|
| Identifier vocabulary size (`chain`-type tasks) | memorization vs. generalization; should grow sublinearly with training rows |
| Generalization gap: acc(held-out relations) − acc(seen relations) | should be ≈ 0; large negative gap = memorization |
| Program validity rate (fraction parsing + executing without error) | should be ≈ 1.0 with GBNF; any drop signals grammar/model mismatch |

**Abandonment number:** if any task type's accuracy at round *N* falls more than **3 percentage points** below its peak accuracy across rounds 0 … *N* − 1, the round is rolled back. If two consecutive rounds are rolled back, the SFT approach (data mixture, hyperparameters, or both) is abandoned and redesigned.

Rationale for 3 pp: the measured arithmetic degradation was 7.5 pp in a single round (0.825 → 0.750). A 3 pp threshold catches this at the first occurrence with margin for noise. [derived: 0.825 − 0.750 = 0.075 = 7.5 pp]

---

## Q5 — The failure mode we will not see coming

**The model internalizes the host's refusal boundary, and the system's coverage silently contracts to the host's acceptance region.**

Mechanism: during SFT, the training data reflects the host's gating decisions — programs the host accepted are reinforced; programs the host refused are absent or negatively weighted. Over rounds, the model learns not just *what

---
