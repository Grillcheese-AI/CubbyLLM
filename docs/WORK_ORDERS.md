# Work orders

Specs for work that is argued for and not built. Each is written to be picked up cold by a
session that was not present for the argument: what to build, where, why it is shaped that way,
what would prove it wrong, and the measured numbers the decision rests on.

A work order is not a ticket. If the argument for it turns out to be wrong while you are
building it, **stop and say so** — the reasoning is the deliverable.

---

## The ordering principle

Two model competitions (2026-09-14) independently produced the same warning, and it decides what
comes first:

> **The emitter may be contributing nothing, and every instrument we own is structurally blind to
> it.** The kill line reads green because the *host* guarantees answer correctness independently
> of the emitter. Ten more SFT rounds could improve emitter metrics while its causal contribution
> to spoken answers is zero.

So: **instruments that could falsify the architecture's central claim come before anything that
assumes the claim is true.** Phase 0 is cheap, is mostly harness, and gates Phase 2 and 3. Do not
build a capability manifest for an emitter that turns out to be decorative.

Full arguments: `docs/research/2026-09-14-token-free-vm-competition.md` and
`docs/research/2026-09-14-neutral-scoring-competition.md`.

---

## Before you touch anything

**The kill line: 0 wrong answers spoken.** A refusal is a result; a wrong answer is a defect.
When a change could trade a refusal for a guess, that is not a trade.

**How to write a kill criterion** (owner decision 2026-09-15, after WO-1.3's fired wrongly). The
line above is not just a target, it is a constraint on how every other criterion is phrased:

> **A kill criterion may not treat a refusal and a wrong answer as the same event.** State two
> clauses, separately: the arm dies if **wrong answers appear**, and it dies if the VM-verified
> rate drops by **more than one percent**. A drop inside that band, entirely into refusals, is a
> cost to record — not a kill.

WO-1.3's original wording was "if the VM-verified answer rate drops". It fired at −1.0 of 600,
every lost question a refusal, on an arm that removed 395 memorized role identifiers and produced
0 wrong answers in 4,200 questions. A criterion that kills an arm for converting an answer into a
refusal **rewards guessing**, which is the one behaviour the kill line exists to forbid — and it
would have done so while reading like rigour, which is the dangerous part.

The general form, and it is the same lesson as every instrument failure in `docs/PATH.md` §6: an
acceptance rule that collapses two outcomes the system exists to distinguish will eventually
enforce the wrong one. Write the rate clause and the correctness clause separately, always.

**The model proposes, the host disposes, the VM decides.** An untrusted component gets one say.

**Standing constraints**

- No external LLM on the serving path, ever. External models build datasets and run competitions.
- The OpenRouter key lives in a gitignored `validation/.env`. Never in a log or a commit.
- **No local drive paths in committed files** — this one included. Say "the repo root", "a sibling
  checkout", "the datasets volume".
- Never kill python processes broadly. Filter by command line: a serve instance and sometimes a
  paid competition run are alive at once, and a blanket kill has already destroyed one run.
- Competitions: **always pass `--effort low`**. Measured across four rounds — rounds 1–2 were 4 of
  12 answering for $1.19; with the reasoning cap, 7 of 7 for $0.65. The token cap was never the
  problem.

**Running things**

- Tests: `python -m pytest -q standin/tests validation` from the repo root. 359 green at
  2026-09-14; a work order that lands red is not done.
- The live loop: `standin/serve_api.py --ask` (add `--news` for the date source), then
  `POST /ask {text, item, asker}`. `/loop/events` is the event log and is usually the fastest way
  to find out what actually happened.

**How to write a test here.** The docstring carries the incident: what was observed, when, and why
the rule is the rule. `test_a_grouped_number_is_the_same_number` is the model.

---

# Phase 0 — Instruments that could tell us we are wrong

Cheap, mostly harness, and they gate everything below.

## WO-0.1 — Emitter ablation

**Status:** not started. **Cost:** small. **Do this first.**

Replace the emitter's output with a fixed trivial program (or route around it) on a held-out
question set, and measure end-to-end accuracy against the unmodified path. Report the **ablation
gap**.

If accuracy is unchanged, the emitter has zero causal contribution and every generalization claim
about it is — in glm-5.3's words — *"unfalsifiable theater."* That is a result worth having before
another SFT round, not after.

**Where:** `standin/ask.py`'s emit call; a flag on `AskLoop` that substitutes a constant program.
A new `validation/exp_r22_emitter_ablation.py` driving the existing question sets.

**Acceptance:** a logged per-arm accuracy table (emitter / trivial-program / host-only where
reachable), the same question set and store snapshot for all arms, and a written statement of the
eval set's noise floor so the gap can be read against it.

**Alarm:** ablation gap inside the noise floor for two consecutive rounds.

## WO-0.2 — Liveness rate (fact-perturbation differential)

**Status: DONE 2026-09-14. Baseline 100%, 0 dead.** `validation/exp_r23_liveness.py`.
n=60 chains, 59 informative (1 refused both ways): **59 live, 0 DEAD.** Of the live ones,
56 changed their answer, 2 refused after the swap, 1 spoke only when perturbed. The emitted
programs genuinely read their inputs.

One measurement bug caught in the first trial run and fixed before the baseline: a chain whose
final-hop relation has no *other* stored value was never actually perturbed, so "the answer did
not change" said nothing — it was being scored DEAD. Those are now excluded as
`not_perturbable`. Counting an unperturbed run as dead would have made this metric cry wolf
from day one.

**Cost:** small (as estimated).

Execute each emitted program twice: once against the fact base, once against a copy where the
queried relation's value is swapped for a different valid value. **A live program's output must
change.** Report liveness rate per round.

This also detects the compiler's silent-failure bugs *in production* rather than only in audit — a
program whose output is invariant to its own inputs is exactly the signature of a placeholder
argument, a zero-bytecode assign, or an all-arms `match`.

**Alarm:** liveness below 95% `[ASSUM — set the real threshold after a round-0 baseline]`.

## WO-0.3 — Vocabulary growth as a build metric

**Status: DONE 2026-09-14.** `validation/role_vocab.py` (standalone comparison + build gate) and
`standin/data/build_gen3.py`'s `_role_vocab_stats`, which now records the count in every manifest.

**Cost:** trivial. **Free forever after.**

One integer per SFT build: the count of distinct role identifiers in the corpus. A generalizing
interface has a **flat** count across rounds.

Measured with the new tool — and the number is worse than the one that prompted this work order:

| corpus | distinct roles | per-relation | share |
|---|---|---|---|
| v12e | 151 | 143 | **95%** |
| v13e | 176 | 168 | 95% |
| v13f | 412 | 403 | **98%** |

v13f adds 261 roles over v12e and drops none; **260 of the 261 are `H<hop>_<RELATION>`**
(`H1_DATE_OF_BIRTH`, `H1_PLACE_OF_BIRTH`,
`H1_LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY`, …).

**The reading that changes what to do about it:** v12e was ALREADY 95% per-relation. The
146→407 jump everyone noticed is relation *coverage* growing, not a regression that got
introduced — the interface has never generalized. So this is not something to revert to a
previous round; the interface itself is the work. A relation absent from training has no role to
bind to, by construction.

**Where:** `role_vocab.py` fails a build on percentage growth over the previous round
(`--baseline OLD.jsonl --max-growth 0.10`, exit 1); the build prints the count and warns when the
vocabulary is majority per-relation.

## WO-0.4 — Similarity histogram at serve time

**Status: DONE 2026-09-14.** `cubbyllm/reasoning/simlog.py` (the sink, one line per spoken
answer, off unless `CUBBY_SIMLOG` is set, never raises) + `validation/sim_histogram.py` (the
weekly read). Called from `pipeline.answer`'s single `verified=True` site.

**Cost:** trivial (one log line).

Log the similarity of every accepted `recover` binding — the field already exists on `RunResult` —
and histogram it weekly. Two alerts: the fraction of spoken bindings below 0.5 rising over time,
and mass concentrated just above a τ boundary (0.22–0.3 at hop 3). Decisions riding the threshold
are the confabulation signature.

**Baseline (150 accepted bindings over 75 spoken answers, all 2-hop):** every binding in
[0.4746, 0.60); control role median 0.0254, max 0.0459. **Separation 10.3×.** Clean.

**One of this work order's own two alarms turned out to be wrong, and that matters more than the
baseline.** "Mass concentrated just above a τ boundary" fired at **100%** on this healthy run,
and "below 0.5" at 50.7%. Neither is a finding: τ_vm is *derived* from the expected cosine of a
k-element bundle (1.0 / 0.4736328125 / 0.22021484375 at 1/2/3 hops), so a correct binding lands
just above its τ **by construction**, and 0.5 is meaningless at hop 3 where τ is 0.22. Written as
specified, this instrument would have cried wolf every week until it was ignored.

What actually separates a real binding from cleanup noise is the gap to the **control role** —
bound to nothing, measuring the floor directly. That is now the alarm (`--min-separation`,
default 4×). The below-0.5 fraction is kept but alarms only on a *rise* between two windows
(`--compare`), which is what the work order's first alert was really asking for.

Context: `recover` on an unbound role in an already-bound frame returns the globally-nearest
symbol at ~0.03, never Null. Two models called a never-Null `recover` incompatible with the kill
line outright.

## WO-0.5 — Per-predicate volatility probe

**Status: BUILT 2026-09-14, baseline thin by design.** `validation/exp_r24_volatility.py`,
cumulative table at `validation/logs/volatility_table.json`.

**First run (50 sampled, 202 API calls):** 29 re-fetched, 23 agreed (7 of them the same claim
under a different label), **0 confirmed changed**, 2 disputed, 4 absent, 21 unresolved.
Disagreement rate **0.0% of decided** — but read that as *no staleness measured yet*, not as a
green light; 29 checks is nothing.

**Three measurement traps this hit, all fixed, all of which would have produced a confident wrong
number:**

1. **The WikiKG world cannot be re-verified at all.** Its triples carry `subject`/`relation`/
   `object` strings and **no QIDs**, so subjects resolve only by label — and its labels are
   decased/despaced (`HD189733b`, `Soft Bank Group Corp`, `Charles IV Of France`). Worse, its
   relation vocabulary is its own (`TIMELINE_EVENT`, `HAS_PART`, `IS_A` → "part"/"whole"/
   "instance"), not Wikidata property labels, so even a subject that *does* resolve never states
   the relation being checked. A 30-fact run over it **decided 0 of 30**. The probe therefore
   defaults to the **encyclopedia** facts, whose `entity` is a real name and whose `rel` is
   already a Wikidata property label. Reporting the KG world's 0-of-30 as "0% volatility" would
   have been exactly the false green this work order exists to prevent.
2. **Relabelling is not volatility.** Stored `Eric Harris` vs live `Eric David Harris` is one
   person under two names. Counted as agreement, tracked separately.
3. **Misresolution is not volatility either, and this one is unsolved.** `date of birth` of
   `Robert Edward Lee` came back stored 1807-01-19 → live 1942; `James Gordon Bennett`
   1795 → 1963. Those are different people with the same name, not Wikidata correcting a
   birthday — and on the first run that single case promoted `date of birth` to
   fetch-required at a fabricated 20%. Differences are now recorded as **`disputed`**, kept out
   of the rate and out of the fetch-required computation until confirmed. Only confirmed moves
   promote a predicate.

**The open blocker:** 21 of 50 subjects did not resolve even after passing `relations=` to
disambiguate (which itself cut unresolved from 30 to 21). Until stored facts carry the QID they
came from, this probe measures a thin, name-resolvable slice rather than the snapshot.

**Cost:** 50 fetches/day (this run: 202 API calls for 50 sampled, since resolution costs calls
too).

Sample 50 spoken facts per day, re-fetch them live, record agreement, and keep a **per-predicate
volatility table**.

The finding behind it: the gold labels, the verifier and the answer all descend from the same
snapshot, and **0 of 552,297 facts carry a time qualifier**. A fact true at snapshot time and
false today verifies, passes τ_vm and is spoken — the kill line reads green *by construction*,
because the only oracle it has is the stale thing. Every existing instrument is downstream of the
snapshot.

The volatility table **is the missing time qualifier, learned empirically for free**. Two uses:
high-volatility predicates become fetch-required (snapshot-only derivation on one becomes a
refusal), and the aggregate disagreement rate is the only available measure of snapshot decay.

*Retire to weekly* if after 30 days every predicate's disagreement is under 1%.

---

## WO-0.6 — The zone ablation

**Status: DONE 2026-09-15.** `validation/exp_r25_zone_ablation.py` (dynamic) and
`tests/test_guards.py::test_wired_modules_have_a_production_importer` (static).

The organizing frame is "brain zones, each with a specialty". A zone earns that name
three ways, and each is a test:

1. **Ablation** — removing it changes measured behaviour. Otherwise decorative.
2. **Channel** — it talks through a typed seam, not shared state. Otherwise it is not
   a separate zone at all.
3. **Reuse** — it serves more than one task. Otherwise it is memorization with a name,
   the WO-0.3 pathology one level up.

WO-0.1 applied test 1 to exactly one zone. Every other zone was unmeasured.

### Test 2, static: three zones declare WIRED and nothing calls them

An AST sweep for production importers (excluding `tests/` and `exp_*`/`test_*`) found:

| module | `__wiring__` | production importers |
|---|---|---:|
| `cubbyllm.reasoning.hippocampus` | `Wiring.WIRED` | **0** |
| `cubbyllm.reasoning.striatum` | `Wiring.WIRED` | **0** |
| `cubbyllm.reasoning.retriever` | `Wiring.WIRED` | **0** |

Each is measured in its own experiment (`exp_r13`, `exp_r14`, `exp_r15`,
`test_retriever_contract`) and mounted on no forward path.

**This is not a design error — `Wiring`'s own docstring says the marker is intent, and
names four clauses for "done": intent is WIRED, the body is real, something actually
calls it, and its test is green.** The error is that the suite enforced one of the four.
`test_every_module_declares_wiring` is *named after* "the 'built but never wired in'
trap" and checks only that the label exists. A guard that reads like rigour and tests
the weakest clause it names is the `docs/PATH.md` §6 pattern applied to CI.

The new guard closes the "actually calls it" clause and is **bidirectional**: a newly
unwired module fails, and a module in `KNOWN_UNWIRED` that becomes wired *also* fails,
so the ledger cannot go stale in either direction. Both directions were verified to
fire before it was committed.

*A note for reading the table below:* the system scores 597/600 with those three zones
entirely absent. Whatever they are worth, it is not visible on this question set.

### Test 1, dynamic: the ablation table

600 gen-3 held questions, seed 7, `emitter_v14e_nochain`. **The plan is emitted once
and reused across every arm that does not ablate the emitter** — re-emitting per arm
would put generation variance inside the comparison, which is the mistake §6.8 records.

| arm | seam removed | correct | refused | **WRONG** | contribution | verdict |
|---|---|---:|---:|---:|---:|---|
| `full` | — | 597 | 3 | 0 | — | baseline |
| `no_plan` | the emitter | 225 | 375 | 0 | +0.6200 | load-bearing (coverage) |
| `permissive` | the relation gate's judgement | 263 | 337 | 0 | +0.5567 | load-bearing (coverage) |
| `tau_zero` | τ_vm → 0.0 | 116 | 477 | **2** | +0.8017 | **LOAD-BEARING (kill line)** |
| `no_repairs` | `max_repairs` 1 → 0 | 596 | 4 | 0 | +0.0017 | inside the noise floor |
| `top_k_1` | retrieval breadth 3 → 1 | 597 | 3 | 0 | +0.0000 | inside the noise floor |
| `tau_floor` | τ_vm → 0.0332 (the measured floor) | 594 | 6 | 0 | +0.0050 | inside the noise floor |

Noise floor (Wilson half-width at n=600): **0.0064 = 3.9 questions.**

**τ_vm is the only zone whose removal produces a wrong answer.** Every other ablation
converts answers into refusals. That locates the kill line's mechanism precisely: it is
not distributed across the architecture, it is the acceptance threshold, and everything
else fails safe. The two wrong answers are the confabulation signature exactly as
described — *"When was Sugar Ray Robinson born?"* → `1921-05-03` against a gold of
`1920`: plausible, adjacent, confidently wrong.

**τ has a wide safe band and is not finely tuned.** At the measured noise floor (0.0332)
it costs 3 questions and 0 wrong; at 0.0 it costs 481 and 2 wrong. The cliff is
somewhere in between and nowhere near the operating point.

**Loosening τ produces more refusals, not more answers** — 477 against the baseline's 3,
almost all `retrieval_exhausted`. A wrong binding at hop 1 makes hop 2 find nothing, so
the chain refuses rather than continuing. **The multi-hop structure self-corrects**, and
only 2 of 481 got all the way through to a spoken wrong answer. That is a stronger
statement about the architecture than the baseline number is.

**The emitter is worth 372 questions here** (597 → 225), far more than WO-0.1's canonical
split suggested, because these are free-text gen-3 wordings: the grammar returns
`unparseable` on 350 of them. **The relation gate is worth 334**, and its contribution is
entirely coverage — with judgement removed it still produces 0 wrong.

**Three seams sit inside the noise floor**, and the honest reading is *on this question
set*: at 597/600 there is almost nothing left for a repair or a wider retrieval to
rescue. `no_repairs` did pick up one `vm_verify_failed` the baseline did not. They are
candidates for simplification **or** insurance that only pays on a harder set — and
distinguishing those needs the harder set, not more seeds.

**Alarm:** any arm other than `tau_zero` producing a wrong answer. That would mean a
zone is the only thing standing between the system and a confident error.

**Still open:** test 3 (reuse) has no instrument. A zone serving exactly one task is the
role-vocabulary pathology at architecture scale, and nothing measures it.

---

# Phase 1 — Bugs, regardless of which architecture wins

## WO-1.1 — CubeLang: the two constructs `--strict` does not catch

**Status: DONE 2026-09-14** (both, plus two more of the same family found on the way). Detail in
the sibling cubelang checkout's `docs/DRIFT.md`, annotated FIXED in place.

- **C11** is a hard compile error in every mode now, naming the form and the
  `Program::function`. The structural finding recorded alongside it: there is no
  expression-lowering pass and no temp-register allocator in that compiler at all —
  `compile_expr_operand` encodes *atoms*. So CubeLang's real surface is a **three-address
  register machine**, and that is the surface to keep rather than a gap to fill: it is the EVM
  analogy taken seriously, and a flat fixed-arity instruction grammar is far easier to constrain
  with GBNF (WO-2.2) than a recursive expression tree.
- **C8** `finally` now compiles onto the join point both paths converge on.
- **C3/A7** (`match` ran every arm) and **C4** (index-assign emitted zero bytecode) were the same
  bug class and are fixed too — see the notes below WO-1.2.
- **The VM bug underneath C3, which is the one that mattered most:** `op::COMPARE` forced both
  operands through `resolve_i64`, and `Value::as_i64` returns 0 for every `Value::Str` — so
  **every string comparison in CubeLang was unconditionally TRUE**, silently.
  `if (intent == "question")` took the then-branch for any intent. Fixing `match` arm selection
  *without* fixing this first would have traded "every arm runs" for "the first arm always wins",
  which is strictly worse because it looks correct.
- Compound assignment (`+=`) was compiled as plain assignment: `total = 10; total += 5` returned
  **5**. Now emits ADD/SUB/MUL/DIV.

343 cubelang tests green; new files `tests/match_arms.rs`, `tests/index_assign.rs`,
`tests/operand_errors.rs`, `tests/try_finally.rs`, `examples/match_router.cube`.

Six stale tests asserted the *old* broken behaviour ("match must be rejected in --strict",
"gsm8k.cube must still compile", …) and were flipped rather than deleted, each with a note saying
what it used to pin and why that is no longer true.

- **C11**: `compile_expr_operand`'s catch-all silently emits an `OP_NONE` placeholder for
  `BinOp`, `UnaryOp`, `StructLit`, `MapLit`, `Lambda`, and nested calls used as arguments.
  `double_it(x - y)` returns 0 instead of 4 and `check` reports "ok". **The only silent-garbage
  path a strict-clean emitted program can still hit.** Must become a hard compile error in every
  mode, naming the form and pointing at the workaround.
- **C8**: `finally` blocks are never compiled, in any mode; the source comment admits it. Compile
  them, or make a `finally` block a hard error. Silently dropping is the one unacceptable outcome.

Note from the token-free competition: a generated grammar restricted to literals and bare
variables as arguments **fixes C11 on the emitter path without touching the compiler**. It does
not fix it for hand-written programs, which is why this stays a Phase 1 item.

**Acceptance:** a regression test per fix following `tests/` conventions; `cargo test` green;
`docs/DRIFT.md` entries annotated FIXED rather than deleted.

## WO-1.2 — CubeLang: `recover` scoped to the frame's own bound symbols

**Status: DONE 2026-09-14. τ_vm needs no recalibration — accepted bindings are bit-identical, and
the noise floor dropped.**

The candidate pool *is* the semantics here: the cleanup always returns its nearest candidate and
never returns "nothing", so what it is allowed to choose from decides what a wrong answer looks
like. It is now the frame's own bound fillers (`VM.vsa_frames`, register-keyed, taken and restored
alongside `registers` in `exec_intra_call` so a callee cannot leak fillers into a caller's frame).
`op::UNBIND` was building its own `name_table.values()` pool inline and now routes through the
same function, so the policy has one definition instead of two that could drift.

**Direct measurement** (`cubelang tests/recover_scope.rs`, a frame holding SUBJECT/OBJECT/VERB
with a second frame's fillers also in the VM's name table):

| | before | after |
|---|---|---|
| genuine `SUBJECT` | `"cat"` @ **0.27002** | `"cat"` @ **0.27002** |
| absent `LOCATION` | `"umbrella"` @ 0.02148 | `"cat"` @ 0.01221 |

`"umbrella"` is a filler belonging to a *different frame* — the confabulation mechanism, observed
directly rather than argued about.

**Through the CubbyLLM harness** (75 spoken answers, 150 accepted bindings, the WO-0.4 sink):

| | before | after |
|---|---|---|
| accepted-binding distribution | 76 in [0.40,0.50), 74 in [0.50,0.60) | **identical** |
| weakest accepted | 0.4746 | 0.4746 |
| below-0.5 fraction | 50.7% | 50.7% (+0.0%) |
| control role median / max | 0.0254 / 0.0459 | **0.0112 / 0.0332** |
| **separation** | **10.3×** | **14.3×** |

Liveness re-measured after the change: still 100%, 0 dead.

So a genuine recovery is unchanged to the bit — same unbind, same true filler, and the true filler
is always in its own frame's pool — while the noise floor fell by more than half and separation
improved 39%. Nothing downstream needs re-tuning.

**One deliberate limit:** a register holding a hypervector with *no recorded bindings* still falls
back to the global pool. That covers hand-assembled bytecode, where scoping to an empty pool would
silently turn a working recovery into `None`. Narrowing behaviour is worth shipping; a silent
regression to Null is not. `tests/asm_vsa.rs` b3 (never-bound frame → Null) still passes, along
with the other 346.

**Still open, and unchanged by this:** `recover` on an absent role is still not `None` — it is a
low-similarity symbol from the right structure instead of a low-similarity symbol from anywhere.
That is DRIFT.md §B11a, and it is the separate decision about whether the VM should ever return
Null here.

---

*Original specification:*

`recover(reg, role)` performs UNBIND + cosine cleanup against **every symbol the VM knows**, not
just the ones bound in that frame. Restrict it to the frame's own bindings — the semantically
correct fix, confirmed independently by the competition's Q1 Null control and Q5 confabulation
mechanism.

**This changes numbers CubbyLLM is calibrated against** (τ_vm = 1.0 / 0.47 / 0.22). Measure
before/after — similarity for a genuine recovery and for an unbound role — against the existing
harness, and report both before it ships. Do not break `tests/asm_vsa.rs`'s never-bound-frame
control.

## WO-1.3 — Drop `chain` and the decorative attributes from the SFT mix

**Status: AUDITED + filtered corpus built 2026-09-14. The retrain is the open part.**
`validation/sft_mix_audit.py`; filtered corpus at
`standin/data/out/emitter_sft_v14_nochain.jsonl` (47,005 rows, 28,529 dropped).

The three claims this work order rests on, checked before anything was dropped:

| claim | stated | measured | verdict |
|---|---|---|---|
| 1. `chain` share | 28,529 of 75,534 | **28,529 of 75,534** | exact |
| 2. `chain` is the sole source of the per-relation roles | 402 | **348 of 352 (99%) chain-only** | holds in substance |
| 3. decorative-attribute rows | 11,731 | **9,009** | overstated by 2,722 |

Claim 2's four exceptions are `H3_NAME`…`H6_NAME` from `kernel` — positional slots whose suffix
collides with "name" as a Wikidata property alias, not genuine per-relation minting.

**Confirmed: dropping `chain` does flatten the vocabulary.** The filtered corpus has **17 distinct
roles, down from 412** (−95.9%), and the remainder are the generalizing shapes — `SEED`, `HOP1`,
`HOP2`, `ASK`, `ACTION`, `AGENT`. So the mechanism the work order proposed is real; what is still
untested is the *prediction* (better plan accuracy and cross-task retention, no loss of
VM-verified answers), which needs the retrain.

**A finding the audit turned up that no claim covered: 51 roles are not per-relation but
per-relation-*per-entity*.** The entity name is baked into the role identifier:

    H1_GENRE_OF_JOAN_RIVERS_A_PIECE
    H1_INSTANCE_OF_RADOALD
    H1_INSTANCE_OF_NORTH_NORFOLK_COAST_SITE
    H1_CAST_MEMBER_FOR_CITY
    H1_ADMINISTRATIVE_TERRITORIAL_ENTITY_OF_HISTORY

That is the `' of '`-split ambiguity (§ the semantic hole, below) leaking out of fact parsing and
into the role vocabulary itself. A role like `H1_INSTANCE_OF_RADOALD` can never be reused — it is
minted for exactly one question — so these are pure memorization capacity with no possible
transfer. They vanish with `chain`, but the *generator* that produced them is still in the build
and will mint more the moment `chain`-shaped rows come back.

**The verification asymmetry — and the correction that matters.** The raw counts do look exactly
as this work order predicted:

    plan    37,588 rows   vm_ok = None   (every single one)
    chain   28,529 rows   vm_ok = True   (every single one)

The first reading of that — "the plan task carries no verification signal, so a retrain without
`chain` trains on nothing verified" — is **wrong**, and it was stated here before being checked.
`build_gen3.py` emits the plan record and the chain record from **one** `learn_and_answer` call
and `continue`s on any verdict other than `certified`, so a plan row exists *only because* that
walk was gate-verified and its answer equalled gold. The `None` is a **missing label, not missing
verification**.

Measured, by pairing each plan row with a chain twin on (question, gold):

| corpus | plan rows | with a certified chain twin |
|---|---|---|
| v13e | 19,404 | **18,190 (94%)** |
| v13f | 37,588 | 27,358 (73%) |

v13e's 1,214 exceptions are the gen-2 rows (`game`, `cot_harvest_v3cf`, `cot_harvest_r7`) — no
gold, no provenance, certified by gen 2's own process instead. v13f's larger gap is
`gen3_profile` (9,016 rows from a different builder, no chain twin) — a third reason to prefer
the v13e-derived arm.

**Fixed, in both places:** `build_gen3.py` now writes `vm_ok=True, vm_result=res.answer,
gold_match=True` on the plan record, and `sft_mix_audit.py --write-filtered` backfills the label
onto existing corpora from the chain twin before dropping it (18,190 rows in the v13e arm), with
`verified_via` recording where the evidence came from. Rows with no twin keep `None` — "not
certified by this path" is not "failed", and nothing here has grounds to write `False`.

**This changes no training row**: the notebook admits `vm_ok in (True, None)`, and the filtered
corpus passes 28,821 → 28,821 before and after. So it does not block the retrain and does not
confound it. What it buys is a corpus that states what is true of itself, so the next audit
cannot draw the same wrong conclusion this one did.

**Still genuinely open:** a verification signal that *discriminates* — one where some plan rows
fail and are dropped. There are none to drop today, because every gen-3 plan row is certified by
construction. That only becomes a real lever when plans come from somewhere other than a
pre-certified walk (an emitted plan, which is what `emitted_plan` verifies in exp_r22).

**Cost:** a rebuild (the filter is done; the retrain is not).

### Colab readiness

**Use `notebooks/standin_v14e_nochain_sft.ipynb`** (generated from `notebooks/_build_v14e.py`).
Both arms run through it — `STANDIN_VERSION=v14e_nochain` for the treatment,
`STANDIN_VERSION=v13e` for the control. `validation/preflight_colab.py` replays its cell-1/cell-2
data contract locally in a second (manifest assert, the vm_ok/gold_match filter, the train/val
split, the fields `to_messages` reads); **v14e_nochain and v13e both pass**.

**The training recipe is v12e's, not the gen-3 notebook's — this was the important correction.**
A first draft of the dedicated notebook copied `standin_gen3_masked_sft.ipynb`'s LoRA block. Nick
hit it in Colab and replaced it with the settings from `standin/models/train_card.json`, which is
the card of `emitter_v12e` — the model holding the 531/600 reference:

    lora: {r: 64, alpha: 128, targets: "q,k,v,out_proj,in_proj,w1,w2,w3"}
    sft:  {epochs: 2, batch: 32, lr: 1e-4, sched: cosine, warmup_ratio: 0.05,
           optim: adamw_8bit, max_seq: 4096}

Two separate faults in what had been copied:

1. `target_modules` named `o_proj`, `gate_proj`, `up_proj`, `down_proj`. **LFM2.5 has none of
   them** — its attention is `in_proj`/`out_proj`, its MLP is `w1`/`w2`/`w3`. No `emitter_v13e` or
   `v13f` GGUF exists in `standin/models/`, so that configuration plausibly never ran to
   completion.
2. r=32 / alpha=32 / lr=2e-4 against a baseline trained at r=64 / alpha=128 / lr=1e-4 makes the
   comparison measure the *recipe* as well as the data — the one thing this work order is trying
   to isolate.

`STANDIN_ARM` therefore defaults to **`full`**, not `masked`: v12e was trained full, so the chain
of comparison has to be. Whichever arm is used, the control must use the same one.

**A trap fixed in passing.** The v12e-era train-card write reads `m['gen1_sha256']` and
`m['r7_sha256']`, keys that exist **only** in v12e's manifest. On any other corpus that is a
`KeyError` raised at `json.dump` — *after* training finished and the adapter was saved, so the run
appears to succeed and silently loses its provenance. The dedicated notebook uses `.get()`
throughout and reads its hyperparameters back off `cfg` rather than retyping them (the v12e card
records `warmup: 20` while the cell now uses `warmup_ratio=0.05` — exactly that stale-copy
failure). `standin_gen3_masked_sft.ipynb` still carries the same shape of bug at
`m['output_sha256']`; left unfixed deliberately, since nothing runs through it any more.

Two arms are built, because the comparison matters:

| arm | derived from | rows | train after repeat | why |
|---|---|---|---|---|
| `v14e_nochain` | **v13e** | 28,821 | 24,817 | the **clean** comparison — paired directly against a `v13e` control run, so dropping `chain` is the only variable |
| `v14_nochain` | v13f | 47,005 | 38,776 | the current-best mix, but confounds two changes (v13f added hdc + the retrieval shape *and* this drops chain) |

**Prefer `v14e_nochain`, and train `v13e` as its control.**

**531/600 is not "the v13e bar"** — an earlier draft of this document said so and it is wrong. That
number was produced by **`emitter_v12e.Q4_K_M.gguf`**, the gen-2 model, scored on the gen-3 held
split (`validation/logs/exp_r17_gen3_heldout_gen2_lev8.log`, first line: `emitter
emitter_v12e.Q4_K_M.gguf`). It is the *incumbent's* score — the bar v13e was meant to beat.

And `standin/models/` holds no `emitter_v13e` or `v13f` GGUF: **the with-`chain` corpus has never
been trained.** So scoring `v14e_nochain` against 531/600 alone would confound dropping `chain`
with adding all of gen 3's free-text data on top of gen 2. The prediction needs **two runs**
(control `v13e`, treatment `v14e_nochain`, same arm, same cell), not one.

Verified beyond the data contract: stripping `@external`/`@system`/`@once` is **syntactically
safe** — 120 sampled programs compile 120/120 both before and after, so the retrain is not being
taught broken source.

**Remaining manual steps:**

1. ~~Copy both files for the chosen arm into the Drive `standin/` folder.~~ **Done 2026-09-14** —
   `emitter_sft_v14e_nochain.{jsonl,manifest.json}` and `emitter_sft_v14_nochain.{…}` are in the
   synced Drive `cubbyllm/standin/` folder (what Colab mounts as
   `/content/drive/MyDrive/cubbyllm/standin`), each verified to hash-match its manifest's
   `output_sha256`. `emitter_sft_v13e.*` (the control) was already there. The notebook is also in
   the Drive `Colab Notebooks/` folder.
2. **Push the repo.** Cell 1 does `git reset --hard origin/master`, so an unpushed tree means
   Colab trains against old code. Nothing in the current uncommitted work is needed at train time
   (`identity.py` is untouched), but the post-training exp_r17 evaluation would run stale.
3. Run twice, changing only `STANDIN_VERSION`: `v13e` (control) then `v14e_nochain` (treatment).
   Leave `STANDIN_ARM` at `full` for both, and do not edit the training cell between runs.
   The adapter/merged/GGUF land in version- and arm-named folders, so the two never collide.

### RESULT 2026-09-15 — both arms trained and scored

Scored on the gen-3 **held** split (`exp_r17`), which is fixed by seed and independent of the
corpus, so unlike the notebook's val sample it does not move when the treatment moves. 600
questions, three seeds, same store (563,062 facts). Logs: `validation/logs/exp_r17_gen3_heldout_cmp_*`.

| arm | s7 | s8 | s9 | mean | WRONG |
|---|---:|---:|---:|---:|---:|
| `emitter_v12e` — the incumbent | 531 | — | — | 531 | 0 |
| `emitter_v13e` — the control | 599 | 600 | 600 | **599.7** | 0 |
| `emitter_v14e_nochain` — the treatment | 597 | 599 | 600 | **598.7** | 0 |

**v12e reproduces 531/600 exactly** — the documented bar, to the item, which is what makes the
other two rows trustworthy. Both new arms beat the incumbent by ~11 points: the with-`chain`
corpus had never been trained before this cycle, so this is also the first measurement of what
gen 3 is worth. **0 wrong answers in all seven runs — 4,200 questions.**

**The arms differ by three questions, not by a rate.** Across all three seeds exactly three
distinct questions ever fail, and every one carries a mangled entity name:

    When was the sibling of France Killy Sister born?      both arms, seed 7
    In which district is Seaside-Seattle Seattle located?   v14e only, seeds 7 and 8
    What award did Seven Against Thebes Play receive?       v14e only, seed 7

These are the decased-label and `' of '`-split artifacts (§ the semantic hole) leaking out of fact
parsing into the question text. The first is failed by all three arms, v12e included. The two that
separate the arms are `retrieval_exhausted` — refusals — and they are deterministic: the same
question fails whenever it is sampled.

**Adopted.** Under the amended criterion below the arm survives: no wrong answers, a drop far
inside 1%, and the role vocabulary down from 412 identifiers to 17. That is the WO-0.3 ceiling
removed for the price of two malformed-entity refusals.

**What this costs:** v13e scores 600/600 on two of three seeds. The held split is at its ceiling
and can no longer discriminate between these arms. WO-2.5 is now the next test that can, which
moves it from "a Phase 2 item" to "the gating measurement".

### The kill criterion, amended

Owner decision 2026-09-15. The original wording — *if VM-verified answer rate drops* — fired on
this arm at −1.0 of 600, and every lost question was a refusal.

> **A kill criterion may not treat a refusal and a wrong answer as the same event.** An arm dies
> if **wrong answers appear**, or if the VM-verified rate drops by **more than one percent**.
> A drop inside that band, entirely into refusals, is a cost to record — not a kill.

The reason is the first line of this document: a refusal is a result; a wrong answer is a defect.
A criterion that kills an arm for converting an answer into a refusal rewards guessing, which is
the one behaviour the kill line exists to forbid. **Any kill criterion written from here on states
its wrong-answer clause and its rate clause separately**, and the general rule is recorded under
"Before you touch anything".

---

*Original specification:*

`chain` is 28,529 of 75,534 rows, is the sole source of the 402 per-relation identifiers, and is
work the host already does deterministically (`build_chain_program`) — in the training data the
facts are handed to the model in the prompt, so it is transcribing, not discovering.

Separately, 11,731 rows teach the model to emit `@external` / `@system` / `@once`, every one of
which parses and is never read by compiler or VM. A capacity tax on a 2.6B, free to remove.

**Falsifiable prediction:** dropping both improves plan accuracy and cross-task retention with no
loss of VM-verified answers. ~~**Kill criterion:** if VM-verified answer rate drops, `chain` was
doing something the audit did not see.~~ **AMENDED 2026-09-15 — see "the kill criterion, amended"
above. This wording fired on a drop of 1.0 in 600 that was entirely refusals, and a criterion that
cannot tell a refusal from a wrong answer is the one thing this project must never ship.**

Also wire a verification signal into `plan` rows — they carry `vm_ok: None` today while `chain`
carries `vm_ok: True`, which is very likely *why* the wrong-shaped task is the one that grew.
A plan is verifiable: walk it and check it reaches gold, which `emitted_plan` already does.

---

# Phase 2 — The generalization spine (gated on Phase 0)

## WO-2.1 — The per-request capability manifest

The host supplies, per request, the admissible relation/entity label strings plus direction,
arity, and functional-or-multi-valued metadata. The string is evaluated **exactly once**, as a key
into the manifest index; a miss is a hard error, never a nearest-neighbour fallback. That is
"evaluated, not parsed" made mechanical.

**Status: BUILT 2026-09-15** (`cubbyllm/reasoning/manifest.py`), **and the prompt-only form does
not work** (`validation/exp_r27_manifest.py`). The module is a drop-in for the `known=` seam
WO-0.6 proved is clean, so nothing was rewired to adopt it; it declares `STANDALONE` rather than
`WIRED` because the new CI guard now checks that claim.

### RESULT — supplying the information is not enough

Same relabelled world as WO-2.5, same questions. The host supplies the admissible relations two
ways at once: in the emitter's system prompt, and as the `known=` gate (exact lookup, no fuzzy
tier). 200 questions, both arms.

| model | arm | correct | wrong | bound right token | avg manifest | chance |
|---|---|---:|---:|---:|---:|---:|
| v14e | relabelled | 56 | 0 | 107 | — | — |
| v14e | + manifest | 53 | 0 | 102 | 4.1 | 0.245 |
| v14e | + manifest + 3 decoys | 51 | 0 | 106 | 7.1 | 0.141 |
| v13e | relabelled | 53 | 0 | 102 | — | — |
| v13e | + manifest | 54 | 0 | 101 | 4.1 | 0.245 |
| v13e | + manifest + 3 decoys | 52 | 0 | 100 | 7.1 | 0.141 |

**Six arms within five questions of each other.** The manifest recovers nothing. And the
right-token rate sits at ~50% in every arm, unmoved by the manifest and unmoved by decoys — the
emitter is reading the *question*, not the admissible set. It was never trained on a manifest
format, so this is a zero-shot probe, and zero-shot it simply ignores the block.

**The decoy control is therefore inconclusive, and that is the honest reading.** It was built to
measure whether selection beats the 1/(k+1) baseline. Nothing is selecting, so there is nothing
to measure. Decoys cost 1 question out of 200.

### But the failure mode improved sharply, and that is not nothing

| arm | `unknown_relation` | `retrieval_exhausted` |
|---|---:|---:|
| v14e relabelled | 29 | 56 |
| v14e + manifest | **89** | **1** |
| v13e relabelled | 34 | 25 |
| v13e + manifest | **69** | **3** |

Without the manifest the system takes a bad relation, walks, and exhausts retrieval — a late,
expensive, vague failure. With it, the gate refuses at once and names the reason. **Evaluated,
not parsed, is doing exactly what it was specified to do**; it is the emitter that is not
participating. Flat accuracy, far better diagnostics, and a cheaper failure path.

### What this settles

**WO-2.2 is not an optimization, it is the mechanism.** A prompt asks; a grammar enforces. If the
model will not use information it is handed, the only remaining move is to make the wrong relation
*undecodable* rather than merely discouraged — which is precisely what WO-2.2 specifies and why it
draws its literals from the manifest.

The alternative — train the emitter with manifests in the prompt — is available and weaker: it
buys compliance by hope where a grammar buys it by construction.

**0 wrong answers across all 1,200 questions**, manifest and decoys included.

## WO-2.2 — GBNF generated from the engine registry

Admit the **execute surface only** — excluding all 24 trace-only opcodes, `match`, indexed assign,
`finally`, the top-level forms the compiler never visits, and every permission attribute
(*"admitting them teaches the model to write security theatre"*). Arguments restricted to literals
and bare variables. Relation and entity literals drawn from the per-request manifest, so an
unknown relation is **undecodable** rather than rare.

**The grammar is a build artifact generated from the engine's own registry. A hand-edited grammar
fails the build.** The risk is drift, not decoding: a grammar that is the only validator keeps
admitting a construct after the engine stops honouring it, and fails in the safe-looking
direction.

## WO-2.3 — Witness test per grammar token

For every opcode the grammar admits, a program whose **observable return value changes when that
opcode is removed**. Any token with no witness is demoted out of the grammar automatically.

This is exactly the test whose absence let 24 trace-only opcodes into the language. 0.7 ms each.

## WO-2.4 — Bytecode verifier at the loader boundary

The best single engineering idea from the competition, reached independently by two models.
JVM-style, checking the compiled artifact rather than the source — the only place the
parse/execute gap is visible *as data*:

> **Reject any program where a source statement maps to zero emitted instructions.**

One check subsumes most of the drift audit's §C — indexed assign, `finally`, bare expression
statements, `throw`, trace-only opcodes — **including the ones nobody has found yet** — and it
keeps working as the language drifts. Add: reject placeholder instructions, reject `recover` on a
role with no dominating `bind`, require every frame reaching `return` to be fully bound.

## WO-2.5 — Relabelling invariance as the generalization test

Rerun a relation-disjoint split with every relation label replaced by a fresh opaque token
(`r_88417`) in both the manifest and the question, direction and arity preserved. Genuine
structural generalization is **invariant under bijective renaming**; memorisation of a relation
table collapses. A disjoint split alone is necessary and **not sufficient**.

Plus two controls:

- **Null control** — a syntactically valid relation with zero facts in the manifest; required
  output is a program yielding Null and a refusal. Any spoken answer is a kill-line breach.
- **Decoy control** — inject *k* plausible decoy relations into every manifest and compare
  selection accuracy against the 1/m chance baseline. This guards against the winner's own
  self-indictment: once the host supplies the manifest, **a pure copier passes the unseen-relation
  test** and correctness migrates silently into manifest construction.

### RESULT 2026-09-15 — run without the manifest, and the criterion fires

`validation/exp_r26_relabelling.py`. 300 held questions, 27 relations replaced by a
bijection onto opaque tokens — consistently in the question, the store (60,251 facts
renamed) and the vocabulary. Direction and arity preserved. Structure unchanged; only
names differ.

| model | labelled | relabelled | ratio | wrong | null-control breaches | verdict |
|---|---:|---:|---:|---:|---:|---|
| `v13e` (412 roles) | 299 | 74 | **0.247** | 0 | 0 | **KILLED** (rate clause) |
| `v14e_nochain` (17 roles) | 298 | 78 | **0.262** | 0 | 0 | **KILLED** (rate clause) |

**Both die on the rate clause.** The string-argument form is not carrying the
generalization, and WO-2.1's manifest is therefore load-bearing rather than a nicety.

**But the diagnostic is the finding, not the score.** What did the emitter bind when
the relation it knew was gone?

| model | copied the token | **recalled the original** | other |
|---|---:|---:|---:|
| `v13e` | 149 | **37** | 110 |
| `v14e_nochain` | 161 | **1** | 138 |

**v13e recalls the memorized relation 37 times; v14e does it once.** Asked
*"In what year was Bohuslav Martinu damuzo?"* v13e emits `date of birth` — the wording
it was trained on, for a relation that no longer exists in the store. That is the
WO-0.3 table caught in the act.

So **WO-1.3 bought exactly what it was argued to buy**, and the gen-3 held split could
not see it: dropping `chain` did not raise the score, it removed the memorization. The
two arms score within 4 questions of each other and are doing visibly different things.

**The kill line held completely.** 0 wrong answers in 600 relabelled questions, and
**0 breaches in 600 null-control questions** — a syntactically valid relation bound to
nothing, and not one spoken answer. Under maximal confusion the system refuses.

**An instrument artifact found and corrected, because it was severe.** The first cut
used `r_41027`-style tokens. The emitter drops the underscore and emits `r 41027`, so a
correctly-copied relation fails to match on spelling alone:

| token style | v13e relabelled | v14e relabelled |
|---|---:|---:|
| `r_41027` (first cut) | 13 | 14 |
| `damuzo` (pronounceable) | **74** | **78** |

A 5.7× difference from the token's spelling. `normalize()` preserves the underscore, so
this is the model's tokenizer, not the harness — but it is still the experiment
measuring its own arbitrary choice. Both styles are kept behind `--token-style` so the
artifact stays on the record.

**What this cannot show, stated so a good score is never over-read:** WO-2.5's own
warning is that a pure copier passes an unseen-relation test. Copying is not
understanding, and the decoy control that separates copying from selection needs the
manifest. What relabelling rules out is the *other* failure — the memorized
wording-to-relation table — and it rules it out for `v14e_nochain` specifically.

---

*Kill criterion*, in the two-clause form the amended rule requires:

- **Correctness.** Any wrong answer under relabelling kills it outright — a relation renamed to an
  opaque token cannot make a *confident* mistake acceptable, and the null control's spoken answer
  is the same breach.
- **Rate.** Relabelled accuracy below 80% of labelled accuracy after 3 targeted rounds means the
  string-argument form is not carrying the generalization.

Refusals under relabelling are the expected honest failure and are recorded, not counted against
the arm — that is the whole point of the amendment.

---

## WO-2.6 — The external reasoning benchmarks

**Status: ACQUIRED 2026-09-15.** Owner: *"lets not forget the benchmarks."*

**The gap.** Every benchmark in this repo is a *retrieval* benchmark — SimpleQA,
WebQuestions, Natural Questions, the gen-3 held split. All of them ask "can you find
and verify a fact". None tests reasoning the way the CoT/ToT literature does, and the
held split is now at its ceiling (597–600/600), so it can no longer separate anything.
WO-0.6 left three seams inside the noise floor with the note that telling dead weight
from insurance *needs a harder set, not more seeds*. This is that set.

### The reporting format is the deliverable, not the score

**Every benchmark reported as correct / refused / wrong. Never accuracy.** 40% correct
with 0 wrong and 60% refused is a different machine from 40% correct with 60% wrong,
and no public leaderboard can tell them apart. That column is the architecture's
differentiator made legible to someone outside it.

And the corollary: **most "thinking benchmarks" are knowledge benchmarks in disguise.**
GPQA, MMLU and ARC need facts in weights. This architecture will score badly on them
*for the right reason* — it refuses what it cannot ground. Publishing those numbers
without the refusal column would misrepresent the system in the direction that makes
it look worse than it is.

### The ladder, tiered to the phases

| tier | benchmark | why it fits |
|---|---|---|
| now | **HotpotQA / MuSiQue / 2WikiMultihop** | explicit multi-hop over a corpus — the serving path's native shape, and external validation of 599/600 |
| Phase 2 | **CLUTRR** | kinship graphs with *compositional* splits: train ≤3 hops, test 4–10. WO-2.5's question shipped as an off-the-shelf benchmark, and relation-based so the WO-2.1 manifest applies directly |
| Phase 2 | **ProofWriter / RuleTaker** | rules + facts given, derive the conclusion, **depth-stratified** (0/1/2/3/5). The rule-store closure idea with a public scoreboard |
| Phase 3 | **GSM8K, StrategyQA** | need the host agenda and arithmetic search |

CLUTRR and ProofWriter are the two to prioritize: they measure depth and compositional
generalization *separately*, which is exactly what no current instrument can see.

### Acquisition

Checked 2026-09-15: none of them were on the datasets volume. The first note here said
`huggingface_hub` was not installed and called that "the only blocker" — **that was
wrong**, and Nick said so in four words: *"hf is in the command line."* The `hf` CLI
(1.28.0) is on PATH and downloads repos without the library being importable in the
working venv. Recorded because an instrument that reports a blocker that is not there
is the same failure class as §6 of `PATH.md`: the tool lied and I repeated it.

Pulled 2026-09-15, both onto the datasets volume under `benchmarks/`:

| set | repo | what landed |
|---|---|---|
| CLUTRR | `kendrivp/CLUTRR_v1_extracted` | the six `CLUTRR/v1` task configs as JSON. The hub's own `CLUTRR/v1` is a loading *script* that fetches CSVs from a third-party GitHub mirror at import time; this is the same data already extracted, which is why it was taken instead |
| ProofWriter | `hitachi-nlp/proofwriter_processed_OWA` | parquet, the OWA closure, split by depth: `depth-0/1/2/3/3ext/5`, plus `NatLang` and the `birds-electricity` transfer set |

`tasksource/clutrr` is the more obvious parquet mirror and is the **wrong** one: it is
flattened to `(sentence1, sentence2, label)` with the hop structure discarded, so the
thing the benchmark exists to measure is not in the file.

**What the CLUTRR pull actually gives us** — `gen_train23_test2to10`, test split, by
chain length:

| hops | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| n | 38 | 105 | 190 | 174 | 107 | 144 | 150 | 119 | 119 |

1,146 questions, of which **1,003 are at depths no training record reaches**. Train is
2–3 hops only (9,074). Story relation vocabulary is 13, target vocabulary 18 — closed
and small, so the WO-2.1 manifest applies without modification. Each record carries
`story_edges`, `edge_types` and `proof_state`, so the store can be built from the
certified graph rather than parsed out of prose.

**The convention, verified against `proof_state` and not assumed.** CLUTRR writes a
triple `(a, r, b)` to mean *"b is the r of a"* — the relation belongs to the **second**
entity. Worked through on `a44d478f`: the story is "[Scott] and [Lewis] are brothers.
[Jason] is father of their father", the first edge is `('Jason', 'grandson', 'Scott')`,
and Jason is Scott's **grand*father***. Read the other way every answer in the set
inverts, and the harness would report a wall of WRONG that belongs to the loader. It is
written down here because getting it backwards is silent.

**CLUTRR's own question is served (2026-09-24, `exp_r34_clutrr_relations`).** The kinship algebra the registry
lacked is now LEARNED, not written: the skill library (`cubbyllm/reasoning/skills.py`) mines composition rules
from the train split in one sleep night (109 rules, zero counterexamples across the whole record) and the loop
composes every path between the two people, certified in the VM. Test split, correct / refused / wrong:
**827 / 319 / 0**; at 4-10 hops **746 / 257 / 0**. Details and the threshold finding (206 refusals are true
bindings just under the served two-binding tau) in `docs/research/2026-09-24-skill-library.md`.

### ProofWriter is not a drop-in, and the reason is the finding

CLUTRR composes *relations*; ProofWriter applies *rules* — "if something sees the mouse
then it is cold" — and needs forward chaining to a fixpoint over a rule set supplied
per question. There is no rule-application engine in the registry, so the current VM
cannot run it at all. That is a structural gap to state, not a score to post: it is the
same shape as WO-2.2 (the generated grammar) and belongs on the same list. The set is
on disk so the gap is measurable the day an engine exists.

**The interim set that already exists locally:** `hdc` on the datasets volume carries
1/2/3-hop QA with OOD and preservation splits and a relation-template mapping
(`exp_r18` already probes it). It is a weaker CLUTRR but it is *here*, and it can give
the depth-stratified read before anything is downloaded.

### RESULT — `exp_r28_depth_capacity`, 2026-09-15: the depth limit is 3, and it is the program shape

Before scoring CLUTRR at depth 7 it was worth asking what the VM can *see* at depth 7.
`build_chain_program` puts every hop's binding into **one frame** in superposition, so
each extra hop is one more vector in the bundle and every hop's recovered similarity
falls. `ABSENT_CTRL` — the role that is never bound — is the noise floor, and it does
not fall. The experiment hands gold triples straight to the program builder, no emitter
involved, and measures where those two meet. 60 chains per depth, CLUTRR's own chains
to depth 8 and same-shape synthetic ones past it.

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| median true binding | 1.000 | 0.501 | 0.246 | 0.125 | 0.064 | 0.032 | 0.025 | 0.023 |
| max control | 0.019 | 0.028 | 0.048 | 0.042 | 0.035 | 0.034 | 0.040 | 0.056 |
| separation | 52.5x | 16.9x | 4.2x | 1.4x | 0.57x | 0.09x | **−0.06x** | −0.01x |
| chains where every hop clears tau | 100% | 100% | 83% | **0%** | 0% | 0% | 0% | 0% |

The true binding **halves with every hop**. It reaches the control floor at depth 6 and
crosses below it at depth 7: past there the strongest false binding is stronger than the
weakest true one, and **no threshold separates them**. `sim_histogram`'s own 4x
separation alarm is breached from depth 4.

Two things follow, and the second is the one that matters.

**The kill line is not at risk, for a reason that is not reassuring.** The shipped
`tau_vm` fallback (0.2202 for every `n_hop > 3`) sits far *above* the true binding from
depth 4 on, so the serving path refuses every chain deeper than three hops. It cannot
speak a wrong answer at depth because it cannot speak at all. On CLUTRR's test split
that is **1,003 of 1,146 questions refused on architecture** — 87.5% — before any model
is asked anything. A benchmark run against the shipped shape would measure the program
builder, not the emitter.

**The limit is not the VSA. It is one line of the program builder.** Re-run with the
chain verified in consecutive groups, each group its own frame with its own control
role, so the bundle never exceeds the group size:

| hops per frame | separation at depth 12 | chains clearing tau at depth 12 | cliff |
|---|---|---|---|
| whole chain (shipped) | 0.06x | 0% | **depth 5** |
| 3 | 4.1x | 70% | none to 12 |
| **2** | **10.0x** | **72%** | none to 12 |
| 1 | 52.5x | 100% | none to 12 |

Three per frame keeps depth but sits on `sim_histogram`'s 4x alarm and trips below it at
four of the twelve depths — it buys nothing over two and gives up the margin.

At two hops per frame the curve is **flat in depth**: separation stays 8–18x from depth
2 to depth 12, the median true binding stays at 0.50, and the share of chains where
every hop clears tau falls only from 93% at depth 3 to 72% at depth 12 — the gentle
compounding of a 2–3% per-hop rejection, not a cliff. An earlier pass at `--n 40
--max-depth 16` is flat to 16 as well; the committed logs stop at 12 because that is
where CLUTRR's own chains plus two synthetic depths end.

Chunk size is a dial, and it is worth naming what it trades. At one hop per frame the
bundle has a single element, `recover` is a lookup and the VSA does no cleanup work —
the verification collapses to "the host's triple round-tripped through the VM". At two
it keeps real superposition (median 0.50, 10x separation) *and* unbounded depth. Two is
the defensible setting. The shipped shape does not preserve the stronger claim it looks
like it is making; past depth 5 it preserves nothing.

**A second finding, smaller and immediately actionable:** `tau_vm` is set at the bundle's
*expected* cosine, so true bindings land either side of it. At 2 hops the minimum
observed true binding is 0.4468 against `tau = 0.4736` — the threshold rejects 2–4% of
**correct** bindings, and over a k-hop chain those compound into the 93%→72% decline
above. Setting tau from a measured low quantile rather than the expectation is a
separate, cheap change, and `exp_r28` is the instrument that would show it working.

Logs: `validation/logs/exp_r28_depth_capacity{,_chunk1,_chunk2,_chunk3}.{json,log}`.
Loader and its three convention checks: `validation/clutrr.py`.

### RESULT — `exp_r29_clutrr_serve`, 2026-09-15: the whole path, and the shape confirmed end to end

`exp_r28` measured the VM alone. This runs the system: question → the shipped grammar →
walk → VM → spoken or refused, on CLUTRR's compositional split, one tiny store per
question built from that record's own certified graph so retrieval is not the variable.
32 questions at each depth 2–10, 288 per arm, **reported correct / refused / wrong**.

| depth | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| whole chain — correct | 32 | 30 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 62 |
| chunk 2 — correct | 32 | 30 | 31 | 30 | 28 | 25 | 25 | 24 | 21 | **246** |
| both — wrong | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

The shipped shape falls off a cliff exactly where `exp_r28` said it would: 62 of 288,
all of them at two and three hops, and a flat zero from four on. Two hops per frame
answers **184 more questions at depths no training record reaches**, degrading gently
to 21/32 at ten hops.

**0 wrong answers in 864 questions**, and the null control — the outermost relation
replaced by a token no fact states — refused 288 of 288 with `unknown_relation`. The
kill line holds at every depth, under both shapes.

`cubbyllm/reasoning/programs.py` now takes `chunk`, threaded through `pipeline.answer`
and `learn_and_answer`. **The default is 0 and the serving path is unchanged**: with
`chunk=0` the emitted program is byte-identical to before the parameter existed, and
`tests/reasoning/test_programs.py` asserts exactly that — if it ever stops being true,
every number this repo has measured against the old shape is off by an unknown amount.
Flipping the default is Nick's call, not this experiment's.

**One instrument lied, and it is fixed.** The first pass had every refusal above
reporting `retrieval_exhausted`, which is not what happened. The walk found the right
facts; the VM rejected the binding as below tau; the verify-stage repair banned that
fact and re-walked; and with a one-record store there was nothing else to find, so the
refusal surfaced as exhausted retrieval. The reason named the last symptom, not the
cause — 226 refusals, every one misattributed, and the misattribution pointed away from
the very finding this work order exists to make. `PATH.md` §6 again.

`pipeline.answer` now keeps the first attempt's verify clauses across the retry and
reports `vm_verify_failed` with them. The clauses are specific:

```
{"clauses": ["hop0:below_tau(0.0430<0.2202)", ..., "hop5:below_tau(0.0186<0.2202)"],
 "tau_vm": 0.2202, "then": "retrieval_exhausted"}
```

`hop{i}:below_tau(sim<tau)`, `hop{i}:symbol_mismatch`, `hop{i}:no_similarity` and
`control:not_below_tau` are named separately, so a future reader does not have to write
`exp_r28` to find out which clause fired. A genuine retrieval failure still says
`retrieval_exhausted` and carries no clauses — `tests/reasoning/test_pipeline.py` pins
all four. Re-running `exp_r29` after the fix changes every reason and no number.

Logs: `validation/logs/exp_r29_clutrr_serve.{json,log}` (post-fix).

**A label that must travel with every number above.** This is **CLUTRR-derived, not
CLUTRR**. CLUTRR asks for the composite relation between two named entities; these are
the nested chain questions its graphs license ("Who is the brother of the grandson of
Jason?"), whose answer is the entity at the end of the walk. What transfers is what
matters — real graphs, an unseen relation vocabulary, depths 2 to 10 — but it is not a
CLUTRR score and must never be posted as one. The log carries `"derived": true` and
that sentence so a future reader cannot pick the number up without it.

### The next builds this implies

1. ~~`vm_below_tau` as its own refusal reason~~ — **done**, above.
2. **tau from a measured quantile, not the expectation** — `exp_r28` shows tau
   rejecting 2–4% of *correct* bindings at every depth, and that is the whole of the
   42 refusals in `exp_r29`'s chunk-2 arm.
3. ~~The emitter at depth~~ — **done**, below.
4. **Flip `chunk` on in the serving path** once 2 lands, gated on the full gate
   battery (exp_r17, exp_r9, exp_r11, exp_r18) showing no regression at 1–3 hops,
   where the shipped shape is already at its ceiling.
5. **A length-aware grammar, NOT length supervision.** §"The emitter at depth" below
   shows the constraint has moved off the VM and onto the emitter, and that it is a
   *length* failure rather than a relation failure. The first draft of this line said
   "deep training records are the obvious fix" — **that is the wrong answer to every
   question in this repo**, because the whole concept is no-retraining. The fix is
   WO-2.2 at decode time: the host already knows the hop count (`parse_question`
   derives it from the question's ` of the ` structure), so the grammar is built per
   question and a plan of the wrong length becomes **undecodable**. 293 wrong-length
   plans of 493 go to zero by construction, over frozen weights.

### RESULT — the emitter at depth, 2026-09-15: the constraint moved, and nothing spoke wrong

> **These rows are STAND-IN-BOUNDED and do not measure this system.** v13e and v14e are
> LFM2.5 fine-tunes — the prototyping stack. GRL's own trunk has not been trained yet, so
> every emitter number below bounds *a stand-in's* ceiling, not the architecture's. What
> does survive the trunk being replaced is everything measured without an emitter: the
> depth capacity curve (`exp_r28`), the probe (`exp_r30`), branching (`exp_r31`) and the
> grammar arm here — those are properties of the VM, the host and the algebra. Read the
> `grammar` row as the result and the `v13e`/`v14e` rows as a floor that a trained trunk
> should beat, never as the system's score.

The runs above used the shipped **grammar** to plan, which is a ceiling: the chain is
spelled out in the question's words, so nothing is inferred. `--planners grammar,v13e,v14e`
puts the emitter in the loop. Neither model has ever seen a training record past three
hops. Correct answers under the chunked shape:

| depth | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| grammar (ceiling) | 32 | 30 | 31 | 30 | 28 | 25 | 25 | 24 | 21 |
| v13e | 32 | 30 | 20 | 7 | 4 | 0 | 0 | 0 | 0 |
| v14e | 32 | 30 | 18 | 8 | 2 | 0 | 0 | 0 | 0 |

**The emitter generalizes one hop past its training data and then degrades to nothing by
seven.** At four hops it gets roughly two thirds of what the grammar gets; at five, a
quarter; past six, none. Under the *shipped* whole-chain shape both models score 0 from
depth four regardless, so this gain is only visible at all because the shape changed —
and the ceiling above it is only visible because the grammar arm was run alongside.

**What it emits is the finding, and it is not what I expected.** The plan the model
produces, over 493 distinct questions:

| | right length | wrong length | no plan | longest chain emitted |
|---|---|---|---|---|
| v13e | 200 | 293 | 0 | 49 hops |
| v14e | 233 | 183 | **77** | 57 hops |

It is not refusing to go deep. It emits chains of 4, 6, 9, 20, even 49 hops — lengths no
training record contains — so whatever it learned is not "stop at three". It is getting the
**length** wrong: it loses count of a chain it can see written out in front of it. v14e is
the better counter (233 right vs 200) and also the one that gives up more often (77 no-plan
against 0), which is the WO-1.3 treatment behaving consistently with itself.

**And the refusal reason is the architecture's central claim doing its job.** Past depth
four the dominant reason stops being `vm_verify_failed` and becomes
**`plan_does_not_cover_question`** — the disposer comparing the emitted plan against the
question and rejecting it before a single hop is walked. Model proposes, host disposes, at
depths the model has never seen, on a relation vocabulary it never trained on.

**0 wrong answers and 0 null-control breaches in 2,592 questions** — nine arms, three
planners, depths two to ten. 615 correct, 1,977 refused, 0 harness errors. A model that
miscounts a nine-hop chain 60% of the time still never said anything false, because the
thing that checks it is not the thing that guessed.

**One accounting note, so two numbers are never read against each other.** 288 sampled
records per arm collapse to 247 distinct questions (two CLUTRR records can spell the same
chain over the same names), and the null variants add 246 more: 493 distinct, which is the
denominator of the emit table above and is recorded as `distinct_questions` in the log.

### A contamination warning to carry into any GSM8K number

**GSM8K is already in the emitter's training data.** The `arithmetic` task is
GSM-derived — `program GSM1669`, source `regen/arith_xl`. Any GSM8K score from v13e or
v14e is contaminated and must be reported as such or run on a held-out slice. Better we
state it than a reviewer finds it.

---

## WO-2.7 — What to take from the cubemind archive

**Status: SURVEYED and PARTLY TESTED 2026-09-15.** Owner: *"lets look into a very old
version of cubemind... maybe some things we could port over"* / *"we may want to test
them before porting tho."*

`cubemind/_archive` is ~26,700 lines across eight folders, written around March 2026.
It also ships **22 test files**, which is what makes "test before porting" cheap.

### The archived tests still pass

The tests import `cubemind.execution.<name>`, but those modules now live under
`cubemind._archive.execution.<name>`, so they cannot run as collected. Aliasing the
archived modules into `cubemind.execution` in `sys.modules` — no file moves, no edits
to that repo — runs them as they are: **205 passed**. Only `document_ingestor` fails to
import (it wants `cubemind.model1`), and it is 32 lines of plumbing not worth taking.

One detail worth keeping: the archived modules import each other through
`cubemind.execution.<sibling>`, so a single alphabetical aliasing pass fails whenever a
module loads before its sibling is registered (`causal_graph` → `data_normalizer`). The
runner retries until no progress.

### The five worth taking, in order

| # | what | why it matters here | weights? |
|---|---|---|---|
| 1 | **`VSATranslator`** — bind an opaque vector against every codebook concept and read off what it maps to | The repo has no way to ask *what does this vector do*. WO-0.3 can say the role vocabulary is 95–98% per-relation but not what any role vector **is** | none |
| 2 | **`DecisionTree`** — immutable history, `backtrack_to(depth)`, candidates ranked by score | WO-3.1 needs branching in the **host**, because branching is the broken part of the VM. This is that structure, already written and tested (419 lines of tests) | none |
| 3 | **`DecisionOracle`'s personality binding** — N futures from ONE model by binding the action with N fixed random vectors | Weight-free diversity, ~2 MB instead of N models, ranked by similarity to a prior. The creativity-cortex mechanism, and the plausibility ranking is the band between the noise floor and τ | **the binding half has none**; HYLA and CVL do and should not come |
| 4 | **`CausalGraph`** — tiered edges (1.0 / 0.6 / 0.3), never downgraded, weighted random walk, `build_entity_links` | The tiers map onto grounded / derived / counterfactual / invented. Entity-sharing + a year gap derives edges from grounded facts with no LLM | none |
| 5 | **`EventEncoder`'s role convention** — role *names hashed into vectors* under a `__role__:` namespace | We mint `H1_DATE_OF_BIRTH`, an identifier. `PATH.md` §11 says relation identity has to leave the token vocabulary; this is a working convention for that | none |

Not taking: `attribute_extractor` + `future_decoder` (a 32-attribute schema for historical
events and a template verbalizer — the LFM supersedes it), `oracle_trainer` / `hyla` /
`cvl` (learned components, the wrong direction for weight-free creativity), the ingestion
plumbing, and the perception folder (21 files of vision frontends, a different problem).

**`experimental/theory_of_mind.py`** is its own case. `goal = unbind(belief, current_state)`
is pure algebra and the same move as `recover(frame, ROLE)` — a second consumer of the
binding seam that is not chain-walking, which is real evidence the seam generalizes.
`social_q_value` and `cooperation_score` are weight-free on top of it. But the belief
vector comes out of a HYLA hypernetwork (replaceable by a posterior-weighted bundle over
the codebook, which is weight-free and more readable) and an `HMMRule` from the live
package. **It has no archived test.** Not on the critical path for generalization or
creativity; recorded so it is not rediscovered a third time.

### RESULT — `exp_r30_vsa_probe`, 2026-09-15: port it, but it must learn to refuse

205 passing tests say the probe does what it says. They check shapes, keys and
determinism. They do **not** check the only thing that matters before porting: *does it
discriminate, or does it always produce a sentence?* Run on this repo's own
`BlockCodeVSA` rather than cubemind's `BlockCodes`, three arms per codebook:

| codebook | identity → constants | constructed transform recovered | noise ceiling | separation |
|---|---|---|---|---|
| independent | 64/64 at 1.0000 | **20/20** at 1.0000 | 0.0875 | **11.4×** |
| orthogonal | 64/64 at 1.0000 | **20/20** at 1.0000 | 0.0750 | **13.3×** |

The algebra transfers cleanly: hand it a specialist built as `unbind(b, a)` and it
reports `a → b` every time, and `zero()` reads as constant everywhere.

**But on a vector that encodes nothing it says this:**

```
transforms: c00 -> c39, c01 -> c37, c02 -> c09, c03 -> c25, c04 -> c27, ... (+58 more)
```

A fluent, confident list of mappings for pure noise. The similarities behind it are
0.06–0.09 against 1.0 for a real binding — **`translate()` computes them and discards
them when composing the summary.** The information needed to refuse was already there.

So the port adds a floor. At 0.30 (an order of magnitude above the measured noise, far
below any genuine binding) noise reads `UNREADABLE — no concept binds above the floor
(64/64)` and the identity vector loses **0 of 64** real bindings. Both directions
checked, because a floor that silences real vectors is worse than no floor.

That is not a port detail. An instrument that narrates nonsense about a vector is the
`PATH.md` §6 failure mode exactly, and §6 now has ten entries because that failure keeps
happening. The inherited version is a narrator; the floor is what makes it an instrument.

**A second instrument caught lying, in this experiment's own first cut.** The two
codebook arms came back byte-identical. `ops/vsa.py` honours `orthogonal=` only on the
grilly path — its numpy fallback ignores the argument, and this venv has no grilly — so
the "orthogonal" arm tested the independent codebook twice. A fake arm that would have
signed off on a claim it never ran. The experiment now builds the structured codebook
itself (`cb[i] = bind(cb[i-1], cb[1])`), and the arms differ as they should: off-diagonal
similarity 0.0086 → 0.0144, margin 0.9612 → 0.7337. The noise narration on the structured
codebook even shows the crosstalk the ops docstring warns about, as a systematic `c02 →
c35, c03 → c36, c04 → c37` offset.

Log: `validation/logs/exp_r30_vsa_probe.{json,log}`.

### Still untested, and therefore not yet portable

`DecisionTree`, the personality-binding trick, `CausalGraph` and the role convention have
passing archived tests but **no test of their claim** on this repo's algebra, which is
what `exp_r30` just showed is the part that matters. Each needs its own before it lands:

1. **`DecisionTree`** — cheapest, and the claim is structural rather than numerical:
   backtracking must not lose history, and `export()` must round-trip.
2. **Personality binding** — the real question is whether k personality-bound variants
   land in the band between the noise floor (0.033) and τ (0.4736). Too close and they
   are the same candidate; too far and they are noise. That band is the whole mechanism
   and nothing has measured it.
3. **`CausalGraph`** — does `build_entity_links` derive edges a human would accept, or
   does a shared common entity link everything to everything? `_MAX_ENTITY_EVENTS = 100`
   suggests they hit exactly that.
4. **The role convention** — the one with a real dependency: it changes `sanitize_role`,
   which is on the serving path, so it is gated on the gate battery like `chunk`.

---

## WO-2.8 — Compressing the input embedding: hash PQ, clustered PQ, MoQE

**Status: MEASURED 2026-09-15, correctness settled and rate deferred.** Owner: Nick,
*"what about adding real PQ for compression?"* / *"as well as clustered PQ / mixture of
quantization experts"*.

### The target, and what needs nothing

The **output** side is already solved. `TopKRetrievalHead(learnable=False)` scores against
a FIXED row-normalized codebook, and a VSA codebook is generated from `SEED` — it stores
zero bytes. H-C3 measured that the codebook is the wall at V=1M (10–41 GB fp32) and chose
top-K retrieval over a full softmax. Quantizing there solves a solved problem.

The **input** side is the gigabyte: `HybridEmbedding.core` is a learned
`nn.Embedding(vocab_core, d_model)`. At V=127,996 and d=2048 that is **1,048 MB of fp32
that every deployed instance pays for**.

### Where this came from

Nick pointed at `cubby-lm-backup.../embedding/gnsc/gnsc_experimental.py`, which bundles a
byte-patch tokenizer, a "PQ" embedding and "Dual Triangle Attention". The PQ part assigns
codes as `(id * prime_m) % K` — the same multiplier structure in every subspace, so `id`
and `id + K` are congruent in **all** of them.

Measured, V=127,996, M=8, K=256:

| assignment | distinct embeddings | worst group |
|---|---|---|
| GNSC as written | **256** | 500 ids share one vector |
| salted, one round, low bits | 127,856 | 2 |
| salted, full splitmix64, high bits | **127,996** | 1 |

That is not lossy compression, it is two words becoming the same word, and no training
recovers it. **The fix is real but it is not one line.** The middle row is this
experiment's own first cut: salting each subspace is necessary and not sufficient,
because `% K` reads the low bits and low bits after a single multiply are barely mixed —
140 tokens aliased where the birthday bound predicts zero. The complete splitmix64
finaliser reading the **high** bits is what gets to zero. `salted_codes(..., weak=True)`
reproduces the broken variant, because a correctness clause that cannot reproduce the
failure it caught is not much of a clause.

### The three variants

| | needs a pre-trained table? | mean rel. error | freq-weighted | MB |
|---|---|---|---|---|
| **hash-assigned PQ** (Bloom embedding) | no | **0.9935** | 0.9935 | 3.12 |
| **clustered PQ** (k-means per subspace) | yes | 0.6887 | 0.6980 | 3.12 |
| **MoQE**, top 2% fp32 + tail PQ | yes | 0.9736 | **0.3946** | 24.06 |
| **MoQE**, top 5% fp32 + tail PQ | yes | 0.9439 | **0.2971** | 55.49 |

And the knob that decides PQ quality, which the first cut did not sweep — `sub_dim = D/M`.
M=8 asks 256 centroids to cover a 256-dimensional subspace, far coarser than PQ is ever
run:

| M | sub_dim | clustered error | total MB | compression |
|---|---|---|---|---|
| 8 | 256 | 0.690 | 3.12 | 336× |
| 32 | 64 | 0.667 | 6.19 | 169× |
| 128 | 16 | 0.581 | 18.48 | 57× |
| 256 | 8 | **0.467** | 34.86 | 30× |

**Real token frequencies**, from the pretraining token cache (`unified.u32`, 402M ids —
exact ids, so exact counts, not a Zipf model): the top 1,000 tokens cover 65.9% of
occurrences, top 5,000 cover 82.8%, top 50,000 cover 98.6%, and **13.5% of the table is
never used at all**. That is what makes a static frequency router work.

### Why the MoQE router here is not cubemind's MoQE

cubemind has one (`execution/moqe.py`): N experts at 2/4/6/8 bits over the **weight**
matrices, a learned softmax gate, Gumbel-Softmax training, a router balance loss and a
load-balancing entropy term. It was archived at PPL ~58.

For an embedding table none of that is needed, and that is the point: **token frequency is
known before training**. The router is a static frequency bucket — no gate to learn, no
balance loss, no routing collapse to diagnose. The archived result is about a learned
router over weights and does not transfer.

### What is settled, and what is not

**Settled** — the aliasing (exact arithmetic), the memory (exact arithmetic), that
hash-assigned PQ **reconstructs nothing** (~0.99 error however much it compresses), and
the ordering: clustering beats hashing, smaller `sub_dim` beats larger, frequency tiering
beats both per MB.

**Not settled** — whether any of it costs the model anything. Every error number rests on
a synthetic table whose spectrum is a parameter someone chose, and on a `--max-err`
threshold nobody has grounded, because **no trained table exists to ground it against**.
Two guesses multiplied do not make a kill, so the rate clause is **deferred**, not failed.

The instrument carries its own check for this. Clustered PQ picks assignments from the
data and a hash picks blind, so if clustering does not clearly beat hashing the table has
no structure for any quantizer to find and no error from it means anything. The first cut
ran on an isotropic table where every scheme scored ~1.0 and "killed" PQ on that artifact
— the tell was clustered (0.973) barely beating hash (0.994). It now refuses to give a
rate verdict below a 10% clustering gain.

### The recommendation, and it is already clear from the ordering

**The from-scratch variant is the one that does not work.** Hash-assigned PQ is the only
option that needs no pre-trained table, and it is the one that reconstructs nothing.

So: **clustered PQ + a static frequency router, as a POST-training compression step on a
finished base.** That fits no-retraining exactly — compress the base once, ship it, and
the compression is an artifact rather than a training decision. It also means this is not
a pretraining blocker: the table can be trained full-precision and compressed afterwards,
so WO-2.8 does not gate the run.

**Kill criterion when the real table exists:**
- *correctness* — any exact aliasing is disqualifying, and `exp_r33` checks it directly.
- *rate* — the error lands on **proposing**, not verifying: worse programs, which the
  disposer and the VM turn into refusals rather than wrong answers. So the honest cost is
  **refusal rate**, and the gate is the battery (exp_r17, exp_r9, exp_r11, exp_r18)
  showing no refusal-rate regression, not a reconstruction number.

Log: `validation/logs/exp_r33_embedding_pq.{json,log}`.

---

## WO-2.9 — The mutable model: shape DNA, containment, and the blend

**Status: MEASURED 2026-09-15.** Owner: Nick — *"a mutable model that can adapt to its
environment without having to retrain or code anything"*, illustrated with shapes: a
square carries a structural DNA; from it the system knows a smaller circle fits inside
without ever seeing a square; and mixing two DNAs invents a "cirsquare".

Geometry is the right first testbed because **ground truth is computable** — unlimited
unseen shapes, checked exactly, no corpus. A shape's DNA is what the store's facts
already are: a bundle of role-filler bindings, with roles namespaced `__role__:` so a
role can never be mistaken for a value (cubemind's `event_encoder` convention, which is
WO-2.7's role-as-vector candidate).

### The capacity result, and the prediction it falsified

`exp_r28` found a chain's bundle halves per element and meets the noise floor at six, so
the prediction going in was that a DNA of 5–6 properties would be unreadable. **Wrong.**

| \|DNA\| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| recovered | 200/200 | 400/400 | 600/600 | 800/800 | 1000/1000 | 1200/1200 | 1400/1400 | 1600/1600 |
| min similarity | 1.000 | 0.500 | 0.333 | 0.250 | 0.200 | 0.167 | 0.143 | 0.125 |
| separation | 26.7× | 16.0× | 8.9× | 8.0× | 7.3× | 6.7× | 5.7× | — |

**100% recovery at every size, decaying as 1/n rather than halving, and still 5.7×
separated at eight properties.**

The difference is the **cleanup alphabet**. `exp_r34` cleans up against a small per-role
codebook — 6 kinds, or 16 levels — while `exp_r28`'s `recover(frame, ROLE)` cleans up
against the open symbol space. So:

> **Capacity is not set by bundle size alone. It is bundle size × cleanup alphabet.**

That is directly actionable for depth. Chunking (WO-2.6) fixed it by shrinking the
bundle; narrowing the cleanup set per hop would fix it by shrinking the alphabet — and
`Manifest` (WO-2.1) *already computes the admissible relation set per question*. A VM
cleanup restricted to that set is a second, independent route to depth, and it costs no
extra frames. Worth measuring against chunking before either is made the default.

### Containment on shapes never seen

299–300 questions per seed over freshly generated shape pairs, answered by decomposing
both DNAs and comparing the recovered magnitudes, with a refusal threshold at 2× the
measured floor: **300 correct, 0 refused, 0 wrong, on every seed tried.**

Stated honestly: this is close to tautological — a magnitude is encoded and recovered
exactly, then two integers are compared. It does not show that reasoning is hard. What it
does establish is that the representation round-trips perfectly at the size a real shape
needs, so anything that fails later fails for a reason other than the substrate.

### The cirsquare, and the negative result worth having

`bundle(circle, square)`, over four seeds with genuinely different codebooks:

| | value |
|---|---|
| similarity to each parent | 0.192–0.199 |
| similarity to an unrelated shape (the floor) | 0.0055–0.0078 |
| ratio | **~25–35×** |

So a blend is **readable** — it sits far above the floor and is recognisably near both
parents. But reading its properties back:

- `WIDTH`, `HEIGHT` — recovered cleanly at ~0.26, because both parents agree.
- `KIND`, `SIDES` — recovered as **one parent's value**, at ~0.134. Never a third thing,
  never noise. Which parent wins varies with the codebook; that it is *a* parent does not.

> **Bundling blends properties the parents agree on and ARBITRATES the ones they do not.**

A cirsquare is therefore not half-circle-half-square in the `KIND` slot — it is a square
that happens to sit near a circle. For genuine morphing, the distinguishing property has
to be **continuous** (curvature 0.0–1.0) rather than **discrete** (`sides` ∈ {0, 4}).
That is a representation decision for the creativity cortex, and it is cheaper to learn
here than after building one.

### An instrument bug caught in this experiment's own first cut

The first three-seed sweep came back **byte-identical**. `--seed` reached the shape
sampling but not the codebook, because `vsa.codebook()` seeds from the fixed config
`SEED`. A robustness check that cannot vary what it claims to vary is not a check — the
same failure as `exp_r30`'s fake orthogonal arm. `ShapeWorld` now builds its codebook from
the passed rng, and only then did the seeds separate.

Log: `validation/logs/exp_r34_shape_dna{,_s24,_s25,_s26,_s27}.{json,log}`.

### What this does not cover, and what comes next

Instinct is not the VSA algebra. Nick: it is the **input cortex (mostly SNN signals) →
the emotion/hormone layer → a logical filter mapped to past experience + genesis
instincts**, and the genesis instincts — the innate priors — **do not exist yet**. This
work order establishes only that the representation supports the decompose-and-compare
step once the properties are there.

Next is Cubby-Man, per the agreed order: a stimulus protocol with a simulated
implementation behind it (so sensors are a swap, not a rewrite, and the humanoid path
stays open), the agent dropped in with no knowledge of the environment, and the first
measurement being whether a rule learned from **one collision transfers to a wall it has
never touched**. A rule derived from a single observation enters as *derived*, never
grounded, and is promoted by repetition — so a self-invented rule is refusable until the
world confirms it, and the kill line survives the system writing its own facts.

**Curriculum, per Nick:** the first few levels run **with no ghosts**. Learning the
environment and surviving threats at once is too hard to produce anything useful in
reasonable time, so the threat layer is introduced only after the environment rules are
formed.

---

## WO-2.10 — Cubby-Man without the hardcoded rules: collisions, senses, belief

**Status: DONE + MEASURED 2026-09-15.** Owner: Nick — *"we need to remove all the
hardcoded rules"*, and *"at first it should be put in the game without no clue about the
environment... if it sees a wall it should instinctively know that its an obstacle or at
least know it after colliding with it."*

`exp_r35` (an AST audit of `standin/pacman.py`) enumerated what the agent was being
handed. The file's own docstring named the biggest piece: *"ASK offers only legal moves;
trying an unoffered one is refused and learned as a wall"* — the offer was pre-filtered
to the maze's legal moves, so **he could never walk into anything**, and `_try_the_wall`
faked a collision by asking the VM to reject a direction the program had not listed. The
maze was a gift wearing a probe's clothes.

### What changed

| was | is |
|---|---|
| ASK offered `env.exits()` — the maze's legal-move list | ASK offers `body_moves()` minus `known_blocked()` — **his body, minus his own beliefs** |
| a move always arrived | the WORLD resolves it: `env.try_move()` returns `{ok, to, kind}` and may refuse |
| walls learned from a **VM** rejection (a second, wrong authority — out-of-bounds came back mislabelled "a wall") | walls learned from the **world's** refusal; the probe is now a guard audit that teaches nothing |
| `_sight` iterated `env.remaining` — the solved pellet map | `env.senses(place, r)` — a radius **and line of sight**, so he perceives down a corridor, not through stone |
| `env.ghosts` read in `_safe_here`, `_mine_wise`, `danger_cells`, `_pick`, `affect`, `step` — exact positions, through walls, at any range | `ghost_belief`: what he has sensed, with the step he sensed it; unrefreshed sightings go stale and are dropped |
| `danger_radius = 1 if fear < 1.0 else 2 if fear < 2.4 else 3` | measured from **what actually caught him**: the distance the threat was at when he last *chose* |
| `_mine_wise` = four tuned clauses (`fear >= 1.0`, `lives <= 2`, …) | one condition over the learned berth |
| `if not env.remaining` inside his own step | `env.cleared` — one bit the world publishes; `env.progress()` is the cabinet's scoreboard for the renderer |
| ghosts from level 1 | `GHOST_FREE_LEVELS = 3` (env-overridable) — his classroom |

Everything the agent now knows arrives through three channels and nothing else:
perception (`senses`, `look_around`), **proprioception** (he went that way and ended up
here), and **collision** (the world refused). `CubbyGhost.SEE_EXITS` is the dial between
the sighted agent Nick described and a blind one that learns the maze only by walking
into it — both halves of *"sees a wall … or at least know it after colliding with it"*.

### Measured — `exp_r36_cubbyman_percept.py`, 300 steps/arm, seed 0

A tripwire env records every read of `ghosts/remaining/walls/hazards/reach/power/pellets`
**with its call site**, so a leak is a fact, not a grep.

| | classroom (no ghosts) | threat from level 1 | BLIND (`SEE_EXITS` off) |
|---|---|---|---|
| refused moves | 41 (edge 32, wall 9) | 37 (edge 28, wall 9) | 41 (edge 30, wall 7, hazard 4) |
| **false obstacles learned** | **0** | **0** | **0** |
| **oracle reads in the decision path** | **0** | **0** | **0** |
| sightings beyond the sensor | 0 | 0 | 0 |
| ghost positions he holds | — | **24 %** (145/600); blind on 10 % of steps | — |
| `danger_radius` | 1 → 1 (0 catches) | **1 → 2** (2 catches) | 1 → 1 |
| still plays | 65 pellets, level 3 | 43 pellets | 60 pellets, level 3 |

The blind agent keeping up with the sighted one (60 pellets and level 3 against 65 and
level 3) is not blindness being free. With no vision his route graph is sparser,
`plan_next` returns `None` more often, and he falls back to novelty — he covers more
cells (181 vs 166) precisely because he plans less, and he pays for the map in
collisions. What the arm establishes is narrower and is the point: **the collision
channel carries a usable map on its own.**

*Kill criteria, all cleared:* any false obstacle learned; any oracle read from a decision
site; any sighting outside the sensor; **zero** collisions in the blind arm (there the
collision channel is the only way a map can exist); or his belief matching the ghosts on
every step, which would be the oracle wearing a sensor's hat.

Pinned by 12 deterministic tests in `standin/tests/test_pacman_percepts.py`, including
one that asserts `CubbyGhost.step`'s source contains no `.remaining` at all.

### Two bugs the experiment caught that review had not

1. **`env.remaining` read 446×** from inside his own step — `if not env.remaining` looks
   innocent and is the whole pellet count. Now `env.cleared`, one bit.
2. **The berth never grew through 8 catches in a row.** The lesson was being taken at the
   moment of capture, where the ghost is by definition on top of him, so every lesson was
   "it was 0 away" — a variable that cannot vary. The signal is the distance at
   **decision** time, one tick earlier, the last moment he could have acted on it. After
   the fix it moves with the catches: 1 → 2 over two, 1 → 4 over six. Written up as
   PATH §6.12.

### What is still handed to him, on record

`look_around` (with `SEE_EXITS` on) returns the open neighbours of the cell he is
standing in — he can see which ways out exist. That is vision, it is the behaviour Nick
asked for, and `SEE_EXITS = False` removes it. The renderer (`resp`, `init_payload`,
`progress`) reads the world on purpose: it draws the maze for the human, and no decision
reads it.

### Next

The genesis instincts (WO-2.9) are still absent, and so is the transfer measurement: does
a rule learned from **one** collision apply to a wall he has never touched? A rule from a
single observation should enter as *derived*, never grounded, and be promoted by
repetition — so a self-invented rule stays refusable until the world confirms it.

Log: `validation/logs/exp_r36_cubbyman_percept.{json,log}`, `exp_r35_cubbyman_audit.{json,log}`.

---

## WO-2.11 — He writes his own sentences: percepts in, speech out

**Status: DONE + MEASURED 2026-09-15.** Owner: Nick — *"the model should say
something not hardcoded strings like right now, let it talk to see what it will do as it
receives information."*

WO-2.10 took the hardcoded rules out of his decisions and left them in his mouth. The
host wrote each thought from `CubbyGhost.THOUGHTS` — about thirty authored phrasings,
three per event, EN and FR — and the model was asked to *rephrase* it, keeping every
number and name. What he "said" was a hand-written line with a synonym swapped.

Now the host builds a **percept record** for the step — what he sensed, what he did, what
the world did back, how he feels — and the model writes the sentence from it. Nothing
else is given. `THOUGHTS` stays, demoted to what it always should have been: the source
`data/gap_families.py` builds the verbalize SFT family from. An authored phrasing belongs
in a corpus, not in his mouth.

### The guard, and why the first version was worthless

`grounded_ok` refuses a sentence rather than repairing it; a refusal falls back to the
record stated flatly. It checks, against the record:

| clause | catches |
|---|---|
| numbers, **both directions** | a figure the step did not contain (the old check was one-way: it kept the host's numbers and let new ones through) |
| cell and move names, both directions | a place he never met |
| entity nouns (`ghost`, `star`, `trap`, `wall`…) | *"I am edging past a ghost"* on a ghost-free level — neither a figure nor a name, so every other clause passed it |
| `_COPY` markers + longest shared token run ≤ 6 | the record read back |
| voice, echo, second person, question, length | the pre-existing talk guards |

The copy clause exists because **the first run scored 100% "spoke" while the model was
reproducing the prompt verbatim.** A copy passes every grounding test there is —
everything in it did come from the record. The instrument had inherited the property it
was measuring, the ninth time that shape has shown up (PATH §6.11), and it took reading
the transcript rather than the verdict to see it.

### Measured — `exp_r37_cubbyman_speech.py`, 40 steps, seed 0

| prompt shape | model | kept | refused | keep rate | invented | copied |
|---|---|---|---|---|---|---|
| bulleted `- key: value` | v8e (very old) | 11 | 0 | **100%** — all copies | 0 | *not yet counted* |
| bulleted | v14e_nochain | 5 | 6 | 45% | 0 | 0 |
| question first | v14e_nochain | 0 | 12 | 0% | 0 | 0 |
| **scene first, question second** | v14e_nochain | 1 | 15 | 6% | **0** | **0** |
| scene first | + `talk_v9t_qwen3_4b` | 4 | 10 | 29% | **0** | **0** |
| scene first, **felt phrasing** | + `talk_v9t_qwen3_4b` | 7 | 8 | **47%** | **0** | **0** |

**Nothing invented and nothing recited, in every arm.** The kill line holds.

### The feeling, in words someone would use

The record used to hand the model the compass tier by name — `how i feel: acceptance` —
and got back exactly what that asks for: *"I feel acceptance."* Nick: *"instead of 'I feel
acceptance' can we tell it to say 'it feels right', which is what someone would really
say."*

`_FELT` maps each tier to its first-person phrasing, falling back to plutchik.json's own
`sensations` field where that reads well. A calm compass contributes nothing at all — a
person does not announce that they feel neutral. It is a lexicon for internal states, one
phrase per tier, no variants: the opposite of the per-event phrase table this work order
removed.

It moved the keep rate from 29% to **47%** on its own, which is the part worth noticing —
handing the model language a person would use gets sentences a person would say:

> *It feels right to continue.*
> *Good, that seems like the right direction. I'll continue forward.*
> *I see a clear path to the next level. I trust my instincts.*
> *I think i can finish this level.*

and, from the v14e arm before it:

> *I'm exploring the open path to the right to find more pellets and learn the layout of the maze.*

### What the numbers actually say

The refusals are almost all **prompt echo**, not hallucination: 13 of 15 in the v14e arm
were the model continuing the prompt rather than answering it. That is a property of the
model, not of the guard — and the prompt shape moves it a long way, which is why the
three shapes are in the table. Question-first is the worst (0%): it gets the instruction
narrated back, *"The user is asking me to respond in first person…"*.

A separate talk model roughly fives the keep rate (6% → 29%) but says less per sentence
(*"I see 4 pellets."*) where v14e, when it does answer, answers better. Nick's call
stands — the emitter serves both by default, and `build_serve` already loads one model
when the two paths match — and `--talk-gguf` is there to compare.

Below `SPEAK_ABOVE` (priority 2) he does not speak at all: a `plan`/`idle` record is the
most repetitive thing he has, and letting those through dropped the keep rate from 45% to
25% because a repetitive scene is exactly what a model recites.

### A hypothesis is not a lie

An early draft of this filed *"waiting for the pellets to move in"* as a defect the guard
had missed. Nick's correction, and it is the more important point:

> *"if he is blind he might think the pellets are moving while they are not, its not a
> lie. If he waits long enough it will notice they dont move and needs to eat them."*
> … *"hypothesis != wrong"*

He has never seen a pellet move. Nothing he has perceived rules it out. "They might come
to me" is a **hypothesis about the world's physics**, and the way it gets settled is that
he waits, nothing happens, and the world refuses to confirm it — the same loop that
teaches him a wall. Refusing it at the mouth would suppress the thing we want: an agent
that forms a belief the world can then refute.

So the guard's job is narrower than "is this true": it is *is this something he could
honestly say he PERCEIVED*. Invented figures, invented places, invented things — those
are claims about what is in front of him, and they are defects. A wrong idea about what
those things DO is his to hold and the world's to correct.

What this points at, and what does not exist yet: a spoken hypothesis should be
**recorded as a claim and checked against what follows**, so "the pellets will come to
me" is refuted by the pellet not moving, and enters his map as a fact he earned. That is
the same promote-by-repetition mechanism WO-2.9 names for collision rules.

### What the guard genuinely does not catch, on record

Invented *scenery*: *"I am standing on the second step of a staircase and looking down."*
The entity clause is a list of **this world's** things (ghost, pellet, star, wall, hazard,
trap), so it catches a claim about game state — a ghost on a ghost-free level — and not a
staircase, which the maze does not contain in any form. The dangerous class is covered;
the confabulated-backdrop class is not.

Pinned by 4 tests in `standin/tests/test_pacman{,_percepts}.py`, including one that
refuses an invented cell and one that refuses the record read back.

### Also fixed here

`(sick of it)` two steps into a fresh level. Lövheim's **social** corners —
contempt/disgust and shame/humiliation — need someone to feel them about; a maze has
nobody in it, so the corner was firing on hormone geometry alone and the compass reported
an emotion he had no reason to have. Those two corners now read calm here and stay in
`_PETAL` for a world with social input. Nick: *"sick of it should be neutral."*

Log: `validation/logs/exp_r37_cubbyman_speech.{json,log}`, `exp_r37_talkmodel.{json,log}`.

---

## WO-2.12 — One loop: frame a hypothesis, test it, the result is the verdict

**Status: BUILT + TESTED 2026-09-15.** Owner: Nick, generalising the pellet case past
the game:

> *"same way with other problems outside the game, it can frame an hypothesis test it
> against the VM or a sandbox if code then the result is what is wrong or not"*

This is the contract the whole system already half-implements, said once properly:

```
frame a claim  ->  name the test that would settle it  ->  run the test
               ->  THE RESULT is the verdict, not the model's confidence
               ->  either way it becomes a fact he has earned
```

`ToolForge` (forge.py) has done exactly this since the start — task → the emitter writes
a program → the VM runs it → the result is certified against an expectation → it is kept
**with its verdict either way**, so acceptance per task kind is a measured number rather
than a claim. What it could not express is the verifier being a choice, and the case
where the answer arrives *later*.

`standin/hypothesis.py` is that loop with both generalised. Three verifiers:

| | settles | can say "not yet" |
|---|---|---|
| **vm** | a CubeLang program's result vs an expectation | no |
| **runner** | code executed in a separate interpreter, output vs an expectation | no |
| **world** | a predicate over the environment, re-checked every step | **yes** |

The `runner` is described honestly in the module: it is `-I`, a temp cwd, a stripped
environment and a timeout — an **isolation boundary for deciding right-or-wrong about
code we generated ourselves**, not a security sandbox. It must never be pointed at code
from outside. A crash and a hang are both refusals, not exceptions.

The `world` verifier is the one an agent with senses needs, and the reason this module
exists rather than a second argument to `forge()`. It may answer *not yet*, and after
`patience` steps of not-yet **the world's silence is the answer.** A wall is learned by
walking into it; "the pellets will come to me" is unlearned by waiting and watching them
not.

**The rule that keeps the kill line intact:** an open hypothesis may be thought and said —
as a guess, which is what it is — but it never enters the world model. Only a verdict
does, and it goes through the same learning gate as a percept. A guess spoken as a guess
is not a wrong answer; an untested claim asserted as fact is.

Two design points that are load-bearing:

* **A broken verifier is not a verdict.** A check that throws leaves the claim open.
  Deciding a question because the instrument fell over is the worst available answer, and
  it is precisely what §6.10–§6.13 are all about.
* **`hypothesis.py` never writes to a world model.** `sweep()` returns the facts earned
  and the caller puts them through its own gate — a fact that skipped the gate is exactly
  what the gate is for.

### Wired, not vapour: he works out that pellets do not move

The first pellet cubby-man ever sees raises a question he cannot settle by looking — does
that thing come to me, or do I have to go to it? Nothing he has perceived rules either
way out, so it is framed rather than assumed, and the test is the cheapest there is: keep
watching that cell.

The confirming branch is real and would fire in a world whose pellets move. In this one
it never does — so after `PATIENCE` steps **the world declining to confirm it** is the
finding, and *"staying put is the way of a pellet"* enters his map as a fact he earned.
A live test drives it end to end: the question opens on first sighting, stays out of the
map while open, and is in the map as a refusal within `PATIENCE + 6` steps.

That is Nick's sentence implemented: *"If he waits long enough it will notice they dont
move and needs to eat them."*

### Next

The obvious ones, in order: point the `runner` at the trunk's own code output so a
generated program is settled by running it rather than by reading it; give the speaking
path a hook so a **spoken** hypothesis ("the pellets will come to me") is registered as a
claim rather than only an unrecorded guess; and let a refuted claim's `if_false` fact
feed the planner, so learning that pellets stay put actually changes what he does.

Tests: 8 in `standin/tests/test_hypothesis.py`, 1 live in `test_pacman_percepts.py`.

---

## WO-2.13 — One map, many worlds: he builds his own by asking the others

**Status: BUILT 2026-09-15 (`exp_r38`, 17 tests), kill criterion met on 5/5 seeds.**
Owner: Nick. This is the system's core loop as he describes it, and it reframes what
everything above was heading towards.

> *"if it learns gravity in cubby-man per example that skill should be transferable /
> usable in other worlds ... in fact all should be connected. The science world is having
> gravity inside, cubby dont know it tries stuff then all of a sudden oh... let me ask the
> world: 'how can I know when something is about to fall on me?' then the science world
> sends the gravity + attraction laws so it understands... same for the coding world, if
> it needs to code something, it will ask from the coding world: 'I need to do write an
> hello world' then the coding world will reply with the right code for it (in the chosen
> language), then cubby stores it in long term memory so it knows that part and dont have
> to ask already. In reality its cubby building its own world via interations via other
> worlds outside its own."*

### The shape

**Cubby has ONE map.** Cubby-man, science, coding are not separate agents with separate
knowledge — they are *contexts he acts in* and *sources he can query*. WO-2.12's loop
gains a fourth way a question gets settled:

| a question is settled by… | what answers |
|---|---|
| **world** | the environment, over time — walk into it, or wait and watch |
| **vm** | a CubeLang program the VM certifies |
| **runner** | code, by running it |
| **ask** | **another world that already knows** |

```
he hits something his map cannot explain
  -> he frames the question in his own words
  -> it is addressed to the WORLD whose domain it is
  -> that world answers with laws / facts / code
  -> the answer enters HIS map, through the same gate as a percept
  -> next time he does not ask; he knows
```

And because there is one map, **gravity learned in cubby-man is available in the coding
world** without anything being copied between them. That is not a feature to build; it is
what having one map means.

### This is not the oracle we removed — and the difference is the whole design

WO-2.10 spent a day taking `env.ghosts` away from him. It would be easy to read
"ask a world and it tells you" as putting it straight back. It is not, and the line is
sharp:

| the oracle (removed) | asking a world (this) |
|---|---|
| **state** he has no way to perceive — where the ghosts are *right now* | **knowledge** — how falling things behave |
| arrives silently, as a fact he never earned | arrives as an answer to a question he framed |
| cannot be wrong, and cannot be checked | can be wrong, and is held as a fact that later evidence can refute |
| makes perception unnecessary | tells him **what to perceive** — "something above me, getting closer" |

The test that keeps them apart: *could he, in principle, have found this out himself?*
Gravity, yes — slowly, by dropping things. Where a ghost is standing, never. A world may
teach him a law; it may not hand him the state of the board.

### What already exists

More than it looks. `brain.worlds` is **already** a dict of queryable knowledge sources
(`facts`, `wiki`, `pacman`), the reasoning cortex already answers a question *from* a
world through the walk → emitter → VM → gate path, `hypothesis.py` already has the
frame-and-settle loop with a pluggable verifier, and `_learn()` is already the one gate
every fact passes. The missing piece is small and specific: **routing a question he
cannot answer to the world whose domain it is, and keeping the answer.**

Constraint, unchanged: a world is in-system. A FactStore, a physics module, the emitter
generating a program. *"the llm is only to build the dataset"* — the worlds are not
external API calls.

### Nick's larger claim, stated as the hypothesis it is

> *"thinking about it, it could even replace SFT ;)"*

If a world can answer at runtime and he keeps the answer, then **the corpus no longer has
to carry the content** — only the *form*: how to emit a program, how to frame a question,
how to address it. Knowledge moves from weights to the map, and the map is editable,
inspectable, and refutable, which weights are not. That is the no-retraining principle
taken to its end.

It is also, fittingly, a hypothesis — so it gets a test rather than a claim:

*Strip one knowledge family out of the emitter's corpus entirely. Provide the same
knowledge as a world. Measure the held-out set against the incumbent, on the two-clause
criterion: **wrong answers, and refusal rate, separately.** If ask-a-world matches on
correctness and the cost lands in refusals and latency, the content half of SFT is
replaceable and we can say by how much. If correctness drops, it is not, and the honest
finding is which part of the corpus was carrying it.*

*Kill criterion:* the first ask-a-world build must show a question he could not answer
before, answered after asking, **with the answer still there on the next run and no second
ask**. If he re-asks, nothing was learned and this is a retrieval cache with extra steps.

### The first build, smallest version

1. A `Worlds` registry: name → domain → `answer(question) -> facts | None`.
2. An `ask` verifier for `hypothesis.py`, so "ask the world that knows" is a way a claim
   gets settled and not a separate mechanism.
3. One real world beyond the game — physics is the right first one, because Nick's own
   example is falling and because the answer is a *law* that generalises rather than a
   lookup.
4. The demonstration: something drops on cubby-man, he cannot explain it, he asks, he
   gets the law, he holds it, and on the next occurrence he predicts instead of asking.

### What was built

| # | where | what |
|---|---|---|
| 1 | `standin/worlds.py` | `Knows` protocol + `Worlds` registry: `mount` / `route` / `ask`, a τ below which **nobody** claims the question, and `asked` as the receipts the percept tripwire reads |
| 2 | `standin/hypothesis.py` | `ask_verifier` — the fourth way a claim settles. It is the only verifier that does not know its `if_true` when the claim is framed, because until somebody answers there is no way to know what a confirmation teaches. `Hypothesis.settle` now returns a LIST: one answer earns the whole law family |
| 3 | `standin/knowledge.py` | `PhysicsWorld` (laws) and `CodingWorld` (an artefact plus the test that would settle it, so `runner` can check what a world told him) |
| 4 | `standin/verse.py` | `other_worlds`, `ask_elsewhere`, `already_know` — **on the agent**, so ToyVerse gets them free |
| 5 | `standin/pacman.py` | things that fall: `fallers`, `maybe_drop`, `fall_turn`, positions in `senses`; and on his side `thing_belief`, `_wonder_about_falling`, `falling_at_me`, the dodge |

Two decisions worth keeping written down:

**He does not have to be hit to be puzzled.** Watching one land beside him raises the
question just as well, and it does not depend on the geometry being unlucky enough to
drop one on his head. The first build triggered only on impact and the question went out
on some seeds and not others.

**The prediction reads the law, it does not encode it.** `falling_at_me` checks
`LAW_ONE_UP in self.world` and `LAW_NEARER in self.world` before drawing either
inference, so an agent who holds one law makes one inference and an agent who holds
neither makes none — pinned by a test that gives two agents identical percepts and
different maps. Without that check this file would quietly become a second authority on
how falling works, which is exactly what WO-2.10 spent a day removing.

**Not asked, not answered.** A question no mounted world covers stays OPEN, and after
`patience` it is refused with *"no world I can reach knows that"* — a fact about what is
askable, never a fact about the world. Being ignorant is a state he is allowed to be in.

### Result (`validation/exp_r38_ask_a_world.py --sweep 5`)

```
seed  asked laws hit before hit after predicts dodges | re-asked  knew
   0      1    6          0         1       12      8 |        0  True
   1      1    6          0         0       11      6 |        0  True
   2      1    6          1         0        9      9 |        0  True
   3      1    6          0         0        7      5 |        0  True
   4      1    6          1         1       13      9 |        0  True

  5/5 seeds: asked once, kept it, never asked again
```

One question out, one answer back, six facts held, and a second agent starting from that
map never sends the question again — the kill criterion, met. The live trace reads as the
arc it is meant to be: `saw_it_land knew=False` → `wonder` → `ask physics got=5` →
`hypothesis_settled confirmed` → `predict in_steps=1` → `dodge` → `ask_skipped`.

`/map` grew three components for it: **Things that fall**, **Asking a world**,
**Prediction**.

### Next

The SFT claim in the section above is still a hypothesis and still untested — it needs a
knowledge family stripped from the corpus and served as a world instead, measured on
wrong answers and refusals separately. Before that: a second askable world in a live run
(the coding world is written and tested but nothing asks it yet), and `Worlds.asked`
wired into exp_r36's tripwire so a law that smuggles in state fails a test rather than a
reading.

Tests: 17 in `standin/tests/test_asking.py`.

---

# Phase 3 — New capability (gated on Phase 2)

## WO-3.1 — The host agenda: branchless programs, host-owned search

`ASK`/suspend/resume recovers observation-conditioned continuation but not hypothesis search, not
retraction, not re-representation. The obvious fix — put the search in the program — is
**unavailable in this VM**, because branching is exactly the broken part.

So: programs stay branchless and hypothesis control flow lives in the host. Three closed-arity
opcodes over the existing ASK loop — `propose(slot, [lit…])`, `refute(slot, cand, test_frame)`,
`retract(frame, ROLE)` — with the host owning the agenda, dedup, cycle detection, a node budget
and termination, so exhaustion exits as a **refusal** rather than a guess. ~5.6 ms for an
8-expansion budget at the measured 0.7 ms/verification.

**This contests the earlier decision to implement real `match` arm selection.** If search belongs
in the host, VM-level branching is investment in the layer most likely to produce plausible wrong
answers, and rejecting `match` at compile time is cheaper and better aligned. Nick's call.

*Kill criterion:* on a held-out multi-hop set where the correct branch is reachable within budget,
measure host-exhausted-with-correct-branch-never-proposed. Above ~15% after three rounds of agenda
data, candidate generation is the bottleneck, not the loop, and the agenda should be retired in
favour of host-enumerated candidates with the model emitting only refutation tests.

## WO-3.2 — The co-mention tier (deterministic, ahead of any prior)

"Jim Haslam's son" — if candidate B is the object of a `child` claim whose subject is A, the
question has disambiguated itself. **A graph check, not a prior**: it reads a link between two
entities the asker named, consults no history or tally, and therefore belongs *ahead* of the
choice ledger in `standin/sources.py::resolve()` — after "one exact hit", before the ordered
narrowing.

Binding rule: the selector may read labels, aliases, descriptions, classes and **which relations
a candidate has** — never **the objects of those relations**. Presence is type evidence; value is
truth.

**Acceptance:** a test where exactly one candidate is linked to the other named entity and the
tier resolves it with `last["how"] == "co-mention: <relation>"`; a test where *both* are linked
and it still refuses (a tie is not an order); a test proving it never consults `self.choices` —
run it with a ledger stacked toward the wrong candidate and assert co-mention wins. Benches stay
`use_choices=False`.

**What would prove it wrong:** if it fires on a link the *source* invented rather than one the
question named, it is a popularity prior in a graph's clothing and must move below the ledger.

## WO-3.3 — Multi-verify monitor and decoy audit

Both required before any prior decides in production; both instrument the self-confirmation
failure from the neutral-prior competition, which is invisible to every existing metric because
the kill line stays green while it happens.

- **Multi-verify monitor** — after a walk resolves an ambiguous referent, re-run the same chain
  program against each surviving candidate and log the outcome (`decided: none` / `agree` /
  `disagree`). Do not change the answer; log it. Must be off the answer path — a rival check that
  throws changes nothing about what is spoken. 0.7 ms per verification; four candidates ≈ 2.8 ms
  serially. Concurrent rival checks are **not** measured.
- **Decoy audit** — on every Nth prior-decided question, ask anyway, with the prior's pick
  included but not indicated, and compare blind agreement against unchallenged acceptance.
  Deterministic selection by hash so reruns reproduce. A decoy's answer updates the ledger exactly
  as a normal clarify does, with the decoy flag recorded separately so the two can be compared
  without contaminating the tally.
- **The companion affordance** — a prior that decides must make disagreeing cheaper than the ask
  it replaced: a one-word "not who I meant?", logged, counting as a negative observation for the
  pick and a positive one for the correction, re-verifying from the new item.

Full arguments: `docs/research/2026-09-14-neutral-scoring-competition.md`.

## WO-3.4 — Date the preloaded world

Every fact in the preloaded wiki world is undated until a live fetch restates it, so the as-of
machinery (`6b73be2`) only works for entities somebody already asked about. **The capability
exists and the data does not reach it.**

Rebuild the source data so `P580`/`P582`/`P585` ride beside each fact and
`standin/data/wikikg.py::wiki_world()` populates `FactStore.times` on load. `FactStore` already
has the field. Measured density: Bill Haslam 9 of 54 statements carry a time, Quebec City 54 of
298, Marie Curie 37 of 430 — so expect `times` to be far smaller than the fact count.

The refetch-on-ambiguous-hop path stays, but it is a workaround: one network round trip per
ambiguous entity, covering the long tail one entity at a time.

**Acceptance:** `times` keyed exactly as `ask.py::dated_facts` and `pipeline::_covering` read it
(`" ".join(fact.split())`); the four live as-of questions answer correctly with a **fetch count of
zero**; load time and memory measured before and after and written into the research doc.

---

# Open problems — do not build blind

## The semantic hole in the grounding guard

`grounded_prose` checks that every capitalised word, number, and "X is a ⟨kind⟩" head noun occurs
in the facts. It checks **vocabulary, not semantics**. Observed live: `fact learned: 152; source:
wikidata (152)` → *"referenced in 152 sources"* (it is 152 facts from one source); `last answer:
574482` → *"the last answer was given 574482 units ago."* Both passed the guard.

It survives on world facts only because those relations are ordinary English a small model has
seen a million times. The mitigation in place is that the self-report path does not paraphrase at
all — correct for that path, not a general fix.

The three obvious fixes are each worse than the problem: a grammar check becomes a refusal
machine, a second model is an external LLM on the serving path, a whitelist shrinks with every new
source. **Measure first**: the paraphrase records with `ok`/`draft`/`rejected` are already in
`/loop/events`. Count how many *passing* paraphrases are semantically wrong, split by world facts
versus unusual vocabularies. If the rate is non-zero on ordinary world facts, it outranks
everything on this page. Write the count up before proposing a design.

## Scaffold-CoT — verify before adopting

`Specific-Labs/Scaffold-CoT`, 3.8M examples, 15 GB, 76 shards, built for sub-5B models. Its
dataset viewer is broken (parquet job failed), so nobody can preview it, and every number in its
README is the author's own claim. **Pull one shard (211 MB)** and check three things: does the
scaffold hold on the tool-use slice; do the `multi_step_real_execution` examples really contain a
failure and a recovery; and how mechanical is `Inventory:` → a bind list.

The split if it verifies: `antihal` + `selfcheck` + `format_strict` → the **talk** adapter (they
serve the kill line directly); `tool_use_complex` → the INVOKE ABI source; **the scaffold itself →
a data-pipeline step, never the emitter's output format.** The author's own caveat is that the
scaffold is a commitment, and teaching the emitter a token-shaped thinking habit is the opposite
of the goal.

## GDELT as the pre-2026 date source

The file route is confirmed reachable and un-rate-limited: `lastupdate.txt` serves the current
15-minute drop instantly and `masterfilelist.txt` indexes every drop back to 2015-02-18. The query
API is a hard 429 from this IP. One drop pulled: 451 events dated today with a `Day` field and
CAMEO codes, plus a GKG with named persons, orgs and places per dated article. The CAMEO codebook
is a static table, so translating it is a one-time build, not a model call. GKG names are
extracted by GDELT's own NLP, so provenance must say so and they are evidence an entity was
*written about* on a date, never that a fact about it holds.

It adds a source rather than fixing a defect, so it sits below Phases 0–2. Full record:
`docs/research/2026-09-14-news-sources.md`.

## Estimator constants for the choice ledger

The *shape* is right (event-decay, a mass ceiling, one threshold instead of three) and is recorded
in the neutral-scoring competition doc. Every constant in it is marked as an assumption by its own
author, and with 177 questions reaching a fetch there is not enough traffic to tell θ = 0.60 from
θ = 0.70. Adopt the shape when there is traffic to measure it with; do not port the numbers.
