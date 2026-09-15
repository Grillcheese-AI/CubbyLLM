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

---

*Original specification:*

`chain` is 28,529 of 75,534 rows, is the sole source of the 402 per-relation identifiers, and is
work the host already does deterministically (`build_chain_program`) — in the training data the
facts are handed to the model in the prompt, so it is transcribing, not discovering.

Separately, 11,731 rows teach the model to emit `@external` / `@system` / `@once`, every one of
which parses and is never read by compiler or VM. A capacity tax on a 2.6B, free to remove.

**Falsifiable prediction:** dropping both improves plan accuracy and cross-task retention with no
loss of VM-verified answers. **Kill criterion:** if VM-verified answer rate drops, `chain` was
doing something the audit did not see.

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

*Kill criterion:* relabelled accuracy below 80% of labelled accuracy after 3 targeted rounds means
the string-argument form is not carrying the generalization.

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
