# From VM to Generalized Thinking

The route from "a thought is a program the VM executes" to "the model writes programs for
questions nobody wrote a rule for" — what has been measured, what turned out to be false, and
what would prove the whole thing wrong.

This is a living document. Each phase appends; nothing is deleted. Where an earlier claim here
was wrong, it is struck and corrected in place rather than removed, because the correction is
usually worth more than the claim was.

Companion documents: `WORK_ORDERS.md` (what to build and why), the sibling CubeLang checkout's
`docs/DRIFT.md` (the language audit), `docs/research/` (the arguments each decision rests on).

---

## 1. The claim

Frontier models bill for thinking that cannot be seen. Reasoning tokens are generated, metered
and charged, inside a box the user cannot audit, cannot cache, and pays for again on every
similar query.

The alternative this project is built on: **a thought is a program, and programs run in a VM.**

A program executes in milliseconds and costs zero tokens. Once certified it is kept and reused at
zero marginal cost. The harder the reasoning, the wider the gap — a frontier model's cost grows
with reasoning depth and a VM call's does not. It holds hosted or air-gapped, because the VM does
not phone anywhere.

That is the economic argument, and it is the easy half. The harder half is epistemic:

> A program can be executed, certified, retired, and audited. A token stream can only be read and
> believed.

Everything below follows from taking the second sentence seriously.

### What "generalized" has to mean here

The trap in this architecture is not that the model fails. It is that the model becomes
decorative and everything keeps reading green.

If a hand-written grammar parses the question, the host builds the walk, and the VM certifies the
answer, then the system is correct *and the model contributed nothing*. Every metric downstream —
precision, coverage, kill-line compliance — stays green while the thing supposedly being trained
does no work at all. Ten more SFT rounds could improve emitter loss while its causal contribution
to spoken answers is exactly zero.

So "generalized thinking" gets a specific, falsifiable definition:

**The model writes a correct program for a question whose phrasing, and eventually whose
relation, no rule in the system was written for — and the answer is certified by a VM that has
no idea a model was involved.**

Three properties, each independently testable:

1. **Causal** — removing the model degrades the answer. (If not, it is theatre.)
2. **Structural** — performance survives replacing every relation label with an opaque token.
   Memorising a relation table collapses under bijective renaming; understanding does not.
3. **Honest** — where it cannot write a correct program, it refuses. A refusal is a result.

---

## 2. The constitution: seven invariants

An idea that violates one is rejected, however good it is otherwise. These are not aspirations;
they are the filter every design passes through before it gets built.

1. **The VM is the only truth gate.** No LLM-as-judge anywhere in the loop. The model never
   scores its own thoughts.
2. **The model proposes, the host disposes.** Identity, provenance, execution, reward,
   retirement, and what the model is shown are host decisions.
3. **The don't-know contract.** An unverified claim is never spoken.
4. **Retire, never delete.** Every record carries a store-snapshot hash and a git revision.
5. **Serve-time budget.** A small model on a consumer GPU; VM calls in milliseconds; training on
   one 80 GB GPU for tens of minutes per round.
6. **No self-play or self-judging shortcuts.**
7. **Nothing depends on another repository at runtime.** Reusable pieces are ported, never
   linked.

Invariant 1 is the load-bearing one. It is also the one that makes this hard: with no model
allowed to judge, every quality signal has to come from execution against a store, and every
measurement has to survive the question *"what would this instrument say if the thing it
measures were broken?"* Most of Phase 0 turned out to be answering that question.

### The kill line

**Zero wrong answers spoken.** A refusal is a result; a wrong answer is a defect. When a change
could trade a refusal for a guess, that is not a trade.

This is stricter than accuracy and it is deliberately lossy. The project's own honest framing:
raw accuracy is lower than a chase-only baseline, because the pipeline refuses what it cannot
verify. Coverage is the weak axis, not correctness. When this pipeline claims, it is right.

---

## 3. What the VM buys, and what it does not

The VM certifies **binding fidelity, not truth.** This is on record and it matters more than any
headline number.

When a chain verifies, what has been established is that the symbols bound in the program are the
symbols present in the store, at the similarity the algebra predicts. It has *not* been
established that the store is right. Every counterfactual catch in the fault-injection work was
caught by symbol mismatch — the mechanism is structural agreement, not knowledge.

Three consequences that shape everything downstream:

- **A stale fact verifies.** The gold labels, the verifier and the answer all descend from the
  same snapshot, and zero of 552,297 facts carry a time qualifier. A fact true at snapshot time
  and false today passes every gate. The kill line reads green *by construction*, because the
  only oracle it has is the stale thing. Every existing instrument is downstream of the snapshot.
  This is what WO-0.5 exists to put a number on.
- **Ingested documents are outside the gate's protection.** Extraction can be mechanically bound
  to source spans, but a wrong relation between two entities that *are* both present passes the
  span check. The honest design carries a provenance tier, not a claim of truth.
- **`recover` on an unbound role does not return Null.** It returns the nearest candidate at low
  similarity. That is the confabulation mechanism, and it is still open (DRIFT §B11a). WO-1.2
  narrowed the candidate pool to the frame's own bindings, which cut the noise floor by more than
  half — but the shape of the failure is unchanged: a low-similarity symbol from the right
  structure instead of a low-similarity symbol from anywhere.

---

## 4. The ordering principle

Two independent model competitions produced the same warning, and it decided what gets built
first:

> The emitter may be contributing nothing, and every instrument we own is structurally blind to
> it.

So: **instruments that could falsify the architecture's central claim come before anything that
assumes the claim is true.** Do not build a capability manifest for an emitter that turns out to
be decorative.

This is the single most useful decision recorded here, and it is worth stating as a general rule:

> When a system has a component whose contribution is *assumed*, the cheapest possible next
> experiment is the one that removes it.

---

## 5. Phase 0 — what the instruments said

Five instruments, all built, all baselined. Each one produced a finding. Three of them also
produced a lesson about the instrument itself, which is recorded in §6 because that turned out to
be the more valuable output.

### 5.1 The ablation — the central result

`validation/exp_r22_emitter_ablation.py`. Four arms over the same 240 questions and the same
store snapshot: the trained **emitter**; the hand-written **grammar**; a **trivial** constant
program; and **shuffled**, a real program borrowed from a different question. The question set is
one item asked four ways.

Correct answers, by question shape:

| question shape | emitter | grammar | trivial | shuffled |
|---|---:|---:|---:|---:|
| canonical | 33 | **58** | 2 | 0 |
| "have" phrasing | 23 | 0 | 2 | 0 |
| relative clause | 35 | 0 | 2 | 0 |
| possessive | 8 | 0 | 2 | 0 |
| **total (of 240)** | **99** | 58 | 8 | 0 |

**Wrong answers: 0 in every arm.** Emitter accuracy 0.4125, noise floor 0.0618, ablation gap
0.4125, 513 VM calls, 159 s.

Read this table carefully, because it says three different things at once.

**The emitter is not decorative.** The ablation gap is 0.4125 against a 0.0618 noise floor — the
trivial arm scores 8 of 240 and the shuffled arm scores nothing at all. Removing the model
destroys the answer. Property 1 of §1 holds, measured, and the "unfalsifiable theatre" warning is
answered.

**The grammar is better than the emitter at what the grammar was written for**, and by a lot:
58 to 33 on canonical phrasing. The thing being replaced still wins on its home turf.

**The grammar scores zero on all three rephrasings.** Not "worse" — zero. On the "have" set it
plans 60 walks and all 60 fail; on relative clauses it refuses all 60 with `unknown_relation`; on
possessives it produces no plan at all. The emitter gets 66 across those three.

That is the whole thesis in one table. The grammar is a better program writer for the sentences
someone sat down and wrote a rule for. The emitter is the only thing in the system that survives
the question being asked differently. Generalization is not a nice property of the emitter — it
is the *only* property the emitter has that the grammar does not, and it is already measurable.

It also sets the honest target. The emitter is not yet good enough to retire the grammar: 99 to
58 overall, but 33 to 58 where they overlap. The path is to keep the rephrasing robustness and
close the canonical gap, and the failure mode to watch for is closing the gap by memorising
canonical phrasings and losing the rest.

### 5.2 Liveness — do the programs read their inputs?

`validation/exp_r23_liveness.py`. Execute each emitted program twice: once against the fact base,
once against a copy where the queried relation's value is swapped for another valid value. **A
live program's output must change.**

n=60 chains, 59 informative: **59 live, 0 dead.** Of the live ones, 56 changed their answer, 2
refused after the swap, 1 spoke only when perturbed.

This is the production-side detector for exactly the compiler bugs §7 describes — a program whose
output is invariant to its own inputs is the signature of a placeholder argument, a zero-bytecode
assign, or an all-arms `match`. It catches them in serving, not only in audit.

### 5.3 Role vocabulary — where generalization is actually failing

`validation/role_vocab.py`, plus a count in every SFT manifest. One integer per build: the number
of distinct role identifiers in the corpus. **A generalizing interface has a flat count across
rounds.**

| corpus | distinct roles | per-relation | share |
|---|---:|---:|---:|
| v12e | 151 | 143 | 95% |
| v13e | 176 | 168 | 95% |
| v13f | 412 | 403 | **98%** |

v13f adds 261 roles over v12e and drops none. 260 of the 261 are of the form
`H<hop>_<RELATION>` — `H1_DATE_OF_BIRTH`, `H1_PLACE_OF_BIRTH`, and so on.

**The reading that changed what to do about it:** v12e was already 95% per-relation. The jump
everyone noticed is relation *coverage* growing, not a regression that got introduced. **The
interface has never generalized.** This is not something to revert to a previous round — the
interface itself is the work. A relation absent from training has no role to bind to, by
construction, which means the system cannot possibly answer a question about a relation it has
not seen, no matter how good the model gets.

The audit turned up a worse sub-case that no claim covered: **51 roles are per-relation
*per-entity***, with the entity name baked into the identifier — `H1_INSTANCE_OF_RADOALD`,
`H1_GENRE_OF_JOAN_RIVERS_A_PIECE`. A role minted for exactly one question can never be reused.
That is pure memorisation capacity with zero possible transfer, and the generator that produces
them is still in the build.

This finding is why Phase 2 is shaped the way it is. WO-2.1 (a per-request capability manifest)
and WO-2.5 (relabelling invariance) exist to move relation identity out of the *token vocabulary*
and into *data the host supplies per request* — the only structural fix.

### 5.4 Similarity at serve time

`cubbyllm/reasoning/simlog.py` — one line per spoken answer, off unless `CUBBY_SIMLOG` is set,
never raises — and `validation/sim_histogram.py` for the weekly read.

Baseline: 150 accepted bindings over 75 spoken answers, all 2-hop. Every accepted binding in
[0.4746, 0.60). Control role median 0.0254, max 0.0459. **Separation 10.3×.** After WO-1.2
narrowed the candidate pool: accepted bindings **bit-identical**, control median 0.0112,
**separation 14.3×**.

That before/after is the cleanest result in Phase 1. A genuine recovery is unchanged to the bit,
the noise floor fell by more than half, and nothing downstream needed recalibrating.

### 5.5 Volatility — the missing time qualifier

`validation/exp_r24_volatility.py`. Sample spoken facts, re-fetch them live, keep a per-predicate
disagreement table. The table **is the missing time qualifier, learned empirically for free**:
high-volatility predicates become fetch-required, and the aggregate rate is the only available
measure of snapshot decay.

First run, 50 sampled, 202 API calls: 29 re-fetched, 23 agreed, **0 confirmed changed**, 2
disputed, 4 absent, 21 unresolved. Disagreement 0.0% of decided.

Read that as *no staleness measured yet*, not as a green light. 29 checks is nothing, and 21 of
50 subjects did not resolve at all. Until stored facts carry the QID they came from, this probe
measures a thin, name-resolvable slice rather than the snapshot.

### 5.6 The zone ablation — what each brain zone is actually worth

The organizing frame is *brain zones, each with a specialty*. A zone earns that name three
ways: **ablation** (removing it changes measured behaviour), **channel** (it talks through a
typed seam, not shared state), and **reuse** (it serves more than one task). §5.1 applied the
first test to exactly one zone. WO-0.6 applied it to the rest.

**The static half first, because it changes how to read the dynamic half.** An AST sweep for
production importers found three modules declaring `Wiring.WIRED` with **zero** callers on any
forward path: `hippocampus`, `striatum`, `retriever`. Each is measured in its own experiment and
mounted nowhere.

That is not a design error. `Wiring`'s own docstring says the marker is *intent*, and names four
clauses for "done" — intent is WIRED, the body is real, something actually calls it, its test is
green. The suite enforced one. And the guard that enforced it is *named after* "the 'built but
never wired in' trap". A guard that reads like rigour and checks the weakest clause it names is
§6's pattern applied to CI, and it let the gap reach three modules silently.

**So the table below is what the system scores with three declared zones entirely absent.**

600 gen-3 held questions, seed 7. The plan is emitted once and reused across every arm that does
not ablate the emitter, so generation variance cannot leak into the comparison (§6.8's lesson,
applied in advance this time).

| arm | seam removed | correct | refused | **wrong** | verdict |
|---|---|---:|---:|---:|---|
| `full` | — | 597 | 3 | 0 | baseline |
| `no_plan` | the emitter | 225 | 375 | 0 | load-bearing (coverage) |
| `permissive` | the relation gate's judgement | 263 | 337 | 0 | load-bearing (coverage) |
| `tau_zero` | τ_vm → 0.0 | 116 | 477 | **2** | **load-bearing (kill line)** |
| `no_repairs` | repairs 1 → 0 | 596 | 4 | 0 | inside the noise floor |
| `top_k_1` | retrieval breadth 3 → 1 | 597 | 3 | 0 | inside the noise floor |
| `tau_floor` | τ_vm → 0.0332 | 594 | 6 | 0 | inside the noise floor |

Noise floor: 0.0064, or 3.9 questions.

**τ_vm is the only zone whose removal produces a wrong answer.** Every other ablation converts
answers into refusals. That locates the kill line's mechanism exactly: it is not distributed
across the architecture — it is the acceptance threshold, and everything else fails safe.

**And loosening τ produces more refusals, not more answers.** 477 against the baseline's 3,
almost all `retrieval_exhausted`. A wrong binding at hop 1 leaves hop 2 with nothing to find, so
the chain refuses instead of continuing. Only 2 of 481 reached a spoken wrong answer. **The
multi-hop structure catches its own errors downstream** — which is a stronger claim than the
baseline score, and it was not designed in deliberately. The two that escaped are the
confabulation signature precisely: *"When was Sugar Ray Robinson born?"* → `1921-05-03` against
a gold of `1920`. Plausible, adjacent, confident.

The emitter is worth 372 questions here, far more than §5.1's canonical split implied, because
these are free-text wordings the grammar simply cannot parse — 350 `unparseable`. The relation
gate is worth 334, and all of it is coverage: with its judgement removed it still produces 0
wrong.

Three seams sit inside the noise floor. The honest reading is *on this question set*: at 597/600
there is nothing left for a repair or a wider search to rescue. Whether they are dead weight or
insurance that only pays on harder questions is not answerable from here — it needs the harder
set, not more seeds.

**Test 3 has no instrument.** Nothing measures whether a zone serves more than one task, and a
zone-per-problem is §5.3's pathology at architecture scale.

### 5.7 Relabelling — the test the held split could not run

§5.3 found the role vocabulary 95–98% per-relation and called it the memorization ceiling.
WO-1.3 removed the task that minted those identifiers, 412 → 17, and the held split saw
nothing: 599.7 against 598.7 of 600, because it is at its own ceiling. This is the test that
is not.

> Genuine structural generalization is invariant under bijective renaming. Memorization of a
> relation table collapses.

Every relation replaced by an opaque token — consistently in the question, the store (60,251
facts renamed) and the vocabulary. Direction and arity preserved. Only names change.

| model | labelled | relabelled | ratio | wrong | null-control breaches | verdict |
|---|---:|---:|---:|---:|---:|---|
| v13e — 412 roles | 299 | 74 | **0.247** | 0 | 0 | **killed** (rate) |
| v14e_nochain — 17 roles | 298 | 78 | **0.262** | 0 | 0 | **killed** (rate) |

**Both fail.** The string-argument form is not carrying the generalization, which makes
WO-2.1's per-request manifest load-bearing rather than a nicety — Phase 2 is confirmed as
the work, empirically rather than by argument.

**And then the diagnostic, which is the actual finding.** What did the emitter bind once the
relation it knew was gone?

| model | copied the token | **recalled the original** | other |
|---|---:|---:|---:|
| v13e | 149 | **37** | 110 |
| v14e_nochain | 161 | **1** | 138 |

Asked *"In what year was Bohuslav Martinu damuzo?"*, v13e emits `date of birth` — the wording
it was trained on, for a relation that no longer exists anywhere in the store. That is §5.3's
table caught in the act, 37 times. v14e does it **once**.

So **WO-1.3 bought precisely what it was argued to buy, and §8's comparison could not see it.**
The two arms score within four questions of each other and are doing visibly different things.
A score that moves by 0.3% was hiding a 37× difference in mechanism.

**The kill line held completely.** 0 wrong in 600 relabelled questions, and 0 breaches in 600
null-control questions — a syntactically valid relation bound to nothing, and not one spoken
answer. Under maximal confusion the system refuses, which is the whole design.

**One more instrument that lied, caught in this run.** The first cut used `r_41027`-style
tokens. The emitter drops the underscore and emits `r 41027`, so a correctly-copied relation
fails to match on spelling alone:

| token style | v13e | v14e |
|---|---:|---:|
| `r_41027` | 13 | 14 |
| `damuzo` | **74** | **78** |

**A 5.7× swing from the token's spelling.** `normalize()` preserves the underscore, so this is
the model's tokenizer rather than the harness — but the experiment was still measuring its own
arbitrary choice, and 0.043 would have gone into the record as a capability number. Both styles
are kept behind a flag so the artifact stays visible. It belongs in §6 and is counted there.

**What this cannot show.** WO-2.5's own warning is that *a pure copier passes an unseen-relation
test*. Copying is not understanding, and the decoy control that would separate them needs the
manifest. What relabelling rules out is the other failure — the memorized wording-to-relation
table — and it rules it out for v14e specifically.

### 5.8 The manifest — supplying the information is not enough

§5.7 concluded that the string-argument form does not carry the generalization, which makes
WO-2.1's per-request manifest load-bearing. This tests that conclusion rather than assuming it.

`cubbyllm/reasoning/manifest.py`. The host supplies the admissible relations for one request,
with direction and arity. One tier: exact lookup, a miss is a miss. `StoreRelations` has a
second, Jaccard ≥ 0.6 paraphrase tier — and that tier is exactly what lets a memorized relation
wear an exact hit's confidence. Removing it *is* the proposal.

Same relabelled world, same questions. The admissible set is supplied twice over: in the
emitter's system prompt, and as the gate.

| model | arm | correct | wrong | bound right token |
|---|---|---:|---:|---:|
| v14e | relabelled | 56 | 0 | 107 |
| v14e | + manifest | 53 | 0 | 102 |
| v14e | + manifest + 3 decoys | 51 | 0 | 106 |
| v13e | relabelled | 53 | 0 | 102 |
| v13e | + manifest | 54 | 0 | 101 |
| v13e | + manifest + 3 decoys | 52 | 0 | 100 |

**Six arms within five questions of each other. The manifest recovers nothing.** The right-token
rate sits near 50% throughout, unmoved by the manifest and unmoved by decoys — the emitter is
reading the *question*, not the admissible set. It was never trained on a manifest format, and
zero-shot it ignores the block entirely.

**So the decoy control is inconclusive, and that is the honest reading.** It exists to measure
whether selection beats the 1/(k+1) baseline. Nothing is selecting, so there is nothing to
measure. Three decoys cost one question in two hundred.

**But the failure mode improved sharply.**

| arm | `unknown_relation` | `retrieval_exhausted` |
|---|---:|---:|
| v14e relabelled | 29 | 56 |
| v14e + manifest | **89** | **1** |

Without the manifest the system takes a bad relation, walks, and exhausts retrieval — a late,
expensive, vague failure. With it the gate refuses at once and names the reason. *Evaluated, not
parsed* does exactly what it was specified to do; the emitter is what is not participating. Flat
accuracy, much better diagnostics, a cheaper failure path.

**What this settles, and it is the useful part.** WO-2.2's generated grammar is not an
optimization — it is the mechanism. A prompt asks; a grammar enforces. When a model will not use
information it is handed, the remaining move is to make the wrong relation **undecodable** rather
than discouraged. A negative result that names the next build precisely is worth more than a
positive one that leaves it optional.

**0 wrong answers across all 1,200 questions**, decoys included.

---

### 5.9 Depth — the limit was in the program, not in the machine

Every depth number in this document is 1, 2 or 3 hops. `tau_vm` has exactly three entries to
match, and a fallback that silently reuses the 3-hop threshold for everything deeper. That was
never a decision; it was the shape of the data. CLUTRR's compositional split — train on two and
three hops, test on **four to ten** — is the first instrument that asks what happens past it, and
the first question is not *how well does it score* but *what can the VM see down there at all*.

`build_chain_program` puts every hop's binding into one frame in superposition. Each extra hop is
one more vector in the bundle, so every hop's recovered similarity falls. `ABSENT_CTRL` — the role
that is never bound — is the noise floor, and it does not fall. Gold triples straight into the
program builder, no emitter, no retrieval, 60 chains per depth:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| median true binding | 1.000 | 0.501 | 0.246 | 0.125 | 0.064 | 0.032 | 0.025 | 0.023 |
| max control | 0.019 | 0.028 | 0.048 | 0.042 | 0.035 | 0.034 | 0.040 | 0.056 |
| separation | 52.5× | 16.9× | 4.2× | 1.4× | 0.57× | 0.09× | **−0.06×** | −0.01× |

**The true binding halves with every hop.** It reaches the noise floor at six and crosses below it
at seven, where the strongest false binding beats the weakest true one and no threshold separates
them at all.

The kill line survives this, for a reason that is not reassuring. The shipped `tau_vm` fallback
sits far *above* the signal from depth four on, so the serving path refuses everything deeper than
three hops. It cannot say anything wrong down there because it cannot say anything. On CLUTRR's
test split that is 1,003 of 1,146 questions refused on architecture, before a model is asked
anything at all.

**And then the fix turned out to be one line.** Verify the chain in consecutive groups, each group
its own frame with its own control, so the bundle never exceeds the group size:

| hops per frame | separation at depth 12 | chains clearing tau at depth 12 |
|---|---|---|
| whole chain (shipped) | 0.06× | 0% |
| **2** | **10.0×** | **72%** |
| 1 | 52.5× | 100% |

At two hops per frame the curve is flat in depth: 8–18× separation from depth 2 to depth 12, the
median true binding steady at 0.50, and the whole-chain clear rate sliding only from 93% at depth
three to 72% at depth twelve — the gentle compounding of a 2–3% per-hop rejection, not a cliff.

Chunk size is a dial and it is worth naming what it trades. At one hop per frame the bundle has a
single element, `recover` is a lookup, and the VSA does no cleanup work: the verification collapses
to *the host's triple round-tripped through the VM*. At two it keeps real superposition **and**
unbounded depth. The shipped shape does not preserve the stronger claim it appears to be making —
past depth five it preserves nothing.

End to end, on the whole serving path — question, grammar, walk, VM, spoken or refused — 32
questions at each depth from two to ten:

| depth | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| whole chain — correct | 32 | 30 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 62 |
| chunk 2 — correct | 32 | 30 | 31 | 30 | 28 | 25 | 25 | 24 | 21 | **246** |
| wrong | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

**0 wrong answers in 864 questions**, and the null control — the outermost relation replaced by a
token no fact states — refused 288 of 288.

Four things have to be said about what this is and is not.

It is **CLUTRR-derived, not CLUTRR**. CLUTRR names two entities and asks for the composite relation
between them, which needs a kinship algebra the engine registry does not have. These are the nested
chain questions its graphs license, answered with the entity at the end of the walk. What transfers
is real graphs, an unseen relation vocabulary, and depths nothing here had reached; the number is
not a leaderboard entry and the log says so in its own field.

The planner in that table was the **grammar**, not the emitter. The chain is spelled out in the
words of the question, so nothing had to be inferred; it is the ceiling, what the VM and the walk
can do when the plan is right. Putting the emitter in the loop is the generalization question, and
it is the next paragraph.

**The emitter at depth.** Neither v13e nor v14e has seen a training record past three hops.
Correct answers under the chunked shape, with the grammar's ceiling above them:

| depth | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| grammar (ceiling) | 32 | 30 | 31 | 30 | 28 | 25 | 25 | 24 | 21 |
| v13e | 32 | 30 | 20 | 7 | 4 | 0 | 0 | 0 | 0 |
| v14e | 32 | 30 | 18 | 8 | 2 | 0 | 0 | 0 | 0 |

It generalizes about one hop past its data and is gone by seven. But *how* it fails is the part
worth having. Over 493 distinct questions v13e emits the right chain length 200 times and the
wrong one 293; v14e, 233 right, 183 wrong, 77 no-plan. And the lengths it emits run to **49 and
57 hops** — lengths no training record contains. It never learned "stop at three". It is losing
count of a chain written out in front of it: a *length* failure, not a relation failure, which is
a much more tractable thing to fix.

Past depth four the dominant refusal stops being `vm_verify_failed` and becomes
**`plan_does_not_cover_question`** — the disposer comparing the emitted plan against the question
and rejecting it before a hop is walked. That is invariant 3 doing its job at depths the model has
never seen, on a vocabulary it never trained on.

**0 wrong answers and 0 null-control breaches in 2,592 questions** across nine arms: 615 correct,
1,977 refused, 0 harness errors. A model that miscounts a nine-hop chain most of the time still
never said anything false, because the thing that checks it is not the thing that guessed. That
sentence is the whole architecture, and this is the first time it has been tested somewhere the
model is genuinely out of its depth.

And the change is **not switched on**. `chunk` defaults to 0, the emitted program is byte-identical
to the shipped one at that setting, and a test asserts it: if that ever stops being true, every
number measured against the old shape is off by an unknown amount. Flipping the default is gated on
the gate battery, not on this result.

The thing worth carrying out of §5.9 is smaller than the numbers. For three sections this document
has treated depth as a property of the architecture. It was a property of a function that builds a
string.

---

## 6. The instruments that would have lied

This is the section worth reading twice. Under invariant 1 there is no model available to sanity-
check a number, so a broken instrument produces a confident wrong measurement and nothing
contradicts it. Every one of these was caught by asking *"what would this say if the thing it
measures were healthy?"* — and several were caught only after they had already produced a number
that got written down.

**A measurement that cannot be wrong is not a measurement.** Each entry below is an instrument
that had to be fixed before its own baseline meant anything.

### 6.1 An alarm that fires at 100% on healthy data

WO-0.4 specified two alerts. One was "mass concentrated just above a τ boundary" — decisions
riding the threshold being the confabulation signature.

It fired at **100%**. The other, "fraction below 0.5", fired at 50.7%.

Neither is a finding. τ_vm is *derived from* the expected cosine of a k-element bundle —
1.0 / 0.4736328125 / 0.22021484375 at one, two and three hops — so a **correct** binding lands
just above its τ by construction. And 0.5 is meaningless at hop 3, where τ is 0.22.

Written as specified, this instrument would have cried wolf every week until somebody stopped
reading it. What actually separates a real binding from cleanup noise is the gap to a **control
role** — bound to nothing, measuring the floor directly. That is the alarm now. The below-0.5
fraction was kept but alarms only on a *rise between two windows*, which is what the original
alert was really asking for.

**The general form:** an alarm derived from the same constant as the thing it watches cannot
discriminate. Check every threshold against a healthy run before trusting it on a sick one.

### 6.2 A dead-program detector that scored healthy programs as dead

WO-0.2's first trial run reported a DEAD program. It was an artifact: the chain's final-hop
relation had no *other* stored value, so the "perturbed" copy was identical to the original.
"The answer did not change" said nothing at all, and it was being scored as death.

Those are now excluded as `not_perturbable`. Counting an unperturbed run as dead would have made
this metric cry wolf from day one — the mirror image of §6.1.

### 6.3 A world that cannot be re-verified, reported as 0% volatility

The volatility probe's first target was the WikiKG world. Its triples carry subject/relation/
object strings and **no QIDs**, so subjects resolve only by label — and its labels are decased and
despaced (`HD189733b`, `Soft Bank Group Corp`, `Charles IV Of France`). Worse, its relation
vocabulary is its own (`TIMELINE_EVENT`, `HAS_PART`, `IS_A`), not Wikidata property labels, so
even a subject that *does* resolve never states the relation being checked.

A 30-fact run over it **decided 0 of 30**.

Reporting that as "0% volatility" would have been precisely the false green the work order exists
to prevent. The probe now defaults to the encyclopedia facts, whose entity is a real name and
whose relation is already a Wikidata property label.

### 6.4 Two things that look like staleness and are not

Also from the volatility probe, and both would have inflated the rate:

- **Relabelling is not volatility.** Stored `Eric Harris` vs live `Eric David Harris` is one
  person under two names. Counted as agreement, tracked separately.
- **Misresolution is not volatility.** `date of birth` of `Robert Edward Lee` came back stored
  1807 → live 1942; `James Gordon Bennett` 1795 → 1963. Those are *different people with the same
  name*, not Wikidata correcting a birthday. On the first run that single case promoted
  `date of birth` to fetch-required at a **fabricated 20% rate**.

Differences are now recorded as `disputed` and kept out of both the rate and the fetch-required
computation until confirmed. Only confirmed moves promote a predicate.

### 6.5 A missing label read as missing verification

The Phase 1 audit found this, and it is the most instructive one because the wrong conclusion was
*written into the work order before it was checked*.

The raw counts looked damning:

    plan     37,588 rows    vm_ok = None    (every single one)
    chain    28,529 rows    vm_ok = True    (every single one)

The obvious reading — "the plan task carries no verification signal, so a retrain without `chain`
trains on nothing verified" — is **wrong**. The builder emits the plan record and the chain record
from *one* certified walk and skips any verdict other than `certified`. A plan row exists **only
because** that walk was gate-verified and its answer equalled gold. The `None` is a missing
label, not missing verification.

Measured by pairing each plan row with a chain twin on (question, gold): **18,190 of 19,404 v13e
plan rows (94%) have a certified chain twin.** The 1,214 exceptions are gen-2 rows certified by
gen 2's own process.

Both the builder and the audit now write the label. This changed no training row — the notebook
admits `vm_ok in (True, None)` and the filtered corpus passes 28,821 rows before and after — so it
neither blocks nor confounds the retrain. What it buys is a corpus that states what is true of
itself, so the next audit cannot draw the same wrong conclusion.

**The general form:** `None` is not `False`. An absent label and a failed check are different
facts, and a pipeline that conflates them will eventually delete good data or train on bad.

### 6.6 A baseline attributed to the wrong model

An earlier draft called **531/600** "the v13e bar". It is not. That number was produced by
`emitter_v12e.Q4_K_M.gguf` — the gen-2 model — scored on the gen-3 held split. It is the
*incumbent's* score, the bar v13e was meant to beat.

And there is no `emitter_v13e` or `v13f` GGUF anywhere: **the with-`chain` corpus had never been
trained.** Scoring the treatment arm against 531/600 alone would have confounded "dropping
`chain`" with "adding all of gen 3's free-text data on top of gen 2".

That is why the comparison needs **two runs** — a `v13e` control and a `v14e_nochain` treatment,
same arm, same cell — not one.

### 6.7 A metric that scores a correct program zero

Found 2026-09-15, on the v13e control run itself.

The training notebook reports `exact_match_by_task`: string equality between the generated program
and the reference. It reported **arithmetic 0.0**.

The VM scored the same generations **6 of 8 correct**.

The gap is entirely artifact. References are named `program GSM1669 implements ISolver` with an id
drawn from the source corpus. Nothing in the prompt determines it, the model cannot guess it, and
every generation is therefore a string mismatch no matter how correct the body is. Two of the
eight were byte-identical to their reference once the name was normalised; the rest computed the
right number by a valid different route (`40 + 56` where the reference wrote `56 + 40`).

A second instance of the same class in the `kernel` task: the decision template carries confidence
literals (100 on approve, 10 on reject) that the prompt never states. A generation that takes the
correct branch but writes 70/20 is scored wrong while making the identical decision.

**Neither artifact is a wrong answer, and together they dominate the reported number.** An
arm-vs-arm comparison read off `exact_match_by_task` would largely be measuring how often each arm
guessed an unguessable token — and because the no-chain arm changes the row mix, the artifact rate
moves on its own between arms.

`validation/score_val_generations.py` now re-scores any run's `val_generations.json` by executing
each program and comparing the VM's return value to gold, with the reference run alongside as a
per-item control. It tallies three ways, not two.

### 6.8 The eval sample moved with the independent variable

Found 2026-09-15, on the arm comparison this whole phase exists to run — and the worst of the
ten, because it silently invalidated the experiment rather than a single number.

The notebook draws its eval sample like this:

    rng = random.Random(1)
    sample = [r for t, rs in sorted(by_task.items())
                for r in rng.sample(rs, min(N_PER_TASK, len(rs)))]

**One RNG, consumed across tasks in sorted order.** The control draws arithmetic, then *chain*,
then kernel, plan, role_binding. The treatment has no `chain` — so from its second task onward it
consumes a different slice of the same stream, and every task after the first samples different
questions.

Verified on the real corpora: the per-task pools hold **identical ids in identical order** for
every surviving task. The divergence is entirely the sampler.

| | shared eval items |
|---|---:|
| old, one shared `Random(1)` | **8 of 32** |
| new, one RNG per task seeded by task name | **32 of 32** |

Only `arithmetic` survived, because it is drawn first, before the streams diverge. That is why
the one task with full overlap is also the one showing zero difference between arms — and why the
kernel "improvement" is between disjoint question sets.

The fix is one line: seed a fresh `Random` per task from the task's own name, so a task's sample
depends only on that task's pool. Now in `notebooks/_build_v14e.py` with the incident in the
comment.

**The general form, and it is the sharpest instance of the §6.9 pattern:** the eval sample was
derived from the corpus, and the corpus was the independent variable. Changing what is being
tested changed what it is tested on. Any A/B where the eval set is computed from the treatment
has this bug available to it.

### 6.9 A token whose spelling was the measurement

Found 2026-09-15, in §5.7's own first run. Relations were relabelled to `r_41027`-style
tokens. The emitter drops the underscore and emits `r 41027`, so a relation it had copied
correctly failed to match the store on spelling alone. Switching to pronounceable tokens
(`damuzo`) moved the result from 13/300 to 74/300 — **5.7×, from nothing but the token's
shape.**

`normalize()` preserves the underscore, so the harness was innocent; the model's tokenizer
did it. That distinction does not rescue the number. 0.043 would have entered the record as
a capability measurement of an architecture, and it was a measurement of a naming choice I
made in five seconds.

**The general form:** when an experiment injects a synthetic symbol, the symbol's *form* is a
free parameter, and a free parameter nobody justified is a confound. Vary it once before
trusting the result.

### 6.10 A refusal that named the last symptom

Found 2026-09-15, in §5.9's own first run. All 226 refusals in the whole-chain arm reported
`retrieval_exhausted`, and retrieval had found exactly the right fact every single time.

The chain was real. The VM rejected each binding as below τ. The verify-stage repair then did
what it is supposed to do — banned the weakest hop's fact and re-walked — and with a one-record
store there was nothing else to find, so the second walk returned empty and *its* failure became
the reason. Each stage behaved correctly and the label came out wrong.

What makes this one worth its own entry is the direction of the error. The misattribution pointed
**away from the finding**: a reader of that log would have concluded the store was too small, gone
looking for more facts, and never found the depth cliff at all. §5.9 exists because the similarity
was measured directly rather than inferred from the refusal reason.

The fix keeps the first attempt's verify clauses across the retry and names them:
`hop0:below_tau(0.0430<0.2202)`, `symbol_mismatch`, `control:not_below_tau`. Re-running changed
every reason and no number — which is the signature of a labelling bug rather than a behaviour
one. A genuine retrieval failure still says `retrieval_exhausted`; relabelling the case the old
name was right for would have been its own lie, so a test pins it.

**The general form:** in a pipeline with a repair stage, the reported failure is whichever stage
failed *last*, and that is almost never the one that caused it. If a stage can mask an earlier
stage's verdict, it will.

### 6.12 The lesson taken at the wrong moment

Found 2026-09-15, by exp_r36, in code written the same afternoon.

Cubby-Man's danger radius used to be a table — `1 if fear < 1.0 else 2 if fear < 2.4 else 3` —
with three numbers nobody had measured. Replacing it with something he *learns* is the whole
"no-retraining" idea in miniature: the berth he keeps should come from what has actually caught
him. So the new rule was "record how far off the ghost was when it caught you, and keep at least
that much room."

It never moved. Six catches, eight catches, the radius stayed at 1.

The reason is embarrassing once seen: **a ghost that has just caught him is on top of him.** The
distance at the moment of capture is always 0 or 1, by definition of capture. Every lesson was
"it was touching me", which is true, useless, and unfalsifiable. The rule was learning from a
variable that cannot vary.

The informative distance is the one at the last moment he could have *acted on it* — where the
threat was when he chose his move, one tick earlier. Same rule, same data, one step back in time:
the radius now moves with the catches (1 → 2 over two of them, 1 → 4 over six) instead of sitting
at 1 through eight.

A table of invented constants at least looks arbitrary. A learned quantity measured at the wrong
instant looks principled and is worse: it produces a number, the number is stable, and stability
reads as convergence.

**The general form:** when something is learned from an outcome, check that the signal still has
variance *at the moment it is sampled*. Measure at the last point where the agent could have
changed the outcome, not at the point where the outcome is already decided.

### 6.13 The copy that passed every test

Found 2026-09-15, by exp_r37, in a guard written twenty minutes earlier.

Cubby-Man's thoughts used to come from a table of about thirty authored phrasings, with the
model asked to rephrase one. Nick's instruction was to stop that — *let it talk to see what it
will do as it receives information* — so the host started handing over the step's percept record
and letting the model write the sentence. The guard was the obvious one: **everything in the
sentence must come from the record.** Every figure, every cell name, every content word.

First run: 11 sentences kept, 0 refused, 0 invented. A 100% grounding rate.

The transcript:

> `# Here is what I just perceived in the maze: - about: invented a move - i am at: level-1 cell 0-1-0 - name: JUMP - i`

It was reproducing the prompt. Every single one.

Of course it scored perfectly. **A verbatim copy is maximally grounded** — there is nothing in
it that did not come from the record, by construction. The guard was measuring overlap with the
source and rewarding the one output that is pure overlap. The number went up as the behaviour
got worse, which is the property that makes this class of bug survive review: the verdict said
PASS, and only reading the transcript said otherwise.

The fix is a clause the guard had no reason to contain until the failure existed: the longest
run of consecutive tokens shared with the record, capped at six. `level-1 cell 0-2-0` is four
tokens and he must be able to say it; twenty is recitation. Plus the shapes a copy has and a
sentence does not — a leading `#`, a bullet, a `key:` field marker.

The same run produced a second one, smaller and in the opposite direction. On a ghost-free
level he said *"without triggering the ghost."* No invented figure, no invented cell — "ghost"
is neither — so every clause passed. Grounding had been defined over numbers and names, and
**entities were not in the vocabulary of the check at all.**

**The general form:** when a guard tests that output resembles its input, ask what the maximally
resembling output is, and whether you would accept it. If the degenerate case scores best, the
metric is inverted. And when a check enumerates what may not be invented, the list is a claim
about what *kinds* of thing exist — write it down and go looking for the kinds you left out.

### 6.14 The pattern

Nine of these twelve are the same shape: **an instrument inherited a property from the thing it
was measuring**, and therefore could not discriminate. The τ alarm inherited τ. The dead-program
detector inherited the store's sparsity. The volatility probe inherited WikiKG's label scheme.
`exact_match` inherited the corpus's arbitrary identifiers. The eval sampler inherited the
corpus's task *set* — the independent variable itself. The relabelling probe inherited the
tokenizer's opinion of an underscore.

The other three are each their own shape and worth keeping separate. §6.10: nothing was inherited
and no stage misbehaved — the pipeline simply reported its last failure instead of its first, and
a correct system described itself incorrectly. §6.12: the code was right, the data was right, and
the *sampling instant* was wrong, so a learned quantity sat at a constant and looked converged.
§6.13: the metric was inverted — the worst possible output scored best — and the verdict line
said PASS while the transcript said the model was reciting its own prompt.

The one defence that worked every time was cheap: **run the instrument on data known to be
healthy, and require it to say so.** An alarm that fires on a clean run is broken, not sensitive.
§6.10 adds a second: **when a measurement is available directly, take it directly** rather than
inferring it from a label some other stage wrote. §6.12 adds a third: **check the signal still
varies where you sample it** — a learned number that never moves is a measurement problem before
it is a learning one. §6.13 adds a fourth, and the cheapest of the lot: **read the transcript,
not the verdict.** Three of these were visible in the first ten lines of output to anyone who
looked at the output instead of the summary.

---

## 7. The language itself was lying

Phase 1 audited CubeLang, on the reasoning that a bug in the language the thinking is written in
is worse than any bug in the model — the model's output is checked, the VM's is not.

Four silent wrong-answer bugs were found. All four had the same character: the program compiled,
ran, reported success, and returned a wrong value. None of them could be caught by any instrument
in §5, because every instrument is downstream of the VM.

**String comparison was unconditionally true.** `op::COMPARE` forced both operands through
`resolve_i64`, and `Value::as_i64` returns 0 for every `Value::Str`. So every string compared
equal to every other string. `if (intent == "question")` took the then-branch for any intent
whatsoever. This is the root one.

**`match` ran every arm.** Arm selection was never compiled; all bodies executed in sequence, so
the last arm's effect won. Fixing this *without* fixing string comparison first would have traded
"every arm runs" for "the first arm always wins" — strictly worse, because it looks correct.

**`container[key] = value` emitted zero bytecode.** Parsed, then silently discarded.

**`+=` was compiled as plain assignment.** `total = 10; total += 5` returned **5**.

347 tests green after the fixes, including new suites for arm selection, indexed and compound
assignment, and recover scoping, plus a pinned-output test over `examples/` so the committed
programs cannot drift from the VM again. Six stale tests had asserted the *old* broken behaviour
and were flipped rather than deleted, each annotated with what it used to pin and why that is no
longer true.

**The structural finding underneath all of it:** there is no expression-lowering pass and no
temp-register allocator in that compiler at all. `compile_expr_operand` encodes atoms. CubeLang's
real surface is a **three-address register machine** — and that is the surface to keep, not a gap
to fill. It is the EVM analogy taken seriously, and a flat fixed-arity instruction grammar is far
easier to constrain with a generated grammar (WO-2.2) than a recursive expression tree would be.

**What this says about the method.** A silently-wrong VM is the one failure the architecture has
no defence against, because invariant 1 makes the VM the only judge. There is no second opinion by
design. That makes VM correctness a different category of concern from model quality, and it
argues for the bytecode verifier in WO-2.4 — *reject any program where a source statement maps to
zero emitted instructions* — which subsumes most of this class including the instances nobody has
found yet.

---

## 8. The experiment in flight

The role-vocabulary finding (§5.3) named a mechanism: the `chain` task is the sole source of the
per-relation identifiers, and in the training data the facts are handed to the model in the
prompt — so it is transcribing, not discovering. Work the host already does deterministically.

Dropping `chain` **does** flatten the vocabulary, and by the full amount: the filtered corpus has
**17 distinct roles, down from 412** (−95.9%), and what remains are the generalizing shapes —
`SEED`, `HOP1`, `HOP2`, `ASK`, `ACTION`, `AGENT`.

So the mechanism is real. The **prediction** is not yet tested:

> **Falsifiable prediction.** Dropping `chain` and the decorative attributes improves plan
> accuracy and cross-task retention with **no loss of VM-verified answers**.
>
> **Kill criterion.** If the VM-verified answer rate drops, `chain` was doing something the audit
> did not see.

Two runs, because one would confound the variable with the corpus: a **`v13e` control** and a
**`v14e_nochain` treatment**, same LoRA recipe, same arm, same cell, changing only the version.

### Control result, re-scored

The v13e control has finished. Re-scored with `score_val_generations.py` (§6.7), on the subset
where `gold` is the program's own return value:

| task | reported `exact_match` | VM: correct | refused | **wrong** |
|---|---:|---:|---:|---:|
| arithmetic | 0.0 | 6/8 | 1 | 1 |
| chain | 1.0 | 8/8 | 0 | 0 |
| kernel | 0.875 | 7/8 | 0 | 1 |
| **total** | | **21/24** | 1 | **2** |

All 24 reference programs reproduce gold under the VM, so the harness is sound and these are
honest model numbers. Final training loss 0.1868 over 40,560 rows, 2 epochs.

The three misses are three different things, and separating them is the point of scoring this way:

- **One genuine wrong answer.** A pizza-delivery word problem: the model billed a flat $2 delivery
  surcharge as `4 × 12 = 48` instead of `2 × 2 = 4`, and returned 108 against a gold of 64. It
  compiles, it runs, `verify()` in the emitted template returns `true` unconditionally, and
  nothing downstream can catch it. **This is the kill line, and it is a real breach.** One in 24.
- **One refusal.** The nine-step program hit the eval's 500-token generation cap mid-body and
  failed to parse. That is a loud refusal — a result, not a wrong answer. The cap is now 900.
- **One undetermined constant.** The `kernel` miss (§6.7) takes the correct branch and differs
  only in a confidence literal the prompt never states. Not a wrong answer in any sense that
  matters.

`plan` (8/8 exact) and `role_binding` (5/8) are not VM-checkable here — `plan` emits a walk
skeleton whose gold is the answer the disposer resolves downstream, and `role_binding`'s gold is
`None`.

**How the treatment must be read.** Not off `exact_match_by_task`, for the reasons in §6.7. The
comparison is the VM tally, and specifically the three columns separately: *correct* is the
capability claim, *refused* is free, and any increase in *wrong* kills the arm regardless of what
happens to the other two.

### Treatment result — and why it does not decide the work order

`v14e_nochain` finished 2026-09-15. 24,817 train rows from a corpus of 17 distinct roles, against
the control's 40,560 rows from 176. Final loss 0.2342 against the control's 0.1868 — higher, and
expected: the easy transcription task is gone, so what remains is harder per row.

The first reading of the two score logs looked like a win — kernel 7/8 → 8/8, and its wrong
answer gone. **That reading is false, and the reason is the eighth instrument failure (§6.9).**

Diffed by item id, the two arms were scored on almost entirely different questions:

| task | control n | treatment n | **shared** |
|---|---:|---:|---:|
| arithmetic | 8 | 8 | **8** |
| chain | 8 | 0 | 0 *(dropped by design)* |
| kernel | 8 | 8 | **0** |
| plan | 8 | 8 | **2** |
| role_binding | 8 | 8 | **1** |

On the only task with full overlap, the arms are **identical**: 6 correct, 1 refused, 1 wrong,
the same wrong answer (the pizza problem, 108 against a gold of 64) from both. The emitted
programs differ textually; the verdicts do not differ at all. The two shared `plan` items are
byte-identical between arms and both exact-match in both.

So the honest reading:

- **The kill criterion is not triggered.** VM-verified answer rate did not drop on any comparable
  item. `chain` is not obviously carrying something the audit missed.
- **The prediction is untested.** "Improves plan accuracy" cannot be measured here — `plan` is at
  1.0 in both arms, a ceiling, on 8 items of which 2 are comparable. "Cross-task retention" rests
  on one shared `role_binding` item.
- **The mechanism landed.** The treatment trained on 17 roles instead of 176. Whether that buys
  generalization is WO-2.5's question, and this eval cannot see it.

Eight comparable items is not evidence of anything except the absence of a catastrophe. The
decision needs the fixed-question evaluations that do not move with the corpus.

### The gen-3 held split — the evaluation that does decide it

`validation/exp_r17_gen3_heldout.py`. 600 questions from the held tenth of the gen-3 split —
entities no training record used — through the emitter, the gate, and the VM. The question set is
fixed by seed, independent of the corpus, so it does not move when the treatment moves.

Three emitters, seed 7, same store (563,062 facts):

| arm | plan | verified | correct | **wrong** |
|---|---:|---:|---:|---:|
| `emitter_v12e` — the incumbent | 599 | 531 | **531** | 0 |
| `emitter_v13e` — the control | 600 | 599 | **599** | 0 |
| `emitter_v14e_nochain` — the treatment | 600 | 597 | **597** | 0 |

**v12e reproduces 531/600 exactly** — the documented bar, to the item. That makes the harness
reproducible and the other two rows trustworthy.

**Both new arms beat the incumbent by roughly 11 points.** The with-`chain` corpus had never been
trained before this cycle (§6.6), so this is the first measurement of what gen 3 is worth: the
incumbent's 69 failures become 1 and 3.

Repeated at seeds 8 and 9:

| arm | seed 7 | seed 8 | seed 9 | mean |
|---|---:|---:|---:|---:|
| v13e control | 599 | 600 | 600 | **599.7** |
| v14e treatment | 597 | 599 | 600 | **598.7** |
| delta | −2 | −1 | 0 | **−1.0** |

**0 wrong answers in all seven runs — 4,200 questions.**

### What the difference actually is

Not a rate. Across all three seeds, exactly **three distinct questions** ever fail:

| question | v13e | v14e |
|---|---|---|
| `When was the sibling of France Killy Sister born?` | seed 7 | seed 7 |
| `In which district is Seaside-Seattle Seattle located?` | — | seeds 7, 8 |
| `What award did Seven Against Thebes Play receive?` | — | seed 7 |

Every one carries a **mangled entity name** — `France Killy Sister`, `Seaside-Seattle Seattle`,
`Seven Against Thebes Play`. These are the decased-label and `' of '`-split artifacts already on
record, leaking out of fact parsing into the question text. The first is failed by all three arms,
v12e included. The two that separate the arms are `retrieval_exhausted` — **refusals, not wrong
answers** — and they are deterministic: the same question fails whenever it is sampled.

So the whole v13e-vs-v14e gap is two malformed-entity questions that v14e refuses and v13e
answers.

### The verdict: the criterion was wrong, not the arm

Read literally, **the kill criterion triggered**: the VM-verified answer rate dropped, by 1.0 of
600 on average. The work order says such an arm is *abandoned rather than explained*, and that
rule exists for good reason, so the trigger was recorded rather than argued away — and then the
criterion itself was examined.

It does not distinguish the two outcomes this whole project is built on separating. The loss is
entirely into **refusals**, on **corrupted inputs**, with **zero wrong answers in 4,200
questions** — bought in exchange for taking the role vocabulary from 412 identifiers to 17, the
memorization ceiling described in §5.3.

**Amended, owner decision 2026-09-15.** The rule is now general and binds every work order, not
just this one:

> **A kill criterion may not treat a refusal and a wrong answer as the same event.** State two
> clauses, separately: the arm dies if **wrong answers appear**, and it dies if the VM-verified
> rate drops by **more than one percent**. A drop inside that band, entirely into refusals, is a
> cost to record — not a kill.

The reasoning is worth stating plainly, because the original wording read like rigour. A criterion
that kills an arm for converting an answer into a refusal **rewards guessing** — and guessing is
the single behaviour the kill line exists to forbid. Written as it was, it would have enforced the
opposite of the invariant it was serving.

It is the same failure as every instrument in §6, one level up: **an acceptance rule that collapses
two outcomes the system exists to distinguish will eventually enforce the wrong one.** §6 is about
instruments that could not measure what they claimed; this is a *decision rule* that could not
decide what it claimed. The defence is the same — check it against a case whose answer you already
know.

Recorded in `WORK_ORDERS.md` under "Before you touch anything", with the original wording struck
through in place rather than deleted. WO-2.5's criterion has been rewritten into the two-clause
form ahead of being run.

**v14e_nochain is adopted** under the amended rule: no wrong answers, a drop far inside one
percent, and the WO-0.3 vocabulary ceiling removed.

**One thing this result costs us.** v13e scores 600/600 on two of three seeds. The held split is
at its ceiling and can no longer discriminate between these arms — the next comparison needs a
harder evaluation, which is exactly what WO-2.5 (relabelling invariance) is for.

---

## 9. The path

Each phase is gated on the one before. The gating is not bureaucracy — it is the §4 principle
applied repeatedly: do not build on a claim an earlier phase could still falsify.

### Phase 1 — bugs, regardless of which architecture wins

Done, except the retrain. The four wrong-answer bugs (§7); `recover` scoped to the frame's own
bindings, with the before/after in §5.4 showing no recalibration was needed; the SFT mix audited
and both arms built.

### Phase 2 — the generalization spine

This is where the §5.3 finding gets addressed structurally. The theme: **move relation identity
out of the model's token vocabulary and into data the host supplies per request.**

- **WO-2.1 — per-request capability manifest.** The host supplies the admissible relation and
  entity strings plus direction, arity, and functional-or-multi-valued metadata. The string is
  evaluated *exactly once*, as a key into the manifest index; a miss is a hard error, never a
  nearest-neighbour fallback. "Evaluated, not parsed", made mechanical.
- **WO-2.2 — a grammar generated from the engine registry.** Admit the execute surface only.
  Arguments restricted to literals and bare variables — which incidentally fixes the last
  silent-garbage path on the emitter side without touching the compiler. Relation literals drawn
  from the per-request manifest, so an unknown relation is **undecodable** rather than rare. *The
  grammar is a build artifact; a hand-edited grammar fails the build* — the risk is drift, and
  drift fails in the safe-looking direction.
- **WO-2.3 — a witness test per grammar token.** For every admitted opcode, a program whose
  observable return value changes when that opcode is removed. Any token with no witness is
  demoted automatically. This is exactly the test whose absence let 24 trace-only opcodes into the
  language. 0.7 ms each.
- **WO-2.4 — bytecode verifier at the loader boundary.** JVM-style, checking the compiled artifact
  rather than the source — the only place the parse/execute gap is visible *as data*. Reject any
  program where a source statement maps to zero emitted instructions; reject placeholders; reject
  `recover` on a role with no dominating `bind`; require every frame reaching `return` to be fully
  bound.
- **WO-2.5 — relabelling invariance, the actual generalization test.** Rerun a relation-disjoint
  split with every relation label replaced by a fresh opaque token, direction and arity preserved.
  **Genuine structural generalization is invariant under bijective renaming; memorisation of a
  relation table collapses.** A disjoint split alone is necessary and not sufficient.

  With two controls, both guarding against the obvious cheat. A **null control** — a syntactically
  valid relation with zero facts — where any spoken answer is a kill-line breach. And a **decoy
  control**, because once the host supplies the manifest, *a pure copier passes the
  unseen-relation test* and correctness migrates silently into manifest construction.

  *Kill criterion:* relabelled accuracy below 80% of labelled accuracy after three targeted
  rounds means the string-argument form is not carrying the generalization.

Note that WO-2.2 as written excludes `match` and indexed assign from the grammar — both of which
now work (§7). That is a live tension, not an oversight; see below.

### Phase 3 — new capability

- **WO-3.1 — host-owned hypothesis search.** `ASK`/suspend/resume recovers observation-conditioned
  continuation, but not hypothesis search, retraction, or re-representation. The obvious fix — put
  the search in the program — is unavailable here, because branching is the part that was broken.
  So programs stay branchless and the agenda lives in the host: `propose`, `refute`, `retract`,
  with the host owning dedup, cycle detection, a node budget and termination, so **exhaustion
  exits as a refusal rather than a guess.**

  **This contests the Phase 1 decision to implement real `match` arm selection.** If search belongs
  in the host, VM-level branching is investment in the layer most likely to produce plausible
  wrong answers, and rejecting `match` at compile time is cheaper and better aligned. Recorded
  here unresolved, because it is a genuine architectural fork and not a detail.
- **WO-3.2 — the co-mention tier.** A graph check, not a prior: it reads a link between two
  entities the *asker* named. Binding rule — the selector may read labels, aliases, descriptions,
  classes and *which relations a candidate has*, never *the objects of those relations*. Presence
  is type evidence; value is truth.
- **WO-3.3 — multi-verify monitor and decoy audit.** Both required before any prior decides in
  production. They instrument a self-confirmation failure that is invisible to every existing
  metric *because the kill line stays green while it happens*.
- **WO-3.4 — date the preloaded world.** The as-of machinery exists and the data does not reach
  it: every preloaded fact is undated until a live fetch restates it. The capability exists and
  the data does not reach it.

---

## 10. What would prove this wrong

Stated in advance, so that meeting them is not negotiable after the fact.

| # | Claim | What falsifies it | Status |
|---|---|---|---|
| 1 | The emitter contributes causally | Ablation gap inside the noise floor for two consecutive rounds | **Holding.** 0.4125 vs 0.0618 |
| 2 | Emitted programs read their inputs | Liveness below 95% | **Holding.** 59/59 |
| 3 | The interface generalizes across relations | Role vocabulary grows with relation coverage | **Failing.** 95–98% per-relation; Phase 2 is the response |
| 4 | Accepted bindings are separable from noise | Separation to the control role collapses toward 1× | **Was breached by depth, now held by a shape.** 14.3× at 1–3 hops, but 0.57× at 5 and below zero at 7 under the shipped whole-chain program — the criterion fired and nothing was watching, because no instrument went past 3 hops. Two hops per frame holds 8–18× to depth 12 (§5.9) |
| 5 | Dropping `chain` costs no verified answers | VM-verified answer rate drops in the treatment arm | **Triggered, marginally.** −1.0 of 600 over 3 seeds on the held split — two malformed-entity questions, both refusals, 0 wrong. See §8 for why the criterion, not the arm, is what should change |
| 6 | Structure, not a relation table, carries it | Relabelled accuracy below 80% of labelled after 3 rounds | **Failing.** 0.247 / 0.262 of labelled (§5.7). The manifest alone does not fix it (§5.8) — the grammar has to enforce |
| 8 | Each brain zone earns its name | A zone's ablation sits inside the noise floor, or it has no production caller | **Mixed.** 3 zones load-bearing; 3 seams inside the floor; 3 modules declare WIRED with 0 callers (§5.6) |
| 9 | Reasoning depth is a property of the machine | A change to the program *shape* lifts it | **Falsified, 2026-09-15.** The limit was `build_chain_program`, not the VSA: 62 → 246 of 288 correct at depths 2–10 (§5.9) |
| 7 | Zero wrong answers spoken | Any confidently wrong spoken answer | **Breached in eval.** The same 1 wrong answer in **both** arms' val sets (§8) |

Row 7 deserves its own note. The ablation run (§5.1) recorded **0 wrong answers across all four
arms and 240 questions** on the serving path, which is the path the kill line governs. §5.9 has
since added **2,592 questions at depths up to ten, with the emitter emitting chains of the wrong
length most of the time, and 0 wrong answers** — the strongest evidence for this row so far, and
the first collected somewhere the model is out of its depth rather than inside it. The breach
in §8 is on the *training eval* — a generated program scored against gold, not an answer spoken to
a user. They are different surfaces and the distinction is real, but it is not a reason to
discount it: the same emitter on the serving path produces answers the host walks and the VM
certifies, and the reason that surface stays clean is the host and the VM, not the emitter. §5.1
says so directly — the grammar scores 58 where the emitter scores 33.

The honest summary of row 7: **the kill line holds where it is enforced, and the emitter is not
what is enforcing it.**

---

## 11. Where this is going

Three things have to become true, in order.

**The emitter has to beat the grammar on the grammar's own ground.** 33 to 58 on canonical
phrasing is the gap. Closing it while *keeping* 66 across the rephrasings is the whole problem —
closing it by memorising canonical phrasings would be a regression disguised as progress, and
§5.1's table is the instrument that would show it.

**Relation identity has to leave the token vocabulary.** While roles are minted per relation, the
ceiling is the relation set in training, and no amount of scale moves it. Phase 2 is a structural
fix, not a tuning pass, and WO-2.5 is the test that says whether it worked.

**The VM has to stop being able to lie.** §7 found four silent wrong-answer bugs in one audit.
The bytecode verifier (WO-2.4) is the general form of that audit, run at load time on every
program forever — the only defence that scales to the bugs nobody has found.

If all three land, the claim in §1 becomes testable end to end: a question in a phrasing nobody
wrote a rule for, about a relation under an opaque name, answered by a program the model wrote,
certified by a VM that never knew a model was involved, at zero token cost — or refused.

That is the thing worth building. Everything on this page is either evidence that it is reachable,
or a record of an instrument that would have told us we were already there.

---

## Changelog

- **2026-09-15** — Created. Phase 0 complete and baselined; Phase 1 complete but for the retrain.
  v13e control trained and re-scored; `score_val_generations.py` added after `exact_match` was
  found to score correct programs zero (§6.7). v14e_nochain treatment running.
- **2026-09-15, later** — Both arms finished. The val-split comparison does **not** decide WO-1.3:
  the eval sampler's single shared RNG made the two arms score different questions (§6.8),
  leaving 8 comparable items on which the arms are verdict-identical. Sampler fixed (per-task
  seeding, 8/32 → 32/32 shared).
- **2026-09-15, gen-3 held split** — The fixed-question evaluation, 3 seeds × 600 questions ×
  3 emitters. v12e reproduces 531/600 exactly; **v13e 599.7 and v14e_nochain 598.7 mean**, so
  both new arms beat the incumbent by ~11 points and the with-`chain` corpus is measured for the
  first time. **0 wrong answers in 4,200 questions.** The arms differ by two specific
  malformed-entity questions that v14e refuses.
- **2026-09-15, the manifest (WO-2.1)** — Built, and the prompt-only form measured: it recovers
  nothing (six arms within five questions), because the emitter ignores an admissible set it was
  never trained to read. The decoy control is inconclusive for the same reason. But the gate
  converts late `retrieval_exhausted` failures into immediate `unknown_relation` ones — 56 → 1 —
  so *evaluated, not parsed* works and the emitter is the part that does not participate. This
  settles WO-2.2 as the mechanism rather than an optimization.
- **2026-09-15, relabelling (WO-2.5)** — The gating measurement, run without the manifest.
  Both arms fail the rate clause (0.247 / 0.262), so the string-argument form does not carry
  the generalization and WO-2.1 is load-bearing. But the diagnostic is the result: v13e
  recalls the memorized relation 37 times, v14e **once** — so WO-1.3 bought what it claimed,
  invisibly to a score that moved 0.3%. 0 wrong and 0 null-control breaches in 1,200
  questions. A token-spelling artifact worth 5.7× was found and corrected mid-run (§6.9).
- **2026-09-15, the zone ablation (WO-0.6)** — Every zone measured, not just the emitter. τ_vm is
  the only one whose removal produces a wrong answer; everything else fails into refusals, and
  loosening τ makes the *chain* refuse downstream rather than speak. Three modules declare
  `Wiring.WIRED` with zero production callers — found by a new bidirectional CI guard, because
  the existing one is named after that exact trap and only checks that the label exists.
- **2026-09-15, the criterion amended** — Owner decision. A kill criterion may no longer treat a
  refusal and a wrong answer as the same event: two clauses, separately, and a drop under one
  percent that lands entirely in refusals is a cost, not a kill. The original wording would have
  rewarded guessing while reading like rigour. **v14e_nochain is adopted.** The held split is now
  at its ceiling (v13e 600/600 on two seeds) and can no longer separate the arms — WO-2.5, whose
  criterion has been rewritten into the two-clause form, is the next real test.
- **2026-09-15, depth (WO-2.6)** — CLUTRR and ProofWriter acquired, and the first instrument to
  look past three hops immediately found a wall: one frame holds the whole chain, so the true
  binding halves per hop, meets the noise floor at six and crosses below it at seven. Falsifying
  row 9 took one parameter — two hops per frame, each with its own control — and the curve went
  flat: **62 → 246 of 288 correct at depths 2 to 10**, separation 8–18× to depth 12. With the
  emitter in the loop it generalizes about one hop past its training data and is gone by seven,
  failing on chain *length* rather than on relations, and the disposer refuses the wrong-length
  plans before a hop is walked. **0 wrong and 0 null-control breaches in 2,592 questions.** The
  change is measured, not switched on: `chunk` defaults to 0 and a test pins the shipped shape
  byte-for-byte. One more instrument caught lying on the way (§6.10). ProofWriter needs a
  rule-application engine that does not exist, which is a gap to state rather than a score to post.
- **2026-09-15, the hardcoded rules (WO-2.10)** — Owner: *"we need to remove all the hardcoded
  rules."* An AST audit (exp_r35) found the file admitting the biggest one in its own docstring:
  the ASK offered **only the maze's legal moves**, so cubby-man could never walk into anything,
  and the probe that "learned walls" was a VM rejection wearing a collision's clothes. Now the
  offer is *what his body can do minus what his own map has ruled out*, the **world** resolves
  the attempt and may refuse, and the refusal is the percept. Ghost positions became a **belief**
  with a range, a line of sight, and a staleness; the pellet map became a sensor; the four tuned
  threat clauses collapsed into one condition over a berth he **measures from what caught him**;
  and `if not env.remaining` — the whole level's pellet count, read 446× from inside his own step
  — became one published bit. Levels 1–3 now run ghost-free, per the owner: his classroom.
  Measured at 300 steps/arm across three arms including a **blind** one that can only learn by
  colliding: **0 false obstacles, 0 oracle reads in the decision path, 0 sightings beyond the
  sensor**, he holds 24% of the true ghost positions, and he still plays (blind reaches level 3).
  The experiment caught two bugs review had not, the second of them a learned quantity sampled at
  the one instant where its signal cannot vary (§6.12).
- **2026-09-15, letting him talk (WO-2.11)** — Owner: *"the model should say something not
  hardcoded strings like right now, let it talk to see what it will do as it receives
  information."* The host no longer writes a sentence for the model to rephrase: it hands over
  the step's **percept record** and the model writes the sentence, with a guard that refuses
  rather than repairs. The first guard was worthless — it scored **100% grounded on a model that
  was reciting its own prompt**, because a verbatim copy is maximally grounded by construction
  (§6.13), and a second clause was missing entirely: *"without triggering the ghost"* on a
  ghost-free level is neither an invented figure nor an invented place. With both closed, and
  across three prompt shapes and two models, **0 invented and 0 recited in every arm** — the keep
  rate is what moves (45% / 6% / 29%), and almost every refusal is prompt echo rather than
  hallucination. What he says now is his: *"I'm exploring the open path to the right to find more
  pellets and learn the layout of the maze."* Also, Lövheim's social corners (contempt, shame)
  now read neutral in a world with nobody in it — they were firing on hormone geometry alone and
  reporting an emotion he had no reason to have.
