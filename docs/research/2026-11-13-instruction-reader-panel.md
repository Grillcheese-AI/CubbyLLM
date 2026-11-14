# Panel: the instruction reader, and what it should write (2026-11-13)

The brief: `2026-11-13-instruction-reader-brief.md` (beside this file). Asked independently, same prompt, temperature 0.3; budgets per model as in `validation/panel_instruction_reader.py`. Model ids checked against OpenRouter's live list at run time. Cached under `standin/data/out/panel_cache/`.

**These are model opinions, not findings.** Nothing here is adopted until an order pre-registers it and its line is read. Cowork's own answer was sealed before this ran: `2026-11-13-instruction-reader-cowork-sealed.md`.

Cost reported by OpenRouter: $2.52. Answers: 8 of 8.

## openai/gpt-6.1-sol-pro

*15944 tokens out, 238.6 s, finish `stop`*

## 1. Diagnosis

Round 2 is not yet an acceptable reader. On the reported dev comparison, span and kind reach 0.889, below the 0.90 line; agreement is 0.685; false passes are 0.0935, above the 0.05 kill line (§§2, 4.3). Its results on unseen compositions and long instructions are unknown (§4.4).

The comparison does not isolate causes: training, grammar and decoding conditions changed together (§4.3). Lower held-out answer loss does not establish semantic correctness (§3.6).

### Negation

The strongest evidence points to the target representation and training distribution.

Of the 30 remaining reversals, 29 are negations and one is a mirrored comparison. They occur inside filters or position tests, at the outer count comparison, and at the size comparison (§4.5). This is not just failure to understand one word such as “unless.”

The gold presents equivalent meanings in different forms, without wording determining which form is used. It also associates some comparison directions much more often with a later `not`, and contains 548 nonzero lower-bound meanings against 1,648 upper-bound meanings (§4.6). These are plausible sources of learned shortcuts. They are not proof of causation.

Writing a program adds avoidable opportunities to place or omit a negation during lowering. That burden is structural to target A. Understanding the instruction’s polarity is not removed by shortening the target.

The data can address spelling inconsistency, directional imbalance, scope coverage and semantic supervision. Whether this model can meet the required accuracy after those changes is unknown. The numbers do not establish a model-size ceiling or show that greedy decoding is the principal cause.

### The join

The observed error is wrong attachment: the reader joins checks 1 and 2 where the instruction joins checks 2 and 3. In one observed case, all parts were individually correct (§4.5).

A free-form main function exposes an unnecessary binding problem. That is structural to A’s target. However, an incorrect group reference remains possible in a description target. Shortening alone does not establish the right scope.

There are only three inspected examples, not a measured rate (§4.5). The evidence cannot distinguish insufficient group training, numbering mistakes, grammar effects or a model limitation. Explicit group references and attachment supervision can address the problem; their effectiveness is unknown.

### Long instructions

Restarting at “check 1” and omitting the fifth function is consistent with a learned output pattern or bookkeeping failure. It is not evidence of a general context-length failure: only two examples were inspected, and round 1 did not have lower agreement on long rows (§§4.1, 4.5).

Writing and calling one function per part makes output length and bookkeeping grow with the instruction. That burden is structural to A. Incorrect labels and silently missing parts need not be structural: the host can assign identifiers and reject incomplete coverage.

Round 2’s length generalisation remains unknown (§4.4). We also do not know the instructions’ token lengths, so the 512-token attention window does not by itself explain these errors (§3.1).

**Overall:** redundant program generation is a structural burden, but the evidence does not show that “this model cannot write a program this long.” The remaining semantic failures are concentrated enough to justify controlled representation and data experiments before changing the model.

## 2. What the reader should write

I would make **B the primary experimental target**, compare it with A under controlled conditions, and retain C as a separate generality experiment.

Use the already proposed canonical spelling in every arm where applicable (§5). Otherwise a target comparison would also be a spelling experiment.

### A. Program under an expanding grammar

**Errors removed:** Grammar can exclude particular invalid constructions and bookkeeping choices. It does not, by construction, remove any entire semantic error class in §4.5. A legal comparison can have the wrong direction; a legal join can attach the wrong parts.

The nine rules already include some semantic restrictions (§3.5). Adding rules after inspecting failures requires a fresh evaluation set.

**Errors left:** Negation, attachment and omitted parts can all remain. Longer programs continue to expose more implementation choices than the requested meaning requires.

**Generality:** A can express checks outside the generator descriptions only where CubeLang and the available reply views support them. Its actual extra coverage is unknown.

**Unseen kinds:** It may compose available views into a new program. That is a possibility, not evidence that line 5 will pass. Unsupported, uncompiled or unenforced checks must be “asked.”

**Time:** The reported program averages 1,546 characters. Reading takes about 25 seconds at home, of which about 0.3 seconds is grammar time; rented-card batch results average 9.9 seconds on the reported dev subset (§4.7). Grammar optimisation alone cannot remove most of that measured time.

### B. Parts and groups, compiled by the host

Emit typed descriptions, explicit group references and coverage labels. The compiler should be a generic translation over views, columns, comparisons, quantifiers and Boolean structure—not host code keyed to particular English instructions.

**Errors removed:** Subject to a verified compiler and schema, the reader no longer writes loops, temporaries, signatures, calls or Boolean lowering code. Host-assigned identifiers can prevent numbering resets. Invalid references can be rejected.

This removes implementation opportunities for error, not semantic negation itself. The reader can still choose `>` instead of `<=`, negate the wrong scope, attach the wrong parts or fail to identify a part.

**Generality:** The current description language has the parts and groups listed in §3.3. It cannot be assumed to express every check that CubeLang can express. Whether it supports nested groups or other compositions is not stated.

That is a genuine possible loss, not merely a formatting change. Extend descriptions only with reusable operations over the existing views, not one host rule per instruction kind.

**Unseen kinds:** An unseen composition within the language remains expressible. An instruction outside it must be “asked.” B must still meet line 5’s compiling-coverage requirement; perfect accuracy on a narrow supported subset is insufficient.

**Time:** Descriptions average 316 characters, 4.9 times shorter than programs (§4.7). That is a character ratio, not a measured token or latency ratio. Token counts, compilation cost and end-to-end speed are unknown.

### C. Descriptions with a free-program fallback

Use descriptions where they represent the intended check exactly. Allow a free program only otherwise.

**Errors removed:** The description branch gets B’s implementation benefits. The free branch retains A’s failure modes. Routing creates another failure mode: forcing an unsupported meaning into a nearby supported description.

A failed description parse is not evidence that the instruction needs a free program. Conversely, successful parsing is not evidence of faithful meaning.

**Generality:** C can retain A’s expressible checks while shortening supported ones, provided routing is correct. That potential has not been measured.

**Unseen kinds:** Report description coverage, fallback coverage and “asked” separately. A fallback counts as enforced only when it compiles, is bound to the instruction and is run by the VM.

**Time:** It depends on routing and fallback frequency. Those are unknown. A sequential “try B, then A” design can add work on difficult cases; it should not be assumed faster.

### A set to measure lost generality

Build a common, frozen evaluation with separate strata:

- Held-out compositions expressible in the current descriptions.
- VM-checkable compositions deliberately outside those descriptions, generated from a separately specified program-construction process.
- Held-out instruction kinds and the public benchmark’s out-of-domain constraints required by line 5.
- Instructions not checkable through the stated views, which should be “asked.”

Gold checks must be generator-built or otherwise authored without model-written programs or targets. Benchmark sentences remain test-only.

Classify representability before seeing reader outputs. Report both the language’s representational ceiling and the reader’s achieved coverage. Otherwise failure to express a check and failure to recognise an expressible check become indistinguishable.

Freeze all arms before one preregistered evaluation. Do not use inspected failures to revise an arm and reread the same held-out rows.

## 3. Negation and direction

The following ranking is a judgment about expected gain per cost, not a measured result. These proposals are additional to §5.

All new numerical lines below are **proposed decision thresholds**, not forecasts. Proposal acceptance does not waive any line in §2.

### First: explicit scope supervision

**Change:** Make the template renderer record which text span supplies each filter comparison, outer bound, position comparison and negation scope. Add a training-only task to recover these alignments. Keep the enforcement target unchanged.

This tests whether the reader is attaching a correctly recognised relation to the wrong semantic slot. It does not add a host English-negation rule.

**Set:** A fresh held-out composition set stratified by the three reversal locations reported in §4.5, with balanced comparison direction and wording forms. Include independently generated replies near the relevant boundaries.

**Pass line:** At least a 25% relative reduction in one-reversal semantic errors against a frozen baseline, no reduction in enforced coverage, and false passes at most 2%.

**Kill line:** Reject if any acceptance condition fails.

**Cost:** Renderer metadata, an auxiliary training task and evaluation. Additional training cost and benefit are unknown. No serving-time task is required.

### Second: VM-outcome supervision during training

**Change:** Add a training task that predicts whether a supplied reply satisfies the instruction. Labels come only from the generator’s check run in the VM. Select replies that distinguish the gold from its single-reversal mutants.

This is semantic-outcome supervision, not another operator-token weighting scheme. At serving time, the VM still decides; model predictions of satisfaction are never enforcement.

**Set:** A separate fresh composition set, with replies produced by generation procedures held out from this auxiliary training. Score the emitted checks, not just the auxiliary predictions.

**Pass line:** At least a 25% relative reduction in negation errors, no reduction in enforced coverage, and false passes at most 2%.

**Kill line:** Reject if any acceptance condition fails, including improvement confined to auxiliary prediction accuracy.

**Cost:** More VM executions, training examples and training work. Their amounts are unknown. There need be no additional serving-time computation.

### Third: independently read satisfaction and violation, then abstain on disagreement

**Change:** Train two structured outputs: the satisfaction predicate and the violation predicate. Targets are mechanically derived from generator checks. At serving time, accept only when a generic normaliser establishes that they are complements.

Start with the description subset for which that comparison is implemented. Do not claim arbitrary program equivalence. Disagreement means “asked,” not choosing the more permissive result.

Both readings can share the same error, so agreement is not proof of correctness.

**Set:** A fresh set balanced across direction and negation scope, plus unseen compositions. Report selective coverage and false passes jointly.

**Pass line:** False passes at most 2%, enforced coverage at least 90% of the frozen single-read baseline, and all applicable unseen-kind requirements retained.

**Kill line:** Reject if any acceptance condition fails.

**Cost:** Another reading and a normaliser. Latency and training cost are unknown. This is primarily a safety filter, not a demonstrated comprehension improvement.

## 4. The join and long instructions

### Read groups as structure, not as generated control flow

Use explicit references to source spans and a typed Boolean structure. Keep linguistic interpretation in the reader and Boolean compilation in the host.

Define operand roles unambiguously:

- `A and B`: both.
- Inclusive `A or B`: either or both.
- `A unless B`: `A or B`, matching the example in §3.3.
- `A if B`: `not B or A`.

Do not infer exclusive-or from every occurrence of “either.” Ambiguous scope or an unsupported exclusive interpretation should be “asked.”

The existing flat groups may suffice for some instructions. General nesting is a proposed extension; current generator support is unknown.

**Compiler line:** Exact truth-table agreement for every supported Boolean constructor. Any mismatch kills that compiler version.

**Reader lines:** Retain at least 95% reply agreement and at most 2% false passes, with the hard kill above 5% (§2). Also report group type, operand attachment and whole-join accuracy separately.

Build attachment contrasts where only grouping changes while part meanings remain fixed. Their gold structures must be generator-built.

### Make coverage independent of model numbering

The host should assign identifiers from accepted source spans. The reader should not invent successive function names.

Maintain a coverage ledger over the whole instruction. Every sentence must be assigned to parts, “asked” or “not an instruction.” An incomplete output becomes an explicit incomplete reading, never a silently truncated successful check.

A ledger prevents silent loss; it does not prove that a “not an instruction” label is correct. Kind accuracy still needs testing.

**Coverage line:** Zero silently dropped sentences, as already required (§2).

### Test length without changing the question

Keep the prescribed training limit of four parts and the held-out five-to-eight-part read (§2). Do not train on those lengths and then claim to have passed length generalisation.

Use B to reduce output growth. If chunked reading is explored, first establish the coverage ledger, retain cross-chunk references, and resolve groups over the complete ledger. Independent clause checks cannot safely resolve an “unless” whose operands cross a chunk boundary.

**Length lines:** Agreement at least 90% of the in-length rate; zero silent drops; report false passes and coverage separately on long rows. Also apply the false-pass safety limits there. A relative agreement line alone can conceal a dangerous result when the in-length baseline is poor.

Performance beyond eight parts is unknown and needs a separate preregistered extension set.

## 5. The measure

The lines in §2 are necessary and should remain. They are not sufficient as aggregate reports.

### Report the full enforcement funnel

For every split, report:

- Sentences accounted for, including incorrect kind labels.
- Requested parts and groups.
- Descriptions accepted.
- Programs compiled.
- Checks actually bound and run.
- “Asked,” with reasons.
- Agreement and false passes on a fixed common reply pool.

The reported dev comparison uses 502 checked replies in round 1 and 492 in round 2 (§4.3). Conditional agreement alone can improve when difficult replies disappear from its denominator.

Report false passes both with the current denominator and conditional on gold-failing replies. Give raw numerators and denominators. Include uncertainty calculated with instruction-level clustering, because replies attached to one instruction are not independent.

### Required slices

Report:

- Lower bounds, upper bounds, equality and inequality.
- Strict versus inclusive comparisons; zero versus nonzero bounds.
- Filter, outer-count, size and position comparisons.
- Negation depth and scope.
- Quantifier, view, wording form and language.
- Number of parts and group type.
- Nesting depth, if supported.
- Description versus free-program branch.

Current support and data volume for deeper negation are unknown.

Use independent evaluation replies targeting boundaries, empty views, missing positions and overlapping filters—not only the replies used to validate each training row. “If there is one” makes missing-position behaviour particularly important (§3.3).

Finite reply agreement does not prove equivalence. Report complete-check correctness as well as part correctness: independently correct parts can still be joined incorrectly.

### Time and cost

I would set a **proposed product line of at most five seconds at the 95th percentile**, end to end on the actual home serving system, measured per instruction. This is a proposed usability bar, not a forecast that B will meet it.

Also report median latency, generated tokens, compilation time, retries, fallback frequency and batch conditions. The rented-card batch measurements are not interchangeable with home serving latency (§4.7).

Finally, measure total system cost at matched task quality against the frontier comparator. The one-hundredth-cost aim applies to the whole system, not just reader decoding (§2). Present evidence does not establish that aim.

## 6. What I would not do, and open questions

I would not:

- Ship round 2 on the strength of the reported dev improvement.
- Diagnose model size from these confounded runs.
- Treat every operator error as a reason for another grammar patch.
- Infer a latency gain from the character-length ratio.
- Count “asked” as a correct enforced check.
- Let fallback conceal reduced description coverage.
- Replace VM enforcement with a model’s judgment of compliance.
- Train on benchmark sentences, use model-written gold, or reuse inspected tests to validate a fix.
- Change model, target, data balance, loss and decoder together and claim to know which helped.

Open questions:

- How expressive is the full CubeLang subset compared with the descriptions?
- What are the join-error rate and round 2’s unseen and long results?
- How many errors select the wrong relation, and how many lower the selected relation incorrectly?
- What precisely distinguishes a false-pass denominator, a compile failure and an unenforced part in current scoring?
- Can the proposed compiler and normaliser be tested independently against the VM?
- What token counts and serving costs accompany each target?
- How is ambiguous natural-language scope labelled?
- Does “no model writes a program” in §2 prohibit authoring gold programs, as the surrounding training rule suggests? If it also prohibits the reader’s proposed program, A and C’s free branch conflict with that rule and need clarification.

The immediate decision is not whether to trust a shorter output. It is whether removing implementation choices improves semantic accuracy without sacrificing the required unseen-kind coverage. B is the cleanest experiment for answering that.

## anthropic/claude-fable-5.2

*24481 tokens out, 279.9 s, finish `stop`*

# Panel answer: the instruction reader and what it should write

A note on the evidence. Round 2 rests on 64 dev rows (4.3), and its comparison with round 1 changes training, grammar and decoder together. The join is three rows seen by eye and length is two (4.5). So only the negation diagnosis is supported by counts; the other two are hypotheses.

## 1. Diagnosis

### Negation

Round 2 leaves 30 reversals in 198 parts (15.2%), which is 30 of the 57 parts not right. Of these, 29 are polarity and 1 is mirrored (4.5).

- **Main cause: the gold's two spellings (4.6).** This is fixable in the data.
  - Summing 3.3 and 4.6, about 4,150 count and size parts plus 2,824 any/none parts are written one of two ways in near-equal shares. That is about 6,970 of 10,064 parts, roughly 69% (my arithmetic), and the wording does not say which way.
  - The example in 3.3 shows the same for a filter: "at least one … at most 2" is written as `not all … > 2`.
  - A reader who understands the part perfectly must still pick a spelling at the operator and keep to it at a `not` written several lines later.
- **The numbers fit this.**
  - Mirrored errors, the true direction mistakes, were 5 of 70 in round 1 and 1 of 30 in round 2 (4.2, 4.5). The reader mostly knows which way the words point; it loses the sign.
  - Between rounds, "wrong with no single reversal" fell from 66 to 14 and "right as written" rose from 59 to 141. Reversals fell only from 49 to 30, so their share of the parts not right rose from 49/137 to 30/57 (4.3).
- **Second cause: the imbalance of bounds, acting through the spelling (4.6).** Also fixable in the data.
  - "At most / fewer than" outnumbers "at least / more than" 1,648 to 548, and half of the former are spelled `not >`. So after `>` or `>=` a `not` follows in 64 to 90% of cases.
  - That gives the reader a shortcut that ignores the words: after `>`, add `not`.
  - This predicts that errors concentrate where the instruction really means "more than / at least". The brief does not give the 30 reversals by gold direction, so I don't know whether this holds. It can be counted on dev rows already read, at no cost, and should be counted before round 3.
- **Third cause: the form of the target.** This is structural.
  - In the program the operator comes first and the `not` is a later two-line block of the same shape as every other block (3.3). The sign of a part is two choices far apart.
  - In the description the `not` is the first token and the operator follows within about 50 characters (my estimate from 3.3).
- **A gap in the round-3 plan.**
  - Half of the reversals (15 of 30) sit in a count's filter or a position's value. The proposed grammar rule refuses a `not` only around a count or size comparison (section 5). One spelling must also cover the quantifier duals (any / not none / not all with the opposite filter), or the largest place stays two-spelled.
  - Position values have their own cause. With "if there is one" the gold writes a `not` around the opposite test (3.3), so "should have the value null" becomes `!=` inside a `not`. The target's operator is the opposite of the words, and round 3 keeps `not` for positions. I don't know how the 15 split between filters and positions.
- **The model.** No number points to it.
  - Programs right as written are 25/64 = 0.39. Independent part errors at 141/198 over about 3.1 parts a row would give about 0.35 (estimate). Errors look per-part, not compounding with program length at 2 to 4 parts.
- **Greedy decoding under a grammar.** No number isolates it.
  - It is a mechanism, not a cause: with two near-equal spellings, the highest-scoring operator is a near-tie.
  - Sampling or beams would not help, because both spellings are valid.
- **Training.** Round 2 helped overall, but 4.3 cannot say how much is training, rule 9 or the decoder.

### The join

This is a hypothesis; it has not been counted.

- **Cause: the form of the target, plus a missing constraint.**
  - The connective is read on the split line ("check 3 condition"), but the binding is written in the main, after every part function, as index copying.
  - Rule 4 controls whether a group may appear, not which checks it ties (3.5).
  - This is fixable by construction. Data alone is the weaker tool.
- **Possible data cause.** Grouped pairs may sit mostly at checks 1 and 2 in training, which would produce exactly the error seen. I don't know the positional distribution; it should be counted.

### Long instructions

This is a hypothesis, from two rows.

- **Cause: a learned counter asked to go past its training range.**
  - "Check 5" was never a target, and line 4 forbids fixing that with data. It is structural to making the model count, and fixable by taking the counting away from it.
- **Round 1's "no loss on long rows" is weak evidence.**
  - Both splits were near 0.5 agreement.
  - The share of replies the VM could check did fall with length, from 0.960 to 0.867 (4.1).
  - Agreement counted only over checkable replies hides that.
- **This is also a line-1 risk.** A fifth check relabelled "check 1" with no part written is a silent drop unless the host catches it.

### Summary

- **Structural to "a model of this size writes a program this long":** the sign as two distant choices, the join as distant index copying, counting past 4, the non-compiling rate, and the 25 s (4.7).
- **Fixable by data:** the two spellings (including duals and positions) and the imbalance.

## 2. What the reader should write

### (A) The program under a growing grammar

- **Removes by construction:** only what each rule names. There were nine rules in one day (3.5), each found from an error.
- **Leaves:** the sign of a part, the join binding until the proposed tie is built, numbering beyond 4, and compile failures not yet covered by a rule.
- **Generality:** in principle anything CubeLang can compute over the views.
  - In practice rule 9, "one view per loop" and "the join in the generator's form" already narrow the program toward the description language.
  - I don't know how many programs outside that language the grammar still allows. If few, A pays 4.9 times the length (4.7) for freedom it no longer has.
- **Unseen kinds (line 5):** not measured, so I don't know. The risk is a compiling but wrong program, which is the false-pass case.
- **Decode time:** about 25 s at home (4.7).

### (B) Descriptions compiled by the host

- **Removes by construction:**
  - compile failures, and with them the uncounted denominators of 4.1;
  - every error that rules 1 to 9 police, since the schema makes it unwritable;
  - inconsistency between operator and `not`, once the description has one spelling;
  - the main, which the host builds from the groups.
- **Does not remove:**
  - a sign read wrongly, which becomes one short choice instead of two distant ones;
  - a group bound to the wrong parts, unless groups are written as in section 4;
  - a wrong column or number.
  - B with today's descriptions inherits the two spellings, since the parts in 3.3 carry them. B is only worth it together with one spelling.
- **Generality cost:** anything not a composition of the slots becomes "asked". Examples of what the slots as stated cannot say:
  - a comparison between two views or two elements ("the first sentence shorter than the last");
  - a count compared to another count, or a proportion;
  - a filter of three comparisons;
  - a position other than first, second or last.
- **A new risk:** the reader squeezes an instruction outside the language into the nearest description. That is a false pass, and it must be measured.
- **Unseen kinds:** a new composition of known slots is handled with a guaranteed-compiling output. A kind outside the slots is "asked".
  - Line 5's "compiling program for at least half" then depends on how much of the held-out kinds and the benchmark's constraints the slots cover. I don't know that share.
- **Decode time:** if time were proportional to characters written and the split is about 400 characters (my estimate from 3.4), B takes about 0.37 of A's time, roughly 9 s at home (estimate). The time to read the prompt is not stated.
- **Rule check:** the compiler is the generator's own and compiles slots, not kinds. To keep it so, a slot may be added only if it is a view, column, quantifier or comparison usable across kinds.

### (C) The mix

- **Removes by construction:** the same as B, on the parts where a description fits.
- **Leaves:** the description path's remainder, plus everything A leaves on the free path.
- **Generality:** none lost in principle. In practice it adds a routing decision, and the free path is where false passes will gather.
- **Unseen kinds:** kinds outside the language reach the free path with the least training support; today's dev false-pass rate, on seen kinds, is 9.35% (4.3).
- **Decode time:** as B for most rows, as A for a part that takes the free path.

### Recommendation

- Build B now, with "asked" as the only exit and each part's description written on its own split line (section 4). This is C with an empty free path.
- Open the free path later, and only if the census below shows that B cannot reach line 5. Give it its own lines: false passes at most 2% on outside kinds, kill above 5%.

### A set to measure the generality lost

1. **A coverage census.** A person, not a model, labels each held-out kind and each benchmark out-of-domain constraint type as one of:
   - expressible in the slots;
   - expressible in CubeLang over the 17 columns but not in the slots;
   - not expressible over the views.
   - Do this on the kinds' definitions, not on counted test rows. Or declare it as the single read of a separate slice.
   - This decides between B and C without a training run.
2. **An "outside" set from the generator.** Checks outside the slots (the families listed under B), with programs written by the generator and run in the VM on built replies, as the rules require.
   - Some families go in training with the target "asked". Others are held out.
   - Report for B the squeeze rate (outside parts given a description) and the false passes among them.
   - Report for A or C the compile rate, agreement and false passes.
   - Lines, written now: false passes at most 2% of checkable replies, kill above 5%; "asked" for at least 90% of outside parts under B (my proposal).

### The time line I would set

- At home, a median of at most 10 s for 2 to 4 parts and at most 20 s for 5 to 8 parts.
- Fail if it is not below today's 25 s.
- This is a judgment. I don't know the reply's own generation time, which should anchor it.

## 3. Negation and direction: three proposals, ranked

The sets: test and test_long are spent by round 2's read. Each proposal is read on a new generator-built set, **test_neg**, never read:

- compositions not in train, in minimal pairs;
- balanced over direction (at least, at most, exactly, other than);
- balanced over place (size, count compared, filter, position value);
- balanced over depth of negation in the words (0, 1, 2).
- At least 300 rows. At about 7.7 checkable replies a row (492/64 in 4.3), that is about 2,300 replies, with a standard error near 0.3 points at 2% before allowing for clustering by row (estimate).

### Proposal 1: the target follows the wording, and the host normalises

- **What changes.**
  - Round 3's plan is one spelling per meaning, which makes the reader compute the negation: "should not have more than 3" becomes `<=`.
  - The alternative is one spelling per wording. The template knows how it said the part, and the target mirrors it: "not … more than 3" becomes `not >`, "at most 3" becomes `<=`.
  - The host pushes the `not` into the comparison exactly. The reader copies signs and the host does the logic.
  - This is generic host code (negation), not code for one kind.
  - It replaces the proposed rule refusing a `not` around a comparison with a host normal form.
- **How to run it.** As a second arm beside one-per-meaning, with the same rows, decoder and grammar.
- **Pass line.** Reversals at most 3% of parts (dev today: 15.2%), no direction above 5%, and false passes at most 2%.
- **Kill line.** Reversals above 8%, or false passes above 5%, or no better than the one-per-meaning arm beyond a row-level interval. In the last case keep the simpler plan.
- **Cost.** A generator change (the spelling drawn from the template) and one extra training run. Round 2 was 1,782 steps; the wall time is not stated.

### Proposal 2: "if there is one" as its own slot

- **What changes.**
  - The `at` part gets a slot, "absent: pass". The operator written is the operator the words say.
  - The compiler emits the vacuous-pass form.
  - This removes the last place where the gold's operator is the opposite of the wording, and lets `not` disappear from positions too.
- **Reach.** At most the position share of the 15 filter-or-position reversals. That share is not stated.
- **Pass line.** On test_neg's `at` parts, reversals at most 3%, and no difference above 3 points between parts with and without "if there is one".
- **Kill line.** Reversals on `at` parts above 8%.
- **Cost.** A generator and compiler change with the rows rebuilt. It can ride in both arms of proposal 1.

### Proposal 3: a second reading of the complement, with disagreement shown as "asked"

- **What changes.**
  - Each part is described twice from two prompts: "passes when" and "fails when".
  - The host complements the second exactly in the slot language and compares normal forms. Equal means enforced; different means "asked".
  - Targets for the second prompt are the generator's own complements, run in the VM with pass and fail swapped.
- **What it does.** It trades wrong checks for "asked", which the rules prefer. It does not catch the same misreading made twice, and I don't know how correlated the two readings are.
- **Pass line.** False passes at most 2%, with "asked" rising by at most 10 points of parts.
- **Kill line.** False passes not at least halved against the single reading on the same rows, or "asked" up by more than 20 points.
- **Cost.** A second short decode per part and one more prompt type in training. It needs B.

## 4. The join and long instructions

### The join

- **Write the join where the connective is read.**
  - The split line already carries "condition". Let it carry the connective and its partner as a relative pointer: "unless, of the previous check".
  - The host builds the main from these, and the reader writes no main.
  - This goes one step beyond the proposed grammar tie: it removes the place where the tie could be broken.
- **Count before fixing.**
  - A join error is a row where every part is right and the program disagrees.
  - Count where grouped pairs sit in train (first pair or later). If skewed, redraw them uniformly.
- **Test replies must cover the join's truth table.**
  - That means all four combinations of the two parts' outcomes for each group. Replies built to pass or fail single parts can leave a misbound group unseen.
  - I don't know whether today's replies do this.
- **Ambiguous scope ("A. B, unless C").**
  - The narrow reading, A and (B or C), implies the wide one, (A and B) or C.
  - So choosing the narrow reading can cause a false fail but never a false pass against the wide reading.
  - Default to the narrow reading and show the user which was taken.
- **What I don't know.** Whether the generator's groups are always two parts, always adjacent, or ever nested. If they are not adjacent, the pointer needs an index, and the index must be host-numbered.
- **Lines, on a fresh held-out join set.**
  - Group kind and both members right for at least 97% of groups.
  - Agreement at least 98% on truth-table replies in rows whose parts are all right.
  - Kill: binding below 90%, or false passes above 5%.
  - The 97% is my estimate of what line 3's 95% leaves room for.

### Long instructions

- **The reader does not count.** The grammar or the host assigns the number of each check line in order. References between lines are relative, which reads the same at check 7 as at check 2.
- **Read each part on its own.** After the split, each part's description is decoded from its own line, plus its partner's line if grouped.
  - Length then touches only the split.
  - The risk is a part that refers to another sentence ("that list"). I don't know whether the generator has such wordings; a set with them should exist before this is trusted.
- **Host accounting, to make line 1 hold by construction.** The host checks two things, and anything failing either becomes "asked":
  - the split lines' text covers every sentence of the instruction;
  - there are as many described parts as check lines.
  - This is generic bookkeeping, not code for a kind.
- **Do not train on 5 to 8 parts.** Line 4 forbids it.
- **Lines, on a new long set (test_long is spent by round 2's read).**
  - Silent drops 0 (kill at 1).
  - Agreement at least 90% of the in-length rate, counted only if line 3 passes in-length.
  - The share of parts right at positions 5 to 8 at least 90% of that at positions 1 to 4.

## 5. The measure

The lines point the right way, but four things let a poor reader pass or hide a good one.

- **Denominators.** Agreement and false passes are counted over replies the VM could check (4).
  - A reader could pass by not compiling the hard rows.
  - Report every rate over all replies as well, with "asked" as its own outcome.
  - Set an "asked" ceiling on in-language sets. I propose at most 10% of parts.
- **The false-pass base.** A false pass can happen only on a reply the true check fails.
  - Report false passes over those replies as well as over all. Otherwise the 2% moves with the mix of built replies, which is not stated.
- **Replies that can tell.** For each part the reply set should separate the gold from each one-reversal mutant and from each rebinding of each group.
  - That means replies at N−1, N and N+1, and the truth table of section 4.
  - Report the share of mutants the replies can detect. Without it, agreement overstates.
- **Line 4 is relative.** Round 1 would have met it (0.548 against 0.508) while failing line 3. Tie it to line 3 as above.

Report as well:

- by direction of the bound: at least or more than, at most or fewer than, exactly, other than;
- by depth of negation in the words: 0, 1, 2;
- by place: size, count compared, filter, position value;
- by part kind and by group kind and position;
- by number of parts (2 to 8) and by the part's index;
- by wording form: plain, goal first, passive;
- by language, since the brief gives no number by language;
- pair accuracy, meaning both members of a minimal pair right. A reader with a fixed sign gets half the members and none of the pairs;
- intervals by row, not by reply. With 50 rows a split (4.1), a 2% line and a 5% kill can hardly be told apart, so read all 300;
- seconds per instruction at home, by number of parts;
- one change per comparison. Reading round 1's adapter under rules 1 to 9 on the same decoder, on dev, would separate what 4.3 confounds.

## 6. What I would not do, and open questions

### I would not

- Keep growing A's grammar as the main line. It converges on B at 4.9 times the length and 25 s.
- Read 4.3 as a training result, or round 2's dev numbers as evidence on unseen compositions or on 5 to 8 parts (4.4).
- Add sampling, beams or voting before the gold has one spelling. Voting between two valid spellings settles nothing.
- Move to a larger model or adapter. No number points there.
- Open the free-program path before the census and its own false-pass line.
- Let a model's wording variations touch the words that carry the sign ("no fewer than", "other than", "unless"). The VM checks the program against built replies; it does not check the wording against the program. Keep those tokens from templates.
- Count "asked" as agreement, or leave it out of the denominator.

### Open questions

1. How do the 30 reversals break down by gold direction, by gold spelling (direct or wrapped), by negation in the words, by language, and between filter and position within the 15?
2. How many tokens are the instruction and the program, against the 512-token window (3.1)?
3. Does any grammar rule hold the numbering of the split lines, or only that of the program's parts?
4. Are groups always two parts, adjacent and un-nested, and where do they sit in train?
5. Do the built replies include boundary and truth-table cases, and what share of them fail the gold?
6. What do the 14 "wrong with no single reversal" and the 13 unmatched parts consist of?
7. What share of the held-out kinds and the benchmark's constraints fits the slots, and has any line-5 set been built?
8. How do the 25 s split between reading the prompt and writing, and can the home GPU batch per-part decodes?
9. Why did checkable replies fall from 502 to 492 on the same 64 rows (4.3)?

## google/gemini-4-argon

*28030 tokens out, 431.1 s, finish `stop`*

### 1. Diagnosis

Round 2 cut non-reversal part errors on the 64 scored dev rows from 66 to 14 and raised parts right as written from 59 / 196 to 141 / 198 (4.3). What remains falls into the three error classes of section 4.5, plus a residual of unmatched spans (13 / 198) and non-reversal errors (14 / 198) (4.3). The numbers point to specific causes for each.

#### 1.1 Negation and direction (30 one-reversal errors on 64 dev rows: 29 negations, 1 mirrored; 4.3, 4.5)

* **Where a count is compared (10 of 30) and at a view's size (5 of 30) (4.5):**
  * **Primary causes:** The gold's two spellings (4.6), the imbalance between bounds (4.6), the form of the target (3.3, 4.7), and greedy decoding under a grammar (3.5).
  * **Why the numbers point here:** Across about 4,150 parts covering 18 count and size meanings, the gold writes every comparison two ways in near-equal shares: directly (e.g., `<=` 244 times) or as a `not` around its opposite (`not >` 260 times) (4.6). In the CubeLang target (3.3), the comparison operator (`>`, `<=`, `==`) is emitted first inside the loop or `c4` check, whereas the `not` is emitted several lines later as a separate variable block (`if (c4 == 0) { assign c5 = 1; }`). Under greedy decoding (3.5), the model must commit to the operator token before deciding whether to emit the `c5` block. Worse, because non-zero bounds in train and dev are skewed 3-to-1 toward upper bounds (1,648 "at most" / "fewer than" vs. 548 "at least" / "more than", 4.6), whenever the gold writes `>` or `>=`, it follows with a `not` in 64% to 68% of size parts and 77% to 90% of count parts, versus only 9% to 32% after `<` or `<=` (4.6). A greedy decoder that emits `>` for a true "more than" instruction faces a 77% to 90% training prior to append a `not` block five lines later, flipping the part into its complement.
* **In a count's filter or a position's value (15 of 30, the largest single bucket; 4.5):**
  * **Primary causes:** The gold's two spellings on quantifiers (3.3, 4.6) and the generator's inverted encoding of `at` ("if there is one", 3.3).
  * **Why the numbers point here:** First, the wording does not follow the spelling (4.6). In the example of section 3.3, the instruction says *"Include at least one paragraph that has at most 2 sentences"*, which is an affirmative filter (`<= 2`). The gold rewrites `any` ("at least one") as `not all` and applies a De Morgan flip to the inner filter, turning `<= 2` into `> 2` (`if (x3 > 2) { add k1, 1; }`) (3.3, 4.6). Likewise, `none` (996) is paired against `not any` (1,051) and `any` (383) against `not none` (394) (4.6). Second, for `at` (1,789 parts in train and dev, the second most frequent kind, 3.3), when said with "if there is one", the generator writes the check as a `not` around the opposite test (3.3). Thus *"should end with '.'"* becomes `not (... != '.')` and *"should not end with '?'"* becomes `not (... == '?')` (3.3). The model must write `==` when the English says "should not" and `!=` when the English says "should". In round 1 on dev, 35 of the 70 reversals were `==` / `!=` swaps (4.2), and in round 2, 15 of the 30 reversals remain in a filter or a position's value (4.5).
* **Mirrored comparisons (`>` for `<`, 1 of 30 in round 2, down from 5 of 70 in round 1; 4.2, 4.5):**
  * **Primary cause:** Training data coverage of phrasing order (3.6). Adding round 2's 3,000 minimal-pair rows with 30% goal-first or passive wording and 4x loss weighting on key tokens (3.6) nearly eliminated mirrored errors (1 left in 198 parts, 4.3, 4.5).

#### 1.2 The join (tying checks 1 and 2 instead of checks 2 and 3; 4.5)

* **Primary causes:** The form of the target (3.3, 3.4, 4.7), the model's 512-token attention window (3.1), the grammar rules (3.5), and group imbalance in training (3.3).
* **Why the numbers point here:** First, non-conjunctive groups are sparse in training: `one` occurs 6,830 times, whereas `either` (838), `if` (396), and `unless` (383) together make up only 19% of groups (estimate from 3.3). Second, in the reader's output today (3.4), the split marks `check 3 condition | , unless...` without naming which check it modifies (check 2 vs. check 1). Third, `public function solve` is written at the very end of the program, on average 1,546 characters (and up to 2,474 characters) after the split (3.3, 4.7). Cubby uses a 512-token windowed attention layer only every third layer, relying on its recurrent mixer beyond that window (3.1). By the time the decoder writes `solve()`, the instruction and split can sit at or beyond the 512-token attention horizon, and grammar rules 4 and 5 (3.5) force an `or` / `if` group to appear when the instruction's words say so without constraining *which* two `pK` variables it joins. With `p1` and `p2` being the most common group arguments in 2-part and 3-part training rows, the decoder defaults to joining `p1` and `p2`.

#### 1.3 Long instructions (labelling the 5th check as "check 1" and dropping the 5th part in round 2; 4.5)

* **Primary causes:** The training regimen in round 2 (3.3, 3.6) interacting with literal part numbering in the target (3.4) and grammar rule 8 (3.5).
* **Why the numbers point here:** We know this is not an inherent inability of the base model to handle 5 to 8 parts because in round 1 (trained for only 788 steps to loss 0.050, 3.6), `test_long` agreed as often as `test` (0.548 vs. 0.508, 4.1). However, every training row has only 2 to 4 parts (3.3). Thus the token sequence `check 5` and function identifier `part5` never appear once in all 12,018 training records (3.3, 3.6). When round 2 trained for 5 epochs (1,782 steps, loss 0.034) with key tokens weighted 4 times in the loss (3.6), it sharpened the transition distribution after `check 4` so strongly that the model cannot emit the unseen number `5` and resets the counter to `check 1`. Once the split has no `check 5`, grammar rule 8 ("parts numbered and called in order", 3.5) prevents emitting a fifth part. (How round 2 performs on the full `test_long` set is not read yet, so the exact rate is unknown, 4.4.)

#### 1.4 Structural vs. data-fixable

* **Fixable by data and target canonicalization:**
  * The 29 negation reversals (4.5) are primarily artifacts of the gold's two spellings (4.6), the 3-to-1 bound imbalance (4.6), the De Morgan rewrite of `any`/`all`/`none` filters (3.3, 4.6), and the inverted `not` encoding of `at` ("if there is one") (3.3).
  * The `check 5 -> check 1` wrap-around (4.5) is an artifact of training for 5 epochs exclusively on literal indices `1..4` (3.3, 3.6).
* **Structural to "a 576M model with a 512-token window writes a 1,546-character program":**
  * **Decode latency:** Reading 445M weights per token (3.1) to auto-regressively emit an average of 1,546 characters of CubeLang (4.7) plus the split takes about 25 s at home and 9.9 s on a rented A100 in batches of 32 (4.7). No data change can fix this; only shortening what the model writes can.
  * **Long-range state tracking across boilerplate:** For 5 to 8 parts (`test_long`), a CubeLang program scales to about 2,500 to 4,000 characters (estimate from 4.7), which exceeds the 512-token attention window (3.1). Keeping track of which part index attaches to which `unless`/`if` condition across 60+ lines of `create`/`assign`/`while` boilerplate forces the recurrent mixer to carry exact symbolic indices over hundreds of redundant tokens.

---

### 2. What the reader should write

The three candidate targets are:
* **(A)** The CubeLang program as today, held to a growing grammar (3.4, 3.5).
* **(B)** The split plus the generator's compact description of parts and groups (3.3), compiled deterministically into CubeLang by the host.
* **(C)** A mix: the 3.3 description for every part that fits the schema, and a free CubeLang `function partK(...)` only for parts that do not.

*(Note on Section 2, Rule 5: A deterministic compiler from the 3.3 compositional schema of 6 quantifiers, 7 views, 17 columns, and 4 group combinators to CubeLang is a single general compiler—already present in the generator to build training rows [2, 3.3]—not host code for one kind of instruction.)*

| Dimension | (A) Full CubeLang + growing grammar | (B) Split + 3.3 descriptions (host-compiled) | (C) Mix: 3.3 descriptions + free CubeLang fallback |
|---|---|---|---|
| **4.5 errors removed by construction** | Removes syntax and naming errors covered by rules 1–9 (3.5). Leaves all three 4.5 classes (negations, join index errors, long-instruction counter reset) unless patched by further grammar rules. | Removes multi-line separation between operator and trailing `c5` negation (3.3); removes all 14 non-reversal CubeLang coding errors (4.3); brings the whole output (~316 chars avg, 4.7) inside a single 512-token window (3.1), removing attention-decay join errors. Leaves semantic reading of the operator from English/French text and split-counter wrap unless training/grammar also change. | Same as (B) for all parts that fit the 3.3 schema (100% of current train/dev/test/test_long). On fallback parts, leaves (A)'s error classes. |
| **Generality cost** | None relative to CubeLang, *except* that grammar rules 1–9 (e.g., rule 7 "one view per loop" and rule 5 "the join in the generator's form", 3.5) already restrict (A) nearly to (B)'s templates. | Cannot write a check that is not a composition of `n`, `count`, `any`, `none`, `all`, `at` with 1–2 constant column filters over the 7 views and 17 columns (3.2, 3.3) (e.g., cannot compare element `i` to element `i+1` or compare two views' sizes to each other). | Zero generality lost relative to (A): uses (B) when a part is a standard composition and falls back to a CubeLang function when it is not. |
| **Unseen kind of instruction (Sec 2, Line 5)** | If the unseen kind is a new composition of the 7 views/17 columns that obeys rules 1–9 (3.5), it can write a program. If it violates rule 7 (e.g., needs nested loops) or fails to compile, it becomes `"asked"` (2). | If the unseen kind is a new composition of the 7 views, 17 columns, and 6 quantifiers, it compiles 100% of the time. If it is an out-of-domain benchmark constraint outside the 3.3 schema, it must be marked `"asked"` (2). If >50% of the Line 5 set falls outside the 3.3 schema, (B) cannot pass Line 5's 50% compile floor. | Unseen compositions of the 3.3 schema compile 100% via descriptions; out-of-schema benchmark constraints trigger the CubeLang fallback to meet Line 5. *(Requires keeping a fraction of raw CubeLang functions in generator training so the reader learns the fallback without violating Sec 2 Rule 3.)* |
| **Decode time** | **25 s** at home (0.3 s grammar); **9.9 s** avg (2 s shortest) on rented A100 (batch 32) for 2–4 parts (4.7). Scales to ~45–55 s at home on 5–8 parts (estimate). | Parts + groups average **316 chars vs. 1,546 chars** (4.9x shorter, 4.7). With the split included, total tokens drop ~3.2x (estimate), cutting decode time to **~6–8 s at home** (estimate) and **~2.5–3 s on A100 batch 32** (estimate). | Same as (B) (**~6–8 s at home**, **~2.5–3 s on A100**, estimate) on all schema-fitting instructions; adds ~5 s at home (estimate) per rare fallback part. |

#### Recommendation on target
Adopt **(C)** (or **(B)** if the generality audit below shows Bucket 2 is empty or below accuracy bars). Even in **(A)**, grammar rules 1 through 9 (3.5) have already stripped away almost all freedom in CubeLang except the exact slot values of (B), while still forcing the 576M model to emit ~1,230 characters of redundant loop boilerplate per row (1,546 minus 316, 4.7). Moving to (B)/(C) cuts output length 4.9x on the check body (4.7), keeps 8-part instructions inside a single 512-token attention window (3.1), and guarantees 100% valid CubeLang compilation on every schema-fitting part.

#### A set to measure the generality lost
Before retiring (A) for (B) or sizing the fallback in (C), build a **Pre-Registered Expressiveness Audit Set** from the public instruction benchmarks and held-out kinds referenced in Section 2 (Line 5) without putting any benchmark sentence in training (Sec 2, Rule 4):
1. Take 200 verifiable constraint parts (estimate of sufficient sample size) from the held-out kinds and public benchmarks.
2. Have a human (or rule classifier over the benchmark metadata, obeying Sec 2 Rule 3) label each constraint into one of three buckets using *only* the VM's 7 views and 17 columns (3.2):
   * **Bucket 1 (In 3.3 schema):** Expressible as a composition of `n`, `count`, `any`, `none`, `all`, `at` with 1–2 constant column comparisons and `one`/`either`/`if`/`unless`.
   * **Bucket 2 (In CubeLang [A], outside 3.3 schema [B]):** Expressible in CubeLang over the 7 views and 17 columns under grammar rules 1–9 (3.5), such as comparing `sentence_n_word[0]` against `sentence_n_word[n_sentence - 1]`, but not in the 3.3 JSON description.
   * **Bucket 3 (Outside the 7 views / 17 columns):** Requires views or columns the host does not provide (e.g., tone, topic, rhyme); neither (A) nor (B) can enforce these in the VM, so both must mark them `"asked"`.
3. **Measure of generality lost:** The ceiling of generality lost by (B) relative to (A) is the share of parts in **Bucket 2**. The *usable* generality lost is the share of Bucket 2 parts on which (A) writes a compiling program that reaches >= 95% reply agreement with <= 2% false passes (Sec 2, Line 5).

---

### 3. Negation and direction: three proposals beyond Section 5

Section 5 already proposes pushing negation into count/size comparisons, refusing `not` around a count or size comparison in the grammar, balancing "at least" and "at most", and weighting key tokens. However, **Section 5's proposals leave two major sources of negation reversals untouched**:
1. Section 5 explicitly says *"`not` kept only for positions"* (5), which preserves the generator's inverted encoding of `at` ("if there is one") as a `not` around the opposite operator (3.3).
2. Section 5 does not remove De Morgan rewrites across `all`/`any`/`none` and their inner `where` filters (where *"at least one paragraph with at most 2 sentences"* is written with filter `> 2`, 3.3, 4.6).
Together, filter and position value reversals account for **15 of the 30 remaining reversals** in round 2 (4.5) and 35 `==`/`!=` reversals in round 1 (4.2).

Here are three proposals beyond Section 5, ranked by expected gain per cost.

#### Proposal 1 (Rank 1 — Highest gain / cost): Remove operator inversion from `at` ("if there is one") and lock `any` / `none` / `all` filter operators to surface polarity

* **What changes:**
  1. **Direct vacuous-pass compilation for `at` (1,789 parts in train/dev, 3.3):** Stop writing `at` ("if there is one") as a `not` around the opposite comparison (3.3). Instead, compile "if there is one" as an explicit bounds guard that returns `1` (pass) when the view has fewer elements than `pos` (`if (n_sentence < 2) { return 1; }`), followed by the **un-negated** comparison that matches the English verb (`==` for *"should end with"*, `!=` for *"should not end with"*) and no `c5` negation block.
  2. **No De Morgan filter flips on `any`, `none`, `all` (4,122 parts in train/dev, 3.3):** In the generator, forbid rewriting `any` ("at least one ... with <= 2") as `not all (... with > 2)` (3.3, 4.6), `none` as `not any`, or `any` as `not none` (4.6). Every `where` filter must use the comparison operator that matches the clause's own words (`<= 2` for "at most 2"). In CubeLang, `k1` always counts elements that *pass* that positive filter, and the final check tests `k1 >= 1` (`any`), `k1 == 0` (`none`), or `k1 == n_view` (`all`), with no `c5` negation block.
* **Which set reads it:** Pre-register on the 36 unread dev rows (out of the 100 dev rows, 4.3), then read once on the held-out 300-row `test` set (2, 3.3, 4.4).
* **Pass line (written now):** Reversals in a count's filter or a position's value drop from 15 / 198 parts (7.6% on the 64 dev rows, 4.3, 4.5) to **<= 3 / 198 parts (<= 1.5%)** on the 64 dev rows and **<= 1.5% of parts** on `test`.
* **Kill line (written now):** Filter and position reversals remain **> 6 / 198 parts (> 3.0%)** on the 64 dev rows, or false passes on `test` exceed **5%** (Sec 2, Line 3 kill).
* **Cost:** Half a day of generator template edits; regenerating the 12,018 training records (3.6); one 1,782-step training run (3.6). Zero serve-time cost (programs become shorter by removing the `c5` negation block from `at` and `all`, 3.3).

#### Proposal 2 (Rank 2): Local operator extraction in the split header + Grammar Rule 10

* **What changes:** Today the split line copies the text span (`check 2 | Your reply should have fewer than 9 lines`, 3.4) and defers deciding the operator (`<` vs. `<=` vs. `>`) until the model writes `part2` hundreds of tokens later. Change the split line to emit the canonical operator(s) immediately before the pipe while attending directly to the sentence:
  `check 2 [op: <] | Your reply should have fewer than 9 lines`
  `check 1 [op: !=] | The first list item, if there is one, should not be marked with "-".`
  Add **Grammar Rule 10** to the decoder (3.5): when writing `partK` (or description `K`), the comparison operator slots are constrained to the exact operator sequence already emitted in `check K`'s bracketed tag.
* **Which set reads it:** The 36 unread dev rows (4.3) for sanity check, then the held-out `test` and `test_long` sets (3.3, 4.4).
* **Pass line (written now):** Total one-reversal parts on the 64 dev rows drop from 30 / 198 (15.2%, 4.3) to **<= 4 / 198 (<= 2.0%)**, and on `test` <= 2.0% of parts are one-reversal errors.
* **Kill line (written now):** One-reversal parts remain **> 10 / 198 (> 5.0%)**, or split span-and-kind accuracy on `test` falls below **90%** (Sec 2, Line 2).
* **Cost:** ~12 extra characters per part in the split (~40 characters per row, adding ~0.5 s decode time at home, estimate from 4.7); a deterministic script to add `[op: ...]` to training targets; ~25 lines of grammar code (3.5); one 1,782-step training run (3.6).

#### Proposal 3 (Rank 3): Operator log-probability margin gating with dual-branch VM check (zero retraining)

* **What changes:** Greedy decoding (3.5) forces the reader to commit to an operator token (`==` vs. `!=`, `<=` vs. `>`, `<` vs. `>=`) even when the model is nearly 50/50 uncertain between an operator and its complement/negation—turning model uncertainty directly into false passes (which sit at 9.35% in round 2, above the 5% kill line, 2, 4.3). Section 2 states: *"A false pass is the worst outcome... A part whose program does not compile, or that the reader is not held to, is 'asked': shown to the user as not enforced. 'Asked' is better than a wrong check."*
  During decoding, at each comparison operator token in `partK`, record the probability margin $\Delta p = p(\text{op}_1) - p(\text{op}_2)$ between the greedy operator and its complement/mirror. If $\Delta p < \tau$ (pre-registered at $\tau = 0.30$, estimate, tuned only on the 64 already-read dev rows):
  * Compile `partK` with $\text{op}_1$ and a shadow `partK_alt` with $\text{op}_2$.
  * When the VM checks a reply, if both `partK` and `partK_alt` agree on the reply, return their shared verdict; if they disagree on that specific reply, abstain on that part as `"asked"` rather than issuing a false pass.
* **Which set reads it:** Tune $\tau$ on the 64 dev rows (4.3); read once on the 36 unread dev rows and on `test` (4.4).
* **Pass line (written now):** False passes drop below **2.0%** on `test` (meeting Sec 2, Line 3) while keeping checkable reply coverage **>= 92%** (estimate).
* **Kill line (written now):** False passes remain **> 5.0%** (Sec 2, Line 3 kill), or abstaining drops checkable replies below **85%**.
* **Cost:** **Zero retraining.** ~40 lines of host/decoder code to inspect logits at the 1–2 operator tokens per part and emit `partK_alt` when $\Delta p < \tau$; negligible VM runtime cost.

---

### 4. The join and long instructions

#### 4.1 Reading the logic between parts (`and`, `either-or`, `if`, `unless`)

Three facts explain why the join failed on the three dev rows seen by eye (4.5):
1. `either` (838), `if` (396), and `unless` (383) are outnumberedmore than 4-to-1 by `one` (6,830) (3.3).
2. In the split today (`check 3 condition | , unless...`, 3.4), the tag `condition`neither distinguishes `either`, `if`, and `unless` nor names which check is the main clause and which is the condition (especially when round 2 puts 30% of checks goal-first or in the passive, 3.6, such as *"Unless check 1, check 2"* vs. *"Check 2, unless check 1"*).
3. In Target A, `solve()` is written ~1,500 characters after the split (4.7).

**Proposal for the join:**
1. **Explicit directed join tags in the split (extending Section 5's proposal):** Replace the ambiguous `condition` tag in the split (3.4) with the exact combinator and its relative target (`prev` or `next`), written right on the clause that carries the connective:
   * Default independent check: `check | ...` (joined by `and`)
   * Disjunction: `check [either prev] | , or ...`
   * Conditional (trailing, *"A, if B"*): `check [if_for prev] | , if ...` (compiles to `(not this) or prev`)
   * Conditional (fronted, *"If A, B"*): `check [if_for next] | If ...` (compiles to `(not this) or next`)
   * Exception (trailing, *"A, unless B"*): `check [unless_for prev] | , unless ...` (compiles to `prev or this`, 3.3)
   * Exception (fronted, *"Unless A, B"*): `check [unless_for next] | Unless ...` (compiles to `next or this`)
2. **Deterministic compilation of `solve()` (or Grammar Rule 11):** Once the split declares `[unless_for prev]` on check $k$, there is zero new information for the model to invent in `solve()` 1,500 characters later. Under Target B/C, the host compiles `solve()` directly from the split tags. Under Target A, **Grammar Rule 11** forces `solve()` to emit `if (p{k-1} == 1) { assign g1 = 1; } if (p{k} == 1) { assign g1 = 1; }` for the exact indices $(k-1, k)$ marked in the split.
3. **Group-balanced training pairs:** Up-sample `either`, `if`, and `unless` in the generator from 19% of groups (3.3) to 40% of groups (estimate), with equal shares of 3-part and 4-part rows where the joined pair is $(1, 2)$, $(2, 3)$, and $(3, 4)$, and equal shares of fronted (*"Unless A, B"*) and trailing (*"B, unless A"*) wordings.

* **Pass line (written now):** On dev and `test`, join wiring accuracy (selecting the right combinator and the right pair of part indices $(i, j)$ whenever all parts are matched) is **>= 98%**, and whole-program reply agreement meets **>= 95%** with false passes **<= 2%** (Sec 2, Line 3).
* **Kill line (written now):** Join wiring accuracy is **< 93%**, or reply false passes exceed **5%** (Sec 2, Line 3 kill).

#### 4.2 Reading instructions longer than any seen in training (5 to 8 parts)

In round 2, the 5th check was labelled `check 1` again and no 5th part was written (4.5) because training contains only 2 to 4 parts (3.3), giving the tokens `check 5..8`, `part5..8`, and `p5..8` **zero occurrences** in 12,018 training records (3.6). Furthermore, an 8-part CubeLang program is ~3,000–3,500 characters (estimate from 4.7), overflowing the 512-token attention window (3.1).

**Two-part proposal for long instructions (while strictly obeying Line 4 of Section 2: trained on at most 4 parts per row, read on 5 to 8):**

1. **Eliminate or force monotonic check counters + Index-Offset Training:**
   * **In the grammar (zero retraining needed, immediate fix for R2):** In the split grammar (3.5), maintain an integer counter $c$ of `check` lines emitted so far. Whenever the model emits the token `check`, constrain the number token strictly to the literal integer $c + 1$. Because the grammar allows only token `5` after the fourth check, the model can **never** wrap `check 5` back to `check 1` (4.5).
   * **In training (for Target A so `part5..8` and `p5..8` are in-distribution without ever training on >4 parts):** When generating a 2-, 3-, or 4-part training row, randomly shift the starting index offset $s \in \{0, 1, 2, 3, 4\}$ on 50% of rows (or better, drop numbers from the model's tokens altogether by writing unnumbered `check | ...` in the split and compiling parts in Target B/C where JSON array elements have no index tokens `part1..part8` at all!).
2. **Per-part windowed compilation (or Target B/C) to stay inside the 512-token window (3.1):**
   * If using **Target B/C**, an 8-part description is only ~650–800 characters (estimate from 316 chars for ~3 parts, 4.7)—well inside a single 512-token attention window (3.1) and using no `part5..8` tokens.
   * If staying in **Target A**, once the split is emitted, each part's function `partK` is forbidden by Grammar Rule 9 from reading any words outside `check K`'s span anyway (3.5). Therefore, the decoder only needs to attend to `check K` when writing `partK` (or can run each `check K` through the adapter as an independent 1-part batch item on the GPU and assemble `part1..partN` + `solve()`), guaranteeing that an 8-part instruction has the exact same per-part accuracy and context length as a 2-part instruction.

* **Pass line (written now):**
  * Counter wrap-around (`check K` reset to `check 1` for $K \in \{5..8\}$): **0%** on `test_long` (kill **> 0%**).
  * Span and kind recovery on `test_long` (5 to 8 parts): **>= 90%** (Sec 2, Line 2).
  * Reply agreement on `test_long`: **>= 90% of the in-length (`test`) agreement rate** (Sec 2, Line 4; i.e., >= 85.5% if `test` is at 95%), with false passes **<= 2%** (kill **> 5%**).
  * VM checkable reply rate on `test_long`: **>= 95%** (up from 0.867 in 4.1; kill **< 90%**).

---

### 5. The measure

#### 5.1 Are the lines of Section 2 (Order 0067) the right bar?

The five lines of Order 0067 (2) set the right structure, but the numbers in Section 4 expose **four measurement loopholes** that should be tightened now:

1. **False passes are divided by *all* checkable replies instead of *gold-failing* replies (4.1, 4.2, 4.3):**
   In 4.3, round 2's false passes are reported as `46 / 492 = 0.0935`, where 492 is *all* replies the VM could check (both passing and failing replies). By definition (Sec 2), a false pass occurs when *"the reader's program passes a reply that the true check fails"*. A reply where the true check *passes* can never be a false pass. If roughly half of the 492 replies are gold-failing (~246 replies, estimate; the exact pass/fail split of the test replies is not stated in the brief), then 46 false passes is actually an **~18.7% false-pass rate on bad replies** (estimate). Dividing by total replies lets a test suite dilute its false-pass rate simply by including more gold-passing replies.
   * **Correction:** Report False Pass Rate over gold-failing replies (`FP / Gold_Fail`), or mandate an exact 50/50 split of gold-pass and gold-fail replies per row and report both denominators.
2. **Lines 3 and 4 have no minimum floor on "replies the VM could check":**
   In 4.1, `test_long` only checked `312 / 360 = 0.867` of replies, and in 4.3, round 2 checked 10 fewer replies on the same 64 dev rows than round 1 (`492` vs. `502`). Because Line 3 computes agreement and false passes only over replies the VM could check (4.0), a reader that fails to compile (or marks `"asked"`) on 40% of in-domain rows could still pass Lines 1, 2, 3, and 4 on the remaining 60%.
   * **Correction:** Add an explicit compile/checkable floor to Lines 3 and 4: **replies the VM can check >= 98% (kill < 95%)** on `test` and **>= 95% (kill < 90%)** on `test_long` (matching Line 5's principle of an explicit compile floor).
3. **Whole-program reply agreement masks part-level errors on conjunctive (`and`) rows:**
   In 4.2, round 1 achieved `0.559` reply agreement even though **only 6 of 100 programs (`0.060`) were right as written** and only `101 / 312` parts (`0.324`) were right as written! In 4.3, round 2 has `0.685` reply agreement with `25 / 64` (`0.391`) programs right as written. Why? Because on a 3- or 4-part `and` program (`one` is 81% of groups, 3.3), whenever one part rightly fails a bad reply, the whole conjunction returns `0` and "agrees" with the gold `0`, hiding broken sibling parts.
   * **Correction:** Add a part-level bar to Line 3: **isolated part-level VM agreement >= 97%** (or **parts right as written / semantically equivalent >= 92%**) and **programs right / equivalent across all parts >= 80%**.
4. **No latency line is set (Section 2, last bullet):**
   At 25 s per instruction on the home GPU and 9.9 s on a rented A100 in batches of 32 (4.7), reading the instruction costs too much time and compute relative to the system aim ("a frontier model's quality on its tasks at one hundredth of the cost or less", 2).
   * **Proposed time line:**
     * **Home GPU (batch 1):** Pass **<= 6.0 s average** (<= 10.0 s on `test_long`); Kill **> 12.0 s average**.
     * **Rented A100 (batch 32):** Pass **<= 2.5 s average**; Kill **> 5.0 s average**.

#### 5.2 What else should be reported

Every evaluation run should report a stratified table across four slices:

1. **By direction of a bound:**
   Report part equivalence, reversal rate, and false passes separately for:
   * **Upper bounds (`<`, `<=`):** "at most", "fewer than" (1,648 in train/dev, 4.6)
   * **Lower bounds (`>`, `>=`):** "at least", "more than" (548 in train/dev, 4.6)
   * **Equality / inequality (`==`, `!=`):** "exactly", "other than", value matches
   Report the **unweighted macro-average** across the three directions so the 3-to-1 majority of upper bounds (4.6) cannot hide high reversal rates on lower bounds.
2. **By depth and place of negation:**
   * **By surface negation depth:** Depth 0 (affirmative wording), Depth 1 (single negation: *"not"*, *"none"*, *"other than"*, *"unless"*, or vacuous *"if there is one"*), and Depth 2+ (compound negations: e.g., *"no paragraph with fewer than..."* or *"unless ... not"*).
   * **By structural place (matching 4.5):** (a) view size (`n`), (b) count comparison, (c) inner `where` filter on `count`/`any`/`none`/`all`, (d) position value (`at`), and (e) group combinator (`if`/`unless`).
3. **By number of parts ($N \in \{2..8\}$) AND by ordinal part position ($k \in \{1..8\}$):**
   Reporting accuracy by **ordinal position $k$ within the instruction** (part 1 vs. part 2 vs. part 4 vs. part 5..8) separates two distinct failure modes: gradual attention decay across long outputs vs. a sharp cliff at $k = 5$ from unseen index tokens (`check 5`, 4.5).
4. **By group combinator:**
   Report join agreement separately on rows containing only `one` (`and`, 6,830 in train/dev) vs. rows containing `either` (838), `if` (396), and `unless` (383) (3.3). Because `if` and `unless` are under 5% of groups each (3.3), a reader could fail every `if` and `unless` row and still pass a 90% aggregate bar.

---

### 6. What I would not do, and open questions

#### 6.1 What I would not do

1. **Do not keep adding grammar rules to patch full CubeLang generation (Target A, 3.5) instead of compiling from descriptions (Target B/C, 3.3, 4.7).** Nine rules were added on 2026-11-13 (3.5). Each rule turns the grammar into a slower, harder-to-maintain parser while still forcing a 576M model to generate 1,546 characters of boilerplate over 25 seconds at home (4.7) and overflowing the 512-token attention window (3.1) on 5 to 8 parts.
2. **Do not keep `not` wrappers as an inverted test on `at` ("if there is one") (3.3, 5).** Section 5's round 3 plan proposes keeping `not` for positions. Because `at` is 1,789 of 10,064 parts (3.3), keeping `not` around the opposite operator on `at` leaves the model writing `==` for *"should not"* and `!=` for *"should"*, perpetuating the 15 filter/position reversals (4.5).
3. **Do not apply the outside models' margin loss on the operator token (Section 5) before eliminating two-spelling gold targets (4.6) and `at`/`all` filter inversions (3.3).** Putting a margin loss on the operator token while the training set itself writes the same meaning with opposite operators 50% of the time (4.6) will punish the model for matching the gold distribution and degrade probability calibration.
4. **Do not train for 5 epochs with 4x token loss weights (3.6) on literal part numbers `1..4` without index-offset augmentation or unnumbered checks.** That over-sharpening is what broke 5-part splitting in round 2 (4.5) when round 1 had no length drop (4.1).
5. **Do not scale up the base model, window size, or LoRA rank (3.1) to fix this task.** The 316-character part-and-group representation (4.7) fits comfortably inside the existing 512-token window and rank-16 LoRA adapter (3.1).

#### 6.2 Open questions (not stated in the brief)

1. **What are round 2's numbers on the remaining 36 of the 100 dev rows, and on `test` and `test_long` (4.3, 4.4)?** Section 4.3 notes that the 64 dev rows scored so far are *"the shorter 64 of the 100"*, and Section 4.4 notes `test` and `test_long` are not read yet. I don't know whether round 2's agreement (0.685 on the shorter 64 dev rows, 4.3) drops on the longer 36 dev rows or on `test_long`.
2. **How are the ~8 test replies per row constructed, and what share of them are gold-failing replies (2, 4.1, 4.3)?** The brief states replies are built to pass and to fail (2, 4.0) and shows ~7.2 to 8.0 replies per row (400 replies for 50 `test` rows in 4.1; 492 for 64 `dev` rows in 4.3), but does not state how many replies fail the gold check or whether each part of a multi-part instruction is tested by a dedicated single-part-failing reply.
3. **What share of the unseen-kinds and public benchmark test set (Section 2, Line 5) can be expressed using the existing 7 views and 17 columns (3.2) and the 6 quantifiers of 3.3?** I don't know what constraints are in that benchmark set; this determines whether Target B alone can pass Line 5's 50% compile floor or whether Target C's fallback is required.
4. **How is the VM check used in the end-to-end user request flow, and on what hardware topology at serve time?** The brief states the reader is a rank-16 LoRA adapter on Cubby's shared base on a home GPU (3.1) and the VM runs the check on the reply (1, 2), but I don't know whether the reader runs concurrently with reply generation, sequentially after reply generation, or inside a rejection-sampling loop that regenerates failed replies.
5. **What is the error breakdown between English and French instructions (2)?** Section 2 states templates are in English and French, but no metrics in Section 4 separate the two languages. I don't know whether the 29 negation reversals in 4.5 are concentrated in one language (e.g., French two-part negations like *ne ... pas / aucun / que*) or split evenly.

## deepseek/deepseek-v4-pro-0813

*18970 tokens out, 540.2 s, finish `stop`*

## 1. Diagnosis

**Negation.** The 30 reversals on the 64 dev rows are 29 negations and 1 mirror (4.5). The numbers point mainly to the gold’s two spellings and the bound imbalance, not to the program form or the grammar. Section 4.6 shows that every count/size meaning is written two ways in near-equal shares, and the wording does not follow the spelling: after `>` or `>=` the gold continues with a `not` in 64–90% of parts, after `<` or `<=` in only 9–32%. The model is trained on both spellings with no reliable signal from the words, so it chooses the wrong one. The imbalance—1,648 upper-bound parts against 548 lower-bound parts (4.6)—likely biases the model toward the more frequent upper-bound forms. This is data-fixable: one spelling per meaning, balanced bounds, and a grammar that refuses the free `not` should remove most of it. It is not structural to a model of this size writing a program this long.

**The join.** The eye finding in 4.5—parts right but the or-group joined checks 1 and 2 instead of 2 and 3—points to the form of the target and greedy decoding under a grammar. The join is written in `main` at the end of a target averaging 1,546 characters (4.7); the model must carry the split’s part indices across that length. Greedy decoding commits to group indices with no backtracking. The training also under-represents non-`one` groups: `either` 838, `if` 396, `unless` 383, against `one` 6,830 (3.3). This is partly structural to a model of this size writing a program this long, and partly data-fixable with more join examples and a grammar that ties the group to the split’s join.

**Long instructions.** The counted round 2 test/test_long read is not available (4.4). Round 1 showed no loss on long rows (4.1), but round 2 eye on two 5-check rows found the fifth check labelled “check 1” and no fifth part written (4.5). Training is only up to 4 parts (3.3). The numbers point to the target form and the training length. This is structural to the program target at this model size, and data can help by training on 5–8 part rows; whether that reaches line 4 is unknown.

The model size is confounded with target length; the brief gives no ablation separating them.

## 2. What the reader should write

**A — the program, as today, held to a grammar that grows a rule at a time.**  
It removes none of the three error classes by construction; it leaves negation, join, and long-instruction errors. It costs no generality beyond the grammar rules themselves: the target is CubeLang under the current grammar. An unseen kind of instruction gets an attempted program; line 5 depends on the model. Decode time is the current one: about 25 s at home, and on a rented A100 in batches of 32 about 2 s shortest and 9.9 s average over the 64 dev rows (4.7). To measure generality lost, build a set of instructions whose correct program uses a form the grammar forbids.

**B — the generator’s own description of each part and of the groups, compiled by the host.**  
It removes by construction the program syntax, type, name, call-order, and long-program-form errors, because the host compiles the description. It leaves negation: the model still chooses the operator and `not` in the description. It leaves join errors at the description level: the model can still write wrong group indices. It reduces but does not remove long-instruction errors, because the target is shorter but the model can still duplicate or omit parts. Generality cost: an instruction whose check is not a composition of the 3.3 slots (`n`, `count`, `any`, `none`, `all`, `at`, `not`, groups `one`, `either`, `unless`, `if`) cannot be expressed. An unseen kind outside those slots cannot be written as a free program, so line 5 is at risk. Decode time: the parts/groups form is 316 characters on average against 1,546 for the program, 4.9 times shorter (4.7). If time scales with output length, this is about 5 s at home and about 2 s on the A100 average (estimate). To measure the generality lost, build a held-out set of instructions whose correct check is not a composition of the description slots, and compare B against A on line 5 and agreement.

**C — a mix: descriptions where one fits, a free program only where none does.**  
For the description subset it removes the same program-form errors as B; for the free-program subset it removes none. It leaves negation in both subsets. It leaves join and long-instruction errors in the free-program subset, and reduces them in the description subset. Generality cost is less than B: only instructions forced into a description when none fits, or forced into a free program when a description would be safer. A new error is the chooser itself. An unseen kind that fits the description is handled; one that does not can fall back to a free program, so line 5 is better than B but depends on the chooser. Decode time is between A and B, depending on the mix; it is not measured. To measure the loss, use the same held-out set as for B, and also score chooser errors: cases where a description fits but the model writes a free program, and cases where no description fits but the model writes a description.

B and C comply with the hard rule that no model writes a program, because the host compiles the description.

## 3. Negation and direction

Ranked by expected gain per cost.

**Proposal 1 — Direction label in the split, grammar-fixed operator and `not`.**  
What changes: the split target adds a direction label for each check: `upper`, `lower`, `equal`, `other`, `position`. For count and size parts, the grammar fixes the comparison operator and `not` to the canonical form for that label; `not` remains only for `at` positions. Training rows derive the label from the generator’s part.  
Which set reads it: a new 300-row held-out set from the same generator, not the 64 dev rows already read; if combined with other proposals, one read of the combined system.  
Pass line: parts fixed by one reversal at most 5% of parts, against 30/198 = 15% on the 64 dev rows (4.3), and false passes at most 2%.  
Kill line: false passes above 5%, or parts fixed by one reversal above 10%.  
Cost: low. One label per part and one grammar rule; no new model. Decode time roughly unchanged (estimate).

**Proposal 2 — Description target with host compilation.**  
What changes: the reader writes the generator’s parts/groups (3.3) instead of the program; the host compiles. The description grammar allows only one spelling per count/size meaning and `not` only on `at` parts. This is target B plus the one-spelling rule.  
Which set reads it: a new 300-row held-out set.  
Pass line: agreement at least 95%, false passes at most 2%, parts fixed by one reversal at most 5%.  
Kill line: false passes above 5%, or agreement below 90%.  
Cost: medium. New target and host compiler; the target is 4.9 times shorter (4.7). Decode time estimate about 5 s at home and about 2 s on the A100 average if length-proportional (estimate).

**Proposal 3 — Negation self-correction pass.**  
What changes: after the first program or description is written, the model reads the instruction and the proposed part’s operator/`not` and outputs `ok` or `flip` for each count/size part. The grammar allows only `ok` or `flip`; `flip` toggles the comparison or `not` in the host. Training rows label the first output against the generator’s part.  
Which set reads it: a new 300-row held-out set.  
Pass line: parts fixed by one reversal at most 5%, false passes at most 2%.  
Kill line: false passes above 5%, or reversals above 10%.  
Cost: medium. A second short decode over the part descriptions; estimate +20–50% decode time (estimate).

## 4. The join and long instructions

The split should recover an explicit join tree, not only per-sentence labels. The words “and”, “either-or”, “if”, and “unless” should attach to the checks they join. The program or description should be constrained to use exactly that tree. The current grammar has “and” by default and an or/if group only when the words say so (3.5), but it does not tie the group to the two checks the split joins; the 4.5 eye error is exactly that.

**Proposal J1 — Explicit join tree in the split.**  
What changes: the split target adds a join tree over part indices in the generator’s group form (`one`, `either`, `unless`, `if`, default `and`). The program/description grammar must use exactly that tree.  
Which set reads it: a new 300-row held-out set enriched with `unless` and `if` joins.  
Pass line: join agreement at least 95%, false passes at most 2%.  
Kill line: false passes above 5%, or join agreement below 90%.  
Cost: low.

**Proposal J2 — Train on 5–8 part rows.**  
What changes: add training rows with 5 to 8 parts, generated the same way. If the program target remains, add a grammar rule that part numbers must be unique and sequential; if the description target is used, the host compiler enforces unique indices.  
Which set reads it: test_long, 300 rows of 5 to 8 parts, once.  
Pass line: on test_long, agreement at least 90% of the in-length rate (line 4), false passes at most 2%.  
Kill line: false passes above 5%, or agreement below 90% of the in-length rate.  
Cost: medium.

**Proposal J3 — Description target for long instructions only.**  
What changes: for instructions with more than 4 parts, the reader writes the parts/groups description (3.3) instead of the program; the host compiles. This is a general length switch, not a rule for one kind of instruction.  
Which set reads it: test_long.  
Pass line: on test_long, agreement at least 90% of the in-length rate, false passes at most 2%.  
Kill line: false passes above 5%, or agreement below 90% of the in-length rate.  
Cost: medium.

Long instructions should not be read by renumbering parts. The fifth check labelled “check 1” is a split error and should be caught by lines 1 and 2 on long rows.

## 5. The measure

The lines of section 2 are a right minimum but not sufficient. Line 1 is a correct kill: silent drops are unacceptable. Line 2 measures span and kind but not direction. Line 3 measures the core but does not separate negation from join. Line 4 is right but should also report false passes and span/kind by part count. Line 5’s “compiling program for at least half” is a low bar; it should also report agreement and false passes on the subset where a program is written.

What else should be reported:

- **By direction of a bound:** upper-bound, lower-bound, equal/other; report parts right as written and reversals separately.
- **By depth of negation:** 0, 1, 2+ negations; report reversals and false passes separately.
- **By number of parts:** 2, 3, 4, 5, 6, 7, 8; report agreement, false passes, span/kind, and asked rate separately.
- **“Fixed by one reversal” and “programs right as written”** as standard diagnostics, not only agreement.
- **Decode time and target length by number of parts.**

I would set a reading-time line of 10 s at home and 5 s on the rented A100 for the chosen target. This is an estimate from 4.7 and the 4.9-times-shorter description target; it should be revisited after the target is chosen.

## 6. What I would not do, and open questions

I would not do the following:

- Do not add host code for one kind of instruction.
- Do not put public benchmark sentences in training rows.
- Do not let a model write a rule, a program, or a training target.
- Do not drop parts silently; “asked” is the fallback.
- Do not read round 2’s test/test_long before the pre-registered read.
- Do not train a fix on the 64 dev rows already read; use new held-out rows.
- Do not adopt the outside models’ advice as such without a pre-registered read.
- Do not add grammar rules that forbid correct programs just to reduce errors; each rule must be checked on a held-out set.

Open questions:

- What are round 2’s counted test and test_long numbers? Unknown (4.4).
- Is the model capacity sufficient for 5–8 part programs even with training? Unknown.
- Does the description language cover enough of the public benchmark’s out-of-domain constraints? Unknown.
- What is the token length of the program relative to the 512-token window? The brief gives characters, not tokens; unknown.
- How much of the error is greedy decoding under the grammar versus the model? No ablation is given.
- The reading-time line above is an estimate; the right line depends on the chosen target.

## qwen/qwen3.8-max-prime

*32427 tokens out, 352.7 s, finish `stop`*

1. Diagnosis

I treat round 2’s unseen test and test_long numbers as unknown until the read happens (4.4). The diagnosis below relies on round 2 dev numbers (4.3), the residual error summary (4.5), and the gold statistics (4.6).

**Negation.**

The numbers point most strongly to:

- **The gold’s two spellings.** Every count or size comparison is written in near-equal shares as itself or as `not` around its opposite (4.6). The wording does not reliably follow the spelling (4.6). This gives the reader two valid target forms for one meaning, while the instruction words may imply only one reading.
- **The imbalance between bounds.** There are 548 lower-bound comparisons and 1,648 upper-bound comparisons among nonzero comparisons (4.6). Also, `not` follows `>` or `>=` far more often than it follows `<` or `<=` (4.6). That can teach spurious associations between direction and negation.
- **The form of the target.** The program is long: average 1,546 characters, longest 2,474 (4.7). Each comparison and each `not` is another token where polarity can be lost.
- **The training.** Round 2 reduced one-reversal parts from 49 to 30 and wrong-with-no-single-reversal parts from 66 to 14 on the 64-row dev comparison (4.3). So training changes can move this error class. But round 2’s remaining reversals are still almost all negation: 29 of 30 (4.5).

The numbers do not isolate a hard model limit. Cubby is small (about 576M weights, 445M read per token, 3.1), but the improvement from round 1 to round 2 (4.3) means the residual negation error is not proven to be unfixable model behavior. Greedy decoding under the grammar also is not the main pointer: the grammar removes syntactic free choices (3.5), but it does not choose logical polarity.

Structural vs fixable by data:

- Structural to “a model of this size writes a program this long”: the long program target gives many places for a negation error.
- Fixable by data/target: the two spellings, the wording-spelling mismatch, and the bound imbalance are data problems. A canonical spelling, balanced bounds, and a shorter target can remove much of this class.

**Join.**

The join error class is not yet counted; it was seen by eye on three dev rows (4.5). So I do not know its exact share.

The numbers that do exist point to:

- **The form of the target.** In the current program, the main function must refer to part results and group variables by order. That makes wrong endpoint binding easy. The observed failure was exactly this: checks 2 and 3 were tied in the instruction, but the reader’s or-group joined checks 1 and 2 while each part was right (4.5).
- **The training.** Non-and groups are rare relative to `one`: `either` 838, `if` 396, `unless` 383, against `one` 6,830 (3.3). The reader has much more practice with default and than with explicit conditional or alternative joins.
- **Greedy decoding under a grammar.** The current grammar says an or/if group appears only when the instruction’s words say so and the join follows the generator’s form (3.5), but the observed error shows that this does not force the group to attach to the correct two check numbers.

The numbers do not point to the gold’s two spellings or bound imbalance for this class. They also do not isolate model capacity; the fact that parts are often right while programs are often wrong (round 2: 141/198 parts right, but only 25/64 programs right, 4.3) could be representation, training, or target form.

Structural vs fixable:

- Structural to free program writing: arbitrary indices and group variables make misbinding possible.
- Fixable by data/target: explicit group anchors, more balanced non-and examples, and host compilation of groups can remove much of this class.

**Long instructions.**

Round 1 did not show lower agreement on long rows: agreement was 0.548 on test_long vs 0.508 on test (4.1). But long rows had lower span-and-kind recovery (0.589 vs 0.656) and lower VM-checkable reply rate (0.867 vs 0.960) (4.1). Round 2’s long unseen read is not known (4.4). The eye evidence from round 2 shows a fifth check labelled “check 1” again and no fifth part written (4.5).

The numbers point to:

- **The form of the target.** Longer instructions require longer programs. The average program is already 1,546 characters (4.7). More parts mean more functions, more calls, more chances to restart numbering or drop a part.
- **The split/counting step.** The observed fifth-check failure is a split or numbering failure, not necessarily a logic failure (4.5).
- **The training.** The reader is trained on up to 4 parts and read on 5 to 8 parts (line 4, section 2). Length generalization is being asked for, not trained directly.

The numbers do not point to bound imbalance or the two spellings as the main long-instruction problem. Greedy decoding may contribute if the model cannot recover after a split mistake, but the current grammar already requires parts numbered and called in order (3.5), and the eye error still happened.

Structural vs fixable:

- Structural: a small model writing a long program is more likely to fail as part count grows.
- Fixable by data/target: shorter descriptions, explicit part counts, unique part ids, and host compilation can reduce the length burden. But some length extrapolation remains because training stops at 4 parts while line 4 tests 5 to 8.

2. What the reader should write

**Target A: the program, as today, held to a grammar that grows a rule at a time.**

This target removes, by construction, the program-text errors covered by the grammar: syntax, types, duplicate parameters, call order, one view per loop, and the current join form (3.5). Each added rule removes one free textual choice.

It leaves:

- Negation errors, because the grammar does not know whether the instruction’s meaning requires `<=` or `not >`, `==` or `!=`, etc.
- Join errors, unless a grammar rule can force the correct endpoints; current rules do not.
- Long-instruction errors, because the output is still a full program per part plus a main.

Generality cost:

- CubeLang plus the reply views is the most general of the three targets, because a new instruction can be a new composition in code.
- But the grammar grows rule by rule (3.5). If an unseen composition needs a form not covered by a rule, the reader may fail to produce a compiling program, and the check becomes “asked”.

Unseen kind, line 5:

- A can in principle meet “compiling program for at least half” on held-out kinds and public benchmark out-of-domain constraints, but only if the grammar and model cover those compositions.
- False passes are the main risk, because a compiling program can still be semantically reversed.

Decode time:

- This is the current cost: about 25 s at home, of which the grammar is 0.3 s, and about 9.9 s average on a rented A100 in batches, with shortest about 2 s (4.7).

Set to measure generality lost:

- A “grammar-gap set”: test-only rows made by the generator using known views, columns, part types and group types in combinations not seen in training, plus the public benchmark out-of-domain constraints used only as tests.
- Report: compile rate, asked rate, agreement, false passes, and silent drops.

**Target B: the generator’s own description of each part and of the groups, compiled by the host.**

The reader writes the slots of 3.3: `n`, `count`, `any`, `none`, `all`, `at`, optional `not`, and groups `one`, `either`, `unless`, `if`. The host compiles these into the program. The VM still decides.

This removes, by construction:

- Program syntax errors.
- Long-program boilerplate errors.
- Duplicate parameters, bad loop structure, bad call order, and other CubeLang text errors.
- Much of the long-instruction target-form burden, because the parts-and-groups form averages 316 characters against 1,546 for programs (4.7).
- If the description language is canonicalized, it can also remove the gold’s two spellings for the same logical meaning.

It leaves:

- Wrong slot choice: `count` instead of `n`, `any` instead of `all`, wrong view, wrong column, wrong number, wrong position.
- Wrong operator or direction: `>` instead of `<=`, `==` instead of `!=`.
- Wrong `not` where a `not` is still allowed.
- Wrong group endpoints: an `unless` group referring to parts 1 and 2 instead of 2 and 3.
- Split errors: missing part, extra part, wrong span or kind.

Generality cost:

- The description language is limited to the slots and groups in 3.3 and the reply views/columns in 3.2.
- An instruction whose check is not a composition of those slots cannot be enforced. It should become “asked”, not silently dropped.
- This is the main cost. It may reduce line 5’s compiling/enforceable rate if many unseen kinds fall outside the description language.

Unseen kind, line 5:

- If the unseen kind is a new composition of existing slots, B should handle it safely.
- If it needs a check outside the slots, B asks. That is safer than a false pass but may miss the “compiling program for at least half” bar.

Decode time:

- Exact time is not stated and I do not know it.
- Because the target is 4.9 times shorter on average (4.7), I estimate decode time would be materially lower than A. I would not set a firm number until measured.

Set to measure generality lost:

- A “description-gap set”: held-out kinds and public benchmark out-of-domain constraints, labelled only for whether the check can be expressed in the slots of 3.3.
- Report: enforceable rate, asked rate, compile rate after host compilation, agreement, false passes.
- If no outside-slot examples are known, I do not know how to build a large set from the current generator alone, because the generator’s language is the description language.

**Target C: descriptions where one fits, a free program only where none does.**

The reader first tries to write a description. If the instruction is outside the description language, it may write a free CubeLang program, held to the grammar and the VM. If neither fits or compiles, the check is asked.

This removes, by construction:

- For the covered subset, the same program-text and length errors as B.
- For the fallback subset, it preserves the generality of A.

It leaves:

- All B errors on the described subset.
- All A errors on the free-program subset: negation, join, long output, syntax unless grammar covers it.
- A new selection error: choosing free program when a description would fit, or choosing description when it does not fit.

Generality cost:

- Lower than B, because fallback can cover instructions outside the description language.
- Higher complexity than B, and higher false-pass risk on fallback programs.

Unseen kind, line 5:

- C has the best chance of meeting the compiling/enforceable half bar, because it can fall back to free programs.
- But fallback programs must be strictly held. If they do not compile or cannot be held, they must be asked.

Decode time:

- I do not know. It depends on the fallback rate.
- Estimate: near B if fallback is rare; near A if fallback is common.

Set to measure generality lost:

- A “fallback set”: instructions known not to be expressible in the description language but believed to be expressible in CubeLang.
- Also an “unnecessary fallback set”: instructions expressible in the description language, to measure how often C chooses free program anyway.
- Report: fallback rate, compile rate, agreement, false passes, asked rate.
- I do not know whether the current generator can build gold free-program rows for the fallback set without violating the rule that no model writes training targets. If it cannot, this set must come from test-only public benchmark constraints or hand-built test rows, not training.

If choosing now, I would make B the main target because the measured residual errors are negation, join, and long-program assembly (4.3, 4.5), and B attacks the program-length and syntax part of those errors. I would keep C only if the description-gap set shows B cannot meet line 5. I would keep A only as a fallback under strict grammar and ask rules.

3. Negation and direction

Ranked by expected gain per cost.

**Proposal 1: deterministic normalization, otherwise ask.**

What changes:

- For count and size comparisons, the host accepts only canonical comparisons.
- If the reader emits a single unambiguous negation, such as `not (count > N)`, the host normalizes it to `count <= N` if and only if the complement is deterministic.
- If negation depth is greater than one, if the wording and operator conflict, or if there is no unique normalization, the part is not enforced and is shown as “asked”.
- This applies cleanly to B and C. For A, it applies only where the host can parse the program comparison reliably; otherwise the part is asked.
- No silent drops.

Which set reads it:

- The held-out test set of 2 to 4 parts.
- test_long of 5 to 8 parts.
- The unseen-kind set from line 5.

Pass line written now:

- False passes at most 2%.
- Silent drops 0.
- Unseen-kind enforceable/compiling rate still at least half, as in line 5.

Kill line written now:

- False passes above 5%.
- Any silent drop.
- Unseen-kind enforceable/compiling rate below half.

Cost:

- Low to moderate host compiler logic.
- No serve-time external model.
- The price is more asked checks. I do not know the asked rate until measured.

Expected gain:

- High for safety. The dominant remaining error is negation (4.5), and this converts uncertain negations into asked instead of false passes.

**Proposal 2: bound-span ledger with copy constraint.**

What changes:

- The split must contain, for every numeric bound, the exact bound phrase and a canonical operator: `<=`, `<`, `>`, `>=`, `==`, `!=`.
- The later description or program must copy that operator and number. The grammar or host validator enforces the copy.
- The “if there is one” position-absence pattern remains separate; its `not` is not treated as an ordinary count/size negation.
- Training rows are built by the generator, with the ledger included and checked before the row is kept. No model writes the target.

Which set reads it:

- The 64 dev rows for tuning.
- Then held-out test rows of 2 to 4 parts.
- A held-out bound-direction set, built by the generator, balanced across the 18 meanings counted in 4.6 and across wording forms.

Pass line written now:

- Operator-and-number agreement at least 95% on the bound-direction set.
- False passes at most 2% on held-out test rows.
- Silent drops 0.

Kill line written now:

- Operator-and-number agreement below 85%.
- False passes above 5%.
- Any silent drop.

Cost:

- Moderate. It changes the output format, grammar or validator, training templates, and scoring.
- Output may be longer than pure B, but I estimate it would still be shorter than full programs. That is an estimate.

Expected gain:

- High. The remaining reversals are concentrated in comparisons: 15 in filters or position values, 10 where a count is compared, 5 at view size (4.5). This proposal directly attacks those tokens.

**Proposal 3: operator-only gate.**

What changes:

- Before reading full programs or descriptions on a held-out set, read isolated numeric parts and score only the operator, number, and allowed position negation.
- Use this as a gate for training rounds. If operator reading fails, do not spend the unseen full-program read.
- This is a measurement and training-gate change, not a serve-time change.

Which set reads it:

- A held-out set of isolated parts, built by the generator, balanced across the meanings and spellings of 4.6.
- The existing dev rows may be used for tuning, but the pass/kill read is held out.

Pass line written now:

- Operator-and-number agreement at least 95%.
- False operator choices that would flip pass/fail at most 2%.

Kill line written now:

- False operator choices above 5%.

Cost:

- Low. It needs a new scored set and a small scorer. It does not change serve time.

Expected gain:

- Moderate. It does not fix negation by itself, but it finds direction errors early and cheaply.

4. The join and long instructions

The logic between parts should be read from explicit split evidence. Default is and (3.3). `either`, `if`, and `unless` should be created only when the instruction words say so (3.5), and the split should name exactly which checks the connective joins. The reader should not infer group membership from proximity alone.

**Proposal J1: explicit group anchors.**

What changes:

- When the split sees `either`, `if`, or `unless`, it must write a group line with the connective and the exact check numbers it joins.
- The description or program must reference those same check numbers.
- The host validator rejects or asks invalid references.

Which set reads it:

- Held-out test rows containing non-and groups.
- A generator-built join set with `either`, `if`, and `unless`, using compositions not seen in training.

Pass line written now:

- Group endpoint accuracy at least 95%.
- False passes at most 2%.
- Silent drops 0.

Kill line written now:

- Group endpoint accuracy below 85%.
- False passes above 5%.
- Any silent drop.

Cost:

- Moderate. It changes split format, training rows, and validation. It does not add one-kind host code; it is generic.

**Proposal J2: host-compiled groups.**

What changes:

- The reader writes parts and groups in the generator’s slot form (3.3).
- The host compiles the main join logic.
- Invalid group references are asked, not enforced.

Which set reads it:

- The same join set as J1.
- The general held-out test set.

Pass line written now:

- Agreement at least 95% on replies the VM can check, as in line 3.
- False passes at most 2%.
- No invalid group compiled silently.

Kill line written now:

- False passes above 5%.
- Any silent drop.

Cost:

- Moderate host compiler work.
- Lower decode cost than A because the model does not write the main function.

This removes by construction a large part of the program-assembly error visible in round 2: many parts were right while few programs were right (4.3).

For long instructions, the reader should treat each accounted sentence or clause as a separate part, assign unique ids, and join unjoined parts by and. If the count or numbering cannot be resolved, the affected check should be asked.

**Proposal L1: part-count checksum.**

What changes:

- The split states the total number of checks.
- Each check gets a unique id from 1 to N.
- The grammar or host validator checks that N parts exist and ids are unique.
- If the count fails, the instruction is not enforced as a whole; the unresolved parts are asked.

Which set reads it:

- test_long, 5 to 8 parts.
- Any future held-out length set.

Pass line written now:

- Line 4: agreement on 5 to 8 parts at least 90% of the in-length rate.
- Span and kind at least 90% of parts, as in line 2.
- False passes at most 2%.
- Silent drops 0.

Kill line written now:

- False passes above 5%.
- Any silent drop.
- A missing part that is not shown as asked.

Cost:

- Low to moderate. It is mostly output format and validation.

**Proposal L2: use the shorter description target for long rows.**

What changes:

- For instructions with more than 4 parts, prefer B or C over A, because the description target is 4.9 times shorter (4.7).
- Keep the same VM decision and ask rules.

Which set reads it:

- test_long.

Pass line written now:

- Same as L1: line 4, line 2, false passes at most 2%, silent drops 0.

Kill line written now:

- False passes above 5%, any silent drop, or missing unasked part.

Cost:

- This is part of the target change, not an extra model.
- Exact decode time unknown, but expected lower than A.

5. The measure

The lines in section 2 are the right product bars, but they need diagnostic support.

- Line 1, no silent drops, is right and should remain a kill line. “Asked” is acceptable; silent dropping is not.
- Line 2, split span and kind at least 90%, is right. Round 2 dev reached 0.889 (4.3), but held-out compositions still need the official read.
- Line 3, agreement at least 95% and false passes at most 2%, is the right safety bar because a false pass is the worst outcome. It is strict relative to current dev numbers: round 2 agreement was 0.685 and false passes 0.0935 (4.3). But the bar should not be lowered; instead, use intermediate gates.
- Line 4, length generalization, is right but cannot be judged yet for round 2 (4.4).
- Line 5, unseen kinds, is right, but it should be reported together with asked rate and reason. A compiling but wrong check is worse than an asked check.

What else should be reported:

**By direction of a bound.**

Report separately:

- Lower bounds: “at least”, “more than”.
- Upper bounds: “at most”, “fewer than”.
- Equality and inequality: `==`, `!=`, and “other than”.
- Operator confusion pairs: `<=` vs `not >`, `<` vs `not >=`, `==` vs `!=`.

This is needed because the gold is unbalanced and spelled two ways (4.6).

**By depth of negation.**

Report:

- No `not`.
- One `not`.
- Two or more `not`s.
- Place of negation: filter or position value, count comparison, view size, as in 4.5.
- Direct spelling vs `not`-wrapped spelling.

**By number of parts.**

Report separately for 2, 3, 4, and 5 to 8 parts:

- Span and kind.
- Parts right as written.
- Parts fixed by one reversal.
- Parts wrong with no single reversal.
- Parts not matched to a span.
- Programs or compiled descriptions right as written.
- Agreement.
- False passes.
- Asked rate.
- VM-checkable reply rate.

**By group type.**

Report:

- `one`, `either`, `if`, `unless`.
- Group endpoint accuracy.
- False passes for rows with non-and groups versus rows with only and.

**By target form.**

For A, B, and C, report:

- Compile or host-compile rate.
- Asked rate and reason.
- Agreement.
- False passes.
- Decode time.

**Time line.**

No time line is set. I would set one only after B is measured, because B’s shorter target may change the cost a lot (4.7). If forced now, I would set an estimated pass line: p95 read time at most 20 seconds on the home GPU for 2 to 4 parts, and at most 30 seconds for 5 to 8 parts. Estimated kill line: p95 above 60 seconds. These numbers are estimates and should be revised after measurement.

6. What I would not do, and open questions

I would not:

- Add host code for one kind of instruction. The reader’s value is composition, not hand rules.
- Put public benchmark sentences into training rows. They are tests only.
- Let a model write rules, programs, or training targets. The generator and VM must build and check rows.
- Trade away the false-pass kill line to get agreement.
- Allow silent drops to improve scores.
- Train on 5 to 8 parts and then claim line 4 unchanged, unless a new pre-registered length bar is written.
- Infer round 2 test or test_long performance from dev. The unseen read has not happened (4.4).
- Add grammar rules that patch one observed sentence without a general composition.
- Use an external model at serve time.

Open questions:

- What are round 2’s test and test_long numbers? I do not know (4.4).
- How often would B ask on unseen kinds? I do not know.
- Can the current generator build enough outside-description examples to measure B’s generality loss? I do not know.
- Can C’s free-program fallback be evaluated without violating the no-model-written-program rule? I do not know how, unless the generator or test-only benchmark constraints provide gold checks.
- What asked rate is acceptable to users? Not stated.
- What decode time will B or C actually have? Not stated; only the current A times are known (4.7).
- Whether the remaining negation error after canonicalization is a model limit or a data/target problem. The current numbers do not decide that.

## z-ai/glm-5.3-prime

*18568 tokens out, 127.6 s, finish `stop`*

# 1. Diagnosis

The three error classes of 4.5 are negation, the join, and long instructions. The candidate causes are the target's form (~1,500 characters), the gold's two spellings, the bounds imbalance, the model, greedy decoding under a grammar, and the training. One caution first: the round-1-to-round-2 comparison (4.3) changes decoder, grammar and hardware along with the training, so no cause is isolated cleanly by it.

**Negation.** The numbers point to the gold's two spellings and the bounds imbalance, with greedy decoding as the amplifier. Round 2 leaves 30 reversals on 198 parts, of which 29 are negations (4.3, 4.5); round 1 left 49 reversals and 66 parts wrong with no single reversal on 196 (4.3). So the data moved this class substantially, which argues the residual is also data-shaped. The gold writes every count and size comparison two ways in near-equal shares across all 18 meanings, about 4,150 parts, and the wording does not follow the spelling (4.6) — the model is asked to pick a spelling the words do not determine. The imbalance is sharper: 548 parts mean "at least/more than" against 1,648 meaning "at most/fewer than" (4.6), and after `>` or `>=` the gold goes on with a `not` in 64–90% of parts against 9–32% after `<` or `<=` (4.6). A greedy decoder facing an undetermined choice takes the majority token; the residual reversals — a comparison for its complement — are exactly flips toward or away from the majority direction. The grammar does not yet constrain this: the rule refusing a `not` around a count or size comparison is proposed, not built (5). This cause is **data-fixable** (one spelling per meaning, balanced bounds — already proposed in 5). What is structural is only the residue: under greedy decoding, a wording the gold itself spells two ways will leave some ambiguity at any model size. It is not structural to program length — a reversal is one token in a 1,546-character program (4.7), and the scaffold around it is now usually right.

**The join.** No counted numbers exist (three rows by eye, 4.5), so this rests on indirect evidence. Two causes fit. First, the training: joined groups are rare — `one` 6,830 against `either` 838, `if` 396, `unless` 383 (3.3), so roughly 1,617 of 8,447 groups, about 19% (computed from 3.3). Second, the form of the target: the instruction's "unless" sits between two checks, but in the program the join is expressed as flag assignments in the main (the `g1`, `g2` sequence in 3.3's example), hundreds of characters from the word that triggered it. The reader's error — an or-group joining checks 1 and 2 where the instruction ties 2 and 3 — is an attachment error, and the program form places the attachment decision at maximal distance from its trigger. That part is **structural to a model of this size writing a program this long**: a 576M-weight model (3.1) must bind an early word to a late variable pair across a long greedy decode. The group-imbalance part is **data-fixable**. The grammar rules added 2026-11-13 (join in the generator's form; an or/if group only when the words say so, 3.5) bind the form but not the attachment, and the error persisted by eye under rules 1–9.

**Long instructions.** The numbers point to training, not structure. Round 1 showed no length loss — agreement 0.548 on 5–8 parts against 0.508 on 2–4 (4.1) — so a model of this size generalizing past trained length is demonstrated, not contradicted. Training never saw 5–8 parts (train and dev are 2–4, 3.3). The round-2 eye-seen failure — a fifth check labelled "check 1" again, no fifth part (4.5) — is a restart in the split, and as far as the brief states the split is not grammar-held (3.5 governs the program's decode; 3.4 shows free split lines). That is a **decoding cause fixable by a grammar on the split**, not by data — and line 4 of section 2 forbids the data fix, since it would trade away the line's meaning. The one structural caveat: compile rate drops with length (0.867 against 0.960, 4.1), and longer rows mean programs longer than the longest seen (2,474 characters, 4.7); whether the 512-token window (3.1) bears on this I don't know, because the brief gives characters, not tokens.

**The model as a cause.** For no class do the numbers isolate the model: every class moved between rounds with data and grammar changes (4.3). Programs right as written rose from 5/64 to 25/64 and parts wrong with no single reversal fell from 66 to 14 (4.3); the remaining 14 may be model-limited, but the evidence does not say. What is clearly structural to "a model of this size writes a program this long" is the **decode time** — about 25 s at home per instruction (4.7) — and the sheer surface on which one greedy token lands. On the current evidence, length is a cost problem more than an accuracy bottleneck.

# 2. What the reader should write

**A — the program, as today.** Removes by construction only what the nine grammar rules remove (3.5): type errors, duplicate names, call-order errors, the default "and". Leaves all three classes of 4.5: the negation choice is unconstrained (the refusing rule is only proposed, 5), the join attachment is bound in form but not in binding, and the split's numbering is outside the program grammar. Generality is the highest of the three: any check CubeLang can express, and an unseen kind is a new composition of the views — the reader's own premise (section 2). Line 5 is best served: nothing in the target bounds the semantics, so "a compiling program for at least half" of unseen kinds is a statement about the model, not the language. Decode: about 25 s at home, 9.9 s average on a rented A100 in batches of 32 (4.7). No generality-measuring set is needed for A itself; A is the ceiling against which B and C are measured.

**B — the generator's descriptions, host-compiled.** The target is 316 characters against 1,546 — 4.9 times shorter (4.7). Removes by construction: the program scaffold entirely — the loops, counters, flag assignments and main of 3.3's example — so the 14 parts wrong with no single reversal (4.3) should mostly disappear, the join becomes a three-token datum (`["unless", 1, 2]`, 3.3) whose attachment is local and grammar-constrainable, and compile failures should fall (the host compiles; the 0.960/0.867 VM-checkable rates of 4.1 were program-compile rates — this is an estimate). Leaves: the negation — the operator and `not` are still choices in the description, and the 29 reversals would mostly persist, though the choice now sits next to the instruction's words and a one-spelling rule is trivial to enforce on a 316-character schema; and the split errors (13 parts unmatched on the 64 rows, 4.3), since the parts list still presupposes a split. The generality cost is real and unmeasured: the description language has the slots of 3.3 (n, count, any/none/all, at, not; filters of one or two column comparisons; four group forms). A check needing anything else cannot be written, and under the hard rules it must be "asked." The brief does not state CubeLang's reach beyond these slots, so the size of the gap is unknown to me. Line 5 is at risk twice over: unseen kinds and the benchmark's out-of-domain constraints may fall outside the slots, and the line's phrase "a compiling program" changes meaning when the host compiles descriptions. There is also a rule-adjacent cost: each new slot is host code, so B trades "a hand rule per instruction kind" for "a slot per check kind" — coarser, and within the letter of the rule, but the growth pressure moves to the host. Decode time: roughly one fifth of A's by the character ratio — about 5 s at home, about 2 s on the A100 (estimates; characters are not tokens, so the true ratio is unknown to me). The measuring set: a held-out set of instructions whose checks the generator can draw and the VM can run but whose form is outside the slots — built the reverse way per the rules (generator draws, words it, VM verifies on pass and fail replies), read once, reporting the fraction B must ask that A writes correctly.

**C — descriptions where one fits, a free program only where none does.** Removes the scaffold errors on the describable fraction — which is all of the current set by construction (the 10,064 parts of train and dev all fit the slots, 3.3) — and keeps A's generality as a fallback, so line 5 keeps a real path. Leaves the negation on both paths, and adds a new error class the other two don't have: the mode choice. Forcing a description where none fits is a false-pass risk (a description that compiles but says the wrong thing); writing a program where a description fits only loses safety. The mode judgment must itself be trained, which requires rows whose checks are outside the slots so the reader learns the boundary — buildable under the rules, since the generator draws them. Decode time sits between A and B, plus the mode decision. The measuring set: the same out-of-slots set as B, with the mode error counted on it — how often the reader writes a description for an out-of-slot check is the dangerous number.

My reading of the numbers: the measured residue is concentrated in single semantic choices (29 negations, 4.3), not in scaffold corruption, so B's main by-construction gain over A is smaller than it looks — until round 2's test and test_long reads (4.4) say otherwise. C captures most of B's safety and none of its ceiling loss, at the cost of the mode error. I would build the out-of-slots set before choosing, since it is the one number that decides between B and C, and it can be built now under the existing rules.

# 3. Negation and direction

Ranked by expected gain per cost. Pass and kill lines are written now, before any read.

**1. Twin-run disagreement → asked (serve time, no training).** What changes: for each part whose check has a well-defined one-reversal twin — the complement of a comparison, the mirrored comparison, or the part's `not` toggled, the classes counted in 4.2 and 4.3 — the VM runs both the written check and its twin on the reply. Where they agree, the part is enforced as written; where they disagree, the part is marked "asked," shown to the user as not enforced. This keys on the check's form, not the instruction's words, so it is generic host code within the "no host code for one kind of instruction" rule, and it honors "asked is better than a wrong check" (section 2). Which set reads it: the dev rows already read, to size the asked rate, then the unread test and test_long rows, pre-registered. Pass line: false passes from reversal-class parts at most 2% on test (line 3's bar), with asked parts at most 20% of parts. Kill line: asked above 40% of parts (the reader has become useless), or false passes unchanged. Cost: no training, no decode change — the model's 25 s is untouched; the VM runs each reply once more per ambiguous part, and the VM's run cost is not stated in the brief, so I don't know it. Expected gain: round 2's false passes are 46/492 = 0.0935 with 30 reversals as the main counted residue (4.3); this converts the disagreeing half of those into "asked" exactly on the replies where the distinction matters.

**2. A direction tag per bound (a minimal description for the comparison only).** What changes: the reader writes, per part, one closed tag for each bound — at-most, fewer-than, at-least, more-than, exactly, other-than, not-equal — plus the number, as a grammar-held line; the program's operator must match the tag. The tag puts the direction choice in a seven-word vocabulary adjacent to the instruction's own words, where round 2's four-times weighting on key tokens (3.6) can bear on it directly. Which set reads it: the 64 dev rows first, then test and test_long. Pass line: complement-class reversals at most 5 per 198 parts on dev (from 29, 4.3), and false passes at most 2% on test. Kill line: reversals not halved, or the tag line itself wrong on more than 10% of parts — the tag has only moved the error. Cost: targets relabelled mechanically from the generator's parts (the generator already knows the direction; no model writes targets), one training round, a small grammar. Expected gain: directly targets the 29 negations of 4.3, which sit 15 in a count's filter or a position's value, 10 where a count is compared, 5 at a view's size (4.5).

**3. A grammar on the split.** What changes: the split's lines are held to a grammar — each line typed (check, condition, process, none), check numbers strictly increasing, join words attached to the checks they tie. Which set reads it: test_long and test, unread rows. Pass line: on test_long, span and kind at least 90% of parts (line 2) and no numbering restarts; agreement at least 90% of the in-length rate (line 4). Kill line: silent drops above 0 (line 1's kill), or restarts on more than 5% of rows. Cost: a grammar smaller than the program's nine rules (estimate), no training. Expected gain: uncertain — the failure it targets is two rows seen by eye (4.5); it ranks last because the evidence is thinnest, though the cost is also smallest.

# 4. The join and long instructions

**The join.** Read it at the split, as an explicit structure over check numbers in the generator's own form (the `["unless", 1, 2]` of 3.3), and compile the program's main from it. The measured failure is attachment — which checks the "unless" ties — and in the program form the attachment lives in flag assignments remote from the trigger words (3.3's main). A join line of three tokens makes the attachment a local, grammar-constrainable choice, and the existing rule "the join in the generator's form" (3.5) then has a direct target. A further virtue: the model need not know the expansion of each group form at all — the example of 3.3 shows `unless` compiled as an or, but the brief does not show the `if` expansion, and under this proposal that knowledge stays in the host compiler, not the model. Before building it, count the join error on dev: three rows by eye is not a measurement (4.5). Lines, written now: pass — the reader's group structure equals the generator's on at least 95% of test rows (line 3's agreement made structural); kill — false passes attributable to join mis-binding above 2%, tighter than line 3's 5% because a mis-bound join over correct parts is precisely a false-pass factory. The data side — more `either`, `if` and `unless` rows, from 1,617 joined groups in 8,447 (computed from 3.3) — is a round-3 generator change, cheap under the reverse-build rules.

**Long instructions.** Three proposals. First, the split grammar above (monotone numbering), targeting the restart seen by eye (4.5). Second, host-side accounting: every sentence must be typed by the split — a part, "asked," or "not an instruction" — and a numbering restart or untyped sentence forces "asked" for the unaccounted parts, never a silent drop; this makes line 1's kill mechanically enforced rather than hoped for. Third, do not train on 5–8-part rows: line 4 is a written line about generalizing from at most 4 parts, and the brief's hard rules forbid trading it away. If line 4 still fails after the first two, the honest outcome is asked parts on long instructions, reported as such. Lines: line 4 as written — agreement at least 90% of the in-length rate on 5–8 parts — plus, per deliverable 5, agreement reported per part-count so a cliff at 5 is visible rather than averaged away.

# 5. The measure

The lines of section 2 are the right bar in kind: they enforce the accounting rule with a kill, rank the false pass as worst, test length generalization rather than length exposure, and test unseen kinds. Three gaps.

First, no line bounds the asked rate. A reader that marks every part "asked" passes lines 3 and 5's false-pass bars trivially; line 1 counts asked parts but caps nothing. I would report the asked rate on every read and pre-register a line for the next one — asked at most 20% of parts on test (my proposed number, a choice, not a measurement).

Second, line 5's "a compiling program for at least half" is well-posed for target A but changes meaning under B or C, where the host compiles descriptions. If the target changes, re-register the line in the new terms before the read.

Third, the aim — a frontier model's quality at one hundredth of the cost or less — has no time line attached. The brief asks what I would set: at most 10 s per instruction at home, and at most 3 s average in a batch of 32 on a rented card, for the next read (my proposed lines; current numbers are about 25 s at home and 9.9 s average on the A100, 4.7, so the line demands roughly a 2.5× speedup — whether that is reachable is unknown to me).

What else should be reported, beyond the section-2 lines:

- **By direction of bound:** agreement, false passes and reversal rate for "at least/more than" parts against "at most/fewer than" parts. The 548-to-1,648 imbalance (4.6) means the aggregate is dominated by the majority direction; the minority direction's error rate is invisible in it.
- **By depth of negation:** no `not`; `not` around a comparison; `not` around any/all/none; `not` around a part; and the "if there is one" construction, which the generator itself writes as a `not` around the opposite test (3.3). The 29 reversals sit at depth one (4.3, 4.5); whether depth two collapses should be known.
- **By number of parts:** 2 through 8, per count. Line 4 gives only a ratio to the in-length rate; a cliff at 5 parts would satisfy no one's curiosity.
- **By group kind:** join accuracy for `one`, `either`, `if`, `unless` separately, since joined groups are about 19% of the total (computed from 3.3) and a large error rate within `unless` is a small number in the aggregate.
- **By gold spelling, on dev only:** reversal rate conditioned on which of the two spellings the gold used and on the wording frame — a direct test of 4.6's claim that the wording does not follow the spelling.
- **Compile rate** ("replies the VM could check," 4.1), reported separately from agreement and by length.
- **Decode time per read** (4.7), since the system's aim is cost.

# 6. What I would not do, and open questions

I would not train on 5–8-part rows; it trades away line 4. I would not move wholly to descriptions (B) before the out-of-slots set exists and has been read once — line 5 and the no-host-code rule make the description language's coverage the binding constraint, and its size is unknown. I would not bundle the margin loss on the operator token with other changes; it is already kept for a round where it is the only change (5). I would not let the twin-run policy drop a part silently — it must mark "asked" (line 1). I would not fix the join from three eye-seen rows without counting it first (4.5). I would not read round 2's test or test_long rows twice, or design a fix after seeing them and read it on the same rows (pre-registration). And I would not project decode times from the 4.9× character ratio (4.7) as though characters were tokens — the token ratio is unknown to me.

Open questions. Round 2 on unseen compositions and on 5–8 parts has not been read (4.4); the counted read should precede any round-3 decision, since the whole diagnosis above rests on dev rows. The reversal residue is not conditioned on gold spelling or direction — 4.6 counted the gold, not the errors against it. The asked rate today is not stated anywhere in the brief; I don't know it. CubeLang's reach beyond the description slots is not stated; the size of B's generality loss is unknown to me. The expansion of `if` groups is not shown in the brief. Whether the split is decoded under any grammar today is not stated. The VM's own run cost — which the twin-run proposal depends on — is not stated. Whether the 512-token window (3.1) bears on long programs I don't know, because the brief gives program lengths in characters, not tokens. And the round-1-to-round-2 comparison (4.3) confounds decoder, grammar and hardware with the training change, so the clean attribution of round 2's gains is unknown.

## x-ai/grok-4.7

*38692 tokens out, 642.3 s, finish `stop`*

The unread test and test_long reads have not happened (4.4). Section 4.3 is the shorter 64 of 100 dev rows, and the decoder and grammar differ as well as the training. I treat 4.3 as a confounded dev comparison, not as a line result. The three join rows and the two long rows are not rates. Where a cause is not separated by the numbers, I say so.

## 1. Diagnosis

**Negation.** The numbers point to the gold’s two spellings (4.6), not to program length, the model, or greedy decoding.

On the 64 dev rows, round 2 left 30 one-reversal parts: 29 negations (a comparison for its complement, or `==` for `!=`) and 1 mirror (4.5). By place, of those 30: 15 in a count’s filter or a position’s value, 10 where a count is compared, 5 at a view’s size. I do not know which place holds the one mirror. Round 1 on 100 dev rows had a wider reversal mix: 35 `==`/`!=`, 29 complements, 5 mirrors, 1 missing `not` (4.2). What remained after round 2 is almost only direction.

The same meaning is written two ways in near-equal shares, and the words do not follow the spelling (4.6). Count at most N is 244 as `<=` and 260 as `not >`; fewer than N is 210 and 194; other than N is 329 and 361; size at most N is 196 and 189; about 4,150 parts across 18 meanings. “None” is 996 against `not any` 1,051; “any” is 383 against `not none` 394. In the 3.3 row the words say “at least one paragraph that has at most 2 sentences” and the part says `not all` with `>`. After `>` or `>=` the gold continues with a `not` in 64 to 68% of size parts and 77 to 90% of count parts; after `<` or `<=`, in 9 to 32%. The operator token is not determined by the words, and the gold’s own use of `not` is direction-asymmetric.

The bound imbalance is real and separate: among comparisons with a number other than 0, 548 mean “at least” or “more than” and 1,648 mean “at most” or “fewer than” (4.6). I do not know which way the 29 negations go, so I cannot say the numbers point to that imbalance as the source. It is a standing bias, not an identified one.

It is not the long target. On those 64 rows, parts right as written rose from 59/196 to 141/198, parts with no single reversal fell from 66 to 14, and programs right as written rose from 5/64 to 25/64 (4.3). A model that writes 25 of 64 programs fully right is not failing at emitting CubeLang. Programs lag parts because a program is right only if every part and the join are right. That multiplication is structural to one long program. The negation itself is a local operator choice. I do not know the part-count mix on the 64, so I will not estimate how much of the 25/64 the multiplication explains.

It is not, on these numbers, the model. The same size of model, the same LoRA job, cut the unstructured errors when the data and the grammar changed. Held-out answer loss fell from 0.050 to 0.034 (3.6). I do not know whether rank 16 caps the operator token.

Greedy decoding under the grammar is not isolated. Section 4.3 changes decoder, grammar, and training together. The nine rules do not choose operator direction; the rule that would refuse a `not` around a count or size comparison is not built (3.5, 5). I cannot blame greedy decoding, and I cannot credit round 2’s training alone for the 4.3 move.

Training can move this class: it already moved the other classes on these dev rows. It has not removed the two spellings. That residue is in the data. Data can fix it. It is not structural to “a model of this size writes a program this long.”

**The join.** No rate exists. On three dev rows, seen by eye and not counted, the instruction tied checks 2 and 3 and the reader’s or-group joined checks 1 and 2, with every part right in one of those rows (4.5). I do not know if that was true in all three, or whether the split lines were right and only main was wrong.

What that observation points to is an index error, not a negation. Two spellings and the bound imbalance do not fit: the parts were right. Group-type imbalance is real — `one` 6,830, `either` 838, `if` 396, `unless` 383 on 3,300 train and dev rows (3.3) — but those rows kept an or-group and missed the indices, so the skew in group type is not what they show.

The form of the target is the cause the setup points to. The join is written again in main, after the part functions, in a program of 1,546 characters on average (3.3, 4.7). The split already has a condition line (3.4). Re-emitting indices at the end is a long-range copy the program form creates. It is not proven structural to this model’s size: there is no count, and 25/64 short programs were fully right (4.3). I do not know whether 1,546 characters exceeds the 512-token window (3.1). Data with more non-initial attachments might reduce it; that is not shown. Relative to the three rows, the form is the better account, and the data might still help. I would not call the class measured.

**Long instructions.** Not counted for round 2 (4.4, 4.5). On two rows of 5 checks, seen by eye, the fifth check was labelled “check 1” again and no fifth part was written. I do not know which set those rows came from. Dev is 2 to 4 parts (3.3), so they are not dev. If they are from test_long, they should be excluded from the official read.

Round 1, on rows no read had seen, does not show an agreement penalty for length: test 195/384 = 0.508, test_long 171/312 = 0.548; false passes 44/384 = 0.115 and 35/312 = 0.112 (4.1). Span and kind was worse on long rows (188/319 = 0.589 against 99/151 = 0.656), and fewer replies could be checked (312/360 = 0.867 against 384/400 = 0.960). Length was not the agreement trouble in round 1. It was a trouble for accounting and for whether the VM got a check.

The two eye rows point to the training cap of 4 parts (line 4) plus the reader owning check numbers, not to spellings or bounds. Round 1 could sometimes write long programs, under grammar rules 1 to 5, so a hard failure of this size on long output is not shown. Round 2 on 5 to 8 parts is unknown (4.4). I do not know the character length of a 5-to-8-part program, or whether it exceeds the window.

More data of 2 to 4 parts, in the same format, is not shown to fix a restart at “check 1.” Training on 5 to 8 parts would abandon line 4. So, under the line as written, this is structural to the cap and to who emits the numbers. It is not structural to model size on the round 1 evidence. A format in which the host owns the numbers can be tried without long training rows. That is a format fix, not a data fix inside the current target.

## 2. What the reader should write

The description language I use below is only what section 3.3 states: `n`, `count`, `any`, `none`, `all`, `at` (first, second, last; “if there is one” as a `not` around the opposite test), a filter of one or two column comparisons, an optional `not`, groups `one`, `either`, `unless`, `if`, and default and, over the views and 17 columns. I do not know what share of user instructions, or of the public benchmark’s out-of-domain constraints, falls outside that.

A false pass is worse than “asked” (section 2). None of the three targets is shown to meet line 3. The best dev figure, confounded, is agreement 337/492 = 0.685 and false passes 46/492 = 0.0935 (4.3). Round 1 test false passes were 0.115 and 0.112 (4.1). Both sit above the kill line of 5%.

**(A) the program, grammar growing a rule at a time.**

Removes by construction: none of the three classes in 4.5. The nine rules remove syntactic free choices (3.5). They do not choose direction. They do not tie an or/if group to the two checks the split names; that rule is not built (5). They did not, on the two eye rows, stop a fifth check being labelled “check 1.” I do not know if the split lines are under the grammar at all.

Leaves: negation, the join index error, and the long-instruction numbering failure, as far as those are known.

Generality: the widest of the three, for any check the views can support and CubeLang can say. I do not know what CubeLang cannot say over those views. Cost in generality against the description language: none. Cost against instructions the views cannot see: the part is asked if the program does not compile. I do not know how often that happens.

Unseen kinds (line 5): this is the path the line names, a compiling program. Where one is written, the false-pass kill is 5% and the pass line is 2%. Current reads are above the kill on seen compositions. An unseen kind is not likely to be better. I do not know that. Chasing “at least half compile” with this false-pass rate hits the kill. Asked is the better failure.

Decode time: about 25 s at home, of which the grammar is 0.3 s; on a rented A100 in batches of 32, about 2 s for the shortest and 9.9 s on average over the 64 dev rows (4.7). I do not know if the 25 s is the same 64, or a mean, or a median.

**(B) the generator’s descriptions, host-compiled.**

The host lowering is the one the training pipeline already uses to keep a row (section 2, 3.3). It is not host code for one kind of instruction, as long as it stays that one lowering and does not gain a branch per benchmark constraint.

Removes by construction: errors that exist only in the CubeLang lowering. The 3.3 part function implements `not all` as a count, a compare to the size, and a flip. A wrong flip with a right meaning is a program bug. The host compiler does not make that bug. B also removes a join that is wrong only in main’s if-nesting, if the group tuple is right.

Does not remove by construction: a wrong operator or a wrong `not` in a slot. The 4.5 reversals are defined on the check’s meaning. The reader can emit the complement in `op` just as it can emit it in a loop. If descriptions keep both spellings (4.6), the ambiguity moves into a shorter text; it does not disappear. Wrong group indices are still writable: `["unless", 0, 1]` instead of `["unless", 1, 2]`. The distance is shorter. That is not removal by construction. The split relabel of a fifth check is left if the split format stays. Line 2 needs spans and kinds, so I assume the split stays. I do not know its character length. The 4.9 times shorter figure is program against parts and groups, 1,546 against 316, not total output (4.7).

Generality cost: any check that is not a composition of those slots cannot be written. The reader must ask, not invent a program. That is the real cost. I do not know its size.

Unseen kinds: a new composition of the slots can be written, compiled, and held to line 5’s agreement and false-pass lines. A kind outside the slots, including a benchmark constraint the slots cannot say, does not get a compiling program unless someone adds host code, which the rule forbids. Those parts are asked. Asking can miss “a compiling program for at least half” without being a false pass. I do not know whether the benchmark stratum is mostly inside or outside. Until that is measured, B can fail line 5’s compile clause by construction on the outside portion, and that is still better than a false pass above 5%.

Decode time: I do not know. The program portion is 4.9 times shorter in characters, and the longest falls from 2,474 to 567 (4.7). The A100 times run from about 2 s on the shortest to 9.9 s on average, so length is part of the time. I will not turn 4.9 into a seconds figure. The grammar’s 0.3 s is not the home bottleneck.

**(C) descriptions where one fits, a free program only where none does.**

Removes by construction: the same lowering errors as B, and only on the rows where a description fits and is what gets compiled. On the free-program residue it removes nothing that A does not remove.

Leaves: slot-level negation and wrong group indices on the description rows; all of 4.5 on the free-program rows; the split-numbering failure unless the split is changed.

Generality: the same as A, if the free arm is real. The cost is the decision. If the reader forces a description that does not match the instruction, that is a wrong check, and a false pass is the worst outcome. If the host treats a description that does not type-check as the signal that none fits, that signal is sound only for ill-formed slots, not for a well-formed description of the wrong check. I do not know a host test that detects “a legal description of the wrong meaning” without being a second reader.

Unseen kinds: the free arm is how C tries to meet line 5’s compile clause. It is not safe to turn on. Seen-composition false passes are already above the kill (4.1, 4.3). Writing a free program for an unseen kind, on that evidence, is the path that hits the kill in order to make the half-compile clause. I would not serve the free arm until a pre-registered read shows false passes at most 2% where a free program is written. Until then C, as I would run it, is B plus asked, and line 5’s compile clause stays unmet for out-of-slot kinds rather than being met by a wrong check.

Decode time: I do not know. It depends on the mix, and the mix is not stated. Each free program costs about what A costs; each description costs what B costs. A second pass, description then program, would add time. I do not know how much.

**Set for the generality lost.** Build it before any of these is chosen, by the same reverse path as training: the generator draws the check, templates word it in English and French, the VM runs replies built to pass and to fail, no model writes the target, no benchmark sentence enters a training row (section 2). Three strata, read once:

1. In-slot checks, compositions not in train, dev, test, or test_long. Control.
2. Out-of-slot checks that use only the existing views and columns, so a generator program still runs in the VM. This is the generality B loses.
3. The public benchmark’s out-of-domain constraints, test only, each labelled in-slot or out-of-slot by whether a description exists. I do not know who labels that, or how often the label is uncertain. Where it is uncertain, say so and do not count the row as in-slot.

Lines written now, for that set, not for the 64: in-slot, agreement at least 95% and false passes at most 2%, kill above 5% (line 3). Out-of-slot, report B’s ask rate as the generality lost; do not kill B for asking. For A, and for C’s free arm, false passes at most 2% where a program is written, kill above 5%; a compile rate is reported and is not a success if false passes sit between 2% and 5%. I do not know how large a set can be built. A stratum of 300 would match the existing test splits (3.3). That is a size I am choosing, not a measured need.

I would write B for every in-slot check, and ask otherwise, until the generality set and an unread quality read exist. I would not write A’s program, and I would not turn on C’s free arm, while false passes on the reads we have are above the kill line.

## 3. Negation and direction

Section 5 already has one spelling, a grammar rule that refuses a `not` around a count or size comparison, balanced bounds, and the parked margin loss. I do not repeat those. The refuse-`not` rule does not stop `>` written for `<=`, and it does not cover the 15 reversals in a filter or a position value (4.5). These three are beyond that. I rank them by expected gain on false passes and on those reversals, divided by cost. I do not know the share of the 46/492 false passes that are negation reversals, so the expected gains are not counts.

**First: ask when the emitted comparison is the complement of a template relation in that part’s own words.**

What changes. At serve time, if the part’s span contains a relation from the generator’s own English or French templates, and the emitted comparison is the complement of that relation, the part is asked, not enforced. Nothing is dropped (line 1). “If there is one” on a position stays as section 3.3 writes it, a `not` around the opposite test; this rule does not flag that pattern. If the span has no template relation, the rule does not fire and does not guess. No retrain. No new view.

Which set reads it. A new held-out set, built before the run, 2 to 4 parts, compositions not in train, dev, test, or test_long. Read once. Not the 64. Not the round-2 test read, which should be spent on round 2 as planned (4.4).

Pass line, written now: on that set, false passes at most 2% of replies that remain enforced; parts asked for this reason at most 10% of matched parts; silent drops 0.

Kill line, written now: false passes above 5% on enforced replies; or this ask fires on more than 25% of matched parts; or any silent drop.

Cost: no training round. I do not know the engineering hours. It is small beside round 2’s 1,782 steps (3.6). Expected gain per cost is the best of the three if the templates cover the spans, because a false pass is the worst outcome and the cost is a check. If the templates do not cover the spans, the gain is small. I do not know the coverage, in English or in French.

**Second: a grammar license on the operator in a filter and in a position value, not only a refused `not` on a count or size.**

What changes. One more grammar rule, same mechanism as “a part says only what its own words say: numbers, positions, views” (3.5). For a filter comparison and for a position’s value, the operator must be the one the part’s own words license, in either language, from the same template list. If the words license nothing the grammar can see, the rule does not force an operator. This is not the section 5 rule. That rule refuses a wrapper. This one licenses `<=` against `>` inside the places where 15 of the 30 reversals sit, and where a count is compared or a size is compared if those words are in the part.

Which set reads it. The same new held-out set, or a second one of the same kind if the first was spent. Read once. Not the dev rows the reversals were seen on.

Pass line, written now: false passes at most 2%; agreement at least 95%; negation reversals at most 5% of matched parts; silent drops 0.

Kill line, written now: false passes above 5%; or negation reversals above 15% of matched parts; or the rule asks or fails to compile more than 20% of parts. The 15% and 20% are lines, not measurements. I set 15% next to the dev residual of 30/198 reversals so a rule that does not beat that residue on a clean set is a kill. I do not treat 30/198 as a test result.

Cost: one grammar rule. The existing grammar is 0.3 s of about 25 s (4.7). I do not know the hours to write the rule. Expected gain is high if the grammar can see the relation word, including in a sentence with two numbers, as in the 3.3 row. I do not know if it can. Over-fire is the risk, which is why the ask share is in the kill line. Gain per cost sits under the first proposal because of that risk and because a post-check is cheaper than a constrained decode if both depend on the same lexicon.

**Third: the reader writes the surface relation; the host lowers it.**

What changes. The training target for a count, a size, a filter, and a position value becomes the relation the words say (“at most,” “fewer than,” “at least,” “other than,” and the French templates), not the pushed spelling. The host applies the generator’s existing lowering to one spelling. The model stops doing the rewrite that 4.6 shows the words do not match. This is not section 5’s one spelling. That still asks the model to emit the pushed form. Here the host pushes. One change in the round, not mixed with a new grammar rule or a new pair scheme (the 4.3 confound).

Which set reads it. A new held-out set, built before the retrain is scored, read once. The 64 are already read.

Pass line, written now: agreement at least 95%; false passes at most 2%; negation reversals at most 3% of matched parts.

Kill line, written now: false passes above 5%; or agreement below 80% on that set.

Cost: one training round on the order of round 2, 12,018 records, five epochs, 1,782 steps (3.6), if only targets change. I do not know whether the record count changes, or the money, or the wall-clock. This has the largest expected agreement gain, because it matches the stated mismatch and does not need a new parser. It has the worst gain per cost of the three because the cost is a full round. I would run it only as its own round, and only if the first two leave the unread quality read short of line 3. I would not run it on top of an unscored grammar change.

## 4. The join and long instructions

Default and should not be written by the reader. Parts and groups not otherwise joined are joined by and (3.3), and the grammar already defaults to and (3.5). The host applies that. The reader writes a group only when the words say either, if, or unless.

**Join.** Section 5 would still have the reader emit the group, tied by a grammar rule to the two checks the split joins. I would go one step past that. The reader does not write main’s join, and does not write the indices a second time. The split already marks a condition clause (3.4). The host builds the group tuple from that attachment and compiles it with the generator’s lowering. The VM runs the compiled program. If the split names a condition but the attachment is not two accounted-for parts, the group is asked, not guessed.

I know one lowering, the 3.3 example: `["unless", 1, 2]` becomes or of those two parts, then and with the unpaired part. I do not know if every unless, if, and either lowers that way. The host should use the generator’s lowering, not a new one I invent here.

This removes, by construction, the error of a right split and a wrong re-encoding in main. It does not remove a wrong attachment in the split. The three eye rows do not say which of those they were. They are not the test.

Pass line, written now, for a fresh held-out of instructions that contain either, if, or unless, built before the run and read once, not those three rows: the group word and the two part indices agree with the generator on at least 95% of such groups; false passes at most 2% of replies; silent drops 0.

Kill line, written now: false passes above 5%; or the indices are wrong on more than 20% of such groups; or any silent drop. The 20% is a line, not a measured rate. No measured rate exists (4.5).

**Longer than any seen in training.** Line 4 stays: trained on up to 4 parts, read on 5 to 8, agreement at least 90% of the in-length rate. I would not put 5-to-8-part rows in training.

How to read them. The host owns the check counter. It numbers accounted-for spans in order. The reader does not emit “check 1.” A repeated “check 1,” as on the two eye rows, cannot reset a counter the reader does not own. Each part is read from its own span, one part at a time, against the views. The host assembles the parts and the groups into one program. The VM runs that program. A sentence with no part, and not marked asked or not-an-instruction, is a silent drop and a kill under line 1. Stopping at four parts and leaving the fifth unmentioned is that kill, not a success.

Part-wise reading keeps each output near a part the model has seen inside 2-to-4-part instructions, instead of asking for a fifth function at the end of a longer program. The input instruction is still longer than any training instruction. I do not know those character lengths, and I do not know if the long instruction exceeds the 512-token window. This proposal does not remove that input-length risk.

Pass line, written now, on a 5-to-8 set that excludes the two eye rows, read once, with no training row above 4 parts: line 4 as written, and, in addition, false passes at most 2% of checked replies, and every sentence a part, asked, or not an instruction.

Kill line, written now: any silent drop; false passes above 5%; or agreement below 90% of the paired in-length rate. Round 1’s long agreement was 0.548 against an in-length 0.508, which is above 90% of a rate that itself missed line 3 (4.1). I would not read that as long-instruction success. The absolute false-pass kill is there so a bad in-length rate cannot carry a bad long rate over line 4.

I do not know the decode time of up to eight short reads against one long one. I would report it, not guess it.

## 5. The measure

The lines in section 2 are the right bar. I would not lower any of them because the confounded dev read is short. Line 3 is not met on any counted read: best dev agreement 0.685, false passes 0.0935 (4.3); round 1 test false passes 0.115 and 0.112, above the kill (4.1). Line 2 is not met on a legal unread read; round 2’s 0.889 is dev (4.3). Line 1 is the right kill, and the eye failures are line 1 failures if the sentence is neither a part nor asked.

Line 4 is the right generalization test and a weak bar on its own. Round 1 would pass it on agreement (0.548 is above 90% of 0.508) while missing line 3 on both splits. Keep the ratio. Report the absolute agreement and the false passes beside it. Do not let the ratio pass a long read whose in-length rate missed line 3.

Line 5 is the right false-pass kill and a dangerous compile clause. “At least half” can push a program out where asked would be right. Keep the line. Do not read a higher compile rate as better if false passes are above 2%.

Report, on every read, dev separate from unread test:

- By direction of a bound: “at least” or “more than” against “at most” or “fewer than,” and the two spellings within each. The gold is 548 against 1,648 on the meanings, and the `not`-after-operator rates are 64 to 68% and 77 to 90% after `>` or `>=` against 9 to 32% after `<` or `<=` (4.6). A pooled agreement can hide a failure on the scarce direction. I do not know the error rates by direction. That is why they must be reported.
- By depth of negation: zero, one, and two or more, and whether the `not` is in the words or only in the gold. The 3.3 row is the second kind. I do not know the current depth mix of errors.
- By number of parts: 2, 3, 4, 5, 6, 7, 8, not only the pooled 2-to-4 and 5-to-8. Round 1 pooled them (4.1).
- By group type: `one`, `either`, `if`, `unless`, with index agreement separate from part agreement. The join is not counted yet (4.5).
- By place: filter, count comparison, size, position value, the 4.5 buckets.
- Parts right, parts fixed by one reversal, parts wrong with no single reversal, parts unmatched, programs right. Round 2 on the 64 was 141/198, 30, 14, 13, and 25/64 (4.3). Part accuracy is not program accuracy.
- Asked, silent drops (kill if not 0), and replies the VM could check. Round 1 long fell to 312/360 = 0.867 checked (4.1). An ask is not a silent drop. Do not let asks disappear from the denominator without a line of their own.
- False passes and false fails, split by replies built to pass and replies built to fail. The lines count false passes. False fails are not the worst outcome; they show a check that is too closed. I do not know the false-fail counts.
- Time, home and rented, median and the long median, grammar separate. About 25 s home and 0.3 s grammar, and 2 s to 9.9 s on the A100 over the 64, are the figures we have (4.7).

One change per read. Section 4.3 cannot be attributed.

Time line I would set. I do not know the frontier model’s time or cost on this task, so I cannot turn “one hundredth of the cost” into a reader time. I would not kill a quality line for time. Report line, not a kill: on the unread in-length test, home median at most the current about 25 s, so the reader does not get slower; report the 5-to-8 median separately. I do not know today’s long-instruction home time, so I will not set that number. If a later home measurement on an unread set exceeds 60 s median for in-length instructions, I would call that a fail of the cost aim for this component. The 60 s is a line I am setting, not a measurement. I would not spend quality to hit it.

## 6. What I would not do, and open questions

I would not train on 5 to 8 parts and then claim line 4. I would not add host code for one kind, or for one benchmark constraint, to lift the compile rate. I would not put a benchmark sentence in a training row. I would not use an external model at serve time, or let a model write a target. Wording drafts only, as the rule says.

I would not drop a part silently, and I would not relax the 5% false-pass kill because 0.0935 looks nearer than 0.1255. I would not turn on a free program for unseen kinds while the reads we have are above that kill. I would not lengthen the program target. It is already 1,546 characters against 316 for the description (4.7).

I would not add another grammar rule designed on the dev rows already read and then score those rows. Rules were added as errors were found (3.5). A rule fit to seen rows is read on rows no one has seen. I would not run another round that changes training, decoder, and grammar together. I would not adopt the parked margin loss in the same round as anything else. I would not balance the 548 against the 1,648 as the first change, and I would not downsample “at most,” until the direction of the reversals is reported. Balancing is already proposed (5); doing it blind can shrink the scarce class and hide the cause.

I would not treat the three join rows or the two long rows as rates. I would not include those two long rows in the official test_long read. I would not fine-tune on the 64. I would not move to a larger model on this evidence. The residual that is counted is a spelling problem the data can fix, and the cost aim is a hundredth. I do not know that a larger model is even available on the home GPU.

I would not read round 2’s test early, and I would not design the next fix from test rows. The next official read is round 2 on the unread splits, one system, lines already written.

Open questions, all unstated:

- What share of the 46/492 false passes are negation reversals, and which way do the 29 go?
- Do filter comparisons and position values have two gold spellings, or only count and size comparisons and any/none (4.6)?
- Does the grammar constrain the split lines, or only the CubeLang?
- What is the token length of a 1,546-character program, and of a 5-to-8-part instruction, against the 512-token window?
- What fraction of the public benchmark’s out-of-domain constraints fit the slots in 3.3?
- Does every unless, if, and either lower as the one example in 3.3?
- Which set were the two 5-check rows from, and were they seen before the official read?
- Why the shorter 64 of the 100, and what do the other 36 show under round 2? I have round 1 on the 100 (4.2) and round 2 only on the 64 (4.3).
- Are the home 25 s and the A100 9.9 s the same rows and the same grammar?
- What share of rows is French, and do the template relation words cover the spans where the 29 reversals sit?
- Of the round 1 drop from 0.960 to 0.867 checked on long rows, how much was a compile failure, how much a missing part, and how much an ask? Section 4.1 does not split them.
- Round 2 on unseen compositions and on 5 to 8 parts is unknown (4.4). I will not fill that in.

## moonshotai/kimi-k3.1-pro

*13117 tokens out, 280.8 s, finish `stop`*

## 1. Diagnosis

### Negation and direction

The remaining part errors are predominantly polarity errors, not general program-writing failure. On the 64 dev rows, round 2 has 30 one-reversal parts, of which 29 are negations or complemented comparisons and only one is a mirror such as `>` for `<`. Fifteen occur inside a filter or position value, ten in a count comparison, and five in a size comparison (4.5). This pattern points first to the target data and then to how the model is trained to choose direction:

- **The gold’s two spellings are a direct cause.** The same meaning appears as a comparison or as `not` around its opposite in near-equal shares, across about 4,150 parts. `none`/`not any` and `any`/`not none` are similarly duplicated. The wording does not determine the spelling: “at least one paragraph … at most 2 sentences” is represented as `not all` in the example (3.3, 4.6). A reader can therefore understand the English and still be penalized for choosing the other target spelling. More importantly, learning `not` as a loosely correlated surface choice interferes with learning the actual comparison.
- **The bound imbalance supplies the observed directional prior.** Among nonzero comparisons, 1,648 parts mean “at most” or “fewer than”, against 548 meaning “at least” or “more than” (4.6). The asymmetric continuation rates after `>`/`>=` versus `<`/`<=` reinforce this prior. This is consistent with complement errors dominating mirror errors, although it does not by itself prove which individual errors it caused.
- **Training matters, but round 2 does not isolate how.** On the same 64 dev rows, parts right as written rise from 59/196 to 141/198, and parts wrong with no single reversal fall from 66 to 14. Nevertheless, 30 reversals remain and false passes are still 46/492 = 0.0935 (4.3). Training, the ninth grammar rule, batching, and the serving environment all changed together, so these gains cannot be assigned to training alone.
- **Program length is secondary for this class.** A comparison’s polarity is a local semantic decision. Writing roughly 1,546 characters gives more places to express it incorrectly, but does not explain why almost every remaining reversal has the same polarity form.
- **Model size and greedy decoding are not isolated.** There is no comparison with another model size or with non-greedy decoding. Greedy decoding can commit early to `not` or an operator without reconsidering the whole clause, but the numbers do not establish that it did so.

The two spellings and bound imbalance are data-fixable. The need to expand a simple comparison into looping, branching, and explicit Boolean boilerplate is structural to target A. It is not demonstrated to be structural to a 576M-weight model: no other size has been tested.

### The join

The evidence is only three dev rows, inspected by eye. In those rows, every part was right in one version, but an or-group meant for checks 2 and 3 joined checks 1 and 2 (4.5). That is a reference/scope error, not evidence that the reader cannot read the individual checks.

The long hand-written `main`, absolute part numbers, and distance between the split and the join are plausible contributors. So is the group distribution: `one` has 6,830 instances, against 838 `either`, 396 `if`, and 383 `unless` (3.3). The present grammar constrains the form of a join but evidently does not guarantee that its operands are the clauses connected by the English. There is not enough evidence to assign the error to model size, greedy decoding, or training. A structured group tree removes the structural reference-writing problem; better clause-to-group alignment is data-fixable. The class must be counted before its prevalence is claimed.

### Long instructions

Round 2’s long result is unknown (4.4). The only round 2 evidence is two inspected rows where the fifth check was again labelled “check 1” and no fifth part was written (4.5). Round 1 actually had slightly higher agreement on test_long than on test, 0.548 versus 0.508, so round 1 showed no general length failure (4.1).

The two round 2 examples point to split accounting and index bookkeeping, with a missing part as a consequence. They do not establish an inherent length limit of this model. Target A makes the risk grow because every additional part adds boilerplate and global references. A per-span description protocol can fix the accounting structurally; better length generalization still has to be demonstrated on unseen rows. Training on longer rows before the preregistered long read would destroy the intended measurement.

## 2. What the reader should write

I would use **C, with B as the normal path and a tightly bounded free-program fallback**. The compiler must be generic over descriptions and groups, not host code for individual instruction kinds. Its equivalence to the generator should be established on VM-checked replies before reader results are interpreted.

| Target | Removed by construction | Left | Generality and unseen kinds | Decode time |
|---|---|---|---|---|
| **A: full program** | No measured 4.5 class. Grammar can remove invalid syntax or declarations, but not wrong meaning. | Negation/direction, choice of group operands, split omissions, and long-row bookkeeping. | Greatest nominal expressiveness within CubeLang and the available views. An unseen kind can be attempted as a new program, but compilability does not imply correctness. | Measured: about 25 s at home, with grammar accounting for 0.3 s; on batched A100, about 2 s for the shortest and 9.9 s average over the 64 dev rows (4.7). |
| **B: descriptions only** | For representable parts, reader-side CubeLang syntax, type, loop, call-order, and hand-written-main errors move into the compiler. The reader no longer manually expands Boolean values and negation into branches. | The semantic choice of `not`, operator, threshold, view, and position. The two gold spellings remain ambiguous unless separately normalized. The reader still chooses group operands, so the observed join error remains. The existing split can still omit or renumber a fifth check. | No loss for checks expressible by the stated slots and groups, including unseen compositions of them. A kind outside that language must become “asked”; it cannot satisfy the compiling-program portion of line 5 through B alone. | Parts and groups average 316 characters versus 1,546 for the program, 4.9 times shorter (4.7). This reduces output substantially, but no B latency is measured; 4.9 times fewer characters is not a latency claim. Split time remains. |
| **C: descriptions plus free fallback** | The same benefits as B for every part expressed as a description. Explicit span accounting can prevent a fallback or omission from being silent. | All A errors remain possible in free-program portions. Negation and group selection remain possible everywhere. Mode selection adds a new error: choosing a description that does not fit, or falling back unnecessarily. | With a genuinely general fallback, nominal expressiveness approaches A. Actual coverage also depends on recognizing when no description fits. An unseen kind can be described, attempted as a free program, or asked; the mode and reason must be explicit. | Unmeasured. It should lie between A and B for similar instructions, but the fallback rate and free-program length determine it. |

For C, I would keep the **split and group tree structured always**. A free CubeLang part function is permitted only when that part’s check has no description. If the join itself cannot be represented, that limitation must be explicit rather than silently replacing the whole output with an unconstrained program. Every uncompiled or unenforced part remains “asked”.

A hypothetical instruction outside B is “The first two sentences must have the same number of words.” That requires a relation between two elements, whereas the stated descriptions compare sizes, counts, or columns with numbers or values. This is an expressiveness example, not a claim that such an instruction occurs in the measured sets.

### A set to measure generality lost

Build a test-only expressiveness set with four reported strata:

1. **In-language held-out compositions:** reverse-generated descriptions and groups, including held-out kinds, English and French, and two through eight parts. The compiled description must agree with the generator’s program on every built reply.
2. **Systematic schema boundaries:** combinations of views, the 17 columns, filters, positions, negation, and group types absent from training but valid under the generator. This measures compositional generalization, not expressiveness beyond the schema.
3. **Beyond-schema candidates:** public-benchmark out-of-domain constraints, kept strictly out of training, plus independently human-authored candidates. A human must classify each as expressible in B, expressible only in free CubeLang with existing views, not VM-checkable with the available views, ambiguous, or not an instruction. Where checkable, a human-supplied gold check is run on replies built to pass and fail.
4. **Fallback stress cases:** instructions mixing representable parts with one beyond-schema part, so C cannot hide a failure by evaluating only its easy portions.

Report, over **all** instructions: enforced coverage, asked coverage, fallback rate, agreement where enforced, false passes, and the fraction both enforced and correct. Conditional agreement among compiling programs is insufficient because B could look accurate by asking on everything difficult. Line 5 remains the minimum bar for unseen kinds.

## 3. Negation and direction: two additional proposals

### 1. Part-only auxiliary reading, with the split supplied

**Expected gain per cost: highest.** Add a separate training round in which the reader receives one already-isolated check clause and writes only its normalized description. Full-instruction rows remain in training. The generator supplies every target and validates every check in the VM. Cover the existing query forms, the 18 comparison meanings, positions with and without the existence wording, English and French, and the existing goal-first and passive templates. This isolates semantic direction from splitting, loop construction, and joining. It is distinct from adding more full-instruction minimal pairs and should be run without another simultaneous change.

- **Sets read:** a fresh isolated-part set, followed by a fresh full-composition set.
- **Pass line:** at least 98% exact isolated parts and at most 1% complement/reversal parts; on full compositions, at least an 80% reduction in reversals relative to the immediately preceding reader on the same fresh rows, while meeting agreement of at least 95% and false passes of at most 2%.
- **Kill line:** stop if full-composition reversals do not fall by at least half, or if false passes exceed 5%.
- **Cost:** additional generated records and training. No additional serve-time decoding. The required record count and training time are unknown.

### 2. Closed-slot scoring instead of greedy polarity selection

After the other slots of a part are proposed, score the legal operator/`not` completions as whole alternatives rather than allowing one greedy token decision to determine polarity. Merge alternatives that compile to the same meaning. If the serving framework can supply a calibrated confidence margin, a part whose alternatives remain unresolved becomes “asked”; whether that capability exists is not stated.

- **Sets read:** a fresh operator-contrast set containing direct and complemented wordings, followed by fresh full compositions.
- **Pass line:** at least an 80% reduction in reversals against greedy decoding of the same reader, agreement of at least 95%, false passes of at most 2%, and compliance with the time line below.
- **Kill line:** stop if reversals do not fall by at least half, false passes exceed 5%, or the added scoring makes the reader miss the time line.
- **Cost:** several scored completions per applicable part, calibration work, and additional serving complexity. It can reduce coverage through “asked”; that coverage must be reported, not treated as free accuracy.

## 4. The join and long instructions

Read the instruction in three explicit stages:

1. Assign a stable identifier to every sentence or clause span and account for it as a part, “asked”, “process”, or “not an instruction”.
2. Read each part independently from its own span. The output is one description or an explicit free-program fallback, never a renumbering of another span.
3. Read a group tree whose leaves are those identifiers. “And” is the default; `either`, `if`, and `unless` must be licensed by the instruction’s words. The host validates arity, references, duplication, and coverage, then compiles the tree. An unresolved group is “asked”, not silently changed to “and”.

This implements the proposed clause-to-group tie and extends it to all groups. It also changes the long-row failure from a silently absent fifth check into an explicit accounting failure that must be asked. It does not guarantee that the fifth check’s meaning is right.

**Join lines, on a fresh join-enriched set:** every group reference valid or explicitly asked; at least 95% exact group trees; reply agreement at least 95%; false passes at most 2%, with a kill above 5%. Report results separately for `either`, `if`, and `unless`.

**Long lines:** retain line 4 exactly—train only through four parts, then read five through eight, with agreement at least 90% of the in-length rate. Silent omissions and duplicate identifiers that suppress a check are both zero, with any occurrence killing the run. Report each length separately. Do not train on longer instructions until that read is complete.

## 5. The measure

The five lines are the right **minimum safety bar**, especially silent dropping at zero and the separate false-pass kill lines. They are not sufficient to diagnose or compare targets. Agreement over VM-checkable replies can hide low coverage, and a single aggregate can hide the exact bound and negation failures now dominating dev.

Add the following, always with denominators and supports:

- **Coverage:** compiling, enforced, asked, fallback, and unmatched-span rates per instruction and per part; instructions both enforced and correct over all instructions.
- **False passes:** the existing reply-level rate, plus the rate among replies built to fail and the fraction of instructions with at least one false pass. Report false passes by part and by group where attributable.
- **Bound direction:** lower bounds (“at least”, “more than”) versus upper bounds (“at most”, “fewer than”); strict versus non-strict; equality and “other than”; zero and nonzero thresholds. The 548/1,648 imbalance makes the first split essential (4.6).
- **Negation:** no negation, one negation, and deeper nesting; separately report surface English/French negation, target `not`, and semantic scope—filter, position value, count comparison, size, whole part, or group. Include `none`/`not any`, `any`/`not none`, and the vacuous position case.
- **Composition:** exact results for two, three, four, five, six, seven, and eight parts; query kind; group kind; group operands adjacent or not; seen versus unseen composition and kind.
- **Wording:** English/French, goal-first/passive, and direct/complemented phrasing. Score minimal families as families as well as individual rows, so getting only the easy wording does not count as success.
- **Length and time:** output characters and tokens, end-to-end latency, split latency, description/program latency, and compile time. Home and batched-A100 results must remain separate.

I would set an initial **time line of at most 10 s mean end-to-end at home**, measured on the same unseen test and test_long rows, with the 95th percentile also reported and no relaxation of accuracy or coverage lines. This is a proposed service target, not a measured B/C result. Batched A100 timing cannot pass the home line.

Round 2’s dev results must not be presented as passing these lines. Its counted test and test_long read has not happened (4.4).

## 6. What I would not do, and open questions

I would not:

- attribute round 2’s improvement specifically to training, the grammar, or the serving change, because they are confounded (4.3);
- claim that model size is the bottleneck without a model-size comparison;
- treat canonicalizing the gold as sufficient: it removes target ambiguity, not necessarily wrong English interpretation;
- select a semantic interpretation by trying candidate checks on replies—the VM can run a check but cannot decide what the user meant;
- add host patches for particular instruction kinds, train on benchmark sentences, or let a model construct training targets;
- allow C’s fallback to become the default merely because it raises conditional agreement;
- tune on the counted test rows, train on five-to-eight-part rows before the long read, or reuse a held-out set after designing a fix from it;
- trade higher enforced coverage for false passes above the written lines.

Open questions:

1. What are round 2’s unseen test, test_long, and public out-of-domain results?
2. How frequent are the join and repeated-identifier errors on counted fresh rows?
3. Which public out-of-domain constraints are VM-checkable, which fit B, and which genuinely require C’s fallback?
4. Can the generic description compiler be shown equivalent to the generator on every built reply, including vacuous positions and all group types?
5. Does the split provide enough clause-boundary information to identify group operands without lexical host rules?
6. Are token or completion scores available from the serving framework for closed-slot scoring, and what latency would they add?
7. What is the latency distribution at home, rather than only the average, and how much of it is the split rather than the program?
8. When natural wording is genuinely ambiguous, which parts should be asked rather than enforced, and how should that ambiguity be represented in evaluation?
