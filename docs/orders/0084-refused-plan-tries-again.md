# 0084 - A refused plan gets more tries: the gate picks among the seat's samples, or tells it why
status: open (Code A, after 0082's report and once 0083's `_ask` change is on master)
written: 2026-11-25 by Cowork, after Seed-Prover (arXiv 2507.23726) · approved: 2026-11-25 by Nick ("yes") · hypothesis: H-E15 (the emitter never copies a name or a number)

## Why
Today the programs seat writes one plan per question, greedily, and the gate disposes of it. When the gate refuses,
the host's levers run (lever 4 asks a source for a relation, lever 6 aliases the question's word, the siblings tier
tries a property's other wording, the walk gets one repair), but the seat itself is never asked again. On 0081's read
of the gen-3 held split (seed 7, 600 questions, the served adapter, `exp_0081_r17_s7_2026-11-24.json`, arm A), the
seat's single plan was answered on 198 and refused on 402: 265 by coverage, 91 retrieval exhausted, 39 ambiguous, 6
with no plan and 1 unknown relation, with 0 wrong.

Seed-Prover (ByteDance Seed AI4Math, 2025) refines each Lean proof over several rounds with the compiler's feedback,
and H-E19 already measured why that can pay here: on the 236 word problems the right answer was among the beam's
finished paths for 0.496 of the questions, while picking one by the model's own score reached 0.309. In the knowledge
loop the picker is not a score but the gate itself, which is exact, so more of the seat's own plans could turn
refusals into answers at no training cost. Two ways are tried:
- **resample:** after a refusal, more plans drawn from the seat, each disposed of in turn, the first certified spoken;
- **feedback:** the gate's reason written back into the seat's prompt, then one more plan. The adapter was never
  trained on such a line, so this reads whether it can use one at all.

More tries are also more chances to certify a wrong plan, so the wrong-answer lines are the gate's first clause.

- **What it brings CubbyLLM:** questions the seat's first plan misses get answered from its second or third, still
  checked by the same gate, with no new training.
- **Over a conventional model:** a model's second try is judged by the same model; here every try is disposed of by
  the store and the VM, so the extra tries cannot buy a wrong answer the gate would not have spoken anyway.
- **What it does for you:** fewer "I can't answer that" on questions Cubby's records do hold.

## Build
1. **The flag** `CB_SEAT_RETRY` = `off` (default), `sample` or `feedback`. Off is byte-identical to today (pinned by a
   test).
2. **Where:** in the ask loop, after the seat's first plan is refused by the gate (any refusal reason but an
   ambiguity the asker must settle), on top of 0083's `_ask` change.
   - **`sample`:** `Cubby450mEmitter.emit_samples` draws K = 4 plans in one batched decode, at temperature 0.7 and
     top-k 40 (the step vote's settings), seeded by the question. Each distinct plan, in the order drawn, goes
     through `learn_and_answer` exactly as the first one did; the first certified is spoken, with the try's index in
     the record.
   - **`feedback`:** one host-written line after the question, fixed before the run and recorded in the report:
     the refusal reason and the plan's relations (for example "Refused: the plan does not cover the question
     (relations: birth date)."); then one greedy plan, disposed of the same way.
   - A try that repeats a plan already disposed of is not run again.
3. **The benches,** each arm on the same rows, the card named in every log (home card or Colab, where grilly2 0.1.0
   runs the seat):
   - the gen-3 held split, seed 7 (600), as 0081 read it (arm `off` = 0081's arm A, rerun on the same build);
   - H-E15's live eight, one loop per arm, `--ask --web off`, the everyday table off (H-E15's condition);
   - WebQuestions' test (2,032) with order 0066's four checks on, the set where Cubby says the most and where its
     wrong answers are (0054: 62 spoken, 1 wrong and 2 debatable).

## Acceptance (pre-registered)
- **Tests:** off is byte-identical; a sampled plan goes through the same disposer; a repeated plan is not run twice;
  the feedback line is written as fixed; the try index is in the record.
- **Numbers to report, per arm and set:** spoken, right, wrong (every changed answer read by hand), refused by reason;
  how many answers came from which try; the median and p95 time per question, and per refused question.
- **Gate:**
  - **Wrong answers:** none new on any set. On WebQuestions, every answer that differs from `off` is read; one more
    wrong or debatable answer fails the gate.
  - **The held split:** `sample` or `feedback` right on at least 30 more of the 600 than `off` (+5 points).
  - **The live eight:** the four `off` answers stay answered.
  - **Time:** the median per question at most 1.5x `off`'s.
- **Kill:** under +2 points on the held split for both arms, or any new wrong answer on any set. The seat then stays
  at one plan.
- **If `feedback` gains and `sample` does not:** the next step is training the seat on feedback rows (refused plan,
  reason, the plan that certified), its own order.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- The gate does not change: `plan_verify`, coverage, the levers, the walk and the VM dispose of every try exactly as
  of the first. A try only proposes.
- **Held-out discipline** (the hypothesis file, section 0): K = 4, temperature 0.7, top-k 40 and the feedback line are
  fixed here, before any read; nothing is chosen on the sets read.
- No training, no new adapter: the served programs adapter as it is (or P, if Nick has made 0082's adapter the served
  one by then; name the one used).
- The home card is shared: check the inbox before a long run.
- Logs: `validation/logs/exp_0084_*_<date>.json`, with no local paths in committed files.

## Out of scope
- Training on feedback rows (a follow-up, above).
- 0081's open question (the question's own words in the plan, or coverage reading lever 6 in reverse).
- The arithmetic step loop (0078's vote already samples tries).
- Turning the flag on where a person talks to Cubby: Nick's call, after the report.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
