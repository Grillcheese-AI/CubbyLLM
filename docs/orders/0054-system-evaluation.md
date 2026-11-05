# 0054 — The system evaluation: right, wrong and refused on four public benchmarks, against Cubby without the VM and two frontier models
status: open
written: 2026-11-05 by Cowork · approved: 2026-11-05 by Nick ("best would be to run the tests now"; "Run them via API"; "Full sets on Colab now") · hypothesis: the system paper (H-E14, H-E15, H-E16, H-A11, H-A13)

## Why
The system paper (Publication Plan: ICML 2027, go/no-go around January 5; NeSy in March otherwise) needs one
consolidated run. Nick's abstract waits on its numbers.

**What exists is scattered, and most of it has no model writing programs:**
- 0038's 65 of 187 WebQuestions spoken, 0 wrong: the host's everyday table, with no model in the programs seat. The
  450M arm with the table on was never measured.
- 0036's stack: 2 of 20 with the 450M.
- CLUTRR's 827 of 1,146, 0 wrong (H-A13): each story's relations given as facts, rules learned overnight, no language
  model.

So this run reports who proposed each answer, beside the three rates.

**What it brings, in plain words:** one scoreboard that says how often Cubby is right, how often it is wrong, and how
often it says "I don't know", next to what the leading chatbots do on the same questions.
- **Over a conventional model:** a conventional score counts right answers only. Here the wrong ones are counted
  beside them, so a reader sees how many confident mistakes each system makes, and what refusing costs.

## Build
A harness (`validation/system_eval/`, names Claude Code's) and one Colab notebook Nick runs.

1. **The benchmarks:** full test splits. The counts are checked against the files, and any difference is reported.
   - **SimpleQA** (4,326);
   - **Natural Questions, open** (test, 3,610);
   - **WebQuestions** (test, 2,032);
   - **CLUTRR** `gen_train23_test2to10` (test, 1,146), with each story's relations given as facts: H-A13's setting,
     the library learned on train as in `exp_r34`.
   - Licences recorded. Any question seen in development is flagged and reported separately as well as in the total:
     0038's scoreboard, and WebQuestions questions whose WebQSP version 0046 read.
2. **The arms:**
   - **C:** the stack as `cubby_serve` mounts it:
     - every proposer (the host's grammar, the everyday table, the rule library, the 450M programs seat);
     - the Wikidata source live, through the cache;
     - the VM certifying. C speaks only what is certified, and each answer records its proposer.
   - **C−VM:** the same proposals, spoken without certification (the first proposal that yields a value). Read from
     C's own records, with no second pass.
   - **M:** the 450M talk adapter alone, closed book, answering in text, always.
   - **F:** GPT-6 and Claude Opus 5.5 by API (exact model ids recorded), closed book, tools off, default settings
     recorded:
     - every benchmark with an instruction that allows "I don't know" (the comparison the claim is about);
     - SimpleQA also with the question alone, to match their published numbers;
     - CLUTRR with the same relations as sentences (the fair input), and with the original story, reported beside.
3. **The grading,** one rule for every arm:
   - A spoken answer that matches a gold or an alias after normalization is right (`exp_everyday_webq`'s normalizer).
   - An unmatched spoken answer goes to SimpleQA's published grader prompt (correct / incorrect / not attempted). One
     pinned grader model scores every arm, recorded. It only scores; nothing it writes becomes a training record.
   - Refused: Cubby's refusal, and F's "I don't know" or not attempted. M never refuses.
   - WebQuestions' golds are Freebase's of 2013: a mismatch on a value that changes (an office, a team, a currency)
     is listed for the read, not counted wrong by the normalizer.
   - **For the read:** a seeded sample of 100 grader verdicts per arm and benchmark, and every C answer graded wrong
     in full, with its program and its certification trace.
4. **Where it runs:**
   - **One Colab notebook Nick runs** (A100), resumable from Drive per benchmark and arm.
   - The trunk and adapters run in torch there. grilly2 is Vulkan, so a parity read at home: C on 100 SimpleQA
     questions on grilly2, the same answers expected (short chunks).
   - The VM: CubeLang cloned at a pinned commit and built, as 0049's notebook does.
   - The Wikidata source: live through the cache on Drive, at a polite rate, with a generic client name. Its time is
     reported.
   - **The API keys:** from Colab secrets Nick sets himself. Never printed, logged, written or committed.
   - **The preflight:** 50 questions per benchmark through every arm. It prints the time per question, the projected
     session time and the projected API cost per arm, then stops for Nick's go.
   - **Frozen and recorded:** CubbyLLM's commit, every adapter's hash, the VM's commit, the store's snapshot, the model
     ids, the grader, the date.

## Acceptance (pre-registered)
- **Tests:**
  - the grading rule on fixtures: a right alias, a wrong value, a refusal, an "I don't know", a stale WebQuestions
    gold;
  - C−VM read from C's records equals a second pass on 50 questions;
  - no key in any output: a test scans the notebook's outputs and the logs for key patterns.
- **Reported, per benchmark and arm:**
  - right, wrong and refused (counts and %);
  - the wrong rate among spoken answers, with a 95% interval;
  - time per question;
  - for C: by proposer, and the VM's time per check;
  - for F: cost;
  - the grader's agreement with the read, per arm.
- **The lines** (the paper's claim, written before the run):
  - **Line:** C's wrong answers at most 1% of its spoken answers on every benchmark (H-E15's line), and fewer than
    C−VM's.
  - **Kill, the claim as stated:** C above 1% on a benchmark. The abstract's claim then narrows to the benchmarks
    where it holds, with the cause.
  - **Reported, no line:**
    - C against F: the wrong rate at each one's refusal rate;
    - how many of C's spoken answers the 450M proposed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated where code reaches `cubbyllm/`, < 300 lines a file, `__wiring__`, tests pin
  the claim).
- **The stack is frozen for the run, and nothing is tuned on these test sets.** Each number is written once. A rerun
  is for a crash only, and is reported.
- No external model on Cubby's serving path or writing a training record. F and the grader are measured or score,
  nothing more.
- Claude Code does not start a paid run. The notebook is Nick's, and the API spend waits for his go at the preflight.
- No secret in a file, log, argument or commit. No local path in a committed file.
- grilly2 at home in short chunks only.

## Out of scope
- **Cubby reading CLUTRR's stories** (it gets the relations here): a story reader is its own order.
- **Frontier models with web search or tools.**
- **Changing the stack because of these numbers:** findings go under Found on the way, then into their own orders.
  The January freeze reruns this harness.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
