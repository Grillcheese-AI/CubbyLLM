# 0054 — The system evaluation: right, wrong and refused on five public benchmarks, against Cubby without the VM, the 450M alone and frontier models' published numbers
status: in progress
written: 2026-11-05 by Cowork · approved: 2026-11-05 by Nick ("best would be to run the tests now"; "Full sets on Colab now"; then "lets use the reported numbers for now and run the frontier model benchmark later with openrouter, 11k questions might cost a lot") · hypothesis: the system paper (H-E14, H-E15, H-E16, H-A11, H-A13)

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
often it says "I don't know", next to what the leading chatbots report on the same benchmarks.
- **Over a conventional model:** a conventional score counts right answers only. Here the wrong ones are counted
  beside them, so a reader sees how many confident mistakes each system makes, and what refusing costs.

## Build
A harness (`validation/system_eval/`, names Claude Code's) and one Colab notebook Nick runs. **No model API is called
in this run.**

1. **The benchmarks:** full test splits. The counts are checked against the files, and any difference is reported.
   - **SimpleQA** (4,326);
   - **Natural Questions, open** (test, 3,610);
   - **WebQuestions** (test, 2,032);
   - **CLUTRR** `gen_train23_test2to10` (test, 1,146), with each story's relations given as facts: H-A13's setting,
     the library learned on train as in `exp_r34`.
   - **AA-Omniscience, public split** (600, v1; Apache 2.0; `ArtificialAnalysis/AA-Omniscience-Public` on the Hub).
     This is the only benchmark here where a current frontier model publishes right, wrong and declined on the same
     questions (F, below). It is specialist knowledge across six domains, so Cubby is expected to refuse most of it;
     the read is what it says when it does answer.
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
   - **F, published numbers only:** each frontier model's own published results on these benchmarks, collected with
     their sources, every source opened and checked (the citation rule).
     - Kept as published: correct, incorrect and not attempted where given; accuracy alone where that is all there
       is, said so.
     - Where GPT-6 or Claude Opus 5.5 publish nothing on a benchmark, that is said. The newest models that do publish
       are listed instead, by name and date.
     - **Found so far:** Claude Opus 5.5's system card (September 22, 2026, section 6.5.4.1, figures 6.5.4.1.A-B),
       on AA-Omniscience's public split, closed book, no web search:
       - Claude Opus 5.5: correct 0.76, incorrect 0.17, declined 0.07 (net 0.58);
       - Claude Opus 5: 0.71 / 0.22 / 0.06; Claude Sonnet 5: 0.47 / 0.24 / 0.28.
       - Read from the figures; checked against the card before any of it reaches the paper.
     - They are labelled in every table as other models, other graders and other settings. They are context, not a
       matched comparison. On AA-Omniscience the questions are the same, and only the grader and settings differ.
3. **The grading,** with no judge model:
   - A spoken answer that matches a gold or an alias after normalization is right (`exp_everyday_webq`'s normalizer).
   - **C's unmatched spoken answers are all read** (C speaks few), with each one's program and certification trace.
   - **C−VM's and M's unmatched answers:** a seeded sample of 200 per arm and benchmark is read. The rest are reported
     as unmatched, with the right share estimated from the sample (with its interval).
   - Refused: Cubby's refusal. M never refuses.
   - WebQuestions' golds are Freebase's of 2013: a mismatch on a value that changes (an office, a team, a currency)
     is read, not counted wrong by the normalizer.
4. **Where it runs:**
   - **One Colab notebook Nick runs** (A100), resumable from Drive per benchmark and arm.
   - The trunk and adapters run in torch there. grilly2 is Vulkan, so a parity read at home: C on 100 SimpleQA
     questions on grilly2, the same answers expected (short chunks).
   - The VM: CubeLang cloned at a pinned commit and built, as 0049's notebook does.
   - The Wikidata source: live through the cache on Drive, at a polite rate, with a generic client name. Its time is
     reported.
   - **The preflight:** 50 questions per benchmark through every arm. It prints the time per question and the
     projected session time, then stops for Nick's go.
   - **Frozen and recorded:** CubbyLLM's commit, every adapter's hash, the VM's commit, the store's snapshot, the date.

## Acceptance (pre-registered)
- **Tests:**
  - the grading rule on fixtures: a right alias, a wrong value, a refusal, a stale WebQuestions gold;
  - C−VM read from C's records equals a second pass on 50 questions;
  - the harness makes no request to a model API.
- **Reported, per benchmark and arm:**
  - right, wrong and refused (counts and %), with C−VM's and M's estimated from the read where unmatched;
  - the wrong rate among spoken answers, with a 95% interval;
  - time per question;
  - for C: by proposer, and the VM's time per check;
  - **for C, where coverage would come from** (Nick: "cubby's coverage can be improved easily"), read from C's
    records with no second pass:
    - refusals by reason: no plan (the question was not turned into a program), a relation the store lacks, a fact
      missing, the walk failed, not certified;
    - spoken answers by the source of their facts: the offline store, or live Wikidata;
    - on AA-Omniscience, 50 refusals for a missing fact, read by hand: is the fact in Wikidata at all?
- **The lines** (the paper's claim, written before the run):
  - **Line:** C's wrong answers at most 1% of its spoken answers on every benchmark (H-E15's line), and fewer than
    C−VM's.
  - **Kill, the claim as stated:** C above 1% on a benchmark. The abstract's claim then narrows to the benchmarks
    where it holds, with the cause.
  - **Reported, no line:**
    - C beside F's published numbers, labelled as not matched;
    - how many of C's spoken answers the 450M proposed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated where code reaches `cubbyllm/`, < 300 lines a file, `__wiring__`, tests pin
  the claim).
- **The stack is frozen for the run, and nothing is tuned on these test sets.** Each number is written once. A rerun
  is for a crash only, and is reported.
- No external model is called, on the serving path or anywhere in this run.
- Claude Code does not start a paid run. The notebook is Nick's.
- No secret in a file, log, argument or commit. No local path in a committed file.
- grilly2 at home in short chunks only.

## Out of scope
- **The frontier run** (Nick: later, through OpenRouter). Its own step, after this one. Its likely shape, recorded so
  the harness is ready for it:
  - a seeded sample per benchmark, cost printed first;
  - "I don't know" allowed;
  - one pinned grader that then re-grades every arm, C and M included.
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
