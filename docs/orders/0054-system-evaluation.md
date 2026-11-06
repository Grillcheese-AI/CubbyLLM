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

## Cowork's note (2026-11-06)
Nick made 0055 (the full knowledge base) the top priority: "so we can have the true numbers for the benchmarks".
- **This order's full run waits for 0055.** The harness and the 50-question preflight can go ahead.
- **The preflight on today's base is kept as the "before".** It shows how coverage grows when the store grows,
  without retraining, and the wrong rate beside it.
## Cowork's addendum (2026-11-06): opposite pairs
Nick, 2026-11-06, from his note on opposites: word embeddings place antonyms close together, so a program can pick
the opposite direction, and the VM then certifies the wrong question. That is a certified wrong answer, the case
0 wrong has to catch.
- **The set:** question pairs that differ only by an opposite:
  - oldest / youngest, first / last, earliest / latest, largest / smallest, highest / lowest;
  - before / after, start / end;
  - an inverse relation (what came before or after X: "follows" / "followed by");
  - English and French (le plus âgé / le plus jeune, premier / dernier, avant / après, ...).
- **How it is built:**
  - at least 100 pairs per language, by templates, from facts the stack reaches today, so it reads direction and not
    coverage;
  - each pair's two golds differ;
  - no external model writes them.
- **Read through C, C−VM and M,** by opposite type and language. Per pair:
  - both right;
  - one answered as the other (a polarity error);
  - refused.
- **Line:** 0 polarity errors spoken by C.
  - Every polarity error, C's or C−VM's, is printed with its program.
  - It counts in the paper's error analysis as "interpretation".
- **It runs now, on today's base:** it does not wait for 0055.
## Opposite pairs, read (Claude Code, 2026-11-06): 0 polarity errors spoken by C
Log: `validation/logs/system_eval_opposites_2026-11-06.json` (every pair's verdict per arm; every opposite and
unmatched side listed, C's with its program). Built by `validation/system_eval/opposites.py` (`25ddc7d`).
- **The set:** 125 pairs a language, EN and FR, 25 a type: follows/followed by, replaces/replaced by, start/end time,
  founded/dissolved, highest/lowest point. Each item's names name it alone; each pair's golds differ (dates by year).
  **Not built:** oldest/youngest, first/last, earliest/latest, largest/smallest: Wikidata states no such property pair
  on one item and the stack has no superlative form, so they would read coverage, not direction.
- **How C ran:** the stack online, today's base, the VM certifying; the programs seat is the serving one (the 450M on
  grilly2, the local card), by the main run's split: the stack's prompts recorded (425), the seat's programs written
  under grilly2's environment, then replayed (no prompt missing). M: the talk adapter (talk_v2_2, the hash Colab used)
  on grilly2.
- **The line passes: C speaks 0 polarity errors; C-VM 0 as well (a tie at 0).**
  - C, sides: 94 right, 6 unmatched, 400 refused. The 6 unmatched are not opposites: 5 are right years written zero-padded
    ("0395" for 395, which the grading's matcher misses), and 1 is C-VM's 1966 against 1967 where C refused.
  - C, pairs: start/end EN 20 both right, 5 mixed; highest/lowest EN 13 both right (the 450M's programs), 1 mixed,
    11 refused; founded EN 25 mixed (founded right in all 25, dissolved refused in all 25); follows and replaces
    refused in both languages; **every French pair refused.**
- **M:** 11 sides right, 273 unmatched, 214 refused or declined; 2 marked polarity, both French: "Qu'est-ce qui suit
  211 av. J.-C. ?" answered with 212 av. J.-C. (a real one), and "Qui a remplacé Gouvernement Brazauskas II ?" (M
  repeats the question's name, which contains the other gold "Gouvernement Brazauskas I": the matcher's, not M's).
- **Found on the way:** run from the repo's environment, the grilly2 seat failed on every call and the stack's loop
  recorded 125 refusals. That run is moved aside, and a failure inside the live seat now stops the run
  (`SeatFailure`).

## Cowork's addendum (2026-11-06): a sixth benchmark that tests understanding apart from coverage
The same source (0055's addendum). Every pair's query was run on QLever against a recent Wikidata dump, and kept only
with results (the paper, section 3). Once Cubby's store is built from one dump, each gold answer can be recomputed on
that same dump. A miss is then a misunderstanding, not a missing fact.
- **When:** after 0055's store and cold tier. The gold answers are recomputed through the cold tier.
- **First:**
  - the licence, checked and recorded;
  - 100 test pairs read by hand: how often does the question not match its query? The questions were written by Qwen
    models and judged by another model, with no human check (the paper). Above 5% mismatch, the score is reported
    with that rate as its caveat.
- **The set:** a seeded sample of 1,000 from the test split, as the paper uses. For comparison, its 7B model trained
  on the dataset reaches F1 57.6, and trained on the older benchmarks, 6.8. Model-written questions are evaluation
  data, which Cubby's rules allow.
- **Reported:**
  - right, wrong and refused, as for every benchmark;
  - F1 beside them for the paper's comparison, labelled as a different metric and setting;
  - refusals by reason, with "the query cannot be expressed in CubeLang" counted apart.
- **The opposite pairs:** this set's queries with sorting and a limit are real superlatives (oldest / youngest,
  largest / smallest), used beside the template pairs.