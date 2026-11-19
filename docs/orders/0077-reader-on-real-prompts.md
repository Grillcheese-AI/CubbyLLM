# 0077 - The description reader on real prompts (round 5: checks inside people's own prompts)
status: review (Code A, 2026-11-19: every line read; arm R kept by line 5, line 6 passed at home; approved by Nick 2026-11-17, "i approve 77")
written: 2026-11-17 by Cowork · approved: Nick 2026-11-17 · hypothesis: none (the preprint's first limitation)

## Why
Every row behind 0073 and 0074 is the generator's. The checks are worded from templates, and the other sentences come
from hand-written lists:
- the task ("Explain the migration of birds.");
- tone and structure phrases;
- context lines.

The preprint says so as its first limitation: real instructions are messier than this. People's prompts are long,
run over several lines, carry their own context, and put a rule in the middle of a sentence. The reader has to copy
all of that line by line and still find the checks.

0067's round 6 already screened real, human-written prompts that carry no rule on the reply (`real_rows.py`):
OpenAssistant's OASST1 first turns (Apache-2.0; English and French) and Databricks' Dolly-15k instructions (CC BY-SA
3.0; English). It put a rule inside each one, so the rule was known by construction. This order does the same for the
description reader. One or more checks from the generator go inside a real prompt; the prompt's own text is a task;
the replies and verdicts are built as before. So the paper can report a held-out read on real wording.

- **What it brings CubbyLLM:** the reader is measured, and trained, on the way people actually write, not only on our
  templates.
- **Over a conventional model:** instruction-following tests mostly use written-for-the-test prompts. Here every
  check inside a real prompt is still known exactly, so a wrong reading is counted, not guessed.
- **What it does for you:** when you paste a real, messy request with rules in it, Cubby still finds and checks
  them.

## The arms
- **Arm R (new):** arm L's target and recipe (0073's arm M, folded descriptions, free values), trained on 0074's
  train set plus this order's real-prompt train rows.
- **Arm L (the baseline, no new training):** 0074's adapter, read on the same new rows. Its numbers on real prompts
  are what the preprint reports first.
- **One change:** real prompts in the training mix. The target, grammar, decoder, recipe, host and scoring are
  0074's.

## Build
1. **The prompts** (`validation/builders/real_prompts_0077.py`, off-repo, counts and hashes only):
   - **sources:** OASST1 first turns in English and French, and Dolly-15k instructions in English, with only the
     prompts used and no answer from either set;
   - **screens:** round 6's screens (`exp_if_follow.rules_of`, `alpaca_rows`' constraint words and counts, and the
     French counterpart), so a kept prompt carries no rule on the reply;
   - **round 6's split, kept** (Nick, 2026-11-17): round 6 used every screened prompt of 200 characters or more, and
     every French one, so fresh prompts would only be short English ones. train_real and dev_real come from round 6's
     train and dev prompts, which no description reader has seen. test_real and test_dolly come from round 6's 900
     held-out English prompts: no adapter was trained on them and none was opened by eye; 0067's v6 reader read them
     once, with other rules in them, and only its counts were looked at. No prompt is used twice;
   - **length:** at most 600 characters a prompt, so an answer, which copies it, stays inside 900 ids. Prompts under
     200 characters are thinned as in round 6;
   - **benchmarks:** no train or dev prompt shares 5 words with IFEval or IFBench.
2. **The rows** (`validation/builders/build_rows_0077.py`, on 0074's builder, seed 77):
   - the generator draws 2 to 8 checks, words them, ties some pairs, and runs every reply in the VM, as in 0074;
   - the real prompt goes before the checks, after them, or with the checks after its first sentence;
   - each line of the prompt is one `task` segment in the gold, so the reader writes the prompt's text as it stands,
     one line per prompt line;
   - sentences of the other kinds may still be added from the lists, with the held-out phrases only in held-out rows.
   - **The sets:**
     - **train_real:** 2,000 rows, OASST1 only, English and French. **Dolly text is never trained on:** an adapter
       trained on CC BY-SA text is research only and is not served;
     - **dev_real:** 300 rows, OASST1, for looking;
     - **test_real:** 300 rows, English OASST1 prompts drawn with seed 77 from round 6's 524 held-out ones of at most
       600 characters;
     - **test_dolly:** 300 rows, Dolly-15k, English: round 6's 300 held-out Dolly prompts of at most 600 characters,
       all of them, a source the reader never saw.

   French is in train_real and dev_real only. Round 6's French held-out prompts can't be told apart without
   rebuilding its rows, so there is no French test set and some of them may be trained on. The counts go in the
   report. The held-out hashes are committed before arm R is trained. The test rows are never opened by eye.
3. **The SFT set:** 0074's 8,000 train records and the 2,000 real-prompt records, plus 300 held (dev_real). Every
   answer reads back strictly, and every record fits the trainer's 1,280 tokens. A record over it stops the build.
4. **The notebooks** (from `_build_view_long.py`):
   - training arm R with 0074's recipe unchanged, about 1,560 steps;
   - the read of arm R and arm L on dev_real first, then on test_real, test_dolly, and 0074's test and test_long,
     once;
   - scoring at home with `exp_view_split_score.py`;
   - the comparison with `exp_view_long_compare.py`'s functions, on the named sets.
5. **The screen checked on dev:** dev_real's 300 prompts are read by eye for a rule on the reply that the screens
   missed. The count is reported. A missed rule is a gold the reader is wrong to follow, so the count bounds what the
   real sets can say.

## Acceptance (pre-registered; test_real and test_dolly read once per arm; their SHA-256 committed before arm R is trained)
Counted as in 0074, with its interval (2,000 resamples over the set's rows, here with seed 77).
1. **Accounted:** silent drops 0 on every set (**kill**).
2. **Real prompts, test_real and test_dolly each**, for arm R:
   - gold checks with their line at least 90%;
   - agreement at least 95%;
   - false passes at most 2% (**kill** above 5%);
   - parts right at least 95%;
   - groups right at least 97% (**kill** below 90%);
   - replies checked at least 95%.
3. **The prompt is not read as checks:** the share of the prompt's `task` lines written as `check` is reported, by
   set and arm, with what they did to agreement.
4. **No loss:** on 0074's test and test_long, arm R's false passes over replies built to fail are not above arm L's.
   Not passed if the whole interval of R - L is above 0.
5. **Which reader is kept:** arm R replaces arm L if it hits no kill, line 4 holds, and either of these is true:
   - arm L does not pass line 2 on a real set where arm R does;
   - arm R's false passes over replies built to fail on test_real are lower, with the whole interval of R - L below
     0.

   Otherwise arm L stays, and the real-prompt read is reported as arm L's.
6. **Time at home:** the median for a real-prompt row is at most 12 s, by number of checks; above 25 s it is not
   served.
7. **Reported for both arms,** raw counts with each rate:
   - by source (OASST1, Dolly) and language;
   - by where the prompt sits (before, after, around the checks);
   - by prompt length (in 100-character bands);
   - the breakdowns of 0074's line 10;
   - dev's screen read.
- **Tests:**
  - a small build that reads back with the checker, with every prompt line a `task` segment and the spans tiling the
    instruction;
  - no held-out prompt of round 6 in train_real or dev_real, and no prompt twice;
  - no Dolly text in any train record;
  - the 600-character bound.

## Constraints
- Rules in `CLAUDE.md` hold. No model writes a rule, a description, a program or a training target: the checks and
  verdicts are the generator's, run in the VM.
- **Licences:**
  - OASST1 is Apache-2.0 and may be trained on;
  - Dolly-15k is CC BY-SA 3.0: test only, its text off-repo, and only counts committed;
  - no prompt text in the repository.
- No benchmark prompt (IFEval, IFBench, M-IFEval, IHEval) in any row.
- A fix designed after this order's test read is read on prompts no one has seen.
- Nothing is wired into serving (`CB_RULES` stays off). The Colab runs are Nick's.

## Out of scope
- Real prompts that carry their own rule (they need a person to write the gold): the screen keeps them out, and dev's
  read counts what it missed.
- Answers to the prompts: the replies are the pool's, built to test the checks, as in 0074.
- Russian and Spanish (after English and French are done).

## Report
<!-- Claude Code (or Cowork on a branch) fills this in, then sets status: review -->
### Builds 1 to 5 (Claude Code, Code A, 2026-11-18): the rows, the set and the notebooks; nothing trained or read
- **Build 1, the prompts** (`validation/builders/real_prompts_0077.py`). Round 6's split is read from its own rows
  (`rule_rows_0067_v4_asked_real`, by `source_id`), and each prompt is taken again from `real_rows.pool()` (round 6's
  screens and thinning; 11,083 prompts, all 10,426 of round 6's found). At most 600 characters: candidates train_real
  5,165, dev_real 563, test_real 524, test_dolly 300 (the order's 524 and 300). One train prompt shares 5 words with a
  benchmark and is out. Each set is in a seeded order (test_real on seed 77 alone) and the builder takes from its front.
  **French:** round 6's French held-out prompts are known by their split (`real_*_test`), so all 180 are kept out of
  train_real and dev_real: no French held-out prompt is trained on (the order allowed that some might be).
- **Build 2, the rows** (`validation/builders/build_rows_0077.py`, seed 77; `long_rows_0077`, off-repo; manifest
  `validation/logs/long_rows_0077_2026-11-18_manifest.json`): train_real 2,000 (OASST1 en 1,692, fr 308), dev_real
  300 (en 253, fr 47), test_real 300 (OASST1 en), test_dolly 300 (all of round 6's held-out Dolly prompts of 600
  characters or less). 2 to 8 checks, each length as often (286 or 285 a length in train, 43 or 42 in the others);
  the prompt after / before / around the checks: train 843 / 875 / 282, dev 128 / 133 / 39, test_real 124 / 125 / 51,
  test_dolly 140 / 140 / 20 (around needs a first sentence followed by more). 28,245 VM runs agree with the
  evaluator, 0 disagree; the lower bounds' share 0.500 to 0.502 in every set. Every row's answer reads back strictly
  in both spellings, every prompt line is its own `task` segment, and no prompt is used twice. 0074's builder is used
  through two hooks that are off by default (`view_compose.compose(task=False)`, `build_rows_0073.finish(wrap=...)`):
  0073's and 0074's row tests pass unchanged.
- **The held-out hashes are in that manifest, committed here before arm R is trained** (by set: dev_real `d04d0009...`,
  test_real `2623d211...`, test_dolly `8f91e5a7...`; the source ids hashed too). The read's file
  (`rules/long_rows_0077/read_rows.jsonl`: ids, splits, instructions; `validation/logs/long_rows_0077_read_rows.manifest.json`,
  sha `2188557b...`).
- **Build 3, the SFT set** (`validation/sft/build_view_real_sft.py`; `rules/view_real_sft_r.jsonl`, sha `9010ae3f...`;
  `validation/logs/view_real_sft_r.manifest.json`): 10,300 records, train 10,000 (0074's 8,000, made again and equal
  record for record to 0074's own set, and the 2,000 real ones), held 300 (dev_real); no Dolly record, no test row;
  every answer reads back strictly, also with `own_text`; 0 records share 5 words with IFEval or IFBench; median 251
  tokens, most 1,255 (the trainer's 1,280); the longest 0077 gold answer 665 ids (dev), 384 (test_real), 396
  (test_dolly), none over the read's 900.
- **Build 4, the notebooks** (`notebooks/_build_view_real.py` -> `view_real_r.ipynb`, `view_real_read_dev.ipynb`,
  `view_real_read_test.ipynb`; they clone `master`). Training: 0074's recipe unchanged, 5 epochs of 10,000 records at
  batch 32 (about 1,560 steps); it refuses a set whose rows are not the committed ones. The read: arms R and L on
  dev_real; then once, after dev's reports are committed: both arms on test_real and test_dolly, and arm R on 0074's
  test and test_long (arm L's answers there are 0074's own read, `rules/reads/0074/view_long_l_test.jsonl`, not read
  again). Each cell prints `read: <phase> (all passes read)` after its last pass.
- **Build 5, dev's screen read** (`validation/logs/view_real_dev_screen_2026-11-18.json`, ids only): 17 of 300 dev_real
  prompts carry a rule on the reply that the screens missed: 5 a count or a length (ten things, five of something,
  two examples each, "detailed"), 9 a form (ASCII art, an outline, a chart, emoji, a script twice, a drawing of nodes,
  a 0/1 grid, a list ordered by price), 3 a style or reading level only. In those rows the gold writes the rule as
  task text, so a reader that writes it as a check is counted wrong; the 5 count rules are the kind a drawn check
  measures and could contradict one. On the test sets the same share would be about 6%.
- **Found on the way:** the first build's train_real held one record of 1,411 tokens (a 432-character prompt that
  tokenizes densely), over the trainer's 1,280: the order says that stops the build, and it did. Rebuilt with
  `--tokenizer`, so a train or dev prompt whose record would pass 1,280 tokens is replaced by the next prompt in its
  order (1 replaced). dev_real, test_real and test_dolly came out byte for byte the same as the first build; the first
  build is moved aside off-repo, never read.
- **Tests:** `tests/validation/test_rows_0077.py` (5): round 6's split kept, no prompt twice, no Dolly in train or dev,
  no French held-out prompt anywhere, the 600-character bound, seeded; a prompt put after, before or around the checks
  gives one task segment a line and the spans tile the instruction; a small build in the VM reads back on both arms
  with its sources apart. With 0073's and 0074's row tests, the guards and the path rules: 43 passed.
- **Next (Nick's):** `view_real_r.ipynb` on Colab (A100 or G4), then `view_real_read_dev.ipynb`. Dev is scored at home
  with `exp_view_split_score.py` and its two reports committed before the test notebook runs.
- **Not verified here:** the training and the reads (Colab).

### Arm R trained, and dev_real read (Claude Code, 2026-11-19; for looking, nothing is decided on dev)
- **Training** (Nick, Colab, `rules/cubby450m_view_real_r`): 1,563 steps in 12.2 min, held answer loss 0.0398 on
  dev_real (arm L's 0.0021 was on 0074's dev, another held set: not comparable).
- **The read** (`view_real_read_dev.ipynb`; `rules/reads/0077/view_real_{r,l}_dev.jsonl`, 300 answers each, sha
  `a0c5af1f...` and `8b001268...`), scored at home on the shared VM (`exp_view_split_score.py --splits dev_real`;
  `validation/logs/view_real_dev_r.json`, `view_real_dev_l.json`):

| dev_real (300 rows) | arm R | arm L |
|---|---|---|
| 1. characters in no line and not flagged | 0 | 0 |
| answers ended at the limit | 2 | 5 |
| 2. gold checks with their line | 1,481 / 1,497 = 0.989 | 1,466 / 1,497 = 0.979 |
| segments with span and kind | 2,566 / 2,699 = 0.951 | 2,198 / 2,699 = 0.814 |
| agreement | 4,015 / 4,034 = 0.995 | 3,531 / 3,734 = 0.946 |
| false passes | 5 / 4,034 = 0.0012 | 9 / 3,734 = 0.0024 |
| false passes over replies built to fail | 5 / 2,208 = 0.0023 | 9 / 2,030 = 0.0044 |
| parts right | 1,473 / 1,497 = 0.984 | 1,440 / 1,497 = 0.962 |
| groups (pairs) right | 304 / 309 = 0.984 | 299 / 309 = 0.968 |
| replies checked | 4,034 / 4,134 = 0.976 | 3,734 / 4,134 = 0.903 |
| 3. prompt lines written, in part or whole, as `check` | 10 / 476 | 105 / 476 |
| prompt lines copied whole as one `task` line | 384 / 476 | 13 / 476 |
| seconds an instruction (Colab) | 1.33 | 1.53 |

- **Read:** arm L, which never saw a real prompt, cuts people's lines into pieces and writes a fifth of them in part as
  checks; its span-and-kind falls to 0.814 and a tenth of the replies go unchecked. Arm R copies most prompt lines
  whole and its line-3 counts are 1/4 of L's. Both stay under 0.5% false passes on dev. The line-3 count is computed
  from the answers (a prompt line overlapped by an answer line of kind `check`), not by the scorer; it goes into the
  test read's script the same way.
- **Next:** `view_real_read_test.ipynb` (Nick's), allowed now that both dev reports are committed: test_real and
  test_dolly for both arms, 0074's test and test_long for arm R, once.

### The held-out read, once (Claude Code, 2026-11-19): arm R is kept by line 5
Read on Colab (Nick, `view_real_read_test.ipynb`; `rules/reads/0077/view_real_{r,l}_test.jsonl`, 600 answers each;
`view_real_r_test0074.jsonl`, 600), scored at home on the shared VM (`validation/logs/view_real_test_{r,l}.json`,
`view_real_test0074_r.json`; arm L on 0074's sets is 0074's own read, scored again with the same counts as
`view_long_test_l.json`), set against the lines by `validation/exp_view_real_compare.py`
(`validation/logs/view_real_compare_2026-11-19.json`; interval: 2,000 resamples, seed 77).

| line 2, arm R (arm L beside it) | test_real | test_dolly |
|---|---|---|
| gold checks with their line (>= 0.90) | 1,481 / 1,497 = 0.989 (L 0.977) | 1,466 / 1,497 = 0.979 (L 0.961) |
| agreement (>= 0.95) | 3,984 / 4,039 = 0.986 (L **0.918**) | 3,910 / 3,968 = 0.985 (L **0.923**) |
| false passes (<= 0.02, kill > 0.05) | 7 / 4,039 = 0.0017 (L 0.0017) | 14 / 3,968 = 0.0035 (L 0.0040) |
| false passes over replies built to fail | 7 / 2,211 = 0.0032 (L 6 / 1,977) | 14 / 2,125 = 0.0066 (L 15 / 1,999) |
| parts right (>= 0.95) | 1,477 / 1,497 = 0.987 (L 0.962) | 1,453 / 1,497 = 0.971 (L **0.943**) |
| groups right (>= 0.97, kill < 0.90) | 266 / 271 = 0.982 (L **0.959**) | 288 / 296 = 0.973 (L **0.949**) |
| replies checked (>= 0.95) | 4,039 / 4,089 = 0.988 (L **0.880**) | 3,968 / 4,050 = 0.980 (L **0.916**) |

- **Line 1:** 0 characters dropped silently on every set, both arms.
- **Line 2:** arm R passes every bound on both real sets; arm L misses agreement, groups and replies checked on both,
  and parts right on test_dolly (bold). No kill on either arm.
- **Line 3:** the prompt's lines written, in part or whole, as `check`: arm R 8 of 454 (test_real) and 5 of 393
  (test_dolly); arm L 157 and 115. Copied whole as one task line: R 390 and 333, L 13 and 13. Arm L cuts people's
  lines into pieces and reads a third of them as checks; that is where its unchecked replies come from.
- **Line 4 (no loss on 0074's sets)** holds: false passes over replies built to fail, R against L, test 0 / 1,629
  against 4 / 1,639 (R - L interval -0.0075 to 0), test_long 2 / 2,522 against 3 / 2,522 (-0.0028 to 0.0016).
- **Line 5: arm R replaces arm L.** No kill, line 4 holds, and arm L misses line 2 on both real sets where arm R
  passes it. (The other condition does not hold: on test_real R - L false passes over replies built to fail is
  -0.0042 to 0.0044.)
- **Line 6 (time at home): passes.** `exp_view_split_time.py --split dev_real` (the flag added here; off, dev as
  before), arm R on grilly2, one instruction at a time, held to the machine, 60 dev_real rows (seed 0) after one
  untimed read, in Cowork's slot of 2026-11-19 15:22-15:30 (no other timing or CPU job; the local site server idle):
  median 7.28 s, 95th percentile 10.32 s over all 60; by number of checks, median / 95th: 2: 4.35 / 6.70 (8 rows),
  3: 4.80 / 5.61 (9), 4: 6.09 / 8.28 (10), 5: 9.28 / 14.53 (5), 6: 7.36 / 8.69 (6), 7: 8.86 / 10.66 (9), 8: 8.78 /
  11.93 (13). Every median is under 12 s and every time under 25 s; all 60 `finished` and read back whole; median 190
  ids an answer; the grammar 5.6% of the time (`validation/logs/view_real_time_r.json`). The adapter was copied from
  Drive as it is (it records no base hash; it loads on `base450m_cont`, as 0074's did).
- **Line 7:** by source (OASST1, Dolly; English only in the test sets), by where the prompt sits and by length band,
  both arms, in the comparison log. Arm R by place, false passes over replies built to fail: test_real before
  0 / 984, after 3 / 868, around 4 / 359; test_dolly before 1 / 965, after 12 / 1,026, around 1 / 134.
- **What the dev screen read bounds:** about 6% of the prompts carry a rule the screens missed (dev: 17 of 300);
  the counts above treat those lines as task text, as the gold does.
- Nothing is wired into serving (`CB_RULES` stays off).
