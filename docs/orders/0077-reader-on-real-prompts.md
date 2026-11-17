# 0077 - The description reader on real prompts (round 5: checks inside people's own prompts)
status: open (approved by Nick 2026-11-17, "i approve 77"; its prompt source revised with him the same day, since round 6 used every screened prompt of 200 characters or more: 0077 reuses round 6's split, Build 1; arm R's rows, SFT set and training wait for 0074's held-out read, and if 0074 keeps arm K this order is revised before they are built)
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
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
