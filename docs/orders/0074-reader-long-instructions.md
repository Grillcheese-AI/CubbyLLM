# 0074 - The description reader learns long instructions (round 4: trained on 2 to 8 parts, read on fresh rows)
status: open (approved by Nick on 2026-11-16: "approve"; builder: Code A, or Cowork on branch `cowork/0074` from `cowork/0073` while Code A is signed out)
written: 2026-11-16 by Cowork, on Nick's "yes" to the next step 0073's line 4 points to · approved: 2026-11-16 by Nick ("approve") · hypothesis: none (0073's line 4)

## Why
0073's reader writes one short description for each check of an instruction, and the host compiles and runs them.
Its kept arm (arm M: folded, free values) passes every line on 2 to 4 parts. On test: agreement 3,252 / 3,279,
false passes 10 / 3,279 = 0.0030, parts right 889 / 903. Round 2's reader, on the same rows, had 410 / 3,168
false passes. It was trained on up to 4 parts, and on 5 to 8 (test_long) line 4 is not passed:
- false passes 96 / 4,470 = 0.0215, over the 0.02 bound;
- parts right at positions 5 to 8: 559 / 754 = 0.7414, against 1,160 / 1,200 = 0.9667 at 1 to 4.

By position the fall comes after the fifth part: 260 / 300 at 5, then 173 / 228, 86 / 150 and 40 / 76 at 6, 7
and 8 (`validation/logs/view_split_test_m.json`). 0073 listed training on 5 or more parts out of scope. This order
is that one change: the same reader, the same target and recipe, trained on rows of 2 to 8 parts, and read on rows
no one has seen, beside 0073's adapter.

- **What it brings CubbyLLM:** a brief with five to eight requirements is checked line by line as reliably as a short
  one, instead of a long instruction being read badly from the sixth check on.
- **Over a conventional model:** every check is still a description the host compiles and runs on the reply, and
  what the reader cannot read is asked, not guessed. The order measures how often that holds on long instructions.
- **What it does for you:** long lists of requirements get checked as well as short ones.

## The arms
- **Arm L (new):** 0073's arm M (folded descriptions, free values: point 7's choice, frozen), trained on this
  order's train set of 2 to 8 parts.
- **Arm K (the baseline, no training):** 0073's kept adapter (`rules/cubby450m_view_split_m`), read on this
  order's rows with the same decoder.
- **One change:** the lengths in the train set. The target, the variant, the grammar, the decoder, the recipe, the
  host and the scoring are 0073's.

## Build
1. **The rows** (`long_rows_0074`, off-repo; manifest, SHA-256 and a read-back check in the repo; the hashes of the
   held-out sets committed before arm L is trained). `validation/builders/build_rows_0074.py` on 0073's builder,
   seed 74, drawn as 0073's: bounds balanced, tied pairs placed evenly, each connective said before and after its
   partner, run in the VM before kept. Sets:
   - **train:** 3,000 rows of 2 to 4 parts and 1,500 minimal pairs (3,000 rows), as in 0073, plus **2,000 rows of
     5 to 8 parts** (500 of each);
   - **dev** 300 (150 of 2 to 4 parts, 150 of 5 to 8), **test** 300 (2 to 4), **test_long** 300 (5 to 8),
     **test_longer** 300 (9 to 12). Each held-out set gets the replies that can tell, as 0073's did;
   - no instruction of 0067's or 0073's rows, and no held-out row with a train row's folded parts and groups. The
     phrases are held out as in 0073. The pool is 80, 160 and 240 replies for 2 to 4, 5 to 8 and 9 to 12 parts;
   - if rows of 9 to 12 parts cannot be drawn, joined or run (the VM, the join, the grammar), the build stops and
     the reason goes under Report. test_longer is then left out, and nothing else changes.
2. **The SFT set** (`validation/sft/build_view_split_sft.py --set` on the new rows): `view_long_sft_l.jsonl` holds
   8,000 train records (the three train parts) and 300 held (dev), in arm M's target. Every answer reads back
   strictly, and every record fits the trainer's 1,280 tokens: a record over it stops the build.
3. **The notebook** `notebooks/view_long_l.ipynb` (from `_build_view_split.py`: the set, its counts, its
   manifest): 0073's recipe unchanged (rank 16, alpha 32, five epochs, lr 2e-4, batch 32, max length 1,280,
   seed 0, key tokens weighted 4), written to `rules/cubby450m_view_long_l`. That is about 1,250 steps
   (8,000 x 5 / 32), against 0073's 938. The run is Nick's, on Colab.
4. **The read** (`view_long_read_dev.ipynb`, `view_long_read_test.ipynb`, from `_build_view_split_read.py`): arm L
   and arm K with `exp_view_split_torch.py`, free, at most 900 ids an answer, as in 0073. Dev comes first, to look
   at. Then test, test_long and test_longer, once. Scored at home on the shared VM with `exp_view_split_score.py`.
   The comparison is a script (`validation/exp_view_long_compare.py`): two arms by name, on the same rows in the
   same order, each line judged as an exact fraction, with the interval of 0073's line 9.
5. **Time at home:** `exp_view_split_time.py` on 60 dev rows (seed 0), now 2 to 8 parts, by number of parts.

## Acceptance (pre-registered; test, test_long and test_longer of `long_rows_0074` read once per arm; their SHA-256 committed before arm L is trained; dev is for looking, held-out rows are never opened by eye)
Counted as in 0073: a row is checked only if it is whole, with nothing asked by the host and nothing flagged; an
`asked` line of the reader leaves its row checked; a rate at its bound passes, a kill is a rate past its kill
bound, and a rate with nothing to count over neither passes nor kills. **The interval:** the rows of the set, the
same rows in the same order for both arms, 2,000 resamples drawn with seed 74, from the 51st to the 1,950th of the
differences in order; a resample with no rate for an arm is left out and counted. Arm L is judged; arm K is read
beside it.
1. **Accounted:** every sentence a line or flagged; silent drops 0 on every set (**kill**).
2. **Short instructions** (test, 2 to 4 parts): gold checks with their line at least 90%; agreement at least 95% of
   the replies checked; false passes at most 2% of them (**kill** above 5%); parts right at least 95%; parts fixed
   by one reversal at most 3% (**kill** above 8%); groups right at least 97% (**kill** below 90%); replies checked
   at least 98%.
3. **No loss on short instructions:** on test, arm L's false passes over replies built to fail are not above arm
   K's: not passed if the whole interval of L - K is above 0.
4. **Long instructions** (test_long, 5 to 8 parts, now trained on): gold checks with their line at least 90%;
   agreement at least 95%; false passes at most 2% (**kill** above 5%); parts right at least 95%; parts right at
   positions 5 to 8 at least 0.90 of the rate at 1 to 4; parts fixed by one reversal at most 3% (**kill** above
   8%); groups right at least 97% (**kill** below 90%); replies checked at least 95%.
5. **Better than the baseline on long instructions** (the order's claim): on test_long, arm L's false passes over
   replies built to fail are below arm K's, with the whole interval of L - K below 0; and arm L's parts right are
   above arm K's, with the whole interval above 0.
6. **Asked:** gold checks asked at most 10%, on test and on test_long.
7. **Time at home:** a median of at most 8 s for 2 to 4 parts and at most 12 s for 5 to 8. Above 25 s the target
   is not served.
8. **Longer than seen** (test_longer, 9 to 12 parts; arm L trained on up to 8): every count of lines 2 and 4 with
   its rate, and parts right by position, for both arms. Reported; it decides nothing in this order.
9. **Which reader is kept:** arm L replaces arm K if it hits no kill and lines 3 and 5 hold. Otherwise arm K
   stays and the order ends as a negative result. Any of lines 2, 4, 6 and 7 not passed is reported with its
   counts and what it would need.
10. **Reported for both arms,** raw counts with each rate: by direction of the bound (with the unweighted mean of
    the four), by negation in the words, by place, by number of parts (2 to 12), by position (1 to 12), by
    connective, by language; and, apart, the false passes on rows where a gold check was written `asked`.
- **Tests:** a small build read back with the checker and a spoiled copy caught (as `test_rows_0073.py`), with rows
  of 9 to 12 parts drawn, joined and run in the VM; the SFT set within the trainer's length; the comparison script
  on two made-up reports (an interval that holds 0, one that does not, a kill, a rate with nothing to count over).

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim; the core opens no connection and
  imports nothing from `validation/`).
- No model writes a rule, a description, a program or a training target: rows are built in reverse and run in the
  VM before kept. No benchmark sentence in a training row (the 5-word test). The words that carry a sign come from
  the templates only.
- One change: the lengths in the train set. Arm K shares the rows, the decoder and the scoring.
- A fix designed after this order's test read is read on rows no one has seen. Rows stay off-repo; ids, counts and
  hashes only.
- Nothing is wired into serving (`CB_RULES` stays off). The Colab runs and any paid compute start on Nick's go.
- The order builds on 0073's code, which is on branch `cowork/0073` until Code A's review merges it. Its work goes
  on master once 0073 is merged, or on a branch from `cowork/0073` before that.

## Out of scope
- **Plain checks written `asked`**, which carry most of 0073's false passes on dev (11 of arm M's 14): their own
  order.
- **The own-text rule for arm M:** point 7 kept free values, and 0073's dev showed once a number its line does not
  have. A second change; its own comparison.
- **Arm W's lost lines on long rows** (0073's test_long: 1,778 / 1,954 gold checks with their line, against arm M's
  1,906).
- Training on 9 or more parts; reading each part in its own pass; a free program for what the descriptions cannot
  say; 0067's lines 5 and 6.

## Report
<!-- Claude Code (or Cowork on a branch) fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
