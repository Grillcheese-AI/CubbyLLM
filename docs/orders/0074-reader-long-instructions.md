# 0074 - The description reader learns long instructions (round 4: trained on 2 to 8 parts, read on fresh rows)
status: done (2026-11-17; approved by Nick on 2026-11-16: "approve"; built by Cowork on branch `cowork/0074`; held-out read 2026-11-17: arm L kept by line 9, with no kill and every line passed; merged to master the same day on Nick's "merge all changes", without Code A's review. Nothing is wired into serving.)
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

### Steps 1 and 2, built: the rows and the SFT set (Cowork, 2026-11-16, branch `cowork/0074`)
Built off-repo with `validation/builders/build_rows_0074.py` (seed 74): `rows.jsonl` (train 5,000, dev 300, test
300, test_long 300, test_longer 300) and `pair_rows.jsonl` (1,500 pairs, 3,000 train rows). Counts and SHA-256 are in
`validation/logs/long_rows_0074_2026-11-16_manifest.json`. The set was read back from its files and checked in
`validation/logs/long_rows_0074_2026-11-16_check.json`. No adapter of this order exists, so the hashes of test,
test_long and test_longer are committed before any training. The held-out rows were not opened by eye: only counts.
- **Run before kept:** 81,575 VM runs, 0 disagreements with the evaluator.
- **Read back** (`validation/exp_rows_0074_check.py`, the evaluator only): 9,200 rows, 38,216 checks, 81,575
  replies, 1,500 pairs. Each of its 19 counts that must be 0 is 0, among them: an instruction of 0067's or 0073's
  rows (13,800 looked up), a held-out row with a train row's folded parts and groups, a train or dev instruction
  sharing 5 words with a benchmark, a row whose number of parts is not its set's.
- **Parts:** train 2,054, 2,031 and 1,915 rows of 2, 3 and 4 parts (with the pairs), and 500 of each of 5 to 8;
  dev 51, 49, 50, 47, 36, 34 and 33 of 2 to 8; test 97, 104 and 99 of 2 to 4; test_long 87, 73, 83 and 57 of 5 to
  8; test_longer 69, 85, 82 and 64 of 9 to 12. **Rows of 9 to 12 parts could be drawn, joined and run:** test_longer
  is built.
- **Bounds** (lower bounds' share of the bounds said with a number other than 0, band 0.45 to 0.55): train with its
  pairs 0.5; dev 0.5019 (0.5013 for its half of 5 to 8); test 0.5016; test_long 0.5003; test_longer 0.4995.
- **Held out:** a held-out row's composition without its values is a train row's for 3 of dev's 51 two-part rows and
  4 of test's 97, and for no row of three parts or more.
- **Replies that can tell** (the 8 as before, then those found; train rows keep 8):
  | set | replies a row | built to fail | numbered parts with N - 1, N and N + 1 | tied pairs with their four combinations | part mutants told by the row's verdict | group mutants told by the row's verdict |
  |---|---|---|---|---|---|---|
  | dev | 13.26 (3,979; 1,583 added) | 2,100 (0.5278) | 636 of 764 (0.8325) | 199 of 276 (0.721) | 2,996 of 3,569 (0.8395) | 1,016 of 1,131 (0.8983) |
  | test | 10.96 (3,287; 895 added) | 1,639 (0.4986) | 371 of 431 (0.8608) | 117 of 156 (0.75) | 1,995 of 2,293 (0.87) | 570 of 601 (0.9484) |
  | test_long | 15.16 (4,548; 2,148 added) | 2,522 (0.5545) | 803 of 994 (0.8078) | 290 of 364 (0.7967) | 4,118 of 4,877 (0.8444) | 1,380 of 1,567 (0.8807) |
  | test_longer | 19.52 (5,855; 3,455 added) | 3,606 (0.6159) | 1,335 of 1,648 (0.8101) | 505 of 680 (0.7426) | 6,123 of 7,761 (0.7889) | 2,377 of 2,952 (0.8052) |

  On test_longer about 21% of the wrong readings of one part pass every reply of their row: a false pass counted
  there is a floor, more so than on the shorter sets.
- **Dropped while building:** too few usable replies on one side 207 (train), 259 (train_long), 280 (pairs, 225 of
  them the twin), 23 (dev), 25, 33 and 91 (test, test_long, test_longer); 5 words shared with a benchmark 35, 45, 13
  and 9 (train, train_long, pairs, dev); a twin with no description or of a kind kept for the test 30.
- **Time:** 2,530 s (train 496, train_long 975, pairs 296, dev 126, test 78, test_long 185, test_longer 374), so 0.49
  s a train row of 5 to 8 parts and 1.25 s a row of 9 to 12.
- **The SFT set** (`validation/sft/build_view_split_sft.py --arms m --name view_long_sft_l.jsonl --order 0074`; on
  Drive under `rules/`; manifest `validation/logs/view_long_sft_l.manifest.json`): 8,300 records, 8,000 train (long_2
  1,011, long_3 1,072, long_4 1,067, long_5 547, long_6 536, long_7 534, long_8 533; pair_2 1,094, pair_3 1,008,
  pair_4 898) and 300 held (dev). Every answer reads back strictly, also with the own-text rule (8,300 of 8,300).
  Tokens a record as the trainer counts them: 236 at the median, 529 at most; none is over 1,280. Key tokens 279,256 of
  1,101,392 answer tokens (0.2535; 0073's arm M: 0.236). No record shares 5 words with IFEval or IFBench.
- **The read's file** (`validation/builders/read_rows_0073.py --splits dev,test,test_long,test_longer --order 0074`):
  `rules/long_rows_0074/read_rows.jsonl`, 1,200 rows, ids, splits and instructions only; its manifest is
  `validation/logs/long_rows_0074_read_rows.manifest.json`.
- **The notebooks** (`notebooks/_build_view_long.py`): `view_long_l.ipynb` refuses to train unless its set is
  order 0074's in arm M's target, 8,000 train and 300 held, built from the rows whose SHA-256 is committed;
  `view_long_read_dev.ipynb` reads arm L and arm K on dev; `view_long_read_test.ipynb` refuses to run until dev's
  two reports are committed. They clone `cowork/0074`.
- **Tests:** `tests/validation/test_rows_0074.py` (4: a small build with rows of 9 to 12 parts, run in the VM, read
  back with the checker, and a spoiled copy caught), `test_view_long_compare.py` (4). 0073's tests pass unchanged.
- **Left:** the push of `cowork/0074` (Nick's yes), arm L's training (Nick, Colab), then the dev read, the timing at
  home, the counted read and line 9.

### Arm L trained, and the dev read (Cowork, 2026-11-16, branch `cowork/0074`)
Nick trained arm L on Colab (`notebooks/view_long_l.ipynb`): 1,250 steps at 0.63 s a step, 0073's recipe unchanged;
held answer loss 2.5692 at step 0 and 0.0021 at the end. Then he read dev on Colab (`view_long_read_dev.ipynb`):
arm L and arm K, free values, 300 rows each, every answer `finished`. Scored at home on the shared VM
(`cubelang-cowork` at b3b3360) with `exp_view_split_score.py --splits dev`. Dev is for looking: nothing is decided on
it, and no held-out set has been read. Logs: `validation/logs/view_long_dev_{l,k}.json`,
`view_long_second_chance_dev_{l,k}.json`. The answers stay off-repo.

| dev (150 rows of 2 to 4 parts, 150 of 5 to 8) | Arm L | Arm K (0073's adapter) |
|---|---|---|
| Silent drops | 0 | 0 |
| Gold checks with their line | 1,400 / 1,402 = 0.9986 | 1,368 / 1,402 = 0.9757 |
| Agreement | 3,976 / 3,979 = 0.9992 | 3,659 / 3,836 = 0.9539 |
| False passes | 3 / 3,979 = 0.0008 | 47 / 3,836 = 0.0123 |
| False passes over replies built to fail | 3 / 2,100 = 0.0014 | 47 / 2,022 = 0.0232 |
| Parts right | 1,397 / 1,402 = 0.9964 | 1,286 / 1,402 = 0.9173 |
| Parts right at positions 1 to 4 / 5 to 8 | 1,047 / 1,049 = 0.9981; 350 / 353 = 0.9915 | 1,019 / 1,049 = 0.9714; 267 / 353 = 0.7564 |
| Parts fixed by one reversal | 1 / 1,402 = 0.0007 | 11 / 1,402 = 0.0078 |
| Groups right | 276 / 276 = 1.0 | 254 / 276 = 0.9203 |
| Replies checked | 3,979 / 3,979 = 1.0 | 3,836 / 3,979 = 0.9641 |
| Gold checks asked (by the reader + by the host) | 2 + 0 of 1,402 = 0.0014 | 10 + 15 of 1,402 = 0.0178 |
| Seconds per instruction, decoded on Colab | 0.9309 | 0.9819 |

- **By number of parts, parts right**, L / K: 101 / 101 of 102 (2), 147 / 143 of 147 (3), 200 / 196 of 200 (4),
  235 / 231 of 235 (5), 215 / 203 of 216 (6), 236 / 203 of 238 (7), 263 / 209 of 264 (8). False passes by number of
  parts: arm L 2 (2 parts) and 1 (7 parts); arm K 2, 4, 1, 4, 5, 10 and 21 (2 to 8).
- **By position**, L / K: 299 / 294 and 299 / 294 of 300 (1, 2), 249 / 243 of 249 (3), 200 / 188 of 200 (4),
  148 / 130 of 150 (5), 103 / 79 of 103 (6), 67 / 42 of 67 (7), 32 / 16 of 33 (8). Arm K on these new rows falls as it
  did on 0073's test_long (0.7564 at 5 to 8 here, 0.7414 there).
- **By language**, L / K: English 812 / 759 of 815, French 585 / 527 of 587.
- **Arm L's five parts not right** (looked at by eye, dev only): two plain checks written `asked` (d400015, a French
  "every sentence ... and does not end with" of two conditions; d400034, "write at most one line"); one negation in a
  French filter under "sauf si" written `==` for `!=`, the only part fixed by one reversal (d400094); a condition's
  column (sentences counted as words, d400149); and a condition's value taken from the next sentence's quote
  (d300040, "lines that are part of a list" written `item == "!"`): the free-value slip 0073's dev showed, and the
  source of 2 of arm L's 3 false passes. The third is on d400034, the row with the `asked` check.
- **The second chance** (arm L): 1 part fixed by one reversal, 2 wrong that no one reversal fixes, 5 right in another
  form than the gold's. Arm K: 11, 71 and 16.
- Nothing here changes the read: the decoder, the grammar and the scoring stay as they are for the held-out sets.
  `view_long_read_test.ipynb` can run: it needs these two reports in the branch it clones.

### Held-out read, the lines judged (Cowork, 2026-11-17, branch `cowork/0074`)
Nick read test, test_long and test_longer on Colab (`view_long_read_test.ipynb`), arm L and arm K, free values, 900
rows each, once. Arm L's 900 answers all ended `finished`; arm K's ended `finished` 862 times and at the 900-id limit
38 times, all on test_longer. Scored at home on the shared VM with `exp_view_split_score.py`, compared with
`exp_view_long_compare.py` (seed 74). Logs: `validation/logs/view_long_test_{l,k}.json`,
`view_long_compare_0074.json`, `view_long_time_l.json`. The answers stay off-repo and no held-out row was opened.

| | test (2 to 4 parts), L | test, K | test_long (5 to 8), L | test_long, K |
|---|---|---|---|---|
| Silent drops | 0 | 0 | 0 | 0 |
| Gold checks with their line | 901 / 902 = 0.9989 | 900 / 902 = 0.9978 | 1,908 / 1,910 = 0.999 | 1,860 / 1,910 = 0.9738 |
| Agreement | 3,269 / 3,287 = 0.9945 | 3,252 / 3,271 = 0.9942 | 4,514 / 4,548 = 0.9925 | 3,975 / 4,349 = 0.914 |
| False passes | 4 / 3,287 = 0.0012 | 4 / 3,271 = 0.0012 | 3 / 4,548 = 0.0007 | 79 / 4,349 = 0.0182 |
| False passes over replies built to fail | 4 / 1,639 = 0.0024 | 4 / 1,631 = 0.0025 | 3 / 2,522 = 0.0012 | 79 / 2,410 = 0.0328 |
| Parts right | 900 / 902 = 0.9978 | 897 / 902 = 0.9945 | 1,901 / 1,910 = 0.9953 | 1,711 / 1,910 = 0.8958 |
| Parts right at positions 1 to 4 / 5 to 8 | | | 1,197 / 1,200 = 0.9975; 704 / 710 = 0.9915 | 1,169 / 1,200 = 0.9742; 542 / 710 = 0.7634 |
| Parts fixed by one reversal | 0 / 902 | 0 / 902 | 2 / 1,910 = 0.001 | 22 / 1,910 = 0.0115 |
| Groups right | 156 / 156 = 1.0 | 154 / 156 = 0.9872 | 363 / 364 = 0.9973 | 341 / 364 = 0.9368 |
| Replies checked | 3,287 / 3,287 = 1.0 | 3,271 / 3,287 = 0.9951 | 4,548 / 4,548 = 1.0 | 4,349 / 4,548 = 0.9562 |
| Gold checks asked (reader + host) | 1 + 0 = 0.0011 | 1 + 2 = 0.0033 | 2 + 0 = 0.001 | 18 + 19 = 0.0194 |
| False passes on rows where a gold check was written `asked` | 4 | 4 | 2 | 14 |

- **Line 1** (silent drops 0): passed on every set, both arms.
- **Line 2** (test): arm L passes every bound, and so does arm K.
- **Line 3** (no loss on test): false passes over replies built to fail 4 / 1,639 (L) and 4 / 1,631 (K); interval of
  L - K -0.0001 to 0.0, not above 0: **passed**. All 4 of each arm's false passes are on one row of 2 parts where the
  reader wrote a plain check `asked`.
- **Line 4** (test_long, now trained on): arm L passes every bound. Agreement on test_long is 0.998 of test, and parts
  right at 5 to 8 are 0.994 of the rate at 1 to 4. Arm K does not pass agreement (0.914), parts right (0.8958),
  groups right (0.9368) or parts right at 5 to 8 over 1 to 4 (0.7836), and hits no kill.
- **Line 5** (the order's claim, test_long): false passes over replies built to fail 3 / 2,522 (L) against
  79 / 2,410 (K), interval of L - K -0.0419 to -0.0219, whole below 0; parts right 1,901 / 1,910 against
  1,711 / 1,910, interval 0.0829 to 0.1157, whole above 0: **passed**.
- **Line 6** (asked at most 10%): arm L 0.0011 on test and 0.001 on test_long: passed.
- **Line 7** (time at home, one at a time on grilly2, 60 dev rows): medians 3.9 s (2 to 4 parts) and 6.53 s (5 to 8),
  95th percentiles 5.06 s and 9.07 s, 5.79% of the time in the grammar: passed, served.
- **Line 8** (test_longer, 9 to 12 parts, decides nothing):

  | test_longer | Arm L | Arm K |
  |---|---|---|
  | Gold checks with their line | 3,093 / 3,141 = 0.9847 | 2,699 / 3,141 = 0.8593 |
  | Agreement | 5,280 / 5,717 = 0.9236 | 2,910 / 4,106 = 0.7087 |
  | False passes | 110 / 5,717 = 0.0192 | 176 / 4,106 = 0.0429 |
  | False passes over replies built to fail | 110 / 3,526 = 0.0312 | 176 / 2,530 = 0.0696 |
  | Parts right | 2,895 / 3,141 = 0.9217 | 1,717 / 3,141 = 0.5466 |
  | Parts right at positions 1 to 4 / 9 to 12 | 1,185 / 1,200 = 0.9875; 599 / 741 = 0.8084 | 1,040 / 1,200 = 0.8667; 156 / 741 = 0.2105 |
  | Parts fixed by one reversal | 21 / 3,141 = 0.0067 | 99 / 3,141 = 0.0315 |
  | Groups right | 659 / 680 = 0.9691 | 472 / 680 = 0.6941 |
  | Replies checked | 5,717 / 5,855 = 0.9764 | 4,106 / 5,855 = 0.7013 |
  | Gold checks asked (reader + host) | 15 + 7 = 0.007 | 49 + 75 = 0.0395 |

  By position, arm L: 299, 298, 297, 291, 291, 281, 272, 267 and 260 of 300 at 1 to 9, then 189 / 231, 106 / 146 and
  44 / 64 at 10, 11 and 12. Arm K: 295, 289, 255, 201, 167, 140, 118, 96 and 78 of 300, then 51 / 231, 20 / 146 and
  7 / 64. Arm L, trained on up to 8, holds well past 8 and slips from the tenth check on; it does not fall the way arm
  K fell after the fifth.
- **Line 9:** arm L hits no kill and lines 3 and 5 hold, so **arm L replaces arm K**. No line of 2, 4, 6 or 7 is
  missed by arm L.
- **Line 10,** parts right, L / K:
  - by direction of the bound, test_long: lower 394 / 356 of 397, upper 394 / 345 of 395, exactly 29 / 23 of 29,
    other 170 / 138 of 173, no number 914 / 849 of 916; unweighted mean of the four 0.9931 / 0.8402 (test
    0.9886 / 0.9871, test_longer 0.9032 / 0.4565);
  - by negation in the words, test_long: none 1,475 / 1,326 of 1,480, one 402 / 362 of 406, two 24 / 23 of 24;
  - by place, test_long: size 481 / 406 of 484, count 364 / 324 of 368, position 142 / 132 of 142;
  - by number of parts: 193 / 193 of 194 (2), 312 / 312 (3), 395 / 392 of 396 (4), 435 / 429 of 435 (5),
    437 / 415 of 438 (6), 580 / 509 of 581 (7), 449 / 358 of 456 (8), 607 / 434 of 621 (9), 796 / 499 of 850 (10),
    826 / 457 of 902 (11), 666 / 327 of 768 (12);
  - by connective, test_long: if 93 / 85 of 93, either 193 / 185 of 193, unless 77 / 71 of 78;
  - by language, test_long: English 1,188 / 1,088 of 1,192, French 713 / 623 of 718;
  - false passes on rows where a gold check was written `asked`: test 4 / 4, test_long 2 / 14, test_longer 25 / 34.
- **Not verified here:** time on a served host (CB_RULES stays off); arm L on real prompts (order 0077, now built on
  arm L as it assumed).
