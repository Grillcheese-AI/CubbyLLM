# 0078 - GSM8K, solved and checked (right, wrong and refused on its test set)
status: open (two builders, Nick 2026-11-19: Code A steps 2, 3, the school (S2) and the speak rule, starting now; Code B step 1 and S1, after 0046's R9 accuracy runs; each writes its own section of the Report)
written: 2026-11-19 by Cowork · approved: 2026-11-19 by Nick ("we also need to finish solving gsm8k"; "yes" to writing it and to the TinyGSM download) · hypothesis: H-E19 (arithmetic is a world and a school), H-E18 (program first)

## Why
GSM8K is grade-school word problems: read the story, plan the steps, compute. In SERA the model never computes: it
proposes one operation at a time on numbers the host placed, the VM computes every value, and the host writes it back
(H-E19, `cubbyllm/reasoning/step_loop.py`). Where it stands (H-E19, 2026-10-06/07), on the 236 program-first held-out
problems (synthetic, not GSM8K):
- the 450M with the emitter adapter: 22.9% one step at a time; 27.5% picking each step from the VM's menu; 30.9% with
  a beam ranked by mean log-prob; 49.6% if the right finished path were always picked (the oracle);
- the path scorer was killed on its pre-registered line; the 2.6B stand-in reached 85.2% in one pass.

GSM8K's own test set (1,319 problems) has never been read. And that emitter saw about 150,000 TinyGSM programs (shard 0
of 17, converted 2026-09-30), while TinyGSM's own paper reached 63.1% on GSM8K with a 125M model trained on all 12.3M.
The data is the first lever, and the VM's new forms (cubelang P6-3 to P6-7: expressions as operands, `break`, `//`
and `%`) let the converter keep programs it rejected in September (57,142 `node BinOp` on shard 0 alone).

Nick's `cubelang/examples/gsm8k.cube` is the solver-that-learns this order makes real: `verify`, `learn(expected,
actual)`, `classify_error` and `patterns`. It replays a gold solution's `<<48/2=24>>` steps, so it is a checker and a
teacher, never a test-time solver (it needs the gold steps). Today it does not compile (cubelang `docs/DRIFT.md`).

- **What it brings CubbyLLM:** a real arithmetic score on the standard benchmark, with every number computed by the VM
  and a refusal wherever Cubby is unsure.
- **Over a conventional model:** a model doing arithmetic in its head makes slips that look like answers; here a slip
  in a value cannot happen, and a wrong plan shows up as disagreement between tries, which is refused.
- **What it does for you:** ask Cubby a word problem and the answer it gives was computed, not guessed; when it cannot
  settle the plan, it says so.

## Build
1. **TinyGSM, all of it** (MIT, arXiv 2312.09241; 17 parquet files, 11.8M rows, 3.8 GB): downloaded off-repo to
   `$CUBBY_DATASETS\tinygsm` on Nick's yes (Cowork, 2026-11-19, with a SHA-256 manifest). Converted with
   `validation/sft/import_tinygsm.py`, shard by shard, resumable, CPU only:
   - the converter takes the P6 forms the VM now runs (expressions as operands, `//`, `%`); each new form pinned by a
     test, and shard 0 converted again, reported beside its September counts;
   - every kept program is run in the VM (not a sample) and must give the converter's value;
   - the eval overlap filter (a question sharing a 10-word run with an eval question) covers GSM8K test, GSM8K's 329
     val problems (dev), the 236 program-first problems, and every other benchmark the emitter is read on;
   - the counts by reject reason and by number of steps, per shard, in one manifest under `validation/logs/`.
2. **GSM8K itself** (MIT, `openai/gsm8k`): train 7,473 and test 1,319, off-repo. The 329 val problems the emitter
   holds out of GSM8K train stay dev. Test is read once per arm, at the end.
3. **`gsm8k.cube` compiles and runs** (cubelang, on a branch; the PR is Nick's to merge):
   - strict compile, no silent placeholder; its intended shape kept (parse the gold's annotations, replay each step
     in registers, `verify` within 0.01, `classify_error` naming the kind: stopped early, reversed, unit conversion,
     one number off, as `school.py` names them; `patterns` counting them);
   - run on GSM8K train's 7,473 golds: the share whose replay equals the gold answer, the rest listed by kind
     (annotation slips in the dataset are counted, never fixed);
   - the same (problem, proposed program) pairs judged by `gsm8k.cube` and by `school.py`'s checker agree on every
     pair (both are exact).
4. **The emitter, three arms, one base** (`base450m_cont`, the step format of H-E19; Colab, Nick's runs, notebooks
   built here):
   - **S0:** today's adapter (v12e with TinyGSM shard 0), read as it is;
   - **S1:** trained on the full converted TinyGSM plus GSM8K train's own programs (outside dev), the same recipe and
     budget per row as S0's, more rows;
     **Amendment (Nick, 2026-11-20): S1's first cut is about 1.5M step rows** (10x S0), the same recipe and budget
     per row: every GSM8K train program outside dev (v12e's, plus the other golds that convert), tagged `gsm8k_id`;
     TinyGSM programs sampled to GSM8K train's mix of step counts (its golds' `<<...>>` steps), whole programs only,
     a fixed seed, until the cut is reached; the counts by step count reported beside GSM8K's. **Scale-up, fixed now:**
     if S1's dev right is short of S0's dev right + 10 points, the next cut is 1M programs (~6.6M rows, the long plans
     first). Dev alone decides; the arm read on test is the last cut trained, once; a smaller cut's test is never read.
   - **S2:** S1, then the school (`validation/train_school_torch.py`, built in H-E19, smoke only until now) on GSM8K
     train: levels, tries until the VM says right, dopamine-weighted rows, each wrong try located at its first wrong
     step and named by `gsm8k.cube`'s `classify_error`; a round is kept only if dev does not drop.
5. **How it speaks** (chosen on dev only, then frozen): k tries at temperature, each run step by step in the VM; the
   answer is spoken when at least m of the k land on the same VM value, otherwise refused. k in {8, 16}, m chosen as
   the smallest share that keeps dev's wrong answers at or under 1%. Frozen in the report before test is read.

## Acceptance (pre-registered)
1. **Conversion:** every kept TinyGSM program gives the converter's value in the VM (**kill** on any disagreement);
   counts by reason and steps reported per shard.
2. **No overlap:** no training row shares a 10-word run with a GSM8K test or dev question (**kill** above 0).
3. **`gsm8k.cube`:** compiles strict; the gold replay's share reported, with every miss by kind; it and `school.py`'s
   checker agree on 100% of pairs (**kill** below).
4. **GSM8K test, read once per arm** (right, wrong, refused; Wilson 95% intervals):
   - wrong at most 1% of all 1,319 questions for the arm that is kept (**kill** above 3%);
   - right at least 50% for that arm;
   - S1 above S0 by at least 10 points of right, the whole interval of the difference above 0 (the data lever works);
   - S2 against S1 reported whatever its size (the school's lift).
5. **Reported for every arm:** by number of gold steps; spoken share and precision; the m and k used; the median time
   per question at home on grilly2 (timed in a quiet slot Cowork opens); the 236 program-first problems beside, for
   H-E19's continuity.
6. **Which arm is kept:** the arm with the most right among those within line 4's wrong line. If none is, none is
   served and the report says what the misses are.
- **Tests:** the converter's new forms; the overlap filter on planted overlaps; `gsm8k.cube` on a few golds and on
  each error kind; the speak rule on fixed tries.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim). Tests run with `python3`.
- Code B: CPU only, six threads at most, no GPU. The training runs and the test reads are Colab notebooks Nick runs.
- TinyGSM's programs were written by a model (GPT-3.5): the VM checks that a converted program computes what its code
  computes, not that it answers its question. Only GSM8K's test says whether the plans are right.
- No benchmark text in the repository: GSM8K and TinyGSM stay off-repo; logs hold ids and counts.
- Held-out data is read once. A fix designed after test is read is read on problems no one has seen.
- No external model at any point of an arm's answer; the 2.6B stand-in is not an arm here.
- Nothing is served differently until Nick's word on the kept arm.

## Out of scope
- Growing the base (450M to about 1B): its own order if S2 misses line 4 and Nick says.
- Other maths benchmarks (MATH, SVAMP, ASDiv): after this one.
- Word problems inside conversation (`CB_RULES`, 0067): later.

## Report
<!-- Claude Code fills this in, then sets status: review -->
### Code A: step 2, GSM8K with ids (2026-11-19)
- **GSM8K** (MIT, `openai/gsm8k` "main", its card beside the files) downloaded to `$CUBBY_DATASETS\gsm8k` and written
  with ids by `validation/builders/gsm8k_sets.py`: `train.jsonl` (7,473, `gsm8k-train-<index>`, a `dev` flag),
  `test.jsonl` (1,319, `gsm8k-test-<index>`), `dev_ids.json`, `manifest.json` (SHA-256 of every file); the manifest,
  ids and counts only, is `validation/logs/gsm8k_2026-11-19_manifest.json`.
- **Dev:** the 329 problems the emitter holds out of GSM8K train (its val split, read by `exp_he19_beam.py`), matched
  by question text: all 329 exactly. Checks: dev questions that are test questions 0; the emitter's arithmetic
  training rows whose question is a GSM8K test question 0, a dev question 0.

### Code A: step 3, `gsm8k.cube` compiles and runs (2026-11-19)
- **Where:** cubelang branch `agent-a/gsm8k` (pushed; the PR is Nick's), its own worktree and target dir, on `main`
  with P6-3 to P6-7 (merged on Nick's yes). Three commits: a VM fix, the example, two reading fixes.
- **Its intended shape, in CubeLang, no native module:** the gold's `<<expr=result>>` steps read from the answer's
  text one character at a time (P6-2's string index, P6-3..7's forms); `parse` counts them; `solve` replays each
  step in registers (precedence, parentheses, decimals, `.5`, thousands commas, `//`); `steps_agree` checks each
  step's stated result; `verify` checks a value against `#### n` within 0.01; `classify_error` names a wrong answer
  as `school.py` does (stopped early, reversed, unit conversion, one number off, other); `learn` counts the kind in
  `patterns`. A CLI argument is text, so a list of values is "v1;v2". Compiles strict.
- **Found on the way, a VM bug, fixed on the branch:** a decimal held in a register compared as a whole number
  (`as_i64` on both sides): `0.678 <= 0.5` held, a register of 0.5 was falsy; literal-to-literal comparisons are
  folded at compile time, which hid it. Now f64 when either side is a decimal, i64 exactly otherwise
  (`tests/float_compare.rs`). Every CubeLang program comparing decimals in registers was affected: worth a look in
  CubbyLLM's other generated programs once it is on `main`.
- **The gold replay** (`validation/exp_gsm8k_cube.py`, `validation/logs/gsm8k_cube_2026-11-19.json`): 7,001 of
  7,473 GSM8K train golds replay to their `####` answer (0.937); 376 state the answer in prose after their last
  step; 95 have no annotation; 1 annotation's stated result is not its expression's (the dataset's slip, counted).
- **Against `school.py`'s checker:** 1,000 sampled problems, each with the gold as a program and five mutants
  (stopped early, reversed, x60, off by a stated number, other): **5,867 of 5,867 pairs judged the same** (school's
  codes 0 to 5: 1,012 / 964 / 1,025 / 976 / 912 / 978). The first full read had 3 disagreements (a question's ".5"
  read as 5) and 2 VM stops (`//`): the kill held the line until both were fixed and pinned; the read was run again.
- **Tests:** cubelang `cargo test` 551 passed (the old pin of the broken example replaced by two tests on a made-up
  problem; no GSM8K text in cubelang).

### Code A: step 5, the speak rule (2026-11-19)
- `cubbyllm/reasoning/speak_vote.py` (`7ab8b28`): k tries' VM values; the leading value is said when it has at least
  m votes and no other value ties it; a try that did not run never votes. `choose_m` picks the smallest m keeping
  dev's wrong answers within 1% and reports right / wrong / refused for every m. Tests:
  `tests/reasoning/test_speak_vote.py` (5). m and k are chosen on dev once S1 exists, frozen here before test.

### Code B: step 1, TinyGSM, all of it (2026-11-19)
- **Where:** `validation/sft/import_tinygsm.py` (the converter), `validation/sft/tinygsm_all.py` (17 shards, resumable,
  every kept program in one VM session), `tests/builders/test_tinygsm_0078.py` (7). VM: cubelang `main`'s P6 build
  (`bbc02e4`, in `cubelang-b`'s target dir; the shared VM, `b3b3360`, cannot run `//`). Rows off-repo,
  `$CUBBY_DATA/tinygsm_0078_nop6/`; the log, ids and counts only: `validation/logs/tinygsm_0078_nop6_2026-11-19.json`.
- **The overlap filter** is built once from v12e's val split (614), the program-first file (815 rows: the 236 and the
  rest of that file), GSM8K test (1,319) and GSM8K dev (329, Code A's ids): 29,980 rows dropped over the 17 shards.
- **All 17 shards, September's dialect: 6,350,518 kept of 11,846,109 (0.536); the VM runs every one and agrees on
  6,350,518 of 6,350,518.** By steps: 1: 1,298,594; 2: 1,930,412; 3: 1,493,373; 4: 681,590; 5: 492,286; 6: 228,496;
  7: 131,282; 8: 57,083; 9: 25,814; 10: 11,588. Per shard and by reject reason in the log.
- **Shard 0 beside September:** 340,876 kept (September 341,150: the filter now also drops GSM8K test and dev, 1,360
  against 793); every kept program run in the VM (September: a 2,000 sample).
- **Found on the way:** shards 8-10, 15 and 16 give the answer by `print(f"<<<{x}>>>")`, not `return x`; read as a
  return (that exact form only, pinned), they keep 316k-390k a shard instead of 38k-201k (+1,271,704 rows; the first
  run is kept aside, `tinygsm_0078_nop6_before_marker`).
- **The P6 forms (`//`, `%`), behind `--p6`, not yet in these rows:** on shard 0, 368,523 kept (+27,647), the VM
  agreeing on all, once `//` and `%` are kept on whole numbers only (Python's `5 // 0.1` is 49.0, the VM's 50: 44 rows
  stopped the first run, the kill). A P6 step is `assign sK = A // B;`, which three readers of the step dialect do not
  read yet (`emitter_data.slot_record`, `school.arith_step_values`, the step loop's proposals): asked of Cowork.

### Code B: S1's set and notebook (2026-11-20)
- **The first cut** (Nick's amendment above; `validation/sft/build_s1.py --mode mix`, seed 78;
  `validation/logs/s1_0078_2026-11-20.json`): **1,499,937 step rows**, one file `s1_0078_step.jsonl` (SHA-256
  `dc0dd167...`, on Drive `cubbyllm/emitter/`). **Acceptance 2 checked on the whole file: 0 training rows share a
  10-word run with a GSM8K test or dev question** (the 1,595 val rows that do are dev's own; not trained). A first
  build had 14 v12e programs that did (9 test, 6 dev, one both): dropped whole, listed by id in the log; that file
  (`d75250fb...`) is kept aside on Drive `emitter/_aside/`.
  - S0's base, v12e_w re-cut: 34,814 rows; its GSM8K rows carry `gsm8k_id` (5,805 train; its val split is the 329 dev);
  - GSM8K train's other golds as programs (`validation/sft/gsm8k_programs.py`): 112 programs, 479 rows, tagged
    `gsm8k_id`. Of the 1,325 v12e lacks, 1,213 are not kept: 1,211 listed by id and kind in the log (a number the
    question does not state 677, the answer stated after the last step 271, no computation 144, no annotation 95, other
    24; with the 14 dropped above, 13 more share a run with test or dev and 1 more states no number), and 2 that
    convert but do not re-cut into steps. **GSM8K train outside dev in S1: 5,917 of 7,144.**
  - TinyGSM: 348,006 whole programs, 1,464,644 rows, drawn to GSM8K train's mix of step counts:

    | steps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
    |---|---|---|---|---|---|---|---|---|---|
    | GSM8K train golds (outside dev) | 399 | 2,082 | 2,039 | 1,348 | 747 | 272 | 120 | 39 | 3 |
    | TinyGSM programs drawn | 19,698 | 102,788 | 100,665 | 66,550 | 36,879 | 13,429 | 5,924 | 1,925 | 148 |

- **For S2 (Code A):** every GSM8K row (and each of its step rows) carries `gsm8k_id`; the gold's `<<...>>` text is in
  `$CUBBY_DATASETS/gsm8k/train.jsonl` by that id. `emitter_data._base` and `step_loop.recut` pass it through.
- **The notebook:** `notebooks/emitter_s1.ipynb` (`_build_emitter_s1.py`): S0's recipe (from the Gate A adapter, one
  epoch, batch 16, lr 2e-4, rank 16 / alpha 32, 1,024 tokens, `--no-gen`), ~93,700 steps, about 3.5-5 h on an A100 in
  one session (the trainer does not resume at a step). Nick's to run; then the dev read here.

### Code B: S1 trained (Nick's Colab run, 2026-11-20): the deviation, the rows, and S0's footnote
- **A deviation from "the same recipe as S0's": batch 18, not 16.** `train.log` (Drive `emitter/cubby450m_s1_0078/`) holds
  four starts, each from the Gate A adapter: 16:31:19 UTC (batch 16, 93,849 steps) and 16:52:01 (batch 16) and 17:03:34
  (batch 20, 75,080 steps), each stopped within minutes; the run that finished started **17:18:55 at batch 18, 83,422
  steps** (one epoch of the rows), lr 2e-4, rank 16 / alpha 32, 1,024 tokens, as the notebook. **Reason (Nick):** "batch
  18 was more token per sec", chosen for throughput before the finishing run (first-step rates in the log: batch 16 at
  2,831 and 4,158 tok/s, batch 20 at 5,422, batch 18 at 5,307, settling at 8,813). The rows and the epoch are unchanged;
  S0 was batch 16 at the same lr. Final held program loss 0.0342 at step 83,422; 378.4 min of training.
- **The trainer's 1,501,584 rows against S1's 1,499,937:** the file is 1,498,057 train rows + 1,880 val rows; the trainer
  takes the train split only (`encode_records(..., "train")`) with each row's `repeat` (v12e's kernel 1,674 -> 3,511, chain
  1,150 -> 2,840): 1,498,057 + 3,527 = 1,501,584, per family as the trainer prints it. **No dev row trained:** no train row
  carries a dev id; the 329 dev problems are in the val split only.
- **S0's footnote (the 14 of the first build):** S0 was trained on v12e's 14 programs that share a 10-word run with GSM8K
  test (9) or dev (6) questions, so S0's reads are reported twice: all questions, and without those ids
  (`base_dropped_overlap` in `s1_0078_2026-11-20.json`). **GSM8K numbers already written from adapters trained on v12e**,
  each read on v12e's val arithmetic (= GSM8K dev, 6 of whose questions S0's line shares runs with), get the same
  footnote: H-E19's control arm, 2 of 64 val arithmetic programs (2026-09-30); Gate A's val 14.6% (killed); the step loop's
  and the beam's reads on the val arithmetic (H-E19, `exp_he19_*` logs); and the branching cutoff set on 80 of the val
  problems. No GSM8K test number has been written yet.

### Code A: S2, the school (not built yet)
- `validation/train_school_torch.py` already keeps a round only if its held-out gate does not drop. Wiring its
  error kinds to `gsm8k.cube` needs the pool's rows to carry their gold's `<<...>>` text: that is S1's set (Code B's
  conversion of GSM8K train). Built when S1's format is fixed.
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
