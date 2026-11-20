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
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
