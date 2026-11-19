# 0075 - The graders checked before their rates are trusted (0054's right, wrong and refused)
status: done (Code B, on `agent-b`; reviewed row by row by Cowork and merged to master on Nick's "merge / check 0075", 2026-11-19)
written: 2026-11-17 by Cowork, on Nick's "yes" to writing it after the field watch's find · approved: 2026-11-17 by Nick ("I approve both") · hypothesis: none (0054's system evaluation)

## Why
On 2026-11-01 an issue was opened on NVIDIA's NeMo Gym, [#5430](https://github.com/NVIDIA-NeMo/Gym/issues/5430)
(open, with a fix proposed in PR #5431). The SimpleQA and AA-Omniscience verifiers there read the judge's grade as
the first of A, B, C, D found anywhere in its text. So `NOT_ATTEMPTED`, `C: PARTIAL_ANSWER` and `Answer: B` all
score 1.0, as right. Two of the benchmarks 0054 reports are those two, and a refusal rate is only as good as the
grader that tells a refusal from an answer.

Cubby's own grader (`validation/system_eval/grade.py`) uses no judge model, but its matcher is lenient in the same
direction. `exp_everyday_webq.matches` counts an answer right when the two are equal after normalization, when
either one holds the other and is longer than 3 characters, or when their first years are the same:
- gold "Paris, Texas", answer "Paris" (France): right;
- an answer that names several candidates, one of them the gold: right;
- "between 1990 and 2000" against "1990": right.

A wrong answer counted right is a hole in the claim that matters most, "0 wrong". It is not a refusal: it is never
read, because only unmatched answers go to the read.

The frontier numbers set beside Cubby (arm F, `validation/logs/system_eval_published_2026-11-05.json`) come from
graders that are not always named: the Claude Opus 5.5 card's AA-Omniscience figure says "grader not named". The
planned frontier run through OpenRouter will need a judge, and so a parser like the one above.

- **What it brings CubbyLLM:** the "0 wrong" and refusal numbers survive a hostile reader, because every "right" is
  traced to the rule that gave it, and a non-answer can never be graded right.
- **Over a conventional model's report:** most benchmark reports give accuracy from a judge model nobody audits.
  Here each verdict names its rule, and the leniency is measured.
- **What it does for you:** when Cubby says "never wrong", the number has been checked against the ways graders
  usually fool themselves.

## Build
1. **Every "right" names its rule** (`grade.py`; no verdict changes in this step). `matches` returns the rule that
   matched, besides the verdict: `equal`, `answer in gold`, `gold in answer`, `year`, or for CLUTRR `relation` and
   `relation word`. A script counts the rules over 0054's stored run records: arms C, C-VM and M, the five
   benchmarks, by arm and benchmark (`validation/exp_grade_rules.py`, log in `validation/logs/`).
2. **The controls** (a fixed set, `tests/validation/test_grade_controls.py`), each with the class it must get:
   - non-answers: an empty answer, "I don't know", "The facts don't say.", `NOT_ATTEMPTED`, "N/A": refused or
     declined;
   - partials and lists against a gold "Paris": "Paris, Texas", "Paris or Lyon", "Paris, Lyon, Nice", "not Paris",
     "Paris?";
   - a year against a gold "1990": "1990s", "between 1990 and 2000", "born 1950, died 1990".

   What each partial must get is written in the test before the matcher is changed. A non-answer graded right is
   the kill.
3. **The read of the lenient matches.** For arm C, every "right" whose rule is not `equal` and not CLUTRR's
   `relation`. For C-VM and M, a seeded sample of 200 per benchmark (seed 75). Each is read as 0054's reads were
   (`reads.jsonl`: right or wrong, and why). The read's wrong answers are reported, and they enter the rates as
   wrong.
4. **A strict matcher behind a flag** (`--strict`):
   - containment only on word boundaries, and only when the answer names one candidate (no "or", no list of
     values);
   - the year rule only when the gold is a date and the answer holds exactly one year;
   - what the strict matcher does not take goes to the read, as unmatched answers do today.

   The rates are recomputed with it and set beside the current ones. Which one the system paper uses is written
   here: the strict one, with its read.
5. **The published numbers' graders.** Each source in `system_eval_published_2026-11-05.json` gets three fields:
   - `grader`: as the source names it, or "not named";
   - `not_attempted`: whether "not attempted" is its own class;
   - `matched`: whether the setting matches 0054's.

   The comparison table says "grader not named" where it is so. No published number is changed.
6. **The judge parser for the later frontier run** (`validation/system_eval/judge_parse.py`, used by nothing yet):
   - only a standalone grade letter counts;
   - several distinct letters, or none, mean not attempted;
   - it is pinned on the issue's own controls (`NOT ATTEMPTED`, `C: PARTIAL_ANSWER`, `Answer: B`,
     `B: INCORRECT`).

   No run in this order: the frontier run is its own decision, with its cost stated first.
7. **Where each gold comes from:**
   - a short table per benchmark: the gold's source, its date, and how it was checked (SimpleQA, NQ open,
     WebQuestions with Freebase golds of 2013, CLUTRR generated, AA-Omniscience);
   - every "wrong" a read gives also gets a reason: `wrong`, `gold stale`, `gold wrong` or `question ambiguous`;
   - the reasons are reported apart, and no gold is edited.

   The idea is from [retrieval-gold-provenance](https://github.com/Vitamin-Village/retrieval-gold-provenance)
   (EACL 2027 Findings), whose benchmarks are not ours.

## Acceptance (pre-registered)
1. **Rules:** every "right" of arms C, C-VM and M on the five benchmarks names one rule. The counts by rule, arm and
   benchmark are reported, with no "right" left without a rule (**kill** above 0).
2. **Controls:** every control gets the class written for it, and a non-answer is graded right 0 times (**kill**).
3. **The lenient matches read:**
   - arm C: all of them, and the wrong ones counted;
   - arms C-VM and M: the sampled wrong share, with its Wilson 95% interval.
   Every wrong answer the read finds in arm C is reported as wrong in 0054's tables, and "0 wrong" is stated again as
   measured.
4. **Strict against current:** right, wrong and refused, by arm and benchmark, under both matchers, raw counts with
   each rate. The difference is reported whatever its size.
5. **Published numbers:** each source has its three fields filled.
6. **Judge parser:** the issue's four controls give not attempted, not attempted, B and B.
7. **Gold reasons:** every read "wrong" carries one reason, and the counts by reason are reported.
- **Tests:** the rule each match names; the controls; the strict matcher on the controls; the judge parser.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim).
- No judge model grades Cubby's answers. A person's read decides an unmatched or lenient match, as in 0054.
- No number rounded or invented. No gold edited. No benchmark question goes into a training row.
- Nothing paid starts here. The frontier run is its own order, with its price stated first.

## Out of scope
- Running frontier models (OpenRouter): the parser is built, but not used.
- New benchmarks, or a change of what the system answers.
- The field watch's other 0054 finds (signed certificates go to 0076).

## Report
<!-- Claude Code fills this in, then sets status: review -->
**Verdict (Code B, 2026-11-19):** the leniency is real, and it was hiding one wrong answer of C's. Every "right" names
its rule; no non-answer is graded right; the read of every lenient match finds **1 wrong in C** (`webq:445`: the
answer restates the question's own country, matched inside a longer gold) and **79 wrong of M's 117** lenient
"rights" (+9 debatable). "0 wrong" stands, as measured, where it stood: SimpleQA (0 of 6), CLUTRR (0 of 827),
AA-Omniscience (nothing spoken). C on WebQuestions is now **5 wrong of 65 (+3 debatable)**; NQ 4 of 15 (+1 debatable).
Wherever a number is quoted, its debatable count goes beside it.

- **Commits** (on `agent-b`): `a050b8b` (in progress), `8838fb0` (steps 1-7, this report).
- **Files:** `validation/system_eval/match_rules.py` (the rule of each match; the strict matcher; `non_answer`),
  `grade.py` (`strict=` on `matches` and `verdict`; off: unchanged), `validation/exp_grade_rules.py` (steps 1, 3, 4:
  the rules, the read list, the strict counts), `validation/exp_grade_rates.py` (the rates under both matchers with
  every read applied), `validation/system_eval/judge_parse.py` (step 6, used by nothing);
  `tests/validation/test_grade_controls.py`, `tests/validation/test_judge_parse.py`. Logs:
  `grade_rules_2026-11-19.json`, `grade_reads_0075_2026-11-19.json` (the read: ids, verdicts, reasons; the reader's
  glosses are off-repo beside `to_read_0075.jsonl`, as benchmark text stays off the repo), `grade_rates_2026-11-19.json`; `system_eval_published_2026-11-05.json` (step 5's fields added).
- **Tests:** 47 pass in the two new files (the rule each match names; the controls under both matchers; strict never
  takes more than lenient; the judge parser on the issue's controls). `tests/validation` and the guards: 420 passed,
  2 skipped.

**The records** (0054's current arms): C is the "after" (`run_0055_after`; NQ its gather-seat rerun `nq_final2`;
CLUTRR the "before"'s), C-VM from C's records, M `run_0054/m`. **Found first:** a record keeps only the first 8 golds
of a question (585 rows have more), and the run graded against all of them, so the rules are counted against
`questions.jsonl`'s full lists. With the truncated lists, 3 stored "rights" had no rule (the kill line, met by the
records, not by the grader): each answer is a 9th or 10th gold.

**1. Rules** (`grade_rules_2026-11-19.json`): every stored verdict recomputes the same; **no "right" without a rule**.

| | equal | answer in gold | gold in answer | year | relation |
|---|---|---|---|---|---|
| C: SimpleQA / NQ / WebQ / CLUTRR | 5 / 6 / 40 / - | 1 / - / 6 / - | - / - / 3 / - | - / 3 / - / - | - / - / - / 827 |
| C-VM: SimpleQA / NQ / WebQ / CLUTRR | 6 / 6 / 64 / - | 2 / - / 8 / - | - / - / 4 / - | - / 3 / - / - | - / - / - / 1,035 |
| M: SimpleQA / NQ / WebQ / AA | - | - | 3 / 45 / 50 / 6 | 8 / 3 / 1 / 1 | - |

None of M's "rights" is `equal`: M writes sentences, and a sentence that restates the question holds the gold
whenever the question does.

**2. Controls:** each gets its written class under both matchers; **a non-answer is graded right 0 times**. The
lenient matcher takes "Paris, Texas", "Paris or Lyon", "Paris, Lyon, Nice", "not Paris", "1990s", "between 1990 and
2000" and "born 1950, died 1990" as right (recorded as measured); the strict one takes none of them.

**3. The lenient matches read** (all of them in every arm: C-VM's and M's are under 200 a benchmark, so no sample):

| arm | read | right | wrong | debatable | wrong share (95%) |
|---|---|---|---|---|---|
| C | 13 | 10 | 1 | 2 | 0.077 (0.014-0.333) |
| C-VM | 17 | 14 | 1 | 2 | 0.059 (0.011-0.270) |
| M | 117 | 29 | 79 | 9 | 0.675 (0.586-0.753) |

Cowork checked the read row by row (2026-11-19): agreed, but for `webq:1023` (the gold answers nothing: debatable,
`gold wrong`) and `nq:254`'s reason (`wrong`). C's debatable: `nq:3075` (a date a day from the gold's; the gold is an
older revision's) and `webq:592` (the gold is the year; the answer's day is not the usual one). M's wrong: question
echoes, non-answers that name the gold in passing, and false statements around the right word.

**4. Strict against current** (`grade_rates_2026-11-19.json`): with every lenient match read, both matchers give the
same right, wrong and refused (the read decides either way). What differs is what reaches the read:

| | lenient: matched (read wrong) | strict: matched (read wrong) |
|---|---|---|
| C: SimpleQA / NQ / WebQ | 6 (0) / 9 (0) / 49 (1) | 5 (0) / 9 (0) / 42 (0) |
| C-VM: SimpleQA / NQ / WebQ | 8 (0) / 9 (0) / 76 (1) | 6 (0) / 9 (0) / 67 (0) |
| M: SimpleQA / NQ / WebQ / AA | 11 (10) / 48 (33) / 51 (29) / 7 (7) | 2 (1) / 20 (12) / 25 (9) / 1 (1) |

C under both, right / wrong (+debatable) / refused: SimpleQA 6 / 0 / 4,320; NQ 10 / 4 (+1) / 3,595; WebQ 57 / 5 (+3)
/ 1,967 (wrong among spoken 5 of 65, 7.7%, 95% 3.3-16.8%); CLUTRR 827 / 0 / 319; AA 0 / 0 / 600. **The strict matcher
alone does not make a match safe:** 23 of M's 48 strict matches read wrong (a gold held whole inside a false answer,
on word boundaries). The paper's numbers are the strict matcher's **with its read**, as the order says. **The reading
taken (Nick's yes, 2026-11-19):** the strict matcher takes no `answer in gold` at all, narrower than the order's
wording, with the read behind it. M's unmatched
answers (SimpleQA 1,119, NQ 1,800, WebQ 660, AA 445) were never read in 0054 and are not estimated here.

**5. Published numbers:** the eight sources have `grader`, `not_attempted` and `matched`, from the settings recorded
on 2026-11-05 (not re-opened). Graders are named only by the AA paper (a judge model), the SimpleQA paper (a prompted
classifier) and the EM reports; the Claude Opus 5.5 card's AA figure is "not named". No setting matches 0054's whole
(the card's is the nearest: the public split, closed book).

**6. Judge parser:** the issue's four controls give not attempted, not attempted, B and B (`test_judge_parse.py`).
Only a standalone letter counts; several or none is not attempted; a letter whose label the text contradicts
("C: PARTIAL_ANSWER" under SimpleQA's labels) is not attempted. On AA's labels "C: PARTIAL_ANSWER" is a partial,
never correct.

**7. Where the golds come from** (as the benchmarks' papers state them; not re-checked here):

| benchmark | source of the gold | date | how checked |
|---|---|---|---|
| SimpleQA | written by the benchmark's trainers (Wei et al.) | 2024 | a second trainer answered each question independently |
| NQ open | spans annotated on Wikipedia pages (Kwiatkowski et al.; the open split, Lee et al.) | 2019 | several annotators on dev and test |
| WebQuestions | Freebase answers chosen by crowd workers (Berant et al.) | 2013 | the crowd's choice; Freebase is closed |
| CLUTRR | generated from kinship graphs (Sinha et al.) | 2019 | by construction |
| AA-Omniscience | the benchmark's authors (Jackson et al., arXiv 2511.13029) | 2025 | as the paper describes |

Every wrong or debatable the read gives carries a reason: C 2 `wrong` and 1 `gold stale`; C-VM the same; M 84 `wrong`, 3 `gold wrong`, 1 `question ambiguous`
(`gold wrong`: `nq:1824`, `nq:3526`, `webq:1023`). No gold edited.

- **Not verified here:** the published sources were not re-opened (step 5's fields come from their 2026-11-05
  records); step 7's table is the papers' own account; the read is mine, for Cowork's check.
- **Found on the way:** the records' 8-gold truncation (above); `webq:1113`'s right answer matched by accident (the
  gold is a word inside one of the answer's words): the lenient containment is not on word boundaries.
- **Questions, answered (Nick, 2026-11-19: "yes" to both):** (1) the narrower strict matcher stays (above);
  (2) future records keep every gold: `run_c.py` and `gpu.py` (`m_record`) write the whole list, pinned by
  `tests/validation/test_records_keep_golds.py`; the existing records are not edited, and the rules keep reading the
  full lists from `questions.jsonl`.
