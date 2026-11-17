# 0075 - The graders checked before their rates are trusted (0054's right, wrong and refused)
status: open (approved by Nick on 2026-11-17: "I approve both"; builder: Code A, or Cowork on a branch while Code A is signed out)
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
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
