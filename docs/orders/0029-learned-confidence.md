# 0029 — Copy or refuse, arm D: a confidence learned on calibration documents, so fewer right answers are refused
status: in progress
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes") · hypothesis: H-K10 (after 0022)

## Why
Order 0022 made copy or refuse the contract, and on the 450M it holds both wrong-answer lines: present wrong 1.7%,
absent answered 0.0% (`exp_refuse_450m_end_2026-10-22.json`). Its kill line fired: 32.9% of present facts refused,
against 25%. The numbers say where the threshold is spent. Absent questions are all refused already, so the cut is set
by present facts answered wrong (12.4% under the mix): one scalar (the best match) tells present from absent well
(0021: 0.95 balanced) and right from wrong among present facts poorly. On the toy, a threshold fit at the line's edge
on calibration documents missed the line on test documents (2.3-3.4%).

What it brings: the same "facts from the copy or not at all", with about a third fewer refusals as the target. A
conventional model has no refusal to tune: it answers, right or wrong.
In plain words: Cubby keeps saying "I don't know" when it might be wrong, but says it less often when it is right.

## Build
- **Step 1, diagnostics (nothing trained).** On 0022's calibration documents, break the refusals and the wrong answers
  down by answer length (pieces), by piece position, by the best match and the margin to the second candidate among
  the document's writers, by the gate's λ, and by error kind (another fact of the same document, the same relation of
  another subject, a name cut short or run on). Report it before step 2: if the errors sit in long answers, the
  per-answer aggregation (today: every piece above theta) is the thing to change.
- **Step 2, the confidence.** A small logistic model per answer (a few hundred parameters at most) over the per-piece
  features aggregated (min, mean, first, last), the margin, the piece count and λ; fitted on one half of the
  calibration documents. Nothing in the trunk or the copy is retrained.
- **Step 3, the threshold**, set on the other half with a conformal risk bound: the smallest refusal whose bounded
  risk, `(n · risk + 1) / (n + 1)`, is at or under each line (2% present wrong, 5% absent answered).
- Measured once on 0022's 200 test documents and on 200 fresh ones (`make_fact_shard.py --absent`, a new seed).
- Flag-gated (`set_refuse(..., scorer=)`), off = bit-identical to 0022's contract.

## Acceptance (pre-registered)
- Both wrong-answer lines on the test documents (present wrong at most 2%, absent answered at most 5%).
- Refused at most 25% of present facts (0022's kill line), exact at least 0.70.
- A risk-coverage curve (refused against present wrong) for 0022's scalar and for the learned confidence, on the same
  documents.
- **Kill:** refusals still above 25% with the lines held: the errors are not visible to these features. Then the next
  step is verification instead of confidence: the store is asked whether the copied fact exists before it is said.
- Tests: off = bit-identical; the scorer and the threshold read calibration documents only; decode = forward.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Calibration and test documents never mix; the test documents are read once.

## Out of scope
- Retraining the copy or the trunk; B's toy cells (they follow if D works on the 450M).

## Report
<!-- Claude Code fills this in, then sets status: review -->
**Step 1, the diagnostics (2026-10-22).** The status stays `in progress`: step 2 waits for your read of this.

- **Commits:** the step-1 commit (`[0029] step 1: ...`).
- **Files:** `validation/refuse_diag_450m.py` (new, 261 lines; `--from-rows` recomputes the summary from the saved rows
  without the model). Logs: `validation/logs/refuse_diag_450m_2026-10-22.json` and its `.rows.json` (one row per answer:
  every piece's best match, runner-up, the writer it picked, lambda, the copy's pick).
- **Tests:** none new. It is a measurement, and nothing in `cubbyllm/` changed.
- **Measured** ([log](../../validation/logs/refuse_diag_450m_2026-10-22.json)). Run 5's final checkpoint, on the calibration half only: 200 documents, 802 present and 175
  absent questions. CPU, 2 threads, 1,414 s. The test half is unread.
  - **It reproduces 0022's calibration exactly** at 0022's pair (theta 0.117, theta_end 0.085): exact 0.6958, present
    refused 0.2843, present wrong 0.020, absent answered 0.0.
  - **Every wrong answer is a boundary error.** All 16 have the pieces right and the length wrong: 13 cut short, 3 run
    on, by 1 to 5 pieces. None is a piece of another fact (the same relation of another subject, or another relation),
    and none is a piece of the right fact in the wrong place.
  - **Most refusals were right answers.** Of the 228 refused present facts, 211 had the copy's picks equal to the gold
    over its length:
    - 157 had a true piece read below theta;
    - 50 had the end read above theta_end;
    - 4 had nothing read.
    Only 17 refused answers had wrong picks.
  - **Refusals climb with length; wrong answers do not.**

    | pieces | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9+ |
    |---|---|---|---|---|---|---|---|---|
    | refused | 16.9% | 18.1% | 23.0% | 31.5% | 38.5% | 53.2% | 46.4% | 71.1% |
    | wrong (count) | 2 | 4 | 1 | 4 | 3 | 1 | 1 | 0 |

    The rule "every piece above theta" gives each extra piece another chance to dip below theta. So the per-answer
    aggregation is what spends the refusals, which is the step-1 question the order asks. It is the refusals that sit
    in long answers, though, not the errors.
  - **Absent questions are told apart at the first piece.** Their first reads all sit below theta (the highest is
    0.1053), and 89.7% sit below theta_end. Only 0.6% of present first reads fall below theta_end. Inside an answer,
    theta is spent on present facts alone.
  - **The margin to the runner-up separates the wrong answers somewhat; lambda does not.**
    - Margin: 11 of the 16 wrong answers have their smallest margin under 0.02, beside 76 refused and 44 exact.
    - Lambda: at a mean of 0.9 or more there are 553 exact, 195 refused and 11 wrong.
- **What this means for step 2.** The two kinds of failure sit in different places:
  - the errors are in where the answer ends;
  - the refusals are in the pieces of long answers.

  So the confidence should score the two separately:
  - the pieces as an aggregate (mean, first and count, not only the min);
  - the end read by its own match and margin.

  The order's feature list (min, mean, first, last, margin, count, lambda) allows this if "last" is the end read. The
  headroom is the 211 right-pick refusals (26% of present facts). The scorer needs to admit about a third of them
  without letting in more boundary errors to bring refusals under 25%.
- **Not verified here:** nothing ran on a GPU. The test documents are unread, as the order requires.
- **Found on the way:** nothing outside the order.
- **Questions:**
  1. In step 2, should the end read (its best match and its margin at the answer's end) be features of its own, apart
     from the last piece's?
  2. May step 2 start?
