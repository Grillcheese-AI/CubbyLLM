# 0029 — Copy or refuse, arm D: a confidence learned on calibration documents, so fewer right answers are refused
status: proposed
written: 2026-10-22 by Cowork · approved: — · hypothesis: H-K10 (after 0022)

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
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
