# 0022 — Copy or refuse: a fact token comes from the copy, or the answer is "I don't know"
status: done
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes") · hypothesis: H-K10 · **after 0021**

## Why
Order 0021 found where the wrong answers come from. The read tells a present fact from an absent one: one threshold
on the best match separates them at a balanced 0.947-0.963. But when the copy's gate closes, the trunk speaks, and at
an asking position the trunk's pick is never the fact (alone it reads 0.000). So every piece under a closed gate is a
wrong answer, not a refusal: training the switch alone left 7.3-9.2% of present facts answered wrong.

On the same frozen toy models, **copy or refuse** (the copy's whole answer when every piece clears the threshold,
NONE otherwise) gave present wrong 0.6-1.3%, absent answered 2.9-4.9%, refused 12-16%
(`validation/logs/k10_policies_<arm>_s<seed>_2026-10-22.json`). The threshold was picked on the evaluation
documents, so that is an upper bound until it is fit on a split of its own. This order makes it the contract, in
the model, measures it honestly, and takes it to the 450M.

- **What it brings:** a fact Cubby speaks comes from what it was shown, whole, or Cubby says it doesn't know. The
  language model never fills in a fact.
- **Advantage over conventional models:** a chat model always produces its likeliest continuation, so a missing
  fact comes out as a confident guess. Here the decision to speak a fact is a measured test on the read, and its
  failure mode is a refusal.
- **In plain words:** Cubby gives you the fact it was told, or says "I don't know"; it never makes one up.

## Build
- **`SpatialCopy` mode `refuse`** (meta `spatial.copy_refuse = theta`, absent = today's mix, bit-identical): at an
  asking position, read every piece of the asked answer (the key is known there: placed, or the pointer's), take
  each piece's best match, and decide once for the whole answer. All pieces clear theta: the answer's tokens come
  from the copy alone (lambda = 1). Any piece below: the answer is refused (the toy's NONE; on the 450M a refusal the
  host speaks as "I don't know"). Decode = forward.
- **theta is fit, then frozen:** on a calibration split of documents disjoint from the test documents, the lowest
  theta that meets both wrong-answer lines; then measured once on the test documents.
- **A. The toy:** U, A and P (three seeds) from 0021's saved joint models.
- **B. The phrasings** (0021's part B): twelve training templates, P on three seeds, under the contract.
- **C. The 450M:** run 5's copy, nothing retrained, on a far-facts probe with absent questions: `facts_far1k_v2`
  plus a variant where 20% of questions ask a fact the document does not hold (`make_fact_shard.py --absent 0.2`,
  readable names). theta fit on its calibration documents, measured on the held-out ones. CPU is enough.
- **D, optional:** 0021's question 2, a gate slope apart from `tau`, trained with the gate loss: does it lower the
  refusals at the same wrong rates?

## Acceptance (pre-registered)
- **A and C:** present answered wrong <= 2%, absent answered <= 5%, at the theta fit on calibration documents.
  Refused and exact are reported beside them.
- **C:** exact on present facts and refused, against run 5's 0.946 with its mix. Language unchanged (the mode acts
  only at asking positions).
- **B:** on unseen subjects in held-out phrasings, exact >= 0.80 under the contract on 2 of 3 seeds, and the
  pointer's accuracy there >= 0.95.
- **Kill:** if meeting both wrong-answer lines costs more than 25% of present facts refused on the 450M, a single
  threshold is too blunt: the next step is a per-piece confidence learned on calibration documents (arm D's
  direction).
- **Tests:** off = bit-identical; a piece under theta refuses the whole answer; a refused answer emits no fact
  token; theta is read from calibration documents only; decode = forward.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Nothing is retrained on the 450M here.

## Out of scope
- Wiring the refusal into the serve stack's reply wording (the host's "I don't know" line already exists).
- Free-text keys on the 450M (a run of its own, after this).

## Report
- **Commits:** `6c917b6` (the contract in the model), `a8fdbce` (A), `a4c95ad` (C's script), and this one (the
  contract says the answer's end; B's per-cell reading; the logs; this report).
- **Files:** `cubbyllm/model/spatial_refuse.py` (the runs, the lookahead reads, the decision, applying it),
  `spatial_copy.py` (`set_refuse(theta, theta_end, refuse_id, step, lookahead, end_id=None)`, the decode state
  `cr_prev` / `cr_L` / `cr_off`), `spatial.py` / `assembly.py` (the state keys), `validation/train_base.py` (meta
  `spatial.copy_refuse`); `validation/k10_refuse.py` (A and B: theta fit on calibration documents, measured once on
  the test documents; `--tag`, B's cells), `validation/make_fact_shard.py` (`--absent`, `--skip-facts`, `--out`; off,
  byte-identical), `validation/exp_refuse_450m.py` (C; `--theta`, `--theta-end`, `--end` for the test half alone);
  `tests/model/test_spatial_refuse.py`.
- **Tests:** `tests/model/test_spatial_refuse.py` + `test_spatial_copy.py`: 26 passed. They pin off = bit-identical, a
  piece under theta refusing the whole answer, a refused answer emitting no fact token, theta read from calibration
  documents only, decode = forward, and a copied answer ending in `end_id` when it is set.
- **Measured** (theta fit on calibration documents, then measured once on the test documents; `k10_refuse_<arm>_s<seed>_2026-10-22.json`, `exp_refuse_450m_2026-10-22.json`, `exp_refuse_450m_end_2026-10-22.json`):

  | | theta | present wrong (<= 2%) | absent answered (<= 5%) | refused | exact |
  |---|---|---|---|---|---|
  | A: U s0 | 0.202 | 3.4% | 1.5% | 35.6% | 0.610 |
  | A: A s0 | 0.197 | 2.3% | 1.5% | 35.9% | 0.617 |
  | A: P s0 / s1 / s2 | 0.200 / 0.208 / 0.210 | 3.1 / 2.4 / 2.6% | 2.4 / 1.0 / 0.0% | 35.6 / 39.7 / 40.0% | 0.614 / 0.579 / 0.574 |
  | C: 450M, the trunk says the end | 0.117 | 5.4% | 0.0% | 32.9% | 0.617 |
  | **C: 450M, the contract says the end** | 0.117 | **1.7%** | **0.0%** | **32.9%** | **0.654** |
  | C: 450M, the mix (today) | -- | 12.4% | 100% | 0% | 0.876 |

  - **A misses the wrong-answer line on the test documents.** Every arm met it on its calibration documents
    (1.8-1.9%), then read 2.3-3.4% on the test documents. The lowest theta that meets a line sits on its edge, so a
    new sample lands past it. Fitting to a margin (say 1.5% on calibration) would hold, at more refusals. Absent
    answered meets its line everywhere (0.0-2.4%).
  - **C meets both wrong-answer lines once the contract says the end.** The first test-half run let the trunk say
    the end after a whole copied answer. The read was right, but run 5's trunk often went on past the answer, and the
    extra tokens counted as wrong: 5.4% against the 1.7% the decision alone predicts (`test_simulated` in the first
    log). With `end_id` the model reads exactly the simulation: exact 0.654, refused 32.9%, wrong 1.7%, absent
    answered 0.0%. Language is unchanged: the largest logit change at a position that is not an asking one is 0.0
    (`max_logit_change_off_the_askers`).
  - **Against run 5's 0.946:** on this probe the mix reads 0.876, not 0.946. The probe differs: 20% of its questions
    ask a fact the document does not hold, and the mix answers all of them (100% absent answered, 12.4% present
    wrong). Under the contract, exact falls to 0.654, and the 0.22 between the two is the price of the refusals.
  - **The kill criterion fires.** Meeting both wrong-answer lines on the 450M costs 32.9% of present facts refused,
    against a line of 25%. As the order says, a single threshold is too blunt. The next step is a per-piece
    confidence learned on calibration documents (arm D's direction).
  - **B fails its exact line; the pointer meets its line.** Unseen subjects in held-out phrasings, under the
    contract: exact 0.541 / 0.627 / 0.580 on the three seeds (line 0.80, so 0 of 3). The pointer's accuracy there is
    1.000 / 1.000 / 0.998 (line 0.95). Under the mix, the same cell reads 0.827 / 0.855 / 0.920
    (`exp_k10_free_text_P_s<seed>_t12_2026-10-22.json`). The key arrives; what costs exact is the refusals (41.5% on
    seed 2's cell). It is the same bluntness as C's, not a key problem.
- **Not verified here:** D (the gate slope apart from `tau`) is not run. Nothing was retrained; C ran on CPU on 200
  calibration and 200 test documents of `facts_far1k_v2_absent`.
- **Found on the way:** the trunk cannot be trusted to end a copied answer: `end_id` is part of the contract now, and
  any later free-text-keys run should carry it. Thresholds fit at the exact edge of a line will miss it out of sample.
- **Questions:** D next, as the kill criterion says (a per-piece confidence learned on calibration documents)? Or
  first a margin on theta, which trades refusals for the line A missed?

## Review (Cowork, 2026-10-22): done; the kill criterion fired, and the finding sets the next step
Checked against `exp_refuse_450m_2026-10-22.json` and `exp_refuse_450m_end_2026-10-22.json` (theta 0.117, 814 present
and 183 absent test questions): the report's numbers are the logs'. **Done.**
- **The contract works on the 450M.** With the contract saying the answer's end, both wrong-answer lines hold on the
  test documents: present wrong 1.7% (14 of 814), absent answered 0.0% (0 of 183), language unchanged off the asking
  positions. `end_id` belongs to every later run: the trunk cannot be trusted to end a copied answer.
- **The price is the kill line's:** 32.9% of present facts refused, against 25%. A single threshold is too blunt.
- **What the numbers say about where.** Absent answered is 0.0% at this theta, so the threshold is set by the
  present-but-wrong answers (12.4% under the mix), not by the absent ones: one scalar separates present from absent
  well (0021: 0.95) and right from wrong among present facts poorly. That is what arm D has to learn.
- **A's miss on the test documents** (2.3-3.4% against 1.8-1.9% on calibration) is a threshold fit at the line's edge.
  Not a margin picked by hand: fit the next threshold with a conformal risk bound, which holds out of sample.
- **B** fails its exact line under the contract (0.54-0.63) while the pointer reads 0.998-1.000: the same refusals,
  not a key problem.
- **The question:** D next, with the margin folded into it (the conformal fit). Proposed as order 0029: diagnostics
  first (refusals and errors by answer length, piece and kind), then a per-answer confidence learned on half the
  calibration documents, its threshold set on the other half, measured once on the test documents.
