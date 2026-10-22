# 0022 — Copy or refuse: a fact token comes from the copy, or the answer is "I don't know"
status: proposed
written: 2026-10-22 by Cowork · approved: — · hypothesis: H-K10 · **after 0021**

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
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
