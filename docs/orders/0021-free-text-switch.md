# 0021 — Free-text facts, part 2: train the switch alone (0005's recipe on 0006's toy), then widen the templates
status: proposed
written: 2026-10-22 by Cowork · approved: — · hypothesis: H-K10 · **after 0006**

## Why
Order 0006 showed that text keys and the pointer work. On subjects the model never saw, asked in phrasings seen in
training, the pointer reads 0.854 / 0.899 / 0.870 (three seeds), where hashed addresses read 0.000. Text keys cost
nothing as addresses: given the address, text keys read 0.848 and today's hashed places 0.857. Its pass lines missed
anyway: U 0.848 against 0.90, P 0.816 / 0.694 / 0.785 on unseen subjects in held-out phrasings against 0.80 on 2 of 3
seeds, and absent facts answered 5.3-9.9% against 5% (`validation/logs/exp_k10_free_text_<arm>_s<seed>_2026-10-21.json`).

The cause is the switch, not the read. The copy path alone, before it is mixed with the language model, picks the
right answer on 0.956 (U) and 0.974 (hashed) of present facts; mixed, 0.846 and 0.857
(`..._alone_2026-10-22.json`). The gate is open 0.90 on average on a present piece, but enough pieces fall under 0.5
that the trained language model's guess wins, and exact needs all five. On empty addresses it is 0.15-0.18 open, and
absent facts get an answer. So 1-0.85-0.03 = about 12% of present facts are answered wrong: more wrong answers than
the absent ones, and under "0 wrong answers spoken" the number that counts most.

Order 0005 fixed the same failure on the 450M by training the copy alone on a frozen trunk: far facts 0.761 to 0.946,
text identical. This order applies that recipe to the toy, then turns to the phrasings: on seed 1, the held-out
phrasings' 0.694 tracks the pointer's own 0.895.

- **What it brings:** free-text questions about names the model never saw, answered as reliably as the copy reads
  them, with absent facts refused, and nothing else in the model changed.
- **Advantage over conventional models:** the knowledge and the decision to speak live in a few hundred parameters
  beside a frozen language model, trained on whether the read holds something, so a fact added later cannot drift
  the model's language.
- **In plain words:** Cubby learns when it truly has the answer and when it doesn't, without retraining the rest
  of itself.

## Build
- `validation/exp_k10_free_text.py`:
  - `--save PATH`: save the model at the end of the joint phase (today's 1,000 steps), with its args and seed.
  - `--copy-only STEPS --resume PATH`: freeze everything but the copy path's gate, tau and beta (the pointer and the
    trunk frozen), train STEPS more on the same documents, absent facts included, and evaluate as today.
  - Report per arm: exact, the four cells, absent answered, **present answered wrong** (1 - exact - refused, as its
    own column), present refused, the copy alone, lambda on present and empty pieces.
- **A. The switch:** U, A and P (three seeds) from their saved joint models, `--copy-only 500`.
- **B. The phrasings** (only after A): training templates from 6 to 12 (the same 2 held out), P on three seeds, joint
  then copy-only as in A. Report the pointer's accuracy on held-out phrasings beside exact.

## Acceptance (pre-registered)
- **A:**
  - U >= 0.93 exact (the copy alone reads 0.956: the switch should close most of the gap).
  - Present answered wrong <= 3%, and absent facts answered <= 5%, for U, A and every P seed. Present refused is
    reported; it may rise, because a refusal is a result.
  - P >= 0.80 on unseen subjects in held-out phrasings, on 2 of 3 seeds.
  - Language: identical to the joint model's (only the gate trains; the copy acts only at asking tokens).
  - **Kill:** if copy-only training cannot bring absent <= 5% with present refused <= 10% on U, the gate's inputs
    (the stream and the best match) cannot tell present from empty here: report the two distributions of the best
    match, and the next step is a gate that sees the read's margin over the second-best candidate.
- **B:** P >= 0.85 on unseen subjects in held-out phrasings, on 2 of 3 seeds, with the pointer's accuracy on them
  >= 0.95. If 12 templates do not lift the pointer, the next step is a pointer that keys on the relation words'
  identity rather than their position (0006's report, question 2).
- **Tests:** copy-only leaves every frozen parameter bit-identical (hash before and after); resume = the saved model;
  the new column equals 1 - exact - refused.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- The toy only (CPU). Nothing runs on the 450M here.

## Out of scope
- The 450M run with text keys and the pointer: an order of its own once A and B pass.
- A host-side parser (0006's fallback): not needed while the pointer delivers the key.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
