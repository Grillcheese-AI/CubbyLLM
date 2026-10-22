# 0021 — Free-text facts, part 2: train the switch alone (0005's recipe on 0006's toy), then widen the templates
status: review
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes") · hypothesis: H-K10 · **after 0006**

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
- **The finding first.** Training the switch alone does not reach A's lines, and the reason is not what the kill
  line supposes.
  - The read's inputs tell present from absent: one threshold on the best match separates them at a balanced
    0.947-0.963 in every arm, and every answer piece clears it, not only the first.
  - What the lines miss on is what the gate hands the answer to when it closes. At an asking position the trunk's
    own pick is never the fact (LM alone 0.000 in 0006), so every piece under the gate becomes a wrong answer, not
    a refusal.
  - A host-side **copy or refuse** on the same frozen models (the copy's whole answer when every piece clears the
    threshold, NONE otherwise) meets both wrong-answer lines in all five arms: present answered wrong 0.4-1.3%,
    absent answered 2.7-4.9%. It does so by refusing 12-21% of present facts, against the trained gate's 9-12%
    wrong.
  - That is the 0006 copy's quality (0.927-0.974 alone), said as "the fact or nothing".
  - Part B was not run (it is sequenced after A, and A's lines fail). See the questions.
- **Commits:** `1111a59` (--save, the switch alone, present wrong, 12 templates), `12f1c84` (the readout split by
  absent questions, the opt-in gate loss), `907f105` (no local path in the joint log), `48a018d` (part A's first
  runs: superseded, see below), and this report's commit (the stream fix, the rerun, `k10_policies.py`).
- **Files:**
  - `validation/exp_k10_free_text.py`: --save, the present-wrong column, --templates.
  - `validation/exp_k10_switch.py` (new): the switch alone on a frozen trunk, `stream`, `best_match`, --gate-loss.
  - `validation/k10_policies.py` (new): the switch, copy-or-refuse and the gate's own mix, scored on the joint
    model.
  - `validation/k10_free_docs.py`: templates 8-13 and their function words, used only with templates == 12.
  - Tests: `tests/test_k10_switch.py` (5).
- **Tests:**
  - A resumed model's tensors equal the saved ones, and it scores the same.
  - The switch-alone steps move only the gate, tau and beta; the frozen hash holds.
  - present wrong = 1 - exact - refused.
  - The gate loss off is the recipe, bit for bit, and it targets the asked pieces only.
  - Twelve templates train on 0-5 and 8-13, never 6-7, and the held-out documents are the same.
  - The switch trains in the joint run's filler language.
  - Off unchanged: tiny joint runs of P and U with HEAD's files and these give identical metrics.
- **Measured** (logs `validation/logs/exp_k10_switch_<arm>_s<seed>[_gl1]_2026-10-22.json`,
  `k10_policies_<arm>_s<seed>_2026-10-22.json`; joint 1,000 steps, then 500 steps of the switch alone):

  | arm | joint exact | switch alone: exact | unseen × held-out | absent answered | present wrong | refused |
  |---|---|---|---|---|---|---|
  | U s0 | 0.848 | 0.879 | 0.849 | 0.083 | 0.092 | 0.029 |
  | A s0 | 0.857 | 0.897 | 0.881 | 0.097 | 0.077 | 0.026 |
  | P s0 | 0.855 | 0.883 | 0.854 | 0.102 | 0.073 | 0.044 |
  | P s1 | 0.849 | 0.877 | 0.824 | 0.069 | 0.089 | 0.034 |
  | P s2 | 0.841 | 0.863 | 0.835 | 0.071 | 0.076 | 0.060 |

  - **U ≥ 0.93: fails** (0.879).
  - **Present answered wrong ≤ 3%: fails** in every arm (7.3-9.2%).
  - **Absent answered ≤ 5%: fails** in every arm (6.9-10.2%).
  - **P ≥ 0.80 on unseen × held-out, 2 of 3 seeds: passes, 3 of 3** (0.854 / 0.824 / 0.835).
  - **Language:** moves at the fourth decimal (2.2607 → 2.2605 on U). The copy acts at asking positions, and the
    answer's closing EOS is predicted at one; the filler is untouched.
  - **The frozen hash is unchanged in every arm.**
  - **Kill line:** on U, absent 8.3% with refused 2.9%, so it triggers as written. Its stated cause does not hold:
    the best match splits present from absent (one threshold 0.17-0.18, balanced accuracy 0.947-0.963).
  - **The gate loss (an opt-in arm, not the recipe):** a BCE on "the asked fact holds a piece here" barely moves
    it (U absent 0.083 → 0.073; P s0 and A unchanged). The gate's slope on the best match is `exp(tau)`, the copy's
    own cleanup temperature, so it cannot sharpen without changing the copy.
  - **Copy or refuse** (`k10_policies`; the lowest-refusal threshold meeting both lines):

    | arm | threshold | exact | present wrong | refused | absent answered |
    |---|---|---|---|---|---|
    | U s0 | 0.18 | 0.840 | 0.013 | 0.147 | 0.029 |
    | A s0 | 0.17 | 0.872 | 0.006 | 0.122 | 0.044 |
    | P s0 | 0.18 | 0.835 | 0.012 | 0.153 | 0.034 |
    | P s1 | 0.17 | 0.853 | 0.013 | 0.134 | 0.049 |
    | P s2 | 0.17 | 0.830 | 0.006 | 0.164 | 0.044 |

    A hard per-piece switch to the trunk at the same thresholds keeps 10.6-13.0% present wrong: the trunk's guess
    is the error.
- **Not verified here:** the 450M (out of scope), and part B.
- **Found on the way:**
  - **Part A's first runs (`48a018d`) trained the switch on another filler language.** `copy_only` drew its
    documents with `Language(seed + 8)`, and the language's grammar is drawn from its seed. Its training loss sat
    at ~10.5 nats against the joint model's 2.6. Fixed (`stream`: `seed + 1`, new draws at `seed + 7`) and every
    arm rerun. The numbers moved by at most 0.003, so the language was not the cause either. The logs above are
    the rerun.
  - The P joint runs this time read 0.849 / 0.841 on seeds 1 / 2 against 0006's 0.783 / 0.826. The code is the
    same for the joint phase; CPU threading order (0006 ran at 2 threads beside other jobs) is the likeliest source.
    Seed-to-seed spread is of that order.
- **Questions:**
  1. Adopt copy-or-refuse as the switch's contract: a fact token comes from the copy or the answer is NONE, never
     from the trunk. It meets the wrong-answer lines on today's models. The cost is 12-21% refused; U's 0.93
     would then rest on the copy's own 0.957. Building it in the model is a mode of `SpatialCopy` (a threshold on
     `best` with a NONE fallback), an order of its own.
  2. Or decouple the gate's slope from `tau` (a separate scale on `best - beta`) and train it with the gate loss?
  3. Part B (twelve templates): run it now, since P's held-out phrasings already pass on 3 of 3 seeds, or after the
     switch's contract is settled?
