# 0082 - The programs adapter learns the second writer's wordings: H-E18's plan and chain records in S0's mix
status: open (Code A, 2026-11-24; the training is Nick's Colab run)
written: 2026-11-24 by Cowork, after an outside review of the hypothesis file · approved: 2026-11-24 by Nick ("yes") · hypothesis: H-E18 (program first, words second), H-E15

## Why
H-E18 (2026-09-30) trained the 450M's emitter with and without the program-first records, at equal steps on
`base450m_cont`, and read both on the held-out eval written by a second writer (wordings, shapes and seeds never
trained on):
- plan 0.225 -> 0.495 (+27.0 points), chain 0.915 -> 0.993 (+7.8), arithmetic 0.021 -> 0.059 (+3.8);
- no v12e val family down;
- slots: the plan and chain families copied no literal and named no unknown slot. The 13 copied literals and 10
  unknown slots were all arithmetic and kernel programs, which is why the verdict was "neither".

Its next step was written the same day: keep the program-first plan and chain records in the emitter's mix, and hold
its arithmetic out. That never happened. The lineage went v12e -> Gate A (`v12e_w_tg`) -> the write-back adapter
(`cubby450m_v12e_w_step_cont`, S0), which is the programs adapter Cubby serves today and 0078's S0. None of those
mixes holds a program-first record, so the served seat still plans knowledge questions the way v12e alone did.

This order puts the 1,079 plan and 1,101 chain records into S0's own recipe, and nothing else changes. The outside
review (2026-11-24) asked for exactly this, and the hypothesis file's held-out rule (section 0) sets the bar it is
read against.

- **What it brings CubbyLLM:** the 450M plans knowledge questions phrased in ways it never saw, the gain H-E18
  measured, in the adapter that is actually served.
- **Over a conventional model:** the new examples come from programs sampled first and checked in the VM, and a
  second writer words them; nothing an LLM says is a label. More wordings teach the plan, and the VM still checks
  every answer.
- **What it does for you:** ask Cubby a question about a chain of facts in your own words, and it plans it instead
  of stumbling on a phrasing it never saw.

## Build
1. **The mix** (`validation/sft/build_0082_mix.py`, new):
   - Start from S0's file, `emitter_sft_v12e_w_tg30_step.jsonl`, unchanged.
   - Add the program-first **plan** and **chain** records of `emitter_sft_v12e_pf_slots.jsonl` (1,079 and 1,101),
     converted to S0's slot form the way v12e_w was (`validation/emitter_data.py`, the options v12e_w used). No
     program-first arithmetic record goes in.
   - Repeat them as S0's file repeats v12e's own plan and chain families, so their share within those families is
     H-E18's. Report the counts and the repeat used.
   - Checks before writing: each added program re-run in the VM equals its record's value; 0 prompts, 0 chain seeds
     and 0 plan shapes shared with `pf_heldout_eval_w_slots.jsonl` (bucket 0, the other writer); slot conversion drops
     counted, never guessed.
   - A manifest in `validation/logs/` (counts by family, SHA-256 of the output, the source files' SHA-256). The data
     stays off-repo, on Drive beside S0's file.
2. **The notebook** (`notebooks/emitter_pf_0082.ipynb`, small, like `emitter_s1.ipynb`; Nick runs it on an A100):
   - **Arm P:** the Gate A adapter (`cubby450m_v12e_w_tg_cont`) continued one epoch on the new mix with S0's recipe
     (`train_emitter_torch.py`: S0's batch, lr, rank, alpha and max length, as H-E19 and `emitter_s1.ipynb` record
     them; `--no-gen`). S0's run took 34.6 min on an A100; this mix is a few percent longer,
     with the repeat.
   - **Then, for P and for S0** (`--gen-only --resume-adapter`): the plan and chain records of
     `pf_heldout_eval_w_slots.jsonl` (285 and 294), and the v12e_w val split's families other than arithmetic.
   - Skips a finished arm when its files exist, so a dropped session resumes.
3. **The reads, at home:**
   - `validation/exp_he18_ab.py --files <S0 files> <P files>`: the real VM, CPU, per family, with the slot verdicts;
   - `validation/exp_he19_step_loop.py`, greedy, on the 236 program-first arithmetic problems, P against S0's 0.229;
   - after 0081's runs are done on the home card: the gen-3 held split (seed 7) and H-E15's live eight through the
     served loop, P against S0, with `CB_EMIT_TRIE` as 0081 left it (off if 0081 was killed).

## Acceptance (pre-registered)
- **Tests:** the mix builder on a small fixture (no arithmetic record added, the repeat applied, a planted held-out
  prompt refused, the VM check run).
- **Numbers to report:** the mix's counts; per arm, on the unseen-writer eval, right, wrong and copied / unknown slots
  for plan and chain; the v12e_w val families; the 236's greedy loop; the gen-3 split and the live eight (planned,
  verified, correct, WRONG, refusals by reason).
- **Gate:**
  - **The unseen-writer eval, read once per arm:** plan >= 0.55 and chain >= 0.99, with 0 copied literals and 0
    unknown slots in both families (the hypothesis file's win bar), and P's plan at least 10 points above S0's.
  - **Nothing forgotten:** no v12e_w val family other than arithmetic more than 2 points below S0; the 236's greedy
    loop within 2 points of S0's.
  - **Wrong answers:** 0 spoken on the gen-3 split and the live eight.
- **Kill:** P's plan less than 3 points above S0's (H-E18's kill line), or the 236's loop more than 2 points below
  S0's. The program-first plan and chain records then stay out of the served mix.
- **Between the kill and the gate:** reported; serving P is Nick's call.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Held-out discipline** (the hypothesis file, section 0): the steps are S0's recipe, fixed before the run; nothing
  is chosen on the unseen-writer eval, the 236 or the gen-3 split; each is read once per arm.
- One change against S0: the added records. Same start adapter, same recipe, same base.
- **0078 is untouched:** P is not one of 0078's arms, and S0 stays 0078's baseline as it is. If 0078 keeps S1 or S2,
  adding the same records to that arm's mix is a follow-up, not this order.
- **Serving:** P replaces the served programs adapter only on Nick's yes, and not while 0078's reads use S0.
- **The home card** is shared: the served-loop reads wait for 0081's runs to finish.
- The data stays off-repo; logs go to `validation/logs/exp_0082_*_<date>.json`, with no local paths in committed
  files.

## Out of scope
- Program-first arithmetic (H-E19 and 0078 own arithmetic).
- A new base, a new rank, or a longer budget than S0's.
- The decode-time constraint (0081).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
