# 0082 - The programs adapter learns the second writer's wordings: H-E18's plan and chain records in S0's mix
status: in progress (Code A, 2026-11-24; the training is Nick's Colab run)
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
### Code A: the mix and the notebook (2026-11-24); the training and the reads wait on Nick's run
- **Files:** `validation/sft/build_0082_mix.py` (new), `notebooks/_build_emitter_pf_0082.py` -> `emitter_pf_0082.ipynb`,
  `tests/validation/test_build_0082_mix.py`; manifest `validation/logs/build_0082_mix_2026-11-24.json`. The mix
  (`emitter_sft_0082_pf.jsonl`, 150,614 records, SHA-256 `09b9c387...`) is on Drive beside S0's file, off the repo.
- **The mix:** S0's file unchanged (146,557 train / 1,880 val records), then **1,076 plan and 1,101 chain**
  program-first train records. No arithmetic. About 1.5% more train rows than S0.
  - **The repeat used: each record's own (1), as in H-E18's mix.** S0's 2,288 plan and chain train records are
    byte-identical (repeats too) to their twins in the program-first file, so S0 repeats v12e's families exactly as
    H-E18's mix did. Keeping the added records at their own repeat therefore gives H-E18's share: 0.486 of plan rows,
    0.279 of chain rows (S0 repeats its chain records 2, 3 or 6 times: 2,840 rows from 1,150 records). A first build
    applied S0's most common chain repeat (2) to the added chains; that gives a share H-E18 never trained, so it was
    replaced before anything ran.
  - **Already in S0's slot form:** the program-first slot file was made by `emitter_data.convert` with v12e_w's
    options, as the identical twins show. The conversion dropped 22 of the raw file's 1,101 plans (counted against
    the raw file, not guessed back); 0 chains.
  - **The VM check:** each added record's filled program and its reference were run in the real VM, giving the same
    value (a chain's is its gold too). 3 plans fail and are left out: all three are seeded on
    `james "beg" stewart`, whose quotes break the program's string literal when the slot is filled. The host would
    fail the same way.
  - **Held-out:** 0 prompts or questions, 0 seeds (chain and plan, normalized), 0 plan shapes (the seed and its
    relations) shared with the unseen-writer eval (`pf_heldout_eval_w_slots.jsonl`: 579 seeds, read from the raw
    file's provenance), and no added seed in bucket 0. The shape check is implied by the seed check here (a shape
    carries its seed); it is kept as its own count.
- **The notebook (Nick's to run, an A100):** arm P = the Gate A adapter continued one epoch on the mix, S0's recipe
  (batch 16, lr 2e-4, rank 16 / alpha 32, max length 1,024, seed 0, the steps from the train rows' repeats, as
  `emitter_s1.ipynb`), `--no-gen`. Then `--gen-only` for P and for S0 on the val split's chain, kernel, plan and
  role-binding records and the unseen-writer eval's plan and chain records (all of them). S0's generations go to
  `emitter/exp_0082/s0`, leaving S0's adapter folder as it is. Each step is skipped when its files exist; every log
  line names the card.
  - **The optional last cell (off):** grilly2 0.1.0 on Colab (Vulkan registered with the two JSON files from
    grilly2's `notebooks/colab.ipynb`), cubelang built at `efa2559`, then the gen-3 split (seed 7) and the live eight
    for S0 and P at one `CB_EMIT_TRIE` arm. It asserts its preconditions: 0081's 450M bench arm on master (on
    `agent-b` now), and the base's grilly2 export on Drive (today it is only at home).
- **Tests:** the builder on a fixture (S0's rows first and unchanged; no arithmetic added; a planted held-out prompt
  and a held-out seed refused; a VM disagreement left out; the conversion drop counted; the share).
- **Next (after Nick's run):** `exp_he18_ab.py --files` on S0's and P's generations; the 236's greedy step loop for P
  against S0's 0.229; the served-loop reads after 0081's runs.
- **Not verified here:** the notebook on Colab; the optional cell (needs the export on Drive and 0081 on master).
- **2026-11-25, the generations batched.** On an L4 the one-at-a-time generations ran at ~12 s a record (864 records
  an arm: ~3 h, ~6 h for both). `train_emitter_torch.py --gen-batch N` decodes the records of one prompt length
  together (at most N) with 0078's CUDA-graph step (`dev_read_torch.greedy_rows`); the 864 prompts have 153 lengths.
  The notebook passes 32, for both arms. On CPU it gives what `generate` gives each record alone (pinned:
  `test_greedy_rows_equal_generate_one_by_one`; and a smoke run, 8 of 8 the same). On a card a batch's matmuls may
  round a near-tie differently, so P and S0 are both decoded this way and each file records its decode. Off (0) the
  old path is unchanged. The base's grilly2 export (`base450m_cont_final.pt`, step 24414) is now on Drive at
  `runs/base450m_cont/grilly/` (Nick's yes), SHA-256 checked.

### Code A: the first run was not S0's recipe; arm P again at batch 32 (2026-11-25)

- **The mismatch.** S0 trained **4,691 steps of 32** (its adapter file and its training log). The notebook said batch
  16, copied from `emitter_s1.ipynb`'s note, which is not what ran; I did not check S0's adapter file. On Nick's L4,
  batch 16 ran out of memory and P ran at **batch 8, 19,033 steps**: 4 times S0's optimizer steps at the same lr.
  So the first run changes two things against S0, the records and the steps, and its gain cannot be credited to the
  records alone. It stays on Drive as a side arm, **P8** (`cubby450m_0082_p`).
- **P8's read** (Cowork's, repeated here with the log: `validation/logs/exp_0082_he18_ab_p8_2026-11-25.json`; the
  real VM; both arms decoded with `--gen-batch 32`):

  | | S0 | P8 | delta |
  |---|---|---|---|
  | unseen-writer plan (285) | 0.270 | 0.586 | +31.6 |
  | unseen-writer chain (294) | 0.908 | 0.990 (291; 0.9898) | +8.2 |
  | val chain / kernel / plan / role-binding | 0.889 / 0.768 / 0.968 / 0.984 | 0.952 / 0.789 / 0.984 / 1.000 | +6.3 / +2.1 / +1.6 / +1.6 |

  Slots: 0 copied, 0 unknown on the unseen-writer plan and chain, both arms; P8 has 1 copied literal in kernel val
  (S0: 2), which is what the script's VERDICT NEITHER reads (H-E18's all-family clause, not this order's gate).
  Against this order's gate, P8 would pass plan (>= 0.55, +31.6 >= +10) and miss chain by one record (0.9898 < 0.99).
  The 236's loop is not run on P8.
- **Arm P again, batch 32 (Nick's choice: on the L4, by accumulation).** `train_emitter_torch.py --micro-tokens N`
  (`train_talk_torch.py`'s): each step's 32 rows are drawn exactly as an unchunked batch of 32 would be (the same order
  for the same seed), then cut into chunks of at most N padded tokens, each chunk's loss weighted by its share of the answer tokens: the batch's mean
  (`tests/validation/test_talk_micro_batches.py`; a CPU smoke run, whole against chunked: the adapters agree to
  1.6e-7). Off (0) is the old loop. The notebook now trains `cubby450m_0082_p32` at 4,759 steps of 32, chunks of
  8,192 tokens under 70 GB (8 x 1,024 fit on the L4), and generates it as before; S0's generations are kept.
- **The held-out discipline, for Cowork to rule on.** The unseen-writer eval has now been read once with the VM on
  P8, and P32 will be read on it once more. The rerun is decided on the recipe (the order's "one change against S0"),
  not on P8's numbers; but the logs' text-exact lines (P8's plan 0.561 against S0's 0.249) were seen before Nick chose,
  and Cowork's VM read was made before the mismatch was raised. I read P32 as the order's arm P and P8 as a labelled
  side arm, unless Cowork rules otherwise.

### Cowork's ruling on the held-out discipline (2026-11-25)
Code A's reading holds: **P32 is the order's arm P; P8 is a labelled side arm.** The order fixed S0's recipe and one
change against S0 before any run, and P8 broke both (4 times S0's optimizer steps). S0's adapter file and P8's log
show it whatever P8's numbers are, so training P again restores the pre-registered arm; it does not choose between
outcomes. Fixed now, before P32 is read:
- **P32's read is the gate read,** whatever it shows. P8 never stands in for P, and the better of the two is never
  reported as P's.
- **No third run of arm P under this order.** The bar stays as written: chain at least 0.99, which is 292 of 294; 291
  is a miss.
- **P32 differs from P8 in batch and steps only:** the same mix, start adapter, lr, rank, alpha, max length and seed,
  read with the same script and decode (`--gen-batch 32`). The report states each from P32's adapter file and log,
  not from the notebook's text.
- **P8 is read on nothing more:** not the 236, the gen-3 split or the live eight. Those are read on P32 only.
- **The report says the unseen-writer eval was read twice for arm P:** once on P8, before the mismatch was raised,
  and that P8's text-exact lines and Cowork's VM read were seen before Nick chose to train again.
- **In the hypothesis file,** P8 appears only as a side arm with a second change beside the records. P32 against P8
  is reported for what it shows (whether the extra steps carry part of the gain); nothing is chosen from it.

### Code A: arm P (P32) read on the unseen-writer eval and the 236 (2026-11-28); the served reads wait
- **P32's settings, from its adapter file and log** (`cubby450m_0082_p32`): 4,759 steps of 32 on
  `emitter_sft_0082_pf.jsonl`, lr 2e-4 (warmup 50), rank 16, alpha 32, max length 1,024, seed 0, from
  `cubby450m_v12e_w_tg_cont`, on `base450m_cont_final.pt` (`7d4bf9d6...`); 73.9 min on an L4; the last held program
  loss 0.0404. S0's adapter file: 4,691 steps of 32, the same lr, rank, alpha and max length. Two first starts ran out
  of memory before step 1 with chunks of 8,192 and 6,144 tokens (the log keeps all three starts); the run that
  trained used chunks of at most 4,096 tokens, as the adapter file records (the batch of 32 is the same; only how it
  is cut differs). So P32 differs from P8 in batch and steps only, as ruled.
- **The gate read, once** (`validation/logs/exp_0082_he18_ab_2026-11-28.json`; the real VM; both arms decoded with
  `--gen-batch 32`, S0's generations those of 2026-11-25):

  | | S0 | P32 | delta | the gate |
  |---|---|---|---|---|
  | unseen-writer plan (285) | 0.270 (77) | **0.477** (136) | **+20.7** | >= 0.55: **missed**; >= +10 over S0: met |
  | unseen-writer chain (294) | 0.908 (267) | **0.976** (287) | +6.8 | >= 0.99 (292): **missed** |
  | copied / unknown slots, plan and chain | 0 / 0 | 0 / 0 | | met |
  | val chain / kernel / plan / role-binding | 0.889 / 0.768 / 0.968 / 0.984 | 0.952 / 0.779 / 0.984 / 1.000 | +6.3 / +1.1 / +1.6 / +1.6 | none down: met |

  The script's VERDICT NEITHER reads H-E18's all-family slot clause: 2 copied literals in kernel val, the same count
  on S0; not this order's gate.
- **The 236's greedy step loop** (`validation/logs/exp_0082_step_loop_p32_2026-11-28.json`, grilly2 on the home card,
  the same command as S0's read of 2026-10-01): **0.284 (67 of 236)** against S0's 0.229 (54; the order's fixed number, not re-run): **+5.5**, within the
  gate's 2 points and clear of the kill. Answered 0.945 (S0 0.970): 13 refused (10 steps the VM could not run, 3 past
  the 12-step cap; S0 7). By length: 3 steps 0.42 (S0 0.34), 4: 0.25 (0.19), 5: 0.07 (0.07), 6 and more 0 (0). The
  adapter records no base hash, so the seat loaded it with a warning; the export it ran on (`81a45418...`) is the one
  0077 records as made from `7d4bf9d6...`, P32's base. Cowork's drafting server for 0079 (idle; its batch ended
  before this read began) stayed loaded on the card during the read; the decode is greedy.
- **Where it stands:** below the gate (plan 0.477 and chain 0.976 under their bars), above the kill (plan +20.7, the
  236 +5.5). Between the two: reported, and serving P32 is Nick's call. The gen-3 split (seed 7) and the live eight,
  P32 against S0 with `CB_EMIT_TRIE` off (0081 was killed), wait for a card slot (Nick, 2026-11-28: the 236's loop
  only, for now); S0 is read again there, since the ask path on master has changed since 0081 read it. They decide
  the gate's "0 wrong answers spoken" line; the status stays in progress until they are read.
- **P32 against P8, for what it shows:** P8 (4 times the optimizer steps) read 0.586 plan and 0.990 chain, P32 0.477
  and 0.976; the val families are the same within a point (kernel 0.789 against 0.779). So the extra steps carry
  about 11 points of the unseen-writer plan gain, and the records alone about 21. Nothing is chosen from this.
- **The unseen-writer eval has been read twice for arm P:** once on P8 (Cowork's VM read and the text-exact lines,
  P8's plan 0.561 against S0's 0.249, both seen before Nick chose to train again), and once on P32, this gate read.
  P32's text-exact lines (plan 0.460, chain 0.844, from its generation log) were seen before this VM read.
