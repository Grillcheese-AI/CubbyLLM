# 0005 — Train the copy alone on a frozen trunk (run 5)
status: done
written: 2026-10-21 by Cowork · approved: 2026-10-22 by Nick · hypothesis: H-K10

## Why
Run 4 continued the whole 450M with the copy added. Far facts read 0.761 and villages 0.953, but the text mix
rose +0.037: the re-warmed learning rate moved every text source by about the same amount, and the copy touches
none of them (`validation/logs/spatial_run4_link_metrics_2026-10-21.jsonl`). Run 3's trunk with run 4's trained
copy reads **0.844** exact, against 0.738 for run 4 as trained on the same 64 documents
(`validation/diag_swap_copy.py`, `diag_swap_copy_run4_2026-10-21.json`). So the gain is in the copy's 1,027
parameters, and the trunk's drift only cost. Training the copy alone keeps the language bit-identical by
construction. It is also the claim worth having: knowledge added to a trained model through ~1k parameters.

## Build
- `train_base`: `CB_TRAIN_ONLY=copy` (absent = every parameter trains, as now). Only the copy path's
  parameters require grad and enter the optimizer, at `CB_COPY_LR` (its own peak, default 3e-3, same
  schedule shape). The trunk runs forward only: no activations kept for its backward, no update. The
  checkpoint still saves the whole model. Say in the log line that the trunk is frozen.
- `fact_probe`: also report first / later / end per piece, and the copy's mean λ at asking tokens whose
  address holds a written piece against those whose address is empty. This is the split the toy reported, so
  the gap from 0.84 to the read's 0.965 can be located.
- `notebooks/_build_spatial.py`: `RUN = 5`, from `spatial_link_3_final.pt`, with the copy flags as run 4,
  `CB_TRAIN_ONLY=copy`, and only the two shards with asking tokens drawn (`spatial_here`,
  `facts_far1k_lead`), since text gives the copy no gradient. Keep `HOURS_PER_ARM` a placeholder for Cowork,
  and the readout's three lines. Regenerate the notebook.

## Acceptance (pre-registered)
- **Tests:**
  - with `CB_TRAIN_ONLY=copy`, one optimizer step changes only the copy's tensors (every other tensor
    bit-identical);
  - logits at non-asking positions are bit-identical to the starting checkpoint's after training steps;
  - the flag absent leaves training unchanged;
  - the probe's split counts sum to its totals.

  Full `pytest tests` unchanged apart from the new tests.
- **CPU smoke** (no GPU): 20 steps from run 3's checkpoint on a few documents. The copy's tensors move, nothing
  else does, and the step runs faster than a full step. Report its seconds per step.
- **Run 5 (Cowork, on the GPU)**, against run 3:
  - far facts ≥ 0.80 exact on the 400 held-out documents;
  - villages ≥ 0.90;
  - text mix within 0.005 (identical at every non-asking position).

  Kill: far facts below run 4's 0.761.

## Constraints
- Rules in `CLAUDE.md` hold. Flag-gated; files under 300 lines where they are (`train_base.py` already is not:
  say why it grows).

## Out of scope
- Any change to the copy's form, the ring, or the gate. B8 (the split scene). The learned route.

## Report
<!-- Claude Code fills this in, then sets status: review -->
**In one line:** `CB_TRAIN_ONLY=copy` trains the copy's 1,027 parameters on a frozen trunk. On the CPU smoke from run 3,
every other tensor stays bit-identical and a step takes 3.51 s against 12.50 s for a full step. The probe now splits
the pieces and the gate. `RUN = 5` is written.

- **Commits:** `e69fc49` claimed · the commit carrying this report.
- **Files:**
  - New: `validation/recall_split.py` (the probe's scorer and its split), `validation/diag_frozen_check.py` (which
    tensors a run moved, against the checkpoint it started from), `tests/model/test_train_only.py`.
  - Changed: `validation/train_base.py`. It adds `CB_TRAIN_ONLY` / `CB_COPY_LR`, `train_only()` and `params_of()`. The
    checkpoint paths (`payload`, `restore`, `init_from`, `init_copy`) take the full list from `params_of`. The step
    skips the backward of a micro-batch the copy never acted in, and `fact_probe` goes through `recall_split`.
    `notebooks/_build_spatial.py` and `spatial_facts_450m.ipynb` get `RUN = 5`.
  - Why the long files grow: `train_base.py` goes from 1,117 to 1,149 lines, because the flag lives where the optimizer,
    the step and the checkpoint are. The probe's split went to its own file. `_build_spatial.py` goes from 375 to 412,
    because run 5's config, readout and notes sit with the runs they continue.
- **Tests:** `tests/model/test_train_only.py`, 7 tests (8 cases):
  - With the flag, one optimizer step changes only the copy's tensors. Every other tensor is bit-identical, checked
    over the full list.
  - The loss's autograd graph reaches only the copy's tensors, so nothing is kept for the trunk's backward.
  - The checkpoint still holds every tensor and loads back into a frozen model. This test fails on the code before
    the fix (Found on the way, 1).
  - After 3 training steps, every position that does not ask has the starting model's logits exactly, and the asking
    positions changed.
  - Without the flag, every tensor trains: the optimizer groups are as before and the trunk moves.
  - An unknown part, or a model without a copy path, refuses.
  - The probe's first token, exact and question count equal `exp_hypertoken_recall.score`'s on the same forwards.
    Its pieces sum to questions + later pieces, which is the answers' token count. Its λ counts sum to the asking
    tokens the copy acted on: 7 present and 3 empty per document, as built. The test's documents include an asker
    whose address is only written after it (counted empty) and one with no writer yet (not counted).
  - `train_base.fact_probe` carries the split and the mean λ, and leaves `CB_NO_FLEX` as it found it.

  Full `pytest tests`: 414 passed, 5 skipped, 1 failed (the known wiring guard, order 0002). That is the earlier 406
  plus these 8.
- **Measured: the CPU smoke** (`validation/logs/train_only_smoke_2026-10-21.log`,
  `diag_frozen_check_smoke_2026-10-21.json`). Setup: `train_base.py` from `spatial_link_3_final.pt` with run 4's copy
  flags and `CB_TRAIN_ONLY=copy`; 20 steps of 1,024 tokens (S = 1024, micro 1); only the two shards drawn; 6 threads.
  - **Only the copy moved:** 326 tensors and 5 router tensors compared to run 3's, 0 moved. τ 3.040 → 3.072, β 0.115
    → 0.089, gate bias −2.197 → −2.171, gate weight norm 0 → 0.598.
  - **Speed:** 3.51 s a step (median of steps 2–20; range 3.02–4.00) against **12.50 s** for a full step (every tensor
    training, same settings, steps 2–5), 3.6× faster. On the GPU the ratio will differ: the trunk forward still runs,
    and the optimizer step and the activation memory shrink.
  - **Probe after 20 steps, 4 documents a shard** (it shows the split runs; it is not a measurement). Far facts:
    first 0.957, exact 0.522, later 0.880, end 0.957; λ 0.641 at addresses holding a piece (146), 0.442 at empty ones
    (23). Villages: exact 0.973, λ 0.009 / 0.001.
  - Acceptance:
    - Tests: met.
    - CPU smoke: met (the copy's tensors move, nothing else does, and the step is faster).
    - Run 5 is for Cowork: far facts ≥ 0.80, villages ≥ 0.90, text within 0.005, kill below 0.761.
  - The readout's text line: run 5 draws only its two shards, so its `_mix` covers those two and is not comparable to
    run 3's. The readout therefore checks every source without places against run 3's last eval, within 0.005 each.
    That is stricter than a mix. The trunk is frozen and every non-asking position is identical, so expect ~0 there,
    up to GPU numerics.
- **Not verified here:**
  - The frozen trunk under `torch.compile` and bf16 autocast (CPU float32 only).
  - The step's speed and memory on the G4.
  - Whether a larger `CB_MICRO` fits now that no trunk activations are kept (left as run 4's).
  - Run 5 itself.
  - `diag_frozen_check.py` on run 5's final against run 3's, which the notebook's notes ask Cowork to run.
- **Found on the way:**
  1. **`CubbyModel.parameters()` yields only the tensors that require grad** (`assembly.py`). Freezing the trunk
     therefore shrank the list that `train_base` uses as the checkpoint's layout. The first smoke stopped at init
     ("not this model without its copy path"), and a save would have kept only the copy's 4 tensors. My first tests
     missed it because they also listed tensors through `parameters()`. Fixed: `train_only` keeps the full list
     before freezing, and every checkpoint path takes it (`params_of`). Anything else that freezes a `CubbyModel`
     has the same trap.
  2. `fact_probe` reloaded the 128k tokenizer from disk at every probe: `__dict__.setdefault(key, from_file(...))`
     evaluates its default first. It now loads once.
  3. A frozen-trunk micro-batch the copy never acts in (no asking token with a candidate) has no graph, and
     `backward()` would raise. The step skips it; it adds no gradient. This is only with the flag.
- **Questions:**
  1. `CB_COPY_LR` is 3e-3, as the order says. Run 4 trained the copy at 3e-4, its trunk's peak. If Cowork wants run
     5's curve comparable step for step, `CB_COPY_LR=3e-4` reproduces that.
  2. Run 5's `CB_MICRO` is run 4's. With no trunk activations kept, a larger micro, or a shorter accumulation, would
     likely fit. Cowork can measure it on the G4 and set it.

## Review (Cowork, 2026-10-22)
- **Checked.** I read the code (`train_only`, `params_of`, the skipped backward, `recall_split`, the `RUN = 5`
  cells). On Nick's machine, `test_train_only.py` and `test_spatial_copy.py` give 25 passed. The smoke's frozen check
  matches the report: 0 of 331 tensors moved. Found 1 (`parameters()` lists only tensors that require grad) is a
  good catch.
- **Answers.**
  1. Keep `CB_COPY_LR = 3e-3`. Run 4's copy moved little at 3e-4, and the smoke at 3e-3 moved it as far in 20
     steps. Run 5 is not meant to match run 4 step for step.
  2. Keep run 4's `CB_MICRO`; speed is not the limit here.
- **Budget:** `HOURS_PER_ARM = 0.15` for run 5. Every step now draws only the two asking shards, about 320
  far-fact questions a step, against run 4's ~30. So 0.15 h is many epochs of the 20,000 far-fact documents, for
  1,027 parameters. The probe every eval shows the curve.
- **Still open, for `done`:**
  - run 5 on the G4 against the readout's lines: far facts ≥ 0.80 exact, villages ≥ 0.90, every text source
    within 0.005 of run 3, kill below 0.761;
  - `diag_frozen_check.py` on run 5's final against run 3's;
  - compile and bf16 with the frozen trunk.
- **Run 5 (2026-10-21, G4, 0.17 h) and `done`.** Far facts **0.946** exact (the line is 0.80; the kill line was 0.761),
  villages 0.953, every text source within 0.005 of run 3 (worst 0.0000). `diag_frozen_check` finds 0 trunk or
  router tensors moved. The GPU checks: compile and bf16 ran with the frozen trunk at 183k tok/s; the probe ~1 min
  an eval.
- **Found by Cowork:** the readout cell crashes *after* its verdict lines for `RUN >= 5`. Its run-3-era block
  `if link and text:` runs and indexes `text` (a list there) by name. This is cosmetic, since the verdict is already
  printed, but it is folded into order 0003 to fix.
