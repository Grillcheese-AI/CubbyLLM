# 0005 — Train the copy alone on a frozen trunk (run 5)
status: in progress
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
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
