# 0003 — `exp_hypertoken_recall` sets `CB_NO_FLEX=1` on import
status: proposed
written: 2026-10-21 by Cowork (found by Claude Code under 0001) · approved: — · hypothesis: none (hygiene)

## Why
Importing `validation/exp_hypertoken_recall.py` sets `CB_NO_FLEX=1` in `os.environ`. The backbone reads that
flag at every forward, so any in-process caller (a probe during training) silently takes a run off flex
attention mid-training: slower, and a different kernel than the run was measured on. `train_base.fact_probe`
works around it (0001, `c4b11ba`). The fix belongs in the file that causes it.

## Build
- Move the default into `main()` (or the CLI path), so importing the module changes no environment.
- Drop the workaround in `train_base.fact_probe` once the import is clean.
- A test: importing the module leaves `os.environ` unchanged.

- **Also (Cowork, 2026-10-22):** `notebooks/_build_spatial.py`'s readout cell runs its run-1-to-3 comparison block
  (`if link and text:`, H-S12's gates) for runs 4 and 5 too, and crashes after the verdict for `RUN >= 5`, where
  `text` is not the text arm's record. Guard it to `RUN < 4` and regenerate the notebook.

## Acceptance (pre-registered)
- The new test passes. `python validation/exp_hypertoken_recall.py ...` behaves as before from the command
  line (same default). The full `pytest tests` counts are unchanged apart from the new test.

## Constraints
- Rules in `CLAUDE.md` hold. No other change.

## Out of scope
- Any change to how the probe scores.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
