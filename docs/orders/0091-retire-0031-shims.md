# 0091 — Retire the 116 order-0031 shims in standin/
status: proposed
written: 2026-11-27 (from docs/AUDIT_FOR_CLAUDE_2026-11-27.md) · approved: — · hypothesis: H-F1 (anti-sprawl layout)

## Why
Order `0031` (2026-10-22) moved Cubby's core modules out of `standin/` into `cubbyllm/` and `cubby_serve/` and left compatibility shims under `standin/` *"for one cycle"*. Five weeks and 56 orders later, `tools/shim_count.py` still reports:
- **336 in-repo code imports** + **1 notebook import** across **116 shim files** in `standin/`.

Keeping two import paths for the same module causes duck-typing workarounds (e.g. `cubbyllm/brain.py:80`), confuses symbol searches, and bloats `standin/` with 116 obsolete files. Nick confirmed on 2026-11-27 that retiring these shims is safe (moving them aside to `_discard/` or backing them up to another drive rather than permanently deleting them).

## Build
1. **Rewrite all remaining in-repo imports** using `tools/moves_0031.json` and `tools/shim_count.py --json`:
   - Update the 336 Python imports (primarily in `tests/`, `validation/`, `standin/`, and `tools/`) to import directly from `cubbyllm.*` or `cubby_serve.*`.
   - Update the 1 notebook import (editing its `notebooks/_build_*.py` builder and regenerating the `.ipynb` per `CLAUDE.md`).
   - Update `CLAUDE.local.md` locally where it still mentions `standin/books.py` / `standin/data/build_books_index.py`.
2. **Verify `python tools/shim_count.py` reports `0` code imports and `0` notebook imports.**
3. **Move the 116 shim files aside:**
   - Move all 116 shim files out of `standin/` into `_discard/standin_shims_0031/` (git-ignored, preserving directory structure for easy backup to another drive).
   - Remove the shim-allowance clause from `tests/test_package_rules.py::test_standin_holds_only_what_stays_and_the_shims` so no new shims can accumulate under `standin/`.
   - Remove the duck-type shim comment/workaround in `cubbyllm/brain.py:80` if no longer needed (verify with tests).

## Acceptance (pre-registered)
- `python tools/shim_count.py` reports `0 in-repo imports of shimmed names in code, 0 in notebooks (0 shims)`.
- `tests/test_package_rules.py`, `tests/test_guards.py`, `tests/test_local_paths.py`, and the full `python -m pytest tests` suite pass with 0 new failures or skips.
- `standin/` contains only the modules explicitly listed in `test_package_rules.py::STAYS` (plus its own `tests/`).

## Constraints
- Rules in `CLAUDE.md` hold (no permanent deletion: move shims to `_discard/standin_shims_0031/`; notebooks regenerated from `_build_*.py`, never hand-edited; no local paths in committed files).
- Do not change runtime behavior of any moved module; this order is purely mechanical import rewriting and shim retirement.

## Out of scope
- Moving or refactoring the modules in `STAYS` (the GGUF stand-in, LFM source, and Cubby-Man game).
- Changing any external sibling repo (`cubbyverse`, etc.).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <shim_count before (336 code / 1 notebook / 116 shims) vs. after (0 / 0 / 0); full pytest pass count>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
