# 0088 — Repo hygiene, fast accurate guards, and lightweight CI
status: proposed
written: 2026-11-27 (from docs/AUDIT_FOR_CLAUDE_2026-11-27.md) · approved: — · hypothesis: H-F1 (anti-sprawl layout)

## Why
The repository currently tracks **629.75 MB** across 3,866 files, including **464 `_`-prefixed scratch files** inside `validation/logs/` committed before `validation/logs/_*` was added to `.gitignore` — among them 8 PyTorch checkpoints (`validation/logs/_k10_ckpt/*.pt`, ~295 MB), 6 SQLite databases (`validation/logs/_worlds_smoke/**/*.sqlite`, ~67 MB), 75 `.err`, 70 `.out`, 25 `.done`, and 15 `.cmd` files, plus root-level `cubby_mfu_pilot.zip` and `cot_harvest_amb.jsonl`.

Separately, the three guard test files (`test_guards.py`, `test_package_rules.py`, `test_local_paths.py`) take **25.1 s** on every run, and `tests/test_guards.py::_production_importers()` scans `repo.rglob("*.py")` including `validation/`. That both wastes ~2/3 of the AST-parsing time and introduces a false-negative bug: a non-`exp_` helper under `validation/` falsely counts as a production importer of a `Wiring.WIRED` module.

Nick ruled on 2026-11-27: use **safe `git rm --cached`** (no history rewrite / no force-push).

## Build
1. **Untrack scratch & binary clutter (safe `git rm --cached`, files stay on disk / moved to `_discard/` or `$CUBBY_DATA`):**
   - `validation/logs/_*` (464 tracked scratch files, including `_k10_ckpt/*.pt` and `_worlds_smoke/**/*.sqlite`).
   - Any tracked `*.err`, `*.out`, `*.done`, `*.cmd`, `*.u32`, `*.i16` inside `validation/logs/`.
   - Root `cubby_mfu_pilot.zip` and `cot_harvest_amb.jsonl` (verify no doc or test cites them; if cited, record the Drive path or leave an explicit note).
   - Ensure `.gitignore` covers `validation/logs/*.{err,out,done,cmd,pt,sqlite,u32,i16}` and root `/_*.log`.
2. **Add a repo-hygiene guard in `tests/test_guards.py`:**
   - Fail if `git ls-files` contains any path matching `validation/logs/_*`.
   - Fail if `git ls-files` contains forbidden binary/scratch extensions (`.pt`, `.pth`, `.safetensors`, `.gguf`, `.sqlite`, `.db`, `.zip`, `.u32`, `.i16`, `.err`, `.out`, `.done`, `.cmd`) outside an explicit, documented allowlist (if any).
   - Fail if any newly tracked file exceeds 5 MB (explicit allowlist for existing intentional assets such as `data/grillcheese_bbpe128k.json` and the two `.npz` files).
3. **Fix `_production_importers()` in `tests/test_guards.py`:**
   - Restrict scanning to production roots (`cubbyllm/`, `cubby_serve/`, `standin/` excluding `standin/tests/` and `standin/data/`).
   - If fixing the false negative surfaces genuinely unwired `Wiring.WIRED` modules, either downgrade them to `Wiring.STANDALONE` (if experimental/standalone) or record them in **Report → Found on the way** for Nick/Cowork to decide.
4. **Fast local-path guard mode + CI:**
   - Add `CB_FAST_GUARDS=1` support in `tests/test_local_paths.py` to scan only staged/modified files (`git diff --name-only HEAD` + `--cached`) during rapid inner loops; default (`0`) stays a full-repo scan.
   - Add `.github/workflows/ci.yml` running `pytest tests/test_guards.py tests/test_package_rules.py tests/test_local_paths.py` and fast CPU unit tests on push/PR to `master`.

## Acceptance (pre-registered)
- `git ls-files validation/logs/_*` returns 0 files.
- Total tracked working-tree size drops from 629.75 MB to under 250 MB without rewriting git history.
- `tests/test_guards.py`, `tests/test_package_rules.py`, and `tests/test_local_paths.py` all pass, with a planted-failure test verifying the new repo-hygiene guard catches a `_`-prefixed log and a `.pt` file.
- Guard suite wall-clock (`pytest tests/test_guards.py tests/test_package_rules.py tests/test_local_paths.py`) reported before vs. after (target: <= 10 s full scan, <= 2 s with `CB_FAST_GUARDS=1`).
- No dated `.log` / `.json` evidence file cited by `README.md`, `CUBBYLLM_HYPOTHESES.md`, or `docs/orders/` is removed.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, no local paths in committed files, no permanent deletion).
- Do **not** rewrite git history (`git filter-repo` / `BFG` / force-push are explicitly out of scope).
- Keep `validation/logs/*.log` and dated non-`_` `.json` evidence files tracked.

## Out of scope
- Moving multi-MB dated JSON logs (`exp_r18_*.json`, etc.) off git to Drive (separate decision if needed later).
- Rewriting `docs/model-competition-for-cubby-finalconsensus.json`.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <tracked MB before/after; files untracked; guard suite wall-clock before/after; any unwired modules surfaced by fixing `_production_importers`>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
