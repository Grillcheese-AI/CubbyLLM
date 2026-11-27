Finish a CubbyLLM work order and prepare it for `review`.

Pre-commit checklist:
1. Verify every modified/created file under `cubbyllm/` and `cubby_serve/` is under 300 lines (or document the explicit reason in the commit message) and declares `__wiring__`.
2. Check no scratch files (`validation/logs/_*`, `*.pt`, `*.sqlite`, `*.err`, `*.out`, `*.done`) are staged (`git status`).
3. Run local-path check on all touched files: `python tools/scrub_local_paths.py <touched files>`.
4. Run targeted unit tests for the changed components, plus the guard suite:
   `python -m pytest tests/test_guards.py tests/test_package_rules.py tests/test_local_paths.py`
5. Fill in the order's **Report** section completely (no placeholders; every measured number links its dated `validation/logs/` file; list commits, files, tests, not-verified-here items, found-on-the-way items, and questions).
6. Set the order's `status:` to `review`.
7. Write long commit messages to `validation/logs/_msg_<order>.txt` and commit with `[NNNN] ...` via `git commit -F`.
8. Leave a concise summary note in `docs/inbox.local.md` (rotate resolved notes > 14 days old into `docs/inbox_archive_*.local.md` if `inbox.local.md` exceeds ~20 KB).
