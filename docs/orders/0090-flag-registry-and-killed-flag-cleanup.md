# 0090 — Central CB_* flag registry, killed-flag cleanup, and packaging hygiene
status: proposed
written: 2026-11-27 (from docs/AUDIT_FOR_CLAUDE_2026-11-27.md) · approved: — · hypothesis: H-F1 (anti-sprawl)

## Why
`cubbyllm/` and `cubby_serve/` currently read **51 unique `CB_*` environment flags** via scattered `os.environ.get(...)` calls (215 repo-wide):
- Killed experiments stay wired into hot production code indefinitely (`CB_EMIT_TRIE` from killed `0081` inside `AskLoop._ask`, `CB_ASK_MEMORY`/`CB_ASK_MEMORY_WRITE` from killed `0083`, `CB_SOURCE_NAMESAKES` from under-bar `0043`), adding dead branches and import overhead.
- Env read timing is inconsistent (`hybrid.py` reads `CB_ATTN_TRITON` / `CB_FLEX_OPTS` at module import time; `ask_memory.py` reads `CB_CERT_FACTS` inside a per-check helper).
- Several small code-hygiene defects accumulated alongside this: `cubbyllm/brain.py:48` mutates global `sys.stdout.reconfigure(...)` at import time; `plan_verify.py` uses 11 inline `__import__("re")` calls; `seats_450m.py` and `cubby_serve/{api,get}.py` leave `open(...)` handles without `with`; and `pyproject.toml` still has `dependencies = []` and no `[tool.ruff]` configuration.

Nick ruled on 2026-11-27: **once a flag-gated experiment is killed and its order is `done`, remove its runtime hook from `cubbyllm/`** (git history + the order's commit hash remain the reproducible record).

## Build
1. **Central flag registry (`cubbyllm/core/flags.py`, < 200 lines, `__wiring__ = Wiring.WIRED`):**
   - Declare every active runtime `CB_*` flag in `cubbyllm/` and `cubby_serve/` with `{default, order, state: "experimental"|"graduated", doc}` and typed getters (`flag_bool`, `flag_str`, `flag_int`).
   - Add an AST guard test in `tests/test_guards.py` failing if any module under `cubbyllm/` or `cubby_serve/` reads an unregistered `CB_*` env var directly.
2. **Retire killed flags from `done` orders:**
   - Remove the runtime hooks and dead-flag branches for killed `done` orders:
     - `0081`: `CB_EMIT_TRIE` hook in `cubbyllm/reasoning/ask.py` (keep `plan_trie.py` only if tests/docs require it, else move aside to `_discard/` or downgrade to `STANDALONE` with a clear deprecation note; check `0081`'s report first).
     - `0083`: `CB_ASK_MEMORY` / `CB_ASK_MEMORY_WRITE` hooks in `cubbyllm/reasoning/ask.py` & `ask_recall.py` (confirm `0083` is `done` and neither `0084` nor `0087` needs the hook).
     - Audit any other `done` + killed flags (`CB_SOURCE_NAMESAKES` if `0043` is closed; list any still-blocked/open flags untouched in the Report).
3. **Code hygiene & packaging fixes:**
   - Move `sys.stdout.reconfigure(...)` out of `cubbyllm/brain.py` module top-level into `cubby_serve/api.py::main` and `cubby_serve/__main__.py`.
   - Replace inline `__import__("re")` / `__import__("pathlib")` calls in `cubbyllm/reasoning/{plan_verify,pipeline}.py`, `cubbyllm/sources/wikidata_aliases.py`, and `cubby_serve/get.py` with normal top-level imports.
   - Wrap bare `open(...)` calls in `cubbyllm/emit/seats_450m.py` and `cubby_serve/{api,get}.py` with `with` statements.
   - Update `pyproject.toml`: real description, `[project.dependencies]` (`numpy`), optional extras (`train`, `serve`, `eval`, `dev`), and `[tool.ruff]` config (`target-version = "py312"`, `line-length = 120`, core lint rules).

## Acceptance (pre-registered)
- Every runtime `CB_*` flag in `cubbyllm/` and `cubby_serve/` is registered in `cubbyllm/core/flags.py`, pinned by a guard test.
- Active runtime `CB_*` count in `cubbyllm/` + `cubby_serve/` reported before vs. after, with every removed killed flag citing its `done` order and commit hash.
- `import cubbyllm.brain` does not touch `sys.stdout` (pinned by a unit test using a mock/capture stdout).
- Full unit + guard test suite (`python -m pytest tests`) passes with 0 new failures; default runtime behavior with all flags at default is unchanged.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, no local paths, `import cubbyllm` stays torch- and grilly-free, no permanent deletion).
- Never remove a flag whose order is still `open`, `in progress`, `review`, or `blocked` (e.g., `0084`, `0087`, `0067`, `0068`).
- Run this order **after** `0084` / `0087` land if they touch `ask.py`, or coordinate via inbox so Code A and Code B do not conflict on `ask.py`.

## Out of scope
- `cubby_serve/` multi-user concurrency hardening (`CubbyBrain` lock, CORS, body limits — deferred per Nick's 2026-11-27 ruling: no deployment soon).
- Splitting `plan_verify.py` (950 lines) or redesigning `covers()` around span-cited plans (separate architectural order).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <runtime CB_* count before/after; killed flags retired with order refs; ruff findings count>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
