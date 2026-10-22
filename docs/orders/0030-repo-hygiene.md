# 0030 — Repo hygiene: the red tests made true, local paths out of committed files, the largest files split
status: in progress
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes") · hypothesis: none

## Why
Reports for 0015 and 0027 found what `CLAUDE.md`'s "known red" does not say: `pytest standin/tests` has 25 red tests
that predate those orders (`test_build_chat_sft.py` 24 and `test_chat.py` 1, on names `build_chat_sft.py` no longer
has since 34e7658), and `test_goemotions_cube.py` does not import (`GOEMOTIONS`, the same file, since 2026-09-25).
Older committed files still carry local paths (the hypotheses file, `data/corpus_sources.json`, some top-level docs),
against the rule. And `serve_api.py` (559 lines) and `emitter.py` (750) keep growing past the 300-line rule.

What it brings: a red test means something again, and the public repo carries no machine's folders.
In plain words: the alarms only ring for real problems, and nothing private leaks through old files.

## Build
- **The red tests:** fix each test or the code it tests; where the code's change was intended, update the test; a
  test that cannot be fixed now is marked `xfail` with its reason and an order number. `CLAUDE.md`'s known-red line
  lists exactly what is red afterwards.
- **Local paths:** replace them in committed files with what they are ("the local books folder", an environment
  name). A guard test scans committed text files for drive-letter paths and user folders and fails on one.
- **The largest files:** split `serve_api.py` and `emitter.py` by responsibility (routes, the board and plugin
  dispatch; the trunk, its roles, the emitters), no behaviour change, the existing tests pinning it.

## Acceptance (pre-registered)
- `pytest tests standin/tests`: red only what `CLAUDE.md` names (today: 0002's wiring test).
- The guard finds no local path in any committed text file (and fails on a planted one in its own test).
- `serve_api.py` and `emitter.py` each under 300 lines after the split; every test that passed before passes after.

## Constraints
- Rules in `CLAUDE.md` hold. No history rewrite: old commits keep what they had.

## Out of scope
- Order 0002 (the wiring guard) stays its own order.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
