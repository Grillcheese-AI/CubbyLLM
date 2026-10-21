# 0002 — The wiring guard is red: two modules without `__wiring__`
status: proposed
written: 2026-10-21 by Cowork · approved: — · hypothesis: none (hygiene)

## Why
`tests/test_guards.py::test_every_module_declares_wiring` fails on `cubbyllm.spatial` (the package
`__init__`) and `cubbyllm.reasoning.step_loop` — on Nick's machine and in a clean clone (seen
2026-10-21). A red guard hides the next real failure in the same test. It is also a small first
order to try the loop end to end.

## Build
- Add `__wiring__` to both, with the right intent: check whether production code imports each
  (`test_wired_modules_have_a_production_importer` enforces `WIRED` = something calls it); a
  package root is an entry point; `STANDALONE` otherwise, with one line saying so.
- No other change.

## Acceptance (pre-registered)
- `python -m pytest tests/test_guards.py` passes; the full `pytest tests` result is unchanged apart
  from that (report both counts).

## Constraints
- Rules in `CLAUDE.md` hold. Do not add anything to `KNOWN_UNWIRED` to silence the second guard.

## Out of scope
- Wiring either module into a default path (if one should be, say so under Found on the way).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
