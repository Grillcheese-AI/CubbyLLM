# 0089 — Slim CLAUDE.md, shard CUBBYLLM_HYPOTHESES.md, and triage the order queue
status: in progress (Cowork, 2026-11-27, on Nick's "cleanup the context rot": steps 1 and 2 done, step 3 in part; the order template and README, `tests/test_queue.py` and step 4's triage are left)
written: 2026-11-27 (from docs/AUDIT_FOR_CLAUDE_2026-11-27.md) · approved: 2026-11-27 by Nick ("cleanup the context rot") · hypothesis: none (workflow efficiency)

## Why
Every fresh Claude Code and Cowork session currently pays heavy context overhead and risks acting on stale state:
- `CLAUDE.md` carries hand-maintained snapshots from 2026-10-21 / 2026-10-31 (missing orders `0048`–`0087`) and still lists a "Known red" guard test that passes (`26 passed`).
- `CUBBYLLM_HYPOTHESES.md` has grown to **526 KB** (~130k tokens) in one file with single paragraphs exceeding 20,000 characters, causing line-based reads to truncate mid-paragraph.
- `TODO.md` (67 KB, single bullets >15,000 chars) is stale from September 2026 and superseded by `docs/orders/`.
- `Architecture.md` and `cubbyllm/core/config.py` still describe July 2026 toy `MinGRUBackbone` as the open decision, omitting `HybridBackbone`, `base450m`, and `spatial.py`.
- 13 of 85 orders in `docs/orders/` are simultaneously marked `in progress`, violating `docs/orders/README.md`'s one-at-a-time rule.

`docs/inbox.local.md` was already rotated (526 KB → 13 KB active) and `tools/queue.py` was added on 2026-11-27.

## Build
1. **Slim `CLAUDE.md` (< 130 lines):**
   - Keep only timeless roles, rules, invariants, boundaries, and commands.
   - Remove the hand-maintained "Where things stand", "Queue", and stale "Known red" sections; replace with `python tools/queue.py` and `docs/RESULTS.md`.
   - Document the new killed-flag retirement rule (Nick, 2026-11-27): once a flag-gated experiment is killed and its order is `done`, remove its runtime hook from `cubbyllm/`.
   - Document the inbox rotation rule (keep `docs/inbox.local.md` under ~20 KB; archive older notes to `docs/inbox_archive_<dates>.local.md`).
2. **Shard `CUBBYLLM_HYPOTHESES.md` without losing a word:**
   - Split into `docs/hypotheses/INDEX.md` (< 15 KB scorecard table: `ID | 1-line claim | Status | Latest order/log | File`) and per-group files `docs/hypotheses/{0_context_rules,group_A,group_B,group_C,group_D,group_E,group_F,group_G,group_K,group_P}.md` with normal line breaks.
   - Replace `CUBBYLLM_HYPOTHESES.md` at repo root with a short pointer + top-level index linking every group file (or keep a small script `tools/check_hypotheses_index.py` verifying zero broken links and zero dropped `H-*` IDs).
3. **Archive `TODO.md` and refresh stale core docs:**
   - Move `TODO.md` → `docs/archive/TODO_2026-09.md` (leave a 5-line redirect note at `TODO.md` or remove root reference from `CLAUDE.md`).
   - Update `Architecture.md` header/status note (pointing to `HybridBackbone`, `base450m`, and `spatial.py`) and fix the stale `"H-D1 still open"` comment in `cubbyllm/core/config.py`.
   - Update `docs/orders/_TEMPLATE.md` and `docs/orders/README.md` to support structured frontmatter (`status`, `assignee`, `depends_on`, `updated`) and explicit `waiting_gpu` / `waiting_nick` / `parked` states.
4. **Queue triage (Cowork + Nick sign-off):**
   - Review the 13 `in progress` orders (`0024`, `0027`, `0029`, `0049`, `0052`, `0053`, `0054`, `0055`, `0067`, `0068`, `0080`, `0082`, `0085`) and `0048` (stuck in `review`) plus `0002` (obsolete `proposed`), setting each to its honest state so at most 1 order per worker is `in_progress`.

## Acceptance (pre-registered)
- `CLAUDE.md` is <= 130 lines and contains zero dated queue lists or false "Known red" claims.
- Every `H-*` ID present in the original `CUBBYLLM_HYPOTHESES.md` appears verbatim in `docs/hypotheses/` and is listed in `docs/hypotheses/INDEX.md` (verified by a unit test or script).
- `python tools/queue.py` runs cleanly with zero WIP warnings after triage.
- `tests/test_guards.py`, `tests/test_package_rules.py`, `tests/test_local_paths.py`, and `tests/test_queue.py` (new unit test for `tools/queue.py`) all pass.

## Constraints
- Rules in `CLAUDE.md` hold (no local paths in committed files, no permanent deletion).
- **Zero content loss** when sharding `CUBBYLLM_HYPOTHESES.md`: preserve every measurement, log link, and kill verdict.
- Do not change any hypothesis status while sharding.

## Out of scope
- Editing local git-ignored files (`CLAUDE.local.md`, `.claude/settings.local.json`) in git commits (update those locally alongside this order).
- Refactoring `cubbyllm/` runtime code (that is orders `0090` and `0091`).

## Report
- **Cowork, 2026-11-27 (steps 1 and 2, step 3 in part):**
  - **`CLAUDE.md`:** 191 lines to 130, with no dated status, queue or "Known red" (the guard suite passes). It points
    to `tools/queue.py`, `docs/RESULTS.md`, `docs/hypotheses/INDEX.md` and cubby-stack's preset for what is served,
    and carries Nick's two new rules (a killed flag leaves the code; the inbox stays under ~20 KB). The old file is
    `docs/archive/CLAUDE_2026-11-27.md`, word for word.
  - **The hypothesis document:** 520,784 characters in one file, its longest line 37,747 characters, now
    `CUBBYLLM_HYPOTHESES.md` (the title, the preface, section 0 and a map: 6 KB) and 12 parts in `docs/hypotheses/`,
    split by section, lines wrapped at 120 characters (tables and headings left as they were). Checked by the
    splitting script before anything was written: each part equals its section with only whitespace changed, the
    parts together equal the original, and all 98 `H-` identifiers are kept. `docs/hypotheses/INDEX.md` lists the 98
    with the status each entry's own text last states (some entries state none; they show a dash). No status
    changed.
  - **`TODO.md`:** moved to `docs/archive/TODO_2026-09.md`; a three-line note is left where it was.
  - **`Architecture.md`:** a note at the top on what the trunk is now; `cubbyllm/core/config.py`'s docstring no
    longer says H-D1 is open (a docstring change only).
  - **`.claude/commands/start-order.md`:** reads the index, then the group's file.
  - **Local, not committed:** Code B's inbox rotated as Code A's was; `CLAUDE.local.md`'s pre-0031 paths fixed.
- **Left:** the order template and `docs/orders/README.md` (the frontmatter and the waiting states),
  `tests/test_queue.py`, and step 4 (the triage, which needs Nick's sign-off).
