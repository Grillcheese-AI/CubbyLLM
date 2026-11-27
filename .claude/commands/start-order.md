Start or resume a CubbyLLM work order cleanly.

Steps:
1. Run `python tools/queue.py --for code-a` (or `--for code-b` if in the `agent-b` worktree) to see active, review, and ready-to-claim orders.
2. Read `docs/inbox.local.md` (active notes only; do not load archive files unless a specific older note is referenced).
3. Verify git branch and clean working tree (`git status`, `git branch --show-current`):
   - Code A must be on `master`.
   - Code B must be on `agent-b` (and merge `master` in before starting).
4. Read the target order in `docs/orders/NNNN-*.md` in full, plus only the hypothesis entries it cites: find each in `docs/hypotheses/INDEX.md`, then read it in its group's file.
5. Confirm scope, constraints, flag name (default off), and pre-registered acceptance/kill criteria before writing any code.
6. Set the order's `status:` to `in progress` and commit that status change before building.
