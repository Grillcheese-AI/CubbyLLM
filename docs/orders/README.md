# Work orders

The queue between **Cowork** (research and dispatch) and **Claude Code** (the coder). One file per
order, `NNNN-slug.md`, numbered in the order written, started from `_TEMPLATE.md`. Who does what
is in `CLAUDE.md`; this file is the format.

## Lifecycle

The second line of every order is `status: <one of these>`:

| status | set by | means |
|---|---|---|
| `proposed` | Cowork | written, waiting for Nick |
| `open` | Nick (or Cowork on his word) | approved; Claude Code may take it |
| `in progress` | Claude Code | claimed (commit the status change, so the other side sees it) |
| `review` | Claude Code | built, tested, pushed, **Report** filled in |
| `done` | Cowork | checked (GPU parts included), numbers recorded where they belong |
| `blocked` | anyone | with the reason and what would unblock it, under **Report** |

Claude Code takes the lowest-numbered `open` order unless Nick names one, and works one order at
a time. Queue: `grep -H "^status:" docs/orders/0*.md`.

## Rules

- **An order is an argument, not a ticket.** If the reasoning turns out wrong while building, stop
  and write why under **Report** (`blocked`); that finding is worth more than the diff.
- **Scope is the order's.** Something else broken or worth doing goes under **Report → Found on the
  way**, not into the same commits. Cowork turns it into its own order.
- **Acceptance is pre-registered.** The numbers and the kill line are in the order before anything
  runs; the report gives the measured values against them, failures included.
- **What this machine cannot check** (CUDA, Triton, flex, Colab-scale runs) is listed under
  **Report → Not verified here**; Cowork checks it on a GPU before `done`.
- **Commits** reference the order (`[0001] ...`); the report lists their hashes.
- Logs a run produces go to `validation/logs/` with the date in the name, and the report links them.
