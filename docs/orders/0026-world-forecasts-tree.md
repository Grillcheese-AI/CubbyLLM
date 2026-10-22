# 0026 — Forecasts with a track record, and a branching tree: what follows if branch A, B or C, or a choice, is the new reality
status: proposed
written: 2026-10-22 by Cowork · approved: — · hypothesis: H-K12 (proposed), with H-P1-H-P5 (forkable state,
counterfactual re-entry), H-E17 (possibility worlds) and MoWM · **after 0025**

## Why
Nick (2026-10-22): predict the state of the world from current and past events, "with branching so we can see also
what might happen if branch A, B, C, D, ... is the new reality, or if a specific choice is made in the reality branch:
a tree". The pieces exist:
- **Forkable state** (H-P1-H-P5, verified at toy scale and on the hybrid): a decode state branches exactly, in O(1),
  isolated; counterfactual re-entry from a choice point works on a world the model fully knows.
- **Possibility worlds** (H-E17, every gate passed at n = 1,500): "which world, and is it possible there?" asked of
  every claim.
- **A decision tree over futures** in the CubeMind work (`decision_tree.py`, `decision_oracle.py`: an immutable history
  of nodes, each holding candidate futures, the user choosing one to branch from). Port the ideas, not the code.
- **Signals and their change points** (order 0025), **dated events** (0024), **ground truth** (UCDP).

What is missing is honesty about prediction. A forecast is not a fact; the only evidence that it is worth anything
is its record against outcomes. So forecasts here are probabilities scored against what then happened, and a branch
is a scenario: "if this, then ...", never a claim about what will be.

- **What it brings:** ask "what happens next between X and Y?" and Cubby gives probabilities it has earned on past
  years, with the analogues behind them; ask "and if talks collapse?" or "and if X decides Y?" and it opens that
  branch, recomputes, and lets you branch again, with every assumption on the path written out.
- **Advantage over conventional models:** a chat model narrates one plausible future with confidence. Here each
  number has a measured track record, each scenario is a separate branch you can compare side by side, and nothing
  is spoken as a forecast until its calibration has been tested on years it never saw.
- **In plain words:** Cubby can sketch where things are heading, with honest odds, and play out "what if" paths
  side by side, always saying which part is assumed.

## Build
- **Questions with outcomes:** a fixed set, each resolvable from data: organized violence above UCDP's threshold in a
  country next month; a tension shift (0025) between a pair next month; a GDACS orange or red alert in a region next
  30 days; and the like. History: GDELT and UCDP from 2015.
- **The forecaster:** base rates (each question's frequency, per country) adjusted by current signals and by
  **analogues** (past situations whose signal trajectory resembles today's, and what followed). Probabilities only,
  with the analogues kept as evidence. No external model.
- **The tree** (`standin/world_tree.py`): a node is a world state at a date: the signals, the held facts as of that
  date and the open questions. A **branch** is an assumption added to its parent: an event ("the ceasefire holds",
  "talks collapse") or a choice ("X decides Y"). Each branch is its own possibility world (H-E17): the forecaster runs
  again inside it, conditioned on the assumption (its analogues are the past cases where that event or choice
  happened), and the branch can branch again. Nodes are immutable; a node shows its whole assumption path. Where the
  model gives the branch event a probability, the branch carries it; a chosen branch (a "what if") carries none.
- **Replies:** forecasts as probabilities with their measured record ("on past years, forecasts like this one were
  right about N in 10"); scenarios as "if ..., then ...", never as what will happen; the audit view shows the
  analogues and the events.

## Acceptance (pre-registered)
- **Calibration, held out:** forecasts made for every month of 2024-2025 using only data available before each month
  (no look-ahead; the code and a test pin it), against the outcomes. Brier skill score against the base rate per
  question type, and a reliability curve. **Gate: BSS >= 0.05 on the held-out months for a question type before any
  forecast of that type is spoken. Kill: BSS <= 0, then that type is never spoken.**
- **Branches help when they are true:** for held-out months where a branch event then happened, the forecast inside
  the matching branch scores better than the unconditioned one (BSS of branch over root > 0). If not, branches are
  shown as "assumed" without their own numbers.
- **The tree:** a fork leaves its parent unchanged (a test); a node's assumption path is complete; 3 nested branch
  levels on a live question in under a minute on Nick's machine.
- **Replies:** 0 forecasts spoken as facts; every scenario says it is one.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model in the forecaster or the tree. No look-ahead anywhere in the backtest.
- Forecasts about conflict concern real people: internal until the gate is passed for that type, and never presented
  as certain.

## Out of scope
- Markets and prices. Forecasts about individuals.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
