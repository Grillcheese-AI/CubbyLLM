# 0026 — Forecasts with a track record, and a branching tree: what follows if branch A, B or C, or a choice, is the new reality
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes") · hypothesis: H-K12 (proposed), with H-P1-H-P5 (forkable state,
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

## Cowork's addendum (2026-11-06): butterfly reach as dominance over recorded causes
Nick brought a guide to "butterfly effect" analysis (2026-11-06). Its method links events by keyword overlap, time and
place under a 0.3 threshold, and treats the best-connected events as the likeliest causes. That finds correlations,
not causes, and its "most influential" signal is fame by another name. It also starts from the model-written causal
fields that `build_history_records.py` leaves out on purpose. So it is not used. Two things from it are kept, done
exactly (Nick: "yes").

**1. What X's absence takes with it** (an operation the branching tree calls):
- **The graph:** the history graph's recorded causal links only. "Before" and "after" links are not causes.
- **The operation:** a super-root linked to every event with no recorded cause, then the dominator tree (Lengauer and
  Tarjan). X dominates Y when every recorded path of causes into Y passes through X.
- **The answer to "what if X had not happened?":**
  - the events X dominates: they lose every recorded cause;
  - the events with X among their causes but not dominated: they keep another cause;
  - "nothing recorded downstream of X" when that is the case.

  Every answer is worded as "per the recorded links", with the paths as its certificate.
- **Cycles in the recorded links** are reported, and that component is refused, never broken by guess.

**2. The model-written causes as a work queue, never as facts:**
- each `causal_link`, `precursor_events` and `downstream_influence` entry of the model-written records becomes a
  candidate link with the status "hypothesis";
- it enters the graph only when two independent sources state it (the web's two-site rule; Wikipedia and DBpedia
  are not two);
- it is never spoken before that.

**Acceptance (added before anything is built):**
- **Tests:**
  - synthetic graphs with known dominators;
  - an event with two independent recorded causes is never listed as lost;
  - a cycle is reported and refused;
  - the same graph gives the same answer twice.
- **Gate:** the existing counterfactual set (reworded questions downstream of X, 18% spoken and correct today) must
  reach spoken-and-correct at least 35% with spoken-and-wrong 0, as pre-registered in `docs/panel_compiler.md`.
- **Reported:**
  - the share of events with at least one recorded cause, which bounds what this can answer;
  - for the queue, the number of candidates, and the confirmation rate on a seeded sample of 200.

## Cowork's note (2026-11-11): a what-if as a fork of the event log, from the field watch, proposed
Nick: "there's a lot more we can borrow in it". **Proposed, for Nick's yes; not built, not measured.** As the
field watch read it.
- **Deterministic Event-Graph Substrates as World Models** (Tesseract, May 2026): an append-only event log; a
  what-if is the log forked at a tick; "why" and "what if" are one walk over an event's causal ancestors. No
  part is learned.
- **For this order:** a forecast's what-if branch is a fork of the recorded log, so each branch replays, and the
  "why" of an outcome is a list of recorded events.
- **What it brings, in plain words:** Cubby can answer "what if this had not happened?" by replaying from that
  point, and show the events that led to the result.
- **Over a conventional model:** a model imagines the alternative; nothing can be replayed or checked.
