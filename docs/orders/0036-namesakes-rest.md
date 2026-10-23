# 0036 — Namesakes, the rest: inverse questions, a chosen item kept as itself, the kind-to-relations table, and an audit of the relation tier
status: in progress
written: 2026-10-23 by Cowork, from 0033's review (fixes toward the first rule; Nick told) · approved: — · hypothesis: none

## Why
Order 0033 stopped two items sharing a label from being merged in an answer (74 of 100 mixed before, 0 after). Three
paths can still mix them, and one decision rule needs an audit:
- inverse questions ("what has X as its capital?") still read every namesake of a shared object label;
- a chosen item's fetched facts go into the wiki world under the bare label, so a later bare question reads that one
  item as if the name were one thing ("What is Ottawa?" already does this);
- the relation tier decides by scanning up to 2,000 members of a kind, which can miss a rare relation;
- the tier's choices (France's capital, Japan's currency) were never audited one by one.
In plain words: when names are shared, Cubby never mixes the things up, whichever way the question is asked.

## Build
1. **Inverse questions:** the same rule as 0033 for objects: a shared object label is decided by the question's own
   context or asked back, never read through every namesake.
2. **A chosen item kept as itself:** facts fetched for a clarify choice are stored under the item (its id), with the
   label as an alias; a later bare question still asks which.
3. **The kind-to-relations table:** built once with the hdc world (which relations each kind's members carry), in place
   of the scan: complete and instant.
4. **The audit:** 50 of the relation tier's decisions, drawn by seed, each with its label, the items and the one
   chosen, read by hand (Cowork reads the list; it is written to the log for that).

## Acceptance (pre-registered)
- An inverse probe like 0033's (50 shared object labels): 0 mixed answers.
- A chosen item, then the bare label asked again: asks which (test).
- The table gives the same decisions as the scan on 0033's probe, and the tier's latency is reported.
- The audit: at least 49 of 50 right; one wrong and the tier goes strict (the asker or a qualifier only).
- 0014's bench: 179 of 179, 0 wrong; the real stack's 20 answers as after 0033.

## Constraints
- Rules in `CLAUDE.md` hold.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
