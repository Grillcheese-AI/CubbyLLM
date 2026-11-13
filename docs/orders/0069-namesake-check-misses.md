# 0069 - The namesake check's misses: R2's test wrongs by cause, the "located in" walk as a veto, then one fresh read
status: in progress (Code B, on the branch `agent-b`, 2026-11-13; after 0070 and cubelang's P6-1, P6-2, P6-8, P6-9)
written: 2026-11-12 by Cowork - approved: 2026-11-12 by Nick ("yes", to writing the qFHRR panel's first two steps as orders) - hypothesis: none (0046's R2 lines)

## Why
The qFHRR panel (`docs/research/2026-11-12-qfhrr-panel.md`, eight models, all agreeing) says no score ever chooses
between namesakes: the symbolic check B1q stays the chooser, and its misses are fixed outside the word table. Cowork's
synthesis (`2026-11-12-qfhrr-synthesis.md`) made "B1q repairs" the first candidate.

**Where B1q stands (0046), which the panel's brief did not give:** the brief quoted B1q's first seed-0 run. Since then:
- the rules of `43715a6` (short names as written, the relation reading's words, "located in" walked in the VM) and
  `acc16df` (two names joined by a comparison or a choice are not each other's cue) are in;
- **seed 1, read once, met R2's fame-set lines:** 0 single picks on 1,000 bare names; 0 wrong and 0.762 resolved on
  952 qualified (`retrieval_R2_dev_2026-11-05_seed1.json`). French refuses 80 qualified rows against English's 26;
- **R2's test, read once, failed** (`retrieval_R2_test_2026-11-05_test.json`):
  - Mintaka EN (450 rows): B1q 68 right, 15 wrong, 83 decided, against today's tiers 98, 13 and 111;
  - WebQSP_EL EN (338): B1q 35 right, 6 wrong (1.8%), 41 decided, against tiers 78, 7 and 85;
  - ShadowLink, where the gold bears the mention: Top 6 of 8, Shadow 6 of 6, a 25-point gap on small counts.

So the panel's two concrete fixes ("né", and the Rosebank level of "located in") were already made. What is open is
why B1q is wrong on 15 of 450 Mintaka rows and 6 of 338 WebQSP_EL rows, why it decides half as many rows as the
tiers, and why French refuses three times as often. One point from the panel bears on a rule already in: **Grok's
"located in" as a veto.** The walk lets a chain make a candidate stand, and on dev it made one wrong pick that way
(Mintaka `941b5290` #0: Rhode Island the island, in Contra Costa County, in California). The comparison rule closed
that row, not the mechanism. As a veto, a chain could only turn a pick into undecided, never create one.

- **What it brings:** the namesake check that already never guesses on a bare name stops being wrong on the
  questions that do name a detail, and decides more of them.
- **Over a conventional model:** a model picks the famous bearer of a name. Here a pick needs a detail in the
  question that holds for one bearer, and the misses are found by cause, not patched row by row.
- **In plain words:** when you ask about "Paz the singer", Cubby picks the Paz who is a singer, or asks which one
  you mean, and this order finds out why it still picks wrong on some questions.

## Build
All in the R2 harness (`validation/retrieval_bench/r2_choosing.py`, `r2_b1q.py`, `r2_b1q_prep.py`, `r2_vm.py`).
Nothing on the serving path changes.

**A. The misses, sorted by cause (no rule is written in this step):**
1. The 21 wrong rows of R2's test (Mintaka 15, WebQSP_EL 6) and the 43 of dev (ShadowLink 29, Mintaka 9, WebQSP 5).
   For each: the mention, the gold, the pick, the cue that made the pick stand, and which rule let it stand. Ids and
   short codes only in the log (licences), no question text.
2. The rows tiers decide and B1q does not, on R2's test (Mintaka and WebQSP_EL), by cause: no cue in the question,
   a cue that is not a name in the index, a cue held by several candidates, the comparison rule, a refusal reason.
3. Seed 1's 227 unresolved qualified rows (121 undecided, 106 refused), EN and FR apart, by the same causes. Count
   apart the cues written as adjectives or demonyms ("French" for France), which exact name lookup misses.
4. Reported as one table: cause by set and language, with counts.

**B. The walk as a veto, on dev only** (seed 0, and R2's dev rows of Mintaka, WebQSP_EL and ShadowLink):
- arm `walk` (as built): a candidate holds a place cue if the cue is reached through stored "located in" statements
  within the VM's bound;
- arm `veto`: a candidate holds the cue only if a statement of its own names it. The walk is run on the other
  candidates: if a chain reaches the cue for another candidate, a pick becomes undecided. A chain never makes a pick;
- both with every other rule as frozen.
- Reported per set: right, wrong, undecided, refused, single picks, resolved.

**C. Latency by candidate count.** B1q's p50 and p95 at 2, 6, 10, 20, 50 and 100 candidates on the layered store
(only 2 and 6+ are measured; a fifth to a third of mentions name more than 20 items). If p95 passes 50 ms anywhere:
the check intersects sorted lists (the candidates' ids against each cue value's subjects, in and out) instead of
walking each candidate, with picks identical row for row (pinned by a test).

**D. Rules, then one fresh read.**
- Cowork reads A and B and writes, in this order, which general rules are built: the veto if B's line below says so,
  and any rule that A shows is general (several rows, a cause that is not one set's quirk). 0046's constraint holds:
  no rule is written for one failing row.
- Then one read with every rule frozen, on rows no arm has read:
  - a fresh fame sample (seed 2, the same construction, its SHA-256 committed before any arm runs);
  - fresh Mintaka EN and WebQSP_EL EN test rows (by a new seed, excluding the rows of R2's test read);
  - ShadowLink rows not yet read, if any are left; otherwise reported as not read.

## Acceptance (pre-registered)
- **Tests:** each rule built in D is pinned by a test on made-up items; the veto cannot create a pick (pinned); the
  sorted-list intersection gives B1q's picks row for row (pinned, if built).
- **B, the veto's line (dev):** the veto replaces the walk if it has no more wrong picks than the walk on every set
  and keeps seed 0's qualified resolved at or above 0.70. Otherwise the walk stays and the table is recorded.
- **D, the fresh read, R2's own lines:**
  - fame seed 2: 0 single picks on bare names; 0 wrong and at least 70% resolved on qualified;
  - Mintaka EN and WebQSP_EL EN: no more wrong than today's tiers on the same rows, at most 1 in 50, and more
    decided than tiers;
  - ShadowLink (where the gold bears the mention): Shadow within 5 points of Top.
  - **Kill:** any bare-name single pick, any qualified wrong, or a Shadow-Top gap over 10 points.
- **C:** p95 at most 50 ms at every candidate count up to 100, reported with the machine.
- **Reported:** A's table; B's table; French and English apart everywhere.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim).
- 0046's constraints hold: nothing on the serving path changes; no external model in any arm; nothing is ranked by
  fame within a kind (sitelinks, degree, link counts, description length); no hand rule for a failing row; the sets'
  licences decide what is committed (ids and counts only).
- No score chooses: qFHRR, BM25 or a margin may not decide a pick in any arm of this order (the panel, unanimous).
- Each held-out sample is read once. A rule changed after the fresh read waits for another fresh sample.

## Out of scope
- Wiring B1q into serving: its own order, once its test lines pass.
- qFHRR as a proposer of cue values inside B1q (the synthesis, a later candidate): only if A shows cues that miss
  the name index by wording, and then as its own arm with its own lines.
- The relation step's missing-fact guard: order 0070.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
