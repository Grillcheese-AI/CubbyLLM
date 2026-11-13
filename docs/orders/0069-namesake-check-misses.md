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

## Cowork's step D (2026-11-13)
- **1. The veto replaces the walk.** B met the pre-registered line (no more wrong on every set; seed 0's qualified
  resolved 0.7330, at least 0.70), and the line says the veto then replaces the walk. It is recorded that on dev it
  removed no wrong pick and cost 46 rights (0.7835 to 0.7330): it is kept for what it rules out (a chain can never make
  a pick), and the fresh read will say what it costs there.
- **2. A namesake cue never qualifies a pick on its own.** A's most frequent cause, across Mintaka, WebQSP_EL and
  ShadowLink (29 of 38 test wrongs, 32 of 43 dev): the qualifier's words, looked up as a name, give several items, and
  the pick holds one of them. Rule: a cue span whose words name more than one item in the index cannot make a candidate
  stand; where it was the cue that held, the pick becomes undecided. B1q never chooses among the cue's items. Pinned on
  made-up items.
- **3. An article that belongs to a name is kept.** The cue rule tries the span with its leading article first ("La
  Jolla", "Le Havre", "The Hague") and strips it only when the name with the article is not in the index. A general
  bug in both languages (3 rows today). Pinned on made-up items.
- **Not built here:** the gold not bearing the name (18 / 31: candidate recall, not B1q's choice); a cue naming the
  qualifier that no candidate holds (24: missing statements); **French labels missing from the name index** (64 of
  seed 1's FR rows): a data change to the index that every arm reads, so its own order, proposed to Nick.
- **Before the fresh read, on dev only, with 1 in place:** rules 2 and 3 each measured (right, wrong, undecided,
  refused per set, FR and EN apart). **Kept** only if, with the rule, no set has more wrong picks and seed 0's qualified
  resolved stays at or above 0.70; otherwise not built, and the table is recorded. Written now, before the run.
- **Then C** in its slot (it measures B1q as frozen after this step), **then the fresh read** exactly as the order
  writes it, every rule frozen, each sample's SHA-256 committed before any arm runs.

### Step D's dev runs (Claude Code (B), 2026-11-13)
Logs: `retrieval_R2_dev_2026-11-13_0069_veto.json` (rule 1 alone), `..._0069_rule2.json` (`--b1q-rules namesake_cue`),
`..._0069_rule3.json` (`--b1q-rules article`); the rebuilt cowork VM (`b3b3360` for rules 2 and 3, `676df1d` for the veto run:
the same verdicts, P6-10 changed only speed), every other rule frozen. Right / wrong / undecided / refused (resolved). The
entity-linking sets' dev rows are English only; the fame set is English and French apart.

| B1q, dev, with the veto | veto alone | rule 2 | rule 3 |
|---|---|---|---|
| fame/qualified | 667 / 0 / 97 / 146 (0.7330) | 182 / 0 / 582 / 146 (0.2000) | 640 / 0 / 90 / 180 (0.7033) |
| fame/en | 373 / 0 / 52 / 41 (0.3861) | 94 / 0 / 331 / 41 (0.0973) | 345 / 0 / 45 / 76 (0.3571) |
| fame/fr | 294 / 0 / 45 / 105 (0.3114) | 88 / 0 / 251 / 105 (0.0932) | 295 / 0 / 45 / 104 (0.3125) |
| fame/bare | 998 held, 2 single | 998 held, 2 single | 998 held, 2 single |
| mintaka | 55 / 9 / 325 / 44 (0.1270) | 15 / 2 / 372 / 44 (0.0346) | 56 / 10 / 323 / 44 (0.1293) |
| webqsp_el | 26 / 5 / 282 / 10 (0.0805) | 4 / 0 / 309 / 10 (0.0124) | 29 / 5 / 279 / 10 (0.0898) |
| shadowlink/Top | 8 / 15 / 76 / 4 (0.0777) | 5 / 5 / 89 / 4 (0.0485) | 8 / 15 / 76 / 4 (0.0777) |
| shadowlink/Shadow | 13 / 14 / 73 / 8 (0.1204) | 4 / 8 / 88 / 8 (0.0370) | 13 / 14 / 73 / 8 (0.1204) |
| shadowlink/Tail | 33 / 0 / 52 / 3 (0.3750) | 11 / 0 / 74 / 3 (0.1250) | 33 / 0 / 52 / 3 (0.3750) |
| mintaka/en | 55 / 9 / 325 / 44 (0.1270) | 15 / 2 / 372 / 44 (0.0346) | 56 / 10 / 323 / 44 (0.1293) |
| webqsp_el/en | 26 / 5 / 282 / 10 (0.0805) | 4 / 0 / 309 / 10 (0.0124) | 29 / 5 / 279 / 10 (0.0898) |
| shadowlink/en | 54 / 29 / 201 / 15 (0.1806) | 20 / 13 / 251 / 15 (0.0669) | 54 / 29 / 201 / 15 (0.1806) |

- **Rule 2: not built** (Cowork, 2026-11-13): seed 0's qualified resolved 0.2000 against the 0.70 line, though it cuts
  wrongs (Mintaka 9 -> 2, WebQSP_EL 5 -> 0, ShadowLink Top 15 -> 5, Shadow 14 -> 8).
- **Rule 3: not built** (Cowork, 2026-11-13): Mintaka wrong 9 -> 10 (qualified resolved 0.7033).
- **Rule 2b** (Cowork, 2026-11-13, written before it ran): a namesake cue qualifies a candidate only when the candidate
  holds the cue's item through instance of, subclass of or occupation (`--b1q-rules namesake_cue_class`; the facts
  `<item> ~class <cue>` in the segment). Measured next, with the veto; the same keep line.

## Report
<!-- Claude Code fills this in, then sets status: review -->
**Where it stands (2026-11-13):** A and B done and committed; C waits for cubelang P6-10 on the shared VM (the P6 build
was 1.5-3.5x slower; P6-10 is in review); D is Cowork's (the rules), then one fresh read.

- **Commits** (on `agent-b`): `cca4879` (in progress), `a08bfe1` (B), `9260781` (A).
- **Files:** `validation/retrieval_bench/r2_misses.py` (each set rebuilt as R2 builds it; B1q's decision repeated from
  the segment's facts, with why each candidate stands), `r2_misses_report.py` (A's tables), `r2_b1q.py` (`WALK`:
  "walk" as built or "veto"; the walk factored out), `r2_choosing.py` (`--b1q-walk`, recorded in the log's args);
  `tests/validation/test_r2_arms.py` (the veto, pinned). Logs: `retrieval_R2_misses_2026-11-13.json`,
  `retrieval_R2_dev_2026-11-13_0069_veto.json`.
- **Tests:** the veto never makes a pick, a chain for another candidate turns a pick into undecided, the walk unchanged
  under "walk" (16 pass in the file).

**A. The misses by cause** (`retrieval_R2_misses_2026-11-13.json`). B1q repeated from the facts equals R2's logs on
every row before any cause is counted: dev 2,965 / 2,965, test 1,088 / 1,088, seed 1 1,952 / 1,952. A diagnostic of
reads already made; nothing is scored anew; ids and codes only.
1. **Wrong picks.** None stood by the "located in" walk, on test or dev.

   | | test (38) | of which Mintaka (15) | WebQSP_EL (6) | ShadowLink (17) | dev (43) |
   |---|---|---|---|---|---|
   | the cue that held names several items | 29 | 13 | 3 | 13 | 32 |
   | the gold does not bear the name | 18 | 2 | 1 | 15 | 31 |
   | neither (a plain wrong pick) | 5 | 2 | 3 | 0 | 1 |
   | the pick stood by its own statement / a statement pointing at it | 32 / 6 | 15 / 0 | 5 / 1 | 12 / 5 | 30 / 13 |

   The two causes overlap (a row can have both). A cue "names several items" when the cue's words, looked up as a
   name, give more than one item and the pick holds one of them: the qualifier is itself a namesake.
2. **Rows the tiers decide and B1q does not** (test, Mintaka and WebQSP_EL: 161): the qualifier held by no candidate
   157 (Mintaka 83, WebQSP_EL 74), no cue span 3, held by several 1. Of the 157: the gold bears the name and every cue
   names an item the gold has no statement with, either way, 146 (the question's other entities, not a detail of the
   gold); the gold does not bear the name 10; a cue names the gold itself 1.
3. **Seed 1's 227 unresolved qualified rows** (EN 89, FR 138): held by several 117 (EN 63, FR 54); held by none 105
   (EN 26, FR 79); no qualifier 4 (FR: the comparison rule took the name 2, no span names an item 2); the relation not
   verified 1 (FR). Where no candidate holds it, checked against the qualifier's own item (the fame row's
   discriminator):
   - **the qualifier's name is a span of the question and the name index lacks it:** FR 64, EN 9 (e.g. "comte de X",
     "personnalite politique", "Azerbaidjan occidental"): **this is most of French's extra refusals**;
   - a cue names the qualifier but no candidate holds it: FR 12, EN 12;
   - the name starts with an article the cue rule strips ("La Jolla" read as "jolla"): FR 2, EN 1;
   - the name is not a span of the question: FR 5, EN 4.
   Demonyms (P1549 of the candidates' countries, 378 known): no unresolved row carries one.

**B. The walk as a veto, on dev** (`retrieval_R2_dev_2026-11-13_0069_veto.json`, against the walk on the same rows
and build, `retrieval_R2_dev_2026-11-13_b1q_p6final.json`; the rebuilt cowork VM `676df1d`, every other rule frozen):

| B1q, dev | walk: right / wrong / undecided / refused | veto: right / wrong / undecided / refused |
|---|---|---|
| fame qualified | 713 / 0 / 103 / 94 (resolved 0.7835) | 667 / 0 / 97 / 146 (resolved 0.7330) |
| fame bare | 998 held, 2 single | 998 held, 2 single |
| Mintaka | 55 / 9 / 325 / 44 | the same |
| WebQSP_EL | 26 / 5 / 282 / 10 | the same |
| ShadowLink Top | 9 / 15 / 75 / 4 | 8 / 15 / 76 / 4 |
| ShadowLink Shadow / Tail | 13 / 14 / 73 / 8; 33 / 0 / 52 / 3 | the same |

- **The line** (no more wrong picks than the walk on every set, seed 0's qualified resolved at or above 0.70):
  **met** (0.7330). On dev the veto removes no wrong pick (A: none stood by the walk); 53 rows change, each a right or
  an undecided turned refused or undecided (46 fame qualified rights, 1 ShadowLink Top right).
- The 2 bare single picks are the shared-today line's (as on every run since 0046).

**C, D:** not yet (C after P6-10 on the shared VM, in a quiet slot; D after Cowork's rules).

- **Not verified here:** —
- **Found on the way:**
  - The harness did not record its arms' mode; `--b1q-walk` now goes into the log's `args` (the veto run's log is
    annotated by hand: it was started with the flag, and 53 rows differ from the walk's).
  - A commit-message slip: `9260781` says "FR 64 of 79"; the base is 83 (79 held by none plus 4 with no qualifier).
- **Questions:** for D: (1) the name index lacking French labels is a data cause, not a rule: is filling it (R1b's
  segment, French labels of items already in it) in this order or its own? (2) "a cue that names several items" is
  the most frequent wrong-pick cause on test (29 of 38): is "a namesake cue never qualifies" a rule you want written?
