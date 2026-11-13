# 0070 - The missing-fact guard: the relation step refuses when the item lacks the fact asked, measured on hidden facts
status: open (queue: after 0054's seven name-statement items, beside 0067; Nick sets the order between them)
written: 2026-11-12 by Cowork - approved: 2026-11-12 by Nick ("yes", to writing the qFHRR panel's first two steps as orders) - hypothesis: none (0046's R3 and R8 lines)

## Why
In 0046's R3o (`retrieval_R3o_dev_2026-11-04.json`), every relation arm chooses among the properties the item
carries. On 38 of 908 scored rows the item does not carry the gold property today. There, the arms hand over a near
property the item does carry:

| arm | handovers on the 38 |
|---|---|
| qFHRR, margin 0 | 36 |
| qFHRR, margin 0.25 | 16 |
| the 450M with the Wikidata-label adapter (0048) | 16 |
| lexicon_content | 6 |
| the 450M, serving adapter | 4 |
| grammar | 0 |

The asked-kind filter (R3k) turns dropped golds into more such rows: qFHRR's handovers rise to 83. A handed-over
property is on record, so the VM checks the hop and certifies a wrong answer: Cubby answers a different question than
the one asked. All eight members of the qFHRR panel (`docs/research/2026-11-12-qfhrr-panel.md`) call it the worst
failure in the brief; Cowork's synthesis makes the guard the second candidate. Two problems stand in the way:
- **No arm can say "none of these":** a ranker over the item's own properties always ranks.
- **38 rows, all dev, decide nothing.** The panel's answer (Fable, Gemini): hide the gold on rows where the item
  holds it and require a refusal, which gives hundreds of missing-fact rows.

- **What it brings:** when the store does not have the fact asked, Cubby says so instead of answering a near
  question with a checked fact.
- **Over a conventional model:** a model fills the gap with the nearest plausible answer. Here a missing fact is a
  refusal the record can show.
- **In plain words:** ask where someone died when Cubby only knows where they were born, and it says it doesn't know
  where they died, instead of naming their birthplace.

## Build
In the R3 harness (`validation/retrieval_bench/r3o.py`, `r3o_props.py`, `r3_arms.py`, `r3_qfhrr*.py`), with the
guard and the set in new files beside them. Nothing on the serving path changes.

**A. The hidden-fact set:**
- From R3o's dev rows where the item carries the gold (870 rows over the five sets), each row is copied with the gold
  property removed from the item's candidate list (the property stays in the whole codebook; only this item no
  longer carries it). A row whose gold has an inverse the item also carries drops both.
- Split by a seed, written now as 0, into **guard-dev** (half) and **guard-test** (half), by row, stratified by set
  and language. The split's row ids and SHA-256 are committed before any guard runs.
- The 38 natural rows are reported apart, always (dev).
- Caveat recorded with every number: a hidden fact is not distributed like a missing one.

**B. Three guards, each in front of the relation arms** (qFHRR at margins 0 and 0.25, lexicon_content, and both
450M arms from their saved programs). A guard either lets the arm's list through or refuses the row ("missing fact").
1. **G1, overall against own:** score the question's relation wording against the whole property codebook as well as
   the item's own properties. Refuse when the best property overall is one the item does not carry and beats the
   item's best by a margin set on guard-dev.
2. **G2, expected but absent:** the properties carried by at least a share s of the items of the item's class
   (`instance of`), counted from the local layered store's statements, join the candidates when this item lacks them.
   Refuse when the best candidate is one of them. The counts are statistics in a file beside the harness, never facts
   in a store. A class with too few items in the local store is "unavailable", never "none".
3. **G3, independent support:** a property from qFHRR or the 450M passes only if the lexicon or the grammar also
   names it; otherwise refuse. Agreement filters here; it never certifies.
- **G1 + G2** together as a fourth guard.
- Each margin and share is set on guard-dev only, recorded with its data and date, and never carried to another
  table or language.

**C. The reads:**
1. guard-dev: set the thresholds. Report the curve (handovers against false refusals) per guard and arm.
2. Present rows (the 870 rows as they are, the gold carried): recall@5 and wrong rate with and without each guard.
   This measures what the guard costs where the fact is there.
3. guard-test: read once, every threshold frozen.
4. The 38 natural rows: reported with each guard.

**D. Only if a guard passes, and only on Nick's word:** the relation step with that guard read once on R3's test
split (unread so far), against R3's gate. That spends R3's one test read.

## Acceptance (pre-registered)
- **Tests:** a hidden-fact row never carries its gold (pinned); a guard refuses when its condition holds and passes
  the list unchanged otherwise (pinned on made-up items); the split is the committed one (pinned by its hash).
- **A guard passes, for a given arm, if all hold:**
  - guard-test: handovers on at most 10% of rows;
  - the 38 natural rows: at most 4 handovers (the serving 450M's count);
  - present rows: recall@5 down by at most 2 points against the same arm without the guard, and SimpleQuestions
    recall@5 at least 0.85 for an arm that had it.
- **Kill, for that arm and guard:** more than 25% handovers on guard-test, more than 16 of the 38, or SimpleQuestions
  under 0.85 where it was above.
- **Reported:** every arm by every guard, per set and language; the thresholds with data and date; the share of
  classes "unavailable" for G2; latency per question of each guard (the lookup-and-check stage stays within p95
  50 ms).

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim).
- 0046's constraints hold: nothing on the serving path changes; no external model in any arm; no hand rule for a
  failing row; the sets' licences decide what is committed (ids and counts only).
- Facts are never model-written: G2's counts are statistics, never statements, and no guard adds a fact.
- CPU only. The class counts come from the local store; any fetch goes through the source's cache, dated.
- Each held-out read happens once.

## Out of scope
- The direction-aware asked kind (the synthesis's third candidate): its own order.
- French (the fourth candidate): its own order. French rows are reported here as they are.
- The 450M's selection seat with a "none of these" answer (the fifth): its own order, and it would be read on this
  order's hidden-fact set.
- Serving: a guard that passes is wired by its own order, after R3's test read.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
