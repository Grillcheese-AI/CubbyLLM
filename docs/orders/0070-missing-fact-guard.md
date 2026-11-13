# 0070 - The missing-fact guard: the relation step refuses when the item lacks the fact asked, measured on hidden facts
status: review (Code B, on the branch `agent-b`, 2026-11-13: no guard passes; the report below)
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
**Caveat first (Cowork's wording, 2026-11-13):** a hidden fact is not distributed like a missing one. A hidden fact sits
in a class where most items carry it, which is where G2 speaks; a fact missing in the world may not. The 38 natural
rows are the check on that, and there no guard moves the serving 450M (4 handovers with and without G1, G2 or G1+G2).

**Verdict: no guard passes for any arm.** The closest is the serving 450M behind G2 (and G1+G2, the same rows): it holds
three lines (natural 4 of 38, present loss 0, no SimpleQuestions line to keep) and misses guard-test's 10% line by
5 rows (49 of 436, 11.2%). D (R3's test read) does not apply.

- **Commits** (on `agent-b`): `74bab81` (the set's split and the guards' rule, before any guard ran), `4e6d0a3`
  (the harness; guard-dev's curves; the thresholds frozen), `43dd3be` (guard-test read once), and this report's
  commit (the "unavailable" breakdown, this report).
- **Files:** `validation/retrieval_bench/r3g_set.py` (the hidden-fact set, the split), `r3g_split.json` (ids +
  SHA-256 `4bef4d41...cb5a`), `r3g_guards.py` (G1, G2, G3, the grids and the threshold rule), `r3g_classes.py` +
  `r3g_classes.json` (G2's statistics), `r3g_arms.py` (R3o's arms scoring any property set, on the CPU), `r3g_eval.py`
  (records, tallies, curves, lines, timing), `r3g.py` (parts dev / test / time / g2), `r3g_thresholds.json`;
  `tests/validation/test_r3g.py`. Logs: `retrieval_R3g_dev_2026-11-13.json`, `retrieval_R3g_test_2026-11-13.json`,
  `retrieval_R3g_g2_2026-11-13.json`.
- **Tests:** 7 new, all pass: a hidden row never carries its gold or the gold's inverse (500 random rows plus two named
  ones); G1, G2 and G3 refuse on their condition and pass the list unchanged otherwise, on made-up items; the
  threshold rule; the split is the committed one (hash pinned, re-derived from its strata by seed 0). The whole suite
  after merging master: in the inbox note.

**How it was built** (what the numbers rest on):
- **The rows.** R3o's dev rows rebuilt in this interpreter and checked before any guard: every list and judgement equal
  to R3o's logs on all 908 scored rows for `lexicon_content`, `450m`, `450m_wd`, `qfhrr@0.0` and `qfhrr@0.25`.
  qFHRR is scored in numpy with the scorer's own codes, rounded to 6 decimals as the grilly2 scorer hands them over
  (without the rounding one row differs: a 7e-9 tie at rank 5). No GPU was used.
- **The hidden-fact set** (A): the 870 rows whose items carry a gold, each with its golds and their inverses (P1696)
  taken out of the candidates (705 rows lose one property, 97 two, 50 three, 13 four, 5 five; none is left empty).
  guard-dev 434 / guard-test 436, by seed 0 within each set and language.
- **The present rows** are the same 870 items with their golds kept: on guard-dev, the thresholds were set on the same
  434 items, hidden and present. Present results are also reported by half (`present_by_half` in the test log).
- **G1** uses each arm's own scorer over the whole codebook (qFHRR: the 12,684 codes; the lexicon and the 450M: every
  property their wordings name). **G2**: `r3g_classes.json`, from the local store's kinds table (snapshot `8a2041cf...`,
  73,205,306 items): 538 classes of R3o's 952 items; 19 not in the store; 171 under 50 items. **G3** keeps what
  `lexicon_content` (any wording found among the candidates) or the grammar names.

**Measured** (guard-test read once, thresholds frozen; `retrieval_R3g_test_2026-11-13.json`). Handovers on guard-test
(line at most 10%, kill over 25%), on the 38 natural rows (line at most 4, kill over 16), present recall@5 over 870 and
its loss (line at most 2 points), SimpleQuestions present recall@5 (line 0.85 where the arm had it: only qfhrr@0.0,
0.8926):

| arm, guard (threshold) | guard-test | natural | present r@5 (loss) | wrong rate | SQ r@5 | lines |
|---|---|---|---|---|---|---|
| 450m, none | 64/436 (14.7%) | 4 | 0.2908 | 0.0931 | 0.4732 | fails 10% |
| 450m, G1 (m 1.0) | 64 (14.7%) | 4 | 0.2908 (0) | 0.0931 | 0.4732 | fails 10% |
| **450m, G2 (s 0.05)** | **49 (11.2%)** | **4** | 0.2908 (0) | 0.0908 | 0.4732 | **fails 10% only** |
| 450m, G1+G2 (1.0, 0.05) | 49 (11.2%) | 4 | 0.2908 (0) | 0.0908 | 0.4732 | fails 10% only |
| 450m, G3 | 37 (8.5%) | 1 | 0.2575 (0.0333) | 0.0529 | 0.4430 | not a passing configuration: loss over 2 points (dev 0.0369) |
| lexicon, none | 105 (24.1%) | 6 | 0.3908 | 0.1448 | 0.5671 | fails 10%, 4 |
| lexicon, G1 (m 0.5) | 97 (22.2%) | 6 | 0.3851 (0.0057) | 0.1425 | 0.5503 | fails 10%, 4 |
| lexicon, G2 (s 0.05) | 78 (17.9%) | 6 | 0.3839 (0.0069) | 0.1425 | 0.5604 | fails 10%, 4 |
| lexicon, G1+G2 (0.5, 0.05) | 73 (16.7%) | 6 | 0.3793 (0.0115) | 0.1402 | 0.5470 | fails 10%, 4 |
| 450m_wd, none | 137 (31.4%) | 16 | 0.4943 | 0.2805 | 0.8054 | killed (over 25%) |
| 450m_wd, G1 (m 0.9) | 136 (31.2%) | 16 | 0.4931 (0.0012) | 0.2805 | 0.8054 | killed |
| 450m_wd, G2 (s 0.3) | 126 (28.9%) | 16 | 0.4943 (0) | 0.2805 | 0.8054 | killed |
| 450m_wd, G1+G2 (0.9, 0.3) | 125 (28.7%) | 16 | 0.4931 (0.0012) | 0.2805 | 0.8054 | killed |
| 450m_wd, G3 | 13 (3.0%) | 1 | 0.2851 (0.2092) | 0.0368 | 0.5235 | not a passing configuration (dev loss 0.2097) |
| qfhrr@0.0, none | 431 (98.9%) | 36 | 0.7138 | 0.2747 | 0.8926 | killed (over 25%, over 16) |
| qfhrr@0.0, G1 (m 0.9) | 411 (94.3%) | 33 | 0.7138 (0) | 0.2655 | 0.8926 | killed |
| qfhrr@0.0, G2 (s 0.2) | 308 (70.6%) | 32 | 0.7080 (0.0058) | 0.2701 | 0.8859 | killed |
| qfhrr@0.0, G1+G2 (0.9, 0.2) | 301 (69.0%) | 30 | 0.7080 (0.0058) | 0.2609 | 0.8859 | killed |
| qfhrr@0.0, G3 | 86 (19.7%) | 5 | 0.4207 (0.2931) | 0.1138 | 0.5906 | killed (SQ under 0.85); not a passing configuration |
| qfhrr@0.25, none | 168 (38.5%) | 16 | 0.4184 | 0.1000 | 0.5772 | killed (over 25%) |
| qfhrr@0.25, G1 (m 0.7) | 148 (33.9%) | 14 | 0.4034 (0.0150) | 0.0908 | 0.5604 | killed |
| qfhrr@0.25, G2 (s 0.05) | 114 (26.1%) | 15 | 0.4000 (0.0184) | 0.0943 | 0.5470 | killed |
| qfhrr@0.25, G1+G2 (0.9, 0.05) | 114 (26.1%) | 14 | 0.4000 (0.0184) | 0.0931 | 0.5470 | killed |
| qfhrr@0.25, G3 | 39 (8.9%) | 4 | 0.2563 (0.1621) | 0.0586 | 0.4094 | not a passing configuration (dev loss 0.1752) |

- **G3 is never a passing configuration:** its recall loss on guard-dev (0.0369 to 0.3133) is over the 2-point rule for
  every arm it ran on, so it had no threshold to set; it is reported as measured.
- **Per set and language** (every arm by every guard, hidden, present and natural): in the test log's `reads`. For the
  closest, the serving 450M behind G2, guard-test handovers by set: SimpleQuestions 14/149, QALD EN 13/92, QALD FR 8/77,
  Mintaka EN 11/60, Mintaka FR 3/58 (without the guard: 22, 14, 11, 13, 4). Its 4 natural handovers (P149, P17, P856,
  P40) are the same 4 with or without G2.
- **The thresholds** (`r3g_thresholds.json`, set 2026-11-13 on guard-dev's 434 rows, split `4bef4d41...`), and the
  curves: `retrieval_R3g_dev_2026-11-13.json`. G1 barely moves any arm. For qFHRR at margin 0, G1 at m = 0 costs 0.4885
  of recall on guard-dev's present rows: on most present rows some property the items do not carry already scores above
  the gold. The 450M's G1 refuses nothing at any margin.
- **G2 "unavailable"** (no class of the row's items has 50 items in the store; `retrieval_R3g_g2_2026-11-13.json`):
  guard-dev 23 of 434 (Mintaka EN 4/60, FR 3/57; QALD EN 6/91, FR 3/77; SimpleQuestions 7/149); guard-test 18 of 436
  (2/60, 3/58; 4/92, 6/77; 3/149); natural 10 of 38 (QALD FR 5 of 12, QALD EN 4 of 17, Mintaka FR 1 of 6).
- **Time per question of each guard's own work** (present rows, 870; CPU, 6 threads; at the start of the read no CubbyLLM
  job of Code A ran, and the CPU was 33% busy with other applications): qFHRR's G1, p50 / p95 23.0 / 26.4 ms (its whole
  codebook, 12,684 codes of 10,240 floats, in numpy); its G1+G2 24.6 / 31.5 ms; its G2 0.55 / 7.9 ms; every guard of the
  lexicon and the 450M p95 under 0.2 ms. All within the 50 ms p95 line. The class lookup is a dict from the class file,
  not the store.
- **"refused" counts in the log** include rows the arm did not read (a guard's condition can hold there); they change
  no handover and no hit. The losses above are hits lost.

- **Not verified here:** the qFHRR scores on grilly2's card (scored in numpy here; equal to R3o's grilly2 lists on all
  908 rows); the timing on a quiet machine (`--part time` reruns it without reading any outcome).
- **Found on the way:**
  - **The inverse rule misses kin:** Wikidata's `inverse property` of P40 (child) is P8810 (parent), and P22 (father) and
    P25 (mother) declare none. So on the 16 rows whose gold is P40 and the 2 whose gold is P22, the fact can still be
    carried the other way (P22 / P25 incoming), and a handover of it there is counted against the arm. The rule was
    written before the read, so it is left as is; at most 18 of 870 rows.
  - **Code B's data folder:** `cubbyllm.local_paths.data_dir()` resolves to this checkout's own `standin/data/out`, which
    has no source cache, so a harness run here would fetch everything again. I set `CUBBY_DATA` to the main checkout's
    `standin/data/out` for each run, which the shared-data rule allows. Adding `CUBBY_DATA` to this checkout's
    `paths.local.json` would make that the default.
  - **qFHRR's tie at 6 decimals:** R3o's lists depend on the grilly2 scorer rounding its scores (`r3_qfhrr_score`), so a
    CPU path has to round too. One of 908 rows differs without it.
- **Questions:**
  1. The serving 450M behind G2 misses only the 10% line (49/436). The rule allows no second read, so this order ends
     here. Is a new order with a different G2 (incoming counts as well as outgoing, or the class chain) worth writing,
     read on a fresh hidden set?
  2. Should the kin inverses (P40 against P22 / P25) be added by hand to a future hidden set's rule?
