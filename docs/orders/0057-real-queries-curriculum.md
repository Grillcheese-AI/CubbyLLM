# 0057 â€” Real Wikidata queries as curriculum: CubeLang's constructs ranked by real use, and training pairs written by rule
status: open (queue: after 0055, 0010, 0059, 0060, 0061 and 0020)
written: 2026-11-06 by Cowork Â· approved: 2026-11-06 by Nick ("yes") Â· hypothesis: H-E18 (program first, words second), with H-A15

## Why
- **The data:** the Wikidata Query Logs dataset (Walter and Bast, SIGIR 2026, arXiv 2602.14594). About 860,000
  distinct real queries people ran against Wikidata in 2017-18.
- **Real queries are richer than Cubby's programs.** Today Cubby's programs are mostly a seed and its hops, while 59%
  of the real queries use more:

  | Construct | Share of real queries |
  |---|---|
  | filters | 62.5% |
  | optional parts | 28.5% |
  | limits | 26.8% |
  | sorting | 19.7% |
  | property paths | 16.9% |
  | aggregates | 12.5% |

- **The paper's own measurement:** a 7B model trained on the older benchmarks scores F1 6.8 on these real queries.
  Trained on them, it scores 57.6, and already 51.6 at 10,000 pairs.
- **Why words by rule:** the dataset's questions were written by models, so Cubby cannot train on them (no external
  model writes a training record). H-E18's method still fits, with the words written by rule: the program first
  (here, a real query), the words second.

**What it brings, in plain words:** Cubby learns from what thousands of people really asked Wikidata, without any
outside AI writing its lessons.
- **Over a conventional model:** models trained on tidy benchmark questions collapse on real ones (6.8 against 57.6 in
  the paper). Cubby's training comes from real use, and every answer is still checked.

## Build
0. **The licence** of the logs and of the released files, checked and recorded. Stop if it does not allow this.
1. **The constructs, ranked by real use:**
   - each real query is parsed and its constructs counted: hops; filters, by kind; optional parts; sorting with a
     limit; aggregates; grouping; property paths; language filters; qualifiers;
   - reported: the share of real queries CubeLang can express today, and how that share grows construct by
     construct. That ordering is the VM's roadmap.
   - Adding a construct to CubeLang is its own order, taken from this ranking.
2. **A translator for what CubeLang already expresses:** SPARQL to CubeLang for the covered subset. Each pair is
   checked by running both, the query on 0055's cold tier and the program in the VM. Only pairs whose answers agree
   exactly are kept.
3. **Questions written by rule, in EN and FR:**
   - templates per query shape, filled with the EN and FR labels of the query's items and properties (the store's
     names), as 0048 did for MKQA and PIAF;
   - no model writes or rewrites a question;
   - a question that two different queries' templates render identically is dropped: it is ambiguous by
     construction.
4. **The training set and one adapter:**
   - the round-trip pairs, split by the dataset's own clusters, so no near-duplicate crosses splits;
   - a LoRA on the frozen trunk with the programs adapter's recipe, in a Colab notebook Nick runs.

## Acceptance (pre-registered)
- **Tests:**
  - the translator, on fixtures per construct: each gives its query's answers;
  - the templates never give two different queries the same question;
  - no record carries text from the dataset's model-written questions.
- **Reported:**
  - step 1's ranking and coverage curve;
  - step 2's translated share and agreement rate;
  - step 3's counts per shape and language.
- **Gate, for the adapter,** on 0054's understanding benchmark (this dataset's test sample):
  - right up by at least 10 points over the current programs adapter;
  - wrong not above it;
  - the existing batteries within 1 point (E8 reworded, the emitter families).
- **Kill:** no gain on that benchmark, or the batteries down by more than 1 point. The finding is then that
  rule-written questions sit too far from real wording.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model writes or rewrites a training record. The dataset's questions are evaluation data only.
- Nothing from the test split's clusters enters training.
- The Colab notebook is Nick's to run. grilly2 only on the local card. No local path in a committed file.

## Out of scope
- **Adding constructs to CubeLang:** their own orders, from step 1's ranking.
- **Cubby's own model writing the questions:** a later comparison against the rule-written ones.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Cowork's addendum (2026-11-06): from the Freiburg group's repositories
Nick: "yes". Licences are checked and recorded first.
- **The translator's fixtures:** for every construct the translator covers, the cases of the SPARQL conformance suite
  (`ad-freiburg/sparql-conformance`) that use only covered constructs. Each must give the query's answers.
- **GRASP's shape index:** per class (the instance-of values), the properties its instances carry, with counts, built
  from 0055's store.
  - Offered to the programs seat as schema context: the property candidates for the asked item's class.
  - Measured as an arm of the gate, the adapter with and without it on the understanding benchmark.
- **GRISP as the baseline** (Walter and Bast; arXiv 2604.21133). A small model writes a query skeleton and the
  entities are filled in by search, which is close to Cubby's slots (H-E15). Its published F1 on the query-log test
  sample sits beside Cubby's numbers in the paper. Running it here is an evaluation, never a training source.
## Cowork's addendum (2026-11-06): QAWiki's hand-crafted pairs as training data
QAWiki (the WikiKGQA 2026 dataset, Zenodo, CC0) is a community resource of hand-crafted question-query pairs, written
by people. Under Cubby's rules that makes them usable as training data, unlike the query-log dataset's model-written
questions.
- **Its training pairs,** in English and Spanish, join the round-trip set: translated to CubeLang where covered, and
  kept only when the answers agree.
- **Nothing from the challenge's test** (the 150 questions, 0054's addendum) enters training.
- **Reported:** the effect of the hand-crafted pairs on the gate, against the rule-written questions alone.
## Cowork's addendum (2026-11-06): questions about two periods of time
Nick: "yes". The idea comes from Semantica (MIT), which implements Allen's interval relations; only the idea is taken.

**What it brings, in plain words:** Cubby answers "did this happen while she was in office?" exactly, and says "can't
tell" when the dates are too vague.
- **Over a conventional model:** a model guesses at overlaps between dates and gives no sign when the dates cannot
  decide.

- **The relations:** Allen's 13 relations between two periods (before, meets, overlaps, starts, during, finishes,
  equals, and their inverses), plus a date against a period.
  - They are computed exactly from the statements' dates: start (P580), end (P582) and point in time (P585).
- **Precision decides:** Wikidata dates carry their precision (year, month, day).
  - A relation is answered only when it holds for every date the precision allows.
  - When the precision leaves it open, Cubby refuses with the reason "dates too coarse". For example, a year against a
    month of that same year.
- **A missing end date is unknown, not "still true".** "Still true" needs the live source (the standing rule for
  time-sensitive facts).
- **The question shapes:**
  - "who held office X when Y happened";
  - "was A still in office when B was founded";
  - "did A and B overlap".

  They are written by rule from held-out statements, in EN and FR. A seeded sample of 100 is read by hand.
- **Measured:** right, wrong and refused, with "dates too coarse" counted apart.
- **Line:** 0 wrong.
