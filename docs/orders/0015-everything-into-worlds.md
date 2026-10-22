# 0015 — Everything in the datasets into worlds: one manifest-driven ingestion, incremental
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("store by worlds everything in datasets, not just use
them as pretraining"; "everything in datasets is stuff Cubby can see once and learn") · hypothesis: H-F2, H-K10
**After 0012 and 0014:** it generalizes their index and loader.

## Why
The policy is `docs/WORLDS.md`, and the per-dataset list is `docs/worlds_manifest.local.tsv` (git-ignored; Cowork
wrote it from an inventory of both dataset trees on 2026-10-22). Orders 0012 (books), 0013 (book reader) and 0014
(the hdc triples) each ingest one source. This order makes ingestion one tool, driven by the manifest, so every
dataset, and every one added later, lands in its world the same way: rows with provenance, tiers, and incremental
re-runs.

- **What it brings:** everything Cubby has been given becomes knowledge it can look up, cite and check: learned
  in one sighting, not a thousand.
- **Advantage over conventional models:** they only pretrain on their data, so it comes back blurred, uncited, and
  frozen at the training date. Here every source is also a store: exact, attributable, updatable without retraining.
- **In plain words:** whatever you give Cubby, it reads once and keeps, and it keeps track of where it learned
  each thing (for an audit, not shown in replies).

## Build
- **`standin/worlds_ingest.py`** (new, < 300 lines) and adapters (< 300 lines each). It reads the manifest's routes:
  - `triples`: 0014's loader, generalized;
  - `passages`: 0012's index, generalized to any text domain, with frames and, after 0013, the reader;
  - `events`: dated, to the year's world;
  - `quotes`: speaker → quote, tier "attributed";
  - `fiction`: one world per work, in-fiction only;
  - `table`: rows;
  - `EXCLUDED`: recorded with its reason, never loaded.
- **Per dataset:** world ids, tier, provenance (source, work, location, date), and an as-of date where the source
  has one. Each work also carries a release status from a local list (see `CLAUDE.local.md`). `--release`
  builds a store from the cleared works only, and reports what it left out. Incremental by file hash: a re-run adds only what is new, and nothing is deleted.
- **One store interface** for the ask loop: lookup by entity and relation across worlds, with a world filter. The
  reality worlds are consulted by default. A fiction world is consulted only when the question names its work. The
  self world is consulted for questions about Cubby.
- **Order of ingestion** (by value, reusing what exists):
  1. hdc (0014);
  2. history books (0012/0013);
  3. Wikipedia, as passages and frames;
  4. dated news events;
  5. dbpedia corpus;
  6. encyclopedia (mount);
  7. reference and scholarly books;
  8. quotes;
  9. code wikis;
  10. real legal text;
  11. fiction worlds;
  12. web text, last, as witness.

  Report per dataset as it lands. The "classify first" rows are classified by Claude Code, with reasons.
- **Topic routing that already exists** (path in `CLAUDE.local.md`): a file → topic map from an earlier
  classification pass. It has 44,761 rows over 83 topics: Gutenberg books (`gbooks`, `gbooks_factuals`, ~2.8k)
  and Wikibooks pages (~41k). Use it to route those files to their domain worlds. Normalize the topic case first
  ('history' and 'History' are one topic), and map the 83 topics onto the worlds' domain list.
- **The Gutenberg catalog** (Project Gutenberg's own catalog CSV; a small download, **on Nick's yes**) gives each
  numeric id its author, title, subjects and copyright status. It does three jobs:
  - it names the ~5.6k "Gutenberg #N" rows in the rights tracker;
  - it confirms their public-domain status;
  - it marks **fiction** (subjects and bookshelves), so novels go to fiction worlds, not reality.

## Acceptance (pre-registered)
- **Coverage:** every manifest row is ingested or excluded with its reason. The report has a per-dataset table:
  rows, entities, world, tier mix, as-of share, time.
- **Fiction isolation:** 50 in-fiction facts asked as reality questions → **0 spoken**. The same 50, asked naming
  their work → answered in-fiction.
- **Exclusion:** no evaluation set's answer-bearing rows are in a serving world. The check runs over every bench
  file in the repo.
- **No regression:** 0012's and 0014's benches give the same results through the generic tool. **0 wrong spoken**
  across all benches.
- **Incremental:** a re-run with nothing new changes nothing. Adding one new book or file adds only it.
- Full `pytest tests`. Synthetic fixtures only; no source text in the repo.

## Constraints
- Rules in `CLAUDE.md` hold. CPU, Nick's machine. Disk: the stores live off-repo on a drive with room (say which and
  how much). No network beyond 0014's label gaps (Nick's yes).

## Out of scope
- Writing worlds into the trunk's scene in bulk (0009). Retraining.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
