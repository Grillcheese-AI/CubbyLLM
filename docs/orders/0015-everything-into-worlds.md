# 0015 — Everything in the datasets into worlds: one manifest-driven ingestion, incremental
status: review
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

- **Domain routing for untyped text and questions:** port `domain_classify.py` / `domain_regex.py` from Nick's `empirical_grilly_next` project (a checkout beside this one; path in `CLAUDE.local.md`)
  (80+ ordered regexes over the 88-domain SVC ontology, no model). Use it where a dataset has no topic map, and
  to pick which domain world a question consults first.

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

## Addendum (Cowork, 2026-10-22): label hygiene when a world is mounted
Order 0011 found, in the wiki world: 528 facts whose subject or object is a Python repr (`<class 'wikikg.relations.X'>`,
`<__main__.DynamicRelation object at 0x...>`), one object `_` (`BryantPark LOCATION_OF _`), and a relation spelled
`SP0USE_OF` (a zero for the O). Mounting a world drops facts whose ends are not names, maps known misspelled relations
to their spelling, and logs both counts per world.

## Addendum 2 (Cowork, 2026-10-22): the time rule on every world
From order 0012's review: `standin/time_rule.py` reads the books' witnesses. It also reads every mounted world's
facts: an undated value of a relation in `CHANGES` (a spouse, a capital, a head of state) is said "at one point" or
carries the world's "as of", never flat as if current. Report how many facts per world it touches.

## Addendum 3 (Cowork, 2026-10-22): when two worlds disagree
Order 0014 found 1,378 facts where the hdc world and the wiki world give another value for the same subject and
relation, against 311 that agree. Sample 50, sort them (coarser or finer, older, a label, an error), and set a rule
per kind; until then a disagreement is held, never said. Order 0013's audit also found the wiki world stating an
ancient name "located in" its modern one ("Antium" in "Anzio"): an identity, not a location. Count those and map
them to a same-place relation when the world is mounted.

## Report
- **Commits:** be2db9e, 0394463, ce80474 (the manifest, the release filter, the topic map and catalog), fe580b9 (in
  progress), 8ce7eab (the tool: routes, registry, fiction worlds, one store interface, wiki hygiene), 9e4562a
  (Addendum 2), b3e7551 (Addendum 3), 2d5f0f0 and 093017f (local paths out of the code and the logs), and this
  commit (the run's logs, the report, one scan fix).
- **Files:** `standin/worlds_manifest.py`, `worlds_adapters.py`, `worlds_fiction.py`, `worlds_ingest.py`,
  `worlds_store.py`, `standin/data/world_hygiene.py`, `standin/data/build_same_place.py`, `standin/hdc_store.py`
  (`disagreements`), `standin/time_rule.py`, `standin/ask.py` (the world branch of the time rule, the contested
  seed), `validation/exp_worlds_bench.py`; tests `standin/tests/test_worlds.py` (6), `test_time_rule.py` (5),
  `test_books_bulk.py` (veto, same-name guard, at one point). Local and git-ignored: the per-dataset manifest, the path
  map (`docs/*.local.tsv`), the registry and the stores. **Disk:** the data disk Nick chose, 3.4 GB in all (fiction
  2.4 GB, passages 0.9 GB, registry, events, quotes and tables under 10 MB; the Gutenberg catalog 21 MB).
- **Tests:** `pytest tests`: 441 passed, 5 skipped; red only the known `test_every_module_declares_wiring` (order
  0002). One more red was mine and is fixed here: the sleep test's reason scan read the time rule's `"as_of"` tier on
  the line after a reason and took it for one (`tests/reasoning/test_sleep.py`: listed with the scan's other
  non-reasons). `pytest standin/tests`: 653 passed; 25 red that predate this order and touch nothing of it
  (`test_build_chat_sft.py` 24 and `test_chat.py` 1: names `build_chat_sft.py` no longer has since 34e7658; and
  `test_goemotions_cube.py` does not import, `GOEMOTIONS`, the same file).
- **Measured** (`validation/logs/worlds_ingest_2026-10-22.json`, `validation/logs/exp_worlds_bench_2026-10-22.json`):
  - **Coverage:** 76 manifest rows, 76 decided: 23 ingested, 42 excluded, 11 queued for a later tranche (Nick's
    choice of this pass: the tool and a first tranche; Wikipedia, web text and code wikis later). Per ingested row
    (manifest row: world, route, rows, entities, files, as-of, frame triples, seconds):
    2 how-to passages 354,435 / 29,578 / 193 files / 58 triples / 222 s; 3 business 9,792 / 2,149 / 20 / 10 s;
    4, 5, 9 reference 114,481, 138,966 and 183,069 rows (66, 71, 137 triples; 161, 17, 248 s); 6 assorted, classified
    37 books, all reality (44,617 rows); 7 psychology 40,137 (tier attributed, low trust); 8 code 34,859; 10
    classified: 224 books to fiction worlds, 25 to science passages (one author's fiction and non-fiction, split by
    their text); 11 exam 5,771; 12 scholarly 53,021 (11 triples); 16, 17, 75, 76 fiction: 4,157,232 rows in all, 189
    works (one world each; row 76 re-read adds 0 files); 18 encyclopedia 109,769 rows, as of 2005; 20 the hdc world
    mounted (3,384,594 facts, as of 2019); 21 quotes 24,022 (attributed); 22 the history books mounted (5,639,095
    rows); 25 a table 10,810 rows; 53 events 3,934 (attributed, not verified); 35 and 61 the self world mounted. The
    whole run took 4,114 s (about 69 minutes), most of it fiction (3,041 s) and its frames.
  - **Fiction isolation:** 50 in-fiction facts (kin a novel states of its own people: child 17, father 12, spouse 10,
    sibling 8, mother 3) asked as reality questions: **0 spoken**, 0 with the fiction's value. Asked naming their work:
    49 answered, 49 correct, 1 not spoken (a refusal, not a wrong answer).
  - **Exclusion:** 0 files of an excluded dataset in the registry. The evaluation sets' 5,642 questions as 5+-word
    phrase queries over the 12 passage worlds: 2 hits, both the same 8-word noun phrase ("third president of the
    United States of America") in the history books' own text and in one novel -- a phrase a book writes, not an
    evaluation row; no answer-bearing row of an evaluation set is in a world. The repo's own bench questions (526):
    0 hits.
  - **No regression:** through the registry's stores, 0012's bench gives what its final log gave (30 spoken, 1 flat,
    29 hedged, 0 wrong flat, 3 wrong hedged -- 0012's own three, unchanged, each said hedged as one source's) and 0014's
    gives 177 spoken, 177 correct, 0 wrong, 0 time-sensitive said flat. **0 wrong said flat** across both; the 3 hedged wrong answers are 0012's, not new.
  - **Incremental:** a re-run of the events and the quotes datasets with nothing new: +0 files, rows unchanged (3,934
    and 24,022). Adding one file adds only it: `test_worlds.py` (synthetic fixtures).
  - **Addendum (hygiene):** in `world_hygiene.py` at the mount: debris, values equal to their subject, `SP0USE_OF`
    respelled (counts in the mount's log, b3e7551).
  - **Addendum 2 (time rule), facts touched per world:** wiki 52,815 of 551,661 (said "at one point"); hdc 908,108 of
    3,384,594 in a changing relation, 155,749 of them time-sensitive (said "as of 2019"); the bulk books' facts 466
    of 2,230 (at one point, hedged when attributed); fiction 296 of 1,239 (at one point, inside its work).
  - **Addendum 3 (disagreements)** (`worlds_disagreements_sample_2026-10-22.json`): compared 1,689, agree 311, agree
    once folded 48, held 1,330; the 50-sample's kinds: both true (many-valued) 18, finer against coarser 11, the same
    value under another label 10, a wiki-world error 9, a namesake 2, older against newer 0; a rule per kind in the
    log. Ancient names in their modern ones: 7 pairs mapped to a same-place relation at the mount
    (`build_same_place_2026-10-22.json`).
- **Not verified here:** the routes not in this tranche (11 queued rows); the release build (`--release`) beyond its
  fixture test -- no release has been cut; the domain classifier port for untyped questions (the topic map routed the
  domains this tranche needed).
- **Found on the way:**
  - The hdc world's label lookup lands on namesakes for ancient places (Nineveh resolves to a village in Nova Scotia,
    Byzantium to a film): the same-name guard by dates (0012) does not apply, as the hdc world carries no dates there.
    Recall of the same-place mapping is low for the same reason (Constantinople is lost to a Wikidata5M country label).
  - My first commits of this order carried local paths (the rights tracker, the catalog, the books index, the eval
    sets, the default roots) in code and one log's store paths and reasons (the reasons name collections): all now
    read from the environment, the committed report keeps a dataset's manifest row only (2d5f0f0, 093017f). The values
    live in the local notes. Older files in the repo still carry local paths from before the rule (the hypotheses
    file, `data/corpus_sources.json`, a few top-level docs): not touched here.
- **Questions:**
  - The held disagreements (1,330) stay unsaid until a rule per kind is wired; the sample's rules are proposals for
    Cowork: which to wire first? ("same value, another label" needs an alias table from the hdc world's names.)
  - The next tranche: Wikipedia (passages and frames) first, as the order's list has it?
