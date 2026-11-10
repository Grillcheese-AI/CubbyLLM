# 0055 — The full knowledge base at home: Wikidata hot and cold, so the benchmarks read the real ceiling
status: in progress
written: 2026-11-06 by Cowork · approved: 2026-11-06 by Nick ("perfect lets put that as a top priority so we can have the true numbers for the benchmarks. Right now its very low as we do not have the full kb") · hypothesis: the system paper; the design is `docs/research/2026-11-06-fact-store-synthesis.md`

**Priority: top** (Nick). It goes before 0054's full run, the rest of 0053, and every other open order. 0054's full run
waits for this order, then reads before (today's knowledge base) and after.

## Why
Today Cubby answers from:
- a 4.3M-fact Wikidata5M world;
- a small wiki world;
- live Wikidata, through its search API.

SimpleQA's "3 of 860" was the wiki world. 0054 run now would measure the store, not the system.

The panel and its check (the synthesis) give the design: **one hot store, Cubby's own, in Rust, and the full graph as
a cold tier.** The full graph's numbers come from Willerval, Diefenbach and Bonifati (ESWC 2024): 19.2 billion
triples, built on a 16 GB desktop in about two days, 340 GB.

**What it brings, in plain words:** Cubby keeps a fast pocket edition of what it needs and the complete Wikidata on
the shelf beside it, so the benchmarks show what the system can do, not what its small store happened to hold.
- **Over a conventional model:** a model's knowledge is whatever fit in its weights at training. Here every fact is
  stored exactly, with its source, and is added or corrected in one write, without retraining.

## The machine today (read 2026-11-06)
- 48 GB of RAM. Five drives with 47-57 GB free each. No Java.
- **A 2 TB NVMe is needed before the full builds:**
  - the dump, if stored (100-150 GB compressed);
  - the cold tier (about 340 GB) and its build space;
  - the hot store (25-90 GB, estimated).

  This is Nick's purchase. Until it is in, steps 0 and 1 run: they need no local room.
- **Update (2026-11-06, later): no purchase needed to start.** Nick is freeing the system NVMe (94 GB free, about
  210 GB once his Steam library has moved). The 1 TB hard drive is back and reads cleanly.
  - **Hot store:** on the system NVMe.
  - **Cold tier:** on the 1 TB hard drive. That drive is slower, which is acceptable for the rarely used tier.
  - The 2 TB NVMe becomes an upgrade, not a blocker.
- **Nick's 5 TB Google Drive holds the dump and every build product.** Builds can run on Colab against it. Serving
  still needs the local NVMe: a lookup from Drive is far too slow for the VM's hops.

## Build
0. **The coverage audit** (outside `cubbyllm/`): 1,000 questions.
   - **Amended 2026-11-06 (Nick: "stop it now", "amend step 0").** Read live, the audit costs about 13 hours (about
     50 s a question on Wikidata's API), and neither of its jobs needs the live API: the split does not gate the
     download (below), and the store check needs a reference, which the same audit gives against the dump. So:
     - **Step 0 is the audit's code** (`validation/fact_store/`, `e41d481`) **and the 51-question live sample** it read
       before it was stopped (SimpleQA only, kept off-repo). The sample is the live reference for the store check.
     - **The 1,000-question split is read in step 1, from the dump** (its statements in place of the API), in
       minutes.
   - **The questions:** SimpleQA 400, Mintaka 300 (with its French), MKQA 300 (French).
   - **Per question:** is the gold a value of a statement on the asked item, elsewhere in Wikidata, or not in Wikidata?
   - **Where a set has no gold ids** (SimpleQA): every candidate item for the name is checked (no fame ordering), and
     ambiguity is counted.
   - 100 disagreements are read.
   - **The split is the ceiling the full base can reach.** A low share means text is the next lever, not a stop:
     Nick's priority stands.
1. **The dump, downloaded once to Nick's 5 TB Google Drive** (Nick, 2026-11-06, answering where the download may go:
   his 5 TB Drive). This is done from a Colab session signed in to the account that owns that Drive. The compressed
   dump is kept there as the source copy, so a parser bug means a reparse, not a re-download. Then one pass over it,
   never unpacked to disk, writes two things:
   - **Counts:** items; statements by rank; qualifiers and references; EN, FR and `mul` terms; bytes by section. This
     settles the panel's conflicting sizes.
   - **A 1% hash sample** (not the first Q-ids). The hot store is built from it, and on it are measured:
     - bytes per statement, and the build's time and peak RAM;
     - lookup latency, in memory and from SSD;
     - name resolution: latency, and EN/FR recall on the audit's entities that fall in the sample;
     - **measured, not gating:**
       - inverse, symmetric and chain folding, from Wikidata's own property constraints, with exceptions counted;
       - place ids in S2 (Hilbert) order against Q-id order: pages read on geography questions;
       - neighbourhood preloading against a plain cache;
       - how often a coarse community map prunes a relation search.
   - **Where:** a Colab session Nick runs, reading the dump from his Drive. The counts and the sample are written
     back to Drive.
2. **The full builds, after the NVMe:**
   - **Hot, Cubby's store** (the synthesis, section 3):
     - dense ids;
     - statements at every rank, flagged;
     - time qualifiers hot, the others cold;
     - references deduplicated;
     - exact values (no floats; dates keep their precision and calendar);
     - EN, FR and `mul` names in an FST, exact and accent-folded, with a bounded typo path;
     - a reverse index;
     - a small overlay for live values.
   - **Cold:** the full Wikidata in HDT. Either built with qEndpoint's k-HDTCat (GPL-3.0, Java 17, used as a
     separate tool and never linked into Cubby's code), or taken from the prebuilt index its image downloads (its
     date and contents recorded). Which one is Nick's call at download time.
   - **Both are resumable in chunks.** This PC has shut down under long jobs before.
3. **Wiring, behind a flag (off by default):** the VM's hops read the hot store, then the cold tier, then live
   Wikidata.
   - The live value wins for time-sensitive facts (the standing rule).
   - The cold tier is read in-process from the HDT file. If only a server works, that is a question for Nick: the
     serving code never touches the network.
4. **0054 reruns with the flag on.** Its full run is the "after"; its 50-question preflight on today's base is the
   "before".

## Acceptance (pre-registered)
- **Tests:**
  - 100,000 entities round-trip with 0 differences against a second, independent parser;
  - 1,000 items re-fetched at their recorded revision agree;
  - a rebuild gives identical chunk hashes;
  - with the flag off, today's answers are unchanged, bit for bit;
  - no stored value is a float;
  - no candidate list is ordered by sitelinks, pageviews or Q-id.
- **Reported:**
  - the stream's counts;
  - the hot store's size and bytes per statement;
  - build time and peak RAM;
  - lookup latency p50, p95 and p99, in memory and from SSD;
  - name resolution latency and recall, EN and FR;
  - the audit's split, per set.
- **The lines:**
  - a fact lookup from SSD at most 1 ms p95; name resolution at most 50 ms p95;
  - every fact the audit found in live Wikidata (step 0's live sample) is found in the store, unless it changed after
    the dump's date (listed);
  - with the flag on, 0 wrong on the existing batteries (E8 reworded, the emitter families, the stack's 20, 0038's
    scoreboard), and more answers spoken.
  - **Size:** the hot store at most 90 GB, the panel's top estimate. Above that, the projection is reviewed with
    Cowork before the full build.
- **Kill:**
  - any stored value that differs from its source: the build is not faithful;
  - or, in 0054 with the full base, Cubby's wrong answers above 1% of its spoken ones.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Where the store lives:** in the VM's datastore (the cubelang repository), or in a crate beside it. Claude Code's
  choice, recorded. Main cubelang's `target/release` binary is never overwritten.
- `cubbyllm/` never touches the network. The audit, the stream and every download run outside it.
- **Downloads over 1 GB:**
  - The dump to Nick's Drive has his yes (2026-11-06).
  - Any other large download (a prebuilt HDT index, a second dump) waits for his yes in chat.
  - The Drive's account is named in no committed file.
- No external model builds stored knowledge.
- **Licences recorded:** Wikidata CC0; qEndpoint GPL-3.0 as a tool only; every crate's licence checked.
- No local path in a committed file.

## Out of scope
- **Text** (Wikipedia, books): passages, extraction, and their compression (hypertokens or zstd). The next order,
  from this one's numbers.
- **Following Wikidata's change stream:** after the first full build.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Where it stands (2026-11-10):** steps 0-2 built and measured: the full hot store (73.2M entities, 758.6M
  statements, 77.2 GB) on the system NVMe, the scholarly articles' cold set (45.4M entities, 972.4M statements,
  47.0 GB) on the hard drive, checked clean, read by the board's Knowledge section. Step 3 is wired behind
  `CB_WD_STORE=1` (below): the store is a world ahead of Wikidata5M. Step 4 (0054 with the flag on) is next, and is
  where step 3's finding (full Wikidata shares most well-known names, so the strict namesake rule reads little by
  name) gets its number. One line is missed: a fact lookup from SSD is 1.5 ms p95, the line 1 ms (below). Status
  stays `in progress` until step 4.
- **Commits:**
  - CubbyLLM: b440256, 9fbc251, 4aeb97b, c8f1bda, a6ffd4b, 1df13c7 (wrong, see below), a4e85c3, 87a40ea, f2775a4,
    f3c3771, d91763f, a259fdb, ecd2bc4, 0936d0f, bacab67, 80b4641, 68d9abf, 1ca2c61, d66cd0c, 8e78b78, 430497d (step 3); the board's Knowledge
    section 13c5551, 9d55930; the sign-in in front of it 2f4a614. Step 0: e41d481, 3d52130.
  - wdstore (cubelang-cowork, branch `cowork/wd-store`, local, BUSL-1.1): 842994a, 52e7bdd, 07b0c80, adc82b1,
    d6e3cce, 7600e51, adce34a, 0c02844, 057bb39, 74c277e, 3589c23, 36dc8e1, 269964a, 2ff5bf7, dccff36, 43c3738,
    6e67789, 0c675a5, 2055bc5, 17fd715, ebb7a3d, b04b36a.
- **Files:** `validation/fact_store/` (the audit, the dump count and its resume, the pool, the cutter, the query-log
  projection, QAWiki's golds, the round-trip check); `notebooks/wikidata_dump.ipynb` (+ its builder);
  `cubby_serve/board/store*.py` and `dashboard/board/store*.js`, `graph.js` (the Knowledge section); the store itself
  in `wdstore/` (FORMAT.md; src: model, turtle, ingest, codec, chunk, nodes, reverse, names, spill, writer, resume,
  build, bz2par, store, snapshot, asof, serve, export). Logs: `validation/logs/wikidata_json_counts_2026-11-06.json`,
  `wikidata_kgqa_counts_2026-11-10.json`, `wikidata_query_logs_projection_2026-11-10.json`,
  `wdstore_roundtrip_rdflib_2026-11-10.json`, `qawiki_store_resolve_2026-11-10.json`, `wdstore_bench_2026-11-10.txt`.
- **Tests (run 2026-11-10):** CubbyLLM `tests/validation/test_fact_store_*.py` (the audit, the count and its resume,
  the pool, the cutter, the Turtle count with redirects, QAWiki, the query logs): 33 passed, 2 skipped (Linux-only
  pipe tests, run on Colab); `tests/serve/test_board_store.py` passes. wdstore `cargo test`: 21 pass, among them
  `no_stored_value_is_a_float`, `a_name_finds_every_entity_it_may_mean_in_id_order_exact_then_folded_then_typo`
  (never by fame), `a_built_store_reads_back_what_was_ingested_and_a_rebuild_writes_the_same_bytes`,
  `a_stopped_build_resumed_writes_the_same_store_as_one_never_stopped`, `one_changed_byte_fails_the_check`,
  `a_triple_the_dump_writes_twice_is_kept_once`, the snapshot's and `serve`'s as-of tests. The round trip against
  rdflib: 100,000 entities, 0 differences.
- **Measured:** <against each acceptance line>
  - **Step 1, Wikimedia's JSON dump** (wikidata-20261026, counted before the source moved to the challenge's dump;
    `validation/logs/wikidata_json_counts_2026-11-06.json`):
    - 122,042,968 items and 14,017 properties; 1,845 GB of JSON, one pass in 3 h 7 min on a Colab CPU (164 MB/s).
    - 1,784,307,309 statements on items: 1,765.5M normal, 6.6M preferred, 12.5M deprecated.
    - Scholarly articles: 46.0M items holding 987.0M statements (55%); the rest hold 797.3M.
    - Main values by datatype: item 886.3M, string 330.6M, external id 271.3M, quantity 106.8M, time 75.9M,
      monolingual text 68.5M.
    - Qualifier snaks 458.8M (37.8M time); 1,334M references attached, about 151M distinct (estimated).
    - Labels: EN 93.2M, FR 18.5M, `mul` 20.7M; enwiki 10.4M, frwiki 3.5M sitelinks.
    - 1% sample: 1,221,273 entities (1.68 GB gzipped).
    - Not yet checked: the parts' sha1 against Wikimedia's. The download's check stopped on a part the Drive showed
      late; the count started on code that did not check its own input (fixed in `9fbc251`). The check runs next.
  - **Step 1, the challenge's dump (Turtle), read from its first 57 MB:**
    - Its date is 2025-12-29T23:00:38Z (`wikibase:Dump`'s `schema:dateModified`): the challenge's base version.
    - The dump writes a shared value or reference node once. About half the value and reference nodes the sampled
      entities use (312 of 723 values, 334 of 684 references) are written elsewhere. So a store built from the 1%
      sample alone lacks those nodes (time precisions, quantity units); the full pass counts this.
  - **Step 1, the challenge's dump, the full count (notebook cell B2), first run (2026-11-06):** stopped at 961 GB and
    116,245,182 entities after 3 h 20 min (80 MB/s; its `progress.json` on the Drive), with no summary and nothing
    kept. The count is resumable since this report's line (below); B2 runs again.
  - **Second and third runs (2026-11-09):** the second checkpointed every 15 minutes to 920.6 GB (111,037,712 items,
    1,186,646 sampled) and stalled at about 958 GB; the third resumed from 920.6 GB (skipping at 150 MB/s) and stalled
    at 957 GB. A fourth (after the cutter's change below) resumed and stopped writing progress at 948 GB, short of the
    same place. Same place each time: the dump's last byte (below). Fixed; B2 resumes from 920.6 GB.
  - **Step 1, the challenge's dump, counted (B2, 2026-11-10, 02:26)** (`validation/logs/wikidata_kgqa_counts_2026-11-10.json`):
    113 parts, md5 as Zenodo's, the stream ended at its closing `.`, the cutter's fallback never used; 971.8 GB of
    Turtle; the dump's date 2025-12-29T23:00:38Z. 118,589,950 items and 13,142 properties; 1,730,739,604 statements
    on items (normal 1,712.3M, preferred 6.4M, deprecated 12.4M; 1,730.0M with a value, 0.86M some value, 0.20M no
    value); scholarly articles 45,397,160 items with 972,433,809 statements (56%), the rest 758,305,795; qualifier
    snaks 434.2M (35.3M time); references attached 1,303.5M, about 137.7M distinct; labels EN 91.1M, FR 18.2M, `mul`
    17.9M; enwiki 10.1M, frwiki 3.4M sitelinks; 140.6M value nodes. **Redirects: 4,779,630** (revisions -
    entities; every entity and every redirect has one revision; the count's own `redirects` field says 0 by the
    bug fixed in f3c3771). Against Wikimedia's JSON dump ten months later: 3.5M fewer items, 53.6M fewer statements.
    At 82 bytes a statement (format 3 on the dump's last part), the hot store is about 62 GB.
  - **Step 2, the full build (2026-11-10)** (format 3, wdstore 43c3738 with mimalloc; the manifest beside the store):
    - Read: the 113 parts (121.3 GB of bzip2, md5 as Zenodo's, re-checked after the copy to the data disk), 123,382,723
      blocks, **0 failed**.
    - **Entities: 73,205,306 hot and 45,397,786 cold** (the scholarly articles), 118,603,092 in all. Statements
      758,600,624 hot and 972,443,219 cold, 1,731,043,843 in all: 3 more than B2's count, whose Turtle reader counts
      a `p:` link written twice once (the store keeps each triple once since 3589c23). Qualifiers 36.1M hot (the time
      ones) and 398.1M cold; references attached 1,303.5M; 139,262,077 reference nodes and 69,587,410 value nodes,
      **0 undefined** of either; 310.2M names; **4,779,630 redirects** (B2's figure exactly); 29.7M reverse keys;
      67.0M exact and 65.4M accent-folded name keys.
    - **Size: hot 77.24 GB** (chunks 37.4, nodes 33.1, indexes 6.7), **under the 90 GB line**; cold chunks 47.0 GB
      on the hard drive. Bytes a statement: 49.3 in the hot chunks, 101.8 with the nodes and indexes (the nodes are
      the whole dump's, shared with the cold set); 48.3 in the cold chunks.
    - **Time: about 3 h 10 min of building** (three sessions, resumed from checkpoints: 1,661 s, 7,950 s, 1,808 s;
      4 h of wall time with the stops). **Peak RAM about 26-27 GB** (read from Task Manager: the runner's own
      recorder wrote 0.00, a bug in its script; the store's code has no limit set). The stops: the session ran
      inside Claude Code's process and its memory guard ended it at 95% of RAM; run outside it, the system disk's
      page file then filled C: (8.8 GB left), so the build moved to the data disk and the hot files were copied to
      the NVMe after (robocopy, 855 files, 73 GB; build logs and the per-number logs left behind).
    - **The dump's date in this store's manifest is wrong:** 2025-12-31T06:44:46Z, the last shard's header; the dump
      is 8 shards with a header each, and its date is the first's, 2025-12-29T23:00:38Z (B2's). Fixed for the next
      build (6e67789); this build's manifest still says the 31st, and the board's stats strip shows it (Questions).
  - **The check: clean** (`wdstore check`, 2026-11-10 09:01-10:26, 1 h 25 min, about 31 GB of RAM at its peak): every
    chunk re-encoded to the same bytes (843 hot, 587 cold), every index the entities' own, every hash as the manifest
    records it, 0 undefined value or reference nodes. This is the acceptance's "a rebuild gives identical chunk
    hashes", done on the full store.
  - **Lookup latency, the full store** (`wdstore bench`, about 20,000 ids spread over the store, each read first from
    disk, then cached; three spreads that share no ids; `validation/logs/wdstore_bench_2026-11-10.txt`). The hot store
    on the system NVMe (WD SN750), the cold set on the data disk (a 5,400 rpm laptop hard drive):
    | | p50 | p95 | p99 |
    |---|---|---|---|
    | a fact by id, hot, from SSD | 409-423 us | **1,530-1,554 us** | 2.6-2.8 ms |
    | a fact by id, cold set, from the hard drive | 9.9-10.1 ms | 16.6 ms | 18.4-18.7 ms |
    | a fact by id, cached (either) | 5 us | 13-14 us | 20-25 us |
    | a name, from SSD | 203-371 us | **744-1,140 us** | 0.9-1.6 ms |
    | a name, cached | 2 us | 4-5 us | 15-24 us |
    - **Line missed: a fact lookup from SSD is 1.5 ms p95, the line 1 ms.** Stable across the three spreads. An
      entity read touches several pages that are not next to each other (the id table, the chunk's record, then
      each value and reference node it names, in the node files), each a random 4 KB read at queue depth 1; the
      large entities' tail sets the p95. Not changed here; the levers, for Cowork and Nick: read the node pages of
      one entity together (prefetch, or several reads in flight), store each entity's small nodes inline (larger
      store), keep the id table (1.9 GB) and the node files' index in memory, or move to a faster disk. Once read, an
      entity is cached: the board's repeated reads are 5-14 us.
    - **Name resolution: 0.74-1.14 ms p95 from SSD, the line 50 ms: met.**
    - The first run mixed the hot store and the cold set (p95 15.9 ms, the hard drive's seeks); `bench` now reports
      them apart (17fd715).
  - **QAWiki's gold ids against the full store** (`validation/logs/qawiki_store_resolve_2026-11-10.json`): 990,750 ids
    by role (643 in queries, 899 in mentions, 989,208 in answers): **0 missing**; 141 answer ids in the cold set;
    **5 mentions resolve only through a redirect** (Q6958747 -> Q268378, Q9200127 -> Q8191099, Q20937557 -> Q133250,
    Q97453439 -> Q21096945, Q103239474 -> Q52946), each followed by the store and said.
  - **Step 3, wired (2026-11-10)** behind `CB_WD_STORE=1` (off: the server builds exactly the worlds it built before,
    the same `Stores(wiki, hdc)`; the new code is never imported). `cubbyllm/worlds/wd_store_world.py` (the world: a
    subclass of the Wikidata5M world `HdcWorld`, so its namesake rules (0033, 0036, 0037) and its hop are the same
    code; only the reads differ), `cubbyllm/worlds/wd_store_values.py` (how a value is said), `cubby_serve/wd_world.py`
    (its own `wdstore serve` pipe, the manifest's date and counts), `cubby_serve/cli.py` (the world goes ahead of
    Wikidata5M in `Stores`). `cubbyllm/` opens no socket and starts no process: it is handed the pipe's `ask`.
    - **Facts:** the truthy statements (best rank, never deprecated) of the 2,877 properties whose values can be said
      (item 1,717, quantity 679, string 348, time 69, monolingual text 64; 9,886 external ids and the rest left out),
      said exactly: a date at its precision ("11 March 1952", "March 1952", "1952", "500 BC"; a Julian date its year
      only; a decade or vaguer not said), a quantity with its unit's label ("1.96 metre"), English or `mul` text. An
      ended statement is dropped only on a relation whose value changes (the time-sensitive list: spouse, employer,
      position held, ...), which the loop says "as of 2025"; "educated at" stays true after the degree. Q42 picked:
      67 facts, among them born 11 March 1952 in Cambridge, died 11 May 2001, educated at St John's College and
      Brentwood School, 1.96 metre.
    - **The store side** (wdstore ebb7a3d, b04b36a): `get` with `"nodes": true` gives the value nodes (a time's
      precision and calendar, a quantity's unit), and `brief` a property's datatype; the world loads all 13,099
      properties in about 0.5 s (2.6 s cold).
    - **Names, as the Wikidata5M world matches them:** label tier, then alias tier; "the" + the name (the planner
      drops the article); an accent or punctuation form kept only when the item's own name normalizes to the key;
      never the store's typo path; a Wikimedia page (disambiguation, category, template, list) is never an item.
    - **Found: full Wikidata shares most well-known names.** "Douglas Adams" is 10 people (and a disambiguation
      page), "Ada Lovelace" 3 items, "Angela Merkel" 6, "Marie Curie" 17, "The Beatles" 20 under the label tier;
      "Paris" is 248 items (24 paintings, 23 photographs, 10 albums, 7 US towns ...), "Canada" 50 and "Albert
      Einstein" 28, all above the 25-item cap. Under the strict rule (carried over unchanged), such a name gives none of its
      items' facts and the loop asks which one (or, above the cap, lists none); a name only one item has works
      ("Justin Trudeau": 74 facts), as do a Q-id, the asker's choice and a qualifier. So by name alone the store
      answers far less than its size suggests; step 4 measures how much, and the rule is a question below.
    - Times on the full store: a name 10-300 ms (the first read of an item from disk is most of it), Q90 (Paris,
      212 facts) 195 ms; before the Wikimedia-page filter and the cap, "Paris" took 31 s (every one of 248 items read).
    - Tests: `tests/worlds/test_wd_store_world.py` (7, over a fake that speaks `wdstore serve`): values said exactly,
      names by tier with the article, punctuation and never a typo, a shared name read only once chosen, a crowded name
      read and listed none, an object's facts its subjects' own, the date and time-sensitive relations, the flag on
      with no store stops the server. tests/worlds, tests/reasoning, tests/serve: 474 pass (the known red, 0002, as
      before).
  - **The dump date corrected in place** (Nick, 2026-11-10): both copies' manifests (the NVMe store and the build
    folder) now say 2025-12-29T23:00:38Z; only that field changed (checked by a diff), the originals kept aside on
    the data disk. The snapshot hash is the same (8a2041cff56d...): it covers the files' names and hashes only, not
    the manifest's other fields, so this report's question was wrong to say it would change.
  - **The first reader:** the board's Knowledge section (audit view only: the store names its dataset) reads the full
    store through `wdstore serve`: an entity 341 ms with its labels (the board's own round trips), the 12,937,830
    entities with P31 = Q5 counted in 7 ms, a two-step path Q42 -> Q35 in 9 entity reads.
- **Not verified here:**
  - **1,000 items re-fetched at their recorded revision agree:** not run (it reads live Wikidata, by revision).
  - **Name recall, EN and FR, on the audit's entities:** not run on the full store. The bench resolves each sampled
    entity's own first label (10,062 to 10,098 a run, 100% found), which is latency, not recall.
  - **The audit's split per set** (the 1,000 questions read from the dump) and **every fact of step 0's live sample
    found in the store:** not run yet; both read the store and go with step 3's wiring.
  - **With the flag off, today's answers unchanged; with it on, 0 wrong on the batteries; 0054's rerun:** steps 3-4,
    not built.
  - **The cold tier is not HDT.** The order's step 2 names a cold tier in HDT (qEndpoint); what exists is the
    scholarly articles' cold set in the hot store's own format, on the hard drive, which the store reads in-process
    with no other tool. The full Wikidata in HDT is not built. A change to the order: Cowork's call.
  - The parts' sha1 against Wikimedia's JSON dump (step 1's first source): not checked; that dump is no longer the
    source.
- **Found on the way:**
  - **The count lost the whole pass when the runtime dropped,** and Colab's Drive mount uploads a file only once it is
    closed: the first run's 1% sample, written into one file left open for 3 h 20 min, never reached the Drive (the
    side file shows there at 0 bytes). `validation/fact_store/dump_resume.py` now checkpoints the count every 15
    minutes (`resume.json`: the decompressed offset at a cut between entities, the counters, the HLL registers), and
    the sample and side files are written in closed segments, staged on the local disk and copied whole. A rerun
    decompresses up to the checkpoint without counting, then goes on; the parts' hash still covers every byte. Tests:
    a pass stopped after a checkpoint and resumed gives the same counts, hash, sample and side files as a pass never
    stopped, on both formats (`tests/validation/test_fact_store_resume.py`).
  - **The stall is at the dump's last byte: the workers held lbzip2's input pipe open.** The pool's workers were
    forked after the pipe into lbzip2 was opened, so each held a copy of its write end; when the feeder closed the
    parent's copy, lbzip2 still had writers, never saw the end of its input and never ended, and the pass waited on
    it with every entity counted. The JSON pass (2026-11-06) ended because it predates the feeder (the pipe came with
    the parts' hash); the tests run on Windows, where nothing forks. `dump_count.pool` now starts the workers from a
    forkserver, which holds none of the parent's files. Tests (`tests/validation/test_fact_store_pool.py`): the pool
    never forks; on Linux, a pipe fed and closed while the pool runs ends, and with a forked pool it does not (the
    cause, pinned). B2 runs them on Colab before the count.
  - **The dump's tail is ordinary entities.** Its last 40 bzip2 blocks (21.7 MB, read from the last part by
    bzip2recover's method) are entities with `data:` headers to the end, the last `Q137645055` (modified
    2025-12-31T06:38:11Z), and the file ends with the stream's end marker. So the earlier reading of the stall (a
    header-free stretch past the last entity, its buffer growing without end) was wrong; that cutter change stays as
    a guard (a fallback cut at a blank line after 128 MB with no header, a stop after 2 GB with no cut; tests in
    `tests/validation/test_fact_store_cut.py`), and `fallback_cuts` in counts.json says whether it ever fired.
  - **The dump is 8 shards one after another, split by the hash the sample uses.** Wikibase's dumper puts an entity in
    shard `int(sha1(id)[:8], 16) % 8`; the sample takes `int(sha1(id)[:8], 16) % 100 == 0`, which is only ever shard 0
    or 4. So the sampled entities sit in two stretches of the dump (0-160 GB and about 430-650 GB), and the sample
    segments in between are empty (53 bytes, an empty gzip). Checked: the 17 segments hold 1,186,646 distinct
    entities, exactly the counter, every one in the sample by its hash; segment 2's are all shard 0, segment 8's all
    shard 4. The sample is the 1% hash sample it was meant to be (the same ids as the JSON dump's); only its place in
    the file is uneven. The side file is small for another reason, a bug, below.
  - **B2 counts no redirect, wrongly; the dump has them.** The dump writes a redirect as its own block: a `data:`
    header for the old id (with a revision), then `wd:Qold owl:sameAs wd:Qnew`. The Turtle count took the old id for
    the block's entity and skipped the line (43 such lines in the dump's last 21.7 MB alone). Fixed
    (`ttl_count.count_block` reads `owl:sameAs` first; the test's redirect is now in the dump's layout). The running
    B2 keeps the old code, and its resume does not recount the first 920.6 GB, so its `redirects` stays 0; the
    count is still in it: every entity and every redirect has one `data:` header, so redirects = revisions -
    entities, 115,664,225 - 111,050,029 = **4,614,196 at the checkpoint**. The list itself comes from the store's
    full build, which reads every block.
  - **A stopped notebook cell left its script running** underneath (`subprocess.Popen`); the cell now kills it.
  - **The hot store by real use (Cowork's addendum), first projection** (`validation/fact_store/query_logs.py`;
    `validation/logs/wikidata_query_logs_projection_2026-11-10.json`). Licence: the logs are CC0 (TU Dresden: "All of
    the above are published under License CC-0"); Freiburg's code is Apache-2.0, and its processed `organic.tar.gz`
    (20 MB) states nothing further. Read: 859,305 distinct queries (2017-06-12 to 2018-03-25), the queries only;
    3,618 properties named, 3,364 of them in the challenge's dump; 89,476 queries name none. Statements per property
    from B2's checkpoint at 920.6 GB (1,639.9M statements, about 95% of the dump; rerun on the final count).
    | hot properties (by queries using them) | statements hot | share | queries whose properties are all hot |
    |---|---|---|---|
    | 100 | 858M | 52% | 74.8% |
    | 300 | 1,198M | 73% | 87.7% |
    | 500 | 1,357M | 83% | 92.9% |
    | 1,000 | 1,414M | 86% | 97.5% |
    | 2,000 | 1,477M | 90% | 99.5% |
    Ranking by use alone shrinks the store little: the scholarly properties are large and rarely asked. P2860 (cites
    work) holds 306M statements (19%) and is used by 5,579 queries (0.65%; rank 41); P2093 (author name string)
    141M, 1,128 queries (rank 148). Which of them go hot is the size line's call (Questions).
  - **Step 2 begun: the hot store, format 1** (`wdstore/` in the cubelang repository, branch `cowork/wd-store` of
    the `cubelang-cowork` worktree, commit 842994a, local). **Where it lives (the order's choice for Claude Code):** a
    crate of its own beside the VM's datastore, with its own workspace and target folder, so cubelang's
    `target/release` is never touched; the VM depends on it in step 3. Ids are Wikidata's own numbers (no id map,
    no ordering but the id). Every statement at every rank with its exact value (no float type in the code; a test
    fails on one), valid-time qualifiers hot and the rest apart, references, each entity's revision (recorded
    time), EN/FR/mul names, enwiki/frwiki titles, value and reference nodes, redirects; a rebuild writes the same
    bytes (`wdstore check`); the manifest carries the inputs', chunks' and code's hashes (cached build steps).
    Measured on one segment of the 1% sample: 121,169 entities, 1,108,786 statements, 0 blocks failed, **104.6 bytes
    a statement**, 10 s. That is about 180 GB for the dump, twice the 90 GB line.
  - **Format 2** (52e7bdd, local): statement ids as 16 bytes, every value and reference node once in its own
    numbered file, redirects in a sorted file, `resolve` through them. Same segment: **65.5 bytes a statement**
    (54.8 in the entity chunks), 0 failed, a rebuild byte-identical. About 113 GB for the whole dump; about 50-55 GB
    with the scholarly articles cold, under the 90 GB line. Found on the way: the dump may write one value node's
    triples in two places (a quantity's amount and unit in one block, its `quantityNormalized` link in another;
    2,373 such nodes in the segment); format 1 kept one part only, format 2 merges them by hash.
  - **The reverse index and the names** (07b0c80, then the names commit, local). Reverse: (property, value entity)
    -> the entities that have it, in id order, each hit with its statement's rank (deprecated ones filtered by the
    caller without reading entities). Names: labels and aliases (EN, FR, `mul`) under exact and accent-folded keys
    in two FSTs, then a typo path (one edit from 5 characters, two from 9); candidates in id order, never by fame;
    over 64 near keys it answers "too many", never a guess. Same segment: 80.2 bytes a statement with every file.
    From memory: an entity by id p95 15.4 us, a name p95 5.8 us (the lines, 1 ms and 50 ms p95, are from SSD:
    measured on the full build).
  - **The full build's pieces** (local commits d6e3cce, 7600e51). bzip2 decoded on every core, block by block (the
    dump's last part: 238 MB/s out on 12 threads, byte for byte the earlier decode). Format 3 builds in the dump's
    order without sorting it (the shards' ids rise, but not strictly), with an id table, nodes numbered by first
    sighting and the indexes sorted in runs on a scratch disk; scholarly articles go to a cold set of the same format
    (`--cold`, the synthesis's default). On the dump's real last part (1.05 GB of bzip2): 1,235,092 hot and 614,755
    cold entities, 16.5M statements, 32,919 redirects resolved, 2 of 1,021,047 references undefined (written in the
    part before), `check` clean; 50.5 bytes a hot statement in the chunks, 82.0 with the nodes and indexes; p95 13.9
    us by id, 4.1 us by name; 135 s, so about 4.2 h for the dump. Projected hot store: about 75-80 GB, under the 90
    GB line.
  - **The build resumes after a stop** (adce34a, local): every file append-only, a checkpoint every 32 batches of
    64 MB of text (each file's length, the bzip2 block the text resumes from), never by the clock. On the dump's last
    part, a build stopped after 8 checkpoints and run again gives the same 72 files, byte for byte, as one never
    stopped; 87 s, so about 2.7 h and about 135 GB written for the whole dump (hot, cold, nodes, indexes, logs).
    Nick's yes (2026-11-10): parts copied to a local data disk, built there, hot files to the system NVMe,
    scholarly articles cold. The full run follows the copy.
  - **Acceptance, the second parser: 100,000 entities, 0 differences** (`validation/fact_store/roundtrip_check.py`;
    `validation/logs/wdstore_roundtrip_rdflib_2026-11-10.json`). rdflib's Turtle reader (shares no code with the
    store's), literal normalization off, against `wdstore dump`, field by field, on the first 100,000 entities of
    the sample's segment 0: 1,226,463 statements. Found on the way, on the first 2,000: Q212222's statement lists
    the same reference three times in the dump's Turtle; RDF reads it once, the store kept three; the store now
    keeps each triple once (3589c23).
  - **The snapshot addendum's store side** (36dc8e1, local): `wdstore snapshot` (the base's content hash, the
    change log's length and head, one hash over them; `--at N` pins), a chained change log (each line carries the
    previous line's hash), a read against an unheld snapshot fails with `snapshot_not_held`; the manifest names the
    dump as a release and carries its date. Its four tests pass.
  - **Step 3's store side** (wdstore 269964a, 2ff5bf7, local): `wdstore serve` answers JSON requests on a pipe
    (stdin, stdout; never a socket): snapshot, get (a redirect followed and said; with `as_of`, each statement's
    holds / does_not / too_coarse / undated, from its start, end and point-in-time qualifiers with their precision
    and calendar, Julian dates converted exactly; with a snapshot the store is not in, only `snapshot_not_held`),
    name, who, pointing_at (every entity pointing at a value, whatever the property: a second key table over the
    reverse index's postings), resolve, labels. The store never reads a clock.
  - **Step 3's wiring, the plan** (from a read of the fact path): the walk resolves every hop through
    `Stores.lookup` (`cubbyllm/worlds/hdc_store.py`), wiki world first, then the Wikidata5M world; live Wikidata
    comes only after a refusal (`learn.py`). The store goes in as a read-only world ahead of Wikidata5M, built in
    `cubby_serve/cli.py` behind `CB_WD_STORE=1` (off: today's answers unchanged, bit for bit), handed a query
    function over `wdstore serve` the way the live source is handed its fetch function (`cubbyllm/` opens no
    socket). The Wikidata5M world's namesake rules (0033) read five kinds of query (facts by subject name, by object
    name, the ids a name stands for, an id's label, an object's members under a relation); they go behind one small
    interface with two backends, today's SQLite one unchanged and the store's. Its `snapshot` (the dump's date) and
    `time_sensitive` let the existing rule (`AskMemory._snapshot`, `source_wins`) keep live Wikidata winning on
    time-sensitive facts. Names stay in id order; the live resolver's search keeps its own `limit: 5` (the API's
    relevance order: a fame cut the store's path does not have; noted, not changed here).
- **Questions:**
  - **The hot store's size line.** By real use, 1,000 properties cover 97.5% of the logged queries but hold 86% of
    the statements, because the scholarly properties rank high enough by use (P2860 at 41, P2093 at 148) to get in.
    Should the hot store keep them by rank, or keep scholarly articles' citation and author-string statements cold
    whatever their rank (about 450M statements, 27%)? Nick's or Cowork's call; nothing is built on it yet. (The full
    build took the synthesis's default meanwhile: every scholarly article cold, so the question is now whether any
    scholarly statement should come back hot.)
  - **The 1 ms line.** A fact from SSD is 1.5 ms p95 on this NVMe. Work on the store's reads (the node pages of one
    entity read together, or small nodes inline), a faster disk, or the line moved to what the walk needs (a hop is
    one entity read; the 50-question preflight would say whether 1.5 ms matters next to the rest of an answer)?
  - **Choosing among a name's items in the full store.** Today: the asker, a qualifier, or a name only one item has.
    Options, each needing an order or Nick's yes: (a) a kinds table built from the store's own classes (P31, P279),
    so 0037's relation rule decides here as it does on Wikidata5M (the item whose kind carries the asked relation and
    fits the question's cue words, no other's does); (b) read the name in Wikidata5M first and the store by the one
    Q-id it gives (fresh values for the 2019 world's items), which is a notability filter, so a fame rule by another
    name, against this order's own line; (c) leave it strict and let step 4 say what it costs. Claude Code's view:
    (a), measured as 0037 was (an audit before it speaks).
  - **The cold tier:** the scholarly articles' cold set in the store's own format, read in-process from the hard
    drive (16.6 ms p95), instead of the full Wikidata in HDT. Keep it, or build the HDT as well?

## Cowork's addendum (2026-11-06): the hot store chosen by real use
Nick found the Wikidata Query Logs dataset (Walter and Bast, SIGIR 2026, arXiv 2602.14594; the
`ad-freiburg/wikidata-query-logs` repository). Its organic split holds about 860,000 distinct real queries that
people ran against Wikidata between June 2017 and March 2018, from Malyshev et al.'s 2018 logs. The queries are
human-written; only the dataset's questions are model-written, and those are not used here.
- **First:** the licence of the logs and of the released files is checked and recorded. The paper says "permissive"
  without naming one. If it does not allow this use, this addendum stops.
- **From the real queries, with no question text:**
  - each property and qualifier, counted by how many distinct queries use it;
  - the query shapes, counted.
- **Used for the projection:**
  - the properties and qualifiers real queries use go hot, ranked by use; the rest go cold;
  - reported: the share of real queries whose properties all sit in the hot store, at the hot store's size.
- Nothing else in 0055 changes.
## Cowork's addendum (2026-11-06): QLever as the reference engine
From the Freiburg group's repositories (`ad-freiburg`; Nick: "yes"). Its licence is checked and recorded first; it
is expected to be Apache-2.0.
- **QLever indexes the whole Wikidata dump** in about 4.5 h with about 20 GB of RAM on a desktop CPU, by its own
  Wikidata recipe (`qlever-control`, `Qleverfile.wikidata`).
- **Its two jobs here:**
  - the reference the hot store and the cold tier are checked against: the stream's counts, and a 100,000-entity
    sample's facts, read both ways;
  - the engine 0054's query-log benchmark uses to recompute its gold answers on the same dump.
- **Serving does not change:** the cold tier stays the HDT file read in-process. QLever runs as a server, and the
  serving code never touches the network unless Nick decides a local server is fine.
- **Space:** its index size is measured. If the 1 TB drive cannot hold both it and the HDT tier, Cowork and Nick
  choose one, and the report says what each would cost.
## Cowork's addendum (2026-11-06): the source dump is the WikiKGQA 2026 challenge's own
Nick chose it ("Challenge dump"; his yes for the download): `wikikgqa-wikidata-all-dump.ttl.bz2`, 121.3 GB, CC0,
from QAWiki v2026.3 on Zenodo (DOI 10.5281/zenodo.22661413).
- **Why:** it is the exact Wikidata the WikiKGQA 2026 challenge was scored on, and the record also holds every team's
  submitted answers. Built from it, Cubby is scored on the challenge's test exactly as the entrants were (0054's
  addendum). Every other benchmark runs on the same snapshot too.
- **The trade-off, accepted:** the snapshot is some months old. Live Wikidata still answers time-sensitive facts (the
  standing rule).
- **What changes in step 1:**
  - the stream parses RDF (Turtle), not JSON: statements, qualifiers and references all appear as the statement
    nodes of Wikidata's RDF format;
  - each item's revision is read from its `schema:version`, for the revision cross-check;
  - QLever's recipe and the HDT tools read this format directly.
- **The dump's date** (the challenge's "Wikidata base version") is recorded in the report.
## Cowork's addendum (2026-11-06): the folding step done by an exact rule engine
Nick: "yes". `reasonable` (gtfierro/reasonable; Rust, BSD-3) is an OWL 2 RL reasoner built on DataFrog, a small
Datalog engine. Licences are checked first.
- **The mapping:** Wikidata states its rules in its own vocabulary, not OWL, so they are mapped to OWL RL axioms:
  - inverse properties (P1696);
  - the symmetric and inverse constraints (P2302 with Q21510862 and Q21510855);
  - transitive properties (instances of Q18647515);
  - subclass of (P279);
  - disjoint union of (P2738);
  - any other mapped constraint, recorded.
- **Schema level only, at build time:**
  - the subclass closure, for the VM's single type check;
  - the inverse, symmetric and chain folding of the 1% sample.

  Deriving everything over every item would explode, so per-item checks stay in the VM, at question time.
- **Measured on the sample,** against the hand-written folding:
  - the derived facts must agree exactly, and every disagreement is listed;
  - time and memory;
  - each derived fact keeps its derivation as its certificate.
- **Violations** (disjoint classes, asymmetry) are reported, never repaired by the build.
- **Check before relying on it:** its README marks some consistency rules as raising exceptions. Whether that means
  "reports a violation" or "fails" is checked and recorded. If the OWL layer adds nothing over DataFrog alone, the
  lighter one is kept.
- **Borrow note (Nick, 2026-11-10): HornDB**, a Rust OWL 2 RL reasoner queried with SPARQL; an early prototype (6
  stars). Two things may help the folding step; neither is a dependency:
  - **A conformance suite:** it passes 100 of the 115 W3C OWL 2 RL test cases. The cases are W3C's own (the OWL 2
    test suite), so whichever engine does the folding (`reasonable`, or DataFrog alone) is run on them, and its pass
    count recorded beside HornDB's 100 / 115; a case it fails is listed, never hidden.
  - **Its join method** (said to be fast): an idea to read only if the folding's measured time on the sample is the
    problem. Its licence is checked before its code is read.
  - **Not relied on:** "every inferred fact traces back to its premises" is listed as a design goal, and the page does
    not say it works yet. Each derived fact keeps its own certificate here (above), whatever HornDB does.
  - Nothing is built yet: the folding step has not started.

## Cowork's addendum (2026-11-06): time, change and merged items in the store
Nick: "yes". These ideas come from three open-source systems with the same store design: Semantica (MIT), CocoIndex
(Apache-2.0) and Utopia (Apache-2.0). Only the ideas are taken; no code, no dependency.

**What it brings, in plain words:** Cubby can replay any past answer exactly, never answers from a fact that has since
changed, and still recognises an entry after Wikidata merges it with a duplicate.
- **Over a conventional model:** a model cannot say what it knew last month, nor which of its answers relied on a fact
  that changed.

**What the first build needs, and what can wait:**
- The fields (two clocks, input hashes) go into the store format now. Adding them later means a rebuild.
- Retirement and merges in the overlay matter from the first overlay change.
- Redirects matter for the benchmarks, because gold answers can use old ids.
- None of it delays the base store, which stays one immutable file built from the dump.

### 1. Two clocks per statement
- **Valid time:** when the fact held in the world. Wikidata's start, end and point-in-time qualifiers (P580, P582,
  P585) already carry it, with their precision.
- **Recorded time:** when the store learned it.
  - For the base: the dump's date and the item's revision.
  - For the overlay: when the change was applied, and the revision it came from.
- **A corrected or removed statement is closed, never overwritten.** It gets a recorded end and stays readable.
- **Queries can be pinned to a recorded date:**
  - benchmarks run pinned to the dump;
  - users get the latest.
- **Test:** an answer given before a seeded overlay change is reproduced exactly when the query is pinned to its date.

### 2. Derived facts retired by their inputs
- **Each derived fact lists the statements it used** (folded inverses, symmetric pairs, chains, the subclass closure).
  This is its certificate already.
- **An overlay change to a statement closes every derived fact that used it.** The rules are then re-run for the
  entities involved only.
- **DataFrog only adds facts.** It cannot take one back, so a re-run covers that small set, not the whole store.
- **Test:** after a seeded batch of overlay changes, the store equals a full rebuild from the changed data, exactly.

### 3. Merged items (redirects)
Wikidata merges duplicate items. The old id then redirects to the one that remains.
- **Where redirects come from:** Wikidata's RDF links a redirected id to its target with `owl:sameAs` (the design is
  in Wikimedia's task T93767).
  - Whether the challenge dump carries redirects is checked on the dump and recorded.
  - If it does not, Claude Code reports it under Questions. The source for redirects is then Cowork's call.
- **The store resolves an old id to the item that remains:** at question time, and when reading benchmark gold answers.
- **A merge in the overlay is a recorded change like any other.** It can be undone, and both ids keep answering.
- **Reported:** how many benchmark gold ids resolve only through a redirect.

### 4. Cached build steps
- Each build step's output is stored with the hash of its input, its code and its parameters.
- A step whose hashes all match is skipped. Any change re-runs that step and every step after it.
- The report lists the hashes behind every number: 0054's "frozen and recorded" line, made checkable.

### 5. The change log
- **Append-only:** one line per change to the store. Each line gives what changed, the source revision, the loader's
  code hash and the time.
- **It is the recorded-time side of the two clocks.** It outlives the statements it describes.

### Out of scope
The rest of those systems:
- extraction by an LLM;
- merging duplicates by embeddings or by a model's judgement;
- answers written by an LLM.

## Cowork's addendum (2026-11-09): certificates re-checked by a separate checker, and probes for late facts
Nick: "yes", after the field watch's summary. Each source below was re-read on 2026-11-09.

### 1. A second, independent check of the rules applied (from Open Ontologies, MIT)
**What it brings, in plain words:** a second program, written separately from the reasoner, re-checks every step
behind every derived fact, the way a second accountant re-adds the books.
- **Over a conventional model:** a model's reasoning cannot be re-checked step by step at all.

- **How Open Ontologies does it** (fabio-rovai/open-ontologies; Rust engine, Lean 4 checker):
  - a certificate is two tab-separated files and a digest: `asserted.tsv` (the asserted triples), `derivations.tsv`
    (one line per step: the rule, the conclusion, the premises) and `asserted.sha256`;
  - `oo-cert`, a Lean 4 program that shares no code with the engine, re-applies each rule to its premises, checks
    that every premise is asserted or derived on an earlier line, and refuses a forged step by its line number;
  - its stated limit: a digest binds a certificate to bytes, not to a state of the world.
- **For Cubby:**
  - write the rules-applied part of each certificate in the same shape: one line per derivation, with the rule's id
    and the ids of its premises and conclusion, bound to the store snapshot by its hash;
  - re-check the derivations the folding step keeps (the addendum on the exact rule engine) with a checker that
    shares no code with that engine. Whether it is oo-cert itself (MIT, Lean 4 toolchain) or a small separate
    checker is Claude Code's call, measured;
  - in CheezML (0064), the derivation file is an artefact referenced by its hash, never embedded.
- **Tests:**
  - an unchanged run passes;
  - one planted forged step per rule family is refused, with its line;
  - on every derived fact of the folding step, the engine and the checker disagree 0 times.

### 2. Late facts, retractions and as-of reads: memory-over-time (tonydzi; MIT code, CC BY 4.0 data)
- **What it is:** a benchmark of agent memory over time (paper under review at TGDK), comparing Mem0, Graphiti and a
  bitemporal fact log.
- **Its probes, used as store fixtures:**
  - the room-402 probe: four changes, a late fact and a retraction, with `--broken` controls that must fail;
  - the CRM as-of set: 30 as-of questions over 15 real changes.
- **For Cubby:** the two-clock store answers every probe exactly (an as-of read, a late fact placed where it belongs
  in the history, a retraction closing a version), and the broken controls fail on Cubby's store too.
- **Licence:** the files derived from LoCoMo are CC BY-NC 4.0. They are research only and never go into a release or
  the reproducibility package.

## Cowork's addendum (2026-11-10): the store names its state, for CheezML v0.2
Nick: "Carry it into the VM and store", then "all good" on this addendum as written, acceptance lines included
(2026-11-10). CheezML v0.2 (0064's addendum) records, for every run that read a fact, the
store state it read (`cheez:storeSnapshot`) and the date the question was about (`cheez:asOf`). These are the store's
two clocks seen from the answer: recorded time and valid time. The store has to give both.

**What it brings, in plain words:** every answer names the exact copy of the knowledge base behind it, so it can be
asked again next year against that same copy.
- **Over a conventional model:** a model's knowledge has no version an answer can point to.

**What it costs the first build:** nothing in the base file, whose manifest already carries its hashes. The chain
below is in the change log, which starts with the first overlay change.

### 1. A snapshot hash
- **The change log is chained:** each line carries the hash of the line before it, so the last line's hash stands
  for the whole log.
- **`wdstore snapshot` gives** the base manifest's hash, the change log's length and its head hash, and one snapshot
  hash over the three. With no overlay change, the snapshot is the base's.
- **A pin to a recorded date** resolves to a position in the log and gives that position's snapshot hash. The
  benchmarks' pin (the dump) is the base's snapshot.
- **A read against a named snapshot** reads that state or fails with a code. It never reads the latest instead.

### 2. One snapshot per question
- The host's walk and the VM's run for one question read the same snapshot. The host passes it to the VM as
  `expect_snapshot` (cubelang P4-10), and a mismatch refuses the run.
- A fact admitted between two runs (0007's gate) makes a new snapshot. The second run names the new one.

### 3. The as-of date
- Every read takes the date the question is about, passed explicitly. The store never reads the clock itself.
- A statement is returned as holding only when its valid time covers that date. "Dates too coarse" stays a refusal
  reason, counted apart (0054's as-of sets).
- A question that names no date uses the time it was asked, and the record says so. On a store pinned to a recorded
  date (the benchmarks, pinned to the dump), it uses that date, so a later run gives the same answer.

### 4. The releases behind the store
- The manifest names the dump as a release: its name, its version (the dump's date and id) and its hash. 0064's
  writers cite it as a `cheez:DatasetRelease`.
- The folding rules are cited by their code hash, which the manifest already holds. Ontology releases (FIBO, AUTO)
  come with the domain worlds (0058).

### Tests
- An identical rebuild gives the same snapshot hash.
- One overlay change gives a new head hash, and a read pinned to the old snapshot returns exactly what it did before.
- A read pinned to a snapshot the store does not hold fails with a code.
- One line changed in the middle of the change log makes the chain check fail.

### Acceptance (added by this addendum)
- **Every answer in 0054's replay** names one snapshot hash and one as-of date: 0 runs without both.
- **Replay:** 100 answers run again against their recorded snapshot and as-of date give the same answer and the same
  report hash.
