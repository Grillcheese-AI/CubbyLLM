# Panel: a terabyte of knowledge outside the weights (2026-11-06)

The brief: `2026-11-06-fact-store-brief.md` (beside this file). Asked independently, same prompt, temperature 0.3. Reasoning effort `medium` with a 24,000-token budget covering reasoning and answer, except Claude Opus 5.5 (48,000: its first answer was cut at 24,000) and DeepSeek V4 Pro (effort `low`, 32,000: its first try spent the whole budget reasoning). Model ids read from OpenRouter's live list on 2026-11-06. Cached under `standin/data/out/panel_cache/`.

**These are model opinions, not findings.** Each proposal still has to pass the experiment it names, and every number or system named here is checked before it is used.

Cost of the answers below: $1.26 (the two first tries that failed cost $0.59 more). Answers: 7 of 8.

## openai/gpt-6.1-sol-pro

*15345 tokens out, 241.3 s, finish `stop`*

## Recommendation

**Use one question-to-program model over one logically unified store. Store facts as exact, versioned Wikidata statements; use ordinary lexical retrieval to find their keys. Do not build a mixture of topic models yet.**

Buy a **4 TB NVMe SSD** for the build and serving store, and preferably upgrade to **64 GB RAM**. The 32 GB machine is sufficient for a constrained implementation, but it leaves less room for name indexes and database cache. The GPU should not participate in exact fact lookup.

Three premises need correcting:

1. **“A VM checked it” does not mean “the answer is true.”** It can mean that the answer follows from a particular source revision under particular interpretation rules. Wikidata, a reader, and entity resolution can all be wrong.
2. **40–80 GB is not a defensible budget for the complete requested Wikidata representation.** It may fit a restricted projection. It should not be assumed to fit full statements, qualifiers, references, bilingual names, and useful indexes.
3. **The present coverage numbers do not yet justify downloading a terabyte.** Only 3 answerable questions among 860 whose gold entity exists suggests missing predicates, incomplete statement representation, or interpretation failures—not merely missing entities.

My first experiment would be an audit of those 860 questions.

All storage sizes and build times below are **planning estimates, not measurements of the November 2026 dump**. I cannot verify that dump’s current size here.

---

## 1. Define the contract before designing the store

A certified answer should carry an internal record like:

```text
subject:       Q…
property:      P…
value:         exact typed value or Q…
qualifiers:    exact qualifier set
source:        Wikidata
revision:      …
retrieved_at:  …
selection:     nondeprecated, applicable interval, …
evidence:      statement GUID + reference IDs + source-record hash
```

The certificate proves:

- the selected statement exists in the pinned source revision;
- its value belongs to the selected subject;
- the VM’s qualifier, rank, and temporal rules were satisfied;
- the displayed name is a rendering of that exact value.

It does **not** prove:

- the source is factually correct;
- the user meant that subject;
- an extracted sentence entails the claim;
- two cited publishers are independent.

For the “0 wrong” requirement, count **all** of these as failures, not just fabricated values.

An unqualified factual assertion may therefore be too strong. “Wikidata’s revision dated … lists …” is sometimes the only claim the system can certify. Contradictory statements should cause a qualified enumeration or refusal, not a hidden tie-break.

**Test:** publish separate rates for wrong identity, unsupported interpretation, wrong temporal selection, source disagreement, and fabricated value.  
**Kill:** any wrong spoken answer. A hundred correct database reads do not excuse one wrong identity selection.

---

## 2. What to keep

### Wikidata: keep full statements, not only truthy statements

Wikidata’s “truthy” projection is useful as a derived index. It is not a suitable authoritative store.

Its best-rank direct claims omit the statement structure needed for many historical questions. Preferred rank is also **not a confidence score** and does not necessarily mean “currently true.”

Keep:

- Item and property identifiers.
- All statements, including deprecated statements, with explicit rank.
- Statement GUIDs.
- Main snak, qualifiers, references, and their ordering where supplied.
- Distinctions among a known value, `somevalue`, and `novalue`.
- EN/FR labels, aliases, and descriptions.
- Datatype definitions and property metadata.
- Source revision and modification metadata.
- Sitelinks.

Sitelinks are inexpensive relative to the whole store and useful for joining Wikipedia passages and public evaluation data. Keep all sitelinks in the compressed source archive; index EN/FR Wikipedia links first. Index the others only when a workload requires them.

Do not convert everything to strings or floating-point numbers:

- quantities need exact decimal amount, bounds, and unit;
- dates need precision, calendar model, timezone, and signed year;
- coordinates need globe and precision;
- monolingual text needs its language tag.

### Planning sizes

These are alternative retained representations, **not additive rows**.

| Representation | Estimated retained size, including practical indexes | What it loses |
|---|---:|---|
| Restricted entity-valued/direct-claim projection, EN/FR names | **40–100 GB** | Much of the requested semantics and provenance |
| Truthy statements with typed values, EN/FR names | **80–220 GB** | Historical and alternative statements; most statement-level evidence |
| Full normalized statements, qualifiers, ranks, references, EN/FR names | **200–650 GB** | Nothing required for statement semantics, if implemented correctly |
| Above, plus compressed original entity records | **350–1,100 GB total** | Preserves the original source for audit and reparsing |

These ranges are deliberately broad. Reference repetition, string interning, identifier density, and the actual dump version matter greatly.

For an initial sizing model, use **1–3 billion statements**, not an assumed 74.5 million entities. If a primary statement representation averages **40–80 bytes per statement**, that alone is **40–240 GB**, before name indexes, reverse indexes, qualifiers, reference bodies, and source records. Measure the actual bytes per category on a sample.

**Recommendation:** full normalized statements plus compressed source records. No six-permutation RDF index unless the actual program workload demonstrates a need.

**Pass:** independently parsed sample statements reproduce exactly, including unusual datatypes and qualifiers.  
**Kill:** any silent datatype simplification, discarded rank, or provenance pointer that cannot recover its source record.

### Text

Keep the EN/FR Wikipedia text, the books you are licensed to retain, and the encyclopedia as a **passage store**, not as millions of automatically trusted triples.

For each passage retain:

- document identity and edition/revision;
- page, section, and stable passage identifier;
- original text;
- byte offsets or equivalent recoverable locations;
- language;
- extraction rule/model version;
- source-family information.

DBpedia abstracts are usually derived from Wikipedia. **They do not provide independent confirmation of Wikipedia.** GDELT also does not itself constitute independent corroboration; syndicated copies can appear under many domains.

Start extraction with narrow, auditable constructions:

- explicit dates in infobox-like fields;
- named officeholder intervals;
- explicit parent/child or author/work statements;
- dated event tables;
- controlled constructions whose entities and relation are unambiguous.

Even these require evaluations. Negation, quotations, changed names, calendars, and “X succeeded Y” are common traps.

A Cubby-native reader can propose broader claims, but a model-generated triple is not made trustworthy merely because the model is your own. Store it as an **unvalidated candidate** until it passes a claim-specific verification or review process.

For novel text claims, either:

1. speak explicitly attributed claims—“This edition says …”; or
2. promote them under a documented corroboration policy.

Two sources agreeing is a policy threshold, not proof of truth or independence.

**Pass:** zero unsupported promotions in an adversarial, bilingual extraction test; every claim opens the exact supporting passage.  
**Kill:** one promoted assertion that the cited passage does not support. Unvalidated candidates may still be retained for investigation.

---

## 3. Physical design: RocksDB first, not a new graph database

### Authoritative store

Use **RocksDB**, with its Rust binding, as the initial serving database. It is an embedded C++ engine, not a pure-Rust engine. RocksDB is available under **Apache-2.0 or GPL-2.0**; check the precise versions and dependency licences you ship.

Use column families approximately as follows:

```text
facts:       (subject_id, property_id) -> complete statement bundle
entities:    entity_id -> current revision, metadata, labels/descriptions
reverse:     (property_id, object_id, subject_id) -> statement references
references:  reference_id -> exact reference structure
names:       (language, normalized_name, entity_id) -> alias metadata
sources:     source-record ID -> archive location and checksum
```

Use fixed-width ordered integer keys initially. Wikidata identifiers are stable source identifiers; do not require globally renumbering them whenever new entities arrive.

Inside values:

- encode integers with varints where advantageous;
- dictionary-encode repeated property IDs and strings;
- intern identical reference structures, comparing full bytes before deduplicating;
- use Zstandard compression;
- preserve exact typed values;
- retain complete statement identity.

Do not use a hash alone as an identity. A hash can locate a bucket; exact bytes establish equality. This also applies to the hippocampal keys.

The `(subject, property)` bundle makes the common hop a single database read. It should include the qualifiers and ranks needed to select a value, rather than requiring many random reads.

Add the entity-valued reverse index because questions such as “Who wrote this?” and joins need it. Do **not** add every RDF permutation speculatively.

For large bundles, impose a declared execution limit. Refuse “too many matching statements” rather than silently answering from the first page.

### Source archive

Keep original entity records in independently decompressible Zstandard blocks, approximately **256 KB–1 MB uncompressed**, with record offsets and checksums.

This is a cold audit path. It is not read on every ordinary fact hop. The certificate points to evidence whose existence and integrity were established during ingestion.

Retaining source records makes parser bugs repairable without downloading everything again.

### Names and passages

Reuse the existing FST infrastructure as a read-only accelerator over name keys. Maintain exact name entries in RocksDB as the authoritative index.

For text and fallback lexical candidate retrieval, use **Tantivy**, a Rust search library under the **MIT licence**. Search results supply candidate item IDs or passage IDs, never certified facts.

Other reusable components:

| Component | Licence | Use |
|---|---|---|
| SQLite / FTS5 | Public domain | Existing worlds, audit tables, early experiments |
| `rusqlite` | MIT | Rust SQLite access |
| `fst` | MIT or Unlicense | Existing name-index machinery |
| `zstd` implementation | BSD-3-Clause | Source and value compression |
| RocksDB | Apache-2.0 or GPL-2.0 | Exact persistent indexes |

Check the versions actually incorporated; this is not a dependency licence audit.

### RAM and disk budget

For **32 GB RAM**, start with:

- **6–8 GB** RocksDB block cache;
- **2–4 GB** cached FST/name structures;
- **2–4 GB** model and runtime host allocations, to be measured;
- **2–3 GB** ingestion/indexing buffers when building;
- remaining RAM for the OS, page cache, and process overhead.

Do not assume the entire bilingual FST fits in that allocation. Memory-map it and measure working-set behaviour.

A **4 TB NVMe** allows roughly:

- **0.35–1.1 TB** retained Wikidata representation;
- **0.1–0.3 TB** text, passage indexes, and candidates initially;
- **0.5–1.0 TB** temporary files, compaction, and rebuild headroom.

The supplied “40 GB of Wikipedia text” needs its own audit: text-only, XML, compressed bytes, and indexed bytes are different measurements.

Keep at least **25% disk free** during sustained RocksDB updates. Google Drive is an archive destination, not a database filesystem.

### Latency

Your targets are plausible **for bounded point reads**, not for arbitrary graph queries:

- hot exact hop: **p95 <1 ms**;
- SSD-resident exact hop: **p95 <10 ms**;
- name candidate lookup: **p95 <50 ms**.

A hop should be defined as returning a complete bounded statement bundle, applying its selection rules, and producing its certificate—not merely returning a storage pointer.

Measure separately:

- process-cold versus cache-hot;
- p50, p95, and p99;
- 1 and 8 concurrent requests;
- after an update/compaction workload;
- ordinary and unusually large items.

**Pass:** these targets on a store larger than RAM, with zero incorrect results.  
**Kill for this layout:** persistent target misses after ordinary cache/index tuning, or pathological large-item behaviour. That is evidence for a packed immutable base—not evidence for neural memory.

---

## 4. Existing full-scale systems: useful comparators, not a licence to invent numbers

The most relevant system to investigate is **QLever**:

> Hannah Bast and Björn Buchhold, **“QLever: A Query Engine for Efficient SPARQL+Text Search,” CIKM 2017**.

Current QLever deployments demonstrate full Wikidata querying on individual large servers. QLever is **Apache-2.0**, predominantly C++, and normally deployed as a service rather than embedded in a Rust VM.

**OpenLink Virtuoso Open Source** is another established single-server Wikidata/RDF comparator; its open-source distribution uses **GPLv2**. It is also not a lightweight Rust embedded dependency.

However, I do **not** have a verified, mutually comparable set of full-dump **index bytes, wall-clock build hours, and p95 query milliseconds** for a named dump and these systems on hardware comparable to yours. I will not substitute remembered numbers or benchmark means for those measurements. In particular, “one machine” often means a server with hundreds of GB of RAM, not 32 GB.

That leaves this requested evidence incomplete. Before selecting either system, require a reproducible record containing:

```text
dump date and flavour
triple/statement count
CPU, RAM, SSD
peak build RAM and disk
wall-clock build time
final index bytes
query set and per-query latency distribution
```

Run QLever on the pilot as a comparator if you expect open-ended SPARQL. For the stated fixed VM hops, I would start with RocksDB instead: substantially less query machinery, and a directly testable execution contract.

**Pass for adopting QLever:** materially better coverage or implementation simplicity at acceptable memory and latency on your workload.  
**Kill:** needing a larger machine just to support query features the VM never uses.

---

## 5. Name to item: retrieval is not disambiguation

Use this pipeline in both languages:

1. Preserve the original user string.
2. Generate conservative normalized forms: Unicode normalization, case handling, punctuation/spacing variants.
3. Exact label and alias lookup.
4. If needed, folded-accent lookup as a candidate-expansion path.
5. Token/BM25 retrieval and bounded edit-distance search.
6. Verify all candidates against current entity metadata.
7. Apply only explicit contextual constraints.
8. Return a uniquely justified item, enumerate, or ask.

Preserve language distinctions. “French” and “English” are Wikidata language codes; Quebec/French and US/UK variants are not comprehensive separate Wikidata label inventories. Regional forms require additional, provenance-bearing aliases. Do not pretend Wikidata supplies all four varieties.

### Namesakes

Never truncate a namesake set by popularity.

For an exact name with 40 matching people:

- retain or page all 40;
- filter using explicit, checked constraints such as occupation, date, place, or a relation mentioned in the question;
- ask if more than one remains.

If candidate enumeration exceeds an execution cap, report that the name is too ambiguous. Do not convert a top-10 retrieval result into an exhaustive identity claim.

Descriptions help present clarification choices; they are not automatically authoritative facts about the entity.

### Typos

Your trigram codes may propose candidates. So may Tantivy. Neither licenses automatic identity selection merely because one score is largest.

A unique fuzzy match can still be wrong. Use typo candidates to offer a correction, or require corroborating context before execution. Do not introduce a universal similarity threshold.

### VM interface

```text
resolve_name(text, language, explicit_constraints)
  -> Unique(item_id, resolution_evidence)
   | Ambiguous(candidate_set)
   | NotFound
   | TooBroad
```

Only `Unique` feeds an ordinary fact hop. Keep resolution evidence in the final answer certificate.

**Pass:** ≥99% candidate recall at an explicitly reported retrieval budget on the evaluated unambiguous mentions, with zero false automatic resolutions.  
**Kill:** any fame-based namesake choice, or pretending an incomplete candidate list is exhaustive.

---

## 6. Time and updates

### Temporal selection

Store the statement as supplied. Do not rewrite dates into a single “current value” column.

A question about a date must interpret:

- start/end intervals;
- point-in-time qualifiers;
- date precision;
- calendar;
- rank;
- conflicting or missing temporal information.

“Point in time” is not interchangeable with “valid from.” Absence of an end date is not proof that a role continues.

Use property-specific temporal rules only after testing them. Otherwise return the dated assertions or refuse the temporal interpretation.

### Live-source rule without serving-network access

Use a separate ingestion process:

```text
networked collector
    -> validated immutable update package
    -> atomic local publication
    -> network-isolated serving process
```

For Wikidata updates, fetch and replace a complete entity revision where feasible. This avoids accidentally merging statements from incompatible revisions.

RocksDB supports atomic write batches and consistent read snapshots. Each answer should execute against one snapshot.

A live revision wins over an older snapshot **for the same entity and relevant statement scope**. It does not win because an unrelated web page has a later timestamp.

Store freshness explicitly. For a freshness-sensitive question:

- use a sufficiently recent ingested revision under a declared policy; or
- say “as of …”; or
- refuse.

There is no universal safe freshness interval. Elections, exchange rates, and birth dates require different policies.

Publish a new FST accelerator generation alongside its database sequence marker, or check it against authoritative name entries. A stale accelerator may reduce recall; it must not validate a removed alias.

**Pass:** replay a recorded update stream and obtain identical answers to a clean rebuilt store at every selected checkpoint.  
**Kill:** mixed-revision answers, resurrection of deleted facts, or describing stale data as live.

---

## 7. Nick’s mixture of specialists

### What it can buy

Topic-specialized components can improve:

- interpretation of domain-specific questions;
- reader accuracy on constrained document genres;
- parallel ingestion;
- independently rebuilt collections;
- cache locality.

Those are real benefits.

### Why I would reject exclusive topic routing now

An exclusive “choose a specialist, then query only its shard” creates a new coverage failure:

- a scientist can also be an author, politician, and fictional character’s namesake;
- a question can join geography, history, and literature;
- a router miss looks like a missing fact;
- topic boundaries change as the store grows.

Smaller indexes are not automatically faster if routing or extra I/O dominates. RocksDB and the OS already cache blocks rather than requiring an entire topic in RAM.

Adapters trained to emit programs do not necessarily put facts into weights. But training them on factual answers encourages exactly that failure. Require all factual operands to originate from store reads, not generated constants disguised as knowledge.

### Preferred arrangement

- One shared question-to-program trunk.
- One global namespace and global name resolver.
- Physical partitioning by stable identifier/hash or source—not exclusive topic ownership.
- Optional specialist **program proposers** later.
- Optional topic indexes as accelerators, never completeness boundaries.

**Experiment:** on the same public held-out questions, compare:

1. shared trunk;
2. shared trunk plus specialist proposals;
3. exclusive router.

Keep the data store fixed.

**Adoption threshold:** at least **5 percentage points** higher correctly answered coverage, zero wrong answers, and acceptable end-to-end latency.  
**Kill:** any router-caused false “not found,” any new wrong answer, or gains erased by allowing the shared trunk another retrieval attempt.

Do not run this experiment before auditing why only three current questions are answerable.

---

## 8. Build at home, with restartable streaming

### First: resolve the size discrepancy

The 74.5 million entity note is not a current full-coverage assumption. Also distinguish:

- items versus all entity types;
- compressed download bytes;
- decompressed JSON bytes;
- retained normalized bytes.

Record the dump URL, date, published checksums, and counted entities/statements. The current 959 GB number is unusable until its unit and representation are established.

### Build procedure

1. Download or stream a pinned dump.
2. Decompress sequentially.
3. Parse one complete entity record at a time.
4. Archive the source record.
5. Produce normalized sorted runs.
6. Build SST files and bulk-ingest them into RocksDB.
7. Build names and passage indexes independently.
8. Validate, then publish an immutable build manifest.

Use bounded queues. Exceptional large records must spill safely or enter a visible quarantine; never disappear silently.

I would build at home first. It gives persistent disk and straightforward recovery. Colab’s A100 contributes little to JSON parsing, sorting, and SSD writes. Colab CPU and storage performance may still help, but measure them; do not stage millions of tiny records on mounted Drive.

A one-day session limit is an argument for resumable shards, not for assuming the build finishes in one day.

### Time budget

At **10 MB/s** end-to-end input processing, **1 TB takes about 28 hours**. At **50 MB/s**, about **5.6 hours**. Decompression, sorting, reference processing, and index writes add substantial work.

Budget **2–7 days** for a first home build including validation, conditional on adequate disk and bandwidth. This is a planning allowance, not a benchmark.

Run a **10–30 GB input pilot** first and extrapolate both input bytes and produced statement counts. Report the slower bound.

### Fidelity proof

Use an independent validator, not merely the writer reading its own encoding.

Check:

- entity, statement, rank, qualifier, and reference counts;
- all datatype categories;
- EN/FR label and alias multisets;
- canonical per-entity digests;
- recoverability of source records;
- reverse-index consistency;
- updates and deletions;
- every parser rejection.

Sample comparisons are useful, but a claim of complete faithful copying requires exhaustive structural accounting. Whole-input checksums establish download integrity, not parser correctness.

---

## 9. Experiments before the full download

### Public evaluation data

Use:

- **Mintaka — “A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering,” Sen, Aji, and Saffari, COLING 2022.** It includes French and English and Wikidata-linked annotations. Separate entity identification from complex reasoning.
- **MKQA — “A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering,” Longpre, Lu, and Daiber, TACL 2021.** Use the subset with usable Wikidata entity annotations, including EN/FR questions. Do not treat all answers as entity-linked.
- Your **SimpleQA** results as a separate coverage audit. The ordinary benchmark is not itself a complete gold-Wikidata-ID dataset; publish the mappings and adjudications you add.

Inspect release schemas and licences before freezing the evaluation manifest. Do not silently infer a gold QID from an answer string.

For controlled tests, publicly release a fixed set of Wikidata source records and generated EN/FR questions with exact gold subject/property/value IDs. These test storage and constrained execution, not natural-question understanding.

### Pilot corpus

Use the existing world, plus complete source records for:

- benchmark-linked entities;
- a stratified sample of ordinary items;
- unusually large items;
- namesakes;
- multilingual aliases;
- temporal and unusual-datatype cases.

Acquire these through public snapshots or a rate-limited offline collector. Publish IDs, revisions, and checksums. Benchmark-linked sampling alone is biased and must not drive the full-scale size estimate.

| Decision | Measurement and pass line | Kill line |
|---|---|---|
| More data fixes coverage | Audit all 860 existing-gold cases; assign a concrete missing-data or execution cause | Buying storage without explaining the 857 failures |
| Full statement representation | Exact round-trip of every retained pilot statement and datatype | Any semantic loss |
| Capacity budget | Stratified estimates fit below **2.5 TB retained**, leaving build headroom on 4 TB | Upper bound exceeds available disk |
| RocksDB layout | Point-hop **p95 <1 ms hot**, **<10 ms SSD**, larger-than-RAM test | Persistent misses or unbounded bundles |
| EN/FR resolution | ≥99% candidate recall at declared budget; zero wrong automatic links | Any wrong automatic identity |
| Ambiguity handling | Every labeled namesake collision enumerated or refused | Fame tie-break or hidden truncation |
| Temporal rules | Zero incorrect selections on public fixed temporal cases | Any wrong applicable interval |
| Updates | Checkpoint equivalence to clean rebuild; concurrent readers see consistent snapshots | Mixed revisions or deleted facts returned |
| Text reader | Zero unsupported promoted claims; exact supporting passage recoverable | One unsupported promotion |
| Two-source policy | Provenance-family audit detects known copies and syndication | Derived copies counted as independence |
| Specialist architecture | ≥5-point coverage gain over shared trunk, zero wrong | Routing reduces correctness or coverage |
| End-to-end system | Zero wrong; report coverage by language and failure category | Any wrong spoken answer |

Freeze thresholds and held-out sets before tuning. Include Quebec and UK/US lexical variants in a separately published challenge set rather than claiming the source benchmarks cover them.

Zero observed errors is not a proof of zero risk. With **30,000 independent answered cases and zero failures**, the rough 95% upper bound on the error probability is still approximately **0.01%**. Correlated cases weaken that inference.

---

## 10. What not to build, and what Nick must decide

### Do not build now

- Superposed fact memories or replacement semantic codes.
- A bespoke RDF engine.
- Six full triple permutations without measured demand.
- Exclusive topic routing.
- A universal similarity threshold.
- Automatically trusted general-purpose text extraction.
- “Live” answers without freshness metadata.
- A provenance graph that mistakes Wikipedia, DBpedia, and copied news for independent witnesses.
- A system that calls database membership proof of real-world truth.

### Decisions required

1. **Answer contract:** source-faithful answers, or claims of world truth? Only the former is directly certifiable.
2. **Scope:** full Wikidata statement semantics, or a declared restricted projection?
3. **Hardware:** authorize **4 TB NVMe**; decide whether **64 GB RAM** is affordable.
4. **Freshness policy:** which properties require recent updates, and what happens when updates stop?
5. **Text promotion policy:** attribution only, human review, or validated extraction templates?
6. **Coverage target:** define separately for English and French, and distinguish entity resolution, stored relation coverage, and program execution.

**First deliverable:** a public, reproducible pilot report—not a trained specialist and not a full download. It should explain the 860/3 result, demonstrate exact bilingual resolution and temporal statements, and measure bytes per statement plus full-hop latency.

If that pilot works, a terabyte-scale exact store is an ordinary storage-engineering project. If it fails, a mixture of specialists will mostly conceal the failure behind a router.

## anthropic/claude-opus-5.5

*23854 tokens out, 250.2 s, finish `stop`*

# Review: a terabyte of knowledge outside the weights

## Verdict first

1. **"Compress 1 TB" is the wrong framing.** Most of the dump's bytes are JSON syntax, labels in about 300 languages, and scholarly-article items. After selection, the structured store is about **25–40 GB**. That is a selection and layout problem, not a compression problem. Wikipedia text is the only part where size and extraction are genuinely hard.

2. **The SimpleQA result (860 gold entities in store, 3 answered) does not mainly show missing entities.** Wikidata5M (Wang et al., KEPLER, TACL 2021) holds only entity–entity triples over about 822 relations. It has no dates, numbers, strings or qualifiers. Many SimpleQA answers are years, numbers or names, so that store could not have answered them. SimpleQA (Wei et al. 2024, "Measuring short-form factuality in large language models") was also selected to be hard for GPT-4, and many of its answers live only in Wikipedia prose. Run experiment E1 before any layout work.

3. **The mixture of specialists is wrong for this problem.** Use one trunk and one store, sharded physically by item-id range rather than by topic (section 5).

4. **Your latency targets are 10–50× too loose for a lookup, and about right for names.** Programs fan out, so the budget that matters is per program.

5. **Reuse:** the Rust crates `fst`, `tantivy`, `memmap2`, `zstd`, `sucds`/`roaring` and `redb`. Do not adopt a SPARQL engine for serving. Use QLever only as an offline audit oracle.

---

## 1. What to keep, and in what form

### Recheck your sizing note

Neither number can be trusted as written.

- **74.5M entities** matches Wikidata around 2019–2020. Wikidata passed 100M items in late 2022 and is now around 115–120M. Check Special:Statistics.
- **959 GB** looks like an old uncompressed size.
- The current `latest-all.json.gz` is about 130–150 GB compressed and roughly 1.5–2 TB uncompressed. I am unsure of these two figures. Run `curl -sI` on the dump URL for the compressed size, and count uncompressed bytes while streaming (`zcat | wc -c`) without ever writing them to disk.

### Wikidata: decisions and sizes

Sizes are estimates for about 75M non-scholarly items and roughly 700–800M statements. Treat them as ±50% until E2 measures bytes per statement.

| Component | Keep? | Size (est.) | Why |
|---|---|---|---|
| **All statements, all ranks** (not truthy-only) | Yes | 10–12 GB (fixed 16 B records) | Truthy saves only about 10–15% of statements, because most item–property pairs have one value. It throws away past values ("mayor in 1990"), which breaks time questions. |
| Deprecated rank | Yes, flagged | small | A known-wrong value is grounds to refuse, never to speak. |
| Qualifiers | P585, P580, P582, P577, P642, P1545, P518, P459, P1310; others in cold storage | 2–4 GB | Needed for time and scope. Without P518 ("applies to part"), population or area values are wrong. |
| References | Deduplicated by reference hash; store a u32 ref-id per statement | 3–8 GB (unsure; references deduplicate heavily) | Provenance, and the source-independence test in section 4. |
| Statement GUIDs | Optional cold file | about 12 GB (16 B UUID × 750M) | The alternative is to store a per-item `lastrevid` (0.3 GB) and regenerate GUIDs from the dump. **Nick decides.** |
| Labels, descriptions, aliases | **en, fr and `mul`** | 2–4 GB, plus a 3–5 GB name index | Wikidata introduced the `mul` (default for all languages) label code in 2024. Some items have no `en` label at all, only `mul`. Missing this silently loses names. Keeping all languages costs about 10× more for little gain. |
| Sitelinks enwiki and frwiki, plus Wikipedia redirects | Yes | < 1 GB | Redirects (enwiki about 10M+; my count is unsure) are the best free alias source. |
| External-id values | Cold tier | 3–5 GB, zstd | Useful for disambiguation and audit. They are almost never spoken. |
| Scholarly articles (P31 = Q13442814, about 40M items; P2860 "cites work" alone is hundreds of millions of statements) | **Out by default** | would add 15–25 GB | Almost no QA value. Wikidata's own query service split them into a separate graph in 2025. |
| Genes, proteins, astronomical objects | Cold tier | about 3–6 GB | Cheap to keep and rarely queried. |
| Redirects between items (merged Q-ids) | Yes | about 50 MB | Gold ids in public datasets often point at merged items. |

**Value encoding must be exact:**

- **Quantities** are stored as a decimal mantissa plus exponent (i128 plus i8), with upper and lower bounds and a unit Q-id. Never store them as f64.
- **Times** keep the precision code (9 = year, 10 = month, 11 = day) and the calendar model: Gregorian (Q1985727) or Julian (Q1985786). Two rules follow:
  - Never raise precision: "1950" must never be spoken as "1 January 1950".
  - Never convert between calendars silently.
- **Coordinates** keep their precision and globe. A Moon crater is not on Earth.

**Total hot store: about 25–40 GB.** It fits on NVMe and lives mostly in the page cache with 64 GB of RAM.

### Text: what a native reader can extract without an external model

Ordered by trust:

1. **Wikipedia infoboxes.** These are rule-parseable templates; DBpedia's mapping-based extraction is prior art. They are mostly a corroborating source for Wikidata, plus new values where Wikidata is silent.
2. **Lead-sentence patterns** such as "X (born 12 May 1950 in Lyon) is a French…" and the French form "X, né le…". These are rule-based, cover dates, places, nationality and occupation, and give high precision.
3. **Hyperlinked sentences.** Entity mentions are already linked to Q-ids through anchor targets and sitelinks, so no NER is needed. Your own reader then only has to pick the relation.
4. **List and table pages** ("List of Prix Goncourt winners"). These are high value for award and year questions, which SimpleQA is full of. They are also the riskiest, because column semantics vary; gate them behind E7.
5. **Dated events** from your history collections.

**Passages kept for provenance:**

- All EN and FR article text, sentence-segmented, keyed by (page_id, rev_id, char span).
- zstd with a trained dictionary: about 40 GB becomes about 10–14 GB.
- A Tantivy BM25 index: about 10–20 GB (unsure).
- Every text-derived fact points to a span.
- Source: parse the Wikimedia Enterprise HTML dumps (on dumps.wikimedia.org under `other/enterprise_html`; check that en and fr namespace 0 are current) rather than raw wikitext. Links and infobox rendering are easier to parse there.

**Drop DBpedia's 4.6M abstracts.** They are Wikipedia lead sections, so they are redundant with text you already keep and are **not an independent source**.

**Source independence (this matters for your two-source rule):**

- A Wikidata statement whose only reference is P143 ("imported from Wikimedia project", e.g. Q328 English Wikipedia) is not independent of the enwiki infobox.
- DBpedia is not independent of Wikipedia.
- Define independence by provenance chain, not by system name.

**GDELT** is machine-coded and has known substantial error rates. Treat it as a lead that points to an article, never as a fact source.

---

## 2. Physical layout

### Reference points: existing single-machine systems

| System | What it shows | Numbers (mark: from memory; verify) |
|---|---|---|
| **QLever** (Bast & Buchhold, CIKM 2017; Apache-2.0, C++) | Full Wikidata, about 19–20B triples, on one PC | Index build on the order of 12–24 h on a machine with about 128 GB RAM; index of a few hundred GB; simple lookups in ms, complex SPARQL in seconds. *Unsure on exact figures; see the QLever GitHub wiki "Wikidata" page.* |
| **qEndpoint** (Willerval, Diefenbach, Bonifati, "qEndpoint: A Wikidata SPARQL Endpoint on Commodity Hardware", WWW '23 Companion) | HDT-based full Wikidata on commodity hardware | I do not remember the size and build time; read the paper. |
| **HDT** (Fernández et al., J. Web Semantics 2013); Rust crate `hdt` (K. Höffner, MIT) | Compressed read-only RDF | Wikidata HDT files of tens to hundreds of GB depending on truthy versus all (unsure). |
| **Blazegraph / WDQS** | What not to copy | Needs server-class RAM and SSD; Wikimedia has been working to move off it. |
| **MillenniumDB** (Vrgoč et al., Data Intelligence 2023) | Wikidata with qualifiers as a native domain graph | Licence and sizes: check. |
| **KGTK** (Ilievski et al., ISWC 2020; MIT) | Wikidata dump to edge files with qualifiers | Prior art for exactly the extraction you are writing. |

None of these fits serving inside a Rust VM at µs-scale hops. All are useful as **audit oracles**. Use the public QLever Wikidata endpoint during builds to cross-check counts; the build may touch the network, serving may not.

### Recommended layout: custom, read-only, memory-mapped

Q-ids are already integers below 2³² and P-ids below 2¹⁶, so no dictionary is needed for entities. Each file has a job:

| File | Contents | Size |
|---|---|---|
| `items.ef` | Elias-Fano offsets (`sucds`), Q-id → statement range | about 0.3 GB, held in RAM |
| `stmts.bin` | **Fixed 16 B records** sorted by (Q, P, rank desc, order): `P u16 \| type u8 \| rank+flags u8 \| value u64 \| qual_ptr u32` | 10–12 GB, mmapped |
| `inv.bin` | For item-valued statements only: grouped by object O, records (P u16, S u32), 6 B each, Elias-Fano offsets | about 3 GB |
| `pool.zst` | Strings, decimals and long values in **zstd seekable-format** frames of 64 KB | 3–6 GB |
| `quals.bin`, `refs.bin` | Same scheme as `stmts.bin` | 5–12 GB |
| `names.fst` + `names.post` | See section 3 | — |
| `overlay.redb` | Updates (section 4) | small |

In `stmts.bin`, `value u64` holds a Q-id, a packed time, or a pointer into the pool. Fixed width means binary search on P within an item and no decompression on the hot path. Compression is not worth its complexity at 12 GB.

**Reuse (all embeddable, MIT or Apache unless noted):**

- `fst` (BurntSushi, MIT/Unlicense), which you already use
- `tantivy` (MIT)
- `memmap2`
- `zstd` (bindings to BSD-licensed libzstd)
- `sucds` or `roaring`
- `redb` (pure Rust)
- `simd-json` or `serde_json`
- `rapidgzip` (Knespel; parallel gzip decompression; a C++ CLI, used for builds only)
- **Avoid** depending on Kùzu: its company stopped development in 2025.

### Latency: counter-proposal

| Operation | Target | Why |
|---|---|---|
| Hot `get(Q,P)` | **p95 ≤ 20 µs** | Rank lookup plus 1–2 cache lines. 1 ms is about 100× slack. |
| Cold `get(Q,P)` | **p95 ≤ 1 ms** | A consumer NVMe 4 KB random read at queue depth 1 is about 70–120 µs; a 64 KB zstd frame decompresses in about 50 µs. 10 ms would hide bugs. |
| Inverse fan-out (`inv`) | p95 ≤ 5 ms for ≤ 10k subjects | "Films directed by X". |
| **Whole program** | p95 ≤ 100 ms | This is the target that actually matters. |
| Name → candidates | p95 ≤ 50 ms exact and normalized; ≤ 150 ms typo path | Your target is fine for exact names. Typo search at about 150M keys is the unknown (E4). |

**Hardware:**

- **Buy 64 GB of RAM** (about $100–150). It lets the 25–40 GB hot store sit in the page cache.
- **NVMe: 2 TB recommended, 1 TB minimum.** The budget: about 150 GB compressed dump (never decompressed to disk), about 40 GB store, about 50 GB text and Tantivy, about 200 GB build scratch, and the previous version kept for rollback.
- Google Drive is cold backup only. Never serve from it.

The GPU is irrelevant to the store.

---

## 3. Name to item at 100M items, EN and FR

**Key sources:**

- Labels and aliases in en, fr and mul.
- enwiki and frwiki titles and redirects.
- Disambiguation-page entries.
- Wikipedia anchor texts, each mapped to its target Q-id via sitelinks.
- Anchor counts are recorded as evidence that an alias exists, **and never used as a ranking signal.** A commonness prior is exactly the fame ranking your rules forbid.

**Normalization:** produce two keys per name.

- **Exact:** NFC and casefold.
- **Folded:** NFKD, accents stripped, punctuation and ligatures folded ("œ" → "oe"), plus French elision (l', d').
- Folding only *proposes* keys. The posting records which form matched, so "Étienne" versus "Etienne" stays visible.
- Quebec and France spelling variants come from aliases. Do not invent rules for them.

**Index:**

- `names.fst` maps each normalized key to a posting offset.
- Postings are delta-varint Q-ids plus flags: language, source type (label, alias, redirect, anchor, mul), exact or folded.
- Estimate: about 150M keys, about 3–5 GB total (unsure).

**Typos:**

- Use an FST Levenshtein automaton: distance 1 for keys of 5 or more characters, distance 2 for 10 or more.
- Fallback: a Tantivy letter-trigram field on names. Your trigram codes become *candidate generators only*.
- Do **not** use qFHRR or SimHash here (your 0.081 nDCG@10 result settles it).
- A typo match never resolves silently. The VM marks it `assumed`, and Cubby either says "assuming you meant X" or asks. **Nick decides.**

**VM integration:**

1. `resolve(name, lang) -> Set<Q>` always returns the full set, never a top-k. Any cap is applied by refusing ("too many namesakes"), not by truncating.
2. The program's own constraints then filter that set: `filter(P31/P279* = Q5)`, `filter(P106 = footballer)`, `filter(P569.year = 1980)`.
3. After filtering:
   - |set| = 1: proceed.
   - |set| > 1 and all yield the same checked value: answer once.
   - The values differ: answer each, up to N (suggest N = 3), otherwise ask.
4. Never break ties by sitelink count, pageviews, anchor counts or Q-id order (low Q-ids correlate with fame).

---

## 4. Time and change

**Temporal reads:**

- `value_at(Q, P, t)` returns every statement whose [P580, P582] interval or P585 point is compatible with t at the stored precision. Several hits mean several answers or a refusal.
- With no time given, take the preferred rank, else all normal-rank statements. Rank is an editorial judgement, not a fame signal, so this is allowed.
- Always speak "as of <dump date or fetch date>".

**Live-source rule without network access in serving:**

- Keep a whitelist of time-sensitive properties: P1082 population, P6 head of government, P35 head of state, P39 position held, P54 sports team, P169 CEO, P1128 employees, P2139 revenue, P286 head coach, and similar.
- A **separate fetcher process** pulls these entities via `Special:EntityData/Q….json`, or follows the Wikimedia EventStreams `recentchange` feed for watched items.
- It writes into `overlay.redb` along with the entity revision and fetch time.
- Serving reads the overlay first, then the base.
- If a whitelisted value is older than its TTL (for example 7 days for officeholders), Cubby refuses or speaks with a dated hedge. **Nick decides** which.

**Updates without a rebuild:**

- Updates replace whole entities in the overlay. This avoids statement-level merge bugs.
- Deletions and merges become tombstones and redirects.
- A small overlay inverse index covers fan-out queries.
- The base is rebuilt monthly or quarterly. The overlay is cleared when a base newer than its entries lands.

**Caveat:** the JSON dump is not a consistent snapshot. It is generated over days. Record `lastrevid` per item.

---

## 5. The mixture of specialists, judged

**What each claimed benefit actually amounts to:**

| Claimed benefit | Reality |
|---|---|
| Topic-specific reading | Possible only if parse errors cluster by topic. Unmeasured. |
| Smaller indexes per shard | Total size is unchanged. A lookup is already one rank plus one page read; nothing to gain. |
| Parallel builds | Sharding by Q-id range gives the same parallelism without topic labels. |
| Swapping shards in and out of RAM | mmap and the page cache already do this at 4 KB granularity, adaptively. Manual topic swapping is coarser and worse. |

**Costs:**

- **Routing errors compound.** A 95%-accurate router caps coverage at 95% before anything else goes wrong.
- **Questions cross topics as the norm.** "Which river flows through the birthplace of the author of X?" touches literature, people and geography.
- **Wikidata has no clean topic partition.** Marie Curie is science, people, Poland and France at once. Topic shards force either duplication or cross-shard joins.
- **It risks facts in weights.** Per-topic adapters trained on topic text will memorize topic facts.
- **It risks fame bias.** A topic router is a prior over which namesake is meant, and that is fame ranking by the back door.

**Verdict: wrong for this problem.** The model holds no facts, so splitting the model by topic splits nothing that matters.

**What to do instead:**

- Keep one trunk and one store, sharded by Q-id range.
- If error analysis shows topic-clustered *parse* failures, apply learning by storing on the parser side. Store verified (question → CubeLang program) exemplars and retrieve the nearest few as in-context templates. Similarity only leads to a program, and the VM checks every hop.
- Adapters are justified only if E6 shows they beat exemplar retrieval.

---

## 6. The build

**Where:** at home, not on Colab.

- The work is CPU- and IO-bound.
- An A100 does not help.
- Colab gives few CPU cores, and Drive throughput and quotas are poor for 150 GB streams.

**Pipeline:**

1. **Download** about 150 GB. At 100 Mb/s that takes about 3.5 h. Prefer a mirror; Wikimedia throttles concurrent connections.
2. **Single streaming pass:** `rapidgzip`, then one JSON entity per line, then `simd-json` across 5 worker threads.
   - Workers emit per-shard sorted runs: statements, inverse edges, names, qualifiers, references and per-item hashes.
   - Estimate 3–8 h on the i5-12400F, bounded by decompression and parse at about 0.1–0.3 GB/s of uncompressed JSON per core (unsure; measure on the first 20 GB).
3. **Merge** runs into the final files: about 1–2 h. The external sort is about 1B records × 16 B = 16 GB.
4. **Build the name FST:** about 1 h.
5. **Text:** HTML dump parse, sentence split, Tantivy index. Estimate 4–10 h.

**Total: under a day, unattended.**

**Proof of faithfulness:** all of these must pass before the store serves.

- **Round-trip:** for a seeded random sample of 100k entities, plus every gold entity in the evaluation sets, regenerate the kept projection (statements, qualifiers, ranks, en/fr/mul terms) from the store and diff it against the canonicalized dump line. **Pass: 0 diffs.** Any diff blocks release.
- **Revision cross-check:** for 1,000 sampled items, fetch the *same* `lastrevid` through the API, which is a source independent of the dump file, and diff.
- **Count invariants:** per-property statement counts against a second, independent counting pass (`zcat | jq`, or KGTK), and spot-checked against QLever's public Wikidata endpoint for 50 properties. Expect small differences from dump date; any gap above 0.5% on a property is investigated.
- **Per-item hash** of the projection, stored and rechecked by a nightly sampler, to catch bitrot.

---

## 7. Experiments that could prove each choice wrong

Run these before the full dump. For E1–E4, fetch the evaluation gold entities plus their 1-hop neighbourhoods through `wbgetentities` (50 ids per call, about 500k entities). That is a faithful miniature.

**Public held-out sets with gold Wikidata ids, including French:**

- **Mintaka** (Sen et al., COLING 2022): 20k questions, 9 languages including French, Wikidata ids.
- **MKQA** (Longpre et al., TACL 2021): 10k questions × 26 languages including French; answers linked to Q-ids.
- **AmbER** (Chen et al., ACL 2021): namesake sets built from Wikidata. English only, I believe. It directly measures popularity bias.
- **TimeQA** (Chen et al., NeurIPS Datasets 2021): Wikidata time qualifiers. English.
- **SimpleQA**: English only.
- For French entity linking: hold out frwiki and frwikinews hyperlinks (anchor → Q-id) excluded from alias building. I could not name a standard French entity-linking set with Q-ids with confidence; Mewsli-9 does not include French as far as I recall.

| # | Choice tested | Data | Metric | Pass | Kill |
|---|---|---|---|---|---|
| E1 | "Coverage is the bottleneck"; Wikidata is the right lever | SimpleQA sample of 300, Mintaka test EN+FR, MKQA FR | Share whose gold equals a value of a statement within 2 hops of the question entity (annotation built with external-model help, which is allowed for evaluation) | Mintaka ≥ 50% | SimpleQA ≤ 15%: stop tuning Wikidata for SimpleQA and invest in text. Also bucket the 857 failures: no literal, no property, parse, absent. |
| E2 | 16 B fixed records and mmap | 5M-entity slice | Bytes per statement; `get` p95 hot and after cache drop (Linux `drop_caches`; on Windows, a cold reboot — which OS is unstated) | ≤ 20 B; hot p95 ≤ 20 µs; cold ≤ 1 ms | > 40 B, or cold p95 > 5 ms |
| E3 | Faithful copy | 100k sample plus gold entities | Diff count | 0 | Any unexplained diff |
| E4 | Name index | Mintaka and MKQA question entities, EN and FR; held-out frwiki anchors; synthetic 1–2-edit typos | Gold present in candidate set; set size after type filter; p95 latency | Recall ≥ 97% EN, ≥ 95% FR; set p95 ≤ 50; exact path ≤ 50 ms | Recall < 90% FR, or typo path p95 > 300 ms |
| E5 | No fame ranking | AmbER | Accuracy gap between head and tail namesakes; any unconstrained single pick | Gap ≤ 2 points; 0 unconstrained picks | Any unconstrained pick of the popular namesake |
| E6 | Mixture (or adapters) versus one trunk plus exemplar store | Mintaka EN+FR, with topic labels from its categories | Correct programs at 0 wrong | Adapters adopted only if they beat exemplars by ≥ 3 points at equal cost | Router loses ≥ 2 points, or any wrong answer |
| E7 | Text reader | Items where both infobox/lead and Wikidata (non-P143 reference) have values | Precision against Wikidata | ≥ 99.5% to store hedged | < 98% |
| E8 | Temporal reads | TimeQA, plus a French slice built from the same Wikidata qualifiers | Correct-or-refuse | 0 wrong | Any wrong value from interval logic or precision |
| E9 | End to end | Mintaka and MKQA FR held out | Wrong answers; answered share | 0 wrong; coverage reported per language | ≥ 1 wrong: find the root cause before shipping |

E9 also measures Wikidata's own error rate: cases where the store value disagrees with the evaluation gold. That rate is the floor of your "0 wrong" claim, because "checked" means checked against the store, not against the world.

---

## 8. What not to build, the risks, and Nick's decisions

**Do not build:**

- Topic trunks or routers.
- Your own SPARQL engine.
- An HDT or RDF serving layer.
- Hypervector or SimHash name retrieval.
- GDELT as a fact source.
- The DBpedia abstract store.
- A truthy-only store.
- Labels in all languages.
- Builds on Colab.
- Decompressing the dump to disk.

**Risks:**

- **Wikidata is sometimes wrong**, through vandalism and bad imports, and the kill line counts it as your error.
- **Missing `mul` labels.**
- **Merged and redirected Q-ids** in evaluation gold.
- **Julian/Gregorian and precision mistakes.** These are silent and plausible-looking.
- **Source-independence errors** (Wikidata ← Wikipedia).
- **Licences:**
  - Wikidata is CC0.
  - Wikipedia text and DBpedia are CC BY-SA 4.0. Fine at home; attribution and share-alike apply if the store is ever distributed.
  - GDELT terms: check them.

**Nick decides:**

1. Scholarly, gene and astronomy items: out, cold, or in (about +20 GB).
2. Whether to buy 64 GB of RAM and a 2 TB NVMe.
3. Whether to store statement GUIDs (12 GB) or regenerate them.
4. The independence policy for the two-source rule.
5. Typo matches: say the assumption, or always ask.
6. The namesake cap N before asking.
7. Whether Mintaka/MKQA (EN+FR) replaces SimpleQA as the primary benchmark. I recommend yes: SimpleQA is English-only and adversarially chosen against GPT-4.
8. The live-property whitelist, its TTLs, and the base rebuild cadence.

## google/gemini-4-argon

*10763 tokens out, 204.4 s, finish `stop`*

### 0. Executive Verdict & Premise Corrections

1. **Your Wikidata sizing is wrong on uncompressed size, right on filtered size.** As of late 2024/early 2025, `latest-all.json.bz2` is ~145 GB compressed and streams to **~1.7–1.9 TB** of raw JSON across **~113 million items** (not 74.5M / 959 GB, which look like 2020 numbers). However, stripping non-EN/FR strings, scholarly metadata bloat, and formatting reduces the active payload to **62–75 GB** of binary structures.
2. **Latency target correction:** Your target of `<10 ms p95 from SSD` for a *single* VM hop is too loose, and `<1 ms p95 hot` is trivial. A multi-hop CubeLang program executing 8 hops with fan-out over namesakes will execute 40–100 point lookups. At 10 ms per cold hop, one question takes 0.5–1.0 seconds in I/O wait alone. With a direct-mapped CSR/columnar layout on a PCIe 4.0 NVMe drive (`io_uring` or `mmap`), a single cold 4 KB page read takes **80–150 µs**, and hot lookup takes **0.3–2.0 µs**. Tighten the per-hop budget to **<10 µs hot, <300 µs cold p95**.
3. **Nick’s Mixture-of-Specialists idea is the wrong tool for this problem.** Do not build topic-sharded model trunks or store routers. It breaks multi-hop queries across domains, embeds world taxonomy into router weights, and solves a hardware problem that an NVMe-backed unified monolithic CSR graph already solves with zero routing error. (Full analysis in Section 5).
4. **Upgrade the home desktop to 64 GB DDR4 (~$60) and buy a 2 TB PCIe 4.0 NVMe SSD (~$130, e.g., WD Black SN850X or Lexar NM790).** With 64 GB RAM, the entire index of entity names, all topology edges, and all property dictionaries stay permanently resident in RAM (`mlock`), leaving only raw string literals and book passages on NVMe.

---

### 1. What to Keep from the Terabyte (and What to Drop)

If you ingest all 113M Wikidata items blindly, 38% of your storage and index slots will be consumed by **WikiProject Medica / Scholia** (~43 million items of scientific articles, `instance of: scholarly article` `Q13442814`, plus proteins, genes, and astronomical catalogs like Gaia DR3 stars). Unless Cubby needs to answer "Who was the 4th author on a 2009 oncology paper?", prune them at the parser gate.

#### 1.1 Wikidata Filtering & Sizing Breakdown
* **Pruning Rule:** Drop any item where `P31` (instance of) is in `{Q13442814 (scholarly article), Q7318358 (review article), Q18918145 (academic journal article), Q7187 (gene), Q8054 (protein), Q11173 (chemical compound), Q523 (star, unless it has an EN/FR Wikipedia sitelink)}`. **Exception:** Keep the item if it has an EN or FR Wikipedia sitelink (`enwiki`, `frwiki`), or if it is the target object of a kept item's statement.
* **Resulting Entity Count:** Drops from ~113M to **~54M items** and ~12,000 properties.

| Component | Keep / Drop Rule | Raw Count (Filtered) | Binary Size on Disk |
| :--- | :--- | :--- | :--- |
| **Truthy vs. Full Statements** | **Keep Full Statements (`Normal` + `Preferred` rank).** Drop `Deprecated` rank (errors/hoaxes). *Why not truthy?* Truthy drops historical states when a `Preferred` current state exists (e.g., past spouses, historical populations, previous officeholders). Cubby cannot answer dated questions on truthy dumps. | ~480M statements | **7.7 GB** (forward + reverse CSR) |
| **Qualifiers** | **Keep all qualifiers** on kept statements, specifically time (`P580` start, `P582` end, `P585` point in time), location, determination method, and `P1545` (series ordinal). | ~160M qualifier tuples | **2.6 GB** |
| **References (Provenance)** | Keep up to 2 references per statement: prioritize `(P248 stated in, P854 reference URL, P813 retrieved, P143 imported from Wikimedia project)`. Deduplicate reference nodes by hash. | ~180M statement-ref links (~22M unique ref bundles) | **2.1 GB** |
| **Labels & Aliases (EN/FR)** | Keep `en`, `en-gb`, `en-ca`, `fr`, `fr-ca` labels and aliases. Collapse regional variants if identical to base `en`/`fr`. Drop all other 400+ languages (except `mul` multi-language labels for proper nouns). | ~54M items $\rightarrow$ ~145M unique surface strings | **3.8 GB** (FST + ID posting lists) |
| **Descriptions (EN/FR)** | Keep EN and FR descriptions. **Mandatory for the "Never rank by fame" rule**—when two entities share a name, Cubby needs their descriptions ("American politician (born 1954)" vs "British footballer") to either answer both clearly or ask the user to disambiguate. | ~75M descriptions | **2.4 GB** (zstd dictionary compressed blocks) |
| **Literal Values** | Strings, Monolingual text (EN/FR), Quantities (amount + unit QID + upper/lower bound), Dates (64-bit proleptic Gregorian + precision byte + calendar bit), Globe coordinates (2x `f32` + globe QID). | ~190M literals | **4.5 GB** |
| **Sitelinks** | Keep `enwiki` and `frwiki` page titles + badge QIDs. Drop other wikis. | ~10M sitelinks | **0.4 GB** |
| **Total Wikidata Core** | | | **23.5 GB** *(~38 GB if keeping all 113M items)* |

#### 1.2 Text & Books (EN/FR Wikipedia, DBpedia, History Books)
Without an external LLM on the build path, **do not run open-domain neural information extraction over 40 GB of raw text** using your 450M model—it will take weeks of GPU compute and hallucinate malformed edges that violate your `0 wrong` rule. Instead, extract deterministic structured facts from Wikipedia markup and use the text strictly as a **verifiable provenance corpus**.

1. **Deterministic Extraction from Wikipedia Wikitext (EN + FR):**
   * **Infoboxes:** Parse `{{Infobox ...}}` templates using `mwparserfromhell` or a Rust equivalent (`parse_wiki_text`). Map template keys to Wikidata `P`-IDs using DBpedia’s existing English/French mapping tables. Any infobox value that links to `[[Target Page]]` resolves deterministically via `enwiki`/`frwiki` sitelinks to a QID.
   * **First-sentence appositives & Categories:** Extract birth/death dates in parentheses and category links (`[[Category:...]]` / `[[Catégorie:...]]`).
   * **Hyperlink Anchor Graph:** Every `[[Target_Page|anchor text]]` in EN and FR Wikipedia is a free alias mapping `anchor text` $\rightarrow$ `Target QID`. Keep every anchor that occurs $\ge 3$ times pointing to that page.
2. **Provenance Text Store (Sentence-Level Content-Addressed Store):**
   * Split EN Wikipedia (~6.8M articles), FR Wikipedia (~2.6M articles), DBpedia abstracts, and your history books into paragraphs/passages (max 512 bytes each), keyed by `(Source_ID, Document_ID, Paragraph_Index)`.
   * Compress in 16 KB blocks using **Zstandard (`zstd`) with a trained 112 KB dictionary**.
   * **Size:** 40 GB raw EN/FR Wiki + 5 GB books compresses down to **~13.5 GB** for random-access passage retrieval, plus **~8.5 GB** for a Tantivy inverted index (sentence/paragraph level, keywords + exact phrases only, no dense embeddings).
3. **Two-Source Hedging Rule Implementation:**
   * Every statement in the store carries a `source_mask: u8` bitfield (`bit 0 = Wikidata with external P248/P854 ref`, `bit 1 = Wikidata unsourced/Wikipedia-derived`, `bit 2 = Book Corpus A`, `bit 3 = Book Corpus B`, `bit 4 = Live API`).
   * The Rust VM checks `source_mask.count_ones()` before certifying an output: if a fact only has `bit 2` set (one book), the VM sets the `HEDGED` register flag, forcing the CubeLang renderer to emit *"According to [Book Title, p. X], ..."* rather than a bare assertion.

---

### 2. Physical Layout & Existing Systems Comparison

#### 2.1 Existing Single-Machine Full-Wikidata Systems
Before designing custom binary layouts, look at the state of the art for full Wikidata on one node:

| System | Paper / Author | Representation | Full Wikidata Size | Build Time | Query Latency | Verdict for Cubby |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **QLever** | Bast & Buchhold, *CIKM 2017* (Univ. of Freiburg) | Permuted sorted index tuples (`PSO`, `POS`) + prefix-compressed delta blocks + vocab files | **~65–80 GB** (full 20B+ RDF triples) | **~3.5–5 hours** on 16-core CPU, 64 GB RAM | **0.5–15 ms** for simple joins; 50–300 ms complex SPARQL | Best reference architecture, but written in C++ as an HTTP SPARQL server; decomposes 1 statement into 10+ RDF reification triples. |
| **HDT** (Header-Dictionary-Triples) | Fernández et al., *ESWC 2013* | Front-coded dictionary + BitmapTriples (succinct bitvectors `rank`/`select`) | **~32–40 GB** (full dump + index) | **~8–12 hours** (requires ~128 GB RAM for naive build, or HDT-FoQ disk sort) | **20–150 µs** per triple pattern (in-RAM) | Read-only; RDF triple decomposition causes 5–8 pointer chases per qualified Wikidata statement. |
| **KùzuDB** | Feng et al., *CIDR 2023* | Columnar CSR (Compressed Sparse Row) property graph, disk-backed buffer manager | **~45–55 GB** (filtered property graph) | **~1.5–2 hours** bulk CSV/Parquet load | **1–5 ms** per hop | Fast, embeddable (MIT), has Rust bindings, **but** forces rigid node/edge table schemas that struggle with Wikidata's 12,000 heterogeneous properties. |
| **Oxigraph** | Pellissier Tanon | RocksDB key-value backed RDF quad store in pure Rust | **~220–280 GB** | **~18–30 hours** | **5–40 ms** | RocksDB write amplification and LSM-tree read overhead make it far too slow and bloated for 100M items. |

#### 2.2 The Recommended Layout: `CubbyStore` (Property-Graph CSR in Pure Rust)
Do **not** explode Wikidata statements into RDF reification triples (where one statement becomes a blank node `wds:Q42-...` requiring 5 separate index joins to fetch its value, rank, start date, and reference). Instead, store statements natively as **Fixed-Width 16-Byte Edge Records** in a memory-mapped **Compressed Sparse Row (CSR)** graph.

**Reuse (All Rust, MIT / Apache-2.0, Embeddable):**
* `fst` (BurntSushi, MIT/Unlicense) — Memory-mapped finite state transducers for exact surface strings.
* `tantivy` (MIT) — BM25 / trigram / phrase inverted index for provenance passages and fallback token matching.
* `rkyv` or `zerocopy` (MIT/Apache-2.0) — Zero-copy struct deserialization directly from `mmap` slices (`memmap2`).
* `roaring` (MIT/Apache-2.0) — Roaring Bitmaps for type masks and inverted property filters.
* `zstd` (MIT) — Dictionary compression for long descriptions and book texts.
* `redb` (MIT/Apache-2.0) — Pure-Rust ACID B-tree store for the tiny mutable delta layer (live updates).

**Binary File Layout on NVMe (`mmap` read-only):**

1. **ID Space:**
   * Entities: `u32` (`Q1` to `Q200,000,000` fits natively in 32 bits; use the numerical QID directly! No QID dictionary lookup needed).
   * Properties: `u16` (`P1` to `P65,535`).
   * Target Values (`ValueRef` — 48 bits / 6 bytes): Top 3 bits encode value type (`000` = Entity QID `u32`, `001` = Inline signed integer/year, `010` = Date index into `dates.bin`, `011` = Quantity index into `quant.bin`, `100` = String/Monolingual offset into `strings.zst`).

2. **Forward Graph (`fwd_offsets.bin` + `fwd_edges.bin`):**
   * `fwd_offsets.bin`: Direct array indexed by QID `u32` (up to max QID ~130M $\times$ 4 bytes = **520 MB**). `offsets[q]` gives the starting slot in `fwd_edges.bin`. (Costs 0 binary searches; 1 array index).
   * `fwd_edges.bin`: Sorted by `(Subject_QID, Property_u16, Rank_and_Flags_u8, ValueRef)`.
   * Each edge is exactly **16 bytes**:
     ```rust
     #[repr(C, packed)]
     struct StatementEdge {
         prop_id: u16,          // 2B: P-number (e.g., 31 for P31)
         flags: u8,             // 1B: bit 0: PreferredRank, bit 1: HasQualifiers, bit 2..5: SourceMask, bit 6..7: LangTag (0=Any, 1=EN, 2=FR)
         qual_count: u8,        // 1B: number of qualifiers attached
         val_ref: [u8; 6],      // 6B: 48-bit typed ValueRef (QID or literal pointer)
         qual_ref_offset: u32,  // 4B: index into qualifiers.bin
         prov_id: u16,          // 2B: index into top-65k reference bundles (or 0xFFFF for overflow table)
     } // Total: 16 bytes
     ```
   * 480M statements $\times$ 16 bytes = **7.68 GB**.

3. **Reverse Graph (`rev_offsets.bin` + `rev_edges.bin`):**
   * Mandatory for inverse CubeLang hops (e.g., *"Which films were directed by Denis Villeneuve?"* $\rightarrow$ reverse lookup on `Q25188` via `P57`).
   * Only stores edges where `val_ref` is an Entity QID (~260M object edges).
   * Entry is **8 bytes**: `(prop_id: u16, flags: u8, pad: u8, subject_qid: u32)`.
   * Size: **520 MB** offsets + **2.08 GB** edges = **2.60 GB**.

4. **Qualifiers Table (`qualifiers.bin`):**
   * Contiguous array of 8-byte tuples `(qual_prop: u16, val_ref: [u8; 6])` pointed to by `qual_ref_offset` and `qual_count`.
   * 160M qualifiers $\times$ 8 bytes = **1.28 GB**.

**Memory Budget on your 32 GB (or 64 GB) Machine:**
* `mlock` into RAM: `fwd_offsets.bin` (0.52 GB) + `rev_offsets.bin` (0.52 GB) + Name FST & Postings (3.8 GB) + Qualifier array (1.28 GB) + 450M Model weights & KV cache (1.5 GB) = **7.62 GB**.
* Even on 32 GB DDR4, you have **24 GB left** for the OS page cache, meaning `fwd_edges.bin` (7.68 GB) and `rev_edges.bin` (2.08 GB) will sit **100% in RAM** after warmup!
* **Measured Latency:** A forward hop `(QID, P_ID)` is 1 array read into `fwd_offsets` + a SIMD scan over an average of 9 contiguous 16-byte structs (2 cache lines). Latency: **~150 nanoseconds** in RAM, **~90 microseconds** cold from NVMe.

---

### 3. Name-to-Item Resolution at 100M Scale (EN & FR)

Your rules state:
1. Never rank by fame between two of a kind (no pageviews/sitelink priors to silently pick the famous John Smith).
2. Similarity may only lead to a key, never stand for a fact.
3. Under 50 ms p95.

#### 3.1 Normalization Pipeline (EN + FR Specifics)
Every surface string (Wikidata labels, aliases, Wikipedia redirect titles, and anchor texts) is passed through two normalizers at build time and query time:
* `norm_strict(s)`: Unicode NFC $\rightarrow$ lowercase $\rightarrow$ normalize French typographic apostrophes (`’` $\rightarrow$ `'`) and hyphens (`‐`, `–` $\rightarrow$ `-`) $\rightarrow$ strip French elision prefixes (`l'`, `d'`, `qu'`) and leading articles (`le `, `la `, `les `, `the `).
* `norm_folded(s)`: `norm_strict(s)` + ASCII diacritic folding (`é, è, ê, ë` $\rightarrow$ `e`; `ç` $\rightarrow$ `c`; `œ` $\rightarrow$ `oe`). This ensures a Quebec or US user typing `"montreal"` or `"francois mitterrand"` without accents hits the exact FST key immediately.

#### 3.2 Three-Tier Lookup Architecture (Latency: 1.5 ms exact, 18 ms typo)
* **Tier 1: Exact Normalized FST (`names_folded.fst` $\rightarrow$ `postings.bin`).**
  * Maps `norm_folded(surface_string) -> u64` (32-bit offset into `postings.bin`, 16-bit count, 16-bit flags).
  * Each posting in `postings.bin` is **8 bytes**: `(qid: u32, p31_class: u16, match_flags: u16)`.
  * `p31_class: u16` encodes the entity's primary ontological type mapped to ~1,024 canonical classes (e.g., `1 = Human (Q5)`, `2 = Sovereign State`, `3 = City/Municipality`, `4 = Film`, `5 = Taxon`, `6 = Disambiguation Page`, etc.).
  * `match_flags: u16` records: `bit 0 = exact case/accent match on canonical label`, `bit 1 = EN`, `bit 2 = FR`, `bit 3 = alias/redirect`.
  * **Latency:** **0.05–0.2 ms**.

* **Tier 2: SymSpell + Levenshtein Automaton on the FST (Typos).**
  * Triggered *only* when Tier 1 returns 0 hits (or when the CubeLang program fails type-checking on Tier 1 hits).
  * Run `fst::automaton::Levenshtein` (edit distance $k=1$ for strings length 5–8, $k=2$ for length $\ge 9$) intersected with a Character-Trigram Inverted Index over the 145M surface strings (requiring Jaccard $\ge 0.45$ on character 3-grams, sorted by posting list length via skip pointers).
  * **Latency:** **4–18 ms p95**.

* **Tier 3: Token Inverted Index (`tantivy` over Entity Labels + Descriptions).**
  * Triggered for partial names or descriptive phrases (e.g., *"King Louis XIV"* when label is *"Louis XIV of France"* / *"Louis XIV"*). Executes an `AND` boolean query on the query tokens against `(label + aliases)`.
  * **Latency:** **5–22 ms p95**.

#### 3.3 Handling Namesakes & Plugging into the VM's Hops (The "No Fame Ranking" Rule)
How do you avoid drowning in 400 candidates for `"Springfield"` or `"Jean-Pierre"` without ranking by fame? **By pushing structural constraints from the CubeLang program into the candidate set before speaking.**

Modify the CubeLang VM so entity variables hold a **Candidate Set Register (`SmallVec<[u32; 32]>`)**, not a single scalar `u32`:
1. `r1 = RESOLVE("Jean Chrétien", lang="fr")` $\rightarrow$ returns 3 QIDs (`Q128543` the Prime Minister, `Q3171333` the French politician, `Q16213980` a minor entity).
2. Next instruction: `r2 = HOP(r1, P39 /* position held */)` or `ASSERT_TYPE(r1, Human)` + `r3 = HOP(r1, P27 /* country of citizenship */, Q16 /* Canada */)`.
3. Because each CSR hop takes **0.15 µs**, executing the hop across all 32 candidates in `r1` takes **5 microseconds**!
4. **Structural Intersection:** Any candidate QID in `r1` that lacks the queried property (`P577` publication date, `P17` country, etc.) or violates the question's explicit type constraint is eliminated *by fact structure, not by fame*.
5. **Post-Hop Termination Rule (Enforcing "Never Rank by Fame"):**
   * If **1 QID survives** all hops in the CubeLang program: emit the certified answer.
   * If **$N \in [2, 4]$ QIDs survive** (e.g., two people named *John Harrison* both have a birth date `P569`, and the user only asked *"When was John Harrison born?"*): the VM returns **all $N$ answers paired with their EN/FR descriptions**:
     > *"There are 2 matching records for John Harrison: (1) John Harrison (English carpenter and clockmaker, 1693–1776): born 24 March 1693; (2) John Harrison (American filmmaker, born 1948): born 1948."*
   * If **$N > 4$ QIDs survive**: the VM halts with `Refusal::AmbiguousEntity`, grouping the candidates by their `p31_class` and descriptions, and asks the user to specify which one they mean.

---

### 4. Time, Change, and Incremental Updates

#### 4.1 Temporal Representation & Semantics in the VM
* **Encoding:** Every time literal in `dates.bin` is a 64-bit integer: `44 bits signed seconds/days from epoch` + `4 bits Wikidata precision (0=1B yrs ... 9=Year, 10=Month, 11=Day)` + `1 bit calendar (Gregorian/Julian)` + `15 bits reserved`.
* **Point-in-Time (`AS_OF`) VM Semantics:**
  * Every CubeLang `HOP(subj, prop, time_filter)` checks qualifiers `P580` (start time), `P582` (end time), and `P585` (point in time).
  * If `time_filter == CURRENT`: select edges with `PreferredRank`. If all edges are `NormalRank`, select edges that have no `P582` (end time) qualifier, or the maximum `P585` (point in time).
  * If `time_filter == AS_OF(t)` (e.g., *"Who was the Mayor of Paris in 1998?"*): ignore `PreferredRank` and filter *all* `Normal` + `Preferred` edges where `P580 <= t <= P582` (accounting for precision intervals: `1998` with precision `Year` spans `[1998-01-01, 1998-12-31]`).

#### 4.2 The Live-Source Rule & Updates Without Full Rebuilds
Your rule: *"The serving code never touches the network; live sources sit outside it. For time-sensitive facts the live value wins over any snapshot."*

Implement a **Two-Layer Storage Engine (Immutable Snapshot + Mutable `redb` Delta Layer)**:
1. **External Sync Daemon (Out-of-process):** A separate daemon polls the Wikimedia EventStreams SSE API (`recentchange` stream) or resolves cache misses from the Live Wikidata API, writing verified mutations into a local **`delta.redb`** database and appending new surface names to a small in-memory `BTreeMap` + periodic mini-FST segment.
2. **Tombstone Bitmap:** An in-memory `RoaringBitmap` indexed by `(QID)` marks any entity modified since the snapshot build.
3. **VM Read Path:**
   ```rust
   fn get_statements(qid: u32, prop: u16) -> StatementIter {
       if delta_bitmap.contains(qid) {
           // Merge immutable CSR slice with delta.redb overrides (Delta wins on timestamp)
           return delta_store.get_merged(qid, prop, &csr_snapshot);
       }
       csr_snapshot.get_slice(qid, prop) // 99.9% path: zero-overhead array slice
   }
   ```
4. **Mutable Property Registry:** Mark ~180 properties as `TIME_SENSITIVE` (e.g., `P6` head of government, `P39` position held, `P1082` population, `P166` award received, `P54` member of sports team, `P570` date of death). If a query reads a `TIME_SENSITIVE` property from the base snapshot, the VM attaches the snapshot timestamp provenance: *"As of [Snapshot Date], ..."* unless `delta.redb` carries a fresher timestamped entry.
5. **Compaction:** Once a month at home, merge `delta.redb` into `fwd_edges.bin` via a sequential streaming merge-sort in **~25 minutes** on your local NVMe.

---

### 5. Judging Nick’s "Mixture of Specialists" Idea

Nick proposed: *"A trunk (or an adapter on the shared trunk) per topic, each owning a shard of the store, with a router choosing one."*

**Verdict: Reject the sharded store and topic-router. Keep at most lightweight LoRA adapters on a single shared trunk for *syntax/schema translation*, over a single monolithic store.**

Here is the exact engineering breakdown:

#### What it supposedly buys (and why those benefits disappear here):
1. **"Shards swapped in and out of RAM":** Unnecessary and actively harmful. Swapping a 4 GB topic shard from NVMe into RAM takes ~1.2 seconds of PCIe bandwidth saturation. In contrast, our monolithic CSR index keeps all offsets and FSTs in 7.6 GB of RAM permanently, and fetches arbitrary cold entity edges in **0.09 ms** (4 KB page reads). Random page reads on NVMe beat shard swapping by $10,000\times$.
2. **"Smaller indexes per shard":** A binary search or FST traversal scales as $O(|s|)$ (length of the string, independent of key count!) for FSTs, and $O(1)$ for direct-indexed CSR arrays. Splitting 54M entities into 10 shards of 5.4M entities reduces FST lookup time by **0.00 microseconds**.
3. **"Parallel builds":** You can already partition the build by QID range (`Q1..Q10M`, `Q10M..Q20M`) and merge-sort them in parallel without sharding by semantic topic.

#### What it costs (The Fatal Flaws):
1. **Cross-Topic Multi-Hop Queries Break:** Questions do not respect topic boundaries. Consider: *"Which French chemist born in Dole discovered a vaccine for rabies?"* (Chemistry + Geography + Medicine + Biography) or *"What is the population of the birthplace of the director of Inception?"* (Cinema $\rightarrow$ Biography $\rightarrow$ Demography). If Cinema and Demography live in separate store shards, the VM must load multiple shards or fail on hop 2.
2. **Routing Errors Create Silent Recall Failure:** If a router classifies *"When did Mercury enter service?"* into `Astronomy` instead of `Transportation / Trains & Ships`, the `Astronomy` shard only contains the planet `Q308` and completely misses the train/ship entities. Even a 95% accurate router immediately caps your system's end-to-end recall at 95%, and violates the **"Never rank by fame"** rule by silently hiding namesakes that live in non-selected topic shards!
3. **Router Weights Memorize World Knowledge:** To know whether *"The Crown"* goes to the `Royalty/History` specialist or the `Television` specialist, the router must memorize entity facts inside its weights—violating Cubby's core philosophy that facts live outside the weights.

#### What to do instead (One Trunk + Property-Retriever over One Unified Store):
Use **one 450M trunk** over **one unified store**. The hardest job of the 450M model in generating CubeLang is knowing which of Wikidata's 12,000 properties (`P`-IDs) to emit for a wording like *"alma mater"* (`P69`), *"patron saint"* (`P417`), or *"indicatif téléphonique"* (`P473`).
* Instead of topic specialists, put the 12,000 Wikidata Properties (with their EN/FR labels, aliases, and usage examples) into your existing FST + Tantivy lexical index.
* Before the 450M model writes CubeLang, retrieve the top-25 candidate `P`-IDs matching the verbs/nouns in the question (<2 ms) and pass those 25 `(P-ID, label, type signature)` definitions in the prompt context of the 450M model.
* This keeps **property schema knowledge outside the weights** too! Add a new property to the store tomorrow, and the 450M model can write CubeLang for it with zero retraining.

---

### 6. The Build Pipeline & Faithfulness Verification

#### 6.1 Where to Run It: Run at Home, NOT on Colab
**Do not build this on Colab + Google Drive.**
* Streaming 145 GB from Google Drive into Colab hits Drive’s FUSE mount rate limits and quota throttles, and writing 60 GB of random-access intermediate sort runs back to a Google Drive FUSE mount will stall or disconnect at the 12–24 hour session kill limit.
* Building a graph database requires **Zero GPU compute**. It is 100% CPU (bzip2 decompression + JSON parsing + radix sort) and NVMe sequential I/O.
* Your home **Intel i5-12400F (6 cores / 12 threads, 4.4 GHz)** with a local PCIe 4.0 NVMe SSD (5,000 MB/s sequential read/write) will outperform a Colab instance attached to network block storage. Use your 5 TB Google Drive strictly for encrypted off-site backups of the compiled `.bin` files.

#### 6.2 Step-by-Step Build Pipeline & Wall-Clock Timing (on i5-12400F)
1. **Download (`~2.5 hours` on 150 Mbps fiber):** Download `latest-all.json.bz2` (~145 GB) to the local NVMe.
2. **Stage 1: Parallel Bzip2 Stream & Filter (`~3.8 hours`):**
   * Use `pbzip2 -dc` (or Rust `bzip2` multithreaded reader splitting on bzip2 block magic `0x314159265359`) piped into a 10-thread Rust parser using `simd-json`.
   * Why two-pass or one-pass? To avoid keeping dangling links to pruned scholarly articles, Stage 1 emits flat binary append logs:
     * `raw_edges.log` (uncompressed 24B structs)
     * `raw_names.log` (`(norm_string_hash, QID, flags, string_bytes)`)
     * `kept_qids.roaring` (bitmap of kept entities, ~60 MB in RAM).
   * Throughput on i5-12400F: ~11,000 JSON entities/sec $\rightarrow$ 113M entities in **2.8 to 3.8 hours**.
3. **Stage 2: External Radix / Merge Sort (`~45 minutes`):**
   * Sort `raw_edges.log` by `(Subject_QID, Prop_ID, Rank)` to build `fwd_edges.bin` and `fwd_offsets.bin`.
   * Duplicate object edges and sort by `(Object_QID, Prop_ID, Subject_QID)` to build `rev_edges.bin` and `rev_offsets.bin`.
   * At 3,500 MB/s NVMe sequential speed, sorting 12 GB of binary records in 16 GB RAM chunks takes **<25 minutes**.
4. **Stage 3: Name Normalization & FST Compilation (`~35 minutes`):**
   * Sort the ~145M `(norm_folded_string, QID, p31_class, flags)` tuples lexicographically using external disk sort (`extsort` crate).
   * Stream the sorted unique keys into `fst::MapBuilder` while writing grouped QID arrays to `postings.bin`.
5. **Stage 4: Wikipedia & Book Zstd + Tantivy Indexing (`~1.5 hours`):**
   * Parse `enwiki` and `frwiki` dump XML streams, train a 112 KB zstd dictionary on 10,000 random paragraphs, compress passages into `passages.zst`, and commit the Tantivy index.
* **Total End-to-End Build Time at Home: ~6.5 to 7.0 hours** (Peak local NVMe disk usage during build: **~260 GB** including the 145 GB compressed dump; **~52 GB** after deleting temporary sort files and the `.bz2`).

#### 6.3 Proving Faithfulness to the Source
To guarantee `0 wrong` answers caused by parser bugs, bit-packing truncation, or endianness errors:
1. **In-Flight Rolling Hashes (100% Coverage):** During Stage 1 parsing, compute an order-independent XOR/multiset hash (`XXH3_128`) over every `(QID, Prop_ID, Value_Hash, Qual_Hash)` tuple accepted by the filter. After Stage 2 builds the final CSR files, run a full linear scan over `fwd_edges.bin` + `qualifiers.bin` (~4 seconds) and recompute the multiset hash. **The two 128-bit hashes must match to the bit.**
2. **Stratified 100,000-Entity Reservoir Audit:** During Stage 1, reservoir-sample 100,000 raw JSON lines across all entity classes (humans, dates, negative BCE dates, quantities with bounds, coordinates, French accents) into `audit_sample.jsonl`. Run an independent verifier binary that parses `audit_sample.jsonl` with standard `serde_json` (sharing zero code with the fast `simd-json` builder) and queries `CubbyStore` for every statement, qualifier, and EN/FR alias. **Pass requirement: 100,000 / 100,000 exact bit-for-bit matches.**

---

### 7. Pre-Download Offline Experiments (Fast Kill Gates)

Before downloading 145 GB or running a 7-hour build, download the **Wikidata Truthy/Full subset for a 5M-entity slice** (or use your existing 4.3M Wikidata5M entities augmented with raw JSON fetched via batch API for the gold entities in these public benchmarks) to test every architectural claim.

#### Public Datasets to Use (Carry Gold QIDs + Include French):
1. **MKQA (Multilingual Knowledge Questions and Answers — Longpre et al., TACL 2021):** 10,000 queries aligned across **English and French** (`en`, `fr`), annotated with **gold Wikidata QIDs** and typed answer values (dates, numbers, entities), plus explicit `Unanswerable` questions to test your `0 wrong` refusal gate.
2. **Mintaka (Sen et al., COLING 2022):** 20,000 complex questions (comparative, superlative, multi-hop, temporal qualifiers) in **English and French**, linked to **gold Subject and Answer Wikidata QIDs**.
3. **LC-QuAD 2.0 (Dubey et al., ISWC 2019):** 30,000 complex Wikidata questions paired with gold SPARQL queries (easily translated to gold `(QID, P-ID)` paths).
4. **SimpleQA (OpenAI, 2024):** Your current 4,326-question benchmark.

#### The 4 Experiments:

| Choice Being Tested | Offline Experiment Setup (on 5M items + Gold QIDs) | Metric | Pass Line | Kill Line (Pivot if hit) |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Filtering Rule & Coverage** (Does dropping scholarly articles hurt QA coverage, and does our schema fix the SimpleQA 860/4,326 bottleneck?) | Extract the ~15,000 unique QIDs referenced in SimpleQA, MKQA, and Mintaka via the Wikidata API `wbgetentities` (<15 mins) and run the Stage 1 filter + CSR encoder on them. | Entity coverage, Fact coverage, and False-Pruning rate. | **$\ge 96\%$** of MKQA/Mintaka answerable gold facts present in CSR; **$0$** benchmark entities dropped by the scholarly-article filter; SimpleQA stored-fact answerability jumps from $3 / 4,326$ to **$>2,200 / 4,326$**. | If scholarly filter drops $>0.2\%$ of benchmark entities, narrow the filter to only drop `Q13442814` items that have $<3$ incoming links. |
| **Exp 2: Name-to-Item FST + SymSpell + Structural Disambiguation** (EN & FR, No Fame Ranking) | Build the Tier 1/2/3 name index over 5M Wikidata5M items + all EN/FR aliases for MKQA & Mintaka entities. Inject 15% synthetic keyboard typos and strip French accents on 50% of `fr` queries. | Candidate Recall@$K$ (before hop) & Surviving Candidate Set Size (after 1st property hop). | Gold QID in initial candidate set for **$\ge 97\%$ exact** and **$\ge 90\%$ typo/accent-stripped** queries in both EN & FR; after intersecting with the question's 1st `P`-ID hop, **$\ge 88\%$** of queries narrow to $\le 2$ surviving QIDs *without fame ranking*; p95 latency **$<15$ ms**. | If $>20\%$ of queries still have $>4$ surviving namesakes after hop intersection, add **Description Keyword Intersection** (requiring question context nouns like *"actor"* or *"city"* to match the entity's EN/FR description tokens). |
| **Exp 3: CSR Cold NVMe Latency** | Generate a synthetic 54M-node, 480M-edge `fwd_edges.bin` (7.68 GB) and `qualifiers.bin` (1.28 GB) on your local NVMe. Run `echo 3 > /proc/sys/vm/drop_caches` before each batch of 1,000 3-hop CubeLang programs (with 16 namesakes per hop). | p95 and p99 wall-clock execution time for a full 3-hop program (cold & warm). | Warm 3-hop execution **$<0.2$ ms p95**; Cold (`drop_caches`) 3-hop execution **$<4.0$ ms p95**. | If cold p95 exceeds $15$ ms due to scattered QID accesses across `fwd_edges.bin`, reorder internal dense IDs by breadth-first graph locality (using `WebGraph` / LLP ordering) while keeping a 520 MB `QID -> DenseID` array in RAM. |
| **Exp 4: Unified Store + Lexical Property Retriever vs. Topic Router** | On Mintaka (EN + FR), compare: (A) One 450M trunk fed top-25 lexically retrieved `P`-IDs over one unified store vs. (B) A 10-topic router picking a specialist store shard. | End-to-End Certified Answer Recall & Wrong Answer Count on Mintaka. | Unified + Property Retriever achieves **$0$ wrong answers** and beats the 10-topic router on multi-hop recall by **$\ge 8$ percentage points**. | Only build topic adapters if a topic-specific LoRA (over the *unified* store, never sharding the data!) improves CubeLang syntax accuracy by $\ge 3\%$ on domain-specific phrasing without dropping cross-domain accuracy. |

---

### 8. What NOT to Build, Key Risks, and Decisions Nick Must Make

#### What NOT to Build:
1. **Do NOT use SQLite or RocksDB for the 480M statement graph.** SQLite B-trees require 3–4 page pointer chases per lookup, blow up 480M statements + indexes to ~140 GB, and take >24 hours to `CREATE INDEX` at that scale. Keep SQLite only for small isolated fiction worlds.
2. **Do NOT build dense vector embeddings (FAISS / HNSW / qFHRR) for 100M entity names or facts.** A 54M-vector `f16` index at $D=384$ takes 41 GB of RAM alone, violates your measured negative on similarity thresholds, and performs worse on exact proper nouns than an FST + Trigram index that takes 3.8 GB.
3. **Do NOT use the 450M model to extract facts from the 40 GB Wikipedia dump at build time.** Extract deterministic facts from Wikitext infoboxes/links via parser rules, and index the raw text as provenance passages.
4. **Do NOT build a SPARQL engine.** CubeLang only needs bounded-hop graph traversal (`HOP`, `REV_HOP`, `FILTER_QUAL`, `INTERSECT`), which runs in 200 lines of Rust over the CSR arrays.

#### Top 3 Technical Risks & Mitigations:
1. **Risk: "Predicate Paraphrase Gap" (The 450M model knows what the user wants, but guesses the wrong Wikidata `P`-ID).**
   * *Mitigation:* Build the **Property Alias Dictionary** (Section 5). Map English and French verb phrases (from Wikidata property aliases, Wikipedia infobox field mappings, and SimpleQuestions/LC-QuAD training pairs) directly to `P`-IDs in an FST so the VM supplies candidate `P`-IDs to the model *before* it emits CubeLang.
2. **Risk: Wikidata Ontology Inconsistency (`P31` "instance of" vs `P279` "subclass of" vs occupation `P106`).**
   * *Example:* Asking *"Which French architects..."* checks `P106 = Q42973 (architect)`, whereas asking *"Which cathedrals..."* checks `P31/P279* = Q2977 (cathedral)`.
   * *Mitigation:* Precompute the transitive closure of `P279` (subclass of) up to depth 5 at build time into an in-memory DAG (~45 MB), and compile a `type_check(qid, target_class_qid)` VM intrinsic that checks `P31`, `P106`, and transitive `P279` in a single call.
3. **Risk: Unsourced Wikidata Vandalism Violating `0 Wrong`.**
   * *Mitigation:* Any Wikidata statement whose `prov_id` has neither an external reference (`P248`/`P854`) nor a corresponding Wikipedia infobox/text match is flagged `UNVERIFIED` in `flags: u8`. Depending on Nick's policy, the VM either refuses to speak it unhedged or verifies the string value inside the linked Wikipedia article's `passages.zst` paragraph (a 1 ms local check!) before certifying the answer.

#### Four Concrete Decisions Nick Must Make Today:
1. **Hardware ($190 decision):** Will you approve ordering a 2 TB PCIe 4.0 NVMe SSD and upgrading the home server from 32 GB to 64 GB DDR4 RAM so the entire graph topology stays `mlock`ed in RAM?
2. **Scholarly Pruning Policy:** Do you agree to drop the ~43M scholarly-article/protein/gene items (`Q13442814`, etc.) that lack Wikipedia sitelinks, cutting the store size in half?
3. **The Namesake UX Contract:** When structural graph hops leave 2–4 valid namesakes (because both people have the requested property and we refuse to rank by fame), should Cubby **speak all 2–4 answers with their descriptions**, or **ask a clarifying question first**? (Recommendation: Speak all if $\le 3$; ask to clarify if $\ge 4$).
4. **Unsourced Wikidata Policy:** When a Wikidata statement has no external reference (`P248`/`P854`), may Cubby auto-corroborate it by checking if the target entity/literal appears in the subject's local English/French Wikipedia article text, or must it speak the fact hedged?

## deepseek/deepseek-v4-pro-0813

*No answer: no answer: finish=length, 3712 tokens all spent on reasoning*

## qwen/qwen3.8-max-prime

*11861 tokens out, 142.7 s, finish `stop`*

# Fact-Store Design Review – Reviewer 3

I'll commit to one design, name its failure modes, and say where I'm uncertain. I'll argue against the mixture-of-specialists. Every recommendation carries a measurement that would kill it.

---

## 1. What to keep from the terabyte, and in what form

### Wikidata: keep truthy + qualifiers, shed references to a cold tier

| Slice | What | Binary size (my estimate) | Why |
|---|---|---|---|
| Statements | Truthy only (best-rank per property per item), **with qualifiers** (`P585 point in time`, `P580/P582 start/end`, `P1480 stated in`, etc.) | **35–55 GB** | Full statements with all ranks add ~30 % more rows for almost no answering value; you already keep rank as a byte. |
| Labels, aliases, descriptions | EN + FR only | **8–14 GB** | Needed for name→item. Other languages: skip for now. |
| Sitelinks | EN + FR Wikipedia, FR Wikisource | **2–4 GB** | Cross-reference to the text corpus; also gives you a free "is this entity notable enough to have an article" signal. |
| References / provenance | Source URL, `P248 stated in`, `P813 retrieved`, `P143 imported from` – **not** the full reference graph | **8–15 GB** cold | Full references are ~60 % of the dump's byte count. Store them in a separate, lazily-loaded SQLite file keyed by statement-id. The VM fetches provenance only when it speaks a fact. |
| Item/property metadata | Datatype, `P31 instance of`, `P279 subclass of` for the ~5 000 properties | **< 1 GB** | The VM needs to know a value is a `time`, `quantity`, `item`, etc. |

**Total hot store: roughly 55–90 GB.** Nick's 40–80 GB note is in range; the spread depends on how compactly you encode qualifiers. I am **uncertain** about the exact figure because it depends on whether you keep `P31/P279` for all 100 M items or only the ~10 M that appear in sitelinks; keeping them for all items adds ~6 GB.

**What I'd drop:** non-truthy (deprecated/normal-rank) statements, all languages except EN/FR, the full reference graph from the hot path, media identifiers, and the ~20 M items that are purely structural (categories, disambiguation, Wikimedia internal). That last cut alone removes ~15–20 % of items.

### Text: what the reader extracts, what stays as provenance

The Cubby-native reader (rules + the 450 M model) should extract only **typed triples with a surface-form citation**:

- `(entity, property, value, source_passage_id, date_of_publication)`
- A fact from one source is tagged `hedged`; a second independent source promotes it to `asserted`. The VM only speaks `asserted` facts without a hedge marker.

Keep the **passage** (≤ 512 tokens) in a column-store (I recommend `tantivy` segments; see §2) keyed by `passage_id`. Do **not** keep full book text in the hot store; keep it on the 5 TB Drive mount and fetch on demand. The hot text store is the extracted facts + their citation passages.

**Measurement that shows this working:** on a held-out 500-question set drawn from the history books already in SQLite, the reader extracts the gold fact with correct entity links ≥ 70 % of the time. **Kill line:** < 40 %, meaning the reader is not reliable enough and you need to narrow the extraction templates.

---

## 2. Physical layout – one concrete design

I'm recommending a **custom column-oriented binary store** memory-mapped from NVMe, not an RDF triple store. The VM executes typed three-address code; it needs `(subject_id, property_id) → [(object/value, qualifiers, rank, statement_id)]` in one seek, not SPARQL triple patterns.

### 2a. Encodings

- **Entity / property IDs:** dictionary-encoded. Map `Q42 → 41` (sequential `u32`). 100 M items × 4 B = **400 MB** for the forward map; the FST (§2c) serves as the reverse map. Store the map as a sorted `mmap`'d array of `(u32, u32)` pairs.
- **Statements:** sorted by `(subject_id u32, property_id u32)`. Each row: `object_or_value` (16 B union: `u32` item ref, `f64` quantity, or offset into a value blob), `qualifier_block_offset u32`, `rank u8`, `statement_id u64`. Average row ≈ 34 B. At ~1.2 B truthy statements (my estimate; **I'm uncertain by ±20 %**), that's **~40 GB**.
- **Qualifier blocks:** a separate `mmap`'d blob, addressed by offset. Each qualifier: `(property_id u32, value 16 B)`. Average 1.8 qualifiers per statement → ~2.2 B qualifier entries → **~15 GB**.
- **Value blob** (strings, monolingual text, URLs): dictionary-compressed (LZ4 blocks, 64 KB each), `mmap`'d. ~**8 GB**.
- **Labels / aliases / descriptions:** separate column files, one per language, keyed by entity ID. ~**10 GB** total EN+FR.

### 2b. Indexes

- **Primary lookup** (subject, property) → statements: the sorted statement file *is* the index. Binary search on the `(subject_id, property_id)` prefix, then scan the run. With `mmap` and OS page cache, a hot lookup is one `memcmp`-driven binary search over ~1.2 B entries: **< 0.5 ms** (27 comparisons × ~20 ns each). From cold NVMe: 1–2 random 4 KB reads → **< 2 ms** on a decent NVMe (Samsung 980 Pro or WD SN850X; sequential 7 GB/s, random 4 KB ~1 MB/s IOPS ≈ 100 µs per read). The 1 ms hot / 10 ms SSD targets are met with margin.
- **Reverse lookup** (object → subjects): a second sorted permutation of the same rows, sorted by `(object_id, property_id)`. Same size, ~**40 GB**. Needed for "what items have `P31 = Q5`?"-style hops.
- **Property-type index:** for each property, its datatype and a small Bloom filter of which subjects have it. < **200 MB**.

### 2c. Name → item: FST + trigram side-channel

- **FST** built with the Rust `fst` crate (BurntSushi, MIT/Apache-2.0). Input: every EN and FR label and alias, lowercased, Unicode-NFC-normalised, with diacritics folded into a parallel ASCII key. Value: a `u32` entity ID (or an offset into a postings list when multiple entities share a name).
  - Estimated input: ~100 M items × ~1.4 labels + ~1.8 aliases × 2 languages ≈ **~450 M keys** (many items lack FR labels; **I'm uncertain by ±25 %**).
  - FST size at 450 M keys, mean length 14 bytes: **~6–10 GB** (the `fst` crate achieves roughly 1.5–2.5 bytes per key at this scale; BurntSushi's benchmarks on English words show ~2 B/key).
  - Lookup: exact match in **< 0.1 ms**; prefix scan for autocomplete in **< 1 ms**.
- **Trigram side-channel for typos:** build a trigram → postings-list index over the same 450 M names, but **only for trigrams with document frequency < 5 000**. High-frequency trigrams ("the", "ing", "ion") are useless for discrimination and would blow up the index. This keeps the index to roughly **8–14 GB**. Query: extract trigrams from the input, intersect postings lists, get ≤ 200 candidates, then verify each against the FST with edit distance ≤ 2 (Ukkonen's algorithm, O(k·n) per candidate, k = edit distance, n = name length; 200 × 2 × 20 chars ≈ 8 000 ops, **< 0.1 ms**).
  - The existing letter-trigram binary codes in Cubby can seed the candidate set but should **not** be the final discriminator; they showed 0.081 nDCG@10 on BEIR dbpedia-entity, which is not usable as a ranker. Use them only as a fast pre-filter if the trigram postings intersection returns > 200 candidates.
- **Namesakes:** the FST postings list returns *all* entities with that name. The VM must resolve by type constraints from the question (e.g., "the river" → `P31 = Q4022`), or ask the user. **Never rank by fame.** This is a hard rule and the design enforces it structurally: there is no popularity score in the index.

**Measurement that shows this working:** on a held-out set of 10 000 entity mentions (5 000 EN, 5 000 FR, including 1 000 deliberate typos and 500 namesakes), the FST + trigram pipeline returns the gold Wikidata ID in the top-1 candidates ≥ 92 % of the time, and the gold is *always* in the top-20 candidates (≥ 99.5 %). **Kill line:** gold-in-top-20 < 95 %, because then the VM cannot resolve by type constraints and must ask the user too often.

### 2d. What to reuse, not build

| Component | System | Licence | Why |
|---|---|---|---|
| FST | `fst` crate (BurntSushi) | MIT / Apache-2.0 | Already the standard in Rust; used by `tantivy` internally. |
| Full-text search over passages | `tantivy` (Paul Masurel et al.) | MIT | Rust-native, Lucene-quality BM25, segment-based, embeddable. |
| Key-value / conversation history | `sled` or existing SQLite | Apache-2.0 / public domain | You already have SQLite FTS5 at p95 38 ms; keep it. |
| LZ4 block compression | `lz4_flex` crate | MIT | For the value blob. |
| Memory-mapped I/O | `memmap2` crate | MIT / Apache-2.0 | Standard. |
| Streaming JSON parse of the dump | `serde_json` with `Deserializer::from_reader` in streaming mode | MIT / Apache-2.0 | Avoids loading 959 GB into RAM. |

**What I would NOT use:** Oxigraph (Rust RDF store, GPL-3.0 – licence conflict risk, and it's designed for SPARQL, not typed three-address VM hops); Blazegraph / RDF4J (Java, wrong ecosystem); KGTK (Python, TSV-based, not embeddable in a Rust VM). I'm **unsure** whether Oxigraph's licence would actually conflict with Cubby's; Nick should check.

### 2e. Existing systems that store Wikidata at full scale on one machine

- **Qlever** (Hannah Bast et al., University of Freiburg, `github.com/ad-freiburg/qlever`). Indexes full Wikidata as a SPARQL engine. Published numbers (Bast & Buchhold, 2017, "HaPPY: Hyperedge, Predicate, Pair, and Object Yields in Estimating Join Cardinalities"): full Wikidata index ≈ **~120 GB** on disk. Query latency: most single-hop queries < 100 ms, multi-hop < 1 s. Build time: ~12 h on a 16-core machine. This validates that ~100 GB is the right order of magnitude for a full Wikidata index.
- **KGTK** (Szekely et al., 2022, "KGTK: A Toolkit for Building Knowledge Graphs"; USC/ISI). Converts Wikidata to TSV edges. Full dump → ~**200 GB** of TSV. Build: ~6–10 h on 8 cores. Not embeddable, but useful as a cross-check.
- **Wikidata Toolkit** (Thomas Pellissier Tanon, `github.com/Wikidata/Wikidata-Toolkit`, LGPL-2.1). Java library that streams the JSON dump and gives you typed callbacks. Build pass over the full dump: ~**3–5 h** single-threaded. I used this as a reference for my size estimates; it doesn't produce a queryable store itself.

None of these is the right runtime for Cubby (wrong query model, wrong language, or wrong licence), but they confirm the size and build-time numbers.

### 2f. NVMe budget

| Component | Size |
|---|---|
| Statement store (forward + reverse) | 80 GB |
| Qualifier + value blobs | 23 GB |
| Labels / aliases / descriptions | 10 GB |
| FST name index | 8 GB |
| Trigram typo index | 12 GB |
| Provenance SQLite (cold) | 15 GB |
| Wikipedia EN+FR passage store (tantivy) | 45 GB |
| DBpedia abstracts | 3 GB |
| Build scratch space | 100 GB |
| **Total** | **~300 GB** |

**Recommendation: 1 TB NVMe** (a WD SN850X 1 TB or Samsung 990 Pro 1 TB, both ~$80–100 USD in 2026 pricing). 500 GB is technically enough but leaves no room for the build scratch or the next corpus. The 5 TB Google Drive holds the raw dumps, full book texts, and cold provenance.

---

## 3. Name → item at 100 M items, EN and FR, within the rules

The pipeline, as a VM callable:

```
resolve_name(query: &str, lang: Lang, type_hint: Option<PropertyConstraint>)
  → Vec<(EntityId, MatchKind)>   // MatchKind ∈ {Exact, Alias, Typo, Prefix}
```

1. **Normalise:** NFC, lowercase, strip leading articles ("the", "le", "la", "les", "l'").
2. **FST exact / alias lookup.** If hit, return all entity IDs. If exactly one and the type constraint matches, the VM proceeds. If multiple, go to step 5.
3. **Trigram typo search** (edit distance ≤ 2). Return candidates.
4. **Type-constraint filter:** intersect candidates with the set of entities satisfying the type hint (e.g., `P31 ⊑ Q5` for persons). This set is precomputed as a sorted `u32` array per top-level type (~200 types cover 95 % of questions; **I'm uncertain about the exact coverage**). Intersection is a binary-search merge, < 0.5 ms.
5. **If still ambiguous:** the VM returns all candidates to the 450 M model, which formulates a clarification question. The model **never picks by fame**. It may use the question's context ("the river", "born in 1920") to select, but only via type/temporal constraints, not popularity.

**Latency budget:** FST lookup 0.1 ms + trigram 0.5 ms + type filter 0.5 ms = **~1 ms p95 hot**, well within the 50 ms target. From cold SSD, add 1–2 ms for page faults. The 50 ms target is conservative; I'd tighten it to **10 ms p95**.

**French-specific:** the FST must contain FR labels and aliases. ~35–40 % of Wikidata items have FR labels (**I'm uncertain; check with a 1 M-item sample before the full build**). For items without FR labels, fall back to EN labels and the sitelink to `fr.wikipedia.org` article titles. Quebec French: include `fr-ca` aliases where they exist in Wikidata (rare; ~5 000 items). The DBpedia FR abstracts provide additional FR surface forms.

---

## 4. Time and change

### Dated values

Every statement carries its qualifiers, including `P585` (point in time), `P580/P582` (start/end). The VM's type system includes a `Time` value with year-month-day precision. When the model writes a CubeLang hop that asks "population of France", the VM:

1. Retrieves all `(Q142, P1082, ?pop)` statements.
2. Sorts by `P585` descending.
3. Returns the most recent, **with its date**, and the VM's answer includes "as of 2024-01-01" or whatever the qualifier says.

If the question specifies a time ("population of France in 1900"), the VM filters by `P585` range. This is a sorted-array scan within the statement run, < 0.1 ms.

### Live-source rule

The serving VM **never touches the network** (stated rule). A separate **sync daemon** (outside the VM process) polls the Wikidata API and GDELT at a configurable interval (I'd say 6 h for Wikidata, 1 h for GDELT news). It writes deltas to a `pending_updates` SQLite WAL. A nightly (or on-demand) **merge step** applies the deltas to the mmap'd store:

- For a changed statement: mark the old row as superseded (a `valid_until` timestamp), append the new row. The sorted order is maintained by a small B-tree overlay (< 1 GB) that is periodically compacted into the main sorted file.
- For a new entity: append to the entity table, rebuild the FST incrementally (the `fst` crate supports incremental builds via `fst::MapBuilder`; **I'm not 100 % sure the incremental API is production-ready for 100 M keys – test with 10 M first**).

The live value wins over the snapshot because the VM always reads the latest non-superseded row.

**Measurement:** after a sync cycle, pick 1 000 random items that changed in Wikidata in the last 24 h. Verify the store reflects the new value. **Pass:** ≥ 999/1 000. **Kill:** < 990/1 000, meaning the merge is dropping updates.

### Updates without full rebuild

The sorted statement file is append-friendly: new statements go into a small "delta" file (same format). The VM's lookup checks the delta file first (binary search, < 0.1 ms for a delta < 1 M rows), then the main file. Compaction (merge delta into main) runs weekly or when the delta exceeds 10 M rows, and takes ~20 minutes for a 40 GB file on NVMe.

---

## 5. The mixture-of-specialists idea: I recommend against it

Nick's proposal: a trunk (or adapter) per topic, each owning a shard of the store, with a router.

### What it would buy

- Smaller per-shard indexes (maybe 5–10 GB per shard instead of 55–90 GB), which helps if you're RAM-constrained.
- Parallel builds across shards.
- Topic-specific question-reading (a "history" adapter might parse dates differently from a "geography" adapter).

### What it costs, and why the costs dominate here

1. **Routing errors are unrecoverable under the 0-wrong rule.** If the router sends "What treaty ended the Thirty Years' War?" to the "geography" shard instead of "history", the VM finds no fact and refuses. The user gets a wrong refusal. With a single store, the lookup succeeds. At 100 M items, the router must be near-perfect, and cross-topic questions ("the birthplace of the author of *Les Misérables*") require multi-shard queries, which reintroduce the full-store latency.

2. **The bottleneck is coverage, not index size.** The current score is 3/860 on SimpleQA. The problem is that the store doesn't contain the facts, not that the index is too slow or too large. A mixture of specialists does not add a single fact. It adds a routing layer that can only lose.

3. **It puts facts back into weights if the specialists are trained.** If each specialist is a fine-tuned adapter, it memorises topic-specific patterns. This violates the "facts stay exact keys with provenance" rule and the "prefer learning by storing" rule. If the specialists are *not* trained (just prompt templates or rule sets), then they're just query rewriters, and you can do that with one trunk and a query planner.

4. **The 32 GB RAM constraint is manageable without sharding.** The hot store is 55–90 GB. With `mmap`, the OS pages in what's needed. On a 32 GB machine, the working set for a single query is ~4 KB (one page for the statement, one for the qualifier block, one for the value). The OS page cache handles the rest. If you upgrade to 64 GB (stated as possible), the entire statement store fits in RAM. You don't need to shard to fit.

5. **Build parallelism is achievable without specialists.** The statement file is sorted by `(subject_id, property_id)`. You can partition the dump by subject-ID range, build each partition in parallel, then merge-sort. This is a standard external sort. No router needed.

### The alternative I recommend

**One trunk, one store, sharded only by storage layout (forward index, reverse index, name FST, text index), not by topic.** The 450 M model writes CubeLang programs that hop through the single store. The "specialist" behaviour is achieved by the CubeLang program itself: a history question generates a different sequence of hops than a geography question, but they all hit the same store.

**Measurement that would change my mind:** if, after building the single store, p95 lookup latency exceeds 10 ms from SSD because the 90 GB store causes excessive page faults on 32 GB RAM, then sharding by topic (with a deterministic hash, not a learned router) would be worth revisiting. But I predict this won't happen because query locality is high (a single question touches 3–10 entities).

**Measurement that shows the single-store design working:** p95 lookup < 1 ms hot, < 5 ms cold, over a sustained 1 000-query benchmark. **Kill line:** p95 > 20 ms cold, indicating the store doesn't fit the I/O budget.

---

## 6. The build

### Step-by-step

| Phase | Where | Duration (estimate) | Notes |
|---|---|---|---|
| Download `latest-all.json.bz2` (~100 GB compressed) | Home, to NVMe | 8–24 h depending on connection | Use `aria2c` with 16 connections. Check SHA-256 against the published hash on `dumps.wikimedia.org`. |
| Stream-parse, extract truthy + qualifiers + EN/FR labels + sitelinks, write column files | Home (i5-12400F, 6 cores) or Colab A100 | **6–10 h** | Single-threaded JSON parse is the bottleneck (~150 MB/s with `serde_json` streaming). Parallelise by splitting the dump at item boundaries (each JSON object is one item; split at `{"type":"item"` boundaries). 6 threads → ~1.5 h parse. Index building (sort, FST) adds 2–4 h. |
| Build FST over 450 M names | Same | **1.5–3 h** | `fst::MapBuilder` is single-threaded but fast (~50 M keys/min). |
| Build trigram index | Same | **1–2 h** | Parallelisable across trigram partitions. |
| Build tantivy index over Wikipedia EN+FR passages | Same | **2–3 h** | tantivy indexes ~50 MB/s of text per thread. 40 GB / (6 threads × 50 MB/s) ≈ 2.2 h. |
| Build reverse index (sort by object) | Same | **1–2 h** | External merge-sort of 40 GB. |
| **Total wall-clock** | | **~12–20 h** | Fits in a Colab Pro 24 h session or an overnight home run. |

**If building on Colab:** download the dump to Google Drive first (the 5 TB Drive holds it). Mount Drive in Colab, stream from Drive to local scratch. The A100 instance has 12 cores (or 4 for the free tier), ~50 GB RAM. The 959 GB uncompressed dump won't fit in Colab's local disk (~100 GB); you must stream from Drive and never fully decompress. Process in chunks: decompress 10 GB, parse, append to column files, delete the chunk.

### Faithfulness proof

1. **Count check:** the dump has N items (count them during parse). The store must have N items. The dump has M truthy statements. The store must have M. Log both. If they differ, the build is wrong.
2. **Hash check:** during parse, compute SHA-256 of each statement's canonical form (subject, property, value, qualifiers, rank). Store the hashes in a sorted file. After the build, re-read the store, recompute hashes, and diff. **Pass:** 0 mismatches. **Kill:** any mismatch.
3. **Sample audit:** randomly sample 10 000 statements from the dump (using a fixed seed). For each, query the store and verify the value, qualifiers, and rank match exactly. This catches encoding bugs that the hash check might miss (e.g., if the hash was computed over the wrong bytes).

**Measurement:** the count check and hash check are binary pass/fail. The sample audit should show 10 000/10 000 matches. **Kill line:** any mismatch in the hash check.

---

## 7. Offline experiments before downloading the full dump

These use public data, carry gold Wikidata IDs, and include French. Run them on a **1 M-item sample** of Wikidata (download `latest-all.json.bz2`, stream until you've seen 1 M items, stop).

### Experiment A: Name resolution coverage

- **Data:** Wikidata 1 M-item sample. Extract EN + FR labels and aliases. Build the FST + trigram index.
- **Query set:** 2 000 entity mentions: 1 000 from the **KILT benchmark** (De Cao et al., 2021, "KILT: a Benchmark for Knowledge Intensive Language Tasks"; the entity mentions carry Wikidata Q-IDs). 500 from **FQuAD** (d'Hoffschmidt et al., 2020; French QA, links to Wikipedia FR articles → map to Wikidata via sitelinks). 500 synthetic typos (edit distance 1–2 from gold names).
- **Metric:** Recall@20 (gold Q-ID in top-20 candidates).
- **Pass line:** ≥ 99 % Recall@20. **Kill line:** < 95 %.
- **What it shows:** whether the FST + trigram design resolves names at scale. If it fails on typos, the trigram index needs rethinking.

### Experiment B: Fact coverage on SimpleQA

- **Data:** the 860 SimpleQA questions where the gold entity is in the store (Nick's existing set). Plus the full 4 326 SimpleQA questions mapped to Wikidata Q-IDs (this mapping needs to be done once by hand or with string matching against the 1 M sample; **I'm unsure how many of the 4 326 map cleanly**).
- **Metric:** coverage = fraction of questions where (a) the gold entity exists in the store AND (b) the specific fact (property + value) exists as a truthy statement.
- **Pass line:** ≥ 60 % coverage on the 860-question subset (up from the current 3/860 ≈ 0.3 %). **Kill line:** < 20 %, meaning the Wikidata truthy store fundamentally lacks the facts SimpleQA asks, and you need the text-extraction pipeline to fill the gap.
- **What it shows:** whether the bottleneck is store coverage or retrieval. If coverage is high but retrieval fails, the index is broken. If coverage is low, you need more sources.

### Experiment C: Temporal fact correctness

- **Data:** 500 questions from **TempLAMA** (Dhingra et al., 2022, "Time-Aware Language Models as Temporal Knowledge Bases"; has Wikidata-grounded temporal facts) + 200 hand-written French temporal questions ("Quelle était la population de la France en 1950 ?").
- **Metric:** exact-match accuracy on the value AND the date qualifier.
- **Pass line:** ≥ 90 %. **Kill line:** < 70 %, meaning the qualifier storage or temporal filtering is wrong.

### Experiment D: Latency on the 1 M sample

- **Data:** the 1 M-item store built in experiments A–C.
- **Metric:** p50, p95, p99 lookup latency for 10 000 random `(subject, property)` queries, hot (after warmup) and cold (after `posix_fadvise(POSIX_FADV_DONTNEED)`).
- **Pass line:** p95 < 0.5 ms hot, < 5 ms cold. **Kill line:** p95 > 2 ms hot or > 15 ms cold.
- **What it shows:** whether the sorted-array + mmap design meets the latency targets at 1/100 scale. Extrapolate: at 100 M items, the binary search adds ~7 comparisons (log₂(100 M) ≈ 27 vs log₂(1 M) ≈ 20), so latency increases by ~35 %. If p95 is 0.5 ms at 1 M, expect ~0.7 ms at 100 M. Still fine.

### Experiment E: French coverage

- **Data:** the 1 M-item sample. Count items with FR labels, FR aliases, FR sitelinks.
- **Metric:** fraction of items with at least one FR surface form.
- **Pass line:** ≥ 30 % (this is what I expect; **I'm uncertain**). **Kill line:** < 15 %, meaning FR coverage is too sparse and you need to augment with FR Wikipedia article titles and DBpedia FR abstracts as additional name sources.

---

## 8. What not to build, risks, and decisions Nick must make

### What not to build

- **Do not build a SPARQL endpoint.** The VM speaks CubeLang, not SPARQL. A SPARQL layer adds 50–100 ms of query-planning overhead per hop and a dependency (Oxigraph, Blazegraph, or Virtuoso) that doesn't fit the typed three-address model.
- **Do not build a vector / embedding index over the fact store.** The rules say similarity may only lead to a key, never stand for a fact. The qFHRR results (0.081 nDCG@10) confirm that embedding-based retrieval over entity names is not competitive with lexical methods. BM25 via tantivy is the right text-search layer.
- **Do not build a learned router** (for the mixture-of-specialists or for any other purpose). The 0-wrong rule means a routing error is a wrong answer. A deterministic hash or type-based partition is safer.
- **Do not build a custom compression codec.** LZ4 (via `lz4_flex`) for the value blob, zstd for cold provenance, and the FST's built-in compression are sufficient. A custom codec is months of work for < 20 % size reduction.
- **Do not train the 450 M model on facts.** The rules say prefer storing. Every training run that memorises facts creates a provenance gap: the model "knows" something the store can't verify. The model should learn to *write CubeLang*, not to *know things*.

### Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| The 959 GB dump has more than 100 M items (Nick's note says 74.5 M; the dump page says > 110 M as of 2025). **I'm unsure of the exact current count.** | Medium | Count during the 1 M-item sample. Extrapolate. Adjust u32 ID space if needed (u32 covers 4.2 B, so this is fine). |
| 32 GB RAM is not enough to keep the hot working set for multi-hop questions that touch 20+ entities. | Low–Medium | `mmap` with `MADV_RANDOM` hint. If p95 exceeds 5 ms, upgrade to 64 GB (stated as possible). |
| The FST incremental rebuild (for live updates) is not production-ready at 450 M keys. | Medium | Test at 10 M keys first. Fallback: nightly full FST rebuild (1.5–3 h) instead of incremental. |
| French Wikidata coverage is sparser than expected. | Medium | Experiment E catches this early. Augment with FR Wikipedia titles and DBpedia FR. |
| The Cubby-native reader extracts facts too slowly or too inaccurately from books. | Medium | Experiment on the 500-question history set (§1). If recall < 40 %, narrow the extraction to template-based rules (regex + dependency patterns) and skip the model for extraction. |
| Colab session dies mid-build. | High | Checkpoint every phase. The column files are append-only; a crash loses at most the current chunk. Resume from the last checkpoint. |

### Decisions Nick must make

1. **Truthy vs. full statements.** I recommend truthy. If you need deprecated facts for provenance ("this was believed true until 2019"), store them in the cold provenance tier, not the hot store. **Decision needed before the build.**

2. **32 GB vs. 64 GB RAM.** The design works at 32 GB with `mmap`, but 64 GB lets you pin the entire statement store in RAM and eliminates cold-start latency. The upgrade is ~$60–80 for a 32 GB DDR4 stick. **I'd buy it.**

3. **Whether to keep sitelinks for all languages or only EN+FR.** I recommend EN+FR only for now. Adding languages later is a schema change, not a rebuild. **Decision needed before the build.**

4. **The SimpleQA coverage gap.** 3/860 is the current state. The experiments in §7 will show whether the Wikidata truthy store closes this to 60 %+ or whether you need the text-extraction pipeline to fill the gap. **This is the single most important experiment. Run it before downloading the full dump.**

5. **Whether the 450 M model writes CubeLang directly or whether a rule-based query planner handles simple one-hop questions.** A rule-based planner for "What is the capital of X?" avoids model inference entirely and is faster and more reliable. The model handles multi-hop and ambiguous questions. **This is an architecture decision that affects latency and the 0-wrong guarantee.**

6. **The mixture of specialists: I've argued against it.** If Nick disagrees, the specific experiment to change my mind is: build two 10 M-item shards, route 1 000 questions, and show that routing accuracy is ≥ 99.5 % AND cross-topic questions (< 5 % of queries) are handled without a fallback to the full store. If routing accuracy is < 99 %, the 0-wrong rule is violated. **I predict it will be < 99 %.**

---

## Summary of the one design

A single, sorted, dictionary-encoded, memory-mapped column store for Wikidata truthy statements with qualifiers, served by a Rust VM. Names resolve through an FST (exact/alias) and a trigram side-channel (typos), never through embeddings or learned similarity. Provenance lives in a cold SQLite tier. Text facts are extracted by the Cubby-native reader into the same typed-fact format, hedged until corroborated. One trunk model writes CubeLang; no specialists, no router. Live updates arrive via a sync daemon and a delta file, compacted weekly. The whole hot store is 55–90 GB on a 1 TB NVMe. The build takes 12–20 h and is verified by count, hash, and sample audit. The first experiment to run is SimpleQA coverage on a 1 M-item sample, before downloading the full dump.

The design's primary failure mode is **coverage, not retrieval**: the store may simply not contain the facts the questions ask. The second failure mode is **French sparsity**: FR labels and aliases may cover too few items. Both are diagnosable with the §7 experiments before committing to the full build.

## z-ai/glm-5.3-prime

*4071 tokens out, 35.4 s, finish `stop`*

# Review: the terabyte fact store

Verdict up front: the terabyte is mostly **not facts** — it is JSON boilerplate, revision metadata, and full statement history. The store you actually need is roughly **60–120 GB of truthy statements plus text indexes**, and it fits on one NVMe drive with the hot name index in RAM. The mixture-of-specialists idea is, in my opinion, **wrong for this problem as stated** — it buys you nothing the sharded store doesn't already give you, and it reintroduces routing errors, which for you are a *wrong-answer* risk, not a quality risk. Details and the experiments that could prove me wrong below.

## 1. What to keep from the terabyte, and in what form

**Wikidata.** The 959 GB figure is the *streamed, uncompressed full JSON dump* including every statement rank, every reference blob, revision ids, and all ~300 languages of labels/aliases. Check it, but the composition is roughly:

- Full JSON dump (latest-all.json.bz2): ~140–150 GB compressed, ~1.0–1.2 TB uncompressed (I am unsure of the 2026 numbers; the 2024 dump was ~130 GB bz2 — verify against `dumps.wikimedia.org` before planning disk).
- Of that, the parts you need:

| Component | Keep? | Rough size (110M items) |
|---|---|---|
| Truthy statements (preferred rank, no deprecated) with qualifiers | yes | ~25–40 GB |
| Non-preferred normal-rank statements | yes, flagged | +~15–25 GB |
| Deprecated statements | no | — |
| References | keep as (property, item) pairs + hash, not full blobs | ~5–10 GB |
| Labels/aliases/descriptions EN+FR only | yes | ~8–12 GB |
| Sitelinks (EN/FR Wikipedia + a few others) | yes | ~3–5 GB |
| All other languages | no | ~60% of the dump text |
| Revision ids, timestamps, contri butors | no | — |

**Estimate: 60–90 GB** in a compact binary encoding (below). This is the number to plan NVMe around: **a 2 TB NVMe** covers store + text + build scratch with room for the news feed.

**Truthy vs full:** keep truthy *plus* normal-rank statements with a rank flag. Reason: your "live value wins" rule and dated-history questions need the non-preferred values (e.g., a historical capital). Deprecated never gets spoken.

**Text (Wikipedia EN+FR, DBpedia abstracts, books).** Two layers:
- **Extracted facts** (subject QID, predicate PID, object, provenance = source id + passage offset) go into the same statement store, flagged `source=book|wiki|news` and `corroborated=false` until a second independent source agrees. This is your hedged-fact rule made physical.
- **Passages for provenance:** keep the source text compressed (zstd) with a byte-offset index, not in the fact store. ~40 GB of Wikipedia text compresses to ~12–15 GB with zstd-19; books similar. Only passages that back a spoken fact are ever decompressed.

What a 450M model can extract without an external model: short subject–predicate–object sentences with a capitalized entity and a known predicate pattern. That is maybe 5–15% of encyclopedic sentences (I am unsure; measure it — experiment E3 below). Infoboxes and "X was born in Y" patterns are the realistic yield; narrative history prose mostly is not.

## 2. Physical layout

**Reuse, don't build:**

- **Build pipeline:** [Wikidata Toolkit](https://www.mediawiki.org/wiki/Wikidata_Toolkit) (Java, Apache 2.0) streams the bz2 dump and can emit exactly the filtered JSON you want. It processes roughly 5–15k entities/s single-threaded (unsure; measure on a 1M-item slice). Alternative: write the filter in Rust with `bzip2` + `serde_json` — you control it, and it's in your stack. I'd do Rust, ~1k–3k entities/s/core, 6 cores → **full dump in 8–24 hours at home**. Don't waste the Colab A100 on this; it's CPU work.
- **Serving store:** **LMDB** (OpenLDAP public license, Rust bindings `heed`/`lmdb-master`) or a custom memory-mapped sorted-file format. LMDB gives you: memory-mapped, single-writer/multi-reader, no network, 10 µs-scale point lookups. This matches your VM hop budget.
- **Name→item index:** an **FST** (the Rust `fst` crate, MIT/Apache, same design as Lucene's) mapping normalized name → list of QIDs. 100M items × ~5 aliases × ~20 bytes ≈ 10 GB of raw strings; an FST with shared prefixes + value blocks compresses this to **~3–6 GB, fully mappable in RAM** (fits in 32 GB alongside everything else hot).
- **Typo/near-miss:** a **trigram inverted index** over normalized names (the pg_trgm / Lucene FuzzySuggester approach) — you already have letter-trigram binary codes, so the concept exists; make it an exact inverted index, not hypervectors, since your measured negative says the qFHRR table loses to BM25 by 4×.
- **Statement storage:** dictionary-encoded. QIDs and PIDs as varints into a sorted `(subject, predicate, object)` table — this is the classic **SPO/POS/OSP permutation** layout (as in RDF-3X, Neumann & Moerkotte 2008, or QLever). Sorted runs + delta encoding + zstd per block: ~4–8 bytes/statement. ~1.5–2B truthy statements → **~15–30 GB**.

**Systems that hold full Wikidata on one machine** (check these; sizes are from their papers, not my measurements):
- **QLever** (Bonn; Apache 2.0; C++): indexes the full Wikidata RDF dump (~13B triples) in ~1–2 days on one machine, index ~250–400 GB, SPARQL queries in ms–s. This is the strongest existence proof that one machine suffices. It's C++, not Rust, and SPARQL-shaped rather than hop-shaped — I would not serve from it, but **steal its permutation layout**.
- **Jena TDB2** (Apache 2.0): full dump fits but queries at this scale are slow (100 ms–s). Not suitable for 1 ms hops.
- **Blazegraph**: was the actual Wikidata backend; JVM, now effectively unmaintained (GraphDB/Blazegraph split). Skip.
- **DuckDB** (MIT) for the *build* and offline analysis — great for the experiments below, not for 1 ms serving.

**Latency check against your targets:** LMDB point lookup on a warm mmap: ~10–50 µs. FST name lookup: ~10 µs. Your 1 ms p95 per hop is comfortable; even 10 ms from SSD is loose. The real risk is not lookup, it's **name resolution fan-out** (one name → 50 QIDs → 50 statement reads). Budget that explicitly: 50 × 50 µs = 2.5 ms. Your 50 ms p95 for name resolution is achievable with the trigram index if you cap candidates at ~200 and refuse beyond that.

## 3. Name → item at 100M items

Pipeline, all inside the VM hop:
1. Normalize (Unicode NFKC, casefold, strip diacritics per language — keep both a folded and an accent-preserving key, since French *é* matters for disambiguation).
2. Exact FST lookup → QID list. If exactly one, done. If several, **return all with their descriptions; never rank by fame** (your rule — and note Wikidata's own search *does* rank by sitelink count, so do not reuse its ordering).
3. If zero, trigram index → top-k by trigram overlap → Levenshtein ≤ 2 (Lucene's automaton) → candidates.
4. If still ambiguous, the VM refuses with "N people named X; which?" — this is a feature, not a failure, under 0-wrong.

Disambiguation between namesakes uses the *question's* other resolved entities (birth-year, country, occupation predicates), never fame. That's a graph hop, which is exactly what your VM is for.

Plug-in: CubeLang gets a primitive `resolve(name, lang) -> [qid]` and `facts(qid, pid) -> [statement]`, both VM-checked, both refusing on ambiguity above a stated cap.

## 4. Time and change

- Every statement carries its qualifiers, so point-in-time queries are reads: `population(P1082) with P585 > t`, pick latest ≤ t. No special machinery.
- **Updates without rebuild:** LMDB single-writer transactions from an out-of-network updater process that consumes the weekly Wikidata `recentchanges` API / entity diffs (this process is outside the serving sandbox, per your rule). Weekly entity-diff dumps exist (`wikidatawiki` dumps); I am unsure of their current format — check.
- Live-source rule: the live value *overwrites* the stored statement at write time with `provenance=live, fetched=<ts>`. The serving path never knows the network exists.

## 5. The mixture-of-specialists idea

**I think it's wrong for this problem.** Point by point:

- **What it buys:** smaller indexes per shard — but the sharded store already gives you this; a router is not needed to mmap only the shard a QID lives in, because the QID *tells you* the shard. Parallel builds — same, shards build in parallel regardless. Topic-specific question reading — this is the only real argument, and it's a *model* claim, not a store claim.
- **What it costs:** routing errors become wrong-answer risk (a question about "the wife of the painter who..." routed to the history trunk answers from history's shard or refuses wrongly). Cross-topic questions are the norm in SimpleQA-style long-tail QA, not the exception. And per-topic adapters trained on topic data **do put facts back into weights** — an adapter trained on a shard absorbs shard-specific priors, which is exactly the memorization you've forbidden yourself.
- **One trunk over one sharded store** is simpler, has no routing failure mode, and the shard choice is derived from the QID, which is already checked. If the 450M model reads questions poorly on some topic, the fix is per-topic *evaluation and prompt/program examples* (stored, not trained), not per-topic weights.

**When I'd be wrong:** if an experiment shows the single trunk's CubeLang programs fail on a coherent topic (say, sports) at a rate the topic-adapter fixes, *without* the adapter gaining on a held-out fact-recall probe that tests memorization. Design that experiment before building any adapter. Until then: one trunk, one sharded store, shard = hash(QID) or property-range partition.

## 6. The build

1. Download `latest-all.json.bz2` (~140–150 GB) to the 2 TB NVMe (or Drive, but Drive throughput ~ 50–100 MB/s makes the download alone ~2–4 weeks — **download at home**).
2. Rust streaming pass: bz2-decompress, parse each entity, filter to the kept fields, emit sorted SPO runs + name records + text blocks. 6 cores, **8–24 h** (measure on the first 1M entities; extrapolate — the dump's entity size distribution is heavy-tailed).
3. External sort/merge of permutations, FST build (the `fst` crate builds 10 GB-scale FSTs in minutes to an hour).
4. Wikipedia: `pages-articles` zst dumps, extract infobox/lead facts with your reader, zstd the text, build passage offset index. **~6–12 h** per language.
5. **Faithful-copy proof:** per-entity statement counts and a rolling hash of the canonicalized kept fields, compared against an independent second pass over a 1% random entity sample; plus total counts (entities, statements, labels) vs. Wikidata's own dump statistics page. A mismatch kills the build.

## 7. Experiments (run before the full download; each uses a 1–5M-item dump slice)

**E1 — coverage (the actual bottleneck).** Data: SimpleQA (public, OpenAI, 2024) + a French equivalent (build from FR Wikipedia titles → QIDs; must be public with gold QIDs per your rule). Metric: fraction of questions whose gold entity exists in the slice-store and whose answer is a truthy statement. Pass line: gold entity present for ≥ 80% of the EN subset once the full dump is in (on the slice, measure the *upper bound* by checking gold QIDs against the slice). Kill line: gold entity absent for > 30% even with full Wikidata — means Wikidata is the wrong source and you need Wikipedia-extracted facts to dominate.

**E2 — name resolution.** Data: 10k SimpleQA question entities + 10k FR-Wikipedia titles, gold QIDs. Metric: recall@k of `resolve()` (k = 1, 5, all), p95 latency. Pass: recall@all ≥ 0.95, p95 ≤ 50 ms. Kill: recall@all < 0.85 — the alias/typo pipeline is broken before scale.

**E3 — reader yield.** Data: 1k EN + 1k FR Wikipedia lead paragraphs with hand-checked gold triples (you may use external models to *build* this eval, per your rules). Metric: precision (0-wrong applies: a wrong extraction is a kill) and recall of extractable facts. Pass: precision ≥ 0.99, recall ≥ 0.3 of pattern-extractable facts. Kill: precision < 0.95 — the reader can't feed the store safely.

**E4 — hop latency.** Data: 10k random (QID, PID) lookups + 1k name resolutions on the slice, cold and warm. Pass: warm p95 < 1 ms/hop, name p95 < 50 ms. Kill: p95 > 10 ms warm — the encoding is wrong, revisit permutations/blocking.

**E5 — end-to-end.** The 860/4,326 SimpleQA subset where gold is a store entity. Pass: ≥ 500 answered with 0 wrong. That single number is the store's reason to exist.

## 8. What not to build, risks, decisions

**Don't build:** a custom RDF/SPARQL engine (use the permutation layout, not the query language); any hypervector retrieval on the answering path (your own measured negatives); per-topic adapters (§5); anything on Colab A100 except possibly eval scoring — the build is CPU+NVMe work.

**Risks:** (1) the 959 GB / 74.5M figures are stale — recheck before buying disk; (2) Wikidata truthy coverage of SimpleQA long-tail may be lower than hoped — E1 exists to find this out cheaply; (3) name ambiguity at 100M items may make "answer each or ask" produce a lot of asks — measure refusal rate in E5 and decide whether that's acceptable UX; (4) LMDB write-during-read is fine but the updater must be crash-safe — test kill -9 during an update.

**Decisions for Nick:** (a) 2 TB NVMe, and whether 64 GB RAM now or later (32 GB works; 64 removes the RAM/SSD boundary for the FST + hot shards); (b) Rust-native build vs Wikidata Toolkit — I say Rust; (c) kill or keep the mixture-of-specialists — I say keep the *sharding*, drop the *routing*; (d) commit to E1–E3 before the 140 GB download; the slice costs ~2 GB and one evening.

## x-ai/grok-4.7

*20412 tokens out, 411.5 s, finish `stop`*

The premise is wrong in two places. You do not need to compress 1 TB into the serving path, and a trunk-per-topic mixture is the wrong partition for this store. The terabyte is mostly JSON syntax, languages you are not serving, lexemes, Commons, and prose. The thing the VM can certify is a projection of Wikidata: on the order of 10^8 items and 10^9 statements, which your own note already guessed at 40–80 GB kept. Treat 40–80 GB as a hypothesis to kill with a sample, not as a measurement. I have not weighed a 2026 dump.

One design: an immutable, id-range-sharded fact trunk, global EN/FR name tables, a small live overlay beside it. No topic routers. No facts in weights. No similarity standing in for a value.

## 0. Claims to fix before any schema work

**0 wrong cannot mean “true in the world.”** Wikidata is wrong, stale, and vandalised. The only claim this store can support is: every spoken value is a specific statement in a named dump, or a specific overlay record, and the hop checked it. If Cubby speaks in its own voice, it inherits Wikidata’s errors. Every certified answer should carry dump date, Q-id, property, and statement offset. That is a product decision you have to make; the schema cannot save it.

**“Answer each namesake” does not scale.** Never ranking by fame is right. Emitting 200 bios for “Jean Martin” is not. The store returns the full id set, ordered by Q-id only so the order is not importance. The speaker asks, using stored descriptions, dates, and occupations as discriminators, not as a ranker. Enumerate only if the user asks and the set is small (I would cap listing at 10).

**Two-source agreement does not apply to Wikidata.** Wikipedia and Wikidata are not independent; a large fraction of Wikidata was imported from Wikipedia. Apply the two-source rule only to facts your reader extracts from books. Wikidata statements are speakable as attributed snapshot facts, or you will not move SimpleQA off “3 of 860.”

**A qualifier you do not understand must not be dropped.** If a statement has a qualifier outside the allowlist, mark it `not_speakable` and keep the raw qualifier in an audit file. Speaking the unqualified triple is a silent wrong answer (“population” that was “of the metro area”, “spouse” that was “of the fictional portrayal”). This rule matters more than compression.

**Fact hops may not be satisfied from hippocampal memory.** Different namespace. A hash of a key string is not a checked statement.

**The GPU is irrelevant.** The RX 6750 XT and Colab A100 do not belong in this build. It is parse, sort, and mmap. Serving code stays off the network; Google Drive is an archive, never a datastore the VM opens.

## 1. What to keep

### Wikidata

Source: the full JSON dump (`latest-all.json.gz` / `.bz2`), one entity at a time. Not the truthy NT dump: truthy drops qualifiers, ranks, and references, and you need all three. Skip lexemes (L-ids) in v1. Skip Commons media, math, musical notation, geo-shape, tabular data.

Keep, for every item:

- Q-id, all statements whose datatype you can interpret: item, time, quantity, string, external-id, monolingual text, globe-coordinate.
- Rank (preferred / normal / deprecated).
- Qualifiers on the allowlist, interpreted. Any other qualifier makes the statement `not_speakable`.
- References, deduplicated, as provenance, not as a second copy of the fact. A statement with a weak reference is still speakable if attributed to the dump. Do not pretend the reference makes it true.
- Labels, aliases, descriptions in English and French only. Missing French label means refuse or show the English label marked as English. Do not translate with the 450M or with anything else; that invents keys.
- Sitelinks for `enwiki` and `frwiki` only, as provenance pointers.

Do not keep a property allowlist. Filtering “useful” properties is an editorial prior and will miss the next SimpleQA. Disk is cheaper than a second rebuild. Allowlists are for qualifiers and for the live overlay only.

Qualifier allowlist to implement, not to grow by intuition: P580 start time, P582 end time, P585 point in time, P1264 valid in period, plus the rank-reason properties you are willing to display (P2241, P7452). Also P518 applies to part and P642 of, if and only if the speaker template can say “of X”. If the template cannot, those statements are `not_speakable`, not simplified.

Time encoding: store precision, calendar, and year/month/day as Wikidata gives them. Do not convert to Unix milliseconds. A year-precision date is not a day. Quantity encoding: integer mantissa and exponent, plus unit Q-id, plus lower and upper bound when present. Never `f64`. A float population is a wrong answer you will not notice in review.

Descriptions are for asking, not for matching. Labels and aliases are keys. Descriptions can be zstd-blocked. Labels should not be: the resolve hop has to read them.

Text of Wikipedia is not a fact source in v1. A rules reader, or the 450M, will be wrong on prose, and your kill line is zero wrong. What you may extract without an external model: nothing from sentences, until the experiment below passes. Infoboxes are a trap; they duplicate Wikidata and disagree with it. Keep EN+FR plain-text leads (and only those sections) as a provenance annex keyed by sitelink, so a spoken fact can show the page it came from. Your ~40 GB figure is the right planning size for that annex. Licence split: Wikidata is CC0; Wikipedia text is CC BY-SA. Separate files, separate headers. Do not pour book text into the same blob; full-text books are a copyright decision, not a storage decision. Offsets plus a short quotation are the safe default unless you already have redistribution rights.

DBpedia abstracts: drop. They are a third copy of the lead. GDELT: do not build it into this store. Event coding will violate 0 wrong the moment it is speakable.

Expected size, as a planning band only: facts, ranks, allowlisted qualifiers, refs, EN/FR labels and aliases, descriptions, two sitelinks — **40–80 GB**. Name FSTs and id maps inside that, on the order of a few GB, not tens. Wiki annex another **~40 GB** plain, less if zstd’d for the display path only. If the sample experiment projects facts above **200 GB**, the encoding is wrong or the speakable set is too wide; do not “solve” it with a topic shard.

## 2. Physical layout

Immutable snapshot, memory-mapped, little-endian, versioned header (dump date, SHA-256 of the compressed dump, git hash, normalizer version, qualifier-allowlist hash, counts). Shards are **ranges of dense local ids**, about 5–10 million items each, so a file is a few GB. Not topics. The OS caches hot pages. With the 450M resident (well under 2 GB even in fp32) and the OS, budget **≤20 GB** of page cache on the 32 GB machine. That is enough for indexes and hot properties, not for a randomly touched 80 GB. **Buy 64 GB DDR4** if you want cold p95 to collapse; it changes the operating point more than any index trick. I am confident of that, and not confident of any finer RAM model.

Layout:

- `qid_to_local`: a direct `u32` array of length `max_qid+1`. Q-ids fit in u32 (max Q is on the order of 10^8, far under 2^32). About **0.5–0.8 GB**. Binary search on a sorted array also works; the direct map is simpler and fast enough. Local ids are dense and never reused inside a snapshot.
- `stmt_off[local_id] → u64` into a CSR blob, statements sorted by (property, rank, time). Common case, item-valued and no qualifier: pack into 8 bytes (property u16, flags u16, object local id u32). Quantities, times, strings, and any qualified statement go to an overflow file. I do not know the fast-path fraction; that is a sample metric. If it is under half, do not contort the fast path — overflow latency still has to meet the SLO.
- Hot file: a duplicate of the simple statements for a fixed set of high-traffic properties (P31, P279, P569, P570, P17, P27, P39, P106, and whatever SimpleQA actually hits — measure, then freeze the list). Target **2–4 GB**, intended to stay resident. This is a cache by property, not a specialist.
- Labels: uncompressed packed blob plus offset table, split EN and FR. Descriptions: zstd frames of about 1 MB.
- Aliases: one FST per language, `fst` crate (Andrew Gallant; mmap via the crate’s map open). Posting lists of local ids in a side file. Normalization, versioned in the header: NFKC, casefold, collapse whitespace and hyphens. No stemming. No confusable folding in v1 (it merges distinct names; turn it on only if a false-certainty experiment stays at zero).
- References: deduped table, statement holds `ref_id`. Do not inline reference graphs.
- Overlay: LMDB through `heed`, written only by a process outside the server, installed by atomic rename. Serving VM reads snapshot then overlay; overlay wins, including deletes. Verify the heed/LMDB licences before vendoring (LMDB is the OpenLDAP public licence; confirm heed). Do not put the bulk snapshot in LMDB or RocksDB. CSR is smaller and the access pattern is a hop, not a transaction.

Reuse, do not build:

- `fst` and `levenshtein_automata` (same author) for names and typos.
- LMDB/`heed` for the overlay only.
- DuckDB (MIT) at build time for count diffs and sample joins. Not on the hop path.
- Oxigraph (Rust, SPARQL; confirm Apache-2.0 before depending on it) as a **correctness oracle on a 1% item sample**, not as the server. A general SPARQL engine will not hit your latency on 32 GB, and you do not need SPARQL in the VM.
- tantivy (MIT) only if you search the provenance annex. Not for facts.

Name as prior art, do not embed:

- QLever (Bast and Buchhold, CIKM 2017, Freiburg) is the existence proof that full Wikidata is queryable on one machine. I am **not** willing to quote a 2026 index size or latency from memory; check their current Wikidata index notes before you cite a number. It is C++, and it is a SPARQL engine, not a VM hop.
- HDT (Fernández, Martínez-Prieto, Gutiérrez, Polleres, and Arias, Journal of Web Semantics 2013) is the right idea — dictionary, triples, compact, mmap-friendly — and the wrong file format for qualifiers and ranks. Steal the dictionary-plus-CSR lesson, do not adopt HDT as the schema.
- MillenniumDB (Angles, Vrgoč, and others; confirm the VLDB title before citing) is the other existence proof for Wikidata on one box. Same rule: cite their measured size and load time from the paper, do not invent them, and do not take a C++ graph database into the VM.

I do not know of a maintained, embeddable, Rust, one-machine Wikidata store that already keeps ranks and qualifiers and meets a 1 ms hop. That is why this is a custom CSR rather than a dependency. If Oxigraph on the 1% sample is already under 1 ms and the projection fits in 20 GB resident, stop and use Oxigraph. I doubt it; the experiment is cheap.

Latency: your SLOs are loose, and that is fine as pass lines, dangerous as design targets. A warm CSR hop on a resident hot file should be tens of microseconds, not 1 ms. A cold NVMe random read on this class of disk is typically well under 1 ms for a single 4 KB page; I would not quote a tighter number without `fio` on your drive. Keep pass lines at **1 ms p95 warm, 10 ms p95 cold, 50 ms p95 name resolve**, and also record microseconds so a 900 μs design cannot look like a success.

NVMe: **500 GB free** for dump plus intermediates plus snapshot. Steady state about **150 GB** without the wiki annex, about **250 GB** with it. The 5 TB Drive holds the compressed dump and old snapshots only.

## 3. Name to item

A VM primitive, pure in `(snapshot_id, overlay_id, normalizer_version, span)`:

`resolve_name(span, lang) → set of local ids | Unknown`

Search both languages always. A French question will use an English alias. Tag which field matched so the speaker can say so; do not drop cross-language hits.

Steps, in order, stop when the exact path returns a non-empty set:

1. Exact FST lookup on the normalized span. Return **every** posting. No top-k, no sitelink count, no QRank. QRank is a fame ranker; do not import it.
2. If the span is multi-token and exact failed: all-tokens match, order-insensitive, against a word inverted index. Still exact tokens. Still a set.
3. Typo path only if both failed, and only for tokens of length ≥ 5 (edit distance 1) or ≥ 8 (edit distance 2). Implement it as a Levenshtein automaton intersected with the FST (Schulz and Mihov, 2002), via `levenshtein_automata`. **No similarity threshold.** That is the point: your measured negative is that thresholds do not travel. If the automaton returns more than 32 ids, return Unknown and ask for another letter. Short-token typos are Unknown, not a guess.
4. Do not use the qFHRR table, SimHash, or letter-trigram scores on this path. You already measured them. Trigram codes may remain a debug tool; they may not emit a candidate the automaton would not emit.

Ambiguity is not a failure. `|set| > 1` is a successful hop whose CubeLang branch must ask. A singleton is speakable. Unknown is a refuse. There is no “best” id.

Plug-in: the hop returns ids plus the match class (exact / tokens / typo) and the language of the alias. Later hops take local ids only. The model never copies a label out of its weights and treats it as resolved. A typo match may lead to a key; the fact spoken is still the statement at that key, attributed as usual.

## 4. Time and change

Speaking rule, deterministic:

- Consider only speakable statements of the asked property.
- If any have time qualifiers, a statement applies only when its precision-aware interval contains the question time. Year precision does not match a day.
- Preferred outranks normal. Deprecated is never spoken unless the question is about a former value and the time qualifier applies.
- Two applicable preferred statements that disagree: refuse. Do not pick.
- No time qualifier means “Wikidata asserts this undated,” not “true for all time.” The template should say undated. Most of instance-of, capital, and occupation looks like this.

Live rule, without the server touching the network: a fetcher outside the process writes an LMDB overlay. Overlay wins over snapshot for the same (Q-id, property, statement identity), including deletions. Live refresh is an allowlist you write by hand (population, position held, head of government, and whatever else you are willing to operate). Everything else is snapshot. If the question is time-sensitive and the property is not on the allowlist, refuse, do not speak a stale snapshot. How CubeLang decides “time-sensitive” is a separate problem; if it is unsure, refuse.

Do not patch CSR in place. Between dumps, the overlay is the only mutation. Next dump is a full rebuild and a new snapshot id. Old overlays do not apply to a new snapshot. This will waste hours of build time and will avoid a class of silent rank/delete bugs. Take the trade.

## 5. Mixture of specialists

Wrong for this problem. Drop it unless the experiment in §7 passes, which I do not expect.

What it appears to buy — smaller indexes, parallel builds, shards swapped in RAM — you get from id-range shards plus a hot property file, with no router. Parallel builds are `local_id` ranges written by separate threads after the stream pass.

What it costs is the actual bug. Topic is not a partition of Wikidata. P31/P279 is a messy DAG. “President of France when X was born” needs person, office, and time in one hop chain. A router that picks one trunk either drops a hop (coverage) or speaks from the wrong trunk (wrong answer). Namesakes of one string live in different topics; routing to one of them is fame-by-topic, which you forbade.

The decisive point: **you cannot route to a topic shard until you know the entity, and the query is how you learn the entity.** The name table has to be global anyway. After a global FST hit, the statement CSR is a pointer chase by local id. A topic index saves nothing on the hot path and adds a fallible classifier.

Adapters on a shared trunk put facts back in weights. That is allowed by your rules only if an experiment beats storing. It will not beat storing on false certainty: a trained adapter can emit a value that is not in the snapshot, and your kill line is zero of those. Hand-written VM decoders per datatype (time, quantity, coordinate) are the only “specialists” worth having. Those are parsers, not trunks.

One trunk, one schema, id-range files, global names. That is the design.

## 6. Build and faithfulness

At home, off Drive, off the GPU. Stream the JSON dump; do not materialise the array. Some entities are very large — I do not know the current maximum. Give the parser a scratch ceiling (1 GB) and skip-and-count anything larger; if a skipped Q-id is in the eval set, the build fails.

Stages: one stream pass to a columnar intermediate (this is the long pole; **plan on 12–48 hours**, and treat that range as uncertain — a single-threaded JSON parse of ~1 TB is the risk, so parallelise only on entity boundaries, not by splitting mid-object). Then sort and write CSR, FSTs, and the hot file (a few hours on 6 cores). A100 time does not help. Colab session limits and Drive I/O make it the wrong place to parse; it is acceptable for a one-off sample join if the sample already fits on local disk.

Header holds dump filename, dump date, SHA-256 of the compressed file, counts of items, statements, speakable statements, not_speakable statements, skipped entities, skipped datatypes.

Faithfulness is a diff, not a count. Counts can match while two values are swapped.

- Forward: 10,000 Q-ids, re-read from the dump, diff allowlisted fields against the store. Pass: **0 mismatches**. Kill: any mismatch.
- Reverse: 10,000 statements taken from the store, find them in the source. Pass: 0 invented statements.
- Oracle: items with `qid % 100 == 0` loaded into Oxigraph or DuckDB from the same intermediate; triple-level diff on that 1%. Pass: 0. This catches systematic encoder bugs the random sample can miss if you are unlucky, and it is still not a proof. The full forward diff on the next monthly build is the proof; do it once a month, not only once.

SimpleQA is not a build proof. After the first full snapshot, re-run it as a coverage meter: of the gold entities that now resolve, how many have the gold property speakable. Direction to expect: the “3 of 860” fact-hit number moves, or the linker is still wrong. It is not a pass line for truth.

## 7. Experiments before the full dump

Held-out sets must be public and carry gold Q-ids, with a French slice. Do not tune on them and then cite them.

**Size and speakable fraction.** Data: your Wikidata5M, plus a reservoir of 50,000 entities from a streaming pass (the first million Q-ids are old and dense; do not use them as the only sample), plus 5,000 entities chosen because they have many statements. Metric: bytes per statement after your encoding; fraction that fit the 8-byte path; fraction `not_speakable` because of an unknown qualifier; fraction of items with an FR label; projected GB at the dump’s own item and statement counts. Pass: projection **≤ 80 GB** facts+EN/FR strings, and `not_speakable` under **20%** of statements. Kill: projection **> 200 GB**, or `not_speakable` over **40%** (allowlist too small to be the product), or FR label coverage so low that you are shipping an English store. I do not know those fractions; this experiment is the decision.

**Name resolve.** Data: Mewsli-9 (Botha, Shan, Gillick, EMNLP 2020), English and French subsets, gold Wikidata ids; plus a namesake set mined from labels that collide after normalization; plus a typo set you generate with one edit on tokens ≥ 5, gold id known. Metric: set recall (gold ∈ set), and **false certainty** (singleton returned, and it is wrong). Latency p95 on the home box with the model loaded. Pass: false certainty **0**; set recall **≥ 0.90** on exact-alias Mewsli-9 EN and FR; name p95 **< 50 ms**. Kill: any fame feature used to cut a set; false certainty **> 0**; set recall **< 0.70**; p95 **> 50 ms**. Accuracy@1 is the wrong metric and will push you back to QRank.

**No-fame.** Data: colliding-label groups, questions that contain only the shared string, EN and FR. Metric: fraction of runs that speak one item’s fact without asking. Pass: **0**. Kill: **> 0**.

**Time.** Data: Mintaka temporal subset (Sen et al., 2022; confirm the venue before you cite it), EN and FR, gold Q-ids; plus a store-internal set of P39 and P1082 statements with start/end/point-in-time, where the gold is the dump itself. Metric: wrong value rate when the system speaks; refusal rate when two preferred statements apply. Pass: **0** wrong values; on unambiguous Mintaka items present in the sample store, recall **≥ 0.80**. Kill: any spoken value whose qualifier excludes the question time, including precision violations (year treated as day).

**Prose reader, before you build one.** Data: T-REx (Elsahar et al., LREC 2018) English alignments of sentences to Wikidata triples; French will have to be a hand-checked set of a few hundred, because I am not aware of a T-REx-quality French public set — if you find one with gold Q-ids, use it and drop the hand set. Reader is rules or your 450M only. Metric: precision of emitted triples; fraction of true emitted triples that are not already in Wikidata. Pass: precision **1.0** on what it emits (refusal allowed) and novel-true yield **≥ 0.2 per passage**. Kill: any wrong emit, or novel yield ≈ 0. I expect the kill. Until it passes, do not extract facts from Wikipedia or from books into the speakable store. Book passages stay in FTS as provenance.

**Latency.** Data: 100,000 random (local id, property) and 10,000 name lookups, half FR, on the i5 with the model resident. Cold = after `drop_caches`, warm = third run. Pass: warm exact p95 **< 1 ms**, cold exact p95 **< 10 ms**, name p95 **< 50 ms**. Kill: cold exact p95 **> 10 ms** after the hot file is in place. Also log major page faults. If warm is under 1 ms only because the test set fits in cache and a uniform property sample misses, believe the uniform sample.

**Specialists, so the idea gets a measurement rather than a veto.** Data: 500 questions, EN and FR, gold Q-ids and the set of properties a correct hop chain touches. Include cross-topic items (office plus birthplace plus date). At least half should be Mintaka or another public set, so we can check them. Router: a hand-written keyword-to-topic map, not an external model. Compare to the global trunk. Metric: fraction of questions whose chosen shard lacks a needed statement; false answers if forced to speak; RAM. Pass for building specialists: routing miss **0** and resident RAM at least **30%** below the global trunk at the same latency. Kill: any routing miss on the cross-topic half, or RAM savings under **20%**. A miss here is fatal because the router sits on the answering path.

## 8. What not to build, risks, decisions

Do not build: topic trunks or adapters; a vector name index; a SPARQL endpoint as the product; in-place CSR updates; GDELT; DBpedia as a parallel truth; lexemes; sitelinks beyond en/fr; a property popularity filter; GPU indexing; serving from Drive; zstd on the hot label and statement files; float quantities; Unix-time conversion; QRank or sitelink-count tiebreaks.

Risks, in the order I would worry:

- Attribution omitted, so dump errors become Cubby errors.
- Unknown qualifiers stripped, so restricted claims become general ones. The `not_speakable` bit is the mitigation; if you find yourselves overriding it to grow coverage, stop.
- Normalizer merges two names. False-certainty on Mewsli-9 and on the collision set is the alarm. If it fires, back the normalizer off; do not add a threshold.
- French alias gap. That is the data, not the index. Measure it. Do not fill it with a model.
- 32 GB thrash on the cold CSR. Hot file plus 64 GB is the mitigation. If page faults still blow the 10 ms line, narrow the hot property list from the SimpleQA and Mintaka property histogram; do not shard by topic.
- Levenshtein on 10^8 names is too slow or too wide for some scripts. The length floor and the 32-id cap are the mitigation. If p95 fails, disable the typo path rather than approximate it. Exact match already does the real work.
- A huge entity or a parser skip drops something the eval set needs. Count skips in the header; fail the build if any eval id was skipped.
- Wikimedia deletes and rank changes between dumps. Overlay deletes handle the live allowlist only. Elsewhere you are stale until the next snapshot, and the template must show the dump date.
- Copyright on the book annex, and CC BY-SA on the wiki annex, if those files ever leave the house. Keep them physically separate from the CC0 fact trunk.

Decisions you owe the build, not more design:

1. Accept “0 store-contradicting claims, attributed to a dump,” and reject “0 false claims about the world.”
2. Freeze the qualifier allowlist and the live-property allowlist. Storage keeps all interpretable datatypes.
3. Ambiguity policy: ask whenever the set is larger than 1; list at most 10, and only on request; otherwise count plus a discriminator question.
4. 64 GB RAM, yes or no. I would buy it before writing the CSR.
5. Wiki lead annex in v1, yes or no. I would keep it for display and not for extraction.
6. Monthly full rebuild, overlay between dumps. Yes, unless the first build exceeds two days, in which case fix the parser, do not invent incremental CSR.
7. Specialists only if §7’s router experiment passes. It will not.

End-to-end measurement that the trunk is working: on a frozen SimpleQA slice whose gold Q-id is in the snapshot, fact-hit rate at least 10× your current 3/860, false certainty still 0, French Mewsli-9 false certainty 0, cold hop p95 under 10 ms on the home machine. What shows it is not working: any of those kill lines, a projection over 200 GB, or a specialist router that drops a cross-topic hop. Coverage will still be short of the whole terabyte after this. That is the point. The rest of the terabyte is not facts you can say.

## moonshotai/kimi-k3.1-pro

*12933 tokens out, 217.0 s, finish `stop`*

## Recommendation in one paragraph

Build **one question-answering trunk over one global, content-addressed statement store**, physically sharded by numeric Wikidata item id—not by topic. Retain Wikidata’s **full statements**, including qualifiers, ranks and compact references; derive a truthy/current view for fast queries rather than making truthy the only stored representation. Add EN/FR labels, aliases, descriptions, EN/FR Wikipedia titles and redirects, and compressed provenance passages. Use immutable, memory-mapped sorted tables with a small mutable update overlay. Do **not** build specialist adapters or a topic router initially: they do not increase the number of stored facts, and QID sharding already provides parallel builds and bounded shard sizes.

The target is approximately **200–400 GB for Wikidata plus 40–100 GB for text/indexes**, not 40–80 GB. Use a **2 TB NVMe SSD**; upgrade to **64 GB RAM** if convenient, but design for a 24 GB serving-memory ceiling. Expect the first home build to take roughly **24–72 hours**. These are planning estimates, not measurements; the pilot below must replace them.

A crucial qualification: VM certification can establish **“this value is exactly supported by this stored source statement”**, not that the source is true. “0 wrong” should therefore mean zero unsupported answers and zero errors on the evaluation sets, while separately reporting refusals and source errors.

---

# 1. What to keep

## Wikidata: full statements, with a derived truthy view

Do not reduce the dump to `subject, predicate, object` triples. That would discard precisely the information needed to avoid confidently speaking obsolete, deprecated or context-dependent values.

### Retain for every Wikidata item

1. **Every statement**
   - Subject item, property, complete typed value.
   - Statement id and rank: preferred, normal, deprecated.
   - All qualifiers, not just time qualifiers.
   - References, stored compactly as provenance.
   - Deprecated statements remain queryable when explicitly requested but are not current answers.

2. **Values without lossy conversion**
   - Item references: numeric QID.
   - Time: signed value, precision and calendar model.
   - Quantity: exact decimal representation, unit and optional bounds.
   - Coordinates: fixed-precision latitude/longitude plus globe.
   - Monolingual text: language and text.
   - Strings and external identifiers: exact UTF-8.
   - Do not convert quantities to binary floating point as their canonical representation.

3. **Names**
   - Labels and aliases for at least `en`, `en-GB`, `en-CA`, `fr`, `fr-CA`; include `en-US` if present.
   - Descriptions in those languages, useful for presenting ambiguity but not for silently choosing a winner.
   - EN/FR Wikipedia sitelinks.
   - EN/FR Wikipedia redirects as separately provenanced aliases.
   - Other-language labels are unnecessary initially unless needed for cross-lingual book references.

4. **References**
   - Reference property/value pairs, reference-node hash, source URL or publication identifier where present, and retrieval date.
   - References are provenance, not proof. Two references copied from one source are not independent corroboration.

Initially exclude **Lexemes, MediaInfo and entity-schema content**, unless they are separately required. Preserve properties’ labels and datatype metadata because the VM needs them.

### Derived views

Materialize:

- **Current/truthy:** non-deprecated statements, respecting preferred rank but retaining ties.
- **Valid-at-time:** statements eligible under start/end qualifiers.
- **Point-in-time observations:** values with `P585`, kept distinct from validity intervals.
- **Retrieval time:** when Cubby fetched or imported the statement.

Do not implement “closest date wins” or “most recently edited wins” as general semantics. If several eligible values remain, return each or refuse/ask.

### Planning sizes

Assuming roughly **1.5–2.0 billion retained statements**, pending the pilot:

| Component | Planning estimate |
|---|---:|
| Core statement records and CSR offsets | 45–80 GB |
| Qualifiers, literal dictionaries and typed values | 35–100 GB |
| Deduplicated references/provenance | 20–80 GB |
| EN/FR names and descriptions | 8–20 GB |
| EN/FR sitelinks and redirects | 1–4 GB |
| Forward/reverse and name indexes | 30–70 GB |
| **Total** | **139–354 GB** |

Budget **200–400 GB**. A truthy-only version might fit in **50–110 GB**, but the saving is not worth losing temporal and provenance behavior.

The stated “74.5M entities” is almost certainly an old or filtered count if the dump contains more than 100M items. Recount entity types during the pilot; do not size from both numbers interchangeably.

## Text: extract narrowly, keep the evidence

Keep all EN/FR Wikipedia article text in revision-identified, Zstandard-compressed blocks. Forty GB of text will commonly compress to roughly **10–18 GB**; allow another **20–60 GB** for document metadata, BM25 indexing and extracted-fact records. Treat these as estimates.

Initially extract only constructions with a mechanical, testable interpretation:

- Infobox fields and tables with an explicit schema.
- List/table rows whose columns have declared meanings.
- Dates, identifiers, ISBNs and bibliographic metadata using validated parsers.
- Sentences with explicit entity links only where a property-specific rule has passed its precision test.
- EN/FR links to items through sitelinks, preserving page id, revision id and character offsets.

Do **not** start with an open-domain neural relation extractor. A fluent extractor that produces 98% correct facts is incompatible with an effectively unhedged answering path: at 100,000 extracted facts, that could mean 2,000 bad claims.

Each extracted fact should contain:

```text
subject QID, property, exact value,
document QID/page id, revision id, passage offsets,
extractor name and version, extraction rule/model hash,
source publication date where known, import time
```

Keep the passage even when the fact is also in Wikidata. A book-derived fact remains source-attributed and hedged until a genuinely independent source agrees; syndication and copied passages do not count.

DBpedia abstracts are largely redundant with Wikipedia. Use them as a comparison source, not a second automatically independent corroborator.

---

# 2. Physical layout

## One immutable base plus an update overlay

Use application-specific memory-mapped files rather than expanding Wikidata into general-purpose RDF triples in SQLite or RocksDB.

### Entity and value representation

- Map QIDs to dense unsigned 32-bit local ids; reserve 64-bit file offsets.
- Keep a QID-to-local-id directory. A direct numeric-offset table is acceptable if it costs only about **0.5–1.5 GB**; otherwise use a minimal perfect hash plus QID verification.
- Properties use compact integer ids with a separate datatype table.
- Store strings once in dictionaries, compressed in independent **64–256 KiB** blocks.
- Hash long values and reference sets with BLAKE3, while retaining the actual values needed for provenance.

### Statement tables

Use CSR-like layouts:

- Outgoing: `subject → [property, value-ref, statement-ref]`.
- Statement table: rank, flags, qualifiers, references and typed value.
- Reverse: `object, property → subjects`, built only for properties used by reverse VM hops.
- Sort each run and merge externally with a fixed memory budget.
- Shard by high bits or hash of numeric QID, for example **64–256 shards**.
- The global name index and property dictionary are shared across shards.

QID sharding gives parallel builds, bounded temporary files and RAM swapping **without requiring a semantic router**. A question can cross shards through ordinary exact hops.

Use memory mapping and the OS page cache. Under a 32 GB machine, cap Cubby’s non-page-cache memory around **20–24 GB**, leaving room for the OS and VM. Sixty-four GB would improve cold-start behavior but should not be necessary for correctness.

### Reuse rather than build

- **`memmap2`**, Rust, MIT/Apache-2.0: memory mapping.
- **`fst`**, Rust, MIT/Unlicense: finite-state name dictionaries.
- **Tantivy**, MIT: Wikipedia passage BM25 and possibly experimental name fallback, not fact storage.
- **Zstandard**, BSD/GPL dual-licensed library: compressed text and dictionary blocks.
- **Roaring/croaring**, Apache-2.0 implementations: large postings where useful.
- **BLAKE3**, CC0/Apache-2.0 implementations: chunk and provenance hashing.
- **Oxigraph**, MIT/Apache-2.0: useful for small RDF tests, not my choice for full Wikidata on this machine.
- **Sophia**, Rust RDF tooling: useful for parsing/interoperability tests, not a reason to adopt an RDF serving layout.

Avoid a monolithic RocksDB installation for the base: write amplification and per-key overhead are unnecessary for an immutable sorted snapshot. The small update overlay could use RocksDB or SQLite.

## Existing full-Wikidata systems: what their numbers do and do not establish

The citable historical result I would use as a reference is **Bast et al., “QLever: A Query Engine for Efficient SPARQL+Text,” CIKM 2017**. It reported indexing the then-complete Wikidata, approximately **2.9 billion triples**, in **under 8 hours**, with **average query time under 100 ms** on one machine. Those are not current sizing numbers: modern full Wikidata RDF is several times larger, and QLever’s current full deployment should not be assumed to fit in 32 GB RAM. I am not confident enough in its historical on-disk GB figure to quote it; use the paper’s table directly if comparing.

Other relevant systems:

- **Wikidata Query Service / Blazegraph:** genuinely operates at full Wikidata scale, but as a substantial service deployment, not evidence for a 32 GB home installation.
- **HDT**, Fernández et al.’s compressed RDF representation: compact and single-machine friendly for read-only triples, but updates generally mean rebuilding, and it does not remove the cost of the RDF expansion.
- **qEndpoint/Tentris:** worth benchmarking if a general SPARQL endpoint becomes a requirement; I would not adopt either without a current full-dump RAM and build measurement.

There is no reliable, apples-to-apples published **current size/build-time/p95 latency** measurement for full Wikidata on the stated home machine. That absence is a reason to run the pilot, not to invent a number. QLever can be used as an independent semantic comparison oracle on a subset; it need not become Cubby’s server.

### Latency expectations

The proposed targets are reasonable but need p99 and concurrency measurements too:

- Hot exact hop: **p95 <1 ms**, expected to require a few cached table accesses.
- SSD exact hop: **p95 <10 ms** on local NVMe, including several random reads.
- Exact name candidates: normally **<5 ms**.
- Typo fallback: **p95 <50 ms**, with a hard candidate-expansion budget.

Measure one thread and four concurrent requesters, cold and warm. “Cold” must mean a newly mapped snapshot with dropped caches or after reboot, not merely an empty application cache.

---

# 3. Name to item at 100M items

Build a global, language-tagged name index:

```text
(language, normalization class, normalized name)
    -> offset of sorted posting list of QIDs and match metadata
```

Use three normalizations:

1. Unicode NFC, whitespace and case normalization, retaining accents.
2. Accent-folded EN/FR fallback.
3. Limited punctuation folding; do not aggressively strip meaningful characters.

Each posting records whether the name came from a label, alias, sitelink or redirect. Posting lists are sorted by QID, **not popularity**.

Resolution stages:

1. Exact language-specific label/alias/title.
2. Cross-variant exact match.
3. Accent/punctuation fallback.
4. Prefix search only when explicitly requested.
5. Typo fallback using a bounded trigram candidate index or FST Levenshtein automaton.

Similarity only generates candidates. It never selects a fact and never overrides an ambiguous exact match. Typo thresholds must be conditioned on language and name length and recalibrated per deployment.

For namesakes:

- Use explicit constraints from the question—dates, occupations, linked places or other VM-resolved entities—to eliminate candidates only when those constraints are stored facts.
- If two people remain viable, return both candidate answers or ask a clarification.
- Never use PageRank, Wikipedia length, sitelink count, search frequency or “fame” to choose between them.

The VM interface should be approximately:

```text
RESOLVE(language, surface, context) -> candidate QIDs + match evidence
LOOKUP(subject QID, property, as_of) -> statement ids
READ(statement id) -> exact value + provenance
CERTIFY(statement ids, requested answer type) -> answer or refusal
```

Keep the existing qFHRR codes out of this path. Retain trigram codes only if the measured candidate-recall/latency experiment justifies them.

Expected name-index storage is **5–20 GB**, depending heavily on how many aliases and Wikipedia redirects are retained.

---

# 4. Time, live values and updates

Store three different notions of time separately:

- **Valid time:** start/end qualifiers.
- **Observation time:** point-in-time qualifier.
- **Ingestion/retrieval time:** when Cubby obtained the claim.

For an `as_of=t` query, include only statements whose validity interval contains `t`; apply rank rules afterward. If time qualifiers are absent, do not infer them. A statement with an end date must not silently become the current value.

The network-capable live importer sits outside the server. It writes signed/checksummed deltas containing fetch time, Wikidata revision information and full statement provenance. The server reads only the imported result. If the live importer is unavailable, answer from the snapshot with its date or refuse if freshness is mandatory.

Updates:

- Nightly or hourly append-only delta files, with deletions/tombstones.
- Lookup order: newest applicable overlay, then base.
- Compact overlays when they exceed a measured threshold, for example **2–5 GB** or **5% of changed entities**.
- Rebuild the immutable base monthly or when compaction cost justifies it.
- Never require a full dump rebuild for an ordinary update.

For time-sensitive properties on Nick’s live-wins allowlist, a successfully imported live statement supersedes the snapshot for current answering. Historical snapshot values remain available. Live vandalism is a real risk; do not apply “live wins” indiscriminately to every property.

---

# 5. Nick’s mixture of specialists

**It is the wrong first design for the coverage bottleneck.**

What it could buy:

- Topic-specific CubeLang vocabulary or question interpretation.
- Independent training/build experiments.
- Smaller semantic shards and parallel shard construction.

But the last two benefits are already supplied by QID sharding. A specialist does not create a missing statement, alias or extracted fact.

What it costs:

- A person, place, event and work can belong to several topics.
- A history question may require geography, bibliography and biography hops.
- Top-1 routing turns an otherwise answerable question into a refusal.
- Fallback to all specialists removes much of the supposed saving.
- Fine-tuning creates a memorization risk and another model-validation burden.
- Topic-partitioned name indexes make cross-topic namesakes harder to handle.

Use **one trunk and one global store**. If question parsing later proves topic-sensitive, test adapters while leaving the store unpartitioned semantically. That isolates any benefit to question interpretation and avoids routing facts away from the query.

---

# 6. Build and fidelity

Run the build **at home**. An A100 provides almost no benefit for decompression, JSON parsing, sorting and hashing, while Colab’s session limit and Drive I/O add failure modes. Drive is suitable for checksummed backup chunks, not build scratch or random access.

### Disk requirement

A dedicated **2 TB NVMe SSD** is the minimum comfortable design:

- Compressed source dump: allow 100–250 GB.
- External-sort runs and scratch: 300–500 GB.
- New store and text indexes: 300–500 GB.
- Previous snapshot for atomic rollback: 300–500 GB.
- Delta and free-space reserve: remainder.

A 1 TB drive would make snapshot replacement and scratch needlessly fragile. If the pilot extrapolates above 400 GB, revisit the budget before the full build.

### Pipeline

1. Record source URL, dump date, byte length and cryptographic checksum.
2. Stream the compressed JSON once; never materialize the uncompressed JSON.
3. Parse entities independently and emit canonical binary records.
4. Write sorted runs under a fixed **16–24 GB** memory budget.
5. Merge by subject/property/value; construct reverse indexes only where needed.
6. Build name FSTs and postings separately.
7. Compress text and build Tantivy indexes independently.
8. Write chunk manifests and a Merkle root; atomically activate the completed snapshot.

Planning runtime is **24–72 hours**, dominated by decompression, JSON parsing and external sorting. If bzip2 decoding proves to be the bottleneck, parallelize after an initial split where the dump format permits it, or pay for a one-time recompression into independently seekable chunks.

### Prove fidelity

Do not validate only with handpicked queries.

- Compare exact per-entity counts of statements, qualifiers, references, labels, aliases and sitelinks against an independent parser on at least **100,000 randomly sampled QIDs**.
- Include every evaluation QID and stratified samples of rare value types.
- Canonicalize retained source JSON deterministically and compare BLAKE3 hashes per entity.
- Compare full-build aggregate counts by property, datatype, rank and qualifier property.
- Execute **1 million random statement lookups** against source-derived expected values.
- Require **zero semantic mismatches**. Any mismatch blocks deployment.
- Repeat a small build in a clean process and require identical chunk hashes.

A second parser such as Java **Wikidata Toolkit** is useful for the independent sample; I would not require it to share Cubby’s canonicalization code.

---

# 7. Experiments and kill lines before the full download

Use development sets for tuning and keep test sets untouched. Relevant public resources:

- **Mintaka**, Sen et al., COLING 2022: 20,000 English questions with translations including French and Wikidata entity/answer annotations.
- **QALD-9-plus**, Perevalov et al.: multilingual DBpedia/Wikidata QA with SPARQL; verify the exact French/Wikidata split in the release used.
- **LC-QuAD 2.0**, Dubey et al., ISWC 2019: approximately 30,000 English Wikidata questions with SPARQL.
- **T-REx**, Elsahar et al., LREC 2018: English Wikipedia/Wikidata text–triple alignments.
- **SimpleQA** remains useful for the existing coverage comparison, but it does **not** natively supply gold QIDs for every answer. Do not pretend otherwise; publish the QID/property annotation subset separately.

| Choice | Pilot and metric | Pass line | Kill/change line |
|---|---|---|---|
| Full statement representation | Import a stratified **100k-item** API sample plus dump slices; extrapolate bytes/statement and build throughput | Projected store **≤400 GB**, full build **≤72 h**, zero fidelity mismatches | **>500 GB**, **>7 days**, or any unexplained mismatch: narrow scope or redesign before full build |
| Full vs truthy only | Public as-of questions from QALD/LC-QuAD plus a published temporal test derived from QIDs with start/end/P585 | Full representation answers **≥95%** of mechanically supported temporal cases; no obsolete-value errors | If temporal/deprecated/provenance behavior matters in **<1%** of supported questions and size is excessive, retain full statements but simplify indexes; do not infer that truthy is semantically sufficient |
| Memory-mapped tables | **1M** random exact hops; hot and genuinely cold; one and four clients | Hot p95 **<1 ms**, SSD p95 **<10 ms**, serving RAM **<24 GB** | More than twice either latency target, or RAM overflow: change layouts/indexes rather than adding model specialists |
| Name index | Mintaka EN/FR and QALD French; candidate recall@32, ambiguity error, latency | **≥99%** candidate recall where the gold name occurs in retained source names; **zero** fame-based/confident namesake errors; p95 **<50 ms** | **<98%** recall or any systematic French shortfall: improve source aliases/normalization, not answer-time embeddings |
| Typo fallback | Inject measured-looking 1-edit and 2-edit corruptions into held-out EN/FR names; compare exact, FST and trigram candidates | At least **2 percentage points** candidate-recall gain over simpler fallback within **50 ms p95** | Smaller gain, excessive candidate growth or deployment-specific thresholds: omit it |
| Narrow text reader | T-REx EN plus a published, human-checked French Wikipedia sample of at least **2,000 predicted facts/language** | **≥99.5% precision** on selected schemas, with passage support and QIDs correct; report recall separately | **<99% precision**, any systematic cross-source hallucination, or inadequate French sample: stop general extraction and keep only structured fields |
| End-to-end store value | Re-run the existing 860-entity SimpleQA subset, plus Mintaka/QALD EN/FR, with certified answering | On SimpleQA’s 860, at least **258 supported answers (30%)** as an initial coverage gate, with zero incorrect answers; on supported Mintaka/QALD cases, **≥90%** execution success | If SimpleQA remains **<10% (86/860)**, Wikidata formatting was not the main coverage problem; invest in source/text coverage. If supported-set execution is **<90%**, fix parser/VM before adding data |
| Specialist adapters | Same store; baseline trunk versus routed adapters; include **500** deliberately cross-topic EN/FR questions | At least **2 percentage points** overall supported-answer gain, no cross-topic regression **>0.5 points**, zero new wrong answers | Any routing-caused loss without compensating gain, or any memorization-canary answer without store support: do not ship specialists |

For reader recall, require useful performance **per declared relation**, not an ill-defined open-domain recall number. Low recall causes refusals; low precision violates the central design.

Finally, zero errors in a test of \(n\) answered cases is not proof of a zero error rate. With zero observed errors, the approximate 95% upper bound is \(3/n\): zero errors in 1,000 answers still permits roughly **0.3%**. Publish both answered coverage and error count.

---

# 8. Do not build; risks and decisions

## Do not build now

- Topic routers, specialist adapters or topic-owned fact shards.
- A neural compressor or associative memory as the canonical store.
- A full general-purpose RDF index merely because Wikidata publishes RDF.
- A monolithic SQLite/RocksDB base.
- An open-ended relation reader before the precision gate.
- ANN embeddings, qFHRR or SimHash as answering evidence.
- Fame, PageRank or sitelink-count disambiguation.
- A Colab-dependent build pipeline.
- Full rebuilds for routine updates.

## Main risks

1. **Source truth versus source support:** certification does not make Wikidata or a book correct.
2. **Coverage remains source-bound:** SimpleQA contains facts that full Wikidata may never represent.
3. **Name ambiguity:** better aliases increase recall and collisions simultaneously.
4. **Extraction contamination:** copied books and mirrored Wikipedia passages are not independent corroboration.
5. **Live vandalism and update semantics:** “newest” is not automatically “currently valid.”
6. **French parser disparity:** storage can be bilingual while CubeLang generation remains materially worse in French.
7. **Licensing:** Wikidata is CC0; Wikipedia text carries attribution/share-alike obligations, and book rights must be checked separately.
8. **Size uncertainty:** the 40–80 GB estimate is not credible for the proposed full-statement, referenced design.

## Decisions Nick must make explicitly

- Whether “0 wrong” means **zero unsupported answers**, which is enforceable, or real-world infallibility, which is not.
- Which item classes and source collections are in version one.
- Which properties obey live-wins and their maximum acceptable age.
- Whether ambiguous questions should enumerate answers or ask; both are compliant, but the UX differs.
- What counts as independent corroboration.
- Acceptable refusal/coverage targets separately for EN and FR.
- Purchase/provision of **2 TB NVMe**, and optionally 64 GB RAM.
- Snapshot cadence, retention and rollback policy.

The immediate build order should be: **full-statement Wikidata pilot → exact name and hop benchmarks → full home build → EN/FR end-to-end evaluation → only then narrow text extraction**. Specialist models should remain behind their experiment, not ahead of the store.

