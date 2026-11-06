# A terabyte outside the weights: what the panel said, checked (Cowork, 2026-11-06)

The brief: `2026-11-06-fact-store-brief.md`. The answers: `2026-11-06-fact-store-panel.md` (7 of 8 models answered;
DeepSeek V4 Pro spent its budget reasoning, twice). Cost: about $1.85.

**In plain words:** Cubby keeps every fact it may say in its own database on Nick's PC, about a hundred times smaller
than the raw downloads. It finds any of 100M+ things by name in English or French in milliseconds, and it always
knows where each fact came from.
- **Over a conventional model:** a model's knowledge is frozen in its weights, cannot be inspected, and costs a
  retraining to update. Here a fact is added, corrected or traced in one write, and the model never has to memorize
  it.

## 1. The mixture of specialists: all seven say no, for one reason
- **You cannot route to a topic before you know the entity, and finding the entity is the query** (Grok). A
  specialist adds no fact the store lacks (Kimi, Qwen); a topic router becomes a fame prior by the back door (Claude).
- **What survives of the idea:**
  - shards by storage (id ranges or hash), not by topic;
  - specialist *program proposers* or schema adapters on the one trunk, only if an experiment beats the single trunk,
    with a memorization canary: any answer the store cannot support kills it (Kimi; GLM similar).

## 2. First, before downloading anything: is the fact in Wikidata at all?
- **A correction to my brief:** the "3 of 860" SimpleQA figure was measured on the small wiki world, not on the
  Wikidata5M world. Claude's diagnosis (Wikidata5M holds entity-to-entity triples only, no dates, numbers or strings)
  is about the latter. The point stands for both: neither measure says anything about full Wikidata.
- **The audit (Claude's E1, GPT's 860 audit, Gemini's Exp1, Qwen, Kimi):** on public question sets with French
  (SimpleQA, Mintaka, MKQA), check whether full Wikidata holds the gold as a statement of the question's entity.
  - It needs no download: the live Wikidata API, or QLever's public endpoint, on about 1,000 questions.
  - **Kill line:** if full Wikidata holds the answer for under about 20% of SimpleQA, the lever is text, not the dump.

## 3. The store, where the answers converge
- **One read-only store, memory-mapped, in Rust** (5 of 7; GPT prefers RocksDB): dense 32-bit ids, sorted forward
  and reverse adjacency, a small writable overlay for live values.
- **What it keeps:**
  - statements at every rank, flagged (deprecated ones kept by four, dropped by three);
  - the time qualifiers (P580, P582, P585 and a few more) hot, the rest cold;
  - references deduplicated, as provenance;
  - EN, FR and `mul` labels and aliases; EN/FR sitelinks;
  - scholarly articles and similar bulk classes in a cold tier.
- **Exact values:** no floats for quantities; dates keep their precision and calendar.
- **Names:** an FST (the `fst` crate) with exact and accent-folded keys, then a bounded typo path, the full candidate
  set filtered by the program's own constraints; otherwise ask. Never ordered by fame.
- **Size:** the hot store's estimates run from 25 to 90 GB without scholarly articles (two answers say 200-650 GB with
  everything). The spread is the reason to build a sample first and measure bytes per statement.
- **Hardware:** a 2 TB NVMe (four of seven), which also replaces the failed 4 TB drive; 64 GB of RAM recommended by
  all seven, though 32 GB can serve.

## 4. The build
- **At home, streaming the compressed dump without unpacking it,** in under a day to a few days (estimates vary).
- **A checked reference:** QLever's own Wikidata recipe indexes the whole dump in about 4.5 h with about 20 GB of RAM
  on a desktop Ryzen 9 5900X, from about 100 GB of compressed downloads. That makes it a usable oracle to check our
  store against, at home (the panel disagreed on QLever's numbers; its own config file is the source here).
- **Proof that the store is faithful:**
  - a second, independent parser (Kimi: Wikidata Toolkit);
  - a revision cross-check: each item's revision recorded, a sample re-fetched at that revision (Claude);
  - a deterministic rebuild that gives identical chunk hashes (Kimi).

## 5. Text (Wikipedia, books)
- **Version 1:** passages kept for provenance in a text index (Tantivy).
- **Facts from prose:** only by deterministic rules (infoboxes, lead sentences, links), behind a precision gate. Grok
  expects even that to fail its 1.0 precision line and keeps prose out of v1.
- Wikipedia and DBpedia are not independent of each other, and much of Wikidata comes from Wikipedia: they cannot be
  each other's "second source".

## 6. Ideas from one answer each, worth keeping
- **`not_speakable` (Grok):** a statement with a qualifier the answer template cannot say is never spoken. This
  blocks a metro-area population being read as the city's.
- **Namesakes that agree (Claude):** if every candidate gives the same checked value, answer once. Fewer questions
  back, no fame.
- **A property retriever (Gemini):** EN/FR labels and aliases of every Wikidata property in the name index, the top
  25 passed to the model. This puts the schema outside the weights, which is 0048's relation-word gap.
- **The subclass closure (Gemini):** precomputed once, a single type check in the VM.
- **Keep the compressed source records (GPT):** a parser bug is fixed by reparsing, not re-downloading.
- **Error categories (GPT):** identity, interpretation, time, source disagreement, fabrication, counted apart.

## 7. Rejected, because they break a rule
- Quantities as floats (Qwen).
- "The latest date wins" (Qwen, GLM).
- A similarity threshold on trigrams (Gemini).
- Keeping or pruning items by their number of sitelinks: fame by another name (Gemini, Qwen).
- Wikipedia text "confirming" a Wikidata value: not an independent source (Gemini).

## 8. Facts checked here
- QLever's Wikidata recipe: about 4.5 h and 20 GB of RAM on a Ryzen 9 5900X; about 100 GB of compressed downloads
  (its `Qleverfile.wikidata`).
- **Not yet checked:**
  - the dump's current sizes and counts (the answers conflict: 100-150 GB compressed, 1.0-1.9 TB uncompressed,
    0.7-2 B statements);
  - every licence the panel named (several conflict);
  - every paper citation.

  Each is checked before it reaches an order or a paper.
