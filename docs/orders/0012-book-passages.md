# 0012 — The books as a source: a passage finder, facts read by frames, attributed until corroborated
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("we absolutely need to add all the verified facts and
there's a lot more to come"; "maybe add a passage finder for the books"; trust tier: attributed until corroborated) ·
hypothesis: H-F2 (temporal worlds), the ask loop, H-K10
**Suggested priority:** after 0011, before 0007. The books become the lookup's first, local, free source.

## Why
Nick's history collection is ~4,700 books, ~3.7 GB of text, in an era / region / topic tree. More are coming. Its
cleaned, twin-deduped copy is the pretraining domain `verified_facts` (paths in `CLAUDE.local.md`). The trunk has read
them, but no fact in them is in the store, so none can be answered by address, checked, or cited. The encyclopedia
already shows the shape: an offline `learn.Source`, facts read by frames, provenance = volume + headword + sentence
(`standin/encyclopedia.py`). This order does the same for the books, at their scale, and adds what the encyclopedia
lacks: a **passage finder** that returns the paragraphs about an entity, with their citation.

- **What it brings:** every claim Cubby takes from a book comes with the passage that says it. A question it can't
  answer looks in Nick's library first, locally, before the network. A fact two independent authors agree on becomes
  a plain fact.
- **Advantage over conventional models:** they absorb books into their weights and cannot say which book said what,
  or whether two sources agree. Here each fact keeps its witness, and agreement is counted, not assumed.
- **In plain words:** it knows what your history books say, keeps track of which book said it, and only states
  a fact flatly once two different authors agree.

## Build
- **The index** (`standin/books.py`, new, < 300 lines; off-repo index file, path from the environment):
  - SQLite FTS5 over paragraphs. Each row carries its book (author and title from the file name; era, region and
    topic from the folder path) and its character offset.
  - An entity-mention table built with the worlds' `NameIndex` (373,689 names), so an entity's paragraphs come back
    without a text search.
  - Incremental by file hash: a re-run adds new books and skips the rest; nothing is deleted.
  - Read both source folders (the cleaned domain where a twin exists there, the original otherwise) and
    `historical_facts.jsonl`'s dated events.
- **The passage finder:** `passages(entity, words=None, era=None, k=10)` returns ranked paragraphs, each with
  `(author, title, era, offset, text)`.
- **`BookSource`** (implements `cubbyllm.reasoning.learn.Source`, like `EncyclopediaSource`):
  - For a stalled (entity, relation), find passages and read facts with **frames**: birth and death years and
    places, founded by / in, ruled from–to, located in, part of, battle of X in year Y, the shapes the encyclopedia's
    frames and the wiki world's relations cover.
  - Provenance is the book, the offset and the sentence.
  - What no frame says, the source does not say.
- **Tiers** (Nick's rule):
  - one book → **attributed**, spoken hedged ("one account says …"), the book never named to a user;
  - two independent sources (different authors, or a book and Wikidata) → **corroborated**, spoken as fact;
  - a contradiction between sources → held, both kept, nothing spoken flat.

  These ride on the store's existing provenance and latent machinery (`standin/worlds.py`, the MoWM worlds). The
  book's era routes the fact to its temporal world (H-F2).
- **In the ask loop:** `BookSource` goes first among the fallbacks (local, free), then Wikidata, then the web.
- **Sources are internal (Nick):** a regular user never sees a citation, a title, an author or a quote; the audit
  view (order 0016) does.
  No book text enters the repo: the index stays off-repo, and tests use synthetic paragraphs.

## Acceptance (pre-registered)
- **Index:** builds over all current books. Report time, size, paragraphs and entities indexed. A re-run with one
  new book adds only it.
- **Passage finder:** on 100 (entity, relation) pairs the wiki world holds, the gold object appears in one of the
  top-5 passages for ≥ **60%**. Report it per era.
- **Frames:** on ≥ 200 triples they read where Wikidata states the same relation, agreement / (agreement +
  contradiction) ≥ **0.95**. Report the yield per frame.
- **Bench:** 100 questions the store cannot answer but a book can.
  - **0 wrong spoken**.
  - Every attributed answer carries, in its record, a citation whose passage holds the subject, the value and the
    relation cue. No reply text names a source.
  - Corroborated answers are spoken flat only with two independent sources.
- Off = unchanged (the ask loop without `BookSource`); full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. CPU, Nick's machine. No model, no network. Nothing from the books in a committed file.

## Out of scope
- The learned reader for what frames miss (order 0013). Training on the books (they are already in the corpus).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
