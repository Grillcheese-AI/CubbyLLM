# 0013 — A Cubby-native reader for the books, and the bulk pass that puts all their facts into the worlds
status: in progress
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick (extractor: "Cubby-native reader"; "add all the verified
facts") · hypothesis: H-F2, H-K10 · **after 0012**

## Why
Frames (order 0012) read the easy shapes with high precision and miss most of what a history book says. Nick's
rule: no external model. So the reader is the 450M with a LoRA adapter, taught by **distant supervision**. A book
sentence that names a subject and an object the wiki world already relates is a labelled example, for free. Then one
pass over every book fills the store, incrementally, since more books are coming.

- **What it brings:** the facts of thousands of history books in Cubby's store, each with its passage, read by its
  own model.
- **Advantage over conventional models:** their reading is baked into weights with no source. Here every fact is a
  row with a witness, checkable and correctable one at a time.
- **In plain words:** it reads your whole history library once, writes down every fact with where it found it, and
  keeps it up to date as you add books.

## Build
- **Training data (`standin/data/build_reader_sft.py`):**
  - **Positives:** sentences from 0012's passages that name `s` and `o` where the wiki world holds `s r o`. The
    example is (sentence, `s`) → (`r`, `o` as it appears in the sentence).
  - **Negatives:** sentences naming both with no known relation → NONE.
  - **Held out:** books, not sentences, so a test book was never seen.
  - The value must occur in the sentence (the `LfmReader` rule).
- **The adapter:** a reader LoRA on `base450m_cont`, the talk adapters' recipe, trained on Colab by Cowork. Its output
  is a relation id (closed set) and a span copied from the sentence.
- **The bulk pass (`validation/books_bulk_read.py`):**
  - Frames first, then the reader, over every paragraph that mentions an entity.
  - Candidates go through the memory gate (dedup, contradiction), take 0012's tiers, and route to the era's world.
  - Re-runnable by file hash. Report yield per era and per relation, and the corroborated share.

## Acceptance (pre-registered)
- **The reader on held-out books:** relation + value exact ≥ **0.90** precision at ≥ 0.50 recall on the aligned
  sentences. NONE on negatives ≥ 0.95.
- **A hand check:** 100 random novel extractions (no wiki world match), each shown with its passage, judged by Cowork
  and Nick. Precision ≥ **0.90**, or the reader's facts stay latent and are not spoken.
- **The bulk pass:**
  - report counts (triples, entities, per era);
  - **0 wrong spoken** on 0012's bench, re-run with the bulk store;
  - a re-run with no new book changes nothing.
- **Kill:** hand-check precision below 0.80. Then the reader does not speak, and frames stay the only reader.

## Constraints
- Rules in `CLAUDE.md` hold. No external model, in the data or the reader. Book text stays off-repo, and the
  training records live off-repo.

## Out of scope
- Writing the facts into the trunk's scene in bulk (order 0009's capacity). Training the trunk on the facts.

## Addendum (Cowork, 2026-10-22): the finder's measure, from 0012's review
Order 0012 found that most of the wiki world's pairs are not stated in the books at all (covered ceiling 43 of 100),
so a recall over all pairs measures the books, not the reader. This order reports recall **conditional on the answer
being in the books** (0012's finder: 29 of 43, 0.67), beside the unconditional rate, and the reader's gate is on the
conditional one. The time rule, finer-is-agreement and the same-name guard from 0012's review apply to what the reader
writes into the store.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
