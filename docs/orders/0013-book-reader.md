# 0013 — A Cubby-native reader for the books, and the bulk pass that puts all their facts into the worlds
status: review
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
- **The finding first: distant supervision does not label these books well enough to train a reader that passes
  the hand check.**
  - Of 100 random training positives, **45 state the relation they are labelled with** (Claude Code's reading,
    `validation/logs/build_reader_sft_audit_2026-10-22.json`; the file with each verdict is off-repo beside the
    index, for Cowork and Nick to confirm).
  - The rest are co-occurrences: lists, citations, index lines, or a wiki-world pair the sentence never states
    (Wang Mang's "spouse" is his aunt in the sentence).
  - Before the cue / gap / junk rules it was about 7 of 25. Those rules lift it to 45 and leave 7,219 positives
    over 48 relations, few for a reader (births by place: 11; founders: 26).
  - A reader trained on these labels would learn to answer on co-occurrence. Its precision against the labels
    could look high while the hand check fell under the 0.80 kill line.
  - So no adapter notebook was built. The data, the reader's contract and the bulk pass are built and tested, and
    frames stay the only reader (the kill line's fallback, reached here from the labels rather than the reader).
- **Commits:** `aec061b` (the data builder, the reader's contract, the bulk pass), `71aa0ec` (frames' place
  cleanup, applied to the stored reads too), and this report's commit (the held facts as a veto).
- **Files:**
  - `standin/data/build_reader_sft.py` (new): distant supervision with the cue / gap / junk rules, held out by book,
    records off-repo, `--audit` / `--score-audit`.
  - `standin/book_reader.py` (new): the prompt and answer form, the parse (a relation of the set, a value the
    sentence writes), dates to the store's form, one store sentence per fact.
  - `validation/books_bulk_read.py` (new): the bulk pass.
  - `standin/ask.py`: `load_held`, and the refusal `contested` (`cubbyllm/reasoning/sleep.py` routes it to
    latent_audit).
  - `validation/exp_books_bench.py`: `--bulk`.
  - Tests: `standin/tests/test_book_reader.py` (4), `standin/tests/test_books_bulk.py` (6).
- **Tests:** the reader's contract; every read takes its tier and era; a re-run with no new book changes nothing,
  and a new book is the only one read; the loop speaks one author's fact hedged; an undated capital is said "at one
  point"; a book whose person does not fit the subject's life is set aside; a held (subject, relation) is never
  said, even from the store.
- **Measured:**
  - **The bulk pass** ([read](../../validation/logs/books_bulk_read_2026-10-22.json),
    [re-run](../../validation/logs/books_bulk_read_rerun_2026-10-22.json)):
    - 5,217 books read (74 more are empty files), 5,134 frame reads, in 12,050 s on 3 workers.
    - **3,263 facts about 1,780 entities**, after the same-name guard and the place cleanup: 298 corroborated,
      1,932 attributed, 1,033 held. Corroborated share 0.134.
    - Per era: AncientClassical 985, MiddleAges 848, Prehistory 47, unknown 349, EarlyModern 1.
    - **A re-run with no new book reads 0 books and writes the same files** (23 s).
  - **0012's bench with the bulk store** ([log](../../validation/logs/exp_books_bench_bulk_2026-10-22.json)):
    - 2,228 facts and 1,033 held loaded.
    - **58/76 spoken (46 flat, 12 hedged), 0 wrong flat**; 1 wrong hedged (Alcuin born 732; the gold says
      "c. 735").
    - 14 refused `contested`; replies naming a source 0; the wrong tense said flat 0/76.
    - The live run without the store spoke 30/76.
  - **The labels** ([log](../../validation/logs/build_reader_sft_2026-10-22.json)): 416,134 sentences seen;
    6,567 train and 652 test positives, as many negatives; 4,216 train books, 481 test books.
- **The addendum's rules on what is written into the store: in.** The time rule reads a store's book facts. Finer
  is agreement in the scoring. The same-name guard runs in the bulk pass's aggregation. The conditional finder
  measure needs a reader to measure, so it is not run.
- **Not verified here:** anything about a trained reader: no adapter, no held-out precision, no hand check of
  extractions.
- **Found on the way:**
  - The bulk pass held Stesichorus's birthplace (Himera in some books, Metauros in others). But a held fact never
    reached the loop, so the live books reader, seeing only Himera in its top passages, said it flat on two
    witnesses: the first bulk-store run had 1 wrong flat.
    - `AskLoop.load_held` now loads the held file as a veto: an answer on a held (subject, relation) is refused
      `contested`, whoever states it.
  - PDF line-end hyphens ("Mem-phis") and headings run into a name ("BICULTURAL KINGDOMAlexandria") were read as
    capitals. They are cleaned in frames, and in the stored reads at aggregation (re-reading every book takes 3.3 h).
  - A frames change does not re-read old books. The pass is incremental by book hash and reader tag, so a rule
    applied to stored values (as here) or a new reader tag is the way.
- **Questions:**
  1. The labels: hand-label a few hundred sentences as a seed set (Nick and Cowork), or keep frames as the reader?
     An external model may not write the data.
  2. Should the bulk worlds be mounted on the serve path (`serve_api`)? Here they are loaded by the bench only.
