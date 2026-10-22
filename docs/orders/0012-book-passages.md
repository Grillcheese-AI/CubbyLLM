# 0012 — The books as a source: a passage finder, facts read by frames, attributed until corroborated
status: in progress
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
- **Natural questions (Nick, 2026-10-22).** The bench and the parser's tests ask the way people ask: "Who
  founded Rome?", "Where was Napoleon born?", "When was Napoleon born?". Not "What is the founded by of Rome?".
  `What is the <relation> of <entity>?` may stay as the host's internal plan string only. Render questions from the
  hdc set's `relation_template_mapping.csv` (a natural template per relation) plus a few hand-written variants
  per relation, and check the parser maps each one to the same (entity, relation).
  **Shape (Nick):** `{wh} {tense} {verb form} {subject} [in {year}]?`.
  - The wh-word comes from the answer's type: person → Who, place → Where, date → When, count → How many, else
    What/Which.
  - The tense comes from the fact's dates: ended → was, ongoing → is, scheduled → will be.
  - The verb form is set per relation: active ("Who founded Rome?"), passive ("Where was Napoleon born?") or a
    noun ("Who is the CEO of …?"). Extend hdc's template table with the three tenses.
  - The parser reads the tense and the year back as a **time constraint**: which year's world, and whether a
    time-sensitive fact needs its "as of" or a live refresh (order 0014). The bench includes the same fact asked
    in the right and the wrong tense: "Who is …?" on a fact that ended must not be answered flat.

## Constraints
- Rules in `CLAUDE.md` hold. CPU, Nick's machine. No model, no network. Nothing from the books in a committed file.

## Out of scope
- The learned reader for what frames miss (order 0013). Training on the books (they are already in the corpus).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:** `f9ca177` (index, frames, BookSource, natural questions), `b52e56d` (question shape and tense, with
  0014), `45a2470`, `6dabcef`, `5507eca`, `1455c06` (frames and finder fixes from the real books), `603d57f` (the
  forms resolve a wording before the lexicon; the covered finder), and this report's commit.
- **Files:** `standin/books.py` (index, mentions, finder), `standin/book_frames.py`, `standin/book_source.py`,
  `standin/question_forms.py`, `standin/data/build_books_index.py`; `cubbyllm/reasoning/learn.py` (the
  `attributed_only` tier, `before=`), `cubbyllm/reasoning/sleep.py` (routes for `attributed_only` and
  `time_mismatch`); `standin/ask.py` (`books=`, `tense=`, the forms as the first resolver), `standin/serve_api.py`
  (`--books`, `--tense`); `validation/exp_books_bench.py`. Tests: `standin/tests/test_books.py` (12),
  `standin/tests/test_question_forms.py` (5).
- **Tests:** the books tests pin twins read once, mirrors first, meta from names, one new book added alone,
  mentions, the finder's ranking, frames firing only on their subject, the three tiers, the loop's hedged reply
  with its citation, a second source lifting it, a contradiction never said, the tense and year checks, the walk
  going back to the proposed wording, and the forms resolving before the lexicon. Off (`books=None`): the loop as
  it was. `pytest tests`: 428 passed, 5 skipped, 1 failed (the known wiring guard, order 0002). `pytest standin/tests`:
  613 passed, 1 skipped. The 25 failures and 1 collection error are in `test_build_chat_sft`, `test_chat` and
  `test_goemotions_cube`; they need GOEMOTIONS / MULTI_PROMPT, missing at HEAD before this order.
- **Measured** (CPU; logs `validation/logs/build_books_index_2026-10-22.json`, `..._rerun_2026-10-22.json`,
  `exp_books_bench_v4_2026-10-22.{json,log}`):
  - **Index: passes.** 5,291 books: 5,286 files plus 5 era books from the events file. 225 duplicates were found
    by content and read once. 5,639,095 paragraphs, 76,740 names mentioned, 19,536,055 mentions, 7.96 GB. The
    first build took ~32 min (5,200 books at 1,912 s; that log was overwritten by the relaunch, which added the
    last 5 in 62 s). A re-run adds 0 and skips 5,516 in 64 s. One new book adds only it (test).
  - **Passage finder: fails** (≥ 60% pre-registered).
    - Uniform pairs: 10/100 gold in the top 5. Only 19/100 have the gold anywhere in the top 200 passages.
    - Book-covered pairs (relations a history book states, subjects in 10+ paragraphs): 29/100, ceiling 43.
    - Per era (covered): AncientClassical 15/39 (ceiling 21), MiddleAges 5/31 (11), unknown 9/29 (11),
      Prehistory 0/1.
    - The premise does not hold: most wiki-world pairs are not stated in these books at all, so no ranking reaches
      60%. The ceiling, not the ranking, is the limit.
  - **Frames: fail** (≥ 0.95 on ≥ 200 pre-registered).
    - Only 79 comparable triples can be found offline: 8,133 subject asks, yield ~1% a frame.
    - Agreement with coarser places counted: 0.821 (dev 0.875 / test 0.763). Strict: 0.795.
    - Per frame: date of death 17/17, founded by 4/4, date of birth 17/20, place of birth 18 agree / 8 contradict /
      2 coarser, place of death 2/4, capital 3/4.
    - Many "contradictions" are the gold's grain (a region for a town) or a second person with the same name. The
      bench's reading below shows them.
  - **Bench** (76 questions: the frames' reads give 76 distinct pairs, not 100; natural questions, the wiki facts
    taken out):
    - 31 spoken: 29 hedged, 2 flat. 45 refused: 21 retrieval exhausted, 21 answer-type mismatch, 3 ambiguous.
    - **Wrong spoken flat: 2 as scored. 0 wrong spoken fails as scored.** Read one by one:
      - *Gaiseric died in Carthage*: the gold is "Vandal Kingdom"; Carthage was its capital, so the book's answer is finer.
      - *The capital of the Kingdom of Kush is Kerma*: the gold is Meroë. Kerma was Kush's earlier capital, and two
        independent authors say so. This is the time problem: an undated state with several values over its life
        was said flat.
    - Wrong hedged: 4 as scored.
      - Stukeley born 1694 (gold 1687): a real error, a book's or a frame's.
      - Stukeley died in London (gold Kentish Town): coarser.
      - Alcuin born 732 (gold "c. 735"): the sources disagree.
      - Samuel Fuller born 1580: the book's Samuel Fuller is the Mayflower physician, the world's is the filmmaker
        (two people, one name).
    - Citations: 29/29 hedged answers carry a citation sentence holding the subject, the value and the cue. Replies
      naming a source: 0. Flat answers without two independent witnesses: 0.
    - **The wrong tense: 0/76 answered flat** (28 refused `time_mismatch`).
- **Not verified here:** a 200-triple frames check (only 79 are comparable without network). The gazetteer for
  "coarser" used the hdc world built on 38% of its labels; the label fetch is still running (order 0014), and a
  full gazetteer may move Stukeley's London and Gaiseric's Carthage to coarser and finer.
- **Found on the way:**
  - The lexicon resolved the question's words before the forms ("die" → "dice", "born" → "max born"): 67 of 76 bench
    questions were refused `unknown_relation` until the forms came first (`603d57f`).
  - An FTS `IN` subquery made a much-mentioned name's passages 380× slower (`5507eca`).
  - The events file holds a raw control character in a text.
  - The History Book Collection keeps each book in its own folder under a format folder, which first read as an
    era (the first index log's eras are sanitized: those six were book titles).
  - "Born about 742" and "X, who died in 814" needed their own frame shapes.
- **Questions:**
  1. The kill line on the time problem: should an undated state with an end (a kingdom's capital) need a date
     before it is said flat, even on two witnesses? That would have refused Kerma.
  2. The finder's 60% assumed the books state the wiki world's pairs. 43% is the ceiling on the covered selection.
     Keep 60% as the line for 0013's reader, which reads every paragraph rather than ranking, or move it?
  3. A finer place than the gold (Carthage for the Vandal Kingdom): is it scored as agreement?

## Review (Cowork, 2026-10-22): back to open
Checked against `build_books_index_2026-10-22.json` and `exp_books_bench_v4_2026-10-22.json`; the numbers stand.
- **Index: passes.** 5,291 books, 5,639,095 paragraphs, 7.96 GB; a re-run adds 0.
- **Passage finder: fails as pre-registered** (10/100; covered pairs 29/100). The premise was wrong: most of the wiki
  world's pairs are not stated in these books (covered ceiling 43/100). Recorded as a fail, not re-lined after the
  fact. Of the covered pairs whose answer the books hold, the finder ranks it top-5 for 29 of 43 (0.67): order 0013
  pre-registers that conditional measure.
- **Frames: not yet judged:** 79 comparable triples, short of the 200 the line needs (agreement 0.821 with coarser
  places, strict 0.795). Re-run when order 0014's labels are complete.
- **Bench: the kill line is breached once.** "The capital of Kush is Kerma" was spoken flat on two witnesses; Kerma
  was an earlier capital. Gaiseric in Carthage is a finer, true place, not a wrong one.

To close:
1. **The time rule (question 1: yes).** A relation whose value changes over a subject's life (capital, ruler or
   head of state, spouse, member of, the country a place belongs to, population, an office's holder) is said flat
   only with a date: in both witnesses, or matching the question's tense and year. Without one it is hedged in
   time ("Kerma was a capital of Kush at one point") or refused. List the relations; seed it from Wikidata's
   properties that take start / end time qualifiers.
2. **Finer is agreement (question 3: yes),** when the gazetteer shows the answer inside the gold (Carthage within the
   Vandal Kingdom). Coarser stays hedged, never flat.
3. **A same-name guard:** a book's person counts for a subject only if the book's dates or era fit the subject's
   life (the Mayflower's Samuel Fuller is not the filmmaker).
4. **Re-run** the bench (0 wrong flat, as pre-registered) and, after 0014's labels, the frames on >= 200 triples.
