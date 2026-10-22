# 0020 — Emotional arcs: a book (or a conversation) as a time series of feeling, and the change points where it turns
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes open 0020") · hypothesis: H-K11 (affect, order 0019), with K2 (surprise-gated
writing: a change-point probability is a boundary signal) · **the pipeline can start now on the human lexicons; the
0019 readers plug in when they pass**

## Why
Nick (2026-10-22) brought two research notes: track a book's feeling as a series of affect vectors, and find where it
really turns with change-point detection, not one label per chapter. The method is established:
- **Arcs.** Reagan et al. (2016, EPJ Data Science) built valence arcs for 1,327 Project Gutenberg novels with the
  labMT hedonometer (human-rated word happiness) over a 10,000-word sliding window, and found six recurring shapes:
  rise, fall, fall-rise, rise-fall, rise-fall-rise, fall-rise-fall. Vishnubhotla, Hammond, Hirst & Mohammad (2024,
  arXiv:2403.02474) split narration from each character's dialogue and found they carry different emotions across a novel.
- **Change points.** CUSUM (Page 1954) for a sustained shift in a stream; Bayesian online change-point detection (Adams &
  MacKay 2007, arXiv:0710.3742), a posterior over the run length since the last regime; PELT (Killick, Fearnhead &
  Eckley 2012) for the optimal set of change points in a finished text. Scores are autocorrelated, so a shift counts
  only with an effect size and persistence, not a p-value alone.

We already hold what this needs: human word ratings (Warriner 2013 valence and arousal, NRC EmoLex petals:
`validation/vad_check.py`), the books with paragraph offsets (`standin/books.py`), the fiction worlds (one per work,
`docs/WORLDS.md`), and the readers coming in order 0019 (the cleaned Ear, the head on the 450M).

- **What it brings:** each fiction world knows how its story feels over time and where it turns. Cubby can answer
  "when does it get dark?", "is it a sad book?", "find me one that starts grim and ends hopeful"; order 0008 can write
  to a planned arc; the same tracker over a conversation notices when someone's mood turns.
- **Advantage over conventional models:** a chat model asked where a story turns guesses from a summary. Here the turn is
  a measured position with its evidence span (shown in the audit view only), and it is reported only when a statistical
  test, an effect size and persistence all agree, so one dark line is not a "turn".
- **In plain words:** Cubby can tell you how a book feels chapter by chapter and where it changes, and notice when a
  conversation's mood shifts.

## Build
- `standin/arc.py` (numpy only, < 300 lines, `__wiring__` STANDALONE):
  - **Scorer, pluggable:** `lexicon` (Warriner valence / arousal and EmoLex petal rates over the rated words of a window,
    as the hedonometer does), `ear` (the cleaned Ear, order 0019 A), `head` (0019 B, once it passes). Same windows, same
    output vector: valence, arousal, eight petal rates.
  - **Segmentation:** the book's own chapter and scene markers where it has them; ~500-word windows, 100-word stride
    inside; position normalised to [0, 1]; each window keeps its character span (from `books.py`'s offsets).
  - **Smoothing and summaries:** EMA per dimension; per chapter the mean, spread, peak and where the peak falls.
  - **Change points:** CUSUM per dimension (online), BOCPD with a Student-t regime model (online), PELT with an l2 cost
    (offline, `min_size` >= 3 windows). A **shift event** is emitted only when BOCPD's P(new regime) > tau, the blocks
    before and after differ by |d| >= 0.5 (Cohen's d on the smoothed series), and the new level holds for >= p windows.
    tau and p are chosen on calibration books, then frozen.
  - **Arc shape:** the valence arc resampled to 100 points, correlated with each of the six shapes: stored as features,
    never forced into one label.
- `validation/exp_arcs.py`: builds arcs for the public-domain fiction we hold (the Gutenberg books we hold)
  and writes `validation/logs/arcs_<date>.json` (counts and scores only: no passage text).
- What a fiction world stores: the window series (float16), chapter summaries, shift events (position, direction per
  dimension, effect size, persistence, probability, a type such as calm-to-threat or relief-after-crisis), the shape
  features. Evidence is character spans into the source; the audit view shows the passage, replies never quote it.

## Acceptance (pre-registered)
- **Tests:** synthetic series with planted step changes are found within +/-2 windows by CUSUM, BOCPD and PELT; a series
  with none raises no shift event on 95% of seeds; constant input gives no change points; every scorer is deterministic;
  spans map back to the right text.
- **Replication (report):** on the Gutenberg fiction, with the 10,000-word window, the share of books whose valence arc
  best fits each of the six shapes, and the share that fits none above r = 0.5. Spot-check 10 books whose arcs are well
  known.
- **Gold set v0 (the gate; model-proposed, checked in our copies, confirmed by Nick):** one book per shape, its
  pivot found in the text we hold, its position as a fraction of the book. Checked 2026-10-22:
  - *Pride and Prejudice* (rise): Elizabeth's self-reproach after Darcy's letter, ch. 36 in our copy, at 0.551.
  - *Romeo and Juliet* (rise-fall): "fortune's fool", III.1, at 0.504-0.510. Our copy is the First Folio spelling
    ("blacke Fate", "Fortunes foole"), which the lexicon cannot read: use a modern-spelling edition or a spelling map.
  - *Frankenstein*, 1818 text (fall-rise-fall): the creature wakes and the dream turns to disgust, ch. IV (ch. 5 in the
    1831 text), at 0.188.
  - *The Count of Monte Cristo* (fall-rise), held, in place of *The Martian* (not held, not public domain): the arrest
    at the marriage feast (fall), ch. 5, at 0.026; the escape, ch. 20, at 0.151.
  - *Madame Bovary*, Marx-Aveling translation (fall): the bailiff's inventory, Part III ch. 7, at 0.846 (the steepest
    point of a steady fall).
  - *Great Expectations* (rise-fall-rise): Magwitch's return, ch. 39, at 0.658.
  - *Romeo and Juliet*: also fetched in modern spelling (Gutenberg #1513): III.1 at 0.505-0.511.
  The three fetched texts (Gutenberg #2413, #1400, #1513; public domain) sit beside the collections (path in
  `CLAUDE.local.md`). Several proposed quotations were not in the texts (two in *Great Expectations*, the whole
  *Madame Bovary* passage in this translation): positions come from the texts, never from the proposals.
  Nick reads each located passage and confirms it is a turn. The proposals came from a model, so the set is labelled
  "model-proposed, human-confirmed", never "human gold".
  - **Calibrate** tau and p on two books (*Pride and Prejudice*, *Frankenstein*); **test** on the rest.
  - **Gate (smoke, n = 6):** Cubby speaks only of a book's strongest shift, so precision@1 is the first clause: the top
    shift lies within +/-3% of the pivot on >= 3 of the 4 test books; the arc's best-fitting shape is the labelled one
    on >= 4 of 6. **Kill:** top shift on the pivot in <= 1 of 4: shifts stay an audit feature, never spoken.
  - A fuller gold (every turn of each shape, marked by a person who has read the book) replaces v0 before any claim.
- **Folders are not genres:** *Pride and Prejudice* sits in the collection's `factual_books` folder (28 of its 2,445
  books are fiction by Gutenberg's catalog). The fiction world takes the catalog's fiction flag over the folder.
- **Readers:** the same books scored by `lexicon` and `ear` (later `head`): the correlation of their valence arcs, and
  the gold-set precision of each.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model scores or labels any book. Scorers are human lexicons or Cubby's own readers.
- Sources are internal: no passage text in a committed file or log; replies say where a story turns in their own
  words (chapter, roughly how far in), never quoting it.
- Fiction worlds answer about the fiction only.

## Out of scope
- Character arcs (narration vs each character's dialogue): needs speaker attribution; a later order.
- Using arcs to change narration (pace, voice, ambience) or for content warnings.
- Wiring the conversation tracker into the stand-in's afferent or the cube: report its read on the Teams corpus at most.
- K2's use of the change probability as a write signal: a lead for that hypothesis, not built here.

## Addendum 1 (2026-10-22, Cowork): human tests for the conversation tracker
The same tracker over a conversation (report only, as above), on two sets fetched for order 0019 (Addendum 2):
- **ESConv:** each seeker's self-reported intensity before and after the conversation (1-5) is a human label for "the
  mood shifted". Report the correlation between the tracker's change over the seeker's turns and the self-reported
  change, and whether a change point falls where the seeker's helpfulness ratings (given every two supporter turns)
  rise.
- **MELD:** emotion shifts between consecutive utterances of a scene, labelled by people: change-point precision and
  recall at the utterance level.
Both are for research use (CC BY-NC 4.0; a TV script): no example text in a committed file or log.

## Addendum 2 (2026-10-22, Cowork): scripts make character arcs possible
The collection's drama folder holds 78 Gutenberg plays (many languages) and 25 screenplays and TV pilots (feature
films, pilots, one teleplay; paths in `CLAUDE.local.md`). A script marks what a novel hides: each scene opens with a
heading (INT. / EXT.), and each line of dialogue under its speaker's name in capitals. A count over the 25 found
125 to 1,497 speaker cues per script, the main characters on top. So the speaker is given, not inferred:
- **Character arcs, scripts only:** an arc per main speaker over their own lines, and one for the action lines (the
  script's narration), segmented by scene headings. This is the narration-vs-character split of Vishnubhotla et al.
  (2024) with no attribution model. Report only: the gold set above stays on novels.
- The English plays mark speakers too (e.g. "ROMEO." before a speech): the same parser, report only.
- **Fiction stays fiction:** some scripts and plays are about real people (a Napoleon screenplay, Shakespeare's
  histories). Their worlds answer about the work, never as evidence about the real person.
- The screenplays are copyrighted: research use only, no example text in any committed file or log.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
