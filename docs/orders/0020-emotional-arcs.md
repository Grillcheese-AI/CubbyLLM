# 0020 — Emotional arcs: a book (or a conversation) as a time series of feeling, and the change points where it turns
status: proposed
written: 2026-10-22 by Cowork · approved: — · hypothesis: H-K11 (affect, order 0019), with K2 (surprise-gated
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
- `validation/exp_arcs.py`: builds arcs for the public-domain fiction we hold (the Gutenberg rows of the rights tracker)
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
- **Human gold (the gate):** Nick marks the turning points of 5 books he knows well (position and direction; about 10
  minutes a book from memory or the table of contents). tau and p are fit on 2 of them; on the other 3, a detected shift
  matches a mark within +/-3% of the book. **Gate:** precision >= 0.6 and recall >= 0.5. A shift reported where there is
  none is a wrong answer, so precision is the gate's first clause.
- **Kill:** precision < 0.4 on the held-out 3: shifts stay an audit feature and are never spoken; the chapter summaries
  may still be.
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

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
