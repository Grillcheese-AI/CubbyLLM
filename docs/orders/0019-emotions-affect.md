# 0019 — Emotions: clean the affect labels, then give the 450M a read of feeling (a small head on the frozen trunk)
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yep") · hypothesis: H-K11 (proposed here; Cowork adds it at the first measured
number), with the cube (`docs/cube_vs_valence_arousal.md`, `docs/cube_vs_human_vad.md`) and the Ear (`standin/ear.py`)

## Why
Nick (2026-10-22): train Cubby on emotions with `amygdala_affect.jsonl`. Today feeling is read only in `standin/`: the
Ear (`standin/ear.py`, one hypervector direction over words, word pairs and trigrams, valence only) and the cube
(`standin/feeling.py`, `neurochem.py`). The 450M trunk has no affect read.

A triage of the affect sets found a defect first. **In `amygdala_affect.jsonl`, 1,831 of 3,180 rows carry the default
(valence 0.0, arousal 0.5):** 0.1% of rows 1-1,260, 95% of the rows after. They are unlabeled, not neutral ("I'm feeling
very low and just need a hug" reads 0.0). About 1,349 rows are labeled; rows 1-1,260 are
`emotion_valence_arousal_realm_phase.jsonl` with its context dropped. `standin/data/build_ear.py` (`corpora()`) teaches
every row at half weight, so the Ear has been taught that sad and happy sentences are neutral. Its held-out `sign`
metric skips zero labels (275 amygdala sentences scored, `validation/logs/ear_build_2026-10-16.json`), so the build log
cannot show the damage.

Then the trunk. The Ear reads a bag of n-grams: it cannot see a negation two clauses back, sarcasm, or a short "ok"
after five messages about a crash (`validation/label_affect.py` explains why context matters in this corpus). The
trunk's stream sees the whole sentence. A small head on the frozen trunk, trained like order 0005's copy path (the
trunk untouched, text identical), tests whether that context is worth a second reader.

- **What it brings:** Cubby reads how a message feels in context (valence, arousal, a Plutchik petal and its
  intensity), and the cube gets a second, context-aware input next to the Ear.
- **Advantage over conventional models:** a sentiment classifier is one opaque verdict. Here two readers (the
  inspectable Ear, the trunk's head) must agree before a feeling is read with confidence; when they disagree, Cubby
  is unsure and says less. The head is a few thousand parameters on a frozen trunk: nothing else moves.
- **In plain words:** Cubby learns to tell whether a message sounds happy, worried or annoyed, from the whole
  sentence, and holds back when it is not sure.

## Build
**A. Clean the labels (CPU, first; it stands on its own).**
- One loader for the affect sets (`validation/affect_sets.py`, or inside `build_ear.py` if it stays small) that drops
  rows equal to the default (valence 0.0 and arousal 0.5), dedupes texts, and reports counts per file:
  `amygdala_affect.jsonl`, `emotion_valence_arousal_realm_phase.jsonl`, `affect_from_convos.jsonl`, `ei_11.jsonl`,
  `emotions.jsonl`. Paths: `CLAUDE.local.md`.
- `build_ear.py` uses it; `--keep-unlabeled` reproduces the old build bit for bit.
- Add to the Ear's report a sign check on labeled rows with |valence| >= 0.5, held out by the existing text hash.

**B. A head on the frozen 450M (CPU feature pass; GPU not needed).**
- `cubbyllm/model/affect_head.py` (`__wiring__` STANDALONE): from a pooled stream (mean over the sentence's tokens, at a
  layer chosen on the validation split, final layer included) to valence and arousal (tanh / sigmoid) and nine
  classes (Plutchik's eight + neutral) with an intensity. Under 300 lines.
- `CB_AFFECT=1` builds it beside the trunk; it reads, never writes: logits are bit-identical with it on or off, and
  old checkpoints load. `meta["affect"]` absent = off.
- `validation/train_affect.py`: one pass of the frozen trunk over the sentences, features cached to disk, then fit the
  head on the cache (minutes on a CPU). Train: cleaned amygdala rows, `emotions.jsonl` (the 1,818 kept,
  `docs/emotions_corpus.md`), the Teams messages with labels (`docs/teams_affect_labels.md`). Split by text hash
  into train / validation / test; the layer and the head's size are chosen on validation only.

## Acceptance (pre-registered)
- Tests: the loader drops exactly the default rows (a fixture with both kinds); `--keep-unlabeled` matches the old
  build; `CB_AFFECT` off and on give identical logits; the head's gradient reaches its own weights only; the
  feature cache equals a fresh forward.
- **A, report:** rows kept per file; the Ear's held-out sign, pair order and seed probes before and after; the new
  |valence| >= 0.5 sign check before and after. Gate: no seed probe falls by more than 0.02 (the 4-point lesson in
  `build_ear.py`). Kill: if the cleaned build is worse on the strong-label check, stop and report: the labels are
  not what the triage says.
- **B, measured on human text only** (the gate never scores model-written lines): Warriner lemmas (valence, arousal,
  Spearman), the Teams test split (petal macro-F1, against the two labellers' agreement as the ceiling), and
  `affect_from_convos.jsonl` (valence Spearman; its labeller is unknown, so report it, gate nothing on it alone).
  - **Gate:** the head beats the cleaned Ear on valence Spearman by >= 0.05 on Warriner and on Teams, and is worse
    nowhere by > 0.02; arousal Spearman on Warriner is reported (the cube's arousal axis is the open question).
  - **Defects:** confident wrong-sign reads (|valence| > 0.5, opposite sign to the label) are counted separately from
    unsure reads. The head's count may not exceed the Ear's. With both readers, a read is confident only when they
    agree in sign; report that combined count too.
  - **Kill:** if the head does not beat the Ear on human text, it stays out and the Ear remains the reader (cheaper,
    inspectable, learns without retraining). Record the numbers either way.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- The trunk is frozen: no gradient reaches it (as `CB_TRAIN_ONLY=copy` in order 0005).
- No new labels from an external model in this order; the existing Teams labels are used as they are.
- Human text and Nick's own messages stay internal: no example text from `affect_from_convos.jsonl` or the Teams
  corpus goes into a committed file or log (counts and scores only).

## Out of scope
- Speaking with feeling: order 0010 (casual talk) can condition on the affect read later.
- Moving the cube's corners (the fear / distress swap waits for a second corpus, `docs/cube_vs_valence_arousal.md`).
- French and Quebec affect beyond what the Teams corpus holds (`docs/fr_affect_pilot.md`).
- Wiring either reader into the serve path.

## Addendum 1 (2026-10-22, Cowork): a human-labelled test
GoEmotions' train split (43,410 Reddit comments labelled by people, CC BY 4.0; path in `CLAUDE.local.md`) is the one
affect set we hold with human text and human labels. It was dropped as the petal source (`docs/emotions_corpus.md`), but
it tests valence sign with no mapping: the GoEmotions paper groups its 27 labels into positive, negative and ambiguous,
plus neutral.
- Sign test: rows whose labels all fall in positive, or all in negative. Neutral rows: report how many read with
  |valence| > 0.5 (not counted as wrong-sign). Ambiguous and mixed rows: skip.
- Report it in part A (the Ear before and after) and in part B (head vs Ear: sign accuracy, confident wrong-sign count).
  The gate stays on Warriner and Teams; on GoEmotions the head may not trail the Ear by more than 0.02.
- Test only: never train on it. Reddit text: no example text in a committed file or log.

## Addendum 2 (2026-10-22, Cowork): conversation sets labelled by people
Nick named three sets. Two carry human text with human labels; their text files were fetched with Nick's yes
(2026-10-22; paths in `CLAUDE.local.md`).
- **MELD** (Poria et al. 2019; github.com/declare-lab/MELD): 13,708 utterances in 1,432 dialogues (train 9,989 /
  dev 1,109 / test 2,610) from the TV series
  *Friends*, each labelled by people with one of 7 emotions (anger, disgust, sadness, joy, neutral, surprise, fear) and a
  sentiment (positive, negative, neutral). It is the in-context test this order is about: score each utterance with the
  turns before it. Sentiment gives a valence-sign test; six of the emotions map one-to-one to petals, plus neutral.
  - Report on its test split, Ear vs head. A second arm adds its train split to the head's training.
  - Rights: the repository is GPL-3.0; the dialogue is the show's script, so research use only, and no example text
    in any committed file or log.
- **ESConv** (Liu et al. 2021, ACL; github.com/thu-coai/Emotional-Support-Conversation): the released file holds 1,300
  conversations, 38,365 utterances (the paper describes 1,053), between crowdworkers in seeker and supporter roles, most
  about the seeker's real, current problems. Each carries the seeker's emotion type (8), problem type (6) and, for
  1,150 of them, a self-reported intensity, 1-5, before and after (mean 4.07, then 2.07). Test: the head's negative intensity on the seeker's opening turns against the before-score.
  CC BY-NC 4.0, "academic research use only".
- **ExTES** (Zheng et al. 2023, arXiv:2308.11584): 11,177 dialogues written by gpt-3.5-turbo from 87 human-built seeds
  and reviewed by people. Model-written, so it is no evidence of how people feel (the lesson in
  `validation/label_affect.py`); no license stated. Not used here.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
