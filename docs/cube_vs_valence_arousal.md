# The external check is not available from these two files

Joined rows with BOTH a Plutchik petal and a real valence/arousal pair: **0**.

The overlap looked total - every one of the 1,726 unique `emotions.jsonl` texts appears in `amygdala_affect.jsonl`. It is the wrong 1,726.

`amygdala_affect.jsonl` is two files stacked. About 1,335 rows carry real judgements; 1,845 carry valence exactly `0.0` and arousal exactly `0.5`, which is a default rather than a reading. The placeholder half is precisely the half that overlaps `emotions.jsonl` - those texts were appended without affect labels and defaulted to neutral.

So the two halves are disjoint in exactly the way that matters:

- real valence/arousal + `realm`/`phase`, NO petal (~1,335 rows, the same texts as `emotion_valence_arousal_realm_phase.jsonl`, which shares 0 texts with `emotions.jsonl`)
- petal + intensity, NO real valence/arousal (~1,726 rows)

No text has both, so nothing here can test whether the cube predicts an affect space labelled independently of it.

## What would work

Label the ~1,260 texts that DO have real valence/arousal with a petal. Their affect values already exist and were not produced by the petal labeller, so agreement would still be evidence rather than an echo - weaker than a pre-existing double-labelling, stronger than anything internal. It is also cheap: one pass over 1,260 short texts.

The first run of this script reported 1,546 joined rows and `rho = nan`, because every measured value in the join was the same number. A test that cannot fail is not a test, and a join that succeeds on placeholders is not a join.
