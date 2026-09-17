# Is the cube geometry, or vocabulary?

**1106 texts**, each carrying a valence/arousal pair that already existed in `amygdala_affect.jsonl` and a Plutchik petal assigned BLIND by `label_va_petals.py` - the labeller never saw the affect values, and the prompt contains no word for either dimension.

A second model labelled a sample; 21 of those rows carry a disagreement, recorded per row rather than averaged away.

Every previous check here was internal: derived coordinates run back through the agent's own classifier, which can only prove self-consistency, because the classifier and the coordinate share one geometry. This is the first check from outside.

## The two predictions

| prediction | Spearman rho | verdict |
|---|---:|---|
| arousal tracks the **NE** coordinate | **-0.004** | fails |
| valence tracks **(5-HT + DA)/2** | **+0.782** | **holds** |

Cross-checks, which should be WEAKER if the axes mean what they say:

- NE vs valence: -0.215
- (5-HT+DA)/2 vs arousal: +0.052
- 5-HT alone vs valence: +0.819
- DA alone vs valence: +0.324

## Per corner: predicted vs measured

| corner | (5HT,DA,NE) | n | corpus valence | corpus arousal |
|---|---|---:|---:|---:|
| fear | (0, 1, 0) | 191 | -0.533 | 0.754 |
| interest | (1, 1, 1) | 191 | +0.680 | 0.670 |
| anger | (0, 1, 1) | 116 | -0.516 | 0.670 |
| surprise | (1, 0, 1) | 54 | +0.348 | 0.615 |
| joy | (1, 1, 0) | 337 | +0.804 | 0.528 |
| distress | (0, 0, 1) | 177 | -0.393 | 0.468 |
| disgust | (1, 0, 0) | 40 | -0.328 | 0.357 |

## The parked question: where does fear actually sit?

Lövheim places fear/terror at LOW noradrenaline, which is why a ghost catch now reads as distress and why two tests are widened pending a decision.

- corners Lövheim marks HIGH NE: mean corpus arousal **0.598** (n=538)
- corners Lövheim marks LOW NE:  mean corpus arousal **0.592** (n=568)


## If fear and distress swap places

Sorted by MEASURED arousal, fear sits top at 0.75 with Lövheim's NE=0, and distress sits near the bottom at 0.47 with NE=1. They look transposed on the noradrenaline axis. Swapping the two corners and recomputing, against the same data:

| | as published | fear<->distress |
|---|---:|---:|
| arousal ~ NE | -0.004 | **+0.464** |
| valence ~ (5HT+DA)/2 | +0.782 | +0.799 |

This is a hypothesis generated from the data it is measured on, so it is a lead rather than a result - it needs a second corpus before it changes `_CORNERS`. What it does settle is the parked question: the threat->DA coupling is not the thing to change, because the arousal axis does not line up with this data whatever the ODE does.

**`fear` itself** (191 texts): mean corpus arousal **0.754**, against a corpus-wide mean of 0.595.

If fear reads HIGH-arousal here, Lövheim's low-NE placement does not describe this data, and the threat->DA coupling is not the thing to change. If it reads LOW, the old test was named after the old wrong table and the current behaviour is correct.
