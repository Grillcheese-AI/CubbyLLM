# The cube against human ratings

Warriner, Kuperman & Brysbaert (2013): valence and arousal for **13,905 English lemmas**, crowd-rated on 1-9 scales. NRC EmoLex v0.92: **4,463 words** tagged with Plutchik's eight by a separate crowd. Published 2013 and 2010, no language model involved in either.

Of the EmoLex vocabulary, **1,647 words carry exactly one petal** and survive to a corner; 2,119 carry more than one and are dropped rather than assigned arbitrarily, because a word tagged both fear and sadness is precisely the case under test.

The first external check found the valence axis strong (+0.782) and the arousal axis absent (-0.004), with fear and distress apparently transposed on noradrenaline. Swapping them lifted arousal to +0.464 - but that hypothesis came from the data it was measured on.

## Results

### NRC EmoLex x Warriner - 1137 words

The large one. Every single-petal EmoLex word that Warriner also rates.

| | as published | fear<->distress |
|---|---:|---:|
| arousal ~ NE | **-0.101** | **+0.184** |
| valence ~ (5HT+DA)/2 | **+0.538** | +0.442 |

### Plutchik tier words - 20 words

The cube's own naming: `apprehension/fear/terror`, `pensiveness/sadness/grief`, straight out of `_PETAL`. No mapping chosen here - these words already name these corners.

| | as published | fear<->distress |
|---|---:|---:|
| arousal ~ NE | **+0.026** | **+0.354** |
| valence ~ (5HT+DA)/2 | **+0.746** | +0.549 |

### GoEmotions labels - 23 words

The label set mapped through `GOEMOTIONS_PETAL`.

| | as published | fear<->distress |
|---|---:|---:|
| arousal ~ NE | **-0.106** | **+0.165** |
| valence ~ (5HT+DA)/2 | **+0.800** | +0.728 |

## Per corner, human-rated (NRC vocabulary)

| corner | (5HT,DA,NE) | words | valence 1-9 | arousal 1-9 |
|---|---|---:|---:|---:|
| anger | (0, 1, 1) | 158 | 3.87 | 4.86 |
| fear | (0, 1, 0) | 222 | 4.03 | 4.85 |
| surprise | (1, 0, 1) | 77 | 5.14 | 4.61 |
| joy | (1, 1, 0) | 93 | 6.93 | 4.61 |
| disgust | (1, 0, 0) | 220 | 3.60 | 4.40 |
| interest | (1, 1, 1) | 167 | 5.45 | 4.31 |
| distress | (0, 0, 1) | 200 | 3.65 | 4.16 |

**How much is there to find.** Corner means span **3.32** points of valence and **0.70** points of arousal, on the same 1-9 scale. Whatever the labelling, the second axis carries 21% of the signal the first one does: emotion WORDS barely encode arousal.


## The ceiling: every possible NE labelling

126 non-degenerate ways to mark 7 corners high or low on noradrenaline (all-high and all-low are not axes), scored against human arousal. This asks whether the AXIS can work at all, not whether one corner is misplaced.

| labelling | high-NE corners | rho | rank |
|---|---|---:|---:|
| **best possible** | anger, fear, joy, surprise | **+0.282** | 1 |
| Lövheim as published | anger, distress, interest, surprise | -0.101 | 96 of 126 |
| fear<->distress swapped | anger, fear, interest, surprise | +0.184 | 13 of 126 |

## The specific question

- **fear** words (222): mean human arousal **4.85** - Lövheim places this corner at NE=0
- **distress** words (200): mean human arousal **4.16** - Lövheim places this corner at NE=1

Difference **+0.70** on a 1-9 scale, permutation p = **0.0000** over 20,000 shuffles. Fear rates HIGHER, as the swap predicts.

## What this settles, and what it does not

**Settled.** Fear outranks distress on human arousal in all three vocabularies here and in the model-labelled corpus before them - four independent samples, same direction, p < 0.001 on the large one. Lövheim's published NE assignment ranks **96 of 126** against human arousal; the swap ranks **13**. The swap is not a rescue, it is a smaller error.

**Not settled, and more important.** The BEST labelling any binary NE axis can manage is +0.282, against +0.538 for valence off the other two axes. The corners span 0.70 points of arousal and 3.32 of valence. The third axis is not mislabelled so much as nearly empty in this instrument - emotion words carry valence and barely carry arousal.

Which is the honest reading: **language is the wrong place to measure the arousal axis.** Valence survives being written down; activation is in the body and in the timing - burst depth, reply latency, hour of day, the channels the writer does not choose. The cube's NE axis should be validated against BEHAVIOUR, not against text, and until it is, no text corpus can either convict or acquit it.

`_CORNERS` is therefore left as published. The swap wins on a dimension this evidence cannot measure well enough to justify moving a corner.

Not in the lexicon: pensiveness
