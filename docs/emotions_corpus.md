# emotions.jsonl as the emotion family's source

1830 rows in, **1818 kept**, 12 rejected.

Replaces GoEmotions, which had to be MAPPED into the taxonomy — 8 petals x 3 tiers = 24 points for 28 labels, so it could not separate them by construction — and whose French half was a machine translation. This corpus arrives in the target taxonomy with continuous intensity, so there is nothing to quantise and the collision does not exist.

## Petals

| petal | n | share | corner |
|---|---:|---:|---|
| joy | 379 | 20.8% | joy |
| sadness | 303 | 16.7% | distress |
| fear | 280 | 15.4% | fear |
| trust | 272 | 15.0% | **none (oxytocin)** |
| surprise | 174 | 9.6% | surprise |
| anger | 174 | 9.6% | anger |
| anticipation | 138 | 7.6% | interest |
| disgust | 98 | 5.4% | disgust |

## Intensity is continuous

0.2: 1, 0.3: 3, 0.4: 21, 0.5: 124, 0.6: 294, 0.7: 430, 0.8: 589, 0.9: 345, 1.0: 11

9 distinct values, not three tiers. That single fact is what dissolves the 24-points-for-28-labels problem rather than solving it.

## The `secondary` field was doing two jobs

Sometimes a second petal (`joy`, `anger`), sometimes the compound's own name (`remorse`, `optimism`, `awe`). Both resolve to a petal via Plutchik's closed dyad table.

- none: 1585
- petal: 29
- dyad-is-primary: 13
- dyad:gratitude: 12
- dyad:optimism: 9
- dyad:anxiety: 9
- dyad:contempt: 7
- dyad:hope: 7
- dyad:excitement: 7
- dyad:outrage: 6
- dyad:remorse: 6
- dyad:disappointment: 5
- dyad:awe: 5
- dyad:regret: 5

## Dyad adjacency check

A dyad midpoint is only a representable point when its two corners differ in exactly ONE axis. Between corners differing in two or three it sits equidistant from four or more and the classifier reads it as something else. Non-adjacent pairs keep the primary and drop the secondary rather than being thrown away.

- non-adjacent — dropped to primary only: 69
- adjacent — kept as a dyad: 33
- not checkable (trust has no corner): 22

## Rejected

- multi-primary: 11
- primary not a petal: 1

Reported rather than silently dropped — a corpus that quietly discards what it cannot represent is how you end up believing a mapping is cleaner than it is.
