# Does being talked to change the run?

20 seeds x 200 steps, three arms on the SAME mazes. `noise` is matched to `oracle` on message count per seed, so the two differ in WHEN things were said and in nothing else.

The arousal axis cannot be read from text (`docs/cube_vs_human_vad.md`), so it has to be read from behaviour. This is behaviour.

## Arms

| arm | deaths | level reached | score | messages | credibility | berth |
|---|---:|---:|---:|---:|---:|---:|
| silent | 2.20 | 1.35 | 38 | 0.0 | 0.50 | 2.45 |
| oracle | 1.75 | 1.35 | 41 | 102.8 | 0.99 | 2.20 |
| noise | 2.10 | 1.25 | 41 | 103.2 | 0.97 | 2.40 |

## Paired tests

Same maze, same ghosts, one difference. Paired permutation, 20,000 sign flips.

| comparison | metric | mean difference | p |
|---|---|---:|---:|
| oracle vs silent | deaths | -0.450 | 0.2618 |
| oracle vs silent | level | +0.000 | 1.0000 |
| oracle vs silent | score | +2.150 | 0.5402 |
| noise vs silent | deaths | -0.100 | 0.8830 |
| noise vs silent | level | -0.100 | 0.7276 |
| noise vs silent | score | +2.200 | 0.4649 |
| oracle vs noise | deaths | -0.350 | 0.3430 |
| oracle vs noise | level | +0.100 | 0.7565 |
| oracle vs noise | score | -0.050 | 1.0000 |

## Reading it

**Null.** Nobody talking to him changed the outcome. Given that `wariness` moves the berth by at most two tiles, the honest reading is that the berth is not what decides catches in this maze — which is a fact about the world, and the next thing to measure.

## Where he spent his time

| corner | silent | oracle | noise |
|---|---:|---:|---:|
| anger | 54.6% | 34.1% | 41.4% |
| distress | 34.0% | 62.8% | 54.3% |
| interest | 1.8% | 1.3% | 0.8% |
| neutral | 9.6% | 1.7% | 3.5% |

The occupancy histogram is the ablation the corner table has been waiting for: if talking to him reaches the body at all, he spends his time in different places.
