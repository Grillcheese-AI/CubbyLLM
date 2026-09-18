# Does being talked to change the run?

30 seeds x 200 steps, three arms on the SAME mazes. `noise` is matched to `oracle` on message count per seed, so the two differ in WHEN things were said and in nothing else.

The arousal axis cannot be read from text (`docs/cube_vs_human_vad.md`), so it has to be read from behaviour. This is behaviour.

## Arms

| arm | deaths | level reached | score | messages | credibility | berth |
|---|---:|---:|---:|---:|---:|---:|
| silent | 2.13 | 1.37 | 41 | 0.0 | 0.50 | 2.57 |
| oracle | 1.93 | 1.43 | 39 | 108.1 | 0.52 | 2.30 |
| noise | 2.37 | 1.20 | 39 | 108.9 | 0.39 | 2.83 |
| sparse | 2.27 | 1.37 | 41 | 24.5 | 0.52 | 2.60 |

## Paired tests

Same maze, same ghosts, one difference. Paired permutation, 20,000 sign flips.

| comparison | metric | mean difference | p |
|---|---|---:|---:|
| oracle vs silent | deaths | -0.200 | 0.5682 |
| oracle vs silent | level | +0.067 | 0.7940 |
| oracle vs silent | score | -1.733 | 0.5538 |
| sparse vs silent | deaths | +0.133 | 0.7769 |
| sparse vs silent | level | +0.000 | 1.0000 |
| sparse vs silent | score | -0.133 | 0.9699 |
| noise vs silent | deaths | +0.233 | 0.4942 |
| noise vs silent | level | -0.167 | 0.2301 |
| noise vs silent | score | -1.433 | 0.5512 |
| sparse vs oracle | deaths | +0.333 | 0.3405 |
| sparse vs oracle | level | -0.067 | 0.7778 |
| sparse vs oracle | score | +1.600 | 0.5429 |
| oracle vs noise | deaths | -0.433 | 0.2121 |
| oracle vs noise | level | +0.233 | 0.0946 |
| oracle vs noise | score | -0.300 | 0.9167 |

## Where he spent his time

| corner | silent | oracle | noise | sparse |
|---|---:|---:|---:|---:|
| anger | 51.1% | 43.3% | 52.6% | 49.1% |
| distress | 36.1% | 53.3% | 43.2% | 43.6% |
| interest | 1.9% | 1.2% | 0.7% | 1.7% |
| neutral | 10.9% | 2.2% | 3.5% | 5.6% |

Shift away from silence, as total variation: **oracle 17.2 points**, **noise 8.6 points**, **sparse 7.5 points**.

## Reading it

**On the body:** talking to him moves **17.2 points** of his time between corners at most (`oracle`). The affect channel reaches him; that is not in question. Whether the shift is GOOD is a separate question the table answers on its own — a move from `anger` toward `distress` is dopamine being suppressed by threat, which is a body being worn down rather than helped.

**On behaviour:**

- oracle vs silent: **-0.20** deaths, p = 0.568
- sparse vs silent: **+0.13** deaths, p = 0.777
- noise vs silent: **+0.23** deaths, p = 0.494
- sparse vs oracle: **+0.33** deaths, p = 0.340
- oracle vs noise: **-0.43** deaths, p = 0.212

**Nothing separates at n=30.** The directions may be consistent and still mean nothing at this sample size; an effect of this magnitude needs several times the seeds before the p-values are worth reading. Underpowered is not the same as null, and neither word should be used for the other.
