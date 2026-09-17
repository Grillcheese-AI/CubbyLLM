# Does labelled intensity track arousal?

697 labelled rows, 495 non-neutral. No new spend - this re-reads the pilot's own output. Permutation test, 20,000 shuffles, two-sided.

The labeller never saw `gap_s` or `burst`. Every comparison below is therefore a check against a channel it could not have copied.

## 1. The original check, reproduced

### all non-neutral messages

| group | n | mean | median |
|---|---:|---:|---:|
| within 30s | 299 | 0.342 | 0.350 |
| after 30s | 195 | 0.344 | 0.300 |

difference **-0.002**, permutation p = **0.8253** -> flat

## 2. Short acknowledgements removed

The hypothesis for the flat result: fast replies here are mostly `ok` / `oui` / `yep`, genuinely low-intensity and numerous enough to swamp the signal.

### messages longer than 3 words

| group | n | mean | median |
|---|---:|---:|---:|
| within 30s | 212 | 0.347 | 0.350 |
| after 30s | 166 | 0.350 | 0.350 |

difference **-0.003**, permutation p = **0.7983** -> flat

### messages longer than 5 words

| group | n | mean | median |
|---|---:|---:|---:|
| within 30s | 172 | 0.342 | 0.350 |
| after 30s | 144 | 0.356 | 0.350 |

difference **-0.014**, permutation p = **0.2494** -> flat

### messages longer than 10 words

| group | n | mean | median |
|---|---:|---:|---:|
| within 30s | 80 | 0.345 | 0.350 |
| after 30s | 86 | 0.360 | 0.350 |

difference **-0.015**, permutation p = **0.3989** -> flat

### bare acknowledgements removed by wordlist

| group | n | mean | median |
|---|---:|---:|---:|
| within 30s | 283 | 0.348 | 0.350 |
| after 30s | 192 | 0.343 | 0.300 |

difference **+0.005**, permutation p = **0.6459** -> flat

## 3. Other arousal proxies the labeller never saw

### deep in a burst vs first of a run

| group | n | mean | median |
|---|---:|---:|---:|
| burst >= 2 | 84 | 0.367 | 0.350 |
| burst == 0 | 325 | 0.339 | 0.350 |

difference **+0.029**, permutation p = **0.0356** -> **separates**

### very fast (<=10s) vs long pause (>5 min)

| group | n | mean | median |
|---|---:|---:|---:|
| <=10s | 106 | 0.352 | 0.350 |
| >300s | 51 | 0.344 | 0.350 |

difference **+0.008**, permutation p = **0.7246** -> flat

### drew a reaction vs did not

| group | n | mean | median |
|---|---:|---:|---:|
| reacted to | 27 | 0.374 | 0.350 |
| no reaction | 468 | 0.341 | 0.300 |

difference **+0.033**, permutation p = **0.1336** -> flat

## 4. What the petals do, not just the intensity

Intensity might be decoration while the PETAL still tracks something. Share of high-arousal petals (anger, fear, surprise, anticipation) by timing:

- within 30s: **36.8%** high-arousal (110/299)
- after 30s: **45.1%** high-arousal (88/195)

## Reading this

If section 2 separates and section 1 does not, timing works as a check and short acknowledgements were the noise - the full run can proceed.

If everything is flat, the intensity number is not carrying arousal. That is a real finding and it costs one pilot rather than a full run: intensity would then need grounding in something measured rather than asserted, and the corpus already holds candidates the labeller cannot see.
