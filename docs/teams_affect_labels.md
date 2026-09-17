# Affect labels over the real Quebec work corpus

Labeller `google/gemini-3.8-flash`, blind second opinion `deepseek/deepseek-v3.2` on every 8th window. 40 windows of 20, **697 messages labelled**, 73,447 tokens.

Labels are a model's reading, not ground truth. What makes them worth anything is that two independent checks exist: a geometric one that rejects dyads the cube cannot represent, and a timing one the labeller never saw.

## Petal distribution

| petal | n | share |
|---|---:|---:|
| neutral | 202 | 29.0% |
| joy | 164 | 23.5% |
| anticipation | 141 | 20.2% |
| trust | 120 | 17.2% |
| surprise | 28 | 4.0% |
| fear | 22 | 3.2% |
| sadness | 10 | 1.4% |
| anger | 7 | 1.0% |
| disgust | 3 | 0.4% |

## Rejected by the verifier

- missing x20
- non-adjacent-dyad:fear+anticipation x15
- non-adjacent-dyad:anticipation+fear x10
- non-adjacent-dyad:surprise+joy x9
- non-adjacent-dyad:joy+surprise x9
- non-adjacent-dyad:fear+surprise x6
- non-adjacent-dyad:surprise+fear x5
- non-adjacent-dyad:disgust+anger x4
- non-adjacent-dyad:anger+joy x3
- non-adjacent-dyad:sadness+joy x3
- non-adjacent-dyad:anger+disgust x3
- non-adjacent-dyad:joy+sadness x2
- non-adjacent-dyad:sadness+fear x2
- non-adjacent-dyad:anger+surprise x2
- non-adjacent-dyad:joy+anger x2
- non-adjacent-dyad:sadness+disgust x1
- non-adjacent-dyad:disgust+sadness x1
- bad-secondary:neutral x1
- non-adjacent-dyad:fear+sadness x1
- non-adjacent-dyad:fear+disgust x1
- non-adjacent-dyad:disgust+fear x1
- non-adjacent-dyad:anticipation+sadness x1
- non-adjacent-dyad:surprise+anger x1

## Second-model agreement

On the 90 messages both models labelled, they chose the same petal **64** times (**71%**).

## The held-out check: timing

`gap_s` was never in the prompt. If the labels are reading real affect rather than surface features, non-neutral intensity should run higher on messages fired off within 30s of the last one than on messages sent after a long pause.

- within 30s: mean intensity **0.342** (n=299)
- after 30s:  mean intensity **0.344** (n=195)

A gap of roughly nothing means the labels are not tracking arousal, and the intensity axis is decorative. That would be a real negative result and it is worth more than a plausible-looking table.
