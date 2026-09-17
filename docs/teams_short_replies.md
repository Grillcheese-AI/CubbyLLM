# Short replies: neutral, or depleted?

440 `[SELF]` messages labelled - **158** of 1-5 words, 282 longer. No spend.

Testing a claim the owner made before seeing the data: a 1-5 word reply means *sick of it, busy, or tired*. The pipeline had been treating those as low-signal.

## 1. What does the labeller call them?

| petal | short (1-5w) | longer |
|---|---:|---:|
| neutral | 70 (44.3%) | 67 (23.8%) |
| joy | 38 (24.1%) | 64 (22.7%) |
| trust | 25 (15.8%) | 38 (13.5%) |
| anticipation | 14 (8.9%) | 66 (23.4%) |
| fear | 5 (3.2%) | 15 (5.3%) |
| surprise | 5 (3.2%) | 16 (5.7%) |
| sadness | 1 (0.6%) | 6 (2.1%) |
| anger | 0 (0.0%) | 7 (2.5%) |
| disgust | 0 (0.0%) | 3 (1.1%) |

- negative petals: short **3.8%** vs longer **11.0%**
- neutral: short **44.3%** vs longer **23.8%**

If short replies come back overwhelmingly `neutral` while the person who wrote them says they mean *sick of it*, the labeller is mapping a real state onto the absence of one - and `neutral` is where the depletion corner's training data went.

## 2. Held-out: hour of day

Never in the prompt. Tiredness has a shape; simple brevity does not.

| hour (UTC) | short | longer | short share |
|---|---:|---:|---:|
| 04 | 0 | 1 | 0% |
| 05 | 0 | 3 | 0% |
| 12 | 1 | 4 | 20% |
| 13 | 4 | 5 | 44% |
| 14 | 6 | 3 | 67% |
| 15 | 52 | 81 | 39% |
| 16 | 30 | 63 | 32% |
| 17 | 30 | 57 | 34% |
| 18 | 28 | 53 | 35% |
| 19 | 2 | 9 | 18% |
| 20 | 3 | 3 | 50% |
| 21 | 2 | 0 | 100% |

## 3. Held-out: silence before the reply

*Busy* should look like a long gap followed by a curt answer.

- before a short reply: median gap **11s** (n=158)
- before a longer one:  median gap **20s** (n=282)
- means capped at 1h: **172s** vs **127s**, p = **0.4391**

## 4. The short replies themselves

Read these against the labels. The question is not whether the label is defensible in isolation - it is whether it is what the writer meant.

- `trust/0.2` +16.3s b0: oki
- `neutral/0.05` +4.5s b0: les 2
- `neutral/0.05` +23.1s b1: c lier a meme placew
- `neutral/0.1` +8.3s b1: dans marques sont toute la
- `trust/0.2` +11.4s b0: oui
- `neutral/0.1` +3.1s b2: fo les faire la
- `trust/0.2` +6.6s b0: oui
- `neutral/0.1` +2.7s b1: ou page
- `neutral/0.0` +2.2s b0: associee a la categ parent
- `trust/0.2` +6.5s b0: okidoo
- `neutral/0.0` +2.6s b1: sur google
- `anticipation/0.25` +17.5s b0: ok quel email
- `neutral/0.0` +12.9s b0: dac
- `neutral/0.15` +4.6s b1: a moins jaile mal compris
- `joy/0.2` +18.2s b0: ouais lol
- `joy/0.25` +33.6s b1: c fait
- `joy/0.28` +14.8s b2: jai mon compte
- `surprise/0.35` +57.1s b0: ben bizarre lol
- `neutral/0.1` +8.1s b1: chu dans inventaire items
- `neutral/0.1` +4.2s b2: je fait f2
- `neutral/0.15` +13.1s b0: ouais je sais
- `neutral/0.1` +5.4s b1: ca c dans litem
- `surprise/0.2` +12.3s b0: ah ok
- `neutral/0.1` +2.9s b1: tk pas grave
- `surprise/0.4` +12.4s b0: biz ..
- `anticipation/0.35` +15.8s b3: ceux qui sont la ?
- `joy/0.45` +1.5s b0: fiou
- `neutral/0.05` +2.6s b1: ok
- `joy/0.3` +16.2s b0: hahaha ok
- `sadness/0.35` +0.2s b0: aaah
- `joy/0.75` +9.9s b0: yes enfin
- `neutral/0.0` +8.9s b0: dac
- `neutral/0.0` +11.3s b0: c camion 2
- `neutral/0.1` +15.6s b1: ou p-e juste pas utiliser
- `neutral/0.1` +16.7s b0: ca doit etre pareil
- `neutral/0.05` +3.4s b3: ca va le mettre aussi
- `neutral/0.05` +4.1s b5: juste quand le magasin la
- `trust/0.15` +15.8s b0: ok
- `neutral/0.1` +9.2s b0: non ma faire ca tanto
- `trust/0.45` +16.1s b0: merci jf 🙂
