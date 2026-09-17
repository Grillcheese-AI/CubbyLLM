# Burst depth vs intensity

697 labelled rows, 495 non-neutral, 440 runs. No spend - re-reads the pilot's own output.

Testing a prediction the owner made before seeing the data: bursts run *"very intense for like 10 messages in a row"*. The first pass bucketed `burst>=2` against `burst==0`, which averages the 2nd message of a two-message run with the 10th of a ten-message run - if intensity climbs with depth, that bucket is the wrong shape.

## How deep do runs actually go?

- runs: **440**, median length **1**, longest **7**
- runs of single message: **295** (67.0%)
- runs of 2-4: **131** (29.8%)
- runs of 5-9: **14** (3.2%)
- runs of 10+: **0** (0.0%)

## 1. Intensity by burst depth

| depth | n | mean intensity | median |
|---|---:|---:|---:|
| 0 | 325 | 0.339 | 0.350 |
| 1 | 86 | 0.335 | 0.300 |
| 2-4 | 71 | 0.351 | 0.300 |
| 5-9 | 10 | 0.382 | 0.335 |
| 10+ | 3 | 0.700 | 0.700 |

depth>=5 vs depth==0: **+0.117**, p = **0.0006**

Spearman rho (intensity vs depth) = **+0.046**

## 2. Intensity by RUN LENGTH

A message 3rd of 10 is not a message 3rd of 3. This asks whether long runs are hot throughout, rather than whether people wind up.

| run length | n | mean intensity |
|---|---:|---:|
| 1 | 231 | 0.340 |
| 2-4 | 213 | 0.342 |
| 5-9 | 51 | 0.359 |

run>=5 vs run==1: **+0.019**, p = **0.2795**

Spearman rho (intensity vs run length) = **+0.009**

## 3. Emoji by depth - social marker or arousal marker?

| depth | n | % with emoji |
|---|---:|---:|
| 0 | 440 | 10.5% |
| 1 | 136 | 7.4% |
| 2-4 | 102 | 2.9% |
| 5-9 | 16 | 0.0% |
| 10+ | 3 | 0.0% |

If emoji thin out as runs deepen, they are punctuation for the social beats of a conversation - greetings, thanks, softening - and not a measure of how worked up anyone is. Nobody stops mid-rant to add a smiley.

## 4. The deepest runs, as text


**run of 7** ([SELF])

- `b0` `neutral/0.05` les barcode?
- `b1` `anticipation/0.2` les barcode et sku devrais etre cherchable
- `b2` `anger/0.25` mon casse tete presentement aussi niaiseu ca puisse paraitre c le menu lol
- `b3` `anticipation/0.3` ma checker une coupe de sites pour minspirer
- `b4` `anticipation/0.35` on va pouvoir creer les collections automatique genre tout ce qui a un type = casse-tete va la
- `b5` `neutral/0.05` les categ de prextra sont dans product type
- `b6` `trust/0.25` fac pour les collection ca devrais pas etre si pire a creer

**run of 6** ([SELF])

- `b0` `neutral/0.05` fac le magasin va tomber a 0
- `b1` `neutral/0.05` a lupdate ca va le mettre
- `b2` `neutral/0.05` pis quand y va etre a 1
- `b3` `neutral/0.05` ca va le mettre aussi
- `b4` `anticipation/0.2` je pense pas quon doivent suivre le full process niveau web
- `b5` `neutral/0.05` juste quand le magasin la

**run of 6** ([SELF])

- `b0` `joy/0.35` tout va bien dans linventaire? 😉
- `b1` `anticipation/0.35` fac main qui en aille qui rentre y vont se mettre tout seul
- `b2` `anticipation/0.25` sinon ce qui arrive c quon a 50,000+ produit pis c dur a dire ceux qui sont pas bon
- `b3` `trust/0.3` c la maniere jai trouver pour les differencier
- `b4` `neutral/0.15` ceux a 0 = jai prends pas donc ca donne a peu pres 14000 produit
- `b5` `trust/0.3` si y a une autre maniere tu me le dira pis ma prendre les qte a 0 aussi
