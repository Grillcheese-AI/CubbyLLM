# Compression as an energy readout

9738 messages, 6758 from `[SELF]`, **3597** of which presented at least one full-vs-short choice. No model, no labels - a ratio over which spelling the writer picked.

The premise: writing the full form costs energy. `il va falloir que je mette` and `fo je mette` are the same sentence; which one comes out is the budget, not the dialect.

## Is the choice actually live?

If the short form is used 100% of the time it is not a choice, it is just how this person writes, and it can carry no state.

| pair | full used | short used |
|---|---:|---:|
| `il (?:va )?falloir que` | 0 | 220 |
| `parce qu[e']` | 0 | 307 |
| `peut-[eê]tre` | 2 | 93 |
| `\bje suis\b` | 30 | 89 |
| `\bje vais\b` | 31 | 342 |
| `en tout cas` | 0 | 126 |
| `\bmaintenant\b` | 4 | 35 |
| `\bet puis\b|\bensuite\b` | 1 | 556 |
| `\bc'est\b` | 10 | 1607 |
| `\bje pense\b` | 42 | 85 |
| `\bj'ai\b` | 0 | 531 |
| `\bil y a\b` | 1 | 245 |
| `\bd'accord\b` | 0 | 286 |
| `\bquelque chose\b` | 0 | 102 |
| `\bquelqu'un\b` | 0 | 57 |
| `\bbeaucoup\b` | 0 | 150 |
| `\bs'il te pla[iî]t\b` | 0 | 2 |
| `\bje te\b` | 1 | 37 |
| `\bje le\b` | 4 | 45 |
| `\btu es\b` | 0 | 89 |

## Compression by burst depth

The confound, head on: depletion and arousal BOTH compress. What should separate them is volume - tired compresses and stops, wound-up compresses and keeps going.

### by depth

| group | n | mean lexical compression | median words |
|---|---:|---:|---:|
| 0 (opens a run) | 1517 | 0.975 | 12 |
| 1 | 783 | 0.975 | 10 |
| 2-4 | 891 | 0.978 | 11 |
| 5+ | 406 | 0.981 | 13 |

## Compression by message length

### by words

| group | n | mean lexical compression | median words |
|---|---:|---:|---:|
| 1-5 words | 582 | 0.982 | 4 |
| 6-12 | 1454 | 0.975 | 9 |
| 13+ | 1561 | 0.976 | 18 |

## Who compresses - state or habit?

If compression is a per-person style rather than a per-message state, the speakers will differ far more than the situations do.

- **[SELF]** - lexical 0.977 (n=3597), accents dropped 0.954 (n=1238)
- **[PEER]** - lexical 0.654 (n=1006), accents dropped 0.067 (n=947)
- **[PEER2]** - lexical 0.429 (n=59), accents dropped 0.006 (n=59)

## Spread

- mean **0.977**, median **1.000**, sd **0.136**
- distribution: 0.0: 56, 0.3: 1, 0.5: 40, 0.7: 19, 0.8: 6, 1.0: 3475

A scale pinned at 1.0 would mean the short form always wins and there is no signal - the opposite failure from the model's intensity, which sat at 0.35 in every group.

## Restricted to the pairs that actually vary

6 of 20 pairs have the full form used 4+ times by this writer. The rest are settled - a choice made the same way every time carries nothing, and averaging it in buries the pairs that do move.

n = 2136, mean **0.954**, sd **0.200**

| group | n | live-pair compression | accents dropped |
|---|---:|---:|---:|
| burst 0 | 862 | 0.945 | 0.962 |
| burst 1-4 | 1018 | 0.958 | 0.960 |
| burst 5+ | 256 | 0.964 | 0.916 |
| 1-5 words | 289 | 0.964 | 0.992 |
| 6-12 words | 889 | 0.954 | 0.966 |
| 13+ words | 958 | 0.951 | 0.940 |
