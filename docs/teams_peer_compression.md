# Compression vs state, in the writer with headroom

The owner sits at 0.977 lexical compression with no room left to move. `[PEER]` sits at 0.654 and keeps his accents. If writing the full form costs energy, it should show in the person who can still afford it.

## [PEER]

2871 messages, 12 of 20 pairs live for this writer, 963 messages presenting a choice.

### lexical compression — mean 0.633, sd 0.456, distinct 6

**by burst depth**

| group | n | mean | median |
|---|---:|---:|---:|
| opens a run (0) | 847 | 0.646 | 1.000 |
| 1 | 103 | 0.568 | 1.000 |
| 2+ | 13 | 0.308 | 0.000 |

first vs last: **-0.338**, p = **0.0076** -> **separates**

**by hour, local**

| group | n | mean | median |
|---|---:|---:|---:|
| before 12:00 | 541 | 0.624 | 1.000 |
| 12:00-15:00 | 288 | 0.651 | 1.000 |
| after 15:00 | 134 | 0.630 | 1.000 |

first vs last: **+0.006**, p = **0.8943** -> flat

**by message length**

| group | n | mean | median |
|---|---:|---:|---:|
| 1-5 words | 344 | 0.773 | 1.000 |
| 6-12 | 338 | 0.553 | 1.000 |
| 13+ | 281 | 0.558 | 0.500 |

first vs last: **-0.215**, p = **0.0000** -> **separates**

### accents dropped — mean 0.067, sd 0.236, distinct 5

**by burst depth**

| group | n | mean | median |
|---|---:|---:|---:|
| opens a run (0) | 820 | 0.064 | 0.000 |
| 1 | 109 | 0.069 | 0.000 |
| 2+ | 15 | 0.222 | 0.000 |

first vs last: **+0.158**, p = **0.0192** -> **separates**

**by hour, local**

| group | n | mean | median |
|---|---:|---:|---:|
| before 12:00 | 510 | 0.071 | 0.000 |
| 12:00-15:00 | 301 | 0.074 | 0.000 |
| after 15:00 | 133 | 0.039 | 0.000 |

first vs last: **-0.032**, p = **0.1530** -> flat

**by message length**

| group | n | mean | median |
|---|---:|---:|---:|
| 1-5 words | 213 | 0.103 | 0.000 |
| 6-12 | 389 | 0.070 | 0.000 |
| 13+ | 342 | 0.042 | 0.000 |

first vs last: **-0.061**, p = **0.0026** -> **separates**

## [SELF]

6758 messages, 6 of 20 pairs live for this writer, 2136 messages presenting a choice.

### lexical compression — mean 0.954, sd 0.200, distinct 5

**by burst depth**

| group | n | mean | median |
|---|---:|---:|---:|
| opens a run (0) | 862 | 0.945 | 1.000 |
| 1 | 485 | 0.960 | 1.000 |
| 2+ | 789 | 0.959 | 1.000 |

first vs last: **+0.014**, p = **0.1712** -> flat

**by hour, local**

| group | n | mean | median |
|---|---:|---:|---:|
| before 12:00 | 1176 | 0.967 | 1.000 |
| 12:00-15:00 | 643 | 0.933 | 1.000 |
| after 15:00 | 317 | 0.947 | 1.000 |

first vs last: **-0.019**, p = **0.0980** -> flat

**by message length**

| group | n | mean | median |
|---|---:|---:|---:|
| 1-5 words | 289 | 0.964 | 1.000 |
| 6-12 | 889 | 0.954 | 1.000 |
| 13+ | 958 | 0.951 | 1.000 |

first vs last: **-0.013**, p = **0.3331** -> flat

### accents dropped — mean 0.956, sd 0.196, distinct 5

**by burst depth**

| group | n | mean | median |
|---|---:|---:|---:|
| opens a run (0) | 464 | 0.962 | 1.000 |
| 1 | 269 | 0.957 | 1.000 |
| 2+ | 503 | 0.949 | 1.000 |

first vs last: **-0.013**, p = **0.2929** -> flat

**by hour, local**

| group | n | mean | median |
|---|---:|---:|---:|
| before 12:00 | 652 | 0.962 | 1.000 |
| 12:00-15:00 | 370 | 0.963 | 1.000 |
| after 15:00 | 214 | 0.926 | 1.000 |

first vs last: **-0.035**, p = **0.0264** -> **separates**

**by message length**

| group | n | mean | median |
|---|---:|---:|---:|
| 1-5 words | 119 | 0.992 | 1.000 |
| 6-12 | 450 | 0.966 | 1.000 |
| 13+ | 667 | 0.943 | 1.000 |

first vs last: **-0.049**, p = **0.0192** -> **separates**

## Reading this

A separation in `[PEER]` and a flat line in `[SELF]` supports the mechanism and explains the owner's null as a ceiling effect - the budget spent long ago, the instrument with nowhere left to read.

Flat in both is the stronger result and the less convenient one: it says compression is a per-person habit rather than a per-message state, which is already what the three-way split across writers (0.977 / 0.654 / 0.429) suggested on its own.
