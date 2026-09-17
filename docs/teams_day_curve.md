# The within-day curve

6758 `[SELF]` messages with a timestamp. Local time: fixed UTC-4 (Mar-Oct) / UTC-5, zoneinfo unavailable.

Testing four claims made before the data was looked at: cadence slows, replies thin out, sentences shorten, and the break sits near 15:00 local.

The earlier hour-of-day check binned raw UTC and found nothing. Quebec is UTC-4/-5, so 15:00 local was landing in the 19:00-20:00 bins that were dismissed as too thin to read.

## Pooled, by local hour

| hour | msgs | median words | mean words | median gap (s) |
|---:|---:|---:|---:|---:|
| 07 | 150 | 8 | 9.5 | 16 |
| 08 | 484 | 8 | 10.7 | 20 |
| 09 | 757 | 9 | 11.8 | 20 |
| 10 | 1059 | 8 | 10.1 | 21 |
| 11 | 1263 | 8 | 9.5 | 18 |
| 12 | 677 | 8 | 9.8 | 19 |
| 13 | 727 | 8 | 9.4 | 19 |
| 14 | 608 | 8 | 10.6 | 21 |
| 15 | 398 | 10 | 11.3 | 24 |
| 16 | 365 | 9 | 11.4 | 21 |
| 17 | 88 | 8 | 11.1 | 40 |
| 18 | 24 | 6 | 9.6 | 22 |
| 19 | 43 | 9 | 10.6 | 11 |
| 20 | 36 | 8 | 10.9 | 17 |
| 21 | 41 | 13 | 15.1 | 21 |

## Before vs after 15:00 local

- **08:00-15:00** - n=5575, median words **8**, mean **10.2**, median gap **20s**, 1-5 word messages **31.3%**
- **15:00-23:00** - n=1002, median words **9**, mean **11.4**, median gap **22s**, 1-5 word messages **26.8%**

## Per-day, normalised against each day's own morning

Pooling lets a handful of very long days set the shape. This asks the question inside each day and then averages, so a fortnight of crises cannot masquerade as a daily rhythm.

15 days had 5+ messages in both windows.

- **words**: afternoon / morning = median **1.035** - shorter on **7/15** days (47%)
- **gap**: afternoon / morning = median **1.237** - slower on **10/15** days (67%)
- **volume**: afternoon / morning = median **0.560** - fewer on **10/15** days (67%)

A ratio of 1.000 means no within-day change at all. Below 1 for words and volume, above 1 for gap, is the predicted shape.

## Control: the other writer

If this is a fatigue curve rather than a property of the workday itself, the two writers should not have to share it.

- **[SELF]** - median words 8 before 15:00, 9 after (5575 / 1002 msgs)
- **[PEER]** - median words 6 before 15:00, 5 after (2313 / 433 msgs)
