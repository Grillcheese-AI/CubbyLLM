# 0068 - The head lab: four ways to stop scoring 128,000 words at every token, tried on the frozen trunk's saved states
status: in progress (Cowork builds this one, on Nick's word of 2026-11-12)
written: 2026-11-12 by Cowork - approved: 2026-11-12 by Nick ("can you do it yourself?") - hypothesis: H-C3 (with H-C2, H-C6)

## Why
Nick, 2026-11-12: "something we did not fix yet is the 128k vocab". It is true in the code: `build.py` sets the
head's shortlist to the whole vocabulary, so every token scores all 128,000 rows. The claim "128k at a 32k head's
cost" does not hold today.

**What 0050 left** (closed 2026-11-05):
- The shortlist without retraining was killed: 0.958 greedy agreement against 0.995, with 8.7% of targets outside it.
  Its index and flag are in the code, off.
- The head is 1.8 to 2.9% of a decode step on an A100 and 7.3 to 9.6% at home. A perfect fix saves little time at
  128k.
- The training memory was fixed for the grown shape (the chunked cross-entropy).

**Why it still matters:**
- the claim has to be true or reworded;
- the head is 29% of the model's per-token weights (131M of 445M) and 524 MB of the export's 2.3 GB;
- the vocabulary is set to grow (Russian, Spanish, the dynamic tail of H-C6), and the head grows with it;
- the larger saving is fewer steps: H-C2 measured 22% fewer per document with merged tokens, and that needs a head
  that takes a growing vocabulary cheaply.

**What it brings, in plain words:** a bigger vocabulary stops costing more for every word Cubby writes, and the
model gets smaller.
- **Over a conventional model:** a model scores its whole vocabulary at every word.
- **For you:** more languages and longer merged tokens without paying for them at every step.

## What is in hand
No trunk training and no new dump are needed. Off-repo, from 0050: the trunk's states with the full head's top
tokens and normalizer (held-in 851,968 positions; held-in tuning 53,248; held-out 53,248), the frozen index of
0050's final read, and the trained head's rows.
- **Held-out positions are spent once.** 0050 read every 4th position after the first 2,048. Route 2 reads the
  next ones (offset 1), routes 3 and 4 the ones after (offset 2). Offset 3 stays unread.

## The four routes and their lines (pre-registered)
The reference is the full head on the same positions. **Rows** are the rows of 1,024 scored per token, the
index's own pass included. A 32k head scores 32,000. The measured-time line (at most a full 32k head's, on
grilly2 and on an A100) is read when a route is wired, never computed in the lab.
1. **The allowed list.** When the host already restricts the next token (the programs seat's slots, 0067's
   enforced rules), score only those rows. Exact by construction.
   - **Line:** the programs are identical on every logged prompt; the mean rows are at most 32,000.
2. **The shortlist with a fallback.** 0050's frozen index (screen plus successors, K = 4,096); the full head is
   used only when the shortlist is unsure. No training of the head.
   - The rule and its threshold are chosen on the held-in tuning split only, among three signals: the margin
     between the shortlist's two best scores; its best probability; a logistic score over those two, its entropy
     and the gap between the two nearest context clusters. The one with the lowest fallback rate at 0.997
     agreement there is frozen.
   - **Lines, on held-out, read once:** overall greedy agreement at least 99.5%; mean rows at most 32,000.
   - **Kill:** either line missed.
   - **Reported:** the fallback rate; the probability inside the shortlist where it is accepted; targets outside
     it there; the loss both ways; the 95% interval. 0050's loss line (+0.02 nats) is read too: if it is missed,
     this route serves greedy seats only.
3. **A two-level head.** Frequent tokens and cluster entries first; a rare cluster's rows only when a cluster
   wins. Its own distribution, so exact for itself. Trained on the held-in states against the full head's
   output, the trunk frozen.
   - **Lines:** held-out loss within +0.02 nats of the full head's on the same positions; mean rows at most
     32,000. **Kill:** beyond +0.05 nats.
4. **A binary-code head (HDC).** Each token is a bit code; the head maps the state to a code and reads the
   nearest by Hamming distance. Trained the same way.
   - **Lines:** as route 3's. **Reported:** the head's bytes against 524 MB.
   - Its growth test (new tokens given codes with no retraining) is written here before it runs.

Routes 3 and 4 that pass go to the serving batteries on a card (0050's list), in a session Nick opens.

## Constraints
- The lab lives under `validation/`. No trained weight is changed, serving is unchanged, and no file under
  `cubbyllm/` is touched until a route passes and Nick says to wire it.
- At home it runs on the CPU with half the cores. No GPU job at home; a Colab session is Nick's to open.
- numpy and the CPU torch already in the environment. No new dependency.
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim). No local path in a
  committed file.

## Out of scope
- Growing the vocabulary itself, and H-C2's merged tokens: after a head passes.
- The input embedding table.

## Report (Cowork)

### Route 2 as pre-registered (K = 4,096): killed (2026-11-12)
- **Log:** `validation/logs/head_lab_fallback_2026-11-12.json`. The rule frozen on held-in (26,624 positions) is the
  logistic score. To reach 0.997 there it hands 67% of positions to the full head (the margin: 70%; the best
  probability: 69%). Even 0.995 needs 60%.
- **Held-out, read once** (12,800 positions, offset 1): fallback 64.7%; overall agreement 0.99367 (95%: 0.99214 to
  0.99491); mean rows 91,017; loss 2.8375 against the full head's 2.6463. The shortlist alone agrees on 0.95984
  (0050 read 0.958 on its own positions).
- **Both lines are missed,** and 0050's loss line with them.
- **What it says:** where this shortlist is wrong, it does not look unsure. A disagreement means the full head's
  best token is outside the shortlist, and the scores inside carry little sign of it: among the 35% of positions
  the rule trusts most, 1.8% still disagree. A fallback keyed on the shortlist's own confidence cannot rescue a
  shortlist that small.

### A larger shortlist, on held-in only (2026-11-12)
0050 stopped at K = 4,096. Its cost line is a full 32k head's, which leaves room for about 27,900 rows beside
the index's 4,096.
- **Log:** `validation/logs/head_lab_sweep_2026-11-12.json` (13,312 held-in tuning positions; no held-out position
  touched).

| K | shortlist alone | probability inside | fallback needed for 0.995 | mean rows at 0.995 |
|---|---|---|---|---|
| 8,192 | 0.98445 | 0.951 | 43.5% | 67,945 |
| 16,384 | 0.99279 | 0.972 | 4.6% | 26,385 |
| 24,576 | 0.99549 | 0.980 | none | 28,673 |

- **Within a 32k head's rows, the frozen index reaches the line on held-in.** The margin is thin: at K = 4,096
  the shortlist alone missed about a tenth more often on held-out than on held-in (4.0% against 3.6%).

### Route 2b, pre-registered before its read (2026-11-12)
- The same frozen index, a larger K, the same three signals. No training.
- **Chosen on held-in only** (26,624 positions): among K in {16,384; 24,576} and targets {0.997; 0.996}, the
  highest target whose mean rows there are at most 30,000; then the fewer rows. If none qualifies, 2b stops
  there and no held-out position is read.
- **Lines, on held-out, read once (offset 2):** overall greedy agreement at least 99.5%; mean rows at most
  32,000. **Kill:** either missed.
- Routes 3 and 4 move to offset 3. No held-out position is then left unread: a later arm needs a new dump.
- **Not claimed by 2b, even if it passes: the measured time.** 24,576 gathered rows are not a contiguous 32k
  block. The time line is read on grilly2 and on an A100 when the route is wired. If the gather costs too much
  there, the next arm keeps a fixed block of frequent tokens, contiguous, and gathers only a short list.

### Route 2b, read: killed by its agreement line (2026-11-12)
- **The choice, on held-in only** (`validation/logs/head_lab_sweep_2026-11-12_choice.json`, 26,624 positions):

| K | shortlist alone | for 0.996: fallback, mean rows | for 0.997: fallback, mean rows |
|---|---|---|---|
| 16,384 | 0.99339 | 5.3%, 27,260 | 10.1%, 33,443 |
| 24,576 | 0.99572 | 0.2%, 28,933 | 1.5%, 30,558 |

  No setting reaches 0.997 within 30,000 rows. At 0.996 both do, and the rule takes the fewer rows: **K = 16,384,
  target 0.996** (the logistic score).
- **Held-out, read once** (`validation/logs/head_lab_fallback_2026-11-12_2b.json`; 12,800 positions, offset 2):
  - fallback 4.3%; **overall agreement 0.99336** (95%: 0.99180 to 0.99463); **mean rows 26,011**;
  - the shortlist alone: 0.99266; loss 2.9093 against the full head's 2.6965.
- **Against the lines:** agreement at least 99.5%: **missed**, by 0.16 points, the whole interval under the line.
  Mean rows at most 32,000: met. 0050's loss line: missed.
- **What it says:**
  - **The larger shortlist carries over.** Alone it agrees on 0.99266 held-out against 0.99339 held-in, at 20,480
    rows with the index's pass: two thirds of a 32k head.
  - **The fallback does not.** On held-in it lifted the agreement by 0.26 points; on held-out by 0.07, where
    handing 4.3% of positions over at random would give 0.03. As at K = 4,096: where the shortlist misses, nothing
    inside it shows it.
  - K = 24,576 alone read 0.99572 on held-in. With the same slip it would sit at the line on held-out. It is not
    read: no setting is tried on held-out after the fact, and offset 3 is kept for routes 3 and 4.
- **Where that leaves the claim:** with no training, the frozen index agrees with the full head on 99.3% of
  positions at two thirds of a 32k head's rows. That is short of the gate, and the gate stands.
- **Next: route 3.** The two-level head gives the first pass what the shortlist lacks: an entry per cluster of
  rare tokens that says how much probability lies outside. It is trained on the held-in states, the trunk frozen.
