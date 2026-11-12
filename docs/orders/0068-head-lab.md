# 0068 - The head lab: four ways to stop scoring 128,000 words at every token, tried on the frozen trunk's saved states
status: in progress (Cowork builds this one, on Nick's word of 2026-11-11)
written: 2026-11-11 by Cowork - approved: 2026-11-11 by Nick ("can you do it yourself?") - hypothesis: H-C3 (with H-C2, H-C6)

## Why
Nick, 2026-11-11: "something we did not fix yet is the 128k vocab". It is true in the code: `build.py` sets the
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

## Report
<!-- Cowork fills this in -->
