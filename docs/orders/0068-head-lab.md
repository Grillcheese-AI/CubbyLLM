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

### Route 3 on held-in: a two-level head does not fit a frozen trunk (2026-11-12)
- **Log:** `validation/logs/head_lab_twolevel_2026-11-12_pilot.json`. Held-in only: 60,000 states to fit, 13,312 to
  read; 16,384 frequent tokens, 4,096 clusters of the other 111,616.
- The entries' first-pass KL stops at 0.23 nats (0.36 in the first epoch). As its own distribution the head loses
  **0.50 nats** to the full head (the kill is 0.05) and agrees with it on 0.900.
- Read as a router (the entries only choose which clusters to open): 0.921 to 0.952 at about 20,500 to 21,200
  rows. The context shortlist gives 0.993 at the same rows.
- **Why:** a cluster's probability is close to its best token's, and which token that is changes with the state.
  One row per cluster cannot follow it on a trunk trained with a flat head. 13.7% of the full head's
  probability, and 8.5% of its best tokens, lie outside the 16,384 most frequent tokens.
- **Route 3 is not read on held-out.** A two-level head needs the trunk trained with it: a question for the next
  base, not for this lab.

### A coarse pass alone: not enough (held-in, 2026-11-12)
A coarse pass looks at all 128,000 rows cheaply; the exact head then scores only its best N. **Log:**
`validation/logs/head_lab_coarse_2026-11-12_heldin.json` (13,312 positions). The cost of a pass is counted in
rows by the bytes it reads.

| coarse head, built from the trained rows | its pass, in rows | the full head's best token in its top 16,384 |
|---|---|---|
| one bit per weight | 4,000 | 0.868 |
| the same, the mean row removed | 4,000 | 0.857 |
| Hamming: one bit per weight and per state dimension | 4,000 | 0.718 |
| rank 64 | 8,000 | 0.940 |
| rank 128 | 16,000 | 0.956 |

- In 1% of positions the best token sits beyond rank 67,000 in every one of them: rare tokens, decided by fine
  directions that a coarse head drops.

### The context list and a Hamming pass together (held-in, 2026-11-12)
They miss different positions. **Logs:** `validation/logs/head_lab_coarse_2026-11-12_union*.json` (13,312
positions). The rows count the index's pass, K, the coarse pass and N in full, with no discount for rows shared.

| K | coarse pass, top N | agreement | caught where the list misses | rows |
|---|---|---|---|---|
| 8,192 | Hamming, 1,024 | 0.99715 | 82% | 17,312 |
| 8,192 | Hamming, 2,048 | 0.99812 | 88% | 18,336 |
| 8,192 | Hamming, 4,096 | 0.99887 | 93% | 20,384 |
| 16,384 | Hamming, 2,048 | 0.99917 | 89% | 26,528 |
| 16,384 | Hamming, 4,096 | 0.99970 | 96% | 28,576 |
| 16,384 | one bit per weight, 1,024 | 0.99805 | 73% | 25,504 |
| 16,384 | rank 64, 4,096 | 0.99579 | 42% | 32,576 |

- **The Hamming pass is the weakest alone and the best in the union.** My reading, not measured: the context
  list misses a token that is unusual for its context, and such a token is mostly a common one that a bit code
  finds; the Hamming pass misses rare tokens, and the context list holds those.
- **This is where a binary code pays:** one bit per weight, compared by counting bits, with no training.

### Route 5, pre-registered before its read (2026-11-12)
- **The union:** 0050's frozen context list of K rows and the top N of a Hamming pass over the trained rows' signs
  (scaled by each row's norm); the union's rows are scored exactly and the best is taken. Nothing is trained.
- **Chosen on held-in:** the fewest rows among the settings at or above 0.998 there: **K = 8,192, Hamming,
  N = 2,048** (18,336 rows).
- **Lines, on held-out, read once (offset 3, the last unread positions):** overall greedy agreement at least 99.5%;
  rows at most 32,000. **Kill:** the agreement missed.
- **Not claimed, even if it passes:**
  - the measured time: a bit-count pass over 128,000 codes, a top-N selection and two gathers. It is read on
    grilly2 and on an A100 when the route is wired;
  - 0050's loss line, for sampled talk: the probability inside the union is not read here.
- Routes 3 and 4 no longer read offset 3. Route 3 stopped on held-in. Route 4 (a head trained on bit codes) needs
  a new dump.

### Route 5, read: both lines met on held-out (2026-11-12)
- **Log:** `validation/logs/head_lab_coarse_2026-11-12_route5.json` (12,800 held-out positions, offset 3, read once;
  K = 8,192, the Hamming pass's top 2,048, as chosen on held-in).
- **Overall greedy agreement with the full head: 0.99758** (95%: 0.99657 to 0.99829). The line is 0.995: **met**, the whole
  interval above it.
- **Rows: 18,336** (the index's pass 4,096, the list 8,192, the Hamming pass 4,000 by bytes, its top 2,048; the two
  lists counted in full). The line is 32,000: **met.** It is 57% of a 32k head and 14% of the full head.
- The context list alone agrees on 0.98445 there (0.98445 on held-in). The Hamming pass catches 84% of what it
  misses (88% on held-in).
- **What is established:** with nothing trained, the 450M's next token can be chosen from 18,336 rows instead of
  128,000 and be the full head's choice on 99.76% of held-out positions.
- **What is not, and comes next:**
  1. **The time.** Rows are bytes read. The Hamming pass is a bit count over 128,000 codes, then a top-N
     selection; the union's rows are gathered, not contiguous. Read on grilly2 and on an A100 once wired, against
     a full 32k head, as 0050's line says.
  2. **The serving batteries** (0050's list: the programs seat's greedy programs identical on at least 99% of
     prompts, E8, H-E15). They need the route wired.
  3. **Sampled talk:** 0050's loss line. The probability inside the union is not read here.
  4. **Wiring it** touches `cubbyllm/model/vocab/shortlist.py` (a Hamming arm and the union, behind
     `CB_HEAD_SHORTLIST`, off by default). That waits for Nick's word.
- **Where the other routes stand:** route 1 (the allowed list) is exact and still to be counted on the programs
  seat's prompts; routes 2 and 2b are killed; route 3 stopped on held-in; route 4 (a head trained on bit codes)
  is not started, and its case is now the growth of the vocabulary, not today's 128k.

### Route 5, wired behind the switch (2026-11-12, on Nick's word: "yes")
- **What is wired.**
  - `cubbyllm/model/vocab/hamming.py` (new, numpy only): the Hamming pass. A row is kept as its signs, one bit a
    weight, packed the way grilly2's bit codes are, and its norm. A state is read as its signs. A row's score is
    its norm times (d minus twice the Hamming distance). `HammingIndex.top` returns the best N rows; ties at the
    Nth place go to the lower id.
  - `cubbyllm/model/vocab/shortlist.py`: "+ham" is a suffix on any arm. The arm's list is joined with the pass's
    best N. `ARMS` stays 0050's five, so 0050's evaluation reads what it read.
  - `cubbyllm/model/vocab/output_head.py`: `attach_shortlist(index, k, arm, ham_n)` builds the pass from the head's
    own rows when the arm ends in "+ham". It is not saved with the index, so it cannot go stale against the rows.
  - The switch is 0050's: `CB_HEAD_SHORTLIST=1` and an index attached. Off by default. Off, the head's output is
    bit-identical (pinned).
- **Tests:** `tests/test_head_hamming.py` (8) and `tests/validation/test_head_lab_wired.py` (4), new;
  `tests/test_head_shortlist.py` (0050's 6) unchanged and passing.
- **The wiring check, fixed before it runs** (`validation/head_lab_wired.py`; held-in only, it refuses a held-out
  dump). On the 13,312 held-in positions of the lab's table (`heldin_val`, every 4th), "screen+succ+ham",
  K = 8,192, N = 2,048:
  1. the wired list holds the full head's best token exactly where the lab's two hit tests say it does, on all but
     at most 5 positions (rounding at the Nth place), so its agreement is the lab's 0.99812 within 0.0004;
  2. through `TopKRetrievalHead` with the switch on, on every 13th of them: the rows kept are the list's on every
     position, their logits are the full head's within 1e-3, and the best token is the full head's exactly where
     the list holds it.

  A miss on either is a wiring fault to fix. It is not a result about route 5.
- **Also read there, as an exploration:** the pass ranked by Hamming distance alone, with no norm. That is what
  grilly2's packed top-k (`cubbyllm/ops/packed.py`, order 0031) computes today. Its ties at the Nth place are
  counted for and against. It decides nothing: a decision needs held-out positions, and all four offsets are spent.
- **Not wired by this step.** The served seats decode in grilly2 (`cubbyllm/ops/grilly_seats.py` calls the model's
  own `generate`), and that loop does not read this index. The torch head computes its lists in numpy on the CPU,
  one row at a time: it is the reference a kernel must match, not the fast path. The time on a card, the serving
  batteries and sampled talk stay as listed above.
- **Found on the way, not from this order:** `tests/test_guards.py::test_every_module_declares_wiring` fails on
  `cubbyllm.reasoning.step_loop` and `cubbyllm.spatial`. Neither declares `__wiring__`; neither is touched here.
  Reported to Claude Code in the inbox.

### The wiring check, read: both lines met (2026-11-12)
- **Log:** `validation/logs/head_lab_wired_2026-11-12_heldin.json` (13,312 held-in positions; six CPU threads).
- **Line 1, met.** The wired list holds the full head's best token on 0.99812 of positions (25 misses). The lab's
  two hit tests say 0.99812 (25 misses). The two differ on **0 of 13,312 positions**. No weight of the head and no
  state value there is an exact zero, so the two readings of a sign cannot differ on them.
- **Line 2, met.** Through `TopKRetrievalHead` with the switch on, on 1,024 positions: the rows kept are the list's
  on all 1,024; the largest difference from the full head's logits on those rows is 6.7e-05; the best token is the
  full head's on 1,022, the same 1,022 where the list holds it.
- **The rows, counted on the lists themselves.** The union holds 10,095 rows on average (median 10,102, most
  10,250): the two lists share few rows. By bytes a token reads 18,191 rows on average (the index's pass 4,096,
  the Hamming pass 4,000, the union 10,095). The lab's 18,336 counted both lists in full, and it left out two
  things: the successors (up to 32 rows a token) and the norms the pass reads (125 rows by bytes). With both, the
  average is 18,316 and the most a token can read is 18,493. The line was 32,000.
- **The time of the reference, on this machine's CPU: slower than the full head, not faster.** Through torch, 21.0 ms
  a position with the switch on against 1.35 ms with it off. In numpy, the context list takes 3.1 ms a position
  and the Hamming pass 10.6 ms, against 0.78 ms for the full head scored 256 positions at a time. The lists are
  built in numpy, one position at a time. None of this is a card's time. **Turning the switch on today makes
  decoding slower. It stays off.**
- **Distance alone (an exploration, held-in).** Ranked by Hamming distance with no norm, the pass's best 2,048 give
  a union agreement of 0.99820 with every tie at the 2,048th place counted against it, and 0.99842 with them
  counted for it (at 4,096: 0.99880 and 0.99910). The norm-weighted pass reads 0.99812 on the same positions. So
  on held-in the norm buys nothing, and grilly2's packed top-k, which ranks by distance alone on the card,
  computes this pass as it is. Not read on held-out: all four offsets are spent, so the decision needs a new dump.
- **What this step establishes:** the model code computes route 5 as the lab read it, position for position, and
  with the switch off it changes nothing.
- **What it does not: any speed.** The ceiling is 0050's: the head is 1.8 to 2.9% of an A100 decode step and 7.3 to
  9.6% at home. The head's memory does not shrink either: the trained rows are still read, and the bits add 16 MB.
- **Next, in this order:**
  1. A new dump of held-out states, to read distance alone once against the same lines (0.995; 32,000 rows). A
     dump needs a card: Nick's call.
  2. The pass on the card through `cubbyllm/ops/packed.py`, timed against a full 32k head (0050's line).
  3. The serving path in grilly2, then the serving batteries and sampled talk.
