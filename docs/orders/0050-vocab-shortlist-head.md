# 0050 — The vocabulary at a small head's cost: a real shortlist for the 450M's 128k output head
status: in progress
written: 2026-11-04 by Cowork · approved: 2026-11-04 by Nick ("I think a priority would be to fix the vocab first") · hypothesis: H-C3 (with H-C1, H-C6)

## Why
The claim in the novelties doc is a 128k+ vocabulary at a 32k head's cost, by bypassing the full softmax. Today it is
not true:
- **`build.py` sets `retrieval_k = v`.** It builds the head as `TopKRetrievalHead(..., learnable=True)` with the
  top-K set to the whole vocabulary, so every token scores all 128k rows. The module's docstring says the head "does
  NOT full-softmax"; the configuration makes it one.
- **Even with a smaller K, the head scores every row and then keeps K.** `TopKRetrievalHead` is the exact reference:
  it saves the softmax, not the matmul. Its docstring calls an ANN index "a drop-in swap ... later".
- **The head is the largest tensor in the model.** At D = 1,024 it is 128k x 1,024 (131M parameters), and decode reads
  all of it at every token (`quantize_`'s note).
- **Training pays a full cross-entropy over the vocabulary** (`validation/train_base.py`).
- **H-C3 measured only the bound,** on CPU: scoring a 1,024 shortlist is about 32x cheaper at the real head shape. But
  a naive index over random codes agrees only 37-80% at top-1. Index quality on trained rows is the open question.

**What it brings, in plain words:** Cubby's model picks each next word from a short list of likely candidates instead
of scoring all 128,000 words every time, so a large vocabulary stops slowing down every word it writes.
- **Over a conventional model:** a model scores its whole vocabulary at every word, so a bigger vocabulary costs more
  at every step. Here the cost follows the shortlist, not the vocabulary.

## Build: stage 1, at inference, with no retraining
Behind `CB_HEAD_SHORTLIST`, off by default.
1. **Measure today first:** the head's share of decode time and memory per token.
   - At home on grilly2 (the RX 6750 XT), and in torch on an A100.
   - At batch 1 and batch 16, in fp32 and in int8 (`quantize_`).
   - **And the head's share of a training step** (forward, backward and the cross-entropy, with its activation
     memory) for the 450M and for the grown 1.36B shape, on the A100. By parameter count the head is about 29% of
     the 450M's per-token matmuls (131M of 445M: backbone 313.65M, the input embedding a lookup) and about 11% of
     the grown arm's. The measured share decides whether stage 2 is worth its order.
2. **A shortlist index over the trained head's rows** (`base450m_cont`'s), built once offline. The arms (Claude Code
   may add one, recorded):
   - **(a) clusters:** k-means over the rows; probe the clusters whose centroids score best;
   - **(b) a learned router:** a small linear map from the trunk's final hidden state to cluster scores, trained on the
     frozen trunk's hidden states against the full head's top-k (minutes, on held-in text);
   - **(c)** (b) plus the previous token's most frequent successors.

   **The step:** score only the shortlist (size K), exactly, with the trained rows, and softmax over the shortlist.
3. **Where decode runs:** grilly2 (never grilly v1) and torch. The full head stays the reference.
4. **Sampling:** greedy decoding needs only the top-1. For sampled talk, the shortlist's probability mass (the share of
   the full head's mass inside it) is reported.

## Stage 2, training: only once stage 1 passes
A head trained with a sampled or shortlist softmax, so that training too pays for the shortlist, not the vocabulary.
Its own order, written from stage 1's numbers. 0045's runs are not changed.

## Acceptance (pre-registered, stage 1)
- **Tests:**
  - with K equal to the vocabulary, the shortlist path equals the full head (within float tolerance);
  - the index builds deterministically from a seed;
  - the flag off leaves every output bit-identical.
- **Index quality**, per arm and for K in {256, 1,024, 4,096}, on the 13 held-out tails:
  - the full head's top-1 inside the shortlist;
  - its top-10 recall;
  - the shortlist's probability mass;
  - held-out NLL with the shortlist softmax, with the mass outside it counted as lost.
- **Gate** (one arm and one K):
  - **greedy agreement:** the shortlist's argmax equals the full head's on at least 99.5% of held-out positions;
  - **NLL:** held-out NLL within +0.02 nats of the full head's;
  - **the serving batteries unchanged:**
    - E8 reworded spoken-and-correct and the emitter families;
    - H-E15's spoken-wrong at most 1%, with 0 copied literals;
    - the programs seat's greedy programs identical on at least 99% of prompts;
  - **speed:** the head's time per token at most a full 32k head's on the same hardware, measured (not computed), on
    grilly2 and on the A100.
- **Kill:** no arm reaches 99.5% greedy agreement at a K whose time beats a full 32k head.
- **Reported beside the gate:** end-to-end decode tokens per second before and after, at home and on the A100, and the
  memory.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Stage 1 does not change any trained weight.** Serving stays as it is until Cowork has read the report.
- grilly2 only on the local card, never grilly v1. No local path in a committed file.
- **0045's control runs as pre-registered,** with the full head. Stage 1 is applied to its final afterwards.
- A paid run is Nick's to start (the A100 measurements can ride a Colab session he opens).

## Out of scope
- **Stage 2** (the training side): its own order.
- **Growing the vocabulary** (256k, dynamic hypertokens, H-C2's LZW): after this one passes, each in its own order.
- **The input embedding table:** a lookup, not compute. Its memory is reported, not changed.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Step 1 on grilly2: the head's share of decode (Claude Code, 2026-11-04)
- **Log:** `head_share_grilly_2026-11-04.json` (`validation/head_share.py`, on grilly2, the RX 6750 XT). The export is
  `base450m_cont` (128,000 x 1,024). A decode step is timed with its greedy token read back to the host (what a seat
  waits for), after a 64-token prompt, p50 of 30 steps. Heads are timed alone on a hidden state of the step's shape,
  p50 of 50.

| | step | 128k head | head's share | full 32k head | K = 256 / 1,024 / 4,096 rows |
|---|---|---|---|---|---|
| fp32, batch 1 | 22.0 ms | 1.61 ms | **7.3%** | 0.50 ms | 0.11 / 0.13 / 0.17 ms |
| fp32, batch 16 | 32.8 ms | 3.13 ms | **9.6%** | 0.96 ms | 0.15 / 0.15 / 0.22 ms |
| int8, batch 1 | 21.2 ms | 0.75 ms | **3.5%** | 0.52 ms | 0.13 / 0.15 / 0.17 ms |
| int8, batch 16 | 32.4 ms | 2.65 ms | **8.2%** | 0.92 ms | 0.15 / 0.15 / 0.21 ms |

- **Memory:** the head is 524 MB in fp32 (of the export's 2.3 GB), 132 MB in int8.
- **What it says:** at home, the head is not where decode's time goes.
  - A step reads 2.3 GB in 22 ms, about 100 GB/s, well under the card's bandwidth. The step is bound by its many
    small launches (32 layers), not by the head's bytes.
  - The ceiling for a perfect shortlist at batch 1 is about 7% of a decode step (1.6 ms of 22), 3.5% with the int8
    head.
  - The speed line is within reach: a gathered K-row head costs 0.11 to 0.22 ms against the 32k head's 0.50 to 0.96.
    That leaves the index itself about 0.3 to 0.8 ms to fit.
- **Still to measure:** the A100 (torch, Nick's session), and the training step's share (Cowork's addition).
  - By parameters, the head is 29% of the 450M's per-token matmuls. A training step pays that in full, plus the
    cross-entropy over 128k, so stage 2's case rests on the A100's training number, not on decode at home.
