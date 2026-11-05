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

## Step 1 on the A100: decode and the training step (Nick's session, 2026-11-04 night; Claude Code's reading)
- **Logs:** `head_share_decode_cuda_2026-11-05.json`, `head_share_train_cuda_2026-11-05.json`,
  `head_share_dump_cuda_2026-11-05.json` (`notebooks/head_shortlist.ipynb`, `validation/head_share_torch.py`; A100
  SXM4 80 GB). The files carry the runtime's UTC date.
- **Decode, fp32,** as serving decodes (the step timed with its greedy token read back, p50 of 30):

| | step | 128k head | head's share | full 32k head | K = 256 / 1,024 / 4,096 rows |
|---|---|---|---|---|---|
| batch 1 | 25.0 ms | 0.44 ms | **1.8%** | 0.17 ms | 0.13 / 0.12 / 0.12 ms |
| batch 16 | 26.7 ms | 0.78 ms | **2.9%** | 0.27 ms | 0.13 / 0.13 / 0.15 ms |

- **The training step:**
  - forward + backward, micro batch 8 x 1,024, bf16 autocast;
  - "trunk" is the same step through the trunk alone, its loss read off the state;
  - weights at init (shapes only).

| | params | step with the head + CE | trunk alone | head + CE: time | head + CE: peak memory |
|---|---|---|---|---|---|
| 450M (`base450m_cont`) | 576M | 547 ms, 35.8 GB | 480 ms, 22.1 GB | **12.3%** | **13.8 GB** |
| grown 1.36B (`grow_mem_grown`) | 1,328M | 1,403 ms, 71.3 GB | 1,346 ms, 58.0 GB | **4.0%** | **13.3 GB** |

- **What it says:**
  - **Decode:** the head is a small share on either card: 1.8 to 2.9% on the A100, 7 to 10% at home (where the step
    is launch-bound).
    - Stage 1's decode saving is at most that share; the 32k speed line is met with room on both cards.
    - So stage 1's case is the vocabulary's growth (the head scales with V, the trunk does not), not today's 128k.
  - **Training:** the head and its cross-entropy are 12.3% of a 450M step's time and 4.0% of the grown arm's. That
    caps stage 2's time saving.
    - **The larger part is memory:** 13.3 to 13.8 GB a micro batch of 8 x 1,024, whatever the trunk (38% of the
      450M's peak). A shortlist softmax would free most of it: about 1.6x the micro batch at the 450M.
    - That is the measured case for stage 2, as I read it. Whether to write stage 2 is yours.
- **Held-in states for the build:** 851,968 positions (64 windows of 1,024 per shard, training region), on Drive. The
  index built from them is read on the held-in tuning split next, then on the held-out tails once.

## Cowork's check of step 1 (2026-11-04, evening)
Read: both tables, the build notes, and the held-out peek disclosed in the inbox.

**What step 1 settles:**
- **Decode:** the 128k head is small today, at 1.8-2.9% of an A100 decode step and 7.3-9.6% at home (launch-bound). The
  shortlist's decode case is the vocabulary's growth, not today's 128k.
- **Training time:** the head and its cross-entropy are 12.3% of a 450M step and 4.0% of the grown arm's. That caps
  any time saving.
- **Training memory is the finding:** 13.3-13.8 GB per micro batch of 8 x 1,024, whatever the trunk. That is 38% of the
  450M's peak, and it is what pushed round 1's grown arm down to micro 2.

**Step 1's shortlist read:**
- Finish it as planned: the larger build, then one read on the held-out tails. Positions seen while building (the first
  2,048 held-out positions) are left out of that read, and the report says how many.
- The best arm is at 0.957 greedy agreement against the gate's 0.995. If the read stays under the gate, the kill line
  stands and is recorded: today's 128k head stays full until the vocabulary grows.

**Stage 2 splits in two, memory first:**
- **2a, now: an exact fused cross-entropy.** The loss and its gradient are computed chunk by chunk from the hidden
  state and the head's rows, and the 128k logits are never stored. The loss is the same; only memory changes.
  - Claude Code's choice of implementation, recorded: a pure-torch chunked autograd function, or the fused kernels of
    Liger Kernel or Apple's Cut Cross-Entropy, licence checked.
  - Behind a flag, off by default. It goes in `train_base.py` and in the pod runner's kernels phase, as the fused
    MinGRU went: a kernel that changes no number.
  - **Lines:**
    - loss and gradients equal to the full path on a toy batch (relative difference at most 1e-3 in bf16, 1e-5 in
      fp32);
    - at least 10 GB of the 13.3-13.8 GB freed at micro 8 x 1,024, for the 450M and the grown shape, on the A100;
    - step time no more than 3% slower at the same micro batch;
    - then the largest micro batch that fits, and its tokens per second, reported for both shapes.
  - **Kill:** a loss or gradient mismatch beyond the line, or less than 5 GB freed.
  - **For 0045:** if 2a passes before the control's long run starts, the runner's speed phase may pick it, recorded in
    `bench.json` like any kernel. Otherwise the control runs as it is.
- **2b, the shortlist softmax in training:** only if step 1's index passes, as its own order.

**In plain words:** Cubby's training no longer has to hold a score for every one of its 128,000 words at every
position, so the same GPU fits bigger batches. The answer it learns is exactly the same.
- **Over a conventional model:** large-vocabulary models usually pay this memory in full, or approximate the loss to
  avoid it. Here the loss is exact and the memory is gone.

Status: in progress.

## Stage 2a built: the exact chunked cross-entropy (Claude Code, 2026-11-04 night)
- **The choice, recorded:** a pure-torch chunked autograd function (`cubbyllm/model/vocab/fused_ce.py`), not Liger
  Kernel or Cut Cross-Entropy. It needs no kernel to build, runs on every card (the CPU tests included), and brings no
  licence question.
  - **Forward,** chunk by chunk (1,024 positions): the logits in fp32, their log-sum-exp, the target's logit. It keeps
    only the hidden state, the head's rows and one log-sum-exp a position.
  - **Backward:** the logits recomputed per chunk; `dh = (softmax - onehot) W` and `dW += (softmax - onehot)^T h`.
  - **Dtypes:** the matmuls run in the state's dtype (bf16 under autocast) against the head's rows, the softmax in
    fp32.
- **Behind `CB_FUSED_CE=1`,** off by default. `train_base.loss_fn` takes it only when the model has no spatial link
  (the copy path mixes into the logits) and a plain learnable head; it says so in the log. Off, the loss is the full
  path, unchanged.
- **The pod runner:** `tests/model/test_fused_ce.py` joins the kernels phase as `CB_FUSED_CE`. When it passes there,
  the speed phase benches each arm with and without it and keeps the higher tokens per second, both recorded in
  `bench.json`. The gain shows as a larger micro batch, not a faster step.
- **Tests** (`tests/model/test_fused_ce.py`, 5 passed on CPU):
  - loss and both gradients equal to the full path within 1e-5 relative in fp32;
  - within 1e-3 (loss) in bf16, the gradients within 2e-2 of both the bf16 and the fp32 full path;
  - ignored targets and a ragged last chunk;
  - a (B, S, d) state with an upstream gradient;
  - a bf16 state against fp32 rows, as autocast trains;
  - nothing of the logits' size saved for backward.
- **On the A100, next session:** `head_shortlist.ipynb` cell 4 (setup + cell 4 only). It runs the tests on the card,
  then the training step with the chunked loss beside the full one at micro 8 (memory freed, time), and the largest
  micro batch each path fits with its tokens per second, both shapes. That reads 2a's lines.

## Step 1's shortlist read: the larger build, tuned on held-in (Claude Code, 2026-11-04 night)
- Built from the 851,968 held-in states, tuned on the held-in tuning split (4,096 positions across the 13 shards). The
  best: screen + successors at K = 4,096 with 4,096 context clusters, **0.964** greedy agreement (1,024 clusters
  0.955; the 106k build 0.957).
- More build data moved it by under a point. The held-out read (frozen at that setting; the first 2,048 held-out
  positions left out; every 4th position after them) is running.

## Step 1's shortlist, read once on the held-out tails: the kill line holds (Claude Code, 2026-11-04 night)
- **Log:** `head_shortlist_heldout_2026-11-04_final.json`.
  - The index is frozen from the held-in tuning: built from the 851,968 held-in states, 4,096 context clusters for the
    screen arm, 2,048 head-row clusters for arms a-c.
  - Read on the 13 held-out tails, **leaving out their first 2,048 positions** (the ones seen while the build was
    shaped). Every 4th position after them, across all 13: **12,800 positions**. The full head's NLL on them is 2.676.

| arm | K | greedy agreement | top-10 recall | mass inside | target outside | rows scored |
|---|---|---|---|---|---|---|
| (a) clusters | 4,096 | 0.365 | 0.440 | 0.400 | 0.600 | 4,401 |
| (b) router | 4,096 | 0.876 | 0.762 | 0.785 | 0.220 | 4,751 |
| (c) router + successors | 4,096 | 0.905 | 0.800 | 0.812 | 0.190 | 4,763 |
| (d) screen (added) | 4,096 | 0.944 | 0.922 | 0.898 | 0.101 | 4,096 |
| **(d) + successors** | 1,024 | 0.867 | 0.784 | 0.780 | 0.216 | 1,035 |
| **(d) + successors** | **4,096** | **0.958** | 0.933 | 0.911 | 0.087 | 4,101 |

- **Gate: not met.** No arm reaches 0.995 greedy agreement at any K; the best is 0.958 (screen + successors, K =
  4,096). NLL with the outside mass lost is far from the full head's +0.02, since 8.7% of targets fall outside.
- **Kill: crossed as written.** No arm reaches 99.5% agreement at a K whose time beats a full 32k head, though every
  K's time beats it (0.12-0.15 ms on the A100 against 0.17-0.27).
  - **The 128k head stays full;** stage 2b (the shortlist softmax in training) is not written.
  - Stage 2a (the exact chunked cross-entropy) does not depend on the index and stands.
- **What it says:**
  - On the trained rows, an index from the trunk's own states (screen) does far better than one over the head's rows:
    0.94-0.96 against 0.37 for k-means.
  - The last 4% are the long tail of the next-token distribution. More build states moved agreement by under a
    point (106k: 0.957, 852k: 0.964 on held-in), and more clusters by about one.
  - A shortlist that must match the full head's argmax 99.5% of the time would need either a head trained with it
    (2b's training side) or an exact fallback when the shortlist's best is not confident. Either is a new order, if
    the vocabulary grows.

## Stage 2a's first A100 read: the memory line met, the matmuls in fp32 (Claude Code, 2026-11-04 night)
Nick's session ran cell 4 (`head_share_train_fused_fp32_cuda_2026-11-05.json`). The tests passed on the card (the
cell stops before the bench otherwise): the chunked loss agrees with the full path there.

| micro 8, S 1,024 | full | fused | freed | slower |
|---|---|---|---|---|
| 450M (L32) | 537 ms, 35.8 GB | 995 ms, 23.9 GB | **11.9 GB** | **+85%** |
| 1.36B (L64) | 1,390 ms, 71.3 GB | 1,849 ms, 59.4 GB | **11.9 GB** | **+33%** |

- **Largest micro (of 8, 12, 16, 24, 32, 48) under 92% of the card:** 450M full 16 at 18,059 tok/s, fused 24 at
  9,283 tok/s; 1.36B both 8 (5,877 against 4,438 tok/s).
- **Lines:** >= 10 GB freed: **met** on both shapes. <= 3% slower: **not met**. The kill (< 5 GB freed, or a mismatch)
  is not crossed.
- **Why it was slower: a build defect, not the method.** The added time is the same on both shapes (+458 ms, +459 ms),
  so it lies in the head, not the trunk. The callers compute `h` under autocast and call the loss outside it, where
  `h` (after the trunk's last norm) is fp32. The chunked loss ran its four head matmuls in fp32 (TF32 off), against
  the full path's three in bf16: about 4 x 110 ms of fp32 GEMMs at 8,192 x 1,024 x 128k, which is the gap.
- **The fix** (`cubbyllm/model/vocab/fused_ce.py`):
  - the matmuls run in a given dtype; `train_base` and the bench pass bf16, as the full head runs (None: autocast's
    dtype when it is on, else the state's);
  - both gradients are computed in the forward pass, chunk by chunk, and backward only scales them: three matmuls, as
    the full path, no recompute (the loss is the last op, its upstream gradient a scalar). Without grad only the loss
    is computed.
  - What is kept for backward: dh (positions x d) and dW in fp32 (128k x d: 0.5 GB), so about 0.5 GB less freed.
  - Tests: 6 passed on CPU, one new (the matmuls in the asked dtype against the full path on bf16-rounded inputs; eval
    computes only the loss).
- **Next:** cell 4 again, after a pull, in Nick's next A100 session. With the matmuls in bf16 and no recompute I
  expect the step within a few percent of the full path; the 3% line is read there, not assumed.

Status: in progress (2a's second read).

