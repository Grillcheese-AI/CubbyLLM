# Training acceleration: a 2B on one A100 in ~20 hours? (2026-09-23)

Status: RESEARCH. Nothing here has been run yet. Each number is one of three kinds:
- **measured** in this repo, with its log linked;
- **sourced** from a primary source, with its arXiv id (the "Sources" section says which ones I re-read myself);
- **computed** by me from those two, marked "(computed)".

## TL;DR

- Twenty A100-hours buy about 1e19 FLOPs. That is roughly 1.1B tokens for the 2B shape, or 0.7 tokens per parameter.
- Reaching the loss of a week of training needs about 8x in effective training efficiency. Reaching the loss of one full pass over the cache needs about 16x.
- The levers with real evidence, stacked, give about 2-7x (midpoint ~4x) from scratch. Most of that comes from data selection. The optimizer is worth only about 1.1x at this scale. So a from-scratch 2B on one A100 in 20 hours does not reach "a week" of quality with any published method.
- Three routes do reach "2B, one card, ~20 hours":

| Route | What it is | Status |
|---|---|---|
| **A. Seed on the A100, grow on an H100** (recommended) | Spend the 20 A100-hours on a d1024/L32 hybrid (~450M). At that compute it is the better model: fitted loss about 3.0 vs 3.2 for the 2B. It is the "talks and emits" prototype. Then double its width to the 2B shape (HyperCloning: 2.2-4x faster than training from scratch) and continue for about 20 hours on one rented H100 (about $40-60). | Pilots P0-P4 |
| **B. Same stack from scratch on one H100** | About 20 H100-hours is about 63 A100-hours; x ~4 from the stack puts it between a week and one pass over the cache. | No prototype on the way |
| **C. Convert an open-weights model** (Qwen3-1.7B-Base, Apache-2.0) onto our tokenizer and hybrid | Fits in 20 A100-hours on paper, about 1B tokens. | Two unproven steps. The weights would be derived from someone else's model, so this is Nick's call |

## 1. The target in FLOPs

**Measured throughput**
- The 2B shape (D2048/L32/window 512, 1.254B trunk) runs at 18,619 tok/s and 45.5% MFU on an A100-80G ([exp_t1_mfu_pilot_a100_2b_flex.log](../../validation/logs/exp_t1_mfu_pilot_a100_2b_flex.log)).
- That run was **trunk only**: `MFU_V=0` is the pilot's default, and the command did not set it.
- The 131k head costs 6·2048·131072 = 1.61 GFLOP per token, against 7.53 for the trunk. That is **17.6% of FLOPs**.
- With the head included, the 2B does about **15.3k tok/s** at the same MFU (computed). The log's hour projections already include the head's FLOPs. The tok/s figure does not, so anyone dividing tokens by 18,619 undercounts time by 18%.

**The budget**

| Budget | Tokens at the 2B shape (computed) |
|---|---|
| 20 A100-hours | ~1.1B tokens, ~1.0e19 FLOPs |
| 1 week (168 A100-hours) | ~9.3B tokens |
| One pass over the 17.74B-token cache | ~320 A100-hours |

**What the published loss fits say.** These are transformer fits (Hoffmann et al. 2022; the Besiroglu et al. 2024 correction). I use them for relative comparisons only.

| Run | Hoffmann | Besiroglu |
|---|---|---|
| 2B @ 20 A100-hours (1.1B tokens) | 3.20 | 3.15 |
| 2B @ 1 week (9.3B tokens) | 2.66 | 2.60 |
| 2B @ one pass over the cache (17.7B tokens) | 2.55 | 2.50 |
| Compute-optimal at 1e19 FLOPs (~230-260M params, 6.5-7.3B tokens) | 2.98 | 2.93 |
| d1024/L32 (~450M) @ 20 A100-hours, 2.9-3.8B tokens | 3.01-3.07 | — |
| 2B with 2x / 4x / 8x effective data | 2.99 / 2.82 / 2.67 | 2.92 / 2.74 / 2.61 |

**Reading**
- At 1e19 FLOPs the 2B spends about half its compute on its size rather than its learning.
- To reach "a week", the 2B needs about 8x effective data.
- The d1024 model at the same 20 hours sits near the compute-optimal line.

## 2. What each lever is worth for us (ranked)

| # | Lever | For us | Evidence | Risk / note |
|---|---|---|---|---|
| 1 | **Data selection aimed at our targets**, then repetition, then the best data in the WSD decay | **1.5-2.5x** | - BETR (2507.12466): 2.1x compute multiplier over DCLM-Baseline (4.7x over unfiltered), 500+ models from 1e19 to 1e22 FLOPs. Smaller budgets want harsher filtering.<br>- Repetition up to ~4 epochs is nearly as good as fresh data (2305.16264).<br>- MiniCPM (2404.06395): high-quality plus SFT data in a ~10% decay. | - FineWeb-Edu and Nemotron-CC HQ are already in the cache, so the famous "10x" (FineWeb-Edu, MMLU only) is mostly already captured.<br>- Scoring the cache is CPU work.<br>- Watch French separately. |
| 2 | **Width growth from our own d1024 model** (HyperCloning) | **2.2-4x** on the grown phase | - HyperCloning (2409.12903): hidden dimension doubled at the same depth, e.g. OPT-350M (d1024) to OPT-1.3B (d2048). That is our exact shape change. Reaches the random-init baseline's final accuracy 2.2-4x faster. Base cost not counted; early forgetting right after expansion.<br>- G_stack (2405.15319), depth growth: a 7B reaches the 300B-token loss in 194B tokens (54.6% speedup). | - The base's cost is not sunk for us, but we want the base anyway (it is the prototype).<br>- MinGRU gates and RMSNorm work per channel, so width cloning should preserve the function exactly (my derivation; P3 step 0 checks it). |
| 3 | **A100 throughput per token** | **1.15-1.35x** | - Cut Cross-Entropy (2411.09009) removes the head's logit memory, which allows batch 16.<br>- torchao int8 mixed-precision training (+20% end to end on A100 for Llama3-8B, per a research agent's read of the torchao PR; not re-checked).<br>- FlashAttention-2 shows A100 GPT training reaching ~72% MFU (2307.08691). | - No FP8 on the A100.<br>- 2:4 sparsity: skip (small end-to-end gains, loss cost).<br>- int8 on the MinGRU gates is untested. |
| 4 | **Optimizer (Muon)** | **~1.1x** (range 1.0-1.3x) | - Wen et al. (2509.02046): the speedup of matrix-based optimizers over a *well-tuned* AdamW "decreas[es] from 1.4x ... for 0.1B ... to merely 1.1x for 1.2B".<br>- Moonlight's ~2x (2502.16982) is against a less-tuned baseline.<br>- Newton-Schulz overhead is about T·m/B (Keller Jordan): 125% at 8k tok/step, 7.8% at 128k (computed). | - Worth using only with gradient accumulation, weight decay, and QK-norm on the window layers.<br>- Kimi Linear (2510.26692; not re-checked) trained a 3:1 recurrent/attention hybrid stably with MuonClip. |
| 5 | **Batch and schedule** | **~1.0x in tokens**, but needed | - Critical batch size "scales primarily with data size rather than model size" (2410.21676, 85M-1.2B). At ~1-3B tokens it sits far above our 8k tok/step.<br>- WSD matches a tuned cosine (2405.18392) with a 10-20% decay, and any stable-phase checkpoint can be cooled down. | - That makes a WSD run insurance against Colab ending the session.<br>- The `CB_LR=3e-3` default was tuned on d512 runs; re-anchor it at the new shape. |
| 6 | **Pretraining distillation from a teacher** | **≤1.1x** in this budget | - Distillation beats supervised training only "in settings involving many students or an existing teacher ... up to a compute level that scales predictably with student size" (2502.08606).<br>- A research agent found no from-scratch pretraining result for cross-tokenizer distillation.<br>- An LFM2.5-2.6B forward pass over 1.1B tokens would take about half the budget. | - Keep the settled plan: ULD-KD comes after pretraining. |
| 7 | **Curricula** (sequence-length warmup, SkyLadder), Rho-1 | ~1.0x | - Their gains come mostly from saving long-context attention cost; our window attention is a small share of FLOPs.<br>- Rho-1 was shown in continued pretraining, not from scratch. | - Skip. |
| 8 | **MoE** (2B total, ~0.5B active) | ~1.0-1.25x | - The panel already settled cycle one as dense.<br>- MoE carries a token tax and an A100 grouped-GEMM kernel risk.<br>- It buys parametric capacity, which we deliberately keep outside the weights. | - Revisit at ≥1e20 FLOPs. |
| 9 | **Convert an open-weights transformer** | 2B-class in ~1B tokens, *if it works* | - Liger (2503.01496): 93% of the transformer recovered at 0.02% of pretraining tokens (1B-8B).<br>- RADLADS (2505.03005): Qwen2.5 7B-72B converted to an RWKV variant with 350-700M tokens. On the 7B, the benchmarks scored by log-likelihood hold (MMLU 68.2 vs 71.7), but **generation and long context do not**: GSM8K 61.4 vs 85.9, HumanEval 50.0 vs 79.9, passkey ≥95% up to 8k vs 34k+. | - Every conversion with ≥90% retention that the agent found uses a matrix-state mixer (GLA, GDN, RWKV-7, Mamba-2). None converts to a vector-state MinGRU.<br>- Moving to our tokenizer adds 0.7-2B tokens (ALM 2503.20083, TokAlign 2506.03523).<br>- No published run does both at once. |

## 3. What the stack adds up to

- These multipliers do not compose cleanly: data selection and growth overlap, and most were measured at other scales.
- **From scratch on one A100:** about 1.8x (low), 4x (mid) or 7x (high, if every pilot passes).
- At the midpoint, 20 A100-hours is worth about 80 A100-hours of today's recipe (fitted loss about 2.82, against 2.66 for "a week"). That falls short.
- **Route A's arithmetic** (computed; assumes the multipliers hold and stack):
  - An H100 does about 3.2x an A100's work per hour at the measured MFU. That is a transfer estimate from the pilot log, and the log itself says to spot-check it on the actual rental card.
  - So 20 H100-hours is about 63 A100-hours.
  - Growth (2.2-4x) times data selection (1.5-2.5x) puts that at about 210-640 A100-hours of today's recipe: at least a week, up to about two passes over the cache.
  - Even the low end clears a week, but only if both multipliers show up at all. P1 and P3 are what establish that.

## 4. The 20-hour A100 run (Route A, phase 1)

- **Model**
  - d1024/L32 hybrid, window attention every 3rd layer (window 512).
  - 4 heads × 256 dims, so doubling the width gives exactly the 2B's 8 × 256.
  - Tied 131k embedding and head, as in the MFU pilot.
  - About 314M trunk + 134M head ≈ 450M parameters.
- **Tokens:** 2.9-3.8B in 20 hours at 35-45% MFU (computed). The 150M pilot hit 32.8%; the d1024 full-stack MFU is unmeasured (P4 step 0).
- **Data**
  - Rank the 17.74B cache against our targets (EN/FR prose, dialogue, program emission), BETR-style.
  - Keep the top ~1-1.5B unique tokens and repeat them 2-3x.
  - The last 15-20% (the decay) is the best slice plus the instruction, agent and program data.
- **Optimizer:** AdamW with β2 ≈ 0.99 and the LR from P2. Use Muon on the hidden matrices only if P2 passes.
- **Batch:** 128k tok/step (8×1024 × 16 accumulation).
- **Schedule and checkpoints**
  - WSD: ~1-2% warmup, ~20% decay to zero.
  - Checkpoint the stable phase every 30 minutes, so any checkpoint can be cooled if the session ends.
  - Colab Pro+ runs "for up to 24 hours if you have sufficient compute units"; Pro and pay-as-you-go stop at 12 hours (Colab FAQ).
- **Throughput:** Cut Cross-Entropy (or a chunked CE) and a bigger micro-batch. int8 only if P4 passes.
- **Serve side, no training cost:** grammar-constrained decoding against the VM grammar for program emission.

## 5. Pilots (each ≤ 2 hours on a Colab A100)

**P4 step 0 comes first (15 minutes).** It calibrates every budget above.
- Run the MFU pilot with `MFU_V=131072` at D2048 and D1024.
- The head has never been inside a measured step.

**P0: size (the reframe)**
- **Claim:** at 1e19 FLOPs, d1024/L32 beats the 2B shape.
- **Test**
  - Train both for 55 minutes each on identical data.
  - Branch WSD cooldowns at 15, 30 and 55 minutes.
  - Fit L(t) and extrapolate to 20 hours.
- **Kill (go back to the 2B)** if the 2B's extrapolated loss is ≥ 0.03 nats lower on the EN, FR *and* program held-out sets.

**P1: data selection**
- **Claim:** target-aimed selection is worth ≥ 1.5x.
- **Test**
  - Use a proxy with the same layer pattern.
  - Run two arms of about 250M tokens each on the same WSD schedule: a random slice, and the top ~10% of a 3B-token shard ending on the best data.
- **Kill** if the selected arm needs more than 1/1.5 of the random arm's tokens to reach the random arm's final loss, or if general web validation loss gets more than 2% worse.

**P2: optimizer, batch and LR**
- **Step 0:** time Muon at the full 2B shape with 16x accumulation. Kill Muon if its overhead is above 8%.
- **Proxy:** d512/L32 (~78M trunk), 80M tokens per arm (about 1 token per parameter, like the real run), 32k tok/step, WSD.
- **Arms**
  1. AdamW at LR {1e-3, 2e-3, 4e-3}.
  2. The current config (3e-3, cosine with floor), to see what fixing the baseline is worth.
  3. Muon at {0.01, 0.02} with weight decay 0.1 and QK-norm.
  4. Muon with the MinGRU gate projections left on AdamW.
  5. The best AdamW run for 100M tokens: the 1.25x yardstick.
- **Kill Muon** if its best run at 80M tokens does not match arm 5, or on loss spikes or runaway attention logits.

**P3: growth**
- **Claim:** d1024 to d2048 by width cloning reaches the scratch 2B's loss ≥ 2x faster.
- **Step 0:** clone the P0 d1024 checkpoint. Validation loss must match the base within bf16 noise. If it doesn't, the clone is wrong; fix it before going on.
- **Test:** grown vs scratch 2B, 50 minutes each, same data order.
- **Kill** if any of these holds:
  - the grown model is < 0.10 nats better at ~50M tokens;
  - the gap shrinks by more than 50% between 15M and 50M tokens;
  - the scratch model passes the base's loss within the pilot.
- This pilot tests the mechanics on a weakly trained base; the real payoff is measured after phase 1.

**P4: throughput**
- Profile the full-stack step.
- Try Cut CE at batch 16, then torchao int8 on the forward and input-gradient matmuls.
- **Kill int8** if end-to-end tok/s improves by less than 1.10x, if the loss gap exceeds 0.01 nats at 200M tokens on the proxy, or on any loss spike.

**P5: conversion (only if Nick says yes to derived weights)**
- **Source:** Qwen3-1.7B-Base (Apache-2.0, d_model 2048, 151,936 vocab, French included).
- **Steps**
  1. 10 minutes: vocabulary overlap, then zero-shot OMP/FVT bits-per-byte on EN and FR.
  2. 70 minutes: align attention layers to MinGRU layers, against a GLA control from the same initialization.
  3. 30 minutes: distillation, then bits-per-byte plus parse rate on 50 program-emission prompts.
- **Kill** if any of these holds:
  - MinGRU per-layer R² < 0.6 while GLA's > 0.8;
  - bits-per-byte after distillation > 1.10x the teacher's;
  - zero-shot tokenizer bits-per-byte > 1.3x the original.

## 6. Decisions that are Nick's

1. **Is the goal "a 2B", or "the best model 20 A100-hours can buy"?** The evidence says the second is a ~450M, and that it is also the best seed for the 2B.
2. **Route A's second phase needs about 20 rented H100-hours:** about $40-60 at the pilot log's $2-3/hour range.
3. **Derived weights (route C).** A model that starts from Qwen3's weights: is it still "our own"?
   - Apache-2.0 allows it as long as the LICENSE and NOTICE are kept.
   - It ends "trained from scratch".
   - It puts Qwen's parametric knowledge into the weights, which cuts against knowledge-outside-the-weights.
   - LFM2/LFM2.5 is the alternative, and its gated short-conv block maps closest to MinGRU. But its license covers commercial use only while annual revenue is under $10M.
   - Pilot P5 only on a yes.

## 7. What this changes in the settled 2B runbook

- **Kept:** dense 1:3 hybrid, WSD, full-vocab CE, torch.compile + flex, no gradient checkpointing, ULD-KD after pretraining, MoE in cycle two.
- **Changed**
  1. Cycle one starts at d1024 and grows to d2048 (seed and grow).
  2. A data-selection pass comes before the run.
  3. About 128k tok/step through gradient accumulation.
  4. Budgets quote full-stack tok/s, not the trunk-only 18,619.

## Sources

**Re-read by me on the abstract or paper page (2026-09-23):**
- [2509.02046](https://arxiv.org/abs/2509.02046) Fantastic Pretraining Optimizers
- [2507.12466](https://arxiv.org/abs/2507.12466) BETR
- [2405.15319](https://arxiv.org/abs/2405.15319) G_stack
- [2409.12903](https://arxiv.org/abs/2409.12903) HyperCloning (speedups and model pairs from the HTML body)
- [2503.01496](https://arxiv.org/abs/2503.01496) Liger
- [2505.03005](https://arxiv.org/abs/2505.03005) RADLADS (Tables 5 and 10 from the HTML)
- [2502.08606](https://arxiv.org/abs/2502.08606) Distillation Scaling Laws
- [2410.21676](https://arxiv.org/abs/2410.21676) Critical batch size
- [Colab FAQ](https://research.google.com/colaboratory/faq.html) runtime limits

**Read by research agents, not re-checked:**
- 2502.16982 Moonlight
- 2505.02222 Muon efficiency
- 2510.26692 Kimi Linear
- 2405.18392 WSD
- 2404.06395 MiniCPM
- 2305.16264 data-constrained scaling
- 2411.09009 Cut Cross-Entropy
- 2307.08691 FlashAttention-2
- torchao int8 training PR
- 2503.20083 ALM
- 2506.03523 TokAlign
- 2506.06607 OMP
- 2410.10254 LoLCATs
- 2502.14458 Llamba
- 2409.02060 OLMoE
- 2404.07965 Rho-1
- 2503.15450 SkyLadder
