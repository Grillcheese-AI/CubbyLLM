# CUBBYLLM — GROUNDED FACT PACK (for the business-model competition)
Date: 2026-09-10. Company: GrillCheese Research Lab (GRL). Founder: solo, Lévis, Québec, Canada.

## 0. RULES OF ENGAGEMENT — READ FIRST
This project has a hard, enforced rule: **no fabricated measurements.** Every number in this pack is tagged:
- `[M]` MEASURED — a script + captured log produced it.
- `[P]` PROJECTED — arithmetic off a measured number.
- `[X]` EXTRAPOLATED — explicitly linear/analytic extension, not measurement.
- `[N]` NARRATIVE — asserted somewhere in the project's history with NO methodology. **These are listed so you can avoid them.**
- `[NM]` NOT MEASURED — a known, admitted gap.
- `[EXT]` EXTERNAL — third-party source with a URL.

**You will be scored. A fabricated number, result or citation costs −5 points each.** If you need a number that is not in this pack, you must either derive it from a pack number (show the arithmetic) or state "no source — assumption" and give the costing basis. Do not invent TAM figures, customer counts, ARR, growth rates, or benchmark scores.

---

## 1. WHAT THIS ACTUALLY IS
CubbyLLM is **not a small language model**. It is an eight-layer infrastructure in which the language model is deliberately the smallest and most replaceable layer.

1. **Compute** — `grilly`: a Vulkan compute framework, 231 GLSL shaders → SPIR-V, C++ dispatch, pybind11, PyTorch-like API. **Runs on any GPU. No CUDA.** 1,820 tests. Published on PyPI. MIT.
2. **Representation** — one algebra: qFHRR sparse block codes, K=80 × L=128, D=10,240. `FastWordEncoder` turns text into these **with no neural model in the path**.
3. **The VM — CubeLang** — a Rust, deny-by-default, **verify-before-execute** instruction machine with real VSA bind/unbind + cosine cleanup. Capability model: core helpers are versioned, name-addressed, unmodifiable precompiles; user code is verified before it runs; a program may `override` only where marked overridable. Principle: **verify containment, not correctness** — the only notion of safety that survives inputs you did not foresee, which matters because the model writes its own code. MIT. ~92 KB language spec. 12 declared interfaces (`ISolver`, `IAgent`, `IMemory`, `IOrchestrator`, `IProxy`, …), access control (`@external`/`@internal`/`@system`/`@restricted`/`@ratelimit`/`@hook`/`@cron`), lifecycle hooks, events/channels, and container types including `RobotContainer`.
4. **The symbolic boundary** — grilly's block codes and the VM's bipolar codes **never exchange raw hypervectors**. Only `(symbol, similarity)` crosses. Deny-by-default applied to representations.
5. **Worlds / knowledge (MoWM)** — a shared Global World of axioms, per-task worlds, centroid+margin routing, spawn-on-OOD, router-mediated inter-world query. `FactStore` + `TripleIndex` is the serving form.
6. **Reasoning** — the verified CoT pipeline: question → 6-WH-frame grammar → per-hop retrieve → **CubeLang chain program** → VM per-hop `recover` verification + absent-role control → verified answer **or an honest refusal**. Every outcome is harvested as training data.
7. **The brain / host** — sense (a real 5-signal neurochemistry ODE with receptor saturation and HPA dynamics) → fast route → cortex router → {Memory, Reasoning, Tool, Plugin, Chat} cortices. Hormones scale **thresholds and budgets only, never facts**. Every spoken word exits through a VM-mediated ASK: **nothing can be said that the VM did not offer.**
8. **The trunk (the LM)** — proposes programs and speaks. It **never judges, scores, or certifies.** Swappable behind one `Emitter` interface.

Cross-cutting: a **signed certificate ledger** (SQLite, HMAC-SHA256, every VM decision keyed by program SHA + VM build; changed program text loads RETIRED; a new VM build retires every certificate at once), an **encrypted vault** (encrypt-then-MAC, 256-bit key, local only), and **the harvest** (today's hand-written grammar generates the training data for the model that replaces it).

### THE SEVEN INVARIANTS — an idea that violates one scores ZERO
1. The VM is the only truth gate. **No LLM-as-judge anywhere in the loop.** The model never scores its own thoughts.
2. The model proposes, the host disposes. Identity, provenance, execution, reward, retirement, and what the model is shown are host decisions.
3. The don't-know contract: an unverified claim is **never spoken**.
4. Retire, never delete. Every record carries a store-snapshot hash and a git revision.
5. Serve-time budget: a small model on a **12 GB consumer GPU**; VM calls in milliseconds; training on one 80 GB GPU for tens of minutes per round.
6. No self-play or self-judging shortcuts.
7. Nothing depends on another repository at runtime. Reusable pieces are **ported, never linked**. (Consequence: the repos are separately sellable units sharing a spec, not a dependency graph.)

### CRITICAL STATUS CAVEAT — do not get this wrong
The serving stack today runs a **third-party GGUF model (LFM2.5-2.6B) as a stand-in**, used **for prototyping only**. GRL has its own architecture which **has not been trained yet**. The stand-in's numbers are host-architecture evidence, **never CubbyLLM model results**. The 2B trunk drops in behind the same `Emitter` interface unchanged. Any business model that depends on the stand-in's model quality is invalid.

---

## 2. THE CENTRAL ECONOMIC THESIS (the founder's, to be stress-tested and quantified)
**Frontier LLMs bill you for thinking you cannot see.** Reasoning/thinking tokens are generated, metered and charged as output tokens, inside a black box. The user cannot audit them, cannot cache them, and pays again for the same reasoning on every similar query.

CubbyLLM moves thinking **out of the token stream and into an external VM**:
- A thought is a **CubeLang program**, executed in milliseconds, costing **zero tokens**.
- Once certified, a program is **kept forever** and reused at **zero marginal token cost**.
- **The harder the reasoning, the wider the gap** — a frontier model's cost grows with reasoning depth; a VM call's does not.
- This holds **whether hosted in the cloud or run locally** — in privacy-critical environments the same architecture runs air-gapped.

Your job includes **quantifying this** from the measured numbers below and the external pricing in §6. Show the arithmetic. Then attack it: where does it break down, and what would falsify it?

---

## 3. MEASURED RESULTS — THE EVIDENCE BASE

### 3.1 Verified reasoning — the flagship, and the strongest asset
- Claimed-answer precision **0.9923** at coverage **0.646**; with the lookup-first TripleIndex: precision **0.9930**, coverage **0.710**, 568 verified / 564 correct (vs 517/513 before). `[M]`
- Absent-role control: **341/341**, later **528/528**. `[M]`
- **Planted-fault catch across 4 fault classes: 1.000.** Counterfactual harvest: 517/517 verified chains → **1,899 corrupted hops planted, 1,899 caught, 0 escaped** (598 wrong-entity / 233 wrong-relation / 662 inverted-direction / 406 wrong-hop-order). `[M]`
- **Identical CoT numbers at 0 and at 100,000 distractors.** `[M]`
- WikiKG functional-hop chains: **300/300 VM-verified** to the one gold answer. `[M]`
- The 4 verified-but-wrong answers were **corpus-dedup collisions, not verifier leaks**. `[M]`
- Latency: repair budget 3→1 gave **45.9 ms/question vs 120.9 ms** with *identical* verified/correct/precision. Counterfactual harvest ~**262 ms/chain, 1,899 VM calls**. Wiki chain verification **87 ms/question**. `[M]`
- **The project's own honest framing:** raw accuracy is CoT 0.390 vs chase-only 0.724 — the pipeline **refuses** what it cannot verify. "When this pipeline claims, it is right." Coverage, not accuracy, is the weak axis.
- Epistemic caveat on record: "an execution engine's similarity measures binding **fidelity, not truth**" — every counterfactual catch was by symbol mismatch. `[M]`
- Caveat: an earlier τ_vm calibration **degenerated to a floor** (AUC 0.0) and was self-reported in the output JSON as `tau_vm_calibration_degenerate`; fixed in v2 with frame-size-conditional floors. This self-catch is itself a credibility asset.

### 3.2 Certified tools / skills without retraining
- **ToolForge**: the model writes a CubeLang program for a live task; the VM certifies it against the example; registered retire-not-delete. Measured v3→v4: decision **0.75→1.00**, compare **0.00→1.00**, chain **0.50→1.00** (8/8 each), and **24/24 on a held-out level-5 maze**. `[M]`
- v7: forge **1.00 / 1.00 / 1.00** (12/12 each). `[M]`
- The game (`cubby-man`): a 3D labyrinth where walls are learned **from refused moves**, and the agent **invents its own CubeLang power-moves** (`COMBO-AABA`, `KNIGHT`, `WARP`), VM-certifies them, and edits one program in play. Measured: JUMP invented at step 2, `COMBO-AABA` used 55×, **165 steps saved**. `[M]`
- `request_tool` meta-call: a request **can never grant itself network or file access** — it composes registered primitives only. `[M]` (data side shipped; host registry object pending)

### 3.3 Model-free semantic index (a standalone product surface)
- `FastWordEncoder` v4 static table, 60,151 words, 1.1 GB: **~33 µs/challenge**; the earlier table **~25 µs at 0.95× its live teacher's macro score**, vs the live MiniLM teacher's **8.9 ms** → **~350× faster**, with **no model in the path**. `[M]`
- 1M-document store encode, compact 80-byte form: **~9 min (543 µs/doc)** vs ~30 min for the batched teacher. `[M]`
- Retrieval at 1M docs, compact form: **hit@1 0.713 / hit@20 0.920**; dense form flat to 100k (0.978→0.973). `[M]`
- `TripleIndex` over **365,923 triples: builds in 16.5 s at 284 MB peak**; an equivalent cosine `FactStore` would need **14.7 GB**. Full wiki world (552,297 facts): **46 s / 613 MB**. `[M]`
- Inter-world delegation matches exhaustive flat search's 0.554 answer-found rate at **63 vs 3,783 comparisons = 60× cheaper**, 2% delegated. `[M]`
- Centroid routing vs max-over-axioms: 0.680 vs 0.589 macro at **~40× fewer comparisons**. `[M]`
- Margin gating: **92% precision on what routes; 87% of never-seen-domain challenges correctly sent to spawn.** `[M]`

### 3.4 Output-head / inference cost
- Full softmax head cost is **linear in V**: measured 7.8× for 8× vocab; **47.9 ms/token** at the real 32k×10240 shape. Retrieval shortlist floor: **~0.2–1.5 ms → a 32× win**. `[M]`
- V=1M fp32 codebook = **10–41 GB** → at large V the softmax bypass is not an optimization, it is the only way the head fits. `[X]` (explicitly extrapolated)
- **Bounded decode state vs KV cache:** TTT per-token cost **constant 0.253 ms**; KV-attention **0.545 ms → 210.4 ms from L=1k → 131k**; measured crossover **L ≈ 1,024**. State at 128k: **1.0 MB vs 536.9 MB**. `[M]`
- On a **real checkpoint** (d=512/L=8): decode state **1,575,427 floats ≈ 6.3 MB**, of which **1,572,864 are the three attention KV caches** and only **2,560 the five recurrent layers**. Honest accounting on record: "the hybrid's fork cost is the attention window, not the recurrence — ~6 MB, not ~10 KB." `[M]`
- Forking a decode state is **exact**: max|forked − independent| = **0.00e+00**. Branching: 96 vs 240 `step()`s for 4 branches. `[M]`
- LZW hypertoken compression on the real tokenizer (8 MB / 2.07M tokens): **22.2% step reduction free** per-document; **53.4→66.6%** with a persistent global dictionary. `[M]`

### 3.5 Memory / continual learning
- **Episodic recall past the attention window**, scale-verified on a ~0.98B-token D512/L8 hybrid: at len 1024, **58.3 / 58.3 / 41.7 / 100 / 100** by needle depth vs mem-OFF **16.7 / 16.7 / 8.3 / 91.7 / 100**; at len 4096 (8× the window) **41.7 / 33.3 / 41.7 / 58.3 / 100** vs a mem-OFF floor. Controls: shuffled-needle floor **0–8%** (below chance — it actively rejects wrong continuations); untrained model **16.7% = chance**. Net: **2–3.5× untrained chance, 4–7× the shuffled floor, holding at 8× the window.** `[M]`
- Windowed-attention hybrid vs pure recurrence: **98% vs 32%** at len 256. At 1024 the depth pattern **is** the window boundary. Honest limit on record: "**no bounded-state model does arbitrary-distance recall.**" `[M]`
- **The learning gate (H-A7) — safe continual learning, measured.** On a real 92k-step checkpoint, A100-40G, 3.28M-token deltas: at LR 3e-5 the gate **separates** — a replay arm PROMOTES (mix +0.005 vs threshold 0.010) while a code-only arm **REJECTS and names the offending source** (mix +0.059, 12/12 unseen sources +0.032…+0.074). Null delta: every paired difference exactly 0. Gate evaluation costs **37–50 s on an A100**; a 3.3M-token proposal 65–77 s — **comfortably nightly**. Decision: nightly LR = 3e-5. `[M]`
- **θ=f(c) (the central bet), toy scale:** hardened, task-0 MSE **0.062 → 0.060** across 6 sequential tasks vs a fixed-weight baseline's **0.060 → 1.778**. Wrong-context probe 1.87 vs right-context 0.06. `[M]`
- **The load-bearing caveats:** the *naive* hypernetwork without hardening ends at **2.734 — worse than the baseline**; hardening costs **4.1× per step at toy scale and grows with the number of protected contexts** — the "fast half" kill clause is **still open at scale** `[NM]`. And **context inference, not generation, is the hard part**: an unsupervised router **collapsed to 1 of 6 slots**; an online-supervised router decayed to 37% confidence — 23× and 18× the oracle. The fix is an **offline-pretrained, frozen-at-inference** router. `[M]`
- Zero-Forgetting Stability Benchmark (built here because it existed nowhere): **no candidate achieves zero forgetting.** SDM is the only order-agnostic method (15/16/15/16 by quartile); NLMS has the best capacity but forgets oldest-first. Replicated on real dated NYT keys (3.1× more correlated than random, costing every candidate 2–3× recall) — **the ranking survived unchanged.** `[M]`

### 3.6 Training economics
- **MFU pilot, Colab A100-80GB, real fwd+bwd+AdamW, bf16:** at the **true 2B shape** (D2048/L32, B8, S1024, compiled, no grad-ckpt): **40.3% MFU**, rising to **45.5% MFU with a FlexAttention sliding-window BlockMask**. At 150M the compiled hybrid reaches **98% of a transformer comparator's MFU**. With grad-ckpt at 2B the hybrid **beats** the transformer (26.2% vs 25.4%). Throughput **~18.3–18.6k tok/s at both 1k and 4k context** — sequence length is MFU-neutral above B×S ≈ 16k tok/step. `[M]`
- Token cache measured on disk: **17,744,604,983 tokens (17.74B)**; manifest hash pinned; **all 13 stored signatures byte-identical to a live recompute**; NSFW excluded structurally by an 11-file allowlist, not by filtering. `[M]`
- **Budgets** `[P]` off measured MFU: one epoch of the real 17.74B cache ≈ **101 H100-hrs**; Chinchilla-optimal ~40B ≈ 2.25 epochs ≈ **228 H100-hrs**, or ~100 B200-hrs, or ~724 A100-hrs. **⚠️ Note: the widely-quoted "82 H100-hrs / $164–246" figure is the SAME run at the stale 14.37B budget; 82 × 1.23 = 101. Quoting both as different scenarios is a diligence hazard.**
- `[NM]` The H100/B200 rows are **MFU-transfer estimates, not measured** — the runbook mandates re-running the pilot on the rented card in the first 15 minutes. Real-run overhead (data loading, eval, checkpointing) adds an estimated 5–15% wall-clock, not included.
- Tokenizer: BPE training cost **~9.9 s at BOTH 65,536 and 131,072 vocab** → the 65k→131k cost ratio is **1.00×**. Vocabulary size is not a cost lever. Final: byte-level BPE 128k with atomic opcodes. `[M]`
- Largest real runs so far: **151M params, 92,000 of 100,000 planned steps, stopped because Colab credits ran out**, reaching **bpc 0.90–0.94 (ppl ~20–24)** with coherent multi-domain generations `[M]`; and a 2B-shape run of **20,000 steps / ~123M tokens (~3.9 h)** reaching ~1.1 bpc, described in-source as "heavily undertrained." `[M]`
- Backbone bake-off in the real assembled model, matched budget, per-component seeds so the backbone is the only variable: **gru ppl 34.0 / MinGRU 34.8 at 25% fewer backbone params / bdh-Q=K 49.1 / softmax attn 48.0.** Gated recurrence beats attention by ~10% bpc at this scale. `[M]` `[NM]` toy scale, CPU; the parallel-scan throughput advantage is argued by construction, not measured at scale.

### 3.7 Consumer hardware — measured on an RX 6750 XT (12 GB, RDNA2, **no CUDA**)
- Stand-in 2.6B Q4 fully offloaded via llama.cpp Vulkan: **~1.5 s per emitted program**; batch-1 decode keeps the card **~85% busy** (memory-bound — the whole 1.7 GB of weights streams per token); **127 tok/s** (LFM2.5-2.6B) vs 86 tok/s (Qwen3-4B). `[M]`
- grilly's Vulkan backend initializes successfully on this card. `[M]`
- `[NM]` **The local GPU cannot train.** Every SFT round runs on a rented Colab A100. "The adapter lifecycle's *train* step cannot run where the need is detected." This is a real architectural gap.

### 3.8 Host-architecture evidence from the stand-in (NOT model results)
- Serve self-test, 25 chains, facts stripped, own retrieval over 2,942 facts: **gold 0.96, spoken 96% correct / 4% don't-know / 0% wrong** (up from 76/20/4 at v3). `[M]`
- **Measured interference — the θ=f(c) motivation:** one 2.6B carrying both program and talk data was measured to interfere: forge decision **1.00 → 0.17**; arithmetic **0.825 → 0.750 → 0.675** as chat volume grew. Each repaired by replaying the hurt family. The two-adapter split is the pre-registered test of whether context-selected parameters remove it. `[M]`
- Hygiene: **GSM8K test excluded** — 5,277 test-derived programs found and dropped from SFT data. `[M]`
- Sobering reality check on record: with the whole 45k-article wikikg mounted, SimpleQA gold is a graph entity for **19.9%** of 4,326 questions and lookup-reachable for **0.1% (3 of 4,326)**. "The store cannot move SimpleQA; the loop must." And **0/300 real `natural_questions` queries parse the grammar.** `[M]`

---

## 4. LICENSING — THE BIGGEST STRATEGIC PROBLEM
| Repo | License | Note |
|---|---|---|
| **CubbyLLM** (incl. the whole serve stack: brain, ledger, vault, ToolForge, game) | **Apache-2.0** | Boilerplate — copyright line never filled in |
| **cubelang** (the VM — the moat) | **MIT** | **No LICENSE file in the repo** |
| grilly | MIT | Public on PyPI; contractually pinned as independent general-purpose OSS |
| optimum-grilly (HuggingFace Optimum backend, CUDA-free `from_pretrained`/`generate`) | Apache-2.0 | Public on PyPI, alpha |
| opcode-vsa-rs (Rust VSA: MAP-Bipolar, AVX2, mmap index, LSH/Hamming ANN) | MIT | |
| **cubemind** | **BSL-1.1 → Apache-2.0 on 2030-03-18** | The ONLY repo with a working commercial-licensing posture (`licensing@grillcheese.ai`). And the architecture explicitly refuses to depend on it at runtime. |
| mowm | BSL-1.1 (no LICENSE file) | |
| **cubbyverse** (the interfaces/profiles/platform layer) | **NONE — all rights reserved** | Accidentally the right posture, undeclared |
| cubby-lm, cubby-concepts (the FPGA/silicon plan) | **NONE** | |

**The three facts to confront:** (1) the moat — the VM, the verified pipeline, the ledger, the vault, ToolForge — is the **most permissively licensed** part and is free for anyone to commercialize; (2) the only commercially-protected repo is the one the architecture refuses to depend on; (3) the strategic layers (`cubbyverse`, `cubby-concepts`) have **no license at all**.

Note: I-RAVEN results and `cubemind/reasoning/rule_detectors.py` are under **NeurIPS 2026 embargo**.

---

## 5. THE RISK REGISTER — assume a technical diligence finds ALL of this
1. **The 2B run has not happened.** The gate opened 2026-08-14, the corpus is pinned, the money has not been spent. All trunk economics are projection.
2. **`[N]` — a prior document projected a "$956M five-year valuation."** It is arithmetically self-consistent (12× a self-asserted $79.72M Y5 revenue) but **every driver is assumed**, the narrative says $58.9M EBITDA while the table says 74.93, and the table is malformed. The project's own standing instruction: "Do not use these specific figures in any real financial planning until CubbyLLM has an actual measured cost-per-query from a running prototype." **DO NOT USE.**
3. **`[N]` — the "100–1000× cheaper" claim** rests on $15,000 vs $15.00 per 1M queries and 700 W vs 20 W, with a retail API price treated as competitor TCO. Zero benchmark data. The citation for it in the hypothesis doc was **wrong and had to be corrected**. **DO NOT USE.**
4. **`[N]` — a ~1,900–2,250× energy claim** built by multiplying four independently-sourced factors that were never measured together. Verdict on record: "treat as ceiling, never a plan number." **DO NOT USE.**
5. **`[N]` — a 9.2× CAPEX / 9.8× OPEX comparison** downstream of custom silicon that does not exist. **DO NOT USE.**
6. **⚠️ THE I-RAVEN HAZARD.** The 86.1–90.3% I-RAVEN scores are **real numbers but are NOT a VSA result**: the evaluator reads ground-truth `Type`/`Size`/`Color`/`Angle` integers straight out of the dataset's XML annotations — **there is no perception** — and VSA participates only as a 0.5-weighted tiebreaker inside a bare `try/except: pass`. The 100% OOD benchmark file imports no VSA at all. The project's own words: "**Any public or fundraising claim of the form 'our VSA reasoning layer scores 86-90% on I-RAVEN' is not supported by this code**… about ten minutes of reading away for any reviewer doing diligence." **DO NOT USE.**
7. **The trunk is not a retrieval encoder** — mean-pooled trunk states route **worse than a random encoder** (−6.0 sd below its floor); a training-free character-trigram sketch beats it. Defensible (no contrastive objective in the loss) but quotable against the project.
8. **Four self-caught data-leakage instances**, including a **retracted** "quantization improves routing" claim whose exported artifacts now carry `retracted` fields inside the file, and "augmented" rows that were 3,187/4,000 paraphrase duplicates (inflating a metric 0.890 → an honest 0.735). Self-caught = credibility asset; but the error class is real.
9. **Software landmines:** grilly ships two functions named `blockcode_bind` with different semantics, one of which **silently returns the identity element and corrupts any bundle** (no test in grilly unbinds from a superposition); the cubelang strict verifier has **two known holes** where a silent stub passes `check --strict`; `cubelang ask` can return the question text in place of a label because strings decode through the symbol space; the MindForge torch port computes **θ = f(h), not θ = f(c)** — no external context signal crosses into it at all.
10. **`VISION.md`'s own cost table: four of six cost pillars have ZERO measured savings**, by the project's own accounting. (Tension: §3.4's 32× head win and 22% LZW reduction exist in the validation report but were never transferred into that table.)
11. Killed hypotheses (a credibility asset AND a burn-rate record): binding aux loss **falsified** — an untrained trunk is already a near-optimal VSA substrate (trained 11.5% vs untrained control 97.5%); chrono-init **killed** at 44× its kill threshold; sparse choice points **do not transfer** to real text (would branch at 70–90% of steps); interpolation compression **falsified**.
12. Solo founder. No revenue from this. Shopify migration consulting is the current cash source.

---

## 6. EXTERNAL MARKET FACTS `[EXT]` — all retrieved 2026-09-10
### GPU rental
Median on-demand **H100 $3.40/GPU-hr** (getdeploying tracker); RunPod SXM $3.49 / **community $2.69**, PCIe $2.89; Lambda $3.99; CoreWeave on-demand $6.16, **spot $2.46**; Vast.ai $1.73 (spot marketplace — unreliable as a single number). **A100 80GB median $1.77**; RunPod $1.00–1.59; CoreWeave spot $1.21. **B200 median $6.25**, lowest spot $3.12. Recommended modelling: **$2.50/H100-hr spot, $3.40–4.00 on-demand.**
### Frontier API pricing, $/M tokens (input/output)
gpt-5.6-luna **0.20/1.20** · gpt-5.6-terra 2.00/12.00 · gpt-6-astra 10.00/50.00 · Claude Haiku 4.5 1.00/5.00 · Sonnet 5 2.00/10.00 · Opus 5 5.00/25.00 · Gemini 3.1 Flash-Lite 0.25/1.50 · Gemini 3.8 Flash 0.75/3.75 (**rises to 1.50/7.50 on 2027-01-01**) · Together: Llama 3 8B Lite 0.14/0.14, gpt-oss-120B 0.15/0.60, Llama 3.3 70B 1.04/1.04. Batch APIs are 50% off at OpenAI, Anthropic and Google.
**Directly citable proof that residency carries a price premium: Anthropic charges a 1.1× multiplier for US-only data-residency inference; OpenAI applies a 10% uplift for regional residency on models released after 2026-03-05.**
Self-hosted anchor `[EXT, WEAK — vendor blog]`: Llama 4 8B on H100 SXM at batch=16 → **$0.03/M output tokens**; 70B batch=8 → $0.18; batch=1 → $0.73 (at $2.50/GPU-hr, 85% utilisation — the author concedes real systems run 40–70%). Stated break-even vs managed API: ~2–5M tokens/day.
### Market size — **the estimates disagree ~8×; cite the range or don't cite it**
Grand View SLM: $7.8B (2023) → $20.7B (2030), 15.1% CAGR. MarketsandMarkets SLM: $0.93B (2025) → $5.45B (2032), 28.7% CAGR. Grand View **Edge AI: $24.9B (2025) → $118.7B (2033), 21.7% CAGR** — the more defensible one.
### Who actually monetizes
- **Mistral — the one existence proof.** ~**$400M ARR** (Sacra, Jan 2026), up ~20× YoY; **$3.5B raised Jun 2026 at $20B**. Four channels including **enterprise subscriptions for on-premises deployments addressing data residency and regulatory requirements**. **60% of revenue from Europe, driven by enterprise and government deployments requiring data sovereignty.**
- **Liquid AI** — closest architectural competitor. $250M Series A led by AMD Ventures (Dec 2024). LFM2.5 (Jan 2026): 1.2B text model trained on **28T tokens**, open-weight, 2,975 tok/s prefill on Ryzen AI 9 HX 370; LFM2.5-350M measured at 140 tok/s decode on iPhone 17 Pro, 30 tok/s on a Raspberry Pi 5, 56–85 MB RAM. **Revenue: no source found.**
- **Ollama** — $65M Series B (Jul 2026), **8.9M monthly active developers**, no revenue disclosed. **Monetizes cloud inference, not local**: Pro $20/mo, Team $500/mo.
- **Phi (MIT), Gemma 4 (Apache-2.0), Apple AFM 3** — free. Apple's AFM 3 Core Advanced is a 20B sparse model activating 1–4B at a time, held in flash.
- **Conclusion:** among commercial small-model vendors, **only Mistral has verifiable large revenue, and it comes from sovereignty/on-prem deals — not from the model being small. The market pays for compliance and control, not parameter count.**
### The regulatory wedge
- **Québec Law 25** (in force 2023-09-22): AMPs up to the greater of **CAD $10M or 2% of worldwide turnover**; penal fines **$25M or 4%**; **statutory punitive damages of not less than $1,000 per claimant** where intentional conduct or gross fault caused harm (the real class-action exposure). **s.17: before communicating personal information outside Québec you must run a Transfer Impact Assessment** weighing sensitivity, purpose, protective measures, and **the legal framework of the destination State**, and may proceed only on a finding of adequate protection, with a written agreement incorporating it. **On-device inference makes s.17 inapplicable — no transfer, no TIA, no adequacy analysis of US law, no contract.** That is the sharpest sales artifact in this pack.
  - **Honest caveat:** the CAI's published framework starts at only **$1,000–$15,000** base for organisations, and **no headline AMP appears to have been issued**. Sell *exposure and legal-risk aversion*, never observed enforcement.
- **Bill C-36** (first reading 2026-06-15) would replace PIPEDA Part 1 with AMPs of the greater of **$10M or 3% of global revenue** and **require a privacy risk assessment for cross-border transfers**. **Pending legislation, not law.** AIDA died on prorogation Jan 2025 — Canada has **no comprehensive AI statute**; privacy law is the primary constraint and **public-sector procurement is becoming the de facto standard-setter**.
- **Official Languages Act s.25** pushes the bilingual obligation **down onto vendors**: a federal institution must ensure services provided **on its behalf** are available in either official language. A monolingual model cannot be the substrate for a federally-procured public-facing service. **But: Mistral, Cohere, Gemma and Apple AFM 3 all handle French — EN/FR is a qualification gate, not a moat.**
- **Canadian Sovereign AI Compute Strategy: $2B over five years** — up to $700M AI Compute Challenge, up to $1B public supercomputing, **up to $300M AI Compute Access Fund** for Canadian businesses purchasing compute. Budget 2025 added **CAD $925.6M** to sovereign AI infrastructure.
- **Published Canadian on-prem AI contract values in health/legal/defence/finance: NO SOURCE FOUND.** No bottom-up TAM anchor exists there. Do not invent one.
- Cohere–Québec MOU (2026-06-09) is **explicitly zero-dollar, "does not constitute a contract."** Intent, not budget.
### Non-dilutive funding, verified current
**NRC IRAP** — open; incorporated for-profit, ≤500 FTE; FY2024-25 disbursed **$393.1M to 3,136 SMEs**; median disclosed contribution **$75K**, typical first award $75K–$200K, up to 80% of salary costs, 75% total government stacking cap. **Mitacs Accelerate** — open, rolling; partner pays $7,500 → Mitacs matches → **$15,000** per 4–6 month intern unit; requires a university partner. **SR&ED** — expenditure limit raised **$3M → $6M**, up to **$2.1M refundable ITC** at 35%; taxable-capital phase-out $15M–$75M; **capital expenditures eligible again**. **Québec CRIC** — refundable, **20% base / 30% on up to $1M** above the exclusion threshold; covers salaries, subcontractors, capital, **and pre-commercialisation** activities in Québec. **Scale AI** — up to 40% of eligible expenses but **scoped to supply chains**. **CDAP is CLOSED** (2024-02-19). **Investissement Québec Programme Innovation — terms expired, unverified; do not model.**
### Funding climate
Carta Q1 2026: **AI startups took 50% of all pre-seed dollars**; $1M–$2.5M rounds → median **$15M cap**; $2.5M+ lead-led → $20–30M cap. **But Canada: Q1 2026 saw 61 startups raising ~$190M CAD, a 40% YoY decline; average seed round $3M CAD**; fundraising outside the top 5 Canadian VCs fell 90% over five years.
**Best comparable: AUI** (neuro-symbolic, "deterministic execution essential for regulated enterprise environments where certainty matters more than fluency") — **$20M bridge SAFE at a $750M cap, Nov 2025**, ~$60M total. **Solo-founder small-model comparables: no source found.**
### Efficiency-claim sanity check — **read before writing any efficiency slide**
- Mamba (arXiv 2312.00752, **preprint**): 5× higher inference throughput than Transformers, linear scaling. RWKV (arXiv 2305.13048, **preprint**): constant compute and memory at inference, scaled to 14B.
- **MinGRU — "Were RNNs All We Needed?" (arXiv 2410.01201, Feng/Tung/Ahmed/Bengio/Hajimirsadeghi, Mila/Borealis, preprint). THREE PROBLEMS YOUR DILIGENCE WILL FIND:** (1) the famous 175×/1324× speedups are **against classic GRU/LSTM, not Transformers** — and **against Mamba minGRU is at parity** (2.72 ms vs 2.71 ms at seq 512); (2) the paper reports **no decode-time measurements at all** — the constant-time-decode claim is architecturally sound but **unsupported by that paper; GRL needs its own benchmark**; (3) language modelling was done on **Shakespeare, 1,003,854 tokens, 3 layers, 384 dim**, and the authors explicitly say they only *hypothesise* it generalises. **There is no published evidence MinGRU holds at 2B. Going to 2B is genuinely novel — present it as risk AND differentiation, not settled science.**
- **On-device energy, peer-reviewed** (Scientific Reports, 2026-04-17, DOI 10.1038/s41598-026-45023-0): best traditional model ~0.88 accuracy at **0.0021 Wh** vs Qwen 2.5 72B at **34.15 Wh — over 16,000×**; the 7B used **one-eighth** the 72B's energy for 0.07 accuracy points. "High accuracy does not necessarily require high energy expenditure."
- **The counter-finding to know before an investor raises it** (arXiv 2505.09598, 30 models): GPT-4.1 nano 0.454 Wh vs o3/DeepSeek-R1 **33–39 Wh (70×)** — but **GPT-4o mini consumed MORE energy than GPT-4o** (2.106 vs 1.788 Wh) because of A100 vs H100 deployment. Conclusion: "**deployment infrastructure can overshadow model size.**" Frame the claim as "*small model on the user's already-powered device, with no datacenter PUE overhead*," never "small models use less energy."
- **A direct measured on-device-vs-cloud dollar/energy comparison for a ~2B model: NO SOURCE FOUND. Generating it would be a genuinely novel, citable asset.**

---

## 7. THE FOUNDER'S THREE STANDING QUESTIONS
**Q1. The business model.** A pre-seed, investor-facing model that is *provable*: every claim traceable to §3 (measured), §6 (cited), or an explicitly-flagged assumption with its costing basis. Wedge, ICP, pricing, unit economics, use of funds, 18-month milestones, moat, and what kills it.
**Q2. Capabilities.** You may propose new capabilities or products, not just packaging. Each must fit the seven invariants, name a metric, a baseline, a kill criterion and a cost.
**Q3. The "top 10" question — answer this honestly.** The founder wants a strategy to be **"in the top 10 of frontier models within 18 months."** Tell him whether that is achievable, on what definition, and what the credible version is. **If you think the literal version is impossible on this budget, say so and say why, with arithmetic** — then give the strongest achievable reframing (which benchmark or category could be *won outright*, what it would cost, and what winning it is actually worth commercially). A flattering answer here scores zero.
