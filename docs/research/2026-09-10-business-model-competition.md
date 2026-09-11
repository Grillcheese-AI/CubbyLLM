# CubbyLLM business-model competition — raw record
Five frontier models via OpenRouter, two rounds, total spend $0.24.
Round 1: full business-model submissions. Round 2: each model attacks the synthesis.

---


# ROUND 1 — SUBMISSIONS


---

## deepseek-v4-pro

# CUBYLLM — BUSINESS MODEL & STRATEGY

**Date:** 2026-09-10  
**Advisor:** [Redacted]  
**Founder:** GrillCheese Research Lab (GRL), Solo, Lévis, Québec  
**Format:** 7-section direct response to the standing questions

---

## 1. THE ONE‑SENTENCE THESIS

GRL sells **provably‑correct, zero‑egress AI reasoning for compliance‑bound enterprises**, exploiting CubbyLLM’s VM‑verified chain‑of‑thought to turn inference from an opaque, token‑billed black box into a auditable, cacheable, on‑premise compute task—the only architecture that simultaneously satisfies Québec’s Law 25 data‑sovereignty requirements and delivers hard‑guaranteed answer fidelity at a marginal cost that frontier APIs cannot match.

**Supporting measured evidence**  
- **[M]** Verified‑answer precision **0.9923–0.9930** at coverage up to 0.710; absent‑role control 528/528; planted‑fault catch **1.000** (1,899/1,899).  
- **[M]** The reasoning step costs **45.9 ms** per question (VM + retrieval) on a software budget that fits a **12 GB consumer GPU** with no CUDA; stand‑in program generation ~1.5 s on the same card.  
- **[M]** Certified CubeLang programs are stored, retired‑never‑deleted, and **re‑executed at zero additional tokens**—a program that cleared a 3‑hop contradiction on Tuesday clears it forever for any user.

---

## 2. THE WEDGE

### First market: Québec legal‑finance compliance teams

**Buyer’s job title:** Chief Privacy Officer / General Counsel at a mid‑sized Québec financial institution or law firm (50–500 fee‑earners).  
**Budget line:** Technology & compliance / legal‑operations software (typically CAD 50 k–150 k annually for point solutions).  
**Specific weekly pain:** They must answer due‑diligence questions on thousands of pages of client documents, contracts, and internal policies, but **Law 25 s.17** forbids exporting personal information to US‑hosted AI without a Transfer Impact Assessment—every query to a mainstream LLM is a potential CAD 10 M (2 % turnover) exposure.  
**What they do today:** Manual review by junior associates/analysts; keyword search; rule‑based document management. A few experiment with on‑prem Mistral, but **no system gives them a provable audit trail that the answer came from the document, not fabrication.**

**Why THIS wedge (and not the alternatives I rejected)**  

| Rejected alternative | Why |
|----------------------|-----|
| Selling a general small‑model API | Competing with free (Llama, Phi) and the Mistral empire without a regulatory moat is a race to the bottom. |
| Healthcare (e.g., HIPAA) | No equivalent of Law 25’s bright‑line transfer prohibition; US‑based AI already dominates. The Québec regulator is more aggressive on paper (AMPs) even if unenforced. |
| Developer‑tool / low‑code | Monetisation models (Ollama) are unproven; would require a different go‑to‑market muscle GRL doesn't have. |

**Bottom‑up justification (no fabricated TAM):** Québec has **~250 law firms with ≥10 lawyers** and **~50 credit unions / small banks**. Each faces a binary choice: use AI and risk sanctions, or don’t. A pilot with 3–5 design partners who handle sensitive family‑law, estate, or commercial‑loan documents would validate willingness to pay.  

**Cheap evidence‑gathering step (cost < CAD 2 k):** Offer a “Law 25 AI compliance workshop” to the Québec bar and the Autorité des marchés financiers; at the event, conduct a structured 15‑question survey on current AI usage, pain, and budget allocation. GRL already has the stand‑in prototype; the demo can run on a laptop with no network.

---

## 3. THE TOKEN‑COST ARBITRAGE — QUANTIFIED

### (a) Cost per verified multi‑hop answer

We compare a hosted CubbyLLM‑API answer (with the real 2 B trunk, once trained) against a frontier reasoning‑model API that charges for hidden thinking tokens. All API prices from §6.

**CubbyLLM path** (local or self‑hosted, amortised over queries):  
- Program generation (LM): ~100 output tokens. Using the §6 self‑hosted anchor cost for a small model on an H100: **$0.03/M output tokens** (conservative batch=1, 85 % util). The 2 B will be cheaper; we use the same number. ⇒ **100 tok × $0.00000003/tok = $0.000003**.  
- Input tokens: question + retrieved triples (≈1 000 tok) ⇒ **$0.03/M input** ≈ $0.00003.  
- VM execution + retrieval: **45.9 ms**; on a single H100 that can handle thousands of concurrent VM threads, the cost is negligible (< $0.0001/query).  
- **Total per verified answer ≈ $0.00014** (dominated by the LM forward pass).  
- If the same question is answered from the certified‑program cache, the LM is never called, and cost drops to **~$0.00001** (retrieval lookup + VM).

**Frontier reasoning‑model path** (e.g., Claude Opus 5, output $25/M):  
- Assumption **[ASSUMPTION]**: A multi‑hop verification‑grade answer requires the model to “think” for **600 output tokens** of internal reasoning, plus 100 tokens of final answer = 700 output tokens.  
  *(Baseline: the MMLU‑Pro “deep‑reasoning” tier commonly generates 500–1500 think tokens; 600 is conservative.)*  
- Cost: 700 tok × $25/1 M tok = **$0.0175**.  
- There is no cache; every query incurs the full reasoning token cost.

**Gap: 125× cheaper** ($0.00014 vs $0.0175) even before caching.

### (b) Break‑even point

The CubbyLLM API breaks even against a frontier API **from the very first query** when hosting is on existing H100 capacity and the user’s alternative is a paid reasoning API.  

If a firm must buy a dedicated GPU (e.g., RX 6750 XT, one‑time cost ~$400, electricity $0.10/kWh, 150 W = $0.015/h), the break‑even query volume versus buying $0.0175/query API is:

\[
\frac{\text{hardware cost}}{\text{API cost per query}} = \frac{\$400}{\$0.0175} \approx 22,860 \text{ queries}
\]

At 100 queries/day (realistic for a 50‑lawyer firm’s internal QA), the hardware pays for itself in **229 days**. With electricity (< $ 130/year), TCO is favourable over any 3‑year horizon.

**Where the VM path LOSES:**  
- **Single‑hop trivial factoids** where a frontier Mini model costs $0.0003/query. For those, the LM‑generation overhead of CubbyLLM (still token‑based) is comparable or slightly higher. But here the enterprise argument shifts from cost to *auditability/zero‑hallucination*—the VM never invents an answer, while a Mini model may.  
- **Queries that fall outside the retrieval corpus** and are refused → fallback to a frontier model (see (d)).

### (c) The reuse curve

Define \( C_0 = \$0.00014 \) (cost per virgin VM query), \( C_{\text{cache}} = \$0.00001 \) (cached hit). With repetition rate \( r \) (fraction of queries previously certified):

\[
C_{\text{avg}} = C_0 (1 - r) + C_{\text{cache}} \, r
\]

| \( r \) | Avg cost/query | vs frontier $0.0175 |
|---------|----------------|-------------------|
| 0 %    | $0.00014       | 125 × cheaper     |
| 20 %   | $0.000112      | 156 ×             |
| 50 %   | $0.000075      | 233 ×             |
| 80 %   | $0.000037      | 473 ×             |

**Required repetition rate** for a defensible commercial advantage? Even \( r = 0 \) is sufficient. In practice, legal‑due‑diligence document queries are **highly repetitive** (e.g., “What is the governing law?” across hundreds of contracts); \( r \) will naturally exceed 0.5.

### (d) Honest counter‑case: the refusal penalty

From §3.1, coverage = 0.710 → **29 % of questions are refused** (the system correctly says “I don’t know”). Assume those refusals fall back to a frontier reasoning API at $0.0175 each, while answered ones cost $0.00014. Effective cost per incoming query:

\[
0.71 \times \$0.00014 + 0.29 \times \$0.0175 = \$0.0000994 + \$0.005075 = \mathbf{\$0.00517}
\]

This is still **3.4 × cheaper** than $0.0175, and the refuge provides *correct refusals*—the frontier model will hallucinate on some of those 29 %. More importantly, the refusal itself is a **trust signal**; many enterprise buyers will prefer a system that admits ignorance over one that guesses.

**Price the worst‑case fallback cost if the entire 29 % goes to a $25/M‑output model:** $0.00517/q → still a win. If frontier models drop price to $1/M output (e.g., a Mini model), the fallback cost becomes $0.0007, making total $0.00031—still cheaper, and the audit advantage remains. The arbitrage is robust.

---

## 4. THE BUSINESS MODEL

### Revenue lines in priority order

**1. CubbyLLM On‑Premise Compliance Appliance (annual license)**  
- *What is sold:* A packaged Docker/VM image containing the full stack (cubelang VM, verified pipeline, tuned trunk, fact‑store builder) deployable on a 12 GB GPU‑equipped server (any vendor) behind the client’s firewall. Includes an encrypted audit ledger for every answer.  
- *Pricing:* **CAD 75 k/year for up to 50 seats** (derivation: half the cost of two junior associates doing manual review, based on Québec average salary + overhead ~CAD 80 k each; the appliance replaces ~30 % of their document‑QA work, saving > CAD 48 k, so the price captures 50 % of savings).  
- *Gross margin:* near 100 % (software only). The client supplies hardware; optionally GRL can pre‑load a certified appliance on a validated mini‑PC for + CAD 5 k one‑time.  
- *Sales motion:* Direct to general counsel/CPO via Québec law‑tech events and referrals. First 3 months:  
  - Month 1: identify 10 target firms, run the “Law 25 workshop” survey.  
  - Month 2: build a custom demo for 3 firms using their own anonymised document samples (free pilot, time‑limited).  
  - Month 3: close first paid PoC (CAD 15 k for a 3‑month evaluation license).  

**2. CubbyLLM Cloud API with Data‑Residency Guarantee**  
- *What is sold:* Hosted API on Canadian sovereign cloud (e.g., Sherweb, OVH, or a bare‑metal box in a Montréal colo) that never transfers data outside Québec. Per‑query pricing with proof of residency.  
- *Pricing:* **$0.0015 per verified answer** (3‑4 × the near‑zero self‑hosted cost, but 10 × cheaper than frontier reasoning APIs). Target SMB law firms that don’t want to manage hardware.  
- *Gross margin:* ~85 % after infrastructure (one H100 can serve > 100 concurrent queries/second).  
- *Sales motion:* Self‑serve sign‑up with Québec‑specific compliance landing page. First 3 months: launch wait‑list; onboard 20 beta testers; convert to paid.

**3. Professional Services & Model Fine‑Tuning**  
- Custom integration with firm’s document management system (iManage, NetDocuments), training on in‑house policy manuals. Priced at **CAD 200/h** (standard tech‑consulting rate), 40–80 h engagements.  
- This line is high‑touch but builds the design‑partner relationships that de‑risk the core product.

### What GRL does NOT sell (and why)
- **Generic LLM access:** CubbyLLM is terrible at creative writing, chat, and open‑ended conversation; attempting to compete with OpenAI/Anthropic would dilute the verified‑reasoning proposition and invite direct comparison on fluency, where the 2 B trunk loses.  
- **Standalone model for others to embed:** The value is in the stack, not the weights. Selling an Apache‑2.0 model would cannibalise the appliance revenue. The trunk remains proprietary, integrated, and distributed only as part of the CubbyLLM appliance or API.

### The Licensing Problem — Direct Answer

- **Current state (worst accident):** `cubelang` has **no LICENSE file** → all rights reserved by default. That is exactly the posture GRL needs. The MIT note in the table is aspirational/premature; fix it immediately by placing a **BSL‑1.1** (or a custom “source‑available, non‑commercial only, commercial license required”) in `cubelang`, `cubbyverse`, and the verified‑pipeline repos.  
- **Cost to change now:** Zero (no external contributors). Add LICENSE files today; any future public push is under that license. Existing MIT/Apache‑2.0 statements in `grilly`, `opcode-vsa-rs` are fine—they are not the moat.  
- **The cubemind BSL repo is irrelevant** because CubbyLLM does not depend on it at runtime; the real moat (VM + pipeline) must carry the restrictive license.  
- **If GRL waits:** a fork of `cubelang` could appear under a permissive license, and GRL would lose exclusive commercial rights. **Deadline: before the next public commit.**  
- **Recommended license architecture:**  
  - `cubelang`, `cubbyverse`, and `cubby-lm` (the future trained trunk) → **BSL‑1.1** (converts to Apache‑2.0 after 4 years) with a commercial‑use exception sold via `licensing@grillcheese.ai`.  
  - All other infrastructure (`grilly`, `opcode-vsa-rs`, `mowm` if used) → MIT or Apache‑2.0 to build developer goodwill and lower the barrier for integrations.

---

## 5. CAPABILITIES TO BUILD

### 1. Compliance Copilot: Document‑ aware, auditable Q&A
- **What it does:** A desktop application where a lawyer uploads a batch of contracts/policies, the system builds a local fact‑store using `FastWordEncoder` + `TripleIndex`, then answers natural‑language questions with VM‑certified reasoning, displaying the exact source clause and the certified CubeLang program. Every answer is signed into the ledger.  
- **Built from:** `FastWordEncoder` (§3.3), `TripleIndex`, verified CoT pipeline (§3.1), ledger, local stand‑in trunk.  
- **Invariants:** All seven; in particular, the VM is the only truth gate and every spoken answer is verified.  
- **Metric:** Answer‑correctness precision on a curated 500‑question legal‑document benchmark. **Baseline:** Manual junior review (time ≈ 15 min/question, precision 0.95). **Kill criterion:** Precision < 0.90 after 3‑week pilot. **Cost to demo:** CAD 5 k (one month of full‑stack dev + benchmark creation).  
- **Why pay:** Replaces 30 % of document‑review billable hours; eliminates Law 25 transfer risk; provides a regulator‑grade audit trail.

### 2. Enterprise Memory: Secure continual fine‑tuning with forgetting guarantee
- **What it does:** A service that ingests a client’s proprietary documents, runs the H‑A7 learning gate (§3.5) nightly to update the trunk **without degrading general knowledge or leaking across clients**. The model stays 100 % local, never leaves the appliance.  
- **Built from:** Learning‑gate measurements (§3.5), zero‑forgetting stability benchmark, trunk fine‑tuning infra.  
- **Invariants:** #2 (model proposes, host disposes), #4 (retire, never delete—each fine‑tuned delta is a retired version).  
- **Metric:** Per‑client knowledge gain on a held‑out 200‑question internal‑policy test vs. zero forgetting on a general benchmark. **Baseline:** Standard fine‑tuning loses 3–5 points on MMLU. **Kill criterion:** General‑knowledge drop > 1 % or client‑knowledge gain < 5 % after one month. **Cost to demo:** 80 H100‑hrs (≈ $200 spot) for a 3‑client simulation.  
- **Why pay:** Banks cannot risk a model that “learns” trade secrets from one client and regurgitates them to another; the gate gives a measurable safety guarantee.

### 3. Model‑Free Semantic Index as a Standalone Service
- **What it does:** Sells `FastWordEncoder` + `TripleIndex` (or a hosted endpoint) for ultra‑fast, private semantic retrieval without any neural model at inference time—targets firms that don’t want an LLM at all but need “Google‑like” internal search on sensitive documents.  
- **Built from:** §3.3 measured numbers: 350 × faster than MiniLM teacher, 1M‑doc retrieval in milliseconds.  
- **Invariants:** #7 (nothing depends on another repository at runtime—the index is self‑contained) and #3 (no fabrication—retrieved facts are pointers only).  
- **Metric:** hit@10 recall on the client’s own internal corpus vs. incumbent Elasticsearch. **Baseline:** Elasticsearch BM25 typically 0.3–0.4 on long‑form legal text; the index gives 0.92 hit@20 at 80‑byte encoding. **Kill criterion:** If retrieval precision on the first client pilot drops below BM25 (unlikely, but verifying ensures market fit). **Cost to demo:** CAD 2 k (a web‑interface wrapper).  
- **Why pay:** Immediate compliance lift—no model, no transfer, no hallucination.

**Ranking by (commercial value) / (cost to demo)**  
1. Compliance Copilot — highest value, direct use case, sells the full stack.  
2. Model‑Free Semantic Index — fast to demo, opens a second, lower‑risk revenue line, pulls clients into the ecosystem.  
3. Enterprise Memory — higher technical risk (gate needs scale validation), but massive lock‑in after adoption.

---

## 6. THE “TOP 10 FRONTIER MODEL IN 18 MONTHS” QUESTION — HONEST ANSWER

**Literal top‑10 general‑capability LLM (e.g., Chatbot Arena, MMLU‑Pro): impossible.**  
- Frontier models (GPT‑6, Claude Opus 5, Gemini 3.8) are trained on **10 T+ tokens** with trillions of parameters, costing **$50 M–$200 M** in compute.  
- GRL’s planned 2 B trunk, even Chinchilla‑optimal at ~40 B tokens, costs **228 H100‑hrs = ~$570** at spot rates (§3.6). The model will have an order‑of‑magnitude fewer parameters and tokens than any top‑10 contender.  
- The proficiency gap in open‑ended generation, world knowledge, and language coverage cannot be closed by architectural efficiency alone. **No bounded‑state model beats a 70 B+ Transformer on MMLU.**

**Strongest achievable reframing: #1 on a “Verified‑Reasoning & Hallucination‑Control” leaderboard.**  
- **Benchmark to dominate:** Create `Verified‑QA‑Bench`, a corpus of 5,000 multi‑hop, adversarial, and absent‑answer questions where the primary metric is **precision@high‑coverage** (or honest‑refusal rate) with **proven step‑by‑step auditability**.  
- **Evidence it’s winnable:**  
  - GRL already has precision **0.9923–0.9930** and planted‑fault catch **1.000** (§3.1).  
  - No frontier model publishes a comparable VM‑mediated guarantee; their “chain‑of‑thought” is opaque and often incorrect.  
  - The CubbyLLM architecture’s **refusal‑driven coverage** (0.710) is a feature, not a bug, in a truth‑centric benchmark.  
- **What it costs:**  
  - Training the 2 B trunk on the pinned 17.74 B‑token cache: **~101 H100‑hrs = $252** (spot).  
  - Building and curating the benchmark: ~60 h of skilled annotator time = $3 k.  
  - Running eval against the frontier APIs: ~$500 in API credits.  
  - Total ≈ **$ 3.8 k**. The real cost is the founder’s time to integrate the whole stack for a public leaderboard.  
- **Commercial value:**  
  - A “#1 in Verified Reasoning” badge (with reproducible, third‑party‑auditable evidence) directly supports the enterprise sales narrative—it removes the “but is it safe?” objection in regulated industries.  
  - It attracts the $2 B Canadian Sovereign AI Compute Strategy attention and opens non‑dilutive funding.  
  - It differentiates CubbyLLM from every other small‑model vendor who competes on speed/price; this is a **moat‑defining metric** that competitors cannot replicate without rebuilding their architecture.  

**If the founder insists on “top‑10 general model,” the honest advice is: kill the goal, or spend the 18 months raising $20 M and hiring a team to train a 70 B model—which would discard every uniqueness of CubbyLLM. The reframed goal is more valuable and achievable.**

---

## 7. WHAT KILLS THIS

### Top‑5 causes of death (probability × severity)

1. **The 2 B trunk fails to generate correct CubeLang programs at > 0.85 forge success rate.**  
   - *Evidence:* The stand‑in 2.6 B achieves forge 1.00 on level‑5 mazes (§3.2), but the real trunk, trained on only 17 B tokens, may underfit.  
   - *Leading indicator:* Forge decision accuracy on a standard 100‑task test set falls below 0.90 after 50 % of the training budget.  
   - *Pre‑registered decision:* Stop training and pivot to a hybrid mode where the small trunk proposes programs and a larger open‑source model (e.g., Llama 4 8B) acts as a mandatory verifier/editor—the VM still gate‑keeps. This retains the invariants but increases cost.

2. **No enterprise buyer will pay for “auditable AI” because they prefer no AI at all.**  
   - *Indicator:* After 12 demos/cold‑emails to qualified Québec firms, zero purchase orders.  
   - *Decision:* Fall back to the model‑free semantic index as a standalone product (capability #3) and license the VM/verified pipeline to AI platform vendors (e.g., Cohere for their on‑prem offering); the company becomes a licensor, not a full‑stack vendor.

3. **Law 25’s enforcement remains theoretical, erasing the urgency wedge.**  
   - *Indicator:* No CAI‑issued AML penalty > CAD 50 k by 2027‑06.  
   - *Decision:* Reframe the wedge around **data‑residency compliance for Canadian federal procurement** (where the bilingual requirement and Sovereignty Compute Strategy create spending) rather than pure legal risk.

4. **The solo founder cannot simultaneously build the product, sell, and raise funding.**  
   - *Indicator:* The 18‑month milestones slip by > 3 months, and no design partner signed within 6 months.  
   - *Decision:* Immediately bring on a co‑founder with enterprise‑sales experience in Québec legal/finance, offering 20 % equity pre‑seed. If impossible, license the IP to an established Canadian AI firm and take royalties.

5. **The certified‑program cache assumption breaks: enterprise queries are diverse, repetition rate < 5 %.**  
   - *Indicator:* Log analysis of a 3‑month pilot shows < 50 cached hits out of 10 k queries.  
   - *Decision:* The token‑cost arbitrage still holds (see §3d), but the “reuse forever” narrative weakens; pivot marketing to the first‑query cost advantage and the irreplaceable auditability.

### Weakness in my own proposal

The **Compliance Copilot** wedge bets on a very narrow, regulation‑driven market that may not exist at scale. The risk is that after an initial handful of Québec firms, the addressable market saturates quickly, and expansion to other jurisdictions (EU, California) requires rebuilt legal‑fact‑stores and possibly different bilingual handling.  

**Falsification criterion:** If after 9 months of focused Québec sales GRL has fewer than 5 paying customers, the wedge is unproven. The fallback is to widen to “any enterprise requiring provable no‑hallucination answers” (e.g., aerospace maintenance manuals, pharmaceutical quality docs) using the same core stack, but with a broader, less geographically‑anchored GTM. That would be riskier but might uncover a larger market.  

**My proposal survives if the first three design partners convert to full‑price annual licences within 6 months.**

---

## kimi-k2-thinking

**1. THE ONE-SENTENCE THESIS**

GRL sells certified, non-repudiable reasoning programs that run on consumer hardware, to legal/compliance officers in regulated Québec firms who must prove AI decision provenance under Law 25 and cannot use black-box LLMs.

- **Evidence**: Verified reasoning precision 0.9930 [M] with absent-role control 528/528 [M] and planted-fault catch 1.000 across 1,899 corrupted hops [M] — zero hallucination escapes.
- **Evidence**: ToolForge certifies programs at 1.00/1.00/1.00 [M] with retire-not-delete ledger [M] — every decision is keyed by program SHA + VM build.
- **Evidence**: Consumer GPU runs at 127 tok/s [M] with 12 GB VRAM [M] and no CUDA dependency [M] — air-gap deployable.

---

**2. THE WEDGE**

**Buyer**: Chief Privacy Officer at mid-market Québec financial services firms ($50M–$500M CAD revenue).

**Budget line**: Legal/compliance operational budget, "privacy risk mitigation" and "regulatory audit preparation."

**Pain**: Law 25 s.17 requires a Transfer Impact Assessment before any personal information leaves Québec. Every customer query touching PI currently requires manual legal review ($200–400/hr) or is blocked. Black-box LLMs cannot be used because reasoning tokens are unauditable cross-border transfers.

**Current solution**: Manual review or complete avoidance of AI for customer-facing PI-handling.

**Why this wedge**: It is the only segment where "zero tokens cross the wire" is a legal requirement, not a cost optimization. The regulatory penalty exposure (CAD $10M or 2% global revenue) dwarfs software cost, making ROI trivial. Consumer hardware is a constraint the buyer already accepts for air-gapped compliance.

**Rejected alternatives**:
- General enterprise SaaS: Competes with GPT-4.1 nano at $0.03/M tokens [EXT] without regulatory mandate; loses on cost.
- Consumer app: No revenue model (Ollama's 8.9M MAU → $65M Series B but no revenue [EXT]).
- Public sector: Bilingual requirement is table stakes, not a moat; no published contract values anchor TAM [EXT].

**Evidence to gather**: FOIA request to CAI for anonymized AMP docket data showing cross-border transfer violations by Québec financial firms. Cost: ~$500 in filing fees and 30 days.

---

**3. TOKEN-COST ARBITRAGE — QUANTIFY IT**

### Assumptions (explicit):
- Reasoning-LLM path: gpt-5.6-terra at $2.00/12.00 per M tokens [EXT]
- Program generation: 100 tokens (one-time per unique query type)
- VM execution: 45.9 ms/question [M]
- Coverage: 0.710 [M]; fallback rate: 29% [P]
- GPU: A100 at $1.77/hr median [EXT] = $0.0004917/s [P]

### (a) Cost per verified multi-hop answer

**VM Path (first query, program generation + execution):**
- Program generation: 100 tokens × ($2.00 + $12.00)/M = **$0.14** [P]
- VM execution: 45.9 ms × $0.0004917/s = **$0.0000226** [P]
- **Total: $0.1400226** [P]

**VM Path (cached program, subsequent queries):**
- VM execution only: **$0.0000226** [P]

**Reasoning-LLM Path (every query):**
- 5 hops × 200 tokens × $2.00/M = $0.0020 [P]
- 50 answer tokens × $12.00/M = $0.0006 [P]
- **Total: $0.0026** [P]

**Arbitrage at 1,000 reuses**: (0.14 + 1000×0.0000226) / (1000×0.0026) = **0.054×** → **18.5× cheaper** [P]

### (b) Break-even point

**Queries needed to amortize program generation:**
N = 0.1400226 / (0.0026 - 0.0000226) = **54.3 queries** [P]

**Reasoning depth crossover** (where VM wins on first query):
Depth = (0.1400226 - 0.0006) / 0.0004 = **348.6 hops** [P]
→ For chains >350 hops, even generating a fresh program is cheaper.

### (c) Reuse curve

Effective cost per query = (1-R) × $0.140 + R × $0.0000226 [P]

| Repetition rate R | Cost/query | vs Frontier $0.0026 |
|-------------------|------------|---------------------|
| 0%                | $0.1400    | 53.8× more expensive |
| 50%               | $0.0700    | 26.9× more expensive |
| 90%               | $0.0140    | 5.4× more expensive  |
| **98.0%**         | **$0.0026**| **Break-even** [P]   |
| 99%               | $0.0014    | 1.9× cheaper         |

**Required**: >98% query repetition for pure cost win.

### (d) Honest counter-case: coverage gap

**Hybrid cost** (71% VM + 29% frontier fallback):
C_hybrid = 0.710 × $0.0000226 + 0.290 × $0.0026 = **$0.000770/query** [P]

**Net arbitrage**: $0.0026 / $0.000770 = **3.4× cheaper** [P]

**But**: 29% of queries are refused. If each refusal costs $10 in customer support escalation [ASSUMPTION], effective cost = $0.000770 + 0.29×$10 = **$2.900770** [P] — arbitrage destroyed.

**Conclusion**: The arbitrage holds only if refused queries are low-value or if coverage can be improved beyond 0.710. The current coverage is the weak axis, not cost.

---

**4. THE BUSINESS MODEL**

**Revenue lines (priority order):**

1. **Certified Reasoning API** — per verified answer
   - **What**: REST API returning (answer, program SHA, VM build hash, certificate)
   - **Pricing**: $0.001/answer (4× cost markup) [P]
   - **Gross margin**: ($0.001 - $0.000770) / $0.001 = **23%** [P]
   - **Sales motion**: Direct outreach to 50 Québec financial CPOs via Law 25 compliance forums
   - **First 3 months**: Close 3 pilot deals at $5K/mo; integrate with their existing chat systems; map their PI-touching queries to grammar

2. **On-Premise Air-Gapped License** — annual fee
   - **What**: Full stack deployable on firm's hardware, no external network calls
   - **Pricing**: $75K/year base + $15K per certified skill [ASSUMPTION: 1.5× API revenue, justified by elimination of transfer risk]
   - **Gross margin**: ~90% (software margin) [ASSUMPTION]
   - **Sales motion**: Enterprise sales with security review; starts after 2 API pilots prove value
   - **First 3 months**: Package stack into Docker + hardware spec (RX 6750 XT or better); prepare security whitepaper

3. **ToolForge Skill Certification** — one-time fee
   - **What**: Model writes CubeLang program for live task; VM certifies; registered in ledger
   - **Pricing**: $5,000 per skill [ASSUMPTION: based on 40 hrs legal review × $125/hr]
   - **Gross margin**: ~95% after initial skill engineering [ASSUMPTION]
   - **Sales motion**: Upsell to API customers; certify their top 5 most-repeated query types
   - **First 3 months**: Convert measured arithmetic/maze skills into certifiable packages; deliver to pilot customers

**What NOT to sell:**
- **Raw model API**: Violates invariant #1 (VM only truth gate) and cannot compete with Together at $0.14/M tokens [EXT]
- **Consumer app**: No revenue model proven (Ollama's 8.9M MAU → $0 revenue [EXT])
- **Training data or fine-tuning services**: Invariant #6 (no self-play) prohibits selling model outputs as training data

**Why saying no matters**: Selling raw API commoditizes the moat and invites price competition; the value is certification, not generation.

**Licensing fix**:
Current state is fatal: the moat (VM, verified pipeline, ledger) is MIT/Apache; the only BSL repo (cubemind) is not runtime-dependent [M].

**Correct architecture**:
- **cubelang VM**: BSL-1.1 → Apache-2.0 on 2029-09-10 (3-year delay, not 2030) [P]
- **CubbyLLM serve stack**: Apache-2.0 (maintains OSS credibility)
- **cubemind**: MIT (no runtime dependency anyway) [M]
- **cubbyverse**: Apache-2.0 (fix "all rights reserved" ambiguity)

**Cost to fix now**: Legal audit $15K + 2 weeks engineering [ASSUMPTION] = **$20K** [P]

**Cost to fix later**: If competitor forks the MIT VM before relicense, moat is permanently lost; enterprise sales become impossible.

---

**5. CAPABILITIES TO BUILD**

**1. Law 25 Compliance Auditor**
- **What**: Automated Transfer Impact Assessment generator; ingests firm's data flow diagram, outputs TIA document with legal citations
- **Built from**: VM certification [M], retire-not-delete ledger [M], TripleIndex for regulation lookup [M]
- **Invariant**: #1 (VM only truth gate), #2 (host disposes), #4 (retire not delete)
- **Metric**: % of TIA sections auto-completed vs manual legal draft
- **Baseline**: 0% (fully manual today)
- **Kill criterion**: <50% sections auto-completable after 3 months development [ASSUMPTION]
- **Cost**: 200 GPU-hours × $1.77/hr = **$354** [P] + 1 month founder time
- **Why customer pays**: Direct regulatory mandate; saves $200–400/hr legal fees

**2. Certified Financial Calculator**
- **What**: Provably correct arithmetic for financial advice scenarios (returns, amortizations, tax calculations)
- **Built from**: ToolForge v7 [M], arithmetic skill [M], VM verification [M]
- **Invariant**: #1, #3 (don't-know contract), #5 (serve-time budget)
- **Metric**: Accuracy on 1,000 FINRA exam questions; interference resistance
- **Baseline**: GPT-4.1 nano arithmetic 0.825→0.675 under interference [M]
- **Kill criterion**: <95% accuracy OR interference drop >0.1 points [ASSUMPTION]
- **Cost**: 50 GPU-hours × $1.77/hr = **$88.50** [P] + packaging
- **Why customer pays**: Required for regulated financial advice; zero hallucination liability

**3. Episodic Client Memory**
- **What**: Recall client conversation history up to 4,096 tokens past attention window
- **Built from**: Episodic recall mechanism [M], bounded decode state [M], forking verification [M]
- **Invariant**: #5 (serve-time budget), #7 (no runtime dependencies)
- **Metric**: Recall accuracy at 4,096-token distance vs mem-OFF baseline
- **Baseline**: mem-OFF floor 0–8% (below chance) [M]
- **Kill criterion**: <40% recall (2× untrained chance of 16.7% [M]) [ASSUMPTION]
- **Cost**: 500 GPU-hours × $1.77/hr = **$885** [P]
- **Why customer pays**: Differentiated service; client feels "remembered" without privacy risk of cloud storage

**Ranking by value/cost**:
1. Certified Financial Calculator: 95% margin, table stakes, $88 cost
2. Law 25 Compliance Auditor: High value, medium cost, direct regulatory wedge
3. Episodic Client Memory: Medium value, high cost, differentiation only

---

**6. THE "TOP 10 FRONTIER MODEL IN 18 MONTHS" QUESTION**

**Arithmetic on literal goal**:
- Top-10 models require >$2M training budgets (Mistral raised $3.5B [EXT]; AUI raised $60M [EXT])
- GRL's measured budget: 228 H100-hrs = $570 at spot rates [P]
- GRL's actual resources: Solo founder, Shopify consulting cash, no outside funding [M]
- **Conclusion: Literally impossible.** The gap is 3–4 orders of magnitude in capital and 1–2 orders in team size.

**Credible reframing**:
GRL cannot win general capability benchmarks (MMLU, HumanEval) but can **create and win a new category**: **Verified Zero-Hallucination QA**.

**Target benchmark**: Constrained SimpleQA subset (financial entities only)
- **Current**: 0.1% lookup-reachable [M]
- **Goal**: 100% coverage on 500 curated financial entity questions
- **Cost**: 100 GPU-hours × $1.77/hr = **$177** [P] + 2 weeks curation
- **Evidence it's winnable**: Precision 0.9930 [M] and 1.000 fault catch [M] prove the verification layer works; coverage is the only gap.
- **Commercial value**: #1 in a category you define is credible if the category solves a real pain point (regulatory compliance). It becomes a sales artifact for the Law 25 wedge.

**Alternative**: Win **I-RAVEN** with honest VSA-only submission (fix the current evaluator to actually use VSA representations). Cost: 50 GPU-hours [ASSUMPTION]. Value: Low — I-RAVEN is not a buyer's benchmark.

**Recommendation**: Abandon "top-10 frontier" as a goal. Replace with **"top-3 in verified reasoning precision on regulated-domain QA"** within 12 months. This is achievable and commercially defensible.

---

**7. WHAT KILLS THIS**

**Ranked by probability × severity:**

1. **The 2B training run fails to converge** [M]  
   - Probability: 40% (first large-scale run for solo founder)  
   - Severity: 100% (no model, no product)  
   - **Leading indicator**: After 10k steps, validation loss plateaus above 1.5 bpc  
   - **Trigger**: Pivot to VM-only compliance tool; abandon model training; become a verification layer for existing models

2. **Founder execution velocity stalls** [M]  
   - Probability: 60% (consulting cash source splits attention)  
   - Severity: 70% (delayed milestones → missed funding windows)  
   - **Leading indicator**: >14 days between commits on `cubelang` core while Shopify workload increases  
   - **Trigger**: Raise $150K pre-seed on SAFE to hire first engineer; if no raise in 30 days, shut down

3. **CUDA dependency becomes unavoidable** [M]  
   - Probability: 30% (llama.cpp Vulkan backend degrades on new architectures)  
   - Severity: 80% (loses regulatory wedge)  
   - **Leading indicator**: Performance on RX 6750 XT drops >50% vs baseline [M]  
   - **Trigger**: Abandon consumer GPU strategy; focus on cloud-only sales to non-regulated sectors

4. **Frontier API price collapse** [EXT]  
   - Probability: 20% (competitive pressure in API market)  
   - Severity: 90% (cost arbitrage evaporates)  
   - **Leading indicator**: OpenAI announces 5× price cut on reasoning models  
   - **Trigger**: Shift value prop from cost to compliance-only; accept 10× smaller TAM

5. **License audit failure** [M]  
   - Probability: 25% (missing LICENSE in cubelang, no copyright in CubbyLLM)  
   - Severity: 60% (enterprise sales blocked during cleanup)  
   - **Leading indicator**: First enterprise security review flags license gaps  
   - **Trigger**: Halt sales; spend 1 month on legal cleanup; lose 2 quarters of pipeline momentum

**New risk not in §5**: **Token-cost inversion from model distillation.** If a competitor distills a frontier model into a 2B parameter model with 95% capability retention and offers it at $0.01/M tokens, GRL's 3.4× cost advantage becomes irrelevant. The moat must be compliance, not cost.

**Biggest weakness of my proposal**: I am selling a regulatory solution that has never been enforced at scale. The CAI's published AMP framework starts at $1,000–$15,000 base [EXT], and no headline penalty has been issued. I am asking buyers to pay $75K/year for risk aversion without evidence of actual fines.

**What falsifies it**: A public CAI ruling that on-device inference of US-trained models still requires TIA, or a court decision that "communication outside Québec" includes processing on hardware with US-origin software. Either collapses the technical-legal distinction my wedge depends on.

---

## glm-5

# CUBBYLLM BUSINESS MODEL — TECHNICAL STRATEGY DOCUMENT

---

## 1. THE ONE-SENTENCE THESIS

GRL sells verified, auditable reasoning to regulated Canadian enterprises that face statutory penalties for unauditable AI outputs and cross-border data transfers—packaged as on-premises software where the reasoning happens in a deterministic VM (milliseconds, zero tokens) rather than inside a billable token stream.

**Three strongest measured supports:**
- **0.9923 precision at 0.646 coverage** on verified multi-hop reasoning — when the system answers, it's right `[M]`
- **45.9 ms/question verified reasoning latency** — the VM path is fast enough for interactive use `[M]`
- **Québec Law 25 s.17**: cross-border transfer requires a written Transfer Impact Assessment weighing the destination state's legal framework; on-device inference makes s.17 inapplicable — no transfer, no TIA, no US-law adequacy analysis `[EXT]`

---

## 2. THE WEDGE

**Buyer:** Privacy Officer or CISO at a Québec enterprise with statutory disclosure obligations (financial services, healthcare, legal services) or federal institution with Official Languages Act compliance requirements.

**Budget line:** Compliance/legal risk mitigation — same bucket as DPO insurance, privacy impact assessments, data mapping tools.

**Pain in their week:** A cross-border data transfer request lands on their desk. Under Law 25 s.17, they must execute a Transfer Impact Assessment weighing the sensitivity of the data, the purpose, protective measures, and **the legal framework of the destination state** (typically US). This requires outside counsel, costs $5,000–$15,000 per assessment `[ASSUMPTION — legal market rates]`, and takes weeks. They either pay repeatedly for similar assessments, or they accept risk and hope no regulator notices.

**What they do today:** Pay lawyers to draft TIAs case-by-case, or approve transfers informally and accept the AMP exposure (up to CAD $10M or 2% of worldwide turnover, plus statutory punitive damages of $1,000+ per claimant for intentional conduct or gross fault `[EXT]`).

**Why THIS wedge:**
- The regulatory trigger is real and in force (unlike Bill C-36 which is pending)
- The penalty exposure is quantifiable and large enough to justify spend
- On-device inference is the *only* technical architecture that makes s.17 inapplicable — cloud providers with US data residency cannot offer this regardless of model quality
- Mistral's $400M ARR comes from exactly this use case (sovereignty/on-prem) `[EXT]`

**Rejected alternatives:**

| Alternative | Why rejected |
|-------------|--------------|
| General AI inference platform | Mistral, Cohere, OpenAI already serve this; no evidence small model size alone commands premium pricing |
| Developer tools (local LLM SDK) | Ollama has 8.9M MAU developers and no disclosed revenue `[EXT]` — developer attention ≠ monetization |
| Small model licensing | Gemma 4, Phi-4, Apple AFM 3 are Apache-2.0 and free; no market for paid small model weights |
| Federal government prime contractor | Procurement cycles are 18–36 months; solo founder cannot survive the timeline |

**Evidence gap:** §6 states "Published Canadian on-prem AI contract values in health/legal/defence/finance: NO SOURCE FOUND." This is real. To validate the wedge, GRL needs:

1. **Discovery step (executable in 2 weeks, <$500):** Cold outreach to 20 Québec privacy lawyers asking: "What do you charge for a Law 25 Transfer Impact Assessment, and how many did you bill last year?" This establishes the outside-counsel cost anchor and frequency.

2. **Validation step (executable in 4 weeks, <$2,000):** One pilot conversation with a Québec enterprise CISO facing an active cross-border request. Offer a free TIA-avoidance architecture review. If they say "we already solved this" or "our risk appetite is higher than you think," the wedge is weaker than claimed.

---

## 3. THE TOKEN-COST ARBITRAGE — QUANTIFYING IT

### (a) Cost per verified multi-hop answer

**CubbyLLM VM Path:**

| Step | Measured value | Source |
|------|----------------|--------|
| VM verification latency | 45.9 ms/question | `[M]` §3.1 |
| Model decode throughput | 127 tok/s on RX 6750 XT | `[M]` §3.7 |
| Answer length assumption | 50 tokens | `[ASSUMPTION]` |
| Decode time | 50 ÷ 127 = 0.394 s | `[P]` |
| Total time per answer | 0.046 + 0.394 = 0.44 s | `[P]` |

**Energy cost:**
- GPU power during inference: 212.5W `[ASSUMPTION — 85% of 250W TDP per §3.7 "85% busy"]`
- Energy per query: 212.5W × 0.44s = 93.5 Ws = 0.000026 kWh `[P]`
- Québec electricity rate: ~$0.10/kWh `[ASSUMPTION — residential rate]`
- Energy cost per query: $0.0000026 `[P]`

**Hardware amortization:**
- RX 6750 XT street price: ~$400 CAD `[ASSUMPTION]`
- Useful life: 3 years = 26,280 hours
- Assumed utilization: 10% = 2,628 active hours
- Cost per active hour: $400 ÷ 2,628 = $0.152/hr `[P]`
- Cost per query: $0.152 × (0.44 ÷ 3600) = $0.000019 `[P]`

**Total CubbyLLM cost per verified answer:**
- Energy + hardware: $0.0000026 + $0.000019 = **$0.000022** `[P]`

---

**Reasoning LLM Path:**

Assumptions for a multi-hop question:
- Input context: 500 tokens `[ASSUMPTION]`
- Reasoning/thinking tokens: 500 `[ASSUMPTION]` — this is the hidden billing
- Output answer tokens: 50 `[ASSUMPTION]`

**Using gpt-5.6-luna (frontier cheap tier):**
- Input: 500 × $0.20/M = $0.00010 `[P]`
- Output: 550 × $1.20/M = $0.00066 `[P]`
- Total: **$0.00076 per query** `[P]`

**Using gpt-6-astra (reasoning model):**
- Input: 500 × $10/M = $0.0050 `[P]`
- Output: 550 × $50/M = $0.0275 `[P]`
- Total: **$0.0325 per query** `[P]`

---

**Arbitrage ratio:**
- vs gpt-5.6-luna: $0.00076 ÷ $0.000022 = **35×** `[P]`
- vs gpt-6-astra: $0.0325 ÷ $0.000022 = **1,477×** `[P]`

**Sensitivity to reasoning token count:**

| Reasoning tokens | gpt-5.6-luna cost | gpt-6-astra cost | Arbitrage (luna) | Arbitrage (astra) |
|------------------|-------------------|-------------------|------------------|-------------------|
| 100 | $0.00030 | $0.015 | 14× | 682× |
| 500 | $0.00076 | $0.0325 | 35× | 1,477× |
| 1,000 | $0.00136 | $0.0575 | 62× | 2,614× |
| 2,000 | $0.00256 | $0.1075 | 116× | 4,886× |

**Key finding: The arbitrage grows linearly with reasoning depth.** The harder the problem, the larger the gap.

---

### (b) Break-even analysis

**Where CubbyLLM WINS:**
- Any query requiring >100 reasoning tokens: wins against gpt-6-astra
- Any query requiring >500 reasoning tokens: wins against gpt-5.6-luna
- Multi-hop KGQA (measured 87 ms/question for Wiki chains `[M]`): clear win

**Where CubbyLLm LOSES:**

1. **Simple single-hop queries**: If reasoning tokens < 50, the VM overhead (45.9 ms) may exceed the savings. A frontier model answering directly might be cheaper.

2. **Queries outside coverage**: CubbyLLM refuses 35.4% of questions. This triggers fallback cost.

3. **High-throughput batch processing**: If the use case is bulk document classification where verification isn't required, the VM adds latency without value.

**Break-even reasoning depth:**

Against gpt-5.6-luna:
- CubbyLLM cost: $0.000022
- Break-even: $0.000022 = tokens × $1.20/M
- tokens = $0.000022 × 1M ÷ $1.20 = **18 tokens**

This is wrong — I need to include the output tokens. Let me recalculate:

Break-even: CubbyLLM cost = gpt-5.6-luna cost
$0.000022 = input_cost + reasoning_cost + output_cost
$0.000022 = 500 × $0.20/M + R × $1.20/M + 50 × $1.20/M
$0.000022 = $0.00010 + R × $0.0000012 + $0.00006
$0.000022 = $0.00016 + R × $0.0000012

This gives negative R, meaning gpt-5.6-luna is always more expensive than CubbyLLM for this query shape. But this ignores the coverage problem.

---

### (c) The reuse curve — certified program caching

**From §2:** "Once certified, a program is kept forever and reused at zero marginal token cost."

**Measured foundation:**
- ToolForge v7: 1.00/1.00/1.00 on decision/compare/chain (12/12 each) `[M]`
- Game power-moves: COMBO-AABA used 55×, saving 165 steps `[M]`

**Reuse model:**

Let r = cache hit rate (fraction of queries matching a certified program)

| Cache hit rate | Effective cost per query | Savings vs no-cache |
|----------------|--------------------------|---------------------|
| 0% | $0.000022 | baseline |
| 10% | $0.000022 × 0.9 + $0.0000026 × 0.1 = $0.000020 | 9% |
| 30% | $0.000022 × 0.7 + $0.0000026 × 0.3 = $0.000016 | 27% |
| 50% | $0.000022 × 0.5 + $0.0000026 × 0.5 = $0.000012 | 45% |
| 70% | $0.000022 × 0.3 + $0.0000026 × 0.7 = $0.000008 | 64% |

**What repetition rate is needed for enterprise use cases?**

This depends entirely on the domain. For:
- **FAQ-style compliance Q&A**: 30–50% repetition is realistic `[ASSUMPTION]`
- **Ad-hoc research queries**: <5% repetition `[ASSUMPTION]`
- **Standard operating procedure execution**: 60–80% repetition `[ASSUMPTION]`

**The reuse curve is linear.** Each 10% cache hit reduces cost by ~9%.

---

### (d) The honest counter-case — pricing the coverage gap

**From §3.1:**
- Coverage: 0.646 (later 0.710 with TripleIndex) `[M]`
- Precision: 0.9923 `[M]`
- Raw accuracy: 0.390 `[M]`
- Chase-only: 0.724 `[M]`

**The system refuses 29–35% of questions.** This is deliberate ("honest refusal") but has a cost.

**Fallback cost model:**

Assume fallback to gpt-5.6-luna for refused queries:

| Coverage | CubbyLLM portion | Fallback portion | Blended cost |
|----------|------------------|------------------|--------------|
| 0.646 | 0.646 × $0.000022 = $0.000014 | 0.354 × $0.00076 = $0.000269 | $0.00028 |
| 0.710 | 0.710 × $0.000022 = $0.000016 | 0.290 × $0.00076 = $0.000220 | $0.00024 |

**Comparison:**
- Pure gpt-5.6-luna: $0.00076
- CubbyLLM (0.710 coverage) + fallback: $0.00024
- **Savings: 68%** `[P]`

Against gpt-6-astra:
- Pure gpt-6-astra: $0.0325
- CubbyLLM (0.710 coverage) + fallback: 0.710 × $0.000022 + 0.290 × $0.0325 = $0.0094
- **Savings: 71%** `[P]`

**Critical finding: The fallback cost does NOT eat the arbitrage.** Even with 29% fallback to frontier models, CubbyLLM delivers 68–71% cost savings.

**But there's a worse scenario:**

What if the fallback model also can't answer? The user gets:
1. CubbyLLM refusal (verified uncertainty)
2. Frontier model attempt (unverified, possibly wrong)

This is actually the honest path: "We couldn't verify this. Here's an unverified attempt from a frontier model. Use at your own risk."

For compliance use cases, this is preferable to hallucination. For general chat, it's a poor experience.

**This is why the wedge must be compliance-focused.**

---

## 4. THE BUSINESS MODEL

### Revenue Line 1: On-Premises Deployment License

**What's sold:** Perpetual or annual license to deploy CubbyLLM stack on customer-owned hardware (air-gapped or network-isolated).

**Pricing derivation:**
- Mistral enterprise subscriptions: no public pricing `[EXT]`
- But Mistral's $400M ARR with 60% from Europe `[EXT]` suggests ~$240M from sovereignty deals
- Assume 200–500 enterprise customers `[ASSUMPTION]`
- Implied ACV: $480K–$1.2M per enterprise `[P]`

**GRL pricing (year 1):**
- Small enterprise (<500 employees): $50K/year `[ASSUMPTION]`
- Mid-market (500–5000 employees): $150K/year `[ASSUMPTION]`
- Large enterprise (>5000 employees): $300K/year `[ASSUMPTION]`

**Gross margin:**
- Zero marginal cost per query (customer-owned hardware)
- Support burden: 10–20% of revenue `[ASSUMPTION]`
- Gross margin: **80–90%** `[P]`

**Sales motion:**
- Direct sales to Privacy Officers/CISOs
- Pilot: 30-day free deployment on customer hardware
- Close: 90-day cycle for small/mid-market, 180-day for large
- Channel: Partner with Québec privacy law firms (they already advise on TIAs; GRL becomes their technical implementation)

---

### Revenue Line 2: Integration Services (Year 1 only)

**What's sold:** Deployment assistance, knowledge base indexing, custom verification programs.

**Pricing:**
- $15K–$50K per engagement `[ASSUMPTION]`
- 3–5 engagements in first year `[ASSUMPTION]`
- Revenue: $45K–$250K total `[P]`

**Gross margin:**
- Founder time (sweat equity)
- Actual margin: 100% cash, but opportunity cost

**Why include this:**
- De-risks the license sale (customer doesn't have to figure out deployment)
- Builds case studies
- Should be transitioned to partners by Year 2

---

### Revenue Line 3: Annual Support & Maintenance

**What's sold:** Updates, security patches, VM spec updates, verification ledger integrity checks.

**Pricing:**
- 20% of license fee per year `[ASSUMPTION — industry standard]`
- Small enterprise: $10K/year
- Mid-market: $30K/year
- Large enterprise: $60K/year

**Gross margin:**
- 70–80% `[ASSUMPTION]`

---

### First Three Months of Execution

| Month | Activity | Target outcome |
|-------|----------|----------------|
| 1 | Fix licensing (see below) | cubelang under BSL-1.1, cubbyverse under BSL-1.1 |
| 1 | Build demo: Law 25 compliance scenario | One recorded demo showing TIA avoidance |
| 2 | Outreach to 20 Québec privacy lawyers | 5 responses, 1 warm intro to enterprise |
| 2 | Apply for NRC IRAP | $75K–$200K non-dilutive |
| 3 | First pilot conversation | Deploy on prospect hardware (their GPU) |
| 3 | First case study draft | Document the pilot outcome |

---

### What GRL Does NOT Sell

| Not sold | Why |
|----------|-----|
| Cloud inference API | Competes with OpenAI/Anthropic/Google on their terms; no margin advantage; Law 25 doesn't apply if GRL hosts |
| Model weights | Gemma 4, Phi-4, AFM 3 are free; no market for paid weights |
| Developer tools | Ollama has 8.9M MAU and no revenue; developer attention is not monetizable directly |
| Training services | The value is in the verified reasoning, not in teaching customers to build their own |
| General-purpose chat | 64.6% coverage is unacceptable for general chat; compliance is the only defensible wedge |

**Saying no matters because:** Every "no" is a resource allocation decision. A solo founder cannot pursue multiple go-to-market motions simultaneously.

---

### The Licensing Problem — Direct Address

**From §4:**
- cubelang (the VM, the moat): MIT, no LICENSE file
- cubbyverse (interfaces/platform): no license
- cubemind: BSL-1.1 → Apache-2.0 2030-03-18, but architecture refuses to depend on it at runtime

**The three facts to confront:**
1. The moat is the most permissively licensed part
2. The only commercially-protected repo is the one the architecture refuses to depend on
3. The strategic layers have no license at all

**Correct license architecture:**

| Repo | Current | Should be | Why |
|------|---------|-----------|-----|
| cubelang | MIT (no file) | **BSL-1.1** | This is the moat. Anyone can commercialize it today. |
| cubbyverse | None | **BSL-1.1** | Platform layer, strategic value |
| grilly | MIT | MIT | Infrastructure, not moat; keeping permissive encourages adoption |
| optimum-grilly | Apache-2.0 | Apache-2.0 | HuggingFace integration, ecosystem value |
| opcode-vsa-rs | MIT | MIT or BSL-1.1 | Depends on whether this is sold separately |
| cubemind | BSL-1.1 | BSL-1.1 | Already correct, but irrelevant because architecture doesn't depend on it |
| mowm | BSL-1.1 (no file) | BSL-1.1 with LICENSE file | Fix the missing file |
| cubby-lm, cubby-concepts | None | BSL-1.1 or proprietary | Future silicon plans, protect now |

**Cost to fix now:**
- Time: 2 hours to add LICENSE files and update headers
- Money: $0
- Risk: Zero — no one is using this commercially yet

**Cost to fix later:**
- If a competitor commercializes cubelang before license change: **Moat eliminated**
- If GRL gains users on MIT terms then changes to BSL: **Community revolt, fork risk**
- After first enterprise deployment: **Contract renegotiation with every customer**

**Recommendation:** Fix this in Week 1. This is the single highest-ROI action available.

---

## 5. CAPABILITIES TO BUILD

### Capability 1: Law 25 Compliance Pack

**What it does:** Pre-built CubeLang verification programs for common Law 25 compliance scenarios: data subject access requests, retention policy enforcement, cross-border transfer logging.

**Built from:**
- §3.1 verified reasoning pipeline (45.9 ms/question) `[M]`
- §3.3 TripleIndex (365,923 triples, 16.5 s build) `[M]`
- §3.2 ToolForge certified program registration `[M]`

**Invariant:** Lives under Invariant 1 (VM is the only truth gate) and Invariant 3 (don't-know contract).

**Metric:** Time to complete a mock data subject access request.
**Baseline:** Manual process with lawyer review: 2–4 weeks `[ASSUMPTION]`
**Target:** <1 hour automated with verification certificate
**Kill criterion:** If verification precision drops below 0.95 on compliance scenarios, abandon.
**Cost:** 20 H100-hours for program library development + 40 hours founder time `[ASSUMPTION]`

**Why customer pays:** Reduces legal spend and AMP exposure. The verification certificate is auditable evidence of compliance.

---

### Capability 2: Bilingual Knowledge Index

**What it does:** TripleIndex deployed with French/English parallel corpus, verified for Official Languages Act s.25 compliance.

**Built from:**
- §3.3 TripleIndex (552,297 facts, 46 s build, 613 MB peak) `[M]`
- §3.3 retrieval metrics (hit@1 0.713, hit@20 0.920) `[M]`

**Invariant:** Lives under Invariant 7 (no runtime dependencies).

**Metric:** French query precision on bilingual corpus.
**Baseline:** Unverified: unknown
**Target:** 0.90+ precision on French queries
**Kill criterion:** If French precision is <0.80, the bilingual claim is false and federal procurement is blocked.
**Cost:** 50 H100-hours for French evaluation + 20 hours founder time `[ASSUMPTION]`

**Why customer pays:** Federal institutions must offer services in both official languages. A monolingual model cannot be the substrate for public-facing services. This is a qualification gate, not a differentiator — but Mistral, Cohere, Gemma, and AFM 3 all handle French, so this is table stakes.

---

### Capability 3: Audit Trail Export

**What it does:** Formatted export of verification ledger for regulatory audits: every decision, program SHA, VM build, and outcome in a regulator-friendly format.

**Built from:**
- §1 signed certificate ledger (SQLite, HMAC-SHA256, program SHA + VM build) `[M]`
- §3.1 verified reasoning outcomes `[M]`

**Invariant:** Lives under Invariant 4 (retire, never delete).

**Metric:** Time to produce a complete audit package.
**Baseline:** Manual log extraction: 4–8 hours `[ASSUMPTION]`
**Target:** <5 minutes automated
**Kill criterion:** If export format is rejected by a privacy regulator, redesign.
**Cost:** 10 hours founder time `[ASSUMPTION]`

**Why customer pays:** Law 25 requires demonstration of compliance. The audit trail is the product.

---

### Capability 4: Coverage Expansion Program

**What it does:** Systematic improvement of verified reasoning coverage from 0.646 → 0.85+.

**Built from:**
- §3.1 verified reasoning pipeline `[M]`
- §3.5 continual learning gate (LR 3e-5 separates replay from novel) `[M]`

**Invariant:** Lives under Invariant 6 (no self-play or self-judging shortcuts).

**Metric:** Coverage at fixed precision 0.99.
**Baseline:** 0.646 `[M]`
**Target:** 0.85
**Kill criterion:** If coverage improvement requires precision sacrifice below 0.95, the trade-off is unacceptable for compliance use.
**Cost:** 200 H100-hours for training iterations + 100 hours founder time `[ASSUMPTION]`

**Why customer pays:** Each 10% coverage improvement reduces fallback cost by ~$0.00005 per query. At 1M queries/year, that's $50/year — negligible. But coverage is also UX: 85% feels usable; 65% feels broken.

---

### Capability 5: Self-Hosted Dashboard

**What it does:** Web UI for monitoring on-prem deployment: query volume, cache hit rate, verification success/failure, ledger integrity.

**Built from:** New code, no existing component.

**Invariant:** Lives under Invariant 7 (no runtime dependencies on external repos).

**Metric:** Dashboard load time.
**Baseline:** None (doesn't exist)
**Target:** <2 seconds
**Kill criterion:** If dashboard requires cloud connectivity, it violates the air-gapped promise.
**Cost:** 80 hours founder time + 5 H100-hours for testing `[ASSUMPTION]`

**Why customer pays:** Enterprise buyers need visibility. A black box they can't monitor won't be procured.

---

**Ranking by (commercial value) / (cost to demo):**

| Rank | Capability | Value | Cost | Ratio |
|------|------------|-------|------|-------|
| 1 | Audit Trail Export | High (compliance necessity) | Very low | Highest |
| 2 | Law 25 Compliance Pack | High (core product) | Medium | High |
| 3 | Self-Hosted Dashboard | Medium (enterprise requirement) | Low | Medium |
| 4 | Bilingual Knowledge Index | Medium (qualification gate) | Medium | Medium |
| 5 | Coverage Expansion | High (UX) | High | Lower |

---

## 6. THE "TOP 10 FRONTIER MODEL IN 18 MONTHS" QUESTION

### The literal goal is impossible on this budget.

**Arithmetic:**

From §3.6:
- 17.74B tokens cached `[M]`
- One epoch at 2B shape: 101 H100-hrs `[P]`
- Chinchilla-optimal for 2B: ~40B tokens = 228 H100-hrs `[P]`

For top-10 general capability:
- Llama 3.1 405B trained on 15T+ tokens `[EXT — industry knowledge]`
- Mistral Large: trained on trillions of tokens
- GPT-4 class: estimated $100M+ training cost `[EXT — industry knowledge]`

To reach top-10, you need:
- Model size: 70B+ parameters (to compete with Llama 3.1 70B, Mistral Large, etc.)
- Training tokens: 1T+ for a 70B model (Chinchilla-optimal would be 1.4T)

**Cost calculation:**

For a 70B model at 1T tokens:
- Parameters scale: 70B / 2B = 35×
- Tokens scale: 1T / 40B = 25×
- Compute scales: 35 × 25 = 875× (roughly — actual scaling is more complex)
- H100-hrs: 228 × 875 = 199,500 H100-hrs `[X]`
- At $2.50/hr spot: **$498,750** `[X]`

This is compute alone. It excludes:
- Data pipeline engineering
- Multiple training runs (nobody succeeds on first try)
- Evaluation infrastructure
- Inference optimization
- Personnel

**GRL's measured budget:**
- Colab credits ran out at 151M params, 92k steps `[M]`
- Total spent: negligible (free tier)
- Available runway: Shopify consulting income (unknown, but solo founder in Québec suggests <$100K/year discretionary)

**Conclusion: A top-10 general-capability frontier model requires $500K–$2M in compute alone. GRL has not spent this and has no stated path to raising it. The literal goal is impossible.**

---

### The strongest achievable reframing

**What GRL can actually win: Verified reasoning benchmark — specifically, multi-hop knowledge graph question answering with verified chains.**

**Evidence:**
- WikiKG functional-hop chains: **300/300 VM-verified** to the one gold answer `[M]`
- Verified precision: **0.9923** at coverage 0.646 `[M]`
- Planted-fault catch: **1,899/1,899** across 4 fault classes `[M]`
- Absent-role control: **528/528** `[M]`

**This is not a mainstream benchmark.** It's a niche. But it's a niche where GRL has measured, reproducible, defensible superiority.

**What it costs to hold this position:**
- Maintain the VM verification pipeline: 20 H100-hrs/month for regression testing `[ASSUMPTION]`
- Expand coverage: 200 H100-hrs/year for training iterations `[ASSUMPTION]`
- Total: ~$6,000/year in compute at spot pricing `[P]`

**What it's commercially worth:**
- Not a consumer-facing claim
- Not a general AI benchmark
- But: for enterprise compliance use cases, "we can prove every answer we give" is the entire value proposition

**The honest reframing:**

> "In 18 months, GRL will hold the #1 position in verified multi-hop reasoning precision on structured knowledge graphs, with coverage above 0.85. This is not a general AI benchmark, but it is the benchmark that matters for auditable enterprise reasoning."

**What this position is worth:**
- It's the technical foundation for the compliance wedge
- It's not a marketing claim for general users
- It's a procurement criterion for regulated buyers who need to prove their AI's outputs

**If the founder wants a consumer-facing benchmark win:**
- That requires $500K+ in compute
- That requires a team (solo founder cannot iterate fast enough)
- That requires abandoning the verification focus to compete on fluency

**My recommendation: Abandon the "top 10 frontier model" framing entirely.** It's the wrong goal. The verification architecture is the moat. Competing on general capability abandons the moat to fight on terrain where OpenAI/Anthropic/Google have 1000× the resources.

---

## 7. WHAT KILLS THIS

### Cause of Death #1: License not fixed (Probability: 0.6, Severity: 10)

**Scenario:** A competitor (Mistral, Liquid AI, or a startup) notices that cubelang is MIT with no LICENSE file, forks it, adds their own model, and commercializes it. The moat — the VM, the verification pipeline, the ledger — is now shared infrastructure.

**Leading indicator:** A GitHub issue or PR on cubelang from someone outside GRL asking about commercial use.

**Pre-registered decision:** If any external entity expresses commercial interest in cubelang before BSL-1.1 is applied, GRL must choose: (a) immediate license change with clear communication, or (b) accept that the moat is now open water and pivot to services-only.

---

### Cause of Death #2: Coverage ceiling (Probability: 0.4, Severity: 9)

**Scenario:** The 0.646 coverage is structural, not fixable. The verification pipeline can't expand coverage above 0.75 without sacrificing precision. The system remains a "refuses a third of questions" curiosity.

**Leading indicator:** After 200 H100-hrs of training iterations, coverage remains below 0.75.

**Pre-registered decision:** If coverage < 0.75 after the Coverage Expansion Program, pivot from "verified reasoning product" to "verification layer for other models" — sell the VM as middleware, not the full stack.

---

### Cause of Death #3: Regulatory indifference (Probability: 0.3, Severity: 8)

**Scenario:** Québec's CAI continues to issue only $1,000–$15,000 fines. No headline AMPs are issued. Enterprises calculate that the expected penalty (probability × severity) is lower than the cost of compliance tooling. The urgency dissolves.

**Leading indicator:** 12 months of CAI enforcement actions with no fine exceeding $50,000.

**Pre-registered decision:** If regulatory enforcement remains weak, pivot from Québec to federal procurement (Official Languages Act) or to sectors with higher risk appetite (financial services, where OSFI guidance creates de facto requirements).

---

### Cause of Death #4: Frontier adds verification (Probability: 0.2, Severity: 10)

**Scenario:** OpenAI or Anthropic adds a "verified reasoning" mode where the model outputs a structured chain that can be audited. They have 100× the engineering resources. The differentiation vanishes.

**Leading indicator:** Any public announcement from a frontier lab about "auditable reasoning" or "verified outputs."

**Pre-registered decision:** If a frontier lab ships verified reasoning, GRL's only remaining differentiation is on-premises deployment. Double down on air-gapped, sovereign deployment where cloud providers cannot follow.

---

### Cause of Death #5: Solo founder burnout (Probability: 0.5, Severity: 10)

**Scenario:** The founder is running Shopify consulting to pay bills while building CubbyLLM. The cognitive switching cost, the isolation, the lack of peer review, and the sheer workload lead to exhaustion or a health crisis.

**Leading indicator:** Three consecutive months of <20 hours/week on CubbyLLM, or >60 hours/week total work.

**Pre-registered decision:** If burnout indicators appear, the founder must either: (a) raise funding to hire help, (b) find a co-founder, or (c) shut down gracefully. Solo heroics are not a strategy.

---

### Cause of Death #6 (NOT in §5): Market timing mismatch

**Scenario:** The Law 25 wedge is real, but the sales cycle is 12–24 months. The founder runs out of runway before the first enterprise deal closes. The product works, the regulation is real, the buyer exists — but the timeline exceeds resources.

**Leading indicator:** Month 9 with no signed pilot or letter of intent.

**Pre-registered decision:** If Month 9 arrives with no enterprise engagement, GRL must either: (a) raise a bridge round based on technical progress, (b) find consulting revenue that directly subsidizes CubbyLLM development, or (c) open-source everything and accept that the contribution is intellectual, not commercial.

---

### The single biggest weakness of MY proposal

**I am betting on regulatory enforcement that has not yet materialized.**

§6 states: "the CAI's published framework starts at only $1,000–$15,000 base for organisations, and no headline AMP appears to have been issued."

My entire wedge depends on enterprises perceiving Law 25 AMP exposure as existential. If the regulator continues to issue token fines, the urgency I'm selling doesn't exist.

**What would falsify this:**

- A single enterprise CISO saying: "We looked at Law 25 and decided our risk appetite is higher than the expected penalty. We'll pay the fine if it comes."
- A privacy lawyer saying: "TIAs are a profit center for us, but clients are pushing back on the cost. They'd rather accept the risk."

If either of these appears in discovery conversations, the regulatory wedge is weaker than I claim, and GRL must pivot to a different value proposition (auditability for its own sake, not regulatory compliance).

---

## gemini-3.6-flash

# CUBBYLLM: GROUNDED COMMERCIAL & TECHNICAL STRATEGY

---

## 1. THE ONE-SENTENCE THESIS

**GrillCheese Research Lab (GRL) sells an air-gapped, zero-token-reasoning execution stack—combining a deterministic Rust VM, a model-free semantic index, and a verified CoT pipeline—to compliance-constrained enterprises that cannot legally send data across borders or tolerate unauditable, hallucinated LLM outputs.**

### Supporting Evidence Base
1. **Unforgiving Precision & Fault Catching:** Achieves **0.9930 claimed-answer precision** at **0.710 coverage** (568 verified / 564 correct) `[M]` with **1.000 planted-fault catching** across 1,899 corrupted hops (0 leaks) `[M]` and identical CoT performance at 0 vs. 100,000 distractors `[M]`.
2. **Sub-100ms Verified Execution:** CoT chain verification runs in **45.9 ms/question** (87 ms/q on full WikiKG chains) `[M]`, replacing multi-second, token-metered neural generation with deterministic execution.
3. **Model-Free Indexing:** `FastWordEncoder` v4 encodes text in **~33 µs/challenge** (**~350× faster** than its MiniLM teacher at 0.95× quality) with zero neural model overhead `[M]`, building a 365,923-triple `TripleIndex` in **16.5 s at 284 MB RAM** (vs. 14.7 GB for cosine `FactStore`) `[M]`.

---

## 2. THE WEDGE

### Target Buyer & Market
* **Buyer Job Title:** Chief Information Security Officer (CISO) / Chief Risk Officer (CRO) / Data Protection Officer (DPO).
* **Target Segment:** Mid-market healthcare networks, legal-tech providers, and regional financial service firms in Québec.
* **Budget Line:** *Data Sovereignty & Privacy Compliance* or *Enterprise Software Risk Management*.
* **Specific Pain:** Under **Québec Law 25 s.17**, communicating personal information outside Québec requires a mandatory Transfer Impact Assessment (TIA) evaluating destination state legal frameworks `[EXT]`. Violations carry administrative penalties up to **CAD $10M or 2% of worldwide turnover**, criminal fines up to **$25M / 4%**, and **statutory punitive damages of at least $1,000 per claimant** without proving specific injury `[EXT]`. Sending user data to US-hosted APIs (OpenAI/Anthropic) triggers mandatory TIA exposure; US-data-residency add-ons charge a **10% pricing uplift** `[EXT]`.
* **Current Workaround:** Manual human review, blocking internal AI deployment entirely, or running unverified open-weight models (e.g., Llama 3 70B) on expensive, power-hungry local hardware without formal execution guarantees.

### Why This Wedge?
Deploying CubbyLLM on-premises or on air-gapped local hardware makes **Law 25 s.17 completely inapplicable**—there is no cross-border transfer, no TIA required, and zero exposure to foreign legal requests `[EXT]`. Furthermore, the VM’s **0.9930 precision** `[M]` satisfies regulatory requirements for auditability that neural black-boxes cannot meet.

### Rejected Alternatives
1. **Developer API for General Cloud SaaS:** Competing directly against OpenAI ($0.20–$2.00/M input) `[EXT]` and Together AI ($0.14/M) `[EXT]` on price/token leads to a margin collapse against subsidized hyper-scaler infrastructure.
2. **Open-Weight SLM Distribution:** Monetizing raw model weights fails. Competitors like Liquid AI ($250M raised) `[EXT]`, Ollama ($65M raised, monetizing cloud inference only) `[EXT]`, Gemma 4, and Phi offer open models for $0. Mistral proves that commercial small-model revenue (**~$400M ARR**) comes from **enterprise/sovereignty on-premises compliance (60% of revenue in Europe)** `[EXT]`, not selling raw weights.

### Bottom-Up Validation Step
No published Canadian on-prem AI contract values currently exist in public databases `[EXT]`. To validate pricing without inventing TAM figures:
* **Execution Step:** Conduct a 14-day direct outreach sprint to 20 Québec health/legal CISOs offering a free "Law 25 AI Exposure Audit." The metric to track is the conversion rate to a 30-day air-gapped pilot at **CAD $2,500/month**.

---

## 3. THE TOKEN-COST ARBITRAGE — QUANTIFY IT

### Baseline Operating Parameters
* **Frontier API Pricing Anchor (Mid-to-High Tier Reasoning Model):** Input: $2.00 / M tokens; Output: $10.00 / M tokens (e.g., Claude Sonnet 5 rates) `[EXT]`.
* **Standard Reasoning Query Profile `[ASSUMPTION]`:** 2,000 input context tokens; 4 reasoning hops generating 1,500 internal "thinking/CoT" tokens + 200 answer tokens = 1,700 output tokens.
* **CubbyLLM Hardware Cost Basis `[ASSUMPTION]`:** A local consumer GPU (RX 6750 XT 12GB, $400 capital cost amortized over 2 years = $0.023/hr) operating at 220W ($0.10/kWh electricity = $0.022/hr). Total operational cost = **$0.045/hour ($0.0000125/second)**. If hosted on a dedicated cloud L4 instance ($0.35/hr), operational cost = **$0.0000972/second**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      QUERY COST COMPARISON (USD)                        │
├──────────────────────────┬──────────────────┬───────────────────────────┤
│ Execution Path           │ Cost / 1k Queries│ Cost / 1M Queries         │
├──────────────────────────┼──────────────────┼───────────────────────────┤
│ Frontier API (Sonnet 5)  │ $21.0000         │ $21,000.00 `[P]`          │
│ CubbyLLM Uncached (Local)│ $0.000023        │ $23.28 `[P]`              │
│ CubbyLLM Uncached (Cloud)│ $0.000150        │ $150.27 `[P]`             │
│ CubbyLLM Cached (Local)  │ $0.00000057      │ $0.57 `[P]`               │
│ CubbyLLM Cached (Cloud)  │ $0.00000450      │ $4.46 `[P]`               │
└──────────────────────────┴──────────────────┴───────────────────────────┘
```

### (a) Cost Per Verified Multi-Hop Answer

#### Path A: Frontier Reasoning LLM API
* Input: $2,000 \times (\$2.00 / 1,000,000) = \$0.00400$
* Output: $1,700 \times (\$10.00 / 1,000,000) = \$0.01700$
* **Total Cost per Answer:** **$0.02100** ($21.00 per 1,000 queries) `[P]`.

#### Path B: CubbyLLM Uncached (First-Time Verification)
* 2.6B trunk emits CubeLang program (~190 tokens @ 127 tok/s = 1.50 s) `[M]`.
* Index lookup & VM chain execution: 45.9 ms = 0.0459 s `[M]`. Total latency = 1.5459 s.
* Local GPU Cost: $1.5459 \text{ s} \times \$0.0000125/\text{s} =$ **$0.0000193** per answer `[P]`.
* Cloud L4 Cost: $1.5459 \text{ s} \times \$0.0000972/\text{s} =$ **$0.0001502** per answer `[P]`.

#### Path C: CubbyLLM Certified & Cached (Program Re-Execution)
* SHA matched in SQLite signed certificate ledger `[M]`. Zero tokens emitted.
* Execution time: 45.9 ms (0.0459 s) `[M]`.
* Local GPU Cost: $0.0459 \text{ s} \times \$0.0000125/\text{s} =$ **$0.000000574** per answer `[P]`.
* Cloud L4 Cost: $0.0459 \text{ s} \times \$0.0000972/\text{s} =$ **$0.000004462** per answer `[P]`.

---

### (b) Break-Even Analysis & Structural Failure Modes

#### Cost Arbitrage Win
* On uncached queries, local CubbyLLM is **1,088× cheaper** per query than Frontier API ($0.0000193 vs. $0.02100) `[P]`.
* On cloud hosting, uncached CubbyLLM is **139× cheaper** ($0.0001502 vs. $0.02100) `[P]`.

#### Where CubbyLLM Loses
1. **Unstructured Casual Chat / Creative Writing:** Generating low-depth conversational text does not trigger CubeLang execution. A 50-token query yielding 50 tokens on Gemini 3.1 Flash-Lite ($0.25/M in, $1.50/M out) `[EXT]` costs **$0.0000875**. A local 2.6B model running at 127 tok/s `[M]` takes 0.39 s ($0.0000048 local compute), but provides strictly inferior conversational quality compared to a 70B+ model.
2. **Out-of-Grammar / Un-indexed Domains:** If facts are missing from `TripleIndex` and cannot be framed into 6-WH structures, CubbyLLM returns an honest refusal `[M]`.

---

### (c) The Reuse Curve
Let $r$ be the query repetition rate (fraction of incoming queries matching a certified program SHA in the ledger).

$$\text{Cost}(r) = (1 - r) \cdot C_{\text{uncached}} + r \cdot C_{\text{cached}}$$

* At $r = 0\%$: Cloud cost = **$0.0001502 / query** `[P]`.
* At $r = 50\%$: Cloud cost = **$0.0000773 / query** `[P]`.
* At $r = 90\%$: Cloud cost = **$0.0000190 / query** `[P]`.

**Impact:** Re-executing certified programs yields a **33.7× cost reduction** over uncached execution and a **4,706× cost reduction** over Frontier API calls `[P]`.

---

### (d) The Honest Refusal Counter-Case & Fallback Costing

Fact pack §3.1 establishes that claimed-answer precision is **0.9930** at **0.710 coverage** (raw accuracy CoT 0.390 vs chase-only 0.724) `[M]`. The system explicitly refuses **29.0%** of queries to preserve truth guarantees.

If an enterprise workload mandates **100% answer coverage**, all refused queries must fall back to a Frontier LLM (Sonnet 5 @ $0.02100/query) `[EXT]`.

#### Blended Cost Formula
$$C_{\text{blended}} = \text{Coverage} \cdot C_{\text{CubbyLLM}} + (1 - \text{Coverage}) \cdot C_{\text{Frontier API}}$$

#### Uncached CubbyLLM + Fallback ($r = 0, \text{Coverage} = 0.710$)
$$C_{\text{blended}} = 0.710 \cdot (\$0.0001502) + 0.290 \cdot (\$0.02100) = \$0.0001066 + \$0.0060900 = \mathbf{\$0.0061966 \text{ per query}} \quad \text{`[P]`}$$

#### Cached CubbyLLM + Fallback ($r = 0.80, \text{Coverage} = 0.710$)
$$C_{\text{CubbyLLM}} = 0.20 \cdot (\$0.0001502) + 0.80 \cdot (\$0.00000446) = \$0.0000336 \quad \text{`[P]`}$$
$$C_{\text{blended}} = 0.710 \cdot (\$0.0000336) + 0.290 \cdot (\$0.02100) = \$0.0000239 + \$0.0060900 = \mathbf{\$0.0061139 \text{ per query}} \quad \text{`[P]`}$$

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 BLENDED COST WITH 29% FALLBACK BACKSTOP                 │
├──────────────────────────┬──────────────────┬───────────────────────────┤
│ Workload Configuration   │ Blended Cost/Query│ Savings vs. Pure API     │
├──────────────────────────┼──────────────────┼───────────────────────────┤
│ Pure Frontier API        │ $0.021000        │ 0.0%                      │
│ CubbyLLM Uncached + API  │ $0.006197        │ 70.5% `[P]`               │
│ CubbyLLM Cached (80%)+API│ $0.006114        │ 70.9% `[P]`               │
│ CubbyLLM (Coverage 0.85) │ $0.003195        │ 84.8% `[P]`               │
└──────────────────────────┴──────────────────┴───────────────────────────┘
```

#### Economic Verdict
The 29% refusal fallback accounts for **99.6% of total system cost** ($0.006090 of $0.006114) `[P]`. However, because CubbyLLM resolves 71% of queries deterministically, **blended costs remain 70.9% lower than relying solely on frontier APIs** `[P]`. Expanding grammar coverage from 0.710 to 0.850 cuts blended cost almost in half again to **$0.003195/query** `[P]`.

---

## 4. THE BUSINESS MODEL

### Revenue Streams (Priority Order)

#### 1. Air-Gapped Enterprise Server License (Primary Stream)
* **What is Sold:** Annual node license for `cubbyverse` / `cubemind` / `cubelang` enterprise serve stack, including model-free `FastWordEncoder`, `TripleIndex` engine, and the HMAC-SHA256 Signed Certificate Ledger.
* **Pricing:** **CAD $30,000 / year per server node** (~$22,000 USD) `[ASSUMPTION]`.
* **Derivation:** A 50-user enterprise team issuing 100,000 reasoning queries/month to Sonnet 5 incurs **$25,200 USD/year** in raw API fees `[P]`. At $30k CAD flat, CubbyLLM breaks even at ~100k queries/month while eliminating Law 25 legal exposure ($1,000/claimant statutory damages) `[EXT]`.
* **Gross Margin:** **~92%** (Pure software delivery; ~$2,000/node support allocation) `[P]`.
* **First 3 Months Execution:** Package local installer scripts for RHEL/Ubuntu air-gapped environments; sign 2 regional pilot customers in Québec health/legal sectors.

#### 2. Domain Knowledge Graph Compilation & Certification
* **What is Sold:** Turnkey ingestion service converting client SQL schemas, documents, and private enterprise data into `TripleIndex` format (365k triples compiled in 16.5s) `[M]` with custom 6-WH domain grammar certification.
* **Pricing:** **CAD $15,000 per domain pack** `[ASSUMPTION]`.
* **Gross Margin:** **~85%** (10 hours founder/engineering setup time) `[P]`.
* **First 3 Months Execution:** Build automated SQL-to-TripleIndex ingestion scripts (`CubeData-Sync`).

#### 3. Sovereign Non-Dilutive Grant Stacking (Cash Flow Engine)
* **NRC IRAP:** $75,000–$200,000 initial grant funding (covers up to 80% of internal salary costs) `[EXT]`.
* **Québec CRIC:** 20% base / 30% enhanced tax credit on R&D expenses up to $1M, including pre-commercialization `[EXT]`.
* **SR&ED:** 35% refundable Tax Credit on eligible R&D labor `[EXT]`.
* **Mitacs Accelerate:** $15,000 per intern unit ($7,500 partner match) `[EXT]`.

---

### What GRL Does NOT Sell
* **No Multi-Tenant Hosted Consumer Cloud API:** Avoids datacenter PUE/egress capital consumption and competitive price wars against hyper-scalers.
* **No Raw Open-Weight Model Downloads:** Monetizing standalone small model weights is unviable (e.g. Liquid AI, Gemma, Phi) `[EXT]`.

---

### The Licensing Strategy

#### Current Vulnerability (§4 Fact Pack)
* `cubelang` (the VM moat) is licensed under **MIT with NO LICENSE file in the repo** `[M]`.
* `CubbyLLM` serve stack is licensed under **Apache-2.0** `[M]`.
* `cubemind` is BSL-1.1 converting to Apache in 2030, but the architecture explicitly refuses to depend on it at runtime `[M]`.
* `cubbyverse` (platform/interfaces) has **NO LICENSE** (all rights reserved) `[M]`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TARGET LICENSING ARCHITECTURE                       │
├────────────────────────┬──────────────────────┬─────────────────────────┤
│ Repository / Layer     │ License Type         │ Commercial Rationale    │
├────────────────────────┼──────────────────────┼─────────────────────────┤
│ grilly (Compute)       │ MIT / PyPI Public    │ Drives adoption & standard│
│ cubelang (VM Engine)   │ MIT (Spec & Core VM) │ Establishes open standard │
│ FastWordEncoder        │ Apache-2.0            │ Wide developer adoption │
│ cubbyverse Stack       │ BSL 1.1 (Business)   │ Monetizes Enterprise Ops│
└────────────────────────┴──────────────────────┴─────────────────────────┘
```

#### Corrective Action Plan
1. Keep `grilly` and the core `cubelang` VM specification MIT/Apache-2.0 to establish an open standard on PyPI.
2. Relicense `cubbyverse` (the serving stack, `FactStore` router, `TripleIndex` engine, and Signed Certificate Ledger) under **BSL 1.1 (Business Source License)**. Production use exceeding 1 node or commercial deployment requires an Enterprise License Key, converting to Apache-2.0 after 36 months.
3. **Refactoring Cost:**
   * **Now (Solo Founder Stage):** **$0 in IP litigation / $0 CLA friction.** The founder owns 100% of commits across all repositories. Relicensing takes < 1 hour.
   * **Later (Post-Investment / Multi-Contributor):** Requires 100% Contributor License Agreement (CLA) sign-offs or expensive clean-room code rewrites ($100k+ legal/eng cost).

---

## 5. CAPABILITIES TO BUILD

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEW CAPABILITIES ROADMAP (RANKED)                    │
├───────────────────┬──────────────────────────┬──────────────┬───────────┤
│ Capability Name   │ Core Component           │ Target Metric│ Est. Cost │
├───────────────────┼──────────────────────────┼──────────────┼───────────┤
│ 1. CubeData-Sync  │ FastWordEncoder/TripleIdx│ Throughput   │ $0 / 0 GPU│
│ 2. CubeAudit-Ledger│ SQLite Certificate Ledger│ Traceability │ $0 / 0 GPU│
│ 3. CubeRoute-Frozen│ FastWordEncoder Centroids│ Routing Prec.│ $37.50    │
│ 4. ToolForge-Local│ CubeLang Rust VM         │ Cert. Rate   │ $25.00    │
└───────────────────┴──────────────────────────┴──────────────┴───────────┘
```

### 1. `CubeData-Sync` — Model-Free Relational Database Indexer
* **Function:** Ingests enterprise SQL schemas/CSVs, converting primary/foreign keys into zero-neural `TripleIndex` facts at ~22,000 triples/sec (extrapolated from 365k triples in 16.5s) `[M]`. Makes enterprise SQL tables queryable via 6-WH CoT without vector databases.
* **Invariant Enforced:** Invariant 3 (Don't-know contract) & Invariant 7 (No external runtime dependencies).
* **Validation Target:**
  * *Metric:* Ingestion throughput (triples/s) & relational CoT accuracy.
  * *Baseline:* 22,177 triples/s build rate, 0.9930 precision `[M]`.
  * *Kill Criterion:* Relational structure corruption dropping precision below 0.980 or build rate below 5,000 triples/s.
  * *Cost:* $0 / 0 GPU-hours (CPU/RAM bound: 284 MB peak for 365k triples) `[M]`.
* **Why Customer Pays:** Replaces $50k/yr managed vector DB clusters (Pinecone/Weaviate) + embedding API fees with zero-RAM local lookup.

### 2. `CubeAudit-Ledger` — Compliance Certificate Exporter
* **Function:** Exports SQLite HMAC-SHA256 signed certificate ledger entries into immutable compliance logs (JSON-LD / W3C Verifiable Credentials format), tying every spoken answer to its exact VM program SHA, dataset snapshot hash, and git revision.
* **Invariant Enforced:** Invariant 4 (Retire, never delete) & Invariant 1 (VM is only truth gate).
* **Validation Target:**
  * *Metric:* Cryptographic verification latency & 100% provenance auditability.
  * *Baseline:* 100% SHA traceability on SQLite ledger `[M]`.
  * *Kill Criterion:* Verification latency > 5 ms per record or un-hashed state mutations.
  * *Cost:* $0 / 0 GPU-hours (20 hours founder CPU dev time).
* **Why Customer Pays:** Provides legal defense artifacts for Québec Law 25 regulatory audits.

### 3. `CubeRoute-Frozen` — Offline Pretrained Dual-Adapter Context Router
* **Function:** Replaces the online-supervised context router (which decayed to 37% confidence) `[M]` with an offline-pretrained, frozen-at-inference router operating over `FastWordEncoder` centroid representations to route between program-proposal and chat adapters without inference decay.
* **Invariant Enforced:** Invariant 5 (Serve-time budget) & Invariant 2 (Host disposes).
* **Validation Target:**
  * *Metric:* Context routing macro precision across program vs. chat domains.
  * *Baseline:* 37% confidence (decayed online router) `[M]`; Target: >95% macro precision.
  * *Kill Criterion:* Routing precision < 0.90 or added routing latency > 2.0 ms.
  * *Cost:* 15 A100 GPU-hours (~$37.50 at $2.50/hr spot) `[EXT]`.
* **Why Customer Pays:** Eliminates multi-task interference (where chat data drops forge decision scores from 1.00 to 0.17) `[M]`.

### 4. `ToolForge-Local` — Sandboxed Local Automation Runtime
* **Function:** Exposes host `request_tool` meta-calls in CubeLang so local scripts (e.g., local PDF parsing, database queries) are generated by the 2.6B trunk, VM-verified against synthetic unit tests, certified in the SQLite ledger, and executed under deny-by-default isolation.
* **Invariant Enforced:** Invariant 1 (VM is only truth gate) & Invariant 2 (Model proposes, host disposes).
* **Validation Target:**
  * *Metric:* Tool certification rate on Level-1 to Level-5 task benchmarks.
  * *Baseline:* 1.00 (24/24 on held-out Level-5 maze, 12/12 v7 forge) `[M]`.
  * *Kill Criterion:* Execution outside deny-by-default sandbox or certification success < 0.90 on custom APIs.
  * *Cost:* 10 A100 GPU-hours (~$25.00 at $2.50/hr spot) `[EXT]`.
* **Why Customer Pays:** Enables deterministic workflow automation without giving an LLM unmonitored code execution privileges.

---

## 6. THE "TOP 10 FRONTIER MODEL IN 18 MONTHS" QUESTION

### Direct Assessment

**The literal goal of building a "Top 10 General-Capability Frontier Model" (e.g., competing on LMSYS Chatbot Arena or MMLU-Pro alongside GPT-5, Claude Sonnet 5, and Gemini 3) within 18 months is IMPOSSIBLE on GRL's capital and compute budget.**

#### Quantitative Proof
1. General frontier models require pretraining on **10 to 28 Trillion tokens** (e.g., Liquid LFM2.5 trained on 28T tokens) `[EXT]`.
2. Fact pack §3.6: Pretraining GRL's 17.74B token cache at 2B shape (40.3–45.5% MFU) `[M]` requires **101 H100-hours** `[M, P]`.
3. Pretraining on 28 Trillion tokens at identical MFU requires:
   $$\frac{28,000 \text{ Billion}}{17.74 \text{ Billion}} \times 101 \text{ H100-hours} = \mathbf{159,413 \text{ H100-hours}} \quad \text{`[P]`}$$
4. At spot pricing ($2.50/H100-hr) `[EXT]`, raw compute costs **$398,534 USD** ($542k–$637k on-demand) `[EXT]`, excluding multi-node cluster interconnect overhead, dataset curation, and training restarts.
5. A solo founder with zero initial venture capital cannot compete in brute-force pretraining against $100M+ funded laboratories `[EXT]`.

---

### The Achievable Reframing

#### Target Objective
Achieve the **#1 Global Ranking in Verified Small-Model Reasoning (≤3B Parameters)** on deterministic, zero-hallucination benchmark tasks.

#### Benchmark Focus
Establish and lead the **Verified Multi-Hop Reasoning Benchmark (WikiKG-CoT / GSM8K-Verified / Counterfactual Fault Injection)**.

#### Why Winnable (Evidence in §3)
* **0.9930 claimed-answer precision** `[M]`.
* **1.000 counterfactual planted-fault catch rate** (1,899/1,899 caught) `[M]`.
* **0 verifier leaks** `[M]`.
* **45.9 ms VM latency** `[M]`.
* Frontier neural models (Sonnet 5, DeepSeek-R1) cannot provide zero-hallucination guarantees or instant counterfactual catching because they lack a deterministic VM truth gate (Invariant 1).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                COMPUTE BUDGET TO WIN VERIFIED CATEGORY                  │
├───────────────────────────────────┬──────────────┬──────────────────────┤
│ Milestone / Stage                 │ Compute Reqd │ Est. Cost (USD)      │
├───────────────────────────────────┼──────────────┼──────────────────────┤
│ 2B Pretraining (17.74B Token Cache)│ 101 H100-hrs │ $252.50 `[P]`        │
│ SFT & ToolForge Alignment         │ 30 H100-hrs  │ $75.00 `[P]`         │
│ Benchmark Evaluation & Validation │ 20 H100-hrs  │ $50.00 `[P]`         │
├───────────────────────────────────┼──────────────┼──────────────────────┤
│ Total Cost to Achieve #1 Position │ 151 H100-hrs │ $377.50 `[P]`        │
└───────────────────────────────────┴──────────────┴──────────────────────┘
```

#### Commercial Value
Holding the #1 position in verified small-model reasoning allows GRL to capture the high-margin enterprise compliance market in sovereign jurisdictions (Québec Law 25, EU AI Act), mirroring Mistral's monetization path (**60% of revenue from European sovereignty/on-prem deals**) `[EXT]`.

---

## 7. WHAT KILLS THIS

### Ranked Risk Register

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FIVE FATAL RISKS (RANKED)                          │
├─┬─────────────────────────────┬──────────┬──────────┬───────────────────┤
│#│ Risk Description            │ Prob.    │ Sever.   │ Pre-Registered Act│
├─┼─────────────────────────────┼──────────┼──────────┼───────────────────┤
│1│ Licensing Moat Collapse     │ High     │ High     │ Relicense to BSL  │
│2│ Refusal Rate Bottleneck     │ High     │ Med-High │ Grammar Expansion │
│3│ Context Router Decay        │ Medium   │ High     │ Lock Frozen Router│
│4│ On-Prem Training Disconnect │ Medium   │ Medium   │ Zero-Training Sync│
│5│ Compliance Liability Squeeze│ Low-Med  │ Critical │ Strict Legal SLA  │
└─┴─────────────────────────────┴──────────┴──────────┴───────────────────┘
```

#### 1. Licensing Moat Collapse (Open-Source Commercial Leakage)
* **Description:** Key components remain under permissive MIT/Apache licenses without commercial restrictors. An enterprise or competitor wraps `cubelang` + `grilly` + `FastWordEncoder` into a proprietary product, generating enterprise revenue while GRL receives $0.
* **Leading Indicator:** A commercial product or GitHub repository launches utilizing `grilly` / `cubelang` without contacting GRL for licensing.
* **Pre-Registered Action:** Immediately execute the Section 4 Relicensing Plan: move `cubbyverse` and the serving stack to BSL 1.1 before public v1.0 release.

#### 2. Refusal Rate Bottleneck (The 30% Coverage Deficit)
* **Description:** Coverage remains capped at 0.646–0.710 `[M]`. Buyers reject the solution because ~30% of enterprise queries require expensive frontier API fallbacks, eroding the cost arbitrage (Section 3d).
* **Leading Indicator:** Pilot customer telemetry indicates query refusal rates exceeding 25% on enterprise document corpora.
* **Pre-Registered Action:** Trigger automated grammar expansion using the harvest loop (§1). If coverage does not surpass 0.82 within 60 days, pivot product positioning exclusively to structured SQL/TripleIndex databases where coverage exceeds 0.95.

#### 3. Context-Inference Router Decay in Full Integrations
* **Description:** When scaling to multi-task deployments, parameter context routing ($\theta=f(c)$) fails at inference time (as measured in §3.5 where the unsupervised router collapsed to 1/6 slots and the online router decayed to 37% confidence) `[M]`, inducing cross-task interference.
* **Leading Indicator:** ToolForge decision accuracy drops below 0.80 when the chat memory module is enabled concurrently.
* **Pre-Registered Action:** Disable online hypernetwork parameter generation; lock the architecture to explicit, frozen-router dual-adapter switching (`CubeRoute-Frozen`).

#### 4. Local GPU Compute Disconnect (The Adapter Lifecycle Failure)
* **Description:** Fact pack §3.7 acknowledges local 12GB GPUs cannot train `[NM]`; SFT requires rented cloud A100s. If enterprise security mandates 100% on-premises model fine-tuning without cloud connectivity, the deployment stalls.
* **Leading Indicator:** An enterprise prospect in security review mandates local adapter SFT and rejects cloud-based training steps.
* **Pre-Registered Action:** Restrict custom enterprise tuning strictly to model-free `TripleIndex` zero-training fact updates, or deploy low-rank CPU/Vulkan gradient updates in `grilly`.

#### 5. Compliance Liability Squeeze (Regulatory Misrepresentation) — *Unseen Risk*
* **Description:** GRL markets local deployment as an absolute shield against Law 25. A client misconfigures local logging or network egress, suffers a data breach, and faces $1,000/claimant statutory punitive damages under Law 25 s.17/s.93 `[EXT]`. The client sues GRL for misrepresentation.
* **Leading Indicator:** An enterprise client requests contractual breach indemnification or a "Law 25 Zero-Liability Guarantee" in the SLA.
* **Pre-Registered Action:** Scope all software licensing explicitly as "Air-Gapped Infrastructure Software" with zero legal compliance indemnification. Cap liability in enterprise contracts strictly at license fees paid.

---

### Critical Self-Attack: Core Weakness & Falsification Condition

#### The Single Biggest Weakness
The proposed strategy relies heavily on enterprise willingness to pay **CAD $30,000/year** for Law 25 compliance avoidance, assuming CISOs actively enforce Law 25 s.17 TIA restrictions on cloud APIs.

#### Falsification Condition
If 20 direct sales outreach conversations with Québec CISOs/DPOs yield **zero pilots** because organizations choose to ignore Law 25 s.17 due to a lack of public enforcement actions by the Commission d'accès à l'information (CAI) `[EXT]`, **the compliance-driven local wedge is falsified.**

#### Pivot Trigger
Upon falsification, abandon the sovereign compliance pitch immediately and pivot GRL entirely to **developer API cost optimization for high-volume, structured database query workloads**.

---

## grok-4.3

**1. THE ONE-SENTENCE THESIS**

GRL sells a verified CoT execution service to regulated Canadian organizations that must produce auditable, refusal-safe answers under Law 25 data-residency constraints, where the buyer pays for a millisecond VM gate that certifies every hop instead of paying per hidden reasoning token.  
- Claimed-answer precision 0.9930 at coverage 0.710 with TripleIndex lookup [M].  
- 1,899 planted faults across four classes caught with zero escapes; 528/528 absent-role control [M].  
- 45.9 ms per verified question on the repair budget-3 path versus 120.9 ms with identical verified/correct/precision [M].

**2. THE WEDGE**

Target: Québec health or legal compliance officer whose budget line is “privacy-risk mitigation / transfer-impact assessments” under Law 25 s.17.  
Pain: every external LLM call on personal data triggers a mandatory TIA weighing destination-state legal framework, with statutory punitive damages exposure of ≥$1,000 per claimant for intentional or gross-fault harm.  
Today they either (a) block external models entirely or (b) pay 1.1× US-residency uplifts plus legal review per vendor.  
This wedge is chosen over general enterprise chat or developer tooling because Law 25 already creates a statutory price on data movement; the on-device/air-gapped path removes the TIA obligation entirely. No public contract-value anchor exists for this segment [EXT]. Evidence to gather: one 10-hour FOIA-style request to three Québec public agencies for redacted LLM procurement documents plus a single paid 4-hour interview with their privacy counsel (total cost <$2k).

**3. THE TOKEN-COST ARBITRAGE — QUANTIFY IT**

(a) Cost per verified multi-hop answer  
VM path (measured): 87 ms/question Wiki-chain verification [M] + 45.9 ms repair path [M] = 132.9 ms wall time on consumer hardware. At 127 tok/s stand-in throughput [M] this equals ~16.9 tokens of equivalent decode time. At Together Llama-3-8B $0.14/M tokens [EXT] the marginal token cost is $0.00000237.  
Reasoning-LLM path (assumption): a frontier model emits 800 reasoning tokens + 200 output tokens per multi-hop query (assumption derived from typical o-series visible traces; no measured CubbyLLM equivalent exists). At gpt-5.6-luna $1.20/M output [EXT] this is $0.00120 per answer.  
VM path is therefore ~506× cheaper per answered query before cache effects.

(b) Break-even  
At 1 query/s the VM path costs ~$0.0076/day in electricity at consumer rates versus ~$103.68/day for the frontier path. The VM path loses when coverage forces fallback: at measured coverage 0.710 the 29% refused queries still require a frontier call, raising blended cost to $0.00035 per query. Break-even against that blended cost occurs at ~3,400 queries/day. The VM path loses at reasoning depth <2 hops (where frontier models are cheaper) or when repetition rate <1.2 queries per certified program.

(c) Reuse curve  
Each certified CubeLang program is stored and reused at 0 marginal token cost. At repetition rate r (queries per unique program), marginal cost falls as 1/r. At r=10 the VM path cost drops below $0.00000024 per query. Measured game data shows one program used 55× inside a single session [M]; production cache hit rate of 4–6× is therefore realistic within 30 days for any narrow domain.

(d) Refusal pricing  
Coverage 0.646–0.710 [M] means 29–35% of queries fall back. At $0.00120 per fallback the blended cost rises to $0.00035–0.00042 per incoming query. This still leaves a 2.9–3.4× advantage versus always-on frontier pricing, but only if the buyer values the 0% wrong-answer rate on the 71% that are answered. If the buyer instead values raw answer volume, the arbitrage collapses.

**4. THE BUSINESS MODEL**

Primary line: per-seat verified-reasoning subscription ($4,800/year) for on-prem or air-gapped deployment; includes VM runtime, ledger, and nightly H-A7 gate runs. Gross margin: 78% after $1.77 A100-hr rental for the 37–50 s gate evaluations [M] and 65–77 s proposals [M].  
Secondary line: per-program certification credits ($0.012 each) sold to developers embedding ToolForge; first 1,000 credits free.  
Tertiary line: Québec-specific compliance attestation reports ($2,400 per engagement) that document the signed-certificate ledger for Law 25 auditors.  
GRL does not sell raw model weights or hosted chat tokens; doing so would require depending on the BSL cubemind repo that the architecture explicitly refuses at runtime.  
License fix: relicense cubelang, grilly, and opcode-vsa-rs under BSL-1.1 with 4-year sunset to Apache-2.0 (cost: one week of founder time plus legal review ~$6k). Delay costs nothing now but forces a later dual-license fork once revenue appears; doing it at $0 ARR is cheaper than at $400k ARR.

**5. CAPABILITIES TO BUILD**

Capability 1: On-device TripleIndex + VM verifier for Québec health records. Built from existing TripleIndex (16.5 s / 284 MB on 365k triples [M]) and CubeLang capability model. Lives under invariants 1, 3, 5. Metric: 0.710 coverage at 0.9930 precision on held-out WikiKG functional hops. Baseline: current 0.710. Kill: coverage <0.55 after adding 50k domain triples. Cost: 12 A100-hrs for domain adaptation of the router. Customer pays because it removes the s.17 TIA requirement.

Capability 2: Certified program marketplace with retire-not-delete provenance. Built from ToolForge (v7: 1.00/1.00/1.00 on 12/12 [M]) and the signed ledger. Lives under invariants 2, 4, 7. Metric: 24/24 held-out level-5 maze success. Baseline: 8/8 v3→v4. Kill: any program accepted that later fails its original test vector. Cost: 8 A100-hrs for ledger integration test. Customer pays to avoid re-verifying every third-party skill.

Capability 3: Nightly H-A7 replay gate as a managed service. Built from measured gate (37–50 s on A100, separates replay arm +0.005 vs code-only +0.059 [M]). Lives under invariants 2, 5. Metric: 12/12 unseen-source rejection. Baseline: 12/12. Kill: any source accepted whose mix delta exceeds 0.010. Cost: $180/month per customer at spot rates. Customer pays to keep continual-learning risk below the 0.059 observed delta.

Capability 4: Air-gapped bilingual (EN/FR) verified agent for federal procurement. Built from existing grammar and 0/300 natural_questions parse rate [M] plus Official Languages Act s.25 requirement. Lives under invariants 1, 3. Metric: 96% spoken-correct rate on 25-chain self-test [M]. Baseline: 76% at v3. Kill: parse rate on new French queries <40%. Cost: 40 A100-hrs for grammar expansion. Customer pays because a monolingual model is disqualified from federal contracts.

Rank by commercial value / demo cost: 1, 4, 2, 3.

**6. THE "TOP 10 FRONTIER MODEL IN 18 MONTHS" QUESTION**

The literal goal is unreachable. Measured training budgets show 151M params at 92k steps reaching bpc 0.90–0.94 [M] and a 2B-shape pilot at 20k steps reaching ~1.1 bpc after 3.9 h [M]. Chinchilla-optimal training on the 17.74B-token cache requires ~228 H100-hrs [P]. Top-10 general models require tens of thousands of H100-equivalent GPU-months; GRL’s measured runway cannot close that gap.  

Strongest credible reframing: win the narrow “verified multi-hop reasoning on knowledge graphs with explicit refusal” category. Evidence: 300/300 WikiKG chains VM-verified to gold answer, 1,899/1,899 planted faults caught, identical numbers at 0 and 100k distractors [M]. Cost to first public leaderboard entry: 60 A100-hrs for domain router hardening plus one 2B training run (~100 H100-hrs). Commercial value: the same regulated buyers already pay 1.1× residency uplifts; a #1 position on an auditable-reasoning benchmark becomes a procurement checkbox rather than a fluency contest.

**7. WHAT KILLS THIS**

1. Coverage remains <0.55 after domain adaptation (leading indicator: router macro score <0.60 on first 5k health triples); triggers immediate pivot to pure ToolForge licensing.  
2. Québec public procurement requires source-available code review that exposes the two known cubelang verifier holes; triggers 6-week verifier-hardening sprint or withdrawal.  
3. A single certified program is later shown to have produced a wrong answer on an unseen input distribution (probability low but severity existential); triggers mandatory retirement of all programs sharing that SHA prefix.  
4. Solo founder burnout or health event before the first paid pilot (probability elevated by current Shopify consulting load); triggers immediate search for a technical co-founder with equity grant.  
5. A larger vendor ships a drop-in verified-CoT SDK under permissive license before GRL closes first three design-partner contracts (new risk not listed in §5); triggers immediate BSL conversion of the three core repos and public “license audit” marketing packet.  

Single biggest weakness of this proposal: the wedge assumes Québec compliance officers will pay for on-prem deployment before any measured end-to-end latency or integration cost on their actual EMR systems exists. It would be falsified by a 10-interview discovery sprint returning zero willingness-to-pay above $800/year per seat.

# ROUND 2 — ADVERSARIAL ATTACKS


---

## deepseek-v4-pro

1. **Finding 1’s coverage sensitivity is computed on a dataset that the grammar already handles. On real questions, coverage is zero (0/300 `natural_questions`).** The arithmetic is sound within the fact pack’s artificial 71% coverage, but that figure is from the curated WikiKG functional-hop chains—not open-domain user queries. The grammar cannot parse a single real `natural_questions` query. So the entire blended-cost model collapses: if coverage on actual customer asks is <1%, then >99% of queries fall back to the frontier API, and the local cost is irrelevant. The load-bearing claim that “coverage is ~50× more valuable than unit cost” becomes meaningless when the true coverage is near zero and the path to raising it is an unsolved NLP hard problem (open-domain semantic parsing). The synthesis ignores this fatal reality and treats coverage as a tunable dial.

2. **Against selling the verifier: the 0/300 grammar-parse result makes the verifier unsellable in its current form.** The verifier requires a CubeLang program that captures the query’s retrieval-and-reasoning structure. Without a model that can generate that program from arbitrary natural language, the customer cannot supply it. The trunk is the component being deferred, yet it’s the only plausible program-proposer. The customer cannot be expected to hand-write CubeLang for every query, nor can they map their unstructured text into the MoWM fact format unaided. The verifier is useless without the full CubbyLLM pipeline, and the pipeline doesn’t work on real questions because the grammar fails. The competitor is not another verifier but the entire neuro-symbolic field that has struggled for decades to parse open-domain queries reliably. The verifier per se is a dead end until the grammar gap is solved, which requires significant research, not just engineering.

3. **Finding 3’s inversion—selling a generated TIA—is likely unauthorized practice of law (UPL) in Québec.** The TIA under Law 25 requires a legal assessment of the destination state’s legal framework and a finding of “adequate protection.” Automating that as a software output for a fee constitutes providing legal advice to the public. The Barreau du Québec aggressively prosecutes UPL. Even if positioned as a tool for lawyers, direct sale to enterprises without legal intermediary exposes the founder to civil and regulatory liability. Moreover, the signed evidence file doesn’t remove the need for a lawyer’s judgment; it would be a template at best, and any error could lead to massive class-action exposure under Law 25’s private right of action (s.17+punitive damages). The avoidance play—local inference making s.17 inapplicable—is legally safer, though weak on the enforcement-deterrence side. The inversion is commercially unworkable due to UPL risk, not because of weak fines.

4. **Creating a benchmark is a consolation prize unless the benchmark becomes a de facto procurement requirement.** The strongest historical counterexample is the DARPA Grand Challenge, which a single agency created and spawned an industry—but that had a large prize and military backing. No startup has created an adopted benchmark in AI without broad academic/industry consortia (e.g., MLPerf, ImageNet). For GRL’s benchmark to work, it must: (a) measure something that regulated customers already demand and that no existing benchmark captures—like guaranteed precision with machine-checkable proof; (b) be adopted by a major procurement body (e.g., a Quebec health authority) that requires vendors to submit CubeLang-verifiable results; (c) be technically easy for competitors to adopt (i.e., CubeLang must be open, trivial to integrate, and the grammar must handle their output). None of these are true. The 0/300 grammar parse means the benchmark cannot even evaluate real-world questions, so it would be irrelevant. The reframe is a dead letter without fixing the grammar first.

5. **The synthesis misses that the grammar is the entire product and it’s fundamentally broken.** The CubbyLLM architecture’s core value chain—query → grammar parse → program → VM verify—collapses at the first step for natural language. The 6-WH-frame grammar was hand-written on the WikiKG schema and cannot parse `natural_questions` at all. Expanding it to open-domain queries is not an engineering task; it is the central unsolved problem of symbolic NLP. All the measured verifier numbers (precision, fault catch, absent-role) are on questions that the grammar already parses, so they say nothing about real-world utility. The project’s own honest caveat “0/300 real queries parse the grammar” is not a transient gap—it’s a kill signal. The synthesis’s suggested pivot to selling the verifier is impossible because the verifier can’t be fed. No amount of cost sensitivity or compliance framing matters until the system can process a single real user question. That is the missing critique, and it invalidates the entire business model.

---

## kimi-k2-thinking

**1. FINDING 1's framing is a projection disguised as measurement.** The coverage of 0.710, the 45.9 ms VM latency, and the 179× cost ratio are all measured on the LFM2.5-2.6B *stand-in*, not the 2B CubbyLLM trunk. The risk register explicitly states: "The 2B run has not happened." The sensitivity analysis is therefore arithmetic on a counterfactual. Worse, the "zero marginal token cost" assumes no resource contention: the RX 6750 XT runs 85% busy during decode, and grilly's 231 GLSL shaders compete for the same GPU. The 6,483× cached claim assumes infinite parallel capacity; measure actual throughput under load and the gap collapses.

**2. FINDING 2 is wrong: the real competitor is RAG with a judge model, and the verifier is unsellable without a services arm.** Modern RAG (LangChain, LlamaIndex) already does retrieval + generation + guardrails. The VM's 0.9930 precision is measured *only on symbolic mismatches* ("by symbol mismatch," per the pack), not semantic errors. It catches wrong-entity but not subtle confabulation. The 0/300 natural_questions parse rate means the verifier cannot handle real query syntax. To sell it, a customer must provide *labeled examples* for ToolForge (the pack: "certifies against the example"), which is a data annotation services business. The "standalone verifier" is a consulting product with a VM runtime, not a scalable software SKU.

**3. FINDING 3's inversion is legal malpractice.** Quebec Law 25 s.17 requires a *legal analysis* of foreign law frameworks and a *written agreement* incorporating protective measures. A software vendor generating a TIA is practicing law without a license. No general counsel will outsource legal opinions to a startup; the liability is uninsurable. The correct customer is the *enterprise legal department*, not law firms, and they will treat the VM as a *tool* for lawyers, not a replacement. The synthesis misreads the regulation.

**4. FINDING 5's benchmark strategy is a consolation prize with zero monetization history.** GLUE, SuperGLUE, HELM were created by academics; no benchmark creator captured enterprise value. AUI's $750M cap came from *enterprise deployments*, not a benchmark. The harness exists, but populating competitive baselines requires GPU time to run *other models*—not "near zero." The strategy pigeonholes GRL as a research project, not a venture-backed company. It is a distraction from the core problem: no one has proven they will pay for verification.

**5. Single most important miss: The architecture's eight-layer complexity is inexecutable by a solo founder, and the pack provides no staffing model.** The synthesis never addresses how one person builds/maintains grilly (1,820 tests), cubelang (92 KB spec), neurochemistry ODEs, ToolForge, game, ledger, vault, and a 2B model. The measured results span *different codebases* with *no unified build system*. The non-dilutive funding suggested (IRAP $75K, Mitacs $15K) covers ~3 months of one engineer. The real capital required to productize this is 10–20 engineer-years. The synthesis treats engineering risk as secondary, but it is *existential*.

---

## glm-5

## ATTACK ON THE SYNTHESIZED POSITION

### FINDING 1 — Arithmetic right, framing wrong on two axes.

**Attack 1: The fallback cost is an arbitrary choice, not a measured constraint.**

The synthesis assumes refused queries fall back to Claude Sonnet 5 at $0.013/query. This drives the "coverage is 50x more valuable" result. But the 29% refused could:
- Fall back to a cheaper model (gpt-oss-120B: $0.000825/query)
- Be handled by humans
- Simply not be answered

If fallback is gpt-oss-120B, blended cost becomes $0.000290/query. Coverage improvement from 71%→81% reduces this to $0.000198 — a 32% reduction. Halving local cost reduces it to $0.000264 — a 9% reduction. The ratio drops from 50x to ~4x. The synthesis's headline number depends entirely on choosing the most expensive fallback.

**Attack 2: Query caching is ignored entirely.**

The synthesis shows cached cost at $0.00000201 — 36x cheaper than first-time. If even 5% of enterprise queries are repetitive (FAQs, standard procedures, compliance checks), the blended local cost drops from $0.00007246 to $0.000070. That's a 3% reduction. But if 20% of queries repeat, the reduction is 12%. The synthesis treats every query as first-time, overstating local cost by an unknown margin.

**Attack 3: The $9,179/year savings is correct but proves too much.**

At 1M queries/year, the savings vs Sonnet 5 is $9,179. That's less than one junior developer's salary. The synthesis correctly identifies that the economics don't support a company — then pivots to "sell the verifier" without addressing whether that product has larger TAM. A $9K/year value proposition doesn't become a company just because you change the product framing.

---

### FINDING 2 — The verifier is unsellable in its current state, and the synthesis knows this but buries it.

**The 0/300 natural_questions result is not a "gap." It's a hard stop.**

The synthesis calls this "the named gap" and says "grammar coverage on real questions is the entire critical path." This is diplomatic language for: the product cannot process a single real query from a standard benchmark. You cannot sell a verifier that returns "parse error" on 100% of natural-language questions.

**The customer's facts problem is handwaved.**

The VM verifies against a knowledge base. The synthesis never addresses who builds that knowledge base or at what cost. Enterprise knowledge bases don't exist in TripleIndex format. If the customer must:
- Extract their knowledge from documents
- Structure it as triples
- Mount it in GRL's format
- Maintain it over time

...that's a services engagement, not a software sale. The synthesis pivots from "sell the model" to "sell the verifier" without addressing whether the verifier's input requirements are even satisfiable by customers.

**Real competitor: Guardrails AI.**

The synthesis names no competitors. Guardrails AI has raised $25M+, has a product that validates LLM outputs against schemas, and integrates with existing models. They don't have formal verification — but they have distribution. The synthesis proposes entering a market where a funded competitor already exists, with a product that currently can't parse natural language.

---

### FINDING 3 — Unauthorized practice of law, and the synthesis ignores it.

Québec Law 25 s.17 requires weighing "the legal framework of the destination State." That is a legal judgment. A software tool that generates a document making that judgment is practicing law.

The Québec Bar has exclusive jurisdiction over the practice of law in Québec. A software vendor selling "generated TIAs" faces:
- Cease-and-desist from the Bar
- Liability for unauthorized practice
- No insurance coverage for the liability

The synthesis says "bills against money already leaving the building for outside counsel." But outside counsel won't sign off on a software-generated TIA — that's them putting their liability behind code they didn't write and can't audit. They'll charge the same fee to review it, or refuse entirely.

The inversion doesn't survive contact with the actual buyer: a law firm or in-house counsel who understands professional liability.

---

### FINDING 5 — Benchmarks are participation trophies, not strategies.

**What would have to be true:**
1. The benchmark becomes the standard for "verified answering" in enterprise AI governance
2. GRL's architecture is the only one that can score well
3. Enterprise buyers select vendors based on benchmark scores
4. GRL can monetize the #1 position

**Historical counterexample: MLPerf.**

Created 2018 by Google, NVIDIA, Baidu, Intel and others. Became the industry standard for ML hardware benchmarking. No single company captured lasting value — it became table stakes. NVIDIA dominates not because they created MLPerf, but because they have the best hardware.

**Another counterexample: GLUE/SuperGLUE.**

Created by NYU, Facebook AI, Google, DeepMind. Became standard NLP benchmarks. The creators did not monetize them. They became public goods that accelerated everyone's research equally.

Creating a benchmark is giving away your evaluation framework to competitors. The synthesis proposes this as a "strategy" when it's actually a concession that you can't compete on the existing axes.

---

### THE SINGLE MOST IMPORTANT THING THE SYNTHESIS MISSES

**Liquid AI exists, has 250x the funding, and has already shipped.**

The synthesis ignores the nearest architectural competitor entirely. Liquid AI:
- Raised $250M Series A (Dec 2024) led by AMD Ventures
- LFM2.5-1.2B trained on 28T tokens (700x GRL's planned 40B)
- Measured on-device: 140 tok/s on iPhone 17 Pro, 30 tok/s on Raspberry Pi 5
- Distribution through AMD hardware partnerships

GRL's 2B model, if successfully trained on 40B tokens, would be outparametered 23x and undertrained 700x relative to Liquid's shipped product. The synthesis treats "train the 2B model" as an execution step when it's actually a competitive cliff.

The synthesis's FINDING 5 says "do not enter a race that is already lost." But it frames the race as "frontier models" (GPT-6, Claude). The actual race GRL entered — small, efficient, on-device models — already has a winner with product in market. The synthesis doesn't acknowledge this because it would make the "create a benchmark" strategy look like what it is: too late.

---

## gemini-3.6-flash

### 1. FINDING 1: Arithmetic & Framing Attack

**The Fallback Premise is a Flawed Strawman:**
1. **Unrealistic Fallback Model:** The blended cost calculation ($0.003821) routes refusals to Claude Sonnet 5 ($10/M output). Routing 29% of refusals to Together's Llama 3.3 70B ($1.04/M output) yields a fallback cost of $0.00104/query. The blended query cost collapses from **$0.003821 to $0.000353** (a **10.8× drop**). At this realistic baseline, a 10% coverage increase saves only **$0.000098/query**, destroying the financial motivation for enterprise adoption.
2. **The Air-Gap Paradox:** Under the stated enterprise privacy/compliance wedge (Québec Law 25), **cloud fallback is prohibited**. If air-gapped, fallback cost is $0 (honest refusal). In an air-gapped deployment, local unit cost represents 100% of financial query cost, and coverage dictates *system utility*, not token arbitrage. 

### 2. FINDING 2: Is the Verifier the Product?

**No. The Verifier is Currently Unsellable for Three Reasons:**

1. **The Ingestion Wall:** `0/300` real `natural_questions` queries parse the 6-WH-frame grammar `[M]`. A verifier that rejects 100% of unstructured real-world natural language questions before reaching the VM is not a product; it is a broken parser.
2. **The Fact-Supply Bottleneck:** Per-hop symbolic recovery requires a pre-built, clean `FactStore`/`TripleIndex`. The pack demonstrates that even over a 45k-article WikiKG, `SimpleQA` entities were lookup-reachable for only **0.1% (3 of 4,326)** `[M]`. Enterprise buyers do not possess structured knowledge graphs. Selling a verifier without an automated graph-construction pipeline requires the buyer to supply structured facts the verifier can bind against—a requirement the enterprise market cannot meet.
3. **License Contamination Risk:** `cubelang` has published `Cargo.toml` and `README.md` files declaring MIT. Reverting to BSL-1.1 without formal copyright assignments from all past commit history creates an immediate IP cloud. Enterprise compliance teams doing open-source license audits will reject BSL conversion claims on previously tagged MIT code.

### 3. FINDING 3: Selling the TIA Compliance Artifact

**This Inversion Creates Lethal Legal Exposure:**

1. **Unauthorized Practice of Law & Liability:** Generating a Transfer Impact Assessment (TIA) under Law 25 s.17 as a paid commercial software output constitutes generating a formal legal risk opinion. If a client relies on a CubbyLLM-generated TIA during a Commission d'accès à l'information (CAI) audit or a class-action suit under s.17 (carrying statutory punitive damages of **≥$1,000 per claimant** `[EXT]`), the solo founder faces immediate misrepresentation and professional liability claims that no standard EULA waiver will clear in Québec civil law.
2. **Category Error:** A TIA requires evaluating the statutory surveillance landscape of a foreign destination state (e.g., US FISA 702 / CLOUD Act). Software cannot evaluate foreign law; it can only log network egress. The sellable product is a **zero-egress cryptographic audit log** proving no transfer occurred, making a TIA legally unnecessary. Selling automated legal opinions is a founder liability trap.

### 4. FINDING 5: Category Creation via Benchmark as Strategy

**Creating a Benchmark Without Scale is an Indication of Defeat:**

1. **The Incumbent Capture Counterexample:** Releasing an open benchmark for "verified reasoning traces" gives frontier labs a free evaluation harness. With $10B+ compute budgets, incumbents (or open-weight labs like DeepSeek/Qwen) can synthesize 100M CubeLang verification traces, fine-tune an 8B model, and achieve 0.99 precision at 0.95 coverage on GRL's benchmark within 30 days. 
2. **Benchmark Distribution Failure:** Benchmarks derive authority from traffic and ecosystem consensus (e.g., LMSYS, MLCommons). A solo founder with no institutional backing or distribution cannot establish a benchmark as a market standard. Without distribution, creating a benchmark is an academic exercise, not a commercial strategy.

### 5. THE SINGLE MOST IMPORTANT MISSING FACTOR

**The Architectural Core ($\theta = f(c)$ Hypernetwork) is Broken and Unscalable, Leaving the Custom Trunk Worthless.**

The central technical justification for building a custom 2B LM trunk instead of using an off-the-shelf Qwen/Llama-3 model is context-conditioned parameter generation ($\theta = f(c)$) to prevent interference. However, the measured evidence in §3.5 proves this architecture is currently non-viable:

1. **Unsupervised Context Routing Has Collapsed:** Unsupervised context inference collapsed to **1 of 6 slots**, and online-supervised routing decayed to **37% confidence** (18–23× worse than oracle) `[M]`. 
2. **Hardening Costs Scale Prohibitively:** Hardening the hypernetwork against catastrophic forgetting costs **4.1× compute per step at toy scale** and scales linearly with the number of protected contexts `[M]`. 

Without a functional router and with a 4.1×+ training overhead, the custom trunk cannot be trained on consumer hardware or modest budget. CubbyLLM is reduced to a standard, heavily undertrained 2B model executing an external C++ VM. **The synthesis fails to acknowledge that the proprietary LM trunk project should be abandoned entirely.** GRL should run the VM over third-party open-weight models (via `grilly` / `llama.cpp`) and eliminate the custom trunk training roadmap.

---

## grok-4.3

**1. FINDING 1 arithmetic**

The sensitivity ratio is derived, not measured. Pack gives only two coverage points (0.646 → 0.710) with no costed interventions between them. The 33.8 % blended-cost reduction therefore assumes a linear, zero-marginal-cost coverage lift that the data do not show. The 0.673 % figure for halving local unit cost is likewise an extrapolation from the single measured 1.5 s emit + 45.9 ms VM latency pair; any real coverage gain requires grammar or world expansion whose token or engineering cost is tagged [NM]. The 50× multiplier is therefore an untested modeling choice, not a recomputed fact.

**2. Verifier product**

Right on the 0/300 natural_questions result: the grammar is the critical path and remains unmeasured on any customer workload. Stronger objection: customers cannot supply the TripleIndex/FactStore the VM requires. Pack shows 365 k triples build in 16.5 s only when the facts already exist in that exact form; real enterprise data arrive as documents or tables. Converting them is unmeasured and outside the seven invariants. The actual competitor is therefore not another verifier but the customer’s existing RAG stack plus whatever lightweight checker they already tolerate; the VM adds a new data-schema obligation with no measured conversion path.

**3. TIA inversion**

A Québec law firm will not buy a machine-generated TIA from a software vendor. Doing so would constitute the unauthorized practice of law under Québec professional rules; the vendor would also assume negligence liability for any later AMP or damages claim. The pack itself notes the CAI framework starts at $1 k–$15 k and records zero headline AMPs; the risk being sold is therefore hypothetical, while the new liability created is real and uninsurable for a solo founder.

**4. Benchmark strategy**

Creating a new verified-answering benchmark is a consolation prize. Historical counter-example: the GLUE creators published the benchmark and the associated leaderboard but never became a top-tier model provider; the firms that topped GLUE (and later SuperGLUE) captured the value. For the strategy to work, two conditions must hold simultaneously: (a) the benchmark must become the de-facto procurement gate for regulated buyers, and (b) GRL must retain sole ability to score at the required precision-at-coverage level. Neither condition is evidenced; both are assumptions.

**5. Single most important miss**

The synthesis treats the VM as a separable attestation layer. Pack invariant 7 and the measured host architecture show that every spoken token must still exit through a VM-mediated ASK inside the full brain (neurochemistry ODE + cortices + router). The verifier cannot be sold or run without that surrounding host; the host remains untrained on the 2 B trunk and is therefore part of the same unshipped artifact the synthesis claims to defer.