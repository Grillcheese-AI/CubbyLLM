# Rung 1 — the generality gate: model competition record

GrillCheese Research Lab · 2026-09-10 · four frontier models via OpenRouter · $0.61 total

Contestants: GPT-5.2, Grok 4.6, Kimi K3, GLM 5.3. Gemini 3.1 Pro and DeepSeek V4 Pro were unreachable.
Kimi K3 and GLM 5.3 were truncated by the token limit; judge accordingly.

---

# THE PACK THEY WERE GIVEN

# THE GENERALITY GATE — a research-design competition
GrillCheese Research Lab · 2026-09-10 · every number below is from a captured log in `validation/logs/`

## 0. RULES
Tags: `[M]` measured, log cited · `[P]` projected · `[ASSUM]` assumption with basis · `[NM]` not measured.
**A fabricated number, result or citation costs −5. An idea that violates an invariant scores 0, however good.**
If you need a number that is not here, derive it and show the arithmetic, or write "no source — assumption" with the basis.

## 1. THE PROBLEM, IN THE FOUNDER'S WORDS
> **Rung 1 — the generality gate.**
> **Know:** Specify what a program emitter trained on the harvest is supposed to do that the grammar cannot: which question shapes are out of basin, how an arbitrary document becomes facts the verifier can bind against, what happens at the edge of expressible. A document, not code — the deliverable of the first two weeks.
> **Build:** Run the harvest at volume, train the emitter, and hold out a genuinely out-of-basin set. **Test:** Emitter vs grammar on that set, plus mean program length, first-attempt certification rate and retry count.
> **Baseline:** the grammar, 0/300 on real questions `[M]`. **Kill:** the emitter does not beat the grammar out of basin → bounded-domain product; re-scope and do not train a general trunk.
> **Proves:** Whether there is a frontier lab here at all. It is also the cheapest experiment in the plan, because the harvest already runs and the supervision is a by-product of serving.

## 2. THE SYSTEM
A question is answered by a **program that a separate machine executes and checks**, never by generated prose.

    question → hand-written grammar (6 WH-frames) → per-hop retrieve/walk over a fact store
             → CubeLang chain program → Rust VM: BIND_ROLE / UNBIND + cosine cleanup → (symbol, similarity)
             → per-hop verify: symbol == walked object AND similarity ≥ τ_vm, plus an ABSENT_ROLE control
             → verified answer + auditable trace  OR  honest refusal + partial trace
             → EVERY outcome harvested as supervision

The trunk (a language model) proposes; it never judges. The VM is the only truth gate.
**The intended endgame:** the harvest trains the trunk to emit programs directly, and the hand grammar retires.
That is the loop this gate tests.

### The harvest record (this is your training data — design against it)
Per question: `question`, `parsed`, `n_hop`, `answer_class`, `program_source`, `verified`, `answer`, `gold_answer`, `correct`, `repairs_used`, `hops_verified_before_failure`, `taus{tau_vm,tau_ret}`, `store_snapshot_hash`, `git_rev`.
Per hop in `trace[]`: `query`, `fact`, `triple{obj,rel,subj}`, `ret_score`, `symbol`, `similarity`, `hop_verified`, `source` (`lookup` = triple index, no threshold | `search` = cosine top-k above τ_ret).
Also per hop: `candidates_topk[]` — the full `(score, text)` list retrieval returned (runners-up are free negatives).
Also: `reason` ∈ {`unparseable`, `retrieval_exhausted`, `vm_verify_failed`}; `banned_facts[]` = each rejected fact **with its replacement** (error-localized DPO pairs); `counterfactuals[]` = for every verified chain, four planted fault classes re-planted on the pipeline's own accepted triples, each recovered through the VM, logged per hop with `caught` / `caught_by` / escaped.
Design rule already settled: **store the trace tree, not the accepted path.**
Standing rules: harvested negatives are a **separate calibration/test split**, never merged into SFT data. A record without `store_snapshot_hash` + `git_rev` cannot be safely reused after any store or verifier change. Harvested text **inherits corpus orthography verbatim** — the source corpus is systematically typo'd ("Huamn" for "Human" in ~95% of occurrences: 568 answers / 1,098 facts; "united stated" ×387) and the typos are chain-consistent, so verification depends on preserving them. Normalization is a consumer-side concern.
Reserved but not yet emitted: `frame_id` (which grammar frame parsed it), `world_ids[]` + routing scores, `rationalized` (STaR-style backward-derived programs on unsolved questions).

## 3. THE MEASURED STATE — this is what you are designing against

### 3.1 The headline baseline, and why it is not what it looks like
`standin_question_style_probe.log`, natural_questions dev, n=300:
- **fact grammar parses 0/300 (0.0%)** `[M]`
- **but: `ends with '?' 0/300`, `lowercase start 300/300`, mean 9.3 words** `[M]`
- the question detector still fires on **220/300**, and the router reads 220 as "a question about the world" `[M]`
NQ queries are *search-engine queries*, not well-formed interrogatives. **How much of the 0/300 is surface form versus semantics is unmeasured, and it is probably the first thing to find out.**

disfl_qa validation, n=1000 `[M]`: grammar parses **clean 150/1000 (15.0%)**, **disfluent 129/1000 (12.9%)** — disfluency costs ~2 points. Of the parsed clean questions, the disfluent twin parses the same **80/150 (53.3%)**. The router agrees clean-vs-disfluent on **992/1000**. Repair cues present: "no" ×565, "sorry" ×127, "rather" ×119, "make that" ×92, "uh" ×82.
**Reading: the grammar is narrow, not fragile.**

### 3.2 In-domain performance (WikiKG-derived questions, 800-question eval)
- Grammar v2 coverage on its own domain: **95.4% parse**, 37 unparseable of 800 `[M]`
- **Verified coverage 0.646 (517/800); claimed-answer precision 0.992 (513/517)** `[M]`
- Absent-role control: **528/528** `[M]`; sub-τ verified violations: **0** `[M]`
- With a lookup-first triple index: coverage **0.646 → 0.710**, precision **0.993**, +51 verified, 0 lost `[M]`
- Latency: **45.9 ms/question** at repair budget 1 (vs 120.9 ms at budget 3, identical verified/correct/precision) `[M]`
- **Identical CoT numbers at 0 and at 100,000 distractors** `[M]`

### 3.3 THE MOST IMPORTANT FILE — why the 246 failures fail
`exp_m3_exhaustion_decomp.log`, classifying all 246 `retrieval_exhausted` walks of 800 `[M]`:

| cause | n | share |
|---|---:|---:|
| **misparsed_chain** — planner got the hop count wrong | 92 | 37% |
| **not_retrieved** | 88 | 36% |
| **threshold_bound** | 36 | 15% |
| far_below | 19 | 8% |
| walk_completes_now (already fixed) | 11 | 4% |

Failing hop index: `{0: 152, 1: 63, 2: 28, 3: 3}` — **62% of failures are at the FIRST hop.**
Misparse pattern (parsed n_hop, gold n_hop): `{(1,2): 25, (2,3): 38, (3,2): 14, (4,3): 9, (1,3): 4, (4,2): 1, (2,1): 1}` — it systematically **under-counts** hops.
Misparsed openers: `what is the source that` ×21, `which is the administrative territorial` ×7, `what is the instance of` ×7, `where is the history of` ×7, `who is the editor of` ×5, `which is the office held` ×5.
**Two free wins measured in this same log:**
- `threshold_bound` serving scores: min **0.5028**, p50 0.5547, max 0.5944 against τ_ret **0.5959** — "a tau of 0.5028 would admit all of them."
- **Backward (object-side) retrieval on the not_retrieved/far_below cases: 59 of 72 reach τ_ret.** The walk is forward-only. Example: hop 1 forward score `None`, backward **0.930**.

### 3.4 Retrieval as lookup
`exp_m4_triple_lookup.log`, 1,126 hops over 800 questions `[M]`: lookup serves **83.8%** of hops vs the cosine ceiling's 78.7%; ambiguity only **0.8%**; **`gold_chain_not_accepted_by_plan` 16.2%**. End-to-end frontier walk on the index alone: lookup_correct 71.6%, lookup_dead 22.8%, unparseable 4.6%. Store: 1,241 facts, **1,188 parse as triples (95.7%)**.

### 3.5 Verifier strength — and its real limits
- Planted faults: **1,899 attempted, 1,899 caught, 0 escaped** (598 wrong-entity / 233 wrong-relation / 662 inverted-direction / 406 wrong-hop-order); `caught_by: {symbol_mismatch: 1899}` — **100% by string mismatch, 0% by the similarity floor** `[M]`
- **Organic confusables: 70 of 91 caught = 0.769** `[M]` ← the realistic number
- Random distractors: 47/47 `[M]`
- Calibrated τ floors by frame size: **n_hop=1: 1.0000 · n_hop=2: 0.4736 · n_hop=3: 0.2202** (control_role_mean ≈ 0.023–0.028) `[M]` — **verification confidence collapses as chains lengthen**
- On-record caveat: "an execution engine's similarity measures binding **fidelity, not truth**."
- A prior τ_vm calibration **degenerated to a floor** (clean store ⇒ no negatives, AUC 0.0) and self-reported as `tau_vm_calibration_degenerate`; fixed in v2 with frame-size-conditional floors.

### 3.6 The reachability ceiling
With the whole 45k-article wikikg mounted: SimpleQA gold is a graph entity for **19.9%** of 4,326 questions and lookup-reachable for **0.1% (3 of 4,326)** `[M]`.
On record: **"The store cannot move SimpleQA; the loop must."**

### 3.7 The trunk is not a retrieval encoder
Mean-pooled trunk states route at macro **0.0962** vs MiniLM-L6-v2's 0.3580, despite a larger body; on micro the trunk is **−6.0 sd below a random encoder's floor**. A training-free character-trigram sketch beats both trunk arms. `[M]`
Defence on record: the trunk has no contrastive or retrieval objective in its loss. But the numbers exist.

### 3.8 Routing / spawn (the world-model gate)
Centroid+margin, 3 rotated open-set splits × 5 held-out domains `[M]`: **route-macro 0.6796, routed@τ 41%, accuracy|routed 92%, open→spawn 87%, spawn-AUC 0.666** — against a **MiniLM reference ceiling of 0.8245**. Precision bought with abstention, not sharp separation.

### 3.9 Compute available
$5M+ over 18 months. Measured **45.5% MFU at D2048/L32, B8/S1024, compiled with a FlexAttention sliding-window BlockMask, on an A100-80GB: 440 ms/step, 18,619 tok/s, 1,254.4M params, 40.0 GB** `[M]`. The learning-gate machinery evaluates a 3.28M-token delta in **65–77 s** and runs a full eval in **37 s** on one A100 `[M]`.
**So a small emitter experiment is hours, not weeks. Cost is not the constraint. Being wrong slowly is.**

## 4. INVARIANTS — violating one scores ZERO
1. The VM is the only truth gate. **No LLM-as-judge anywhere in the loop.** The model never scores its own thoughts.
2. The model proposes, the host disposes: identity, provenance, execution, reward, retirement and what the model is shown are host decisions.
3. The don't-know contract: an unverified claim is **never spoken**.
4. Retire, never delete. Every record carries a store-snapshot hash and a git revision.
5. Serve-time budget: VM calls in milliseconds; a training round is tens of minutes on one 80 GB GPU.
6. **No self-play or self-judging shortcuts.**
7. Nothing depends on another repository at runtime; reusable pieces are ported, never linked.

## 5. ALREADY TRIED — do not re-propose these
- **Binding auxiliary loss — FALSIFIED.** Goodharted: trained-trunk retrieval **11.5%** (chance 6.25%) vs **untrained control 97.5%**. "An untrained trunk is already a near-optimal VSA substrate." `bind_weight` set to 0. `[M]`
- **Chrono-init — KILLED** at **+0.444 nats** held-out CE, 44× its 0.01-nat threshold. `[M]`
- **Multi-horizon prediction head — free but redundant.** A 300-step post-hoc linear probe on the *baseline* trunk already reads the 500-token future at cosine **0.414**; the trained objective adds ~0.02. `[M]`
- **Sparse choice points via an absolute entropy threshold — does not select on real text.** Toy AUROC 1.000 flagging exactly the true 12.5%; on a real checkpoint over 4,095 positions, median entropy **4.32 nats** and the fraction above 1/2/3 nats is **0.914 / 0.830 / 0.701**. `[M]`
- **Interpolation compression — operators falsified.** `[M]`
- **Semantic-encoding gate — 3 of 4 bars failed.** `[M]`
- **Hyper-encoder embeddings — quality clause fails** (+0.50 nats vs a static table). `[M]`

## 6. WHAT YOU MUST PRODUCE
A design for the generality gate that is **executable in weeks and falsifiable**. Sections:

**A. THE DIAGNOSIS.** From §3, what is actually blocking generality? Rank the causes by (share of failures) × (tractability). Say plainly which of the 0/300 you believe is surface form and which is semantics, and **name the cheap measurement that would settle it before anything is trained.** Disagree with the founder's framing if the data says to.

**B. THE SPEC — the "Know" deliverable.** What must a trained program emitter do that the grammar cannot? Define *out-of-basin* operationally so the held-out set can actually be constructed and nobody can move the goalposts afterwards. Say how an arbitrary document becomes triples the verifier can bind against, and what the system does at the edge of expressible.

**C. THE FREE WINS FIRST.** §3.3 contains at least two recoverable failure classes measured but not acted on. Quantify what fixing them does to coverage — show the arithmetic on the 246 — and say why that must happen before any emitter is trained, or argue that it must not.

**D. THE TRAINING DESIGN.** Exactly how to turn this harvest into an emitter. Which fields, which objective (SFT / RFT / DPO / process supervision / something else), how the negatives and counterfactuals are used **without merging them into SFT data**, how `store_snapshot_hash` invalidation is handled, how the typo'd corpus is handled, and what stops the emitter simply memorizing the grammar's basin. Be specific about data volume, model size and GPU-hours.

**E. THE BOOTSTRAP ARGUMENT.** The harvest can only manufacture supervision for questions the grammar can already reach, so the SFT set inherits the grammar's shape. **Give the strongest mechanism you know for escaping that basin**, and the strongest reason it might fail here specifically. STaR-style rationalization on unsolved questions is the obvious candidate and `rationalized` is a reserved field — but note invariant 6.

**F. THE EXPERIMENT.** Metric, baseline, kill criterion, cost in GPU-hours, and wall-clock. Pre-registered predictions: say what you expect each arm to produce, in numbers, before it runs.

**G. WHAT WOULD FALSIFY YOUR OWN DESIGN.** One paragraph. Your best shot at yourself.

Constraints: tag every number. Prefer a narrow design that can be run in three weeks over a broad one that cannot. Where you disagree with the founder, say so and give your reasoning. Do not restate the pack back at me.


---

# SUBMISSIONS


---

## gpt-5.2

## A. THE DIAGNOSIS (ranked by share × tractability)

### A1) What is blocking “generality” in the *measured* system?
**Within the grammar’s own domain (800 eval):** almost all miss mass is `retrieval_exhausted`, not verifier weakness or latency.

- Verified coverage **517/800 = 0.646** `[M]`
- Unparseable **37/800 = 0.046** `[M]`
- Retrieval-exhausted **246/800 = 0.308** `[M]`
- There are no other failure buckets here: \(800 - 517 - 37 - 246 = 0\) `[M arithmetic]`

**Inside those 246 retrieval exhaustions** (`exp_m3_exhaustion_decomp.log`) `[M]`:

Rank by (share) × (likely fixability in <3 weeks):

1) **misparsed_chain (92/246 = 37%)** `[M]`  
   *Tractability: high.* This is a planner/hop-count error with clear patterns and a strong under-count bias. Fixing it does not require new ML.

2) **not_retrieved (88/246 = 36%)** `[M]`  
   *Tractability: medium-high.* You already measured that **backward retrieval** solves a large fraction (see §C). That’s an algorithmic change.

3) **threshold_bound (36/246 = 15%)** `[M]`  
   *Tractability: very high.* The log already shows min score **0.5028** vs current **τ_ret = 0.5959** and “a tau of 0.5028 would admit all of them” `[M]`. This is a one-line change plus a precision regression check.

4) **far_below (19/246 = 8%)** `[M]`  
   *Tractability: medium.* Might become “not_retrieved” under backward / lookup-first changes; otherwise harder.

5) **walk_completes_now (11/246 = 4%)** `[M]`  
   Already fixed `[M]`.

Also: **62% of retrieval failures are at hop 0** `{0: 152, 1: 63, 2: 28, 3: 3}` `[M]`, so improvements that help hop-0 retrieval/planning will disproportionately move end-to-end coverage.

### A2) What is blocking “generality” in the founder’s headline 0/300 (NQ probe)?
You do **not** yet know whether the **0/300 parse** on NQ is mainly *surface-form mismatch* or *semantic out-of-basin*. The log gives strong surface hints:

- `ends with '?' 0/300` `[M]`
- `lowercase start 300/300` `[M]`
- “NQ queries are search-engine queries” (your note) and the question detector still fires on **220/300** `[M]`

My belief (explicitly uncertain): **the majority of the 0/300 is surface-form**, because the grammar is *narrow, not fragile* (disfluency costs ~2 points) `[M]`, and NQ has obvious non-interrogative formatting `[M]`. But semantics could still dominate if the store/relations don’t support NQ (cf. reachability ceiling).

### A3) Cheap measurement to settle surface vs semantics *before training anything*
Run a **deterministic “wrapper parse” probe** on the same 300 NQ strings, no learning:

For each raw query `q`, generate 6 wrapped variants aligned to your 6 WH-frames (example forms; exact strings must match your grammar):
- `what is q ?`, `who is q ?`, `where is q ?`, … (6 total)

Then measure:
1) parse rate, 2) verified rate, 3) failure reasons.

This cleanly answers: “Is it mostly missing WH/punctuation casing?” without moving semantics. If parse remains ~0, it’s semantics/expressibility/store, not typography.

If you want one more cheap discriminator: pick **n=50** `[ASSUM]` from the 300 and do a 2-person manual label: “answerable by a 1–3 hop KG chain over current store?” This is hours, not weeks, and tells you if NQ is *in principle* in-scope.

---

## B. THE SPEC — the “Know” deliverable (what the emitter must do, and what “out-of-basin” means)

### B1) What the program emitter must do that the grammar cannot
The grammar is currently doing **three jobs** that the emitter must subsume:

1) **Frame selection + slotting**: map arbitrary question surface forms (including non-questions like search queries) into a canonical program skeleton.
2) **Hop-count planning**: choose `n_hop` correctly (you systematically under-count today) `[M]`.
3) **Query construction** for retrieval at each hop (including directionality), so the VM can bind and verify.

The deliverable “Know” doc should specify:
- The **CubeLang subset** the emitter is allowed to emit in this gate (I recommend: single-chain programs, `n_hop ∈ {1,2,3}` to match your verifier floor collapse with length `[M]`).
- A **normal form** for programs so program equivalence is checkable (e.g., alpha-renaming of roles, canonical ordering of fixed fields).
- A **failure taxonomy** at the program level: `unparseable_question`, `not_expressible`, `retrieval_exhausted`, `vm_verify_failed` (you already have three; I would add `not_expressible` so “edge of expressible” is explicit rather than confounded with retrieval).

### B2) Operational definition of OUT-OF-BASIN (so you can actually build the held-out set)
Define *two* out-of-basin sets, because “surface OOB” and “semantic OOB” are different claims:

**OOB-S (surface out-of-basin):** questions where semantics is known in-basin but surface form is out.  
Construction rule (goalpost-proof):
- Start from a set of in-domain questions with verified programs under a fixed `(store_snapshot_hash, git_rev)` `[M invariant-4]`.
- Apply **deterministic, non-LLM** transformations that are *designed to break the grammar* while preserving the answer, e.g.:
  - remove `?` (already present in NQ: `0/300` ends with `?`) `[M]`
  - lowercase the first character (matches NQ: `300/300`) `[M]`
  - delete WH-word, drop auxiliary (“is/was”), collapse to “entity relation” query style  
- The label program is the *original verified program* (same store snapshot).

This tests whether an emitter can generalize across **question shape** without requiring new knowledge.

**OOB-R (relation/template out-of-basin but still expressible):** questions that require programs in your allowed CubeLang subset, but whose *linguistic pattern* is not one of the 6 grammar frames.  
Construction rule:
- Generate questions from your triple store using alternative templates (possessives, noun phrases, appositives) that you intentionally do **not** include in grammar.
- Hold out entire template families at training time.

This tests whether the emitter can replace the hand grammar’s pattern library.

**Explicit non-goals for this 3-week gate:** “arbitrary natural questions like NQ across the open web.” Your own reachability numbers for SimpleQA (**0.1% lookup-reachable**) `[M]` say the store won’t support that yet; making it a kill condition now would confound “emitter vs grammar” with “KB doesn’t contain the answer.”

### B3) “How an arbitrary document becomes facts the verifier can bind against”
You already have a measured triple parse rate: **1,188/1,241 = 95.7%** facts parse as triples `[M]`. For this gate, freeze the ingestion contract:

- **Document → (subject, relation, object) triples** via your existing parser.
- Every triple is stored verbatim (typos preserved; you state verification depends on that) `[M]`.
- `store_snapshot_hash` is part of the training/eval key; the gate uses *one frozen snapshot*.

At the **edge of expressible**:
- If the question implies operators outside the allowed CubeLang subset (aggregation, comparison, conjunction), the correct behavior is **honest refusal** with a partial trace, not a guessed program (invariant 3).

---

## C. THE FREE WINS FIRST (quantified on the 246 retrieval exhaustions)

You said §3.3 contains “at least two recoverable failure classes measured but not acted on.” Agreed; and I think you must fix them **before** emitter training, because otherwise you harvest mislabeled “failures” that are actually solvable—poisoning both SFT and preference data.

### C1) Free win #1: lower τ_ret to admit all `threshold_bound`
From the log:
- `threshold_bound` count **36** `[M]`
- min serving score among them **0.5028** vs current **τ_ret 0.5959** `[M]`
- statement: “a tau of 0.5028 would admit all of them” `[M]`

**Upper-bound coverage gain if all 36 become verified:**  
Current verified = **517** `[M]`  
New verified = **517 + 36 = 553** `[ASSUM: all convert]`  
Coverage = \(553/800 = 0.69125\) `[ASSUM arithmetic]`  
So **+0.045** absolute coverage possible from this bucket alone `[ASSUM]`.

You must re-check precision after this change (claimed-answer precision is **0.992** now) `[M]`; lowering τ_ret could increase confusables that slip past retrieval but VM should still catch by symbol mismatch in many cases. However your organic confusable catch rate is **0.769 (70/91)** `[M]`, so retrieval could surface more confusables that the VM does *not* always catch. This is why it’s a free win only if precision stays flat.

### C2) Free win #2: add backward (object-side) retrieval for forward failures
Measured:
- On `not_retrieved` + `far_below` cases, backward retrieval reaches τ_ret for **59 of 72** `[M]`.

If we assume those 59 correspond to previously failed walks and they verify when retrieved (not guaranteed), **upper-bound** gain:
- **+59 verified** `[ASSUM]`
- Coverage gain = \(59/800 = 0.07375\) `[ASSUM arithmetic]`

### C3) Free win #3 (not “free” but very high ROI): fix hop under-counting
`misparsed_chain` is **92** `[M]` and systematically under-counts hops `[M]`. If fixed, it prevents dead-end planning at hop>0 where you can’t recover.

Upper-bound gain:
- **+92 verified** `[ASSUM]`
- Coverage gain = \(92/800 = 0.115\) `[ASSUM arithmetic]`

### C4) Combined upper bound from the measured “recoverables”
Total potential added verified from these three buckets:  
\(36 + 59 + 92 = 187\) `[ASSUM arithmetic based on measured bucket sizes]`

Upper-bound new verified: \(517 + 187 = 704\) `[ASSUM]`  
Upper-bound coverage: \(704/800 = 0.88\) `[ASSUM arithmetic]`

I am **not** claiming you will get 0.88; I’m saying: the logs prove there is *a lot* of headroom from pipeline fixes, and training an emitter before taking it is very likely to (a) waste data and (b) make the emitter look better than it is by letting it “fix” problems that weren’t genuinely about generality.

---

## D. THE TRAINING DESIGN (turn the harvest into an emitter without violating the split rules)

### D1) Data selection (hard constraint: hash+rev)
Use only records with:
- `store_snapshot_hash` present AND
- `git_rev` present  
(as you require) `[M invariant-4]`.

For the 3-week gate, freeze to **one** `(store_snapshot_hash, git_rev)` pair (latest stable), and discard other hashes for this run. That prevents “training on one store, evaluating on another.”

### D2) What goes into SFT vs what stays out (per your standing rule)
**SFT dataset (only verified positives):**
- Input: `question` (raw, typo’d, original orthography preserved) `[M]`
- Output: the verified CubeLang program (from the executed trace / program field), normalized to a canonical form (spec’d in B1).
- Filter: `verified == true` and `correct == true` if you trust `gold_answer` (you have claimed-answer precision **0.992** on verified) `[M]`. I would still include `verified==true` but track correctness separately.

**NOT in SFT (separate calibration / preference datasets):**
- `candidates_topk[]` negatives
- `banned_facts[]` with replacements (these are localized DPO pairs)
- `counterfactuals[]` (planted faults)

This respects your rule: “harvested negatives are a separate calibration/test split, never merged into SFT data.”

### D3) Objectives (minimal set that is actually run fast)
I recommend **two training stages** (both cheap):

1) **SFT (program emission):** maximize likelihood of the full program given the raw question.  
   This is the core “grammar replacement” skill.

2) **Preference fine-tune (DPO or similar) on hop-local decisions** using `banned_facts[]` pairs, *without* teaching the model to judge truth:  
   - Prompt = `(question, hop_index, current bound roles, retrieval query text)` (all host-provided context from the trace tree)
   - Chosen = the query/fact that led to the replacement
   - Rejected = the banned fact/query  
   This trains “don’t make *this* mistake again” while still letting the VM be the only truth gate at runtime.

I am deliberately **not** proposing any “LLM judge,” any self-scoring, or merging negatives into SFT.

### D4) Handling the typo’d corpus
Do **not** normalize in training. Treat the surface forms as immutable because:
- you state “verification depends on preserving them” `[M]`.

To reduce brittleness, you can add *consumer-side* normalization later, but that is outside this gate.

### D5) Prevent “just memorizing the grammar basin”
You can’t prevent memorization by wish; you need **evaluation construction** that makes memorization insufficient:

- Train on original in-domain phrasing.
- Evaluate on **OOB-S** deterministic rewrites that break the grammar but preserve answers (B2).
- Additionally, train with a small fraction of *grammar-broken* rewrites but hold out entire rewrite families, so you can measure compositional generalization rather than “saw this rewrite.”

No LLM paraphrasing required.

### D6) Data volume, model, and GPU-hours (bounded + explicit assumptions)
You already have a **1,254.4M param** trunk and measured throughput **18,619 tok/s** on one A100-80GB at a specific config `[M]`. Use that trunk with a small fine-tune (full fine-tune or LoRA; either is fine—LoRA is safer operationally).

Assume (no source) `[ASSUM]`:
- SFT examples: **100,000** `[ASSUM]` verified programs from harvest
- Tokens per example (question+program+format): **256 tokens** `[ASSUM]`

Total tokens ≈ \(100{,}000 \times 256 = 25.6\) million tokens `[ASSUM arithmetic]`

Time on 1×A100 at measured throughput:
- seconds ≈ \(25.6\text{M} / 18{,}619 \approx 1{,}375\) s `[ASSUM arithmetic using M throughput]`
- hours ≈ \(1{,}375/3600 \approx 0.38\) GPU-hours `[ASSUM arithmetic]`

Even if this estimate is off by ~5× due to optimizer/seq-length differences, it’s still “hours not weeks,” consistent with your compute note `[M]`. The *real* schedule risk is data correctness/splits, not FLOPs.

---

## E. THE BOOTSTRAP ARGUMENT (escaping the grammar basin without self-judging)

### E1) Strongest mechanism that fits your invariants: **VM-verified program search (rationalization)**
You already reserved `rationalized`. Use it, but do it in a way that is not “self-play”:

- For questions the grammar cannot parse (or parses to the wrong hop count), run a **host-controlled program search** over the *allowed CubeLang subset* (e.g., hop counts 1–3).
- Use **non-LLM heuristics** to seed entities/relations (important because trunk embeddings are weak: routing macro **0.0962** vs MiniLM **0.3580** `[M]`; the trunk is not a retriever).
  - You already have a “character-trigram sketch beats both trunk arms” `[M]`. Use that as the seed matcher.
- Every candidate program is executed and **only VM-verified successes become new training data**.

This escapes the grammar frame library because the generator is not the grammar; it is a program-space search with VM certification.

### E2) Strongest reason it may fail *here specifically*
Two measured bottlenecks could make rationalization unproductive:

1) **Reachability ceiling / store mismatch**: SimpleQA is lookup-reachable for **0.1% (3/4,326)** `[M]`. If your OOB target resembles that distribution, no amount of program search will help because the answers are not in the store in bindable form.

2) **Verifier confidence collapse with chain length**: τ floors drop with hop count (**n_hop=1: 1.0000; 2: 0.4736; 3: 0.2202**) `[M]`. That means longer chains are easier to “verify” spuriously unless symbol binding remains stringent; but your planted faults were caught **100% by symbol mismatch** `[M]`, which suggests verification is mostly string-identity anyway. If the search finds symbol-identical but semantically wrong chains due to aliasing/typos, you may “verify” garbage. Your organic confusable catch rate **0.769** `[M]` is the warning sign.

---

## F. THE EXPERIMENT (falsifiable, 3-week executable)

### F1) Arms
All arms run on a **frozen store_snapshot_hash + git_rev**.

1) **Grammar baseline** (current): grammar → retrieve/walk → VM  
2) **Grammar + free wins**: apply (a) τ_ret change, (b) backward retrieval, (c) hop-count fix  
3) **Emitter** (trained): emitter proposes program(s) directly; host executes + VM verifies. No grammar parse step.

You need arm (2) because otherwise a win by the emitter could just be “it learned around avoidable pipeline bugs.”

### F2) Evaluation sets (pre-registered)
1) **In-domain holdout**: a held-out slice of WikiKG-derived questions under same store snapshot `[NM: size not provided]`  
   Purpose: ensure you didn’t break precision/coverage.

2) **OOB-S (surface out-of-basin)**: deterministic grammar-breaking rewrites of in-domain holdout questions (B2).  
   Purpose: isolates “generality across question shape.”

3) **NQ-300 raw**: the exact probe set where grammar is **0/300 parse** `[M]`.  
   Purpose: reality check; but interpret results only after the wrapper-parse diagnosis in A3.

### F3) Metrics (all already aligned with your founder spec)
- **Verified answer rate** = verified / total
- **Correct|verified** (if gold is available)
- **Mean program length** (tokens or AST nodes; predefine)
- **First-attempt certification rate** (top-1 program verifies)
- **Retry count** (how many candidate programs executed until verify/refusal)
- **Refusal rate** (explicit “don’t know” with reason)

### F4) Kill criterion (pre-registered)
Because the grammar is 0/300 on NQ `[M]`, you must not define the kill solely on NQ unless you first establish answerability. For this gate I would set:

- Primary kill test on **OOB-S**:  
  If **Emitter verified rate ≤ Grammar+free-wins verified rate**, the “retire the grammar” path is not justified → kill the general trunk plan (per founder intent).

- Precision safeguard: emitter must keep **Correct|verified ≥ 0.99** `[P target based on current 0.992 M]` on in-domain holdout; otherwise you’re trading coverage for wrong verified claims (violates product intent even if VM verifies binding).

### F5) Cost + wall-clock
- Pipeline fixes: engineering time, not GPU.
- Training: as in D6, **O(1)** A100-hours `[ASSUM]` for SFT + small DPO.
- Evaluation: you measured “full eval in **37 s** on one A100” `[M]` (on your machinery); even multiplying by several eval sets and retries keeps it minutes-hours, not days.

### F6) Pre-registered predictions (numbers tagged)
These are projections; you should hold me to them.

1) **Grammar baseline on OOB-S:** near **0 verified** `[P]` if rewrites successfully break parsing (by construction).  
2) **Grammar+free-wins on in-domain:** verified coverage improves from **0.646** `[M]` to **0.70–0.80** `[P]` (because the upper bound from recoverables is 0.88 `[ASSUM]`, but not all will convert).  
3) **Emitter on OOB-S:** verified rate **≥ 0.50** `[P]` if trained on enough verified programs and if CubeLang subset matches; otherwise it will collapse to refusal or wrong programs.  
4) **Emitter on NQ-300 raw:** **unknown; my prediction is low** (e.g., **<0.10 verified**) `[P]` unless the wrapper-parse probe in A3 shows that surface wrapping alone recovers parse/verify. If wrapping recovers a lot, emitter should too.

---

## G. WHAT WOULD FALSIFY MY OWN DESIGN (best shot against it)

If, after implementing the free wins (τ_ret, backward retrieval, hop-count fix), the **Grammar+free-wins** arm already achieves high coverage on **OOB-S** via simple wrapper/normalization tricks (A3), then the “emitter replaces grammar” claim is *not* the frontier question; surface-form brittleness was the real issue and the emitter is unnecessary for rung-1. Conversely, if the emitter fails to beat Grammar+free-wins on OOB-S, then either (i) the harvest supervision is too shape-locked to the grammar to teach general mapping from raw text to programs, or (ii) program emission is not the right abstraction and you need an intermediate semantic representation. Finally, if NQ remains near-zero verified even after wrapper parsing *and* after emitter training, that falsifies the assumption that “question shape” is the blocker; it points instead to **store reachability / expressibility** as the dominant constraint, meaning the generality gate as framed cannot be passed with emitter training alone.

If you want, I can turn this into a one-page preregistration checklist (exact dataset construction scripts + exact acceptance thresholds) so nobody can move goalposts mid-run.

---

## grok-4.6

# Generality gate — design
Disagree with the founder on the kill and on the order of work. The 0/300 is not a trunk test. An emitter vs grammar on NQ/SimpleQA, with the store as it stands, is a store-emptiness test that both systems fail for the same reason. That kill would re-scope a lab because a 1,241-fact store cannot bind SimpleQA (lookup-reachable **3/4,326 = 0.1%** `[M]`). Wrong object.

Three-week object: **shape generality on facts already in store**, after taking the measured free wins so SFT does not distill a known-broken planner. Document→triples is specified (B) and not the kill metric, because no non-judge extractor is measured.

---

## A. THE DIAGNOSIS

**Blocking generality, ranked by (share of the actual OOB failure) × tractability.** In-domain 246-decomp is not “generality”; it is listed second because it poisons any later emitter.

1. **Store non-reachability (existential, low 3-week tractability).** SimpleQA gold is a graph entity for **19.9%** of 4,326 and lookup-reachable for **0.1%** `[M]`. On record: the store cannot move SimpleQA. A perfect CubeLang emitter still refuses. This dominates every “real question” number and is not an emission problem.
2. **Grammar basin is six WH-frames, not “questions” (large share, medium tractability).** disfl_qa clean parses **150/1,000 = 15.0%** `[M]`. That is the best measured estimate of frame coverage on naturalistic interrogatives. NQ **0/300** `[M]` is a different, cheaper mixture (next paragraph).
3. **In-domain planner/retrieval bugs that will become SFT** (share of 246 `[M]`: misparsed_chain **92/246 = 37%**, not_retrieved **88/246 = 36%**, threshold_bound **36/246 = 15%**; **152/246 = 62%** fail at hop 0). High tractability. Training before fixing distills under-counted hops and forward-only walks.
4. **Verifier is fidelity, not truth** (limits how far any emitter can be trusted). Planted faults **1,899/1,899** caught, **all** `symbol_mismatch`, **0** by the similarity floor `[M]`. Organic confusables **70/91 = 0.769** `[M]`. Frame-conditional τ_vm: **n_hop=1: 1.0000, n_hop=2: 0.4736, n_hop=3: 0.2202** `[M]`. Longer programs the grammar cannot emit are exactly where the gate goes soft.
5. **Trunk-as-retriever is irrelevant here** (macro **0.0962** vs MiniLM **0.3580** `[M]`). The host retrieves. Do not put a retrieval objective on the emitter; already falsified a binding aux against an untrained control `[M]`.

**Surface vs semantics on the 0/300.** NQ is not an interrogative corpus: `ends with '?' 0/300`, `lowercase start 300/300`, mean **9.3** words `[M]`. The detector still fires on **220/300** `[M]`. I believe **most of the 0/300 is surface+frame**, not “cannot represent the question in CubeLang.” I do **not** believe surface rewrite will produce in-domain **95.4%** parse `[M]`. disfl_qa says the ceiling on real interrogatives is ~15% for this grammar. NQ after becoming WH will still mostly sit outside the six frames.

**Cheap measurement, before anyone trains (day 1–2, CPU, the 300 log):**
- **M0a.** Deterministic surface norm: capitalize first token, append `?`. Re-run the same grammar. Splits “not an interrogative” from everything else.
- **M0b.** On the **220** detector-positive items only, apply a frozen WH-rewrite table (who/what/where/when templates, no model judge). Re-parse. Splits “wrong wrapper” from “not one of six frames.”
- **M0c.** Same two transforms on a **200**-question slice of the 800 (facts exist). If grammar stays high on the 800-transforms and stays ~0 on NQ-transforms, NQ failure is **semantics/frame**, not `?`.

Pre-registered guesses (not results): M0a on NQ **≤ 5/300** parse `[P]`; M0b on the 220 **~15%** parse, i.e. ~33, matching disfl_qa not WikiKG `[P]`; M0c on the 800 search-query transform **near 0** parse `[P]` (that is the shape-OOB grammar baseline).

Founder framing I reject: “emitter vs grammar on real questions” as the existence proof for a frontier lab. Real-question failure is currently **facts not in store**, then **frame shape**, then emission. Cheap to unmix; expensive to confuse.

---

## B. THE SPEC — what the emitter must do that the grammar cannot

Grammar: six WH-frames → a hop plan → CubeLang. It cannot (measured or by construction):
- emit a plan for a non-frame shape (NQ surface, `how`/`why`/`how many`, declaratives);
- accept the gold triple the index already has (**`gold_chain_not_accepted_by_plan` 16.2%** of 1,126 hops `[M]`);
- count hops (systematic undercount: `(2,3): 38, (1,2): 25, …` `[M]`);
- walk backward (forward score `None`, backward **0.930** on the logged example `[M]`).

**Operational out-of-basin (frozen before train; no goalpost moves).** A question is **shape-OOB** iff it is in one of the pre-registered transform classes below **and** the grammar, frozen at `git_rev` R, returns unparseable. Gold is a **store-grounded walk**: `gold_answer` equals a symbol reachable from the current `store_snapshot_hash` by some CubeLang program (host search, §E), not by the grammar.

Construct from the WikiKG 800 (facts exist), never from NQ/SimpleQA until ingestion exists.

| id | transform (deterministic, host) | why grammar should fail | valid kill item? |
|---|---|---|---|
| S1 | search-query: drop WH, drop `?`, lowercase | NQ surface, **0/300** `[M]` | yes |
| S2 | disfluency overlay (`uh`/`sorry`/`make that` from the measured cue list) | disfl_qa cost ~2 pts in-frame `[M]`; weak OOB | no (too weak) |
| S3 | declarative / cloze (“X’s Y is _”) | not a WH-frame | yes |
| S4 | WH outside the six (`why`, `how`, `how many`) | inexpressible or unparsed | **no** — edge of expressible, must refuse |
| S5 | leave-one-**relation-family** out of *training*, not of the grammar | tests program topology, not “beats grammar” | diagnostic only |

**Held-out set:** 200 chains drawn from the 800, **unseen at train**, each issued in canonical WH (in-basin control) and in S1 and S3 (shape-OOB). Same `store_snapshot_hash`. Grammar score on S1/S3 is a measurement, not a hope; kill uses that measurement.

**Genuinely knowledge-OOB** (NQ, SimpleQA, arbitrary docs) is **out of scope for the three-week kill**. Register it as Gate 1b, blocked on ingestion.

**Arbitrary document → bindable triples (host proposes/disposes; VM still the only truth gate for answers).**
1. Snapshot the raw bytes; new `store_snapshot_hash`. Orthography **verbatim** (corpus is chain-consistent typos: “Huamn” **568** answers / **1,098** facts; “united stated” **×387** `[M]`). Normalization is consumer-side only.
2. **Proposer** (trunk or a ported pattern extractor — no runtime other-repo link) emits candidate `(subj, rel, obj)` **as spans**.
3. **Host dispose:** insert iff every span is a verbatim substring of the document. No LLM-as-judge of truth; provenance is the span. Rejects become `banned_facts`-style pairs, calibration split only.
4. Triple index + cosine search as now. Answers still require VM: `symbol == walked object AND similarity ≥ τ_vm`, plus ABSENT_ROLE.
5. Retire the previous snapshot; do not delete. Records with the old hash are not reusable (standing rule).

Unmeasured risk `[NM]`: VSA bind quality on extracted spans vs the 1,188 WikiKG triples. Do not assume τ_vm transfers.

**Edge of expressible.** If no CubeLang program type exists (count, compare, why-causal, answer not a store symbol), the host **refuses** with a new reason `inexpressible`. Unverified claims are never spoken. Retrieval miss → `retrieval_exhausted` + partial trace. VM fail → `vm_verify_failed`. Prose answers do not exist in this system.

---

## C. THE FREE WINS FIRST

They must happen **before** any emitter is trained. Positives in the harvest are `verified=true` programs. Threshold, direction, and hop-count change **which** programs exist and **what they look like**. SFT on today’s 517 verified traces teaches forward-only, grammar-counted chains.

246 `retrieval_exhausted` of 800 `[M]`; verified **517/800 = 0.646** `[M]`.

**Win 1 — τ_ret.** threshold_bound **n=36**; serving scores min **0.5028**, p50 **0.5547**, max **0.5944** against τ_ret **0.5959** `[M]`. Admitting them: **517+36 = 553**, coverage **553/800 = 0.69125** `[P]` **if** precision holds. Not free without a precision remeasure: organic catch is **0.769** not 1.0 `[M]`. Control_role_mean ≈ 0.023–0.028 `[M]` is still ≪ 0.50, so ABSENT_ROLE likely survives `[ASSUM, basis: measured gap]`. **Protocol:** drop τ_ret to 0.5028 on a clone of the 800, require claimed-answer precision **≥ 0.99** (in-domain **0.992** `[M]`) and 0 sub-τ violations (currently **0** `[M]`). Else keep τ and treat the 36 as not a win.

**Win 2 — backward retrieval.** **59 of 72** not_retrieved/far_below reach τ_ret `[M]`. Those buckets are **88+19 = 107**, so 72 is a subset. **Do not** scale 59/72 onto 107 (that would be **107×59/72 ≈ 87.7**, no source that the 35 untested look like the 72). Claim only **+59** candidates. If all 59 VM-verify: **517+59 = 576**, coverage **576/800 = 0.720** `[P]`. Combined with a precision-clean +36, disjoint buckets: **517+36+59 = 612**, **612/800 = 0.765** `[P]`. `walk_completes_now` **11** are already fixed `[M]`; do not add them again.

**Not a retrieval win — misparsed_chain 92 (37%).** Planner undercount. Fix templates / hop compiler, re-harvest. Those 92 are not SFT positives today; leaving them broken means the emitter never sees the hard hop-count cases.

**Lookup-first already taken:** coverage **0.646 → 0.710** (**+51**, 0 lost, precision **0.993**) `[M]`. **0.710×800 = 568** `[M]` derived. Residual vs lookup_correct **71.6%** `[M]` is planner refusal of gold, not missing triples.

**Arithmetic cap on the 246 if every measured retrieval win verifies and precision holds:** 36+59+11 = 106 of 246; **140** left, of which **92** are misparses. Coverage ceiling from retrieval-only on this eval ≈ **0.765**, not 0.95. Grammar v2 parse is already **95.4%** `[M]`; verified is the gap.

**Why before training.** (1) Program text changes (backward hops). (2) Positive count changes (353–612 vs 517). (3) `gold_chain_not_accepted_by_plan` **16.2%** `[M]` is exactly “what an emitter might do that the grammar cannot” **in-basin**; you cannot see it if you freeze the planner’s rejects out of SFT. Cost: one eval is **37 s** on one A100 `[M]`; serving **45.9 ms/q** at repair budget 1 `[M]`. Days, not weeks.

---

## D. THE TRAINING DESIGN

**Do not train until:** M0a–c logged; Win 1/2 accepted or rejected on precision; `store_snapshot_hash` + `git_rev` frozen; train harvest is **not** the 800 eval.

**Fields → positives (SFT).** `question` → `program_source` where `verified=true` AND `correct=true` (**513/517** in-domain `[M]`; use `correct`, not `verified` alone) AND `store_snapshot_hash` and `git_rev` match the freeze. Target is the program, not the answer (trunk proposes, VM judges). Feed **no** `parsed`, **no** `frame_id`, **no** gold chain. Preserve typos verbatim in question and in any symbol literals in the program.

**Trace tree:** do not SFT on the accepted path only as a second target; that leaks retrieval internals. After SFT, optional process loss **only** on host-checked hop queries that the VM later verified — still not model-as-judge.

**Volume.** Harvest N on the frozen snapshot is **`[NM]` — measure first** (`N_sft = count(verified & correct & hash match)`). 513-scale is template memorization, not an emitter. If `N_sft < 4,000`, **host-generate** more in-basin questions from the **1,188** triples `[M]` with the **fixed** grammar+backward walker, then serve and harvest. No source for reachable-walk count — assumption: 1–3 hop walks on 1,188 triples can supply ≥4,000 questions `[ASSUM, basis: 1,188 triples, 6 frames, hop 1–3]`. Cap generation at 20k questions to stay inside three weeks.

**Negatives / counterfactuals — never SFT.** Standing rule.  
- `reason=unparseable|retrieval_exhausted|vm_verify_failed` → **calibration/test** for refusal, not targets.  
- `banned_facts[]` (rejected fact + replacement) → **DPO on hop-choice only**, split by question id so no SFT overlap.  
- `counterfactuals[]` (1,899 planted, 0 escaped `[M]`) → **eval** that emitter+VM still catches; do not train on planted faults (the catch is string mismatch, zero learning signal for similarity).  
- Organic confusables **21/91** escaped `[M]` derived: **hold out as a named precision trap set**.

**Objective sequence (narrow).**
1. **SFT** question→program on positives only.  
2. **VM-RFT** (not self-play, not self-judging): sample K∈{4,8} programs; **host** reward `+1` iff VM verifies **and** `answer == gold_answer`; `0` on honest refuse; `−1` on VM-pass / gold-mismatch (the organic hole). Invariant 1 holds: VM+gold string compare, no LLM judge. Invariant 6 holds: no model-vs-model.  
3. **DPO** from `banned_facts` only after SFT, never mixed into SFT minibatches.

**Invalidation.** Any store or verifier change → new hash/rev; old records retired from train; re-serve. No mixing snapshots in one run.

**Typos.** Train and eval on raw text. A normalizer in the emitter is a bug: binds will miss “Huamn”.

**What stops basin memorization.** (i) Train questions are canonical WH **only** (no S1/S3 in train — otherwise OOB is leaked). (ii) Entity/chain holdout of the 200. (iii) No `frame_id` input. (iv) Kill metric is S1/S3 on **held-out chains**, not train paraphrases. (v) If `N_sft` is template-heavy, report unique `(rel sequence)` count; if that count ≈ 6, say so and expect kill.

**Model / GPU-hours.** Use the compiled trunk **1,254.4M, 40.0 GB, 18,619 tok/s, 440 ms/step** `[M]`. Fits one A100-80GB.  
Arithmetic `[P]` from that throughput: 20k programs × 512 tok = 10.24M tok ≈ **550 s/epoch** at 18,619 tok/s; 5 epochs SFT ≈ **0.8 GPU-h**. RFT: 8k train questions × 8 samples × (generate + **45.9 ms** VM `[M]`) — generation dominates; budget **≤ 8 GPU-h**. Total train **< 10 GPU-h**. Wall-clock bound is harvest/eval construction, not MFU.

Do not revive binding aux, chrono-init, multi-horizon head, entropy choice points, interpolation compression, semantic-encoding gate, hyper-encoder `[M]` all already killed.

---

## E. THE BOOTSTRAP ARGUMENT

SFT on harvest **cannot** leave the grammar’s program basin. Surface S1 at **test** is the only thing WH-only SFT can be asked to beat, and it may still emit the same six plans.

**Strongest legal escape (use this, not STaR-as-judge).** Host-side **VM-oracle program search** on questions that have `gold_answer` and a store path: enumerate short CubeLang walks (bidirectional, hop 1–4), execute, keep those that VM-verify to gold. Write them to the reserved `rationalized` field. Filter: hash/rev match, not merged from negatives, organic-confusable trap set held out. This is **search with the VM as oracle**, host dispose. It is not self-play and not LLM-as-judge. That is how the 92 misparses and the **16.2%** gold-not-accepted hops become supervision.

Inverse generation (sample walks, then template a question, including S1) expands **train** shape only if we allow it; **do not**, or the S1 kill is leaked. Use inverse generation only to grow **canonical-WH** volume.

**Why it fails here specifically.** (1) Oracle search only finds programs for **reachable** golds. SimpleQA **0.1%** `[M]` → `rationalized` stays empty off-store. (2) Gold-conditioned search Goodharts: any walk that lands on the gold **string** (typo’d orthography included) certifies; τ_vm at 3 hops is **0.2202** `[M]` and planted faults were caught by **symbol mismatch**, not similarity — confusable entities can VM-pass. (3) CubeLang’s expressible set is role-walks over this relation vocabulary; search will not invent aggregation. (4) Invariant 6 forbids the usual STaR move (model writes a rationale, model accepts it). If `rationalized` is filled by the trunk without VM success, that run is invalid.

---

## F. THE EXPERIMENT

**Order (three weeks, one A100).**  
W0 (days 1–3): M0a–c; Win 1/2 with precision veto; freeze hash/rev. **No train.**  
W1: re-serve harvest on frozen stack; VM-oracle fill of `rationalized` on the 246 + gold-not-accepted hops; build 200-chain holdout with S1/S3.  
W2: SFT ± RFT as specified; eval.

**Splits (pre-registered).**
- **T_in:** held-out 200, canonical WH, same store. Grammar baseline = remeasure after free wins, not 0.646 if wins land.
- **T_S1, T_S3:** same 200 chains, transforms. Grammar expected ~0 parse.
- **T_trap:** 91 organic confusables `[M]`.
- **T_real:** NQ 300 and SimpleQA 4,326 **logged, not kill**.

**Metrics (all `[M]` when run):** verified coverage; claimed-answer precision; first-attempt certification rate; retry count (budget 1 vs 3 already **identical** verified/correct/precision `[M]` — expect retries ≈ 0 value); mean program length in hops and tokens `[NM]` today; refusal reasons; grammar parse rate on the same set.

**Baselines.** Frozen grammar after free-win decision. Emitter SFT-only. Emitter SFT+RFT. Emitter SFT+`rationalized` (oracle programs). No LLM judge arm.

**Kill (replaces the founder’s).** On **T_S1 ∪ T_S3** (facts in store, 200 chains × 2 transforms = 400 items):
- grammar verified coverage is the measured floor (expect ≈ 0);
- **kill the general trunk** if best emitter verified coverage **< 0.10** **or** precision **< 0.95**;
- **also kill** if T_in precision **< 0.99** (in-domain **0.992** `[M]` — do not buy OOB with lying);
- **do not kill** on T_real. If T_real stays ~0, that is Gate 1b (ingest), already predicted by **3/4,326**.

If T_S1 ≥ 0.10 at precision ≥ 0.95, there is a **program emitter** worth training further. There is not yet a general QA lab. Write that sentence in the report.

**Cost.** ≤ **10 GPU-h** train `[P]` + eval **37 s** per full pass `[M]` + VM at **45.9 ms/q** `[M]`. Wall-clock: **15 working days** if W0 is not skipped; **< 48 h** of GPU.

**Pre-registered predictions (stated before run; not results).**
- M0a NQ parse **0–5/300** `[P]`; M0b **~15%** of 220 `[P]`; M0c S1 on 800-slice parse **≈ 0** `[P]`.
- Win 1: **+36** verified only if precision ≥ 0.99; else **+0** `[P]`. Win 2: **+40–59** verified of the 59 `[P]` (VM still has to pass). Combined coverage **0.70–0.765** `[P]`, not 0.95.
- SFT emitter on T_in: within **0.03** of post-win grammar coverage `[P]` (distillation).
- SFT emitter on T_S1: **0.00–0.15** verified `[P]`, mode near 0 if unique rel-sequences stay ~6 — **I expect to be near the kill bar, not safely above it.**
- SFT+`rationalized` on T_in: can beat grammar by up to the **16.2%** gold-not-accepted slice, i.e. **≤ ~0.07** absolute if those hops convert end-to-end `[P, basis: 71.6% lookup_correct − 64.6% verified, and 16.2% hop-level]`.
- T_real NQ/SimpleQA verified **≈ 0** for grammar **and** emitter `[P]`.
- Retries: no coverage gain, as budget 1 vs 3 `[M]`.
- T_trap catch **≤ 0.769** `[P, cannot beat organic without a new verifier]`.

---

## G. WHAT WOULD FALSIFY THIS DESIGN

If M0a puts NQ parse near disfl_qa **15%** or M0c shows the 800 still parse after S1, I misread 0/300 as surface and the cheap probe I demanded was the wrong split — then a real-question kill is fairer than I claimed, **but only after ingesting facts**, which this design still refuses to skip. If Win 1/2 move coverage a lot and the SFT emitter then beats grammar on T_S1 **without** oracle search, the bootstrap worry is overstated and WH-only SFT already abstracts off frames — the interesting result, and I should have predicted it higher than 0.15. If oracle-filled `rationalized` programs VM-verify to gold but fail T_trap worse than **21/91**, the escape hatch is Goodhart and must be retired, not trained on. If T_S1 ≥ 0.10 while mean emitted hop length rises into the n_hop=3 τ_vm **0.2202** regime and precision falls below 0.95, the gate is not “generality,” it is the verifier collapsing, and training a bigger trunk will not fix it. The design is also wrong if document span-extraction (B) cannot produce triples the VM will bind at in-domain precision — in that case shape-OOB on WikiKG is a toy and I will have spent three weeks proving a grammar can be paraphrased, which is not a lab.

---

## kimi-k3

# Generality Gate — Design Submission

**Up-front position:** the gate as framed compares the wrong things. "Emitter vs grammar out of basin" is uninformative as stated — the grammar scores 0 by construction on a true out-of-basin set, so the emitter wins by emitting anything, and ties at 0 if the store can't bind. The gate only means something stratified by store reachability. Details below; the kill criteria in F are rewritten accordingly.

---

## A. THE DIAGNOSIS

**The 0/300 — my call, and the measurement that settles it.**

The named cheap measurement (cost: <1 engineer-day, zero GPU, run before anything trains):

1. **Surface probe:** mechanical repair of the NQ-300 (capitalize first token, append `?`), re-run through grammar v2 *unchanged*, count parses. Twenty lines of code.
2. **Expressibility audit:** two annotators label all 300 as {in-frame / in-CubeLang-but-not-in-frame / needs-an-operator-CubeLang-lacks}, recording the relation sequence for expressible ones. Same audit on 200 of DisflQA's 850 unparsed clean questions [M: 150/1000 parsed]. ~3 hours.

My belief, stated plainly: **surface is ≤10% of the 0/300; semantics is ≥60%.** Basis: DisflQA clean questions are well-formed interrogatives and still parse at only 15.0% [M]; NQ's profile (0/300 `?`, 300/300 lowercase, 9.3 words [M]) describes verbal-relation queries ("who played X in Y") against nominal frames ("who is the X of Y"). Pre-registered probe predictions: post-repair parse 3–10% [P]; audit: 25–45% in-frame, 40–60% needing missing operators [P]. If post-repair parse ≥40%, I'm wrong, surface dominates, and the emitter is an over-built solution to a preprocessing bug — that outcome *downgrades the whole rung*, which is why the probe runs first.

**Ranked blockers (share × tractability):**

| # | Blocker | Share evidence | Tractability | Verdict |
|---|---|---|---|---|
| 1 | Frame expressibility ceiling | gates ~100% of OOB: 0/300 [M], 15.0% [M] | the emitter question itself | primary target |
| 2 | Store reachability | SimpleQA: 19.9% entity-present, 0.1% reachable [M] | low at web scale, medium for curated corpora | caps everything; founder underweights this |
| 3 | Planner hop-count | 92/246 = 37% [M]; under-counts in 67/92 [M-derived: 25+38+4] | high (retry or emitter) | free-ish; also the emitter's best in-domain target |
| 4 | Retrieval direction + threshold | 88+19+36 = 143/246 = 58% [M]; 59/72 backward-recoverable [M]; τ 0.5028 admits all 36 [M] | high, measured, unacted | fix this week |
| 5 | Verifier strength at depth | τ floors 1.0 / 0.4736 / 0.2202 at 1/2/3 hops [M]; organic catch 0.769 [M]; planted catches 100% by string, 0% by floor [M] | medium | quiet cap: fixing #3 pushes traffic into the weak-verification regime |
| 6 | Trunk as encoder | 0.0962 macro [M] | — | do not spend; retrieval stays symbolic |

**Where I disagree with the founder:**

- **The gate metric is mis-specified.** On a genuinely out-of-basin set the grammar is 0 by construction; "emitter beats grammar" rewards emission rate, not verified coverage. The comparison must be **verified coverage on the store-reachable stratum** (H3 below), with precision and refusal correctness as contract gates.
- **"Supervision is a by-product of serving" is half-false.** Serving traffic is router-filtered (router reads 220/300 NQ as world-questions [M]; router agrees clean/disfluent 992/1000 [M]) — the harvest inherits the router's and grammar's biases. Out-of-basin supervision must be *manufactured* (§E), which is a deliberate cost line, not a by-product.
- **"Proves whether there is a frontier lab here" overclaims.** A failed gate may indict the store, not the emitter; a passed gate on a tiny reachable stratum proves little. The gate proves exactly one thing: *whether a learned emitter adds verified coverage over the best grammar on store-reachable, out-of-frame questions.* I add a store-kill (K2) alongside the emitter-kill.

---

## B. THE SPEC (the "Know" deliverable)

**The emitter must do four things the grammar cannot:**

1. Map arbitrary interrogative surface (no `?`, lowercase, disfluent, verbal relations) to CubeLang chains — decouple parsing from the six frame templates.
2. Choose chain length from the question's relational structure (this is the 92/246 class [M]; the grammar under-counts in 67/92 of them [M-derived]).
3. Compose *any* relation sequence in the store's relation vocabulary, not just frame-licensed ones.
4. Emit a reserved `UNSUPPORTED_OP` token when the question needs operators CubeLang lacks (aggregation, temporal, comparison) — never a best-effort program. Trained from the audit labels (§A), which makes this the first measurable use of the probe.

**Out-of-basin, operationally (frozen before training):** basin := the set of (frame_id, relation-sequence) pairs grammar v2.1 can emit. This requires `frame_id`, a reserved-but-unemitted field — **turn it on now; cost ≈ 0, and the basin definition, the splits, and the goalposts all depend on it.** Held-out sets, content-addressed by `store_snapshot_hash`, construction script published pre-training:

- **H2 (surface OOB):** NQ-300 post-probe + DisflQA disfluent n=1000 [M sizes].
- **H3 (semantic OOB, reachable):** external QA whose gold answer entity-links to the store AND has ≥1 gold chain in the triple index. Linking is host-side string matching (consumer-side normalization permitted, logged).
- **H4 (unreachable):** gold entity absent from store. Metric is refusal correctness — the don't-know contract made measurable.

**Document → bindable facts:** two-tier store. Tier 1 is the current curated triple store (1,241 facts, 95.7% triple-parseable [M]). Tier 2 is a claim store: host-side extraction (ported in-repo per invariant 7), each claim = (subj, rel, obj, doc_id, span offsets, snapshot_hash). Admission = parses-as-triple + symbols resolve or are host-minted with provenance. **No truth is asserted** — the VM checks binding fidelity, not truth (on-record caveat [M]); tier-2 answers carry span provenance for audit; unextractable content yields no claims and therefore refusals. Orthography is inherited verbatim (the "Huamn" load-bearing typo pattern: 568 answers / 1,098 facts [M]). Rung 1 runs on tier 1 only; tier 2 is a spike whose only job is sizing H3.

**Edge of expressible:** `UNSUPPORTED_OP` → host speaks "cannot express," distinguished from "cannot verify." Both are refusals; conflating them corrupts the harvest labels.

---

## C. THE FREE WINS FIRST

Consistency check: 517 verified + 246 exhausted + 37 unparseable = 800 [M-derived]. Arithmetic on the 246:

| Fix | Class | Recovery | Cumulative coverage |
|---|---|---|---|
| F1: admit retrieval in [0.5028, 0.5959) — measured min of the class [M] | threshold_bound (36) | +36 | 553/800 = **0.691** [P from M] |
| F2: backward (object-side) walk | not_retrieved + far_below (107) | +59 measured (59/72 [M]); extrapolated 107 × 59/72 = 87.6 → +87 [P] | 612 = **0.765**; 640 = **0.800** [P] |
| F3: on walk-exhaustion, retry n_hop±1 (67 need +1, 25 need −1 [M-derived]) | misparsed_chain (92) | rate [NM]; pre-registered 30–50% [P] → +28..46 | **0.80–0.86** central; ceiling 763/800 = 0.954 (parse-limited) [P] |

Caveats: interaction with the lookup-first +51 [M] is unmeasured [NM] — re-run the decomposition after landing fixes. F1 touches τ_ret, not τ_vm; precision is protected by the downstream VM gate (symbol match + absent-role, 528/528 [M]), but re-run the organic-confusable suite (70/91 [M]) at the new τ_ret before declaring victory.

**Why before any emitter trains:** (i) the fixes re-label the harvest — every recovered question flips from negative to positive, and training pre-fix teaches the emitter that threshold_bound-shaped programs *fail*; (ii) snapshot invalidation (invariant 4) is cheap now and expensive after a training corpus exists; (iii) the emitter's bar must be the *best* grammar (~0.80 [P]), not 0.646 [M], or the gate is gamed in the emitter's favor; (iv) sequencing cost is days of host-side code and zero GPU — mis-sequencing buys a confounded gate. The only work that may run in parallel is outcome-independent: schemas, harness, the perturbation factory.

---

## D. THE TRAINING DESIGN

**SFT records.** Input: question verbatim. Target: the accepted CubeLang chain (relation tokens from the store vocabulary; entity anchors in store orthography, typos included — binding depends on it [M]). Filters: `verified=true`; `store_snapshot_hash` + `git_rev` == training manifest; dedup by (program skeleton, answer entity); `repairs_used` ≤ 1 weighted 1.0, >1 weighted 0.5 [choice]. `n_hop`, `answer_class`, `hops_verified_before_failure` are stratification/consistency fields, never targets.

**Negatives, without merging into SFT (standing rule preserved):**
- **Program-level DPO:** from trace trees (this is what trace-tree storage is *for*): chosen = verified program, rejected = same question's VM-failed earlier attempt (`repairs_used` > 0 traces). Judge = VM. Invariant 6 satisfied.
- **Counterfactual negatives:** mutate verified programs along the four measured fault classes [M]; keep as DPO-rejected only where the VM catches (`caught=true`; planted escape rate is 0/1,899 [M], but organic escape is 0.231 [M] — any escaping mutation is logged as a verifier hole and *never* used as a negative). Four free negatives per positive [M].
- **`banned_facts[]`** are retrieval-level: they train a ranker/τ calibration, never the emitter — §3.7 [M] forbids making the trunk a retriever.
- `unparseable` / `retrieval_exhausted` records: calibration/test split only.

**Snapshot invalidation:** manifest keyed to (store_snapshot_hash, git_rev). On change, records retire (never delete). Re-verification pass re-labels under the new snapshot: at 45.9 ms/question [M], 100k records ≈ 77 min single-core, minutes parallelized [P from M].

**Typos:** verbatim everywhere, store-side and anchor-side. Eval includes a typo-consistency slice. Any consumer-side normalization is a separate, logged arm.

**Anti-basin-memorization:** three held-out splits — (frame_id × relation-sequence), entity-cluster, and H2/H3 — plus canary surface perturbations of train questions. **Arm 2 (SFT-only) exists precisely as the memorization control**: it is *expected* to reproduce the basin; the gap between Arm 2 and Arm 4 is the measurement of escape.

**Supervision factories (host-side, no self-play):** (i) grammar-as-generator over the triple index for unlimited in-domain pairs, VM-verified; (ii) mechanical perturbation — case/`?` stripping, injection of the measured disfluency cues ("no" ×565, "sorry" ×127, "rather" ×119, "make that" ×92, "uh" ×82 per 1000 [M]) — applied to factory questions; (iii) §E's answer-anchored search.

**Volume / size / compute.** Program length ~10–40 tokens [ASSUM: per-hop BIND/UNBIND, n_hop ≤ 4]. Target 50k–200k SFT records [P requirement; current harvest size NM — if short, the factories close the gap]. 100k × ~30 tokens ≈ 3M tokens; at 18,619 tok/s [M] → ~160 s/epoch; 10 epochs ≈ 27 min on one A100 — inside invariant 5's "tens of minutes." RFT: k=8 samples × 50k questions = 400k programs; decode 1–2 h [P]; VM filtering 400k × 46 ms ≈ 5.1 h single-core → <40 min parallel [P from M]. DPO < 2 h [P]. **Total core loop ≤ 8 GPU-hours; ≤ 30 with 3 seeds** [P]. Cost is not the constraint; being wrong slowly is.

---

## E. THE BOOTSTRAP ARGUMENT

**Strongest mechanism: answer-anchored backward chain synthesis — model-free STaR.** For any unsolved question with a known gold answer: enumerate chains backward from the gold answer's symbol through the triple index (the store is 1,241 facts [M] — exhaustive enumeration to depth 4 is trivial), execute each candidate in the VM, and keep chains that verify *and* pass the absent-role control *and* are the unique verifier. The model proposes nothing here; the index and VM do everything. This is legal under invariant 6 because the gold answer is data, the judge is the VM — the illegitimate version (trunk-scored rationales, unverified rationalizations in SFT) is what invariant 6 forbids, and this is not that. It escapes the grammar's basin because

---

## glm-5.3

Let me carefully work through this. The task: produce a rigorous, falsifiable research design for "the generality gate" — sections A through G. Every number must be tagged [M], [P], [ASSUM], [NM], or derived with arithmetic. No fabrication. Must respect invariants. Must not re-propose falsified things. Must disagree with the founder where the data says so.

Let me first digest the data and figure out the real story.

Key facts:

**Baseline:** grammar parses 0/300 NQ natural questions. But: `ends with '?' 0/300`, `lowercase start 300/300`, mean 9.3 words. Question detector fires on 220/300. NQ queries are search queries, not interrogatives. So the 0/300 conflates surface form (no question mark, lowercase, keyword-query style) with semantics (question shapes out of the grammar's 6 WH-frames). The founder's own text says "How much of the 0/300 is surface form versus semantics is unmeasured, and it is probably the first thing to find out."

**disfl_qa:** clean 15% parse, disfluent 12.9%. So even on well-formed questions the grammar parses only 15%. This is the crucial number! On disfl_qa, which ARE well-formed questions (with disfluency variants), the grammar parses only 15% clean. So the grammar is narrow — it parses only its own WikiKG-derived template shapes. The 0/300 on NQ is partly surface form, but the 15% on disfl_qa tells us the semantic coverage is also narrow. Actually wait — disfl_qa is derived from SQuAD, so questions about passages. The grammar has 6 WH-frames tuned to WikiKG relation shapes. 15% parse on natural SQuAD-style questions. So even if you fixed surface form (add question marks, capitalize), you'd expect maybe ~15% parse, not much more. That's a key inference: surface-form repair can at most move NQ from 0% toward the disfl_qa-like ceiling of ~15% [P/derived]. Actually we can't directly say that — NQ and disfl_qa are different distributions. But it's a reasonable projection: the grammar parses ~15% of natural questions in one natural distribution; NQ is another natural distribution; surface normalization would likely land in a similar range, not near 100%. So the claim "the emitter must beat the grammar out of basin" — the grammar is at ~0 on NQ, so almost anything beats it. The gate as stated is nearly unfalsifiable on the emitter side: any emitter that parses >0/300 beats the grammar. That's a problem with the founder's kill criterion: it's a weak gate. I should disagree: the baseline of 0/300 makes "beat the grammar" trivially achievable; the gate needs a second baseline — e.g., the grammar after the two free wins + surface normalization (a "grammar-maxed" arm), and a calibrated don't-know/refusal baseline. Also the real question is verified-correct, not parse.

Let me think about the failure decomposition (§3.3): 246 retrieval_exhausted of 800. Causes: misparsed_chain 92 (37%), not_retrieved 88 (36%), threshold_bound 36 (15%), far_below 19 (8%), walk_completes_now 11 (4%).

Free wins:
1. threshold_bound: 36 cases, serving scores min 0.5028 vs τ_ret 0.5959. Lowering τ_ret to 0.50 (or per-frame calibration) admits all 36. But caution: lowering τ_ret globally might hurt precision. The measured precision is 0.992–0.993; the verifier (symbol match) is the truth gate, so admitting more retrieval candidates risks... actually retrieval threshold τ_ret gates which facts are considered; the VM verify is symbol==walked object AND similarity ≥ τ_vm. If we admit lower-scored retrievals, we may pull in wrong facts, which the VM would catch (symbol mismatch) → more refusals? Or the walk might bind a wrong-but-passing fact? The planted-fault data says symbol_mismatch catches 100% of planted wrong-entity/wrong-relation faults. Organic confusables 0.769. So lowering τ_ret admits more confusables; 76.9% get caught. So some risk. But the measured claim: "a tau of 0.5028 would admit all of them" — that's about admitting the threshold_bound cases. The right move: per-frame or per-hop τ_ret calibration, or a two-stage: admit top-k above a lower floor and let the VM arbitrate. Quantify: 36/246 = 14.6% of failures; coverage 0.646 → potentially +36 verified if all convert → 0.646 + 36/800 = 0.691. But not all will convert — some may fail at later hops. Give a range.

2. Backward retrieval: 59 of 72 not_retrieved/far_below reach τ_ret with object-side retrieval. 72 = 88 + 19 = 107? Wait: not_retrieved 88, far_below 19 → 107. "59 of 72" — hmm, 72 ≠ 107. Let me re-read: "Backward (object-side) retrieval on the not_retrieved/far_below cases: 59 of 72 reach τ_ret." Hmm, maybe the backward probe was run on a subset of 72? The log says 59 of 72. Maybe far_below 19 + not_retrieved 88 = 107, but only 72 were probed? Or maybe "72" is the count of cases where backward retrieval was attempted (some may have been excluded because the hop had no object anchor?). I should be careful: the text says 59 of 72. I can't reconcile 72 with 107. I should flag this discrepancy explicitly — that's the honest move. Possibly the 72 is the subset with a well-formed object anchor, or the probe sample. I'll note: the log reports 59/72; the union of not_retrieved+far_below is 107 [M]; either the probe covered 72 of 107 or the classes overlap; the design should re-run the backward probe on all 107 before banking the win. Good — that's the kind of rigor the persona wants. Actually, let me reconsider: maybe far_below is a subset of not_retrieved? The table lists them separately: not_retrieved 88, far_below 19. 88+19=107. 72 might be... hmm, 88 - 16 = 72? Or the probe was on the 72 first-hop cases? Failing hop index {0:152, 1:63, 2:28, 3:3} — total 246. First-hop failures 152. Hmm, 72 could be the not_retrieved cases at hop 0? Unknown. I'll flag it as an unreconciled number and require a re-measure. That's honest and it's a good look for a rigorous engineer.

So free wins arithmetic on 246:
- threshold_bound: 36 cases → if τ_ret lowered to 0.50 (per-frame calibrated), admit all 36. Conversion assumption: some fraction then verify. Upper bound coverage: (517+36)/800 = 0.691 with lookup-first it's (561+36)/800? Wait — §3.2 says with lookup-first triple index: coverage 0.646 → 0.710, +51 verified, 0 lost. So 0.710 = 568/800. Hmm: 517+51 = 568, 568/800 = 0.710. Yes. So the current system with lookup-first is at 0.710. The 246 decomposition is from the 800-question eval — is it pre- or post-lookup-first? The exhaustion decomposition log is exp_m3; the lookup is exp_m4. The 246 failures are of 800 at coverage 0.646 presumably (pre-lookup). With lookup-first, coverage 0.710, so failures drop to 232. Do the free wins overlap with the lookup gains? The threshold_bound and not_retrieved classes might partially be served by the lookup index. Hmm. §3.4: lookup serves 83.8% of hops vs cosine ceiling 78.7%; gold_chain_not_accepted_by_plan 16.2%. End-to-end on index alone: lookup_correct 71.6%, lookup_dead 22.8%, unparseable 4.6%.

So the honest arithmetic: I should present the free wins on the 246 from exp_m3, note the possible overlap with the m4 lookup gains, and require a single re-baseline run that applies: (a) lookup-first, (b) backward retrieval, (c) τ_ret recalibration, and measures the joint coverage — because gains may overlap (a case fixed by lookup may also be a threshold_bound case). Pre-register an expected joint coverage: upper bound if disjoint: 0.710 + (36 + 59 + maybe more of the 107)/800... Let me compute: post-lookup coverage 0.710 (568/800, 232 failures). If threshold_bound 36 and backward-fixable 59 are disjoint from lookup gains and from each other: 568+36+59 = 663 → 0.829. But misparsed_chain 92 remains (that's the emitter's job — actually no, misparse is the planner/grammar's hop-count error; an emitter could fix it, but also a better parser could). Actually misparsed_chain is a grammar/planner failure — the emitter's core target. So after free wins, remaining failures ≈ 232 − 36 − 59 − (some far_below remainder) ≈ 137ish, of which misparsed_chain ~92 is the dominant class → the emitter's measurable target on in-domain is hop-count planning. Hmm, but the gate is about out-of-basin. Still, the free wins matter because: (1) they change the baseline the emitter must beat; (2) they change the harvest distribution (more verified chains → more supervision, and the newly recovered chains are exactly the borderline cases that make better training data); (3) if you don't fix them, the emitter gets blamed/blamed-confused for failures that are retrieval's fault, and the SFT data is biased toward easy retrieval paths.

Wait, actually there's a subtlety: should the free wins happen before the emitter? The prompt says "say why that must happen before any emitter is trained, or argue that it must not." Arguments for before: (a) the baseline must be the strongest grammar, else the gate is unfair/trivially passed — no wait, a stronger baseline makes the gate harder, which is what you want for falsifiability; (b) harvest quality: fixing retrieval changes which chains verify, so the SFT set changes; training before the fix means training on a distribution that will be retired (violates retire-never-delete / snapshot discipline: records without store hash+git rev can't be reused after store/verifier change — fixing τ_ret and adding backward retrieval is a verifier/pipeline change → old records invalidated → wasted training). That's a strong argument: invariant 4 + the standing rule means any harvest taken before the fix is invalidated by the fix. So fix first, then harvest, then train. Cost: the fixes are retrieval-side, milliseconds, no training. Wall-clock: days.

Also misparsed_chain 92: is that a free win? The misparse is the planner under-counting hops. Could be fixed by grammar changes (the openers listed: "what is the source that" ×21 etc.). But that's grammar engineering — the founder's endgame is to replace the grammar with the emitter. Fixing the grammar's hop counter expands the basin by grammar engineering, which is exactly the "bounded-domain product" path. Hmm. But actually for the gate's fairness, we might want a "grammar-maxed" baseline: grammar + free wins. The misparse fix is more invasive; I'd argue: implement the two measured free wins (τ_ret recalibration, backward retrieval) because they're retrieval-side and cheap; do NOT chase the 92 misparses with more grammar frames, because (a) each new frame is hand-written basin expansion — it moves the goalposts in the grammar's favor and consumes the two weeks on exactly the work the emitter is supposed to replace; (b) the misparse cases are perfect emitter test material — keep them as a held-out slice. Actually wait — but if we keep them as failures, the emitter can win on them, which is fine and is the point: hop-count planning is a semantic skill the emitter should learn from the harvest (the harvest has n_hop, parsed, gold n_hop... hmm, does the harvest have gold n_hop? It has `parsed` and `n_hop` and `answer_class`... The misparse table says "parsed n_hop, gold n_hop" — so gold hop counts exist for at least the eval. For harvest records, `n_hop` is the parsed one presumably. Hmm — for training the emitter to plan hop counts, we need gold hop counts. On verified chains, the executed program's hop count is validated by the VM (each hop verified), so verified chains give gold n_hop for free. On failed chains, we don't have gold n_hop unless... the misparse analysis derived gold n_hop somehow (maybe by retrying with different hop counts — a search over hop counts). So one mechanism: hop-count search at harvest time (try n_hop ∈ {parsed, parsed+1, ...} within budget) converts misparsed failures into verified chains with correct n_hop → those become supervision AND error-localized DPO pairs (parsed=1 vs gold=2). That's a nice concrete mechanism: budget-limited hop-count retry. Cost: a few extra VM walks per failing question, milliseconds each. That could recover a chunk of the 92. But careful: is that "self-play"? No — the VM verifies; retry with different hop counts is host-side search, the VM is the truth gate. It's like repair budget. The system already has repairs_used and budget 1 vs 3. So hop-count retry is a repair class. This is arguably a third free win, but it wasn't measured — I should tag it [P] and propose measuring it in the re-baseline run. Actually the misparse table itself (parsed vs gold n_hop) implies someone measured gold n_hop — likely by executing with corrected hop counts. The pattern {(1,2):25, (2,3):38, ...} — if gold n_hop was found by search, then the recovery rate is knowable. Not stated. So: propose a cheap probe: for the 92 misparsed cases, re-run with gold n_hop (search ±1) and measure how many verify. Pre-register an expectation. If most verify, that's another free win — but wait, that's grammar-basin expansion via search, and it directly feeds the emitter's training data with correct hop counts. Hmm, but if a hop-count search makes the grammar+search baseline much stronger, the emitter's job shrinks. But the emitter's real job is out-of-basin anyway. For the in-domain baseline, grammar+search is a legitimate strong baseline.

Hold on — is hop-count retry allowed under "the model proposes, the host disposes"? Yes: host-side retry is a repair, like budget 3. Fine.

Now the diagnosis section A. Rank causes by share × tractability:

Out-of-basin failure (NQ 0/300): 
- Surface form: no '?', lowercase, 9.3 words, keyword style. The question detector fires 220/300 — so 80/300 aren't even detected as questions. The router reads 220 as world-questions. So surface normalization (question-mark, capitalization, query→interrogative rewrite) is a measurable, cheap fix. Cheap measurement to settle surface vs semantics: take the 300 NQ queries, apply a minimal surface normalization (host-side, rule-based: capitalize first token, append '?', maybe expand abbreviations?) — no, minimal: just append '?' and capitalize. Then re-run the grammar. Parse rate = surface-recoverable share. But we already know from disfl_qa that well-formed natural questions parse at only 15% — so expected post-normalization NQ parse is low, maybe ~10-15% [P]. Actually, better measurement: reverse direction — take the grammar's 6 WH-frames, and check how many NQ queries match a frame after normalization. Hmm, the grammar is the parser; just run it.

Actually, the cleanest cheap measurement: three-way decomposition on the 300:
1. Normalize surface (add '?', capitalize, lowercase-start fix — wait they're already lowercase start; the normalization is: append '?', capitalize first letter). Re-run grammar → parse rate p1. p1 − 0 = surface-attributable share.
2. For still-unparsed, ask: does the question's answer require a fact shape the store can express (a triple walk)? This is harder. Alternative: have the grammar log WHY it failed (which frame matched partially, where it diverged) — failure-reason logging. Or: hand-label a 50-query stratified subsample: is this answerable by a 1-3 hop triple walk over a general KB? That's a human annotation, cheap (50 queries, an afternoon), and it directly measures "semantically in-grammar vs out-of-expressible." Hmm, but "hand-label" — is that LLM-as-judge? No, human annotation is fine; invariant bans LLM-as-judge in the loop, not human labeling of an eval set. But careful: the deliverable must be falsifiable; human labels on 50 with inter-annotator agreement (two annotators, report kappa) is standard.

Actually there's an even cheaper, fully mechanical measurement: the store-side reachability. §3.6: SimpleQA gold is a graph entity 19.9%, lookup-reachable 0.1%. For NQ, we don't have that measured. But the point: even a perfect emitter can only answer what the fact store can bind. The 0/300 conflates: (a) surface, (b) frame coverage (grammar's 6 WH-shapes), (c) store coverage (facts not present), (d) expressiveness (question needs aggregation/comparison/negation that triple walks can't express). The cheap measurement should separate grammar-parse failure from store failure: for each NQ query, after normalization, attempt a relaxed frame match (e.g., WH-word + anchor entity present in store?) — measure: fraction of NQ queries whose answer is even present in the 1,241-fact store... wait, what store is mounted for the NQ probe? The standin probe used natural_questions dev — with which store? Unclear [NM]. If the WikiKG store (1,241 facts? no — §3.4's store is 1,241 facts for the 800-question eval; §3.6 says 45k-article wikikg mounted for SimpleQA). The NQ probe's store is not specified — flag as [NM], must be pinned before the gate runs. This matters enormously: if the NQ probe ran against a store with no NQ-relevant facts, then 0/300 is uninformative about the grammar and the emitter alike — everything fails at retrieval regardless. The gate design must mount a store that covers the eval questions' facts, else the gate measures store coverage, not emitter generality. This is a crucial design point: out-of-basin must mean "question shapes out of the grammar's basin" while facts are in-store; otherwise the emitter is being tested on retrieval impossibility. So construct the held-out set with a store-projection guarantee: every held-out question's gold chain is present in the mounted store snapshot (verifiable by construction: generate questions from the store's own triples with held-out shapes/relations, or filter NQ by gold-answer-in-store). Hmm, but generating from the store risks re-entering the WikiKG template basin. Two-part held-out set: (i) store-anchored natural questions (NQ/disfl_qa filtered to gold-in-store) — measures emitter vs grammar on natural shapes with reachable facts; (ii) shape-held-out synthetic from held-out relations/domains — measures generalization to unseen relation types with guaranteed reachability. Both need gold chains for verification... wait, verification is per-hop against walked facts; the VM verifies binding, not truth. Correctness vs gold_answer is the eval's job (harvest has gold_answer, correct). For held-out, we need gold answers: NQ has gold answers; synthetic-from-store has gold by construction.

OK so now the "Know" spec (section B): what must the emitter do that the grammar cannot?

The grammar: 6 WH-frames, hand-written, forward-only walks, fixed hop planning (under-counts), fails on surface variants (0/300 NQ; 15% disfl_qa), 95.4% parse in-domain.

The emitter must:
1. Parse surface-variant questions (no '?', lowercase keyword queries) — map arbitrary question shapes to a program. This is the surface robustness the grammar lacks (0/300 vs 15% suggests surface is a real but not the whole gap).
2. Plan correct hop counts (fix the 37% misparsed_chain class) — semantic planning.
3. Choose retrieval direction per hop (forward vs backward) — the walk is forward-only; 59/72 recoverable backward. The emitter should emit direction-annotated hops (or the host executes both and VM arbitrates — but better: emitter learns direction from harvest where backward-recovered chains exist as positives).
4. Emit ABSENT_ROLE probes when the question implies absence (don't-know contract) — actually the host does that; emitter should decide when to probe absence.
5. Refuse at the edge of expressible: emit a typed "not expressible as a walk" token (aggregation, comparison, temporal arithmetic, negation beyond absent-role, multi-entity set questions) → honest refusal with partial trace, rather than a forced wrong program. The don't-know contract requires the emitter to have an explicit refusal program form. Edge-of-expressible behavior: the emitter emits a typed refusal with the reason class (needs_aggregation | needs_comparison | needs_negation | needs_temporal | entity_not_in_store) — and the eval scores refusal calibration: of questions labeled inexpressible, refusal rate; of expressible, refusal = false refusal.

Out-of-basin operational definition — must be constructible and goalpost-proof:
- Basin definition: the grammar's parse is the basin marker. A question is in-basin iff grammar v2 (with free wins) parses it AND its frame_id ∈ the 6 frames. Out-of-basin: (a) grammar-maxed fails to parse it (surface-normalized), AND (b) its gold answer chain is present in the mounted store snapshot (reachability by construction), AND (c) it's answerable by the program algebra the VM can execute (≤ k hops, walk + absent-role; no aggregation) — verified by construction for synthetic, by annotation for natural. Pre-register the exact filters and freeze the set with a hash before training. Nobody can move goalposts because the set is hashed and the membership rule is mechanical.

Also need to guard against leakage: emitter trained on harvest from grammar-verified chains; held-out must not overlap harvest questions. Dedup by question text and by (subject, relation) chain? The harvest comes from serving traffic; held-out from NQ/disfl_qa/synthetic. Overlap risk: low, but dedup by normalized question string.

Now section C arithmetic on the 246:

Free win 1 — τ_ret recalibration (threshold_bound, n=36): measured serving scores min 0.5028, p50 0.5547, max 0.5944 vs τ_ret 0.5959. Lowering τ_ret to 0.50 admits all 36. But global lowering risks precision: measured claimed-answer precision 0.992 at τ_ret=0.5959. The VM catches planted faults 100% via symbol mismatch; organic confusables 76.9% caught. So admitted near-threshold facts that are wrong will mostly be caught → converted to refusals, not wrong answers. Expected effect: coverage +up to 36/800 = +4.5 points (0.646 → 0.691 pre-lookup; on post-lookup base 0.710 → up to 0.755 if disjoint), precision drop bounded by organic-confusable escape rate 23.1% of admitted confusables... hmm, need care: precision risk = (fraction of newly admitted facts that are wrong) × (1 − 0.769) × (they pass symbol match?) — wait, organic confusables caught 70/91 = 0.769 by the VM. If a wrong fact is admitted and not caught, it can produce a wrong verified answer. But symbol_mismatch caught 100% of planted faults; organic 76.9%. So expected precision after lowering τ_ret: monitor; pre-register: precision ≥ 0.98 required, else per-frame τ_ret instead of global. Better: per-frame calibrated τ_ret like the τ_vm v2 fix (frame-size-conditional floors) — the τ_vm degeneration lesson applies: calibrate τ_ret per frame on the harvest's candidates_topk (runners-up are free negatives!). That's exactly what candidates_topk is for. So: calibrate τ_ret per frame using candidates_topk negatives + verified outcomes; pre-register expected recovery 25–36 of 36 (some may fail later hops).

Free win 2 — backward retrieval (not_retrieved 88 + far_below 19 = 107; log says 59/72 reach τ_ret): recoverable ≈ 59 (if 72 was the probed subset, re-probe all 107; expected 59 × (107/72) ≈ 88 if the rate holds [P] — but I must be careful: 59/72 = 82% recovery rate; applied to 107 → ~88 [P]. But flag the 72 vs 107 discrepancy honestly.) Coverage: +59 to +88 on 800.

Joint: post-lookup-first coverage 0.710 (568/800, 232 failures) [M]. Free wins upper bound: 568 + 36 + 59 = 663 → 0.829 [P upper]; if the 72-subset rate extends: 568 + 36 + 88 = 692 → 0.865 [P speculative]. Overlap risk: threshold_bound and not_retrieved classes may partially be already served by the m4 lookup index (the 246 decomposition is from m3, pre-lookup?). Actually is it? exp_m3 vs exp_m4 — m3's 800-question eval at coverage 0.646 (pre-lookup), m4 shows lookup-first → 0.710. So the 246 includes cases the lookup fixes (+51). So free-win gains overlap with the +51. Honest statement: re-run the decomposition under lookup-first; pre-register joint coverage 0.78–0.83 [P]. The point: don't train the emitter until the retrieval stack is at its measured ceiling, because (a) baseline fairness, (b) harvest validity (invariant 4: any pipeline change invalidates unhashed/old records — actually records have hash+git rev, so they're re-usable only if the store and verifier are unchanged; changing τ_ret and adding backward retrieval changes the verifier config → old records retired), (c) the newly recovered chains are the informative supervision (borderline retrieval = hard positives; and backward-recovered chains teach direction).

Also misparsed_chain 92: propose the hop-retry probe (host-side, budget-limited ±1 hop search) as a measurement first: re-run the 92 with gold n_hop from the misparse table (the table already has gold n_hop pairs — someone derived them; presumably by search). If e.g. 60/92 verify with corrected hop count [P], that's both a baseline repair (search-augmented grammar) and supervision generation (correct-hop chains + error-localized DPO pairs parsed-vs-gold). But note: adding hop-search to the grammar baseline strengthens the baseline the emitter must beat — good for falsifiability. But it's also basin expansion by host search... it doesn't expand frames, just repairs planning. I'd include it as a measured probe in week 1, and decide by number.

Hmm wait, actually I need to double check the misparse numbers: {(1,2):25, (2,3):38, (3,2):14, (4,3):9, (1,3):4, (4,2):1, (2,1):1} sums to 25+38+14+9+4+1+1 = 92. Good, matches 92. Under-counts: (1,2)25+(2,3)38+(1,3)4+(4,3)9? No wait (4,3) is under-count (gold 3, parsed 4? no: (parsed, gold) — (4,3): parsed 4, gold 3 = over-count. (3,2): parsed 3, gold 2 = over-count. (4,2): over. (2,1): over. Under-counts: (1,2)25, (2,3)38, (1,3)4 = 67. Over-counts: (3,2)14, (4,3)9, (4,2)1, (2,1)1 = 25. So 67 under, 25 over. "systematically under-counts" — yes, 67/92 = 73% under. Fine.

Failing hop index {0:152, 1:63, 2:28, 3:3} sums 246. 62% at first hop (152/246 = 61.8%).

Now section D — training design.

Data: the harvest. Fields to use:
- SFT positives: verified=true records → (question, program) pairs. Program = the CubeLang chain (the accepted path from the trace tree — note design rule: store the trace tree, not the accepted path; so SFT target = accepted path extracted from tree, but the tree gives process supervision: per-hop candidates, why rejected).
- Volume: how much harvest exists? [NM] — not stated. The eval was 800 questions; harvest runs "at volume" per the founder. I must not fabricate. Say: volume unknown to me; the design needs ≥ ~20–50k verified chains for a 1–3B emitter [ASSUM]; if harvest < that, the gate can still run at smaller scale with a smaller model (e.g., 125M–350M) since compute shows 1.25B params trains at 18.6k tok/s... Actually §3.9 gives throughput for the trunk (1,254.4M params, 18,619 tok/s, 440 ms/step at D2048). So a 10k-example SFT with programs ~200 tokens ≈ 2M tokens... The learning-gate machinery evaluates a 3.28M-token delta in 65–77s. So an SFT round of, say, 3 epochs over 20k examples × 300 tokens = 18M tokens ≈ 18M/18.6k tok/s ≈ 968s ≈ 16 min per epoch... wait 18,619 tok/s → 18M tokens / 18,619 = ~967 s ≈ 16 min for one epoch of the whole set. So SFT is tens of minutes. Consistent with invariant 5. Good — I can derive GPU-hours with shown arithmetic and tag assumptions about data volume.

Objective: staged.
1. SFT on verified chains (question → program). Include the surface variants: augment with the harvest's own diversity; do NOT synthesize paraphrases with an LLM (that would be... using an LLM to generate training data is not LLM-as-judge; invariant 1 bans judging. Generating paraphrases with the trunk itself = self-play? Invariant 6 bans self-play/self-judging shortcuts. Safer: no synthetic LLM paraphrases; rely on natural harvest diversity + the disfl_qa-style natural variation. Hmm, but the harvest only contains questions the grammar parsed — the SFT set inherits the basin. This is section E's core problem.)
2. Process supervision / hop-level: the trace tree gives per-hop (query, candidates_topk, chosen, verified). Train hop-level selection? That's a retrieval/reranking objective — but §3.7 warns the trunk is a terrible retrieval encoder (macro 0.0962 vs MiniLM 0.3580; −6 sd below random-encoder floor on micro). So do NOT train the emitter to do retrieval scoring internally — the VM/retriever does that. The emitter emits program structure (frames, hops, direction, roles, absent-probes), not scores. Important design consequence: the emitter's output space is discrete program tokens; the host executes; similarity lives in the VM. This respects invariant 1 and avoids the falsified binding-auxiliary path (§5: binding loss falsified; untrained trunk already near-optimal VSA substrate — that was about VSA binding, different).
3. DPO on error-localized pairs: banned_facts[] gives (rejected fact, replacement) per hop — preference pairs at the fact level; misparse probe gives (parsed n_hop, gold n_hop) program-level pairs; candidates_topk runners-up give negatives for hop-level preference: chosen hop vs top runner-up. Use DPO or simply rejection-sampling / RFT: sample k programs from the emitter per question, execute all, keep verified ones (VM is the judge — not the model), add to SFT (STaR-like but VM-verified, so not self-judging: the VM judges). Wait — invariant 6: no self-play or self-judging shortcuts. Is RFT with VM filtering self-play? The model proposes, the VM disposes — that's the system's core loop, and the founder's own endgame says the harvest trains the trunk. STaR with VM verification is not self-judging because the judge is the VM. But rationalization (STaR backward derivation) on unsolved questions: the reserved `rationalized` field. The concern flagged in the prompt: "note invariant 6." Rationalization = derive a program backward from the gold answer, then train on it. If the derivation is done by the model itself and accepted without VM verification, that's self-judging. If the derived program is then executed forward and VM-verified against the store (bind check), it's VM-adjudicated — the model proposed, the host disposed. I think the clean line: rationalized programs enter training ONLY if forward-executed and VM-verified on the store snapshot; otherwise discarded. And the rationalizer itself: if it's the trunk generating candidate programs backward from gold, filtered by VM — that's just search with VM filter, allowed. The "shortcut" banned is accepting unverified self-derivations. State this explicitly.

But here's the bootstrap problem (E): harvest supervision exists only where the grammar already succeeded (or where host search + VM can still verify). For out-of-basin shapes, no programs exist to imitate. Escape mechanisms:
- (i) Rationalization with VM filter (above): for held-out-ish natural questions where gold answer is known (e.g., from datasets with gold answers — NQ has gold answers), search program space (beam over frames/hops/directions) until VM verifies; the verified program becomes supervision. This manufactures supervision for shapes the grammar can't parse, because search doesn't need the grammar — it needs the VM. This is the strongest mechanism: it decouples supervision from the grammar's parse basin. Cost: search per question; VM calls are ms; beam of, say, 200 candidates × 3 hops × ms = sub-second per question [ASSUM]. For 10k questions ≈ hours on CPU-ish VM. This is the key move: the grammar retires as data-generator, not just at serve time. But wait — does this violate "the harvest can only manufacture supervision for questions the grammar can already reach"? The harvest as currently defined is a by-product of serving (grammar-parsed traffic). Rationalization extends manufacture beyond. The constraint is: gold answers needed. Sources of gold: datasets (NQ, disfl_qa have gold answers), and store-derived synthetic (gold by construction). So supervision for out-of-basin shapes comes from (a) store-derived questions in held-out shapes (gold known), (b) dataset questions with gold answers filtered to store-reachable. This is exactly how to build the SFT set that isn't grammar-shaped. And it's VM-adjudicated, so invariant-clean.

Strongest reason it might fail here specifically: §3.6 — reachability. SimpleQA gold is a graph entity 19.9%, lookup-reachable 0.1%. If the store can't reach the gold, rationalization search finds no verifying program, and supervision manufacture collapses back to the reachable sliver — the basin re-forms around store reachability instead of grammar shape. I.e., you escape the grammar's basin only to hit the store's basin. On the 45k wikikg mount, NQ reachability is unmeasured [NM] — must measure before relying on rationalization. Mitigation: build the held-out and rationalization pool from store-derived questions (reachability by construction), and accept that the gate then measures shape-generalization, not world-coverage. Be explicit: the gate tests the emitter vs grammar on shape, holding store fixed. The founder's "the store cannot move SimpleQA; the loop must" — the loop can only move what the store can bind; SimpleQA at 0.1% lookup-reachable means the endgame claim needs store growth, which is out of the gate's scope. Disagree with founder where needed: the gate as framed ("emitter beats grammar out of basin") is passable by trivial surface normalization; the real gate is (a) shape generalization with reachability held fixed, and (b) refusal calibration at the edge of expressible. Also the founder's baseline claim "grammar 0/300" is a surface artifact partly — the honest baseline is grammar-after-normalization, which I predict lands ~10–20% parse on NQ [P], based on disfl_qa 15% clean parse [M] as the analogue.

Hmm wait, actually let me reconsider: is disfl_qa clean 15% the right analogue for NQ post-normalization? disfl_qa questions are SQuAD questions — well-formed, '?', capitalized. Grammar parses 15% of those. NQ normalized would be interrogative-ish but keyword-y ("who got married on the today show last week" → still not a WH-frame match necessarily). So post-normalization NQ parse ∈ [5%, 20%] [P, wide]. The cheap measurement settles it. Good.

Also: the question detector fires 220/300 but grammar parses 0/300 — the detector and parser disagree; 80/300 not even detected. Router agrees clean-vs-disfluent 992/1000 — router is robust to disfluency, parser isn't (−2.1 points). Fine.

Now the training design specifics:

Model: don't train the 1.25B trunk for the gate — or do? The founder's endgame: harvest trains the trunk to emit programs. But for a 3-week falsifiable gate, a smaller dedicated emitter (e.g., 150–400M params) trained from scratch or from an open base is cleaner: isolates the question "can a program emitter generalize in shape" from trunk politics; and the falsified aux-loss history shows trunk-training side quests go badly. But invariant 7: nothing depends on another repository at runtime; porting an open base model is a link-time thing, not runtime — fine, but "reusable pieces are ported, never linked." Using an open-weights base is a dependency at train time; runtime is the served artifact. I'd propose: train a small emitter from scratch (the lab has the training stack measured at 1.25B/45.5% MFU; a 125–350M model trains faster than that). Actually simpler: use the existing trunk architecture at small scale. Data volume: [ASSUM] harvest ≥ 50k verified records; if less, scale down. Let me do arithmetic: suppose 30k verified chains for SFT (tag [ASSUM] — actual harvest volume is [NM] in the pack; the 800-question eval suggests at least thousands). Program length mean ~? [NM]. Assume 150 tokens/question+program → 4.5M tokens/epoch; 3 epochs = 13.5M tokens; at 18,619 tok/s (measured for 1.25B; a 350M model will be faster, but use the measured number as conservative) → 13.5M/18,619 ≈ 725 s ≈ 12 min. So SFT is <1 GPU-hour. DPO round: pairs from banned_facts + misparse + runner-ups; assume 10k pairs × 400 tokens = 4M tokens, 1–2 epochs ≈ <1 GPU-hour. RFT/rationalization search: VM-bound, CPU/VM time, hours wall-clock, negligible GPU. Total gate: < 10 GPU-hours training + evals (37 s each [M]). So the whole gate is a few GPU-hours — consistent with "cost is not the constraint." Good; show arithmetic, tag assumptions.

Typos: harvested text inherits corpus orthography verbatim ("Huamn" ~95%, "united stated" ×387); typos are chain-consistent; verification depends on preserving them; normalization is consumer-side. So: the emitter must emit symbols that match store orthography — do NOT normalize training text; keep verbatim; the emitter learns to copy orthography from context (the question may contain the typo'd or correct form?). Risk: emitter canonicalizes. Mitigation: (a) train on verbatim harvest text only; (b) eval includes an orthography-consistency check: emitted symbol strings must byte-match the store's symbols — the VM does this anyway (symbol==walked object). So a canonicalizing emitter fails verification → refusals, caught by the eval. Also augmentation: since typos are chain-consistent, a "Huamn"→"Human" surface variant in the question should still bind to the typo'd store symbol — that's a consumer-side normalization at retrieval query time, not at program-symbol time. Keep normalization at the retrieval layer (query side), never in the program's symbol slots. State this.

store_snapshot_hash invalidation: group harvest records by (store_snapshot_hash, git_rev); training runs pin one snapshot; any pipeline change (τ_ret fix, backward retrieval) bumps git_rev → old records retired from SFT (kept, per invariant 4); re-harvest or re-validate: cheap re-validation = re-execute stored programs on the new pipeline; keep records whose programs still verify (the trace tree makes this a replay, ms per record). So: replay-based re-certification of the harvest after the free wins — cost: 50k × few ms = minutes [ASSUM on volume]. Nice — the trace tree enables cheap re-certification. That's a strong point: replay the accepted path under the new snapshot; verified → keep in SFT; else retire.

Negatives/counterfactuals: never merged into SFT (standing rule). Uses: (a) candidates_topk runners-up → DPO/calibration negatives; (b) banned_facts (rejected, replacement) → error-localized DPO; (c) counterfactuals (planted faults, caught/caught_by) → verifier regression suite + emitter adversarial eval (does the emitter's program, when faulted, get caught — same 1,899-fault suite replayed against emitter-emitted programs); (d) harvested negatives as calibration/test split (standing rule) — used for τ calibration and refusal-calibration eval.

What stops the emitter memorizing the grammar's basin: (1) training set includes rationalized out-of-basin-shape programs (VM-verified search products) — the share matters: pre-register ≥ 30% of SFT from non-grammar-parsed shapes [ASSUM/design choice]; (2) held-out is mechanically out-of-basin (grammar-maxed fails to parse) and frozen pre-training; (3) frame_id reserved field: emit frame_id for grammar-parsed training data; held-out excludes all 6 frame_ids... wait, out-of-basin defined by grammar-maxed parse failure already excludes frames. But leakage via paraphrase: harvest questions near-duplicate held-out questions? Held-out sources (NQ dev slice, disfl_qa validation, synthetic held-out relations) — dedup against harvest by normalized string; also hold out entire relations/domains for the synthetic part (frame-level holdout: held-out relation types never appear in training programs). Strongest anti-memorization: relation-type holdout — train on relations R_train, test on R_test disjoint; the emitter must compose programs over relation tokens unseen in training programs. If the emitter generalizes across relation holdout, that's real compositionality evidence.

Hmm, but wait: can the emitter emit programs referencing relations it's never seen in training? The relation vocabulary appears in the store/questions; the emitter copies relation tokens from the question/context. Copying from input is a mechanism (pointer-ish). A small from-scratch model may be weak at copying; fine — that's what the gate measures.

Now section F — the experiment:

Arms:
1. Grammar v2 (as-is): NQ-300 → 0/300 [M known]; on the constructed out-of-basin set → expected ~0 verified-correct [P].
2. Grammar-maxed: surface normalization + free wins (τ_ret recal, backward retrieval, lookup-first) + hop-retry search. Expected: in-domain coverage 0.78–0.83 [P]; out-of-basin verified-correct: low single digits % — only questions that normalize into a frame [P].
3. Emitter (SFT on re-certified harvest + rationalized out-of-basin programs), maybe + DPO round.
4. Emitter + rationalization iteration (round 2 RFT).

Metrics on the frozen out-of-basin set (n ≈ 300–500):
- Primary: verified-correct rate (VM-verified AND matches gold). Grammar-maxed expected ≈ 0.02–0.05 [P]; emitter must beat it by a pre-registered margin. What margin makes "frontier lab" vs "bounded-domain product"? The kill criterion should be quantitative: emitter verified-correct ≥ grammar-maxed + Δ where Δ = e.g., +15 points absolute on out-of-basin AND ≥ 2× grammar-maxed, with refusal calibration ≥ some bar. Hmm, let me think about what's principled. The gate question: "does a trained emitter do what the grammar cannot?" The grammar cannot parse out-of-basin at all; the emitter's win could be huge (surface robustness alone). But we've argued surface robustness is partly a normalization fix away — so grammar-maxed absorbs that. The emitter's real claim: semantic planning (hop counts, direction, refusal) on shapes the grammar can't parse. So the primary metric should be on the surface-normalized-but-frame-unparseable subset: questions that even grammar-maxed can't parse. There, grammar-maxed ≈ 0 by construction (it can't parse → can only refuse). Then "beats the grammar" is trivially any verified-correct > 0... no wait — grammar-maxed could still refuse honestly; verified-correct 0. So any emitter success beats it. That's the trivial-gate problem again. Fix: the gate's bar must be absolute, not relative: e.g., emitter verified-correct ≥ 40% on out-of-basin-with-reachable-facts, with false-refusal ≤ X and wrong-answer rate ≤ Y (don't-know contract: unverified never spoken → wrong answers only via VM-passing-but-wrong: organic confusable escapes 23.1% of confusables...). Hmm, what absolute number is defensible? The rationalization search itself sets a ceiling: if beam search over programs finds a verifying program for q% of out-of-basin questions, then q is the supervision-manufacture ceiling and roughly the emitter's achievable ceiling. So pre-register: measure search-ceiling q on the held-out (search is expensive-ish but this is the eval's own upper bound — actually running search on held-out at eval time is fine as a "oracle planner" arm). Arms: (1) grammar-maxed, (2) emitter greedy, (3) emitter sampled-k (k=8?) with VM arbitration — the system's natural serve mode (propose multiple, VM disposes), (4) oracle search (upper bound). Kill criterion: emitter(sampled-k) verified-correct < 0.5 × oracle-search verified-correct AND < grammar-maxed + 10 points → the emitter adds nothing over grammar+search → bounded-domain product. Hmm, but "emitter vs grammar" is the founder's stated test. I'll keep the founder's comparison but add the search-oracle as the honest ceiling and grammar-maxed as the honest baseline, and make the kill criterion two-sided: (a) must beat grammar-maxed by ≥ 10 absolute points on out-of-basin verified-correct; (b) must reach ≥ 50% of oracle-search ceiling. If (a) fails → kill (bounded domain). If (a) passes but (b) fails badly → the win is surface/cheap, not semantic — re-scope. Also in-domain non-regression: emitter ≥ grammar-maxed coverage − 2 points on the 800-eval, precision ≥ 0.98.

Also measure: mean program length, first-attempt certification rate, retry count (founder's stated metrics) — these come free from the harvest schema.

Pre-registered numeric predictions (must give numbers before running — I'll give ranges with tags):
- Surface-normalization probe on NQ-300: parse 10–20% (basis: disfl_qa clean 15% [M] as nearest measured analogue) [P].
- Free wins joint: in-domain verified coverage 0.710 → 0.78–0.83 [P]; precision ≥ 0.98 [P].
- Hop-retry probe on 92 misparsed: ≥ 50% verify with corrected hop count [P] (basis: the misparse table shows gold n_hop was determinable, implying chains existed; under-count pattern (1,2),(2,3) suggests one extra hop suffices in 67/92).
- Out-of-basin set construction yield: from NQ-300, after normalization, expect ~240–270 remain unparseable; of those, gold-in-store fraction [NM] — depends on mount; if 45k wikikg, NQ reachability unknown; predict low (<20%) [P, wide uncertainty] — hence the synthetic held-out relations are the primary out-of-basin instrument, NQ as secondary.
- Oracle-search ceiling on synthetic out-of-basin: 70–90% (by construction reachable; search over ≤3-hop programs with backward/forward) [P].
- Emitter round 1 (SFT only): out-of-basin verified-correct 25–45% on synthetic; 5–15% on NQ-reachable slice [P].
- Emitter + DPO/RFT round 2: +5–15 points; first-attempt certification rate 0.5–0.7 [P].
- Grammar-maxed on out-of-basin: ≤ 5% (only normalization-rescued frames) [P].

Cost: GPU-hours: SFT <1h [derived], DPO <1h, evals 37s each × arms × sets ≈ minutes, rationalization search = VM/CPU hours [ASSUM]. Total < 5 GPU-hours training; wall-clock 3 weeks dominated by set construction, re-baseline, harvest re-certification, and analysis, not compute. Good.

Kill criterion (founder's): emitter does not beat grammar out-of-basin → bounded-domain. My amendment: "grammar" must mean grammar-maxed (post free wins + normalization), and "beat" must mean ≥ +10 points verified-correct on the frozen out-of-basin set with precision ≥ 0.98 and false-refusal controlled; plus the search-oracle ratio. Otherwise the gate is passable by a trivial normalizer and tells us nothing about a frontier lab.

Section G — what would falsify my design: The design's load-bearing assumptions: (1) that out-of-basin shape generalization is learnable from rationalized search products — if the emitter only performs on shapes present in its SFT (i.e., rationalization products have their own template signature: all ≤3-hop, store-shaped), then "out-of-basin" is illusory: the basin re-forms around the search prior. Detection: relation-holdout slice performance ≈ 0 while in-SFT-shape out-of-basin high → the emitter is a second grammar, memorized. (2) The store-reachability assumption: if NQ-reachable yield is ~0, the natural-question arm of the gate is unfalsifiable (no positives possible) and the gate silently reduces to synthetic-shape tests, which the founder could dismiss as in-distribution-by-construction. (3) The VM's symbol-mismatch-only catching (100% symbol, 0% similarity floor; organic 76.9%) means verified-correct on out-of-basin may be inflated by confusable escapes — if wrong-answer rate on out-of-basin exceeds ~2%, the don't-know contract is being violated at the edge and the gate's "success" is actually a precision failure. Also: my design leans on replay-re-certification; if the free wins change walk semantics such that replays diverge (e.g., backward retrieval changes accepted paths), SFT targets shift underfoot.

Also section A ranking: causes by share × tractability for the generality problem (not just the 246):
1. Surface form (NQ 0/300; but detector fires 220/300) — share: large on NQ (unknown split), tractability: trivial (host-side normalization). 
2. Frame narrowness (6 WH-frames; 15% parse on well-formed natural questions) — share: the dominant semantic cause; tractability: medium — this is the emitter's job.
3. Hop planning (37% of in-domain failures) — share: large in-domain; tractability: high (search + supervision).
4. Forward-only walks (36%+8% of in-domain failures; 59/72 backward-recoverable) — share: large; tractability: very high (free win).
5. Threshold miscalibration (15%) — share: medium; tractability: very high (free win; candidates_topk negatives exist).
6. Store reachability (SimpleQA 19.9% entity / 0.1% lookup-reachable) — share: dominant for world-coverage ambitions; tractability: out of scope for the gate (store growth is a different workstream) — must be held fixed by construction in the gate.
7. Verifier confidence collapse with chain length (τ floors 1.0/0.47/0.22) — share: latent (longer chains = weaker verification); tractability: medium; the emitter will be tempted to emit longer chains where verification is weak — monitor mean program length vs correctness by n_hop. Interesting: the emitter could Goodhart by producing 3-hop chains verified at τ 0.22 — weak bindings. Add a guard: report verified-correct stratified by n_hop and flag if emitter's gains concentrate at n_hop=3 where the VM floor is 0.2202 and control mean is 0.023–0.028. Hmm wait, the τ floors by frame size: n_hop=1: 1.0000, n_hop=2: 0.4736, n_hop=3: 0.2202. So 1-hop requires similarity 1.0 (exact), 3-hop only 0.22. So multi-hop answers are verified with much sloppier binding. Organic confusable escape 23.1%. So emitter wins at n_hop=3 are the least trustworthy. Pre-register: stratify.

Also the "cheap measurement that settles surface vs semantics before anything is trained": the normalization probe — take the 300 NQ queries, apply mechanical normalization (capitalize, append '?', and optionally a token-order-preserving interrogative wrapper? No — minimal: capitalize + '?'), re-run grammar v2. Parse rate p1. Then, for a stratified 50-query subsample of the still-unparsed, two annotators label: "answerable by ≤3-hop triple walk over a general KB, yes/no" with agreement; cross-tab with gold-in-store lookup on the mounted snapshot (mechanical: is gold answer a store symbol? is the needed relation present?). This yields the 2×2: surface/semantic × in-store/out-of-store. Cost: hours, no training. Also run the grammar on disfl_qa clean as the well-formed ceiling — already measured: 15%.

Wait, one more consideration on the diagnosis: "the question detector still fires on 220/300, and the router reads 220 as 'a question about the world'" — so 80/300 aren't detected as questions at all; those are presumably navigational/keyword queries. Even a perfect emitter can't answer non-questions; the eval set should exclude or separately track the 80. Actually NQ dev contains some non-question queries; the gate's NQ slice should filter to detected questions (220) or keep all and report both. Keep all, report stratified.

Also note: mean 9.3 words, no '?' — NQ queries. Fine.

Now, disagreements with the founder (need to be explicit and data-grounded):
1. The 0/300 baseline is not a meaningful baseline for the gate. It conflates surface form with semantics, and the pack itself admits this is unmeasured. Using 0/300 as "the grammar's out-of-basin score" makes the gate trivially passable. Fix: grammar-maxed baseline + oracle-search ceiling + absolute bars.
2. "Beat the grammar out of basin → frontier lab" is too weak a proof: a rule-based normalizer + the grammar might beat 0/300. The gate must test semantic generalization (relation holdout, hop planning, refusal calibration), not just parse robustness.
3. The free wins must be banked before training — not because they're nice, but because invariant 4 + snapshot discipline makes pre-fix harvest records retire on the fix; training on soon-to-be-retired records wastes the round and biases the emitter toward easy-retrieval paths. Also the baseline must include them or the emitter gets credit for retrieval fixes.
4. The reachability ceiling (§3.6) means the gate cannot be run on SimpleQA-style world questions; the founder's "the loop must move SimpleQA" is not testable at this store scale — the gate should hold store fixed and test shape, and say plainly that world-coverage is a store workstream. Actually the founder said the store can't move SimpleQA and the loop must — but the loop can only move what's reachable (0.1% lookup-reachable). So even a perfect emitter moves SimpleQA by ≤0.1% + whatever embedding-retrieval (search mode, cosine top-k above τ_ret) can reach beyond exact lookup. Hmm — lookup-reachable 0.1% is exact-triple lookup; the cosine search mode might reach more. Gold is a graph entity 19.9% — so 19.9% have the answer entity in the graph; reaching it via walks from question anchors is the question. So emitter ceiling on SimpleQA ≤ 19.9% [M-derived bound]. The founder's loop could in principle move up to ~19.9% if walks can reach gold entities. That's a bound worth stating: even perfect planning caps at 19.9% entity-presence. So "the loop must move SimpleQA" has a measured hard ceiling of 19.9% (entity presence), and 0.1% via exact lookup. The gate on SimpleQA is thus mostly a store/retrieval problem, not an emitter problem. Disagree with running the gate there.
5. Trunk-as-retrieval-encoder numbers (§3.7): the emitter must not be asked to do retrieval scoring; keep scoring in the VM/retriever. The falsified binding loss supports this.

Also should I address "which question shapes are out of basin" (founder's Know bullet): out-of-basin shapes, concretely: (a) keyword queries without interrogative form; (b) WH-frames outside the 6 (how many/how much — count/aggregation → likely inexpressible; when/where temporal; why — causal, inexpressible as walk); (c) multi-constraint questions (intersection of two walks — is intersection expressible? The program algebra: per-hop retrieve/walk, BIND/UNBIND, absent-role. Intersection = two chains binding same symbol? Possibly expressible as two programs + host-side equality check? Unclear — mark as edge-of-expressible, needs a spec decision); (d) negation ("who did NOT...") — absent-role covers role-absence on an entity, not set-complement negation → inexpressible; (e) comparatives/superlatives ("largest", "oldest") — require ordering/aggregation → inexpressible unless store has comparative relations; (f) count questions → aggregation → inexpressible. So the edge-of-expressible spec: the emitter emits a typed refusal {inexpressible_class, best_partial_trace}; the eval has an inexpressible subset with gold labels (by construction: count/superlative/negation/why questions generated from the same store) and scores refusal calibration: true-refusal rate on inexpressible (target ≥ 0.9 [P]), false-refusal on expressible (target ≤ 0.1 [P]), and wrong-spoken rate (must be ~0; don't-know contract: any spoken answer must be VM-verified; wrong-spoken = verified-but-wrong vs gold — expect ≤ 2% given organic confusable escape 23.1% of the ~small confusable rate... hmm, organic confusables: 70/91 caught; the 21 escapes — did they produce wrong answers? "Organic confusables: 70 of 91 caught = 0.769" — 21 escaped. What happened to escapes? [NM]. Precision 0.992 measured in-domain suggests escapes rarely become spoken wrong answers (517 claimed, 513 correct → 4 wrong of 517 = 0.77% wrong). So measured wrong-spoken ≈ 0.8% in-domain [M-derived]. Pre-register out-of-basin wrong-spoken ≤ 2%.

Document → triples (founder's Know bullet): "how an arbitrary document becomes facts the verifier can bind against." Current: store 1,188/1,241 facts parse as triples (95.7%) [M] — but that's the WikiKG-derived store, already triple-shaped. Arbitrary documents: need an extraction path that yields (obj, rel, subj) with verbatim orthography, plus provenance (source doc id, span) so the VM's symbol binding is auditable. But invariant 1: no LLM-as-judge — can an LLM extract? Extraction is generation, not judging; the VM verifies bindings against the extracted store, but who checks the extraction is faithful to the document? If the trunk extracts triples and the VM verifies only internal consistency (symbol match within store), then document→store fidelity has no truth gate — the VM verifies binding fidelity, not truth (on-record caveat). So the spec must say: document ingestion is out of the truth gate's protection; the honest design is (a) extraction emits (triple, source span) pairs; (b) a host-side mechanical check binds triple to span (string/anchor overlap — mechanical, not LLM); (c) provenance carried in the record; (d) ingestion marked low-trust vs the curated store, and the emitter/verifier treat ingested facts with a provenance tier. Hmm, but is mechanical span-binding enough? A conservative spec: documents become "candidate facts" with span provenance; the VM can bind against them (symbol match), but answers resting on ingested facts carry the provenance tier in the trace; the don't-know contract is not violated (the claim is verified against the store-as-bound), but the store's truth is inherited from the corpus — same as now (the corpus is the truth source; typos and all). Actually the current system already inherits corpus truth verbatim. So the spec: same contract, one tier deeper: fact store = curated snapshot (hashed); document ingestion produces a new snapshot (hashed) with per-fact provenance; nothing at runtime links to another repo (facts are ported into the store, invariant 7). For the gate itself, document ingestion is NOT required — the gate holds the store fixed. The Know deliverable specifies it, the gate doesn't test it. Say that plainly: the gate tests the emitter vs grammar; ingestion is specified but explicitly out of the gate's measurements, because reachability must be controlled for the gate to be interpretable. Actually the founder asked in Know: "how an arbitrary document becomes facts the verifier can bind against" — so I must spec it (section B), but I can scope its empirical test out of the gate.

Edge of expressible (founder's third Know bullet): enumerate the program algebra's expressible set: {1–k-hop walks over (obj,rel,subj), forward/backward direction per hop, role-binding, absent-role probe, conjunction by chaining (entity shared across hops), identity}. Inexpressible: aggregation (count/sum/max), comparison across entities, set-complement negation, temporal arithmetic, causation, hypotheticals. At the edge: partial expressibility — e.g., "who is the oldest X" → walk to all X's with birth-year relation exists per entity, but max() is host-side? If host-side max is allowed, it's a VM operator question. Decide: the emitter may emit a bounded operator set (walk, bind, absent, and a small set of host operators?) — no: keep the gate's algebra fixed (walk/bind/absent) and define edge = refusal with typed reason + best partial trace. Adding operators mid-gate moves goalposts. Spec: edge behavior = typed refusal ∈ {needs_aggregation, needs_comparison, needs_negation, needs_temporal, needs_causation, entity_not_in_store, shape_unsupported} + partial trace of the longest verified prefix. The emitter is scored on edge calibration as a first-class metric. Good — that makes the "don't-know contract" measurable, which the grammar never had to be good at (it just fails to parse).

Now the harvest → SFT mapping concretely:
- Input: question (verbatim, no normalization — the emitter must handle raw surface), plus maybe the router's world hint? Keep input = question text only, to force surface robustness. Hmm, or include detected-question flag? Keep minimal: question text.
- Target: CubeLang program of the accepted path: frame-independent canonical form: HOP i: (direction, role, anchor-binding) ... plus ABSENT probes. Actually the emitter should emit a canonical program IR, not the grammar's frame tokens — otherwise it memorizes frames. Canonical ops: BIND(symbol from question span), WALK(role, dir), CHECK_ABSENT(role), ANSWER(symbol). The emitter learns to copy entity strings verbatim (orthography preservation) and to select relation tokens — relation vocabulary comes from the store's relation set, surfaced in the question text or in context? Hmm — how does the emitter know the relation token? In the harvest, hops have `triple{obj,rel,subj}` — the rel string. The question ("who is the editor of X") contains relation words. The emitter must map question words → store relation tokens (verbatim orthography). That's the core semantic skill. For held-out relations, the emitter must copy the relation token from the question (if present) — synthetic held-out questions will be generated to mention the relation (like WikiKG questions do: "what is the X of Y"). Hmm, if held-out questions mention relation tokens verbatim, copying suffices — is that "generality"? It's compositional copying — still nontrivial (which role slot, direction, hop structure). To make it harder, some held-out questions paraphrase relations ("who runs" → relation "chief executive officer"?) — no, that requires retrieval-side synonym handling, out of scope. Keep: relation tokens appear in question; test structure generalization (hop counts, direction, multi-constraint, surface form). Be honest about what's tested.

- Volume: [NM] — I'll write the design parametrically: N_verified = harvest count of verified records under the post-fix snapshot; require N ≥ 20k for the 3-week gate at ~350M params [ASSUM]; if N < 5k, drop to a 125M emitter and synthetic-heavy SFT. Show token arithmetic with a concrete assumed N and label it.

- Stages:
  1. Re-baseline + free wins (week 1): implement τ_ret per-frame calibration (using candidates_topk negatives — free), backward retrieval fallback, lookup-first (already measured +51), hop-retry probe (measure, then decide inclusion in grammar-maxed). Re-run 800-eval → pre-registered 0.78–0.83 coverage. Replay-re-certify harvest under new git_rev.
  2. Set construction (week 1–2): out-of-basin held-out: (a) synthetic: held-out relation types × {1,2,3}-hop × surface variants {interrogative, keyword-query, disfluent} × expressible/inexpressible labels, all store-anchored, frozen + hashed; (b) natural: NQ-300 normalized + gold-in-store filter, disfl_qa slice; (c) edge set: inexpressible-typed questions. Dedup vs harvest.
  3. Supervision manufacture (week 2): SFT set = re-certified verified chains (grammar-shaped) + rationalized programs on non-grammar-parsed questions (search + VM filter). Ratio target ≥ 2:1 grammar-shaped : rationalized? Or the reverse? Hmm — rationalized products are the out-of-basin shapes; we want enough: ≥ 30% of SFT [design choice]. Search cost: per question beam ~100–500 programs × VM ms → seconds/question; 10k questions → hours [ASSUM]. 
  4. Train (week 2–3): SFT (1 GPU-h), eval; DPO round on banned_facts + misparse pairs + runner-ups (1 GPU-h), eval; RFT round 2: sample k=8 on unsolved held-out-like pool (NOT the held-out set itself — use a separate dev pool), VM-filter, retrain. Careful: RFT on the dev pool, never touching test. 
  5. Gate run (week 3): all arms on frozen sets, stratified report, kill decision.

- Pre-registered predictions per arm (numbers before running) — I listed above; make them crisp with ranges and bases.

- Kill criterion: On the frozen out-of-basin primary set (synthetic relation-holdout, expressible subset, n≈300): emitter sampled-k verified-correct ≥ grammar-maxed + 10 points absolute AND ≥ 50% of oracle-search ceiling AND wrong-spoken ≤ 2% AND in-domain non-regression (≥ grammar-maxed coverage − 2 pts, precision ≥ 0.98). If emitter < grammar-maxed + 10 → kill: bounded-domain product; re-scope; do not train general trunk. Also secondary kill: if emitter beats grammar only on the surface-variant stratum (keyword/no-'?') but ≈ 0 on relation-holdout → the win is a normalizer, not an emitter → same conclusion, re-scope (this is the "second grammar" failure).

Also the founder's metrics: mean program length, first-attempt certification rate, retry count — include as secondary with predictions: emitter mean program length ≈ gold n_hop + 1 ops... eh, give qualitative-ish but numeric: first-attempt certification 0.4–0.6 round 1, 0.55–0.75 round 2 [P]; retries mean ≤ 1.5 at k=8 sampling [P].

GPU-hours total: SFT ~0.5–1 h [derived from 18,619 tok/s], DPO ~0.5 h, RFT ~1 h, evals: 37 s/full eval [M] × ~20 evals ≈ 12 min, search: VM-side (not GPU) — hours wall-clock. Total GPU < 4 h [P]. Wall-clock: 3 weeks, dominated by engineering and set construction, not compute — matches "cost is not the constraint; being wrong slowly is."

Also handle: "how store_snapshot_hash invalidation is handled" — pin (hash, git_rev) tuples; free-win changes bump git_rev → replay-re-certify (execute stored accepted paths under new config; keep records that re-verify; retire the rest, don't delete). SFT data manifest = list of record ids + snapshot hash; the trained model carries the manifest hash. Any eval against a different snapshot requires re-certification of the eval set too. This is the "retire, never delete" implementation.

Typo handling: verbatim everywhere in training targets; normalization only at retrieval-query formation (consumer-side, host); eval includes an orthography adversarial slice: questions with corrected spelling ("Human") against typo'd store ("Huamn") — retrieval must normalize query-side, program symbols must stay verbatim; measure both directions. Prediction: if emitter canonicalizes, symbol mismatch → refusal, visible as false-refusal spike on that slice [P].

Negatives/counterfactuals without merging into SFT: candidates_topk runners-up → DPO preference (accepted hop vs runner-up) and τ_ret calibration; banned_facts → hop-level DPO (replacement ≻ rejected); counterfactuals (planted faults) → verifier regression suite (re-run 1,899-fault suite against emitter programs; expect 100% caught by symbol mismatch [P — same mechanism]) + emitter robustness eval (fault-injected emitter programs must be caught — they will be, since catching is VM-side); harvested-negative calibration/test split → refusal-calibration eval and τ floors re-check. None of these enter SFT targets.

What stops basin memorization — recap: (1) rationalized out-of-basin SFT share ≥ 30%; (2) relation-holdout eval; (3) surface-variant augmentation from natural data only (no LLM paraphrase — invariant 6 caution; actually is LLM paraphrasing self-play? Generating paraphrases with the trunk and training on them without VM verification would be a self-shortcut; with VM verification (paraphrase → program still verifies) it's filtered... but the paraphrase itself is unjudged text quality — simpler to ban synthetic paraphrase in the gate; natural variation suffices: disfl_qa gives disfluent twins [M]); (4) frozen hashed sets + mechanical membership rules.

Bootstrap (E) — strongest mechanism: VM-filtered backward rationalization (search from gold answer to question anchors), i.e., manufacture supervision where the grammar never reached, with the VM as the only acceptor. Why it might fail here: (a) reachability — manufacture collapses to store-reachable sliver (SimpleQA 0.1% lookup / 19.9% entity [M]); NQ reachability unmeasured [NM] — if ~0, the natural-question rationalization pool is empty and the SFT set re-forms around synthetic shapes; (b) search prior signature: all rationalized programs come from the same beam prior → the emitter learns the search's template, a second grammar (detect via relation-holdout ≈ 0); (c) gold-answer leakage: rationalization uses gold answers from datasets — the emitter could learn "emit a program that binds the gold-looking string" — no, the program is derived from store structure, not from gold text alone; but the ANSWER symbol in training targets equals gold — the emitter sees question→program where program's answer = gold; that's fine, it's supervised learning; the risk is the emitter learning to echo plausible answer strings from question context rather than composing walks — detectable on held-out where answers aren't in the question. (d) Invariant 6: rationalization must be VM-adjudicated forward execution, never self-accepted backward derivations — state the line: a rationalized program counts only if, generated as a candidate and executed forward on the snapshot, the VM verifies every hop; the model never scores its own derivation.

Also note the founder's line: "the harvest can only manufacture supervision for questions the grammar can already reach" — I should push back: that's true of the harvest-as-byproduct, but false of the loop: the VM + search can manufacture supervision wherever the store can reach, independent of the grammar. The grammar's monopoly is on parsing, not on verification. The bootstrap escape is to shift supervision manufacture from parse-gated to verify-gated. That's the central design move. And its failure mode is reachability (measured: severe).

One more check — invariant 5: serve-time budget ms; training rounds tens of minutes on one 80GB GPU — my design fits (SFT < 1 h... "tens of minutes" — 13.5M tokens ≈ 12 min; fine).

Also "Nothing depends on another repository at runtime" — synthetic set generation scripts are host-side; fine.

Let me also double-check arithmetic I'll present:
- 246 decomposition shares: 92/246 = 37.4%; 88/246 = 35.8%; 36/246 = 14.6%; 19/246 = 7.7%; 11/246 = 4.5%. Matches stated 37/36/15/8/4. Good.
- First-hop: 152/246 = 61.8% ≈ 62%. Good.
- Misparse under-count: 67/92 = 72.8%.
- Free win 1 upper: coverage 0.646 + 36/800 = 0.646 + 0.045 = 0.691. Post-lookup base: 0.710 + 36/800 = 0.755 (if disjoint from lookup gains).
- Free win 2: 59 recovered (log) → 0.710 + 59/800 = 0.784. Both: 0.710 + 95/800 = 0.829. If 59/72 rate (81.9%) extends to all 107: 107 × 0.819 ≈ 88 → 0.710 + (36+88)/800 = 0.710 + 0.155 = 0.865. Present 0.78–0.83 as joint pre-registered range with overlap discount; note the 72-vs-107 discrepancy must be resolved by re-probe.
- τ_ret: measured threshold_bound scores min 0.5028 < p50 0.5547 < max 0.5944 < τ_ret 0.5959. A τ_ret of 0.50 admits all 36. Per-frame calibration preferred; candidates_topk supplies negatives for free.
- Precision risk: lowering τ_ret admits more organic confusables; caught rate 0.769 [M] → escapes 23.1% of confusables; in-domain wrong-spoken measured ≈ 4/517 = 0.77% [M-derived]. Pre-register precision ≥ 0.98 post-fix; if violated, per-frame τ_ret.
- Emitter training arithmetic: assume N=30k verified chains [ASSUM], mean example (question+program) ≈ 180 tokens [ASSUM] → 5.4M tokens/epoch; 3 epochs = 16.2M; at 18,619 tok/s [M, measured for the 1.25B trunk — a smaller emitter is faster, so conservative] → 16.2M/18,619 ≈ 870 s ≈ 15 min. DPO: 10k pairs [ASSUM] × 2×(prompt+completion) ≈ 10k × 360 = 3.6M tokens × 2 epochs = 7.2M → ~6.5 min. RFT sampling: k=8 × 10k = 80k generations × ~120 tokens out = 9.6M tokens generated; generation slower than training, assume 5× cost → ~43 min [ASSUM]. Total ≈ 1–1.5 GPU-h. Evals: 37 s each [M]. Fine.

Also the "grammar-maxed" hop-retry: 92 misparsed; if corrected-hop replay verifies ≥ 50% [P], grammar-maxed coverage +~46/800 ≈ +5.7 pts → up to ~0.89 with everything — hmm, that makes in-domain coverage quite high; then the emitter's in-domain job is mostly done by grammar+search, and the gate is purely out-of-basin. That's fine and clarifying: the in-domain story is retrieval+planning repairs; the generality story is the emitter. Actually wait — should hop-retry be in grammar-maxed? It's host search, ms per retry, budget-limited. The system already has repair budgets (budget 1 vs 3 measured). Hop-retry is a new repair class. Include in grammar-maxed (it strengthens the baseline honestly) but ALSO the recovered chains become emitter supervision with correct n_hop. Both. But careful about serve-time budget: hop-retry multiplies walk cost by (1 + retries); at 45.9 ms/question budget-1 [M], budget-3 is 120.9 ms [M] — hop-retry ±1 ≈ up to 3× walks ≈ within budget-3 envelope. Fine for harvest-time; for serve, keep budget 1.

Now — the surface-vs-semantics cheap measurement, make it very concrete: 
Probe S1 (mechanical): NQ-300 → normalize {capitalize first char, append '?'} → grammar v2 parse. Output: p_norm. Prediction 0.05–0.20 [P]. 
Probe S2 (mechanical): for all 300, gold-answer-in-mounted-store lookup (is gold string a symbol in the snapshot? mechanical). Output: reachability r. [NM currently]. 
Probe S3 (annotation, 50 stratified): two annotators label "answerable by ≤3-hop walk over a general KB" + "requires inexpressible op (count/compare/negate/temporal/causal)"; report agreement. 
Then: surface share ≈ p_norm; semantic-frame share ≈ (well-formed but unparseable); expressiveness share from S3; store share from S2. Cost: S1/S2 minutes of compute; S3 one afternoon. No training. This settles the diagnosis before the emitter exists. Also note disfl_qa already gives the well-formed ceiling: 15.0% [M] — so even if surface were fully fixed, expect ≤ ~15% parse on natural questions; i.e., I predict the majority of the 0/300 is frame narrowness + store reachability, not the missing '?' — the '?' is a red herring the founder's baseline over-weights. Actually hold on: 0/300 with 220 detected — the grammar needs interrogative form? The 6 WH-frames presumably require WH-word patterns; NQ queries often start with who/what/when ("who got married...") — mean 9.3 words, lowercase. Do the frames require capitalized '?'? The 0/300 with detector firing 220/300 suggests the parser rejects on form. disfl_qa clean parses 15% — those are full interrogatives. So surface normalization should rescue some NQ; how many NQ queries are WH-initial? [NM] — typical NQ has many "who/what/when/where" queries but also "the capital of france" style. Predict p_norm 5–20%. The rest: frames too narrow (6 WH-frames vs the long tail of natural question shapes) + store reachability. So my stated belief: of the 300, roughly: surface-blocked maybe 10–20% (i.e., would parse after trivial normalization), frame-narrowness 40–60%, store/expressiveness the rest — but honestly tag as belief with the probe to settle it. The prompt demands: "Say plainly which of the 0/300 you believe is surface form and which is semantics." So: I believe the minority is surface (≤ 1/5), majority semantics (frames + reachability + expressiveness), basis: disfl_qa 15% clean parse [M] shows the grammar fails on 85% of well-formed natural questions — surface can't explain that. Good, that's a data-grounded plain statement.

Ranking by share × tractability (diagnosis): 
1. Forward-only + τ_ret miscalibration: 44% of in-domain failures (124/246), tractability near-total → fix first. (36+88+19 = 143, hmm: not_retrieved 88 + far_below 19 + threshold_bound 36 = 143/246 = 58%.) Let me rank properly:
   - Retrieval mechanics (not_retrieved + far_below + threshold_bound = 143, 58% of in-domain failures): tractability very high (two measured free wins). Rank 1.
   - Hop planning (92, 37%): tractability high via search+supervision; overlaps with emitter's purpose. Rank 2.
   - Frame narrowness (85% of well-formed natural questions unparseable [M-derived from 15% parse]): share dominant for generality; tractability medium (emitter + rationalization). Rank 3 for the gate (it IS the gate), but rank by tractability it's the hardest.
   - Surface form: share small-to-moderate (unmeasured; probe S1); tractability trivial. Rank 4 (do it, but it's not the story).
   - Store reachability: share dominant for world-coverage (SimpleQA 19.9% entity/0.1% lookup [M]); tractability: out of gate scope. Rank 5 with a flag: it caps everything; the gate must control for it.
   - Verifier confidence collapse by chain length (τ floors 1.0/0.47/0.22 [M]): latent share; tractability medium (report stratified; maybe cap n_hop at 2 for gate conclusions? No — measure, don't cap). Rank 6 but a mandatory stratification.

Now section B spec — "what must a trained emitter do that the grammar cannot": 
1. Map arbitrary surface forms (keyword queries, no '?', disfluent) to programs — grammar: 0/300 NQ, 12.9% disfluent [M].
2. Plan hop structure (n_hop, direction per hop) from semantics — grammar systematically under-counts (67/92 under-count [M-derived]).
3. Select retrieval direction per hop — grammar is forward-only; 59/72 recoverable backward [M].
4. Calibrate refusal at the edge of expressible with typed reasons + partial trace — grammar has no refusal semantics for inexpressible shapes (it just fails to parse; the don't-know contract at the emitter needs an explicit REFUSE program).
5. Preserve verbatim orthography in symbol slots while normalizing only query-side.
6. Emit ABSENT_ROLE probes when question semantics imply absence-checks.
Operational out-of-basin definition (mechanical, freeze-able):
A question Q is out-of-basin iff: (i) grammar-maxed (v2 + normalization + free wins + hop-retry) fails to parse or fails to produce a verified answer for Q; (ii) Q's gold chain is present in snapshot H (for synthetic: by construction; for natural: gold symbol ∈ store and required relations ∈ store — mechanical check); (iii) Q is labeled expressible (≤3-hop walk, no aggregation/comparison/negation/temporal) — mechanical for synthetic, annotated for natural; (iv) Q not near-duplicate of any harvest record (normalized-string dedup + (subject,relation)-pair dedup). Set frozen with hash + git_rev before training. Goalpost-proof because membership is checkable by script.

Edge of expressible: REFUSE(reason_class) ∈ {aggregation, comparison, complement_negation, temporal_arithmetic, causation, not_in_store, unsupported_shape} + longest verified prefix trace. Contract: never speak unverified; a refusal is spoken with partial trace.

Document→facts: pipeline spec (not gated): chunk → extract candidate (obj, rel, subj, span, doc_id) with a generator (proposal, not judgment — invariant 1 intact: nothing scores thoughts; extraction is host-driven generation whose output is mechanically bound to source spans) → mechanical span-binding check (all triple arguments must anchor to their spans by exact/normalized containment) → ingested into a NEW hashed snapshot with provenance tier `ingested` vs `curated`; VM binds identically; answers resting on ingested facts expose tier in trace; rollback = retire snapshot. Note the on-record caveat: VM measures binding fidelity, not truth — ingested facts inherit corpus truth exactly as the typo'd corpus does today; the tier makes that visible. For the gate: store fixed; ingestion spec delivered as part of "Know" but explicitly untested by the gate.

Hmm, one subtle thing: "extraction with a generator" — is using the trunk to extract triples from documents a violation of "no LLM-as-judge"? The invariant: "No LLM-as-judge anywhere in the loop. The model never scores its own thoughts." Extraction is generation of candidate facts, not scoring. The mechanical span check is the gate on extraction fidelity. I think it's compliant, but I should flag the residual risk: extraction hallucination → a fact not in the document → span check fails → rejected. Hallucinated-but-span-anchored misreads (wrong relation between two present entities) pass mechanical checks — acknowledged residual, tier-labeled. Honest.

Section D specifics — objective choice: SFT first (program synthesis), then DPO on error-localized pairs (banned_facts, misparse, runner-ups). Why not RL? Process supervision via VM is available (per-hop verified flags) — RL with VM reward = the endgame, but for a 3-week gate, SFT+DPO+RFT is enough and cheaper; RL (e.g., GRPO-style with VM reward) is the round-2 upgrade if the gate passes. Actually the founder's loop is exactly RL-with-VM-reward at scale; the gate should establish the SFT/RFT floor first. Also note: RFT with VM filter IS the supervised version of the RL loop; if RFT round-2 gains ≈ 0, that's evidence the RL loop won't have gradient either — a cheap early read on the endgame. Nice point.

Also process supervision: train hop-by-hop with per-hop verified targets from trace trees (teacher-forced accepted path with per-hop candidates as contrastive context?) — keep simple: SFT on full programs; DPO on hop-level pairs. The trace tree's value: (a) replay re-certification; (b) DPO pairs; (c) τ calibration; (d) candidates as free negatives.

Also: how the emitter handles the router/world_ids — reserved fields; out of gate scope; emitter input = question text only. Fine.

Model size: 350M [ASSUM] — why: 30k–100k examples, small vocab output space (program IR + copied spans), measured training throughput at 1.25B is 18.6k tok/s; 350M trains ~3–4× faster [P]; total gate compute trivial either way. Alternative: fine-tune the existing trunk? The trunk is the endgame target, but trunk fine-tuning history is littered with falsified auxiliaries; for the gate, a clean small emitter isolates the question. If the gate passes, port the recipe to the trunk (invariant 7: port, don't link). State this.

Data volume honesty: harvest volume is [NM] in the pack — the 800-question eval and the harvest description imply the machinery exists, but I cannot know the serving volume. Design parametrically: gate runs if N_verified ≥ 10k; if 3k ≤ N < 10k, shrink model to 125M and lean on rationalized synthetic; if N < 3k, the gate tests rationalization-only training (still meaningful: it tests shape generalization from search products). Good — a design that degrades gracefully instead of fabricating volume.

Also dedup/leakage: harvest questions come from serving traffic — presumably WikiKG-derived template traffic (the grammar's basin). The NQ/disfl_qa probes were "standin" logs, not necessarily harvested. Held-out NQ/disfl_qa slices must be excluded from any harvest (check by string match). Synthetic held-out relations: exclude their relation tokens from all training programs (mechanical filter). Freeze.

Eval stratification requirements (pre-registered): by n_hop (1/2/3) — because τ_vm floors collapse (1.0/0.47/0.22 [M]); by surface variant; by expressible/inexpressible; by relation-holdout vs seen-relation; by direction-required (backward-needed) subset. Report verified-correct, wrong-spoken, refusal, false-refusal per stratum. The 59/72 backward subset gives a natural "direction-required" eval slice for in-domain; for out-of-basin, construct backward-required synthetic questions.

Metric definitions:
- verified-correct: VM-verified chain AND final symbol == gold (gold comparison is host-side, mechanical string/normalized match — normalization consumer-side: compare after canonical casing/typo-fold? The typo issue: gold answers in datasets are clean ("Human") vs store "Huamn" — the comparison layer needs the consumer-side normalizer; the don't-know contract is about spoken answers; the eval's correctness check uses normalized equality with the same normalizer for all arms — fair. Flag it.)
- wrong-spoken: spoken (verified) but ≠ gold → the dangerous metric; target ≤ 2%.
- refusal rate on inexpressible (target ≥ 0.9) and on expressible-unreachable (should also refuse — target ≥ 0.95; the store-absent case: ABSENT_ROLE control works 528/528 [M], so reachable-machinery for honest absence exists).
- false-refusal on expressible-reachable: target ≤ 0.15 round 1.

Baseline arms and predictions table (pre-registered):
- Grammar v2 as-is: out-of-basin verified-correct ≈ 0 (it can't parse) [P]; in-domain 0.646/0.710 [M].
- Grammar-maxed: in-domain 0.78–0.83 [P]; out-of-basin (surface-normalized rescuable stratum only) ≤ 5% [P]; NQ-300 normalized: parse 5–20% [P], verified-correct ≤ 3% (reachability-limited) [P].
- Oracle search (beam over program space, VM filter): synthetic out-of-basin expressible: 70–90% [P]; NQ-reachable slice: 60–80% of the reachable subset [P, wide].
- Emitter SFT (round 1): synthetic out-of-basin: 25–45% [P]; relation-holdout stratum: 10–30% [P]; NQ-reachable: 20–50% of reachable [P, wide]; in-domain: 0.70–0.80 (may lag grammar-maxed — acceptable; non-regression bar −2 pts vs grammar-maxed... hmm if grammar-maxed is 0.83 and emitter 0.75, that's a regression — the emitter serves out-of-basin, grammar serves in-basin; the product would route. The gate's non-regression bar: emitter ≥ 0.75 in-domain? Set: emitter in-domain verified-correct ≥ grammar v2's 0.710 − 2 pts = 0.69, not vs grammar-maxed — because grammar-maxed's hop-retry search is a serve-time cost the emitter avoids. Hmm, actually the fair comparison: emitter at serve budget (1 program, k samples) vs grammar at budget 1 (45.9 ms). Emitter inference is one forward pass + VM verify — comparable ms. Grammar-maxed with hop-retry is budget-3-ish (120.9 ms). So compare emitter vs grammar-maxed at matched latency? Over-engineering; state: report both, gate on out-of-basin primarily, in-domain non-regression vs grammar v2 (0.710 lookup-first) − 2 pts.)
- Emitter + DPO + RFT (round 2): +5–15 pts over round 1 on out-of-basin [P]; first-attempt certification 0.55–0.75 [P]; mean retries ≤ 1.5 [P].

Kill criteria (final): 
K1: emitter round-2 out-of-basin verified-correct < grammar-maxed + 10 pts → KILL (bounded-domain product; re-scope; no general trunk).
K2: emitter ≥ grammar-maxed + 10 but < 40% of oracle ceiling → the emitter is a surface normalizer / second grammar; kill the general-trunk plan unless relation-holdout stratum specifically ≥ 30% — hmm, keep it simpler: K2: relation-holdout stratum < 15% → win is surface-only → re-scope (kill general trunk).
K3: wrong-spoken > 2% on any stratum → don't-know contract broken at the edge → the "success" is void regardless of coverage (invariant 3) — fix refusal calibration before claiming the gate passed.
Also in-domain: emitter < 0.69 → the emitter can't even hold the basin → kill (a general emitter that loses in-domain is worse than the grammar).

Cost: GPU-hours: SFT ~0.25 h + DPO ~0.2 h + RFT ~0.75 h + evals ~0.1 h ≈ 1.5 GPU-h [P, derived]. Wall-clock: 3 weeks: wk1 re-baseline+probes+set construction; wk2 supervision manufacture + round-1 train/eval; wk3 round-2 + gate run + report. Search/rationalization: VM-side hours, not GPU-bound [ASSUM].

Section G — self-falsification paragraph: The design's weakest joint is the assumption that "out-of-basin" as I construct it is not secretly in-basin for the rationalization search: synthetic questions generated from the same store, same relation grammar, same ≤3-hop template, searched by the same beam that will later train the emitter — the emitter may just be distilling the search prior, i.e., a learned grammar, and the gate would then measure search→distillation quality, not generality. My own pre-registered numbers embed this: if round-2 gains come entirely from seen-relation strata while relation-holdout stays < 15%, my design's headline "emitter beats grammar out of basin" is true and meaningless. Second joint: reachability — if the NQ gold-in-store yield is near zero (as SimpleQA's 0.1% suggests it may be [M-analog]), the natural-question arm has no positives and I will be tempted to read the synthetic arm as the verdict on world-questions, which it is not. Third: replay re-certification assumes accepted paths are stable under the free wins; if backward retrieval changes which path verifies, my SFT targets and my baseline move together and the delta I measure is partly an artifact of re-certification, not learning. The honest guard is that all three failure modes have pre-registered numbers attached (relation-holdout ≥ 15%, NQ yield measured before training, re-certification delta reported) — if I find myself explaining around those numbers after the run, the design has already failed.

Also, must "disagree with the founder where the data says to" — I have several: (1) 0/300 is not a baseline, it's a confound (the founder half-knows this — "probably the first thing to find out" — but still calls it "the baseline" in the gate; I make it non-load-bearing); (2) "beat the grammar out of basin" as the kill test is too weak — trivially passable; needs grammar-maxed + oracle ceiling + absolute bars; (3) "the harvest can only manufacture supervision the grammar can reach" is false as stated — the VM can manufacture supervision wherever the store reaches; the grammar gates parsing, not verification; the real constraint is reachability (0.1%/19.9% [M]); (4) the founder's "cheapest experiment in the plan" undersells the prerequisite work: the free wins + re-certification are mandatory before training or the gate confounds retrieval repair with emitter learning; (5) SimpleQA/world-coverage ambitions are store-bound; the gate must hold the store fixed and say so.

Also the founder's metrics (mean program length, first-attempt certification, retry count) — include, with predictions.

One more check on invariant 6 and STaR: "No self-play or self-judging shortcuts." My rationalization: candidate programs generated (by search over program space — is the generator the model or the host? If the host enumerates programs mechanically (frames × hops × directions × relation tokens from the store), that's pure host search — no model involved — clean. If the model proposes and VM filters, also clean (model proposes, host disposes). The banned thing: model generates, model accepts. I'll use host-enumerated search + optional model-proposed candidates, both VM-adjudicated by forward execution. State the line explicitly. Also `rationalized` field: emit only VM-verified rationalizations.

Also — "hops_verified_before_failure" field exists; process supervision could train a value function? No — that would be the model judging its own thoughts (a critic scoring hops). Invariant 1 bans model-judging; hop verification is VM-side. Don't train a critic. Good catch — explicitly reject process-reward-model designs on invariant grounds. Process supervision instead = teacher-forcing the VM-verified accepted path with per-hop structure (SFT), and DPO pairs from VM outcomes — the VM is the process judge.

Also the "ABSENT_ROLE control 528/528" — absence detection works; the emitter should learn when to emit absent-probes from harvest (answer_class field probably includes absent/refusal classes).

Now — length and format. This is a dense technical response; the user is a research-lab judge/founder. Format: headers per section A–G, tables where useful (predictions table), tight prose, tags on every number. It should be long but not bloated — this is a competition deliverable with 7 mandated sections. I'll aim for a thorough but disciplined document. Use markdown headers, a few tables. Avoid restating the pack. Every number tagged. Show arithmetic.

Tag discipline: [M] for measured from pack; derived arithmetic shown = [M-derived] or "derived"; [P] projected; [ASSUM] assumptions; [NM] not measured. The rules say tags are [M], [P], [ASSUM], [NM] — I'll use those, with arithmetic shown for derived numbers (the rules say "derive it and show the arithmetic" — so derived numbers can be tagged by their inputs; I'll mark derived as [derived: M+M] or similar. Keep it readable: e.g., "0.691 = 0.646 + 36/800 [derived from M]".

Let me also make sure I don't re-propose falsified things: binding aux loss (no), chrono-init (no), multi-horizon head (no), entropy-threshold choice points (no), interpolation compression (no), semantic-encoding gate (no), hyper-encoder (no). My design: SFT/DPO/RFT on program emission with VM adjudication — none of the falsified list. Also don't propose LLM-judging. Don't propose trunk retrieval-scoring (given §3.7, and binding-loss falsification). Good.

Also the 72 vs 107 discrepancy — flag it, require re-probe. Also NQ probe store mount unknown — flag [NM], require pinning. Also harvest volume [NM] — parametric design. These honesty flags are exactly the persona.

Let me draft the structure:

A. DIAGNOSIS
- The 0/300 is three confounds stacked: surface, frame narrowness, reachability. Plain statement of belief with basis (disfl_qa 15% clean [M] ⇒ surface can't be the main story). The cheap measurement: Probe S1 (mechanical normalization re-parse), S2 (gold-in-store mechanical), S3 (50-question two-annotator expressibility label). Prediction table.
- Ranked causes table: share × tractability.
- Disagreements: baseline is a confound; "beat grammar" too weak; harvest-manufacture claim false as stated.

B. SPEC
- Emitter must-do list (6 items) vs grammar's measured failures.
- Out-of-basin operational definition (mechanical membership rules, frozen hash).
- Document→triples spec (provenance tiers, mechanical span-binding, new snapshot, retire-not-delete), explicitly untested by the gate.
- Edge of expressible: REFUSE(reason) + partial trace; typed classes; contract.

C. FREE WINS
- Win 1: τ_ret recalibration, arithmetic, precision guard.
- Win 2: backward retrieval, arithmetic, 72-vs-107 flag.
- (Probe) Win 3: hop-retry on 92 misparsed — measure first.
- Joint arithmetic: 0.710 → 0.78–0.83 [P]; overlap caveat; why before training: (i) baseline honesty, (ii) snapshot invalidation makes pre-fix harvest retire (invariant 4), (iii) recovered chains are the informative supervision (backward direction, borderline τ), (iv) otherwise emitter credit is confounded with retrieval repair. Also replay re-certification cost: minutes [ASSUM on volume].

D. TRAINING DESIGN
- Data: fields used; SFT set composition (re-certified verified chains + rationalized out-of-basin products ≥ 30%); volume parametric [NM]; model 350M [ASSUM]; token arithmetic; GPU-time.
- Objectives: SFT → DPO (banned_facts, misparse pairs, runner-ups) → RFT round 2 (VM-filtered sampling on dev pool). No PRM/critic (invariant 1). No RL in gate (RFT is its cheap proxy; if RFT round-2 delta ≈ 0, the RL endgame lacks gradient — early read).
- Negatives/counterfactuals: uses without SFT merging (calibration, DPO, verifier regression, refusal eval).
- Snapshot invalidation: (hash, git_rev) pinning; replay re-certification; manifests.
- Typos: verbatim targets; query-side normalization only; orthography adversarial slice.
- Anti-memorization: relation-holdout, rationalized share, frozen sets, dedup, no synthetic paraphrase (invariant 6 caution).

E. BOOTSTRAP
- Mechanism: shift supervision manufacture from parse-gated to verify-gated (host-enumerated program search + VM forward-execution filter). The grammar's monopoly is parsing, not truth.
- Invariant-6 line: rationalization counts only if forward-executed and VM-verified; model never accepts its own derivation.
- Why it may fail here: reachability (0.1% lookup / 19.9% entity [M]; NQ yield [NM] — measure in week 1); search-prior signature → second grammar; gold-echo shortcut.

F. EXPERIMENT
- Sets: primary synthetic relation-holdout out-of-basin (n≈300–500), secondary NQ/disfl natural slices, edge set, in-domain 800.
- Arms: grammar v2, grammar-maxed, oracle search, emitter r1, emitter r2.
- Metrics + stratification.
- Pre-registered predictions table with numbers.
- Kill criteria K1–K3 + in-domain floor.
- Cost: ~1.5–2 GPU-h [P derived]; wall-clock 3 weeks plan.

G. SELF-FALSIFICATION
- One paragraph as drafted.

Length: this will be long. The prompt is a competition with 7 sections and demands specificity; a long, dense answer is appropriate. But I should keep it tight — no restating the pack, no fluff. Use tables for rankings and predictions. Maybe ~1800–2600 words. Let me write it.

Let me fix concrete numbers for the out-of-basin set construction:
- Synthetic: held-out relations: take the store's relation set, hold out ~20% of relation types (never in any training program). Generate questions: per relation × hop-structure {1,2,3} × surface {interrogative, keyword (lowercase, no '?'), disfluent-with-repair-cues} × {expressible, inexpressible-typed}. Target n: 300 expressible primary + 100 edge/inexpressible + 100 backward-required. All gold by construction; frozen hash.
- Natural: NQ-300 (all, stratified by detector 220/80) + gold-in-store filter (yield measured in wk1); disfl_qa parsed-clean slice as in-basin control.
- Also in-domain 800 as non-regression.

Predictions (crisp):
- S1 normalized NQ parse: 5–20% (base: disfl 15% clean [M]) [P].
- S2 NQ gold-in-store on 45k mount: 5–25% [P, wide; SimpleQA entity-presence 19.9% is the only measured analogue [M]].
- Grammar-maxed in-domain: 0.78–0.83 [P].
- Hop-retry probe: ≥ 50/92 verify [P].
- Oracle ceiling synthetic: 75–90% [P].
- Emitter r1 synthetic out-of-basin: 25–45% [P]; relation-holdout stratum 10–30% [P]; NQ-reachable 20–50% [P wide]; in-domain 0.69–0.79 [P].
- Emitter r2: +5–15 pts [P]; first-attempt certification 0.55–0.75 [P]; mean retries ≤ 1.5 at k=8