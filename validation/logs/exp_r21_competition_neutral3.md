# Model competition — 2026-09-14-neutral-scoring-pack

GrillCheese Research Lab · 2026-09-13 · 3 of 4 models answered · the pack is `docs\research\2026-09-14-neutral-scoring-pack.md`

## deepseek/deepseek-v4-pro

*(no answer: empty)*

---

## moonshotai/kimi-k3

# THE NEUTRAL PRIOR — design submission

The core claim: **a prior may only ever choose *which question gets asked of the verifier*, never what the verifier accepts.** Everything below follows from that one sentence.

---

## Q1. The neutrality boundary

**The rule, checkable in code review:**

> A prior may influence only the *selection of candidates presented to the VM*. It may never influence (a) whether a candidate passes, (b) the similarity threshold τ_vm, (c) the ABSENT_ROLE control, (d) the COVER check, (e) the content of any stored fact or its provenance, or (f) whether a refusal is converted into an answer.

Concretely, the prior is a function `prior(query_string, candidate_items) → item | ASK | REFUSE` that runs **before the fetch**, in the host, and its output is one of three things: pick one item to fetch, ask the user, or refuse. That is the entire surface.

**Where scoring stops and verifying begins:** at the fetch boundary. Everything upstream of "fetch claims about item X" is proposal space — the emitter already lives there and is already untrusted. Everything downstream — walk, chain, VM gates, per-hop symbol equality, τ_vm, ABSENT_ROLE — is decision space and must be *prior-free by construction*: the VM binary should not even import the prior module. Enforce this structurally, not by convention: the prior's output type is `ItemId | Clarify | Refuse`, and the VM's input type is `ItemId + question`. There is no channel by which a score can flow downstream. A code review checks: (1) no call from VM/chain code into prior state; (2) τ_vm and the controls are constants in VM scope; (3) the prior's state is write-only from the serving path except at the single decision point.

**Is choosing which item to fetch a violation?** No — and this is the crux. Fetching item X is *asking the verifier a different question*, not asserting an answer. The fetched facts are gated hop by hop; if the prior picks the wrong James Young, the walk either fails (refusal, honest) or — the dangerous case, already observed `[M]` 2026-09-12 — *succeeds on the wrong item*. So the invariant that actually protects the kill line is not "prior never picks"; it is:

> **A prior-picked item must survive the same verification as any other, and the answer must be labeled as prior-selected so it can be audited, challenged, and measured separately.**

The existing four tiers are *deterministic* (they decide or they don't). The prior is a **fifth tier, strictly after tiers 1–4, strictly before asking the user**, and it may only fire when tiers 1–4 all failed. It never overrides a tier that decided. That ordering is the hostile-reading test: if the prior can ever outrank "linked" or "the question decides," it has become a recommender.

**The legitimacy test (one sentence):** *Swap the prior for a uniform random picker over the surviving candidates. If the system's correctness guarantee changes, the prior was touching verification. If only the refusal/ask rate changes, it was legitimate.* This is implementable as an actual shadow-mode A/B (see Q4).

---

## Q2. The estimator

The tally `[M]` is the right idea; it needs three fixes: sparsity, drift, and the unique-maximum rule's behavior on ties and single observations.

**State.** Per ambiguous label L: a set of candidate items {i}, each with a decayed count. Store `(s_i, w_i)` where `s_i` = effective count, plus `last_update`. No probabilities, no smoothing priors beyond a uniform pseudocount.

**Update.** On observation of a pick of item i at time t:
- Decay all counts for L: `s_j ← s_j · 2^(-(t - last_update)/H)` with half-life **H = 90 days** (no source — assumption; basis: drift in public salience is a months-scale process, and H must be ≫ the inter-observation gap for the estimator to be stable; H is a tunable, logged constant, not a measured value).
- `s_i += 1`. On a *prior-decided* answer that the user does not challenge (Q4's challenge channel), add a fractional observation `s_i += 0.25` (no source — assumption; basis: an unchallenged auto-pick is weaker evidence than an explicit choice, and the weight must be < 1 so the prior cannot bootstrap itself to the threshold unaided — see Q5).

**Decision rule.** Let `s_(1) ≥ s_(2)` be the top two decayed counts, `N = Σ s_j`. The prior decides item (1) iff **all** of:
1. `s_(1) ≥ n_min` with **n_min = 4** effective observations (no source — assumption; basis: below this, one user's habit is the whole signal; 4 is the smallest count at which a 2:1 margin is even expressible);
2. margin: `s_(1) ≥ 2 · s_(2)` (a 2:1 likelihood-ratio style rule — cheap, interpretable, and it makes the 51/49 case refuse by construction);
3. recency: at least one observation of item (1) within the last H days (kills the stale-sense case).

Otherwise: **ask** (the existing `clarify` record). Never refuse from the prior — refusal stays available but asking is strictly better when candidates exist, since asking is itself the data-collection mechanism.

**The three required cases:**
- **One observation:** `s_(1) = 1 < 4` → ask. The tally never decides on a single pick.
- **51/49 split:** margin 1.02 < 2 → ask, forever, until drift resolves it. Correct: a coin-flip sense pair *should* be asked.
- **Popular last year, not now:** decay shrinks old counts below `n_min` and the recency gate fails → ask, and the new observations accumulate on the new sense at full weight while the old sense bleeds off. The estimator tracks drift without any explicit change detection.

**Cost:** per label, a handful of floats; update is O(#candidates for that label). Trivially CPU-cheap per question.

---

## Q3. Signals for guessing context, ranked

All signals below are evidence about *which item the asker means* because none of them enters the fact store or the VM; they only select the fetch target, and the fetch is verified.

1. **The question's own next relation (already built, tier 3).** Strongest signal because it is *internal to the question*: "when was X born" presupposes an item with a birth-date claim. It is not a popularity judgment at all — it is a type constraint. Keep it ahead of everything.
2. **The asker's own history (per-user tally, not global).** The current design `[M]` uses a global tally. A per-user slice of the same estimator (same rules, separate state, higher `n_min` or same threshold applied to a sparser table) is purer: what *this* user meant before is evidence about what *this* user means, uncontaminated by other users' senses. At 1.69 MB per per-user VM `[M]`, a per-user tally table is negligible additional state.
3. **Session context.** Earlier verified answers in the same session constrain the domain: a session whose verified chain is about chemistry makes *Marie Curie the physicist* more likely than *the ferry*. Mechanism: take the set of relations/properties already verified this session; among surviving candidates, prefer the item whose claims overlap that relation set (same "the question decides" trick, generalized from one relation to the session's relation profile). Evidence about intent, not truth — the chosen item still faces the full walk.
4. **Global asker tally (the existing signal).** Use it, but *below* per-user and session signals: it is the biased one (biased by who asks), and it is the one that most resembles a popularity ranking, so it should be the last resort before asking.
5. **Linked-fact structure (already tier 1).** Not new, but note it is the only signal keyed by facts rather than names — the model for what "neutral" means.

**Signals I would NOT use:**
- **Corpus frequency / Wikipedia sitelink counts / pageviews.** This is the trap. It is a popularity prior over the *world*, not over *this asker's intent*, and it is exactly the mechanism by which the system quietly becomes a recommender ("the famous one"). It also cannot drift-correct honestly — fame decays slowly and unevenly. Excluded on principle, not on cost.
- **The emitter LM's confidence or preference.** The emitter proposes; letting its preference pick among ambiguous items gives the untrusted component a second, unaudited vote. The emitter already had its say by emitting the program.
- **Anything from an external LLM** — invariant 5 forbids it on the serving path anyway.

---

## Q4. Keeping it honest

**What the user sees when the prior decided.** The spoken answer carries, in the same record format as the existing `how: asker history (n of m)` `[M]`:
- `how: prior (asker history, n of m, margin 2.3:1)` — the signal class, the counts, the margin;
- the rejected candidates (the same `choices` list that would have been shown in a `clarify`);
- a one-word challenge affordance: "not who I meant?" — a challenge is logged, counts as a *negative* observation for the picked item and a positive one for the user's correction, and the corrected answer is re-verified from the new item. The challenge is the conscience of the whole design: a prior that decides must make disagreeing cheaper than the ask it replaced.

**What the log sees.** Every prior decision records: candidates surviving tier 4, all decayed counts, the decision rule's three gates' values, and whether the user challenged. This makes the prior fully replayable offline.

**The measurement that could falsify the design.** Shadow mode first, then an interleaved live arm:
- **Hold fixed:** the four deterministic tiers, τ_vm, the VM, the fetch logic, the refusal format. Only the post-tier-4 disposition varies.
- **Control arm:** current behavior — ask the user whenever tiers 1–4 fail. **Treatment arm:** prior decides under the Q2 rule; ask otherwise.
- **Primary metric: challenged-and-wrong prior decisions** — a prior pick the user rejects. This is the kill-line-adjacent number: a wrong item that *verified* (like the 2026-09-12 James Young case `[M]`) is the only way the prior manufactures a wrong spoken answer. Secondary metrics: ask-rate reduction (the payoff), challenge rate, and the fraction of prior decisions that would have been asked under control (should be 100% of treatment decisions by construction — a sanity invariant).
- **Switch-off number:** if the challenged-wrong rate on prior decisions exceeds the wrong-spoken rate the system tolerates — and the stated kill line is **0 wrong answers spoken** `[M]` — then the honest threshold is: **one challenged-wrong prior decision that reached a spoken, verified answer triggers automatic demotion of the prior to shadow mode and a pinned regression test, same day** (extending the existing rule `[M]`). Softer switch-off: if the prior's decisions are challenged at a rate that makes the ask-rate reduction worthless (challenges are just asks with extra steps), it is also failing — but the hard line is the spoken-wrong one.

Also run the Q1 swap test continuously in shadow: a random-picker arm over the same candidates. If the prior's challenge rate is not clearly below the random arm's, the prior is doing nothing and should be removed for complexity alone.

---

## Q5. The failure the designers won't see coming

**Self-confirmation: the prior manufactures its own evidence.** Once the prior decides, the user mostly accepts (challenging costs effort; the answer is verified and *plausible* — it passed the VM). Each unchallenged acceptance, if counted at full weight, feeds the estimator; the margin grows; the threshold gets easier to clear; the ask rate collapses — and the system has drifted from "verifier with a tie-breaker" to "recommender with a verifier's coat" *without any single decision being wrong*. The James Young failure mode `[M]` — a wrong item that verifies cleanly — is exactly the case where the user is least likely to challenge, because the answer looks certified. The bias is invisible in the challenge rate, because the unchallenged mass is the problem.

This is why Q2 caps the auto-pick observation at 0.25: the prior cannot reach `n_min = 4` on its own acceptances alone; it needs explicit human picks. But even that is not the instrument — it's the mitigation.

**Cheapest instrument that detects it early:** a **decoy audit**. On a small, fixed fraction of prior-decided questions (say every Nth, N logged), the system *asks anyway* — presents the `clarify` choices with the prior's pick included but not indicated. If users, asked blind, pick the prior's item at a rate no better than its share of the choices, the prior's "acceptances" were inertia, not agreement. The divergence between *unchallenged acceptance rate* (near 100%, expected) and *blind re-ask agreement rate* is the early-warning signal, and it costs one extra question every N decisions. No new infrastructure: the clarify record and the logging already exist `[M]`; the decoy is a scheduling flag.

Secondary instrument, nearly free: track the **effective-observation half-life of margins** — if the median margin of prior decisions is monotonically increasing month over month while the blind re-ask agreement is flat, the estimator is inflating itself. One plotted line, reviewed weekly.

---

## z-ai/glm-5.3

# The Neutral Prior — design

## Q1. The neutrality boundary

**Rule (code-review checkable):** *A prior may select among candidate referents; it may never select among candidate answers.*

Formally: the prior's output must be an **item ID from the disambiguation candidate set**, consumed *before* the walk begins, and the VM's verification inputs must be byte-identical regardless of which item the prior picked. If you delete the prior, the set of possible *spoken answers* changes only in that some questions move from "ask" to "refuse/verify-other-item" — never in the truth-value of any answer that is spoken.

Where scoring stops and verifying begins:

- **Prior may touch:** candidate pruning (which of N exact-label hits to fetch first), fetch ordering, the `choices` list ordering shown to the asker, and the decision *ask vs. proceed* when tiers 1–4 are silent.
- **Prior may never touch:** τ_vm, the similarity gate, the ABSENT_ROLE control, the host's coverage check, what counts as a "near" answer, the fact store's contents, and anything downstream of `BIND_ROLE`.
- **Is choosing which item to fetch a violation?** No — *provided* the fetched facts are gated by the full VM path and the prior's pick is recorded as a *hypothesis*, not a conclusion. The test: **the counterfactual-fetch test.** For every question the prior decided, run the same walk on the *other* candidates. If any other candidate would also have verified cleanly, the prior's choice was between two true-but-different answers, and the system must ask (see Q2 threshold). If exactly one candidate verifies, the prior merely saved a fetch — it decided nothing about truth. This test can run offline on every prior-decided question at 0.7 ms × k candidates per hop, i.e. essentially free.

**The distinguishing test for review:** *Does this use of the prior change what the VM would accept?* If yes → illegitimate. If it only changes which program is *attempted*, and a failed attempt yields refusal or ask → legitimate.

## Q2. The estimator

Cheap, CPU-per-question, honest about sparsity and drift. Use a **decayed Beta posterior over the candidate set**, one Beta per (label-string, item) pair.

**State:** for each ambiguous label L and candidate item i: `α_i, β_i` (floats, not counts, so decay is multiplication). Initialize α=β=0.5 (Jeffreys). Store per-item, not per-label, so candidates can be added without resetting.

**Update:** asker picks item i → `α_i += 1`; every other candidate j in that question's choice set → `β_j += 1`. Then a global decay applied lazily on read: `α, β ← α·γ^(Δt/h), β·γ^(Δt/h)` where Δt is time since last update of that pair and h is a half-life unit. Pick γ=0.5, h = one quarter — no source, assumption: fast enough that "popular last year" halves in a quarter, slow enough that a handful of picks still decide. Decay on *read*, not on write, so cost is O(candidates) per question.

**Decision rule — the prior may auto-proceed only if ALL hold:**
1. Posterior probability `p_i = α_i/(α_i+β_i)` has a unique maximum.
2. `p_i ≥ 0.95` (i.e., the posterior is confident, not just the leader).
3. **Effective sample size** `n_eff = Σ(α_j+β_j) − 1 ≥ 8` after decay. One observation gives n_eff≈1 → cannot decide. This is the anti-single-vote rule.
4. **The counterfactual-fetch test passes** (Q1): no other candidate verifies cleanly.

If 1–3 hold but 4 fails → ask, with the prior's leader listed first in `choices` (disclosed as such).

**Edge cases:**
- **One observation:** α=1.5, β=0.5 → p=0.75 < 0.95 → ask. Correct.
- **51/49 split, n=200:** leader p≈0.51 < 0.95 → ask. Correct — a coin-flip referent is genuinely ambiguous.
- **Popular last year, not now:** decay pulls n_eff below 8 within ~a quarter of silence → falls back to ask. If new picks flip, the posterior follows within a handful of updates because β grows on the old leader. No retraining, no batch job.
- **New candidate appears** (a new item acquires the label): starts at Jeffreys prior, cannot win until it earns picks; the old leader may auto-proceed meanwhile — acceptable, because rule 4 (counterfactual fetch) still protects the kill line: if the new item also verifies, the system asks.

Why this beats a raw tally: the tally's "unique maximum" fires on 2-vs-1; the posterior demands n_eff≥8 *and* 95% — and decay makes staleness self-correcting.

## Q3. Other context signals, ranked

All of these are evidence about *which item was meant* because each one is a constraint supplied by the asker or the conversation — none of them asserts anything about the world, so none can make a false answer true or a true answer false; they can only steer which walk is attempted, and the VM still gates the result.

1. **Linked facts already in the conversation (tier 1, generalized).** If a prior answer's chain contained item X, and the new question's label matches an alias of X, the asker almost certainly means X. Strongest signal; already keyed by fact, not name. Evidence-about-referent because it comes from the asker's own discourse.
2. **The question's own next relation (tier 3, generalized to *any* mention, not just the walk's next hop).** E.g. "Marie Curie's husband" — the string "husband" filters candidates before the walk. Already built; extend to surface verbs/nouns in the question as soft candidate filters.
3. **Co-mentioned entities.** "Jim Haslam's son" — "son" plus the other candidate *being* the son's alias. This is a graph check, not a prior: if candidate B is the object of a `child` claim from candidate A, the question structure itself disambiguates. Deterministic, so it should be promoted to a tier-5 rule before any prior fires.
4. **Ask-choice history (the decayed Beta above).** Weakest of the admissible; only signal about *population* intent, not this asker's.
5. **Session-internal tally** (this user's own past picks), folded into the same Beta with a small extra weight — no source, assumption: an asker's own prior picks are more predictive of their next pick than the population's.

**Would NOT use:**
- **Wikidata sitelink counts / pageviews / Wikipedia popularity.** That's a recommender by construction: it encodes what the world finds interesting, not what the asker meant, and it would silently bias toward famous referents exactly when the asker means the metro station. Also violates the spirit of "label shared by two or more items is refused, never picked by rank" — this *is* a rank.
- **The emitter LM's guess.** The emitter writes programs; letting its implicit prior pick referents would move a popularity prior inside the proposal path where it can't be audited, and the serving path must not gain LM-dependent referent choice.
- **Anything derived from the fact store's claim counts** (item with more facts wins) — same objection, plus it's a proxy for notability.

## Q4. Keeping it honest

**User-visible:** when the prior auto-proceeds, the answer carries `how: asker history (p=0.97, n_eff=23 of 41)` — extend the existing `how` field with the posterior and effective n, plus `alternatives: [other candidate labels]` so the asker can see a different Marie Curie existed. When the prior only orders the choices, `choices` are annotated `ordered by: asker history`.

**Log-visible:** every prior decision logs (label, chosen item, all candidates, p_i, n_eff, counterfactual-fetch outcome per candidate). The benches never read or write it — keep that rule.

**The falsifying measurement — an interleaved holdback:**
- **Hold fixed:** the emitter, host, VM, τ_vm, fact store snapshot, question set (re-run SimpleQA 600 plus the accumulated pinned regression set).
- **Control arm:** identical binary with the prior's *decision rule* disabled (it may still log). Treatment arm: prior enabled. Randomize per-question with a deterministic hash so re-runs are reproducible.
- **Primary metric:** wrong answers spoken (must stay 0 in both arms — any nonzero kills the prior immediately, regardless of anything else).
- **Secondary:** correct answers spoken, asks issued, refusals. The prior is worth keeping only if correct↑ and asks↓ with wrong=0.
- **Switch-off numbers:** any wrong answer in the treatment arm → off that day. If asks decrease but correct answers do *not* increase proportionally — i.e., the prior is converting would-be asks into refusals or nears — the prior is guessing badly: off. Concretely, define conversion = Δcorrect/Δasks_decided; if it falls below the rate at which askers answer their own clarify prompts (measurable from existing logs: fraction of `clarify` records that received a choice — derive it, don't assume), the prior is worse than asking → off.

## Q5. The failure you expect

**Feedback-loop capture of the tally.** The prior auto-proceeds on the popular item; users who meant the *other* item get a wrong-seeming (though VM-true) answer about the wrong referent, never ask, and never register a pick. Meanwhile, every genuine clarify that *does* fire reinforces the same leader. The Beta counts only record disagreement that survived to a prompt — so the estimator's own confidence is generated by the cases it already decided, and it drifts toward permanent auto-proceed on labels where it was marginally wrong. The designers won't see it because the kill line stays at 0: the system speaks only verified truths about the *wrong Marie Curie*.

**Cheapest instrument:** a **silent counterfactual arm** on every prior-decided question: run the walk on the non-chosen candidates too (0.7 ms/hop), and log `multi-verify events` — questions where ≥2 candidates verified cleanly. If the multi-verify rate on prior-decided labels trends up while the clarify rate trends down, the prior is suppressing the asks that would have corrected it. Alarm threshold: multi-verify rate on prior-decided questions exceeding the multi-verify rate on tier-surviving asked questions by a factor of 2 — no source, assumption: a 2× gap indicates the prior is selecting into exactly the ambiguity it was supposed to detect. Second, near-zero-cost: track the *answer* of prior-decided questions against follow-up questions in the same session that presuppose the other referent (the asker's next question is free supervision).

---

## qwen/qwen3.8-max-0902

# THE NEUTRAL PRIOR — Design Proposal

---

## Q1. The Neutrality Boundary

### The Rule (checkable in code review)

**The prior is a function whose output enters the pipeline at exactly the same point and with exactly the same treatment as a user's clarification response.** Concretely:

```
disambiguate(query, candidates) → item | ASK
```

If it returns `ASK`, the system asks the user (existing behavior). If it returns an `item`, that item is injected into the pipeline at the same code path where a user's clarification choice is injected. No downstream code knows or cares whether a human or the prior made the selection.

### What the prior may touch

- The selection among candidate items **when all four deterministic tiers have failed** (i.e., the system would otherwise ask).
- Nothing else.

### What the prior may never touch

| Layer | May the prior influence it? | Why |
|-------|----------------------------|-----|
| Tier 1–4 decisions | No | These are deterministic and correct by construction |
| Which facts are fetched | Only indirectly, via item selection | Same as a user's choice |
| VM verification threshold `τ_vm` | No | Verification is truth-gating, not preference |
| Whether to speak or refuse after verification | No | The VM decides |
| The ABSENT_ROLE control | No | Integrity check |
| What is logged as `how:` | Only its own tag | Provenance |

### Is choosing which item to fetch already a violation?

No, but it is a **risk**. The fetched facts are gated and verified, so no *false* fact is spoken. However, a *true but wrong-target* fact can be spoken (the `james young` incident). The prior can therefore cause a kill-line violation — not by corrupting verification, but by directing verification at the wrong entity.

This means the prior is not free. It is a **bet** that must be measured and bounded.

### The Test (for code review)

**Counterfactual Swap Test:** For any answer `A` spoken because the prior selected item `I`:

> *If we replaced the prior's output with a user clarification response of `I`, would the system produce the identical answer `A`?*

If yes → legitimate. The prior saved the user a question; nothing else changed.
If no → the prior is touching something downstream. Reject in review.

**Implementation:** This is enforced architecturally, not by testing each case. The prior's output is a value of type `ClarificationChoice`, the same type the user's response produces. There is no code path that distinguishes them. A reviewer checks: does any branch exist that reads `source == "prior"` after the disambiguation step? If yes, reject.

---

## Q2. The Estimator

### State

Per ambiguous label `L` with candidate items `{i₁, …, iₖ}`:

```rust
struct PriorState {
    counts: Vec<f64>,       // length k, initialized to α each
    last_update: Timestamp,
}
```

Global parameters (constants, not learned):
- `α = 1.0` (uniform Dirichlet prior)
- `λ = 0.9923` (half-life ≈ 90 days; `λ = 0.5^(1/90)`)
- `n_min = 3` (minimum effective observations before the prior may decide)
- `p_dom = 0.85` (dominance threshold)

### Update (on each user pick)

When user picks item `iⱼ` for label `L`:

```
Δt = now - last_update    // in days
for m in 0..k:
    counts[m] = α + (counts[m] - α) * λ^Δt
counts[j] += 1.0
last_update = now
```

This is O(k) per update, k ≤ ~10 for any realistic label. Trivially cheap.

### Decision Rule

```
total = sum(counts)
effective_n = total - k * α      // subtract the prior's contribution
j = argmax(counts)

if effective_n < n_min:          → ASK
if counts[j] / total < p_dom:   → ASK
else:                            → return item[j]
```

### Edge Cases

| Scenario | Counts (α=1, k=2) | effective_n | Dominance | Decision |
|----------|-------------------|-------------|-----------|----------|
| One observation of A | (2, 1) | 1 | 2/3 = 0.67 | **ASK** (< n_min) |
| 51 picks A, 49 picks B | (52, 50) | 100 | 52/102 = 0.51 | **ASK** (< p_dom) |
| 10 picks A last year, 5 picks B this week | After decay: A ≈ 2.6, B = 6 | ≈ 6.6 | 6/8.6 = 0.70 | **ASK** (< p_dom) |
| 10 picks A last year, 15 picks B this week | After decay: A ≈ 2.6, B = 16 | ≈ 16.6 | 16/18.6 = 0.86 | **Decide B** |
| 20 picks A, 2 picks B, all recent | (21, 3) | 22 | 21/24 = 0.875 | **Decide A** |

The dominance threshold of 0.85 means the prior needs strong evidence. With a uniform prior over k=2 candidates, you need roughly a 6:1 ratio of effective observations to cross 0.85. This is conservative by design.

### Why not a more sophisticated model?

A Beta-Binomial or hierarchical model would be more statistically efficient, but:
- The data is too sparse to benefit (most labels have < 10 observations)
- The cost of a wrong decision is high (kill line)
- The cost of asking is low (one extra turn)
- Simplicity makes the neutrality boundary auditable

---

## Q3. Context Signals (Ranked)

All signals below are evidence about **which item the user meant**, not about what is true. They constrain the referent, not the fact.

### Signals to use, ranked by expected utility:

| Rank | Signal | Why it's about intent, not truth | Availability |
|------|--------|----------------------------------|--------------|
| 1 | **The question's next-hop relation** (type constraint) | "When was X born?" requires P569, which a metro station lacks. This constrains which item *can* answer the question — pure intent. | Already in tier 3. Could be strengthened: instead of requiring the property to exist, require the item's *type* (P31) to be compatible with the relation's domain. |
| 2 | **Co-occurring entities in the question** | "Marie Curie's husband" → the person, because metro stations don't have husbands. The co-occurring entity constrains the semantic type of the target. | Available in the question text. Requires a lightweight type check: does the candidate item have a relation to the co-occurring entity? |
| 3 | **Session context** (previous questions in this conversation) | If the user asked about Nobel prizes 30 seconds ago, "Marie Curie" almost certainly means the physicist. This is evidence about the user's current topic, not about truth. | Available in session state. Implementation: maintain a set of recently-walked relations and entity types; score candidates by type overlap. |
| 4 | **The asker's historical picks** (the tally from Q2) | Population-level evidence about which item is usually meant. Not about truth — the metro station is equally real. | Already built. |
| 5 | **The asker's individual history** (distinct from population) | If *this specific user* always picks the physicist when asked about Marie Curie, that's strong evidence about their intent. | Available from logged picks, keyed by asker ID. |

### Signal I would NOT use:

| Signal | Why not |
|--------|---------|
| **Wikidata pageviews / sitelink count** | This is a pure popularity signal external to the system. It tells you what *most people* mean, but it doesn't adapt to this user, this session, or this question. Worse, it creates a feedback loop: popular items get selected more, confirming their popularity, making unpopular items invisible. It is the exact mechanism by which a verifier becomes a recommender. It also requires an external fetch, adding latency and a dependency. |
| **QID magnitude** (lower QID = older = more famous) | Same problem as pageviews — a popularity proxy with no user-specificity. |

### How to combine signals 1–5

They are not independent, so don't multiply probabilities. Instead, use them as **filters in sequence**, each narrowing the candidate set:

```
candidates = all exact-match items
candidates = filter_by_type_constraint(candidates, next_hop_relation)   // signal 1
candidates = filter_by_cooccurrence(candidates, question_entities)      // signal 2
if |candidates| == 1: return candidates[0]
candidates = score_by_session_context(candidates, session_state)        // signal 3
if top_score / second_score > θ_session: return top
candidates = score_by_prior(candidates, asker_id)                       // signals 4+5
apply decision rule from Q2
```

Each filter is conservative: it only eliminates candidates that are *impossible* given the constraint (e.g., a metro station cannot have a date of birth). It never ranks by popularity.

---

## Q4. Keeping It Honest

### What the user sees

When the prior decides, the response includes a disambiguation notice:

```
[Assumed "Marie Curie" → Marie Curie (physicist, Q7186). 
 Based on 14 prior selections. Say "other" to see alternatives.]
```

This is non-optional. The user must always be able to see that a choice was made for them and override it. The override re-enters the pipeline as a user clarification.

### What the log sees

Every prior decision is logged as:

```json
{
  "label": "Marie Curie",
  "chosen_item": "Q7186",
  "alternatives": ["Q12345", "Q67890"],
  "estimator_state": {"counts": [15.2, 1.1, 0.9], "effective_n": 14.2},
  "dominance": 0.88,
  "how": "prior",
  "session_context_used": true,
  "timestamp": "..."
}
```

### The measurement

**Design:** Stratified A/B on ambiguous queries where the prior would decide.

- **Treatment arm (90%):** Prior decides. Answer is spoken with disambiguation notice.
- **Control arm (10%):** Prior is suppressed. User is asked (existing behavior).

**What is held fixed:** The fact store, the VM, the verification pipeline, the tiers. Only the disambiguation step differs.

**Metric:** *Agreement rate* — in the control arm, what fraction of users pick the same item the prior would have chosen?

**Secondary metric:** *Re-ask rate* — in the treatment arm, what fraction of users ask about the same label again within the same session or within 24 hours? A re-ask is a proxy for "the prior got it wrong and the user is trying again."

**Kill criterion:**

| Metric | Threshold | Action |
|--------|-----------|--------|
| Agreement rate (control arm) | < 85% over rolling window of 30 decisions | Disable prior globally |
| Re-ask rate (treatment) − re-ask rate (control) | > 5 percentage points over rolling window of 100 | Disable prior globally |
| Any single label's agreement rate | < 70% over 10 decisions | Disable prior for that label only |

The agreement rate is the ground truth: it directly measures whether the prior predicts what the user would have chosen. The re-ask rate is a continuous signal that doesn't require the control arm.

---

## Q5. The Failure Designers Won't See Coming

### The failure: **Silent coverage narrowing via confirmation loop.**

The prior picks the popular item. Users who wanted the unpopular item receive an answer about the wrong entity. Most will not correct it — they'll rephrase, leave, or simply not notice the answer is about the wrong thing (especially if the wrong entity has a plausible-sounding answer). The system logs this as a successful interaction. The prior's confidence in the popular item increases (no correction = implicit confirmation). Over weeks, the unpopular item's effective count decays toward α, and the prior becomes locked in.

The system now *cannot* reach the unpopular item without the user explicitly saying "other." The prior has silently narrowed the system's coverage. No wrong answer was spoken (the VM verified a true fact about the popular item), but the system is now a recommender: it gives you the answer it expects you to want, not the one you asked for.

This is invisible in the kill-line metric because no false fact was spoken. It is invisible in the agreement rate until the control arm is large enough to notice the drift. It looks like success.

### The cheapest instrument

**Track the "other" override rate per label.**

Every time a user sees the disambiguation notice and says "other" (or re-asks with a different item), log it as an override. Compute:

```
override_rate(label) = overrides / (overrides + accepted_prior_decisions)
```

over a rolling 30-day window.

**Alarm threshold:** If `override_rate(label) > 0.15` for any label, the prior is causing harm for that label. Disable the prior for that label and revert to asking.

**Why this catches the failure early:** The confirmation loop takes weeks to lock in. Override rate rises within days of the prior making bad decisions, because the users who are affected notice immediately. It's a leading indicator, while agreement rate (which requires the control arm) is a lagging indicator.

**Cost:** One counter per label, incremented on two events. No additional infrastructure.

### Secondary failure: **Prior poisoning by a single asker.**

If one user repeatedly picks the obscure item for a label (maliciously or because they genuinely always mean the obscure one), they can skew the prior for all other users. 

**Instrument:** Cap any single asker's contribution to any label's counts at 3 effective observations. Implementation: track per-asker counts per label; when updating, if `asker_count[label][user] >= 3`, do not increment. This is O(1) per update and prevents Sybil-style manipulation without requiring identity verification.

---
