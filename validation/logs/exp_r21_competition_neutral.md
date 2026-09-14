# Model competition — 2026-09-14-neutral-scoring-pack

GrillCheese Research Lab · 2026-09-13 · 3 of 7 models answered · the pack is `docs\research\2026-09-14-neutral-scoring-pack.md`

## openai/gpt-5.6-terra-pro

## Q1. Neutrality boundary

### Rule

A prior may choose an **interpretation route**; it may never contribute to a **fact judgement**.

Formally, split serving into two records:

```text
InterpretationRecord:
  normalized_query
  candidate_item_ids
  deterministic-tier outcomes
  prior features / scores
  chosen_item_id | clarify
  reason

ProofRecord:
  bound_item_id
  fixed walk plan
  fetched fact IDs + provenance
  VM program
  VM checks, including τ_vm and ABSENT_ROLE control
  certified answer | refusal
```

The prior may read and write only `InterpretationRecord`. The VM, chain walker, fact fetch admission, fact store, plan coverage check, relation traversal, cosine threshold, and answer rendering from certified symbols may read no prior score, count, or feature.

The only value crossing the boundary is:

```text
bound_item_id
```

and it must be one member of the frozen exact-match candidate set.

### Is selecting an item to fetch allowed?

Yes, but only as interpretation routing, under all of these conditions:

1. The host first enumerates the complete candidate set from the same deterministic exact-match process used for clarification.
2. All deterministic tiers run first. A prior is reachable only if they leave multiple candidates.
3. The prior selects only one existing candidate ID, never an item returned by broader search, embedding retrieval, source order, or a generated query.
4. Selecting an item does not relax `COVER`, source provenance requirements, walk rules, VM thresholds, or any refusal condition.
5. If the chosen item cannot produce a certified answer, the system refuses with that partial trace. It must not try the next-highest prior candidate until one verifies.
6. The response says that the item was an interpretation selected from ambiguity, not a fact established by the prior.

Thus the prior is allowed to determine *which proposition is attempted*, but not whether that proposition is accepted.

### Code-review test

Use a counterfactual noninterference test:

> Fix `bound_item_id`, the plan, source snapshot, fact-store contents, and VM configuration. Replace every prior feature and score with arbitrary values. The resulting proof record and VM decision must be byte-for-byte unchanged.

And the complementary test:

> Fix the candidate set and replace the prior’s selected ID with another candidate ID. The only allowed differences downstream are those causally implied by binding that different ID: its fetched facts, its walk, its proof, and its refusal or certified answer.

Fail review if a prior value can affect any of:

- candidate generation;
- source ordering or retrieval cutoff;
- plan coverage;
- relation selection after the item is bound;
- which facts are admitted;
- cosine cleanup or `τ_vm`;
- equality checks;
- provenance;
- retry/fallback behavior after VM refusal;
- wording that hides the interpretation choice.

This preserves the distinction: **the host may score candidate meanings; only the VM certifies an answer conditional on the bound meaning.**

---

## Q2. Estimator

### Scope the tally correctly

Do not maintain one global count for a label such as `james young`. Maintain state by an ambiguity key:

```text
ambiguity_key =
  source_version
  language
  normalized_surface_form
  candidate_set_fingerprint   # sorted stable item IDs
  intent_signature            # requested next property, if any; otherwise ⊥
```

`candidate_set_fingerprint` matters. If Wikidata changes the candidate set, old counts must not silently transfer to a materially different ambiguity.

`intent_signature` permits separate evidence for, for example, “birth date of Marie Curie” and “Marie Curie book ISBN,” while preserving the existing deterministic “question decides” tier ahead of the prior.

### State [P]

For each key, store:

```text
last_decay_time
explicit_choice_weight[item_id]
last_explicit_choice_time[item_id]
distinct_chooser state sufficient for one contribution per chooser/session window
```

Only an explicit clarification selection updates this state. Do **not** update from:

- an auto-selected prior decision;
- the fact that an answer verified;
- lack of correction;
- clicks on an answer;
- downstream use of the answer.

Those are not clean observations of intended referent.

### Update [P]

At an explicit selection of item `j` at time `t`:

```text

---

## x-ai/grok-4.6

# Neutral prior — design

The prior lives **only on the host**, in a store that is not the fact store. The VM binary, its inputs, `τ_vm`, BIND/UNBIND, cosine cleanup, ABSENT_ROLE, COVER, and the benches never read it. A spoken sentence is still exactly one VM-certified chain plus an auditable `how`.

---

## Q1. Neutrality boundary

**Scoring** = binding a *seed item* (or deciding to ASK) among items that already survived tiers 1–4. **Verifying** = walking facts for that binding and running the VM. Scoring stops when the seed is bound. Verifying begins at the first hop.

### May touch
- Residual set `S` after the four deterministic tiers (linked / unique exact / question-property / label>alias + meta drop).
- Order of `clarify.choices`.
- Host-side proposal: `select(S, context, prior) → item ∈ S | ASK`.
- The answer record field `how` (already used).

### Must never touch
- COVER.
- Which facts exist, their provenance, hop order, recovered-symbol check, `τ_vm`, BIND_ROLE/UNBIND, ABSENT_ROLE.
- Tiers 1–4 (prior is not a fifth ranker that overrides them).
- Emitter logits / SEED preference among homonyms (host disposes; serving path has no external LLM).
- Benches (already: never written or read by benches).
- The fact store (no prior-derived facts, no popularity predicates).

Choosing which item to **fetch** is **not** a verification violation: fetch is candidate selection; truth is still gated by the VM. It **is** an identity claim the moment that chain is **spoken as the answer to the question**. Wrong item + true facts about it = the `james young` defect. So fetch-from-prior is allowed; **silent identity** is not (see Q4).

### Code-review rule (checkable)
1. `certify(program, facts, τ)` does not take prior state. A PR that adds prior to VM arguments is rejected.
2. `S = tiers_1_to_4(query)` does not take prior state.
3. `select` may return only `ASK` or an element of `S`.
4. No speak unless `certify` is true.
5. If `S` today ≠ the item-set the estimator is keyed on, `select` must return `ASK` (invalidate; do not carry an old unique max onto a new namesake).

### Distinguishing test
Invert or zero the prior and replay the same questions against a frozen fact store.
- **Legitimate:** the set of VM accept/reject decisions for each `(program, item, facts)` is identical; only *which* residual item is attempted, or the order of clarify choices, may change; every speak still has `how`.
- **Illegitimate:** a chain that failed `certify` becomes speakable, `τ`/COVER/hops change, an item outside `S` is fetched, or a fact is added/removed.

---

## Q2. The estimator

Cheap, host-side, per residual ambiguity, CPU. Not a model.

**Do not** increment on auto-select (that freezes the prior). **Do not** count questions; count **askers**. One vote per `(asker_id, key)`.

### Key
```
label_norm = unicode-fold(query entity string)
S         = frozenset of item IDs that survived tiers 1–4
key       = (label_norm, S)
```
If `S` changes (new Wikidata namesake, item deleted), it is a different key. Old counts do not decide.

### State
```
UserVote[asker_id][key] = { item_id, t }
Global[key]             = derived; not stored as raw increments
```
`Global[key][i]` = decayed mass of askers whose **last** explicit clarify pick for `key` is `i`.

Decay (no source — assumption: namesake fame and who-asks mix move on a quarterly scale):
```
w(t) = exp(-λ Δt),  λ = ln(2) / 90 days,  Δt = now - vote.t
```
Mass:
```
c_i = Σ_{askers last-picking i} w(t_asker)
N   = Σ_i c_i
k   = |S|
```

### Update
On **explicit** clarify only: write `UserVote[asker][key] = {chosen, now}` (last write wins, so a person can change sense). Recompute `c_i` from votes whose `item_id ∈ S`. Ignore votes for items no longer in `S`.

Cap: one stored vote per asker per key (ballot stuffing by repeating the question does nothing).

### Decision rule (auto-bind, not sort)
Laplace only for reporting; **decide** on decayed votes, not smoothed fake mass:
```
i* = unique argmax_i c_i     else ASK
c2 = second-max c_i or 0
auto-bind i* iff all of:
  (D1) i* ∈ S and key.S == S today
  (D2) distinct askers with a live vote for this key ≥ n_a
  (D3) N ≥ n_mass
  (D4) c_i* ≥ c2 + g          // not 51/49
  (D5) c_i* / N ≥ p
  (D6) shadow-promoted (Q4)
```
No source — assumption, basis = beat a 51/49 and a single lab vote: `n_a = 3`, `n_mass = 3`, `g = 2`, `p = 0.7`.

Sort of clarify choices (always allowed, even when D1–D6 fail): `c_i` desc, then label.

### Edge cases the rule already names
| Situation | What happens |
|---|---|
| One observation | `n_a = 1` → ASK. Sort that item first. |
| 51/49 | Unique max but D4/D5 fail → ASK. |
| Popular last year, not now | Old votes have `w ≈ 0`; they drop out of `N` and `n_a` (treat `w < 1/16` as dead, i.e. 1 year at this λ). Argmax follows recent last-picks. |
| Same asker repeats | Last pick wins; does not stack. |
| `S` gains a new item | New key; ASK until that set has its own votes. |
| Auto-select happened | No update. |

Personal context is **not** this estimator; it is Q3 signal 1 and may bind with `n=1`.

---

## Q3. Signals (same neutral channel: bind seed or ASK, never truth)

All of these are evidence about **which item the asker meant**, not about whether a claim holds. Ranked.

| Rank | Signal | Why it is *meant*, not *true* | Use |
|---|---|---|---|
| 1 | **This asker’s last explicit pick** for the same `key`, item still in `S` | They already said which namesake they mean. | Auto-bind A. `how: same-asker (1 of 1)`. |
| 2 | **Tier 1 linked, this session** — item is the object of a fact already served in this conversation (or the seed of a verified chain the question anaphorically continues) | The question is continuing a binding they already accepted. | Already built as linked; keep it above any tally. |
| 3 | **Question tokens minus the name**, unique hit on one residual item’s label/alias/description/occupation string (exact token, not embedding) | Extra words the asker typed. Deterministic filter, same shape as tier 3. | Auto-bind only if **exactly one** item in `S` hits; else ignore. |
| 4 | **Global last-pick histogram (Q2)** | What *askers of this string with this residual set* meant. | Sort always; auto-bind only if D1–D6. |
| 5 | **Same-asker last pick for `label_norm` with a different `S`**, if that item still ∈ `S` today | Weak context when Wikidata’s competitor set flickered. | Sort boost only, never auto-bind. |

**Guess when there is none:** cascade 1 → 2 → 3 → (4 only if promoted) → B ask with choices sorted by 4 then 1.

### Do not use
- Sitelinks, fact-count, fetch size (~180), graph degree, PageRank — those score **how famous the world is**, the recommender the kill line already forbade.
- Emitter/SEED preference among homonyms — ranker; host disposes.
- Bench / SimpleQA frequencies — benches must not feed serving.
- VM cosine / `τ_vm` to pick a namesake — that number means “this hop recovered the walked object”, not “this is who you meant”. Using it to choose identity mixes scoring into verifying.
- Trying every item in `S` and speaking the first whose needed role is present — that is tier 3 if it **uniquely** eliminates; if two people both have a birth year it silently prefers fetch order (the original defect).

---

## Q4. Keeping it honest

### What must be visible
**Log** (every prior-touched outcome): `key`, `S`, snapshot `{item: c_i}`, `n_a`, `N`, rule fired (1/2/3/4 or ASK), `how`, asker-id hash, `prior_decided: bool`.

**User** (if a prior or context **bound** the seed): the spoken answer **names the item** with the same `{label, detail}` the clarify UI would have shown, and the `how` string. Example shape (not a recorded utterance): birth date, then `James Young — Scottish chemist (how: asker history, 7 of 9)`. A verifier coat with the namesake stripped off is a recommender.

**User** (ASK): choices **sorted** by `c_i`, not silently skipped. Sorting is the “score most requested” path; it never speaks an identity the user did not confirm.

### Measurement that can falsify (shadow first)
Hold **fixed**: fact store snapshot, VM, `τ_vm`, tiers 1–4, emitter, COVER. Split is not on questions the deterministic tiers already resolved.

- **Control arm:** residual ambiguity always ASK (today).
- **Shadow arm (default):** run `select`; **do not bind**; still ASK. Record `shadow_item` vs `user_pick`.
- **Live arm:** only after promotion, auto-bind under D1–D6 + named identity in the utterance.

Promotion (no source — assumption: must not add identity errors the control cannot add): promote key-by-key when that key has ≥ 20 shadow pairs **and** `shadow_item == user_pick` on ≥ 18, **and** distinct askers ≥ `n_a`. Demote that key (and disable global auto-bind globally) on the first confirmed mismatch.

**Confirmed mismatch:** same asker, later explicit pick of a different item in `S` for that `key`; or they re-ask and choose another clarify row.

**Switch off auto-bind (sort may remain) when any of:**
- One spoken answer whose bound item is a confirmed mismatch — kill line; pin a regression that day (invariant 4).
- Shadow agreement on a promoted key falls below the promotion bar on a rolling 20.
- Residual `S` changed for that label (Q2 key break is the mechanism; the measurement is a counter of would-have-auto-bound-on-stale-S, which must stay 0).

Do not use fact-correctness benches to watch this. They will pass on the `james young` failure (the year is true of the wrong man). The instrument is **identity agreement**, harvested from clarify and from later picks, never from VM scores.

---

## Q5. Failure you will not see coming

**Stale unique-max after a new namesake, plus starved updates.** Auto-bind removes explicit picks, so `UserVote` stops moving. Wikidata then adds a new item that belongs in `S`. If `select` is keyed only on `label_norm` (or uses stale `S`), the old chemist still unique-maxes, the VM certifies a true chain, the trace looks perfect, benches that check facts look clean — same class as `james young`, invisible to verification.

Second, quieter form: **lab-population lock-in**. Three internal askers satisfy `n_a`; the first external asker is auto-bound to the lab’s sense. Distinct-asker votes without a **population tag** will not show this.

**Cheapest instrument:** on every Wikidata refresh, for each stored key, recompute `S'` from tiers 1–4; if `S' ≠ key.S`, **delete the key from auto-bind** (force ASK) and increment `prior.stale_S`. Alert if `prior.stale_S > 0` and any live auto-bind fired that day. Stratify shadow agreement by asker-cohort (internal vs not). Neither instrument needs an LLM or a new bench.

---

## google/gemini-3.8-flash

### Q1. The Neutrality Boundary

#### 1. The Code-Review Rule
A prior may only rank **input candidate groundings** (which entity $E \in \{E_1, \dots, E_k\}$ the surface text refers to) prior to host program synthesis; it must never mutate the **derivation**, the **fact store**, or the **VM predicate**.

In code review, the rule is enforced by an explicit type and pipeline separation:
```rust
// PERMITTED: Pure ranking over candidate entity IDs prior to program dispatch
fn rank_candidates(candidates: &[EntityId], prior: &PriorStore) -> Vec<(EntityId, PosteriorScore)>;

// FORBIDDEN: Any prior passed to VM state, vector-symbol unbind, or threshold logic
fn vm_verify(program: &CubeLangProgram, store: &FactStore /* NO PRIOR ALLOWED */) -> Result<Trace, Refusal>;
```

#### 2. Where Scoring Stops and Verifying Begins
* **Scoring stops** the instant a single candidate entity ID $E^*$ is selected to bind the seed role in the emitted `CubeLang` program.
* **Verifying begins** when the host loads facts anchored at $E^*$ from the fact store. The VM then performs vector-symbolic unbinding (`BIND_ROLE` / `UNBIND` + cleanup at cosine $\ge \tau_{\text{vm}}$) and checks the `ABSENT_ROLE` negative control. 

#### 3. Is Choosing Which Item to Fetch a Violation?
**No**, provided the execution is conditioned on an explicit hypothesis. A fact store query is simply evaluating the conditional proposition: *"Assuming the user meant entity $E^*$, does there exist a certified proof chain for question $Q$?"* 

However, because Wikidata contains true facts about the wrong entity (e.g., the real birth year of the wrong *James Young* `[M]`), invariant #4 (0 wrong answers spoken) means choosing an item to fetch *can* lead to a verified falsehood if the question itself is unspecific. 

#### 4. The Distinguishing Test: Invariance Under Counterfactual True Facts
To test whether a candidate usage is legitimate:
> **The Counterfactual Substitution Test:** Suppose candidate $E_1$ has true fact $F_1 = (E_1, R, V_1)$ and candidate $E_2$ has true fact $F_2 = (E_2, R, V_2)$, where both $V_1$ and $V_2$ satisfy all VM role unbindings for relation $R$. If injecting an arbitrarily high prior weight for $E_1$ can flip the VM's acceptance of an answer derived from $E_2$ *without changing the input string or the fact store*, the usage is **illegitimate**. If the prior merely dictates which entity hypothesis is submitted to the VM first—and the system remains contractually bound to refuse if ambiguity is unresolved—the usage is **legitimate**.

---

### Q2. The Estimator

To remain robust against sparsity, small-sample bias, and temporal drift on a CPU without external libraries, we use an **Online Beta-Binomial / Dirichlet Posterior with Exponential Time-Decay and a Posterior Margin Test**.

#### 1. Mathematical Specification
For an ambiguous label string $s$ resolving to candidate set $\mathcal{E} = \{E_1, E_2, \dots, E_k\}$:

* **State per entity $E_i$ under string $s$**:
  $$S(s, E_i) = \left( w_i, t_i \right)$$
  where $w_i \in \mathbb{R}_{\ge 0}$ is the decayed selection weight (initialized to $0.0$), and $t_i$ is the Unix timestamp of the last update. Let $\tau_{\text{half}} = 90\text{ days} = 7.776 \times 10^6\text{ s}$ `[ASSUM: entity relevance drifts on seasonal/quarterly scales]`.

* **Decay Operator**: At current time $t$:
  $$w_i(t) = w_i \cdot 2^{-\frac{t - t_i}{\tau_{\text{half}}}}$$

* **Update Rule**:
  When an asker explicitly disambiguates and selects item $E^* \in \mathcal{E}$ via clarify mode (B):
  $$w_{E^*}(t) \leftarrow w_{E^*}(t) + 1.0; \quad t_{E^*} \leftarrow t$$
  Non-chosen candidates $E_j \ne E^*$ merely have their decay applied:
  $$w_j(t) \leftarrow w_j(t); \quad t_j \leftarrow t$$

* **Posterior Parameterization**:
  Use a symmetric Dirichlet prior with concentration $\alpha_0 = 1.0$ across all $k$ candidates (Laplace smoothing). The expected posterior probability $p_i$ is:
  $$\hat{p}_i = \frac{w_i(t) + 1.0}{\sum_{j=1}^k w_j(t) + k}$$

#### 2. The Decision Rule & Thresholds
The prior is permitted to bypass asking the user **if and only if** all three criteria hold:
1. **Effective Sample Size (Total Observations)**:
   $$N_{\text{eff}} = \sum_{j=1}^k w_j(t) \ge N_{\min}, \quad \text{where } N_{\min} = 10 \text{ `[ASSUM]`}$$
2. **Absolute Dominance**:
   $$\hat{p}_{\text{top}} \ge 0.80 \text{ `[ASSUM]`}$$
3. **Margin over Runner-Up (Bayes Factor / Margin Ratio)**:
   $$\frac{\hat{p}_{\text{top}}}{\hat{p}_{\text{second}}} \ge 4.0 \text{ `[ASSUM]`}$$

If any criterion fails, the host **must refuse to guess and output clarify `choices:[...]`**.

#### 3. Specific Edge-Case Behavior
* **On one observation ($N=1$, choice $A$ over $B$, $k=2$)**:
  $$\hat{p}_A = \frac{1 + 1}{1 + 2} = 0.667, \quad \hat{p}_B = \frac{0 + 1}{1 + 2} = 0.333$$
  $N_{\text{eff}} = 1.0 < 10$. **Decision: Do not decide. Present clarify choices.**
* **On a 51/49 split ($w_A = 51, w_B = 49$, $k=2$)**:
  $$\hat{p}_A = \frac{52}{102} = 0.5098, \quad \frac{\hat{p}_A}{\hat{p}_B} = \frac{52}{50} = 1.04 < 4.0$$
  Fails both dominance and margin. **Decision: Do not decide. Present clarify choices.**
* **On a sense popular last year ($t - t_{\text{last}} = 365\text{ days} \approx 4\tau_{\text{half}}$)**:
  Its weight decays by $2^{-4} = 0.0625$. An initial weight of $w=80$ decays to $w(t) = 5.0$. If a new candidate receives 10 recent picks ($w_{\text{new}} = 10.0$), the stale candidate cannot block the shift or force an invalid decision; $N_{\text{eff}}$ drops, prompting clarification until the new distribution stabilizes.

---

### Q3. Guessing the Context When There Is None

The system has deterministic graph and session signals already present in memory. Ranked strictly by epistemic validity regarding *which item was meant* (referential intent):

```
Rank 1: Session Entity Graph Proximity (Co-reference / Locality)
Rank 2: Graph Degree on the Target Relation (Schema Salience)
Rank 3: Host-side Historical Selection Prior (Q2 Estimator)
Rank 4: Global Fact Degree (Out-degree Centrality in Stored Slice)
```

#### Detailed Breakdown
1. **Rank 1: Session Co-reference / Graph Distance within Current User Memory (1.69 MB VM session `[M]`)**
   * *Why it is referential evidence*: If the current user session previously verified facts involving *Pierre Curie*, the probability that the label *Marie Curie* refers to the physicist (Q7186) rather than the ferry or metro station is an act of conversational co-reference, not an assertion about the physical truth of her birth date.
2. **Rank 2: Schema Compatibility Degree for the Walk Target**
   * *Why it is referential evidence*: Tier 3 currently checks whether an entity carries property $P$ `[M]`. When multiple candidates carry $P$, the cardinality/qualification of $P$ (e.g., an author having 40 books vs a corporate entity having 1 accidental publication tag) indicates how central the role is to that entity's definition.
3. **Rank 3: Decayed Historical Selection Tally (The Q2 Estimator)**
   * *Why it is referential evidence*: Measures empirical human linguistic convention for unqualified names (e.g., unqualified "Paris" universally referring to the capital of France).
4. **Rank 4: Global Fact Degree in the Stored Fact Slice (552,297 facts `[M]`)**
   * *Why it is referential evidence*: Reflects representation depth in the curated world. An entity with 180 facts `[M]` versus an entity with 1 fact indicates that the world slice was materialized around the former.

#### Signals NOT to Use
* **Do NOT use relation frequency / path length across the full graph as a confidence booster for the answer**: This conflates "how well connected this node is" with "whether this answer is true," violating Invariant #1.
* **Do NOT use string similarity metrics (e.g., Levenshtein or token embedding cosine) below exact hit**: A near-match string on a wrong entity yields verified falsehoods (`james young` `[M]`).
* **Do NOT use language model generation likelihood / perplexity from the emitter**: LMs hallucinate common entities over correct rare entities; using LM score as a prior re-introduces the recommender failure mode that the VM was designed to eliminate.

---

### Q4. Keeping It Honest

#### 1. Visibility Contract
When a prior breaks an ambiguity tie, the system must emit:

* **To the User**: The answer must explicitly qualify the resolved referent and expose an instant undo mechanism:
  > *"Assuming you mean **Marie Curie (physicist, 1867–1934)**: 7 November 1867. [Not who you meant? See other choices]"*
* **To the Audit Log**:
  ```json
  {
    "provenance": {
      "grounding_tier": "prior_disambiguation",
      "prior_type": "asker_history_decayed",
      "p_top": 0.882,
      "margin_ratio": 5.41,
      "n_eff": 28.4,
      "runner_up": {"id": "Q12345", "label": "Marie Curie (ferry)"},
      "vm_trace": "0.7ms, verified, cosine=0.94"
    }
  }
  ```

#### 2. The Verification Experiment & Control Arm
To ensure the prior does not silently introduce defects, evaluate it against a shadow control arm:

* **Split**: Continuous A/B split on ambiguous queries meeting the Q2 threshold:
  * **Arm A (Prior-Decided)**: Automatically select $E^*$ using the prior; output qualified answer with "Not who you meant?" button.
  * **Arm B (Control - Always Ask)**: Emit tier 4 clarify prompt (`choices: [...]`) as done today `[M]`.
* **Telemetry**:
  * In Arm A, measure the **Explicit Rejection Rate ($R_A$)**: fraction of queries where user clicks "Not who you meant?" or immediately re-prompts with a disambiguating term within 60 seconds.
  * In Arm B, measure the **Selection Concordance ($C_B$)**: fraction of users who pick the candidate that Arm A's prior would have selected.

#### 3. The Kill Line Switch-Off Metric
The prior subsystem is **automatically disabled** (reverting the entire system to Tier 4 user-asking) if:
$$\text{Defect Rate} = \frac{\text{Prior Mis-disambiguations}}{\text{Total Prior-Decided Turns}} > 0.005 \text{ (0.5\%) `[ASSUM]`}$$
Any single verified wrong answer spoken where the error stemmed from prior grounding immediately triggers the kill line (Invariant #4: pinned regression test on day 0, prior disabled for that surface string permanently).

---

### Q5. The Failure You Expect

#### 1. The Non-Obvious Failure Mode: "Disambiguation Trapping via Self-Fulfilling Feedback Loops"
When the prior reaches the threshold ($N \ge 10$, $\hat{p} \ge 0.80$, margin $\ge 4.0$), it **stops asking the user**. 

Once the system stops presenting choices, subsequent users asking about the minority entity (e.g., *James Young* the basketball player rather than *James Young* the chemist) never see the disambiguation prompt. They receive an answer about the chemist. Most users do not hit "Not who you meant?"—they either assume the system is buggy, read the wrong fact without noticing the entity qualification, or abandon the session. 

Because the clarify UI is suppressed, **no new negative feedback or minority picks can ever be logged**. The decay operator only decays relative to time, but as long as queries for the majority entity arrive at a steady rate, the top candidate's weight remains refreshed while the minority candidate remains permanently suppressed at weight $0$. The prior converts a temporary popularity bias into an irreversible epistemic monopoly.

#### 2. The Cheapest Instrument to Detect It Early
**The 5% Stochastic Clarification Probe

---

## deepseek/deepseek-v4-pro

*(no answer: empty)*

---

## moonshotai/kimi-k3

*(no answer: empty)*

---

## z-ai/glm-5.3

*(no answer: empty)*

---

## qwen/qwen3.8-max-0902

*(no answer: empty)*

---
