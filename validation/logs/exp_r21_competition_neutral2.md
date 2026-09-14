# Model competition — 2026-09-14-neutral-scoring-pack

GrillCheese Research Lab · 2026-09-13 · 1 of 5 models answered · the pack is `docs\research\2026-09-14-neutral-scoring-pack.md`

## deepseek/deepseek-v4-pro

*(no answer: empty)*

---

## moonshotai/kimi-k3

*(no answer: empty)*

---

## z-ai/glm-5.3

*(no answer: empty)*

---

## anthropic/claude-opus-5

# THE REFERENT/CLAIM SPLIT
*A prior may choose which question is put to the verifier. It may never change the answer the verifier gives to a fixed question.*

Everything below is one mechanism with that sentence as its invariant. Parameters marked `[ASSUM]` are knobs with no measured basis; each has a kill-switch in Q4. Arithmetic marked `[derived]` is shown.

---

## Q1 — The neutrality boundary

### 1.1 The split
Every decision in the pipeline is exactly one of:

* **(R) a reference decision** — *which object in the world is this question about?*
* **(C) a claim decision** — *does this proposition hold, and may it be spoken?*

**A prior may inform R-decisions only.** R-decisions are enumerable and must be enumerated in code — one `#[referent_decision]` attribute per site, CI fails if the prior module is called from an unmarked site:

| R (prior may inform) | C (prior may never touch) |
|---|---|
| seed item selection among tier-survivors | host coverage / COVER-refusal |
| hop-target selection when a hop's *object string* is ambiguous | τ_vm, cosine cleanup, symbol equality |
| binding the session context frame | ABSENT_ROLE control |
| **ordering** of `clarify.choices` shown to the user | which claim to walk when one item has several objects for the same relation |
| | speak / refuse |
| | the answer's language |

The fourth C-row is the trap the lab will hit next and it is not in §3: **multi-valued relations**. Once the item is unique, an item can still carry several objects for the needed relation (offices held, spouses, editions). Picking among *claims* is a claim decision. Reuse the `clarify` UI there if you like, but the tally is forbidden as its input; only qualifiers carried with the fact (time scope, source rank, provenance) may order it, and if nothing does, refuse. Invariant 4 dies the day someone reuses `prior::pick()` in that call site, because that is literally "the popular answer."

### 1.2 Is *choosing which item to fetch* already a violation?
**No — conditionally.** Fetching is not asserting; the fetched facts still face an unmodified gate. But the choice *does* determine what gets spoken (that is exactly the `james young` defect `[M]`), so it is legitimate only under all five tests below. Conversely, **selecting after the fetch, using the fetched values, is always a violation** — that is post-hoc selection over answers, i.e. a ranker.

The line inside a fetch: the selector may read a candidate's **relation keys** (does this item *have* a date-of-birth claim — a book edition does not), its `instance_of` classes, labels, aliases, descriptions. It may **never read the objects** of claims. Property *presence* is type evidence about what kind of thing this is; property *value* is truth. Tier 3 as built `[M]` sits on the legal side of that line; enforce it with a type, not a convention:

```rust
struct CandidateView {          // the ONLY thing the selector sees
  item: Qid, labels: Vec<Str>, aliases: Vec<Str>, description: Str,
  instance_of: Set<Qid>,
  relation_keys: Set<Pid>,      // keys only — no objects, ever
  incidence: Set<Qid>,          // items linked to this one, MINUS blindfolded relations (§1.3)
}
```

### 1.3 The five code-review tests

**T1 — Substitution invariance (the blindfold).** The selector takes `CandidateView`, which cannot express a claim object. Unit test: take a fixture store, mutate every claim *object* while holding ids/classes/relation-keys fixed, assert the selection is byte-identical. Additionally, the **target-relation blindfold**: `incidence` is computed with every relation in the chain program's remaining hop-set removed. If the only thing linking a candidate to the session context is the very relation the question asks about, that link is invisible to the selector — because using it would be choosing the item that produces the answer. Fall through to ask.

**T2 — No-lowering / fallback test.** `assert!(prior_called ⇒ tiers_returned_ambiguous)` and `assert!(selection ∈ tier_survivors)`. A prior may only convert **REFUSE→ASK** or **ASK→ANSWER**. It may never override a tier, widen the candidate set, re-run a failed gate, or touch a refusal whose cause was anything other than referent ambiguity (no-coverage, hop missing, τ_vm miss are off-limits). Downstream of selection the code path is the deterministic path, unmodified.

**T3 — No outcome feedback.** The prior's state has exactly one writer, with signature `fn update(ClarifyChoice)`. A grep test in CI asserts no other call site. **Verification success is not an input.** If "this item's walk verified" could raise its weight, the estimator would learn to pick whichever item yields *an* answer — answer-seeking bias — and the kill line would be silently converted into a selection pressure. This is the single most important test in the document.

**T4 — Disclosure.** The serializer refuses to emit an answer with `resolved_by = prior(..)` unless it carries the surviving alternatives *and* the rival answers (Q4.1). A prior decision that the user cannot see and undo is not permitted to exist.

**T5 — Bench firewall.** Benches run prior-off; the headline number is the prior-off number. Prior-on may only be reported as a delta. (Extends the rule already in §3 `[M]`.)

### 1.4 The one-line discriminator
> **Legitimate** if the prior changes *which question* reaches the verifier. **Illegitimate** if it changes *what the verifier would say* about a question already fixed.

Hostile reading, answered: "but choosing the referent changes the spoken answer." Yes — and that is why a prior decision is not allowed to be the last word. Three brakes, all cheap:

1. **Rival check.** Before speaking a prior-selected answer, run the *same* chain program against each surviving candidate. At 0.7 ms per VM verification `[M]`, four candidates cost 2.8 ms `[derived]`; run serially in one VM, so no allocation beyond the measured 1.69 MB `[M]` (`[ASSUM]`: serial reuse; concurrent rival checks unmeasured `[NM]`). Outcomes:
   * exactly one candidate verifies → **the prior did not decide anything**; this is tier 3 evaluated at full walk depth. Speak with `how: only surviving candidate that verifies`. No prior disclosure needed.
   * several verify and **agree on the final object** → the answer is invariant to the ambiguity. Speak with `how: ambiguity-invariant (k candidates, same object)`. **This is strictly neutral** and recovers refusals with zero prior involvement. (Not in §4 — see Q3.)
   * several verify and **disagree** → this is the `james young` class. Only here does the prior actually decide, and only if it clears the threshold *and* T4 disclosure fires; otherwise ask.
2. **Provenance unchanged.** Invariant 3 already forces the chain onto the answer; the chain names the chosen item, so the prior's fingerprint is always in the audit trail.
3. **The prior has no path to τ_vm.** T2 asserts it structurally.

---

## Q2 — The estimator

Discounted Dirichlet with a per-asker contribution cap, a one-sided Wilson lower bound as the decision rule, and two abrupt-drift guards. Per question: one hashmap lookup, ≤k multiply-adds, one `sqrt`. `[NM]` not benchmarked; bounded O(k) floats, k = surviving candidates.

### 2.1 State
Keyed by `key = (normalized_query_span, next_relation_pid)` — see Q5.2 for why the relation is in the key.

```
PriorKey -> { weights: [(Qid, f32)],      // pruned at w < 0.05
              last_obs: date,
              candset_fp: u64,            // hash of sorted surviving-candidate ids
              voters: [u64; 8],           // ring of recent asker-id hashes
              recent: [Qid; 3] }          // last 3 picks, for the freeze rule
```

### 2.2 Update — only on a `ClarifyChoice` (T3)
```
u = if asker_hash ∈ voters { 0.0 } else { 1.0 }   // one asker, one vote per key per window
if u == 0 { return }
for (_, w) in weights { w *= γ }                  // event-decay, on this key only
weights[chosen] += u
push(voters, asker_hash); push(recent, chosen); last_obs = today
```
`γ = 0.98` `[ASSUM]`. Two properties fall out for free:
* **Bounded confidence.** Steady-state total mass = `1/(1-γ) = 50` `[derived]`. No key can ever accumulate unbounded certainty, however much traffic it sees.
* **Decay is driven by observations, not the calendar.** A key seen three times in a year must not rot to nothing — sparsity is not staleness. Calendar staleness is handled at *read* time (2.3).

### 2.3 Decision rule
Let `W = Σw`, `p̂ = w_top/W`, and a read-time staleness haircut on the sample size only:
```
n_eff = W · 0.5^(days_since_last_obs / 365)          [ASSUM]
LCB   = Wilson_lower(p̂, n_eff, z),  z = 1.2816 (90% one-sided)
```
Decide iff **all** of:
1. `LCB ≥ θ`, `θ = 0.60` `[ASSUM]`;
2. ≥3 distinct askers contributed (structural, since `u` caps at 1.0 per asker);
3. `candset_fp` unchanged since the weights were accumulated — **novelty hold**: if a new candidate appeared (the store grew another *James Young*), the prior abstains for that key until it earns the threshold again on the new candidate set. A tally is only meaningful relative to the menu it was collected on;
4. `recent[0..3]` are not all ≠ `top` — **3-strike freeze**, the abrupt-drift guard: three consecutive picks against the standing prior freeze the key into ask-mode until it re-earns the threshold. γ alone is too slow for a regime break (see 2.4.3);
5. the rival check (Q1.4) left more than one verifying, disagreeing candidate — otherwise the prior is not consulted at all.

Otherwise: **ask** (B), or refuse if asking is impossible. Refusal is a result (invariant 2).

Why these constants cohere. With unanimity (`p̂ = 1`), Wilson collapses to `LCB = n/(n+z²)`, `z² = 1.6425` `[derived]`. `n/(n+1.6425) ≥ 0.60 ⇒ 0.4n ≥ 0.9855 ⇒ n ≥ 2.46 ⇒ n ≥ 3` `[derived]`. So the numeric threshold and the distinct-asker gate bite at the same point — three unanimous askers is the cheapest thing that can ever decide, by construction. (Sanity check that the threshold is reachable at all: at `z = 1.96, θ = 0.80`, unanimity at the mass ceiling n=50 gives `LCB = 50/53.84 = 0.929` — reachable, but requires 50 distinct askers per key; at the scale in §3 (177 questions reaching a fetch `[M]`) that is a dead code path. `θ=0.60, z=1.2816` is

---

## qwen/qwen3.8-max-0902

*(no answer: empty)*

---
