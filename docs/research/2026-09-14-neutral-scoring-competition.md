# The neutral-prior competition — what four frontier models said about scoring without ranking

GrillCheese Research Lab · 2026-09-14 · `validation/exp_r21_competition.py`

Nick asked for an algorithm that lets the VM stay neutral while scoring the most
requested answers, so the loop can eventually guess the context when none is
given — and then asked for it to be put to a model competition, because it is a
hard question and getting it wrong is how a verifier quietly becomes a ranker.

The pack is `docs/research/2026-09-14-neutral-scoring-pack.md`: five questions
(Q1 the neutrality boundary, Q2 the estimator, Q3 other admissible signals,
Q4 honesty and measurement, Q5 an unforeseen failure mode), the seven invariants,
and the measured numbers the lab already has.

## Contestants

| round | model | answered | completion | of which reasoning | cost | wall |
|---|---|---|---|---|---|---|
| 1 | openai/gpt-5.6-terra-pro | 4,357 ch | 10,756 | 8,541 | $0.1604 | 60 s |
| 1 | x-ai/grok-4.6 | 10,633 ch | 9,241 | 6,291 | $0.0587 | 186 s |
| 1 | google/gemini-3.8-flash | 11,422 ch | 3,496 | 295 | $0.0144 | 30 s |
| 1 | deepseek/deepseek-v4-pro | — | 3,500 | 3,500 | $0.0077 | 97 s |
| 1 | moonshotai/kimi-k3 | — | 3,500 | 3,502 | $0.0512 | 52 s |
| 1 | z-ai/glm-5.3 | — | 3,500 | 3,500 | $0.0178 | 14 s |
| 1 | qwen/qwen3.8-max-0902 | — | 3,500 | 3,500 | $0.0246 | 112 s |
| 2 | anthropic/claude-opus-5 | 10,247 ch | 16,000 | 12,047 | $0.4093 | 242 s |
| 2 | deepseek/deepseek-v4-pro | — | 16,000 | 15,999 | $0.0280 | 413 s |
| 2 | moonshotai/kimi-k3 | — | 16,000 | 16,000 | $0.2434 | 509 s |
| 2 | z-ai/glm-5.3 | — | 16,000 | 16,000 | $0.0723 | 84 s |
| 2 | qwen/qwen3.8-max-0902 | — | 16,000 | 16,000 | $0.0989 | 480 s |

**4 of 12 attempts produced an answer, at $1.19 total, of which $0.51 bought
nothing.** The failure is the same in both rounds and it is not a budget: four
models spent 100% of the completion budget on reasoning tokens and emitted zero
visible characters, at 3,500 and again at 16,000. Raising the cap raised the
spend and not the yield — kimi went from $0.05 to $0.24 for the same silence.
The harness's `--effort` flag (a `reasoning` cap in the OpenRouter body) is the
lever to pull, not `--max-tokens`; round 3 tests that. **Bench note for the lab:
a model that returns nothing is a measurement about the harness, not about the
model, and it must be priced as such before the next competition.**

Opus 5's answer was itself truncated by the cap mid-sentence in Q2.3, so its
Q3–Q5 are missing. What survives is still the sharpest single contribution.

## Q1 — Where the boundary is

All four converge, independently, on the same split, in three different
vocabularies:

- **gpt-5.6-terra-pro**: an `InterpretationRecord` and a `ProofRecord`. The prior
  may read and write only the first. Exactly one value crosses: `bound_item_id`.
- **grok-4.6** and **gemini-3.8-flash**: the same separation, reached from the
  invariants rather than from record design.
- **opus-5**: the **referent/claim split**, with the cleanest one-line form of it:

  > A prior may choose *which question* is put to the verifier. It may never
  > change *the answer the verifier gives* to a fixed question.

And all four propose the same review test — a **counterfactual non-interference
check**: fix the bound item, the plan, the source snapshot and the store, then
replace every prior score with arbitrary values; the proof record and the VM
decision must be byte-for-byte identical.

This is the part to adopt verbatim. It is also, encouragingly, what the loop
already does: the ledger in `standin/sources.py` is consulted only inside
`resolve()`, and nothing it returns reaches the gate, `τ_vm`, the cosine cleanup
or the walk.

**Is choosing which item to fetch already a violation?** All four say no,
conditionally, and the conditions agree: the deterministic tiers must run first
and the prior is reachable only when they leave several candidates; the prior may
pick only from that frozen set; picking may not relax `COVER`, provenance, or any
refusal condition; and — gpt-5.6's condition 5, which is the one most likely to
be violated by accident — **if the chosen item cannot produce a certified answer,
the system refuses with that partial trace and does NOT try the next candidate
until one verifies.** Retrying down the ranking is answer-seeking, and it is how
the kill line dies quietly.

Opus 5 draws the line *inside* the fetch more precisely than the others, and it
is worth keeping as a type rather than a convention: the selector may read a
candidate's labels, aliases, description, `instance of` classes, and **which
relations it has** — but never **the objects of those relations**. Property
*presence* is evidence about what kind of thing this is; property *value* is
truth. The lab's ordered narrowing (2026-09-13, with the `of_a_kind` guard added
2026-09-14) reads presence only, so it sits on the legal side — but nothing in
the code says so, and a future edit could cross it silently.

## Q2 — The estimator, and the four defects they found in the lab's first one

The first ledger keyed a tally on the bare name, counted questions, never
decayed, and could fire on a single observation. Round 1 found all four
independently, and they were fixed on 2026-09-14 (`08ce7bf`) before round 2 ran:

| defect | who found it | the fix now in `sources.py` |
|---|---|---|
| keyed on the bare name — a new namesake inherits the tally | grok-4.6 (its Q5) | key is `(name, candidate-set fingerprint)` |
| counts questions, not askers | all three | one vote per asker |
| no decay | all three | 90-day exponential |
| fires at n=1 | all three | ≥3 askers, margin ≥2, share ≥0.7 |

Round 2 then found a defect in **the fix**. Opus 5's estimator decays **per
observation** (`γ = 0.98` applied on each update to that key) rather than by the
calendar, with the argument:

> A key seen three times in a year must not rot to nothing — **sparsity is not
> staleness.**

That is a real hit on the lab's 90-day half-life. A rare but perfectly stable
ambiguity (one asker every four months, always meaning the same person) decays to
nothing under a calendar rule and never binds, while a burst of traffic in one
week binds immediately. Event-decay separates the two: calendar age becomes a
*read-time haircut on the sample size*, not an erasure of the evidence.

Two more from Opus 5 worth taking:

- **Bounded confidence.** With event-decay at `γ`, steady-state total mass is
  `1/(1-γ)` — 50 at `γ = 0.98`. No key can ever accumulate unbounded certainty
  however much traffic it sees. The lab's ledger has no such ceiling.
- **A Wilson lower bound as the decision rule** instead of three hand-set
  thresholds, and the observation that they then cohere by construction: at
  unanimity `LCB = n/(n+z²)`, so `θ = 0.60, z = 1.2816` bites at exactly `n ≥ 3`
  — the same place the distinct-asker gate does. One constant instead of three,
  and the "three unanimous askers is the cheapest thing that can decide" property
  falls out rather than being asserted.

And the **novelty hold**: when the candidate set changes, abstain on that key
until the threshold is re-earned on the new set. The lab's fingerprint-in-the-key
achieves the same end by a different route (a changed set is simply a new key,
with no history) — Opus 5's version keeps the key and freezes it, which also
records that something changed. Either is sound; keep ours, note the difference.

Where all four agree, and the lab already complies: **only an explicit
clarification choice updates the state.** Not an auto-selected prior decision,
not the fact that an answer verified, not the absence of a correction, not a
click. Opus 5 calls the second one out as the single most important test in its
document, and gives the reason plainly: if "this item's walk verified" could
raise an item's weight, the estimator learns to pick whichever item yields *an*
answer, and the kill line becomes a selection pressure. **This deserves a CI
test, not a comment** — one writer for the ledger, asserted by grep.

## Q5 — The failure mode the lab had not considered

This is what the competition was for, and it arrived:

> Once the item is unique, an item can still carry **several objects for the same
> relation** (offices held, spouses, editions). Picking among *claims* is a claim
> decision. Reuse the clarify UI there if you like, but **the tally is forbidden
> as its input**; only qualifiers carried with the fact — time scope, source rank,
> provenance — may order it, and if nothing does, refuse.
> — opus-5, Q1.1

The lab was about to walk straight into this. The open thread from 2026-09-14 is
the `ambiguous_hop` residue ("who was governor in 2015", "what is the capital of
France"), and the obvious next move — now that `world.times` can tell dated values
apart — was to break the tie with the same asker ledger that breaks referent ties.
That would have been the day invariant 4 died, and it would have looked like a
feature: *the popular answer*, served by a verifier.

The correct fix is the narrow one, and the machinery for it landed the same day:
**time qualifiers order multi-valued relations, and nothing else may.** The
profile ask already does exactly this — `547 is the population of Quebec City`
(1608) and `574482` (2024) are both true, and the latest-dated rule picks by
`point`/`start`, never by popularity. The walk must use that same rule, and where
no qualifier separates the values, it refuses. A refusal is a result.

## What to adopt

1. **Name the boundary in the code.** The referent/claim split as a type, not a
   habit: the selector sees labels, aliases, description, classes and relation
   *keys*, never claim *objects*.
2. **The counterfactual non-interference test**, as a real test: scramble every
   prior score, assert the proof record is byte-identical.
3. **A CI test that the ledger has exactly one writer**, and that verification
   success is not it.
4. **Never retry down the ranking.** If the selected item does not certify,
   refuse with the partial trace.
5. **Event-decay instead of calendar decay**, with calendar age as a read-time
   haircut — sparsity is not staleness.
6. **A mass ceiling** (`1/(1-γ)`), so no key becomes unfalsifiable.
7. **Time qualifiers, and only time qualifiers, order multi-valued relations.**

## What to reject

- **Every "popularity" signal, unanimously.** All four explicitly refuse
  sitelinks, fact count, PageRank and incoming-link counts as priors, in the same
  terms: that is scoring how famous the world is, not what this asker meant, and
  it imports someone else's editorial judgement into a verifier.
- **Wilson + `γ` as a drop-in**, for now. It is better than what the lab has, but
  every constant in it is marked `[ASSUM]` by its own author and the lab has 177
  questions that reach a fetch — not enough traffic to distinguish `θ = 0.60`
  from `θ = 0.70` empirically. Adopt the *shape* (event-decay, a mass ceiling, one
  threshold instead of three); leave the numbers where measurement can reach them.

## Not answered

Opus 5's Q3 (other admissible signals) and Q4 (honesty and measurement) were cut
off by the token cap, and the four silent models never answered anything. The
**rival check** it proposed in Q1.4 is the most interesting unexplored idea and
belongs in the next pack: run the same chain program against *each* surviving
candidate — if exactly one verifies, the prior decided nothing; if several verify
and agree on the final object, the answer is invariant to the ambiguity and can
be spoken with **zero** prior involvement. At 0.7 ms per VM verification and
1.69 MB per session (both measured, `exp_r20`), four candidates cost 2.8 ms. That
is a free recovery path for refusals that needs no prior at all, and the lab does
not have it.
