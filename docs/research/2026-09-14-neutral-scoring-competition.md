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
| 3 | moonshotai/kimi-k3 | 13,033 ch | 3,075 | **0** | $0.0455 | 80 s |
| 3 | z-ai/glm-5.3 | 10,192 ch | 2,473 | **22** | $0.0133 | 10 s |
| 3 | qwen/qwen3.8-max-0902 | 13,039 ch | 8,062 | 4,724 | $0.0517 | 199 s |
| 3 | deepseek/deepseek-v4-pro | — | 16,000 | 16,000 | $0.0292 | 408 s |

**7 of 16 attempts produced an answer, at $1.33 total, of which $0.54 bought
nothing.** Rounds 1–2 alone were 4 of 12 for $1.19. The failure is the same in both rounds and it is not a budget: four
models spent 100% of the completion budget on reasoning tokens and emitted zero
visible characters, at 3,500 and again at 16,000. Raising the cap raised the
spend and not the yield — kimi went from $0.05 to $0.24 for the same silence.
The harness's `--effort` flag (a `reasoning` cap in the OpenRouter body) is the
lever, not `--max-tokens` — and **round 3 confirmed it decisively**. The same
four silent models, same pack, same 16,000 cap, with `--effort low`:

| | round 2 | round 3 (`--effort low`) |
|---|---|---|
| answered | 0 of 4 | **3 of 4** |
| cost | $0.44 | **$0.14** |
| kimi-k3 | 0 chars, 16,000 reasoning tokens, $0.24 | 13,033 chars, **0** reasoning tokens, $0.046 |
| glm-5.3 | 0 chars, 84 s | 10,192 chars, **10 s**, $0.013 |

glm answered in ten seconds for a penny after burning 16,000 tokens on silence.
Only deepseek-v4-pro still spends the whole budget thinking and says nothing, at
any setting tried. **Bench note for the lab: a model that returns nothing is a
measurement about the harness, not about the model. Three of the four "failures"
in rounds 1–2 were the harness, and pricing them as model failures would have
retired three capable contestants and kept the bug.**

Opus 5's answer was itself truncated by the cap mid-sentence in Q2.3, so its
Q3–Q5 are missing — and round 3 is where Q3, Q4 and the second Q5 finally
arrived, from kimi-k3, glm-5.3 and qwen3.8-max.

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

## Q3 — which signals are admissible (round 3)

All three round-3 answers rank the same way, and all three give the same reason
for admitting anything at all: these are constraints the **asker** supplied, so
none of them asserts anything about the world and none can make a false answer
true.

Ranked, best first:

1. **The question's own next relation** — already built (tier 3). kimi's framing
   is the one to keep: *"it is not a popularity judgment at all — it is a type
   constraint"*. "When was X born" presupposes an item with a birth-date claim.
2. **Co-mentioned entities, as a DETERMINISTIC tier before any prior fires.**
   glm: "Jim Haslam's son" — if candidate B is the object of a `child` claim
   from candidate A, the question structure itself disambiguates. **This is a
   graph check, not a prior**, and it belongs ahead of the ledger. The lab does
   not have it; it is the most concrete buildable thing in round 3.
3. **The asker's own history, above the global tally.** What *this* asker meant
   before is evidence about what *this* asker means, uncontaminated by other
   people's senses. The lab already binds a same-asker choice at n=1, so this is
   a confirmation — but the ordering is a correction: per-asker should rank
   *above* the population tally, not beside it.
4. **Session context** — the relation profile of what has already verified this
   session. A session whose verified chains are about chemistry makes Marie
   Curie the physicist likelier than the ferry. Same "the question decides"
   trick, generalised from one relation to the session's relation set.
5. **The global asker tally** — admissible, but *last*, because it is the one
   that most resembles a popularity ranking.

**Refused, unanimously across all six answering models**: sitelinks, pageviews,
corpus frequency, and fact-count-per-item ("a proxy for notability"). And one
the lab had not thought to exclude:

> **The emitter's own preference.** "The emitter proposes; letting its preference
> pick among ambiguous items gives the untrusted component a second, unaudited
> vote. The emitter already had its say by emitting the program." — kimi-k3

That is invariant 1 stated as a rule about *referents*, and it closes a door
nobody had noticed was ajar.

## Q4 — how it stays honest (round 3)

The disclosure requirement all three converge on: the spoken answer carries the
signal class, the counts and the margin (`how: prior (asker history, n of m,
margin 2.3:1)`), plus the rejected candidates — the same list a clarify would
have shown. Then kimi adds the part that is a design requirement rather than a
log format:

> A prior that decides must make **disagreeing cheaper than the ask it
> replaced** — a one-word "not who I meant?" that is logged, counts as a
> negative observation for the picked item and a positive one for the
> correction, and re-verifies from the new item.

The falsifying measurement, on which all three agree in shape: shadow mode
first, then an interleaved arm with a deterministic per-question hash so reruns
reproduce; hold the four tiers, `τ_vm`, the VM, the fetch and the refusal format
fixed, and vary only the post-tier disposition. Primary metric is **wrong
answers spoken, which must stay 0 in both arms**. Two switch-off numbers worth
writing down:

- **One** challenged-wrong prior decision that reached a spoken, verified answer
  demotes the prior to shadow mode the same day, with a pinned regression test.
- **Conversion** = Δcorrect / Δasks-decided. If it falls below the rate at which
  askers answer their own clarify prompts — *derivable from the existing clarify
  logs, not assumed* — then the prior is worse than asking, and off.

And kimi's cheap dignity check: run a **random-picker arm** in shadow over the
same candidates. If the prior's challenge rate is not clearly below random's,
the prior is doing nothing and should be removed for complexity alone.

## Q5, again — and this one is worse

Opus 5's Q5 was the multi-valued-relation trap (above), which is now fixed.
kimi and glm independently found a different one, and it is more dangerous
because **the kill line stays at 0 the entire time it is happening**:

> **Self-confirmation.** The prior decides; the asker mostly accepts, because
> challenging costs effort and the answer is verified and plausible — it passed
> the VM. Each unchallenged acceptance feeds the estimator, the margin grows,
> the threshold gets easier, the ask rate collapses — and the system has drifted
> from "verifier with a tie-breaker" to **"recommender with a verifier's coat"**
> without any single decision being wrong. The system speaks only verified
> truths about the wrong Marie Curie.

The James Young case is exactly where this bites: a wrong item that verifies
cleanly is the case an asker is *least* likely to challenge, because the answer
looks certified. And the bias is invisible in the challenge rate, because the
unchallenged mass is the problem.

The lab is half-protected by accident: the ledger records **only explicit
clarify choices**, never auto-picks, so the prior cannot feed itself directly.
Both models say that is the necessary mitigation and explicitly *not* the
instrument. Two instruments, both nearly free:

- **The decoy audit** (kimi). On every Nth prior-decided question, ask anyway —
  show the clarify choices with the prior's pick included but not indicated. If
  blind askers pick it at no better than its share of the list, the
  "acceptances" were inertia, not agreement. The gap between unchallenged
  acceptance (near 100%, expected) and blind re-ask agreement is the early
  warning. No new infrastructure: the clarify record and the logging exist; the
  decoy is a scheduling flag.
- **The silent counterfactual arm** (glm). Walk the non-chosen candidates too
  and log **multi-verify events** — questions where two or more candidates
  verified cleanly. If multi-verify on prior-decided labels trends up while the
  clarify rate trends down, the prior is suppressing the asks that would have
  corrected it.

That second one is Opus 5's **rival check** arriving from a different direction
and repurposed as a monitor rather than a recovery path. Two models, two rounds,
two motivations, one mechanism — which is the strongest argument in the whole
competition for building it.

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
   *(Done, 2026-09-14, `6b73be2`.)*
8. **Co-mentioned entities as a deterministic tier ahead of the ledger** — the
   question's own graph structure, not a prior.
9. **The emitter gets no vote on referents.** It proposed; that was its say.
10. **The decoy audit and the multi-verify monitor**, before the prior is
    allowed to decide anything in production.

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

## Still not answered

Opus 5's own Q3 and Q4 were cut off by the token cap and deepseek-v4-pro never
answered at any setting. The
**rival check** it proposed in Q1.4 is the most interesting unexplored idea and
belongs in the next pack: run the same chain program against *each* surviving
candidate — if exactly one verifies, the prior decided nothing; if several verify
and agree on the final object, the answer is invariant to the ambiguity and can
be spoken with **zero** prior involvement. At 0.7 ms per VM verification and
1.69 MB per session (both measured, `exp_r20`), four candidates cost 2.8 ms. That
is a free recovery path for refusals that needs no prior at all, and the lab does
not have it.
