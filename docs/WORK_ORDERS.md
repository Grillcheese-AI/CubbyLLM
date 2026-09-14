# Work orders

Specs for work that is **argued for and not built**. Each one is written to be picked up cold by
a coding session that was not present for the argument: what to build, where, why it is shaped
that way, what would prove it wrong, and the measured numbers the decision rests on.

A work order is not a ticket. If the argument for it turns out to be wrong while you are
building it, **stop and say so** — the reasoning is the deliverable, and a work order that
survives contact with the code only because nobody checked is worse than none.

---

## Before you touch anything

**The kill line: 0 wrong answers spoken.** Everything below is subordinate to it. A refusal is a
result; a wrong answer is a defect. When a change could trade a refusal for a guess, it is not a
trade — it is the defect.

**The rule that keeps being the one that matters:** the model proposes, the host disposes, the
VM decides. An untrusted component gets exactly one say.

**Standing constraints**

- No external LLM on the serving path, ever. External models are for building datasets and for
  competitions, nothing else. The served model never calls out.
- The OpenRouter key lives in a gitignored `validation/.env`. It never appears in a log, an
  error message, or a commit.
- **No local drive paths in committed files** — this one included. Describe locations
  generically ("the repo root", "a checkout beside this one", "the datasets volume").
- Never kill python processes broadly. Filter by command line: a serve instance and sometimes a
  long competition run are alive at the same time, and a blanket kill has already destroyed one
  paid run.

**Running things**

- Tests: `python -m pytest -q standin/tests validation` from the repo root. 359 green as of
  2026-09-14; any work order that lands red is not done.
- The live loop: `standin/serve_api.py` with `--ask` (and `--news` for the date source), then
  `POST /ask {text, item, asker}`. `/loop/events` is the event log and is usually the fastest way
  to find out what actually happened — several bugs this month were diagnosed from it in one
  call after an hour of guessing.

**How to write a test here.** The docstring carries the incident: what was observed, when, and
why the rule is the rule. `test_a_grouped_number_is_the_same_number` is the model. A test whose
docstring only restates its assertions is not pulling its weight.

---

## WO-1 — The co-mention tier

**Status:** not started. **Cost:** small. **Do this one first.**

### What

A deterministic resolution tier that uses the question's own graph structure to disambiguate,
placed **ahead of the choice ledger** in `standin/sources.py::resolve()`.

When a question mentions two entities and the store (or the source) holds a claim linking
candidate B to candidate A, that link resolves the ambiguity. "Jim Haslam's son" — if B is the
object of a `child` claim whose subject is A, the question has disambiguated itself.

### Why it is shaped this way

This is **a graph check, not a prior**. It reads a relation that exists between two things the
asker named; it consults no history, no tally, no popularity. That is what lets it sit ahead of
everything else: it can only fire when the question already contains the answer to the ambiguity.

From the neutral-prior competition (`docs/research/2026-09-14-neutral-scoring-competition.md`,
glm-5.3, Q3): *"This is a graph check, not a prior... Deterministic, so it should be promoted to
a tier-5 rule before any prior fires."*

### Where

- `standin/sources.py::resolve()` — the tier cascade. Current order: linked (via the fact that
  reached it) → one exact hit → the question's relations narrowed in order (kind-guarded) →
  label over alias → asker history → ask the asker. The new tier goes **after "one exact hit"
  and before the ordered narrowing**.
- The candidate view rule from the competition is binding here: the selector may read labels,
  aliases, descriptions, `instance of` classes, and **which relations a candidate has** — never
  **the objects of those relations**. Property *presence* is type evidence; property *value* is
  truth. A co-mention check reads a link between two named candidates, which is presence, not
  value. Keep it that way and say so in the docstring.

### Acceptance

- A test in `standin/tests/test_sources.py` with two same-named candidates where exactly one is
  linked to the other entity in the question, and the tier resolves it with
  `last["how"] == "co-mention: <relation>"`.
- A test where **both** candidates are linked, which must still refuse — a tie is not an order.
- A test proving the tier never consults `self.choices`: run it with a ledger stacked in favour
  of the wrong candidate and assert the co-mention wins.
- The benches still run with `use_choices=False` and are unaffected.

### What would prove it wrong

If it fires on a link the *source* invented rather than one the question named — i.e. if it
starts resolving "Marie Curie" by any edge at all rather than by an edge to another entity in
the same question. Then it is a popularity prior wearing a graph's clothes, and it must move
below the ledger or come out.

---

## WO-2 — The multi-verify monitor

**Status:** not started. **Cost:** small-to-medium. **Required before any prior decides in
production.**

### What

After a walk resolves an ambiguous referent, re-run **the same chain program** against each
surviving candidate and log the outcome. Do not change the answer — log it.

Three outcomes, all worth recording:

- exactly one candidate verifies → the prior decided nothing; this was tier 3 evaluated at full
  walk depth.
- several verify and **agree on the final object** → the answer is invariant to the ambiguity.
- several verify and **disagree** → this is the James Young class, and the only case where the
  prior actually decided anything.

### Why it is shaped this way

Two models reached this mechanism from opposite directions, which is the strongest signal in the
whole competition. Opus 5 proposed it as a **recovery path** (an ambiguity-invariant answer can
be spoken with zero prior involvement). glm-5.3 proposed it as a **monitor**: if the
multi-verify rate on prior-decided labels trends up while the clarify rate trends down, the
prior is suppressing the asks that would have corrected it.

Build the monitor first. The recovery path is a behaviour change and needs the monitor's numbers
before it can be justified.

### Cost, measured

0.7 ms per VM verification, 1.69 MB per session, 5,211 programs/s from 8 threads, p99 2.77 ms
(`validation/exp_r20_vm_per_user.py`, 2026-09-14). Four candidates cost ~2.8 ms serially in one
VM. Concurrent rival checks are **not** measured — if you run them concurrently, measure first.

### Where

- `cubbyllm/reasoning/pipeline.py` — the walk already has the program and the candidates.
- Log through `cubbyllm/reasoning/events.py` as its own event kind (`rival`), with the candidate
  set, which verified, and whether the verified ones agreed.
- It must be **off the answer path**: a rival check that fails, times out, or throws changes
  nothing about what is spoken.

### Acceptance

- Fixture with three candidates where one verifies → `rival` event says `decided: none`.
- Fixture where two verify and agree → `agree`, and the spoken answer is byte-identical to
  before the change.
- Fixture where two verify and disagree → `disagree`.
- A timing test asserting the added wall time is under a stated budget on the fixture set.

---

## WO-3 — The decoy audit

**Status:** not started. **Cost:** medium. **Required before any prior decides in production.**

### What

On a fixed fraction of prior-decided questions (every Nth, with N logged), **ask anyway** —
present the clarify choices with the prior's pick included but not indicated — and record what
the asker chose blind.

The signal is the gap between **unchallenged acceptance rate** (expected near 100%) and **blind
re-ask agreement rate**. If blind askers pick the prior's item at no better than its share of
the list, the acceptances were inertia, not agreement.

### Why this exists at all

This is the instrument for the failure mode the competition surfaced in round 3, and it is worth
restating because it is the most dangerous thing on this page:

> **Self-confirmation.** The prior decides. The asker accepts, because challenging costs effort
> and the answer is verified and plausible — it passed the VM. Each unchallenged acceptance
> feeds the estimator, the margin grows, the threshold gets easier, the ask rate collapses — and
> the loop has drifted from *a verifier with a tie-breaker* to **a recommender with a verifier's
> coat**, without any single decision being wrong. It speaks only verified truths about the
> wrong Marie Curie.

**The kill line stays at 0 the entire time this is happening.** That is why it needs its own
instrument: no existing metric can see it. The bias is invisible in the challenge rate, because
the unchallenged mass is the problem.

The loop is currently half-protected by accident — the ledger in `standin/sources.py` records
**only explicit clarify choices**, never auto-picks, so the prior cannot feed itself directly.
Both models that raised this say that is the necessary mitigation and explicitly **not** the
instrument. Do not let the accident stand in for the audit, and do not remove the accident.

### Where

- The clarify path in `standin/ask.py` and the panel's clarify bar in
  `dashboard/control_panel.html` already exist. The decoy is a scheduling flag on that path, not
  new infrastructure.
- The ledger must record a decoy answer as an **ordinary explicit choice** (it is one), and the
  audit log must record separately that this question was a decoy, so the two can be compared
  later without contaminating the tally.

### Acceptance

- Deterministic selection: the same question at the same index is a decoy on a rerun (hash, not
  random).
- A test that the prior's pick is not marked, ordered first, or otherwise identifiable in the
  choices handed to the asker.
- A test that a decoy's answer updates the ledger exactly as a normal clarify does.
- A report script under `validation/` that prints acceptance rate vs blind agreement rate.

### The companion affordance

From the same round: *a prior that decides must make disagreeing cheaper than the ask it
replaced.* A one-word "not who I meant?" on a prior-decided answer, logged, counting as a
negative observation for the picked item and a positive one for the correction, and re-verifying
from the new item. Build it with the audit; it is the same plumbing.

---

## WO-4 — Date the preloaded world

**Status:** not started. **Cost:** large (a rebuild). **This is the biggest structural gap.**

### What

The preloaded wiki world carries **no time qualifiers at all**. Every one of its 552,297 facts
is undated until some live fetch happens to restate it.

Rebuild the world's source data so `P580` (start), `P582` (end) and `P585` (point in time) ride
beside each fact, the way `standin/sources.py::_qual_times` already produces them for live
fetches, and have `standin/data/wikikg.py::wiki_world()` populate `FactStore.times` on load.

### Why it matters now

As of `6b73be2` the walk can order multi-valued relations by the source's own dates — that is
the only thing permitted to break such a tie — and questions that name a year resolve correctly:

```
what was the population of quebec city in 2011?  ->  516622
...position held of bill haslam in 2015?         ->  Governor of Tennessee
```

But this only works for entities somebody has already asked about, because the dates arrive with
a live refetch. **The capability exists and the data does not reach it.** Every dated question
about an entity nobody has asked about still refuses.

The workaround that makes the above work — an ambiguous hop is `LEARNABLE`, so the loop refetches
the entity and walks again — is a workaround. Keep it, but do not mistake it for the fix: it
costs a network round trip per ambiguous entity and only ever covers the long tail one entity at
a time.

Measured density, so the size of the job is known: Bill Haslam 9 of 54 statements carry a time,
Quebec City 54 of 298, Marie Curie 37 of 430. Roughly 10–20%, so expect the `times` map to be
far smaller than the fact count.

### Where

- The export/build step that produces the triples the world loads.
- `standin/data/wikikg.py::wiki_world()` — populate `FactStore.times` from it.
- `standin/worlds.py::FactStore` already has the `times` field and needs no change.

### Acceptance

- `wiki_world()` returns a store whose `times` is non-empty, keyed exactly as
  `standin/ask.py::dated_facts` and `cubbyllm/reasoning/pipeline.py::_covering` read it
  (`" ".join(fact.split())`).
- The four live questions above answer correctly **without** any refetch — assert the fetch count
  is zero.
- Load time and memory measured before and after and written into the research doc. A world that
  answers dated questions and takes a minute longer to load is a trade, and the trade must be a
  number.

---

## WO-5 — The semantic hole in the grounding guard

**Status:** open problem, no agreed design. **Do not build blind.**

### What is wrong

`standin/ask.py::grounded_prose` checks that every capitalised word, every number, and every
"X is a ⟨kind⟩" head noun in a paraphrase occurs in the facts. It checks **vocabulary, not
semantics**, and that gap is real. Observed live, 2026-09-14, both passing the guard:

- facts `fact learned: 152; source: wikidata (152)` → *"referenced in 152 sources"*. It is 152
  facts from **one** source.
- facts `last answer: 574482` → *"the last answer was given 574482 units ago."*

Every name and number in both sentences occurs in the facts. The guard did exactly what it was
written to do.

It holds up on world facts only because those relations are ordinary English a small model has
seen a million times — "occupation", "date of birth", "capital". It fails on relations that are
not — "fact learned", "call", "provenance" — because the model confabulates what they mean.

### The mitigation already in place

The self-report path does not paraphrase at all (`standin/ask.py::introspect`). The counts are
the answer. That is correct for that path and is **not** a general fix: the same hole is open
anywhere the relation vocabulary is unusual, which now includes the news source's `headline` and
will include anything new.

### Why this is not a work order yet

The obvious fixes are all worse than the problem:

- Checking that each number is used with its own relation's wording is a grammar problem, and a
  grammar that is wrong 5% of the time becomes a refusal machine.
- Having a second model check the first is an external LLM on the serving path. Forbidden, and
  it would only move the trust problem.
- Restricting paraphrase to a whitelist of known-safe relations is honest and shrinks with every
  new source — which is the wrong direction for a system that is supposed to learn new sources.

**What to do first is measure, not build.** Take the paraphrases the loop has already produced
(they are in `/loop/events` as `paraphrase` records with `ok`, `draft` and `rejected`), and hand
count how many that PASSED the guard are semantically wrong. If the rate is near zero on world
facts and high on unusual vocabularies, the rule "do not paraphrase relations the adapter was
not trained on" is the whole fix and it is cheap. If it is non-zero on world facts, that is a
kill-line problem and it outranks everything else on this page.

Write the count up before proposing a design.

---

## Not on this page, and why

- **Rebuilding the ledger's estimator** (Wilson lower bound, event-decay, mass ceiling). The
  *shape* is right and is recorded in the competition doc. The constants are all marked as
  assumptions by their own author, and with 177 questions reaching a fetch there is not enough
  traffic to tell θ = 0.60 from θ = 0.70. Adopt the shape when there is traffic to measure it
  with; do not port the numbers.
- **GDELT as the pre-2026 date source.** The file route is confirmed reachable and un-rate-limited
  (`docs/research/2026-09-14-news-sources.md`); the query API is a hard 429. It is a real piece of
  work and it is blocked on nothing, but it adds a source rather than fixing a defect, so it sits
  below WO-1 through WO-4.
