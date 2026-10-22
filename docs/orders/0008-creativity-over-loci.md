# 0008 — Creativity over known facts: mix them into hypotheses, test them, keep what is confirmed; art stays free
status: proposed
written: 2026-10-22 by Cowork · approved: — · hypothesis: H-K10 with H-A13 (skill library) and `standin/hypothesis.py`

## Why
Nick, 2026-10-22: "we need to ensure the creativity cortex is still working: while Cubby knows the facts it can
test them and create new ones by mixing facts together if confirmed (a hypothesis as an example, art as another)."
Orders 0005–0007 make Cubby *know*: a fact is written once, recalled through the loci path, and looked up once on
a miss. Nothing yet makes it *make* something from what it knows. The pieces exist apart:
- `standin/hypothesis.py`: frame a claim → name the test → run it → the result is the verdict. Its verifiers are
  VM, RUNNER, WORLD and ASK. Its rule: an open hypothesis may be said as a guess, never stored as a fact.
- The skill library (`cubbyllm/reasoning/skills.py`, H-A13): composition rules, adopted only with zero
  counterexamples.
- `standin/verse.py`: Cubby-man writes his own programs to join known facts into certified new ones.
- The tiers grounded / derived / counterfactual / invented (`docs/WORK_ORDERS.md`, the archive's CausalGraph).

This order joins them on top of the loci path, and guards the line between a confirmed fact, a labelled guess,
and art.

- **What it brings:** Cubby proposes new facts by mixing ones it knows, tests each, and keeps only the
  confirmed ones, recalled next time like any other. It can still make art freely, which is never mistaken for a
  claim.
- **Advantage over conventional models:** they mix facts too, but cannot tell a guess from knowledge, so the
  mix comes out as confident fiction. Here every mix carries its status, and only a verdict makes it a fact.
- **In plain words:** it can have ideas of its own, check them, and remember the ones that turn out true,
  without ever passing a guess off as fact.

## Build
- **A mixer** (`standin/mixer.py`, new, < 300 lines; off by default). From the facts the store holds about one or
  two entities, as the loci reader recalls them, it proposes candidate claims the store does not hold:
  1. **Composition.** `A r1 B` and `B r2 C` give `A (r1∘r2) C`. When the skill library has the rule, the VM
     certifies it (derived). When it does not, the claim is OPEN.
  2. **Analogy.** `A r B` with `C` sharing `A`'s kind (a shared relation value) gives `C r ?` as an OPEN question
     for a source.

  Each candidate becomes a `Hypothesis` with its verifier: VM for a composition under a known rule, ASK (order
  0007's Wikidata lookup) for an empirical claim. A confirmed hypothesis goes into the store as **derived**, with
  the test as its provenance, and is then recalled through the loci path like any fact. A refused one is recorded
  and never stored. An open one may be spoken only in its guess form ("I wonder if…"), never as a fact.
- **Art** (a creative ask): it may take facts as material from the loci reader. Its output is never written to
  the store, never checked by the fact gate, and never refused for being unverified. It carries the `invented`
  tier in the loop's record.

## Acceptance (pre-registered)
- **The bench** (`validation/bench_mixer.py`): a seed store of 200 facts about 50 entities. The mixer proposes ≥ 100
  candidates absent from the store. Report how many it proposed, how many were confirmed or refused per verifier,
  and how many stay open.
- **Hard lines** (these are the pass):
  - **0** open or refused candidates stored as facts;
  - **0** candidates spoken as fact before a verdict;
  - confirmed candidates recalled next time through the loci path at ≥ **0.80**, with 0 source calls;
  - creative asks produce the same routing and replies with the mixer on and off;
  - no art output reaches the store.
- **Kill:** fewer than 10% of candidates reach a verdict, either way. Then the mixer proposes untestable
  claims, and its proposal rule changes.
- Tests: the mixer off leaves the loop unchanged; the `Hypothesis` rule (open ≠ fact) holds through the store
  gate; full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. Wikidata only for ASK, cached. No external LLM on the path.

## Out of scope
- Consolidating derived facts into weights (the sleep cycle). The DecisionOracle's personality binding (N
  futures by binding): its own order if the mixer needs diversity. Art quality.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
