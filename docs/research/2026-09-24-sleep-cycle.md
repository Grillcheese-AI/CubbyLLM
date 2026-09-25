# 2026-09-24 — the sleep cycle: the day's refusals become the night's curriculum

**The ask** (Nick, 24 Sep): self-evolution in the Cubby system, both halves. The two halves are the sleep cycle and
the skill library, and the sleep cycle comes first.

## The frame

Invariants 1 and 6 (the VM is the only truth gate; no self-play or self-judging) rule out the model improving
itself. What they leave is the version conventional models can't have:

- **the model proposes, the host disposes, and only a gate's verdict promotes a change;**
- **every change is on a ledger and can be retired, never deleted.**

Self-evolution happens at five speeds:

| Level | What changes | What gates it | Status |
|---|---|---|---|
| 1. Knowledge | store facts, certified chains | VM + source provenance | the ask loop learns live; **durable as of tonight** |
| 2. Policy | which proposer is tried first | reward-prediction error on gate rewards | striatum; **updated nightly as of tonight** |
| 3. Skills | composition rules; CubeLang helpers later | every past episode re-verifies | **rules built** ([skill library](2026-09-24-skill-library.md)); helpers not yet |
| 4. Adapters | LoRA / θ=f(c) | the H-A7 regression battery, per family | queues feed it; the round itself is v2 |
| 5. Structure | worlds, experts, width | same-budget A/B | later |

Why a sleep, concretely. By day the loop answered or refused and then forgot:

- the ask loop's history lived in memory;
- facts fetched from Wikidata lived in the running store and were gone at the next restart;
- a refusal was a status line.

## What was built

**`cubbyllm/reasoning/sleep.py`**
- STANDALONE, stdlib only. Run it as `python -m cubbyllm.reasoning.sleep --day <records> --out <dir> --night <id>`.
- It runs six phases over the day's records:

| Phase | What it does |
|---|---|
| collect | reads the ask loop's history `.jsonl`, or a search-and-learn bench `.json` |
| replay | each certified chain becomes a hippocampal episode (a plan candidate, through the same gate). An episode whose chain lost a store fact is retired with the facts named |
| consolidate | facts the gate **accepted** from a **trusted, non-latent** source go to an append-only `learned_facts.jsonl`, with relation, time, fetch time and the store's snapshot hash. A `{"retire": fact}` line retires one |
| triage | every refusal goes by its reason to the queue for the level that must change (table below). Items merge across nights, per night, so a recurring item says so |
| audit | a spoken answer that gold or the asker calls wrong is a **defect** (the kill line). Every outcome goes to the striatum: certified +1, refused 0, defect −5 |
| report | writes `nights/<id>/report.md` and `summary.json`: the refusal mix against the previous night, what the night did, the top queue items, and the shapes where the emitter is weakest |

- **Ledger** (`ledger.jsonl`): each night records the git revision and the input's SHA-256. Every action has a deterministic id, so re-running a night writes nothing new (checked on real data below).

**`standin/ask.py`**
- `AskLoop(history_path=…)`: every record is appended as one JSON line with an id, the time, the store's snapshot hash, and for each gated fact its full provenance (fetch time, pre-write snapshot, clash, relation, time).
- `load_learned(path)` puts the durable facts back at boot, each with its relation declared, its source as provenance, and its time.

**`standin/serve_api.py`**
- `--history` (default `standin/data/out/ask_history.jsonl`) and `--learned` (default `standin/data/out/sleep/learned_facts.jsonl`).
- Together they close the loop: day, then night, then the next day knows.

**Guards**
- `hippocampus` and `striatum` leave `KNOWN_UNWIRED`: the sleep cycle writes and updates them every night.
- The ask loop does not *read* either at question time yet. That is v2's first step.

**Tests** (12 tests, all green):
- `tests/reasoning/test_sleep.py`. Among its cases:
  - a scan: every reason literal the loop writes must be routed or an answered kind, so the routing table grows by decision;
  - a certified chain gives one episode however often the night runs;
  - only trusted, accepted facts become durable;
  - a re-run changes nothing;
  - queues merge across nights;
  - a defect costs its shape;
  - a profile answer is not a refusal;
  - a stale episode is retired, not deleted.
- `standin/tests/test_sleep_loop.py`: **day 1 fetches, the night consolidates, and day 2 answers the same question with the source emptied.** Same walk, same VM, zero source calls.

## Routing

Every reason the code can write is routed. The test fails on a new, unrouted reason.

| Reason | Queue | What must change |
|---|---|---|
| plan_does_not_cover_question, no_plan, emit_error, unparseable | emitter_curriculum | the emitter, for that question shape |
| unknown_relation | relation_wordings | the relation map / lexicon |
| ambiguous_relation, ambiguous_hop, ambiguous_entity, date_too_broad | clarify | the ask/clarify path |
| retrieval_exhausted, profile_empty | source_gaps | the store (a fetch) |
| entity_unresolved | entity_resolution | name → item resolution |
| answer_type_mismatch, vm_verify_failed, profile_unrecovered | vm_review | the walk or the thresholds |
| no_events_for_date, no_self_record | capability_gaps | a capability that doesn't exist yet |
| latent_only | latent_audit | nothing; evidence about the latent tier |
| (a spoken answer called wrong) | defects | retire the chain's facts, after review |

**Done automatically:** episodes, durable facts, striatum values, queue merges and the report. None of them changes
what may be spoken: an episode only proposes, and a durable fact is looked up and VM-verified like any other.

**Queued for a host decision:** everything in the queues.

## First run: three recorded days

I replayed the SimpleQA search-and-learn bench (530 questions, `exp_r11_search_learn_wikidata_{alias,lev6,lev9}.json`)
as three nights, in the order the levers were built. Output is in `standin/data/out/sleep_bench/`, which is gitignored.

| Night | Answered | Defects | not covered | unknown relation | retrieval exhausted | ambiguous |
|---|---:|---:|---:|---:|---:|---:|
| 09-11 (alias) | 2 | 1 | 464 | 60 | 3 | 1 |
| 09-12 (lev6) | 4 | 1 | 464 | 56 | 3 | 3 |
| 09-13 (lev9) | 6 | 0 | 338 | 130 | 40 | 16 |

Readings:

- **The report shows where refusals *moved*, not only how many there are.** Lever 9 took 126 questions past the
  emitter's coverage check, and most of them now stop at the relation map (`unknown_relation` went from 56 to 130).
  So the next lever is the lexicon, not the emitter.
- **The defect surfaced by itself.** The question was *In which district is "Kafr al-Awamid" located?*:
  - the answer given was *Az-Zabdani Subdistrict* where gold says *Al-Zabadani*, a subdistrict given for a district;
  - it was spoken on two nights and gone on the third;
  - the queue item keeps both nights and the chain fact, which is the retirement candidate.
- **The top recurring relation wordings are the lexicon's to-do list:** `year` (14), `municipality` (11),
  `episodes`, `first husband`, `second husband`, `married`, `first time`, `month and year`.
- **Durable facts:** 134 accepted Wikidata facts, plus 7 episodes. The 5 facts from the alias run were skipped because
  that run recorded no source: unknown provenance is not trusted.
- **Idempotent:** the ledger held 2,308 lines before and after re-running all three nights.
- **The striatum's weakest shapes are the curriculum's first candidates:**
  - `place|in|flat` is the only negative one (−0.016 over 30 tries, from the defect);
  - after it come the most-tried shapes that were refused every time: `any|what|chain` (216 tries), `name|who|flat` (114).

## v2: acting on the queues

Each step goes through its own gate.

1. **Read side.** The hippocampus becomes a proposer in the ask loop, and the striatum orders grammar, hippocampus and
   emitter by shape. That is when the episodes start paying back.
2. **`relation_wordings`.** Match each recurring wording to the source's property labels. A match is adopted only if its
   recorded examples now verify and the benches don't regress. Shadow first.
3. **`source_gaps`.** Fetch them overnight through the ask loop's own gate, then re-ask.
4. **`emitter_curriculum`.** Build gen-3 records from the store for the top shapes, run an adapter round (SFT or
   evolution strategies), apply the H-A7 gate, then shadow and promote.
5. **`defects`.** The chain facts become retirement candidates, reviewed before a `retire` line is written.
6. **VM replay.** On a new VM build, re-verify the episodes (the certification ledger's `vm_build_id`).
7. **The skill library** (level 3). Composition rules are built and mined every night (the `skills` phase,
   [write-up](2026-09-24-skill-library.md)); recurring sub-programs as CubeLang helpers are the next kind.

**Scheduling.** A nightly Windows Task Scheduler entry would run the module over `ask_history.jsonl` into
`standin/data/out/sleep`. It is not created, because it is persistent configuration and Nick's call.

## Kill criterion

Over five nights on a fixed battery, the refusal share must fall and defects must stay at 0.

If the queues grow while the battery doesn't improve, the loop is collecting, not learning. Stop and find which
queue's action isn't landing.
