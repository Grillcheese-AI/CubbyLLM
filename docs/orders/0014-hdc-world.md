# 0014 — The hdc set's 4.3M Wikidata facts as a world
status: done
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("check if we use those; we need to add them to the
worlds") · hypothesis: H-G6 (the wiki world), H-F2 · **suggested after 0012**

## Why
The local `hdc` folder (path in `CLAUDE.local.md`) is nace-ai's `hypernet-scaling-law-data` (Apache-2.0): Wikidata5M
facts as (subject Q-id, property, object Q-id) with question prompts, in OOD, scaling-law and preservation splits. It is
used today only to **build talk SFT** (`standin/data/build_hdc_sft.py`) and in experiments (`exp_m3_*`,
`exp_r18_hdc_rephrased`, the plan-verify tests). It is **not a world**: the served wiki world is wikikg's 365,923 triples.

Across its files, the triplet columns hold **4,294,788 unique triples**: 551 relations, 3,489,229 entities, counted
2026-10-22. That is ~12× the current world. The scaling-law files' `facts` column adds multi-hop facts as template
sentences, which `build_hdc_sft` already parses back through `relation_template_mapping.csv`.

- **What it brings:** Cubby's store grows from ~0.37M to ~4.6M facts, each with its Wikidata ids and source, answered by
  lookup. And every book fact (0012) has a Wikidata line to be corroborated against.
- **Advantage over conventional models:** a model trained on these facts holds them blurred in weights; a store holds
  each one exactly, cited, and replaceable when Wikidata changes.
- **In plain words:** Cubby gets millions more facts it can look up and cite, instead of half-remembering them.

## Build
- **The triples** (`standin/data/hdc_world.py`, new, < 300 lines): read every file's triplet columns and parse the
  `facts` sentences back to triples, as `build_hdc_sft` does. Deduplicate.
- **Labels:**
  - Each Q-id gets its exact English label, from `standin/data/out/hdc_names.tsv.gz` (alias → Q-id → name). Each
    property gets its label from `wikidata_properties_en_fr.json`. Add FR labels where they exist.
  - Report coverage. Resolve the rest by the smallest route; any dump download needs Nick's yes first.
  - Drop entities with no usable label (Wikimedia lists, disambiguation pages), as `build_hdc_sft` does.
- **A disk-backed world:**
  - SQLite rows (subject, relation, object, Q-ids, provenance `Wikidata5M via hypernet-scaling-law-data, snapshot
    ~2019`), not in-memory dicts.
  - Lookup-first by (subject label, alias or Q-id; relation).
  - The name index grows to ~3.5M names: keep it lazy or disk-backed, and measure.
- **Mount:** `serve_api --world hdc` beside wikikg. In the ask loop, the stores come first, then the books (0012), then
  live Wikidata, then the web.
- **Tiers:**
  - A Wikidata5M fact is one source, "Wikidata". It counts as Wikidata for the corroboration rule: book + Wikidata
    → corroborated.
  - **Time-sensitive relations** (heads of state and government, officeholders, populations, team memberships,
    CEOs, and the like; pick them from the 551 and list them) carry the snapshot's date. They are spoken "as of
    2019", or refreshed from live Wikidata on ask, and never as current without a refresh.
  - Against wikikg: deduplicate by label or Q-id. Report overlaps and contradictions; contradictions are held, with
    both kept.

## Acceptance (pre-registered)
- **Report counts:** triples loaded, entities with a usable English label (≥ 90%, or report why not), relations
  mapped.
- **Lookup:** p95 ≤ **50 ms** per (subject, relation) on Nick's machine; the mounted world ≤ **4 GB** of memory.
- **Bench:**
  - 200 one-hop questions from `eval_ood_scaling_law.pq` (held-out domains) through the ask loop with the world
    mounted: spoken-correct reported, **0 wrong spoken**;
  - 50 time-sensitive questions: each spoken with its "as of" or after a live refresh, never flat.
  - The tense decides the time asked (order 0012's question shape): "Who was … in 2019?" may be answered
    from the snapshot; "Who is …?" on a time-sensitive relation needs a live refresh or an "as of".
- Off = unchanged; full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. No model. The set's files stay off-repo; only code and counts are committed.

## Out of scope
- Training on it (the talk mix already samples it). Writing it into the trunk's scene (0009).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:** `77d466f` (the world and its store), `cac0cca` (a hop asked by an alias; the bench), `71aa0ec`
  (profile_ask without a head), and this report's commit (the emitter asks a label by its own name).
- **Files:**
  - `standin/data/hdc_world.py`: triplet rows and fact sentences parsed back to triples, deduplicated, labelled,
    written to SQLite; `TIME_SENSITIVE`; the comparison against wikikg.
  - `standin/data/fetch_hdc_labels.py`: the missing English / French labels from Wikidata's API (Nick's route),
    cached, 50 Q-ids a request.
  - `standin/hdc_store.py`: `HdcWorld` (read-only, lookup by label, alias or Q-id, an LRU cache) and `Stores` (the
    wiki world first, written to; the hdc world behind it; `held_by` names the world a fact comes from).
  - `standin/ask.py` (`_snapshot`: a time-sensitive fact said "as of 2019", another year refused);
    `standin/serve_api.py` (`--world hdc`); `standin/question_forms.py` (the emitter's label fix below).
  - `validation/exp_hdc_bench.py`.
  - Tests: `standin/tests/test_hdc_store.py` (4).
- **Tests:** label, alias and Q-id lookups; the merged store writes to the wiki world and names a hdc fact's
  snapshot; the loop says a changing value "as of 2019", answers the snapshot's own year and refuses another; a
  contradiction between the worlds is never said. Off (no `--world`) is the wiki world alone, as before.
- **Measured** ([world](../../validation/logs/hdc_world_2026-10-22.json), [bench](../../validation/logs/exp_hdc_bench_2026-10-22.json)):
  - **Counts: pass.**
    - 4,932,890 unique statements read; 3,384,594 facts loaded over 2,458,891 entities and 313 relations.
    - **Label coverage 0.989**: 2,454,437 of 2,481,268 Q-ids have an English label after 31,995 API requests;
      1,217,481 have a French one.
    - Dropped: 364,705 facts whose subject or object is an alias the name table cannot place; 36,931 entities
      without an English label; 437 relations without a label.
    - 17 time-sensitive relations hold 155,749 facts.
    - Against wikikg: 311 facts agree, 1,378 give another value for the same subject and relation (kept, both).
    - The world is 1.02 GB on disk and builds in 187 s.
  - **Lookup: pass.** p50 0.17 ms, **p95 0.31 ms** (line: 50 ms) over 1,000 subjects. Process memory with the
    world mounted: **0.9 GB** (line: 4 GB).
  - **One-hop bench: passes.** 200 questions from the held-out domains as the set writes them:
    - **177 spoken, 177 correct, 0 wrong**; 5 of them said "as of".
    - 23 refused: 7 retrieval exhausted, 6 ambiguous relation, 5 entity unresolved, 4 ambiguous hop, 1 no plan.
    - 88 drawn prompts were skipped as not one-hop (nested prompts; see below).
  - **Time-sensitive: passes.** 50 asked in the present: **0 said flat**, 21 said "as of 2019", 29 refused
    (14 entity unresolved, 13 not covered by the plan, 2 retrieval exhausted). The final run is on the emitter of
    order 0007's report (a label it cannot name back is bound as the label).
  - **The tense decides:** "Who was … in 2019?" is answered from the snapshot. Another year is refused
    (`time_mismatch`), and the present gets "as of" (test).
- **Not verified here:** a live refresh on ask (the API route exists for labels; a fact refresh is not wired, so a
  time-sensitive answer is always "as of 2019" or refused). Corroboration of a book fact by a hdc fact: the place
  and date relations the books state are mostly absent from Wikidata5M (no P19, P20, P119, P112, and no dates).
- **Found on the way:**
  - The first bench refused 141 of 200, and the cause was not the world. The set asks "What is the instance of
    X?"; since order 0012 the template emitter binds the question's own word ("instance"), and the wiki world
    holds a relation named "instance", so nothing translated it and the walk never reached "instance of". The
    emitter now binds the label when the word is the label without its own "of". 44 → 177 spoken, 0 wrong either
    way.
    - A store-side fix ("R of" serving "R") was tried and reverted: "capital of" reads the opposite way from
      "capital", and it would have answered "What is the capital of Strelsau?" with Ruritania.
  - The same collision remains for a label named otherwise: "Who is the owner of …?" (the wiki world's "owner";
    Wikidata's "owned by"), 4 of the 7 retrieval-exhausted questions. Fixing it means translating a held wording
    when the walk on it finds nothing, which is the loop's relation resolution, beyond this order.
  - A nested prompt ("the X of the Y of Z") was answered wrong in the smoke through the paraphrase tier's
    word-overlap tolerance at hop 0. The bench keeps one-hop prompts only (the 88 skipped). That tolerance is a
    risk for any multi-hop question over this world.
  - `profile_ask` crashed on a subject ending in "of" (an IndexError on the first time-sensitive question); fixed,
    with a test.
  - Some of the set's own prompts read badly ("What is the opposite of of Mitzvah?", "What Predecessor does … have?").
- **Questions:**
  1. Should the time-sensitive answers get a live refresh from Wikidata on ask, or is "as of 2019" the line for now?
  2. The 1,378 values that differ from wikikg are kept both. Should either world win on a relation, or is that
     order 0015's?

## Review (Cowork, 2026-10-22): done, passes
Checked against `hdc_world_2026-10-22.json` and `exp_hdc_bench_2026-10-22.json`: labels 0.989; lookup p95 0.31 ms;
0.9 GB mounted; one hop **177/200 spoken, 177 correct, 0 wrong**; time-sensitive 0 said flat (21 "as of 2019").
- The risk found on the way is the order's most important line: a nested prompt was answered wrong through the
  paraphrase tier's word-overlap tolerance at hop 0. The bench skipped the 88 nested prompts. They are order 0023,
  with the relation-wording miss ("owner" / "owned by") and question 1.
- Question 1: "as of 2019" is the line now. A live refresh reuses order 0007's lookup (Wikidata, cached, the answer
  dated by the fetch): order 0023, part C.
- Question 2: neither world wins blindly. 1,378 differing values against 311 agreeing is a signal about the worlds,
  not noise: order 0015 samples them, sorts them (a coarser or finer value, an older one, a label, an error), and
  sets a rule per kind; until then a disagreement is held, never said.
