# 0014 — The hdc set's 4.3M Wikidata facts as a world
status: open
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
- Off = unchanged; full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. No model. The set's files stay off-repo; only code and counts are committed.

## Out of scope
- Training on it (the talk mix already samples it). Writing it into the trunk's scene (0009).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
