# CoT harvest schema — verified-supervision records

**Status:** canonical, v1 (2026-08-08). Emitted by `validation/exp_m3_cot_pipeline.py`
as `validation/logs/cot_harvest{tag}.jsonl` (one JSON object per eval question).
Design settled by a 3-model literature panel (Q23; see the research agenda doc):
**store the trace tree, not the accepted path** — every downstream consumer reads
from the tree.

## Why every field exists (the three-times-paying rule)

The same record serves three consumers, so under-harvesting is irreversible:

1. **Calibration (Q21):** rejected candidates and failed hops are *organic
   near-miss negatives* — the negative class a clean store cannot produce.
2. **Routing/absence discrimination (Q22):** retrieval scores and failure
   patterns are the training labels for wrong-partition-vs-fact-absent.
3. **Fine-tuning (Q23):** verified records are STaR/RFT data; failed-then-
   repaired pairs are DPO preference pairs (near-miss failures carry the most
   contrastive signal — mask bad steps rather than discard trajectories, per
   SRFT: ~61% of discarded trajectories are informative).

## Record layout

| Field | Type | Notes / consumer |
|---|---|---|
| `question` | str | |
| `parsed` | bool | grammar coverage metric; unparsed records still logged |
| `n_hop` | int | difficulty metadata (curriculum) |
| `answer_class` | str\|null | grammar v2's leading class noun (future verify signal) |
| `program_source` | str\|null | the CubeLang program, when built |
| `verified` | bool | the claim gate |
| `answer` | str\|null | claimed answer (only meaningful when `verified`) |
| `gold_answer` | str | dataset answer |
| `correct` | bool | normalized exact match vs gold |
| `trace[]` | list | **per hop**: `query`, `fact`, `triple{obj,rel,subj}`, `ret_score`, `symbol`, `similarity`, `hop_verified` — per-step verifier outcomes are free process-supervision (PRM) data |
| `candidates_topk[]` | list per hop | the `(score, text)` lists retrieval returned — runners-up are free negatives; discarding them forces artificial re-mining later |
| `reason` | str\|null | failure code: `unparseable` / `retrieval_exhausted` / `vm_verify_failed` — the rejection *reason* is dense feedback (SDPO), and the aggregate distribution diagnoses which component is the bottleneck |
| `repairs_used` | int | |
| `banned_facts[]` | list | failed-then-repaired pairs: each rejected fact with its replacement — the error-localized DPO pair (record the *delta*, not just the outcome) |
| `hops_verified_before_failure` | int | graded outcome — enables graded-reward training without re-harvesting |
| `taus` | obj | `tau_vm`, `tau_ret` in force for this record |
| `store_snapshot_hash` | str | sha256 over sorted store fact texts — **the invalidation key**: when the store changes, stale tuples are found and evicted by hash, never silently retrained on |
| `table_path` | str | encoder provenance |
| `git_rev` | str | pipeline version |
| `timestamp` | str | |

## Rules

- **Never merge harvested negatives into SFT data** — they are a separate
  calibration/test split (Q21 assignment: planted faults calibrate, organic
  harvested confusables test).
- **Provenance is load-bearing, not bookkeeping**: a record without
  `store_snapshot_hash` + `git_rev` cannot be safely reused after any store or
  verifier change.
- **Schema is versioned**: breaking changes bump a `schema_version` field
  (absent = v1) and get a new section here.
- Curriculum consumers order by (`n_hop`, `repairs_used`) — empirical
  difficulty beats heuristic proxies; filtering (verified-only, error-tagged)
  is the bigger lever than ordering.

## Future fields (reserved, not yet emitted)

- `frame_id` — which grammar frame parsed the question (needs planner support).
- `counterfactuals[]` — which planted faults this verified chain rejects
  (hard negatives for the verifier's successor; log at calibration time).
- `world_ids[]` / routing scores — when the world-backed retriever lands (Q22
  labels).
- `rationalized` — flag for STaR-style backward-derived programs on unsolved
  questions (answer injected, program derived, hint stripped).
