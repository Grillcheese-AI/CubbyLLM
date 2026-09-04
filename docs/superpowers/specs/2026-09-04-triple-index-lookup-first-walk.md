# Triple index + lookup-first walk — design (2026-09-04)

**Status:** built same day; VM-verified read on the 800-question harvest recorded below.

## The finding it answers

The chain walk (`cubbyllm/reasoning/pipeline._walk`) accepts a fact with an **exact** test on the parsed triple — the
tail string at hop 0, relation + subject afterwards — but its candidates came from **cosine top-k above `tau_ret`**.
`validation/exp_m4_triple_lookup.py` (log `validation/logs/exp_m4_triple_lookup.log`) ran the same acceptance test
as an index lookup on the harvest's own store:

| per hop (1,126 hops over 800 questions, gold-chain entities) | serves the hop |
|---|---|
| cosine top-3 above tau (the live operating point) | 78.7% |
| cosine top-50, any score (the search ceiling) | 83.8% |
| (relation, subject) lookup | 83.8% |
| lookup ambiguous (>1 candidate) | 0.8% |

End to end the index alone reaches 581/800 (72.6%) against the harvest's 517 verified (64.6%): 64 of the 246
`retrieval_exhausted` walks recovered. The other 182 are the **planner** rejecting the gold chain's own fact (chain
segmentation of long relation phrases; non-template and typo facts in the pool) — a grammar thread, not retrieval.

## Design

- **`cubbyllm/reasoning/index.py::TripleIndex`** (WIRED). `add(fact)` parses with `planner.parse_fact`; template facts are
  indexed three ways — by `normalize("rel of subj")` (hop 0), by `normalize(subj)` (hops ≥ 1), by `normalize(obj)`
  (the object side, for a bidirectional walk later). Non-template facts are counted, not indexed: they stay reachable
  by search. `hop(plan, hop, entity)` returns every indexed fact that `planner.accepts` for that hop, insertion order.
- **`planner.accepts`** is the ONE acceptance test (moved out of the pipeline; `pipeline._accept` stays as an alias for
  the validation scripts). A lookup hit is accepted by construction.
- **`pipeline._walk(…, lookup=None)`**: at every hop the lookup runs first — no threshold, no k, `ret_score` 1.0,
  `HopTrace.source = "lookup"`. Search (cosine + `tau_ret`, `source = "search"`) runs only for a hop the index
  misses. Ambiguity is broken by the search's own ranking of the hop query, then by fact text (deterministic). The
  repair loop is unchanged: a fact the VM rejected is banned, and the ban removes it from the lookup too, so the next
  candidate is tried on the retry.
- **`standin/worlds.py::FactStore`** keeps a `TripleIndex` maintained at `add()` — a fact learned at turn N is looked up
  at turn N+1 — and exposes `lookup(plan, hop, entity)`. `ReasoningCortex.walk_facts` passes
  `getattr(retriever, "lookup", None)`; a bare retriever keeps today's behaviour. The walk trace records
  `walk_sources` per hop.
- **Harvest runner** `validation/exp_m3_cot_pipeline.py --lookup` builds the index over the eval store; the harvest's
  `trace[].source` field is added to `docs/schemas/cot-harvest-schema.md`.

## What cosine keeps

Paraphrase (questions the grammar does not parse), world routing (`route_world`), the flat fallback
(`gather_facts`), the tie-break above, and the game world. Object-side retrieval and the `tau_ret` 0.50 experiment
from the GoT build order become moot for template facts.

## Kill criterion

The VM-verified lookup arm must not lose a single verified chain against the `_rb1` baseline (517 verified / 513
correct) and must recover most of the 64 walks the offline count found. Anything else means the index and the
acceptance test disagree somewhere.

## Read (VM-verified, `validation/exp_m3_cot_pipeline.py --lookup --no-counterfactuals --tag _lookup`)

`validation/logs/exp_m3_cot_pipeline_lookup.{json,log}` + `cot_harvest_lookup.jsonl`, against `_rb1` (the same
runner, search only):

| | rb1 (search) | lookup-first |
|---|---|---|
| verified chains | 517 | **568** |
| claimed correct | 513 | **564** |
| claimed-answer precision | 0.9923 | 0.9930 |
| verified coverage | 64.6% | **71.0%** |
| `retrieval_exhausted` | 246 | 195 |
| per hop: 1 / 2 / 3 | 0.865 / 0.566 / 0.220 | 0.917 / 0.648 / 0.283 |
| hops served by the index / by search | – | 937 / 0 |

**+51 verified, all 51 correct, 0 lost, no answer changed among the chains both runs verified.** Every hop the walk
took came from the index — search never supplied a hop the index lacked, as the offline ceiling predicted. The offline
frontier count (581) sits 13 above the pipeline because the pipeline takes ONE candidate per hop with a repair budget
of one: an ambiguous hop resolved to the wrong branch dies downstream — the frontier walk's case. The other 195
exhausted walks are the planner's (the gold fact is rejected by the plan's own acceptance test). The verdict block
(CoT beats retrieval-only per hop, precision ≥ 0.90, zero sub-tau verified, control below tau_vm) is PASS. Mean CoT
wall 45.9 → 83.9 ms/question, not comparable this run: a 14 s stall at questions 500–525 and 10% more VM-verified
programs; re-time in the next full run.
