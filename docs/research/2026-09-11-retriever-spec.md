# The retriever — what the measurements say it has to be

**2026-09-10** · derived from `exp_r1_gate_diagnostics`, `exp_r2c_split_probe`, `exp_r4_index_walk`.
Zero GPU. Every number traces to a captured log.

## Where things actually stand

`cubbyllm/reasoning/index.py` (`TripleIndex`) is the **only** retrieval component that
exists as part of the package — exact normalized lookup, no threshold, no model. The
cosine retriever is not a package component: `make_retriever` lives inside
`validation/exp_m3_cot_pipeline.py` and is rebuilt from scratch per run, a 20-line dense
numpy matmul over the whole store. So retrieval-as-search exists only as experiment
scaffolding, and every measured coverage number in the project rests on it.

## The measurement that specifies the job

`exp_r4_index_walk` rebuilds a `TripleIndex` from the observed store (953 indexed facts,
a 78% subset of the real 1,241) and runs an index-only walk over the 195 failures that
survive lookup-first:

| | |
|---|---:|
| hop-0 tail lookup produces **no seed at all** | **118 / 195 (61%)** |
| hop-0 lookup produces a seed | 77 / 195 |
| — of those, forward walk reaches gold | 23 |
| — of those, bidirectional reaches gold | 25 |
| — **newly recovered by the backward edge** | **2** |
| gold answer **is** an entity in the index | **100 / 195 (51%)** |

**The walk dies at the seed, not in the graph.** Half of these questions have their
answer sitting in a 78%-complete index and no way in.

## Three consequences

**1. `by_object` is not the free win it looks like.** `TripleIndex.by_object()` is built
and documented as "the backward entry for a bidirectional walk; unused by the forward
walk today", and the decomposition found backward retrieval clearing tau on 43 surviving
failures. Wiring it in recovers **2**. The 43 came from *fuzzy cosine matching on object
text* — that is the retriever, not the index. Keep `by_object`; it costs nothing and
pays once fuzzy seeding lands. Do not book it as a fix now.

**2. Hop-0 entry is the product.** 61% of remaining failures never produce a seed.
Later-hop traversal is comparatively healthy and the index already serves it (83.8% of
hops). A retriever optimised for chain traversal solves a problem that is not the
bottleneck.

**3. Exact match is precisely what is failing.** The hop-0 lookup is an exact normalized
match on `"rel of subj"`. Any difference between the question's surface form and the
fact's kills it, and `exp_r2c_split_probe` showed the parser miscounts hops on 92
questions with no cheap grammar fix available (best candidate: +12 of 92, net +8).
Those two failures compound at the same point.

## The spec

- **Primary job: seed the walk at hop 0 from a question whose surface form differs from
  the fact's.** Not traversal. Not re-ranking. Entry.
- **It must degrade gracefully rather than threshold.** The lookup path has no threshold
  and that is why it beat cosine (0.710 vs 0.646, +51 verified, 0 lost). A retriever that
  reintroduces a hard tau at hop 0 reintroduces the `threshold_bound` class that
  lookup-first just eliminated (34 of 36 already recovered).
- **It should return a ranked candidate set, and that set must be logged.**
  `candidates_topk` is empty for 152 of 246 failures, so the diagnostic that would tell
  you why retrieval failed is unavailable for 62% of the cases where it failed.
- **Bidirectional from day one, since the edge already exists** — but expect it to pay
  only in combination with fuzzy seeding.

## What to measure when it exists

Baseline is **0.710** (lookup-first, measured), not 0.646 and not the 0.83 an earlier
draft assumed. The ceilings, both upper bounds:

| | |
|---|---:|
| current | 0.710 |
| + everything backward-reachable by cosine | 0.764 |
| + every surviving failure whose gold is in the store | 0.839 |

The gap between 0.710 and 0.839 is the retriever's entire addressable space on this
eval. If a built retriever does not move coverage meaningfully inside that band, the
problem is not retrieval and the emitter conversation reopens.
