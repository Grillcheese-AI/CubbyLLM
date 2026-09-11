# The retriever contract — 16 pins, mutation-checked

`validation/test_retriever_contract.py` · 2026-09-11 · runs in 0.05s, no table, no corpus, no VM, no GPU.

Written before the retriever exists, following the pattern in
`test_cot_counterfactuals.py`: inject a fake for the expensive part and pin the decisions.
Green proves the **contract is coherent and satisfiable**, not that any real retriever works.
When `cubbyllm/reasoning/retriever.py` lands, point the fixture at it and these become a
regression suite.

## Every pin is a measured failure mode

| pin group | the measurement behind it |
|---|---|
| `seed_*` | 61% of failures surviving lookup-first never produce a hop-0 seed (`exp_r4_index_walk`) |
| `no_threshold` | lookup-first beat cosine 0.710 vs 0.646, +51/0 lost, *because* it has no threshold; 34 of 36 `threshold_bound` absorbed (`exp_r1` Q5) |
| `distractor` | both computable margins had a distractor over gold — 0.2876 vs 0.9445 (`exp_r1` Q2) |
| `candidates_*` | `candidates_topk` empty for 152 of 246 failures (`exp_r1` Q1) |
| `backward_*` | `TripleIndex.by_object` exists, unused, worth ~2 on exact match (`exp_r4`) |
| `lookup_beats_search` | an exact index hit must not fall through to search |
| determinism | ambiguity is 0.8% of hops and must break the same way twice |

## Mutation check — do the pins bite?

A test that cannot fail is worse than no test. Each contract property was deliberately
broken and the suite re-run:

| mutation | caught by |
|---|---|
| threshold at hop 0 (τ=0.5) | `test_seed_hop0_from_exact_tail` |
| score gates instead of acceptance | `test_seed_hop0_from_exact_tail` |
| candidates not logged on a miss | `test_candidates_logged_on_FAILURE` |
| backward edge never consulted | `test_backward_edge_is_consulted_when_forward_misses` |
| search wins over lookup | `test_seed_hop0_from_exact_tail` |
| seed miss raises instead of returning None | `test_seed_failure_is_a_value_not_an_exception` |
| tie-break by insertion order | `test_store_order_does_not_decide_a_TIE` |
| tie-break ignores score | `test_score_still_breaks_a_tie_when_it_can` |

**One pin was found vacuous and fixed.** The original determinism test reversed the store,
but with only one accepted candidate there was no tie to break, so it passed against a
nondeterministic implementation. Fixed by adding a genuine ambiguity to the fixture — two
facts serving the same hop-0 tail, which is the 0.8% ambiguous-hop case the index docstring
names — plus `test_ambiguity_is_real_in_the_fixture`, a guard on the guard: if those two
facts ever stop both serving the hop, the determinism pins would silently go vacuous again.

A second lesson worth keeping: the first mutation attempt targeted the wrong line. Determinism
here comes from the `max(...)` key, not from the candidate sort, so breaking the sort changed
nothing. **A mutation that fails to break the property proves nothing about the pin.**

## Not yet pinned

- Fuzzy seeding itself — the actual product. It needs a real encoder, so it belongs in an
  end-to-end run, not here. These pins constrain the *shape* of the answer, not its quality.
- Latency. The current path is 45.9 ms/question end to end; a retriever that seeds fuzzily
  at hop 0 must stay inside that envelope, and that is a benchmark, not a unit pin.
