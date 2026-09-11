# Rung 1 — Week 1 "Know" results

**Run 2026-09-10** · `validation/exp_r1_gate_diagnostics.py` over `cot_harvest_v3cf.jsonl`,
`cot_harvest_lookup.jsonl` and `exp_m3_exhaustion_decomp.json` (decomp git_rev `96db562f1f71`).
Zero GPU. Every number below is a measurement over those three files.

## Headline: the free wins were already spent, and the real problem is smaller and different

The design memo assumed lowering `tau_ret` and adding backward retrieval would lift coverage
from 0.710 toward ~0.83, and that the emitter should be trained against that "grammar-maxed"
baseline. **That baseline does not exist.** The lookup-first index has already taken most of it.

| finding | result |
|---|---|
| threshold_bound already recovered by lookup-first | **34 of 36 (94%)** |
| far_below already recovered by lookup-first | **16 of 19 (84%)** |
| value of lowering `tau_ret` now | **2 cases** |
| failures surviving lookup-first | **195 of 246** |
| of which misparsed_chain + not_retrieved | **180 (92% of the remainder)** |
| backward retrieval clears tau on the remainder | **43** → ceiling **0.764** (upper bound) |
| surviving failures whose gold answer is in the store at all | **≥103 of 195 (53%)** → ceiling **0.839** (upper bound) |
| surviving failures at hop 0 | **121 of 195 (62%)** |

**The emitter's bar is 0.710, not 0.83.** Grammar-maxing is worth at most +0.054 (backward
retrieval), not +0.12.

## Q3 — cause x failing hop: the joint, computed for the first time

| cause | hop 0 | hop 1 | hop 2 | hop 3 | total | @hop0 |
|---|---:|---:|---:|---:|---:|---:|
| misparsed_chain | 69 | 15 | 8 | 0 | 92 | **75%** |
| not_retrieved | 49 | 28 | 11 | 0 | 88 | 56% |
| threshold_bound | 21 | 14 | 1 | 0 | 36 | 58% |
| far_below | 13 | 6 | 0 | 0 | 19 | 68% |
| walk_completes_now | 0 | 0 | 8 | 3 | 11 | 0% |

75% of misparses die at hop 0 — before hop-count planning could matter.
**Honest caveat:** this is consistent with *both* readings. A wrong frame produces a wrong
hop-0 query, so a parse failure can present as a hop-0 retrieval failure. Q6 says how to
settle it.

## Q1 — is the gold answer already in `candidates_topk`?

Of 246 exhausted walks, **only 93 have candidate data logged; 152 are empty.**
Of the 93 scannable: gold **present 25 (27%)**, **absent 68 (73%)**. Where present, median
rank 0.

So it is mostly a genuine retrieval miss, not a re-ranking problem — but the diagnostic is
missing for 62% of failures. **Instrumentation action: always populate `candidates_topk`.**

## Q2 — margin on the threshold_bound hops: the "free win" is a precision trap

Only 2 of 36 had both candidate data and a locatable gold. **Both have negative margin:**
gold 0.2876 vs distractor 0.9445; gold 0.5247 vs distractor 0.9964. Lowering `tau_ret` to
0.5028 would admit the distractor first. n=2 is weak — but Q5 makes it moot, because 34 of
these 36 are already recovered.

## Q4 — the six-opener override table: DEAD

The six openers match **52 of 92 misparses (57%)**, but the opener → hop-count mapping is
**not deterministic**:

| opener | gold n_hop | |
|---|---|---|
| `what is the source that` | {1: 14, 2: 7} | ambiguous |
| `which is the administrative territorial` | {1: 6, 2: 1} | ambiguous |
| `what is the instance of` | {2: 6, 4: 1} | ambiguous |
| `where is the history of` | {3: 4, 4: 3} | ambiguous |
| `which is the office held` | {3: 3, 4: 2} | ambiguous |
| `who is the editor of` | {2: 5} | deterministic |

**Only 5 of 92 (5%) are fixable by a prefix table.** The proposal is withdrawn.

## Q5 — the decomposition re-run under lookup-first

800 questions in both runs. Verified 517 → 568. **Recovered 51, lost 0.**
Still failing: **195**. By original class: misparsed_chain 92/92 (100% open),
not_retrieved 88/88 (100% open), walk_completes_now 10/11, far_below 3/19, threshold_bound 2/36.

## Q6 — backward retrieval on what remains

| kind | n | probed | reaches tau |
|---|---:|---:|---:|
| misparsed_chain | 92 | **0** | 0 |
| not_retrieved | 88 | 51 | **40 (45%)** |
| walk_completes_now | 10 | 0 | 0 |
| far_below | 3 | 3 | 3 |
| threshold_bound | 2 | 0 | 0 |

## Q7 — is the answer even in the store?

**≥103 of 195 (53%)** of surviving failures have their gold answer somewhere in the observed
store, so they are retrieval/routing failures rather than absence. Caveat: the observed store
is 966 of 1,241 facts (only facts that surfaced anywhere are visible), so "in store" is a
lower bound and "absent" is an over-count.

## THE NEXT QUERY — one script, and it decides the emitter's scope

**The 92 misparses were never backward-probed.** The original decomposition ran the backward
probe on `not_retrieved`/`far_below` only. So for the single largest surviving failure class,
we do not know whether the gold fact was findable in the other direction.

- If backward retrieval reaches them, the misparses are **direction** failures and belong with
  the retrieval fix — the emitter is not needed for them.
- If it does not, the parse produced an unanswerable query and they are **genuinely the
  emitter's target**.

That one probe moves the emitter's scope by up to 92 of 195 remaining failures — roughly half
the problem. Run it before anything is trained.

## Revised numbers for the ladder

- Emitter baseline: **0.710** (not 0.646, not 0.83)
- Realistic grammar-maxed ceiling: **0.764** (backward retrieval, upper bound)
- Absolute ceiling of this eval: **0.839** (every in-store case recovered, upper bound)
- Emitter's real target: 92 misparses + the ~48 not_retrieved backward does not reach
