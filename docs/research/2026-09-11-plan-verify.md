# The plan never crossed the gate — and disposing of it catches 90 of 92 misparses

**2026-09-11** · `cubbyllm/reasoning/plan_verify.py` (new) · `validation/exp_r6_plan_verify.py --observed` ·
763 grammar-parsed questions, 126 distinct relations from the observed store. **Wall 0.09s.** No GPU, no encoder, no VM.

## The hole

CubeLang's `ISolver` is `parse / solve / verify`. The program `programs.build_chain_program` emits is
**bind-only**: `solve`, `hop_i`, `control` — the walked objects go into a frame and the VM verifies each
recovery against them. The **plan** — which relations, in which order, ending at which tail — is produced
by the Python regex grammar in `planner.py` and handed straight to the walk. It never crosses into the VM.
Nothing checks it.

So the plan is the one input in the pipeline no gate ever sees, and every misparse in the harvest
(92 `misparsed_chain`, the 16 compound-relation residue of exp_r5) walked through that hole with the
VM faithfully verifying bindings against a wrong chain.

## The fix is a disposer, not a better parser

`planner.py` is scaffolding — it generates the training data for the emitter that replaces it — and the
emitter will propose plans no regex ever produced. So the check has to be **independent of how the plan
was made**: model proposes, host disposes. `plan_verify.verify_plan(question, plan, known)` runs two
pure checks and returns a `PlanVerdict` that always carries its reason:

| check | what it asks | catches |
|---|---|---|
| `covers()` | does the plan reconstruct the question? (canonical body is a substring of the normalized question; 1-hop: relation and entity both present) | an emitter that drops a hop, invents one, or answers a different question |
| `unknown_relations` | is every relation the plan asks for — `relations[1:]` and the tail's relation prefix — one the store holds? | **the compound-relation misparse**: `want 'award received by the director of photography'`, `have ['director of photography']` — no such edge, plan rejected before a walk is spent |

`known` is injected (`KnownRelations`: anything with `__contains__`). Validation passes `StoreRelations(store)`;
the 12 pins pass a frozenset; **MoWM's world can pass itself** when "possible edge" replaces "edge in
store" — the `unknown_relations != []` branch is the spawn-a-latent-world path of the same verdict.

## Measured

**Q1 — `covers()` on grammar plans: 763 / 763 (100%)** in every outcome bucket. The grammar is
deterministic on the text, so this proves the check is emitter-facing, not grammar-facing. Joining
inverts splitting, so coverage *cannot* catch a mis-split of the same words — that is Q2's job.

**Q2 — `unknown_relation` as a misparse detector:**

| harvest outcome | rejected | of |  |
|---|---:|---:|---:|
| verified | 2 | 517 | 0% |
| **misparsed_chain** | **90** | **92** | **98%** |
| not_retrieved | 69 | 88 | 78% |
| threshold_bound | 1 | 36 | 3% |
| far_below | 1 | 19 | 5% |
| walk_completes_now | 0 | 11 | 0% |

**Catches 90/92 misparses and wrongly rejects 2/517 verified chains — precision 0.98 against
verified, in 0.09s, before any walk, retrieval or VM call.** The rejected plans are exactly the
compound relations the splitter manufactures when a question joins its hops with `contained within the`,
`that describes the`, `received by the` instead of `of the`:

```
'administrative territorial entity contained within the country of citizenship'
'source that describes the instance'
'award received by the creator'
```

The `not_retrieved` row (69/88) is where the store caveat lives: 126 relations from 78% of the facts.
Those rejections are either a vocabulary gap (`--real` will close it) or the question asking for an edge
type the store does not have — which under the don't-know contract is a *correct* refusal, not a failure.
Either way it is not a retrieval problem, which is what the label said it was.

**Q3 — can the disposer repair, not just reject?** Vocabulary-guided re-segmentation of the chain body
(`segment()`), two modes:

| | verified: unique / ambiguous / none | unique & gold n_hop | misparsed repaired |
|---|---|---|---:|
| Q3a exhaustive (entity may contain ` of the `) | 249 / 170 / 0 | 248 right, 1 wrong | 1 / 92 |
| **Q3b strict** (`joint_in_entity=False`) | **418 / 0 / 1** | **418 right, 0 wrong** | 0 / 92 |

Q3b is the clean result. Reading the body **from the relation vocabulary alone, with no ` of the ` guess**,
strict segmentation is unique on 418 of 419 verified chain-frame questions, agrees with the gold hop count
on every one, and is never ambiguous. On the 82 misparsed chain-frame questions it returns **no reading at
all for 80** — it refuses rather than guesses. It repairs nothing (0/92), because the misparses use joints
it does not know; but it is a second, independent detector with **zero false positives** on verified.

The one verified `none` and the one exhaustive `BROKE` are the same question — an entity that contains
` of the ` ("LGBT rights in the United Kingdom" chained under "country of") — and are the price of not
guessing.

**Dead repair probe** (recorded in the script so nobody re-runs it): relation-led segmentation over
*arbitrary* joints fixes 10/92 but produces 14 confidently-wrong unique readings, because a known
relation can be a prefix of the true one (`award` of `award received by`). Greedy-prefix over a
vocabulary is unsafe by construction.

## Where it sits — and which half the VM runs

```
question -> (grammar | emitter) -> plan -> verify_plan -> Retriever.seedable -> walk -> VM
                                              |
                                              +-- ok=False -> honest fail, reason carried into the harvest
```

**Correction (same day):** my first draft said the VM had "no LOOP/CALL yet" on the strength of
`CUBELANG_OPCODES_PLAN.md`. That file is stale — RETURN/CALL/LOOP/arrays are in the compiler. I then
read `cubelang/src` (staged from disk) to find out what the VM can actually do for this check:

**The vocabulary check runs in the VM now.** `QUERY` (`src/vm/engine.rs`, `src/vm/knowledge.rs`) is an
executing opcode: an exact, normalized hash lookup over an in-VM knowledge store loaded with
`cubelang run --knowledge facts.jsonl`, pushing the chunk array — **zero chunks on a miss, never a
nearest neighbour**. That is "is this relation one the store holds?" with the don't-know contract's own
semantics. So:

- `plan_verify.write_vocab_jsonl(store, path)` turns the store's relation vocabulary into knowledge
  (`{"key": rel, "text": rel, "source": "store"}`, one per relation).
- `bridges/programs/plan_verify.cube` is `query mention; pop hits; return hits` — the proven shape from
  `cubelang/tests/query_grounding.rs`.
- `plan_verify.VMRelations(knowledge, exe)` is a `KnownRelations` whose `__contains__` is one QUERY
  per distinct relation (memoized: ~130 calls across 763 questions, not ~1,500).
- `cubelang_client.run_program` gained a `knowledge=` kwarg that passes `--knowledge`.
- `exp_r6 --vm` routes every membership question to **both** the VM and the host-side
  `StoreRelations`, drives the numbers from the VM's answer, and **asserts they never disagree** —
  the host module is the reference the VM is checked against, as the docstring promised.

**The coverage check stays host-side, and not by choice.** The VM has no string semantics: `COMPARE`
resolves both operands through `resolve_i64`, and `Value::Str.as_i64()` is `0`, so `input == "abc"` is
`0 == 0` — true for any strings; `s.contains(x)` lowers to `CALL contains`, an unresolved function
(`src/compiler.rs`, `MethodCall` arm). Both are pinned as `#[ignore]`d contracts in
`cubelang/tests/str_semantics.rs` (2026-08-30). A substring check inside the VM today would be a silent
stub that passes verify-before-execute — the exact failure class strict mode exists to catch. **The gap
is strings, not control flow.** When a string-equality/containment opcode executes, `covers()` moves in
too and `test_plan_verify.py` is its reference.

## Wired into the pipeline and harvested — before/after on the same store, same seeds

`pipeline.answer(..., known=)` disposes of the plan before the walk (opt-in; `known=None` is the old
path byte-for-byte). `exp_m3_cot_pipeline.py --lookup --verify-plan vm` ran it with the **VM's QUERY**
answering membership (165 relations as knowledge, 181 QUERY calls, memoized).

| base → vp | n |
|---|---:|
| verified → verified | 566 |
| retrieval_exhausted → **unknown_relation** | **161** |
| retrieval_exhausted → retrieval_exhausted (residue) | 34 |
| unparseable → unparseable | 37 |
| **verified → unknown_relation** | **2** |

Predicted 161 / 34 / 2 from Q4; got 161 / 34 / 2. Precision 0.993 → 0.993, control 1.000, counterfactual
escapes 0 → 0, repair burns 196 → 35. Nothing verified that had not before. All four kill-criteria PASS.

**The two losses were a bug in principle, now fixed.** Both chains asked for
`office held by THE head of government`; the store holds `office held by head of government`. The walk
accepts that at hop ≥ 1 through `relation_matches` (Jaccard ≥ 0.6) — that is how both verified in base.
The exact-only disposer was therefore *stricter than the walk*, which is the wrong relationship: a
disposer that refuses what the walk would verify is throwing away answers, not cost.

`verify_plan` now has **exactly the walk's tolerance**: exact at hop 0 (the walk's `accepts` at hop 0 is
string equality on the whole tail), `relation_matches` at hops ≥ 1. A refusal is then a proof that no fact
in the store could be accepted at that hop under the walk's own rule — **0 verified chains lost, by
construction**. The paraphrase tier is recorded on the verdict as `(asked, matched)` pairs and never hidden
inside "known" — the same rule `knowledge.rs` states for QUERY: a guess must not wear an exact hit's
confidence. For `VMRelations` the exact tier is the VM's QUERY and the paraphrase tier reads the same
vocabulary file the VM was given.

Re-measured on the harvest (observed vocab, same as `--real` on Q2):

| | exact-only | walk-tolerant |
|---|---:|---:|
| misparsed chains refused | 90 / 92 | 86 / 92 |
| verified chains wrongly refused | 2 / 517 | **0 / 517** |
| refused at plan time (of 232 surviving) | 161 | 155 |
| residue | 34 | 40 |

The 6 misparses that now walk are `office held by the head of government` paraphrases whose fact is not in
the store — dead walks, cost only, and the VM still gates them.

**Re-harvested (`--lookup --verify-plan vm --tag _lookup_vp2`):**

```
outcomes: verified=568  unknown_relation=155  retrieval_exhausted=40  unparseable=37
precision 0.993 (564/568) · control 582/582 · counterfactual escapes 0/2157 · repairs {0: 759, 1: 41}
```

**568 — identical to base.** Zero verified chains lost, as the rule guarantees; the counterfactual block is
byte-for-byte the base run's (same 568 chains, same 2,157 planted faults, all caught). Predicted 154 / 41,
got 155 / 40. Every one of the 800 questions is now accounted for with a reason: 568 verified, 155 refused
at plan time because the store has no such edge, 40 refused after a walk that never found a seed fact,
37 the grammar never claimed. Under the don't-know contract that is 568 answers and 232 honest refusals —
and the 155 carry the relation they could not find, which is exactly what a harvest for the emitter needs.

**Wall time, honestly.** `mean_wall_ms(cot)` went *up*, 59.4 → 65.4. The 161 refused walks saved time,
but each `VMRelations` membership is a `cubelang.exe` subprocess (181 spawns ≈ 30 ms each, inside the
timed region) and that cost more than the walks it prevented. `--verify-plan host` would show the net
saving; for the VM path to be net-positive the VM must stay resident — `run-proto` needs a knowledge
field in `RunRequest` and a persistent process. That is a transport task, not a reasoning one.

## The VM as a resident process — the wall-time objection, closed

`run-proto` now serves until stdin closes (fresh VM per request, same wire, same decoder), with
`knowledge_path` on the request and the parsed store cached by `(path, mtime)`. `CubelangSession`
keeps one process open; `exp_m3 --resident` routes **every** VM call through it.

`--lookup --verify-plan vm --resident --tag _lookup_vp_res`, same store, same seeds:

| | base | vp (spawn per call) | **vp_res (one process)** |
|---|---:|---:|---:|
| mean_wall_ms (cot, per question) | 59.4 | 65.7 | **2.0** |
| eval, 800 questions | 124 s | 128 s | **10 s** |
| counterfactuals, per chain | 123 ms | 121 ms | **3 ms** |
| VM calls | ~2,900 spawns | ~3,100 spawns | **4,694 requests, 1 process** |
| outcomes | 568 / 195 / 37 | 568 / 155 / 40 / 37 | **568 / 155 / 40 / 37** |
| precision · escapes | 0.993 · 0 | 0.993 · 0 | **0.993 · 0** |

Identical outcomes, identical counterfactual block, all four kill-criteria PASS — and the verified
chain-of-thought is now **faster than the unverified chase-only baseline** (2.0 vs 5.0 ms). The
"thinking fee" for a VM-verified 3-hop answer on this corpus is ~2 ms of CPU. (The organic-confusable
held-out catch moved 0.78 / 0.80 / 0.75 across the three runs; it did so between vp and vp2 as well,
before the transport changed — sampling variance in that probe, not the transport.)

## Contract

`validation/test_plan_verify.py` — 13 pins, 0.04s, fake vocabulary only. **10 mutations, 10 caught** —
the tenth (1-hop coverage ignoring the entity) only after the pin it exposed was added; the mutation is
what found the missing pin.

`validation/test_pipeline_plan_refusal.py` — 4 pins on `answer(..., known=)`: refusal before any
retrieval/VM call, `known=None` unchanged, a verifying plan walks identically with or without `known`,
coverage failure carries its own reason. Fake retriever + fake VM.

`validation/test_plan_verify_vm.py` — 4 pins: two run without a VM (the jsonl schema; the transport
contract via an injected fake), two need a cubelang build and skip cleanly otherwise. **Those two, and
`exp_r6 --vm`, have not run yet** — the cloud sandbox cannot build cubelang (`static.crates.io` is
blocked by egress policy) and cannot run your `.exe`. They run on your machine.

## What changes upstream

- The harvest's `misparsed_chain` bucket is now **detectable at plan time** with a reason string, so it
  can be labelled in the harvest instead of inferred afterwards from `n_hop` disagreement.
- The 16 "relation mismatch" cases of exp_r5 and 90 of the 92 misparses are the same population, seen
  from two sides. The retriever was never going to fix them; the disposer refuses them for 0.09s.
- `Retriever.seedable(plan)` should take a verified plan. Wiring `verify_plan` into `pipeline.answer`
  before the walk is the next code change; it is one call and one new `reason` value.

## Run it

```
python validation/exp_r6_plan_verify.py --observed          # reproduces the above
python validation/exp_r6_plan_verify.py --real              # closes the 126-relation caveat
python validation/exp_r6_plan_verify.py --real --vm         # membership answered by the VM; asserts host == VM
python -m pytest validation/test_plan_verify.py validation/test_plan_verify_vm.py -q
```

## The emitter's plans, walked (exp_r7) — and what the loop produced

Coverage v1 first run on arm B: 9 walked, 7 verified, **3 correct, 4 wrong** — every wrong one a dropped
outer hop that v1's 1-hop rule could not see. Under coverage v2:

| arm | built | refused | walked | verified | correct | wrong |
|---|---:|---:|---:|---:|---:|---:|
| A — training questions (sanity) | 93 | 6 | 87 | 84 | 83 | 1 |
| **B — the 92 misparses (unseen)** | 79 | 63 | **16** | **13** | **13** | **0** |
| C — unparseable | 0 | — | — | — | — | — |

16 walked, exactly as predicted from the plans; 13 VM-verified, 13 correct, 0 wrong. The verified set includes
`administrative territorial entity contained within the country of …` — the `3→2` joint family that the
grammar, fuzzy matching, suffix backoff and vocabulary-guided segmentation had all failed on. **These are the
first verified examples of that shape in existence, and they came out of the loop**: emitter plan → disposer →
grammar walk → VM. `cot_harvest_r7_v2.jsonl` holds 97 verified records, 13 on questions outside gen 1's data.

Gen 2 (`emitter_sft_v12e`, `build_gen2_partition.py`): gen 1 unchanged + those 13 chains ×6 (gold-correct 13/13,
reported not filtered) + 1,214 question-only `plan` records → 11,845 records, 14,760 train rows. The 13 questions
are written to `gen2_exclusions.json`; the gate excludes them.

## Arm C — the 37 the grammar declared out of scope (same day, later)

C is two populations. **19 one-hop questions in unseen frames** (`What is X a participant of?`, `What award
did X receive?`, `Which languages spoken, written or signed by X?`): the emitter plans them; exp_r7 only
failed to find the seed. A residual seed rule (strip the relations and the frame words; what is left is the
entity) plus a 1-hop coverage rule that lets the entity precede the relation: **3 verified, 3 correct, 0
wrong** (r7 v3), 13 still refused because the emitter's role names paraphrase the question — gen 2's plan
target keeps the question's words.

**18 nested questions** (`Which list includes the list that includes the component of the instance of X?`):
exp_r8, the first GoT-1 slice — a host-composed decomposition, the inner chain and the outer hop each
VM-certified, bounded alternatives for the ambiguous outer form, an ambiguous frontier counted never chosen.

| | |
|---|---:|
| inner chain verified | 17 / 18 |
| composite verified | **14 / 18**, 14 correct, 0 wrong |
| outer fact absent from the store (correct refusal) | 3 |
| inner relation unknown (correct refusal) | 1 |
| VM calls | 78 (4.3 / question; the flat 3-hop walk spends ~3) |

Two disposer bugs were found by this run, not by the emitter: the class noun in `Which LIST includes …` was
read as a dropped `list` hop (the one correct plan refused — fixed: the class-noun strip now knows
`includes / contains / has`), and a 1-hop inversion frame states its entity before the relation (fixed).
Neither cost a verified chain elsewhere (517/517, 86/92 unchanged).

What the neighbourhood print settled: the outer fact is a plain forward edge (`Living beings is the list of
Books/Organism`), so with the class-noun fix the **flat** three-hop plan `[instance, component, list]` covers
the whole question — `that includes` is a joint, like `contained within the`. This family is a chain with a
joint the grammar lacks, not a genuinely nested sub-question; the decomposition is the ceiling of the general
mechanism, and it held. GoT-1's decomposition remains the road for inner parts that are not a chain the walk
can take — this eval's 800 questions contain none.

**Arm C today:** 17 of 37 with a VM verdict (3 flat, 14 composed), 0 wrong, from 0 this morning. The 14
composites are gen-3 material — verified three-hop chains on a joint no training record has yet.

## Gen 2 (emitter_v12e) — the pre-registered gate, run

The Colab run landed as `standin/models/LFM2.5-2.6B.Q4_K_M.gguf` (Unsloth's default export name — every
program it emits is a `CotPlan`, which only v12e has seen; copied to `emitter_v12e.Q4_K_M.gguf`). Gate run
with the 13 r7 questions excluded, so gen 2 is measured on the questions gen 1 never verified.

**Like-for-like on the remaining 79 B questions** (same r3 disposer rule, gen 1 re-scored on the same 79):

| | gen 1 (v8e) | **gen 2 (v12e)** |
|---|---:|---:|
| well-formed plans | 68 | 69 |
| gold hop count | 32 | **39** |
| disposer-accepted | 26 | 16 |
| accepted **and** gold hop count | 5 | **10** |
| VM-verified after the walk | 0 (its 13 were the excluded ones) | **4**, 4 correct, 0 wrong |

Arm C: gold hop count **31/37** (gen 1: 13/37); verified 3, the same three. Arm A (training set) 95/100
verified. Canary 0/143 object matches. **The bar — more accepted, gold-hop plans on the remaining B than
gen 1, at gen 1's precision or better — is met: 10 vs 5, 0 wrong.** Small numbers, honestly; 4 new
VM-verified answers is transfer, not a leap.

What gen 2 does differently, read from its refusals: 27 `unknown_relation` on B (gen 1: 7) — it shortens
compound relations to their head noun (`source`, `water body`, `office`, `history`) where the store holds
the long form; and on C it emits the question's relation words at hop 0 (`languages spoken written signed`)
where the walk's hop-0 rule is exact string equality, so 25 of 37 C plans are *of the question* but only 6
are *answerable*. That is now the binding constraint on C, and it is the walk's, not the emitter's: exp_r5
measured a hop-0 paraphrase tier (relation_matches over `by_subj`) worth 23/118 on the real store. It has a
reason to exist now.

`cot_harvest_r7_gen2.jsonl`: 102 verified records, 7 on questions outside gen 2's data — gen 3's material.


## Matched pairs on the wiki world (exp_r9) — the generality test the memo asked for

Everything above lives on one 1,241-fact store with 165 relations, which the emitter was fine-tuned on.
The rung-1 memo's own falsifier: if the mechanism does not hold on a genuinely different corpus, throw the
memo away. exp_r9 is that test, in Grok's matched-pair design. The world is the wikikg-trajectories graph
as served (552,297 facts, **2,967 relations** — the emitter has seen 165 of them), TripleIndex-only walk,
resident VM, the harvest's tau floors. 200 functional two-hop chains (both hops served by exactly one
stored fact, so the walk's end is the one gold), each issued in four surface forms against the same store:

| form | shape | grammar | **emitter (gen 2)** |
|---|---|---:|---:|
| canonical | `What is the P2 of the P1 of E?` | **189**/200 correct, 0 wrong | 107/200 correct, 0 wrong |
| have | `What P2 does the P1 of E have?` | 0 (200 plans, all walks failed — 1-hop misparse) | **77**/200, 0 wrong |
| relative | `What is the P2 of the thing that is the P1 of E?` | 0 (200 plans, all refused) | **124**/200, 0 wrong |
| possessive | `E's P1 — what is its P2?` | no plan | 0 (180 refused `plan_does_not_cover_question`) |

**Out of basin (3 forms × 200): grammar 0 correct, emitter 201 correct, 0 wrong.** 1,581 VM calls; wall
459 s; 0 wrong on every form for both planners. A win here is attributable to question shape and nothing
else — same chains, same facts, same gold, same gates.

Three honest readings. The emitter's canonical score (107) is below the grammar's (189): it is a bare
2.6B stand-in fine-tuned on 165 relations meeting ~2,900, and 48 of its canonical plans fail `covers()`
(it drops or shortens a relation), 2 name relations the store lacks. The possessive zero is the
disposer's, not the emitter's: `covers()` assumes the answer-side relation comes first in the question
and the possessive states the inner relation first — a known rule to add, and the refusals were the
right verdict under the rule as written. And the number that matters is the last column: **nothing the
VM verified was wrong, on 1,600 questions and a world 445× the training store.** The gate held
where the memo said it had to.

`validation/exp_r9_matched_pairs.py`, logs `exp_r9_matched_pairs.{json,log}`.

## SimpleQA through the whole gate (exp_r10) — free text, nobody's template

exp_r9 was still our wording. SimpleQA is 4,326 human-written factoid questions with one short gold each, and
exp_g4b had already measured the wiki world's lookup ceiling on it: the gold is a graph entity for 860, and a
stored fact "gold is the R of E" with E named in the question exists for **three**. So this run does not
measure accuracy. It measures the don't-know contract at scale: what a 2.6B stand-in proposes when handed free
text, and whether the disposer, the walk and the VM let any of it through.

| | all | gold in graph | gold not |
|---|---:|---:|---:|
| questions | 4,326 | 860 | 3,466 |
| no plan | 468 | 60 | 408 |
| plan (1-hop / 2-hop / 3+) | 3,858 (2,140 / 1,409 / 309) | 800 | 3,058 |
| seed names a graph entity | 273 | 49 | 224 |
| every relation known to the store | 311 | 34 | 277 |
| refused by the disposer | **3,825** (3,382 `plan_does_not_cover_question`, 443 `unknown_relation`) | 795 | 3,030 |
| walked | 33 — all `retrieval_exhausted` (no seed fact) | 5 | 28 |
| verified / correct / near / **wrong** | **0 / 0 / 0 / 0** | 0 | 0 |

**Every one of the 3,858 plans was refused or failed with a named reason; 0 VM calls; 0 verified; 0 wrong.**
Wall 2,657 s, of which the emitter is 2,559 s (0.60 s per question); the gate itself is 14 ms per question.

What the emitter did with free text is worth reading, because it is the part that transfers: `Who received the
IEEE Frank Rosenblatt Award in 2010?` → `['received']` seed `ieee frank rosenblatt award 2010`; `Who appointed
the Chief Justice of India, Mirza Hameedullah Beg, in 1977?` → `['appointed the chief justice of india in 1977']`
seed `mirza hameedullah beg`; `What is the name of the former Prime Minister of Iceland who worked as a cabin
crew member until 1971?` → `['former prime minister', 'cabin crew member until 1971']` seed `iceland`. These are
plans *of the question* — the decomposition is right and the seed is the right entity — for a store that does
not hold the edge. The disposer said so 3,825 times, the walk said so 33 times, and nothing was spoken.

Two readings. First, the contract held on 4,326 questions nobody here wrote, with the same gates and floors as
every run above: the wrong count is zero because nothing reached the VM, and nothing reached the VM because the
store cannot answer SimpleQA — 3 keyed facts out of 4,326 — which is the truthful state of affairs. Second, the
gap between `rels_all_known` (311) and `walked` (33) is `covers()` refusing plans whose relations the store holds
but whose wording the residual rule does not accept (`received` … `in 2010`): the rule is conservative on free
text, as it should be while the alternative is a wrong answer, and each of those 278 is a harvestable refusal
with the question, the plan and the reason on record. Coverage on SimpleQA is a store problem before it is a
planner problem; the search-and-learn path (a fact the store does not hold, fetched, verified, then stored) is
where that number moves, and this run is its baseline: 0.

`validation/exp_r10_simpleqa.py`, logs `exp_r10_simpleqa.{json,log}` (smoke n=60: `exp_r10_simpleqa_smoke.*`).

## Coverage, lever 1 — hop 0 gets the paraphrase tier (and what the first run taught)

Hop 0 was the one hop with a single tier: `accepts` compared the whole tail string to the fact's `rel of subj`,
and the disposer mirrored it (exact known prefix). Hops ≥ 1 always had two tiers — exact, then
`relation_matches` (Jaccard ≥ 0.6 on relation words) with the subject exact. exp_r5 had measured a hop-0 tier
worth up to 23/118 on the entry residue, and gen 2's C plans were blocked on it (25 of the question, 6
answerable). The change gives hop 0 the same two tiers in all four places: `planner.accepts` (every `' of '`
split of the tail, subject exact, relation by `relation_matches`), `TripleIndex.hop` (exact tier first, then
`by_subj` at each split), `pipeline._walk` (an exact hit is never displaced by a paraphrase; the trace says
`lookup_paraphrase`), and the disposer (`tail_match`, recorded in `paraphrased`).

**The first harvest verified a wrong answer.** `What is the award received by the genre of Joan Rivers: A
Piece of Work?` → `Documentary series` (gold: Genesis Awards). The greedy fact parse reads `Documentary series
is the genre of Joan Rivers: A Piece of Work` as relation `genre of Joan Rivers: A Piece` | subject `Work`; the
plan's tail mis-splits at the same `' of '`; the two "relations" overlap at Jaccard 0.625; the VM verified a
1-hop chain that is internally consistent and semantically wrong. That is exactly the failure class the VM
cannot see (it certifies the binding, not the wording), so the guard has to be the host's: **a paraphrase is
only trusted for a relation the store states in ≥ 2 facts** (`StoreRelations.reused`, `TripleIndex._rel_n`;
`write_vocab_jsonl` now carries `n`). A relation is reused by nature; a one-off "relation" is usually an
entity fragment. Pinned in `validation/test_hop0_paraphrase.py` (5 pins, including the Joan Rivers case and
the guard's honest limit: the same fragment stated twice counts as a relation).

Measured, same store, same seeds, same gates:

| | before (lookup_vp_res) | hop 0 tier, no guard | **hop 0 tier + reuse guard** |
|---|---:|---:|---:|
| verified / 800 | 568 | 570 | **568** |
| claimed-answer precision | 0.993 (564/568) | 0.991 (565/570) — **1 new wrong** | **0.993 (564/568)** |
| refusals `unknown_relation` | 155 | 148 | 150 |
| `retrieval_exhausted` | 40 | 43 | 43 |
| gen 2 (exp_r7, --exclude): arm C verified | 3/37 | — | **5/37**, 5 correct, 0 wrong |
| gen 2 arm A | 95/100 | — | 96/100 (wrong 1, unchanged) |
| gen 2 arm B | 4/79 | — | 4/79 |

On the eval store the lever nets zero: five plans that were refused at plan time now walk and die at
retrieval (43 vs 40), and the one legitimate gain of the unguarded run (`successor of the fictional universe of
Stature` → MC2, gold) is lost to the guard because `from fictional universe` occurs once in 1,241 facts. On
gen 2 it is worth +2 on arm C at 0 wrong. The residue exp_r5 called "relation mismatch" was mostly misparses,
which the disposer already refuses; genuine relation paraphrases are rare on a 165-relation store.

What the run surfaced is bigger than the lever: **88 of the 1,188 parsed facts carry a singleton relation from
the greedy `' of '` split, and for 49 of them another split yields a reused relation** (`capital` | `county of
clarion, pennsylvania`, `location` | `final assembly of ss canberra`). Those facts are indexed under the wrong
subject and are unreachable by entity at every hop — a parse ambiguity that the exact tier at hop 0 happens to
tolerate and nothing else does. Lever 1b, measured before it is built: a relation-aware fact split (prefer the
split whose relation the store reuses) is a store-side change to `parse_fact`/`TripleIndex.add`, and it touches
every hop, so it gets its own before/after.

`cot_harvest_hop0.jsonl`, `exp_m3_cot_pipeline_hop0.{json,log}`, `exp_r7_emitter_planned_walk_gen2_hop0.*`,
`cot_harvest_r7_gen2_hop0.jsonl`.

## Coverage, lever 2 — `covers()` v3: either reading order, either wording of a paraphrase

Two holes in v2, both found by runs rather than by thought. **Order:** v2 read the question answer-side first
only ("What is the P2 of the P1 of E?"); exp_r9's possessive form ("E's P1 — what is its P2?") states the
entity first and the hops inner-first, and every one of its 180 plans of the question was refused. **Wording:**
v2 looked for the plan's relation string verbatim, so a relation the vocabulary tier had already accepted as a
paraphrase (`languages spoken written signed` for the store's `languages spoken, written or signed`) failed
coverage because the question carries the store's wording — 6 of gen 2's 21 arm-C coverage refusals. v3
accepts either reading (answer-side first then the entity, or the entity then the hops in walk order) and
either wording of a paraphrased relation; the residual rule — no relation word may be left over — is unchanged
and runs on both readings, so a dropped hop still fails and a swapped chain fails in both orders. Three new
pins in `test_plan_verify.py` (32 total across the disposer suites), `its it s thing` added to the frame words.

| | before (lever 1) | **covers v3** |
|---|---:|---:|
| harvest, 800 questions | 568 / 0.993 | **568 / 0.993, byte-identical** |
| gen 2 arm C (exp_r7 --exclude) | 5/37 verified | **11/37 verified, 11 correct, 0 wrong** (coverage refusals 21 → 15) |
| gen 2 arms A / B | 96 / 4 | 96 / 4 |
| exp_r9 canonical / have / relative (emitter) | 107 / 77 / 124 | 110 / 80 / 127 |
| exp_r9 possessive (emitter) | 0 (180 refused for coverage) | **21/200 correct, 0 wrong** (154 still refused) |
| exp_r9 out of basin, emitter correct / wrong | 201 / 0 | **228 / 0** |
| grammar, all forms | 189 / 0 / 0 / — | unchanged |

The possessive residue is not the disposer's any more; it is the emitter's, and the refusals are *correct*.
Read on 25 possessive questions: gen 2, never shown this shape, folds the inner hop into the seed entity —
`King Danjong's loyalist — what is its group?` → `['group']` seed `king danjong loyalist` — a 1-hop plan that
dropped a hop, and the residual rule catches `loyalist` as a relation word. The 21 that verified are the 21
where it emitted the two-hop plan (`Karate's member — what is its type?` → `['member', 'type']` seed
`karate`). One limit worth recording: a seed that swallows the whole question prefix (`rajaraja narendra s
adaptation what is its`) hides the dropped hop from the residual rule and passes coverage; the walk then fails
it for want of a seed fact, at the cost of one lookup, never an answer. The disposer cannot judge entity
strings without a store lookup, and that lookup is the walk.

Where lever 2 leaves the ladder: the out-of-basin count is 228 at 0 wrong; arm C is 11/37 by emitter plan
alone (plus exp_r8's 14 composed through the grammar); and the possessive form is a gen-3 training item — the
shape with its two-hop plan, of which the harvest now holds 21 VM-verified examples.

`exp_m3_cot_pipeline_cov3.*`, `cot_harvest_cov3.jsonl`, `exp_r7_emitter_planned_walk_gen2_cov3.*`,
`cot_harvest_r7_gen2_cov3.jsonl`, `exp_r9_matched_pairs_cov3.{json,log}`.

## Coverage, lever 1b — the relation-aware fact split (+14 verified, 0 wrong)

`parse_fact` read `OBJ is the REL of SUBJ` greedily: the last `' of '` splits relation from subject. Right for
`country of citizenship of jean`, wrong for every subject that contains `' of '` — `genre of Joan Rivers: A Piece`
| `Work`, `capital of Economy` | `kyrgyzstan`, `location of final assembly` | `ss canberra` — and a fact split
wrong is indexed under the wrong subject, unreachable by entity at every hop. The exact tier at hop 0 hid this
(it compares the whole string); nothing else tolerated it. 88 of the eval store's 1,188 parsed facts, 49 with a
better split.

The fix is the disposer's own rule applied to facts: **the longest prefix that is a relation the store reuses
wins** (`parse_fact(f, known=)`, `reused_relations(facts)` = relations stated in ≥ 2 facts under the greedy
pass). `TripleIndex(facts)` seeds the reused set with one greedy pass, then indexes; `add()` keeps the set
current, so a live-learned fact splits the same way once its relation is reused (order-dependent only for the
first two facts of a relation). `StoreRelations` parses identically, so the vocabulary drops the fragments:
**165 → 116 relations** on the eval store — the 49 predicted. One pin in `test_hop0_paraphrase.py` (6 total);
the stand-in's own suites pass unchanged.

| | before (covers v3) | **relation-aware split** |
|---|---:|---:|
| verified / 800 | 568 | **582** (0.728) |
| claimed-answer precision | 0.993 (564/568) | **0.993 (578/582)** — 15 gained, all correct; 1 lost |
| `retrieval_exhausted` | 43 | **29** |
| CoT by hop (1 / 2 / 3) | 0.917 / 0.648 / 0.283 | 0.917 / **0.672 / 0.333** |
| gen 2 arm B / C / A | 4 / 11 / 96 | **7** / 11 / 95 |

CoT overall is now 0.723 against chase-only's 0.724 — the verified path has caught the unverified one on the
same store. The gains are the predicted ones: `Clarion (PA) is the capital of county of clarion, pennsylvania`
now serves a hop from `county of clarion, pennsylvania`; `Etymology of Austria` is a subject; a `location of
formation` chain through `Los angelas` runs to its time zone. The one loss is worth its line: a 3-hop chain
whose three VM similarities sat at 0.226 / 0.230 / 0.268 against the 3-hop floor 0.2202 — renaming the hop-0
role from the fragment to the real relation moved one of them under the floor. The chain is right; the floor is
at the edge of what the VM measures for 3-hop frames, which is a calibration fact, not a parse one. (Gen 2's
one arm-A loss is the mirror: an emitter plan that folded `Economy` into the relation used to match the
mis-split fact exactly, and verified a right answer for the wrong structural reason.)

`cot_harvest_split.jsonl`, `exp_m3_cot_pipeline_split.{json,log}`, `exp_r7_emitter_planned_walk_gen2_split.*`,
`cot_harvest_r7_gen2_split.jsonl`. exp_r9 is not re-run on this lever: its 200 chains are sampled through the
index, and a changed index samples different chains — a like-for-like rerun needs the chain set pinned first.

## Coverage, lever 3 — search-and-learn (exp_r11), and the ambiguity rule it forced

The store lever: a refusal becomes a fetch, a gate, a write with provenance, and a second walk.
`cubbyllm/reasoning/learn.py` is the host loop (`learn_and_answer`): `answer()` → the entity the walk stalled on
(the seed at hop 0, else the last object reached) → `source.facts(entity)` (data, never a judgement) → `gate`
(parse, duplicate, sibling) → `store.add` + `known.add` with a `Provenance` record (source, entity, time, the
store's snapshot hash before the write) → `answer()` again. Bounded: at most `max_entities` rounds, one entity
each, and a round that admits nothing ends it. Only `retrieval_exhausted` and `unknown_relation` are learnable;
a plan that does not cover its question stays refused. `Source` is a protocol; the network lives in the serve
stack (`standin/sources.py`, `WikidataSource`: claims rendered as `OBJ is the <property label> of <entity>`,
identifiers and media skipped, responses cached so a rerun is offline and byte-identical). 4 pins, then 5.

**Held-out arm** (the eval store, the hop-0 fact of 200 verified chains withheld, a source that serves exactly
the withheld facts by subject). Clean: 200 refused at first (191 no seed fact, 9 unknown relation) → 197 learned
→ **197 verified, 196 correct**, 1,221 VM calls, 2 s. Then the source poisoned with a contradicting object for
each withheld fact, served *after* the truth: 195 refused by the gate as contradictions, 194 correct; served
*before* it: **119 wrong**, with provenance. The gate as first written kept the store consistent, not true — a
novel falsehood contradicts nothing.

**Wikidata arm, run 1** (600 SimpleQA questions, gen-2 plans, the wiki world, the loop): 530 plans, 462 refused
for coverage (not learnable), 68 fetched, 1,919 facts fetched, 800 admitted, **1,069 refused as
"contradictions"** — and one verified answer, **wrong**: `As of 2022, what is the population of Mersin
Province?` → `1814468` (gold 1,916,432), one census out of the many Wikidata states.

Three faults, all the host's, and all of a kind: the pipeline assumed every relation is functional. The walk
broke a multi-valued hop by retriever rank; the gate refused the second citizenship as a contradiction of the
first (so only one population value was ever stored, and the walk spoke it); and `covers()` let `2022` through
because a year is not a relation word. Three rules, each pinned:

1. **An ambiguous hop is a refusal.** `pipeline._walk`: several candidate facts with *different* objects for one
   hop → `reason="ambiguous_hop"`, the candidates named in `refused` (an ASK), never a pick by rank.
2. **The gate admits multi-valued facts and records the sibling** (`clash`); `functional=True` keeps the memory
   cortex's contradiction rule for a user-taught fact. Truth is decided at answer time, by rule 1.
3. **A number left over in the question is an unbound constraint** → not covered (`covers()` v3.1).

| after the three rules | before | **after** |
|---|---:|---:|
| harvest, 800 questions: verified / precision | 582 / 0.993 (4 wrong) | **574 / 1.000** (9 `ambiguous_hop`; the 4 wrong were all tie-breaks) |
| gen 2, outside the grammar (exp_r7 --exclude) | 18/116, arm A 1 wrong | **18/116, 0 wrong anywhere** (A 92/92) |
| held-out, clean | 196 correct, 1 wrong (a tie-break) | **195 correct, 0 wrong**, 2 ambiguous |
| held-out, poison after / before the truth | 194 correct / **119 wrong** | **197 ambiguous refusals / 197 ambiguous refusals, 0 wrong** |
| Wikidata arm, run 2 (same 600, cache) | 1 verified, 1 wrong | 66 fetched, 1,790 facts, **1,715 admitted**, 0 verified, **0 wrong** |

Every wrong answer this pipeline had ever spoken on the eval — the 4 in the harvest since the first day, gen
2's one on arm A, the held-out arm's one — was the walk choosing between several true facts. Refusing to
choose costs 8 verified answers on 800 and buys precision 1.000, and a poisoned source now yields a refusal
that names both facts and their provenance instead of an answer: retire-never-delete has something to act on.

**What the Wikidata arm says about SimpleQA now.** The store grew by 1,715 admitted facts about 66 entities and
none of the 66 questions verified: after the fetch the reasons are still `unknown_relation` (64) and
`retrieval_exhausted` (2). The store *holds* the answers — `1932-03-23 is the date of birth of Masaki Tsuji`
was admitted for `On what day, month, and year was Masaki Tsuji born?` — but the emitter named the relation in
the question's words (`day`, `month`, `year`; `go undefeated in all of his road races`), and `born` shares no
word with `date of birth`. The binding constraint on free text has moved for the third time today: from the
plan's shape (the disposer), to the store (this lever), to the **relation vocabulary the emitter names** against
the vocabulary a source states. That is gen 3's job — plans in the source's property labels — or a relation
alias layer the host owns (MoWM's "possible edge" oracle is the principled place for it). SimpleQA baseline
after lever 3: 0 verified, 0 wrong, 1,715 facts learned with provenance.

`exp_r11_search_learn_heldout{,_amb}.*`, `exp_r11_search_learn_wikidata{,_amb}.*`, `exp_m3_cot_pipeline_amb.*`,
`cot_harvest_amb.jsonl`, `exp_r7_emitter_planned_walk_gen2_amb.*`, `cot_harvest_r7_gen2_amb.jsonl`.

## Coverage, lever 4 — the source names the relation (and the first SimpleQA answers)

After lever 3 the store held the answers and the plans could not name them: the emitter says `born`, Wikidata
says `date of birth`, and no paraphrase tier bridges two wordings that share no word. The principled resolver
is the source itself: it resolves an entity label to an item, and it can resolve a relation wording to its
property labels the same way (`WikidataSource.relations("born")` → `["date of birth"]`, exact alias hits only —
`born` vs `born in` is not an alias). The host keeps the labels the store *holds*, rewrites the plan into that
wording, and records the translation (`LearnResult.aliased`; `answer(..., aliases=)` so `covers()` still sees
the original words in the question). Exactly one label must survive: `city` naming both `location` and
`located in the administrative territorial entity` is an `ambiguous_relation` refusal with the candidates,
never a pick; a wording the source cannot name (`day month year`, `first husband`) leaves the refusal as it was.
The step costs no fetch round; it runs after the facts about the seed were learned, since that is when the
label enters the vocabulary. `learn.resolve_relations`; 3 pins.

**Alias run 1** (same 600 SimpleQA, same cache): 3 plans rewritten, 2 verified — `established` → `inception` →
**2000, correct**, the first SimpleQA answer ever through search-and-learn; and `district` → `located in the
administrative territorial entity` → "Az-Zabdani Subdistrict" against the gold "Al-Zabadani". `born` →
`date of birth` resolved but never rewrote: the learned string `date of birth of Masaki Tsuji` had been split
as `date` | `birth of Masaki Tsuji`, because the wiki world reuses a relation `date` (2 facts), had never seen
`date of birth`, and lever 1b's rule prefers a known prefix. Right rule, wrong information: the source *knows*
where its relation ends, and the template string throws that away. So a source now hands over `Triple`s and
the store is told the relation before the string is split (`TripleIndex.declare_relation`,
`StoreRelations.declare`); pinned.

**Alias run 2:** 5 plans rewritten, **4 VM-verified: 3 correct, 0 false facts.** `1932-03-23` and `1983-02-16`
for two "on what day, month, and year was X born?" questions (the gold spells them "March 23, 1932"; the
experiment's match now normalizes dates deterministically — a normalizer, not a judge), `2000` for the award's
inception, and the one the strict metric counts wrong: the village is in *Az-Zabdani Subdistrict* (Wikidata's
label and granularity) and the gold says *Al-Zabadani* (the district). The fact is true; the disagreement is a
transliteration and one level of administrative granularity. 8 VM calls in total, 0 API calls (all cached).

Where free text stands at the end of the day, on the 600-question sample: 530 plans; 464 refused for coverage
(the emitter dropping a hop or a qualifier, or a shape the disposer does not read — the largest bucket, and the
emitter's); 65 fetched; 1,715 facts learned with provenance; 58 still refused after learning because the
relation the emitter named is not one a source states (`year` ×7 for "in what year did X marry Y" — the store
has `spouse` with a qualifier, which is not a hop; `first husband`, `episodes`, `go undefeated in all of his
road races`); 3 walks exhausted; 1 ambiguous relation; **4 verified, 0 false**. From 0 verified this morning to
4 is not a number to celebrate; the 0 in "0 false facts across everything the pipeline said today, on 800 +
1,600 + 4,326 + 600 questions" is.

`exp_r11_search_learn_wikidata_alias.*`, `exp_r11_search_learn_wikidata_alias2.*` (rows: every plan, first
and final reason, what was learned, what was rewritten).

## 2026-09-12 — lever 5, the synonym oracle (WordNet + WOLF), and two probes before the next builds

**Lever 5.** WOLF (WordNet Libre du Français) keys its French synsets by the Princeton WordNet 3.0 offset and
carries the French literals but not the English lemmas; WordNet 3.0's `dict/data.*` carry the lemmas by the
same offset. `standin/data/build_lexicon.py` joins them: 119,161 synsets, 56,475 with French literals, one
jsonl line each (both inputs off-repo). `cubbyllm/reasoning/lexicon.py` reads it with stdlib and is a second
relation resolver beside the source's in `learn.resolve_relations`: `birthplace` ↔ `place of birth`,
`conjoint` → `spouse`, `pays` → `country`. Exact-phrase tier only — a word's synonyms are never applied
word-by-word to a phrase (that is how `country` would become `state`) — and a polysemous wording that names two
held relations is still an `ambiguous_relation` refusal. French frame words joined the disposer's list. Pinned
(`test_lexicon.py`): a French question, `Quel est le lieu de naissance de Jean?`, planned in the question's
words, verifies against an English store end to end.

Measured on the 600-question SimpleQA sample (Wikidata source, cache): **4 verified, 3 correct, 0 false —
unchanged**, with 3 `ambiguous_relation` refusals instead of 1 (the lexicon fans polysemous words out to two
held relations; refused, correctly). On English factoids the source's own alias search already covers what
WordNet adds. The lever's yield is the bilingual axis, which no English benchmark measures; it is in the pins
and waits for French questions.

**Probe: the subdomain taxonomy as a router** (`exp_r12`; 160 subdomains, regex patterns, off-repo). It tags
96% of the eval questions and 89% of SimpleQA — but 3 to 9 tags each, `general` / `world` / `international`
on top. As a router it is too coarse to pick a world or a cortex by itself; it needs a specificity rule
(rarest matching subdomain wins, the catch-alls dropped) before it can decide anything. As a *domain tag on
harvest records* — the 2B runbook's tag-routed MoE upcycle, the adapter lifecycle — it is usable as is, and it
is deterministic, which the invariants like. Not wired anywhere today.

**Probe: DBpedia as a local Source** (`full-dbpedia`, BeIR's 4.6M abstracts, off-repo). Titles index in 8 s.
Exact-title coverage: 44% of the seeds the walk stalled on and 44% of SimpleQA's gold answers have an
abstract; 72% of abstracts open with an `X is a/an …` frame, 2.9% carry `(born DATE)`. That is the ceiling
of a regex-only local source with provenance and no model in the loop: the "who/what is X" and "when was X
born" shapes, offline. Anything beyond it is a fact extractor over prose, and a wrong extraction is a stored
falsehood — the place the MoWM possibility-branch gate belongs. Measured, not built.

**Next (agreed 2026-09-12): a frontier model as a proposer, never a judge.** (A) The ceiling probe: an
OpenRouter model in the emitter's seat on the same 600 SimpleQA questions, same disposer, walk, VM and kill
line — does a stronger proposer get plans through, and at 0 wrong? That decides whether the free-text holes
are the stand-in's capacity or the pipeline's. (B) The gen-3 dataset built the reverse way: certified chains
in, *questions* out (possessive, "have", relative, free text, French, synonyms), the plan never the model's,
`covers()` and the VM the filters, provenance on every record. The final model never calls OpenRouter.

`exp_r11_search_learn_wikidata_lex.*`, `exp_r12_local_sources_probe.*`.

## 2026-09-12 — the offline Source: frames over local articles, with provenance

Search-and-learn had one Source, the Wikidata API. The wiki world was built from local corpora, and the same
corpora are a Source if a sentence can be read as a fact *without a model in the loop*: `standin/wikitext.py`
runs regex frames over the opening of an article (600 characters — the lead, where encyclopedias state the
facts a factoid asks for) and hands back `Triple`s with the sentence they came from as provenance. Corpora:
the BeIR `full-dbpedia` abstracts (one parquet, 4.6M rows) and the twenty `wikipedia_dedup` domain files
(IT, business_management, cognition, … transport) — **4,704,261 titles**, indexed once per file
(`<file>.titles2.json`, 8 s for the parquet) and read by row group on demand. Frames, English: `(born DATE)`
and `(DATE – DATE)` → `date of birth` / `date of death`, `born … in PLACE` → `place of birth`,
`(founded|established|formed|incorporated) … in YEAR` → `inception`, `located in` / `capital` / `directed by`
/ `written by`. French: `née le`, `mort(e)/décédé(e) le`, `né(e) à`, `fondée en`, `est une commune … dans …`.
The source also names its own relations (`TRIGGERS`: `born` → `date of birth`, `established` → `inception`,
…), so lever 4's alias step works offline exactly as it does against the API. The `wiki_full` dump the 2B
manifest points at is on a drive that is not mounted on this machine; nothing here needs it.

**Run 1 stored a false fact.** 77 questions fetched, 11 facts, 10 admitted, 0 verified, 0 wrong *spoken* —
and `British is the author of The Roar` in the store. Two hazards, both in the source, both of the kind the
invariants exist to catch: the author frame accepted a single capitalised token, so a nationality read as a
name; and the title index dropped leading articles, so the seed `roar` opened the article *The Roar*. And
`aliased 0`: the source had no `relations()`, so every `born` plan died as `unknown_relation`. Fixes: person
frames need two or more capitalised tokens and no lowercase continuation; `title_key` keeps articles; the
trigger table above. All three pinned in `standin/tests/test_wikitext.py` with run 1's sentences.

**Run 2 (same 600 questions, same emitter, same disposer/walk/VM; `_wikitext2`):** 530 plans, 451 coverage
refusals (identical to the Wikidata runs — the disposer does not know which source is behind it), 77 fetched,
10 facts read, **9 admitted** (1 duplicate), 6 plans rewritten by the alias step, **2 verified, 2 correct,
0 wrong**, 0 API calls, 4 VM calls, 457 s. Every admitted fact re-read against its provenance sentence after
the run: nine dates and inceptions (Belafonte, Kiefer, the two O'Connors, Pereira da Silva, the Penny Crane
award, the V&A), all true. The two verified are the same two dates Wikidata verified; the Wikidata run's other
two (a 1932 birth date and the Az-Zabdani subdistrict) are facts the frames do not read — Wikidata states
thousands of relations, the frames read eleven. That is the trade: **offline, provenance per sentence, and
0 false facts, at half the API source's verified count.** 68 `unknown_relation` refusals remain the emitter's
plans naming relations no source states (`year` ×7, `first husband`, `episodes`).

The day's two hard resets (no bugcheck, no WHEA event) were a physically blocked GPU fan, found and cleared;
the runs above were relaunched after it and completed. `exp_r11_search_learn_wikidata_wikitext.*` (run 1),
`exp_r11_search_learn_wikidata_wikitext2.*` (run 2).

## 2026-09-12 — the ceiling probe, run 1: what a frontier proposer exposed (mostly about the host)

`google/gemini-3.8-flash` in the emitter's seat, same 600 questions, same disposer, walk, VM and kill line
(`standin/openrouter.py`, `exp_r11 --proposer openrouter:<id>`; the key stays in a gitignored file; the
serving model never calls it). 600 calls, 180k prompt + 133k completion tokens, **$0.63**, 28 minutes.
**121 plans, 479 no plan, 114 coverage refusals, 6 unknown relations, 7 fetched, 173 facts admitted,
0 verified, 0 wrong.** Zero verified is not the ceiling; it is two findings, both about the host.

*Finding 1 — the harness starved a thinking model.* The output budget was the stand-in's 300 tokens; Gemini
3.8 Flash spends ~210 of them thinking (125,583 reasoning tokens over 600 calls), so 45 answers hit
`MAX_TOKENS` and 71 more came back cut mid-JSON — 116 of the 479 "no plan" are truncations, not judgements.
The other 377 are the contract's explicit `{"seed": null, "hops": []}`: the model declined a chain for 63% of
SimpleQA, under a budget that left it little room to look for one. Fix: `--proposer-effort low
--proposer-max-tokens 1500`; the proposer now counts finish reasons, reasoning tokens and explicit declines
separately, so the next run reports what the model *decided* and what the wire lost.

*Finding 2 — the disposer is wording-bound in one direction.* Of the 121 plans, 114 were refused for
coverage, and reading them they are mostly right: `date of birth` for "on what day, month, and year was X
born", `date of death` for "die", `inception` for "founded" / "launched", `publication date` for "release".
A frontier model names relations by their canonical label, which is what the sources speak. The store
*holds* those labels, so lever 4 never fires (it resolves *unknown* wordings to labels); and the question
carries none of the label's words, so `covers()` cannot find the hop in it. The stand-in emitter never hit
this because it copies the question's words ("born"), reaches `unknown_relation`, and is aliased forward.
**Lever 6** (`learn.resolve_wordings`) is lever 4 in reverse: on a coverage refusal of a plan whose relations
the store holds, the host asks the resolvers (the source's property aliases, the lexicon, the frame triggers)
which wording *in the question* names each label — the question's content n-grams, shortest first, frame
words, the seed entity and numbers excluded — records the first that does as the label's alias, and walks
once more. A wording that names two held labels is refused as `ambiguous_relation`, the model's pick between
them being no evidence; the residual rule is untouched, so a dropped hop still fails. Pinned
(`test_search_learn.py`: covered and verified, ambiguous and refused, dropped hop still refused).

What run 1 did say about the ceiling: the 7 questions that reached a source fetched 192 facts (a frontier
proposer's seeds resolve to richer items — 94 facts for diazepam, 30 for barium sulfate), the 173 admitted
were all gate-clean, and the 5 relations still unknown after learning are identifiers Wikidata stores as
external IDs the source deliberately skips (`ChemSpider ID`, `KEGG ID`, `DOI`). Run 2 (effort low, 1500
tokens, lever 6, the emitter re-run under lever 6 beside it) is what the ceiling will be read from.
`exp_r11_search_learn_wikidata_gemini.*`.
