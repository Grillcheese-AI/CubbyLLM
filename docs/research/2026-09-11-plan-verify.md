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

