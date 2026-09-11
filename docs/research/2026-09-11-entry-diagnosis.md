# The store gap is confounding every hop-0 number I have quoted

**2026-09-11** · `validation/exp_r5_entry_diagnosis.py --observed` · 966 facts / 953 indexed,
195 failures surviving lookup-first. No GPU, no encoder, no corpus. **Wall 0.75s.**

## What I set out to measure

Two cheap fixes were on the table for hop-0 entry. Both are now measured and both are dead:

| candidate fix | recovers |
|---|---:|
| fuzzy tail — the same Jaccard ≥ 0.6 rule `relation_matches` already uses at hop k>0 | **5 / 195** |
| suffix backoff — drop leading `" of "` segments off the parsed tail | **1 / 195** |

The suffix idea had a real rationale: when `_split_chain` under-counts it swallows the joint,
so the tail it emits is too long and the correct one should be a *suffix* of it. It is not.
Depth histogram `{2: 1}` — one case, at depth 2. That hypothesis is dead.

## What I actually found

Decomposing the 118 no-seed cases by **where the seed entity lives in the index**:

| | | |
|---|---:|---:|
| seed entity IS a subject (relation mismatch) | 16 | 14% |
| seed entity only appears as an object (`by_object` territory) | 0 | 0% |
| **seed entity ABSENT from the index entirely** | **102** | **86%** |

**86% of entry failures are not a retrieval problem at all — the fact is not in the store I
was measuring against.** And that store is *my reconstruction*: 953 indexed facts assembled
from what surfaced in the harvest logs, against a real eval store of 1,241. 78% coverage.

So the absent bucket and the coverage gap are the same size, and I cannot separate them.

## The correction this forces

Every hop-0 number produced against the observed store is confounded, including the headline
I put in `2026-09-11-hop0-search-is-dead-code.md`:

> "The strictest test in the pipeline sits at the one point that fails **61%** of the time."

That 61% is measured on a store missing 22% of its facts. It is an upper bound on the real
entry-failure rate, not the rate. The script now refuses to let this pass quietly — it prints,
at runtime, whenever the absent bucket exceeds half:

```
*** The absent bucket dominates and this store is incomplete. Re-run
*** with --real before quoting any number above. ***
```

**`--real` must be run on your machine before any hop-0 number is quoted again.** It rebuilds
the eval store exactly as the harvest did (`m.load_sample(800, seed=0)`), which needs
`E:\valid_scaling_law_with_facts.pq` and `I:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz` —
1.1 GB, over the 400 MB device-bridge cap, so it cannot run here.

## What survives the confound

Three findings are **store-independent** and stand as measured:

1. **The hop-0 search fallback is dead code.** `accepts(plan, 0, …)` is the same exact string
   equality `_by_tail` is keyed on — 0 of 763. No store can change that; it is an identity.
2. **Fuzzy and suffix are not the fix.** Both are measured *within* the no-seed set, so the
   ratio is not affected by which facts are missing from it — 5 and 1 respectively.
3. **`by_object` is worth ~nothing at the entry.** Zero cases where the seed entity appears
   only as an object. (It still earns its place for backward hops; not for seeding.)

## The 16 relation mismatches — the informative residue

These are the cases where the entity *is* in the index and the relation is what misses. Every
one is the parse producing a **compound** relation where the store holds an **atomic** one:

```
want 'source that describes the instance'              have ['instance']
want 'award received by the director of photography'   have ['director of photography']
want 'languages spoken, written or signed by the child' have ['child']
want 'ethnic group of the ethnic group of the country'  have ['country']
want 'award received by the administrative territorial…' have ['administrative territorial entity']
```

The pattern is unmistakable: `_split_chain` splits on the literal `" of the "` and these
questions join their hops with `by the`, `that describes the`, `signed by the`. The splitter
under-counts, the hops collapse into one, and the collapsed relation is asked of the store as
if it were a single edge.

**That is a parser finding, not a retriever finding.** It points at the same place finding 3
of the hop-0 note pointed: tail *derivation* is the open problem, and the retriever cannot
fix it from downstream. What is new here is that it is now the *only* diagnosable bucket at
the entry — the other 86% is invisible until `--real` runs.

## Run it

```
python validation/exp_r5_entry_diagnosis.py --observed   # reproduces the above, 0.75s
python validation/exp_r5_entry_diagnosis.py --real       # the unconfounded answer
```

The `--real` mode writes `exp_r5_entry_diagnosis.{json,log}`; `--observed` writes
`exp_r5_entry_diagnosis_observed.{json,log}`, so the two can never be mistaken for each other
in the log directory.
