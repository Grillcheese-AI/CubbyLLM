# The hop-0 search fallback is dead code — and fuzzy matching is not the fix either

**2026-09-11** · proven over 763 parsed questions, observed store (966 facts / 953 indexed).
No GPU, no encoder, no corpus.

## 1. Search at hop 0 cannot return anything lookup missed

`planner.accepts` at hop 0 is exact string equality:

```python
if hop == 0:
    return normalize(f"{t.rel} of {t.subj}") == normalize(plan.tail)
```

`TripleIndex._by_tail` is keyed on **that same normalized string**. So the index lookup and
the acceptance filter are the same predicate, and the search path is gated by the filter.

Measured, score-free upper bound (every fact in the store, acceptance-filtered, compared
against the index's answer):

| | |
|---|---:|
| questions parsed | 763 |
| **search found something lookup did not** | **0** |
| identical | 763 |

**No scorer, no encoder and no `top_k` can change that number.** Building a better retriever
and dropping it into the hop-0 search slot would have moved nothing, and nothing in the
pipeline would have reported that it did nothing.

## 2. The asymmetry

| | test |
|---|---|
| **hop 0** | `normalize("rel of subj") == normalize(tail)` — **exact** |
| hop k>0 | `relation_matches()` = Jaccard ≥ 0.6 over words — **fuzzy**, plus exact subject |

The strictest test in the pipeline sits at the one point that fails 61% of the time. Later
hops already tolerate paraphrase; the entry does not.

> **CAVEAT added 2026-09-11 (later the same day).** That 61% is measured against the *observed*
> store — 953 indexed facts reconstructed from the harvest logs, against a real eval store of
> 1,241 (78% coverage). `exp_r5_entry_diagnosis --observed` then showed that **86% of the
> no-seed cases have the seed entity absent from the index entirely**, which is the same size
> as the coverage gap and cannot be separated from it. So 61% is an **upper bound** on the real
> entry-failure rate, not the rate. Run `exp_r5_entry_diagnosis.py --real` before quoting it
> again. Findings 1 and 3 below are unaffected — they are identities and within-set ratios.

## 3. But loosening hop 0 is not the fix — and this corrects the spec I wrote yesterday

`2026-09-11-retriever-spec.md` said the retriever "must be fuzzy on the tail, where the
question's surface form and the fact's differ." That was a guess. Measured:

> Of the 195 failures surviving lookup-first, applying **the same Jaccard ≥ 0.6 rule hop k>0
> already uses** to the tail recovers **5**.

Five. So the entry tails are not near-misses that a fuzzier comparator would catch — they are
**wrong**, which is a different problem with a different owner.

That reconnects to the parse: 75% of the 92 `misparsed_chain` failures die at hop 0
(`exp_r1_gate_diagnostics` Q3), and a misparsed chain produces a wrong tail by construction.
The tail is not phrased differently. It is the wrong tail.

## 4. What this means for the order of work

- **Do not** ship a fuzzy hop-0 comparator on the strength of the asymmetry alone. It is worth
  ~5 of 195 by the only rule already proven acceptable elsewhere in the pipeline.
- **Do** delete or gate the hop-0 search fallback, or make it reachable — right now it is dead
  code that looks like a safety net.
- **Do** treat tail *derivation* as the open problem. Getting the right tail out of the
  question is the parser's job, and the emitter's if the parser cannot be fixed — which is
  where the ladder pointed before the retriever detour, and the detour has now justified it
  with a number instead of an assumption.
- Still worth keeping from the spec: no threshold at the entry, score never gates, a miss must
  explain itself, `by_object` wired. Those are pinned and hold regardless.

## 5. What was built anyway, and why it still earns its place

`cubbyllm/reasoning/retriever.py` — the component that did not exist. The scorer is **injected**,
so it has no dependency on any encoder, table or corpus: validation passes a fastword cosine
closure, the tests pass a table of fixed scores, and it is the same object under test in both.
It unifies the three retrieval paths (lookup / backward / search) behind one seam, returns a
`HopResult` that always carries its candidates and a `reason` on a miss, and defaults `tau` to
`None` so nobody silently reinstates the threshold that lookup-first eliminated.

Contract: `validation/test_retriever_contract.py` — 19 pins, 0.06s, **10 mutations, 9 caught.**
The tenth ("default tau reinstated") was vacuous *because* of finding 1: no fixture reaches the
search path, since at hop 0 nothing can. The vacuous pin was how the dead code was found.
