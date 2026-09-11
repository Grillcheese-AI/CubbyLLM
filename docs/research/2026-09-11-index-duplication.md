# Two TripleIndex implementations — one measured, one wired

**2026-09-11** · checked against the observed store (966 facts, 953 indexed), 1,041 hops.

## The situation

`validation/exp_m4_triple_lookup.py` defines its **own** `TripleIndex` class. So does
`cubbyllm/reasoning/index.py`. Every lookup number in the project — 83.8% of hops served,
coverage 0.646 → 0.710, +51 verified 0 lost — was measured with the **validation** copy.
The **package** copy is the one `pipeline._walk` actually calls.

That is the duplicated-implementation pattern `PACKAGE_LAYOUT.md` exists to prevent, and it
was worth checking before building a retriever on top of either.

## They agree — verified, not assumed

Both classes were built over the same store and their `hop()` outputs compared question by
question:

| | |
|---|---:|
| hop-0 calls compared | 763 |
| hop-1 calls compared | 278 |
| **divergences** | **0** |

So the measured lookup numbers **do** transfer to the wired component. That was the thing
worth knowing, and it is now known rather than hoped.

## Where they differ, and why it still matters

The difference is capability, not behaviour:

| | validation copy | package copy |
|---|---|---|
| `by_tail`, `by_subj` | yes | yes |
| `by_obj` / `by_object()` | **no** | yes (unused) |
| acceptance filter at hop 0 | none — returns every tail match | `accepts(plan, 0, None, t)` |
| dedup, `n_facts`, incremental `add()`, `__contains__` | no | yes |

The hop-0 acceptance difference is the live one: on this store every tail match also passed
`accepts`, so the two agreed. That is a property of the store, not a guarantee. **My check
ran on the observed store — 966 of the real 1,241 facts, 78%.** The remaining 22% could
contain a tail match that `accepts` rejects, and the two implementations would then part
company silently.

## Recommendation

Delete the duplicate: have `exp_m4_triple_lookup.py` import
`cubbyllm.reasoning.index.TripleIndex` rather than redefine it. The equivalence above makes
that safe today, and doing it now is what stops the measurement and the product drifting
apart later — which is precisely how the two sibling repos ended up with duplicated trunk
implementations.

Re-run the equivalence check on the **full** store before the swap, since mine was on 78%.
If they diverge there, the divergence is a finding in its own right: it would mean some
published lookup numbers were measured against a filter the shipped walk does not apply.
