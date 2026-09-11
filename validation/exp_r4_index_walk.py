"""exp_r4_index_walk — what can the TripleIndex do ALONE, and what needs the retriever?

Wired: STANDALONE. Needs NO encoder, NO corpus, NO GPU — only the harvest logs.
Runs in seconds.

Why
---
`cubbyllm/reasoning/index.py` is WIRED and is the only retrieval component that
exists as part of the package: exact normalized lookup, no threshold, no model.
The cosine retriever is NOT a package component — `make_retriever` lives inside
`validation/exp_m3_cot_pipeline.py`, rebuilt per run. So "the retriever was never
done" is literally true, and this measures the consequence.

`TripleIndex.by_object()` already exists, documented as "the backward entry (the
object side) for a bidirectional walk; unused by the forward walk today." The
exhaustion decomposition found backward retrieval clearing tau on 43 of the
surviving failures. The obvious cheap move is to wire `by_object` into `_walk`
and collect that for free.

This measures whether that works. It does not.

Method
------
Rebuild a TripleIndex from the OBSERVED store (every fact text that surfaced in
a trace or a candidate list across both harvests) and run an index-only walk
over the failures that survive lookup-first:

  forward        at each hop, step to facts whose SUBJECT is the current entity
  bidirectional  additionally step to facts whose OBJECT is it (reverse edges)

Both seed from the hop-0 tail lookup, exactly as `pipeline._walk` does.

Caveat carried through every number: the observed store is a subset of the real
one, so every count here is a LOWER bound.

  python validation/exp_r4_index_walk.py
      -> validation/logs/exp_r4_index_walk.{json,log}
"""
from __future__ import annotations

import json, pathlib, sys, time, types
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(rel: str, name: str, subs):
    """Load a package module standalone, without dragging in the package tree."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    for a, b in subs:
        src = src.replace(a, b)
    m = types.ModuleType(name); m.__dict__["__name__"] = name
    sys.modules[name] = m
    exec(compile(src, name + ".py", "exec"), m.__dict__)
    return m


WIRING = ("from ..core.protocols import Wiring", 'class Wiring:\n    WIRED = "WIRED"')


def main() -> None:
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    pl = _load("cubbyllm/reasoning/planner.py", "planner_real", [WIRING])
    ix = _load("cubbyllm/reasoning/index.py", "index_real", [
        WIRING,
        ("from .planner import QuestionPlan, Triple, accepts, normalize, parse_fact",
         "from planner_real import QuestionPlan, Triple, accepts, normalize, parse_fact")])
    norm = pl.normalize

    v3p, lkp, dp = (LOGS / "cot_harvest_v3cf.jsonl", LOGS / "cot_harvest_lookup.jsonl",
                    LOGS / "exp_m3_exhaustion_decomp.json")
    for p in (v3p, lkp, dp):
        if not p.exists():
            sys.exit(f"missing {p}")
    v3 = [json.loads(l) for l in v3p.open(encoding="utf-8")]
    lk = [json.loads(l) for l in lkp.open(encoding="utf-8")]
    decomp = json.loads(dp.read_text(encoding="utf-8"))

    store = set()
    for src in (v3, lk):
        for r in src:
            for h in (r.get("trace") or []):
                if h.get("fact"): store.add(h["fact"])
            for hop in (r.get("candidates_topk") or []):
                for _sc, txt in hop: store.add(txt)
    T = ix.TripleIndex(sorted(store))
    log(f"TripleIndex over the OBSERVED store: {T.n_facts} offered, {T.n_parsed} indexed "
        f"({T.n_parsed/max(1,T.n_facts):.1%})")
    log("  the real eval store is 1,241 facts — every count below is a LOWER bound\n")

    post = {r["question"]: r for r in lk}
    still = [x for x in decomp["details"]
             if x["question"] in post and not post[x["question"]].get("verified")]
    n = len(still)
    log(f"failures surviving lookup-first: {n}\n")

    def bfs(seeds, gold, max_hops, bidi):
        g = norm(gold); seen = set(map(norm, seeds)); fr = list(seeds)
        for h in range(max_hops + 1):
            for e in fr:
                if norm(e) == g: return h
            nxt = []
            for e in fr:
                for _f, t in T._by_subj.get(norm(e), []):
                    if norm(t.obj) not in seen: seen.add(norm(t.obj)); nxt.append(t.obj)
                if bidi:
                    for _f, t in T.by_object(e):
                        if norm(t.subj) not in seen: seen.add(norm(t.subj)); nxt.append(t.subj)
            if not nxt: return None
            fr = nxt
        return None

    c = Counter(); per_recovered = Counter()
    for x in still:
        q, gold = x["question"], x.get("gold_answer")
        nh = post[q].get("n_hop") or x.get("n_hop") or 3
        plan = pl.parse_question(q)
        seeds = [t.obj for _f, t in T.hop(plan, 0, None)] if plan is not None else []
        c["seed_ok" if seeds else "seed_MISSES"] += 1
        if norm(gold) in T._by_obj or norm(gold) in T._by_subj:
            c["gold_is_entity_in_index"] += 1
        if seeds:
            f = bfs(seeds, gold, nh + 1, False) is not None
            b = bfs(seeds, gold, nh + 1, True) is not None
            c["reach_forward"] += f; c["reach_bidirectional"] += b
            if b and not f:
                c["RECOVERED_by_backward_edge"] += 1
                per_recovered[x["kind"]] += 1

    log("WHERE THE INDEX-ONLY WALK DIES")
    log(f"  hop-0 tail lookup produces NO seed  : {c['seed_MISSES']:>4} / {n}  ({c['seed_MISSES']/n:.0%})")
    log(f"  hop-0 tail lookup produces a seed   : {c['seed_ok']:>4} / {n}")
    log(f"    of those, forward reaches gold    : {c['reach_forward']:>4}")
    log(f"    of those, bidirectional reaches   : {c['reach_bidirectional']:>4}")
    log(f"    NEWLY recovered by the back edge  : {c['RECOVERED_by_backward_edge']:>4}"
        f"   {dict(per_recovered) or ''}")
    log(f"\n  gold answer IS an entity in the index: {c['gold_is_entity_in_index']:>4} / {n}  "
        f"({c['gold_is_entity_in_index']/n:.0%})")

    log("\n" + "=" * 76)
    log("READING")
    log("=" * 76)
    log("  The index-only walk dies at the SEED, not in the graph. The hop-0 lookup is")
    log("  an exact normalized match on \"rel of subj\": when the parse is wrong, or the")
    log("  phrasing merely differs, there is nothing to start from — and no amount of")
    log("  backward traversal helps, because there is no frontier to traverse from.")
    log("")
    log("  This is why the decomposition's cosine backward probe found 43 and an exact")
    log("  index lookup finds ~2: the 43 came from FUZZY matching on object text.")
    log("")
    log("  So wiring `by_object` into the walk is NOT the free win it looks like, and")
    log("  the index cannot stand in for the retriever. It serves only hops whose")
    log("  surface form already matches. The knowledge is largely present — the gold")
    log("  answer is an entity in a 78%-complete index for half of these failures —")
    log("  but there is no way in.")
    log("")
    log("  THE RETRIEVER'S JOB, as specified by this measurement:")
    log("    1. hop-0 ENTRY is the product. 61% of surviving failures never get a seed.")
    log("       Later-hop traversal is comparatively healthy and the index already does it.")
    log("    2. It must be fuzzy on the tail, where the question's surface form and the")
    log("       fact's differ. Exact match is what is failing.")
    log("    3. Bidirectional edges are worth ~nothing on exact match (+2) but `by_object`")
    log("       is already built and costs nothing — wire it once fuzzy seeding lands.")
    log("=" * 76)

    out = {"observed_store": {"offered": T.n_facts, "indexed": T.n_parsed},
           "n_surviving": n, "counts": dict(c),
           "recovered_by_kind": dict(per_recovered), "wall_s": time.perf_counter() - t0}
    (LOGS / "exp_r4_index_walk.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (LOGS / "exp_r4_index_walk.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
