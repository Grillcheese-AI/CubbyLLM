"""exp_r5_entry_diagnosis — why does hop 0 fail to seed, and can anything cheap fix it?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Runs in two modes:
  --real       rebuild the eval store exactly as the harvest did (needs the .pq
               corpus and the fastword table). The unconfounded answer.
  --observed   rebuild an approximate store from facts that surfaced in the
               harvest logs. No corpus, no table, no GPU — but INCOMPLETE, and
               the incompleteness dominates the result (see below).

WHY BOTH MODES EXIST, AND WHY --real MATTERS
--------------------------------------------
Run in `--observed` mode on 2026-09-11 (953 of the real 1,241 facts, 78%):

    no-seed cases                          118
      seed entity IS a subject in the index   16  (14%)  relation mismatch
      seed entity only appears as an object    0  ( 0%)
      seed entity ABSENT entirely            102  (86%)  <-- dominates

**86% of the entry failures are the store gap.** So every hop-0 number measured
that way — including "61% of surviving failures never seed" — is confounded and
must be re-run here with `--real` before it is quoted again.

WHAT THE OBSERVED RUN DID ESTABLISH (store-independent)
-------------------------------------------------------
Two candidate cheap fixes at hop 0, both measured, both worth ~nothing:

    fuzzy tail (the same Jaccard >= 0.6 rule hop k>0 already uses)   5 / 195
    suffix backoff (drop ' of ' segments off the front of the tail)  1 / 195

The suffix idea came from the shape of the failure: when `_split_chain`
under-counts it swallows the joint, so the tail it produces is too long and the
correct one should be a suffix of it. It is not. That hypothesis is dead.

And one thing that is true regardless of the store: the hop-0 search fallback
cannot fire at all, because `accepts(plan, 0, ...)` is the same exact string
equality `TripleIndex._by_tail` is keyed on (0 of 763 questions, see
`docs/research/2026-09-11-hop0-search-is-dead-code.md`).

The 16 relation-mismatch cases are the informative residue: the parse produced a
COMPOUND relation where the store holds an atomic one --

    want 'source that describes the instance'   have ['instance']
    want 'award received by the director of ph' have ['director of photography']

which is misparse residue, not a retrieval problem.

  python validation/exp_r5_entry_diagnosis.py --observed   # no corpus needed
  python validation/exp_r5_entry_diagnosis.py --real       # the real answer
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time, types
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "validation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_W = ("from ..core.protocols import Wiring", 'class Wiring:\n    WIRED = "WIRED"')
_PLN = ("from .planner import QuestionPlan, Triple, accepts, normalize, parse_fact",
        "from planner_real import QuestionPlan, Triple, accepts, normalize, parse_fact")


def _load(rel, name, subs):
    src = (ROOT / rel).read_text(encoding="utf-8")
    for a, b in subs:
        src = src.replace(a, b)
    m = types.ModuleType(name); m.__dict__["__name__"] = name
    m.__dict__["__file__"] = str(ROOT / rel)      # find_cubelang_exe locates the sibling repo from it
    sys.modules[name] = m
    exec(compile(src, name + ".py", "exec"), m.__dict__)
    return m


def jaccard(pl, a: str, b: str) -> float:
    A, B = set(pl.normalize(a).split()), set(pl.normalize(b).split())
    return len(A & B) / len(A | B) if A and B else 0.0


def suffixes(tail: str):
    parts = tail.split(" of ")
    return [" of ".join(parts[i:]) for i in range(1, len(parts))]


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--real", action="store_true", help="rebuild the eval store from the corpus (needs .pq + table)")
    g.add_argument("--observed", action="store_true", help="approximate store from the harvest logs (no corpus)")
    ap.add_argument("--jaccard", type=float, default=0.6, help="the rule relation_matches already uses")
    a = ap.parse_args()
    if not (a.real or a.observed):
        a.observed = True
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    pl = _load("cubbyllm/reasoning/planner.py", "planner_real", [_W])
    ix = _load("cubbyllm/reasoning/index.py", "index_real", [_W, _PLN])
    norm = pl.normalize

    decomp = json.loads((LOGS / "exp_m3_exhaustion_decomp.json").read_text(encoding="utf-8"))
    lk = [json.loads(l) for l in (LOGS / "cot_harvest_lookup.jsonl").open(encoding="utf-8")]

    if a.real:
        import exp_m3_cot_pipeline as m
        _q, _ans, _h, _chains, store = m.load_sample(800, seed=0)
        mode, note = "real", "the eval store, rebuilt exactly as the harvest did"
    else:
        v3 = [json.loads(l) for l in (LOGS / "cot_harvest_v3cf.jsonl").open(encoding="utf-8")]
        s = set()
        for src in (v3, lk):
            for r in src:
                for h in (r.get("trace") or []):
                    if h.get("fact"): s.add(h["fact"])
                for hop in (r.get("candidates_topk") or []):
                    for _sc, txt in hop: s.add(txt)
        store = sorted(s)
        mode, note = "observed", "APPROXIMATE — facts that surfaced in the logs, NOT the whole store"
    T = ix.TripleIndex(store)
    log(f"mode {mode}: {len(store)} facts, {T.n_parsed} indexed — {note}")

    post = {r["question"]: r for r in lk}
    still = [x for x in decomp["details"]
             if x["question"] in post and not post[x["question"]].get("verified")]
    log(f"failures surviving lookup-first: {len(still)}\n")

    tails = list(T._by_tail.keys())
    c = Counter(); depth = Counter(); mism = []
    for x in still:
        plan = pl.parse_question(x["question"])
        if plan is None:
            c["unparseable"] += 1; continue
        if T.hop(plan, 0, None):
            c["already_seeds"] += 1; continue
        c["no_seed"] += 1

        if max((jaccard(pl, plan.tail, t) for t in tails), default=0.0) >= a.jaccard:
            c["fuzzy_tail_would_hit"] += 1
        for i, s in enumerate(suffixes(plan.tail), 1):
            if norm(s) in T._by_tail:
                c["suffix_backoff_would_hit"] += 1; depth[i] += 1; break

        e = norm(plan.tail.split(" of ")[-1].strip())
        if T._by_subj.get(e):
            c["entity_is_a_subject"] += 1
            if len(mism) < 8:
                want = plan.tail.rsplit(" of ", 1)[0] if " of " in plan.tail else plan.tail
                mism.append((want, sorted({t.rel for _f, t in T._by_subj[e]})[:3]))
        elif T._by_obj.get(e):
            c["entity_only_an_object"] += 1
        else:
            c["entity_absent"] += 1

    n = max(1, c["no_seed"])
    log("ENTRY FAILURE, DECOMPOSED")
    log(f"  already seed at hop 0              : {c['already_seeds']:>4}")
    log(f"  unparseable                        : {c['unparseable']:>4}")
    log(f"  NO SEED                            : {c['no_seed']:>4}\n")
    log("  could a cheap matcher fix it?")
    log(f"    fuzzy tail (Jaccard >= {a.jaccard})        : {c['fuzzy_tail_would_hit']:>4}  ({c['fuzzy_tail_would_hit']/n:.0%})")
    log(f"    suffix backoff                   : {c['suffix_backoff_would_hit']:>4}  ({c['suffix_backoff_would_hit']/n:.0%})"
        f"   depth {dict(sorted(depth.items()))}")
    log("\n  or is the fact simply not there?")
    log(f"    seed entity IS a subject         : {c['entity_is_a_subject']:>4}  ({c['entity_is_a_subject']/n:.0%})"
        f"   -> RELATION mismatch (misparse residue)")
    log(f"    seed entity only an object       : {c['entity_only_an_object']:>4}  ({c['entity_only_an_object']/n:.0%})"
        f"   -> direction; by_object territory")
    log(f"    seed entity ABSENT               : {c['entity_absent']:>4}  ({c['entity_absent']/n:.0%})"
        f"   -> {'STORE GAP — this mode cannot tell you more' if mode=='observed' else 'genuinely not in the store'}")
    if mism:
        log("\n  relation mismatches — what the parse wanted vs what the index holds:")
        for want, have in mism:
            log(f"    want {want[:52]!r}")
            log(f"      have {have}")
    if mode == "observed" and c["entity_absent"] / n > 0.5:
        log("\n  *** The absent bucket dominates and this store is incomplete. Re-run")
        log("  *** with --real before quoting any number above. ***")

    out = {"mode": mode, "store": len(store), "indexed": T.n_parsed,
           "surviving": len(still), "counts": dict(c), "suffix_depth": dict(depth),
           "jaccard": a.jaccard, "wall_s": time.perf_counter() - t0}
    tag = "" if mode == "real" else "_observed"
    (LOGS / f"exp_r5_entry_diagnosis{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (LOGS / f"exp_r5_entry_diagnosis{tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
