"""exp_r29 -- CLUTRR through the serving path, by depth. WO-2.6, part two.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

WHAT THIS IS
------------
`exp_r28` measured the VM alone: gold triples straight into the program builder,
similarity against the never-bound control role, no retrieval and no emitter. It
found the depth limit -- the true binding halves per hop, meets the noise floor at
6 and crosses it at 7 -- and that **chunking the frame removes it**.

This runs the whole thing: question -> grammar -> walk -> VM -> spoken or refused,
on CLUTRR's compositional split, stratified by hop depth, under both program
shapes. `exp_r28` says what the VM *can* see; this says what the system *does*.

REPORTED AS CORRECT / REFUSED / WRONG, NEVER ACCURACY
-----------------------------------------------------
WO-2.6's own rule. 40% correct with 0 wrong and 60% refused is a different machine
from 40% correct with 60% wrong, and one number cannot tell them apart.

WHAT IS AND IS NOT BEING TESTED
--------------------------------
The store is the record's OWN certified graph, one tiny store per question, so
retrieval is not the variable -- the chain is. The question is the nested chain
CLUTRR's graph licenses ("Who is the brother of the grandson of Jason?"), parsed
by the shipped grammar with no hand-built plan, and its gold answer is the entity
at the end of the walk.

This is **CLUTRR-derived, not CLUTRR**. CLUTRR's own question names two entities
and asks for the composite relation between them -- a query type that needs a
kinship algebra the engine registry does not have (WO-2.6). What is kept is the
part that transfers: real graphs, a relation vocabulary the emitter never trained
on, and chain depths 2 to 10 where everything this repo has measured stops at 3.
Any number here is labelled as derived, in the log and in the write-up. It is not
a CLUTRR score and must never be posted as one.

THE ARMS
--------
    whole_chain    the shipped shape: one frame holds every hop
    chunk2         two hops per frame, tau moved to match (exp_r28's proposal)
    null_relation  the outermost relation replaced by a token with ZERO facts

`null_relation` is the kill-line control: the token is syntactically fine and
binds to nothing, so **any spoken answer is a breach**, not a scoring miss.

KILL CRITERION, two clauses
---------------------------
  * correctness -- any wrong answer at any depth, or any spoken answer on the
    null control, kills it outright. Depth must not buy wrongness.
  * rate -- chunk2 correct at depth >= 4 no better than whole_chain means the
    program shape is not the constraint after all and exp_r28 mis-read it.

Refusals are the expected honest failure and are recorded, not counted against
an arm.

    python validation/exp_r29_clutrr_serve.py --bench-root <dir> --n-per-depth 40
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
TAU_FALLBACK = 0.2202
NULL_TOKEN = "zubnog"                     # exp_r26's free token: no facts, anywhere

ARMS = {
    "whole_chain": {"chunk": 0},
    "chunk2": {"chunk": 2},
    "null_relation": {"chunk": 2, "null": True},
}


def tau_for(n_hop: int, chunk: int) -> float:
    """The frame holds `chunk` bindings under chunking, `n_hop` without -- so the
    threshold is the one for the BUNDLE, not for the chain. Passing chunk without
    moving tau measures nothing (exp_r28)."""
    k = min(chunk, n_hop) if chunk > 0 else n_hop
    return TAU_VM.get(k, TAU_FALLBACK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=None)
    ap.add_argument("--n-per-depth", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import clutrr as C
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import normalize
    from build_gen3 import NullSource
    from exp_r11_search_learn import match
    from sources import PropertyAliases
    from worlds import FactStore

    rng = random.Random(a.seed)
    raw = C.load_raw("test", root=a.bench_root)
    conv = C.verify_convention(raw)
    items = C.items("test", root=a.bench_root)
    by_depth: dict[int, list] = {}
    for it in items:
        by_depth.setdefault(it["hops"], []).append(it)
    log(f"CLUTRR gen_train23_test2to10/test: {len(raw)} records, {len(items)} simple "
        f"paths; convention forward {conv['forward']}/{conv['gendered_edges']}")
    log("available by depth: "
        + json.dumps({k: len(v) for k, v in sorted(by_depth.items())}))

    sample: dict[int, list] = {}
    for k in range(2, a.max_depth + 1):
        pool = by_depth.get(k, [])
        if pool:
            sample[k] = rng.sample(pool, min(a.n_per_depth, len(pool)))
    log("sampled: " + json.dumps({k: len(v) for k, v in sorted(sample.items())})
        + f"  ({sum(len(v) for v in sample.values())} questions per arm)\n")

    src = NullSource(PropertyAliases())
    session = cc.CubelangSession(exe=a.exe)

    def run_fn(source, fn):
        return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    results: dict[str, dict[int, collections.Counter]] = {}
    breaches: dict[str, list] = {}
    for arm in [x for x in a.arms.split(",") if x in ARMS]:
        cfg = ARMS[arm]
        chunk, is_null = cfg["chunk"], cfg.get("null", False)
        results[arm] = {}
        breaches[arm] = []
        ta = time.perf_counter()
        for k, its in sorted(sample.items()):
            c = collections.Counter()
            for it in its:
                q, gold = it["question"], it["gold"]
                facts = list(it["facts"])
                if is_null:
                    # The OUTERMOST relation -- the one the tail hop needs -- becomes a
                    # token no fact in this store states. The rest of the chain is
                    # untouched, so a system that walks what it can and then refuses
                    # is behaving correctly; a system that answers is not.
                    outer = it["chain"][-1][1]
                    q = q.replace(f"Who is the {outer} of ", f"Who is the {NULL_TOKEN} of ", 1)
                    if NULL_TOKEN not in q:
                        c["skipped"] += 1; continue
                    gold = None
                store = FactStore(name=f"clutrr:{it['id']}")
                for f in facts:
                    store.add(f)
                known = StoreRelations(store.index._seen)
                try:
                    lr = learn_and_answer(q, no_search, run_fn, store=store, known=known,
                                          source=src, tau_vm=tau_for(k, chunk), top_k=3,
                                          max_repairs=1, chunk=chunk)
                except Exception as e:                            # noqa: BLE001
                    c["harness_error"] += 1
                    if len(breaches[arm]) < 3:
                        breaches[arm].append({"error": repr(e)[:160], "q": q[:110]})
                    continue
                res = lr.result
                if not res.verified:
                    c["refused"] += 1; c[f"why:{res.reason or 'failed'}"] += 1
                    continue
                c["verified"] += 1
                if is_null:
                    c["BREACH"] += 1
                    if len(breaches[arm]) < 6:
                        breaches[arm].append({"q": q, "spoke": res.answer, "depth": k})
                else:
                    m = match(res.answer, gold, normalize)
                    c[m] += 1
                    if m == "WRONG" and len(breaches[arm]) < 6:
                        breaches[arm].append({"q": q, "gold": gold, "got": res.answer,
                                              "depth": k})
            results[arm][k] = c
        log(f"  {arm:<14} done in {time.perf_counter() - ta:.0f}s")

    session.close()

    # ---- the table: correct / refused / wrong, never accuracy ---------------
    for arm, per_depth in results.items():
        log(f"\n{arm}  (tau = the bundle's, chunk={ARMS[arm]['chunk']})")
        log(f"  {'depth':>6}{'n':>5}{'correct':>9}{'refused':>9}{'WRONG':>7}"
            f"{'BREACH':>8}{'tau':>9}   top refusal reason")
        log("  " + "-" * 76)
        for k, c in sorted(per_depth.items()):
            n = sum(c[x] for x in ("correct", "WRONG", "refused", "BREACH", "verified"))
            n = c["correct"] + c["WRONG"] + c["refused"] + c["BREACH"]
            why = {x[4:]: v for x, v in c.items() if x.startswith("why:")}
            top = max(why.items(), key=lambda kv: kv[1])[0] if why else "-"
            log(f"  {k:>6}{n:>5}{c['correct']:>9}{c['refused']:>9}{c['WRONG']:>7}"
                f"{c['BREACH']:>8}{tau_for(k, ARMS[arm]['chunk']):>9.4f}   {top}")

    # ---- the comparison the experiment exists for --------------------------
    if "whole_chain" in results and "chunk2" in results:
        log(f"\nthe shape, side by side -- correct answers at each depth")
        log(f"  {'depth':>6}{'whole chain':>14}{'chunk 2':>10}{'gain':>8}")
        log("  " + "-" * 40)
        deep_w = deep_c = 0
        for k in sorted(sample):
            w = results["whole_chain"][k]["correct"]
            ch = results["chunk2"][k]["correct"]
            if k >= 4:
                deep_w += w; deep_c += ch
            log(f"  {k:>6}{w:>14}{ch:>10}{ch - w:>+8}")
        log(f"  {'>=4':>6}{deep_w:>14}{deep_c:>10}{deep_c - deep_w:>+8}   "
            f"<- the depths no training record reaches")

    wrong = sum(c["WRONG"] for arm in results for c in results[arm].values())
    breach = sum(c["BREACH"] for arm in results for c in results[arm].values())
    if wrong or breach:
        verdict = f"KILLED (correctness clause): {wrong} wrong, {breach} null-control breaches"
    elif "whole_chain" in results and "chunk2" in results:
        verdict = ("survives; the chunked shape gains "
                   f"{deep_c - deep_w} correct at depth >= 4 with 0 wrong"
                   if deep_c > deep_w else
                   "KILLED (rate clause): chunking gains nothing at depth -- exp_r28 "
                   "mis-read the constraint")
    else:
        verdict = f"survives (correctness): 0 wrong, 0 breaches"
    log(f"\nVERDICT: {verdict}")

    for arm, ex in breaches.items():
        if ex:
            log(f"\n{arm} -- breaches / errors:")
            for e in ex:
                log("   " + json.dumps(e)[:200])

    stem = f"exp_r29_clutrr_serve{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "derived": True,
        "note": "CLUTRR-DERIVED chain questions, not CLUTRR's own relation-naming "
                "query. Never post as a CLUTRR score.",
        "n_per_depth": a.n_per_depth, "max_depth": a.max_depth, "seed": a.seed,
        "convention": conv, "sampled": {k: len(v) for k, v in sorted(sample.items())},
        "tau_table": TAU_VM, "tau_fallback": TAU_FALLBACK, "null_token": NULL_TOKEN,
        "results": {arm: {k: dict(v) for k, v in per.items()}
                    for arm, per in results.items()},
        "verdict": verdict, "breaches": breaches,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
