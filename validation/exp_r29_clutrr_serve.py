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

MODELS = {
    "v13e": "emitter_v13e.Q4_K_M.gguf",
    "v14e": "emitter_v14e_nochain.Q4_K_M.gguf",
}

# THE PLANNER is the second dimension, and the one that carries the question.
# `grammar` is the shipped parser reading the question it was written for -- a
# CEILING, not a generalization result: the chain is spelled out in the words, so
# nothing has to be inferred. A model name puts the emitter in the loop, where
# the question is whether something trained on one to three hops can emit seven.
# The plan is emitted ONCE per (model, question) and reused across the shapes,
# which do not affect emission (exp_r25's pattern).


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
    ap.add_argument("--planners", default="grammar",
                    help="comma-separated: 'grammar' (the shipped parser -- a "
                         "ceiling) and/or model names from MODELS (the emitter in "
                         "the loop -- the generalization question)")
    ap.add_argument("--max-new", type=int, default=700,
                    help="a ten-hop program is ~12 bind lines; 300 truncates it")
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
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from build_gen3 import NullSource
    from emitter import LlamaCppEmitter
    from eval_emitter_vm import strip_fences
    from exp_r9_matched_pairs import emitted_plan
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

    planners = [p for p in a.planners.split(",") if p == "grammar" or p in MODELS]
    emitters: dict[str, object] = {}
    for p in planners:
        if p == "grammar":
            continue
        gguf = ROOT / "standin" / "models" / MODELS[p]
        if not gguf.is_file():
            log(f"!! {p}: missing {gguf.name}, dropping"); continue
        emitters[p] = LlamaCppEmitter(str(gguf))
    planners = [p for p in planners if p == "grammar" or p in emitters]
    log(f"planners: {', '.join(planners)}")

    # Emitted plans, cached per (planner, question): the shapes do not affect
    # emission, so emitting once and reusing is both faster and the only way the
    # two shapes are compared on the SAME plan rather than on two samples of it.
    plan_cache: dict[tuple[str, str], object] = {}
    emit_stats: dict[str, collections.Counter] = {p: collections.Counter() for p in planners}

    def plan_for(planner: str, q: str, n_hop: int):
        """None means 'let the grammar parse it'; False means the emitter failed."""
        if planner == "grammar":
            return None
        key = (planner, q)
        if key in plan_cache:
            return plan_cache[key]
        st = emit_stats[planner]
        try:
            ep = emitted_plan(strip_fences(emitters[planner].emit(q, max_new_tokens=a.max_new)),
                              normalize)
        except Exception:                                     # noqa: BLE001
            st["emit_error"] += 1; plan_cache[key] = False; return False
        if not (ep and ep[0] and ep[1]):
            st["no_plan"] += 1; plan_cache[key] = False; return False
        plan = QuestionPlan(relations=[None] + ep[0][1:], tail=f"{ep[0][0]} of {ep[1]}",
                            n_hop=len(ep[0]))
        # The number that IS the generalization result: did it emit a chain of the
        # right LENGTH? A model that has only ever seen 1-3 hops can copy relations
        # out of the question and still stop at three.
        st[f"hops:{len(ep[0])}"] += 1
        st["right_length" if len(ep[0]) == n_hop else "wrong_length"] += 1
        plan_cache[key] = plan
        return plan

    results: dict[str, dict[int, collections.Counter]] = {}
    arm_of: dict[str, str] = {}                     # results key -> arm name
    breaches: dict[str, list] = {}
    for planner in planners:
        for arm in [x for x in a.arms.split(",") if x in ARMS]:
            cfg = ARMS[arm]
            chunk, is_null = cfg["chunk"], cfg.get("null", False)
            key = arm if planner == "grammar" else f"{planner}/{arm}"
            results[key] = {}; arm_of[key] = arm; breaches[key] = []
            ta = time.perf_counter()
            for k, its in sorted(sample.items()):
                c = collections.Counter()
                for it in its:
                    q, gold = it["question"], it["gold"]
                    if is_null:
                        # The OUTERMOST relation -- the one the tail hop needs --
                        # becomes a token no fact in this store states. The rest of
                        # the chain is untouched, so a system that walks what it can
                        # and then refuses is behaving correctly; one that answers
                        # is not.
                        outer = it["chain"][-1][1]
                        q = q.replace(f"Who is the {outer} of ",
                                      f"Who is the {NULL_TOKEN} of ", 1)
                        if NULL_TOKEN not in q:
                            c["skipped"] += 1; continue
                        gold = None
                    plan = plan_for(planner, q, k)
                    if plan is False:               # the emitter produced nothing usable
                        c["refused"] += 1; c["why:no_plan"] += 1; continue
                    store = FactStore(name=f"clutrr:{it['id']}")
                    for f in it["facts"]:
                        store.add(f)
                    known = StoreRelations(store.index._seen)
                    try:
                        lr = learn_and_answer(q, no_search, run_fn, store=store,
                                              known=known, source=src,
                                              tau_vm=tau_for(k, chunk), top_k=3,
                                              max_repairs=1, chunk=chunk, plan=plan)
                    except Exception as e:                        # noqa: BLE001
                        c["harness_error"] += 1
                        if len(breaches[key]) < 3:
                            breaches[key].append({"error": repr(e)[:160], "q": q[:110]})
                        continue
                    res = lr.result
                    if not res.verified:
                        c["refused"] += 1; c[f"why:{res.reason or 'failed'}"] += 1
                        continue
                    c["verified"] += 1
                    if is_null:
                        c["BREACH"] += 1
                        if len(breaches[key]) < 6:
                            breaches[key].append({"q": q, "spoke": res.answer, "depth": k})
                    else:
                        m = match(res.answer, gold, normalize)
                        c[m] += 1
                        if m == "WRONG" and len(breaches[key]) < 6:
                            breaches[key].append({"q": q, "gold": gold,
                                                  "got": res.answer, "depth": k})
                results[key][k] = c
            log(f"  {key:<22} done in {time.perf_counter() - ta:.0f}s")

    session.close()

    # ---- the table: correct / refused / wrong, never accuracy ---------------
    for key, per_depth in results.items():
        chunk = ARMS[arm_of[key]]["chunk"]
        log(f"\n{key}  (tau = the bundle's, chunk={chunk})")
        log(f"  {'depth':>6}{'n':>5}{'correct':>9}{'refused':>9}{'WRONG':>7}"
            f"{'BREACH':>8}{'tau':>9}   top refusal reason")
        log("  " + "-" * 76)
        for k, c in sorted(per_depth.items()):
            n = c["correct"] + c["WRONG"] + c["refused"] + c["BREACH"]
            why = {x[4:]: v for x, v in c.items() if x.startswith("why:")}
            top = max(why.items(), key=lambda kv: kv[1])[0] if why else "-"
            log(f"  {k:>6}{n:>5}{c['correct']:>9}{c['refused']:>9}{c['WRONG']:>7}"
                f"{c['BREACH']:>8}{tau_for(k, chunk):>9.4f}   {top}")

    # ---- did the emitter emit a chain of the right LENGTH? -----------------
    # This is the generalization result, and it is separate from the score: a
    # model that copies relations out of the question but stops at three hops
    # fails here while looking fine on the depths it can reach.
    for p, st in emit_stats.items():
        if p == "grammar" or not st:
            continue
        hops = {int(x.split(":")[1]): v for x, v in st.items() if x.startswith("hops:")}
        log(f"\n{p} -- the plan it EMITTED (once per question, reused across shapes)")
        log(f"  right length {st['right_length']}, wrong length {st['wrong_length']}, "
            f"no plan {st['no_plan']}, emit errors {st['emit_error']}")
        log(f"  chain lengths emitted: {json.dumps(dict(sorted(hops.items())))}")

    # ---- the comparison the experiment exists for --------------------------
    deep: dict[str, dict[str, int]] = {}
    for planner in planners:
        pre = "" if planner == "grammar" else f"{planner}/"
        kw, kc = f"{pre}whole_chain", f"{pre}chunk2"
        if kw not in results or kc not in results:
            continue
        log(f"\nthe shape, side by side ({planner}) -- correct answers at each depth")
        log(f"  {'depth':>6}{'whole chain':>14}{'chunk 2':>10}{'gain':>8}")
        log("  " + "-" * 40)
        deep_w = deep_c = 0
        for k in sorted(sample):
            w = results[kw][k]["correct"]
            ch = results[kc][k]["correct"]
            if k >= 4:
                deep_w += w; deep_c += ch
            log(f"  {k:>6}{w:>14}{ch:>10}{ch - w:>+8}")
        log(f"  {'>=4':>6}{deep_w:>14}{deep_c:>10}{deep_c - deep_w:>+8}   "
            f"<- the depths no training record reaches")
        deep[planner] = {"whole_chain": deep_w, "chunk2": deep_c}

    totals = collections.Counter()
    for per in results.values():
        for c in per.values():
            for x in ("correct", "refused", "WRONG", "BREACH", "harness_error"):
                totals[x] += c[x]
    n_all = totals["correct"] + totals["refused"] + totals["WRONG"] + totals["BREACH"]
    log(f"\nacross every arm: {n_all} questions | correct {totals['correct']} | "
        f"refused {totals['refused']} | WRONG {totals['WRONG']} | "
        f"BREACH {totals['BREACH']} | harness errors {totals['harness_error']}")

    wrong, breach = totals["WRONG"], totals["BREACH"]
    if wrong or breach:
        verdict = f"KILLED (correctness clause): {wrong} wrong, {breach} null-control breaches"
    elif not deep:
        verdict = "survives (correctness): 0 wrong, 0 breaches"
    elif all(d["chunk2"] <= d["whole_chain"] for d in deep.values()):
        verdict = ("KILLED (rate clause): chunking gains nothing at depth under any "
                   "planner -- exp_r28 mis-read the constraint")
    else:
        gains = ", ".join(f"{p} +{d['chunk2'] - d['whole_chain']}"
                          for p, d in deep.items())
        verdict = (f"survives; 0 wrong in {n_all} questions, and the chunked shape "
                   f"gains at depth >= 4 under every planner ({gains})")
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
        "planners": planners, "max_new": a.max_new,
        # The emit-stat denominator. Sampling 288 records per arm yields fewer
        # DISTINCT questions -- two CLUTRR records can spell the same chain over
        # the same names -- and the plan cache keys on the question text, so
        # right_length + wrong_length + no_plan counts distinct questions, not
        # arm size. Recorded so the two numbers are never read against each other.
        "distinct_questions": len(plan_cache),
        "emit_stats": {p: dict(st) for p, st in emit_stats.items() if st},
        "totals": dict(totals), "deep_ge4": deep,
        "results": {arm: {k: dict(v) for k, v in per.items()}
                    for arm, per in results.items()},
        "verdict": verdict, "breaches": breaches,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
