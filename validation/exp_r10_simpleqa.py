"""exp_r10_simpleqa — free text through the whole gate: SimpleQA on the wiki world.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

exp_r9 showed the emitter escapes the grammar's basin on templated forms the grammar
cannot parse, at 0 wrong. That is still OUR wording. SimpleQA (4,326 human-written
factoid questions, one short gold answer each) is nobody's template. exp_g4b measured
the lookup ceiling on this store: the gold is a graph entity for 860 questions, and a
stored fact "gold is the R of E" with E named in the question exists for THREE. So the
honest expectation is not accuracy -- it is the don't-know contract at scale:

    the emitter proposes a plan for every question it can; the disposer refuses every
    plan whose relations the store does not hold; the walk fails every plan the store
    cannot serve; the VM verifies only what the store says -- and nothing verified is
    wrong against the gold.

What is reported: emitted / no_plan / plan (1-hop, 2-hop, 3+) / seed names a graph
entity / relations all known / refused (by reason) / walked / walk_failed / verified /
correct / near / WRONG, split by whether the gold is even in the graph. Every verified
answer is printed in full, with its gold, because there will be few and each one is
either the mechanism working on free text or a gate that let something through.

    python validation/exp_r10_simpleqa.py --resident [--n 0 (all)] [--gguf ...]
"""
from __future__ import annotations

import argparse, collections, csv, io, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
SIMPLEQA = ROOT / "standin" / "data" / "out" / "simpleqa" / "simple_qa_test_set.csv"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="sample size (0 = all 4,326)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v12e.Q4_K_M.gguf"))
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning import answer
    from cubbyllm.reasoning.planner import QuestionPlan, normalize, parse_fact
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from exp_r9_matched_pairs import emitted_plan
    from emitter import LlamaCppEmitter
    from eval_emitter_vm import strip_fences

    # ---- the world, the questions ------------------------------------------------
    wk.ensure_data(("triplets",))
    tw = time.perf_counter()
    world = wk.wiki_world()
    known = StoreRelations(world.index._seen)
    entities: set[str] = set()
    for f in world.texts:
        t = parse_fact(f)
        if t is not None:
            entities.add(normalize(t.subj)); entities.add(normalize(t.obj))
    rows = list(csv.DictReader(io.StringIO(SIMPLEQA.read_text(encoding="utf-8"))))
    qcol = next(c for c in rows[0] if c.lower() in ("problem", "question"))
    acol = next(c for c in rows[0] if c.lower() in ("answer", "gold", "target"))
    if a.n:
        rows = random.Random(a.seed).sample(rows, a.n)
    for r in rows:
        r["_gold_in_graph"] = normalize(r[acol]) in entities
    n_gold = sum(r["_gold_in_graph"] for r in rows)
    log(f"world: {len(world)} facts, {len(known)} distinct relations, {len(entities)} distinct entity strings | "
        f"built in {time.perf_counter() - tw:.0f}s")
    log(f"SimpleQA: {len(rows)} questions ({'all' if not a.n else f'sample seed {a.seed}'}); gold is a graph entity for {n_gold}\n")

    # ---- the planner, the gates ------------------------------------------------------
    em = LlamaCppEmitter(a.gguf)
    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = collections.Counter()
    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    def no_search(q, k):
        return []
    TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
    def walk(q, plan):
        return answer(q, no_search, run_fn, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), tau_ret=0.0,
                      top_k=3, max_repairs=1, lookup=world.lookup, known=known, plan=plan)
    def build_plan(rels, seed):
        if not rels or not seed:
            return None
        return QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))

    def matches(ans: str, gold: str) -> str:
        x, g = normalize(ans), normalize(gold)
        if x == g: return "correct"
        if g and (g in x or x in g): return "near"
        return "WRONG"

    stats = {"gold_in_graph": collections.Counter(), "gold_not_in_graph": collections.Counter()}
    verified_ex, plan_ex, refused_ex = [], [], []
    t_em = 0.0
    for i, r in enumerate(rows):
        q, gold = r[qcol], r[acol]
        st = stats["gold_in_graph" if r["_gold_in_graph"] else "gold_not_in_graph"]
        st["questions"] += 1
        te = time.perf_counter()
        try:
            out = strip_fences(em.emit(q, max_new_tokens=a.max_new))
        except Exception:                                    # noqa: BLE001
            st["emit_error"] += 1; continue
        t_em += time.perf_counter() - te
        ep = emitted_plan(out, normalize)
        plan = build_plan(*ep) if ep else None
        if plan is None:
            st["no_plan"] += 1; continue
        rels, seed = ep
        st["plan"] += 1; st[f"plan_{min(plan.n_hop, 3)}hop"] += 1
        if seed in entities: st["seed_in_graph"] += 1
        if all(x in known for x in rels): st["rels_all_known"] += 1
        if len(plan_ex) < 8: plan_ex.append((q, rels, seed))
        res = walk(q, plan)
        if res.refused is not None:
            st["refused"] += 1; st[f"refused:{res.reason}"] += 1
            if len(refused_ex) < 6: refused_ex.append((q, rels, seed, res.reason, res.refused))
        elif res.verified:
            st["walked"] += 1; st["verified"] += 1
            m = matches(res.answer, gold); st[m] += 1
            verified_ex.append({"q": q, "plan": rels, "seed": seed, "answer": res.answer, "gold": gold,
                                "match": m, "gold_in_graph": r["_gold_in_graph"]})
        else:
            st["walked"] += 1; st["walk_failed"] += 1; st[f"walk_failed:{res.reason}"] += 1
        if (i + 1) % 250 == 0:
            done = stats["gold_in_graph"] + stats["gold_not_in_graph"]
            log(f"  {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s, emitter {t_em:.0f}s, VM calls {calls['vm']}) "
                f"plan {done['plan']} refused {done['refused']} verified {done['verified']} correct {done['correct']} "
                f"near {done['near']} WRONG {done['WRONG']}")

    # ---- the report ------------------------------------------------------------------
    total = stats["gold_in_graph"] + stats["gold_not_in_graph"]
    keys = ["questions", "no_plan", "plan", "plan_1hop", "plan_2hop", "plan_3hop", "seed_in_graph", "rels_all_known",
            "refused", "walked", "walk_failed", "verified", "correct", "near", "WRONG"]
    log("\nSIMPLEQA through the gate -- gen-2 emitter plans, disposer, lookup walk, resident VM, on the wiki world")
    log(f"{'':16s}{'all':>8}{'gold in graph':>15}{'gold not':>10}")
    for k in keys:
        log(f"{k:16s}{total[k]:8d}{stats['gold_in_graph'][k]:15d}{stats['gold_not_in_graph'][k]:10d}")
    log(f"\nrefusals: { {k[8:]: v for k, v in total.items() if k.startswith('refused:')} }")
    log(f"walk failures: { {k[12:]: v for k, v in total.items() if k.startswith('walk_failed:')} }")
    log(f"VM calls {calls['vm']} | emitter {t_em / max(1, len(rows)):.2f}s per question | "
        f"{1000 * (time.perf_counter() - t0) / max(1, len(rows)):.0f} ms per question end to end")
    log(f"\nthe line: verified {total['verified']}  correct {total['correct']}  near {total['near']}  WRONG {total['WRONG']}"
        f"  | answered-or-refused {total['refused'] + total['walk_failed'] + total['verified']} of {total['plan']} plans, "
        f"no plan {total['no_plan']}")
    log("\nevery verified answer:")
    for v in verified_ex:
        log(f"  [{v['match']:7s}] {v['q'][:90]!r}\n           plan {v['plan']} seed {v['seed']!r} -> {v['answer']!r} (gold {v['gold']!r})")
    log("\nplans the emitter proposed (first 8):")
    for q, rels, seed in plan_ex:
        log(f"  {q[:80]!r} -> {rels} seed {seed!r}")
    log("\nrefusals (first 6):")
    for q, rels, seed, reason, detail in refused_ex:
        log(f"  {q[:80]!r} -> {rels} seed {seed!r}: {reason} {detail}")
    if session: session.close()
    wall = time.perf_counter() - t0
    (LOGS / f"exp_r10_simpleqa{a.tag}.json").write_text(json.dumps({
        "n": len(rows), "gold_in_graph": n_gold, "world_facts": len(world), "relations": len(known), "gguf": a.gguf,
        "stats": {k: dict(v) for k, v in stats.items()}, "total": dict(total), "vm_calls": calls["vm"],
        "verified": verified_ex, "wall_s": round(wall, 1)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r10_simpleqa{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r10_simpleqa{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
