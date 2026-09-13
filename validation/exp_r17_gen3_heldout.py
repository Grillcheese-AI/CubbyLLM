"""exp_r17 -- the gen-3 held split through the gate: does an emitter plan free-text wordings over
entities it never trained on?

The gen-3 builder (standin/data/build_gen3.py) splits its certified questions by entity hash;
the `held` tenth is entities no training record used. Each held question goes: emitter (any
GGUF: gen 2, gen 3 masked, gen 3 full) -> the emitted plan -> the same gate the builder used
(no source to fetch from, the local property table, coverage, the typed answer class, the
resident VM) over the same store (wiki world + the encyclopedia's facts). Scored: planned /
verified / correct (the VM's answer is the record's gold) / near / WRONG, and the refusal
reasons by wording. The kill line is the same: 0 wrong.

  python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v13e_masked.Q4_K_M.gguf --tag _masked
  python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v12e.Q4_K_M.gguf --tag _gen2
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen3", default=str(ROOT / "standin" / "data" / "out" / "gen3_free_text.jsonl"))
    ap.add_argument("--encyclopedia", default=str(LOGS / "exp_r16_encyclopedia_all.json"))
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v12e.Q4_K_M.gguf"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import wikikg as wk
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

    held = [json.loads(l) for l in open(a.gen3, encoding="utf-8")]
    held = [r for r in held if r["task"] == "plan" and r["split"] == "held"]
    rows = random.Random(a.seed).sample(held, min(a.n, len(held)))
    wk.ensure_data(("triplets",))
    world = wk.wiki_world(); known = StoreRelations(world.index._seen)
    n_enc = 0
    if a.encyclopedia and pathlib.Path(a.encyclopedia).is_file():
        for r in json.loads(pathlib.Path(a.encyclopedia).read_text(encoding="utf-8"))["facts"]:
            fact = f"{r['obj']} is the {r['rel']} of {r['entity']}"
            if fact not in world:
                known.declare(r["rel"]); world.index.declare_relation(r["rel"]); world.add(fact); known.add(fact); n_enc += 1
    src = NullSource(PropertyAliases())
    session = cc.CubelangSession(exe=a.exe); calls = collections.Counter()

    def run_fn(source, fn):
        calls["vm"] += 1; return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    em = LlamaCppEmitter(a.gguf)
    log(f"gen-3 held split: {len(held):,} plan records, {len(rows)} sampled (seed {a.seed}) | store {len(world):,} facts "
        f"({n_enc:,} encyclopedia) | emitter {pathlib.Path(a.gguf).name}")
    c = collections.Counter(); by_wording: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    out_rows = []; wrong_ex = []; ok_ex = []
    for i, r in enumerate(rows):
        q, gold, w = r["prompt"], r["gold"], r.get("wording", "?")
        try:
            ep = emitted_plan(strip_fences(em.emit(q, max_new_tokens=a.max_new)), normalize)
        except Exception:                                # noqa: BLE001
            c["emit_error"] += 1; by_wording[w]["emit_error"] += 1; continue
        plan = QuestionPlan(relations=[None] + ep[0][1:], tail=f"{ep[0][0]} of {ep[1]}", n_hop=len(ep[0])) if ep and ep[0] and ep[1] else None
        if plan is None:
            c["no_plan"] += 1; by_wording[w]["no_plan"] += 1
            out_rows.append({"q": q, "gold": gold, "wording": w, "plan": None, "final": "no_plan"}); continue
        c["plan"] += 1
        lr = learn_and_answer(q, no_search, run_fn, store=world, known=known, source=src,
                              tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1, plan=plan)
        res = lr.result
        if res.verified:
            m = match(res.answer, gold, normalize)
            c["verified"] += 1; c[m] += 1; by_wording[w]["verified"] += 1; by_wording[w][m] += 1
            (wrong_ex if m == "WRONG" else ok_ex).append((q, ep[0], ep[1], res.answer, gold, [h.fact for h in res.trace]))
            verdict = m
        else:
            verdict = res.reason or "failed"; c[f"final:{verdict}"] += 1; by_wording[w][verdict] += 1
        out_rows.append({"q": q, "gold": gold, "wording": w, "plan": ep[0], "seed": ep[1], "aliased": lr.aliased,
                         "final": verdict, "answer": res.answer, "n_hop": r.get("n_hop")})
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s) plan {c['plan']} verified {c['verified']} "
                f"correct {c['correct']} near {c['near']} WRONG {c['WRONG']}")
    session.close()
    log(f"\nGEN-3 HELD SPLIT: {dict(sorted(c.items()))} | VM calls {calls['vm']}")
    log("\nby wording (verified / asked, then the refusals):")
    for w, v in sorted(by_wording.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(x for k, x in v.items() if k not in ("correct", "near", "WRONG"))
        log(f"  {v.get('verified', 0):4d} / {n:4d}  {w!r}  correct {v.get('correct', 0)} near {v.get('near', 0)} WRONG {v.get('WRONG', 0)} | "
            + ", ".join(f"{k} {x}" for k, x in v.items() if k not in ("verified", "correct", "near", "WRONG")))
    log(f"\nevery WRONG ({len(wrong_ex)}):")
    for q, rels, seed, ans, gold, trace in wrong_ex:
        log(f"  {q!r} plan {rels} seed {seed!r} -> {ans!r} (gold {gold!r}) via {trace}")
    log(f"\nverified examples (first 8 of {len(ok_ex)}):")
    for q, rels, seed, ans, gold, trace in ok_ex[:8]:
        log(f"  {q!r} plan {rels} seed {seed!r} -> {ans!r} (gold {gold!r})")
    wall = time.perf_counter() - t0
    out = {"gguf": a.gguf, "n": len(rows), "seed": a.seed, "counts": dict(c), "by_wording": {w: dict(v) for w, v in by_wording.items()},
           "rows": out_rows, "vm_calls": calls["vm"], "wall_s": round(wall, 1)}
    (LOGS / f"exp_r17_gen3_heldout{a.tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r17_gen3_heldout{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r17_gen3_heldout{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
