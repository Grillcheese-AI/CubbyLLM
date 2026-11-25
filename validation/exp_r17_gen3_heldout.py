"""exp_r17 -- the gen-3 held split through the gate: does an emitter plan free-text wordings over
entities it never trained on?

The gen-3 builder (standin/data/build_gen3.py) splits its certified questions by entity hash;
the `held` tenth is entities no training record used. Each held question goes: emitter (any
GGUF: gen 2, gen 3 masked, gen 3 full; or, order 0081, the 450M programs seat under each
``CB_EMIT_TRIE`` arm on the same rows) -> the emitted plan -> the same gate the builder used
(no source to fetch from, the local property table, coverage, the typed answer class, the
resident VM) over the same store (wiki world + the encyclopedia's facts). Scored: planned /
verified / correct (the VM's answer is the record's gold) / near / WRONG, and the refusal
reasons by wording. The kill line is the same: 0 wrong.

  python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v13e_masked.Q4_K_M.gguf --tag _masked
  python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v12e.Q4_K_M.gguf --tag _gen2
  <grilly2's python> validation/exp_r17_gen3_heldout.py --cubby450m <export> --emitter-adapter <dir> --seed 7
      (order 0081: arms off, rel, rel+slot; -> validation/logs/exp_0081_r17_s<seed>_<date>.{json,log})
"""
from __future__ import annotations

import argparse, collections, datetime, json, pathlib, random, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main() -> None:
    from cubbyllm.local_paths import data_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen3", default=str(data_dir("gen3_free_text.jsonl")))
    ap.add_argument("--encyclopedia", default=str(LOGS / "exp_r16_encyclopedia_all.json"))
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v12e.Q4_K_M.gguf"))
    ap.add_argument("--cubby450m", default=None, help="order 0081: the base export; the 450M seat replaces --gguf")
    ap.add_argument("--emitter-adapter", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--arms", default="off,rel,rel+slot", help="with --cubby450m: CB_EMIT_TRIE arms, on the same rows")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true", help="with --cubby450m: skip the rows the progress file holds")
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

    if a.cubby450m:                                      # order 0081: the 450M seat, each trie arm on the same rows
        import exp_0081_seat as seat81
        seat = seat81.build_seat(a.cubby450m, a.emitter_adapter, a.tokenizer, [world], known)
        arms = a.arms.split(",")
        what = f"cubby450m {seat.described['export']} + {seat.described['adapter']} ({seat.described['names']:,} names) on {seat.described['card']}"
    else:
        from emitter import LlamaCppEmitter
        em = LlamaCppEmitter(a.gguf)
        arms, what = ["gguf"], f"emitter {pathlib.Path(a.gguf).name}"
    log(f"gen-3 held split: {len(held):,} plan records, {len(rows)} sampled (seed {a.seed}) | store {len(world):,} facts "
        f"({n_enc:,} encyclopedia), {len(known):,} relations | {what} | arms {arms}")
    stem = (f"exp_0081_r17_s{a.seed}{a.tag}_{datetime.date.today().isoformat()}" if a.cubby450m
            else f"exp_r17_gen3_heldout{a.tag}")
    progress = LOGS / f"_exp_0081_r17_s{a.seed}{a.tag}.rows.jsonl"   # scratch: a crash resumes here (--resume)
    out_rows = [json.loads(l) for l in open(progress, encoding="utf-8")] if a.cubby450m and a.resume and progress.exists() else []
    if a.cubby450m and not a.resume:
        progress.write_text("", encoding="utf-8")
    done = {r["i"] for r in out_rows}
    for i, r in enumerate(rows):
        q, gold, w = r["prompt"], r["gold"], r.get("wording", "?")
        if i in done:
            continue
        row = {"i": i, "q": q, "gold": gold, "wording": w, "n_hop": r.get("n_hop"), "arms": {}}
        for arm in arms:
            if arm == "gguf":
                try:
                    e = {"plan": None}
                    ep = emitted_plan(strip_fences(em.emit(q, max_new_tokens=a.max_new)), normalize)
                    e["plan"], e["seed"] = (ep[0], ep[1]) if ep else (None, None)
                except Exception:                        # noqa: BLE001
                    e = {"plan": None, "refused": "emit_error"}
            else:
                e = seat81.emit(seat, q, arm, a.max_new, known)
            ep = (e["plan"], e.get("seed")) if e.get("plan") else None
            plan = QuestionPlan(relations=[None] + ep[0][1:], tail=f"{ep[0][0]} of {ep[1]}", n_hop=len(ep[0])) if ep and ep[0] and ep[1] else None
            keep = {k: e.get(k) for k in ("plan", "seed", "unheld", "trie", "ms", "why")}
            if plan is None:
                row["arms"][arm] = {"final": e.get("refused") or "no_plan", "planned": False, **keep}
                continue
            lr = learn_and_answer(q, no_search, run_fn, store=world, known=known, source=src,
                                  tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1, plan=plan)
            res = lr.result
            verdict = match(res.answer, gold, normalize) if res.verified else (res.reason or "failed")
            row["arms"][arm] = {**keep, "planned": True, "verified": bool(res.verified), "final": verdict,
                                "answer": res.answer, "aliased": lr.aliased, "trace": [h.fact for h in res.trace]}
        out_rows.append(row)
        if a.cubby450m:
            with open(progress, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if len(out_rows) % 50 == 0:
            log(f"  {len(out_rows)}/{len(rows)} ({time.perf_counter() - t0:.0f}s) " + " | ".join(
                f"{arm}: verified {sum(x['arms'][arm].get('verified', False) for x in out_rows)} "
                f"WRONG {sum(x['arms'][arm]['final'] == 'WRONG' for x in out_rows)}" for arm in arms))
    C = {arm: collections.Counter() for arm in arms}
    by_wording = {arm: collections.defaultdict(collections.Counter) for arm in arms}
    ms = {arm: [x["arms"][arm]["ms"] for x in out_rows if x["arms"][arm].get("ms") is not None] for arm in arms}
    wrong_ex, ok_ex = [], []
    for x in out_rows:
        for arm in arms:
            e, c, bw = x["arms"][arm], C[arm], by_wording[arm][x["wording"]]
            if not e["planned"]:
                c[e["final"]] += 1; bw[e["final"]] += 1
                continue
            c["plan"] += 1
            if e["verified"]:
                c["verified"] += 1; c[e["final"]] += 1; bw["verified"] += 1; bw[e["final"]] += 1
                (wrong_ex if e["final"] == "WRONG" else ok_ex).append(
                    (arm, x["q"], e["plan"], e["seed"], e["answer"], x["gold"], e["trace"]))
            else:
                c[f"final:{e['final']}"] += 1; bw[e["final"]] += 1
    session.close()
    for arm in arms:
        log(f"\n[{arm}] GEN-3 HELD SPLIT: {dict(sorted(C[arm].items()))}")
        if ms[arm]:
            log(f"[{arm}] emission ms: median {statistics.median(ms[arm]):.0f}, p95 {pct(ms[arm], 0.95):.0f}")
        log(f"[{arm}] by wording (verified / asked, then the refusals):")
        for w, v in sorted(by_wording[arm].items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(x for k, x in v.items() if k not in ("correct", "near", "WRONG"))
            log(f"  {v.get('verified', 0):4d} / {n:4d}  {w!r}  correct {v.get('correct', 0)} near {v.get('near', 0)} WRONG {v.get('WRONG', 0)} | "
                + ", ".join(f"{k} {x}" for k, x in v.items() if k not in ("verified", "correct", "near", "WRONG")))
    log(f"\nVM calls {calls['vm']}\nevery WRONG ({len(wrong_ex)}):")
    for arm, q, rels, seed, ans, gold, trace in wrong_ex:
        log(f"  [{arm}] {q!r} plan {rels} seed {seed!r} -> {ans!r} (gold {gold!r}) via {trace}")
    log(f"\nverified examples (first 8 of {len(ok_ex)}):")
    for arm, q, rels, seed, ans, gold, trace in ok_ex[:8]:
        log(f"  [{arm}] {q!r} plan {rels} seed {seed!r} -> {ans!r} (gold {gold!r})")
    wall = time.perf_counter() - t0
    out = {"emitter": what, "n": len(rows), "seed": a.seed, "arms": arms, "relations": len(known),
           "counts": {arm: dict(C[arm]) for arm in arms},
           "by_wording": {arm: {w: dict(v) for w, v in by_wording[arm].items()} for arm in arms},
           "emission_ms": {arm: {"median": statistics.median(ms[arm]), "p95": pct(ms[arm], 0.95)} for arm in arms if ms[arm]},
           "rows": out_rows, "vm_calls": calls["vm"], "wall_s": round(wall, 1)}
    if a.cubby450m:
        out["trie_options"] = seat81.options_described(seat, known)
        out["card"] = seat.described["card"]
    else:
        out["gguf"] = a.gguf
    (LOGS / f"{stem}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
