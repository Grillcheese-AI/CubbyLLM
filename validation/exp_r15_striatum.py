"""exp_r15_striatum -- proposer arbitration by reward prediction error, measured against the fixed order.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Three proposers on the matched pairs (exp_r9's 200 chains x 4 forms, the wiki world): the grammar,
the hippocampus (the 100 seen chains' canonical episodes, as in exp_r13), the gen-2 emitter. Every
proposer is tried on every question and its outcome and cost recorded (exhaustive, once). Then three
policies replay the same stream:

    fixed      grammar -> hippocampus -> emitter, until one certifies (today's order)
    striatum   best-first by expected value per shape, learning online from the gate's outcome
    oracle     the cheapest certifying proposer first (hindsight; the floor on cost)

Every policy certifies the same questions (each tries until one certifies), so the measure is COST:
proposals tried, emitter calls, seconds -- and the wrong count, audited against gold, which must stay
0 under every policy. The striatum's learned table is the second result: what it expects of each
proposer, per shape, after 800 questions. With `--simpleqa`, the SimpleQA 600 join the stream from
recorded rows (emitter: exp_r11 lev6b; hippocampus: exp_r14; frontier: exp_r11 gemini4).

  python validation/exp_r15_striatum.py --n 200 --resident [--simpleqa]
"""
from __future__ import annotations

import argparse, collections, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_r9_matched_pairs import emitted_plan, forms  # noqa: E402

COST_S = {"grammar": 0.0, "hippocampus": 0.05, "emitter": 1.0, "frontier": 3.0}     # per proposal, measured orders of magnitude


def replay(stream, policy, striatum=None, proposers=("grammar", "hippocampus", "emitter")):
    """`stream`: [(question, gold, {proposer: (outcome, correct_or_None, seconds)})]. Returns the tally."""
    from cubbyllm.reasoning.striatum import Striatum
    t = collections.Counter(); seconds = 0.0
    for q, gold, outcomes in stream:
        avail = [p for p in proposers if p in outcomes]
        if policy == "fixed":
            order = avail
        elif policy == "oracle":
            cert = [p for p in avail if outcomes[p][0] == "certified"]
            order = (sorted(cert, key=lambda p: outcomes[p][2])[:1] if cert else []) + [p for p in avail if p not in cert]
        else:
            order = striatum.order(q, avail)
        for p in order:
            outcome, correct, secs = outcomes[p]
            t["proposals"] += 1; t[f"tried:{p}"] += 1; seconds += secs
            if policy == "striatum":
                striatum.reward(q, p, outcome)
            if outcome == "certified":
                t["certified"] += 1; t[f"certified:{p}"] += 1
                if correct is False:
                    t["WRONG"] += 1
                    if policy == "striatum":
                        striatum.reward(q, p, "wrong", source="audit")
                break
        else:
            t["unanswered"] += 1
    t["seconds"] = round(seconds, 1)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v12e.Q4_K_M.gguf"))
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--simpleqa", action="store_true")
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning import answer, parse_question
    from cubbyllm.reasoning.hippocampus import Hippocampus
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from cubbyllm.reasoning.plan_verify import StoreRelations, tail_relation
    from cubbyllm.reasoning.striatum import Striatum, question_shape
    from emitter import LlamaCppEmitter
    from eval_emitter_vm import strip_fences

    wk.ensure_data(("triplets", "paths"))
    world = wk.wiki_world()
    import pyarrow.parquet as pq
    tbl = pq.read_table(wk.PATHS).slice(0, a.paths)
    paths = list(zip(tbl.column("entities").to_pylist(), tbl.column("relations").to_pylist(), tbl.column("directions").to_pylist()))
    chains = wk.chain_questions(paths, world.index, n=a.n, seed=a.seed)
    known = StoreRelations(world.index._seen)
    items = []
    for c in chains:
        plan = parse_question(c["question"]); p2 = plan.relations[1]; tr = tail_relation(plan.tail, known)
        if tr is None: continue
        e = plan.tail[len(tr):].strip(); e = e[3:] if e.startswith("of ") else e
        items.append({"p1": tr, "p2": p2, "e": e, "gold": c["answer"]})
    seen = items[0::2]
    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = collections.Counter()
    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
    def walk(q, plan):
        return answer(q, lambda _q, _k: [], run_fn, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), tau_ret=0.0,
                      top_k=3, max_repairs=1, lookup=world.lookup, known=known, plan=plan)
    def outcome_of(q, plan, gold):
        if plan is None: return ("refused", None)
        r = walk(q, plan)
        if r.verified: return ("certified", normalize(r.answer) == normalize(gold))
        return ("refused", None)

    # the hippocampus: the seen chains' canonical episodes, certified by the grammar (exp_r13 phase 0)
    hip = Hippocampus()
    for it in seen:
        q = forms(it["p1"], it["p2"], it["e"])["canonical"]; plan = parse_question(q); r = walk(q, plan) if plan else None
        if r is not None and r.verified and normalize(r.answer) == normalize(it["gold"]):
            hip.write(q, [it["p1"], it["p2"]], it["e"], [h.fact for h in r.trace], r.answer, {"experiment": "exp_r15"})
    em = LlamaCppEmitter(a.gguf)
    log(f"world {len(world)} facts | {len(items)} chains x 4 forms = {4 * len(items)} questions | hippocampus {len(hip)} episodes | "
        f"proposers grammar, hippocampus, emitter")

    # ---- exhaustive outcomes, once --------------------------------------------------------------
    stream = []; t_em = 0.0
    for i, it in enumerate(items):
        for form, q in forms(it["p1"], it["p2"], it["e"]).items():
            outs = {}
            t = time.perf_counter(); outs["grammar"] = (*outcome_of(q, parse_question(q), it["gold"]), time.perf_counter() - t)
            t = time.perf_counter(); cands = hip.propose(q, k=3)
            res = ("refused", None)
            for plan, _ep, _kind in cands:
                res = outcome_of(q, plan, it["gold"])
                if res[0] == "certified": break
            outs["hippocampus"] = (*res, time.perf_counter() - t)
            t = time.perf_counter()
            try:
                ep = emitted_plan(strip_fences(em.emit(q, max_new_tokens=a.max_new)), normalize)
            except Exception:                                           # noqa: BLE001
                ep = None
            plan = QuestionPlan(relations=[None] + ep[0][1:], tail=f"{ep[0][0]} of {ep[1]}", n_hop=len(ep[0])) if ep and ep[0] and ep[1] else None
            outs["emitter"] = (*outcome_of(q, plan, it["gold"]), time.perf_counter() - t); t_em += outs["emitter"][2]
            stream.append((q, it["gold"], outs))
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(items)} chains ({time.perf_counter() - t0:.0f}s, emitter {t_em:.0f}s, VM calls {calls['vm']})")
    per = {p: collections.Counter(o[p][0] for _q, _g, o in stream) for p in ("grammar", "hippocampus", "emitter")}
    wrong_any = {p: sum(1 for _q, _g, o in stream if o[p][0] == "certified" and o[p][1] is False) for p in per}
    log(f"\nexhaustive: " + " | ".join(f"{p}: certified {per[p]['certified']}, wrong {wrong_any[p]}" for p in per))

    # ---- SimpleQA from the record ---------------------------------------------------------------
    if a.simpleqa:
        by_q: dict[str, dict] = collections.defaultdict(dict); golds = {}
        def load_rows(name, proposer, secs):
            path = LOGS / f"{name}.json"
            if not path.is_file(): return 0
            d = json.loads(path.read_text(encoding="utf-8")); n = 0
            ver = {v["q"]: v for v in d.get("verified", [])}
            for row in d.get("rows", []):
                q = row["q"]; golds[q] = row["gold"]
                if row.get("verified"):
                    m = ver.get(q, {}).get("match"); by_q[q][proposer] = ("certified", m == "correct" or m == "near", secs)
                else:
                    by_q[q][proposer] = ("refused", None, secs)
                n += 1
            return n
        n1 = load_rows("exp_r11_search_learn_wikidata_lev6b", "emitter", COST_S["emitter"])
        n2 = load_rows("exp_r14_hippocampus_simpleqa_semantic", "hippocampus", COST_S["hippocampus"])
        n3 = load_rows("exp_r11_search_learn_wikidata_gemini4", "frontier", COST_S["frontier"])
        sq = [(q, golds[q], o) for q, o in by_q.items()]
        log(f"SimpleQA from the record: {len(sq)} questions (emitter rows {n1}, hippocampus rows {n2}, frontier rows {n3})")
        stream_all = stream + sq
        proposers = ("grammar", "hippocampus", "emitter", "frontier")
    else:
        stream_all = stream; proposers = ("grammar", "hippocampus", "emitter")

    # ---- the three policies on one stream -------------------------------------------------------
    results = {}
    for policy in ("fixed", "striatum", "oracle"):
        st = Striatum(seed=a.seed) if policy == "striatum" else None
        results[policy] = replay(stream_all, policy, st, proposers)
        if st is not None:
            striatum = st
    log(f"\n{'policy':10s}{'certified':>10}{'WRONG':>7}{'proposals':>11}{'emitter':>9}{'frontier':>9}{'hippo':>7}{'grammar':>9}{'seconds':>9}")
    for policy, t in results.items():
        log(f"{policy:10s}{t['certified']:10d}{t['WRONG']:7d}{t['proposals']:11d}{t['tried:emitter']:9d}{t['tried:frontier']:9d}"
            f"{t['tried:hippocampus']:7d}{t['tried:grammar']:9d}{t['seconds']:9.0f}")
    fx, sr = results["fixed"], results["striatum"]
    log(f"\nstriatum vs fixed: {fx['proposals'] - sr['proposals']} fewer proposals ({100 * (1 - sr['proposals'] / max(1, fx['proposals'])):.0f}%), "
        f"{fx['tried:emitter'] - sr['tried:emitter']} fewer emitter calls, {fx['seconds'] - sr['seconds']:.0f}s saved; certified {fx['certified']} vs {sr['certified']}; WRONG {fx['WRONG']} vs {sr['WRONG']}")
    log("\nwhat the striatum expects (proposer | shape -> expected reward, n), top 16 by n:")
    for k, v in sorted(striatum.expected.items(), key=lambda kv: -striatum.n[kv[0]])[:16]:
        log(f"  {k:48s} {v:+.2f}  (n={striatum.n[k]})")
    if session: session.close()
    wall = time.perf_counter() - t0
    striatum.save(LOGS / f"exp_r15_striatum{a.tag}.json")
    (LOGS / f"exp_r15_striatum{a.tag}_stream.json").write_text(json.dumps(
        {"n_items": len(items), "stream": [{"q": q, "gold": g, "outcomes": o} for q, g, o in stream_all],
         "results": {k: dict(v) for k, v in results.items()}, "vm_calls": calls["vm"], "wall_s": round(wall, 1)},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r15_striatum{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | VM calls {calls['vm']} | wrote exp_r15_striatum{a.tag}.{{json,log}} + _stream.json")


if __name__ == "__main__":
    main()
