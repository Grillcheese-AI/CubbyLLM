"""exp_r13_hippocampus -- the episodic side cortex on the matched pairs: does a certified chain,
remembered, give the out-of-basin forms and the unseen entities -- at 0 wrong?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Same world, chains, forms and gates as exp_r9 (552,297 facts; N functional two-hop
chains x four surface forms; the disposer, the lookup walk, the resident VM, the tau
floors). Two planners were measured there: the grammar (out of basin: 0) and the gen-2
emitter (out of basin: 228 of 600, possessive 21 of 200 -- it folds the inner hop into
the seed on a shape it never saw). This adds a third proposer that is not a model at
all: the hippocampus (cubbyllm/reasoning/hippocampus.py), an explicit store of chains
the VM certified, recalled by the question's content words.

  Phase 0  the SEEN half (S, every other chain): the canonical question is planned by
           the grammar and walked; the chains the VM certifies are written as episodes.
  Phase A  S, the three out-of-basin forms: the hippocampus proposes (its recalled plan
           first, the rebound shape second), each candidate through the same gate;
           first verified answer wins. The question: does memory of the canonical
           chain give have / relative / possessive at 0 wrong?
  Phase B  the UNSEEN half (U), all four forms: no episode names these entities. What
           can be recalled is the SHAPE (P1, P2) of another entity's chain; the
           hippocampus rebinds it to the entity the question names. The question: does
           analogical transfer through the disposer stay at 0 wrong?

Reported per phase and form: candidates / refused / verified / correct / WRONG, which
kind of candidate verified (recalled | rebound), and the recall distances. The kill
line is the same as everywhere: a rising WRONG count.

  python validation/exp_r13_hippocampus.py --n 200 --resident [--k 3] [--paths 200000]
"""
from __future__ import annotations

import argparse, collections, json, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_r9_matched_pairs import forms  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=3, help="episodes recalled per question")
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
    from cubbyllm.reasoning.planner import normalize
    from cubbyllm.reasoning.plan_verify import StoreRelations, tail_relation

    wk.ensure_data(("triplets", "paths"))
    world = wk.wiki_world()
    import pyarrow.parquet as pq
    tbl = pq.read_table(wk.PATHS).slice(0, a.paths)
    paths = list(zip(tbl.column("entities").to_pylist(), tbl.column("relations").to_pylist(), tbl.column("directions").to_pylist()))
    chains = wk.chain_questions(paths, world.index, n=a.n, seed=a.seed)
    known = StoreRelations(world.index._seen)
    items = []
    for c in chains:
        plan = parse_question(c["question"])
        p2 = plan.relations[1]; tr = tail_relation(plan.tail, known)
        if tr is None:
            continue
        e = plan.tail[len(tr):].strip()
        e = e[3:] if e.startswith("of ") else e
        items.append({"p1": tr, "p2": p2, "e": e, "gold": c["answer"], "facts": c["facts"]})
    seen, unseen = items[0::2], items[1::2]
    log(f"world: {len(world)} facts, {len(known)} relations | {len(items)} matched chains (seed {a.seed}): "
        f"{len(seen)} seen, {len(unseen)} unseen | k={a.k}")

    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = collections.Counter()
    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
    def walk(q, plan):
        return answer(q, lambda _q, _k: [], run_fn, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), tau_ret=0.0,
                      top_k=3, max_repairs=1, lookup=world.lookup, known=known, plan=plan)

    # ---- phase 0: certify the seen chains, write the episodes -------------------
    hip = Hippocampus(); certified = 0
    for it in seen:
        q = forms(it["p1"], it["p2"], it["e"])["canonical"]
        plan = parse_question(q)
        r = walk(q, plan) if plan is not None else None
        if r is not None and r.verified and normalize(r.answer) == normalize(it["gold"]):
            hip.write(q, [it["p1"], it["p2"]], it["e"], [h.fact for h in r.trace], r.answer,
                      {"world": "wikikg", "planner": "grammar", "experiment": "exp_r13", "seed": a.seed})
            certified += 1
    log(f"phase 0: {certified}/{len(seen)} canonical chains VM-certified and written as episodes "
        f"({time.perf_counter() - t0:.0f}s, VM calls {calls['vm']})")
    shapes = {tuple(e.relations) for e in hip.episodes}
    shared = sum(1 for it in unseen if (normalize(it["p1"]), normalize(it["p2"])) in shapes)
    log(f"         {len(shapes)} distinct chain shapes remembered; {shared}/{len(unseen)} unseen chains have a "
        f"remembered shape (the ceiling for rebinding)")

    # ---- phases A and B ---------------------------------------------------------
    def run_phase(name, its, form_names):
        stats = {f: collections.Counter() for f in form_names}
        dists = {f: [] for f in form_names}
        wrong_ex, ok_ex = [], []
        for it in its:
            for form, q in forms(it["p1"], it["p2"], it["e"]).items():
                if form not in form_names:
                    continue
                st = stats[form]
                cands = hip.propose(q, k=a.k)
                if not cands:
                    st["no_candidate"] += 1; continue
                st["planned"] += 1
                rec = hip.recall(q, k=1)
                if rec: dists[form].append(rec[0][1])
                done = False
                for plan, ep, kind in cands:
                    st["candidates"] += 1
                    r = walk(q, plan)
                    if r.refused is not None:
                        st[f"refused:{r.reason}"] += 1; continue
                    if r.verified:
                        st["verified"] += 1; st[f"verified:{kind}"] += 1
                        ok = normalize(r.answer) == normalize(it["gold"])
                        st["correct" if ok else "WRONG"] += 1
                        if ok: hip.reinforce(ep)
                        if not ok and len(wrong_ex) < 8: wrong_ex.append((name, form, kind, q, r.answer, it["gold"], ep.question))
                        elif ok and len(ok_ex) < 3 and kind == "rebound": ok_ex.append((name, form, q, r.answer, ep.question))
                        done = True; break
                    st["walk_failed"] += 1
                if not done:
                    st["unanswered"] += 1
        return stats, dists, wrong_ex, ok_ex

    oob = ["have", "relative", "possessive"]
    sa, da, wa, oa = run_phase("A", seen, oob)
    log(f"phase A done ({time.perf_counter() - t0:.0f}s, VM calls {calls['vm']})")
    sb, db, wb, ob = run_phase("B", unseen, ["canonical"] + oob)
    log(f"phase B done ({time.perf_counter() - t0:.0f}s, VM calls {calls['vm']})")

    def table(title, stats, dists, n):
        log(f"\n{title}")
        log(f"{'form':12s}{'planned':>8}{'cands':>7}{'refused':>9}{'verified':>10}{'recalled':>10}{'rebound':>9}{'correct':>9}{'WRONG':>7}   recall dist (min/median/max)")
        for f, s in stats.items():
            refused = sum(v for k, v in s.items() if k.startswith("refused:"))
            d = dists[f]
            ds = f"{min(d)}/{statistics.median(d):.0f}/{max(d)}" if d else "-"
            log(f"{f:12s}{s['planned']:8d}{s['candidates']:7d}{refused:9d}{s['verified']:10d}{s['verified:recalled']:10d}"
                f"{s['verified:rebound']:9d}{s['correct']:9d}{s['WRONG']:7d}   {ds}   (of {n})")
        for f, s in stats.items():
            rs = {k[8:]: v for k, v in s.items() if k.startswith("refused:")}
            if rs: log(f"  refusals on {f}: {rs}")

    table(f"PHASE A -- seen chains ({len(seen)}), out-of-basin forms: memory of the canonical chain", sa, da, len(seen))
    table(f"PHASE B -- unseen chains ({len(unseen)}), all forms: another entity's shape, rebound", sb, db, len(unseen))
    a_ok = sum(sa[f]["correct"] for f in oob); a_wrong = sum(sa[f]["WRONG"] for f in oob)
    b_ok = sum(sb[f]["correct"] for f in sb); b_wrong = sum(sb[f]["WRONG"] for f in sb)
    log(f"\nout of basin, seen: correct {a_ok} of {3 * len(seen)} (WRONG {a_wrong}) | "
        f"unseen, all forms: correct {b_ok} of {4 * len(unseen)} (WRONG {b_wrong}) | "
        f"exp_r9 same forms, emitter: 228 of 600 out of basin (possessive 21/200), grammar 0")
    log(f"episodes: {len(hip)} live, utility>0: {sum(1 for e in hip.episodes if e.utility)} | VM calls {calls['vm']}")
    for ex in wa + wb:
        log(f"  WRONG [{ex[0]}/{ex[1]}/{ex[2]}] {ex[3][:70]!r} -> {ex[4]!r} (gold {ex[5]!r}) via episode {ex[6][:60]!r}")
    for ex in oa + ob:
        log(f"  ok    [{ex[0]}/{ex[1]}/rebound] {ex[2][:70]!r} -> {ex[3]!r} via episode {ex[4][:60]!r}")
    if session: session.close()
    wall = time.perf_counter() - t0
    hip.save(LOGS / f"exp_r13_episodes{a.tag}.jsonl")
    (LOGS / f"exp_r13_hippocampus{a.tag}.json").write_text(json.dumps({
        "n_items": len(items), "seen": len(seen), "unseen": len(unseen), "certified": certified, "k": a.k,
        "phase_a": {f: dict(s) for f, s in sa.items()}, "phase_b": {f: dict(s) for f, s in sb.items()},
        "dist_a": da, "dist_b": db, "wrong": wa + wb, "vm_calls": calls["vm"], "wall_s": round(wall, 1)},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r13_hippocampus{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r13_hippocampus{a.tag}.{{json,log}} + exp_r13_episodes{a.tag}.jsonl")


if __name__ == "__main__":
    main()
