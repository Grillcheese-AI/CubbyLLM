"""exp_r9_matched_pairs — the generality test: matched pairs on a corpus the emitter never saw.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Everything measured so far lives on one WikiKG-derived 800-question eval with a
1,241-fact store and 165 relations. The rung-1 memo's own falsifier says: if the
mechanism does not hold on a genuinely different corpus, throw the memo away.
This is that test, in the design the memo scored highest (Grok's matched pairs):

  * the WORLD is the wikikg-trajectories graph as served (standin/data/wikikg.py):
    552,297 template facts, 2,902 relation types, TripleIndex-only walk (no cosine
    encoder at this size), exp_g4b: 300/300 canonical two-hop questions VM-verified.
  * N chains whose BOTH hops are functional (one stored fact serves each), so the
    walk's end is the one gold answer. Each chain is issued in FOUR surface forms
    against the SAME store snapshot -- the canonical WH chain the grammar parses,
    and three shapes it does not:

        canonical   What is the P2 of the P1 of E?
        have        What P2 does the P1 of E have?             (grammar: 1-hop misparse)
        relative    What is the P2 of the thing that is the P1 of E?   (compound tail)
        possessive  E's P1 -- what is its P2?                  (unparseable)

  * two planners on every form: the GRAMMAR (planner.parse_question) and the
    EMITTER (gen 2, emitter_v12e, question-only prompt -> CotPlan). Both go through
    the same disposer (plan_verify, vocabulary = this world's relations), the same
    lookup walk, the same resident VM, the same tau floors. The emitter is a bare
    stand-in fine-tuned on 165 relations; here it meets ~2,900 it has never seen.

A win is attributable to QUESTION SHAPE and nothing else: same chains, same facts,
same gold, same gates. What is reported per form and per planner:

    plan / refused / walked / verified / correct / WRONG / VM calls

and the number that is the generality claim: on the three out-of-basin forms,
verified-correct answers the grammar cannot produce, at 0 wrong. A rising WRONG
count on any form is the kill: it means a gate let a wrong plan through.

  python validation/exp_r9_matched_pairs.py --n 200 --resident \
      [--gguf standin/models/emitter_v12e.Q4_K_M.gguf] [--paths 200000]
"""
from __future__ import annotations

import argparse, collections, json, pathlib, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BIND = re.compile(r'bind\s+frame\s*,\s*(?P<role>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"(?P<obj>(?:[^"\\]|\\.)*)"\s*;')
HOP = re.compile(r"^HOP(?P<hop>\d+)$")
ROLE = re.compile(r"^H(?P<hop>\d+)_(?P<rel>.+)$")


def forms(p1: str, p2: str, e: str) -> dict[str, str]:
    return {"canonical": f"What is the {p2} of the {p1} of {e}?",
            "have": f"What {p2} does the {p1} of {e} have?",
            "relative": f"What is the {p2} of the thing that is the {p1} of {e}?",
            "possessive": f"{e}'s {p1} -- what is its {p2}?"}


def emitted_plan(text: str, normalize):
    """CotPlan (SEED + HOPk) or CotChain (H-roles) -> (rels inner->outer, seed) or None."""
    seed, hops, roles = None, {}, []
    for m in BIND.finditer(text or ""):
        r, o = m.group("role"), m.group("obj")
        if r == "SEED" and seed is None:
            seed = o
        elif (h := HOP.match(r)):
            hops.setdefault(int(h.group("hop")), o)
        elif (h := ROLE.match(r)) and r not in roles:
            roles.append(r)
    if hops:
        rels = [hops[k] for k in sorted(hops)]
    elif roles:
        rels = [ROLE.match(r).group("rel").lower().replace("_", " ") for r in sorted(roles, key=lambda r: int(ROLE.match(r).group("hop")))]
    else:
        return None
    return [r.strip() for r in rels], (normalize(seed) if seed else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--paths", type=int, default=200_000)
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
    from cubbyllm.reasoning import answer, parse_question
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from cubbyllm.reasoning.plan_verify import StoreRelations, tail_relation

    # ---- the world, the questions ------------------------------------------------
    wk.ensure_data(("triplets", "paths"))
    tw = time.perf_counter()
    world = wk.wiki_world()
    import pyarrow.parquet as pq
    tbl = pq.read_table(wk.PATHS).slice(0, a.paths)
    paths = list(zip(tbl.column("entities").to_pylist(), tbl.column("relations").to_pylist(), tbl.column("directions").to_pylist()))
    chains = wk.chain_questions(paths, world.index, n=a.n, seed=a.seed)
    known = StoreRelations(world.index._seen)          # every fact the world holds, parsed
    log(f"world: {len(world)} facts, {len(world.index)} indexed, {len(known)} distinct relations | "
        f"{len(chains)} functional two-hop chains (seed {a.seed}) | built in {time.perf_counter() - tw:.0f}s")

    # the canonical question's plan gives P1, P2, E exactly as the store spells them
    items = []
    for c in chains:
        plan = parse_question(c["question"])
        p2 = plan.relations[1]; tr = tail_relation(plan.tail, known)
        if tr is None:
            continue
        e = plan.tail[len(tr):].strip()
        e = e[3:] if e.startswith("of ") else e
        items.append({"p1": tr, "p2": p2, "e": e, "gold": c["answer"], "facts": c["facts"]})
    log(f"matched items: {len(items)}  e.g. {forms(items[0]['p1'], items[0]['p2'], items[0]['e'])['have']!r} -> {items[0]['gold']!r}\n")

    # ---- the planners, the gates ---------------------------------------------------
    from emitter import LlamaCppEmitter
    from eval_emitter_vm import strip_fences
    em = LlamaCppEmitter(a.gguf)
    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = collections.Counter()
    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    def no_search(q, k):
        return []
    TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}     # the harvest's frame-size floors (lookup_vp_res)
    def walk(q, plan):
        return answer(q, no_search, run_fn, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), tau_ret=0.0,
                      top_k=3, max_repairs=1, lookup=world.lookup, known=known, plan=plan)

    def build_plan(rels, seed):
        if not rels or not seed:
            return None
        return QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))

    def seed_from(q, rels):
        """gen 1 fallback when a plan carries no SEED: the residual after the relations and frame words."""
        body = normalize(q)
        for r in sorted((normalize(r) for r in rels), key=len, reverse=True):
            body = body.replace(r, " ", 1)
        stop = set("what which where who is was does do did the a an of to in by for that thing its have has s".split())
        w = [x for x in body.split() if x not in stop]
        return " ".join(w) if w else None

    stats = {f: {p: collections.Counter() for p in ("grammar", "emitter")} for f in forms("", "", "")}
    wrong_ex, ok_ex = [], []
    t_em = 0.0
    for i, it in enumerate(items):
        for form, q in forms(it["p1"], it["p2"], it["e"]).items():
            # grammar
            st = stats[form]["grammar"]
            gplan = parse_question(q)
            if gplan is None:
                st["no_plan"] += 1
            else:
                st["plan"] += 1
                r = walk(q, gplan)
                if r.refused is not None: st["refused"] += 1
                elif r.verified:
                    st["walked"] += 1; st["verified"] += 1
                    ok = normalize(r.answer) == normalize(it["gold"]); st["correct" if ok else "WRONG"] += 1
                    if not ok and len(wrong_ex) < 6: wrong_ex.append(("grammar", form, q, r.answer, it["gold"]))
                else: st["walked"] += 1; st["walk_failed"] += 1
            # emitter
            st = stats[form]["emitter"]
            te = time.perf_counter()
            try:
                out = strip_fences(em.emit(q, max_new_tokens=a.max_new))
            except Exception as ex:                          # noqa: BLE001
                st["emit_error"] += 1; continue
            t_em += time.perf_counter() - te
            ep = emitted_plan(out, normalize)
            if ep is None:
                st["no_plan"] += 1; continue
            rels, seed = ep
            plan = build_plan(rels, seed or seed_from(q, rels))
            if plan is None:
                st["no_plan"] += 1; continue
            st["plan"] += 1
            if plan.n_hop == 2: st["two_hop"] += 1
            r = walk(q, plan)
            if r.refused is not None: st["refused"] += 1; st[f"refused:{r.reason}"] += 1
            elif r.verified:
                st["walked"] += 1; st["verified"] += 1
                ok = normalize(r.answer) == normalize(it["gold"]); st["correct" if ok else "WRONG"] += 1
                if not ok and len(wrong_ex) < 6: wrong_ex.append(("emitter", form, q, r.answer, it["gold"]))
                elif ok and form != "canonical" and len(ok_ex) < 4: ok_ex.append((form, q, r.answer))
            else: st["walked"] += 1; st["walk_failed"] += 1
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(items)} ({time.perf_counter() - t0:.0f}s, emitter {t_em:.0f}s, VM calls {calls['vm']})")

    log("\nMATCHED PAIRS on the wiki world -- same chains, same facts, same gold, same gates; only the surface form moves")
    log(f"{'form':12s}{'planner':9s}{'plan':>6}{'refused':>9}{'walked':>8}{'verified':>10}{'correct':>9}{'WRONG':>7}")
    for form in stats:
        for pl in ("grammar", "emitter"):
            s = stats[form][pl]
            log(f"{form:12s}{pl:9s}{s['plan']:6d}{s['refused']:9d}{s['walked']:8d}{s['verified']:10d}{s['correct']:9d}{s['WRONG']:7d}")
    oob = [f for f in stats if f != "canonical"]
    g_ok = sum(stats[f]["grammar"]["correct"] for f in oob); e_ok = sum(stats[f]["emitter"]["correct"] for f in oob)
    e_wrong = sum(stats[f]["emitter"]["WRONG"] for f in oob); g_wrong = sum(stats[f]["grammar"]["WRONG"] for f in oob)
    log(f"\nout-of-basin forms ({len(oob)} x {len(items)} questions): grammar correct {g_ok} (wrong {g_wrong}) | "
        f"emitter correct {e_ok} (wrong {e_wrong})")
    log(f"canonical: grammar correct {stats['canonical']['grammar']['correct']}/{len(items)} | emitter correct {stats['canonical']['emitter']['correct']}/{len(items)}")
    log(f"VM calls {calls['vm']} over {4 * len(items)} questions x 2 planners | emitter {t_em / max(1, 4 * len(items)):.2f}s per plan")
    for pl, form, q, ans, gold in wrong_ex:
        log(f"  WRONG [{pl}/{form}] {q[:80]!r} -> {ans!r} (gold {gold!r})")
    for form, q, ans in ok_ex:
        log(f"  ok    [emitter/{form}] {q[:80]!r} -> {ans!r}")
    for form in stats:
        s = stats[form]["emitter"]
        rs = {k: v for k, v in s.items() if k.startswith("refused:")}
        if rs: log(f"  emitter refusals on {form}: {rs}")
    if session: session.close()
    wall = time.perf_counter() - t0
    (LOGS / f"exp_r9_matched_pairs{a.tag}.json").write_text(json.dumps({
        "n_items": len(items), "world_facts": len(world), "relations": len(known), "gguf": a.gguf,
        "stats": {f: {p: dict(c) for p, c in d.items()} for f, d in stats.items()}, "vm_calls": calls["vm"],
        "wrong_examples": wrong_ex, "wall_s": round(wall, 1)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r9_matched_pairs{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r9_matched_pairs{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
