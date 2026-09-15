"""
Wired: STANDALONE (validation script; never imported by cubbyllm/).

WO-0.1 — the emitter ablation. Two model competitions (2026-09-14) independently
returned the same warning, and it is the one thing no existing instrument can see:

    "The emitter becomes decorative. The kill line reads green -- every spoken
    answer is verified true against gated facts -- while the emitted program
    contributes nothing to any answer. Every instrument measures ANSWER
    CORRECTNESS, and answer correctness is exactly what the host guarantees
    independently of the emitter."   -- glm-5.3, Q5

So: hold the world, the chains, the gold, the gates and the VM fixed, vary ONLY
where the plan came from, and read the gap.

Four arms on the identical question:

  emitter      the real gen-2 plan (the baseline being tested)
  grammar      planner.parse_question -- the deterministic host path, i.e. what
               the answer rate would be with no model at all
  trivial      a fixed constant plan, the same for every question -- the floor
  shuffled     THE DECISIVE ARM. The emitter's own relations, taken from a
               DIFFERENT item, with this item's correct seed.

`shuffled` is the arm that matters and a constant program is not a substitute for
it. A constant plan obviously fails, which proves nothing; the informative test is
whether a *plausible, well-formed, emitter-produced* plan for the wrong question
does as well as the right one. If it does, the emitter is not choosing relations
for THIS question -- it is producing well-shaped output the host then rescues, and
every generalization claim about it is unfalsifiable.

Holding the seed correct is deliberate: it isolates the emitter's actual
contribution (which relations, in which order) from the part the host could
recover anyway.

    ABLATION GAP = emitter correct - shuffled correct

Read it against the eval set's noise floor, which this script prints. A gap inside
the noise floor for two consecutive rounds is the alarm.

    python validation/exp_r22_emitter_ablation.py --n 60 --resident

A rising WRONG count on any arm is a separate kill: a gate let a bad plan through.
"""
from __future__ import annotations

import argparse, collections, json, math, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_r9_matched_pairs import emitted_plan, forms       # noqa: E402  (same harness, same shapes)

ARMS = ("emitter", "grammar", "trivial", "shuffled")


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% Wilson interval -- the eval set's noise floor, so the
    ablation gap can be read against something rather than eyeballed."""
    if n <= 0:
        return 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(half, abs(centre - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
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

    wk.ensure_data(("triplets", "paths"))
    tw = time.perf_counter()
    world = wk.wiki_world()
    import pyarrow.parquet as pq
    tbl = pq.read_table(wk.PATHS).slice(0, a.paths)
    paths = list(zip(tbl.column("entities").to_pylist(), tbl.column("relations").to_pylist(),
                     tbl.column("directions").to_pylist()))
    chains = wk.chain_questions(paths, world.index, n=a.n, seed=a.seed)
    known = StoreRelations(world.index._seen)
    log(f"world: {len(world)} facts, {len(known)} distinct relations | {len(chains)} chains "
        f"| built in {time.perf_counter() - tw:.0f}s")

    items = []
    for c in chains:
        plan = parse_question(c["question"])
        p2 = plan.relations[1]; tr = tail_relation(plan.tail, known)
        if tr is None:
            continue
        e = plan.tail[len(tr):].strip()
        e = e[3:] if e.startswith("of ") else e
        items.append({"p1": tr, "p2": p2, "e": e, "gold": c["answer"]})
    log(f"matched items: {len(items)}\n")

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
    TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
    def walk(q, plan):
        return answer(q, no_search, run_fn, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), tau_ret=0.0,
                      top_k=3, max_repairs=1, lookup=world.lookup, known=known, plan=plan)

    def build_plan(rels, seed):
        if not rels or not seed:
            return None
        return QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))

    # ---- pass 1: the emitter speaks once per (item, form); nothing is scored yet ----
    log("pass 1 -- collecting emitter plans")
    emitted: dict[tuple[int, str], tuple[list[str], str | None]] = {}
    t_em = 0.0
    for i, it in enumerate(items):
        for form, q in forms(it["p1"], it["p2"], it["e"]).items():
            te = time.perf_counter()
            try:
                out = strip_fences(em.emit(q, max_new_tokens=a.max_new))
            except Exception:                                # noqa: BLE001
                emitted[(i, form)] = None; continue
            t_em += time.perf_counter() - te
            emitted[(i, form)] = emitted_plan(out, normalize)
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(items)} ({time.perf_counter() - t0:.0f}s, emitter {t_em:.0f}s)")

    # the trivial arm's constant plan: the commonest relation pair in the item set,
    # so the floor is a PLAUSIBLE constant rather than a nonsense one
    common = collections.Counter((it["p1"], it["p2"]) for it in items).most_common(1)[0][0]
    log(f"\ntrivial arm's constant plan: {list(common)!r}")

    stats = {f: {arm: collections.Counter() for arm in ARMS} for f in forms("", "", "")}
    wrong_ex = []

    def score(form, arm, q, plan, gold):
        st = stats[form][arm]
        if plan is None:
            st["no_plan"] += 1; return
        st["plan"] += 1
        r = walk(q, plan)
        if r.refused is not None:
            st["refused"] += 1; st[f"refused:{r.reason}"] += 1
        elif r.verified:
            st["walked"] += 1; st["verified"] += 1
            ok = normalize(r.answer) == normalize(gold)
            st["correct" if ok else "WRONG"] += 1
            if not ok and len(wrong_ex) < 8:
                wrong_ex.append((arm, form, q, r.answer, gold))
        else:
            st["walked"] += 1; st["walk_failed"] += 1

    # ---- pass 2: four arms over the same questions --------------------------------
    log("\npass 2 -- scoring four arms")
    for i, it in enumerate(items):
        j = (i + 1) % len(items)                              # the donor item for `shuffled`
        for form, q in forms(it["p1"], it["p2"], it["e"]).items():
            score(form, "grammar", q, parse_question(q), it["gold"])

            ep = emitted[(i, form)]
            rels, seed = (ep if ep else (None, None))
            score(form, "emitter", q, build_plan(rels, seed or normalize(it["e"])), it["gold"])

            score(form, "trivial", q, build_plan(list(common), normalize(it["e"])), it["gold"])

            # the decisive arm: another item's emitter relations, THIS item's seed
            dp = emitted[(j, form)]
            drels = dp[0] if dp else None
            same = drels is not None and rels is not None and [normalize(x) for x in drels] == [normalize(x) for x in rels]
            if same:
                stats[form]["shuffled"]["donor_identical"] += 1
            score(form, "shuffled", q, build_plan(drels, normalize(it["e"])), it["gold"])
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(items)} ({time.perf_counter() - t0:.0f}s, VM calls {calls['vm']})")

    # ---- the table, and the gap ----------------------------------------------------
    log("\nEMITTER ABLATION -- same world, same chains, same gold, same gates; only the PLAN'S ORIGIN moves")
    log(f"{'form':12s}{'arm':10s}{'plan':>6}{'refused':>9}{'verified':>10}{'correct':>9}{'WRONG':>7}")
    for form in stats:
        for arm in ARMS:
            s = stats[form][arm]
            log(f"{form:12s}{arm:10s}{s['plan']:6d}{s['refused']:9d}{s['verified']:10d}{s['correct']:9d}{s['WRONG']:7d}")

    n_q = len(items) * len(stats)
    tot = {arm: sum(stats[f][arm]["correct"] for f in stats) for arm in ARMS}
    wrong = {arm: sum(stats[f][arm]["WRONG"] for f in stats) for arm in ARMS}
    log(f"\ntotals over {n_q} questions: " + " | ".join(f"{a_} {tot[a_]} (wrong {wrong[a_]})" for a_ in ARMS))

    p_em = tot["emitter"] / max(1, n_q)
    noise = wilson_halfwidth(p_em, n_q)
    gap = (tot["emitter"] - tot["shuffled"]) / max(1, n_q)
    log(f"\nemitter accuracy {p_em:.3f} | 95% Wilson half-width (the noise floor) +/-{noise:.3f}")
    log(f"ABLATION GAP (emitter - shuffled) = {gap:+.3f}  ({tot['emitter'] - tot['shuffled']} questions)")
    log(f"gap vs host-only (emitter - grammar) = {(tot['emitter'] - tot['grammar']) / max(1, n_q):+.3f}")
    verdict = ("INSIDE the noise floor -- the emitter's plan is not carrying question-specific "
               "information on this set. WO-0.1's alarm condition." if abs(gap) <= noise else
               "outside the noise floor -- the emitter's plan is doing work.")
    log(f"VERDICT: {verdict}")
    donor_same = sum(stats[f]["shuffled"]["donor_identical"] for f in stats)
    log(f"(donor plan identical to the real one on {donor_same}/{n_q} questions -- these dilute the gap "
        f"toward zero and are not a defect of the emitter)")

    for arm, form, q, ans, gold in wrong_ex:
        log(f"  WRONG [{arm}/{form}] {q[:70]!r} -> {ans!r} (gold {gold!r})")
    if session:
        session.close()
    wall = time.perf_counter() - t0
    (LOGS / f"exp_r22_emitter_ablation{a.tag}.json").write_text(json.dumps({
        # Record the gguf RELATIVE to the repo when it lives inside it: a run
        # artifact is committed, and an absolute path bakes one machine's
        # layout into the repo.
        "n_items": len(items), "n_questions": n_q, "world_facts": len(world),
        "gguf": (str(pathlib.Path(a.gguf).relative_to(ROOT)).replace("\\", "/")
                 if str(pathlib.Path(a.gguf)).startswith(str(ROOT))
                 else pathlib.Path(a.gguf).name),
        "stats": {f: {p: dict(c) for p, c in d.items()} for f, d in stats.items()},
        "totals": tot, "wrong": wrong, "emitter_accuracy": round(p_em, 4),
        "noise_floor": round(noise, 4), "ablation_gap": round(gap, 4),
        "donor_identical": donor_same, "vm_calls": calls["vm"],
        "wrong_examples": wrong_ex, "wall_s": round(wall, 1)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r22_emitter_ablation{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r22_emitter_ablation{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
