"""
Wired: STANDALONE (validation script; never imported by cubbyllm/).

WO-0.2 -- the liveness rate (fact-perturbation differential).

The question every other instrument fails to ask: **does the emitted program
actually read its inputs?**

Every existing measurement is a form of "is the spoken answer correct". A
program that ignores its own inputs entirely can still score perfectly, because
the host supplies the facts, the gate verifies them, and the answer comes out
right regardless of what the program did with them. WO-0.1's ablation found the
emitter's PLAN is load-bearing; this finds whether the PROGRAM is.

The test is a differential, and it is cheap:

    run the program against the fact base            -> answer A
    run it against a copy of the fact base where the
    queried relation's value is swapped for a
    DIFFERENT VALID value for that same relation     -> answer B

    the program is LIVE iff A != B

A program whose output is invariant to its own inputs is exactly the signature
of a placeholder argument, a zero-bytecode assign, or an all-arms `match` --
the three silent-failure classes cubelang's docs/DRIFT.md C3/C4/C11 documented
and that were fixed on 2026-09-14. So this is also the production-side detector
for that whole bug family: it catches them in the answer path rather than only
in a compiler audit, and it keeps catching the next one of its kind.

Why the decoy is a *valid* value for the same relation, not garbage: a garbage
object would be caught by type checks and coverage rules that have nothing to
do with whether the program read it. Swapping `date of birth` for another
person's real date of birth changes only the thing under test.

    python validation/exp_r23_liveness.py --n 60 --resident

Reading it:

  live                A and B differ -- the program's output depends on its input.
  DEAD                A and B are the SAME non-null answer despite the fact
                      changing underneath. This is the alarm. A dead program
                      that speaks is a wrong answer waiting for a world where
                      the host does not happen to agree with it.
  both_refused        Neither run spoke. Carries no liveness information, and is
                      excluded from the rate rather than counted as a pass --
                      counting refusals as live is how this metric would rot.

**Alarm:** liveness below 95% [ASSUM -- set the real threshold after this
round-0 baseline].
"""
from __future__ import annotations

import argparse, collections, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}


def build_relation_objects(index, cap_per_rel: int = 64) -> dict[str, list[str]]:
    """{normalized relation -> a few real objects stated for it}.

    The decoy pool. One pass over the index's subject buckets, capped per
    relation so a relation with 100k facts does not dominate memory; the cap is
    generous because all we ever need is one object that differs from the
    fact's own.
    """
    from cubbyllm.reasoning.planner import normalize

    pool: dict[str, list[str]] = collections.defaultdict(list)
    for bucket in index._by_subj.values():
        for _fact, t in bucket:
            key = normalize(t.rel)
            objs = pool[key]
            if len(objs) < cap_per_rel:
                objs.append(t.obj)
    return pool


def make_perturbed_lookup(world, pool, n_hop: int, swaps: collections.Counter):
    """Wrap `world.lookup` so the FINAL hop's candidates carry a different
    (but real) object for the same relation.

    The final hop is the one whose object becomes the spoken answer, so it is
    "the queried relation's value" in WO-0.2's sense. Earlier hops are left
    alone: perturbing them tests entity resolution, which is a different
    question and would muddy this one.

    The fact TEXT is rewritten alongside the Triple, not just the Triple, so
    that anything downstream reading the sentence (the verifier, the
    provenance record, the program's own string payload) sees a consistent
    fact rather than a sentence that disagrees with its own parse. A fact whose
    object does not appear verbatim in its own text is skipped rather than
    half-rewritten -- a half-rewritten fact would fail for the wrong reason and
    show up as false liveness.
    """
    from cubbyllm.reasoning.planner import Triple, normalize

    last = max(0, n_hop - 1)

    def lookup(plan, hop, entity):
        cands = world.lookup(plan, hop, entity)
        if hop != last or not cands:
            return cands
        out = []
        for fact, t in cands:
            decoy = None
            for cand_obj in pool.get(normalize(t.rel), ()):
                if normalize(cand_obj) != normalize(t.obj):
                    decoy = cand_obj
                    break
            if decoy is None or t.obj not in fact:
                out.append((fact, t))          # nothing valid to swap in
                swaps["unswappable"] += 1
                continue
            out.append((fact.replace(t.obj, decoy), Triple(obj=decoy, rel=t.rel, subj=t.subj)))
            swaps["swapped"] += 1
        return out

    return lookup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
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
    from cubbyllm.reasoning.planner import normalize
    from cubbyllm.reasoning.plan_verify import StoreRelations

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

    pool = build_relation_objects(world.index)
    log(f"decoy pool: {len(pool)} relations with at least one stored object\n")

    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = collections.Counter()
    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    def no_search(q, k):
        return []                                   # lookup is the ONLY fact source, so the
                                                    # perturbation is total

    def walk(q, plan, lookup):
        return answer(q, no_search, run_fn, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), tau_ret=0.0,
                      top_k=3, max_repairs=1, lookup=lookup, known=known, plan=plan)

    stats: collections.Counter = collections.Counter()
    swaps: collections.Counter = collections.Counter()
    dead_ex: list[tuple[str, str]] = []
    rows: list[dict] = []

    log("running the differential (each question twice: real world, then perturbed)")
    for i, c in enumerate(chains):
        q = c["question"]
        plan = parse_question(q)
        if plan is None:
            stats["no_plan"] += 1
            continue

        base = walk(q, plan, world.lookup)
        # Per-chain swap accounting. A chain where nothing was actually swapped
        # was never perturbed, so "the answer did not change" says nothing about
        # liveness -- counting it DEAD would be a false alarm, and the first
        # trial run of this script produced exactly one. Excluded below.
        mine: collections.Counter = collections.Counter()
        pert = walk(q, plan, make_perturbed_lookup(world, pool, plan.n_hop, mine))
        swaps.update(mine)

        a_ans = base.answer if base.refused is None and base.verified else None
        b_ans = pert.answer if pert.refused is None and pert.verified else None

        row = {"question": q, "gold": c["answer"], "base": a_ans, "perturbed": b_ans,
               "base_reason": base.reason, "perturbed_reason": pert.reason,
               "swapped": mine["swapped"], "unswappable": mine["unswappable"]}

        if not mine["swapped"]:
            # The final hop's facts had no alternative stored value for their
            # relation, so the "perturbed" world is identical to the real one.
            stats["not_perturbable"] += 1
            row["verdict"] = "not_perturbable"
        elif a_ans is None and b_ans is None:
            # No liveness information: the program never spoke either way.
            stats["both_refused"] += 1
            row["verdict"] = "both_refused"
        elif a_ans is None:
            # Spoke only when the fact changed. Strange, but it IS input-dependent.
            stats["live"] += 1; stats["live:only_perturbed_spoke"] += 1
            row["verdict"] = "live"
        elif b_ans is None:
            # The usual healthy shape: the swap breaks verification and the
            # program refuses instead of speaking a stale answer.
            stats["live"] += 1; stats["live:refused_after_swap"] += 1
            row["verdict"] = "live"
        elif normalize(a_ans) != normalize(b_ans):
            stats["live"] += 1; stats["live:answer_changed"] += 1
            row["verdict"] = "live"
        else:
            # THE ALARM: the fact changed underneath and the program said the
            # same thing anyway.
            stats["DEAD"] += 1
            row["verdict"] = "DEAD"
            if len(dead_ex) < 10:
                dead_ex.append((q, a_ans))

        rows.append(row)
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(chains)} ({time.perf_counter() - t0:.0f}s, VM calls {calls['vm']})")

    # ---- the rate -------------------------------------------------------------------
    informative = stats["live"] + stats["DEAD"]
    rate = stats["live"] / informative if informative else float("nan")

    log("\nLIVENESS -- does the program's output depend on the facts it was given?")
    log(f"  chains run            : {len(chains)}")
    log(f"  no plan               : {stats['no_plan']}")
    log(f"  not perturbable       : {stats['not_perturbable']}   (the relation has no other stored "
        f"value, so the 'perturbed' world is the real one; excluded)")
    log(f"  both runs refused     : {stats['both_refused']}   (no liveness information; excluded)")
    log(f"  informative           : {informative}")
    log(f"    live                : {stats['live']}")
    log(f"      answer changed    : {stats['live:answer_changed']}")
    log(f"      refused after swap: {stats['live:refused_after_swap']}")
    log(f"      only perturbed spoke: {stats['live:only_perturbed_spoke']}")
    log(f"    DEAD                : {stats['DEAD']}")
    log(f"  swap coverage         : {swaps['swapped']} candidate facts swapped, "
        f"{swaps['unswappable']} left alone (no other stored value for the relation)")
    log(f"\n  LIVENESS RATE = {rate:.1%}  (alarm below 95%)")

    if stats["DEAD"]:
        log("\nDEAD examples -- same answer with the fact changed underneath:")
        for q, ans in dead_ex:
            log(f"  {q!r} -> {ans!r} (unchanged)")
        log("\nA dead program that speaks is a wrong answer waiting for a world where the")
        log("host does not happen to agree with it. Check for a placeholder argument, an")
        log("assignment that compiles to nothing, or a branch that always takes one arm.")
    elif informative:
        log("\nNo dead programs in this sample: every program that spoke changed its")
        log("output when its facts changed.")

    if swaps["unswappable"] and not swaps["swapped"]:
        log("\nWARNING: nothing was swappable, so this run measured nothing. The decoy")
        log("pool found no alternative value for any queried relation.")

    LOGS.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    out = LOGS / f"exp_r23_liveness{tag}.json"
    json.dump({"n": len(chains), "rate": rate, "stats": dict(stats), "swaps": dict(swaps),
               "rows": rows, "wall_s": round(time.perf_counter() - t0, 1)},
              open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    (LOGS / f"exp_r23_liveness{tag}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwrote {out}")
    if session:
        session.close()


if __name__ == "__main__":
    main()
