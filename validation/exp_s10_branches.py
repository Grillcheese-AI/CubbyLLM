"""exp_s10_branches -- H-S10 G7 outside the game: the conditional mind over possibility-world branches.

The game's G7 priced a never-tried floor; the mind H-S10 describes evaluates the branches an
uncertain step forks into. Here the step is a hop of a real question whose store holds two
candidates: the true fact and a PLANTED one (the object swapped for another row's object of the
same relation, as exp_f2m4's plants), so the walk forks (graph-of-thought, pipeline._explore) and
every branch is walked to its own answer. Three policies on the same forks:

  habit         act on the best prediction: the candidates tie, so a branch at random is spoken
  ask           the loop as it stands: speak only when every branch agrees, else refuse (never wrong)
  conditional-strict  as conditional, but a branch the VM rejected that names another answer leaves
                the question open (refuse): the VM's doubt counts as a possibility too
  conditional   price each branch by its possibility: every fact of the branch to the possibility
                oracle (StorePossibility over the TRUSTED world, the one that passed H-F2 M4), a
                branch holding an IMPOSSIBLE fact is worth a wrong answer (strongly negative), one of
                POSSIBLE facts only is worth a right one; speak when the branches worth speaking name
                one answer, refuse when none does or they disagree (refusal is neutral: the dopamine rule)

Questions: hdc Wiki5M 2- and 3-hop chains (validation/exp_r18_hdc_rephrased.py's parse). The plan
is the chain's own relations in walk order (what the emitter proposes when it is right: the fork is
what is measured here, not the planner or the parser -- exp_r18 is that gate); the trusted world
holds every chain's true facts; the walk's world holds those plus the plants.
Controls: questions with no plant (no fork) through the same policies. Reported per hop count and
policy: correct, wrong, refused, and milliseconds per question (the walk; the oracle's checks).
The kill line is the serving gate's: 0 wrong for the conditional policy.

    python validation/exp_s10_branches.py --n 600 [--plant-rate 0.8] [--exe cubelang.exe] [--tag _x]
Log: validation/logs/exp_s10_branches{tag}.json
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_r18_hdc_rephrased import HDC, TAU_VM, fact_parsers  # noqa: E402

NO_SEARCH = lambda q, k: []  # noqa: E731
WRONG, RIGHT = -1.0, 1.0        # what a wrong and a right answer are worth; a refusal 0


def vm_runner(exe: str | None):
    """The real VM when cubelang.exe is found, else the faithful stand-in (recorded in the log)."""
    try:
        from cubbyllm.bridges import cubelang_client as cc
        session = cc.CubelangSession(exe=exe)
        return (lambda source, fn: session.run(source, fn=fn)), "cubelang"
    except Exception as e:                                        # no exe beside this checkout
        from test_pipeline_plan_refusal import faithful_vm
        return faithful_vm([]), f"faithful stand-in ({type(e).__name__})"


def branch_value(oracle, b: dict) -> tuple[float, list[str]]:
    """What the possibility oracle says of a branch's facts: -1 if any is impossible, +1 if all are
    possible, 0 when the world does not know."""
    from cubbyllm.bridges.possibility import Verdict
    vs = [oracle.possible(f).verdict for f in b.get("facts", [])]
    if any(v == Verdict.IMPOSSIBLE for v in vs):
        return WRONG, [str(v) for v in vs]
    if vs and all(v == Verdict.POSSIBLE for v in vs):
        return RIGHT, [str(v) for v in vs]
    return 0.0, [str(v) for v in vs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(HDC / "ood_validation_rephrased.pq"))
    ap.add_argument("--mapping", default=str(HDC / "relation_template_mapping.csv"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plant-rate", type=float, default=0.8, help="share of questions given a planted candidate")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()
    import pyarrow.parquet as pq
    from cubbyllm.bridges.possibility import StorePossibility
    from cubbyllm.reasoning.pipeline import answer
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from worlds import FactStore

    parse, _mapping = fact_parsers(pathlib.Path(a.mapping))
    table = pq.read_table(a.file).to_pylist()
    c = collections.Counter()
    chains: dict[int, list[tuple[str, str, str]]] = {}
    trusted, walk_world = FactStore(name="hdc"), FactStore(name="walk")
    for i, row in enumerate(table):
        if int(row["n_hop"]) < 2:
            continue
        hs = [parse(f) for f in (row["facts"] or [])]
        if not hs or any(h is None for h in hs):
            c["row:unparsed_fact"] += 1; continue
        if any(normalize(hs[k][2]) != normalize(hs[k + 1][0]) for k in range(len(hs) - 1)):
            c["row:chain_broken"] += 1; continue
        if normalize(hs[-1][2]) != normalize(str(row["answer"])) or len(hs) != int(row["n_hop"]):
            c["row:answer_or_hops"] += 1; continue
        chains[i] = hs
        for subj, rel, obj in hs:
            fact = f"{obj} is the {rel} of {subj}"
            for w in (trusted, walk_world):
                w.index.declare_relation(rel)                      # (a source hands over triples: the split knows the label)
                w.add(fact)
    by_template: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, hs in chains.items():
        by_template[tuple(r for _, r, _ in hs)].append(i)
    rng = random.Random(a.seed)
    pool = [i for i in chains if len(by_template[tuple(r for _, r, _ in chains[i])]) >= 2]
    rng.shuffle(pool)
    picked = pool[:a.n]
    plants: dict[int, tuple[int, str]] = {}                       # question -> (fork hop, planted fact)
    for i in picked:
        if rng.random() >= a.plant_rate:
            continue
        hs = chains[i]
        k = rng.randrange(len(hs))
        others = [j for j in by_template[tuple(r for _, r, _ in hs)] if j != i and normalize(chains[j][k][2]) != normalize(hs[k][2])]
        if not others:
            continue
        j = rng.choice(others)
        planted = f"{chains[j][k][2]} is the {hs[k][1]} of {hs[k][0]}"
        if walk_world.add(planted):
            plants[i] = (k, planted)
    print(f"hdc: {len(table):,} rows, {len(chains):,} chains ({dict(c)}), {len(picked)} questions, {len(plants)} planted; "
          f"trusted {len(trusted):,} facts, walk world {len(walk_world):,} | {time.perf_counter() - t0:.0f}s", flush=True)

    vm, vm_name = vm_runner(a.exe)
    oracle = StorePossibility({"hdc": trusted})
    policies = ("habit", "ask", "conditional", "conditional-strict")
    tally = {p: collections.Counter() for p in policies}
    per_hop = {p: collections.defaultdict(collections.Counter) for p in policies}
    ms_walk, ms_oracle, n_branches, caught, reasons = [], [], [], collections.Counter(), collections.Counter()
    examples, wrongs, supported = [], [], collections.Counter()
    for n_i, i in enumerate(picked):
        row, hs = table[i], chains[i]
        q, truth, hops = row["question_prompt_original"], normalize(str(row["answer"])), len(hs)
        rels = [r for _, r, _ in hs]
        plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {hs[0][0]}", n_hop=hops)
        t1 = time.perf_counter()
        res = answer(q, NO_SEARCH, vm, tau_vm=TAU_VM.get(hops, 0.4), tau_ret=0.5, lookup=walk_world.lookup,
                     plan=plan, branch=12)
        ms_walk.append((time.perf_counter() - t1) * 1000)
        forked = i in plants
        verdict = {}
        if res.verified and res.answer:                            # no fork, or branches that agreed
            for p in policies:
                verdict[p] = "correct" if normalize(res.answer) == truth else "wrong"
        elif res.reason == "ambiguous_hop" and res.branches:
            finished = [b for b in res.branches if b.get("status") == "verified" and b.get("answer")]
            n_branches.append(len(finished))
            if not finished:
                reasons["fork:none finished"] += 1
                verdict = {p: "refused" for p in policies}
            else:
                pick = rng.choice(finished)
                verdict["habit"] = "correct" if normalize(pick["answer"]) == truth else "wrong"
                verdict["ask"] = "refused"
                rejected = {normalize(b["answer"] or "") for b in res.branches if b.get("status") == "vm_failed" and b.get("answer")}
                t2 = time.perf_counter()
                valued = [(branch_value(oracle, b), b) for b in finished]
                ms_oracle.append((time.perf_counter() - t2) * 1000)
                winners = [b for (v, _), b in valued if v == RIGHT]
                losers = [b for (v, _), b in valued if v == WRONG]
                if forked:
                    caught["planted branch impossible" if any(plants[i][1] in b.get("facts", []) for b in losers) else "planted branch not caught"] += 1
                named = {normalize(b["answer"]) for b in winners}
                if len(named) == 1:
                    verdict["conditional"] = "correct" if named == {truth} else "wrong"
                else:
                    verdict["conditional"] = "refused"
                # strict: a branch the VM rejected that names ANOTHER answer leaves the question open
                verdict["conditional-strict"] = verdict["conditional"] if not (rejected - named) else "refused"
                ex = {"q": q, "truth": truth, "planted": plants.get(i, (None, None))[1],
                      "branches": [{"answer": b["answer"], "value": v, "verdicts": vs, "status": b.get("status"), "facts": b.get("facts")}
                                   for (v, vs), b in valued],
                      "unfinished": [{"status": b.get("status"), "answer": b.get("answer"), "stalled": b.get("stalled")}
                                     for b in res.branches if b not in finished],
                      "verdict": verdict}
                if len(examples) < 12:
                    examples.append(ex)
                if verdict["conditional"] == "wrong":
                    # the spoken branch's facts are all members of the trusted world: the key names
                    # another value of a multi-valued relation, the world does not call it false
                    ex["world_supported"] = all(f in trusted for b in winners for f in b.get("facts", []))
                    supported["world-supported" if ex["world_supported"] else "false fact spoken"] += 1
                    if len(wrongs) < 30:
                        wrongs.append(ex)
        else:
            reasons[f"walk refused: {res.reason}"] += 1
            verdict = {p: "refused" for p in policies}
        for p in policies:
            tally[p][verdict[p]] += 1
            per_hop[p][f"{hops}-hop {'forked' if forked else 'plain'}"][verdict[p]] += 1
        if (n_i + 1) % 100 == 0:
            print(f"  {n_i + 1}/{len(picked)}: " + " | ".join(f"{p} {dict(tally[p])}" for p in policies) +
                  f" | walk {sum(ms_walk) / len(ms_walk):.0f} ms", flush=True)

    def rate(cnt):
        n = sum(cnt.values()) or 1
        return {k: round(v / n, 3) for k, v in cnt.items()} | {"n": n}
    out = {"bench": "exp_s10_branches: the conditional mind over possibility-world branches, hdc 2/3-hop chains with a planted candidate",
           "date": time.strftime("%Y-%m-%d"), "n": len(picked), "planted": len(plants), "plant_rate": a.plant_rate, "vm": vm_name,
           "policies": {p: rate(tally[p]) for p in policies},
           "per_hop": {p: {k: rate(v) for k, v in sorted(per_hop[p].items())} for p in policies},
           "value": {p: round((tally[p]["correct"] * RIGHT + tally[p]["wrong"] * WRONG) / max(1, sum(tally[p].values())), 3) for p in policies},
           "kill_line_0_wrong": {p: tally[p]["wrong"] == 0 for p in policies},
           "branches_per_fork": round(sum(n_branches) / len(n_branches), 2) if n_branches else None,
           "planted": dict(caught), "refusals": dict(reasons), "conditional_wrong_kind": dict(supported),
           "ms": {"walk_mean": round(sum(ms_walk) / len(ms_walk), 2), "walk_p99": round(sorted(ms_walk)[int(0.99 * (len(ms_walk) - 1))], 2),
                  "oracle_per_fork_mean": round(sum(ms_oracle) / len(ms_oracle), 2) if ms_oracle else None},
           "examples": examples, "conditional_wrong": wrongs, "secs": round(time.perf_counter() - t0, 1)}
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"exp_s10_branches{a.tag}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("policies", "value", "kill_line_0_wrong", "branches_per_fork", "planted", "refusals",
                                            "conditional_wrong_kind", "ms", "vm")},
                     indent=1, ensure_ascii=False), flush=True)
    print(f"log: {path}")


if __name__ == "__main__":
    main()
