"""exp_r34 -- CLUTRR's own question, through the skill library. WO-2.6, part three.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

WHAT THIS IS
------------
exp_r29 walked CLUTRR's chains ("Who is the brother of the grandson of Jason?") and said so: CLUTRR-derived,
not CLUTRR. CLUTRR's own question names two people and asks how they are related -- the relation the chain
between them COMPOSES to -- and that needed a kinship algebra the engine did not have (WO-2.6). Here the
algebra is not written. It is learned, by the path a live loop would take:

  day 1   the TRAIN split (9,074 records, 2-3 hops) as relation asks the loop refused, each with the
          relation CLUTRR's gold states
  night   the sleep cycle (reasoning/sleep.py): the skill library (reasoning/skills.py) adopts every
          composition rule those episodes support with zero counterexamples across the whole record
  day 2   the TEST split (1,146 records, 2-10 hops): each record's own story graph is its store; the
          loop's `relate` finds every path from A to B, composes each over every bracketing, and
          certifies the facts and the rule steps in the real cubelang VM

REPORTED AS CORRECT / REFUSED / WRONG by depth, never accuracy (WO-2.6's rule). Depth is the number of
story edges -- CLUTRR's k. No test record is seen before day 2, and no gold of a test record is read
except to score it.

THE ARMS
--------
    rules        the night's library
    pairs_only   a library mined from the 2-hop train records alone -- what closure over the 3-hop ones buys
    null         every test story's edges into B relabelled with a token no episode ever used: a spoken
                 answer is a BREACH
    shuffled     instrument check -- the library's conclusions permuted. The harness must SEE wrongs here,
                 or a zero-wrong result in the other arms says nothing

KILL CRITERION
--------------
Any wrong answer in `rules` or `pairs_only`, or any spoken answer in `null`, kills it. `shuffled` reporting
zero wrong means the instrument is blind, and nothing here counts.

    python validation/exp_r34_clutrr_relations.py --bench-root <dir>
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
OUT = ROOT / "standin" / "data" / "out" / "exp_r34"          # gitignored: the night's state
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NULL_TOKEN = "zubnog"                     # exp_r26's free token: no facts, no episodes, anywhere
TAU = {1: 1.0, 2: 0.4736328125}           # by bindings per frame (standin/ask.TAU_VM); relate() uses two


def graph(rec: dict):
    """(names, edges, relations, q0, q1) of a CLUTRR record, or None."""
    import clutrr as C
    try:
        edges = ast.literal_eval(rec["story_edges"])
        types = ast.literal_eval(rec["edge_types"])
        q0, q1 = ast.literal_eval(rec["query_edge"])
    except (ValueError, SyntaxError, KeyError):
        return None
    names = C._names(rec)
    if not edges or len(edges) != len(types) or max(max(e) for e in edges + [(q0, q1)]) >= len(names):
        return None
    return names, edges, types, q0, q1


def day_rows(recs: list[dict]) -> tuple[list[dict], collections.Counter]:
    """Train records as the day's relation asks: refused, with every path from A to B and the gold."""
    import clutrr as C
    rows, skipped = [], collections.Counter()
    for rec in recs:
        g = graph(rec)
        if g is None:
            skipped["unparseable"] += 1
            continue
        names = g[0]
        if len(set(n.lower() for n in names)) != len(names):
            skipped["repeated_name"] += 1
            continue
        ps = C.paths(rec) or []
        if not ps:
            skipped["no_path"] += 1
            continue
        rows.append({"id": rec.get("id"), "question": f"How is {names[g[4]]} related to {names[g[3]]}?",
                     "kind": "relation", "verified": False, "reason": "no_rule",
                     "paths": [[r for _a, r, _b in p] for p in ps], "gold": rec["target_text"]})
    return rows, skipped


def night(rows: list[dict], out: pathlib.Path, label: str):
    from cubbyllm.reasoning import sleep as S
    from cubbyllm.reasoning.skills import Library
    if out.exists():
        for f in sorted(out.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    day = out / "day.jsonl"
    day.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    t0 = time.perf_counter()
    rep = S.sleep([day], out, night=label)
    return Library(out / "skills.jsonl"), rep["skills"], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=None)
    ap.add_argument("--task", default="gen_train23_test2to10")
    ap.add_argument("--shuffled-n", type=int, default=150, help="test records for the instrument check")
    ap.add_argument("--limit", type=int, default=0, help="test records per arm (0 = all)")
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--tau2", type=float, default=TAU[2],
                    help="the threshold for a two-binding frame. The default is the served one (exp_r11's); another "
                         "value is a DIAGNOSTIC of how many refusals the threshold alone costs, never a result")
    ap.add_argument("--host-diag", type=float, default=0.0,
                    help="DIAGNOSTIC arm: what the library would answer if the host adopted every contested premise "
                         "at least this one-sided (e.g. 0.95), discounting its minority. Never a result: adopting "
                         "is a host decision the harness does not make")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    import clutrr as C
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.index import TripleIndex
    from cubbyllm.reasoning.planner import normalize
    from cubbyllm.reasoning.skills import MAX_HOPS, MAX_PATHS, Library, Rule, relate

    train = C.load_raw("train", a.task, a.bench_root)
    test = C.load_raw("test", a.task, a.bench_root)
    conv = C.verify_convention(train)
    C.verify_convention(test)
    log(f"CLUTRR {a.task}: train {len(train)}, test {len(test)}; convention forward "
        f"{conv['forward']}/{conv['gendered_edges']}")

    # ── day 1 and the night ─────────────────────────────────────────────────────────────────────────
    rows, skipped = day_rows(train)
    log(f"day 1: {len(rows)} relation asks refused with gold ({dict(skipped) or 'none skipped'}); "
        f"hops {dict(sorted(collections.Counter(len(r['paths'][0]) for r in rows).items()))}")
    lib, sk, dt = night(rows, OUT / "rules", "clutrr-train")
    log(f"night (all train): {sk['adopted']} rules adopted in {sk['rounds']} rounds from {sk['episodes_total']} "
        f"episodes, {sk['retired']} retired; rejected {sk['rejected']} ({dt:.1f}s)")
    pairs_rows = [r for r in rows if len(r["paths"][0]) == 2]
    lib2, sk2, dt2 = night(pairs_rows, OUT / "pairs_only", "clutrr-train-2hop")
    log(f"night (2-hop train only): {sk2['adopted']} rules from {sk2['episodes_total']} episodes; "
        f"rejected {sk2['rejected']} ({dt2:.1f}s)")
    rng = random.Random(a.seed)
    shuffled = Library()
    rules = [e["rule"] for e in lib.rules.values()]
    concl = [r.conclusion for r in rules]
    rng.shuffle(concl)
    for r, c in zip(rules, concl):
        shuffled.adopt(Rule(r.first, r.second, c), support=0, evidence=[], night="shuffled")
    moved = sum(1 for r, c in zip(rules, concl) if r.conclusion != c)
    log(f"shuffled library: {moved}/{len(rules)} conclusions moved\n")

    # ── day 2 ───────────────────────────────────────────────────────────────────────────────────────
    session = cc.CubelangSession(exe=a.exe)
    vm_calls = collections.Counter()

    def run_fn(source, fn):
        vm_calls["n"] += 1
        return session.run(source, fn=fn)

    items = []
    for rec in test:
        g = graph(rec)
        if g is None:
            continue
        names, edges, types, q0, q1 = g
        if len(set(n.lower() for n in names)) != len(names):
            continue
        items.append({"id": rec.get("id"), "depth": len(types), "a": names[q0], "b": names[q1],
                      "gold": normalize(rec["target_text"]),
                      "facts": [C.fact(names[i], r, names[j]) for (i, j), r in zip(edges, types)],
                      "null": [C.fact(names[i], NULL_TOKEN if j == q1 else r, names[j])
                               for (i, j), r in zip(edges, types)]})
    if a.limit:
        items = rng.sample(items, min(a.limit, len(items)))
    log(f"day 2: {len(items)} test questions; by depth "
        f"{dict(sorted(collections.Counter(i['depth'] for i in items).items()))}")

    tau = {1: TAU[1], 2: a.tau2}
    if a.tau2 != TAU[2]:
        log(f"!! DIAGNOSTIC RUN: tau for a two-binding frame {a.tau2} (served: {TAU[2]})")
    host_lib = None
    if a.host_diag:
        from cubbyllm.reasoning.skills import Episode, host_adopt
        host_lib = Library(OUT / "rules" / "skills.jsonl")
        host_lib.path = None                                   # in memory: the diagnostic writes no ledger
        eps = [Episode(**{k: e[k] for k in ("id", "relations", "conclusion")}, source=e.get("source"))
               for e in map(json.loads, (OUT / "rules" / "skill_episodes.jsonl").read_text(encoding="utf-8").splitlines())]
        took = []
        for c in map(json.loads, (OUT / "rules" / "skills_contested.jsonl").read_text(encoding="utf-8").splitlines()):
            if c["why"] == "conflict" and c["share"] >= a.host_diag:
                first, _, second = c["premise"].partition(" then ")
                res = host_adopt(eps, host_lib, (first, second), c["majority"], "diagnostic", "diag")
                took.append((c["premise"], c["majority"], c["share"], "adopted" if "adopted" in res else res["refused"][:80]))
        log(f"!! DIAGNOSTIC host_diag (share >= {a.host_diag}): {took}")
    arms = {"rules": (lib, "facts", items), "pairs_only": (lib2, "facts", items), "null": (lib, "null", items),
            "shuffled": (shuffled, "facts", rng.sample(items, min(a.shuffled_n, len(items))))}
    if host_lib is not None:
        arms["host_diag"] = (host_lib, "facts", items)
    results: dict[str, list[dict]] = {}
    for arm, (library, which, pool) in arms.items():
        ta = time.perf_counter()
        n0 = vm_calls["n"]
        rs = []
        for it in pool:
            ix = TripleIndex(it[which])
            ps, over = ix.paths(it["a"], it["b"], max_hops=MAX_HOPS, limit=MAX_PATHS)
            res = relate(ps, library, run_fn, tau_vm=tau, chunk=2, overflow=over)
            said = normalize(res["answer"] or "")
            if res["verified"]:
                outcome = "correct" if said == it["gold"] else "WRONG"
                if arm == "null":
                    outcome = "BREACH"
            else:
                outcome = "refused"
            rs.append({"id": it["id"], "depth": it["depth"], "gold": it["gold"], "answer": res["answer"],
                       "outcome": outcome, "reason": res["reason"], "n_paths": len(res["paths"]),
                       "paths": [{k: p.get(k) for k in ("relations", "status", "conclusion", "steps", "failed",
                                                        "conclusions")} for p in res["paths"]]})
        results[arm] = rs
        log(f"[{arm}] {len(rs)} questions, {vm_calls['n'] - n0} VM calls, {time.perf_counter() - ta:.1f}s")

    # ── the report ──────────────────────────────────────────────────────────────────────────────────
    def table(arm: str) -> None:
        rs = results[arm]
        by = collections.defaultdict(collections.Counter)
        for r in rs:
            by[r["depth"]][r["outcome"]] += 1
        log(f"\n{arm} -- correct / refused / wrong by depth")
        log("   depth      n   correct   refused   WRONG   BREACH")
        tot = collections.Counter()
        for d in sorted(by):
            c = by[d]
            tot.update(c)
            log(f"   {d:>5} {sum(c.values()):>6} {c['correct']:>9} {c['refused']:>9} {c['WRONG']:>7} {c['BREACH']:>8}")
        hi = collections.Counter()
        for d in by:
            if d >= 4:
                hi.update(by[d])
        log(f"   {'all':>5} {sum(tot.values()):>6} {tot['correct']:>9} {tot['refused']:>9} {tot['WRONG']:>7} "
            f"{tot['BREACH']:>8}")
        log(f"   {'>=4':>5} {sum(hi.values()):>6} {hi['correct']:>9} {hi['refused']:>9} {hi['WRONG']:>7} "
            f"{hi['BREACH']:>8}   <- depths no training record reaches")
        reasons = collections.Counter(r["reason"] for r in rs if r["outcome"] == "refused")
        log(f"   refusals: {dict(reasons.most_common())}")

    for arm in results:
        table(arm)

    wrong = sum(1 for arm in ("rules", "pairs_only") for r in results[arm] if r["outcome"] == "WRONG")
    breach = sum(1 for r in results["null"] if r["outcome"] == "BREACH")
    seen = sum(1 for r in results["shuffled"] if r["outcome"] == "WRONG")
    for arm in ("rules", "pairs_only"):
        ex = [r for r in results[arm] if r["outcome"] == "WRONG"][:5]
        for r in ex:
            log(f"   WRONG [{arm}] {r['id']} depth {r['depth']}: said {r['answer']} gold {r['gold']} {r['paths'][:2]}")
    verdict = ("KILLED" if wrong or breach else ("INSTRUMENT BLIND" if not seen else "survives"))
    log(f"\nVERDICT: {verdict} -- {wrong} wrong in rules + pairs_only, {breach} breaches in null, "
        f"{seen} wrong in shuffled (the instrument sees wrongs: {'yes' if seen else 'NO'})")
    log(f"library: {len(lib)} rules; total {time.perf_counter() - t0:.1f}s, {vm_calls['n']} VM calls")

    LOGS.mkdir(parents=True, exist_ok=True)
    stem = "exp_r34_clutrr_relations" + (f"_{a.tag}" if a.tag else "")
    (LOGS / f"{stem}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "task": a.task, "night": sk, "night_pairs_only": sk2, "rules": lib.table(), "results": results,
        "verdict": verdict}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
