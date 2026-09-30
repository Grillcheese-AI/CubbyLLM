"""exp_f2m4_possibility -- H-F2 M4: MoWM as the world model behind the possibility oracle (2026-09-29).

Pre-registered screen for the gates in CUBBYLLM_HYPOTHESES.md (H-F2 M4). Model-free: the production
FastWordEncoder table (v4) encodes text; no LLM is involved. CPU, minutes.

Worlds: the chain-QA fact pool (`exp_m3_cot_pipeline.load_sample`) split into one fact world per relation
family (the `--worlds` most frequent relations, the rest in 'misc'): the same worlds for every oracle, so
the routing rules are compared on identical members.

Measures, per oracle (A: `StorePossibility` over FactStores, today's best-top-score rule; B:
`MoWMPossibility` over MoWM fact worlds -- `mowm_member`, best member per world, the oracle's default; and
`mowm_centroid`, the open-set domain rule with margin gate and inter-world delegation):
  routing     the world holding a question's hop-0 fact vs the routed world -- accuracy, and for B the
              abstain rate (below tau_match or an ambiguous margin: the fall-through to route_world)
  planted     `--plant` stored facts with the object swapped for another object of the same relation:
              the fraction found IMPOSSIBLE (caught), and where the rest went
  held out    `--holdout` true facts REMOVED from the worlds before the screen: the fraction refused
              (IMPOSSIBLE -- the cost of treating every relation as functional), possible, unknown
  members     stored facts asked back: POSSIBLE/member rate (a sanity floor)
  latency     wall time per possible() -- mean and p99, ms

Gates (pre-registered, see the hypothesis): mowm_member routing >= A - 2 pts; planted caught >= 90% for A
and mowm_member; true facts refused <= 2%; members asked back possible >= 98%; A's possible() <= 5 ms mean
at these store sizes (B reported). mowm_centroid is measured, not gated: the smoke run showed the domain
rule's absolute floor does not transfer to fact worlds (65% abstain), which is why 'member' is the default.

Usage (from the repo root, an environment with cubbyllm, numpy, pyarrow and -- for B -- mowm):
    python validation/exp_f2m4_possibility.py --n 1500 --worlds 8 --plant 400 --holdout 400
Log: validation/logs/exp_f2m4_possibility{tag}.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "standin"))              # appended: standin/hypothesis.py must not shadow the package

from cubbyllm.bridges.possibility import StorePossibility, Verdict     # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact           # noqa: E402
from worlds import FactStore, route_world                              # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# the same data and table as exp_m3_cot_pipeline (its `load_sample`, copied so this screen does not import
# that module's VM and haystack dependencies)
PQ_FILE = pathlib.Path(r"E:\valid_scaling_law_with_facts.pq")
V4_TABLE = pathlib.Path(r"I:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz")


def load_sample(n: int, seed: int, pq_file=PQ_FILE):
    """-> (questions, answers, hops, chains, store): `chains[i]` indexes question i's own facts in `store`."""
    import pyarrow.parquet as pq
    t = pq.read_table(pq_file, columns=["question_prompt", "facts", "answer", "n_hop"])
    rows = [(q, f, a, int(h)) for q, f, a, h in zip(
        t.column("question_prompt").to_pylist(), t.column("facts").to_pylist(),
        t.column("answer").to_pylist(), t.column("n_hop").to_pylist())
        if q and f and a and len(q) > 15]
    rng = np.random.default_rng(seed)
    if len(rows) > n:
        pick = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[j] for j in sorted(pick)]
    fact_id: dict[str, int] = {}
    store: list[str] = []

    def fid(s: str) -> int:
        s = " ".join(s.split())
        if s not in fact_id:
            fact_id[s] = len(store)
            store.append(s)
        return fact_id[s]

    chains = [[fid(x) for x in f] for _, f, _, _ in rows]
    return [r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows], chains, store


def build_worlds(store: list[str], n_worlds: int) -> tuple[dict[str, list[str]], dict[str, str]]:
    """facts by world (one per frequent relation, the rest 'misc'), and fact -> world."""
    rel_of: dict[str, str] = {}
    counts: dict[str, int] = {}
    for f in store:
        t = parse_fact(f)
        r = normalize(t.rel) if t is not None else ""
        rel_of[f] = r
        if r:
            counts[r] = counts.get(r, 0) + 1
    top = [r for r, _n in sorted(counts.items(), key=lambda rn: (-rn[1], rn[0]))[:max(1, n_worlds - 1)]]
    worlds: dict[str, list[str]] = {r: [] for r in top}
    worlds["misc"] = []
    home: dict[str, str] = {}
    for f in store:
        w = rel_of[f] if rel_of[f] in worlds else "misc"
        worlds[w].append(f)
        home[f] = w
    return worlds, home


def plant_contradictions(facts: list[str], n: int, rng) -> list[tuple[str, str]]:
    """(planted, original): the object swapped for another object of the same relation (a different one)."""
    by_rel: dict[str, list] = {}
    parsed = []
    for f in facts:
        t = parse_fact(f)
        if t is not None:
            parsed.append((f, t))
            by_rel.setdefault(normalize(t.rel), []).append(t)
    rng.shuffle(parsed)
    out = []
    for f, t in parsed:
        pool = [u.obj for u in by_rel[normalize(t.rel)] if normalize(u.obj) != normalize(t.obj)]
        if not pool:
            continue
        other = pool[int(rng.integers(len(pool)))]
        out.append((f"{other} is the {t.rel} of {t.subj}", f))
        if len(out) >= n:
            break
    return out


def timed(fn, items):
    """-> (results, wall seconds per item)"""
    out, walls = [], []
    for x in items:
        t0 = time.perf_counter()
        out.append(fn(x))
        walls.append(time.perf_counter() - t0)
    return out, np.asarray(walls)


def verdicts(ps) -> dict:
    d = {v.value: 0 for v in Verdict}
    whys: dict[str, int] = {}
    for p in ps:
        d[p.verdict.value] += 1
        whys[p.why] = whys.get(p.why, 0) + 1
    n = max(1, len(ps))
    return {**{k: round(v / n, 4) for k, v in d.items()}, "n": len(ps), "why": whys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--plant", type=int, default=400)
    ap.add_argument("--holdout", type=int, default=400)
    ap.add_argument("--members", type=int, default=400)
    ap.add_argument("--table", default=str(V4_TABLE))
    ap.add_argument("--pq", default=str(PQ_FILE))
    ap.add_argument("--tau-match", type=float, default=0.35)
    ap.add_argument("--tau-margin", type=float, default=0.02)
    ap.add_argument("--tau-answer", type=float, default=0.6)
    ap.add_argument("--no-mowm", action="store_true")
    ap.add_argument("--routing", default="member,centroid", help="MoWM routing rules to measure, comma-separated")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    from mowm.encoding.semantic_words import FastWordEncoder     # the table format lives in mowm; numpy-only
    enc = FastWordEncoder.from_npz(args.table)
    questions, _answers, _hops, chains, store = load_sample(args.n, args.seed, args.pq)
    worlds_facts, home = build_worlds(store, args.worlds)

    # hold out true facts (removed from every world), plant contradictions on what stays
    parsed_ids = [i for i, f in enumerate(store) if parse_fact(f) is not None]
    rng.shuffle(parsed_ids)
    held = [store[i] for i in parsed_ids[:args.holdout]]
    held_set = set(held)
    kept = {w: [f for f in fs if f not in held_set] for w, fs in worlds_facts.items()}
    planted = plant_contradictions([f for fs in kept.values() for f in fs], args.plant, rng)
    members = [f for fs in kept.values() for f in fs]
    rng.shuffle(members)
    members = members[:args.members]
    gold = [home[store[c[0]]] for c in chains]                    # the world holding each question's hop-0 fact
    print(f"worlds: { {w: len(fs) for w, fs in kept.items()} }  held out {len(held)}  planted {len(planted)}",
          flush=True)

    report = {"n": args.n, "seed": args.seed, "worlds": {w: len(fs) for w, fs in kept.items()},
              "held_out": len(held), "planted": len(planted), "questions": len(questions),
              "taus": {"match": args.tau_match, "margin": args.tau_margin, "answer": args.tau_answer},
              "table": str(args.table), "oracles": {}}

    # A: the stores (today's rule)
    stores = {w: FactStore(fs, enc=enc, name=w) for w, fs in kept.items()}
    A = StorePossibility(stores, tau_match=args.tau_match, tau_margin=0.0, tau_answer=args.tau_answer)
    v0 = [route_world(stores, q)[0] for q in questions]
    routed_A, wall_r = timed(lambda q: A.route(q)[0], questions)
    pl_A, wall_p = timed(lambda pf: A.possible(pf[0]), planted)
    ho_A, wall_h = timed(A.possible, held)
    me_A, wall_m = timed(A.possible, members)
    walls = np.concatenate([wall_p, wall_h, wall_m])
    report["oracles"]["store"] = {
        "routing_acc": round(float(np.mean([r == g for r, g in zip(routed_A, gold)])), 4),
        "routing_acc_v0": round(float(np.mean([r == g for r, g in zip(v0, gold)])), 4),
        "route_parity_with_v0": round(float(np.mean([a == b for a, b in zip(routed_A, v0)])), 4),
        "planted": verdicts(pl_A), "held_out": verdicts(ho_A), "members": verdicts(me_A),
        "possible_ms": {"mean": round(float(walls.mean() * 1e3), 3), "p99": round(float(np.quantile(walls, 0.99) * 1e3), 3)},
        "route_ms": round(float(wall_r.mean() * 1e3), 3),
    }
    print("store:", json.dumps(report["oracles"]["store"]), flush=True)

    # B: MoWM fact worlds under both routing rules -- 'member' (the oracle's default: best member per world)
    # and 'centroid' (the open-set domain rule, with margin gate and inter-world delegation)
    for routing in ([] if args.no_mowm else args.routing.split(",")):
        from mowm import MoWMRouter
        from mowm.bridges import MoWMPossibility
        router = MoWMRouter(k=enc.k, l=enc.l, max_worlds=len(kept) + 8, top_k=2, tau_spawn=0.5, seed=args.seed)
        B = MoWMPossibility(router, encode=enc.encode, tau_match=args.tau_match,
                            tau_margin=(args.tau_margin if routing == "centroid" else 0.0),
                            tau_answer=args.tau_answer, predict=False, routing=routing, n_hylas=0)
        t0 = time.perf_counter()
        for w, fs in kept.items():
            B.add_world(w, fs)
        build_s = time.perf_counter() - t0
        routed_B, wall_r = timed(lambda q: B.route(q)[0], questions)
        # the gate's abstentions: what possible() would send back to route_world
        def _abstains(q):
            name, s, m = B.route(q)
            return s < args.tau_match or (m is not None and m < B.tau_margin)
        abst = [_abstains(q) for q in questions]
        pl_B, wall_p = timed(lambda pf: B.possible(pf[0]), planted)
        ho_B, wall_h = timed(B.possible, held)
        me_B, wall_m = timed(B.possible, members)
        walls = np.concatenate([wall_p, wall_h, wall_m])
        report["oracles"][f"mowm_{routing}"] = {
            "routing_acc": round(float(np.mean([r == g for r, g in zip(routed_B, gold)])), 4),
            "routing_acc_when_routed": round(float(np.mean([r == g for r, g, a in zip(routed_B, gold, abst)
                                                            if not a] or [0.0])), 4),
            "abstain_rate": round(float(np.mean(abst)), 4),
            "planted": verdicts(pl_B), "held_out": verdicts(ho_B), "members": verdicts(me_B),
            "delegated": B.delegated, "spawned": len(B.spawned),
            "possible_ms": {"mean": round(float(walls.mean() * 1e3), 3), "p99": round(float(np.quantile(walls, 0.99) * 1e3), 3)},
            "route_ms": round(float(wall_r.mean() * 1e3), 3), "build_s": round(build_s, 2),
        }
        print(f"mowm_{routing}:", json.dumps(report["oracles"][f"mowm_{routing}"]), flush=True)

    # the gates (pre-registered in CUBBYLLM_HYPOTHESES.md, H-F2 M4)
    S = report["oracles"]["store"]
    gates = {"planted_caught_store>=0.90": S["planted"]["impossible"] >= 0.90,
             "true_refused_store<=0.02": S["held_out"]["impossible"] <= 0.02,
             "possible_ms_store<=5": S["possible_ms"]["mean"] <= 5.0}
    if "mowm_member" in report["oracles"]:
        M = report["oracles"]["mowm_member"]
        gates.update({"routing_mowm>=store-0.02": M["routing_acc"] >= S["routing_acc"] - 0.02,
                      "planted_caught_mowm>=0.90": M["planted"]["impossible"] >= 0.90,
                      "true_refused_mowm<=0.02": M["held_out"]["impossible"] <= 0.02,
                      "members_possible_mowm>=0.98": M["members"]["possible"] >= 0.98})
    report["gates"] = gates
    report["passed"] = all(gates.values())
    logs = ROOT / "validation" / "logs"
    logs.mkdir(exist_ok=True)
    out = logs / f"exp_f2m4_possibility{args.tag}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("gates:", json.dumps(gates), "->", "PASS" if report["passed"] else "FAIL")
    print("wrote", out)


if __name__ == "__main__":
    main()
