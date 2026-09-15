"""exp_r28 -- how deep can the VM's own verification see? WO-2.6, part one.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE QUESTION
------------
Every depth number this project has is 1, 2 or 3 hops, and `tau_vm` has exactly
three entries to match: {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}, with a
`.get(n, 0.2202)` fallback that silently reuses the 3-hop threshold for
everything deeper. CLUTRR's test split runs to **10 hops**. Before any score at
depth 7 means anything, one thing has to be established:

    `build_chain_program` puts EVERY hop's binding into ONE frame in
    superposition. Each extra hop is one more vector in the bundle, so every
    hop's recovered similarity falls as the chain gets longer. The control role
    -- `ABSENT_CTRL`, never bound -- is the noise floor, and it does not fall.
    Somewhere those two meet. Past that depth **no threshold separates a true
    binding from noise**, and a system that answers there is guessing.

That depth is a property of the VSA and the program shape. It is not a property
of the emitter, so it is measured here with the emitter switched off: the gold
triples are handed straight to `build_chain_program`. If the cliff is at 6, then
a 10-hop CLUTRR score is meaningless whatever the emitter emits, and the honest
report is a refusal, not a number.

WHAT IS MEASURED
----------------
Per depth k, over n chains:

    min / median hop similarity     the weakest TRUE binding in the chain
    max control similarity          the strongest FALSE binding seen
    separation  = min_true / max_ctrl
    headroom    = min_true - tau(k) under the shipped tau table

`separation <= 1.0` is the cliff: at that depth some true binding is weaker than
some control, so any threshold admitting the truth also admits noise.

THE KILL CRITERION, two clauses as the amended rule requires
------------------------------------------------------------
  * correctness -- if the shipped `tau_vm` fallback (0.2202 for every n_hop > 3)
    sits at or below the measured control ceiling at any depth the serving path
    will accept, the fallback is a **wrong-answer generator** and must be
    replaced before CLUTRR is scored at all. This clause is about the threshold,
    not about any model.
  * rate -- separation below 4.0 at a depth, which is `sim_histogram`'s own
    `--min-separation` alarm, marks that depth as unsafe to serve.

Refusing everything past the cliff is the correct behaviour, not a failure.

    python validation/exp_r28_depth_capacity.py --n 40 --max-depth 14
"""
from __future__ import annotations

import argparse, json, pathlib, random, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
TAU_FALLBACK = 0.2202                     # what the serving path uses past hop 3
MIN_SEPARATION = 4.0                      # sim_histogram's own alarm level


def tau(n_hop: int) -> float:
    return TAU_VM.get(n_hop, TAU_FALLBACK)


def synth_chain(k: int, pool_names, pool_rels, rng) -> list[tuple[str, str, str]]:
    """A k-hop chain of the same SHAPE from CLUTRR's own names and relations.

    Only used past the depth CLUTRR supplies. The VM sees strings and a bundle
    size; where the strings came from does not change the capacity being
    measured. Marked `synthetic` in the output so no one reads it as a score.
    """
    ns = rng.sample(pool_names, k + 1) if len(pool_names) > k else \
        [f"{rng.choice(pool_names)}{i}" for i in range(k + 1)]
    return [(ns[i], rng.choice(pool_rels), ns[i + 1]) for i in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=None)
    ap.add_argument("--n", type=int, default=40, help="chains per depth")
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument("--chunk", type=int, default=0,
                    help="0 = the shipped shape (ONE frame holds the whole chain). "
                         "c > 0 = verify the chain in consecutive groups of c hops, "
                         "each group its own frame with its own control role. The "
                         "bundle never exceeds c, so the crosstalk that kills the "
                         "shipped shape at depth 5 does not accumulate. This is a "
                         "MEASUREMENT of a proposed program shape, not a change to "
                         "the serving path.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import clutrr as C
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.planner import Triple
    from cubbyllm.reasoning.programs import build_chain_program

    rng = random.Random(a.seed)
    raw = C.load_raw("test", root=a.bench_root)
    conv = C.verify_convention(raw)
    log(f"CLUTRR gen_train23_test2to10/test: {len(raw)} records; convention check "
        f"{conv['forward']}/{conv['gendered_edges']} gendered edges forward "
        f"(inverse {conv['inverted']}, chance)")

    items = C.items("test", root=a.bench_root)
    by_depth: dict[int, list] = {}
    for it in items:
        by_depth.setdefault(it["hops"], []).append(it)
    log(f"{len(items)} of {len(raw)} records are a simple path from the query's "
        f"first entity to its second; the rest carry forks or noise edges and are "
        f"a different experiment")
    log("available by depth: "
        + json.dumps({k: len(v) for k, v in sorted(by_depth.items())}))

    names = sorted({n for it in items for e in it["chain"] for n in (e[0], e[2])})
    rels = sorted({e[1] for it in items for e in it["chain"]})
    log(f"entity pool {len(names)}, relation pool {len(rels)}: {', '.join(rels)}\n")

    session = cc.CubelangSession(exe=a.exe)
    rows = []
    for k in range(1, a.max_depth + 1):
        pool = by_depth.get(k, [])
        synthetic = len(pool) < a.n
        chains = [it["chain"] for it in rng.sample(pool, min(a.n, len(pool)))] if pool else []
        while len(chains) < a.n:
            chains.append(synth_chain(k, names, rels, rng))

        hop_sims: list[float] = []
        min_per_chain: list[float] = []
        ctrl_sims: list[float] = []
        errors = 0
        for ch in chains:
            triples = [Triple(obj=b, rel=r, subj=aa) for aa, r, b in ch]
            size = a.chunk if a.chunk > 0 else len(triples)
            groups = [triples[i:i + size] for i in range(0, len(triples), size)]
            sims, ctrls = [], []
            try:
                for g in groups:
                    source, fns = build_chain_program(g, [t.rel for t in g])
                    for fn in fns[:-1]:
                        out = session.run(source, fn=fn)
                        s = out.get("similarity")
                        if s is None:
                            raise RuntimeError("no similarity")
                        sims.append(float(s))
                    cs = session.run(source, fn=fns[-1]).get("similarity")
                    ctrls.append(0.0 if cs is None else float(cs))
            except Exception:                                    # noqa: BLE001
                errors += 1
                continue
            hop_sims.extend(sims)
            min_per_chain.append(min(sims))
            ctrl_sims.append(max(ctrls))       # the chain's worst control, not each group's

        if not min_per_chain:
            log(f"depth {k:>2}: no chain ran ({errors} errors)")
            continue
        min_true = min(min_per_chain)
        med = statistics.median(hop_sims)
        max_ctrl = max(ctrl_sims) if ctrl_sims else 0.0
        sep = min_true / max_ctrl if max_ctrl > 0 else float("inf")
        # Under chunking the frame holds `chunk` bindings, not `k`, so the
        # applicable threshold is the one for the GROUP size -- that is the whole
        # point of the shape, and scoring it against tau(k) would hide it.
        t = tau(min(a.chunk, k)) if a.chunk > 0 else tau(k)
        # The number that predicts what the SERVING path does, which `min_true`
        # does not: tau is set at the bundle's expected cosine, so a true binding
        # lands either side of it. Every hop below tau refuses the whole chain,
        # so a chain of k hops survives only if all k clear -- and `hop_below`
        # is the per-hop rate that compounds into that.
        hop_below = sum(s < t for s in hop_sims) / len(hop_sims)
        chain_clears = sum(m >= t for m in min_per_chain) / len(min_per_chain)
        rows.append({"depth": k, "n": len(min_per_chain), "synthetic": synthetic,
                     "min_true": round(min_true, 6), "median_true": round(med, 6),
                     "max_ctrl": round(max_ctrl, 6),
                     "separation": None if sep == float("inf") else round(sep, 3),
                     "tau": t, "headroom": round(min_true - t, 6),
                     "hop_below_tau": round(hop_below, 4),
                     "chain_clears_tau": round(chain_clears, 4),
                     "tau_above_ctrl": t > max_ctrl, "errors": errors})

    session.close()

    shape = (f"CHUNKED, {a.chunk} hops per frame" if a.chunk > 0
             else "the shipped shape: ONE frame holds the whole chain")
    log(f"program shape: {shape}\n")
    log(f"{'depth':>6}{'n':>5}{'min true':>10}{'median':>9}{'max ctrl':>10}"
        f"{'sep':>8}{'tau':>9}{'hop<tau':>9}{'chain ok':>10}  source")
    log("-" * 82)
    for r in rows:
        sep = "inf" if r["separation"] is None else f"{r['separation']:.2f}"
        log(f"{r['depth']:>6}{r['n']:>5}{r['min_true']:>10.4f}{r['median_true']:>9.4f}"
            f"{r['max_ctrl']:>10.4f}{sep:>8}{r['tau']:>9.4f}"
            f"{r['hop_below_tau']:>9.2%}{r['chain_clears_tau']:>10.2%}"
            f"  {'synthetic' if r['synthetic'] else 'CLUTRR'}")
    log("  hop<tau: share of TRUE bindings the threshold rejects. chain ok: share of "
        "chains\n  where every hop clears -- the ceiling on what the serving path can "
        "verify at that depth.")

    cliff = next((r["depth"] for r in rows
                  if r["separation"] is not None and r["separation"] <= 1.0), None)
    unsafe = [r["depth"] for r in rows
              if r["separation"] is not None and r["separation"] < MIN_SEPARATION]
    tau_breach = [r["depth"] for r in rows if not r["tau_above_ctrl"]]
    below_tau = [r["depth"] for r in rows if r["headroom"] < 0]

    log("")
    log(f"cliff (separation <= 1.0, no threshold can separate): "
        f"{cliff if cliff else 'not reached by depth ' + str(a.max_depth)}")
    log(f"unsafe to serve (separation < {MIN_SEPARATION}): {unsafe or 'none'}")
    log(f"tau at or below the control ceiling -- a wrong-answer generator: "
        f"{tau_breach or 'none'}")
    log(f"depths where the weakest TRUE binding is already below tau (the chain "
        f"refuses, which is correct): {below_tau or 'none'}")

    verdict = ("KILLED (correctness clause): the shipped tau fallback does not "
               "clear the control ceiling at " + str(tau_breach)) if tau_breach else (
        f"survives (correctness); depths {unsafe} are below the "
        f"{MIN_SEPARATION}x separation alarm and must refuse" if unsafe else
        "survives both clauses")
    log(f"\nVERDICT: {verdict}")

    stem = f"exp_r28_depth_capacity{a.tag or ('_chunk%d' % a.chunk if a.chunk else '')}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "n_per_depth": a.n, "max_depth": a.max_depth, "seed": a.seed, "chunk": a.chunk,
        "convention": conv, "available": {k: len(v) for k, v in sorted(by_depth.items())},
        "tau_table": TAU_VM, "tau_fallback": TAU_FALLBACK,
        "rows": rows, "cliff": cliff, "unsafe": unsafe, "tau_breach": tau_breach,
        "verdict": verdict, "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
