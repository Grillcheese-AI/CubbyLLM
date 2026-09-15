"""exp_r25 -- the zone ablation. WO-0.6.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

WHY
---
`exp_r22` ablates one zone: the emitter. Every other zone in the architecture --
the relation gate, the repair loop, the index, the acceptance threshold -- has
never been removed and measured. The organizing frame is "brain zones, each with
a specialty", and a zone earns that name three ways:

  1. ABLATION  -- removing it changes measured behaviour. Otherwise decorative.
  2. CHANNEL   -- it talks through a typed seam, not shared state. Otherwise it
                  is not a separate zone at all.
  3. REUSE     -- it serves more than one task. Otherwise it is memorization
                  with a name (the WO-0.3 pathology, one level up).

Test 2 is static and lives in `tests/test_guards.py`
(`test_wired_modules_have_a_production_importer`, added the same day: three
modules declared `Wiring.WIRED` with no production importer at all). This script
is test 1.

THE SEAMS
---------
`learn_and_answer` takes every zone as an argument, so the ablation is a
parameter change rather than a code path:

    plan=        the emitter's program        -> None: the host parses instead
    known=       the relation vocabulary gate -> None: nothing disposes of the plan
    max_repairs= the repair loop              -> 0:    one shot, no second try
    top_k=       retrieval breadth            -> 1:    the first candidate only
    tau_vm=      the VM acceptance threshold  -> 0.0:  accept any binding

THE PLAN IS EMITTED ONCE AND REUSED across every arm that does not ablate the
emitter. Re-emitting per arm would put generation variance inside the comparison,
which is the mistake the val-split sampler made (docs/PATH.md 6.8): the thing
being measured must not move with the thing being varied.

READING IT
----------
Three columns, never one. A zone whose removal turns answers into REFUSALS is
load-bearing for coverage; a zone whose removal turns answers into WRONG is
load-bearing for the kill line, and that is the stronger result. `tau_zero` is
in here as a positive control: if dropping the acceptance threshold to 0.0 does
NOT produce wrong answers, then tau is not doing anything and every number that
rests on it needs re-reading.

    python validation/exp_r25_zone_ablation.py --n 600 --seed 7
"""
from __future__ import annotations

import argparse, collections, json, math, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}

#: arm -> the seam it removes. `full` changes nothing and is the baseline.
ARMS = {
    "full":        {},
    "no_plan":     {"plan": None},
    "permissive":  {"known": "permissive"},
    "no_repairs":  {"max_repairs": 0},
    "top_k_1":     {"top_k": 1},
    "tau_zero":    {"tau_vm": 0.0},
    "tau_floor":   {"tau_vm": 0.0332},
}


class PermissiveRelations:
    """The relation gate with its judgement removed but its plumbing intact.

    First cut of this arm passed `known=None`, which is a documented-optional
    argument -- and it raised `argument of type 'NoneType' is not iterable` on
    15 of 20 questions. That arm was measuring a crash, not a zone, and it would
    have reported 5/20 as if the gate were carrying 75% of the answers.

    So: delegate everything real to the store's own vocabulary, but claim to
    contain every relation and to match every relation to itself. The gate is
    present, asked, and always says yes. That isolates *disposal* -- whether
    refusing an unrecognised relation is what keeps wrong answers out.
    """

    def __init__(self, inner):
        self._inner = inner

    def __contains__(self, rel) -> bool:
        return True                      # every relation is admissible

    def match(self, rel):
        return rel                       # ... and resolves to itself

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)   # declare / add / words / reused


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """The eval set's noise floor, so a contribution can be read against
    something rather than eyeballed. Same function as exp_r22."""
    if n <= 0:
        return 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(half, abs(centre - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen3", default=str(ROOT / "standin" / "data" / "out" / "gen3_free_text.jsonl"))
    ap.add_argument("--encyclopedia", default=str(LOGS / "exp_r16_encyclopedia_all.json"))
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v14e_nochain.Q4_K_M.gguf"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import random
    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from build_gen3 import NullSource
    from emitter import LlamaCppEmitter
    from eval_emitter_vm import strip_fences
    from exp_r9_matched_pairs import emitted_plan
    from exp_r11_search_learn import match
    from sources import PropertyAliases

    held = [json.loads(l) for l in open(a.gen3, encoding="utf-8")]
    held = [r for r in held if r["task"] == "plan" and r["split"] == "held"]
    rows = random.Random(a.seed).sample(held, min(a.n, len(held)))

    wk.ensure_data(("triplets",))
    world = wk.wiki_world(); known_full = StoreRelations(world.index._seen)
    n_enc = 0
    if a.encyclopedia and pathlib.Path(a.encyclopedia).is_file():
        for r in json.loads(pathlib.Path(a.encyclopedia).read_text(encoding="utf-8"))["facts"]:
            fact = f"{r['obj']} is the {r['rel']} of {r['entity']}"
            if fact not in world:
                known_full.declare(r["rel"]); world.index.declare_relation(r["rel"])
                world.add(fact); known_full.add(fact); n_enc += 1
    src = NullSource(PropertyAliases())
    session = cc.CubelangSession(exe=a.exe)

    def run_fn(source, fn):
        return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    arms = [x for x in a.arms.split(",") if x in ARMS]
    log(f"exp_r25 zone ablation | {len(rows)} held questions (seed {a.seed}) | store {len(world):,} facts "
        f"({n_enc:,} encyclopedia) | emitter {pathlib.Path(a.gguf).name}")
    log(f"arms: {', '.join(arms)}\n")

    # ---- emit ONCE, reuse across arms -------------------------------------
    em = LlamaCppEmitter(a.gguf)
    plans: list[QuestionPlan | None] = []
    te = time.perf_counter()
    for i, r in enumerate(rows):
        try:
            ep = emitted_plan(strip_fences(em.emit(r["prompt"], max_new_tokens=a.max_new)), normalize)
        except Exception:                                      # noqa: BLE001
            plans.append(None); continue
        plans.append(
            QuestionPlan(relations=[None] + ep[0][1:], tail=f"{ep[0][0]} of {ep[1]}", n_hop=len(ep[0]))
            if ep and ep[0] and ep[1] else None)
        if (i + 1) % 200 == 0:
            log(f"  emitted {i + 1}/{len(rows)} ({time.perf_counter() - te:.0f}s)")
    n_planned = sum(1 for p in plans if p is not None)
    log(f"  emitted {len(rows)} plans in {time.perf_counter() - te:.0f}s; {n_planned} well-formed\n")

    # ---- one pass per arm, identical plans --------------------------------
    stats: dict[str, collections.Counter] = {}
    wrong_ex: dict[str, list] = {}
    for arm in arms:
        over = ARMS[arm]
        c = collections.Counter(); wrong_ex[arm] = []
        ta = time.perf_counter()
        for r, plan in zip(rows, plans):
            q, gold = r["prompt"], r["gold"]
            use_plan = None if "plan" in over else plan
            if use_plan is None and "plan" not in over:
                c["no_plan"] += 1; continue
            tau = over["tau_vm"] if "tau_vm" in over else TAU_VM.get(
                getattr(use_plan, "n_hop", 2) if use_plan else 2, 0.2202)
            try:
                lr = learn_and_answer(
                    q, no_search, run_fn, store=world,
                    known=PermissiveRelations(known_full) if "known" in over else known_full,
                    source=src, tau_vm=tau, top_k=over.get("top_k", 3),
                    max_repairs=over.get("max_repairs", 1), plan=use_plan)
            except Exception as e:                             # noqa: BLE001
                c["harness_error"] += 1
                if len(wrong_ex[arm]) < 3:
                    wrong_ex[arm].append({"q": q, "error": str(e)[:160]})
                continue
            res = lr.result
            if res.verified:
                m = match(res.answer, gold, normalize)
                c["verified"] += 1; c[m] += 1
                if m == "WRONG" and len(wrong_ex[arm]) < 8:
                    wrong_ex[arm].append({"q": q, "gold": gold, "got": res.answer})
            else:
                c["refused"] += 1; c[f"why:{res.reason or 'failed'}"] += 1
        stats[arm] = c
        log(f"  {arm:<12} correct {c['correct']:>4}  refused {c['refused']:>4}  "
            f"WRONG {c['WRONG']:>3}  near {c['near']:>3}  ({time.perf_counter() - ta:.0f}s)")
    session.close()

    # ---- the table --------------------------------------------------------
    base = stats[arms[0]]["correct"] if arms else 0
    floor = wilson_halfwidth(base / max(1, len(rows)), len(rows))
    log(f"\nnoise floor (Wilson half-width at n={len(rows)}): {floor:.4f} "
        f"= {floor * len(rows):.1f} questions\n")
    log(f"{'arm':<12}{'correct':>8}{'refused':>9}{'WRONG':>7}{'contribution':>14}{'  verdict'}")
    log("-" * 66)
    for arm in arms:
        c = stats[arm]
        contrib = (base - c["correct"]) / max(1, len(rows))
        if arm == arms[0]:
            verdict = "baseline"
        elif c["WRONG"] > stats[arms[0]]["WRONG"]:
            verdict = "LOAD-BEARING (kill line)"
        elif contrib > floor:
            verdict = "load-bearing (coverage)"
        else:
            verdict = "inside the noise floor"
        log(f"{arm:<12}{c['correct']:>8}{c['refused']:>9}{c['WRONG']:>7}{contrib:>+14.4f}  {verdict}")

    # WHY the refusals, per arm. An ablation that converts answers into refusals
    # and one that converts them into wrong answers are different results, and
    # the reason code is what tells them apart -- on the smoke run `tau_zero`
    # refused 16 of 20 where a loosened acceptance threshold was predicted to
    # produce wrong answers instead, and the reason codes are the only way to
    # find out whether the chain caught itself downstream or never got started.
    log("\nrefusal reasons by arm (the shape of the loss, not just its size):")
    for arm in arms:
        why = {k[4:]: v for k, v in stats[arm].items() if k.startswith("why:")}
        extra = {k: stats[arm][k] for k in ("harness_error", "no_plan") if stats[arm][k]}
        if why or extra:
            log(f"  {arm:<12}{json.dumps(dict(sorted(why.items(), key=lambda kv: -kv[1])))}"
                f"{'  ' + json.dumps(extra) if extra else ''}")

    for arm in arms:
        if wrong_ex[arm]:
            log(f"\n{arm} -- first wrong/error examples:")
            for w in wrong_ex[arm]:
                log("   " + json.dumps(w)[:190])

    stem = f"exp_r25_zone_ablation{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "n": len(rows), "seed": a.seed, "gguf": pathlib.Path(a.gguf).name,
        "world_facts": len(world), "noise_floor": floor,
        "arms": {k: dict(v) for k, v in stats.items()},
        "wrong_examples": wrong_ex,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
