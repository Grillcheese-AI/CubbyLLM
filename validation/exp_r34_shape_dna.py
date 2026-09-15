"""exp_r34 -- can a shape's DNA be read back, composed, and reasoned over?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE IDEA UNDER TEST
-------------------
Nick's framing (2026-09-15, explicitly "do not take this literally"): a square
carries a structural DNA of properties; from that DNA the system knows a smaller
circle fits inside it without ever having been trained on squares; and mixing two
DNAs invents a new shape -- a "cirsquare". Experience then crystallises into a
capability.

Geometry is the right first testbed because it needs no corpus: ground truth is
COMPUTABLE, so unlimited shapes the system has never seen can be generated and
checked exactly. That is a much cleaner instrument than a benchmark.

A shape's DNA here is exactly what the store's facts already are -- a bundle of
role-filler bindings::

    dna(square) = bundle([bind(KIND, SQUARE), bind(SIDES, 4),
                          bind(WIDTH, L7), bind(HEIGHT, L7), ...])

WHICH MAKES THE FIRST QUESTION A CAPACITY QUESTION
---------------------------------------------------
That bundle is the same object `exp_r28` measured for chain depth, and it found
the recovered similarity HALVES with every element: 1.000, 0.501, 0.246, 0.125,
0.064, 0.032 at 1..6, meeting the noise floor at six. If that transfers, a shape
with six properties is already unreadable -- which would cap the whole
mutable-model idea at five-property objects, and nobody has checked.

So arm 1 is not about shapes at all. It asks how many properties a DNA can hold
before `unbind` stops recovering them, using the same control-role noise floor
`exp_r28` used.

THE ARMS
--------
    capacity    property recovery vs |DNA|, against an unbound-role noise floor
    reasoning   containment questions over shapes NEVER SEEN, answered by
                decomposing both DNAs and comparing -- correct / refused / wrong
    blend       bundle(a, b): where does a "cirsquare" land, and is it READABLE
                or merely noise that scores well

KILL CRITERION, two clauses
---------------------------
  * correctness -- a property recovered ABOVE the noise floor but WRONG is a
    defect, because that is the case nothing downstream can catch. Recovery
    below the floor is a refusal and a result.
  * capacity -- if a DNA cannot hold the properties a plain shape needs (kind,
    sides, width, height is already four) with separation above 4x, the
    representation does not support the idea and chunking is required the way
    it was for chain depth.

    python validation/exp_r34_shape_dna.py
"""
from __future__ import annotations

import argparse, itertools, json, math, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIN_SEPARATION = 4.0                       # sim_histogram's own alarm level

# The DNA's roles, in the order a shape acquires them. KIND and SIDES are the
# identity; WIDTH/HEIGHT are the magnitudes containment needs; the rest are the
# properties a richer world would add, and exist here to push |DNA| up.
ROLES = ["KIND", "SIDES", "WIDTH", "HEIGHT", "FILL", "ROTATION", "COLOUR", "STROKE"]
KINDS = ["circle", "square", "triangle", "hexagon", "pentagon", "octagon"]
SIDES = {"circle": 0, "square": 4, "triangle": 3,
         "hexagon": 6, "pentagon": 5, "octagon": 8}
N_LEVELS = 16                              # quantized magnitude levels


class ShapeWorld:
    """Role-filler DNA over the project's own block-code algebra.

    Values are symbols from one codebook, roles from another, so a role can
    never be mistaken for a value -- cubemind's `__role__:` namespacing trick
    (`_archive/execution/event_encoder.py`), which is the same idea as WO-2.7's
    role-as-vector candidate.
    """

    def __init__(self, vsa, rng) -> None:
        self.vsa = vsa
        names = (["__role__:" + r for r in ROLES]
                 + ["kind:" + k for k in KINDS]
                 + [f"level:{i}" for i in range(N_LEVELS)]
                 + [f"sides:{n}" for n in sorted(set(SIDES.values()))])
        # Built HERE from the passed rng, not via `vsa.codebook()`, which seeds
        # from the fixed config SEED and would make every --seed produce the
        # identical codebook. The first cut did exactly that: a three-seed sweep
        # came back byte-identical and "confirmed" a blend result drawn from one
        # codebook. Same failure as exp_r30's fake orthogonal arm -- a knob that
        # does not reach what it claims to vary.
        import numpy as np
        cb = np.zeros((len(names), vsa.k, vsa.l), dtype=np.float32)
        idx = rng.integers(0, vsa.l, size=(len(names), vsa.k))
        np.put_along_axis(cb, idx[:, :, None], 1.0, axis=2)
        self.vec = {n: cb[i] for i, n in enumerate(names)}
        self.names = names
        self.cb = cb
        # the value codebook used for cleanup, per role
        self.values = {
            "KIND": [("kind:" + k, k) for k in KINDS],
            "SIDES": [(f"sides:{n}", n) for n in sorted(set(SIDES.values()))],
            "WIDTH": [(f"level:{i}", i) for i in range(N_LEVELS)],
            "HEIGHT": [(f"level:{i}", i) for i in range(N_LEVELS)],
            "FILL": [(f"level:{i}", i) for i in range(N_LEVELS)],
            "ROTATION": [(f"level:{i}", i) for i in range(N_LEVELS)],
            "COLOUR": [(f"level:{i}", i) for i in range(N_LEVELS)],
            "STROKE": [(f"level:{i}", i) for i in range(N_LEVELS)],
        }

    def dna(self, props: dict) -> "object":
        parts = [self.vsa.bind(self.vec["__role__:" + r], self.vec[self._key(r, v)])
                 for r, v in props.items()]
        return self.vsa.bundle(parts)

    @staticmethod
    def _key(role: str, value) -> str:
        if role == "KIND":
            return "kind:" + value
        if role == "SIDES":
            return f"sides:{value}"
        return f"level:{value}"

    def recover(self, dna, role: str):
        """unbind then clean up against that role's value codebook.

        Returns (value, similarity). The caller decides whether the similarity
        clears a threshold -- this never guesses on the caller's behalf.
        """
        import numpy as np

        probe = self.vsa.unbind(dna, self.vec["__role__:" + role])
        keys = self.values[role]
        stack = np.stack([self.vec[k] for k, _ in keys])
        sims = self.vsa.similarity_batch(probe, stack)
        i = int(sims.argmax())
        return keys[i][1], float(sims[i])

    def noise_floor(self, dna) -> float:
        """The best similarity an UNBOUND role recovers -- the control.

        Same trick as `programs.py`'s ABSENT_CTRL: a role the DNA never bound
        must come back low, and how low is the floor everything else is read
        against.
        """
        import numpy as np

        best = 0.0
        for r in ROLES:
            probe = self.vsa.unbind(dna, self.vec["__role__:" + r])
            stack = np.stack([self.vec[k] for k, _ in self.values[r]])
            best = max(best, float(self.vsa.similarity_batch(probe, stack).max()))
        return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--l", type=int, default=128)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import numpy as np
    from cubbyllm.ops.vsa import BlockCodeVSA

    vsa = BlockCodeVSA(k=a.k, l=a.l)
    rng = np.random.default_rng(a.seed)
    log(f"BlockCodeVSA k={a.k} l={a.l} (D={a.k * a.l}), backend={vsa.backend()}")
    world = ShapeWorld(vsa, rng)
    log(f"DNA roles: {', '.join(ROLES)}")
    log(f"kinds: {', '.join(KINDS)};  {N_LEVELS} magnitude levels\n")

    def random_props(n_roles: int) -> dict:
        kind = KINDS[int(rng.integers(len(KINDS)))]
        props = {"KIND": kind, "SIDES": SIDES[kind]}
        for r in ROLES[2:n_roles]:
            props[r] = int(rng.integers(N_LEVELS))
        return {r: props[r] for r in ROLES[:n_roles]}

    # ---- arm 1: capacity -- how many properties can a DNA hold? -----------
    log("=" * 74)
    log("arm 1: capacity -- property recovery against an unbound-role floor")
    log("=" * 74)
    log(f"{'|DNA|':>6}{'recovered':>11}{'min sim':>10}{'median':>9}"
        f"{'floor':>9}{'separation':>12}")
    log("-" * 60)
    cap_rows = []
    for n_roles in range(1, len(ROLES) + 1):
        ok = 0; sims = []; floors = []; wrong_above = 0
        for _ in range(a.trials):
            props = random_props(n_roles)
            dna = world.dna(props)
            floor = 0.0
            # the floor for THIS dna: roles it never bound
            unbound = [r for r in ROLES if r not in props]
            for r in unbound:
                probe = vsa.unbind(dna, world.vec["__role__:" + r])
                stack = np.stack([world.vec[k] for k, _ in world.values[r]])
                floor = max(floor, float(vsa.similarity_batch(probe, stack).max()))
            floors.append(floor)
            for r in props:
                val, sim = world.recover(dna, r)
                sims.append(sim)
                if val == props[r]:
                    ok += 1
                elif sim > floor:
                    wrong_above += 1          # the defect class: confident and wrong
        total = a.trials * n_roles
        mn, med = min(sims), statistics.median(sims)
        fl = max(floors) if floors else 0.0
        sep = mn / fl if fl > 0 else float("inf")
        cap_rows.append({"n_roles": n_roles, "recovered": ok, "of": total,
                         "min_sim": round(mn, 4), "median_sim": round(med, 4),
                         "floor": round(fl, 4),
                         "separation": None if fl <= 0 else round(sep, 3),
                         "wrong_above_floor": wrong_above})
        s = "inf" if fl <= 0 else f"{sep:.2f}x"
        log(f"{n_roles:>6}{ok:>6}/{total:<4}{mn:>10.4f}{med:>9.4f}{fl:>9.4f}{s:>12}")
    log("  recovered = unbind + cleanup returned the value that was bound.")
    log("  floor = the best an UNBOUND role scores on the same DNA (the control).")

    cliff = next((r["n_roles"] for r in cap_rows
                  if r["separation"] is not None and r["separation"] < MIN_SEPARATION), None)
    log(f"\n  separation drops below {MIN_SEPARATION}x at |DNA| = "
        f"{cliff if cliff else 'never (within ' + str(len(ROLES)) + ' roles)'}")
    log(f"  a plain shape needs KIND, SIDES, WIDTH, HEIGHT = 4 properties.")

    # ---- arm 2: containment over shapes never seen ------------------------
    log("")
    log("=" * 74)
    log("arm 2: containment on UNSEEN shapes, decided by decomposing both DNAs")
    log("=" * 74)
    n_props = 4                                  # KIND, SIDES, WIDTH, HEIGHT
    tau = None
    # threshold from arm 1: the median of the noise floor at this size, doubled.
    row = next(r for r in cap_rows if r["n_roles"] == n_props)
    tau = max(2.0 * row["floor"], 1e-6)
    log(f"refusal threshold: 2x the measured floor at |DNA|={n_props} -> {tau:.4f}")
    log("  (a property recovered below this is REFUSED, not guessed)")
    c = {"correct": 0, "WRONG": 0, "refused": 0}
    breaches = []
    seen_pairs = set()
    for _ in range(a.questions):
        pa, pb = random_props(n_props), random_props(n_props)
        key = (tuple(pa.items()), tuple(pb.items()))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        da, db = world.dna(pa), world.dna(pb)
        # "does A fit inside B?" -- ground truth is computable, exactly
        gold = (pa["WIDTH"] < pb["WIDTH"]) and (pa["HEIGHT"] < pb["HEIGHT"])
        vals = {}
        refused = False
        for nm, dna, props in (("a", da, pa), ("b", db, pb)):
            for r in ("WIDTH", "HEIGHT"):
                v, s = world.recover(dna, r)
                if s < tau:
                    refused = True
                vals[(nm, r)] = v
        if refused:
            c["refused"] += 1
            continue
        got = (vals[("a", "WIDTH")] < vals[("b", "WIDTH")]) and \
              (vals[("a", "HEIGHT")] < vals[("b", "HEIGHT")])
        if got == gold:
            c["correct"] += 1
        else:
            c["WRONG"] += 1
            if len(breaches) < 5:
                breaches.append({"a": {k: pa[k] for k in ("WIDTH", "HEIGHT")},
                                 "b": {k: pb[k] for k in ("WIDTH", "HEIGHT")},
                                 "recovered": {f"{n}.{r}": vals[(n, r)]
                                               for n, r in vals},
                                 "gold": gold, "got": got})
    n_q = sum(c.values())
    log(f"\n  {n_q} questions over shape pairs generated fresh (no training, no corpus)")
    log(f"  correct {c['correct']}   refused {c['refused']}   WRONG {c['WRONG']}")
    for b in breaches:
        log(f"    WRONG: a={b['a']} b={b['b']} gold={b['gold']} got={b['got']}")
        log(f"           recovered {b['recovered']}")

    # ---- arm 3: the cirsquare --------------------------------------------
    log("")
    log("=" * 74)
    log("arm 3: the blend -- bundle(circle, square), and is it READABLE?")
    log("=" * 74)
    circle = {"KIND": "circle", "SIDES": 0, "WIDTH": 8, "HEIGHT": 8}
    square = {"KIND": "square", "SIDES": 4, "WIDTH": 8, "HEIGHT": 8}
    dc, ds = world.dna(circle), world.dna(square)
    blend = vsa.bundle([dc, ds])
    log(f"  similarity(blend, circle) = {vsa.similarity(blend, dc):.4f}")
    log(f"  similarity(blend, square) = {vsa.similarity(blend, ds):.4f}")
    log(f"  similarity(circle, square) = {vsa.similarity(dc, ds):.4f}   "
        f"(the two parents, for scale)")
    other = world.dna({"KIND": "hexagon", "SIDES": 6, "WIDTH": 3, "HEIGHT": 12})
    log(f"  similarity(blend, an unrelated shape) = {vsa.similarity(blend, other):.4f}"
        f"   <- the floor")
    log("")
    log(f"  {'role':<10}{'recovered':>12}{'sim':>9}   what the blend says it is")
    for r in ("KIND", "SIDES", "WIDTH", "HEIGHT"):
        v, s = world.recover(blend, r)
        note = ""
        if r in ("KIND", "SIDES"):
            note = ("a parent's value -- the blend picks a SIDE, it does not average"
                    if v in (circle[r], square[r]) else "neither parent's value")
        else:
            note = "both parents agree here" if circle[r] == square[r] else ""
        log(f"  {r:<10}{str(v):>12}{s:>9.4f}   {note}")
    blend_rows = {r: list(world.recover(blend, r)) for r in ("KIND", "SIDES",
                                                             "WIDTH", "HEIGHT")}

    # ---- verdict ----------------------------------------------------------
    wrong_above = sum(r["wrong_above_floor"] for r in cap_rows)
    log("")
    if c["WRONG"] or wrong_above:
        verdict = (f"KILLED (correctness): {c['WRONG']} wrong containment answers, "
                   f"{wrong_above} properties recovered above the floor but wrong")
    elif cliff is not None and cliff <= 4:
        verdict = (f"KILLED (capacity): separation falls below {MIN_SEPARATION}x at "
                   f"|DNA|={cliff}, and a plain shape already needs 4 properties. "
                   f"The DNA needs chunking, exactly as chain depth did (exp_r28)")
    else:
        verdict = (f"survives: {c['correct']} correct, {c['refused']} refused, 0 wrong "
                   f"on unseen shapes; DNA holds {(cliff - 1) if cliff else len(ROLES)} "
                   f"properties above {MIN_SEPARATION}x separation")
    log(f"VERDICT: {verdict}")

    stem = f"exp_r34_shape_dna{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "k": a.k, "l": a.l, "seed": a.seed, "roles": ROLES, "kinds": KINDS,
        "n_levels": N_LEVELS, "trials": a.trials,
        "capacity": cap_rows, "capacity_cliff": cliff,
        "containment": {**c, "tau": round(tau, 6), "n": n_q, "breaches": breaches},
        "blend": blend_rows,
        "blend_similarity": {
            "to_circle": round(float(vsa.similarity(blend, dc)), 4),
            "to_square": round(float(vsa.similarity(blend, ds)), 4),
            "parents": round(float(vsa.similarity(dc, ds)), 4),
            "to_unrelated": round(float(vsa.similarity(blend, other)), 4)},
        "verdict": verdict, "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
