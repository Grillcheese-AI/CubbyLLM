"""exp_r30 -- does the probe discriminate, or does it always produce a sentence?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

WHAT THIS IS
------------
cubemind's `_archive/execution/vsa_translator.py` (March 2026) has an idea worth
taking: to find out what an opaque VSA vector DOES, bind it against every concept
in a codebook and read off which concept the result lands on. A vector that maps
every concept to itself is a constant; one that maps `a -> b` is a transform. It
turns an unreadable hypervector into a sentence, with **no model in the loop** --
which is the thing this repo has no way to do, and the reason WO-0.3 can say the
role vocabulary is 95-98% per-relation but not what any role vector actually is.

Its own 205 archived tests pass. They check shapes, keys and determinism. They do
NOT check the only thing that matters before porting:

    A probe that always returns a confident sentence is not an instrument.

So this is the §6 test applied before the port rather than after: run it on data
known to be healthy and require it to say so, run it on noise and require it to
say THAT, and run it on this repo's algebra rather than cubemind's.

THE ARMS
--------
    identity     specialist = zero()             -> every concept is a CONSTANT
    transform    specialist = unbind(b, a)       -> the probe must report a -> b
    noise        specialist = a fresh random vec -> must NOT look like either
    orthogonal   the same three, on a codebook built with orthogonal=True

`orthogonal` is here because `ops/vsa.py` warns in its own docstring that the
structured codebook produces systematic unbind crosstalk onto a neighbour and
role/filler binding wants `orthogonal=False`. If the probe only works on one of
them, that is a porting constraint, and it is cheaper to find it here.

KILL CRITERION, two clauses
---------------------------
  * correctness -- the probe must recover a CONSTRUCTED transform (`a -> b` where
    the specialist was built as `unbind(b, a)`) at a similarity clearly above what
    a random vector achieves. If it cannot read a mapping it was handed, it cannot
    read one it was not.
  * separation -- the best match on a random specialist must be far below the best
    match on a real one. `sim_histogram`'s own alarm level is 4x. A probe whose
    noise floor is close to its signal will narrate nonsense about real vectors,
    and the §6 pattern says that is exactly how an instrument lies.

    python validation/exp_r30_vsa_probe.py --n-concepts 64
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIN_SEPARATION = 4.0                       # sim_histogram's own alarm level
CONSTANT_THRESHOLD = 0.85                  # cubemind's; re-measured here, not assumed
# The floor the port must add. Measured, not chosen: a random specialist's best
# match tops out around 0.09 on a 64-concept codebook while a real one reaches
# 1.0, so 0.30 sits an order of magnitude above the noise and far below any
# genuine binding. The arms below check both directions -- it must silence noise
# AND leave the identity vector fully readable.
FLOOR = 0.30


class Probe:
    """The port, as small as it can be. `vsa` is a `BlockCodeVSA`.

    cubemind's version is a class over `BlockCodes` with the same two methods;
    the only changes are the facade's name and taking the codebook as an array
    plus a name list instead of a dict, because `ops.vsa.codebook()` returns an
    array. The algebra is identical.
    """

    def __init__(self, vsa, names: list[str], vectors) -> None:
        self.vsa = vsa
        self.names = names
        self.vectors = vectors                      # (n, k, l)

    def probe(self, specialist, concept_idx: int) -> dict:
        probed = self.vsa.bind(self.vectors[concept_idx], specialist)
        sims = self.vsa.similarity_batch(probed, self.vectors)
        best = int(sims.argmax())
        order = sims.argsort()[::-1]
        runner_up = float(sims[order[1]]) if len(order) > 1 else 0.0
        return {"input": self.names[concept_idx], "name": self.names[best],
                "similarity": float(sims[best]), "runner_up": runner_up,
                "margin": float(sims[best]) - runner_up}

    def translate(self, specialist, threshold: float = CONSTANT_THRESHOLD,
                  floor: float | None = None) -> dict:
        """`floor`: the minimum similarity a probe must reach to be REPORTED.

        cubemind's version has no floor, and that is the one change the port must
        make. Its `translate()` computes a similarity for every concept and then
        throws it away when composing the summary, so a vector that encodes
        nothing still comes back as `transforms: c00 -> c39, c01 -> c37, ...` --
        a fluent sentence about noise. Measured here: the best match on a random
        specialist is 0.06-0.09 against 1.0 for a real one, so the information
        needed to refuse was present and discarded.

        With a floor, a probe below it is `unreadable` and the summary says so.
        A refusal is a result; a confident wrong answer is a defect.
        """
        probes, transforms, constants, unreadable = [], [], [], 0
        for i, name in enumerate(self.names):
            r = self.probe(specialist, i)
            if floor is not None and r["similarity"] < floor:
                r["kind"] = "unreadable"; unreadable += 1
            elif r["name"] == name and r["similarity"] >= threshold:
                r["kind"] = "constant"; constants.append(name)
            else:
                r["kind"] = "transform"; transforms.append(f"{name} -> {r['name']}")
            probes.append(r)

        def listing(label, items):
            return (f"{label}: " + ", ".join(items[:6])
                    + (f" (+{len(items) - 6} more)" if len(items) > 6 else ""))

        parts = []
        if transforms:
            parts.append(listing("transforms", transforms))
        if constants:
            parts.append(listing("constants", constants))
        if unreadable:
            parts.append(f"unreadable: {unreadable}/{len(self.names)} below the floor")
        if not transforms and not constants and unreadable:
            summary = (f"UNREADABLE -- no concept binds above the floor "
                       f"({unreadable}/{len(self.names)}); this vector encodes "
                       f"nothing the codebook can name")
        else:
            summary = "; ".join(parts) or "no mappings detected"
        return {"probes": probes, "summary": summary, "unreadable": unreadable}


def stats(probes: list[dict]) -> dict:
    sims = [p["similarity"] for p in probes]
    return {"best_mean": round(statistics.mean(sims), 4),
            "best_max": round(max(sims), 4),
            "best_min": round(min(sims), 4),
            "margin_mean": round(statistics.mean(p["margin"] for p in probes), 4),
            "constants": sum(p["kind"] == "constant" for p in probes),
            "transforms": sum(p["kind"] == "transform" for p in probes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-concepts", type=int, default=64)
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--l", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=20, help="constructed transforms to test")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import numpy as np
    from cubbyllm.ops.vsa import BlockCodeVSA

    vsa = BlockCodeVSA(k=a.k, l=a.l)
    log(f"BlockCodeVSA k={a.k} l={a.l} (D={a.k * a.l}), backend={vsa.backend()}")

    def build_codebook(kind: str):
        """Build the codebook HERE rather than trusting `vsa.codebook(orthogonal=)`.

        The first cut of this experiment passed the flag through and the two arms
        came back byte-identical -- which is how the bug showed. `ops/vsa.py`
        honours `orthogonal` only on the grilly path; its numpy fallback ignores
        the argument entirely, and this venv has no grilly (backend is reported
        above). So the 'orthogonal' arm was testing the independent codebook
        twice: a fake arm that would have signed off on a claim it never tested.
        The structured construction is one line, so the arm is now real on any
        backend.
        """
        base = vsa.codebook(a.n_concepts, orthogonal=False)
        if kind == "independent":
            return base
        cb = np.empty_like(base)
        cb[0], cb[1] = base[0], base[1]
        for i in range(2, a.n_concepts):          # cb[i] = bind(cb[i-1], cb[1])
            cb[i] = vsa.bind(cb[i - 1], cb[1])
        return cb

    results: dict[str, dict] = {}
    for cb_kind in ("independent", "orthogonal"):
        vectors = build_codebook(cb_kind)
        names = [f"c{i:02d}" for i in range(a.n_concepts)]
        # a codebook whose entries are near-duplicates cannot discriminate anything,
        # so report how distinct it actually is before reading any probe off it
        off = [float(vsa.similarity(vectors[i], vectors[j]))
               for i in range(min(16, a.n_concepts))
               for j in range(min(16, a.n_concepts)) if i != j]
        pr = Probe(vsa, names, vectors)
        rng = np.random.default_rng(a.seed)
        log(f"\n{'=' * 70}\ncodebook: {cb_kind} ({a.n_concepts} concepts)\n{'=' * 70}")
        log(f"off-diagonal similarity: mean {statistics.mean(off):.4f}, "
            f"max {max(off):.4f}  (near 0 = distinct concepts)")

        # ---- identity: bind(x, zero) == x, so everything must be a constant ----
        ident = pr.translate(vsa.zero())
        s_ident = stats(ident["probes"])
        log(f"identity   best {s_ident['best_mean']:.4f} (min {s_ident['best_min']:.4f})  "
            f"constants {s_ident['constants']}/{a.n_concepts}")
        log(f"  summary: {ident['summary'][:120]}")

        # ---- transform: specialist built so that a MUST map to b --------------
        # specialist = unbind(b, a)  =>  bind(a, specialist) = b
        recovered = 0
        t_sims, t_margins, misses = [], [], []
        for t in range(a.pairs):
            i, j = int(rng.integers(a.n_concepts)), int(rng.integers(a.n_concepts))
            if i == j:
                j = (j + 1) % a.n_concepts
            specialist = vsa.unbind(vectors[j], vectors[i])
            r = pr.probe(specialist, i)
            t_sims.append(r["similarity"]); t_margins.append(r["margin"])
            if r["name"] == names[j]:
                recovered += 1
            elif len(misses) < 4:
                misses.append({"wanted": names[j], "got": r["name"],
                               "sim": round(r["similarity"], 4)})
        t_mean = statistics.mean(t_sims)
        log(f"transform  recovered {recovered}/{a.pairs}  best {t_mean:.4f}  "
            f"margin {statistics.mean(t_margins):.4f}")
        for m in misses:
            log(f"    miss: wanted {m['wanted']}, got {m['got']} at {m['sim']}")

        # ---- noise: a specialist that encodes nothing -------------------------
        noise_best, noise_margin = [], []
        noise_summary = noise_floored = ""
        for t in range(a.pairs):
            # a fresh random one-hot code per trial, independent of the codebook
            spec = np.zeros((a.k, a.l), dtype=np.float32)
            idx = rng.integers(0, a.l, size=a.k)
            np.put_along_axis(spec, idx[:, None], 1.0, axis=1)
            out = pr.translate(spec)                      # no floor: cubemind's version
            sn = stats(out["probes"])
            noise_best.append(sn["best_max"]); noise_margin.append(sn["margin_mean"])
            if not noise_summary:
                noise_summary = out["summary"]
                noise_floored = pr.translate(spec, floor=FLOOR)["summary"]
        n_mean, n_max = statistics.mean(noise_best), max(noise_best)
        log(f"noise      best_max mean {n_mean:.4f}, worst {n_max:.4f}  "
            f"margin {statistics.mean(noise_margin):.4f}")
        log(f"  WITHOUT a floor (cubemind's version) it says:")
        log(f"    {noise_summary[:150]}")
        log(f"  WITH floor={FLOOR}:")
        log(f"    {noise_floored[:150]}")

        # and the floor must not silence a REAL vector: identity must survive it
        ident_floored = pr.translate(vsa.zero(), floor=FLOOR)
        log(f"  the same floor on the identity vector: "
            f"{ident_floored['unreadable']}/{a.n_concepts} unreadable "
            f"(must be 0 -- a floor that refuses real vectors is worse than none)")

        sep = t_mean / n_max if n_max > 0 else float("inf")
        log(f"\nseparation (constructed transform / worst noise): {sep:.2f}x")
        results[cb_kind] = {
            "identity": s_ident, "identity_summary": ident["summary"],
            "transform_recovered": recovered, "transform_of": a.pairs,
            "transform_best_mean": round(t_mean, 4),
            "transform_margin_mean": round(statistics.mean(t_margins), 4),
            "noise_best_max_mean": round(n_mean, 4), "noise_best_max_worst": round(n_max, 4),
            "noise_summary": noise_summary,
            "noise_summary_with_floor": noise_floored,
            "floor": FLOOR,
            "identity_unreadable_at_floor": ident_floored["unreadable"],
            "codebook_offdiag_mean": round(statistics.mean(off), 4),
            "codebook_offdiag_max": round(max(off), 4),
            "separation": None if sep == float("inf") else round(sep, 3),
            "misses": misses,
        }

    # ---- the verdict ------------------------------------------------------
    log(f"\n{'=' * 70}")
    log(f"{'codebook':<14}{'recovered':>12}{'signal':>9}{'noise':>9}{'sep':>8}   verdict")
    log("-" * 70)
    verdicts = {}
    for cb_kind, r in results.items():
        ok_correct = r["transform_recovered"] == r["transform_of"]
        sep = r["separation"] or float("inf")
        if not ok_correct:
            v = "KILLED (correctness): cannot read a mapping it was handed"
        elif sep < MIN_SEPARATION:
            v = f"KILLED (separation): noise is within {sep:.1f}x of signal"
        elif r["identity_unreadable_at_floor"]:
            v = f"KILLED (the floor): it silences {r['identity_unreadable_at_floor']} real bindings"
        else:
            v = "usable WITH the floor"
        verdicts[cb_kind] = v
        log(f"{cb_kind:<14}{r['transform_recovered']:>5}/{r['transform_of']:<6}"
            f"{r['transform_best_mean']:>9.4f}{r['noise_best_max_worst']:>9.4f}"
            f"{sep:>8.2f}   {v}")

    ok = [c for c, v in verdicts.items() if v.startswith("usable")]
    log("")
    if len(ok) == len(verdicts):
        log("VERDICT: the probe discriminates on both codebooks -- port it, WITH the")
        log("floor. As inherited it has none: it computes a similarity for every")
        log("concept and discards it when composing the summary, so a vector that")
        log("encodes nothing still comes back as a fluent list of transforms. The")
        log("information needed to refuse was already there. That is not a port")
        log("detail -- it is the difference between an instrument and a narrator.")
    elif ok:
        log(f"VERDICT: usable on {ok} only. That is a PORTING CONSTRAINT, not a "
            f"detail -- a probe applied to the wrong codebook narrates nonsense.")
    else:
        log("VERDICT: do not port as-is. It produces a confident sentence either way, "
            "which is the §6 failure mode exactly.")

    stem = f"exp_r30_vsa_probe{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "source": "cubemind _archive/execution/vsa_translator.py (March 2026); its own "
                  "205 archived tests pass, and they do not test discrimination",
        "k": a.k, "l": a.l, "n_concepts": a.n_concepts, "pairs": a.pairs, "seed": a.seed,
        "backend": vsa.backend(), "min_separation": MIN_SEPARATION,
        "constant_threshold": CONSTANT_THRESHOLD,
        "results": results, "verdicts": verdicts,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
