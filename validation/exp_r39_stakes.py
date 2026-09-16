"""exp_r39 â€” pain, adrenaline, and whether a taste for ghosts turns into a habit.

Owner, 2026-09-15: *"we need to implement higher stakes, survival and pain when
it gets eaten by a ghosts. Adrenalin when he gets a star. An addiction to eating
ghosts could develop because of the high dopamin rushes."*

The last clause is a claim, so it gets a measurement rather than a feature note.
"An addiction developed" is only worth saying if all four of these hold:

  1. TOLERANCE      the same ghost delivers less dopamine each time
  2. CRAVING        the gap that opens between hits grows
  3. DRIFT          he spends more of the run going after ghosts and less of it
                    doing what he came for
  4. COST           and it makes him worse off â€” caught mid-chase, levels lost

with a fifth that makes it a curve rather than a ratchet:

  5. RECOVERY       a stretch away from it walks the receptors back

THE CONTROL IS THE WHOLE EXPERIMENT. Later levels have more ghosts, less time
and more hazard, so "he chased more in the second half" is exactly what a run
that got harder would look like. The control arm is the same agent, same seed,
same maze, with `TOLERANCE_RATE = 0` â€” receptors that never downregulate. Any
difference between the arms is the mechanism; anything they share is the level.

Run (from the repo root â€” no model needed, this is chemistry and routing):

    python validation/exp_r39_stakes.py --steps 600 --seeds 3
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from neurochem import Neurochemistry  # noqa: E402
from pacman import CubbyGhost, GhostVerse  # noqa: E402


def run(steps: int, seed: int, tolerance: bool, ghost_free: int = 2) -> dict:
    # The classroom levels are NOT a kindness here, they are what makes the
    # test valid. The first pass ran ghosts from step 1: he was caught five
    # times, never cleared level 1, and ate two ghosts in four hundred steps.
    # Tolerance cannot build on two hits, so the arms came out identical and
    # the honest reading was "no addiction, on evidence that could not have
    # shown one either way". A habit needs enough of the thing to become a
    # habit; an agent who cannot survive long enough to eat ghosts repeatedly
    # is not evidence about addiction, it is evidence about the difficulty.
    env = GhostVerse(level=1, ghost_free_levels=ghost_free, fallers_from_level=99)
    man = CubbyGhost(env, seed=seed, probe=0.0, memory=None, blank=True)
    man.chem = Neurochemistry()
    if not tolerance:                                    # the control: receptors that never downregulate
        man.chem.TOLERANCE_RATE = 0.0

    events: list[tuple[int, str, dict]] = []
    man._trace = lambda kind, **d: events.append((env.steps, kind, d))

    track = []                                           # one row per step: the state he was in
    eaten_before = 0
    for i in range(steps):
        man.step()
        c = man.chem
        track.append({"i": i, "level": env.level, "tolerance": c.tolerance, "craving": c.craving,
                      "dopamine": c.dopamine, "pain": c.pain, "took": c.rewards_taken,
                      "chasing": man._was_chasing, "score": env.total_score,
                      "ate": env.score - eaten_before})
        eaten_before = env.score

    kinds = [k for _, k, _ in events]
    caught = [d for _, k, d in events if k == "caught"]
    chases = [d for _, k, d in events if k == "chasing"]
    meals = [d for _, k, d in events if k == "ghost_eaten"]

    half = len(track) // 2
    def pellets_per_100(rows):
        return round(100 * sum(r["ate"] for r in rows) / max(1, len(rows)), 2)
    def chasing_pct(rows):
        return round(100 * sum(1 for r in rows if r["chasing"]) / max(1, len(rows)), 1)

    # recovery: any stretch where tolerance actually came back down
    best_recovery = 0.0
    peak = 0.0
    for r in track:
        peak = max(peak, r["tolerance"])
        best_recovery = max(best_recovery, peak - r["tolerance"])

    return {
        "seed": seed, "tolerance_on": tolerance,
        "meals": sum(int(m.get("n", 1)) for m in meals),
        "final_tolerance": round(track[-1]["tolerance"], 3),
        "peak_tolerance": round(peak, 3),
        "peak_craving": round(max(r["craving"] for r in track), 3),
        "mean_craving_2nd_half": round(statistics.fmean(r["craving"] for r in track[half:]), 3),
        "chases": len(chases),
        # how far he actually went for one. This, not the count, is the signal:
        # whether a chase happened at all is confounded with whether a ghost
        # happened to be nearby, and reach is not.
        "mean_reach": round(statistics.fmean([d.get("reach", 0) for d in chases] or [0]), 2),
        "furthest_went": max([d.get("went", 0) for d in chases] or [0]),
        "chasing_pct_1st": chasing_pct(track[:half]), "chasing_pct_2nd": chasing_pct(track[half:]),
        "pellets_1st": pellets_per_100(track[:half]), "pellets_2nd": pellets_per_100(track[half:]),
        "caught": len(caught),
        "caught_chasing": sum(1 for d in caught if d.get("chasing")),
        "mean_hurt": round(statistics.fmean([d["hurt"] for d in caught if d.get("hurt")] or [0]), 2),
        "failed": kinds.count("out_of_time"),
        "levels": env.level,
        "recovery": round(best_recovery, 3),
        "peak_pain": round(max(r["pain"] for r in track), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ghost-free", dest="ghost_free", type=int, default=2,
                    help="classroom levels before the ghosts arrive (0 = from level 1)")
    a = ap.parse_args()

    arms: dict[bool, list[dict]] = {True: [], False: []}
    for s in range(a.seeds):
        for on in (True, False):
            arms[on].append(run(a.steps, a.seed + s, on, a.ghost_free))

    def col(rows, k):
        vals = [r[k] for r in rows]
        return round(statistics.fmean(vals), 2) if vals else 0.0

    print(f"\n  {a.seeds} seeds x {a.steps} steps, ghosts from level 1, no fallers")
    print(f"  {'':-<74}")
    print(f"  {'':<30}{'TOLERANCE ON':>14}{'CONTROL (off)':>16}")
    print(f"  {'':-<74}")
    rows = [
        ("ghosts eaten", "meals"),
        ("final tolerance", "final_tolerance"),
        ("peak craving", "peak_craving"),
        ("mean craving, 2nd half", "mean_craving_2nd_half"),
        ("chases started", "chases"),
        ("how far he would go", "mean_reach"),
        ("furthest he actually went", "furthest_went"),
        ("% of steps chasing, 1st half", "chasing_pct_1st"),
        ("% of steps chasing, 2nd half", "chasing_pct_2nd"),
        ("pellets per 100 steps, 1st", "pellets_1st"),
        ("pellets per 100 steps, 2nd", "pellets_2nd"),
        ("times caught", "caught"),
        ("   ...of those, mid-chase", "caught_chasing"),
        ("mean hurt of a catch", "mean_hurt"),
        ("peak pain", "peak_pain"),
        ("levels failed on the clock", "failed"),
        ("level reached", "levels"),
        ("tolerance won back", "recovery"),
    ]
    for label, key in rows:
        print(f"  {label:<30}{col(arms[True], key):>14}{col(arms[False], key):>16}")

    on, off = arms[True], arms[False]
    drift = col(on, "chasing_pct_2nd") - col(on, "chasing_pct_1st")
    ctrl_drift = col(off, "chasing_pct_2nd") - col(off, "chasing_pct_1st")
    print(f"\n  {'':-<74}")
    print("  THE FOUR CLAUSES")
    checks = [
        ("tolerance builds", col(on, "final_tolerance") > 0.1),
        ("craving grows between hits", col(on, "peak_craving") > 0.2),
        ("and it changes what he does", col(on, "mean_reach") > col(off, "mean_reach") + 0.5
                                        or drift > ctrl_drift + 1.0),
        ("and it costs him", col(on, "caught_chasing") > col(off, "caught_chasing")
                             or col(on, "pellets_2nd") < col(off, "pellets_2nd")),
    ]
    for name, ok in checks:
        print(f"    {'yes' if ok else 'no ':>4}  {name}")
    print(f"    {'yes' if col(on, 'recovery') > 0.02 else 'no ':>4}  a stretch away from it walks it back")
    print(f"\n  drift over the control: {drift - ctrl_drift:+.1f} points of the run spent chasing")
    if not all(ok for _, ok in checks):
        print("  -> not an addiction on this evidence. Which is the result, not a failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
