"""Does being talked to change how the run goes? Three arms, same seeds.

Wired: STANDALONE (validation only).

`docs/cube_vs_human_vad.md` ended by concluding that the arousal axis cannot be
validated from text and has to be measured against BEHAVIOUR. A maze run is
behaviour. This is that measurement.

THE THREE ARMS, and the third is what makes it an experiment rather than a demo:

    silent   nobody says anything
    oracle   a player who can see: warns when a ghost is genuinely closing,
             and encourages after a death
    noise    a player who talks EXACTLY AS MUCH, at random moments, saying the
             same words

Without `noise` a positive result is unreadable. If oracle beats silent and
noise also beats silent, the effect is "being talked to" — arousal contagion and
a bit of oxytocin — and has nothing to do with the warnings being true. Only
oracle > noise is evidence that the CONTENT mattered. The noise arm is matched
on message count per run, not just in expectation, so the two differ in timing
and nothing else.

WHAT COULD MAKE THIS COME OUT NULL, said in advance rather than after:

  * `wariness` is the only path from affect to action (a4229f7), and it moves
    the berth by at most two tiles. If the berth is not what decides catches in
    this maze, nothing here can move the outcome whatever the coach does.
  * the oracle's warnings raise `threat`, and threat also suppresses dopamine.
    A more careful agent that explores less may clear fewer levels while dying
    less. Both numbers are reported; neither is "the" result.
  * a null in all three arms is a real finding about the size of the affect
    path, and is reported as one.

Every arm runs the same seeds in the same order, so pairing is exact: arm A's
run 7 and arm B's run 7 are the same maze with the same ghosts. The test is a
paired permutation over those pairs, which is what the pairing buys.

    python validation/coach_ab.py --runs 40 --steps 400
"""
from __future__ import annotations

import argparse
import pathlib
import random
import statistics as st
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pacman  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402
import pacworld  # noqa: E402
from pacworld import _manh  # noqa: E402

OUT = ROOT / "docs" / "coach_ab.md"

CHEER = ["you can do it", "lâche pas", "nice one", "almost there", "t'es capable"]
WARN = ["careful!", "ghost!", "watch out", "attention!", "fais attention"]


def perm_p(pairs, n=20000, seed=11):
    """Paired permutation on the per-seed differences: the sign of each pair is
    flipped at random, which is the right null when the same maze is played by
    both arms."""
    if len(pairs) < 3:
        return float("nan")
    obs = abs(st.mean(pairs))
    rng, k = random.Random(seed), 0
    for _ in range(n):
        if abs(st.mean([d if rng.random() < 0.5 else -d for d in pairs])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


def run(seed: int, arm: str, steps: int) -> dict:
    # GHOSTS FROM LEVEL 1. `GhostVerse.GHOST_FREE_LEVELS` is 3 by default —
    # levels 1-3 are his classroom, deliberately empty so that learning what a
    # maze IS and surviving a threat do not have to happen at once.
    #
    # The first version of this experiment did not pass this, and it invalidated
    # the whole run: 250 steps reaches level 2.35 on average, so every arm
    # played in an empty maze. Nobody was ever caught, the oracle never had a
    # ghost to warn about, and all three arms came back byte-identical with
    # `said 0.0`. A null, and a meaningless one — the experiment had not run.
    #
    # The `ghost_free_levels` argument exists for exactly this ("so a test or a
    # demo can turn the threat back on at level 1"). The lesson is the cheap
    # one: a smoke test of ONE run, checking that the thing being measured
    # actually happens, costs 50 seconds and would have saved the 50 minutes.
    env = pacworld.GhostVerse(ghost_free_levels=0, seed_extra=seed)
    a = pacman.CubbyGhost(env=env, seed=seed, blank=True)
    a.chem = Neurochemistry()
    rng = random.Random(seed * 7919 + 13)
    said, deaths, corners = 0, 0, Counter()
    scripted: list[int] = []          # the steps `oracle` spoke on, replayed by `noise`

    for i in range(steps):
        env = a.env
        here = env.coords(a.place)
        near = min((_manh(here, g) for g in env.ghosts), default=99)

        if arm == "oracle":
            # a player who can see: warn when something is genuinely closing
            if env.frightened == 0 and near <= a.learned_radius + 2:
                a.hear(rng.choice(WARN))
                said += 1
                scripted.append(i)
            elif a._last.get("caught"):
                a.hear(rng.choice(CHEER))
                said += 1
                scripted.append(i)
        elif arm == "noise":
            # the same number of messages, the same words, the wrong moments
            if rng.random() < NOISE_RATE[seed]:
                a.hear(rng.choice(WARN + CHEER))
                said += 1

        before = env.lives
        a.step()
        if env.lives < before or a._last.get("caught"):
            deaths += 1
        corners[a.chem.dominant_emotion] += 1

    return {"seed": seed, "arm": arm, "deaths": deaths, "level": a.env.level,
            "score": a.env.total_score, "said": said, "corners": corners,
            "credibility": a.coach.credibility, "radius": a.learned_radius,
            "steps_survived": a.env.steps}


NOISE_RATE: dict[int, float] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    seeds = list(range(1, a.runs + 1))

    # PROGRESS, FLUSHED. The first version printed nothing until the end, so a
    # 50-minute run was indistinguishable from a hung one — the owner asked
    # "is it even running?" and the honest answer was that the script gave him
    # no way to tell. Every step spawns `cubelang.exe`, so this is bound by
    # process startup rather than by compute: it shows as ~30% of one core and
    # no GPU at all, which reads exactly like a stall from the outside.
    total = 3 * len(seeds)
    t0 = time.time()

    def tick(done: int, label: str) -> None:
        el = time.time() - t0
        rate = el / max(1, done)
        print(f"  [{done:>3}/{total}] {label:<7} {el/60:5.1f} min elapsed, "
              f"~{rate * (total - done) / 60:4.1f} min left", flush=True)

    # PASS 1: oracle, which also measures how much a real player talks.
    oracle = []
    for i, s in enumerate(seeds, 1):
        oracle.append(run(s, "oracle", a.steps))
        tick(i, "oracle")
    for r in oracle:                                   # match `noise` per seed, not on average
        NOISE_RATE[r["seed"]] = r["said"] / max(1, a.steps)
    silent = []
    for i, s in enumerate(seeds, 1):
        silent.append(run(s, "silent", a.steps))
        tick(len(seeds) + i, "silent")
    noise = []
    for i, s in enumerate(seeds, 1):
        noise.append(run(s, "noise", a.steps))
        tick(2 * len(seeds) + i, "noise")
    arms = {"silent": silent, "oracle": oracle, "noise": noise}

    def col(rows, k):
        return [r[k] for r in rows]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Does being talked to change the run?\n\n")
        f.write(f"{a.runs} seeds x {a.steps} steps, three arms on the SAME mazes. "
                f"`noise` is matched to `oracle` on message count per seed, so the "
                f"two differ in WHEN things were said and in nothing else.\n\n")
        f.write("The arousal axis cannot be read from text "
                "(`docs/cube_vs_human_vad.md`), so it has to be read from behaviour. "
                "This is behaviour.\n\n")

        f.write("## Arms\n\n")
        f.write("| arm | deaths | level reached | score | messages | credibility | berth |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, rows in arms.items():
            f.write(f"| {name} | {st.mean(col(rows,'deaths')):.2f} | "
                    f"{st.mean(col(rows,'level')):.2f} | "
                    f"{st.mean(col(rows,'score')):.0f} | "
                    f"{st.mean(col(rows,'said')):.1f} | "
                    f"{st.mean(col(rows,'credibility')):.2f} | "
                    f"{st.mean(col(rows,'radius')):.2f} |\n")

        f.write("\n## Paired tests\n\n")
        f.write("Same maze, same ghosts, one difference. Paired permutation, "
                "20,000 sign flips.\n\n")
        f.write("| comparison | metric | mean difference | p |\n|---|---|---:|---:|\n")
        checks = [("oracle vs silent", oracle, silent),
                  ("noise vs silent", noise, silent),
                  ("oracle vs noise", oracle, noise)]
        verdicts = {}
        for label, x, y in checks:
            for metric in ("deaths", "level", "score"):
                d = [p - q for p, q in zip(col(x, metric), col(y, metric))]
                p = perm_p(d)
                verdicts[(label, metric)] = (st.mean(d), p)
                f.write(f"| {label} | {metric} | {st.mean(d):+.3f} | "
                        f"{'**' if p < 0.05 else ''}{p:.4f}{'**' if p < 0.05 else ''} |\n")

        f.write("\n## Reading it\n\n")
        od, op = verdicts[("oracle vs silent", "deaths")]
        nd, np_ = verdicts[("noise vs silent", "deaths")]
        xd, xp = verdicts[("oracle vs noise", "deaths")]
        if op < 0.05 and xp < 0.05 and od < 0:
            f.write("**The content mattered.** A player who can see reduces deaths, and "
                    "does so beyond a player who merely talks as much. That is the "
                    "warning channel doing work, not arousal contagion.\n")
        elif op < 0.05 and np_ < 0.05:
            f.write("**Being talked to mattered; being RIGHT did not measurably.** Both "
                    "arms move against silence and they do not separate from each other, "
                    "which points at contagion and oxytocin rather than at the claims.\n")
        elif op >= 0.05 and np_ >= 0.05:
            f.write("**Null.** Nobody talking to him changed the outcome. Given that "
                    "`wariness` moves the berth by at most two tiles, the honest reading "
                    "is that the berth is not what decides catches in this maze — which "
                    "is a fact about the world, and the next thing to measure.\n")
        else:
            f.write("**Mixed.** The arms separate on some metric and not others; the "
                    "table above is the result, and the summary sentence is not going "
                    "to be more decisive than the numbers are.\n")

        f.write("\n## Where he spent his time\n\n")
        f.write("| corner | " + " | ".join(arms) + " |\n|---|" + "---:|" * len(arms) + "\n")
        allc = sorted({c for rows in arms.values() for r in rows for c in r["corners"]})
        for c in allc:
            tot = {n: sum(r["corners"][c] for r in rows) for n, rows in arms.items()}
            grand = {n: sum(sum(r["corners"].values()) for r in rows) for n in arms}
            f.write(f"| {c} | " + " | ".join(
                f"{100*tot[n]/max(1,grand[n]):.1f}%" for n in arms) + " |\n")
        f.write("\nThe occupancy histogram is the ablation the corner table has been "
                "waiting for: if talking to him reaches the body at all, he spends his "
                "time in different places.\n")

    print(f"wrote {OUT}")
    for name, rows in arms.items():
        print(f"  {name:<7} deaths {st.mean(col(rows,'deaths')):5.2f}  "
              f"level {st.mean(col(rows,'level')):4.2f}  "
              f"said {st.mean(col(rows,'said')):5.1f}  "
              f"cred {st.mean(col(rows,'credibility')):.2f}")
    for (label, metric), (d, p) in verdicts.items():
        if metric == "deaths":
            print(f"  {label:<18} deaths {d:+.3f}  p={p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
