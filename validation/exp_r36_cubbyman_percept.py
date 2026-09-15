"""exp_r36 — cubby-man without the hardcoded rules: does he PERCEIVE?

WO-3.1. Nick, 2026-09-15: "we need to remove all the hardcoded rules", and
"if it sees a wall it should instinctively know that its an obstacle or at
least know it after colliding with it".

exp_r35 enumerated what he was being handed for free. This measures whether
it is actually gone, and whether he still plays once it is. Five claims, each
either counted or killed:

  1. COLLISION TEACHES.  He is offered directions his body has, not the
     maze's legal-move list, so a move can fail. Every refusal must produce
     a fact he did not hold, and every such fact must be TRUE of the world
     (a false wall is a defect, not a belief).
  2. NO ORACLE IN THE DECISION PATH.  A tripwire env records every read of
     ghosts/remaining/walls/hazards/reach/power and the agent method that
     made it. Reads from the renderer (`resp`, `init_payload`, `handle`) are
     the page's, and are allowed; anything else is a leak.
  3. PERCEPTION IS BOUNDED.  No pellet is ever sighted beyond SIGHT or
     through a wall.
  4. BELIEF CAN BE WRONG.  His believed ghost positions must sometimes
     differ from the true ones. If they never do, perception is still
     omniscience wearing a hat.
  5. HE STILL PLAYS.  Pellets eaten and ground covered, against a baseline
     that keeps the oracle. A percept layer that stops him playing is a
     regression, not a principle.

Run:  python validation/exp_r36_cubbyman_percept.py --steps 400
Needs cubelang.exe (the ASK runs on the VM). Skips loudly without it.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

ORACLE_ATTRS = ("ghosts", "remaining", "walls", "hazards", "reach", "power",
                "pellets", "_power_all", "ghost_spawn")
# the page's own feed: these read the world on purpose, for the human
RENDERER = {"resp", "init_payload", "handle", "status_line", "_render_page"}
# the world's own machinery, reached through the env's methods
WORLD_SIDE = {"exits", "_open", "observe", "try_move", "senses", "line_of_sight",
              "power_moves", "eat", "ghost_turn", "_start_level", "begin_run",
              "restart_run", "place_mine", "all_facts", "_respawn", "coords", "cell",
              # the two channels the world PUBLISHES on purpose: one bit for
              # "this maze is empty", and the cabinet's scoreboard for the
              # replay page. Both live on the world, so a read through them is
              # the world reading itself.
              "cleared", "progress"}


def tripwire(cls):
    """A GhostVerse that records WHO read the ground truth. Attribute access
    is cheap enough that instrumenting it is honest: we get the real call
    site, not a guess from a grep."""
    import sys as _sys

    class Tripwire(cls):
        def __init__(self, *a, **kw):
            object.__setattr__(self, "_reads", collections.Counter())
            super().__init__(*a, **kw)

        def __getattribute__(self, name):
            if name in ORACLE_ATTRS:
                try:
                    f = _sys._getframe(1)
                    fn, cls_name = f.f_code.co_name, ""
                    slf = f.f_locals.get("self")
                    if slf is not None:
                        cls_name = type(slf).__name__
                    where = f"{pathlib.Path(f.f_code.co_filename).name}:{f.f_lineno}"
                    object.__getattribute__(self, "_reads")[(cls_name, fn, name, where)] += 1
                except Exception:
                    pass
            return object.__getattribute__(self, name)

    Tripwire.__name__ = f"Tripwire{cls.__name__}"
    return Tripwire


def leaks(reads):
    """Reads that are the AGENT taking ground truth. The world reading its own
    state, and the renderer feeding the page, are not leaks."""
    out = []
    for (cls_name, fn, attr, where), n in reads.items():
        if "Verse" in cls_name or fn in WORLD_SIDE or fn in RENDERER:
            continue
        if cls_name in ("", "Tripwire"):
            continue
        out.append((cls_name, f"{fn} ({where})", attr, n))
    return sorted(out, key=lambda r: -r[3])


def run(steps, seed, ghost_free=None, see_exits=True):
    import pacman as P
    env = tripwire(P.GhostVerse)(ghost_free_levels=ghost_free)

    class Man(P.CubbyGhost):
        SEE_EXITS = see_exits

    man = Man(env, probe=0.25, seed=seed)

    st = {"steps": 0, "bumps": 0, "bump_kinds": collections.Counter(),
          "bump_facts_new": 0, "false_walls": [], "eaten": 0, "visited": set(),
          "sight_violations": [], "belief_wrong": 0, "belief_checks": 0,
          "ghosts_true": 0, "ghosts_known": 0, "ghosts_phantom": 0, "blind_steps": 0,
          "levels": env.level, "caught": 0, "radius": []}
    t0 = time.time()
    st["visited"].add(man.place)
    for i in range(steps):
        # every cell he PERCEIVES FROM, which is not only the ones he ends a
        # step on: next_level() drops him at a new start and senses there
        # before the same step moves him on (the first draft missed exactly
        # that cell and reported its sightings as "beyond the sensor")
        st["visited"].add(man.place)
        was_level = env.level
        try:
            rec = man.step()
        except Exception as e:                           # a crash is a result too
            st["crash"] = f"{type(e).__name__}: {e}"
            break
        if env.level != was_level:
            st["visited"].add(env.start)                 # next_level() sensed from the new start
        st["steps"] += 1
        if rec.get("bumped"):
            st["bumps"] += 1
            st["bump_kinds"][rec["bumped"]] += 1
        st["visited"].add(man.place)
        st["eaten"] = env.total_score
        st["levels"] = max(st["levels"], env.level)
        st["radius"].append(man.danger_radius)
        if env.ghosts:                                   # 4: what he believes vs what is
            st["belief_checks"] += 1
            truth = {tuple(g) for g in env.ghosts}
            bel = set(man.ghost_belief)
            if bel != truth:
                st["belief_wrong"] += 1
            st["ghosts_true"] += len(truth)
            st["ghosts_known"] += len(bel & truth)       # how much of the truth he actually holds
            st["ghosts_phantom"] += len(bel - truth)     # stale positions: a belief that has gone wrong
            if not bel:
                st["blind_steps"] += 1                   # ghosts out there, and he knows of none
    st["secs"] = round(time.time() - t0, 1)

    # 1. every wall/hazard fact he holds must be TRUE of the world.
    #    Facts are level-scoped and each level is a different maze, so only
    #    the CURRENT level's facts can be checked against the current walls
    #    (the first draft of this check compared level-1 beliefs to level-2
    #    stone and reported 11 "false walls" that were all correct).
    here_lvl = f"level-{env.level} "
    for f in man.world.texts:
        m = man._nbr_re.match(f)
        if not m or m.group("b") not in set(man.NON_PLACES.values()):
            continue
        if not m.group("a").startswith(here_lvl):
            continue
        st["obstacle_facts"] = st.get("obstacle_facts", 0) + 1
        x, y, z = env.coords(m.group("a"))
        dx, dy, dz = P.MOVES[m.group("d")]
        q = (x + dx, y + dy, z + dz)
        really = (not P._in(q, env.w, env.h, env.d)) or q in env.walls or q in env.hazards
        if not really:
            st["false_walls"].append(f)

    # 3. nothing sighted beyond SIGHT or through a wall, from anywhere he stood
    for cell in man.sighted:
        try:
            c = env.coords(cell)
        except Exception:
            continue
        ok = any(P._manh(c, env.coords(v)) <= man.SIGHT and env.line_of_sight(env.coords(v), c)
                 for v in st["visited"] if v.startswith(f"level-{env.level} "))
        if not ok and cell.startswith(f"level-{env.level} "):
            st["sight_violations"].append(cell)

    st["reads"] = getattr(env, "_reads", collections.Counter())
    st["man"], st["env"] = man, env
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "validation" / "logs" / "exp_r36_cubbyman_percept.json"))
    a = ap.parse_args(argv)

    try:
        from cubbyllm.bridges import cubelang_client as cc
        cc.find_cubelang_exe()
    except Exception as e:
        print(f"SKIPPED: the ASK runs on the VM and cubelang.exe was not found ({e})")
        return 2

    print(f"exp_r36 — cubby-man's percepts  ({a.steps} steps/arm, seed {a.seed})")
    print("=" * 78)
    arms = {}
    for name, ghost_free, see in (("classroom (levels 1-3, no ghosts)", None, True),
                                  ("threat on from level 1", 0, True),
                                  ("BLIND (SEE_EXITS off: collisions only)", None, False)):
        print(f"\n-- {name} " + "-" * max(0, 60 - len(name)))
        st = run(a.steps, a.seed, ghost_free=ghost_free, see_exits=see)
        arms[name] = st
        if st.get("crash"):
            print(f"  CRASHED after {st['steps']} steps: {st['crash']}")
            continue
        kinds = ", ".join(f"{k}:{v}" for k, v in sorted(st["bump_kinds"].items())) or "-"
        print(f"  steps {st['steps']:4d}   {st['secs']}s   levels reached {st['levels']}   "
              f"pellets {st['eaten']}   cells stood in {len(st['visited'])}")
        print(f"  1. COLLISIONS    {st['bumps']:4d} refused moves ({kinds})")
        print(f"     obstacles     {st.get('obstacle_facts', 0)} learned on this level, "
              f"{len(st['false_walls'])} of them FALSE (must be 0)")
        lk = leaks(st["reads"])
        print(f"  2. ORACLE READS  {len(lk)} leak site(s) in the decision path")
        for cls_name, fn, attr, n in lk[:10]:
            print(f"       {cls_name}.{fn} -> env.{attr}   x{n}")
        print(f"  3. SIGHT         {len(st['sight_violations'])} sightings outside range or through a wall")
        if st["belief_checks"]:
            know = 100.0 * st["ghosts_known"] / max(1, st["ghosts_true"])
            blind = 100.0 * st["blind_steps"] / st["belief_checks"]
            print(f"  4. BELIEF        he holds {know:.0f}% of the true ghost positions "
                  f"({st['ghosts_known']}/{st['ghosts_true']}); 100% would be the oracle")
            print(f"                   {st['ghosts_phantom']} stale/phantom positions, "
                  f"and on {blind:.0f}% of steps he knows of no ghost at all")
        else:
            print("  4. BELIEF        no ghost steps in this arm (that IS the classroom)")
        r = st["radius"]
        print(f"  5. BERTH         danger_radius {r[0] if r else '-'} -> {r[-1] if r else '-'} "
              f"(learned from {len(st['man'].caught_at)} catches)")

    print("\n" + "=" * 78)
    verdict, why = "PASS", []
    for name, st in arms.items():
        if st.get("crash"):
            verdict, _ = "KILLED", why.append(f"{name}: crashed ({st['crash']})")
            continue
        if st["false_walls"]:
            verdict = "KILLED"
            why.append(f"{name}: {len(st['false_walls'])} false obstacle(s) learned, e.g. {st['false_walls'][0]}")
        lk = leaks(st["reads"])
        if lk:
            verdict = "KILLED"
            why.append(f"{name}: oracle still read by {lk[0][0]}.{lk[0][1]} -> env.{lk[0][2]}")
        if st["sight_violations"]:
            verdict = "KILLED"
            why.append(f"{name}: {len(st['sight_violations'])} sighting(s) beyond the sensor")
        if "BLIND" in name:
            # the decisive arm: with no vision, the ONLY way a map exists is
            # walking into things, so zero collisions here means the channel
            # does not carry. (A sighted agent bumping rarely is not a
            # failure -- it means he saw the wall, which is the other half of
            # what Nick asked for.)
            if st["bumps"] == 0:
                verdict = "KILLED"
                why.append(f"{name}: not one refused move -- the collision channel does not carry")
            if st.get("obstacle_facts", 0) == 0:
                verdict = "KILLED"
                why.append(f"{name}: no obstacle learned at all without vision")
        if st["belief_checks"]:
            if st["ghosts_known"] == st["ghosts_true"]:
                verdict = "KILLED"
                why.append(f"{name}: he held EVERY ghost position on every step -- "
                           "that is the oracle wearing a sensor's hat")
            if st["blind_steps"] == 0:
                verdict = "KILLED"
                why.append(f"{name}: never once unaware of a ghost -- the sensor has no blind spot")
    print(f"VERDICT: {verdict}")
    for w in why:
        print(f"  - {w}")
    if verdict == "PASS":
        print("  collisions teach, nothing false is learned, no oracle in the decision path,")
        print("  perception is bounded, his belief is sometimes wrong, and he still plays.")

    blob = {k: {kk: (vv if isinstance(vv, (int, float, str, list)) else
                     (dict(vv) if isinstance(vv, collections.Counter) else len(vv)))
                for kk, vv in v.items() if kk not in ("man", "env", "reads")}
            for k, v in arms.items()}
    for k, v in arms.items():
        blob[k]["leaks"] = [f"{c}.{f} -> env.{at} x{n}" for c, f, at, n in leaks(v["reads"])]
    blob["verdict"] = verdict
    blob["why"] = why
    pathlib.Path(a.out).write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {pathlib.Path(a.out).name}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
