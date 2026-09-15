"""exp_r35 -- what does cubby-man KNOW that he was never told? WO-2.10.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE QUESTION
------------
Nick: drop the agent in with no clue about the environment, let it study the
place and form its own rules -- *"right now some parts are still hardcoded"*,
and *"we need to remove all the hardcoded rules"*.

Before rewriting a 2,053-line file, it is worth knowing exactly which rules are
hardcoded, because "some parts" is not a work item and a list of line numbers is.
There are two distinct leaks and they need different fixes:

  ORACLE READS   the agent reaches into the environment for ground truth it was
                 never shown -- `env.ghosts`, `env.walls`, `env.remaining`. A
                 real body has none of that; it has percepts. Every one of these
                 is a sensor that does not exist yet, standing in for itself.

  HAND POLICY    the agent's choice is decided by a rule somebody wrote --
                 flee inside a fear radius, eat a frightened ghost, drop a trap
                 when two are near. None of it is learned, none of it is
                 refusable, and none of it would transfer to a world whose rules
                 differ.

This audit finds both, mechanically, so the removal list is evidence rather than
an impression. It changes nothing -- it only counts.

WHAT COUNTS AS A LEAK
---------------------
`env.coords()` and `env.cell()` are naming helpers, not knowledge: they translate
between a place-name and a coordinate the agent already holds. `env.step()` and
the move interface are the body's actuators. Everything else on the environment
is ground truth the world would not volunteer, and reading it is the leak.

    python validation/exp_r35_cubbyman_audit.py
"""
from __future__ import annotations

import argparse, ast, collections, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Environment attributes that are GROUND TRUTH -- the world knows them, a body
# in it does not, and a sensor would have to earn them.
ORACLE = {
    "walls": "where every wall is, including ones never touched",
    "hazards": "where every hazard is, unvisited ones included",
    "ghosts": "every ghost's exact position, through walls and at any range",
    "ghost_spawn": "where ghosts will appear before they do",
    "ghost_speed": "the ghosts' movement probability",
    "n_ghosts": "how many ghosts exist",
    "pellets": "every pellet's position at level start",
    "remaining": "which pellets are still uneaten, anywhere",
    "power": "where the power pellets are",
    "_power_all": "where the power pellets were",
    "mines": "every trap on the floor",
    "reach": "which cells are reachable at all -- the solved connectivity",
    "frightened": "the exact remaining frightened timer",
    "budget": "the level's time limit",
    "lives": "lives remaining",
    "energy": "the energy meter",
}
# Interface, not knowledge: naming helpers and actuators.
INTERFACE = {"coords", "cell", "step", "ghost_turn", "begin_run", "place_mine",
             "level", "w", "h", "d", "start", "score", "total_score", "steps",
             "attempt", "game_over", "mines_left"}

# Function names whose BODY is a hand-written tactic rather than a learned one.
# Listed by name so the audit reports them even when they read nothing oracular.
POLICY_HINTS = ("_pick", "_ghost_safe", "_mine_wise", "danger_cells",
                "danger_radius", "_flee", "_hunt", "_should")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    src_path = pathlib.Path(a.src) if a.src else (ROOT / "standin" / "pacman.py")
    src = src_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    src_lines = src.splitlines()

    # ---- which classes are the WORLD and which are the agent? -------------
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    world = set()
    for name, node in classes.items():
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        # a class that DEFINES the oracle attributes is the world itself
        assigns = {t.attr for st in ast.walk(node) if isinstance(st, ast.Assign)
                   for t in st.targets if isinstance(t, ast.Attribute)}
        if assigns & set(ORACLE) or bases & world:
            world.add(name)
    # transitive: subclasses of a world class are world too
    for _ in range(3):
        for name, node in classes.items():
            if {b.id for b in node.bases if isinstance(b, ast.Name)} & world:
                world.add(name)
    agent = [n for n in classes if n not in world]
    log(f"source: {src_path.name}  ({len(src_lines)} lines)")
    log(f"world classes : {', '.join(sorted(world)) or '(none found)'}")
    log(f"agent classes : {', '.join(sorted(agent))}\n")

    # ---- the oracle reads --------------------------------------------------
    leaks: list[dict] = []
    for cname in agent:
        node = classes[cname]
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Attribute):
                continue
            attr = sub.attr
            if attr not in ORACLE:
                continue
            # is the base an environment handle? `env.X` / `self.env.X`
            base = sub.value
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name not in ("env", "e", "verse", "world"):
                continue
            leaks.append({"class": cname, "line": sub.lineno, "attr": attr,
                          "why": ORACLE[attr],
                          "code": src_lines[sub.lineno - 1].strip()[:96]})

    by_attr = collections.Counter(l["attr"] for l in leaks)
    log("=" * 78)
    log("ORACLE READS -- ground truth the agent takes instead of perceiving")
    log("=" * 78)
    log(f"{'attribute':<16}{'reads':>6}   what it hands him for free")
    log("-" * 78)
    for attr, n in by_attr.most_common():
        log(f"{attr:<16}{n:>6}   {ORACLE[attr]}")
    log(f"\n{len(leaks)} oracle reads across {len(set(l['class'] for l in leaks))} "
        f"agent class(es)")

    log("\nthe worst offenders, by line:")
    for l in sorted(leaks, key=lambda x: x["line"])[:18]:
        log(f"  {l['line']:>5}  {l['class']}.{l['attr']:<14} {l['code']}")
    if len(leaks) > 18:
        log(f"  ... {len(leaks) - 18} more")

    # ---- the hand-written policies ----------------------------------------
    log("")
    log("=" * 78)
    log("HAND-WRITTEN POLICY -- decisions made by a rule nobody learned")
    log("=" * 78)
    pols = []
    for cname in agent:
        for sub in ast.walk(classes[cname]):
            if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(h in sub.name for h in POLICY_HINTS):
                continue
            doc = (ast.get_docstring(sub) or "").split("\n")[0][:90]
            n_lines = (getattr(sub, "end_lineno", sub.lineno) - sub.lineno)
            oracle_here = sorted({s.attr for s in ast.walk(sub)
                                  if isinstance(s, ast.Attribute) and s.attr in ORACLE})
            pols.append({"class": cname, "name": sub.name, "line": sub.lineno,
                         "lines": n_lines, "doc": doc, "reads": oracle_here})
    for p in sorted(pols, key=lambda x: -x["lines"]):
        log(f"  {p['line']:>5}  {p['class']}.{p['name']}  ({p['lines']} lines)")
        if p["doc"]:
            log(f"         \"{p['doc']}\"")
        if p["reads"]:
            log(f"         reads: {', '.join(p['reads'])}")

    # ---- the ordering -----------------------------------------------------
    log("")
    log("=" * 78)
    log("WHAT TO REMOVE FIRST")
    log("=" * 78)
    log("  1. `walls` / `hazards` -- the biggest lie. He is OFFERED only legal")
    log("     moves, so he can never walk into a wall and learn one. Replace")
    log("     the offer with an attempt that can FAIL: the move is tried, the")
    log("     world refuses it, and the refusal is the percept. That is the")
    log("     'know it after colliding with it' Nick asked for, and it is the")
    log("     one change that makes the others measurable.")
    log("  2. `ghosts` -- perfect position knowledge through walls at any range.")
    log("     Needs a sensor with a range and a line of sight; until one exists")
    log("     the threat layer is simulated omniscience. Which is why the first")
    log("     levels should run WITH NO GHOSTS AT ALL (Nick): learning the")
    log("     environment and surviving an omniscient threat model at once")
    log("     produces neither.")
    log("  3. `remaining` / `power` / `reach` -- the solved map. He should know")
    log("     only the cells he has stood in and what he saw from them.")
    log("  4. the hand-written policies -- once 1-3 are percepts rather than")
    log("     reads, a policy written against ground truth cannot run anyway,")
    log("     so they fall out rather than needing removal.")
    log("")
    log("  Order matters: fixing the policy first would leave it reading an")
    log("  oracle; fixing the oracle first makes the policy fail loudly.")

    stem = f"exp_r35_cubbyman_audit{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "source": src_path.name, "n_lines": len(src_lines),
        "world_classes": sorted(world), "agent_classes": sorted(agent),
        "oracle_reads": leaks, "by_attribute": dict(by_attr),
        "hand_policies": pols,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
