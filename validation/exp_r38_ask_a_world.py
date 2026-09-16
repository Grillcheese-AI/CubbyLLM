"""exp_r38 — he meets something his map cannot explain, and asks the world that knows.

WO-2.13's kill criterion, run headless and measured rather than asserted:

    "the first ask-a-world build must show a question he could not answer
     before, answered after asking, WITH the answer still there on the next run
     and no second ask. If he re-asks, nothing was learned and this is a
     retrieval cache with extra steps."

The arc the run has to produce, in order:

  1. something comes down on him and he has no account of it  -> hit_by_falling, knew=False
  2. he frames the question in his own words and sends it out  -> wonder / ask
  3. physics answers with laws, and they enter his map through -> hypothesis_settled
     the same gate as a percept                                  + the facts in his store
  4. the NEXT thing overhead means something                   -> predict, then dodge
  5. he never asks again, in this run or the next              -> asks == 1, ask_skipped

Run (from the repo root, no model needed — this is the loop, not the speech):

    python validation/exp_r38_ask_a_world.py --steps 260
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pacman import CubbyGhost, GhostVerse  # noqa: E402


def run(steps: int, seed: int, level: int, quiet: bool) -> dict:
    events: list[tuple[int, str, dict]] = []
    env = GhostVerse(level=level, ghost_free_levels=9, fallers_from_level=1)

    man = CubbyGhost(env, seed=seed, probe=0.0, memory=None, blank=True)
    man._trace = lambda kind, **d: events.append((env.steps, kind, d))

    for _ in range(steps):
        man.step()

    kinds = collections.Counter(k for _, k, _ in events)
    asks = [d for _, k, d in events if k == "ask"]
    hits = [d for _, k, d in events if k == "hit_by_falling"]
    preds = [d for _, k, d in events if k == "predict"]
    settled = [d for _, k, d in events if k == "hypothesis_settled"
               and d.get("state") == "confirmed" and d.get("by") == "ask"]
    laws = [f for f in man.world.texts if "fall" in f or "above me" in f]

    if not quiet:
        print(f"\n{'':-<72}")
        print(f"{steps} steps, level {level}, seed {seed} — blank slate, no ghosts, fallers from level 1")
        print(f"{'':-<72}")
        for at, kind, d in events:
            if kind in ("hit_by_falling", "saw_it_land", "wonder", "ask", "hypothesis_settled",
                        "predict", "dodge", "ask_skipped"):
                bits = {k: v for k, v in d.items() if k in
                        ("knew", "times", "at", "claim", "question", "world", "got", "state",
                         "in_steps", "move", "because", "why")}
                print(f"  step {at:>4}  {kind:<18} {bits}")
        print(f"\n  laws he now holds ({len(laws)}):")
        for f in laws:
            print(f"    · {f}")

    return {
        "events": len(events), "kinds": dict(kinds),
        "asks": len(asks), "answered": sum(1 for a in asks if a.get("got")),
        "hits": len(hits), "hits_before_knowing": sum(1 for h in hits if not h.get("knew")),
        "hits_after_knowing": sum(1 for h in hits if h.get("knew")),
        "predictions": len(preds), "dodges": kinds.get("dodge", 0),
        "ask_skipped": kinds.get("ask_skipped", 0),
        "laws": laws, "settled": len(settled),
        "held": list(man.world.texts),
    }


def second_run(held: list[str], steps: int, seed: int, level: int) -> dict:
    """THE KILL CRITERION. A new agent, blank, given only the facts the first
    one ended with — the way a run that persisted its map would start. If the
    question goes out again, nothing was learned."""
    env = GhostVerse(level=level, ghost_free_levels=9, fallers_from_level=1)
    events: list[tuple[int, str, dict]] = []
    man = CubbyGhost(env, seed=seed + 1, probe=0.0, memory=None, blank=True)
    man._trace = lambda kind, **d: events.append((env.steps, kind, d))
    for f in held:
        man.world.add(f)
    knew = man.knows_falling()                  # BEFORE he takes a step, or it measures this run
    for _ in range(steps):
        man.step()
    kinds = collections.Counter(k for _, k, _ in events)
    return {"asks": kinds.get("ask", 0), "ask_skipped": kinds.get("ask_skipped", 0),
            "hits": kinds.get("hit_by_falling", 0), "predictions": kinds.get("predict", 0),
            "dodges": kinds.get("dodge", 0), "knew_from_the_start": knew}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sweep", type=int, default=0, help="run N seeds and print one line each")
    a = ap.parse_args()

    if a.sweep:
        print(f"\n{'seed':>4}  {'asked':>5} {'laws':>4} {'hit before':>10} {'hit after':>9} "
              f"{'predicts':>8} {'dodges':>6} | {'re-asked':>8} {'knew':>5}")
        bad = 0
        for s in range(a.sweep):
            f = run(a.steps, a.seed + s, a.level, quiet=True)
            l8r = second_run(f["held"], a.steps, a.seed + s, a.level)
            ok = (f["asks"] == 1 and len(f["laws"]) >= 4 and l8r["asks"] == 0
                  and l8r["knew_from_the_start"])
            bad += not ok
            print(f"{a.seed + s:>4}  {f['asks']:>5} {len(f['laws']):>4} "
                  f"{f['hits_before_knowing']:>10} {f['hits_after_knowing']:>9} "
                  f"{f['predictions']:>8} {f['dodges']:>6} | {l8r['asks']:>8} "
                  f"{str(l8r['knew_from_the_start']):>5}  {'' if ok else '<-- FAIL'}")
        print(f"\n  {a.sweep - bad}/{a.sweep} seeds: asked once, kept it, never asked again")
        return 1 if bad else 0

    first = run(a.steps, a.seed, a.level, a.quiet)
    later = second_run(first["held"], a.steps, a.seed, a.level)

    print(f"\n{'':=<72}")
    print("FIRST RUN — he starts knowing nothing about falling")
    print(f"{'':=<72}")
    print(f"  things came down on him        {first['hits']}"
          f"   ({first['hits_before_knowing']} before he knew, {first['hits_after_knowing']} after)")
    print(f"  questions sent out             {first['asks']}  ({first['answered']} answered)")
    print(f"  claims settled by an answer    {first['settled']}")
    print(f"  laws now in his map            {len(first['laws'])}")
    print(f"  predictions he made            {first['predictions']}")
    print(f"  times he stepped out of it     {first['dodges']}")

    print(f"\n{'':=<72}")
    print("SECOND RUN — a fresh agent holding the first one's map")
    print(f"{'':=<72}")
    print(f"  knew it before taking a step   {later['knew_from_the_start']}")
    print(f"  questions sent out             {later['asks']}")
    print(f"  questions he skipped asking    {later['ask_skipped']}")
    print(f"  predictions / dodges           {later['predictions']} / {later['dodges']}")
    print(f"  things came down on him        {later['hits']}")

    ok = (first["asks"] == 1 and first["answered"] == 1 and len(first["laws"]) >= 4
          and first["predictions"] >= 1 and later["asks"] == 0 and later["knew_from_the_start"])
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} — "
          + ("asked once, held the answer, predicted with it, never asked again"
             if ok else "see the counts above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
