"""forge_probe — can the stand-in trunk write the game's programs? MEASURED.

Poses the three live task kinds (flee decision, safer-exit compare, map
orientation chain) on situations drawn from a real GhostVerse, through the
v3 emitter, and reports acceptance per kind — the VM ran the program and
the result matched the expectation cubby-man can check himself.

  python standin/scripts/forge_probe.py --gguf standin/models/emitter_v3.Q4_K_M.gguf --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"),
          os.path.join(ROOT, "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from forge import ToolForge, flee_task, orientation_task, safer_exit_task  # noqa: E402
from pacman import GhostVerse, ProgramLibrary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n", type=int, default=12, help="tasks per kind")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    args = ap.parse_args()
    from standin.emitter import LlamaCppEmitter
    emitter = LlamaCppEmitter(args.gguf, n_ctx=2048, n_gpu_layers=args.n_gpu_layers)
    lib = ProgramLibrary()
    events = []
    forge = ToolForge(emitter, lib, trace=lambda kind, **d: events.append({"kind": kind, **d}))
    rng = random.Random(0)
    env = GhostVerse()
    tasks = []
    for _ in range(args.n):
        near, radius = rng.randint(0, 4), rng.randint(1, 3)
        tasks.append(flee_task(near, radius, {"near": near, "radius": radius}))
        d1, d2 = rng.sample(range(1, 9), 2)
        tasks.append(safer_exit_task(d1, d2, {"d1": d1, "d2": d2}))
        cell = env.cell(*rng.choice(sorted(env.reach)))
        facts = [f for f in env.observe(cell) if " neighbor of " in f and "hazard" not in f]
        if facts:
            f = rng.choice(facts)
            obj, rest = f.split(" is the ", 1)
            direction = rest.split(" neighbor of ")[0]
            tasks.append(orientation_task(direction, cell, facts, obj, {"cell": cell}))
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        r = forge.forge(t, step=i)
        print(f"  {i}/{len(tasks)} {t.kind:9s} ok={r['ok']} got={r['got']!r} expected={t.expected!r}"
              + (f" err={r['error'][:60]}" if r["error"] else ""), flush=True)
    acc = forge.acceptance()
    print(f"\n[stand-in] forge acceptance ({len(tasks)} tasks, {time.time() - t0:.0f}s): "
          + ", ".join(f"{k} {v:.2f} ({a}/{n})" for k, (a, n) in forge.stats.items() for v in [acc[k]]))
    out = os.path.join(ROOT, "standin", "data", "out", "forge_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"acceptance": acc, "stats": forge.stats, "events": events}, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
