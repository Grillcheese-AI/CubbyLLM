"""build_game_sft — v4: the emitter learns the GAME's programs (+ v3 replay).

Wired: STANDALONE (stand-in tooling). Output in standin/data/out/ (gitignored).

The forge probe measured the v3 stand-in on the game's live tasks: decision
0.75 / chain 0.50 / compare 0.00 — and 0 / 1 / 0 out of 8 when the tasks
were phrased in the game's own words. v4 closes the loop the notebook made
possible: the game GENERATES its training data. Every record here is a
(prompt, program) pair whose program is a template cubby-man already runs
(the decision/compare kernels in the corpus shape, his CotChain map queries,
his JoinCount exit counter, his Superpower compositions), phrased the way
the game phrases it, with gold from the world's ground truth — and every
program is re-verified through the real Rust VM before it is kept.

Families (task, subtype):
  kernel  game:decision   flee / go-for-it decisions on live numbers
                          (confidence 90 when the condition holds, 30 when not)
  kernel  game:compare    safer exit (max) and shorter path (min) on small numbers
  chain   game:where      1- and 2-hop neighbor queries over level-scoped cells
  kernel  game:count      exit counts computed by an add-per-fact program
  kernel  game:compose    "compose the moves … into a superpower named X"
  + the certified tools from his notebook (data/out/cubbyman_programs.json),
    when it exists — programs he generated in play that the VM certified.

The v3 set (emitter_sft.jsonl) is merged in unchanged as REPLAY: a v4 that
forgot arithmetic or identity would be a regression, not an upgrade (the
same reason H-A7's nightly gate replays). Game records carry `repeat` 2 in
train. Writes emitter_sft_v4.jsonl + emitter_sft_v4.manifest.json.

  python standin/data/build_game_sft.py            # ~3k game records, VM-verified
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE, os.path.join(ROOT, "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_emitter_sft import OUT_DIR, sha256_file, split_of, verify_all  # noqa: E402
from identity import EMITTER_SYSTEM  # noqa: E402

GAME_SEED = 20260902
N = {"decision": 500, "compare": 500, "where": 700, "count": 250, "compose": 250}
GAME_REPEAT = 2
V3_PATH = os.path.join(OUT_DIR, "emitter_sft.jsonl")
PROGRAMS_PATH = os.path.join(OUT_DIR, "cubbyman_programs.json")

MOVES = ["right", "left", "up", "down", "forward", "back"]
OPP = {"right": "left", "left": "right", "up": "down", "down": "up", "forward": "back", "back": "forward"}


# ── program templates (the shapes he already runs) ──────────────────────────
def decision_program(title: str, obs: int, gate: int, yes: str, no: str) -> str:
    return f"""# {title}
program Dec0 implements ISolver {{
    type Input = quantity;
    type Output = quantity;

    storage {{
        threshold: mutable u64 = 0;
        decisions_made: mutable u64 = 0;
        last_decision: mutable str = "";
    }}

    @system @once
    public function constructor() {{
        assign threshold = {gate};
        assign decisions_made = 0;
        assign last_decision = "none";
    }}

    @external
    public function parse(raw: str): Input {{ return raw; }}

    @external
    public function solve(input: Input): Output {{
        create observation : quantity;
        assign observation = {obs};
        store observation, "current_observation";
        create gate_value : quantity;
        assign gate_value = {gate};
        store gate_value, "gate_threshold";
        create confidence : quantity;
        assign confidence = 0;
        if (observation > gate_value) {{
            add confidence, 90;
            assign last_decision = "{yes}";
            store confidence, "{yes}_confidence";
        }} else {{
            add confidence, 30;
            assign last_decision = "{no}";
            store confidence, "{no}_confidence";
        }}
        query confidence;
        remember confidence;
        add decisions_made, 1;
        sum confidence;
        return confidence;
    }}

    @external
    public pure function verify(input: Input, output: Output): bool {{ return true; }}
}}
"""


def compare_program(title: str, lhs: int, rhs: int, want_max: bool) -> str:
    big, small = ("lhs", "rhs") if want_max else ("rhs", "lhs")
    return f"""# {title}
program Cmp0 implements ISolver {{
    type Input = quantity;
    type Output = quantity;

    storage {{
        comparisons_made: mutable u64 = 0;
        last_winner: mutable str = "";
    }}

    @system @once
    public function constructor() {{
        assign comparisons_made = 0;
        assign last_winner = "none";
    }}

    @external
    public function parse(raw: str): Input {{ return raw; }}

    @external
    public function solve(input: Input): Output {{
        create lhs : quantity;
        assign lhs = {lhs};
        store lhs, "candidate_left";
        create rhs : quantity;
        assign rhs = {rhs};
        store rhs, "candidate_right";
        create result : quantity;
        assign result = 0;
        if (lhs > rhs) {{
            add result, {big};
            assign last_winner = "{'left' if want_max else 'right'}";
        }} else {{
            add result, {small};
            assign last_winner = "{'right' if want_max else 'left'}";
        }}
        store result, "selected_{'max' if want_max else 'min'}";
        query result;
        remember result;
        add comparisons_made, 1;
        sum result;
        return result;
    }}

    @external
    public pure function verify(input: Input, output: Output): bool {{ return true; }}
}}
"""


def chain_program(hops: list[tuple[str, str]]) -> str:
    """hops = [(direction, answer_cell), ...] in walk order; the v3 CotChain
    shape: solve() recovers hop 1, hop_k() recovers hop k, control() an
    absent role."""
    binds = "".join(f'        bind frame, H{i + 1}_{d.upper()}_NEIGHBOR, "{cell}";\n'
                    for i, (d, cell) in enumerate(hops))
    fns = ["    public function solve(mention: str): str {\n        create frame: number;\n"
           f"{binds}        return recover(frame, H1_{hops[0][0].upper()}_NEIGHBOR);\n    }}\n"]
    for k in range(2, len(hops) + 1):
        fns.append(f"\n    public function hop_{k}(): str {{\n        create frame: number;\n"
                   f"{binds}        return recover(frame, H{k}_{hops[k - 1][0].upper()}_NEIGHBOR);\n    }}\n")
    fns.append("\n    public function control(): str {\n        create frame: number;\n"
               f"{binds}        return recover(frame, H9_CONTROL);\n    }}\n")
    return "use vsa;\n\nprogram CotChain implements ISolve {\n" + "".join(fns) + "}\n"


def count_program(n: int) -> str:
    adds = "        add n, 1;\n" * n
    return ("program JoinCount implements ISolve {\n    public function solve(mention: str): str {\n"
            f"        create n : symbol;\n        assign n = 0;\n{adds}        return n;\n    }}\n}}\n")


def compose_program(name: str, steps: list[str]) -> str:
    binds = "".join(f'        bind frame, H{i + 1}_STEP, "{s}";\n' for i, s in enumerate(steps))
    return ("use vsa;\n\nprogram Superpower implements ISolve {\n"
            "    public function solve(mention: str): str {\n        create frame: number;\n"
            f"{binds}        bind frame, H{len(steps) + 1}_NAME, \"{name}\";\n"
            f"        return recover(frame, H{len(steps) + 1}_NAME);\n    }}\n}}\n")


# ── the game's phrasings ────────────────────────────────────────────────────
DECISION_TEMPLATES = [
    # (phrase(obs, gate), yes, no) — the story is the game's; the semantics are obs > gate
    (lambda a, n: f"The alarm level is {a}; flee if it exceeds the ghost distance of {n}.", "flee", "stay"),
    (lambda a, n: f"A ghost is {n} cells away; sound the alarm if the alarm level {a} exceeds that distance.",
     "flee", "stay"),
    (lambda a, n: f"My fear radius plus one is {a} and the nearest ghost is {n} cells away; "
                  f"run if the first number exceeds the second.", "run", "hold"),
    (lambda t, d: f"There are {t} steps left in the level and the pellet is {d} steps away; "
                  f"go for it if the time exceeds the distance.", "go", "skip"),
    (lambda e, c: f"My energy is {e} and a jump costs {c}; jump if the energy exceeds the cost.", "jump", "rest"),
    (lambda a, n: f"A sensor reads {a} units; flag the sample if the reading is above the safety threshold {n}.",
     "approve", "reject"),
]
COMPARE_TEMPLATES = [
    (lambda a, b: f"Two exits lead {a} and {b} cells away from the nearest ghost; report the safer, higher distance.", True),
    (lambda a, b: f"Two moves land {a} and {b} cells from the ghost; report the farther one.", True),
    (lambda a, b: f"Two paths to the pellet take {a} and {b} steps; report the shorter one.", False),
    (lambda a, b: f"Two pellets are {a} and {b} steps away; report the nearer distance.", False),
    (lambda a, b: f"Two trials measured {a} and {b}; report the higher measurement.", True),
]
COMPOSE_TEMPLATES = [
    lambda name, steps: f"Compose the moves {', '.join(steps)} into a superpower named {name}.",
    lambda name, steps: f"Write a superpower called {name} from the steps {' then '.join(steps)}.",
    lambda name, steps: f"Bind the steps {', '.join(steps)} and recover the new move {name}.",
]
FLAVOR = ["DASH", "BLINK", "COMBO", "KNIGHT", "WARP", "SPRINT", "ZIGZAG", "STAIRS", "HOOK", "ELBOW", "JUMP"]


def gen_decisions(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        phrase, yes, no = rng.choice(DECISION_TEMPLATES)
        obs, gate = rng.randint(0, 12), rng.randint(0, 12)
        if phrase is DECISION_TEMPLATES[4][0]:           # energy vs cost: game-range numbers
            obs, gate = rng.randint(0, 100), rng.choice([10, 20, 25, 30])
        prompt = phrase(obs, gate)
        gold = 90 if obs > gate else 30
        out.append({"id": f"game:decision:{i}", "task": "kernel", "subtype": "game:decision", "source": "game",
                    "prompt": prompt, "program": decision_program(prompt, obs, gate, yes, no), "gold": gold})
    return out


def gen_compares(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        phrase, want_max = rng.choice(COMPARE_TEMPLATES)
        a, b = rng.sample(range(1, 13), 2)
        prompt = phrase(a, b)
        out.append({"id": f"game:compare:{i}", "task": "kernel", "subtype": "game:compare", "source": "game",
                    "prompt": prompt, "program": compare_program(prompt, a, b, want_max),
                    "gold": max(a, b) if want_max else min(a, b)})
    return out


def gen_where(rng: random.Random, n: int) -> list[dict]:
    from pacman import GhostVerse
    envs = [GhostVerse(level=L) for L in (1, 2, 3)]
    out, i = [], 0
    while len(out) < n:
        env = rng.choice(envs)
        cell = env.cell(*rng.choice(sorted(env.reach)))
        exits = env.exits(cell)
        if not exits:
            continue
        d1 = rng.choice(sorted(exits))
        mid = exits[d1]
        facts = [f"{nbr} is the {d} neighbor of {cell}" for d, nbr in sorted(exits.items())]
        hops = [(d1, mid)]
        question = f"What is the {d1} neighbor of {cell}?"
        if rng.random() < 0.35:                          # 2-hop through the intermediate cell
            exits2 = env.exits(mid)
            if exits2:
                d2 = rng.choice(sorted(exits2))
                facts += [f"{nbr} is the {d} neighbor of {mid}" for d, nbr in sorted(exits2.items())]
                hops.append((d2, exits2[d2]))
                question = f"What is the {d2} neighbor of the {d1} neighbor of {cell}?"
        rng.shuffle(facts)
        prompt = question + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts[:6])
        if not all(any(f"{cell_} is the {d} neighbor" in f for f in facts[:6]) for d, cell_ in hops):
            continue                                     # the walked facts must be in the block
        out.append({"id": f"game:where:{i}", "task": "chain", "subtype": f"game:where:n_hop={len(hops)}",
                    "source": "game", "prompt": prompt, "program": chain_program(hops), "gold": hops[-1][1]})
        i += 1
    return out


def gen_counts(rng: random.Random, n: int) -> list[dict]:
    from pacman import GhostVerse
    envs = [GhostVerse(level=L) for L in (1, 2, 3)]
    out = []
    for i in range(n):
        env = rng.choice(envs)
        cell = env.cell(*rng.choice(sorted(env.reach)))
        exits = env.exits(cell)
        facts = [f"{nbr} is the {d} neighbor of {cell}" for d, nbr in sorted(exits.items())]
        prompt = f"How many exits does {cell} have?\nFacts:\n" + "\n".join(f"- {f}" for f in facts)
        out.append({"id": f"game:count:{i}", "task": "kernel", "subtype": "game:count", "source": "game",
                    "prompt": prompt, "program": count_program(len(exits)), "gold": len(exits)})
    return out


def gen_composes(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        k = rng.choice([2, 2, 3, 3, 3, 4, 5])
        steps = [rng.choice(MOVES)]
        while len(steps) < k:
            d = rng.choice(MOVES)
            if d != OPP[steps[-1]]:
                steps.append(d)
        name = rng.choice(FLAVOR) if rng.random() < 0.6 else "COMBO-" + "".join(rng.choice("ABC") for _ in range(k))
        prompt = rng.choice(COMPOSE_TEMPLATES)(name, steps)
        out.append({"id": f"game:compose:{i}", "task": "kernel", "subtype": "game:compose", "source": "game",
                    "prompt": prompt, "program": compose_program(name, steps), "gold": name})
    return out


def notebook_tools() -> list[dict]:
    """Programs he generated in play that the VM certified (the notebook)."""
    try:
        entries = json.load(open(PROGRAMS_PATH, encoding="utf-8")).get("entries", {})
    except (OSError, ValueError):
        return []
    out = []
    for name, e in entries.items():
        r = e.get("reasoning", {})
        if e.get("kind") != "tool" or not r.get("ok") or not r.get("rationale", "").startswith("prompt: "):
            continue
        task = "chain" if name.startswith("WHERE") else "kernel"
        out.append({"id": f"game:notebook:{name}", "task": task, "subtype": f"game:notebook:{name.split('#')[0]}",
                    "source": "game_notebook", "prompt": r["rationale"][len("prompt: "):],
                    "program": e["program"], "gold": r.get("got")})
    return out


def build(seed: int = GAME_SEED, n: dict | None = None) -> list[dict]:
    rng = random.Random(seed)
    n = n or N
    recs = (gen_decisions(rng, n["decision"]) + gen_compares(rng, n["compare"]) + gen_where(rng, n["where"])
            + gen_counts(rng, n["count"]) + gen_composes(rng, n["compose"]) + notebook_tools())
    seen = set()
    out = []
    for r in recs:                                       # dedupe on the prompt
        if r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
        r["system"] = EMITTER_SYSTEM
        r["state"] = None
        r["split"] = split_of(r["prompt"])
        r["repeat"] = GAME_REPEAT if r["split"] == "train" else 1
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="records per family (debug)")
    ap.add_argument("--no-replay", action="store_true", help="game records only (debug; v4 needs the replay)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    n = {k: min(v, args.limit) for k, v in N.items()} if args.limit else N
    game = build(n=n)
    print(f"game records: {len(game)} | by subtype: {dict(Counter(r['subtype'].split(':n_hop')[0] for r in game))}")
    if args.no_verify:
        for r in game:
            r["vm_ok"], r["vm_result"], r["vm_error"], r["gold_match"] = None, None, None, None
        vstats, vwall = {}, 0.0
    else:
        print("=== verifying every game program through the real cubelang VM ...", flush=True)
        vstats, vwall = verify_all(game)
        print(f"  done in {vwall:.0f}s: {dict(sorted(vstats.items()))}")
    kept = [r for r in game if r["vm_ok"] is None or (r["vm_ok"] and r["gold_match"] is not False)]
    dropped = Counter(r["subtype"] for r in game if r not in kept)
    replay = []
    if not args.no_replay:
        replay = [json.loads(l) for l in open(V3_PATH, encoding="utf-8")]
    records = replay + kept
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "emitter_sft_v4.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        git_rev = "unknown"
    inputs = {V3_PATH: sha256_file(V3_PATH) if os.path.exists(V3_PATH) else None,
              PROGRAMS_PATH: sha256_file(PROGRAMS_PATH) if os.path.exists(PROGRAMS_PATH) else None}
    manifest = {
        "built": datetime.now(timezone.utc).isoformat(), "git_rev": git_rev, "version": "v4",
        "config": {"GAME_SEED": GAME_SEED, "N": n, "GAME_REPEAT": GAME_REPEAT, "verified": not args.no_verify,
                   "replay": not args.no_replay,
                   "schema_note": "v4 = v3 (replay, unchanged) + the game's VM-verified (prompt, program) pairs; "
                                  "game records carry subtype game:* and repeat 2 in train"},
        "inputs_sha256": inputs,
        "n_game_generated": len(game), "n_game_kept": len(kept), "dropped_by_vm": dict(dropped),
        "n_replay": len(replay), "n_records": len(records),
        "by_task": dict(Counter(r["task"] for r in records)),
        "by_subtype_game": dict(Counter(r["subtype"] for r in kept)),
        "by_split_game": dict(Counter(r["split"] for r in kept)),
        "verification": {"stats": dict(sorted(vstats.items())), "wall_s": vwall},
        "output": out_path, "output_sha256": sha256_file(out_path), "wall_s": time.perf_counter() - t0,
    }
    mp = os.path.join(OUT_DIR, "emitter_sft_v4.manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), indent=1)
    print(f"\nkept {len(kept)}/{len(game)} game records (dropped by VM: {dict(dropped)}) + {len(replay)} replay "
          f"= {len(records)}")
    print(f"  game by split: {manifest['by_split_game']}")
    print(f"wrote {out_path}\nwrote {mp} ({manifest['wall_s']:.0f}s) sha {manifest['output_sha256'][:12]}")


if __name__ == "__main__":
    main()
