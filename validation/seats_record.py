"""seats_record -- the 450M seats' outputs on fixed prompts, on grilly2 (order 0031: the seats move into the package and
must decode the same, token for token, before and after).

One trunk with two roles (the emitter adapter as ``programs``, a talk adapter as ``talk``): three programs, a talk
decode (a swap), the first program again (a swap back), two sampled programs (temperature 0.7, seed 7), and with
``--records`` the first val record of each task, slotted as the host frames it (``gold_prefix``: the output is the
start of the record's program, i.e. the adapter is in). The plain questions are not slotted, so their "programs" are
not programs; they pin the decode, not the emitter's quality. Written:
the outputs, the swap count, the seats' name (adapter folder names only), the wall time. Run in grilly2's
environment, with the seats' code given as a file (an older copy to compare against) or a module:

    <grilly2 python> validation/seats_record.py --seats standin/emitter_450m.py --export <export dir> \
        --tokenizer <tokenizer.json> --programs <emitter adapter dir> --talk <talk adapter dir> --out <json>
    python validation/seats_record.py --compare <a.json> <b.json>

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time

__wiring__ = "STANDALONE"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAMS = ("What is the capital of France?", "What is 3 plus 4?", "Who is the mother of Justin Trudeau?")
TALK = "Hello! Who are you?"


def seats(spec: str):
    """The seats' module: a file path (loaded under its own name, so an older copy sits beside the new) or a module."""
    if spec.endswith(".py"):
        s = importlib.util.spec_from_file_location("seats_under_test", spec)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        return m
    return importlib.import_module(spec)


def val_records(path: str) -> list:
    """The first val record of each task in a slotted emitter set (``emitter_data``'s records)."""
    seen = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") == "val" and r["task"] not in seen:
            seen[r["task"]] = r
    return [seen[t] for t in sorted(seen)]


def record(a) -> dict:
    m = seats(a.seats)
    t0 = time.perf_counter()
    trunk = m.Cubby450mTrunk(a.export, a.tokenizer, {"programs": a.programs, "talk": a.talk})
    em = m.Cubby450mEmitter(trunk)
    out = {"programs": [em.emit(q, max_new_tokens=a.max_new) for q in PROGRAMS]}
    out["talk_ids"] = trunk.generate("talk", [trunk.tokenizer.encode(TALK).ids], a.max_new)[0]
    out["program_again"] = em.emit(PROGRAMS[0], max_new_tokens=a.max_new)
    out["sampled"] = em.emit_samples(PROGRAMS[1], n=2, max_new_tokens=a.max_new, temperature=0.7, seed=7)
    if a.records:                                        # slotted prompts as the host frames them: the adapter at work
        out["slotted"] = []
        for r in val_records(a.records):
            p = em.emit(r["prompt"], max_new_tokens=a.max_new)
            out["slotted"].append({"id": r["id"], "task": r["task"], "program": p,
                                   "gold_prefix": r["program"].strip().startswith(p.strip())})
    return {"seats": trunk.name, "swaps": trunk.swaps, "max_new": a.max_new, "outputs": out,
            "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seats")
    ap.add_argument("--export")
    ap.add_argument("--tokenizer")
    ap.add_argument("--programs")
    ap.add_argument("--talk")
    ap.add_argument("--records", help="a slotted emitter set (jsonl): the first val record of each task is emitted too")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(p, encoding="utf-8")) for p in a.compare)
        same = x["outputs"] == y["outputs"] and x["swaps"] == y["swaps"]
        diff = [k for k in x["outputs"] if x["outputs"][k] != y["outputs"].get(k)]
        print(json.dumps({"same": same, "differ": diff, "swaps": [x["swaps"], y["swaps"]],
                          "wall_s": [x["wall_s"], y["wall_s"]]}))
        return 0 if same else 1
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    r = record(a)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, ensure_ascii=False)
    print(r["seats"], "swaps", r["swaps"], "wall", r["wall_s"], "s;", r["outputs"]["programs"][0][:80].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
