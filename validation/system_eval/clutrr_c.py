"""validation/system_eval/clutrr_c.py -- order 0054's arm C (and C-VM) on CLUTRR, at home on CPU.

H-A13's setting, as ``exp_r34_clutrr_relations.py`` runs it: the night learns the rule library from CLUTRR's train
split (relation asks refused with their gold, the sleep cycle adopting every rule with no counterexample); each test
story's relations are its store; ``relate`` finds every path from A to B, composes each, and speaks one relation only
if every composing path agrees and certifies in the VM. The proposer is the rule library ("rules"), no model.
  C-VM   the first path that composes, its conclusion (a path split between rules: its first conclusion), spoken
         uncertified; nothing composes: refused
A test record whose story cannot be read, or repeats a name, is a refusal (``unreadable``), kept in the count.
Writes ``<out>/c_clutrr.jsonl`` (one record a question; resumable).

    python validation/system_eval/clutrr_c.py --out <dir> [--n 50]

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL), os.path.join(os.path.dirname(VAL), "standin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

KINDS = {"no_path": "fact_missing", "relation_paths_overflow": "walk_failed", "no_rule": "walk_failed",
         "rule_split": "walk_failed", "vm_verify_failed": "not_certified", "unreadable": "no_plan"}


def cvm_of(res: dict) -> str | None:
    """C-VM's answer: the first composing path's conclusion (a split path: its first), or None."""
    for p in res.get("paths") or []:
        if p.get("conclusion"):
            return p["conclusion"]
        if p.get("conclusions"):
            return sorted(p["conclusions"])[0]
    return None


def main(argv=None) -> None:
    import clutrr as C
    from exp_r34_clutrr_relations import TAU, day_rows, night
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.index import TripleIndex
    from cubbyllm.reasoning.skills import MAX_HOPS, MAX_PATHS, relate
    from system_eval.bench import CLUTRR_TASK, _path, clutrr_items
    from system_eval.grade import verdict
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--exe", default=None)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    root = _path("benchmarks")
    train = C.load_raw("train", CLUTRR_TASK, root)
    rows = clutrr_items(C.load_raw("test", CLUTRR_TASK, root))
    if a.n:
        rows = rows[:a.n]
    lib, sk, dt = night(day_rows(train)[0], pathlib.Path(a.out) / "clutrr_night", "clutrr-train")
    print(f"night: {sk['adopted']} rules from {sk['episodes_total']} episodes ({dt:.0f}s)", flush=True)
    session = cc.CubelangSession(exe=a.exe)
    timing = {"n": 0, "s": 0.0}

    def run_fn(source, fn):
        t = time.perf_counter()
        try:
            return session.run(source, fn=fn)
        finally:
            timing["n"] += 1
            timing["s"] += time.perf_counter() - t

    path = os.path.join(a.out, "c_clutrr.jsonl")
    done = {json.loads(x)["id"] for x in open(path, encoding="utf-8") if x.strip()} if os.path.exists(path) else set()
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in done:
                continue
            t, n0, s0 = time.perf_counter(), timing["n"], timing["s"]
            m = r["meta"]
            if not m["facts"]:
                res = {"answer": None, "verified": False, "reason": "unreadable", "paths": []}
            else:
                ps, over = TripleIndex(m["facts"]).paths(m["a"], m["b"], max_hops=MAX_HOPS, limit=MAX_PATHS)
                res = relate(ps, lib, run_fn, tau_vm=dict(TAU), chunk=2, overflow=over)
            spoken = bool(res["verified"])
            cvm = res["answer"] if spoken else cvm_of(res)
            f.write(json.dumps({
                "id": r["id"], "bench": "clutrr", "question": r["question"], "golds": r["golds"], "seen": False,
                "depth": m["depth"], "spoken": spoken, "answer": res["answer"], "said": res["answer"],
                "verdict": verdict(spoken, res["answer"], r["golds"], "clutrr"), "reason": res["reason"],
                "refusal": None if spoken else KINDS.get(res["reason"], f"other:{res['reason']}"),
                "proposer": "rules" if (spoken or cvm) else None, "facts_from": "story" if spoken else None,
                "cvm_answer": cvm, "cvm_verdict": verdict(cvm is not None, cvm, r["golds"], "clutrr"),
                "n_paths": len(res.get("paths") or []), "vm_calls": timing["n"] - n0,
                "vm_s": round(timing["s"] - s0, 3), "wall_s": round(time.perf_counter() - t, 2)},
                ensure_ascii=False) + "\n")
            f.flush()
    json.dump({"rules": sk["adopted"], "episodes": sk["episodes_total"], "night_s": round(dt, 1), "rows": len(rows),
               "vm_calls": timing["n"], "wall_s": round(time.perf_counter() - t0, 1)},
              open(os.path.join(a.out, "c_clutrr.meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
