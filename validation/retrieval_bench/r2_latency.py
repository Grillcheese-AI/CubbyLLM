"""r2_latency -- order 0069, C: B1q's time per question at 2, 6, 10, 20, 50 and 100 candidates, on the VM
(``$CUBELANG_EXE``, the layered store), B1q as frozen after step D (the veto alone, ``r2_b1q.WALK = "veto"``).

The rows are R2's dev rows that carry a qualifier (``r2_misses.load("dev")``: the same rows, cues and segment facts the
harness prepared). For a count k, each sampled row's candidates are its own that bear the name, filled up to k with
other dev rows' candidates (seed 0) -- real items, whose facts are in the same segment -- so a branch costs what a
real one costs. The time is the host's wall clock around one ``b1q`` call (every VM round trip included), per row.

    python validation/retrieval_bench/r2_latency.py [--per 40]
-> validation/logs/retrieval_R2_latency_<date>.json (p50 / p95 / max ms per k, and the machine)

Wired: STANDALONE (order 0069's bench).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

SIZES = (2, 6, 10, 20, 50, 100)


def segment(facts: dict) -> str:
    """The prepared facts written back as a knowledge segment (one fact a value, its source the item)."""
    path = os.path.join(tempfile.mkdtemp(), "r2_latency_segment.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for key, texts in facts.items():
            for t in texts:
                f.write(json.dumps({"key": key, "text": t, "source": f"wikidata:{key.split(' ', 1)[0]}"}, ensure_ascii=False) + "\n")
    return path


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, int(round(q * len(xs))) - 1))], 3)


def main() -> None:
    from cubbyllm.bridges import vm_identity
    from cubbyllm.bridges.cubelang_client import CubelangSession
    from retrieval_bench import r2_b1q as Q
    from retrieval_bench import r2_misses as M
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=40, help="rows timed at each candidate count")
    a = ap.parse_args()
    Q.WALK, Q.RULES = "veto", set()
    d = M.load("dev")
    rows = [r for r in d["rows"] if r.get("named") and Q.longest([c for c in r.get("cues") or () if (r.get("cue_items") or {}).get(c)])]
    pool = sorted({q for r in d["rows"] for q in r.get("named") or ()})
    rng = random.Random(0)
    kpath = segment(d["facts"])
    session = CubelangSession()
    out = {}
    try:
        Q.b1q(session, kpath, rows[0]["named"], rows[0]["pids"][0] if rows[0]["pids"] else None, rows[0]["cues"],
              rows[0]["cue_items"])                                  # one warm-up call (the segment is loaded once)
        for k in SIZES:
            ms = []
            for r in rng.sample(rows, min(a.per, len(rows))):
                named = list(dict.fromkeys(r["named"]))[:k]
                extra = [q for q in rng.sample(pool, min(len(pool), 4 * k)) if q not in named]
                named += extra[:k - len(named)]
                pid = r["pids"][0] if r["pids"] else None
                t0 = time.perf_counter()
                Q.b1q(session, kpath, named, pid, r["cues"], r["cue_items"])
                ms.append((time.perf_counter() - t0) * 1000)
            out[k] = {"rows": len(ms), "p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95), "max_ms": round(max(ms), 3)}
            print(k, out[k], flush=True)
    finally:
        session.close()
    rep = {"what": "order 0069 C: B1q's time per question by candidate count, B1q frozen (the veto alone)",
           "date": _dt.date.today().isoformat(), "vm": vm_identity.identity(),
           "machine": {"system": platform.system(), "release": platform.release(), "processor": platform.processor(),
                       "cpus": os.cpu_count(), "python": platform.python_version()},
           "rows_with_a_qualifier": len(rows), "candidate_pool": len(pool), "per_count": out,
           "line": "p95 at most 50 ms at every count up to 100",
           "passes": all(v["p95_ms"] <= 50 for v in out.values())}
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R2_latency_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_count"}), "->", path)


if __name__ == "__main__":
    main()
