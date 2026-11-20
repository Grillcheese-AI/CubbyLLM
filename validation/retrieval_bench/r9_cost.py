"""r9_cost -- order 0046, R9's cost part: bind, bundle of 16 and similarity per arm, and bytes per vector. Timed only in
a quiet slot Cowork opens (the plan in 0046, R9).

- the like-for-like pair, both in numpy, on which the 2x line is read: MAP-bipolar as int8 arrays (bind = product,
  bundle = sum then sign, ties to +1 as cubelang's ``sign_from_sum``; similarity = dot / D) beside cubemind's
  ``BlockCodes`` on its numpy path (bind = per-block FFT convolution, bundle normalized per block, similarity
  (1/k) sum(a * b));
- each arm's native time, labelled: cubelang's Rust (``r9_bench cost``: bind, bundle, ``cosine_sim`` and the packed
  cosine the HammingIndex uses) and BlockCodes' numpy (the same as the pair's);
- sizes: MAP D = 4096 and 640, block k = 512 and 80 (l = 128); the median over ``--reps`` samples of 100 calls.
Bytes per vector: MAP D / 8 packed; a block-code key k (one index a block), a frame k * l * 4 (float32).

  python validation/retrieval_bench/r9_cost.py [--reps 200] [--out validation/logs/retrieval_R9_cost_<date>.json]

Wired: STANDALONE (order 0046's bench).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieval_bench.r9_vectors import BENCH, L, block_codes  # noqa: E402

__wiring__ = "STANDALONE"

MAP_DIMS, BLOCK_KS, PAIR = (4096, 640), (512, 80), ((4096, 512), (640, 80))   # equal storage: 512 B, 80 B


def med(f, reps: int) -> float:
    xs = []
    for _ in range(reps):
        t = time.perf_counter_ns()
        for _ in range(100):
            f()
        xs.append((time.perf_counter_ns() - t) / 100)
    return statistics.median(xs)


def map_numpy(dim: int, reps: int) -> dict:
    rng = np.random.default_rng(0)
    v = rng.choice(np.array([-1, 1], dtype=np.int8), size=(16, dim))
    a, b = v[0], v[1]
    sign = lambda s: np.where(s >= 0, 1, -1).astype(np.int8)
    return {"arm": "map", "impl": "numpy", "dim": dim, "bytes": dim // 8,
            "bind_ns": med(lambda: a * b, reps), "bundle16_ns": med(lambda: sign(v.sum(axis=0, dtype=np.int32)), reps),
            "similarity_ns": med(lambda: int(np.dot(a.astype(np.int32), b)) / dim, reps)}


def block_numpy(k: int, reps: int) -> dict:
    bc = block_codes(k)
    v = bc.codebook_discrete(16, seed=0)
    a, b = v[0], v[1]
    return {"arm": "block", "impl": "numpy", "k": k, "l": L, "bytes": k, "frame_bytes": k * L * 4,
            "bind_ns": med(lambda: bc.bind(a, b), reps), "bundle16_ns": med(lambda: bc.bundle(list(v)), reps),
            "similarity_ns": med(lambda: bc.similarity(a, b), reps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.strftime("%H:%M:%S")
    rust = [json.loads(subprocess.run([BENCH, "cost", str(d), str(a.reps)], capture_output=True, text=True, check=True).stdout)
            for d in MAP_DIMS]
    mapn = {d: map_numpy(d, a.reps) for d in MAP_DIMS}
    blk = {k: block_numpy(k, a.reps) for k in BLOCK_KS}
    ratio = {f"{d} vs {k}": {op: round(blk[k][op] / mapn[d][op], 2) for op in ("bind_ns", "bundle16_ns", "similarity_ns")}
             for d, k in PAIR}
    rep = {"what": "order 0046 R9's cost: block codes against MAP-bipolar, the numpy pair (the 2x line) and each native",
           "date": _dt.date.today().isoformat(), "run": [t0, time.strftime("%H:%M:%S")],
           "machine": {"system": platform.system(), "processor": platform.processor(), "cpus": os.cpu_count(),
                       "python": platform.python_version(), "numpy": np.__version__},
           "numpy_pair": {"map": list(mapn.values()), "block": list(blk.values())},
           "block_over_map_numpy": ratio, "native": {"map_rust": rust, "block_numpy": list(blk.values())},
           "line": "block codes no more than 2x slower than MAP-bipolar (the numpy pair, at equal storage)"}
    out = a.out or os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R9_cost_{rep['date']}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(ratio, indent=1), "->", out)


if __name__ == "__main__":
    main()
