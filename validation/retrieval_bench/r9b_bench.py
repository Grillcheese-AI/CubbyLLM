"""r9b_bench -- order 0046, R9b: arm 4's accuracy read again (its bundle is a vote), and the cost of arms 1-4 and MAP.

  accuracy  R9's two tasks for arm 4 (the index form), seeds 0, 1, 2, k = 80 and 512, l = 128. Recall by key is R9's
            block computation unchanged (keys as indices, equality share; read again so the row is arm 4's own); role
            recovery's frame is the per-block vote of the bound pairs (``idx_bundle``), unbound by index subtraction and
            cleaned up against the 1,000 fillers by the share of equal blocks.
  cost      timed only in Cowork's slot: R9's single calls (bind, bundle of 16, similarity; the median over samples of
            100 calls) for every CPU arm, and the index job: one query against N = 10k, 100k, 1M stored keys, top-1 and
            top-10, query batches of 1 and 1,000, for every arm that holds N keys in memory (the dense arms' codebook up to
            2 GB: k = 80 at 10k only; spectral keys cost the same as dense); time per query and bytes held. cubelang's Rust MAP arm is ``r9_bench cost`` / ``r9_bench index``.

  python validation/retrieval_bench/r9b_bench.py accuracy [--out validation/logs/retrieval_R9b_accuracy_<date>.json]
  python validation/retrieval_bench/r9b_bench.py cost [--reps 200] [--out validation/logs/retrieval_R9b_cost_<date>.json]

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
from retrieval_bench import r9_vectors as R  # noqa: E402
from retrieval_bench import r9b_arms as A  # noqa: E402

__wiring__ = "STANDALONE"

L, KS, PAIRS, FRAMES, SEEDS = 128, (80, 512), (2, 4, 8, 16), 1000, (0, 1, 2)


def roles_index(k: int, pairs: int, seed: int, frames: int = FRAMES) -> dict:
    roles = R.discrete_indices(1000, k, L, seed + 101).astype(np.uint8)
    fillers = R.discrete_indices(1000, k, L, seed + 202).astype(np.uint8)
    rng = np.random.default_rng(seed ^ 0xF4A3E)
    hits, margin, total = 0, 0.0, 0
    for _ in range(frames):
        ri, fi = rng.choice(1000, pairs, replace=False), rng.choice(1000, pairs, replace=False)
        frame = A.idx_bundle(np.stack([A.idx_bind(roles[a], fillers[b], L) for a, b in zip(ri, fi)]), L)
        for a, b in zip(ri, fi):
            sims = A.idx_scores(A.idx_unbind(frame, roles[a], L), fillers)
            t2 = A.top(sims, 2)
            hits += int(t2[0] == b)
            margin += float(sims[t2[0]] - sims[t2[1]])
            total += 1
    return {"arm": "index", "task": "roles", "k": k, "pairs": pairs, "seed": seed, "right_share": hits / total,
            "margin": margin / total}


def accuracy(out: str) -> None:
    rows = []
    for seed in SEEDS:
        for k in KS:
            for n in R.NS:
                for s in R.SS:
                    rows.append({**R.block_recall(k, n, s, seed), "arm": "index"})
            for p in PAIRS:
                rows.append(roles_index(k, p, seed))
            print(seed, k, flush=True)
    rep = {"what": "order 0046 R9b: arm 4 (the index form, the vote bundle) on R9's two tasks", "date": _dt.date.today().isoformat(),
           "recall": R.summary([r for r in rows if r["task"] == "recall"], ("arm", "k", "n", "s"), ("recall_at_1", "margin")),
           "roles": R.summary([r for r in rows if r["task"] == "roles"], ("arm", "k", "pairs"), ("right_share", "margin")),
           "rows": rows}
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print("->", out)


def med(f, reps: int) -> float:
    xs = []
    for _ in range(reps):
        t = time.perf_counter_ns()
        for _ in range(100):
            f()
        xs.append((time.perf_counter_ns() - t) / 100)
    return statistics.median(xs)


def single_calls(k: int, reps: int) -> list[dict]:
    bc = R.block_codes(k)
    cb = bc.codebook_discrete(16, seed=0)
    a, b = cb[0], cb[1]
    frame = bc.bundle(list(cb))
    sa, sb, sv, sf = A.to_spec(a), A.to_spec(b), A.to_spec(cb), A.to_spec(frame)
    ix = cb.argmax(axis=-1).astype(np.uint8)
    A.idx_bind(ix[0], ix[1], L); A.idx_bundle(ix, L); A.idx_similarity(ix[0], ix[1])       # numba: compile first
    A.frame_against_key(frame, ix[2]); A.frame_shift(frame, ix[2])
    out = [{"arm": "numpy", "bind": med(lambda: bc.bind(a, b), reps), "bundle16": med(lambda: bc.bundle(list(cb)), reps),
            "similarity": med(lambda: bc.similarity(a, b), reps)},
           {"arm": "scipy", "bind": med(lambda: A.scipy_bind(a, b), reps), "bundle16": med(lambda: A.bundle_dense(cb), reps),
            "similarity": med(lambda: A.similarity_dense(a, b), reps)},
           {"arm": "spectral", "bind": med(lambda: A.spec_bind(sa, sb), reps), "bundle16": med(lambda: A.spec_bundle(sv), reps),
            "similarity": med(lambda: A.spec_similarity(sf, sb, L), reps), "to_spec": med(lambda: A.to_spec(a), reps),
            "from_spec": med(lambda: A.from_spec(sf, L), reps)},
           {"arm": "index", "bind": med(lambda: A.idx_bind(ix[0], ix[1], L), reps), "bundle16": med(lambda: A.idx_bundle(ix, L), reps),
            "similarity": med(lambda: A.idx_similarity(ix[0], ix[1]), reps),
            "frame_against_key": med(lambda: A.frame_against_key(frame, ix[2]), reps),
            "frame_bound_with_key": med(lambda: A.frame_shift(frame, ix[2]), reps)}]
    return [{**r, "k": k, "unit": "ns per call"} for r in out]


def index_job(size: int, arm: str, n: int, batch: int) -> dict:
    """Time per query to score ``batch`` queries against ``n`` stored keys, then top-1 and top-10 on those scores (the
    median of 20 runs for one query, one run of 1,000); bytes held."""
    rng = np.random.default_rng(1)
    if arm == "index":
        keys = rng.integers(0, L, size=(n, size), dtype=np.uint8)
        one = lambda q: A.idx_scores(q, keys)
    elif arm == "map_packed":                                # drawn packed: 1M x 4096 as int8 would be 4 GB
        keys = rng.integers(0, 2**63, size=(n, size // 64), dtype=np.uint64) ^ rng.integers(0, 2, size=(n, size // 64),
                                                                                          dtype=np.uint64) << np.uint64(63)
        one = lambda q: A.map_scores_packed(q, keys, size)
    else:                                                    # dense block codes (arms 1, 2): one matrix product a batch
        keys = R.block_codes(size).codebook_discrete(n, seed=1).reshape(n, -1)
        one = None
    qs = keys[rng.choice(n, batch, replace=False)]
    scores = (lambda Q: (Q @ keys.T) / size) if one is None else (lambda Q: np.stack([one(q) for q in Q]))
    scores(qs[:1])
    runs = 20 if batch == 1 else 1
    ts, out = [], None
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        out = scores(qs)
        ts.append((time.perf_counter_ns() - t0) / batch)
    tops = {}
    for t in (1, 10):
        t0 = time.perf_counter_ns()
        A.top(out, t)
        tops[f"top{t}_ns_per_query"] = (time.perf_counter_ns() - t0) / batch
    return {"arm": arm, "size": size, "n": n, "batch": batch, "score_ns_per_query": statistics.median(ts), **tops,
            "bytes_held": int(keys.nbytes)}


def cost(out: str, reps: int) -> None:
    import numba
    import scipy
    t0 = time.strftime("%H:%M:%S")
    singles = [r for k in KS for r in single_calls(k, reps)]
    rust = [json.loads(subprocess.run([R.BENCH, "cost", str(d), str(reps)], capture_output=True, text=True, check=True).stdout)
            for d in (4096, 640)]
    jobs = []
    for n in (10_000, 100_000, 1_000_000):
        for batch in (1, 1000):
            jobs += [index_job(k, "index", n, batch) for k in KS]
            jobs += [index_job(d, "map_packed", n, batch) for d in (4096, 640)]
            jobs += [index_job(k, "dense", n, batch) for k in KS if n * k * L * 4 <= 2**31]   # a codebook up to 2 GB
            jobs += [json.loads(subprocess.run([R.BENCH, "index", str(d), str(n), str(batch), "10"], capture_output=True,
                                               text=True, check=True).stdout) for d in (4096, 640)]   # cubelang's HammingIndex
        print(n, flush=True)
    rep = {"what": "order 0046 R9b's cost: block codes four ways on the CPU, MAP beside them", "date": _dt.date.today().isoformat(),
           "run": [t0, time.strftime("%H:%M:%S")],
           "machine": {"system": platform.system(), "processor": platform.processor(), "cpus": os.cpu_count(),
                       "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                       "numba": numba.__version__, "numba_threads": numba.get_num_threads()},
           "single_calls": singles, "map_rust": rust, "index_job": jobs}
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print("->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("accuracy", "cost"))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R9b_{a.what}_{_dt.date.today().isoformat()}.json")
    accuracy(out) if a.what == "accuracy" else cost(out, a.reps)


if __name__ == "__main__":
    main()
