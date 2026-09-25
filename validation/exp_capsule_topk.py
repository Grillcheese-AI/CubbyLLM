"""Capsule store similarity at scale: grilly2's packed top-k on the GPU against numpy on the CPU, same codes,
same integers. N random 1,024-bit codes (a fact's code is a majority vote of hashed features, so random
codes are the right stand-in for timing); k = 10; median of 20 queries after 3 warm-ups.

    python validation/exp_capsule_topk.py [--n 1000000]
"""
import argparse, json, os, statistics, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "standin")]
from capsules import hamming_numpy  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1_000_000); a = ap.parse_args()
rng = np.random.default_rng(0)
book = rng.integers(-2**31, 2**31, size=(a.n, 32), dtype=np.int64).astype(np.int32)
qs = [book[i] ^ rng.integers(0, 2**31, size=32, dtype=np.int64).astype(np.int32) & 0x0101 for i in rng.integers(0, a.n, 23)]


def cpu(q):
    d = hamming_numpy(q, book)
    i = np.argpartition(d, 10)[:10]
    return i[np.argsort(d[i], kind="stable")], d


import grilly
from grilly.vsa import packed
gbook = grilly.from_numpy(book)


def gpu(q):
    idx, dist, _ = packed.topk(grilly.from_numpy(q), gbook, k=10, dim=1024)
    return idx.numpy(), dist.numpy()


res = {}
for name, fn in (("numpy", cpu), ("grilly2", gpu)):
    ts = []
    for j, q in enumerate(qs):
        t0 = time.perf_counter(); fn(q); dt = time.perf_counter() - t0
        if j >= 3:
            ts.append(dt)
    res[name] = statistics.median(ts) * 1000
same = all(np.array_equal(np.sort(cpu(q)[1][cpu(q)[0]]), np.sort(gpu(q)[1])) for q in qs[:5])
line = (f"capsule top-10 over {a.n:,} codes x 1024 bits: numpy {res['numpy']:.1f} ms, grilly2 {res['grilly2']:.1f} ms "
        f"per query ({res['numpy'] / res['grilly2']:.1f}x); same nearest distances: {same}")
print(line)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
open(os.path.join(HERE, "logs", "exp_capsule_topk.log"), "w", encoding="utf-8").write(line + "\n")
