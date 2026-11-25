"""r9_vectors -- order 0046, R9: which kind of vector the VM should use. The accuracy part (the plan in 0046, R9).

Two arms, each from its own code: MAP-bipolar from cubelang (``examples/r9_bench.rs`` on its branch ``agent-b/r9``,
the crate's ``Codebook``, ``Hypervec`` and ``HammingIndex``), block codes from cubemind's ``BlockCodes``
(``cubemind/ops/block_codes.py``, its numpy path: grilly's import is blocked, so no GPU is touched). The sibling
checkouts are found beside this repo (``cubemind``, ``cubelang-b``).

- recall by key, N = 1k, 10k, 100k; queries: every key at 1k, 1,000 sampled at 10k and 100k; the query keeps an
  expected similarity s = 1.0, 0.8, 0.5, 0.2 to its key (MAP flips (1 - s) / 2 of the signs; block codes re-draw
  (1 - s) / (1 - 1/l) of the blocks); recall@1, the margin (top-1 minus top-2), the measured query-key similarity.
  Keys are ``codebook_discrete``'s draws held as one index a block; the cleanup's similarity is the share of equal
  blocks, which is ``(1/k) sum(a * b)`` for one-hot codes (pinned equal to ``similarity_batch`` in the tests).
- role recovery, 2, 4, 8, 16 pairs, 1,000 frames: ``bind`` each pair, ``bundle`` the frame, ``unbind`` each role, and
  the cleanup against 1,000 fillers (``(1/k) sum(a * b)`` as one matrix product over the fillers, pinned equal to
  ``similarity_batch``).
- arms: MAP D = 4096 (512 B) and D = 640 (80 B); block k = 80 (80 B) and k = 512 (512 B), l = 128; seeds 0, 1, 2.

  python validation/retrieval_bench/r9_vectors.py [--seeds 0 1 2] [--out validation/logs/retrieval_R9_<date>.json]

Wired: STANDALONE (order 0046's bench).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import functools
import json
import os
import statistics
import subprocess
import sys

import numpy as np

__wiring__ = "STANDALONE"

HERE = os.path.dirname(os.path.abspath(__file__))
SIBLINGS = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CUBEMIND = os.path.join(SIBLINGS, "cubemind")
BENCH = os.path.join(SIBLINGS, "cubelang-b", "target", "release", "examples", "r9_bench.exe")
L = 128
NS, SS, PAIRS, FRAMES = (1000, 10000, 100000), (1.0, 0.8, 0.5, 0.2), (2, 4, 8, 16), 1000
MAP_DIMS, BLOCK_KS = (4096, 640), (80, 512)


def block_codes(k: int, l: int = L):
    """cubemind's BlockCodes on its numpy path (grilly's import blocked: Code B uses no GPU)."""
    import importlib
    if CUBEMIND not in sys.path:
        sys.path.insert(0, CUBEMIND)
    held = {k: sys.modules.pop(k) for k in [k for k in sys.modules if k == "grilly" or k.startswith("grilly.")]}
    sys.modules["grilly"] = None                         # blocked even when an earlier import loaded it (a test suite)
    try:
        B = importlib.import_module("cubemind.ops.block_codes")
        if B._BRIDGE_AVAILABLE or B._BLOCK_CODE_OPS_AVAILABLE:
            B = importlib.reload(B)                      # it was first imported with grilly in reach
    finally:
        del sys.modules["grilly"]
        sys.modules.update(held)
    assert not B._BRIDGE_AVAILABLE and not B._BLOCK_CODE_OPS_AVAILABLE, "BlockCodes must run its numpy path"
    return B.BlockCodes(k=k, l=l)


@functools.lru_cache(maxsize=2)
def discrete_indices(n: int, k: int, l: int, seed: int) -> np.ndarray:
    """The indices ``BlockCodes.codebook_discrete(n, seed)`` draws on its numpy path (the same calls, in order)."""
    rng = np.random.default_rng(seed)
    return np.array([[rng.integers(0, l) for _ in range(k)] for _ in range(n)], dtype=np.uint16)


def block_recall(k: int, n: int, s: float, seed: int) -> dict:
    keys = discrete_indices(n, k, L, seed)              # cached across the four s (read only)
    rng = np.random.default_rng(seed ^ 0x5EED)
    qs = np.arange(n) if n <= 1000 else np.sort(rng.choice(n, 1000, replace=False))
    r = int(round((1 - s) / (1 - 1 / L) * k))
    hits, margin, qsim = 0, 0.0, 0.0
    for i in qs:
        q = keys[i].copy()
        blocks = rng.choice(k, r, replace=False)
        q[blocks] = rng.integers(0, L, size=r)
        sims = (keys == q).mean(axis=1)                  # (1/k) sum(a * b) for one-hot codes
        top2 = np.argpartition(-sims, 1)[:2]
        top2 = top2[np.argsort(-sims[top2])]
        hits += int(top2[0] == i)
        margin += float(sims[top2[0]] - sims[top2[1]])
        qsim += float((q == keys[i]).mean())
    m = len(qs)
    return {"arm": "block", "task": "recall", "k": k, "l": L, "n": n, "s": s, "seed": seed, "queries": m,
            "p": r / k, "recall_at_1": hits / m, "margin": margin / m, "query_key_sim": qsim / m}


def block_roles(k: int, pairs: int, seed: int, frames: int = FRAMES) -> dict:
    bc = block_codes(k)
    roles, fillers = bc.codebook_discrete(1000, seed + 101), bc.codebook_discrete(1000, seed + 202)
    flat = fillers.reshape(1000, -1)
    rng = np.random.default_rng(seed ^ 0xF4A3E)
    hits, margin, total = 0, 0.0, 0
    for _ in range(frames):
        ri, fi = rng.choice(1000, pairs, replace=False), rng.choice(1000, pairs, replace=False)
        frame = bc.bundle([bc.bind(roles[a], fillers[b]) for a, b in zip(ri, fi)])
        un = np.stack([bc.unbind(frame, roles[a]) for a in ri]).reshape(pairs, -1)
        sims = un @ flat.T / k                           # similarity_batch's (1/k) sum(a * b), all fillers at once
        for j, b in enumerate(fi):
            top2 = np.argsort(-sims[j])[:2]
            hits += int(top2[0] == b)
            margin += float(sims[j, top2[0]] - sims[j, top2[1]])
            total += 1
    return {"arm": "block", "task": "roles", "k": k, "l": L, "pairs": pairs, "seed": seed, "frames": frames,
            "right_share": hits / total, "margin": margin / total}


def map_run(*args) -> dict:
    out = subprocess.run([BENCH, *map(str, args)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def summary(rows: list[dict], keys: tuple, fields: tuple) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    out = []
    for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        row = dict(zip(keys, g))
        for f in fields:
            xs = [r[f] for r in rs]
            row[f] = {"mean": round(statistics.mean(xs), 4), "spread": round(max(xs) - min(xs), 4), "seeds": len(xs)}
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        for n in NS:
            for s in SS:
                rows += [map_run("recall", d, n, s, seed) for d in MAP_DIMS]
                rows += [block_recall(k, n, s, seed) for k in BLOCK_KS]
                print(seed, n, s, flush=True)
        for p in PAIRS:
            rows += [map_run("roles", d, p, seed, FRAMES) for d in MAP_DIMS]
            rows += [block_roles(k, p, seed) for k in BLOCK_KS]
            print(seed, "roles", p, flush=True)
    for r in rows:
        r["bytes"] = r["dim"] // 8 if r["arm"] == "map" else r["k"]
    rep = {"what": "order 0046 R9: MAP-bipolar (cubelang) against block codes (cubemind), accuracy, as pre-registered",
           "date": _dt.date.today().isoformat(),
           "code": {"map": "cubelang examples/r9_bench.rs (branch agent-b/r9 from main d8a4bae)",
                    "block": "cubemind cubemind/ops/block_codes.py, numpy path",
                    "cubemind_commit": subprocess.run(["git", "-C", CUBEMIND, "rev-parse", "--short", "HEAD"],
                                                      capture_output=True, text=True).stdout.strip(),
                    "cubelang_commit": subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.dirname(BENCH)))), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()},
           "seeds": a.seeds,
           "recall": summary([r for r in rows if r["task"] == "recall"], ("arm", "bytes", "n", "s"),
                             ("recall_at_1", "margin", "query_key_sim", "p")),
           "roles": summary([r for r in rows if r["task"] == "roles"], ("arm", "bytes", "pairs"), ("right_share", "margin")),
           "rows": rows}
    out = a.out or os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R9_{rep['date']}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
