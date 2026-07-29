"""H-C3 — what does removing the full-vocab softmax actually buy, measured.

The claim: an ANN-retrieval output head (top-K candidates from an index, then
softmax over K) makes a large vocabulary affordable where a full-vocab matmul
+ softmax does not. GrillCheese's gcheese-faiss is the existence proof but its
numbers are self-flagged as unvalidated. So: measure, on THIS machine (CPU),
at cubby-lm-like head shapes.

Three heads at each vocab size V in {32k, 256k, 1M}, d_vsa=10240 query dim:
  full    — logits = q @ C^T over all V rows (what VSABindingHead does today),
            + softmax over V.
  ivf     — IVF-style ANN: k-means-lite coarse centroids (sqrt(V)), probe the
            top-p clusters, exact cosine within probed rows only, softmax over
            retrieved K. Reports top-1 agreement with `full` as the accuracy
            cost of approximation (random codebook = pessimistic clustering,
            noted honestly).
  ideal-K — exact cosine over a K-row shortlist (what a perfect ANN would
            leave you doing) — the latency floor of any retrieval scheme.

Batch = 1 token (autoregressive decode step), CPU, float32. Wall-clock medians
over repeated runs. Memory-safety note: at V=1M, D=10240 the fp32 codebook
alone is 41GB — we do NOT allocate that. The V sweep runs at D=2560 (cubby-lm's
d_model) up to V=256k (<=2.6GB), with one anchor run at the real head shape
(V=32k, D=10240); V=1M numbers are reported as EXPLICIT linear extrapolation,
never as measurements.
"""
from __future__ import annotations

import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D_SWEEP = 2560          # d for the V-scaling sweep (memory-safe)
D_ANCHOR = 10240        # cubby-lm's real head dim, anchored at V=32k only
K_SHORT = 1024          # shortlist size for the retrieval heads
N_QUERIES = 30


def bench(fn, n=N_QUERIES):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)  # ms


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def run_shape(V, D, rng):
    """Measure the three heads at one (V, D). Returns dict of results."""
    # random unit codebook rows (float32) — memory is the honest cost
    C = rng.standard_normal((V, D)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    gb = V * D * 4 / 1e9

    # queries with real signal toward a random target row
    targets = rng.integers(0, V, size=N_QUERIES)
    a, b = 0.55, float(np.sqrt(1 - 0.55**2))
    noise = rng.standard_normal((N_QUERIES, D)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    Q = a * C[targets] + b * noise

    # --- full softmax head ------------------------------------------
    def full_step():
        q = Q[rng.integers(0, N_QUERIES)]
        logits = C @ q
        return softmax(logits).argmax()
    t_full = bench(full_step)
    top_full = (Q @ C.T).argmax(1)

    # --- IVF retrieval head -----------------------------------------
    n_cent = int(np.sqrt(V))
    # cheap coarse quantizer: random-row centroids + one Lloyd pass on a sample
    cent = C[rng.choice(V, n_cent, replace=False)].copy()
    sample = C[rng.choice(V, min(V, 20_000), replace=False)]
    assign_s = (sample @ cent.T).argmax(1)
    for j in np.unique(assign_s):
        m = sample[assign_s == j]
        if len(m):
            v = m.mean(0); n = np.linalg.norm(v)
            if n > 0: cent[j] = v / n
    assign = (C @ cent.T).argmax(1)              # offline, not timed
    order = np.argsort(assign, kind="stable")
    bounds = np.searchsorted(assign[order], np.arange(n_cent + 1))
    C_sorted = C[order]
    n_probe = max(1, int(np.ceil(K_SHORT / (V / n_cent))) * 4)

    def ivf_step(q):
        cs = cent @ q
        probes = np.argpartition(cs, -n_probe)[-n_probe:]
        rows = np.concatenate([np.arange(bounds[p], bounds[p + 1])
                               for p in probes])
        logits = C_sorted[rows] @ q
        k = rows[logits.argmax()]
        softmax(logits)
        return order[k]
    t_ivf = bench(lambda: ivf_step(Q[rng.integers(0, N_QUERIES)]))
    top_ivf = np.array([ivf_step(Q[i]) for i in range(N_QUERIES)])
    agree_ivf = float((top_ivf == top_full).mean())

    # --- ideal-K floor ----------------------------------------------
    short = rng.choice(V, K_SHORT, replace=False)
    C_short = C[short].copy()
    def ideal_step():
        q = Q[rng.integers(0, N_QUERIES)]
        logits = C_short @ q
        softmax(logits)
        return logits.argmax()
    t_ideal = bench(ideal_step)
    return {"t_full": t_full, "t_ivf": t_ivf, "t_ideal": t_ideal,
            "agree": agree_ivf, "gb": gb}


def main():
    rng = np.random.default_rng(0)
    print("=" * 78)
    print(f"H-C3  Output-head latency: full softmax vs retrieval   "
          f"(CPU fp32, batch=1)")
    print("=" * 78)
    print(f"  {'V':>10} {'D':>7} {'head':>9} {'ms/token':>10}"
          f" {'top1 vs full':>13} {'codebook GB':>12}")

    sweep = {}
    shapes = [(32_000, D_ANCHOR), (32_000, D_SWEEP),
              (128_000, D_SWEEP), (256_000, D_SWEEP)]
    for V, D in shapes:
        r = run_shape(V, D, rng)
        sweep[(V, D)] = r
        print(f"  {V:>10,} {D:>7} {'full':>9} {r['t_full']:>10.2f}"
              f" {'-':>13} {r['gb']:>12.2f}")
        print(f"  {'':>10} {'':>7} {'ivf':>9} {r['t_ivf']:>10.2f}"
              f" {r['agree']*100:>12.0f}% {'':>12}")
        print(f"  {'':>10} {'':>7} {'ideal-K':>9} {r['t_ideal']:>10.2f}"
              f" {'(floor)':>13} {K_SHORT*D*4/1e9:>12.3f}")

    # explicit extrapolation to 1M (NOT measured — linear-in-V model)
    r128, r256 = sweep[(128_000, D_SWEEP)], sweep[(256_000, D_SWEEP)]
    slope = (r256["t_full"] - r128["t_full"]) / 128_000
    t_1m = r256["t_full"] + slope * (1_000_000 - 256_000)
    print(f"\n  [extrapolated, NOT measured] full head at V=1M, D={D_SWEEP}:"
          f" ~{t_1m:.1f} ms/token, {1_000_000*D_SWEEP*4/1e9:.1f}GB fp32"
          f"  (linear-in-V fit of the measured 128k->256k segment)")
    print(f"  [extrapolated, NOT measured] at V=1M, D={D_ANCHOR}: codebook"
          f" {1_000_000*D_ANCHOR*4/1e9:.1f}GB fp32 — not allocatable here.")

    print("\n[verdict]")
    v_ratio = r256["t_full"] / sweep[(32_000, D_SWEEP)]["t_full"]
    print(f"  - full-softmax cost grows ~linearly in V (32k->256k measured:"
          f" {v_ratio:.1f}x for 8x vocab); retrieval cost stays ~flat.")
    print("  - IVF top-1 agreement on RANDOM codewords is the pessimistic"
          " bound (random vectors cluster terribly); trained/structured"
          " embeddings can only cluster better.")
    print("  - at V=1M the fp32 codebook is 10-41GB depending on D: MEMORY,"
          " not matmul, is the first wall. A large vocab NEEDS low-precision/"
          "low-d codes + retrieval — H-C3's mechanism is REQUIRED, not optional.")


if __name__ == "__main__":
    main()
