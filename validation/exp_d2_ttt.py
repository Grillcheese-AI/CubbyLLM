"""H-D2 — TTT vs KV-cache: measure the COMPUTE side of the trade.

The blueprint argues in-place Test-Time-Training fast weights beat the KV-cache
on MEMORY (constant vs O(L) state). H-D2's validate clause notes it never
measures the COMPUTE side: a TTT update is a gradient step per token. So:
measure ms/token and state bytes for both, at matched d, across context length.

  kv-attn   one decode step of single-head attention over an L-entry cache:
            softmax(q K^T / sqrt(d)) V.     compute O(L*d), state O(L*d).
  ttt       one fast-weight update + read: predict, local error, rank-1
            NLMS-style update of W_fast (d x d).
            compute O(d^2) CONSTANT in L, state O(d^2) CONSTANT in L.

The crossover L* where TTT becomes cheaper should be ~O(d) with real-machine
constants — this measures those constants on CPU. What this does NOT measure
(stated plainly): whether TTT fast weights actually RETAIN 128k tokens of
usable context. That is a model-quality question requiring a trained model —
still open after this experiment. Standalone.
"""
from __future__ import annotations

import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = 512
REPS = 200


def bench(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def main():
    rng = np.random.default_rng(0)
    print("=" * 76)
    print(f"H-D2  TTT fast-weights vs KV-cache attention — compute side"
          f"  (d={D}, CPU fp32)")
    print("=" * 76)

    x = rng.standard_normal(D).astype(np.float32)
    W = np.zeros((D, D), dtype=np.float32)

    def ttt_step():
        # read + rank-1 local update (NLMS-style, the cheap TTT variant)
        y = W @ x
        e = x - y                                   # local self-pred error
        W[:] += 0.1 * np.outer(e, x) / (x @ x + 1e-8)
        return y
    t_ttt = bench(ttt_step)

    print(f"\n  {'L (ctx)':>10} {'kv-attn ms/tok':>15} {'ttt ms/tok':>12}"
          f" {'kv state MB':>12} {'ttt state MB':>13}")
    crossover = None
    for L in (1_024, 8_192, 32_768, 131_072):
        K = rng.standard_normal((L, D)).astype(np.float32)
        V = rng.standard_normal((L, D)).astype(np.float32)
        q = rng.standard_normal(D).astype(np.float32)

        def kv_step():
            s = K @ q / np.sqrt(D)
            s = np.exp(s - s.max()); s /= s.sum()
            return s @ V
        t_kv = bench(kv_step)
        kv_mb = 2 * L * D * 4 / 1e6
        ttt_mb = D * D * 4 / 1e6
        if crossover is None and t_ttt < t_kv:
            crossover = L
        print(f"  {L:>10,} {t_kv:>15.3f} {t_ttt:>12.3f} {kv_mb:>12.1f}"
              f" {ttt_mb:>13.1f}")
        del K, V

    print(f"\n[verdict]")
    print(f"  TTT per-token cost is constant ({t_ttt:.3f}ms at d={D});"
          f" kv-attn grows with L.")
    print(f"  measured crossover: TTT cheaper from L ~ "
          f"{crossover:,} up (on this machine/shape)." if crossover else
          "  TTT never cheaper in tested range.")
    print("  memory side confirmed trivially (O(d^2) vs O(L*d)).")
    print("  STILL OPEN (needs a trained model): whether local-gradient fast"
          " weights RETAIN long-context information competitively — the"
          " compute win measured here says nothing about quality.")


if __name__ == "__main__":
    main()
