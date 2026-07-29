"""H-A3 vs H-A4 — the Zero-Forgetting Stability Benchmark.

cubby-concepts' own CLAUDE.md *names* a "Zero-Forgetting Stability Benchmark"
but no such file/function/test exists in any sibling repo (verified). This
builds it and runs the three candidates head to head, exactly as §9 step 3 asks:

  1. Hebbian associative memory with FLAT decay  — the H-A6 failure class:
     W <- (1-lambda)*W + eta*(value (x) key)  ;  recall = sign(W @ key).
     This is literally GrillCheese's `hebbian_update` form (ΔW=η<pre⊗post>-λW),
     which the survey confirmed is NOT Oja-normalized. lambda=0 gives the pure
     Hopfield-style outer-product memory as a second reference point.
  2. NLMS adaptive associator  — AURA_GENESIS's error-corrective, input-energy-
     normalized update, generalized to a weight MATRIX (the AURA file itself
     notes it deleted its matrix variant): predict, then
     W <- (1-l2)*W + mu * e key^T / (||key||^2 + eps).
  3. SDM  — cubby-concepts' Kanerva Sparse Distributed Memory, imported as-is.

Task: N random bipolar key->value associations, presented ONLINE in a single
sequential pass (a true sequential-learning / forgetting setting, not batch
least-squares). Two views:
  (A) capacity curve   — final recall accuracy vs N;
  (B) recency profile  — at a fixed N past capacity, recall bucketed by write
      order, to expose *which* memories each method forgets.

Parameter budget is matched: the D×D associators and the M=D, D SDM each hold
~D^2 numbers. Standalone; SDM is the only imported candidate, the rest are the
minimal textbook forms of the exact rules named above.
"""
from __future__ import annotations

import sys
import time
import numpy as np

# Windows console defaults to cp1252; keep output ASCII-safe regardless.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-concepts")
from vsa.sdm import SparseDistributedMemory  # noqa: E402
from vsa.core import generate_random_hypervector  # noqa: E402

RECALL_SIM = 0.90   # a pair is "recalled" if cos(retrieved, value) > this


def make_pairs(n, d, rng):
    keys = generate_random_hypervector(dim=d, num=n, rng=rng).astype(np.float64)
    vals = generate_random_hypervector(dim=d, num=n, rng=rng).astype(np.float64)
    return keys, vals


def cos_rows(A, B):
    num = np.einsum("ij,ij->i", A, B)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


# ── Candidate 1: Hebbian associative memory with flat decay (H-A6 form) ──────
def hebbian_recall(keys, vals, d, lam, eta=1.0):
    W = np.zeros((d, d), dtype=np.float64)
    for i in range(len(keys)):                      # ONLINE, one pass
        W = (1.0 - lam) * W + eta * np.outer(vals[i], keys[i])
    retrieved = np.sign(keys @ W.T)                 # (N, d)
    retrieved[retrieved == 0] = 1
    return retrieved


# ── Candidate 2: NLMS adaptive associator (AURA rule, matrix form) ───────────
def nlms_recall(keys, vals, d, mu=0.5, l2=0.0, eps=1e-8):
    W = np.zeros((d, d), dtype=np.float64)
    for i in range(len(keys)):                      # ONLINE, one pass
        k = keys[i]
        y = W @ k
        e = vals[i] - y
        denom = eps + float(k @ k)
        W = (1.0 - l2) * W + mu * np.outer(e, k) / denom
    retrieved = np.sign(keys @ W.T)
    retrieved[retrieved == 0] = 1
    return retrieved


# ── Candidate 3: SDM (imported from cubby-concepts) ─────────────────────────
def sdm_recall(keys, vals, d, m):
    sdm = SparseDistributedMemory(num_locations=m, dim=d, rng=12345)
    sdm.write(keys.astype(np.int8), vals.astype(np.int8))   # batched online write
    retrieved = sdm.read(keys.astype(np.int8)).astype(np.float64)
    return retrieved


def accuracy(retrieved, vals):
    return float((cos_rows(retrieved, vals) > RECALL_SIM).mean())


def main():
    d = 2048
    m = 2048                      # SDM locations == D  => ~D^2 params, matched
    rng = np.random.default_rng(7)
    Ns = [64, 128, 256, 512, 1024, 2048, 4096]

    print("=" * 78)
    print(f"H-A3/H-A4  Zero-Forgetting Stability Benchmark   (D={d}, SDM M={m},"
          f" recall@cos>{RECALL_SIM})")
    print("=" * 78)
    print("Each method holds ~D^2 numbers. Online single-pass sequential writes.\n")

    methods = [
        ("Hebbian λ=0 (Hopfield)", lambda k, v: hebbian_recall(k, v, d, lam=0.0)),
        ("Hebbian λ=0.02 (flat decay)", lambda k, v: hebbian_recall(k, v, d, lam=0.02)),
        ("NLMS (error-corrective)", lambda k, v: nlms_recall(k, v, d)),
        ("SDM (Kanerva)", lambda k, v: sdm_recall(k, v, d, m)),
    ]

    # ---- View A: capacity curve -------------------------------------------
    header = "  N     " + "".join(f"{name[:22]:>24}" for name, _ in methods)
    print("[A] Final recall accuracy vs #associations")
    print(header)
    results = {name: [] for name, _ in methods}
    for n in Ns:
        keys, vals = make_pairs(n, d, rng)
        row = f"  {n:<6}"
        for name, fn in methods:
            acc = accuracy(fn(keys, vals), vals)
            results[name].append(acc)
            row += f"{acc*100:>23.1f}%"
        print(row)

    # ---- View B: recency profile at N past capacity -----------------------
    n_stress = 1024
    print(f"\n[B] Recency profile at N={n_stress} (past most methods' capacity):"
          f" recall by write-order quartile")
    keys, vals = make_pairs(n_stress, d, rng)
    q = n_stress // 4
    buckets = [("oldest 25%", slice(0, q)), ("2nd", slice(q, 2*q)),
               ("3rd", slice(2*q, 3*q)), ("newest 25%", slice(3*q, 4*q))]
    print("  method                      " + "".join(f"{b[0]:>13}" for b in buckets))
    recency = {}
    for name, fn in methods:
        retrieved = fn(keys, vals)
        sims = cos_rows(retrieved, vals) > RECALL_SIM
        vals_by = [float(sims[sl].mean()) for _, sl in buckets]
        recency[name] = vals_by
        print(f"  {name:<26}" + "".join(f"{v*100:>12.1f}%" for v in vals_by))

    # ---- Verdict -----------------------------------------------------------
    print("\n[verdict]")
    # forgetting signature = newest recall >> oldest recall
    for name in ["Hebbian λ=0.02 (flat decay)", "NLMS (error-corrective)",
                 "SDM (Kanerva)"]:
        old, new = recency[name][0], recency[name][-1]
        gap = new - old
        sig = "RECENCY-BIASED forgetting" if gap > 0.15 else "order-agnostic"
        print(f"  {name:<28} oldest={old*100:4.0f}% newest={new*100:4.0f}%"
              f"  -> {sig}")
    return results, recency


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n(wall {time.time()-t0:.1f}s)")
