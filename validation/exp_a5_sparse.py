"""H-A5 — does DG-style sparse expansion reduce pressure on a weight memory?

GrillCheese's hippocampal design (capsulememoryintegration.md) claims a
Dentate-Gyrus-style sparse expansion (random projection up, ~2% winner-take-all)
before storage makes the downstream associative memory interfere less. That's
the classic pattern-separation argument. H-A5's validate clause asks for exactly
this prototype "as a side channel", measured for whether it reduces pressure on
the weight-based memory.

Protocol: same associative task as exp_a_forgetting (N bipolar key->value pairs,
online single pass, Hebbian outer-product storage), at N past dense capacity.

  dense        W (D_out x D):      store v k^T with raw keys      [the baseline]
  dense-wide   W (D_out x D_exp):  keys randomly projected UP to D_exp, dense
               (no WTA) — the size-matched CONTROL: any gain here is just
               parameters, not sparsity.
  dg-sparse    W (D_out x D_exp):  keys projected up then top-k WTA binarized
               (~2% active) — the actual DG mechanism at the same size.

If dg-sparse > dense-wide at matched W size, the SPARSITY is doing real work
(H-A5 supported); if they tie, the benefit was just capacity. Standalone.
"""
from __future__ import annotations

import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = 1024
RECALL_SIM = 0.90


def make_pairs(n, d, rng):
    keys = (rng.integers(0, 2, size=(n, d)) * 2 - 1).astype(np.float64)
    vals = (rng.integers(0, 2, size=(n, d)) * 2 - 1).astype(np.float64)
    return keys, vals


def cos_rows(A, B):
    num = np.einsum("ij,ij->i", A, B)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


def hebbian_store_recall(codes, vals):
    """Online Hebbian outer-product store over arbitrary key codes."""
    W = np.zeros((vals.shape[1], codes.shape[1]))
    for i in range(len(codes)):
        W += np.outer(vals[i], codes[i])
    out = np.sign(codes @ W.T)
    out[out == 0] = 1
    return out


def expand(keys, E, wta_frac):
    """Random projection up; optional top-k winner-take-all binarization."""
    z = keys @ E.T                                # (N, D_exp)
    if wta_frac is None:
        return z / np.linalg.norm(z, axis=1, keepdims=True) * np.sqrt(E.shape[0])
    k = max(1, int(E.shape[0] * wta_frac))
    out = np.zeros_like(z)
    idx = np.argpartition(z, -k, axis=1)[:, -k:]
    np.put_along_axis(out, idx, 1.0, axis=1)      # sparse binary code
    return out


def main():
    rng = np.random.default_rng(3)
    print("=" * 76)
    print(f"H-A5  DG sparse expansion vs dense storage  (D={D}, Hebbian store,"
          f" recall@cos>{RECALL_SIM})")
    print("=" * 76)
    print(f"  {'N assoc':>8} {'dense D':>9} {'dense-wide':>11} {'dg-sparse 2%':>13}"
          f"   (D_exp = 8*D = {8*D})")

    D_exp = 8 * D
    E = rng.standard_normal((D_exp, D)) / np.sqrt(D)
    for n in (128, 256, 512, 1024, 2048):
        keys, vals = make_pairs(n, D, rng)
        acc_dense = float((cos_rows(hebbian_store_recall(keys, vals), vals)
                           > RECALL_SIM).mean())
        wide = expand(keys, E, None)
        acc_wide = float((cos_rows(hebbian_store_recall(wide, vals), vals)
                          > RECALL_SIM).mean())
        sparse = expand(keys, E, 0.02)
        acc_sp = float((cos_rows(hebbian_store_recall(sparse, vals), vals)
                        > RECALL_SIM).mean())
        print(f"  {n:>8} {acc_dense*100:>8.1f}% {acc_wide*100:>10.1f}%"
              f" {acc_sp*100:>12.1f}%")

    print("\n[verdict]")
    print("  If dg-sparse beats dense-wide at matched W size, sparsity (pattern"
          " separation) is doing real work beyond mere parameter count ->"
          " H-A5's mechanism validated as a complement (NOT a replacement) for"
          " the memory-rule fix.")


if __name__ == "__main__":
    main()
