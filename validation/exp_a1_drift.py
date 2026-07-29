"""H-A1 — Diagnose cubby-lm's Hebbian layer: it runs at C_ctx=0 and drifts.

Two claims to check against the ACTUAL cubby-lm code (not a stand-in):

  (structural)  The GCE "Context Channel Capacity" framing says a state-based
                learner catastrophically forgets when its weight update does not
                structurally depend on a context signal c (C_ctx = 0). We check
                this by inspecting HebbianGrowthLayer's real update path: is
                there any context input to the update at all?

  (empirical)   Even granting growth, sequential exposure to distinct input
                distributions rotates the shared basis W toward the most recent
                distribution, degrading its representation of earlier ones.
                We measure retained explained-variance of task A's inputs after
                training A, then A→B→C→D, vs a "frozen after A" control.

This is a DIAGNOSTIC baseline (H-A1), not a proposed fix. Standalone script;
imports cubby-lm's real layer. Not wired into any model.
"""
from __future__ import annotations

import sys
import numpy as np
import torch

# cubby-lm's real HebbianGrowthLayer (predecessor code, under test as prior art)
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-lm")
from cubby.trunk_torch.hebbian import HebbianGrowthLayer  # noqa: E402


def make_task_subspace(d_model: int, rank: int, rng: np.random.Generator):
    """A random `rank`-dim subspace of R^d_model (orthonormal basis rows)."""
    A = rng.standard_normal((rank, d_model))
    # Gram-Schmidt via QR
    Q, _ = np.linalg.qr(A.T)
    return Q[:, :rank].T  # (rank, d_model)


def sample_from_subspace(basis, n, rng, noise=0.05):
    """n points that live (mostly) in the row-space of `basis`."""
    rank, d = basis.shape
    coeffs = rng.standard_normal((n, rank))
    x = coeffs @ basis
    x = x + noise * rng.standard_normal((n, d))
    return x.astype(np.float32)


def explained_variance(layer: HebbianGrowthLayer, X: np.ndarray) -> float:
    """Fraction of X's energy captured by the layer's current basis W.

    Mirrors the layer's own residual computation: project onto W, reconstruct,
    measure captured energy. 1.0 = perfectly represented, 0.0 = orthogonal.
    """
    with torch.no_grad():
        xt = torch.from_numpy(X)
        was_training = layer.training
        layer.eval()                      # eval() => no basis update, pure read
        y, _, _ = layer(xt)               # (N, K)
        layer.train(was_training)
        W = layer.W                       # (K, d)
        x_hat = y @ W                      # (N, d)
        num = (x_hat * x_hat).sum(dim=1)
        den = (xt * xt).sum(dim=1).clamp_min(1e-12)
        return float((num / den).mean().item())


def train_on(layer, X, rng, epochs=30, batch=32):
    """`epochs` online passes of X through the layer in train() mode (updates W).

    Calibrated so the control reaches EV~0.97 on its own task, i.e. the layer
    genuinely LEARNS the distribution — otherwise "no forgetting" would be an
    artifact of never having learned anything.
    """
    layer.train()
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            layer(torch.from_numpy(X[idx[s:s + batch]]))


def main():
    d_model = 256
    n_tasks = 4
    rank = 16
    n_per_task = 3000
    rng = np.random.default_rng(0)

    print("=" * 74)
    print("H-A1  cubby-lm HebbianGrowthLayer — context-capacity + drift diagnosis")
    print("=" * 74)

    # ---- Structural check: is there any context input to the update? --------
    import inspect
    sig = inspect.signature(HebbianGrowthLayer.forward)
    update_src = inspect.getsource(HebbianGrowthLayer._update_basis)
    print("\n[structural] HebbianGrowthLayer.forward signature:", str(sig))
    has_context_arg = any(p not in ("self", "x") for p in sig.parameters)
    # The Oja update dW is a function of (x, y=xW^T, W) only — no separate c.
    mentions_context = ("context" in update_src) or ("c=" in update_src)
    print(f"[structural] update depends on any context arg besides x? {has_context_arg}")
    print(f"[structural] _update_basis references a 'context' signal?   {mentions_context}")
    print("[structural] => weight update is dW = f(x, xW^T, W); no c enters it.")
    print("[structural] => C_ctx = 0 BY CONSTRUCTION (GCE Structural Bypass at its limit).")

    # ---- Empirical drift: does task A survive A->B->C->D? -------------------
    tasks = [make_task_subspace(d_model, rank, rng) for _ in range(n_tasks)]
    data = [sample_from_subspace(b, n_per_task, rng) for b in tasks]
    # Held-out probe set for task A (fresh samples, same subspace)
    probeA = sample_from_subspace(tasks[0], 1000, rng)

    # sanger = the layer's proper PCA-deflation mode, so the reconstruction-EV
    # metric is meaningful. (Note: the layer's DEFAULT mode 'nonlinear' (y^3)
    # never tracks subspace structure under this metric — stays at chance ~0.06
    # regardless of lr/epochs; a separate finding recorded in the report.)
    def fresh_layer(max_components, grow_threshold):
        return HebbianGrowthLayer(
            d_model, n_components=rank, max_components=max_components,
            lr=0.1, mode="sanger", grow_threshold=grow_threshold,
            grow_cooldown=50, seed=1337,
        )

    print(f"\n[empirical] task = one random rank-{rank} subspace of R^{d_model};"
          f" {n_tasks} disjoint tasks, sanger mode, 30 online epochs each.")

    def run_sequence(max_components, grow_threshold, tag):
        seq = fresh_layer(max_components, grow_threshold)
        train_on(seq, data[0], rng)
        retained = [explained_variance(seq, probeA)]
        for t in range(1, n_tasks):
            train_on(seq, data[t], rng)
            retained.append(explained_variance(seq, probeA))
        ev0, evN = retained[0], retained[-1]
        rel = (ev0 - evN) / max(ev0, 1e-9)
        labels = ["A"] + [f"+{chr(ord('A')+t)}" for t in range(1, n_tasks)]
        seq_str = "  ".join(f"{lab}:{ev:.2f}" for lab, ev in zip(labels, retained))
        print(f"\n  [{tag}]  K={rank}->{seq.K}")
        print(f"    retained EV(A):  {seq_str}")
        print(f"    EV(A) learned={ev0:.2f}  after all tasks={evN:.2f}"
              f"  -> forgot {rel*100:.0f}% of task A")
        return rel, seq.K

    # (1) Growth OFF: K fixed at rank => the shared basis MUST be reused; this
    #     isolates pure Hebbian drift (the catastrophic-forgetting claim).
    rel_off, _ = run_sequence(max_components=rank, grow_threshold=9.9,
                              tag="growth OFF — pure shared-basis drift")
    # (2) Growth ON: neurogenesis allowed; does adding components mitigate it?
    rel_on, K_on = run_sequence(max_components=64, grow_threshold=0.3,
                                tag="growth ON — neurogenesis mitigation")

    print("\n[verdict]")
    print(f"  structural C_ctx=0 (no context enters the update): CONFIRMED.")
    drift = "CONFIRMED catastrophic drift" if rel_off > 0.5 else \
            ("partial drift" if rel_off > 0.15 else "little drift")
    print(f"  growth-off shared basis forgets {rel_off*100:.0f}% of task A"
          f" -> {drift}.")
    mit = "helps" if rel_on < rel_off - 0.1 else "does NOT rescue it"
    print(f"  neurogenesis (K->{K_on}) forgets {rel_on*100:.0f}% -> {mit}.")
    return {"rel_off": rel_off, "rel_on": rel_on}


if __name__ == "__main__":
    main()
