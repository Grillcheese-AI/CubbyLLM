"""H-P4 — keyframe compression: interpolate between stored states, or recompute?

CLAIM (the AURA note's goal, restated as a test). Store the state every N steps
and regenerate the in-betweens. Two regenerators are on trial:
  * INTERPOLATION between keyframes — what the note proposes. Lossless only if
    the trajectory is close to straight between keyframes (an empirical property
    of the model, not a theorem), and by the sampling theorem only for units
    whose time constant exceeds the gap.
  * RECOMPUTE from the keyframe plus the tokens in between — what H-P1 makes
    cheap. Exact by construction.
Plus the numerical facts about the note's operators, pinned so the critique is
a measurement rather than an argument: with two keyframes and no time axis,
the Hilbert and Fourier strategies as written are IDENTICAL to linear
interpolation (Re and F are linear), and the rank-one Hamiltonian flow
preserves the analytic-signal norm and does not arrive at the second keyframe.

MEASURE. Along a decode trajectory (T x n_gru_layers x d): for gaps N in
{2, 4, 8, 16, 32} the relative RMS error of linear interpolation vs the
hold-last-keyframe baseline (the do-nothing interpolator); per-unit error at
N=8 against per-unit tau from the gate (Spearman rho — slow units should
interpolate better); recompute error (fp noise); bytes per step for full
states vs keyframes + tokens.

DECISION RULE. If linear interpolation at N=4 is not clearly under the hold
baseline, interpolation-based compression is dead on this model and recompute
wins outright. rho < 0 confirms the multi-resolution scheme: store slow units
sparsely, recompute (or drop) fast ones.

With CB_CKPT (+ CUBBY_SPM and CB_TEXT/CB_CORPUS) the same numbers on the real
model, which is the measurement that actually decides the store's design.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _common as C  # noqa: E402

GAPS = (2, 4, 8, 16, 32)


# ── the AURA note's operators, as written ────────────────────────────────────

def analytic(m: torch.Tensor) -> torch.Tensor:
    """Analytic signal along the latent index — the only axis two keyframes offer."""
    n = m.shape[-1]
    X = torch.fft.fft(m.to(torch.complex128))
    h = torch.zeros(n, dtype=torch.float64)
    if n % 2 == 0:
        h[0], h[n // 2], h[1:n // 2] = 1.0, 1.0, 2.0
    else:
        h[0], h[1:(n + 1) // 2] = 1.0, 2.0
    return torch.fft.ifft(X * h)


def interp_linear(m0, m1, a):
    return (1 - a) * m0 + a * m1


def interp_hilbert(m0, m1, a):
    return ((1 - a) * analytic(m0) + a * analytic(m1)).real


def interp_fourier(m0, m1, a):
    f0, f1 = torch.fft.fft(m0.to(torch.complex128)), torch.fft.fft(m1.to(torch.complex128))
    return torch.fft.ifft((1 - a) * f0 + a * f1).real


def hamiltonian_flow(m0, m1, tau, eps=1e-8):
    """A(tau) = exp(-i H tau) A0 with H = d d^dag / (|d|^2 + eps), d = A1 - A0.
    H is rank one and Hermitian, so the flow only rotates the phase of A0's
    component along d and leaves the rest fixed. Returned complex."""
    a0, a1 = analytic(m0), analytic(m1)
    d = a1 - a0
    d2 = (d.abs() ** 2).sum()
    lam = d2 / (d2 + eps)
    dhat = d / torch.sqrt(d2)
    coef = (torch.exp(torch.tensor(-1j * float(lam) * tau)) - 1.0) * (dhat.conj() @ a0)
    return a0 + coef * dhat


def operator_pins(m0: torch.Tensor, m1: torch.Tensor) -> dict:
    m0, m1 = m0.double(), m1.double()
    a = 0.42
    lin = interp_linear(m0, m1, a)
    ham1 = hamiltonian_flow(m0, m1, 1.0)
    return dict(
        hilbert_vs_linear=float((interp_hilbert(m0, m1, a) - lin).abs().max()),
        fourier_vs_linear=float((interp_fourier(m0, m1, a) - lin).abs().max()),
        ham_norm_drift=float(abs(ham1.abs().norm() - analytic(m0).abs().norm()) / analytic(m0).abs().norm()),
        ham_endpoint_rel_err=float((ham1.real - m1).norm() / m1.norm()),
        linear_endpoint_rel_err=float((interp_linear(m0, m1, 1.0) - m1).norm() / m1.norm()),
    )


# ── keyframe measurements on a real trajectory ──────────────────────────────

def interp_errors(traj: torch.Tensor, N: int):
    """traj (T, D). Returns (rel_lin, rel_hold, per_unit_rel_lin (D,))."""
    T, D = traj.shape
    num_lin = torch.zeros(D, dtype=torch.float64)
    num_hold = torch.zeros(D, dtype=torch.float64)
    mean = traj.double().mean(0)
    den = torch.zeros(D, dtype=torch.float64)
    for k0 in range(0, T - N, N):
        m0, m1 = traj[k0].double(), traj[k0 + N].double()
        for j in range(1, N):
            true = traj[k0 + j].double()
            lin = (1 - j / N) * m0 + (j / N) * m1
            num_lin += (lin - true) ** 2
            num_hold += (m0 - true) ** 2
            den += (true - mean) ** 2                      # variance around the unit's mean
    rel_lin = math.sqrt(float(num_lin.sum() / den.sum()))
    rel_hold = math.sqrt(float(num_hold.sum() / den.sum()))
    ok = den > 1e-12 * den.max()                          # skip dead / constant units
    per_unit = torch.full((D,), float("nan"), dtype=torch.float64)
    per_unit[ok] = torch.sqrt(num_lin[ok] / den[ok])
    return rel_lin, rel_hold, per_unit


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def recompute_error(model, ids: torch.Tensor, traj: torch.Tensor, k0: int, N: int) -> float:
    """Fork at keyframe k0 (state after ids[k0]) and re-run ids[k0+1 .. k0+N];
    compare with the recorded in-between states."""
    _, st = C.prefix_state(model, ids[:k0 + 1])
    track = []
    C.continue_from(model, st, ids[k0 + 1:k0 + N + 1], track=track)
    rec = torch.stack(track).reshape(N, -1)
    return float((rec - traj[k0 + 1:k0 + N + 1]).abs().max())


def measure(model, ids: torch.Tensor, V: int, verbose: bool, tag: str) -> dict:
    traj3 = C.state_trajectory(model, ids)                  # (T, n_gru, d)
    T, n_gru, d = traj3.shape
    traj = traj3.reshape(T, -1)
    # per-unit tau, concatenated in the same (layer, unit) order as traj's columns
    prof = C.retention_profile(model, ids[:-1].view(1, -1))
    tau = torch.cat([C.unit_time_constants({i: prof[i]})[i] for i in sorted(prof)])
    rows, r = [], {"gaps": {}}
    for N in GAPS:
        if T < 3 * N:
            continue
        rel_lin, rel_hold, per_unit = interp_errors(traj, N)
        r["gaps"][N] = (rel_lin, rel_hold)
        if N == 8:
            r["rho_tau_vs_err"] = spearman(np.log(tau.numpy()), per_unit.numpy())
        rows.append(f"gap {N:>2}: linear interp rel-RMS {rel_lin:.3f} | hold-last {rel_hold:.3f} "
                    f"| ratio {rel_lin / max(rel_hold, 1e-12):.2f}")
    r["recompute_err"] = max(recompute_error(model, ids, traj, k0, 8) for k0 in (3, 17))
    state_bytes = n_gru * d * 4
    tok_bytes = math.ceil(math.log2(V)) / 8
    r["bytes_per_step"] = {"states": state_bytes,
                           **{N: state_bytes / N + tok_bytes for N in GAPS}}
    if verbose:
        C.banner(f"[{tag}: T={T} steps, {n_gru} recurrent layers x d={d}]", rows + [
            f"Spearman(log tau, per-unit interp error @gap 8) = {r.get('rho_tau_vs_err', float('nan')):+.3f} "
            "(negative = slow units interpolate better)",
            f"recompute from keyframe + tokens: max |err| = {r['recompute_err']:.1e}",
            "bytes/step (recurrent state only; a hybrid's attention KV cache is on top of this) - "
            "full states: %d | keyframes+tokens at gap 8: %.1f | gap 32: %.1f"
            % (state_bytes, r["bytes_per_step"][8], r["bytes_per_step"][32]),
        ])
    return r


def run(quick: bool = False, verbose: bool = True) -> dict:
    model, g, _ = C.trained_toy(steps=150 if quick else 400)
    toks, _ = g.sample(24 if quick else 64, np.random.default_rng(77))
    ids = torch.from_numpy(toks)
    r = measure(model, ids, g.V, verbose, "toy model, branch grammar")

    traj = C.state_trajectory(model, ids[:64]).reshape(64, -1)
    pins = operator_pins(traj[0], traj[8])
    r["pins"] = pins
    if verbose:
        C.banner("[AURA note operators, two real keyframes (t=0, t=8)]", [
            f"Hilbert-interp vs linear: max |diff| = {pins['hilbert_vs_linear']:.1e}  (identical)",
            f"Fourier-interp vs linear: max |diff| = {pins['fourier_vs_linear']:.1e}  (identical)",
            f"Hamiltonian flow at tau=1: analytic norm drift {pins['ham_norm_drift']:.1e} (conserved), "
            f"rel err to keyframe 1 = {pins['ham_endpoint_rel_err']:.3f} (linear: {pins['linear_endpoint_rel_err']:.1e})",
        ])
    assert pins["hilbert_vs_linear"] < 1e-9 and pins["fourier_vs_linear"] < 1e-9
    assert pins["ham_norm_drift"] < 1e-8 and pins["ham_endpoint_rel_err"] > 0.3
    assert r["recompute_err"] < 1e-4, "recompute from a keyframe is not exact — H-P1 broken?"
    lin4, hold4 = r["gaps"][4]
    assert lin4 < hold4, "linear interpolation does not even beat hold-last at gap 4"

    if C.CKPT:
        m, meta = C.load_checkpoint(C.CKPT)
        real = C.real_tokens(1024)
        if real is None:
            print("CB_CKPT set but no CB_TEXT/CB_CORPUS (+CUBBY_SPM): skipping checkpoint arm")
        else:
            r["ckpt"] = measure(m, torch.from_numpy(real), int(meta["vocab"]), verbose,
                                f"checkpoint {os.path.basename(C.CKPT)}")
    return r


def main():
    print("H-P4 keyframe compression — interpolate between stored states, or recompute?\n")
    r = run()
    lin, hold = r["gaps"][4]
    verdict = ("interpolation has signal on the toy; the real decision needs the checkpoint arm"
               if lin < 0.5 * hold else "interpolation barely beats hold-last: recompute wins")
    print(f"PASS: {verdict}.")


if __name__ == "__main__":
    main()
