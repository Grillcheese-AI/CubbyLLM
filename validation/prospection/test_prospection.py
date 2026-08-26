"""Smoke suite for the Group P (prospection) experiments — every script's
``run(quick=True)`` in one pytest session, sharing one toy training run.

    python -m pytest validation/prospection -q

This is deliberately OUTSIDE ``tests/`` (which mirrors ``cubbyllm/`` one-to-one):
these are hypothesis experiments, not package tests. The full-strength numbers
come from running each ``exp_p*.py`` directly; ``quick=True`` trains for fewer
steps and uses looser floors so the whole file runs in well under a minute on
CPU and catches a broken helper, a changed gate formula, or a drifted decode
path before anyone reads a number.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common as C  # noqa: E402


@pytest.fixture(scope="module")
def toy():
    """One trained toy model for the whole module (steps must match the scripts'
    quick setting so the per-process cache is hit, not repopulated)."""
    return C.trained_toy(steps=150)


def test_branch_grammar_has_exactly_one_choice_per_segment():
    import numpy as np
    g = C.BranchGrammar()
    toks, ks = g.sample(50, np.random.default_rng(0))
    m = g.choice_mask(toks)
    assert m.sum() == 50 and toks[np.where(m)[0] + 1].min() >= 1
    firsts = {int(t[g.shared]) for t in g.templates}
    assert len(firsts) == g.K                      # fragile token distinct per branch
    assert all((t[:g.shared] == g.templates[0][:g.shared]).all() for t in g.templates)


def test_fork_equivalence_all_backbones():
    r = importlib.import_module("exp_p1_fork_equivalence").run(quick=True, verbose=False)
    assert set(r) >= {"mingru", "hybrid", "hybrid+mem"}
    assert all(v["max_diff"] < 1e-4 for v in r.values())


def test_entropy_localizes_choice_points(toy):
    r = importlib.import_module("exp_p2_choice_points").run(quick=True, verbose=False)
    assert r["auroc_step"] > 0.85 and r["h_choice"] > r["h_other"]


def test_gate_spectrum_facts(toy):
    r = importlib.import_module("exp_p3_gate_spectrum").run(quick=True, verbose=False)
    assert r["pin"] < 1e-5
    assert abs(r["cap"] - C.TAU_CAP) / C.TAU_CAP < 1e-6
    assert all(s["slow100"] >= 0.2 for s in r["chrono"].values())


def test_keyframe_operators_and_recompute(toy):
    r = importlib.import_module("exp_p4_keyframe_interp").run(quick=True, verbose=False)
    assert r["pins"]["hilbert_vs_linear"] < 1e-9 and r["pins"]["fourier_vs_linear"] < 1e-9
    assert r["pins"]["ham_endpoint_rel_err"] > 0.3
    assert r["recompute_err"] < 1e-4


def test_counterfactual_reentry(toy):
    r = importlib.import_module("exp_p5_counterfactual_probe").run(quick=True, verbose=False)
    assert r["acc_counterfactual"] > 0.8 and r["overdetermined_agree"] > 0.8
    assert r["fragile_flip"] > 0.8


def test_multihorizon_targets_match_naive_recursion():
    """H-P6's target = a reversed log-domain scan; pin it against the naive
    O(S^2) definition y_t = sum_k gamma^(k-1) phi_{t+k} / truncated mass."""
    mh = importlib.import_module("exp_p6_multihorizon_pilot")
    g = torch.Generator().manual_seed(0)
    phi = torch.nn.functional.normalize(torch.randn(2, 13, 5, generator=g), dim=-1)
    B, S, d = phi.shape
    for gamma in (0.0, 0.5, 0.9, 0.98):
        got = mh.MultiHorizonHead.targets(phi, gamma)
        want = torch.zeros(B, S - 1, d)
        for t in range(S - 1):
            num, mass = torch.zeros(B, d), 0.0
            for k in range(1, S - t):
                num += (gamma ** (k - 1)) * phi[:, t + k]
                mass += gamma ** (k - 1)
            want[:, t] = num / mass
        assert torch.allclose(got, want, atol=1e-4), f"gamma={gamma}: max err {(got-want).abs().max()}"
    assert got.shape == (B, S - 1, d)


def test_chrono_wrapper_and_impulse_response_discriminate():
    """H-P7 mechanics: the wrapper turns chrono on; chrono init sets a slow
    spectrum the runner can read back; and the impulse-response probe reads a
    substituted token's perturbation of the recurrent state at long lags — non-zero
    for a chrono-initialised model, ~0 past the window for the default init."""
    os.environ["CB_CHRONO"] = "1"
    import numpy as np
    mh = importlib.import_module("exp_p6_multihorizon_pilot")
    p7 = importlib.import_module("exp_p7_chrono_pilot")
    assert p7.runner is mh and mh.CHRONO_TMAX < C.TAU_CAP
    g = C.BranchGrammar()
    toks, _ = g.sample(64, np.random.default_rng(3))                 # 512 tokens
    x = torch.from_numpy(toks[:512]).view(2, 256)
    dev = torch.device("cpu")

    def one_batch():
        while True:
            yield x, x

    plain = C.build_model(V=g.V, d=32, L=3, backbone="hybrid", window=16)
    chrono = C.build_model(V=g.V, d=32, L=3, backbone="hybrid", window=16)
    for _, mx in C.mingru_mixers(chrono.backbone):
        C.chrono_init_(mx, 1.0, 500.0, weight_scale=1.0)
    lags = [1, 8, 32, 64, 128]
    ir_plain = mh.impulse_response(plain, one_batch(), dev, lags, n_seqs=2)
    ir_chrono = mh.impulse_response(chrono, one_batch(), dev, lags, n_seqs=2)
    assert ir_plain["1"] > 0 and ir_chrono["1"] > 0
    assert ir_plain["128"] < 1e-3, ir_plain                           # default init forgets past the window
    assert ir_chrono["128"] > 10 * max(ir_plain["128"], 1e-6), (ir_plain, ir_chrono)
    spec = mh.gate_spectrum(chrono, one_batch(), dev)
    assert all(v["slow100"] > 0.1 for v in spec.values())
