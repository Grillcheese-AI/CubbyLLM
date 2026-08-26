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
