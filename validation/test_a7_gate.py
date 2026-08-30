"""Unit pins for the H-A7 promotion gate's pure decision logic
(exp_a7_learning_gate.py) — no checkpoint, no corpus. Run:
python -m pytest validation/test_a7_gate.py -q
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

VAL = pathlib.Path(__file__).resolve().parent
for p in (str(VAL.parent), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp_a7_learning_gate as g  # noqa: E402

SRC = ["books", "wiki", "code"]
N = 16


def windows(mean, jitter=0.0, seed=0, n=N):
    rng = np.random.default_rng(seed)
    return list(mean + jitter * rng.standard_normal(n))


def ev(mix=3.0, src=(3.0, 3.0, 3.0), gsm=2.5, ret=0.8, jitter=0.0, seed=0):
    """An evaluate() dict whose windows sit at the given means (+ optional jitter)."""
    return {"ce_mix": mix, "ce_mix_windows": windows(mix, jitter, seed),
            "ce_by_source": dict(zip(SRC, src)),
            "ce_src_windows": {s: windows(m, jitter, seed + 1 + i) for i, (s, m) in enumerate(zip(SRC, src))},
            "ce_gsm8k": gsm, "ce_gsm_windows": windows(gsm, jitter, seed + 9),
            "retrieval_acc": ret}


def shifted(base, mix=0.0, src=(0.0, 0.0, 0.0), gsm=0.0, ret=None, noise=0.0, seed=5):
    """The SAME windows as `base`, shifted per metric (+ optional per-window noise):
    the paired comparison the real gate makes."""
    rng = np.random.default_rng(seed)
    out = {"ce_mix_windows": [w + mix + noise * rng.standard_normal() for w in base["ce_mix_windows"]],
           "ce_src_windows": {s: [w + src[i] + noise * rng.standard_normal() for w in base["ce_src_windows"][s]]
                              for i, s in enumerate(SRC)},
           "ce_gsm_windows": [w + gsm + noise * rng.standard_normal() for w in base["ce_gsm_windows"]],
           "retrieval_acc": base["retrieval_acc"] if ret is None else ret}
    out["ce_mix"] = float(np.mean(out["ce_mix_windows"]))
    out["ce_by_source"] = {s: float(np.mean(out["ce_src_windows"][s])) for s in SRC}
    out["ce_gsm8k"] = float(np.mean(out["ce_gsm_windows"]))
    return out


def eps_for(base):
    return g.noise_floor(base, base)


def test_paired_term_threshold_is_max_of_min_eps_and_three_se():
    t = g.paired_term([1.0] * 8, [0.0] * 8, min_eps=0.01)
    assert t["se"] == 0.0 and t["threshold"] == 0.01 and t["d"] == 1.0 and t["fired"]
    t = g.paired_term([0.0, 0.2, 0.0, 0.2], [0.0] * 4, min_eps=0.01)   # d=0.1, sd=0.1155, se=0.0577
    assert abs(t["se"] - 0.05774) < 1e-4 and abs(t["threshold"] - 0.1732) < 1e-3 and not t["fired"]


def test_paired_comparison_ignores_shared_window_variance():
    """Wildly different windows (jitter 1.0 nat) but a small uniform shift:
    unpaired noise would swamp it; paired SE is ~0, so the shift is detected."""
    base = ev(jitter=1.0)
    forgot = shifted(base, src=(0.08, 0.0, 0.0))     # books +0.08 on every window
    dec = g.gate_decision(forgot, base, eps_for(base))
    assert dec["fired"] == ["forgetting_source:books"]
    assert dec["margins"]["worst_source"] == "books"
    assert eps_for(base)["unpaired_raw"]["mix"] == 0.0   # same eval twice -> no unpaired noise either


def test_noisy_small_shift_does_not_fire():
    base = ev(jitter=0.5)
    d = shifted(base, mix=0.02, noise=0.3)            # +0.02 buried in 0.3-nat per-window noise
    dec = g.gate_decision(d, base, eps_for(base))
    assert "general_capability_mix" not in dec["fired"]
    assert dec["margins"]["eps_mix"] > 0.1             # 3 x SE with sd 0.3, n=16 -> ~0.22


def test_null_delta_always_promotes():
    base = ev(jitter=0.7)
    dec = g.gate_decision(base, base, eps_for(base))
    assert dec["promote"] is True and dec["fired"] == [] and dec["margins"]["d_ce_mix"] == 0.0


def test_each_term_fires_independently():
    base = ev()
    eps = eps_for(base)
    assert g.gate_decision(shifted(base, mix=0.2), base, eps)["fired"] == ["general_capability_mix"]
    assert g.gate_decision(shifted(base, gsm=0.1), base, eps)["fired"] == ["general_capability_gsm8k"]
    assert g.gate_decision(shifted(base, ret=0.6), base, eps)["fired"] == ["binding_health"]


def test_retrieval_floor_is_relative_to_base_with_collapse_floor():
    base = ev(ret=0.80)
    eps = g.noise_floor(base, ev(ret=0.76))            # seed-to-seed 0.04 -> eps 0.12 (3x, above min 0.05)
    assert abs(eps["ret"] - 0.12) < 1e-9
    dec = g.gate_decision(shifted(base, ret=0.70), base, eps)   # 0.70 >= 0.80-0.12=0.68 -> ok
    assert "binding_health" not in dec["fired"]
    dec = g.gate_decision(shifted(base, ret=0.66), base, eps)
    assert "binding_health" in dec["fired"]
    # a high-retrieval base still enforces the absolute collapse floor
    hi = ev(ret=0.99)
    eps_hi = g.noise_floor(hi, ev(ret=0.60))           # huge seed noise -> eps 1.17, floor = max(0.5, -0.18)
    assert g.gate_decision(shifted(hi, ret=0.45), hi, eps_hi)["margins"]["retrieval_floor"] == 0.5
    assert "binding_health" in g.gate_decision(shifted(hi, ret=0.45), hi, eps_hi)["fired"]


def test_improvement_never_fires():
    base = ev(jitter=0.3)
    better = shifted(base, mix=-0.2, src=(-0.1, -0.2, -0.3), gsm=-0.2, ret=0.9)
    assert g.gate_decision(better, base, eps_for(base))["promote"] is True


def test_separation_verdict_requires_all_three():
    P = {"promote": True}
    R = {"promote": False}
    assert g.separation_verdict({"A": R, "B": P, "C": P})["gate_separates"] is True
    assert g.separation_verdict({"A": P, "B": P, "C": P})["gate_separates"] is False   # let forgetting through
    assert g.separation_verdict({"A": R, "B": R, "C": P})["gate_separates"] is False   # blocked the clean one
    assert g.separation_verdict({"A": R, "B": P, "C": R})["gate_separates"] is False   # blocked the null


def test_not_applicable_terms_are_declared():
    base = ev()
    na = " ".join(g.gate_decision(base, base, eps_for(base))["not_applicable"])
    assert "claimed_answer_precision" in na and "routing_precision" in na and "nyt_forgetting" in na
