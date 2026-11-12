"""Order 0068, route 3 (the two-level head): the lab's accounting on a made-up head and made-up states.
No model, no dump."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_spec = importlib.util.spec_from_file_location("head_lab_twolevel",
                                               pathlib.Path(__file__).parents[2] / "validation" / "head_lab_twolevel.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)

V, D, N, F, C = 120, 12, 240, 30, 6


def quiet(*_a):
    return None


def world(seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(V, D)).astype(np.float32)
    h = rng.normal(size=(N, D)).astype(np.float16)
    full = h.astype(np.float32) @ w.T
    top = np.argsort(-full, axis=1)[:, :16]
    lse = lab.lse(full.astype(np.float64))
    tgt = top[np.arange(N), rng.integers(0, 16, size=N)]
    d = {"hidden": h, "target": tgt.astype(np.int32), "top": top.astype(np.int32),
         "toplogit": np.take_along_axis(full, top, axis=1), "lse": lse.astype(np.float32),
         "tlogit": full[np.arange(N), tgt]}
    prob = np.exp(d["toplogit"] - lse[:, None])
    lay = lab.build_layout(w, top, prob, F, C, seed=0, iters=5)
    return w, d, full, lay


def test_the_layout_is_one_permutation_of_the_rows():
    w, _d, _full, lay = world()
    assert sorted(lay.perm.tolist()) == list(range(V)) and lay.f == F and lay.c == C
    assert (lay.cluster_of[lay.head] == -1).all() and (lay.cluster_of[lay.tail] >= 0).all()
    assert lay.sizes.sum() == V - F and lay.sizes.min() >= 1


def test_prep_gives_each_cluster_its_true_log_sum_exp():
    w, d, full, lay = world()
    pre = lab.prep(w[lay.perm], lay, d["hidden"], chunk=64, log=quiet)
    for c in range(C):
        ids = lay.tail[lay.starts[c]:lay.starts[c + 1]]
        assert np.allclose(pre["t"][:, c], lab.lse(full[:, ids].astype(np.float64)), atol=1e-4)
    assert np.allclose(pre["lse_h"], lab.lse(full[:, lay.head].astype(np.float64)), atol=1e-4)
    assert np.allclose(pre["max_h"], full[:, lay.head].max(1), atol=1e-5)
    whole = np.logaddexp(pre["lse_h"].astype(np.float64), lab.lse(pre["t"].astype(np.float64)))
    assert np.allclose(whole, lab.lse(full.astype(np.float64)), atol=1e-4)


def test_exact_entries_are_the_full_head():
    w, d, _full, lay = world()
    pre = lab.prep(w[lay.perm], lay, d["hidden"], chunk=64, log=quiet)
    res = lab.evaluate(d, pre, pre["t"], lay, margins=(0.0,))
    assert abs(res["two_level"]["nll_gap"]) < 1e-6 and res["two_level"]["agree"] == 1.0 and res["two_level"]["unsafe"] == 0
    assert res["router"]["0.0"]["agree"] == 1.0 and res["router"]["0.0"]["mass_inside"] <= 1.0 + 1e-9
    assert res["two_level"]["rows_mean"] >= F + C and res["lse_check_vs_dump"] < 1e-3


def test_entries_far_too_low_open_nothing():
    w, d, _full, lay = world()
    pre = lab.prep(w[lay.perm], lay, d["hidden"], chunk=64, log=quiet)
    res = lab.evaluate(d, pre, pre["t"] - 50.0, lay, margins=(0.0,))
    assert res["two_level"]["rows_mean"] == F + C and res["router"]["0.0"]["rows_mean"] == F + C
    assert res["router"]["0.0"]["agree"] == res["best_in_frequent_set"] < 1.0
    assert res["two_level"]["nll_gap"] > 0.0


def test_a_wider_margin_opens_more_and_agrees_at_least_as_often():
    w, d, _full, lay = world(1)
    pre = lab.prep(w[lay.perm], lay, d["hidden"], chunk=64, log=quiet)
    e0, b0 = lab.init_entries(w[lay.perm], lay)
    res = lab.evaluate(d, pre, d["hidden"].astype(np.float32) @ e0.T + b0, lay, margins=(0.0, 2.0, 8.0))["router"]
    assert res["0.0"]["rows_mean"] <= res["2.0"]["rows_mean"] <= res["8.0"]["rows_mean"]
    assert res["0.0"]["agree"] <= res["2.0"]["agree"] <= res["8.0"]["agree"]


def test_the_fit_lowers_the_first_pass_kl():
    w, d, _full, lay = world(2)
    wp = w[lay.perm]
    pre = lab.prep(wp, lay, d["hidden"], chunk=64, log=quiet)
    e0, b0 = lab.init_entries(wp, lay)
    hid = d["hidden"].astype(np.float32)
    before = lab.first_pass_kl(hid, pre, e0, b0)
    e, b = lab.train(hid, pre, e0, b0, epochs=40, lr=2e-2, batch=64, threads=1, log=quiet)
    after = lab.first_pass_kl(hid, pre, e, b)
    assert 0.0 <= after < before
