"""Order 0068, the coarse pass: the lab's accounting on a made-up head and made-up states. No model, no dump."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_spec = importlib.util.spec_from_file_location("head_lab_coarse",
                                               pathlib.Path(__file__).parents[2] / "validation" / "head_lab_coarse.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)

V, D, N = 300, 32, 200


def quiet(*_a):
    return None


def world(seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(V, D)).astype(np.float32)
    return w, {"hidden": rng.normal(size=(N, D)).astype(np.float16)}


def test_the_exact_rows_recall_everything_at_any_n():
    w, d = world()
    res = lab.read(w, d, ["exact"], [1, 16, V], chunk=64, log=quiet)["exact"]
    assert res["n"]["1"]["recall"] == 1.0 and res["rank_max"] == 0
    assert res["n"][str(V)]["mass_inside"] == 1.0 and res["n"]["1"]["mass_inside"] < res["n"]["16"]["mass_inside"] <= 1.0


def test_a_full_rank_projection_is_the_exact_head():
    w, d = world()
    res = lab.read(w, d, [f"pca{D}"], [1], chunk=64, log=quiet)[f"pca{D}"]
    assert res["n"]["1"]["recall"] == 1.0          # the mean row adds the same amount to every token's score


def test_recall_grows_with_n_and_a_coarser_head_recalls_less():
    w, d = world(1)
    res = lab.read(w, d, ["sign", "hamming", "pca4"], [1, 8, 64, V], chunk=64, log=quiet)
    for s in res.values():
        r = [s["n"][k]["recall"] for k in ("1", "8", "64", str(V))]
        assert r == sorted(r) and r[-1] == 1.0
    assert res["hamming"]["n"]["8"]["recall"] <= res["sign"]["n"]["8"]["recall"] + 0.1   # a bit per state loses more
    assert res["pca4"]["rows_eq_coarse"] == round(V * 4 / D, 1) and res["sign"]["rows_eq_coarse"] == round(V / 32, 1)


def test_the_cost_is_the_coarse_pass_plus_the_n_exact_rows():
    w, d = world()
    res = lab.read(w, d, ["sign_c"], [16], chunk=64, log=quiet)["sign_c"]
    assert res["n"]["16"]["rows_eq_total"] == round(V / 32 + 16, 1)
