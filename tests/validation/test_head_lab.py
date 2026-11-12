"""Order 0068, route 2 (the shortlist with a fallback): the lab's accounting, on a made-up head and made-up states.
No model, no dump: a toy shortlist that keeps the first token ids."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_spec = importlib.util.spec_from_file_location("head_lab", pathlib.Path(__file__).parents[2] / "validation" / "head_lab.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)

V, D, N = 60, 16, 400


class Toy:
    """A shortlist that is the first `keep` token ids, whatever the state."""
    screen = None
    n = 1

    def __init__(self, keep: int) -> None:
        self.keep = keep

    def shortlist(self, h, k, arm, prev):
        return [np.arange(min(self.keep, k))] * len(h)


def world(seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(V, D)).astype(np.float32)
    h = rng.normal(size=(N, D)).astype(np.float16)
    full = h.astype(np.float32) @ w.T
    return w, {"hidden": h, "target": full.argmax(1).astype(np.int32), "prev": np.zeros(N, dtype=np.int32)}, full


def quiet(*_a):
    return None


def test_the_whole_vocabulary_is_the_full_head():
    w, d, _ = world()
    sc = lab.scan(Toy(V), w, d, V, "toy", log=quiet)
    assert sc["agree"].all() and sc["inside"].all()
    assert np.allclose(sc["mass"], 1.0, atol=1e-5) and np.allclose(sc["nll_short"], sc["nll_full"], atol=1e-4)


def test_half_the_vocabulary_misses_what_is_outside_it():
    w, d, full = world()
    sc = lab.scan(Toy(V // 2), w, d, V, "toy", log=quiet)
    outside = full.argmax(1) >= V // 2
    assert (sc["agree"] == ~outside).all() and (sc["inside"] == ~outside).all()
    assert np.allclose(sc["nll_short"][outside], -np.log(lab.FLOOR))
    assert (sc["rows"] == V // 2).all() and (sc["mass"][outside] < 1.0).all()


def test_a_threshold_hands_the_riskiest_positions_to_the_full_head():
    w, d, _ = world()
    sc = lab.scan(Toy(V // 2), w, d, V, "toy", log=quiet)
    r = lab.risk("p1", sc)
    thr, rate = lab.pick_threshold(r, sc["agree"], 1.0)
    res = lab.read(sc, r, thr, V, index_rows=4)
    assert res["agree_overall"] == 1.0 and rate >= float((~sc["agree"]).mean())
    assert abs(res["rows_mean"] - (4 + V // 2 + res["fallback_rate"] * V)) < 0.11      # rounded to one decimal
    none = lab.read(sc, r, float("inf"), V, index_rows=4)
    assert none["fallback_rate"] == 0.0 and none["agree_overall"] == none["agree_shortlist_alone"]
    assert none["rows_mean"] == 4 + V // 2


def test_no_fallback_is_asked_for_when_the_shortlist_already_agrees():
    w, d, _ = world()
    sc = lab.scan(Toy(V), w, d, V, "toy", log=quiet)
    assert lab.pick_threshold(lab.risk("margin", sc), sc["agree"], 0.995) == (float("inf"), 0.0)


def test_a_looser_target_never_needs_more_fallback():
    w, d, _ = world(1)
    sc = lab.scan(Toy(V // 3), w, d, V, "toy", log=quiet)
    r = lab.risk("margin", sc)
    rates = [lab.pick_threshold(r, sc["agree"], t)[1] for t in (0.9, 0.95, 0.99, 1.0)]
    assert rates == sorted(rates)


def test_tuning_freezes_one_signal_and_reads_back_at_its_target():
    w, d, _ = world(2)
    sc = lab.scan(Toy(V // 2), w, d, V, "toy", log=quiet)
    frozen = lab.tune(sc, 0.99)
    assert frozen["signal"] in lab.SIGNALS and set(frozen["table"]) == set(lab.SIGNALS)
    odd = {k: v[1::2] for k, v in sc.items()}
    thr = float("inf") if frozen["threshold"] is None else frozen["threshold"]
    res = lab.read(odd, lab.risk(frozen["signal"], odd, frozen["model"]), thr, V, index_rows=0)
    assert res["agree_overall"] >= 0.99
