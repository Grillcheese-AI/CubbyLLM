"""Order 0053 (``cubbyllm/selfimprove/gif.py``, ``fastlearn.py``), each claim exactly:
- the GIF port gives cubemind's recorded spike counts (a fixture, no cubemind import);
- with every weight zero the choice is the base's, bit for bit;
- the trace reaches only the chosen candidate's synapses from firing neurons;
- a right verdict raises the chosen bias, a wrong one lowers it, and with delta = 0 nothing moves;
- the bounds hold over a long random stream;
- a replay reproduces under its seed;
- the encoder's calibration reaches its rest rate."""
from __future__ import annotations

import json
import pathlib

import numpy as np

from cubbyllm.selfimprove.fastlearn import Encoder, FastLearner, Settings
from cubbyllm.selfimprove.gif import GIF, project

FIX = pathlib.Path(__file__).parent / "fixtures" / "gif_cubemind_spikes.json"
S = Settings(neurons=256, T=4, eta=0.5, tau=1.0, norm=2.0)


def test_the_gif_port_gives_cubeminds_recorded_spikes():
    for c in json.load(open(FIX))["cases"]:
        w = project(8, 16, c["seed"])
        assert np.array_equal(w, np.asarray(c["weight"], dtype=np.float32))
        x = np.asarray(c["input"], dtype=np.float32)
        spikes, (v, theta) = GIF(L=c["L"], tau=c["tau"], threshold=c["threshold"], alpha=c["alpha"]).run(x @ w.T)
        assert np.array_equal(spikes, np.asarray(c["spikes"], dtype=np.float32))
        assert np.allclose(v, c["v"]) and np.allclose(theta, c["theta"])


def _learner(rule="three", seed=0, dense=False):
    enc = Encoder(32, S, seed=seed, dense=dense)
    return FastLearner(enc, ["P1", "P2", "P3"], S, rule=rule, seed=seed), enc


def test_with_every_weight_zero_the_choice_is_the_bases():
    fl, enc = _learner()
    rng = np.random.default_rng(1)
    for _ in range(20):
        scores = rng.normal(size=5)
        i, _ = fl.choose(scores, enc(rng.normal(size=32)), [["P1"], ["P2"], ["P3"], [], ["P1", "P2"]], greedy=True)
        assert i == int(np.argmax(scores))


def test_the_trace_reaches_only_the_chosen_candidates_synapses_from_firing_neurons():
    fl, enc = _learner()
    act = np.zeros(S.neurons)
    act[[3, 7]] = [2.0, 1.0]
    fl.learn(act, "P2", "right", "k")
    moved = np.argwhere(fl.W != 0)
    assert set(map(tuple, moved)) == {(3, 1), (7, 1)}


def test_right_raises_wrong_lowers_and_a_zero_delta_moves_nothing():
    fl, enc = _learner()
    act = np.zeros(S.neurons)
    act[5] = 1.0
    fl.learn(act, "P1", "right", "k")                       # expected 0: delta +1
    up = fl.W[5, 0]
    assert up > 0
    fl2, _ = _learner()
    fl2.learn(act, "P1", "wrong", "k")                      # delta -5
    assert fl2.W[5, 0] < 0
    fl3, _ = _learner()
    fl3.learn(act, "P1", "refused", "k")                    # reward 0 = expected 0: delta 0
    assert not fl3.W.any()


def test_plain_stdp_strengthens_whatever_the_verdict():
    fl, _ = _learner("stdp")
    act = np.zeros(S.neurons)
    act[2] = 1.0
    fl.learn(act, "P3", "wrong", "k")
    assert fl.W[2, 2] > 0


def test_the_bounds_hold_over_a_long_random_stream():
    fl, enc = _learner()
    rng = np.random.default_rng(2)
    for i in range(400):
        act = enc(rng.normal(size=32))
        fl.learn(act, f"P{1 + i % 3}", rng.choice(["right", "wrong", "refused"]), f"k{i % 2}")
    assert np.linalg.norm(fl.W, axis=1).max() <= S.norm + 1e-9


def _day(seed):
    fl, enc = _learner(seed=seed)
    rng = np.random.default_rng(seed + 50)
    picks = []
    for i in range(60):
        act = enc(rng.normal(size=32))
        j, p = fl.choose(rng.normal(size=4), act, [["P1"], ["P2"], ["P3"], ["P1", "P3"]], greedy=False)
        fl.learn(act, p, ["right", "wrong"][j % 2], "k")
        picks.append(j)
    return picks, fl.W.copy()


def test_a_replay_reproduces_under_its_seed():
    a, b = _day(3), _day(3)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
    assert _day(4)[0] != a[0]


def test_the_encoder_calibrates_to_its_rest_rate_and_the_dense_control_passes_values():
    enc = Encoder(32, S, seed=0)
    H = np.random.default_rng(5).normal(size=(64, 32))
    share = enc.calibrate(H)
    assert abs(share - S.rest) < 0.01
    dense = Encoder(32, S, seed=0, dense=True)
    assert np.allclose(dense(H[0]), dense.currents(H[0])[0])
