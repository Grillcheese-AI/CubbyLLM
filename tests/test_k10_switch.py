"""Order 0021's switch-alone phase on 0006's toy (validation/exp_k10_switch.py): a resumed model is the saved one,
the switch-alone steps leave every frozen parameter bit-identical and move only the gate, tau and beta, present
answered wrong is 1 - exact - refused, and twelve templates widen training without moving the held-out set.
Tiny configuration, a few steps: the mechanics, not the numbers.
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

import exp_k10_free_text as ft  # noqa: E402
import exp_k10_hypertoken as kh  # noqa: E402
import exp_k10_switch as sw  # noqa: E402
import k10_free_docs as fd  # noqa: E402


def _args(tmp_path, arm="P", templates=6, steps=3):
    return SimpleNamespace(arm=arm, steps=steps, eval_every=1000, d=32, batch=2, held=24, seq=256, facts_per_doc=4, gap=120,
                           vocab=8192, pieces=5, absent=0.2, layers=2, at=1, window=64, window_ptr=16, pointer_loss=1.0,
                           gate_bias=0.0, lr=2e-3, seed=0, threads=0, tag="", date="t", templates=templates,
                           save=str(tmp_path / f"{arm}.pt"))


@pytest.fixture(scope="module")
def joint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("k10")
    a = _args(tmp)
    torch.manual_seed(0)
    rep = ft.run(a, lambda m: None)
    return a, rep


def test_a_resumed_model_is_the_saved_one(joint):
    a, rep = joint
    model, b, lay, held = sw.load(a.save)
    saved = torch.load(a.save, weights_only=False)["state"]
    assert all(torch.equal(t, saved[n]) for n, t in sw.tensors(model)) and len(saved) == len(sw.tensors(model))
    assert any(n.startswith("embedding.") for n in saved) and any(n.startswith("context_source.") for n in saved)
    ev = ft.evaluate(model, b, held, lay)
    for k in ("exact", "absent_not_none", "present_refused", "copy_alone", "pointer_acc"):
        assert ev[k] == rep["final"][k], k
    assert ev["language"] == pytest.approx(rep["final"]["language"], rel=1e-6)    # float32 sums, another layout


def test_the_switch_alone_moves_only_the_gate_tau_and_beta(joint):
    a, _rep = joint
    model, b, lay, held = sw.load(a.save)
    before = {n: p.detach().clone() for n, p in sw.named_params(model)}
    out = sw.copy_only(model, b, held, lay, steps=3, lr=1e-1, log=lambda m: None, eval_every=1000)
    assert out["frozen_hash"]["unchanged"]
    moved = {n for n, p in sw.named_params(model) if not torch.equal(p, before[n])}
    assert moved and moved <= set(sw.SWITCH)
    ev = out["final"]
    assert ev["present_wrong"] == pytest.approx(1.0 - ev["exact"] - ev["present_refused"], abs=1e-12)
    bm = sw.best_match(model, b, held, lay)
    assert {"present", "absent"} <= set(bm) and bm["present"]["best"]["p5"] <= bm["present"]["best"]["p95"]
    assert 0.5 <= bm["one_threshold"]["balanced_accuracy"] <= 1.0


def test_the_gate_loss_is_off_by_default_and_targets_the_asked_pieces(joint):
    a, _rep = joint
    m1, b, lay, held = sw.load(a.save)
    m2, *_ = sw.load(a.save)
    torch.manual_seed(1)
    r1 = sw.copy_only(m1, b, held, lay, steps=2, lr=1e-1, log=lambda m: None, eval_every=1000)
    torch.manual_seed(1)
    r2 = sw.copy_only(m2, b, held, lay, steps=2, lr=1e-1, log=lambda m: None, eval_every=1000, gate_loss=0.0)
    assert r1["final"]["exact"] == r2["final"]["exact"] and all(torch.equal(x, y) for (_n, x), (_k, y) in zip(sw.tensors(m1), sw.tensors(m2)))
    m3, *_ = sw.load(a.save)
    r3 = sw.copy_only(m3, b, held, lay, steps=2, lr=1e-1, log=lambda m: None, eval_every=1000, gate_loss=1.0)
    assert r3["frozen_hash"]["unchanged"]
    spans = [(10, [5, 6], {"absent": False}), (10, [9], {"absent": True})]
    S = 20
    assert sw.gate_targets(torch.tensor([10, 11, 12, 30, 31]), spans, S).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_twelve_templates_widen_training_and_leave_the_held_out_set(tmp_path):
    lay = fd.FreeLayout(8192)
    six, twelve = _args(tmp_path), _args(tmp_path, templates=12)
    t6 = {info["template"] for _h, _o, info in fd.batch(np.random.default_rng(1), kh.Language(np.random.default_rng(2)), lay, 300, six)[5]}
    t12 = {info["template"] for _h, _o, info in fd.batch(np.random.default_rng(1), kh.Language(np.random.default_rng(2)), lay, 300, twelve)[5]}
    assert t6 == set(fd.TRAIN_T) and t12 == set(fd.TRAIN_12) and not t12 & set(fd.HELD_T)
    held = []
    for a in (six, twelve):
        hr = np.random.default_rng(99)
        h = fd.batch(hr, kh.Language(np.random.default_rng(1)), lay, 16, a, fd.held_cells(hr, 16, a.absent))
        held.append(h)
    assert torch.equal(held[0][0], held[1][0])                       # the same held-out documents, token for token


def test_the_switch_alone_trains_in_the_joint_runs_language(joint):
    """The filler language's grammar comes from its seed: the switch's documents must be in the language the trunk
    learned (seed + 1), only new draws of it."""
    a, _rep = joint
    rng, lang = sw.stream(a)
    assert np.array_equal(lang.next, kh.Language(np.random.default_rng(a.seed + 1)).next)
    assert not np.array_equal(lang.next, kh.Language(np.random.default_rng(a.seed + 8)).next)
