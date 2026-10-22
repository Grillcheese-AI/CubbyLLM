"""Training the copy alone on a frozen trunk (``CB_TRAIN_ONLY=copy``, order 0005) and the probe's
split (``validation/recall_split.py``), each claim exactly.

Pinned: with the flag, one optimizer step changes only the copy's tensors (every other tensor bit
for bit), the loss's graph reaches only the copy's tensors (the trunk keeps nothing for a backward),
the checkpoint still holds every tensor and loads back (``CubbyModel.parameters()`` yields only what
requires grad -- the first CPU smoke stopped there), and after training steps every position that
does not ask has the starting model's logits exactly;
without the flag every tensor trains, in the optimizer groups as before; an unknown part, or no copy
path, refuses. The probe: its first token and exact are ``exp_hypertoken_recall.score``'s; its piece
counts sum to the answers' pieces; its lam counts sum to the asking tokens the copy acted on, split by
whether the asked address holds a write at or before the asker; ``train_base.fact_probe`` carries it.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest
import torch

from cubbyllm.model.spatial import NO_PLACE
from test_grow import META
from test_spatial_copy import _linked, _roles

CPU = torch.device("cpu")
VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "validation")


def _tb():
    sys.path.insert(0, VALIDATION)
    import train_base
    return train_base


def _rs():
    """``recall_split`` without leaking its import's ``CB_NO_FLEX=1`` default into later tests."""
    had = os.environ.get("CB_NO_FLEX")
    sys.path.insert(0, VALIDATION)
    import recall_split
    if had is None:
        os.environ.pop("CB_NO_FLEX", None)
    return recall_split


def _batch(tb, seed=4):
    x, pos, placed, writes, asks, starts = _roles()
    flags = writes.to(torch.uint8) * tb.WRITE | asks.to(torch.uint8) * tb.ASK | starts.to(torch.uint8) * tb.START
    y = torch.randint(0, META["vocab"], x.shape, generator=torch.Generator().manual_seed(seed))
    return tb.Tokens(x, pos.long(), flags), y, asks


def _step(tb, model, opt, loss_fn, x, y):
    loss = loss_fn(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tb.CLIP)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss


def _leaves(t: torch.Tensor) -> set:
    """The tensors ``t``'s backward would write a gradient into."""
    out, seen, todo = set(), set(), [t.grad_fn]
    while todo:
        f = todo.pop()
        if f is None or f in seen:
            continue
        seen.add(f)
        if hasattr(f, "variable"):
            out.add(id(f.variable))
        todo.extend(g for g, _ in f.next_functions)
    return out


def test_with_the_flag_one_step_changes_only_the_copys_tensors(monkeypatch):
    tb = _tb()
    monkeypatch.setattr(tb, "TRAIN_ONLY", "copy")
    m = _linked("ring")
    every = list(m.parameters())                                           # nothing frozen yet: the whole model
    own = {id(p) for p in m.spatial.copy_path.parameters()}
    peak = tb.train_only(m)
    assert peak == tb.COPY_LR and [id(p) for p in tb.params_of(m)] == [id(p) for p in every]
    assert {id(p) for p in every if p.requires_grad} == own
    opt = tb.make_opt(m, peak, CPU)
    assert {id(p) for g in opt.param_groups for p in g["params"]} == own
    loss_fn = tb.make_fns(m, META["vocab"], CPU)[0]
    x, y, _ = _batch(tb)
    assert _leaves(loss_fn(x, y)) == own                                   # the trunk builds no graph: nothing kept for it
    before = {id(p): p.detach().clone() for p in every}
    _step(tb, m, opt, loss_fn, x, y)
    for p in every:
        assert torch.equal(p, before[id(p)]) != (id(p) in own), "only the copy's tensors move"
    assert len(every) > 2 * len(own)


def test_with_the_flag_the_checkpoint_still_holds_the_whole_model(tmp_path, monkeypatch):
    """``CubbyModel.parameters()`` yields only what requires grad: the checkpoint's list must not
    shrink to the copy's when the trunk is frozen, and it loads back into a frozen model."""
    tb = _tb()
    monkeypatch.setattr(tb, "TRAIN_ONLY", "copy")
    m = _linked("ring")
    every = [p.detach().clone() for p in m.parameters()]
    opt = tb.make_opt(m, tb.train_only(m), CPU)
    ck = tb.payload(m, opt, 5, META, {}, {})
    assert len(ck["params"]) == len(every) and all(torch.equal(a, b) for a, b in zip(ck["params"], every))
    torch.save(ck, tmp_path / "frozen.pt")
    m2 = _linked("ring", seed=4)
    tb.train_only(m2)
    tb.restore(str(tmp_path / "frozen.pt"), m2, tb.make_opt(m2, tb.COPY_LR, CPU), CPU, META)
    assert all(torch.equal(a, b) for a, b in zip(tb.params_of(m2), every))


def test_with_the_flag_the_positions_that_do_not_ask_keep_the_starting_logits(monkeypatch):
    tb = _tb()
    monkeypatch.setattr(tb, "TRAIN_ONLY", "copy")
    m = _linked("ring")
    start = copy.deepcopy(m)
    opt = tb.make_opt(m, tb.train_only(m), CPU)
    loss_fn = tb.make_fns(m, META["vocab"], CPU)[0]
    x, y, asks = _batch(tb)
    for _ in range(3):
        _step(tb, m, opt, loss_fn, x, y)
    with torch.no_grad():
        a, b = tb.forward_batch(m, x), tb.forward_batch(start, x)
        assert torch.equal(a[~asks], b[~asks]) and torch.equal(m.forward(x.ids), start.forward(x.ids))
        assert not torch.equal(a[asks], b[asks])                           # the copy did train


def test_without_the_flag_every_tensor_trains_as_before(monkeypatch):
    tb = _tb()
    monkeypatch.setattr(tb, "TRAIN_ONLY", "")
    m = _linked("ring")
    ps = list(m.parameters())
    assert tb.train_only(m) == tb.LR and all(p.requires_grad for p in ps)
    opt = tb.make_opt(m, tb.LR, CPU)
    assert [[id(p) for p in g["params"]] for g in opt.param_groups] == [
        [id(p) for p in ps if p.dim() >= 2], [id(p) for p in ps if p.dim() < 2]]
    assert [g["weight_decay"] for g in opt.param_groups] == [tb.WD, 0.0]
    x, y, _ = _batch(tb)
    before = [p.detach().clone() for p in ps]
    _step(tb, m, opt, tb.make_fns(m, META["vocab"], CPU)[0], x, y)
    n_copy = len(list(m.spatial.copy_path.parameters()))
    assert any(not torch.equal(a, p) for a, p in zip(before[:-n_copy], ps[:-n_copy])), "the trunk trains"


@pytest.mark.parametrize("part, copy_mode", [("copy", None), ("trunk", "ring")])
def test_an_unknown_part_or_no_copy_path_refuses(monkeypatch, part, copy_mode):
    tb = _tb()
    monkeypatch.setattr(tb, "TRAIN_ONLY", part)
    with pytest.raises(SystemExit):
        tb.train_only(_linked(copy_mode))


# -- the probe's split -------------------------------------------------------------------------
ANSWER, NL, SPACE, DOC, N_DOCS = 3, 4, 5, 24, 10                       # two batches of hr.BATCH = 8


class _Tok:
    """The two strings ``exp_hypertoken_recall.answers`` looks for, and a whitespace token."""

    def encode(self, s, add_special_tokens=False):
        return type("E", (), {"ids": {"Answer:": [ANSWER], "\n": [NL]}[s]})()

    def decode(self, ids):
        return " " if ids == [SPACE] else "w"


def _docs(seed=0):
    """Per document: writers at 1-6 and 20; askers whose address was written before them (10, 11,
    12, 15, 16, 21, 22), written only later (7), never written (13, 17), and one with no writer yet
    (0: the copy does not act); answers at 11-13, 16-17 and 22-23 (the last runs to the end: no newline)."""
    g = np.random.default_rng(seed)
    x = g.integers(10, META["vocab"], (N_DOCS, DOC))
    P = np.full((N_DOCS, DOC, 3), NO_PLACE, dtype=np.int64)
    R = np.zeros((N_DOCS, DOC), dtype=np.int64)
    x[:, [10, 15, 21]], x[:, [14, 18]] = ANSWER, NL
    for r in range(N_DOCS):
        W = [(i, r, 1) for i in range(6)]
        P[r, 1:7], R[r, 1:7] = W, 1
        P[r, 20], R[r, 20] = (7, r, 2), 1
        for t, p in {0: W[1], 7: (7, r, 2), 10: W[0], 11: W[1], 12: W[2], 13: (9, 9, 9), 15: W[3], 16: W[4],
                     17: (9, 9, 8), 21: W[5], 22: W[0]}.items():
            P[r, t], R[r, t] = p, 2
    return x, (P, R)


def test_the_probe_split_sums_to_its_totals_and_its_exact_is_the_recall_probes(monkeypatch):
    rs, m = _rs(), _linked("ring")
    monkeypatch.setattr(rs.hr, "STRIP", False)
    x, places = _docs()
    out = rs.score(m, _Tok(), x, places, CPU)
    ref = rs.hr.score(m, _Tok(), x, places, True, CPU)
    assert {k: out[k] for k in ref} == ref                                 # first token, exact, questions: the same
    assert out["questions"] == 3 * N_DOCS and out["n_later"] == 4 * N_DOCS and out["n_end"] == 2 * N_DOCS
    assert out["pieces"] == out["questions"] + out["n_later"] == sum(k - j + 1 for r in x for j, k in rs.hr.answers(_Tok(), r))
    assert out["n_present"] == 7 * N_DOCS and out["n_empty"] == 3 * N_DOCS
    assert out["asks"] == out["n_present"] + out["n_empty"] and 0 < out["lam_empty"] < 1 and 0 < out["lam_present"] < 1


def test_fact_probe_carries_the_split_and_the_mean_lam(tmp_path, monkeypatch):
    tb, rs, m = _tb(), _rs(), _linked("ring")
    x, (P, R) = _docs(1)
    x.astype(np.uint32).tofile(tmp_path / "toy.u32")
    P.astype(np.int16).tofile(tmp_path / "toy.pos.i16")
    R.astype(np.uint8).tofile(tmp_path / "toy.role.u8")
    for f, v in (("_align.json", DOC), ("_holdout.json", N_DOCS * DOC)):
        (tmp_path / f).write_text(json.dumps({"toy": v}))
    monkeypatch.setattr(rs.hr, "CORPUS", rs.hr.CORPUS)                     # fact_probe points it at the cache
    for k, v in (("CORPUS", str(tmp_path)), ("PROBE_SHARDS", "toy"), ("PROBE_DOCS", N_DOCS)):
        monkeypatch.setattr(tb, k, v)
    monkeypatch.setitem(tb.fact_probe.__dict__, "tk", _Tok())
    flex = os.environ.get("CB_NO_FLEX")
    pr = tb.fact_probe(m, CPU)
    assert os.environ.get("CB_NO_FLEX") == flex                            # the probe leaves the flex switch as it was
    assert pr["toy"] == rs.score(m, _Tok(), x, (P, R), CPU)
    v = pr["toy"]
    assert abs(pr["_lam"] - (v["lam_present"] * v["n_present"] + v["lam_empty"] * v["n_empty"]) / v["asks"]) < 1e-12
    assert "lam here" in rs.line(pr) and "end" in rs.line(pr)
