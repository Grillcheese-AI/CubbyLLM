"""Copy or refuse (cubbyllm/model/spatial_refuse.py, order 0022), each claim pinned.

Pinned: off (no ``copy_refuse``), the copy path is today's mix bit for bit; the decision -- every piece clears theta
and an end reads below theta_end: copied; a piece between them, no piece, or no end: refused; a run's positions get
the run's decision (``runs``); a whole answer copied takes its pieces from the copy alone and leaves the end to the
trunk; a piece under theta refuses the whole answer and no fact token is emitted anywhere in it; an asker at an empty
address is refused; the positions that do not ask keep the model's logits; decode = forward in both cases; the
pointer's query rebound to piece j + k is the text key of piece j + k.
"""
from __future__ import annotations

import pytest
import torch

from cubbyllm.model import spatial_refuse as sr
from cubbyllm.model.spatial import SpatialLink
from test_grow import META, _build

D, S, STEP, RID = 64, 14, (0.0, 0.0, 6.0), 39
M64 = {**META, "D": D}
P0 = torch.tensor([2.0, -3.0, 0.0])


def _model(refuse=None, seed=0):
    m = _build(M64, seed=seed)
    torch.manual_seed(seed + 100)
    with torch.no_grad():
        for p in list(m.parameters()) + list(m.context_source.parameters()):
            p.copy_(torch.randn_like(p) * 0.3)
    m.spatial = SpatialLink(D, scale=1.0, seed=5, at=M64["L"] // 2, query="placed", bind="writes", copy="ring",
                            copy_refuse=refuse)
    return m


def _batch(ask_at=P0):
    """Three pieces written at P0 + k STEP (positions 0-2), filler, then an answer run asking P0 + k STEP, k = 0..3
    (positions 8-11; the last asks an empty address)."""
    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, 30, (1, S), generator=g)
    x[0, :3] = torch.tensor([31, 32, 33])
    pos = torch.zeros(1, S, 3)
    writes, asks = torch.zeros(1, S, dtype=torch.bool), torch.zeros(1, S, dtype=torch.bool)
    for k in range(3):
        pos[0, k], writes[0, k] = P0 + k * torch.tensor(STEP), True
    for k in range(4):
        pos[0, 8 + k], asks[0, 8 + k] = ask_at + k * torch.tensor(STEP), True
    return x, pos, writes | asks, writes, asks, torch.zeros(1, S, dtype=torch.bool)


def _bests(m, batch):
    """The reads ahead from the run's first position (what `decide` sees), captured with a permissive contract."""
    got = []
    real = sr.decide
    sr.decide = lambda b, t, e: (got.append(b.clone()), real(b, t, e))[1]
    try:
        m.spatial.copy_path.set_refuse(0.0, 0.0, RID, STEP, 6)
        with torch.no_grad():
            m.forward(*batch)
    finally:
        sr.decide = real
    return got[0][0]


def _decode(m, x, pos, placed, writes, asks, starts):
    state, out = None, []
    for t in range(S):
        lg, state = m.step(x[:, t], state, position=pos[:, t], placed=placed[:, t], write=writes[:, t],
                           ask=asks[:, t], start=starts[:, t])
        out.append(lg)
    return torch.stack(out, 1)


def test_off_is_the_mix_bit_for_bit():
    a, b = _model(None), _model(None)
    assert a.spatial.copy_path.refuse is None and "cr_L" not in a.init_state(1)
    with torch.no_grad():
        assert torch.equal(a.forward(*_batch()), b.forward(*_batch()))


def test_decide():
    b = torch.tensor([[0.9, 0.8, 0.1, 0.0],      # two pieces, an end: copied, length 2
                      [0.9, 0.5, 0.1, 0.0],      # the second piece unsure: refused
                      [0.1, 0.9, 0.9, 0.9],      # the first read already ended: refused
                      [0.9, 0.9, 0.9, 0.9]])     # no end within the lookahead: refused
    assert sr.decide(b, 0.7, 0.3).tolist() == [2, 0, 0, 0]


def test_runs():
    idx = torch.tensor([3, 4, 5, 9, 10, 14, 15])               # S = 12: 14 and 15 are the next row's 2 and 3
    start, off = sr.runs(idx, 12)
    assert start.tolist() == [0, 0, 0, 3, 3, 5, 5] and off.tolist() == [0, 1, 2, 0, 1, 0, 1]
    start, _ = sr.runs(torch.tensor([10, 11, 12, 13]), 12)     # 11 ends row 0, 12 starts row 1: two runs
    assert start.tolist() == [0, 0, 2, 2]


def _thresholds(b):
    pieces, end = b[:3], b[3]
    assert float(end) < float(pieces.min()), "the empty address must read below every written piece"
    return float(pieces.min()), float(end)


def test_a_whole_answer_comes_from_the_copy_and_the_end_from_the_trunk():
    m, batch = _model(), _batch()
    lo, end = _thresholds(_bests(m, batch))
    m.spatial.copy_path.set_refuse(lo - 1e-4, (lo + end) / 2, RID, STEP, 6)
    plain = _model(None)
    plain.spatial.copy_path = None
    with torch.no_grad():
        out, trunk = m.forward(*batch), plain.forward(*batch)
    cand = torch.tensor([31, 32, 33])
    for t in (8, 9, 10):                                       # the copy alone: all the mass on the written ids
        assert torch.logsumexp(out[0, t, cand], 0).abs() < 1e-4
    assert torch.allclose(out[0, 11], torch.log_softmax(trunk[0, 11], -1), atol=1e-5)    # the end: the trunk's
    for t in [i for i in range(S) if i not in (8, 9, 10, 11)]:                            # the rest: untouched
        assert torch.allclose(out[0, t], trunk[0, t], atol=1e-5)
    assert torch.allclose(_decode(m, *batch), out, atol=1e-4, rtol=1e-4)


def test_a_piece_under_theta_refuses_the_whole_answer():
    m, batch = _model(), _batch()
    b = _bests(m, batch)
    lo, end = _thresholds(b)
    hi = float(b[:3].max())
    m.spatial.copy_path.set_refuse((lo + hi) / 2 if hi > lo else lo + 1e-3, (lo + end) / 2, RID, STEP, 6)
    with torch.no_grad():
        out = m.forward(*batch)
    for t in (8, 9, 10, 11):
        assert int(out[0, t].argmax()) == RID and torch.exp(out[0, t, RID]) > 0.999         # no fact token anywhere
    assert torch.allclose(_decode(m, *batch), out, atol=1e-4, rtol=1e-4)


def test_an_empty_address_is_refused():
    m = _model()
    lo, end = _thresholds(_bests(m, _batch()))
    m.spatial.copy_path.set_refuse(lo - 1e-4, (lo + end) / 2, RID, STEP, 6)
    batch = _batch(ask_at=P0 + torch.tensor([40.0, 40.0, 0.0]))                   # nothing written there
    with torch.no_grad():
        out = m.forward(*batch)
    assert int(out[0, 8].argmax()) == RID


def test_theta_end_above_theta_is_rejected():
    with pytest.raises(ValueError):
        _model({"theta": 0.2, "theta_end": 0.5, "refuse_id": RID})


def test_the_pointers_query_rebound_is_the_key_of_the_later_piece():
    link = SpatialLink(D, scale=1.0, seed=5, query="pointer", bind="writes", key="text", copy="ring")
    table = torch.randn(40, D)
    p = torch.tensor([[3.0, 7.0, 1.0]])                                           # subject 3, relation 7, piece 1
    q = link.encode(p, table)                                                     # what the pointer gives at piece 1
    got = sr.queries(link, p, q, table, 4, STEP)                                  # pieces 1..4
    want = link.encode(torch.tensor([[[3.0, 7.0, 1.0 + k] for k in range(4)]]), table)
    assert torch.allclose(got, want, atol=1e-4)
