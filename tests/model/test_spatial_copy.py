"""The spatial link's copy path (cubbyllm/model/spatial_copy.py), each claim to float precision.

Pinned: off, nothing is built and the forward and decode are what they were; added, every
position that does not ask keeps the model's logits exactly (forward and decode, both modes) and
an asking one becomes a normalized log-mixture; with the gate open and one token written at an
address, the asking token's top token is that token; decode = forward with the copy open, roles,
a document restart and a ring smaller than the writes; the ring holds each document's last
``ring`` writers; an asking token with nothing written keeps the trunk's logits; the gate and the
temperature get gradient; the meta keys build it, their absence not.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from cubbyllm.model.spatial import SpatialLink
from cubbyllm.model.spatial_copy import ring_candidates
from test_grow import META
from test_spatial import D, S, _model, _tokens

MODES = ["ring", "table"]


def _linked(copy=None, seed=0, ring=256):
    m = _model(True, seed)
    link = SpatialLink(D, scale=1.0, seed=5, at=META["L"] // 2, query="placed", bind="writes", copy=copy, copy_ring=ring)
    torch.manual_seed(seed + 7)
    with torch.no_grad():
        link.o.weight.normal_(0, 0.3)
        link.q.weight.normal_(0, 0.3)
        link.gate.bias.zero_()
    m.spatial = link
    return m


def _roles(seed=9):
    g = torch.Generator().manual_seed(seed)
    x = _tokens(seed)
    pos = torch.randint(-5, 6, (2, S, 3), generator=g).float()
    placed = torch.ones(2, S, dtype=torch.bool)
    writes = torch.zeros(2, S, dtype=torch.bool)
    writes[:, [0, 1, 2, 5, 6]] = True
    starts = torch.zeros(2, S, dtype=torch.bool)
    starts[:, 5] = True                                                    # a second document at 5
    return x, pos, placed, writes, placed & ~writes, starts


def _decode(m, x, pos, placed, writes, asks, starts):
    state, steps = None, []
    for t in range(S):
        lg, state = m.step(x[:, t], state, position=pos[:, t], placed=placed[:, t],
                           write=writes[:, t], ask=asks[:, t], start=starts[:, t])
        steps.append(lg)
    return torch.stack(steps, 1), state


def _open(m, bias=0.0, tau=1.0, seed=3):
    torch.manual_seed(seed)
    with torch.no_grad():
        m.spatial.copy_path.gate.weight.normal_(0, 0.5)
        m.spatial.copy_path.gate.bias.fill_(bias)
        m.spatial.copy_path.tau.fill_(tau)


def test_off_builds_nothing_and_changes_nothing():
    link = SpatialLink(D, scale=1.0, seed=5)
    assert link.copy_path is None
    assert sorted(link.state_dict()) == ["gate.bias", "gate.weight", "norm_w", "o.weight", "phases", "q.weight"]
    m = _linked(None)
    batch = _roles()
    with torch.no_grad():
        full = m.forward(*batch)
        assert torch.equal(full, m.logits_from(m.features(*batch)))           # the forward is the features' head
        steps, state = _decode(m, *batch)
        assert "ring" not in state and torch.allclose(full, steps, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_added_the_positions_that_do_not_ask_keep_the_models_logits(mode):
    off, on = _linked(None), _linked(mode)
    on.spatial.load_state_dict(off.spatial.state_dict(), strict=False)    # the same link, the copy added
    names = [n for n, _ in on.spatial.named_parameters()]
    assert names[-3:] == ["copy_path.tau", "copy_path.gate.weight", "copy_path.gate.bias"]   # last: old checkpoints load
    batch = _roles()
    asks = batch[4]
    with torch.no_grad():
        a, b = on.forward(*batch), off.forward(*batch)
        assert torch.equal(a[~asks], b[~asks])                               # bit for bit where nothing asks
        assert torch.equal(_decode(on, *batch)[0][~asks], _decode(off, *batch)[0][~asks])
        mixed = a[asks]                                                      # every asker here has a writer before it
        assert torch.allclose(mixed.logsumexp(-1), torch.zeros(len(mixed)), atol=1e-5)   # log p, normalized
        assert not torch.allclose(mixed, b[asks], atol=1e-3)                 # lam starts at 0.1: the copy is on


@pytest.mark.parametrize("mode", MODES)
def test_with_the_gate_open_the_one_written_token_is_the_asking_tokens_top_token(mode):
    m = _linked(mode)
    with torch.no_grad():
        m.spatial.copy_path.gate.bias.fill_(20.0)                            # lam ~ 1
    x = _tokens(11)
    pos = torch.zeros(2, S, 3)
    pos[:, 1] = torch.tensor([3.0, -2.0, 1.0])
    pos[:, 7] = pos[:, 1]                                                   # token 7 asks where token 1 wrote
    placed = torch.zeros(2, S, dtype=torch.bool)
    placed[:, [1, 7]] = True
    writes = torch.zeros_like(placed)
    writes[:, 1] = True
    starts = torch.zeros_like(placed)
    with torch.no_grad():
        full = m.forward(x, pos, placed, writes, placed & ~writes, starts)
        steps, _ = _decode(m, x, pos, placed, writes, placed & ~writes, starts)
    assert torch.equal(full[:, 7].argmax(-1), x[:, 1]) and torch.equal(steps[:, 7].argmax(-1), x[:, 1])


@pytest.mark.parametrize("mode,ring", [("ring", 256), ("ring", 2), ("table", 256)])
def test_decode_matches_the_forward_with_the_copy_open(mode, ring):
    m, off = _linked(mode, ring=ring), _linked(None)
    _open(m)
    batch = _roles()
    with torch.no_grad():
        full = m.forward(*batch)
        steps, state = _decode(m, *batch)
        assert torch.allclose(full, steps, atol=1e-4, rtol=1e-4)
        assert not torch.allclose(full, off.forward(*batch), atol=1e-3)     # the copy reaches the output
    if mode == "ring":                                                     # the second document's writers only
        x = batch[0]
        assert torch.equal(state["ring_n"], torch.tensor([2, 2]))
        assert torch.equal(state["ring"][:, :2].sort(1).values, x[:, [5, 6]].sort(1).values)


def test_the_ring_holds_each_documents_last_writers():
    g = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, META["vocab"], (3, S), generator=g)
    writes = torch.rand(3, S, generator=g) < 0.6
    starts = torch.rand(3, S, generator=g) < 0.2
    idx = torch.arange(3 * S)
    for ring in (256, 3):
        ids, ok = ring_candidates(tokens, writes, starts, idx, ring)
        for i in idx.tolist():
            b, t = divmod(i, S)
            u = t
            while u > 0 and not starts[b, u]:                               # this position's document start
                u -= 1
            want = [int(tokens[b, v]) for v in range(u, t + 1) if writes[b, v]][-ring:]
            assert sorted(ids[ok[i]].tolist()) == sorted(want), (ring, b, t)


def test_an_asking_token_with_nothing_written_keeps_the_trunks_logits():
    off, on = _linked(None), _linked("ring")
    on.spatial.load_state_dict(off.spatial.state_dict(), strict=False)
    _open(on, bias=5.0)
    x, pos, placed, writes, asks, starts = _roles()
    writes[:, :5] = False                                                  # document 1 writes nothing
    asks = placed & ~writes
    with torch.no_grad():
        a, b = on.forward(x, pos, placed, writes, asks, starts), off.forward(x, pos, placed, writes, asks, starts)
    assert torch.equal(a[:, :5], b[:, :5]) and not torch.allclose(a[:, 7:], b[:, 7:], atol=1e-3)


@pytest.mark.parametrize("mode", MODES)
def test_the_gate_and_the_temperature_get_gradient(mode):
    m = _linked(mode)
    batch = _roles()
    logits = m.forward(*batch)
    x, asks = batch[0], batch[4]
    torch.nn.functional.cross_entropy(logits[asks], x[asks]).backward()
    cp = m.spatial.copy_path
    assert float(cp.gate.weight.grad.abs().sum()) > 0 and float(cp.gate.bias.grad.abs()) > 0
    assert float(cp.tau.grad.abs()) > 0
    assert abs(float(torch.sigmoid(cp.gate.bias)) - 0.1) < 1e-6                # lam starts at 0.1


def test_the_meta_keys_build_the_copy_path_and_their_absence_does_not():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "validation"))
    import train_base
    meta = {"D": D, "L": META["L"], "spatial": {"scale": 1.0, "seed": 5, "at": 2, "query": "placed"}}
    assert train_base.spatial_link(meta).copy_path is None
    for mode in MODES:
        meta["spatial"].update(copy=mode, copy_tau=2.5, copy_ring=64)
        cp = train_base.spatial_link(meta).copy_path
        assert cp.mode == mode and cp.ring == 64 and float(cp.tau) == 2.5
    meta["spatial"]["copy"] = "nearest"
    with pytest.raises(ValueError):
        train_base.spatial_link(meta)
