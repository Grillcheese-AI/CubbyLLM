"""The spatial link's text keys and pointer (cubbyllm/model/spatial_key.py, order 0006), each claim
to float precision.

Pinned: off, the link builds neither and its tensors are what they were, and the new arguments
change nothing; text keys of the same ids are identical and of different ids near-orthogonal
(the random-phase floor); a writer is bound at its key and an asker given that key reads the written
token back; a one-hot pointer gives the write's key, and a model whose pointer is forced onto the
gold tokens reads and copies exactly as one given the address (swapping the two pointers changes
nothing -- binding commutes, so a key is symmetric in subject and relation -- and one token off
changes the answer); decode = forward with the pointer,
the copy and a document restart; the pointer's logits get gradient; the meta keys build it.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

from cubbyllm.model.spatial import SpatialLink
from cubbyllm.model.spatial_key import TextKeys
from test_grow import META
from test_spatial import D, _model

S = 14
SUBJ, REL, OTHER = 5, 6, 9                           # token ids; a second fact (OTHER, REL)


def _keyed(query="pointer", copy="ring", seed=0):
    m = _model(True, seed)
    link = SpatialLink(D, scale=1.0, seed=5, at=META["L"] // 2, query=query, bind="writes", copy=copy,
                       key="text", pointer_window=8, pointer_dim=8)
    torch.manual_seed(seed + 7)
    with torch.no_grad():
        link.o.weight.normal_(0, 0.3)
        link.q.weight.normal_(0, 0.3)
        link.gate.bias.zero_()
        if link.pointer is not None:
            for p in link.pointer.parameters():
                p.normal_(0, 0.5)
    m.spatial = link
    return m


def _doc(restart=False):
    """Two facts written at their keys (pieces 0-1 of (SUBJ, REL), piece 0 of (OTHER, REL)), filler,
    then the question ``SUBJ REL`` and three askers (pieces 0, 1, 2 -- the last an empty address);
    ``restart``: a second document from position 9 (its askers' window holds only the question)."""
    x = torch.tensor([[1, 20, 21, 2, 22, 30, 31, 32, 33, SUBJ, REL, 3, 20, 21]] * 2)
    pos = torch.zeros(2, S, 3)
    writes = torch.zeros(2, S, dtype=torch.bool)
    asks = torch.zeros(2, S, dtype=torch.bool)
    for t, (s, r, i) in {1: (SUBJ, REL, 0), 2: (SUBJ, REL, 1), 4: (OTHER, REL, 0)}.items():
        pos[:, t], writes[:, t] = torch.tensor([s, r, i]), True
    for t, i in {11: 0, 12: 1, 13: 2}.items():
        pos[:, t], asks[:, t] = torch.tensor([SUBJ, REL, i]), True              # (s, r) for "placed"; the pointer reads i
    starts = torch.zeros(2, S, dtype=torch.bool)
    if restart:
        starts[1, 9] = True
    return x, pos, writes | asks, writes, asks, starts


def test_off_builds_neither_and_the_new_arguments_change_nothing():
    a = SpatialLink(D, scale=1.0, seed=5, at=2, query="placed", bind="writes", copy="ring")
    assert a.keys is None and a.pointer is None
    assert [(n, p.shape) for n, p in a.named_parameters()] == [
        ("norm_w", (D,)), ("q.weight", (D, D)), ("gate.weight", (1, D)), ("gate.bias", (1,)), ("o.weight", (D, D)),
        ("copy_path.tau", ()), ("copy_path.gate.weight", (1, D)), ("copy_path.gate.bias", (1,))]
    p = torch.randint(-5, 6, (3, 3)).float()
    e = torch.randn(3, D)
    table = torch.randn(40, D)
    assert torch.equal(a.encode(p, table), a.encode(p)) and torch.equal(a.bind_input(e, p, None, table), a.bind_input(e, p))
    for bad in (dict(key="text"), dict(query="pointer"), dict(key="hash", bind="writes")):
        with pytest.raises(ValueError):
            SpatialLink(D, **bad)


def test_text_keys_of_the_same_ids_are_identical_and_of_different_ids_near_orthogonal():
    d, V = 256, 300
    keys, table = TextKeys(d, seed=1), torch.randn(V, d)
    p = torch.stack([torch.arange(V), torch.full((V,), 7), torch.zeros(V, dtype=torch.long)], -1)
    k = torch.fft.irfft(keys.address(table, p), n=d)
    assert torch.equal(k, torch.fft.irfft(keys.address(table, p.clone()), n=d))
    assert torch.allclose(k.norm(dim=-1), torch.ones(V), atol=1e-4)                  # unitary: unit vectors
    c = (F.normalize(k, dim=-1) @ F.normalize(k, dim=-1).T)[~torch.eye(V, dtype=torch.bool)].abs()
    assert float(c.mean()) < 0.053 and float(c.quantile(0.99)) < 0.2                # the random floor: sqrt(2/pi/d) = 0.050
    p2 = p.clone()
    p2[:, 2] = 1                                                                     # the next piece: another address
    k2 = torch.fft.irfft(keys.address(table, p2), n=d)
    assert float(F.cosine_similarity(k, k2, dim=-1).abs().mean()) < 0.06


def test_a_writer_is_bound_at_its_key_and_an_asker_given_it_reads_the_token_back():
    m = _keyed("placed", copy=None)
    link, E = m.spatial, m.embedding.core.weight
    p = torch.tensor([[SUBJ, REL, 0]]).float()
    e = torch.randn(1, D)
    bound = link.bind_input(e, p, torch.tensor([True]), E)
    assert torch.allclose(link.lookup(bound, link.encode(p, E)), e, atol=1e-5)
    assert torch.equal(link.bind_input(e, p, torch.tensor([False]), E), e)          # not a writer: enters as itself


def test_a_one_hot_pointer_gives_the_writes_key():
    m = _keyed()
    link, E = m.spatial, m.embedding.core.weight
    win = torch.tensor([[33, REL, 20, SUBJ, 21, 22, 30, 31]])                        # offsets 1..8
    a = torch.zeros(1, 2, 8)
    a[0, 0, 3] = a[0, 1, 1] = 1.0                                                    # the subject 4 back, the relation 2 back
    for j in range(3):
        got = link.pointer.key(link.keys, a, E[win], torch.tensor([j]))
        assert torch.allclose(got, link.encode(torch.tensor([[SUBJ, REL, j]]).float(), E), atol=1e-6)


@pytest.mark.parametrize("copy", [None, "ring"])
def test_a_pointer_forced_onto_the_gold_tokens_reads_and_copies_as_the_address_given(copy):
    """The asker at t points 2 + j back at SUBJ and 1 + j back at REL (t = 11 + j): bias the pointer's
    logits there, and the model is the one given (SUBJ, REL, j) -- the read, the copy, the logits."""
    ptr, given = _keyed("pointer", copy), _keyed("placed", copy)              # the same weights but the pointer's
    theirs = dict(given.spatial.named_parameters())
    assert all(torch.equal(p, theirs[n]) for n, p in ptr.spatial.named_parameters() if not n.startswith("pointer."))
    assert all(torch.equal(p, q) for p, q in zip(list(ptr.parameters())[:20], list(given.parameters())[:20]))
    x, pos, placed, writes, asks, starts = _doc()
    gold = {11: (2, 1), 12: (3, 2), 13: (4, 3)}                                     # (subject, relation) offsets

    def point(subj, rel):                                                            # logits: the bias alone, one-hot
        with torch.no_grad():
            ptr.spatial.pointer.q.weight.zero_()
            ptr.spatial.pointer.bias.fill_(-1e4)
            ptr.spatial.pointer.bias[0, subj - 1] = ptr.spatial.pointer.bias[1, rel - 1] = 0.0

    for t, (os_, or_) in gold.items():
        one = asks & (torch.arange(S) == t)
        point(os_, or_)
        with torch.no_grad():
            a, b = ptr.forward(x, pos, placed, writes, one, starts), given.forward(x, pos, placed, writes, one, starts)
            assert torch.allclose(a, b, atol=1e-5), t
            point(or_, os_)                                                          # swapped: binding commutes, the same key
            assert torch.allclose(ptr.forward(x, pos, placed, writes, one, starts), b, atol=1e-5), t
            point(os_ + 1, or_)                                                      # the subject pointer one token off
            assert not torch.allclose(ptr.forward(x, pos, placed, writes, one, starts)[:, t], b[:, t], atol=1e-3), t


@pytest.mark.parametrize("copy, restart", [(None, False), ("ring", False), ("ring", True)])
def test_decode_matches_the_forward_with_the_pointer(copy, restart):
    m = _keyed("pointer", copy)
    x, pos, placed, writes, asks, starts = _doc(restart)
    with torch.no_grad():
        full = m.forward(x, pos, placed, writes, asks, starts)
        state, steps = None, []
        for t in range(S):
            lg, state = m.step(x[:, t], state, position=pos[:, t], placed=placed[:, t], write=writes[:, t],
                               ask=asks[:, t], start=starts[:, t])
            steps.append(lg)
    assert torch.allclose(torch.stack(steps, 1), full, atol=1e-4)
    assert not torch.allclose(full[1, 11:], full[0, 11:], atol=1e-4) or not restart  # the restart changed what it reads


def test_the_pointer_gets_gradient():
    m = _keyed("pointer", "ring")
    x, pos, placed, writes, asks, starts = _doc()
    m.forward(x, pos, placed, writes, asks, starts).logsumexp(-1).sum().backward()
    for n, p in m.spatial.pointer.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, n
    idx, lg = m.spatial.pointer.last
    assert lg.shape == (len(idx), 2, 8) and int(asks.sum()) == len(idx)


def test_the_meta_keys_build_the_keys_and_the_pointer():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "validation"))
    import train_base
    meta = {"D": D, "L": META["L"], "spatial": {"scale": 1.0, "seed": 5, "at": 2, "query": "placed", "bind": "writes"}}
    assert train_base.spatial_link(meta).keys is None
    meta["spatial"].update(key="text", query="pointer", pointer_window=12, pointer_dim=8)
    link = train_base.spatial_link(meta)
    assert link.keys is not None and link.pointer.w == 12 and link.pointer.dk == 8
