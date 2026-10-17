"""The trunk's route to space (cubbyllm/model/spatial.py), each claim to float precision.

Pinned: a model given the link computes exactly what it computed before until it is handed a
place (text goldens hold); a place of zero is the identity; moving a token is binding; the
place can be read back from the bound embedding; the lookup is the algebra of the host's SSP
(the same phases, so a Scene's vector reads in the trunk); with nothing placed the read is
zero whatever the trunk's state; the gate and the query get gradient through the read once
something is placed; the scene is causal (a later token changes nothing before it); and the
decode step binds, accumulates and reads what the forward does.
"""
from __future__ import annotations

import numpy as np
import torch

from cubbyllm.model.spatial import SpatialLink
from cubbyllm.spatial.scene import Scene
from test_grow import META, _build

D, S = META["D"], 9


def _model(spatial: bool, seed: int = 0):
    m = _build(META, seed=seed)
    torch.manual_seed(seed + 100)
    with torch.no_grad():
        for p in list(m.parameters()) + list(m.context_source.parameters()):
            p.copy_(torch.randn_like(p) * 0.3)
    if spatial:
        m.spatial = SpatialLink(D, scale=1.0, seed=5, at=META["L"] // 2)
    return m


def _tokens(seed=3):
    return torch.randint(0, META["vocab"], (2, S), generator=torch.Generator().manual_seed(seed))


def _placed():
    placed = torch.zeros(2, S, dtype=torch.bool)
    placed[:, :4] = True                                                 # the first four tokens are things
    return placed


def test_the_link_changes_nothing_until_a_place_is_given():
    plain, linked = _model(False), _model(True)
    x = _tokens()
    with torch.no_grad():
        a, b = plain.forward(x), linked.forward(x)
        assert torch.equal(a, b)                                          # bit-identical: text goldens hold
        e = torch.randn(2, S, D)
        assert torch.allclose(linked.spatial.bind(e, torch.zeros(2, S, 3)), e, atol=1e-6)   # the origin is the identity
        c = linked.features(x, positions=torch.zeros(2, S, 3))           # ... and through the trunk, to float32
        assert torch.allclose(plain.features(x), c, atol=1e-4)
        d = linked.features(x, positions=None, placed=_placed())         # o is zero at step 0: the read adds nothing
        assert torch.equal(plain.features(x), d)


def test_moving_a_token_is_binding_and_the_place_reads_back():
    link = SpatialLink(256, scale=2.0, seed=1)
    e = torch.randn(4, 256)
    p, d = torch.tensor([[3.0, -2.0, 1.0]] * 4), torch.tensor([[1.0, 4.0, -7.0]] * 4)
    assert torch.allclose(link.bind(link.bind(e, p), torch.zeros(4, 3)), link.bind(e, p), atol=1e-5)
    assert torch.allclose(link.bind(e, p + d), link.unbind(link.bind(e, p), link.encode(-d)), atol=1e-4)
    probe = link.unbind(link.bind(e, p), e)                                 # unbind by the token: the place remains
    lattice = torch.tensor([[x, y, z] for x in range(-4, 5) for y in range(-4, 5) for z in range(-2, 3)], dtype=torch.float32)
    assert torch.equal(lattice[(probe @ link.encode(lattice).T).argmax(-1)], p)
    host = link.space().encode(p.numpy().astype(np.float64))                # the same space as the host's
    assert np.abs(host - link.encode(p).numpy()).max() < 1e-5


def test_the_lookup_is_the_scenes_algebra():
    """A Scene built in the link's space: the lookup at a place returns the code of what is there;
    the trunk's own scene -- the running sum of placed, bound embeddings -- reads back the
    embedding of the token at the place."""
    link = SpatialLink(512, scale=1.0, seed=2)
    sc = Scene(link.space())
    sc.place("well", (4, 7, 0)); sc.place("oak", (5, 7, 0)); sc.place("forge", (4, 9, 0))
    V = torch.tensor(sc.vector, dtype=torch.float32).unsqueeze(0)
    for name, cell in sc.things.items():
        q = link.encode(torch.tensor([cell], dtype=torch.float32))
        got = link.lookup(V, q)[0].numpy()
        assert max(sc.things, key=lambda n: float(sc.code(n) @ got)) == name and float(sc.code(name) @ got) > 0.8
    assert torch.allclose(link.unitary(3.0 * q), link.unitary(q), atol=1e-5)  # magnitudes do not change the place
    e = torch.nn.functional.normalize(torch.randn(1, 3, 512), dim=-1)
    cells = torch.tensor([[[4, 7, 0], [5, 7, 0], [4, 9, 0]]], dtype=torch.float32)
    scene = link.scene(link.bind(e, cells), torch.ones(1, 3, dtype=torch.bool))     # (1, 3, 512), running
    for t in range(3):
        back = link.lookup(scene[:, 2], link.encode(cells[:, t]))               # the whole scene, asked at cell t
        assert int((back @ e[0].T).argmax()) == t and float(back @ e[0, t]) > 0.8
    assert torch.allclose(scene[:, 0], link.bind(e, cells)[:, 0])                # causal: V_0 is the first thing alone


def test_the_read_is_zero_without_a_scene_and_routes_gradient_with_one():
    link = SpatialLink(D, scale=1.0, seed=5)
    with torch.no_grad():
        link.o.weight.normal_(0, 0.3)
        link.gate.bias.zero_()
    h = torch.randn(2, S, D, requires_grad=True)
    assert torch.equal(link.read(h, None), torch.zeros(2, S, D))
    out = link.read(h, torch.randn(2, S, D))
    assert out.abs().max() > 0
    out.pow(2).sum().backward()
    assert link.gate.weight.grad.abs().sum() > 0 and link.q.weight.grad.abs().sum() > 0
    assert link.o.weight.grad.abs().sum() > 0 and h.grad.abs().sum() > 0


def test_the_scene_is_causal_and_the_decode_step_matches_the_forward():
    m = _model(True)
    with torch.no_grad():
        m.spatial.o.weight.normal_(0, 0.3)
        m.spatial.gate.bias.zero_()
    x = _tokens(4)
    pos = torch.randint(-5, 6, (2, S, 3)).float()
    placed = _placed()
    with torch.no_grad():
        full = m.features(x, positions=pos, placed=placed)
        state, steps = None, []
        for t in range(S):
            logits, state = m.step(x[:, t], state, position=pos[:, t], placed=placed[:, t])
            steps.append(logits)
        full_logits = m.logits_from(full)
        assert torch.allclose(full_logits, torch.stack(steps, 1), atol=1e-4, rtol=1e-4)
        assert torch.allclose(state["scene"], m.spatial.scene(m.spatial.bind(m.embedding.embed(x, None), pos), placed)[:, -1], atol=1e-5)
        moved = pos.clone(); moved[:, 6:] += 3.0                           # a later place changes nothing before it
        later = m.features(x, positions=moved, placed=placed)
        assert torch.allclose(full[:, :6], later[:, :6], atol=1e-5) and not torch.allclose(full, later, atol=1e-3)
        other = m.features(x, positions=pos + 1.0, placed=placed)
        assert not torch.allclose(full, other, atol=1e-3)                  # the places reach the trunk
