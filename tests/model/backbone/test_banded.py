"""Banded sliding-window attention equals the dense-mask reference, values and gradients."""
from __future__ import annotations

import pytest
import torch

from cubbyllm.model.backbone.banded import banded_attention, dense_attention


@pytest.mark.parametrize("S,W", [(9, 4), (12, 4), (13, 4), (64, 16), (100, 32), (5, 4)])
def test_banded_is_the_dense_reference(S, W):
    torch.manual_seed(S * 7 + W)
    q, k, v = (torch.randn(2, 3, S, 8, requires_grad=True) for _ in range(3))
    a = dense_attention(q, k, v, W)
    ga = torch.autograd.grad(a.pow(2).sum(), (q, k, v))
    b = banded_attention(q, k, v, W)
    gb = torch.autograd.grad(b.pow(2).sum(), (q, k, v))
    assert b.shape == a.shape and torch.allclose(a, b, atol=1e-5)
    assert all(torch.allclose(x, y, atol=1e-4) for x, y in zip(ga, gb))


def test_the_mixer_uses_it_and_decode_still_matches(monkeypatch):
    from cubbyllm.model.backbone.hybrid import HybridBackbone
    monkeypatch.setenv("CB_BANDED", "1")
    torch.manual_seed(0)
    bb = HybridBackbone(16, 3, attn_every=1, window=4, heads=2)
    x = torch.randn(2, 13, 16)
    with torch.no_grad():
        full = bb.forward(x)
        states, ys = None, []
        for t in range(13):
            y, states = bb.step(x[:, t], states)
            ys.append(y)
    assert torch.allclose(full, torch.stack(ys, 1), atol=1e-4)
    monkeypatch.setenv("CB_BANDED", "0")
    with torch.no_grad():
        assert torch.allclose(bb.forward(x), full, atol=1e-5)                 # the dense path, the same function
