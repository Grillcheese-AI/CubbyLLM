import dataclasses

import pytest

import cubbyllm.core.context as mod
from cubbyllm.core.context import Context, ContextSource, FrozenSlotRouter
from cubbyllm.core.protocols import Wiring


def test_context_is_frozen_dataclass():
    c = Context(vector=[0.0], source_id="test", confidence=None)
    assert c.source_id == "test"
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.source_id = "x"  # type: ignore[misc]


def test_frozen_slot_router_infers_context():
    import torch

    router = FrozenSlotRouter(input_dim=32, n_slots=6, ctx_dim=16)
    assert isinstance(router, ContextSource)
    x = torch.randn(4, 32)
    ctx = router.infer(x)
    assert ctx.vector.shape == (4, 16)
    assert ctx.confidence.shape == (4,)
    assert ctx.source_id == "frozen_slot_router"


def test_freeze_makes_is_frozen_true_and_stops_grad():
    router = FrozenSlotRouter(input_dim=8, n_slots=3, ctx_dim=4)
    assert router.is_frozen is False
    router.freeze()
    assert router.is_frozen is True
    assert all(not p.requires_grad for p in router.parameters())


def test_wiring_declared():
    assert mod.__wiring__ in (Wiring.WIRED, Wiring.STANDALONE)
