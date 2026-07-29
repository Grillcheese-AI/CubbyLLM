import cubbyllm.bridges.world_model as mod
from cubbyllm.bridges import WorldModelBridge
from cubbyllm.core.protocols import Wiring


def test_bridge_carries_all_three_channels():
    for method in ("push_context", "pull_context", "emit_novelty", "register_specialist"):
        assert hasattr(WorldModelBridge, method), method


def test_wiring_declared():
    assert mod.__wiring__ is Wiring.STANDALONE
