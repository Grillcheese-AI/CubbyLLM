import numpy as np
from cubbyllm.core.protocols import Wiring
from cubbyllm.bridges.world_model import Challenge, ChallengeResult, WorldModelBridge
from cubbyllm.bridges import fake_world_model as fwm


def _ctx():
    return np.zeros((4, 32), dtype=np.float32)


def test_fake_spawns_on_novel_then_reuses():
    fake = fwm.FakeWorldModel()
    r1 = fake.attempt(Challenge(context=_ctx(), tag="Z"))
    assert isinstance(r1, ChallengeResult)
    assert r1.spawned is True                      # novel -> spawn a specialist
    r2 = fake.attempt(Challenge(context=_ctx(), tag="Z"))
    assert r2.spawned is False                     # same challenge -> reuse
    assert r2.world_id == r1.world_id              # ...the same specialist
    assert fake.registered == [r1.world_id]        # exactly one registered


def test_fake_routes_seeded_tag_without_spawning():
    fake = fwm.FakeWorldModel(seeded_tags=("X",))
    r = fake.attempt(Challenge(context=_ctx(), tag="X"))
    assert r.spawned is False                       # known domain -> no spawn
    assert fake.registered == []


def test_fake_satisfies_protocol():
    assert isinstance(fwm.FakeWorldModel(), WorldModelBridge)


def test_wiring_is_standalone():
    assert fwm.__wiring__ is Wiring.STANDALONE
