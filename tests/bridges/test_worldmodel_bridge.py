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


def test_attempt_challenge_drives_the_bridge_and_flags_new_specialists():
    from cubbyllm.bridges.world_model_client import attempt_challenge, newly_specialized
    fake = fwm.FakeWorldModel(seeded_tags=("X",))
    known = attempt_challenge(fake, _ctx(), tag="X")
    assert newly_specialized(known) is False
    novel = attempt_challenge(fake, _ctx(), tag="Z")
    assert newly_specialized(novel) is True
    again = attempt_challenge(fake, _ctx(), tag="Z")
    assert again.world_id == novel.world_id and again.spawned is False


def test_import_cubbyllm_stays_light():
    # A bare `import cubbyllm` must not pull torch or mowm. Check in a FRESH
    # interpreter: pytest's collection imports sibling test modules (some import
    # torch at module scope), so an in-process sys.modules check is polluted.
    # Pin the child's import path to the repo root (from __file__, not cwd) so
    # the check doesn't depend on where pytest was invoked from.
    import pathlib, subprocess, sys
    root = pathlib.Path(__file__).resolve().parents[2]  # tests/bridges/ -> repo root
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import cubbyllm\n"
        "assert 'torch' not in sys.modules, 'import cubbyllm pulled torch'\n"
        "assert not any(m == 'mowm' or m.startswith('mowm.') for m in sys.modules), 'import cubbyllm pulled mowm'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
