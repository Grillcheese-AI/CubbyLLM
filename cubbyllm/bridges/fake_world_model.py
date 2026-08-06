"""FakeWorldModel — a deterministic, dependency-free WorldModelBridge for tests.

Wired: STANDALONE — a test double, not the forward path.

Implements the challenge round-trip without importing mowm/cubemind/torch: it
mints a new world_id (spawned=True) the first time it sees a challenge tag,
routes to the recorded world (spawned=False) on repeats, and treats a set of
seeded tags as already-existing worlds (route, never spawn). Lets CubbyLLM's
attempt_challenge() flow + its assertions run under plain pytest, no GPU.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .world_model import Challenge, ChallengeResult

__wiring__ = Wiring.STANDALONE


class FakeWorldModel:
    def __init__(self, seeded_tags: tuple[str, ...] = ()) -> None:
        self._worlds: dict[str, str] = {t: f"seed:{t}" for t in seeded_tags}
        self._spawn_counter = 0
        self.registered: list[str] = []

    def _key(self, challenge: Challenge) -> str:
        return challenge.tag if challenge.tag is not None else repr(challenge.context)

    def attempt(self, challenge: Challenge) -> ChallengeResult:
        key = self._key(challenge)
        if key in self._worlds:                       # known / already-spawned
            return ChallengeResult(
                result=challenge.context, world_id=self._worlds[key],
                confidence=1.0, spawned=False,
            )
        self._spawn_counter += 1                      # novel -> spawn a specialist
        world_id = f"spawn:{self._spawn_counter}"
        self._worlds[key] = world_id
        self.register_specialist(world_id)
        return ChallengeResult(
            result=challenge.context, world_id=world_id,
            confidence=0.5, spawned=True,
        )

    # underlying channels (minimal; attempt() drives register_specialist)
    def push_context(self, ctx) -> None: ...
    def pull_context(self): return None
    def emit_novelty(self, event: object) -> None: ...
    def register_specialist(self, handle: object) -> None:
        self.registered.append(str(handle))
