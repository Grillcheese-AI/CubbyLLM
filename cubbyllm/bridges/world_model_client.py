"""world_model_client — CubbyLLM's side of the world-model bridge.

Wired: STANDALONE — a bridge helper, not the forward path.

Builds a Challenge from a block-code context and drives any WorldModelBridge
(the FakeWorldModel in tests; MoWM's CubbyBridge in real use). Import-light: no
mowm/cubemind/torch import here — the concrete bridge is passed in by the caller.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .world_model import Challenge, ChallengeResult

__wiring__ = Wiring.STANDALONE


def attempt_challenge(bridge, context, tag: str | None = None) -> ChallengeResult:
    """Pose `context` (a (k,l) block-code) to `bridge` as a Challenge."""
    return bridge.attempt(Challenge(context=context, tag=tag))


def newly_specialized(result: ChallengeResult) -> bool:
    """True iff the world model grew a new specialist for this challenge."""
    return result.spawned
