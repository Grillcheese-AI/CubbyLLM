"""WorldModelBridge — the first-class model<->world interface (H-F2).

Wired: STANDALONE — a bridge, not the forward path.

H-F2 found the predecessor's two bridges too narrow for automatic
specialization: ``NoveltyToWorldBridge`` is a one-way duck-typed callback, and
the CubeLang bridge is a subprocess text boundary. H0's validated mechanism
says the bridge must carry three things as first-class traffic:
  (a) a context embedding, BOTH directions,
  (b) novelty events,
  (c) specialist handles.
This protocol names all three so a real bridge can't quietly drop any of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.protocols import Wiring

if TYPE_CHECKING:
    from ..core.context import Context


@dataclass(frozen=True)
class Challenge:
    """A task CubbyLLM offloads to the world model."""
    context: object            # (k, l) grilly block-code — the task/query
    tag: str | None = None     # optional symbolic label (deterministic tests/audit)


@dataclass(frozen=True)
class ChallengeResult:
    """The world model's answer to a Challenge."""
    result: object             # (k, l) block-code prediction the world produced
    world_id: str              # which expert world answered
    confidence: float          # routing/prediction confidence in [0, 1]
    spawned: bool              # True iff a NEW specialist world was grown for this


@runtime_checkable
class WorldModelBridge(Protocol):
    """Carries context both ways + novelty events + specialist handles."""

    def attempt(self, challenge: "Challenge") -> "ChallengeResult":
        """Route the challenge to an expert world; spawn a specialist if none
        fits; return its block-code result + world_id + confidence + spawned."""
        ...

    def push_context(self, ctx: "Context") -> None:
        """Send an inferred context out to the world model."""
        ...

    def pull_context(self) -> "Context | None":
        """Receive a context suggestion back from the world model (or None)."""
        ...

    def emit_novelty(self, event: object) -> None:
        """Publish a novelty event (the existing one-way channel, generalized)."""
        ...

    def register_specialist(self, handle: object) -> None:
        """Register a specialist handle produced by context-driven specialization."""
        ...


__wiring__ = Wiring.STANDALONE
