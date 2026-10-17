"""The host's route to the scene: which questions the scene is the source for.

Wired: STANDALONE (2026-10-17). The ask loop has one source per question (``learn_and_answer``'s
``source``); a scene is the source when the plan asks for something only a scene states --
a relation of its table (``position``, ``<direction> neighbour``, ``nearest <kind>``,
``region``) at any hop, or a seed that is a thing in it. Exact and cheap: the same decision
the trunk's gate learns (``cubbyllm.model.spatial``), made here from the plan's words so the
loop routes correctly before any training has happened.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from ..reasoning.planner import normalize, of_prefixes
from .scene import Scene

__all__ = ["route", "spatial_words"]


def spatial_words(plan) -> list[str]:
    """The relation wordings the plan carries: the chain's hops and every ' of ' prefix of
    the tail (``tail_relation``'s candidates), normalized."""
    words = [normalize(r) for r in (plan.relations or []) if r]
    words += [normalize(p) for p in of_prefixes(plan.tail or "")]
    return words


def route(plan, scene: Scene | None) -> bool:
    """True when ``scene`` should answer ``plan``: one of its relation words is a relation
    the scene states, or the tail's entity (after its last ' of ') is a thing in it."""
    if scene is None or plan is None:
        return False
    if any(scene.relations(w) for w in spatial_words(plan)):
        return True
    tail = plan.tail or ""
    i = tail.rfind(" of ")
    entity = tail[i + 4:] if i >= 0 else tail
    return scene.where(entity) is not None


__wiring__ = Wiring.STANDALONE
