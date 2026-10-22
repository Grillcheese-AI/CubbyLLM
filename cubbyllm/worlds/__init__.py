"""cubbyllm/worlds -- the stores Cubby knows from and their routing (order 0031: moved here from standin/).

``base`` holds the store and the routing (``FactStore``, ``Worlds``, ``Knows``, ``CubbyPlugin``, ``route_world``),
re-exported here; the worlds themselves (the wiki world, the hdc world, the books, the history graph, fiction, the
encyclopedia, wikitext, capsules, the self source, the science worlds) and their ingestion are its siblings.

Wired: WIRED (the serve stack mounts its worlds from here).
"""
from .base import CubbyPlugin, FactStore, Knows, Worlds, route_world  # noqa: F401

__wiring__ = "WIRED"
