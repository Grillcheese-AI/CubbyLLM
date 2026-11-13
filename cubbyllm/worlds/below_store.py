"""cubbyllm/worlds/below_store.py -- order 0066, check 2's reach below the store (Cowork's answer to question 2,
2026-11-11), under ``CB_ORIGINAL_ASIDE=1`` (off by default; off, nothing changes).

An item the Wikidata store set aside for a name, in this question (a work that points at another of the name's works:
`wd_store_names.Named._step_aside`), is no candidate for that name in any world below the store either. The worlds
below it that use the store's ids (Wikidata5M) read the name over the items the store kept: one kept item, its rows;
none, nothing; several, none read and the holders noted, as a shared name is today. Facts about the kept items are
still stated there. "The Gospel According to John" stood for the commentary alone in Wikidata5M, and answered
"D. A. Carson" when the store asked about the name.

Wired: WIRED (``hdc_store.Stores.__init__`` wires it; ``hdc_namesakes.Namesakes.hop_rows`` reads it).
"""
from __future__ import annotations

__wiring__ = "WIRED"


def wire(stores) -> None:
    """Hand each read-only world the set-asides of the stores above it (a world that sets aside is not filtered)."""
    for i, w in enumerate(stores.others):
        ups = [o for o in stores.others[:i] if hasattr(type(o), "set_aside")]
        if ups and not hasattr(type(w), "set_aside"):
            w._set_aside_above = lambda name, ups=ups: set().union(*(u.set_aside(name) for u in ups))


def rows(world, name: str, keep):
    """The rows `world`'s hop reads for `name` over the items the stores above kept, or None (nothing set aside here:
    the world reads the name as before)."""
    above = getattr(world, "_set_aside_above", None)
    gone = above(name) if above is not None else set()
    if not gone:
        return None
    _tier, by = world._named(name)
    if not set(by) & gone:
        return None
    kept = {q: [(f, t) for f, t in world._triples(r, q) if keep(f, t)] for q, r in by.items() if q not in gone}
    if len(kept) == 1:
        return world._hand(name, next(iter(kept)), "kept below", kept) if hasattr(world, "_hand") else next(iter(kept.values()))
    holders = [q for q, r in kept.items() if r]
    if holders:
        world._notes()[name] = holders
    return []


__all__ = ["rows", "wire"]
