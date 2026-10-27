"""cubbyllm/worlds/aside.py -- one world's rows on one relation set aside for a walk (order 0037, item 3; Nick,
2026-10-27: the live source's value wins over a snapshot's).

``aside(stores, world, rel)``: inside, the merged store's hop (``hdc_store._Index.hop``) serves none of `world`'s rows
whose relation is `rel`, so a walk that would have read the snapshot's value stalls there and asks the source as any
walk does (``reasoning.ask_source_first``). Per thread. The world itself is untouched, and ``in`` / ``held_by`` still
see the rows: the gate calls the source's restatement a duplicate the world behind holds, and the first world takes it
(``learn.learn_and_answer``), so the walk's next turn reads it there.

Wired: WIRED (``hdc_store.Stores`` / ``_Index.hop``; ``reasoning.ask_source_first``).
"""
from __future__ import annotations

import contextlib

from cubbyllm.reasoning.planner import normalize

__wiring__ = "WIRED"


@contextlib.contextmanager
def aside(stores, world, rel: str):
    """Inside, `world`'s rows on `rel` are not served by `stores`' hop (per thread)."""
    slot = stores._aside
    old = getattr(slot, "v", None)
    slot.v = (world, normalize(rel))
    try:
        yield
    finally:
        slot.v = old


def kept(stores, world, rows) -> list:
    """`world`'s hop rows, less those set aside for the walk under way (none, outside `aside`)."""
    a = getattr(getattr(stores, "_aside", None), "v", None)
    if not a or a[0] is not world:
        return rows
    return [(f, t) for f, t in rows if normalize(t.rel) != a[1]]


__all__ = ["aside", "kept"]
