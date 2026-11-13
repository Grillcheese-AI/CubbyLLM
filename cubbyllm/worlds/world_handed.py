"""cubbyllm/worlds/world_handed.py -- the items a world handed facts for (order 0054, Cowork's read of the name
statement, 2026-11-11, item 1): a record, never a rule (no answer reads it).

The name statement took its item by matching the answer's label against the name's items, and 31 of 86 statements
listed another item with the same label: "the treasure of the sierra madre" named the novel while the namesake rule
had kept the film. A world now keeps, beside the facts it hands out, the id each one is about:

- ``fact_items(fact)``: the ids of the items a handed fact was read from (two namesakes can state the same sentence);
- ``handed(name)``: the item whose facts a hop on a shared name read, and how it was chosen (``kept one``: the
  namesake rule kept one; ``first_by_id``: several kept, every one giving the same value, the first in id order said;
  ``kept below``: the one item the stores above left; ``hop_pick``: the relation tier). Per thread, cleared with
  ``clear_withheld`` at each question.

Wired: WIRED (`hdc_namesakes.Namesakes` takes it as a mixin; `reasoning.name_statement` reads it).
"""
from __future__ import annotations

__wiring__ = "WIRED"

KEPT = 200_000                                           # the facts whose ids are remembered (oldest dropped)


class Handed:
    """For `Namesakes` (uses its `_pins`, a threading.local)."""

    def _note_ids(self, out: list, item) -> list:
        """Remember the item each of `out`'s facts is about (`item`: one id for all, a list of ids by row, or None)."""
        if item is None:
            return out
        ids = self.__dict__.setdefault("_fact_ids", {})
        for i, (f, _t) in enumerate(out):
            q = item[i] if isinstance(item, list) else item
            if q:
                ids.setdefault(" ".join(f.split()), set()).add(q)
        while len(ids) > KEPT:
            ids.pop(next(iter(ids)))
        return out

    def fact_items(self, fact: str) -> list:
        """The ids of the items this world read `fact` from, in id order ([] when it never handed it)."""
        from .hdc_relation import qnum
        return sorted(self.__dict__.get("_fact_ids", {}).get(" ".join((fact or "").split()), ()), key=qnum)

    def _hand(self, name: str, item: str, how: str, held: dict) -> list:
        """`held[item]`, the rows a hop reads, recorded as handed for `name`."""
        from cubbyllm.reasoning.planner import normalize
        if getattr(self._pins, "handed", None) is None:
            self._pins.handed = {}
        self._pins.handed[normalize(name or "")] = (item, how)
        return held[item]

    def handed(self, name: str) -> tuple | None:
        """(item, how) for the shared name a hop of this question read, or None."""
        from cubbyllm.reasoning.planner import normalize
        return (getattr(self._pins, "handed", None) or {}).get(normalize(name or ""))


__all__ = ["Handed", "KEPT"]
