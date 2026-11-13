"""cubbyllm/worlds/wd_store_crowd.py -- a name many items share, in the hot Wikidata store's world, under the kinds table
(order 0055, step 3, option (a)): the store world's mixin for `hdc_relation.decide` (order 0037) on crowded names.

Full Wikidata puts 50 items under "France", 33 under "Japan", 248 under "Paris". Reading every one of them for each
hop costs seconds, so the world reads none past `NAMESAKE_CAP` (`wd_store_world`). With the kinds table loaded, a
hop on such a name first narrows the items to those the rule could pick, from one `brief` request (each item's
recorded classes) and nothing else:
- an item whose recorded kind carries the hop's relation (``Table.carries``: France the country carries "capital");
- an item with no recorded kind (the rule reads its kind from its own facts, so it is read);
- an item that shares a kind with one of the first (two of a kind compete together).
An item whose recorded kinds carry none of the relation cannot fit, so the rule's outcome is the one it would reach
over every item: the one that fits is read, two that fit are each answered, none that fits asks back. Only the
candidates are read (at most `FIT_MAX`; more: none read). Not narrowed, and not read: a place relation ("country",
"located in": any item placed somewhere competes, whatever its kind, `hdc_relation.PLACING`) and a name of more
than `CROWD_MAX` items. The classes come from `brief`, which gives an item's first three: an item recorded with a
fourth class that carries the relation is missed (an item that fits then counts as the only one; noted, rare).

The asker's choice reads the item chosen whatever the name's count (`item_facts`).

Wired: WIRED (`cubbyllm/worlds/wd_store_world.WdStoreWorld`'s base).
"""
from __future__ import annotations

from cubbyllm.reasoning.planner import normalize

from .hdc_relation import PLACING

__wiring__ = "WIRED"

CROWD_MAX = 400                                          # items under one name narrowed at all
FIT_MAX = 25                                             # candidates read after the narrowing


class Crowded:
    """For `WdStoreWorld` (uses its `ask`, `_labels`, `_tiers`, `_entity`, `_kinds`, `_pins`, NAMESAKE_CAP)."""

    def _candidates(self, ids: list, rels: set) -> list | None:
        brief: dict = {}
        for i in range(0, len(ids), 200):
            brief.update(self.ask({"op": "brief", "ids": ids[i:i + 200]}) or {})
        types = {q: (brief.get(q) or {}).get("types", []) for q in ids}
        labs = self._labels({t for ts in types.values() for t in ts})
        kinds = {q: {normalize(labs[t]) for t in ts if t in labs} for q, ts in types.items()}
        carry = {q for q, ks in kinds.items() if any(self._kinds.carries(k, r) for k in ks for r in rels)}
        shared = set().union(*(kinds[q] for q in carry)) if carry else set()
        keep = [q for q in ids if q in carry or not kinds[q] or kinds[q] & shared]
        return keep if len(keep) <= FIT_MAX else None

    def hop_rows(self, name: str, keep, rel: str | None = None) -> list:
        key = normalize(name or "")
        if self._kinds is None or not rel or self._pin(key):
            return super().hop_rows(name, keep, rel)
        labels, aliases = self._tiers(key)
        ids = labels if len(labels) > self.NAMESAKE_CAP else aliases if not labels and len(aliases) > self.NAMESAKE_CAP else None
        if ids is None:
            return super().hop_rows(name, keep, rel)    # not crowded: every item read, as for any name
        rels = self._world_rels(rel)
        if not rels or rels & PLACING or len(ids) > CROWD_MAX:
            return []                                    # (not narrowed: module doc)
        cand = self._candidates(ids, rels)
        if not cand:
            return []
        old = getattr(self._pins, "crowd", None)
        self._pins.crowd = {**(old or {}), key: cand}
        try:
            return super().hop_rows(name, keep, rel)
        finally:
            self._pins.crowd = old

    def namesakes(self, name: str) -> dict:
        """A crowded name's candidates during its hop (even one: the name is shared, the others cannot fit)."""
        cand = (getattr(self._pins, "crowd", None) or {}).get(normalize(name or ""))
        if cand is None:
            return super().namesakes(name)
        return {q: self._triples(rows, q) for q in cand for _s, rows, _k in [self._entity(q)] if rows}

    def item_facts(self, name: str, item: str) -> list:
        """One item's facts when `name` stands for it, however many items share the name (the asker chose it)."""
        got = super().item_facts(name, item)
        if got:
            return got
        labels, aliases = self._tiers(normalize(name or ""))
        return self._triples(self._entity(item)[1], item) if item in labels or item in aliases else []
