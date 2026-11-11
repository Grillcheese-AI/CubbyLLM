"""cubbyllm/worlds/wd_store_refine.py -- the hot Wikidata store's refinements of order 0037's decision on a shared name
(order 0055, Cowork's reading of option (a)'s audit, approved by Nick 2026-11-10), applied after `hdc_relation.decide`
through `hdc_namesakes.Namesakes.hop_rows`' hook (the Wikidata5M world has none: its rule is unchanged):

1. **A property that fits anything never decides a pick.** "Different from", "said to be the same as", the topic's
   main category, the Commons category and "described by source" say nothing of what kind of thing is meant (hop 42:
   Thespis the opera was picked for holding "different from"). With only such a property to go on, every item fits,
   so none is picked: the loop asks which, or answers each, labelled.
2. **Fictional items step aside when a real item fits.** An item of a fictional class (under "fictional entity",
   Q14897293, by subclass of: `wd_store_cues`) leaves the fitting items when a real one fits the same relation; it
   stays among the name's items, so the closing line names it and the asker can still choose it; when no real item
   fits, it fits as before ("France": the country's capital, the Star Trek location named after it).
3. **One answer when every item agrees.** When two or more items fit and every one holds the same values for the
   hop, the values are said once (hop 8: four Calhoun Townships, all in the United States); the items are listed in
   the decision (`agree`), and the loop records them (`ask_namesakes.after_walk`). No item is picked, so no pick can
   be wrong.

Wired: WIRED (`cubbyllm/worlds/wd_store_world.WdStoreWorld`'s base).
"""
from __future__ import annotations

from cubbyllm.reasoning.planner import normalize

from .hdc_relation import qnum

__wiring__ = "WIRED"

NEUTRAL = (1889, 460, 910, 373, 1343)                    # different from, said to be the same as, topic's main
#                                                          category, Commons category, described by source


class Refined:
    """For `WdStoreWorld` (uses its `_pid_label`, `_fictional`, `_p31`, `_entity`)."""

    def _neutral(self) -> set:
        return {normalize(self._pid_label[p]) for p in NEUTRAL if p in self._pid_label}

    def _refine(self, d: dict, by: dict, held: dict) -> dict:
        rels = set(d.get("rels") or ())
        if rels and rels <= self._neutral():
            d["fit"], d["neutral"] = sorted(by, key=qnum), True     # (1) no pick on a property that fits anything
            return d
        fit = list(d["fit"])
        fiction = [q for q in fit if self._classes(q) & self._fictional]
        if fiction and len(fiction) < len(fit):                      # (2) a real item fits: the fictional step aside
            d["fit"], d["stepped_aside"] = [q for q in fit if q not in fiction], fiction
        fit = d["fit"]
        if len(fit) >= 2 and all(held.get(q) for q in fit):
            if len({frozenset(t.obj for _f, t in held[q]) for q in fit}) == 1:
                d["agree"] = list(fit)                               # (3) every item agrees: said once
        return d

    def _classes(self, q: str) -> set:
        self._entity(q)                                  # (reads the item, which records its classes)
        return self._p31.get(q, set())
