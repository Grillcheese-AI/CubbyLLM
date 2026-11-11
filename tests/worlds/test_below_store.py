"""Order 0066, question 2 (CB_ORIGINAL_ASIDE=1): an item the store set aside for a name is no candidate for it in a
world below the store. The store's fixture world is test_wd_store_names's; the world below holds the name for the set
-aside item alone, as Wikidata5M held "The Gospel According to John" for the commentary."""
from __future__ import annotations

import importlib.util
import pathlib
import threading

from cubbyllm.reasoning.planner import normalize
from cubbyllm.worlds import below_store
from cubbyllm.worlds.hdc_namesakes import Namesakes

_spec = importlib.util.spec_from_file_location("_wd_names", pathlib.Path(__file__).with_name("test_wd_store_names.py"))
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)
world = N.world                                         # the fixture


class Below(Namesakes):
    """A read-only world with the store's ids: "Hymn of the Land" is the arrangement (Q8031) alone, "Twin Tale" both."""
    ROWS = {"hymn of the land": {"Q8031": [("Hymn of the Land", "author", "Herbert Arranger")]},
            "song of the sea": {"Q8035": [("Song of the Sea", "author", "Herbert Arranger")],
                                "Q8036": [("Song of the Sea", "author", "D. A. Carson")]}}

    def __init__(self):
        self._pins, self._lock, self.name = threading.local(), threading.Lock(), "below"
        self._by_subj = {k: self._triples(next(iter(by.values()))) for k, by in self.ROWS.items() if len(by) == 1}

    def _named(self, key):
        by = self.ROWS.get(key, {})
        return ("label" if by else "none"), by

    def fact(self, s, r, o):
        return f"{o} is the {r} of {s}"


def read(w, name):
    return {t.obj for _f, t in w.hop_rows(normalize(name), lambda f, t: True, "author")}


def test_a_set_aside_item_is_no_candidate_below_the_store(world, monkeypatch):
    below = Below()
    below_store.wire(type("S", (), {"others": [world, below]})())
    monkeypatch.setenv("CB_ORIGINAL_FIRST", "1")
    N.fresh(world)
    world._tiers("hymn of the land")                      # the store reads the name: the arrangement steps aside
    assert world.decided("hymn of the land") == "Q8030"
    assert read(below, "Hymn of the Land") == {"Herbert Arranger"}                      # the proposal off: as before
    monkeypatch.setenv("CB_ORIGINAL_ASIDE", "1")
    assert read(below, "Hymn of the Land") == set()                                     # the arrangement: no candidate
    N.fresh(world)
    world._tiers("song of the sea")                       # the score steps aside; the painting, unrelated, stays
    assert read(below, "Song of the Sea") == {"D. A. Carson"}                           # the kept item's facts, by id
    assert below_store.rows(below, "twin tale", lambda f, t: True) is None              # nothing set aside: as before
