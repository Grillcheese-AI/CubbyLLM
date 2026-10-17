"""A scene read two ways (cubbyllm/spatial/scene.py): exactly, for the walk and the VM, and
through its one Hilbert-space vector, for the trunk -- and the ask loop learning spatial facts
from it like any other source.

The vector reading's agreement with the exact one is a capacity, measured here and printed:
d = 1024 (the trunk's width), 16 things in a 32 x 32 x 4 box, read back over every cell of the
box. The exact reading is what the VM verifies; the vector is what the model is shown.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from cubbyllm.spatial import hilbert
from cubbyllm.spatial.scene import DIRECTIONS, Scene
from cubbyllm.spatial.ssp import SSP

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT / "validation") not in sys.path:
    sys.path.insert(0, str(ROOT / "validation"))


def village() -> Scene:
    sc = Scene(SSP(d=256, scale=1.0, seed=1))
    sc.place("the well", (10, 7, 0), "water")
    sc.place("Oak", (11, 7, 0), "tree")
    sc.place("the forge", (10, 9, 0), "building")
    sc.place("hut", (14, 7, 0), "building")
    sc.place("lantern", (10, 7, 1), "light")
    sc.region("the square", (8, 5, 0), (12, 9, 1))
    return sc


def test_the_exact_reading_is_plain_geometry():
    sc = village()
    assert sc.where("The Well") == (10, 7, 0) and sc.at((11, 7, 0)) == "oak" and sc.at((0, 0, 0)) is None
    assert sc.adjacent("well") == {"east": "oak", "above": "lantern"}
    assert sc.offset("well", "east", 3) == (13, 7, 0) and sc.offset("nobody", "east") is None
    assert sc.nearest("well", "building") == "forge" and sc.nearest("well") == "lantern"
    assert sc.nearest("hut", "tree") == "oak" and sc.nearest("well", "dragon") is None
    inside = sc.within((8, 5, 0), (12, 9, 1))
    assert set(inside) == {"well", "lantern", "oak", "forge"}
    assert inside == [sc.at(tuple(c)) for c in hilbert.order([sc.where(n) for n in inside])]   # along the curve
    assert sc.within((20, 20, 0), (30, 30, 3)) == []
    assert sc.regions_of("oak") == ["square"] and sc.regions_of("hut") == []
    assert Scene.distance((10, 7, 0), (14, 9, 2)) == 8
    assert set(DIRECTIONS) == {"east", "west", "north", "south", "above", "below"}


def test_a_path_is_a_shortest_walk_over_free_cells_to_a_thing():
    sc = village()
    walk = sc.path((10, 8, 0), (14, 7, 0))
    assert walk[0] == (10, 8, 0) and walk[-1] == (14, 7, 0) and len(walk) == 6
    for a, b in zip(walk, walk[1:]):
        assert Scene.distance(a, b) == 1 and (b == (14, 7, 0) or sc.at(b) is None)
    walls = Scene(SSP(d=64))
    for y in range(-2, 3):
        walls.place(f"wall {y}", (1, y, 0))
    walls.free = lambda c: c not in walls.cells and -2 <= c[1] <= 2 and c[2] == 0
    assert walls.path((0, 0, 0), (2, 0, 0)) is None                      # fenced off
    walls.remove("wall 2")
    assert len(walls.path((0, 0, 0), (2, 0, 0))) == 7                    # round the gap


def test_moving_a_thing_moves_it_in_the_vector_too():
    sc = village()
    v0 = sc.vector.copy()
    sc.place("oak", (20, 20, 0), "tree")
    assert sc.where("oak") == (20, 20, 0) and sc.at((11, 7, 0)) is None
    sc.place("oak", (11, 7, 0), "tree")
    assert np.abs(sc.vector - v0).max() < 1e-9                           # back where it was, exactly
    with pytest.raises(ValueError):
        sc.place("other", (11, 7, 0))


def test_serialize_reads_the_box_along_the_curve_with_each_things_cell():
    sc = village()
    cells, names = zip(*sc.serialize((8, 5, 0), (14, 9, 1)))
    assert set(names) == {"well", "lantern", "oak", "forge", "hut"} and len(names) == 5
    assert [sc.where(n) for n in names] == list(cells)
    assert list(names) == sc.within((8, 5, 0), (14, 9, 1)) and names[-1] == "hut"   # the far hut comes last


def test_the_vector_reading_agrees_with_the_exact_one_at_the_trunks_width():
    rng = np.random.default_rng(0)
    n, trials, hits_at, hits_where, hits_box = 16, 4, 0, 0, 0
    for t in range(trials):
        sc = Scene(SSP(d=1024, scale=1.0, seed=t))
        cells: set = set()
        while len(cells) < n:
            cells.add(tuple(int(v) for v in rng.integers([0, 0, 0], [32, 32, 4])))
        for i, c in enumerate(sorted(cells)):
            sc.place(f"thing {i}", c)
        for i, c in enumerate(sorted(cells)):
            hits_at += sc.recall_at(c)[0] == f"thing {i}"
            hits_where += sc.recall_where(f"thing {i}")[0] == c
        box = ((4, 4, 0), (20, 20, 3))
        hits_box += set(sc.recall_within(*box)) == set(sc.within(*box))
    at, where, within = hits_at / (n * trials), hits_where / (n * trials), hits_box / trials
    print(f"vector reading, d=1024, {n} things in 32x32x4: at {at:.3f}, where {where:.3f}, within {within:.2f}")
    assert at >= 0.95 and where >= 0.95 and within >= 0.75
    assert sc.recall_at((31, 31, 3))[1] < 0.5 or sc.at((31, 31, 3)) is not None  # an empty cell reads weakly
    assert Scene().recall_at((0, 0, 0)) == (None, 0.0) and Scene().recall_where("x") == (None, 0.0)


def test_the_facts_are_the_loops_template_and_follow_the_relations_asked():
    sc = village()
    facts = sc.facts("The Well")
    assert facts[0] == "10 7 0 is the position of well"
    assert "oak is the east neighbour of well" in facts and "lantern is the above neighbour of well" in facts
    assert "forge is the nearest building of well" in facts and "square is the region of well" in facts
    assert sc.facts("nobody") == []
    assert sc.facts("well", relations=["east neighbour"])[0] == "oak is the east neighbour of well"
    assert sc.relations("east neighbour") == ["east neighbour"] and sc.relations("east") == ["east neighbour"]
    assert sc.relations("nearest building") == ["nearest building"] and sc.relations("mother") == []
    assert sc.kind("position") == "place" and sc.kind("east neighbour") == "thing"
    assert sc.wordings("east neighbour") == ["east neighbour", "east of"]


def test_the_ask_loop_learns_a_spatial_fact_from_the_scene_and_the_vm_verifies_it():
    """The search-and-learn loop with the scene as its source: a refusal becomes a fetch of
    the exact reading, gated and stored with provenance, walked again and verified; a second
    question chains a learned neighbour to its position, fetching only what is missing."""
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from test_pipeline_plan_refusal import faithful_vm
    from test_search_learn import LookupStore

    sc = village()
    store, known = LookupStore([]), StoreRelations([])
    run = lambda q: learn_and_answer(q, lambda _q, _k: [], faithful_vm([]), store=store, known=known,  # noqa: E731
                                     source=sc, tau_vm=0.5)
    r = run("What is the east neighbour of well?")
    assert not r.first.verified and r.first.reason == "unknown_relation"         # an empty store: refused
    assert list(r.entities) == ["well"] and r.fetched == len(sc.facts("well"))   # one round, the well's facts
    assert {p.status for p in r.learned} == {"accepted"} and {p.source for p in r.learned} == {"scene"}
    assert r.result.verified and r.result.answer == "oak"
    assert "oak is the east neighbour of well" in store and "10 7 0 is the position of well" in store
    r2 = run("What is the position of the east neighbour of well?")
    assert r2.result.verified and r2.result.answer == "11 7 0"
    assert list(r2.entities) == ["oak"] and r2.fetched == len(sc.facts("oak"))  # only the hop that was missing
    r3 = run("What is the east neighbour of well?")
    assert r3.first.verified and r3.fetched == 0                                 # learned once, then known
