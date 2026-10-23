"""A scene as a file (order 0034; cubbyllm/spatial/scene_file.py): a scene read back from its file is the same scene,
exact reading and vector alike; a file that breaks the format is refused whole; a crop is the largest square round the
centre within the thing budget, moved to the origin, its regions cut to it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from cubbyllm.spatial.scene import Scene
from cubbyllm.spatial.scene_file import FORMAT, crop
from cubbyllm.spatial.ssp import SSP


def village() -> Scene:
    sc = Scene(SSP(d=256, scale=1.0, seed=1))
    sc.place("the well", (10, 7, 0), "water")
    sc.place("Oak", (11, 7, 0), "tree")
    sc.place("the forge", (10, 9, 0), "building")
    sc.place("Ada Brook", (12, 7, 0), "villager")
    sc.place("lantern", (10, 7, 1), None)
    sc.region("the square", (8, 5, 0), (12, 9, 1))
    sc.meta = {"exporter": "a test"}
    return sc


def test_a_scene_read_back_from_its_file_is_the_same_scene(tmp_path):
    sc = village()
    d = sc.to_json()
    assert d["format"] == FORMAT and d["meta"] == {"exporter": "a test"}
    path = tmp_path / "v.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    for obj in (d, json.dumps(d), str(path)):
        back = Scene.from_json(obj, SSP(d=256, scale=1.0, seed=1))
        assert back.things == sc.things and back.kinds == sc.kinds and back.regions == sc.regions
        assert np.allclose(back.vector, sc.vector) and back.meta == sc.meta
        assert back.facts("well") == sc.facts("well") and back.to_json() == d


@pytest.mark.parametrize("bad, why", [
    ({"format": "something/2", "things": []}, "not a cubby.scene/1"),
    ({"format": FORMAT, "things": [{"name": "a", "cell": [0, 0]}]}, "three integers"),
    ({"format": FORMAT, "things": [{"name": "a", "cell": [0, 0, 0.5]}]}, "three integers"),
    ({"format": FORMAT, "things": [{"name": "a", "cell": [0, 0, 0]}, {"name": "b", "cell": [0, 0, 0]}]}, "already holds"),
    ({"format": FORMAT, "things": [{"name": "The Well", "cell": [0, 0, 0]}, {"name": "well", "cell": [1, 0, 0]}]},
     r"already things\[0\]"),
])
def test_a_file_that_breaks_the_format_is_refused_whole(bad, why):
    with pytest.raises(ValueError, match=why):
        Scene.from_json(bad)


def test_a_crop_is_the_largest_square_round_the_centre_within_the_budget_moved_to_the_origin():
    sc = Scene(SSP(d=64, seed=0))
    for i in range(-6, 7):                                  # a row of 13 things through the centre, and one far off
        sc.place(f"post {i + 7}", (100 + i, 50, 3), "way")
    sc.place("tower", (130, 50, 3), "building")
    sc.place("bird", (100, 50, 9), None)                    # above the band: never kept
    sc.region("the green", (95, 45, 3), (98, 55, 5))
    cut = crop(sc, (100, 50, 3), max_things=7, side=24, height=3)
    m = cut.meta["crop"]
    assert m["half_width"] == 3 and m["kept"] == 7 and m["left_out"] == 8 and m["shift"] == [97, 47, 3]
    assert min(c[0] for c in cut.cells) == 0 and cut.where("post 7") == (3, 3, 0)
    assert cut.regions == {"green": ((0, 0, 0), (1, 6, 2))}              # cut to the box, moved with it
    wide =crop(sc, (100, 50, 3), max_things=100, side=24)
    assert wide.meta["crop"]["half_width"] == 11 and "tower" not in wide.things and len(wide.things) == 13
