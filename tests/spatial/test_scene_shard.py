"""The village shard from scene files (``validation/make_scene_shard.py``, order 0035 C): one document's questions.

Pinned: every question is placed (neighbour, offset, at), asked once, and answered from the scene's exact reading at
its address; the answers alternate between a cell that holds something and an empty one; among the things, villagers
and other things alternate while both last; the same seed draws the same questions.
"""
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation", ROOT / "tests" / "spatial"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from make_scene_shard import asked  # noqa: E402
from test_s12_scene_files import village  # noqa: E402


def test_a_documents_questions_are_placed_exact_and_alternate_full_and_empty_then_villagers_and_things():
    sc = village()
    qa = asked(sc, random.Random(11))
    assert qa == asked(sc, random.Random(11)) and len(qa) >= 8
    assert len({(q, cell) for q, _a, cell, _c in qa}) == len(qa)
    for q, a, cell, cls in qa:
        assert q == "What is at here?" or " neighbour of " in q or " of " in q
        assert a == (sc.at(cell) or "nothing")
        assert cls == ("nothing" if sc.at(cell) is None else "name" if sc.kinds[sc.at(cell)] == "villager" else "kind")
    empty = [c == "nothing" for _q, _a, _cell, c in qa]
    assert all(x != y for x, y in zip(empty, empty[1:]))                 # full and empty alternate
    things = [c for _q, _a, _cell, c in qa if c != "nothing"]
    k = min(things.count("name"), things.count("kind"))
    assert k and all(x != y for x, y in zip(things[:2 * k], things[1:2 * k]))   # villagers and things alternate
