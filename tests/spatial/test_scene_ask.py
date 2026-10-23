"""The scene's own question shapes (order 0035, part A; cubbyllm/spatial/scene_ask.py and Scene.facts).

Pinned: "where is X", "what is n <direction> of X" and "what is at x y z" are read off the question and restated in the
scene's grammar; a name the scene does not hold is no scene question; the scene states the thing n steps away and a
cell's occupant from its exact reading, and nothing for an empty cell; the ask loop answers all three, verified.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests" / "worlds", ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.spatial.scene import Scene  # noqa: E402
from cubbyllm.spatial.scene_ask import scene_ask  # noqa: E402


def village() -> Scene:
    sc = Scene()
    sc.place("the well", (5, 5, 0), "water")
    sc.place("oak", (6, 5, 0), "tree")
    sc.place("hut", (8, 5, 0), "building")
    sc.place("Ada Brook", (5, 7, 0), "villager")
    return sc


def test_the_three_shapes_are_read_off_the_question_and_restated():
    sc = village()
    assert scene_ask("Where is the well?", sc) == ("What is the position of well?", "position", "well")
    assert scene_ask("where's Ada Brook", sc) == ("What is the position of ada brook?", "position", "ada brook")
    assert scene_ask("What is 3 east of the well?", sc) == ("What is the 3 east of well?", "3 east", "well")
    assert scene_ask("what is 1 east of the well?", sc) == ("What is the east neighbour of well?", "east neighbour", "well")
    assert scene_ask("What is 2 steps north of the well?", sc)[1] == "2 north"
    assert scene_ask("What is at 8 5 0?", sc) == ("What is the occupant of 8 5 0?", "occupant", "8 5 0")
    assert scene_ask("what's at 8, 5, 0?", sc)[2] == "8 5 0"
    for q in ("Where is Paris?", "What is 3 east of Paris?", "What is the capital of France?", "What is at the market?"):
        assert scene_ask(q, sc) is None, q


def test_the_scene_states_the_thing_n_steps_away_and_a_cells_occupant():
    sc = village()
    facts = sc.facts("well")
    assert "hut is the 3 east of well" in facts and "ada brook is the 2 north of well" in facts
    assert sc.facts("8 5 0") == ["hut is the occupant of 8 5 0"] and sc.facts("9 9 0") == []
    assert sc.relations("3 east") == ["3 east"] and sc.relations("occupant") == ["occupant"] and sc.relations("1 east") == []
    assert any("is the 7 east of" in f for f in [*sc.facts("well", relations=["7 east"])]) is False   # nothing 7 east


def test_the_ask_loop_answers_all_three_verified_and_refuses_an_empty_cell():
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.question_forms import Forms, TemplateEmitter
    from cubbyllm.worlds.base import FactStore
    from test_books import exact_vm

    class NoSource:
        name = "none"

        def facts(self, entity, **kw):
            return []
    loop = AskLoop(TemplateEmitter(Forms({})), world=FactStore([]), source=NoSource(), lexicon=False, scene=village(),
                   run_fn=exact_vm())
    for q, a in (("Where is the hut?", "8 5 0"), ("What is 3 east of the well?", "hut"), ("What is at 5 7 0?", "ada brook")):
        rec = loop.ask(q)
        assert rec["verified"] and rec["answer"] == a and rec["world"] == "scene", q
    rec = loop.ask("What is at 9 9 0?")
    assert not rec["verified"] and rec["answer"] is None
