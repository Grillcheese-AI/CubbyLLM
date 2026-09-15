"""Pins for the toy cubbyverse (standin/verse.py): the world model is
DISCOVERED, walls come from VM rejections, and joins are VM-certified
programs he writes himself. Live tests skip without the exe.
Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "validation"), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import identity as idn  # noqa: E402
import serve as sv  # noqa: E402
from verse import CubbyMan, ToyVerse  # noqa: E402

from cubbyllm.reasoning.planner import parse_fact  # noqa: E402

from test_serve import ChainEmitter  # noqa: E402

F = idn.load_facts()
DK_EN = idn.T(F, "dont_know_line", "en")


def test_toyverse_is_deterministic_and_every_observation_is_walkable():
    a, b = ToyVerse(seed=3), ToyVerse(seed=3)
    assert a.all_facts() == b.all_facts() and a.start == b.start
    for f in a.all_facts():
        assert parse_fact(f) is not None, f"observation not template-parseable: {f}"


def test_ask_labels_are_short_and_distinct_for_near_identical_moves():
    moves = sorted(["north", "combo-aabaa_right_right_down_right_right", "combo-aabaa_right_right_right_down_right",
                    "combo-aabaa_right_right_down_right_down", "knight_up_up_left"])
    labels = [CubbyMan.ask_label(i, m) for i, m in enumerate(moves)]
    assert len(set(labels)) == len(moves) and all(len(l) <= 13 for l in labels), labels
    assert CubbyMan.ask_label(7, "combo-aabaa_right_right_down_right_right") == "07-carrdrr"
    assert CubbyMan.ask_label(7, "combo-aabaa_right_right_down_right_right", salt=1) == "07a-carrdrr", \
        "a re-roll changes every label (a VM decode collision depends on the exact strings)"
    assert "north" not in labels and "up" not in labels, "a raw direction is never an offered label"


def test_cubbyman_starts_knowing_only_the_basics():
    man = CubbyMan(ToyVerse(seed=0), probe=0.0)
    assert man.coverage() < 0.35, "at start he must NOT know the world"
    start_obs = set(man.env.observe(man.env.start))
    assert all(f in man.world for f in start_obs), "he can see where he stands"
    far = [f for f in man.env.all_facts() if f not in start_obs]
    assert all(f not in man.world for f in far), "everything else is undiscovered"


def _exe_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")


def test_live_exploration_learns_the_world_vm_mediated():
    _exe_or_skip()
    man = CubbyMan(ToyVerse(w=3, h=2, seed=0), probe=1.0, seed=0)
    # 14, not 10: since 2026-09-15 the world's SHAPE is something he has to
    # find out rather than be handed, so some of his steps go into walking
    # off the edge and learning where it is. Those steps buy no `coverage`
    # (the boundary is not in the env's all_facts) — the claim is unchanged,
    # it just costs more steps to earn.
    rep = man.explore(steps=14)
    assert rep["coverage"] >= 0.8, f"exploration must map most of the world: {rep}"
    assert rep["anomalies"] == [], "the resume guard must reject every unoffered direction"
    assert all(r["label"].split("-", 1)[1] == CubbyMan.ask_label(0, r["chosen"]).split("-", 1)[1]
               for r in man.log), "each move was chosen through its offered label (whatever the salt)"
    assert man.walls, "walking into something must teach him where it is"
    # this grid has no stone, only an edge — and that is what he learns it as,
    # instead of the mislabelled "a wall" the old VM-guard probe produced
    assert any(f.startswith("the edge is the") for f in man.world.texts)
    assert not any(f.startswith("a wall is the") for f in man.world.texts), \
        "nothing may be learned as stone in a world that has none"
    # his own join programs: VM-computed exit counts + symmetry about
    # places learned before standing in them
    assert rep["counted"] >= 1 and any("exit count of" in f for f in man.derived)
    ec = next(f for f in sorted(man.derived) if "exit count of" in f)
    t = parse_fact(ec)
    assert len(man.env.exits(t.subj)) == int(t.obj), "the VM-computed count must match the env"
    assert any("neighbor of" in f for f in man.derived), "symmetry joins must land"


def test_live_the_demo_arc_cant_answer_then_explores_then_answers():
    _exe_or_skip()
    env = ToyVerse(w=3, h=2, seed=0)
    man = CubbyMan(env, probe=0.0, seed=0)
    # a treasure he has NOT seen yet (any place but the start)
    target = next(p for p in env.pos if p != env.start)
    gold = env.treasure[target]
    q = f"What is the treasure of {target}?"
    s = sv.CubbyServe(ChainEmitter(), sv.FactStore([]), route_tau=0.35)
    s.mount(man)
    before = s.turn(q)
    assert before["reply"] != gold, "before exploring, the answer must not be speakable"
    rec = s.turn("go explore for 14 steps")               # see the note above: the edge costs steps
    assert rec["kind"] == "plugin:game" and rec["reply"].startswith("I explored")
    assert man.coverage() >= 0.8
    after = s.turn(q)
    assert after["kind"] == "task" and after["route"]["world"] == "cubbyverse"
    assert after["reply"] == gold, "after exploring, the same question must answer from HIS world"
    # and a derived (never-observed) fact answers too
    cnt = s.turn(f"What is the exit count of {env.start}?")
    assert cnt["reply"] == str(len(env.exits(env.start)))
