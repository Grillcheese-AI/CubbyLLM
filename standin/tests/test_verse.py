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
    rep = man.explore(steps=10)
    assert rep["coverage"] >= 0.8, f"exploration must map most of the world: {rep}"
    assert rep["anomalies"] == [], "the resume guard must reject every unoffered direction"
    assert man.walls, "trying outside the offered scope must teach him where walls are"
    assert any(f.startswith("a wall is the") for f in man.world.texts)
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
    rec = s.turn("go explore for 10 steps")
    assert rec["kind"] == "plugin:game" and rec["reply"].startswith("I explored")
    assert man.coverage() >= 0.8
    after = s.turn(q)
    assert after["kind"] == "task" and after["route"]["world"] == "cubbyverse"
    assert after["reply"] == gold, "after exploring, the same question must answer from HIS world"
    # and a derived (never-observed) fact answers too
    cnt = s.turn(f"What is the exit count of {env.start}?")
    assert cnt["reply"] == str(len(env.exits(env.start)))
