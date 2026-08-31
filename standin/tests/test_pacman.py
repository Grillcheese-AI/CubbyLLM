"""Pins for cubby-man in the pac maze (standin/pacman.py): the ported maze,
eating on arrival, six-direction walls/joins, and the replay render through
pacman_3d.py's own template. Live tests skip without the exe.
Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "validation"), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import serve as sv  # noqa: E402
from pacman import PACMAN_3D, CubbyPac, PacVerse, render_replay  # noqa: E402

from cubbyllm.reasoning.planner import parse_fact  # noqa: E402

from test_serve import ChainEmitter  # noqa: E402


def small():
    return PacVerse(w=3, h=3, d=1, seed=3, wall_p=0.2, n_pellets=3)


def test_maze_is_deterministic_and_observations_are_walkable():
    a, b = PacVerse(), PacVerse()
    assert a.walls == b.walls and a.pellets == b.pellets and a.start == b.start == "cell 0-0-0"
    assert len(a.pellets) == 16 and a.walls, "the seed-7 map: 16 pellets, real walls"
    env = small()
    for f in env.all_facts():
        assert parse_fact(f) is not None, f"observation not template-parseable: {f}"
    for m, nbr in env.exits(env.start).items():
        assert env.coords(nbr) not in env.walls and m in {"right", "left", "up", "down"}


def _exe_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")


def test_live_cubbypac_eats_learns_walls_and_records_the_run():
    _exe_or_skip()
    man = CubbyPac(small(), probe=1.0, seed=0)
    rep = man.explore(steps=12)
    assert man.env.score >= 1, f"exploration must eat at least one pellet: {rep}"
    assert any("is the discovery of" in f for f in man.world.texts), "eating is a learned fact"
    assert man.walls and rep["anomalies"] == [], "unoffered moves are refused and learned"
    # d=1: forward/back are always outside — the walls he learns say so
    assert any(re.search(r"a wall is the (forward|back) neighbor", f) for f in man.walls)
    assert man.traj and set(man.traj[0]) == {"to", "move", "eaten", "score", "remaining"}, \
        "the trajectory must match pacman_3d's replay schema"
    assert any("neighbor of" in f for f in man.derived), "six-direction symmetry joins land"


def test_live_the_pac_arc_through_the_brain():
    _exe_or_skip()
    env = small()
    man = CubbyPac(env, probe=0.0, seed=0)
    s = sv.CubbyServe(ChainEmitter(), sv.FactStore([]), route_tau=0.35)
    s.mount(man)
    rec = s.turn("play pacman for 12")
    assert rec["kind"] == "plugin:pacman" and "pellets" in rec["reply"]
    gold = env.exits(env.start).get("right") or next(iter(env.exits(env.start).values()))
    move = next(m for m, n in env.exits(env.start).items() if n == gold)
    after = s.turn(f"What is the {move} neighbor of {env.start}?")
    assert after["kind"] == "task" and after["route"]["world"] == "pacman"
    assert after["reply"] == gold, "a discovered maze fact must answer through the reasoning cortex"


def test_replay_uses_the_pacman3d_template_with_his_trajectory(tmp_path):
    if not PACMAN_3D.exists():
        pytest.skip("cubbyverse checkout not on this machine")
    env = small()
    traj = [{"to": [1, 0, 0], "move": "right", "eaten": None, "score": 0, "remaining": 3}]
    out = render_replay(env, traj, out=tmp_path / "replay.html")
    page = out.read_text(encoding="utf-8")
    assert "explored live by cubby-man" in page and "/*DATA*/" not in page
    data = json.loads(page.split("const D = ", 1)[1].split(";\n", 1)[0])
    assert data["traj"] == traj and data["total"] == 3 and data["w"] == 3
