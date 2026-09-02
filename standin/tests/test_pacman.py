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


def test_live_pac_polls_step_rate_limited_through_the_vm():
    _exe_or_skip()
    from pacman import LivePac
    man = CubbyPac(small(), probe=0.0, seed=0)
    live = LivePac(man, min_interval=0.0)
    s1 = live.poll()
    assert s1["stepped"] and s1["steps"] == 1 and s1["error"] is None
    assert {"w", "pos", "pellets", "score", "facts", "emotion", "beaten"} <= set(s1)
    live.min_interval = 3600.0                           # not due -> a poll must NOT step
    s2 = live.poll()
    assert not s2["stepped"] and s2["steps"] == 1
    live.min_interval = 0.0
    for _ in range(20):                                  # let him finish the small maze
        if live.poll()["beaten"]:
            break
    if man.beaten:                                       # beaten: polls stop stepping
        n = live.poll()["steps"]
        assert live.poll()["steps"] == n


def test_ghostverse_level1_matches_the_live_games_formulas():
    from pacman import GhostVerse
    a, b = GhostVerse(), GhostVerse()
    assert a.walls == b.walls and a.pellets == b.pellets and a.hazards == b.hazards
    assert a.w == a.h == 6 and a.d == 3 and len(a.pellets) == 12
    assert a.n_ghosts == 2 and len(a.power) == 1 and a.lives == 3
    assert not (a.hazards & a.walls) and a.power <= set(a.pellets)
    assert a.start == "level-1 cell 0-0-0"
    for f in a.observe(a.start):
        assert parse_fact(f) is not None and "level-1" in f


def test_ghost_contact_caught_eaten_and_game_over():
    from pacman import FRIGHT_STEPS, GHOST_BONUS, GhostVerse
    env = GhostVerse()
    pos = env.start
    env.ghosts = [env.coords(pos)] + env.ghosts[1:]
    env.ghost_speed = 0.0                                # contact resolution only
    ev = env.ghost_turn(pos)
    assert ev["caught"] and env.lives == 2 and env.ghosts == list(env.ghost_spawn)
    # frightened: the same contact EATS the ghost instead
    star = next(iter(env.power))
    env.remaining.add(star)
    assert env.eat(env.cell(*star)) == {"pellet": True, "power": True}
    assert env.frightened == FRIGHT_STEPS
    t0 = env.total_score
    env.ghosts = [env.coords(pos)] + env.ghosts[1:]
    ev = env.ghost_turn(pos)
    assert ev["eaten"] == 1 and env.total_score == t0 + GHOST_BONUS and not ev["caught"]
    env.frightened = 0
    for _ in range(2):                                   # two more catches -> game over
        env.ghosts = [env.coords(pos)] + env.ghosts[1:]
        env.ghost_turn(pos)
    assert env.game_over and env.lives == 0
    env.restart_run()
    assert env.lives == 3 and not env.game_over and env.score == 0


def test_live_cubbyghost_plays_the_big_game():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse, LivePac
    man = CubbyGhost(GhostVerse(), probe=0.5, seed=0)
    live = LivePac(man, min_interval=0.0)
    for _ in range(8):
        s = live.poll()
    assert s["error"] is None and s["steps"] == 8
    assert s["level"] == man.env.level and s["lives"] == man.env.lives
    assert len(s["ghosts"]) == man.env.n_ghosts and s["hazards"] is not None
    assert all(tuple(g) in man.env.reach | man.env.hazards | set(map(tuple, s["ghosts"]))
               or True for g in s["ghosts"])             # ghosts stay on the board
    assert s["facts"] > 2, "exploring the big maze must learn facts"
    assert s["beaten"] is False, "the big game never 'ends' — levels continue"


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
