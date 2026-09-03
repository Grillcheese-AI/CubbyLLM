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


def test_frontend_is_lifted_verbatim_and_every_patch_applies():
    from pacman import PACMAN_LIVE, load_frontend
    if not PACMAN_LIVE.exists():
        pytest.skip("cubbyverse checkout not on this machine")
    html, missed = load_frontend()
    assert html and missed == [], f"upstream page changed under a patch: {missed}"
    assert "fetch('/pac/state')" in html and "'/pac/assets/" in html
    assert 'id="emocompass"' in html and "drawCompass" in html, "the Plutchik cone comes along"
    assert "cubby_uv_template" in html, "the face textures are the originals"
    assert "world model:" in html and "R-STDP" not in html.split("<body>")[1].split("<script")[0]
    # the console panel rides along under the stage number, polling our event feed
    assert 'id="cubbycon"' in html and "fetch(`/events?since=${since}`)" in html
    assert html.index('id="cubbycon"') > html.index('id="bigstage"'), "the panel is injected after the lifted page"
    assert html.count("</body>") == 1 and html.index('id="cubbycon"') < html.index("</body>")


def test_plan_next_runs_over_his_own_facts_not_the_env():
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0)
    L = "level-1 cell "
    # a map he "learned": start -> A -> B, with a pellet he saw at B
    for f in (f"{L}1-0-0 is the right neighbor of {L}0-0-0",
              f"{L}2-0-0 is the right neighbor of {L}1-0-0",
              f"a pellet is the sighting of {L}2-0-0"):
        man.world.add(f)
    man.sighted.add(f"{L}2-0-0")
    assert man.plan_next() == f"{L}1-0-0", "first step of the path to the seen pellet"
    man._eaten_run.add(f"{L}2-0-0")                     # eaten this run -> frontier instead
    assert man.plan_next() in {f"{L}1-0-0"} | set(man.env.exits(man.place).values())
    man._eaten_run.clear()
    assert man.plan_next(avoid={f"{L}1-0-0"}) != f"{L}1-0-0", "a hunting ghost blocks the path"


def test_pattern_moves_instantiate_slots_over_open_cells():
    from pacman import GhostVerse, ProgramLibrary, _assignments, pattern_name
    assert pattern_name("AAA") == "DASH" and pattern_name("ABC") == "WARP"
    assert pattern_name("AACB") == "COMBO-AACB", "a pattern with no flavor name is still a move"
    asg = list(_assignments("AB"))
    assert len(asg) == 24 and all(a["A"] != a["B"] for a in asg)
    assert not any(a["B"] == {"right": "left", "left": "right", "up": "down", "down": "up",
                              "forward": "back", "back": "forward"}[a["A"]] for a in asg), \
        "a slot pair never maps to opposite directions"
    env = GhostVerse()
    lib = ProgramLibrary()
    lib.add("DASH", "AAA", "program …", "pattern", 0, {"why": "test"})
    moves = env.power_moves(env.start, lib, 100)
    for move, land in moves.items():
        assert move.startswith("dash_") and env.coords(land) not in env.walls | env.hazards
        x, y, z = env.coords(land)
        assert abs(x) + abs(y) + abs(z) == 3, "DASH lands three cells away along one open line"


def test_moves_program_is_one_program_with_a_function_per_active_combo():
    from pacman import ProgramLibrary
    lib = ProgramLibrary()
    lib.add("JUMP", None, "…", "jump", 0, {"why": "stuck"})
    lib.add("KNIGHT", "AAB", "…", "pattern", 0, {"why": "curious"})
    lib.add("COMBO-AABA", "AABA", "…", "pattern", 0, {"why": "curious"})
    lib.add("FLEE#1", None, "…", "tool", 0, {"why": "test"})       # a tool is not a move
    lib.entries["COMBO-AABA"]["retired"] = True                     # retired: no function
    src = lib.moves_program()
    assert src.count("program Moves implements ISolve") == 1
    assert "public function jump()" in src and "public function knight()" in src
    assert "combo_aaba" not in src and "flee" not in src
    assert 'bind frame, H1_MOVES, "2"' in src, "solve() is the catalogue smoke: two active moves"
    assert ProgramLibrary.fn_name("COMBO-AAB") == "combo_aab"


def test_live_a_new_combo_is_certified_as_a_function_of_the_one_program(tmp_path):
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0, seed=1, memory=tmp_path / "programs.json")
    a = man._compose("DASH", "out_of_time", list("AAA"), "AAA", "pattern", "test")
    b = man._compose("KNIGHT", "curious", list("AAB"), "AAB", "pattern", "test")
    assert a == "DASH" and b == "KNIGHT"
    for name in ("DASH", "KNIGHT"):
        e = man.library.entries[name]
        assert e["reasoning"]["function"] == name.lower() and "certified: Moves." in e["reasoning"]["verdict"]
        assert "program Moves implements ISolve" in e["program"]
    assert "public function dash()" in man.library.entries["KNIGHT"]["program"], \
        "the second certification ran inside the program that already held the first"
    cube = tmp_path / "cubbyman_moves.cube"
    assert cube.exists() and "public function knight()" in cube.read_text(encoding="utf-8")
    # editing: a modified combo carries its lineage
    child = man._mutate("curious")
    assert child and man.library.entries[child]["reasoning"].get("parent") in ("DASH", "KNIGHT")
    assert man.library.fn_name(child) in cube.read_text(encoding="utf-8")


def test_consolidate_keeps_the_best_and_retires_the_rest_never_deletes():
    from pacman import ProgramLibrary
    lib = ProgramLibrary()
    lib.add("JUMP", None, "…", "jump", 0, {"why": "stuck"})
    for i in range(10):
        lib.add(f"P{i}", "AAB" + "ABC"[i % 3], "…", "pattern", 0, {"why": "curious"})
        lib.entries[f"P{i}"]["used"] = i
        lib.entries[f"P{i}"]["saved"] = 2 * i
    gone = lib.consolidate(level=1, keep=3)
    assert len(gone) == 7 and all(lib.entries[n]["retired"] for n in gone)
    assert {n for n in lib.names() if n != "JUMP"} == {"P9", "P8", "P7"}, "the most valuable stay active"
    assert len(lib.entries) == 11, "retired, never deleted"
    assert "consolidated after level 1" in lib.entries["P0"]["retired_reason"]


def test_power_moves_ignore_tools_and_retired_entries():
    """Live bug (2026-09-02): the persisted notebook holds forge TOOL entries
    (no pattern) -> power_moves iterated them -> 'NoneType' object is not
    iterable on every /pac poll."""
    from pacman import GhostVerse, ProgramLibrary
    env = GhostVerse()
    lib = ProgramLibrary()
    lib.add("FLEE#1", None, "program Dec0 …", "tool", 0, {"why": "test", "ok": True})
    lib.add("WHERE#1", None, "use vsa; …", "tool", 0, {"why": "test", "ok": True})
    lib.add("DASH", "AAA", "program …", "pattern", 0, {"why": "test"})
    lib.entries["DASH"]["retired"] = True
    assert env.power_moves(env.start, lib, 100) == {}, "tools and retired patterns offer no moves"
    lib.add("COMBO", "AB", "program …", "pattern", 0, {"why": "test"})
    moves = env.power_moves(env.start, lib, 100)
    assert moves and all(m.startswith("combo_") for m in moves)


def test_program_library_persists_a_readable_notebook_and_never_forgets(tmp_path):
    from pacman import ProgramLibrary
    p = tmp_path / "programs.json"
    lib = ProgramLibrary(p)
    lib.add("KNIGHT", "AAB", "use vsa;\nprogram Superpower …", "pattern", 3,
            {"why": "curious", "because": "felt like it", "situation": {"level": 1},
             "rationale": "sampled length 3", "verdict": "certified"})
    lib.note_used("KNIGHT", 2, {"step": 9, "level": 1, "move": "knight_right_right_up", "landed": "pellet"})
    again = ProgramLibrary(p)                            # a restart reloads what he learned
    assert "KNIGHT" in again and again.entries["KNIGHT"]["used"] == 1
    md = p.with_suffix(".md").read_text(encoding="utf-8")
    for needle in ("## KNIGHT", "trigger:** curious", "sampled length 3", "certified",
                   "```cubelang", "knight_right_right_up"):
        assert needle in md, f"the notebook must carry the reasoning trace: {needle}"
    for i in range(12):                                  # crowd the active set with dead patterns
        again.add(f"P{i}", "AAAB", "…", "pattern", 0, {"why": "curious"})
    gone = again.retire(step=100)
    assert gone and again.entries[gone]["retired"] and gone in again.entries, \
        "retired, not deleted — he does not forget what he learned"
    assert gone not in again.names()


def test_live_he_generates_a_program_with_a_reasoning_trace():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0, seed=1)
    name = man._propose("out_of_time")
    assert name and name in man.library
    e = man.library.entries[name]
    r = e["reasoning"]
    assert r["why"] == "out_of_time" and "time budget" in r["because"]
    assert r["verdict"].startswith("certified") and "sampled length" in r["rationale"]
    assert r["situation"]["level"] == 1 and "program Moves implements ISolve" in e["program"]
    assert any(f"is the recipe of {name}" in f for f in man.world.texts), "the recipe is a learned fact"
    assert man.env.power_moves is not None and name in man.powers


KERNEL = ("program {cls} implements ISolver {{\n    type Input = str;\n    type Output = quantity;\n"
          "    @external\n    public function parse(raw: str): Input {{ return raw; }}\n"
          "    public pure function verify(input: Input, output: Output): bool {{ return true; }}\n"
          "    @external\n    public function solve(input: Input): Output {{\n"
          "        create r : quantity;\n        assign r = {value};\n        query r;\n        return r;\n    }}\n}}\n")


class KernelEmitter(ChainEmitter):
    """A fake trunk that answers the game's task families: a decision prompt
    -> 90/30 by actually evaluating 'exceeds'; a compare prompt -> the max;
    chain prompts as ChainEmitter. `lie=True` returns wrong numbers so the
    certification path is exercised."""
    name = "fake-kernel"

    def __init__(self, lie: bool = False) -> None:
        self.lie = lie

    def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
        from forge import numbers
        if "flag the sample if the reading is above" in prompt:
            a, b = numbers(prompt)[:2]
            v = 90 if a > b else 30                      # any of the trained classes would do
            return KERNEL.format(cls="Dec0", value=(30 if v == 90 else 90) if self.lie else v)
        if "report the higher measurement" in prompt:
            a, b = numbers(prompt)[:2]
            return KERNEL.format(cls="Cmp0", value=(min(a, b) if self.lie else max(a, b)))
        return super().emit(prompt, max_new_tokens, system, prefix)


def test_live_he_writes_live_decision_programs_and_acts_only_on_certified_ones():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0, seed=0)
    s = sv.CubbyServe(KernelEmitter(), sv.FactStore([]), route_tau=0.35)
    s.mount(man)
    assert man._forge_flee(near=1) is True and man._forge_flee(near=4) is False
    man._last_forge = -99
    assert man._forge_safer_exit({"right": 3, "up": 5, "down": 1}) == "up"
    tools = {n: e for n, e in man.library.entries.items() if e["kind"] == "tool"}
    assert len(tools) == 3 and all(e["reasoning"]["ok"] for e in tools.values())
    assert any("flag the sample" in e["reasoning"]["rationale"] for e in tools.values())
    assert man.forge.acceptance() == {"decision": 1.0, "compare": 1.0}
    md = man.library.notebook()
    assert "## FLEE#1" in md and "PASS" in md and "certified" in md
    # a lying trunk: the VM runs the program, the certification rejects it, he falls back to the rule
    liar = CubbyGhost(GhostVerse(), probe=0.0, seed=0)
    s2 = sv.CubbyServe(KernelEmitter(lie=True), sv.FactStore([]), route_tau=0.35)
    s2.mount(liar)
    assert liar._forge_flee(near=1) is None
    liar._last_forge = -99
    assert liar._forge_safer_exit({"right": 3, "up": 5}) is None
    bad = [e for e in liar.library.entries.values() if e["kind"] == "tool"]
    assert bad and all(not e["reasoning"]["ok"] for e in bad)
    assert "REJECTED" in liar.library.notebook()
    assert liar.forge.acceptance() == {"decision": 0.0, "compare": 0.0}


def test_live_orientation_task_checks_his_trunk_against_his_own_map():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0, seed=0)
    s = sv.CubbyServe(KernelEmitter(), sv.FactStore([]), route_tau=0.35)
    s.mount(man)
    man._forge_orientation()
    where = [e for n, e in man.library.entries.items() if n.startswith("WHERE#")]
    assert len(where) == 1 and where[0]["reasoning"]["ok"], where
    assert "Facts:" in where[0]["reasoning"]["rationale"]


def test_emotion_maps_to_the_plutchik_compass():
    from neurochem import Neurochemistry
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0)
    man.chem = Neurochemistry()
    for _ in range(6):
        man.chem.update(threat=1.0)
    emo = man.emotion()
    assert emo["angle"] == 90 and emo["name"] in {"apprehension", "fear", "terror"}
    assert emo["color"].startswith("#") and 0 <= emo["intensity"] <= 1.2


def test_being_caught_raises_fear_learned_and_hormonal():
    from pacman import CubbyGhost, GhostVerse
    from neurochem import Neurochemistry
    man = CubbyGhost(GhostVerse(), probe=0.0)
    man.chem = Neurochemistry()
    fear0, r0 = man.fear, man.danger_radius
    ne0, c0 = man.chem.noradrenaline, man.chem.cortisol
    man._on_caught()
    assert man.fear == fear0 + 0.7, "the learned fear climbs (pacman_live's +0.7)"
    assert man.chem.noradrenaline > ne0 + 0.3, "the shock: noradrenaline surges"
    assert man.chem.cortisol > c0, "and the slow cortisol moves"
    assert man.emotion()["name"] in ("apprehension", "fear", "terror"), man.emotion()
    assert man._mood() in ("(uneasy) ", "(scared) ", "(terrified) ", "(on edge) ")
    man._on_caught()
    assert man.danger_radius > r0, "two catches: he keeps a wider berth from now on"


def test_traps_are_one_per_level_cumulative_ghost_only_and_used_wisely():
    from pacman import TRAP_BONUS, CubbyGhost, GhostVerse
    env = GhostVerse()
    assert env.mines_left == 1, "one trap at level 1"
    env._start_level(2)
    assert env.mines_left == 2, "unused: it carries over"
    env.place_mine(env.start)
    assert env.mines_left == 1 and env.coords(env.start) in env.mines
    assert not env.place_mine(env.start), "no double trap on one cell"
    # a ghost stepping on it is sent home; the trap is spent; ghosts only
    env.ghosts = [env.coords(env.start)] + env.ghosts[1:]
    env.ghost_speed = 0.0
    t0 = env.total_score
    env.mines.add(env.ghosts[0])                          # (re-arm on the ghost's cell after the level reset)
    cubby_far = env.cell(*next(c for c in sorted(env.reach) if c not in env.ghost_spawn and c != env.coords(env.start)))
    out = env.ghost_turn(cubby_far)
    assert out["trapped"] == 1 and env.ghosts[0] == env.ghost_spawn[0] and not env.mines
    assert env.total_score == t0 + TRAP_BONUS and not out["caught"]
    env._start_level(3)
    assert env.mines_left == 2, "level 3: the unused one + one new"
    # wisely: only under real threat
    man = CubbyGhost(GhostVerse(), probe=0.0)
    here = man.env.coords(man.place)
    man.env.ghosts = [(here[0] + 5, here[1] + 5, here[2])]
    assert not man._mine_wise(), "a far ghost is not worth a trap"
    man.env.ghosts = [(here[0] + 1, here[1], here[2])]
    assert man._mine_wise(), "an adjacent hunting ghost is"
    man.env.frightened = 5
    assert not man._mine_wise(), "frightened ghosts flee: no trap"


def test_live_a_trap_is_dropped_on_the_way_out_and_thought_aloud():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0, seed=0)
    env = man.env
    here = env.coords(man.place)
    env.ghosts = [(here[0] + 1, here[1], here[2])] + env.ghosts[1:]
    env.ghost_speed = 0.0
    rec = man.step()
    assert here in env.mines or env.mines_left == 0, "the trap went down on the cell he left"
    assert any(e["kind"] == "mine" for e in man.log_events()) if hasattr(man, "log_events") else True
    assert any(f"a trap is the marker of" in f for f in man.world.texts)
    r = man.resp()
    assert r["mines_left"] == 0 and r["mines"] and r["mined"] == list(here)


def test_energy_costs_moves_combos_more_and_only_rest_or_pellets_bring_it_back():
    from pacman import (JUMP_COST, MOVE_COST, PELLET_ENERGY, REST, REST_BELOW, CubbyGhost, GhostVerse,
                        ProgramLibrary, pattern_cost)
    assert pattern_cost("AAB") == 6 and pattern_cost("AAAAA") == 10 and pattern_cost(None) == JUMP_COST
    man = CubbyGhost(GhostVerse(), probe=0.0)
    assert man.move_cost("right") == MOVE_COST and man.move_cost(REST) == 0 and man.move_cost("jump_up") == JUMP_COST
    man.library.add("KNIGHT", "AAB", "…", "pattern", 0, {"why": "t"})
    assert man.move_cost("knight_right_right_up") == 6, "a combo costs more than a step"
    env = man.env
    env.energy = 5
    assert not any(m.startswith("knight_") for m in env.power_moves(env.start, man.library, env.energy)), \
        "an unaffordable combo is not offered"
    env.energy = 100
    # rest is offered only when tired AND safe; it restores energy
    env.ghosts = []
    env.energy = REST_BELOW - 1
    assert REST in man.candidate_moves(env.exits(man.place)) and man._rest_wanted()
    env.ghosts = [env.coords(man.place)]                 # a ghost on top of him: not safe
    assert REST not in man.candidate_moves(env.exits(man.place))
    env.ghosts = []
    env.energy = 90
    assert REST not in man.candidate_moves(env.exits(man.place)), "hysteresis: rested enough, back to work"
    # eating gives a little back
    p = next(iter(env.remaining))
    env.energy = 50
    man.on_arrive(env.cell(*p))
    assert env.energy == 50 + PELLET_ENERGY


def test_live_tired_cubby_rests_in_a_safe_spot_and_energy_climbs():
    _exe_or_skip()
    from pacman import REST, REST_BELOW, REST_GAIN, CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0, seed=0)
    man.env.ghosts = []                                  # nobody hunting: everywhere is safe
    man.env.ghost_spawn = []
    man.env.energy = REST_BELOW - 5
    e0 = man.env.energy
    rec = man.step()
    assert rec["chosen"] == REST and man.env.energy == e0 + REST_GAIN and man.traj[-1]["move"] == REST
    assert man.traj[-1]["to"] == list(man.env.coords(man.place)), "resting does not move him"
    steps = 0
    while man._rest_wanted() and steps < 12:
        man.step(); steps += 1
    assert man.env.energy >= 60 and not man._rest_wanted()
    e1 = man.env.energy
    man.step()
    assert man.env.energy <= e1, "a move costs; nothing regenerates by itself"


def test_mood_prefix_follows_the_compass_not_a_dopamine_switch():
    from pacman import CubbyGhost, GhostVerse
    from neurochem import Neurochemistry
    import identity as idn
    F = idn.load_facts()
    man = CubbyGhost(GhostVerse(), probe=0.0)
    man.chem = Neurochemistry()
    assert man._mood() == "", "resting: no mood word"
    seen = set()
    for signals in ({"threat": 1.0}, {"novelty": 1.0, "valence": 0.8}, {"valence": -0.9}, {"social": 1.0, "valence": 0.5}):
        man.chem = Neurochemistry()
        for _ in range(6):
            man.chem.update(**signals)
        m = man._mood()
        seen.add(m)
        assert m == "" or (m.startswith("(") and idn.voice_ok(m, F)), m
    assert len(seen) >= 3, f"different states must read differently: {seen}"
    for word_en, word_fr in CubbyGhost._MOOD.values():
        assert idn.voice_ok(word_en, F) and idn.voice_ok(word_fr, F)
    man.chem = Neurochemistry()
    for _ in range(80):
        man.chem.update(threat=1.0)                      # sustained stress -> cortisol climbs
    assert man._mood() in ("(on edge) ", "(scared) ", "(terrified) ", "(uneasy) ")


def test_routine_learning_reads_calm_a_burst_reads_curious():
    """The ODE's zero-input dopamine equilibrium is ~0.51 by construction (a
    baseline drive of 0.15), which is why a dopamine switch at 0.55 said
    'excited' on everything. The claim that matters: a steady trickle of
    learned facts leaves NO mood word; a burst above expectation does."""
    from pacman import CubbyGhost, GhostVerse
    from neurochem import Neurochemistry

    def feed(man, new):                                  # verse.py's habituated novelty feed, verbatim
        ema = getattr(man, "_expect_new", None)
        expected = 1.0 if ema is None else ema
        surprise = max(0.0, new - expected) / (expected + 1.0)
        man._expect_new = new if ema is None else 0.8 * ema + 0.2 * new
        man.chem.update(novelty=min(1.0, 0.7 * surprise), valence=0.1 * min(1, new))

    man = CubbyGhost(GhostVerse(), probe=0.0)
    man.chem = Neurochemistry()
    for _ in range(12):
        feed(man, 3)                                     # the same 3 facts every step
    assert man._mood() == "", f"routine learning must not read as excitement: {man._mood()!r} {man.emotion()}"
    for _ in range(3):
        feed(man, 14)                                    # a burst: a new wing of the maze
    assert man.chem.affect_arousal > 0.3 and man.emotion()["name"] != "calm", man.emotion()


def test_thoughts_are_first_person_voice_safe_and_bilingual():
    from pacman import CubbyGhost
    import identity as idn
    F = idn.load_facts()
    cases = {"plan": dict(to="level-1 cell 2-0-0", goal="pellet"), "flee": dict(ghost_distance=1, radius=2),
             "caught": dict(place="level-1 cell 3-3-0"), "probe": dict(tried="up"), "eat": dict(score=3, total=12),
             "power": dict(steps=14), "program": dict(name="KNIGHT", pattern="AAB"),
             "modify": dict(parent="DASH", child="SPRINT", edit="appended a slot"),
             "superpower_move": dict(name="DASH", saved=2), "out_of_time": dict(level=1, attempt=2, learned="DASH"),
             "level_up": dict(cleared=1, next=2), "forge": dict(ok=True, answer="flee"), "idle": dict(to="level-1 cell 0-0-0")}
    for lang in ("en", "fr"):
        for kind, d in cases.items():
            for pick in range(3):                        # every phrasing of every event, both languages
                line = CubbyGhost.think(kind, lang, "", pick=pick, **d)
                assert line and idn.voice_ok(line, F), (kind, lang, pick, line)
    assert CubbyGhost.think("flee", "en", "(nervous) ", pick=0, ghost_distance=1, radius=2) \
        == "(nervous) A ghost 1 cells away — too close for my nerves (2), I'm running."
    assert "pastille" in CubbyGhost.think("plan", "fr", "", pick=0, to="x", goal="pellet")
    lines = {CubbyGhost.think("eat", "en", "", pick=i, score=3, total=12) for i in range(3)}
    assert len(lines) == 3, "a repeated event is not a repeated sentence"
    assert "?" in CubbyGhost.think("mine", "en", "", pick=0), "a missing field renders as ? instead of raising"


def test_live_a_step_thinks_out_loud_and_the_model_may_phrase_it():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse

    class Phraser(ChainEmitter):
        """Rephrases a thought when asked; keeps the numbers/names."""
        def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
            if prompt.startswith("Say this in your own words"):
                line = prompt.split(": ", 1)[1]
                return "Hmm - " + line.replace("Tried", "I tried").replace("Noted.", "noted!")
            return super().emit(prompt, max_new_tokens, system, prefix)

    man = CubbyGhost(GhostVerse(), probe=1.0, seed=0)
    s = sv.CubbyServe(Phraser(), sv.FactStore([]), route_tau=0.35)
    s.mount(man)
    for _ in range(6):
        man.step()
    thoughts = [e for e in s.events if e["kind"] == "thought"]
    assert thoughts and all(e["text"] for e in thoughts)
    assert man.resp()["thought"] == thoughts[-1]["text"], "the page's bubble carries the step's thought"
    probes = [e for e in thoughts if e["about"] == "probe"]
    if probes:
        assert any(e["verbalized"] and e["text"].startswith("Hmm") and "raw" in e for e in probes), \
            "a verbalized thought keeps the host line beside it"
    assert man.resp()["word"] is None and man.resp()["collected"] == "", "the letter mechanic is gone"


def test_live_the_frontend_protocol_init_state_stale_next():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse, LivePac
    man = CubbyGhost(GhostVerse(), probe=0.5, seed=0)
    live = LivePac(man, min_interval=0.0)
    init = live.init_payload()
    assert {"w", "h", "d", "level", "walls", "hazards", "pellets", "power", "ghosts",
            "ghost_colors", "lives", "fear", "budget", "emotion_angle", "word"} <= set(init)
    r = live.poll()
    assert r["steps"] == 1 and "stale" not in r and r["to"] and r["move"]
    assert {"ghosts", "lives", "level", "frightened", "fear", "thought", "says",
            "emotion_intensity", "w_dopa", "lay_low"} <= set(r)
    live.min_interval = 3600.0                           # polled too soon -> cached frame, stale
    assert live.poll()["stale"] is True and live.poll()["steps"] == 1
    live.min_interval = 0.0
    assert live.next_level()["level"] == 1, "/next before the level is cleared must not advance"
    man.env.remaining.clear()                            # a cleared level: beaten frame, no step
    r2 = live.poll()
    assert r2["beaten"] is True and r2["steps"] == 1
    nxt = live.next_level()
    assert nxt["level"] == 2 and nxt["w"] == 7 and man.env.level == 2
    assert man.fear == 0.3, "surviving a level keeps fear at the floor"


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
    ate = env.eat(env.cell(*star))
    assert ate["pellet"] and ate["power"] and env.frightened == FRIGHT_STEPS
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
    assert live.error is None and s["steps"] == 8 and "error" not in s
    assert s["level"] == man.env.level and s["lives"] == man.env.lives
    assert len(s["ghosts"]) == man.env.n_ghosts
    assert s["memories"] > 2, "exploring the big maze must learn facts"
    assert s["steps"] <= s["budget"]


def test_live_game_commands_route_to_the_game_cortex():
    _exe_or_skip()
    from pacman import CubbyGhost, GhostVerse
    man = CubbyGhost(GhostVerse(), probe=0.0, seed=0)
    s = sv.CubbyServe(ChainEmitter(), sv.FactStore([]), route_tau=0.35)
    s.mount(man)
    st = s.turn("status")
    assert st["kind"] == "plugin:pacman" and st["reply"].startswith("Level 1") and not man.traj, \
        "status reports without stepping"
    go = s.turn("explore for 3 steps")
    assert go["kind"] == "plugin:pacman" and len(man.traj) == 3 and "level 1" in go["reply"].lower()
    fr = s.turn("statut ? score")
    assert fr["kind"] == "plugin:pacman" and fr["reply"].startswith("Niveau 1")


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
