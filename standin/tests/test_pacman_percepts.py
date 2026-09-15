"""The percept layer (2026-09-15, WO-3.1): cubby-man is offered what his BODY
can try, the WORLD refuses what is solid, and the refusal is what teaches him.

Nick: "we need to remove all the hardcoded rules" / "if it sees a wall it
should instinctively know that its an obstacle or at least know it after
colliding with it".

These pin the mechanism deterministically. exp_r36 measures it over a live
run; this file makes sure the parts cannot quietly stop working."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pacman import MOVES, CubbyGhost, GhostVerse  # noqa: E402


def _exe_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")


def _blocked_dir(env, place):
    """A direction from `place` that the world will refuse, and its kind."""
    x, y, z = env.coords(place)
    for m, (dx, dy, dz) in MOVES.items():
        q = (x + dx, y + dy, z + dz)
        if not (0 <= q[0] < env.w and 0 <= q[1] < env.h and 0 <= q[2] < env.d):
            return m, "edge"
        if q in env.walls:
            return m, "wall"
        if q in env.hazards:
            return m, "hazard"
    return None, None


def test_the_world_refuses_a_solid_move_and_he_stays_put():
    env = GhostVerse()
    d, kind = _blocked_dir(env, env.start)
    assert d, "the level-1 start must have at least one blocked direction"
    out = env.try_move(env.start, d, {})
    assert out["ok"] is False and out["to"] == env.start and out["kind"] == kind
    # and an open one is resolved, not refused
    open_dir = next(iter(env.exits(env.start)))
    ok = env.try_move(env.start, open_dir, {})
    assert ok["ok"] and ok["to"] == env.exits(env.start)[open_dir]


def test_a_believed_exit_the_world_disagrees_with_is_still_refused():
    """`offered` is what he BELIEVES; a base direction is always re-resolved.
    Otherwise a wrong belief would walk him through stone."""
    env = GhostVerse()
    d, _ = _blocked_dir(env, env.start)
    lie = {d: env.cell(99, 99, 0)}                       # he thinks that way is open
    assert env.try_move(env.start, d, lie)["ok"] is False


def test_he_is_offered_directions_his_map_has_not_ruled_out():
    """A blind agent has ruled nothing out yet, so every direction his body
    has is offered — including the ones the maze calls illegal. That is the
    whole point: the offer is his belief, not the maze's legal-move list.

    (A SIGHTED agent at the level-1 start already rules out `forward`: the
    cell above glows, and he can see it. Vision doing that is the other half
    of what Nick asked for, which is why this test uses the blind one to
    isolate the offer.)"""
    class Blind(CubbyGhost):
        SEE_EXITS = False

    man = Blind(GhostVerse(), probe=0.0)
    env = man.env
    offered = man.candidate_moves(env.exits(man.place))
    legal = set(env.exits(man.place))
    body = set(man.body_moves())
    assert legal < body, "the maze's legal list must be a STRICT subset of what his body can try"
    assert body - legal <= set(offered), \
        "nothing he has not ruled out is withheld — he must be able to walk into it"
    # and the sighted one rules out only what he has actually perceived
    sighted = CubbyGhost(GhostVerse(), probe=0.0)
    assert set(sighted.candidate_moves(env.exits(sighted.place))) >= set(env.exits(sighted.place)), \
        "he is never offered LESS than the truly open ways out"


def test_a_collision_teaches_and_the_direction_stops_being_offered():
    man = CubbyGhost(GhostVerse(), probe=0.0)
    env = man.env
    d, kind = _blocked_dir(env, man.place)
    before = len(man.world)
    out = env.try_move(man.place, d, {})
    assert not out["ok"]
    # the refusal, learned exactly as CubbyMan._step learns it
    fact = f"{man.NON_PLACES[out['kind']]} is the {d} neighbor of {man.place}"
    man._learn([fact])
    assert len(man.world) > before and fact in man.world
    assert d in man.known_blocked(man.place), "his map now says that way is solid"
    assert d not in man.candidate_moves(env.exits(man.place)), \
        "and he stops being offered it — his belief doing the filtering, not a rule"


def test_sight_is_stopped_by_walls_and_bounded_by_range():
    env = GhostVerse()
    here = env.coords(env.start)
    seen = env.senses(env.start, 2)
    for c in seen["pellets"] + seen["stars"] + seen["ghosts"]:
        assert sum(abs(a - b) for a, b in zip(c, here)) <= 2, "nothing beyond the radius"
        assert env.line_of_sight(here, c), "and nothing through stone"
    # a wall between two cells blocks the ray
    w = next(iter(env.walls))
    assert not env.line_of_sight((w[0] - 1, w[1], w[2]), (w[0] + 1, w[1], w[2]))


def test_a_ghost_he_cannot_sense_is_a_ghost_he_does_not_know_about():
    man = CubbyGhost(GhostVerse(ghost_free_levels=0), probe=0.0)
    env = man.env
    here = env.coords(man.place)
    env.ghosts = [(here[0], here[1], here[2])]           # right on top of him
    man._sense()
    assert man.believed_ghosts() == [tuple(here)]
    man.ghost_belief.clear()
    env.ghosts = [(env.w - 1, env.h - 1, env.d - 1)]     # the far corner
    man._sense()
    assert man.believed_ghosts() == [], "out of range: he knows of no ghost"
    assert man._safe_here(), "and acts on what he knows, not on what is"


def test_a_belief_goes_stale_and_is_dropped():
    man = CubbyGhost(GhostVerse(ghost_free_levels=0), probe=0.0)
    env = man.env
    here = env.coords(man.place)
    env.ghosts = [here]
    man._sense()
    assert man.believed_ghosts()
    env.ghosts = []                                      # it left; he does not see it leave
    env.steps += man.GHOST_MEMORY + 1
    assert man.believed_ghosts() == [], "an unrefreshed sighting stops being evidence"


def test_the_berth_is_measured_from_what_caught_him_not_a_constant():
    man = CubbyGhost(GhostVerse(ghost_free_levels=0), probe=0.0)
    assert man.danger_radius == 1, "before anything catches him, a body knows only 'touching me'"
    man._on_caught()                                     # never saw it coming
    assert man.danger_radius == 2, "his senses failed him: keep a wider berth"
    man._on_caught()
    assert man.danger_radius == 3
    # caught by one he WAS tracking at decision time, already inside the
    # berth: nothing new about distance
    r = man.danger_radius
    man._threat_seen_at = 1
    man._on_caught()
    assert man.danger_radius == r, "being caught by something he already respected teaches nothing"
    # tracked, but further out than his berth: THAT is the lesson
    man._threat_seen_at = r + 1
    man._on_caught()
    assert man.danger_radius == r + 1, "it reached him from further than he was allowing for"


def test_the_lesson_is_the_distance_at_decision_time_not_at_capture():
    """A ghost that has just caught him is on top of him, so the distance at
    capture is always ~0 and teaches nothing. exp_r36 measured it: 8 catches
    in a row left the berth at 1. The signal is where the threat was when he
    last chose a move."""
    man = CubbyGhost(GhostVerse(ghost_free_levels=0), probe=0.0)
    here = man.env.coords(man.place)
    man.ghost_belief = {tuple(here): man.env.steps}      # sitting ON him, i.e. distance 0
    man._threat_seen_at = 0                              # ...and it was 0 away when he chose, too
    man._on_caught()
    assert man.danger_radius == 1, "a capture at zero range cannot widen the berth by itself"
    man._threat_seen_at = 3                              # but this one he saw coming from 3
    man._on_caught()
    assert man.danger_radius == 3


def test_a_thought_may_only_name_things_the_step_contained():
    """exp_r37 caught him saying "without triggering the ghost" on a level
    with no ghosts. It passed every other check — "ghost" is neither a figure
    nor a cell name — so an entity clause was added: a thing he names must be
    in the record or in what he learned this step."""
    from identity import load_facts
    from pacman import grounded_ok
    facts = load_facts()
    rec = "i am at: level-1 cell 1-0-0; pellets i can see: 2; how i feel: calm"
    assert grounded_ok(rec, "I can make out a couple of pellets from here.", facts)
    assert not grounded_ok(rec, "I am edging past a ghost to reach the pellets.", facts), \
        "no ghost in the record: he may not name one"
    # ...unless he actually met one
    caught = "i am at: level-1 cell 1-0-0; place: level-1 cell 1-0-0; ghost distance: 1"
    assert grounded_ok(caught, "Something caught me; a ghost was right on top of me.", facts)


def test_a_thought_that_recites_the_record_is_refused():
    """A verbatim copy passes every grounding test there is — everything in it
    came from the record. exp_r37's first run scored 100% "spoke" on exactly
    that, so copying is now its own clause."""
    from identity import load_facts
    from pacman import MAX_RUN, grounded_ok, longest_run
    facts = load_facts()
    rec = "i am at: level-1 cell 1-0-0; tried: up; pellets i can see: 2; how i feel: calm"
    assert not grounded_ok(rec, rec, facts), "the record read back is not a thought"
    assert not grounded_ok(rec, "Here is what I just perceived in the maze: " + rec, facts)
    assert not grounded_ok(rec, "about: tried a way that was not offered", facts), \
        "a field marker is the shape of a copy, not a sentence"
    good = "I gave up going that way and had a look round instead."
    assert longest_run(good, rec) <= MAX_RUN and grounded_ok(rec, good, facts)


def test_the_world_publishes_cleared_so_he_never_counts_the_pellets():
    env = GhostVerse()
    assert env.cleared is False
    env.remaining.clear()
    assert env.cleared is True and env.progress()["cleared"] is True
    import inspect
    src = inspect.getsource(CubbyGhost.step)
    assert "env.remaining" not in src and ".remaining" not in src, \
        "his own step must not read the level's true remainder"


def test_ghost_free_early_levels_then_the_live_games_count():
    env = GhostVerse()
    assert env.ghost_free_levels == 3
    for lvl in (1, 2, 3):
        env._start_level(lvl)
        assert env.n_ghosts == 0 and env.ghosts == [] and env.ghost_spawn == []
    env._start_level(4)
    assert env.n_ghosts == 4 and len(env.ghosts) == 4
    env._start_level(2)                                  # and a ghost-free level still resolves a turn
    assert env.ghost_turn(env.start) == {"caught": False, "eaten": 0, "trapped": 0}


def test_live_he_bumps_into_something_and_writes_it_down():
    """End to end, with SEE_EXITS off so a collision is the ONLY way a map can
    exist: within a short run he must walk into something and learn it."""
    _exe_or_skip()

    class Blind(CubbyGhost):
        SEE_EXITS = False

    man = Blind(GhostVerse(), probe=0.0, seed=1)
    bumps = 0
    for _ in range(40):
        if man.step().get("bumped"):
            bumps += 1
    assert bumps > 0, "a blind agent that never collides can never learn the maze"
    facts = [f for f in man.world.texts
             if any(f.startswith(n + " is the ") for n in man.NON_PLACES.values())]
    assert facts, "every collision is written down"
    # and every one of them is TRUE of the world he is standing in
    env = man.env
    for f in facts:
        m = man._nbr_re.match(f)
        if not m or not m.group("a").startswith(f"level-{env.level} "):
            continue
        x, y, z = env.coords(m.group("a"))
        dx, dy, dz = MOVES[m.group("d")]
        q = (x + dx, y + dy, z + dz)
        solid = (not (0 <= q[0] < env.w and 0 <= q[1] < env.h and 0 <= q[2] < env.d)
                 or q in env.walls or q in env.hazards)
        assert solid, f"he learned an obstacle that is not there: {f}"
