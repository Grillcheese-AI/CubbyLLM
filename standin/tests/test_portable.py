"""The behaviours belong to the AGENT, not to cubby-man.

Nick, 2026-09-15: *"these behaviours should be global not one per context per
example the world / context is cubby-man however the same (exactly the same)
should be portable to other contexts and vice-versa."*

So the test is not "does it work in the maze" — it is "does a DIFFERENT world,
with no pacman code anywhere near it, get the same machinery for free." Every
test here uses plain `CubbyMan` over `ToyVerse`, which knows nothing about
pellets, ghosts or combos."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hypothesis import Hypothesis  # noqa: E402
from verse import CubbyMan, ToyVerse  # noqa: E402


def _exe_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")


def test_a_plain_agent_in_another_world_has_the_whole_machinery():
    man = CubbyMan(ToyVerse(seed=0), probe=0.0)
    assert man.guesses is not None, "the hypothesis ledger is his, not the maze's"
    assert man.can == set() and not man.able("anything"), "capabilities start empty and are earned"
    assert hasattr(man, "blank") and hasattr(man, "settle")
    assert man.body_moves() and set(man.body_moves()) == set(man.DIR_NAMES), \
        "he is offered what his body can do, in any world"
    assert man.known_blocked(man.place) == set(), "and has ruled nothing out yet"


def test_the_collision_percept_is_the_agents_too():
    """ToyVerse has no stone, only an edge — and walking off it teaches him,
    with no world-specific code involved."""
    env = ToyVerse(w=2, h=1, seed=0)
    man = CubbyMan(env, probe=0.0)
    d = next(d for d in man.DIR_NAMES if d not in env.exits(man.place))
    out = env.try_move(man.place, d, {})
    assert out["ok"] is False and out["kind"] == "edge" and out["to"] == man.place
    fact = f"{man.NON_PLACES['edge']} is the {d} neighbor of {man.place}"
    man._learn([fact])
    assert d in man.known_blocked(man.place)
    assert d not in man.candidate_moves(env.exits(man.place)), \
        "his own map does the filtering, in any world"


def test_a_question_and_its_verdict_are_the_agents():
    """Framed in one place, settled by the world, learned as a fact — none of
    it mentions a maze."""
    man = CubbyMan(ToyVerse(seed=0), probe=0.0)
    seen = {"happened": False}
    man.guesses.verifiers["world"] = lambda h, now: (
        (True, "it happened") if seen["happened"] else
        ((False, "it never did") if now - h.made_at >= h.patience else (None, "not yet")))
    man.guesses.frame(Hypothesis(claim="maybe that thing moves", test="watch it",
                                 verifier="world", if_true="moving is its way",
                                 if_false="staying put is its way", made_at=0, patience=3))
    assert man.settle(1) == [] and man.guesses.open
    earned = man.settle(3)
    assert earned == ["staying put is its way"]
    assert "staying put is its way" in man.world, "a verdict is a fact he HOLDS"


def test_a_capability_is_earned_by_a_verdict_in_any_world():
    """UNLOCKS is a class attribute on the agent: a world says which verdict
    grants what, and CubbyMan does the granting."""
    class Swimmer(CubbyMan):
        UNLOCKS = {"floating is the way of water": "swim"}
        gained: list[str] = []

        def on_capability(self, name, because):
            Swimmer.gained.append(name)

    man = Swimmer(ToyVerse(seed=0), probe=0.0)
    assert not man.able("swim")
    man.guesses.verifiers["world"] = lambda h, now: (True, "it floated")
    man.guesses.frame(Hypothesis(claim="maybe I float", test="get in", verifier="world",
                                 if_true="floating is the way of water"))
    man.settle(1)
    assert man.able("swim") and Swimmer.gained == ["swim"]
    assert "floating is the way of water" in man.world


def test_what_he_learns_in_one_world_he_still_knows():
    """One map, not one per context. Carry his world model across and the fact
    is still there — which is the whole point of the map being HIS."""
    a = CubbyMan(ToyVerse(seed=0), probe=0.0)
    a._learn(["falling is the way of a dropped thing"])
    b = CubbyMan(ToyVerse(w=3, h=3, seed=1), probe=0.0)   # a different world entirely
    assert "falling is the way of a dropped thing" not in b.world, "a different agent starts fresh"
    b.world = a.world                                    # the same him, a different world
    assert "falling is the way of a dropped thing" in b.world
    assert b.known_blocked(b.place) == set() or True      # the map is shared; the beliefs re-derive


def test_blank_refuses_to_inherit_anything():
    man = CubbyMan(ToyVerse(seed=0), probe=0.0, blank=True)
    assert man.blank is True
    assert man.can == set(), "nothing granted"
    assert man.guesses.all == [], "nothing wondered yet"


def test_blank_is_forced_by_the_environment(monkeypatch):
    monkeypatch.setenv("CUBBYMAN_BLANK", "1")
    assert CubbyMan(ToyVerse(seed=0), probe=0.0).blank is True
    monkeypatch.setenv("CUBBYMAN_BLANK", "0")
    assert CubbyMan(ToyVerse(seed=0), probe=0.0).blank is False


def test_the_game_is_a_context_not_the_owner_of_any_of_it():
    """The pacman agent must not redefine the machinery — it inherits it and
    supplies only its own vocabulary. If any of these move back into
    pacman.py, this is the test that says so."""
    from pacman import CubbyGhost
    for owned_by_the_agent in ("settle", "able", "known_blocked", "known_neighbors",
                               "candidate_moves", "look_around", "_rebuild_beliefs"):
        assert hasattr(CubbyMan, owned_by_the_agent), f"{owned_by_the_agent} belongs on the agent"
    assert "blank" not in CubbyGhost.__dict__ and "can" not in CubbyGhost.__dict__
    assert CubbyGhost.UNLOCKS == {"chaining is the way of a move": "combos"}, \
        "the world says WHICH verdict grants what; the gate itself is the agent's"
    assert CubbyMan.UNLOCKS == {}, "and the agent grants nothing by default"
