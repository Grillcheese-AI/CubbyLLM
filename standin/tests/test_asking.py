"""WO-2.13 — he builds his own map by asking the worlds that already know.

Nick, 2026-09-15: *"let me ask the world: 'how can I know when something is
about to fall on me?' then the science world sends the gravity + attraction
laws so it understands ... then cubby stores it in long term memory so it knows
that part and dont have to ask already."*

The two things these tests are really guarding:

  1. asking is not the oracle coming back. A world sends LAWS he could have
     found himself and that evidence can refute; it never sends the state of
     the board.
  2. he asks ONCE. The second time the question comes up, his own map answers
     it — which is what makes this learning rather than a retrieval cache.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hypothesis import Hypotheses, Hypothesis, ask_verifier, runner_verifier  # noqa: E402
from knowledge import CodingWorld, PhysicsWorld  # noqa: E402
from verse import CubbyMan, ToyVerse  # noqa: E402
from worlds import Knows, Worlds  # noqa: E402

FALLING = "how can I know when something is about to fall on me"


def _mounted(trace=None) -> Worlds:
    w = Worlds(trace=trace)
    w.mount(PhysicsWorld())
    w.mount(CodingWorld())
    return w


# ── the registry ──────────────────────────────────────────────────────────
def test_a_question_goes_to_the_world_whose_domain_it_is():
    w = _mounted()
    assert w.route(FALLING)[0].name == "physics"
    assert w.route("write hello world in python")[0].name == "coding"
    assert len(w) == 2 and set(w.domains()) == {"physics", "coding"}


def test_a_question_nobody_covers_is_answered_by_nobody():
    """The important negative. A registry that always finds someone is an
    oracle with a routing table in front of it."""
    w = _mounted()
    world, score = w.route("what is the ghost doing right now")
    assert world is None and score < w.tau
    rec = w.ask("what is the ghost doing right now")
    assert rec["facts"] == [] and rec["world"] is None


def test_only_something_askable_can_be_mounted():
    w = Worlds()
    try:
        w.mount(object())
        assert False, "a bare object is not a world"
    except TypeError:
        pass
    assert isinstance(PhysicsWorld(), Knows) and isinstance(CodingWorld(), Knows)


def test_every_ask_leaves_a_receipt():
    """`asked` is what the percept tripwire reads: if a law ever arrives that
    he could not have found himself, this is where it shows up."""
    w = _mounted()
    w.ask(FALLING)
    w.ask("nothing anyone knows about")
    assert [r["world"] for r in w.asked] == ["physics", None]
    assert w.asked[0]["facts"] and not w.asked[1]["facts"]


# ── what a world is allowed to send back ──────────────────────────────────
def test_physics_answers_with_laws_not_with_the_board():
    """Every law has to be true of ANY world where things fall, and none of
    them may name a thing in cubby-man. Physics has never heard of the maze."""
    laws = PhysicsWorld().answer(FALLING)
    assert len(laws) >= 4
    assert any("nearer than it was" in f for f in laws), "it must tell him what to PERCEIVE"
    assert any("one step up lands on me next" in f for f in laws), "and let him PREDICT"
    board = ("ghost", "pellet", "level", "cell", "maze", "place", "north", "south",
             "east", "west", "row", "column", "score")
    assert not [f for f in laws for t in board if t in f.lower()], \
        "a law that names the board is the oracle wearing a lab coat"


def test_physics_says_nothing_about_what_it_does_not_know():
    p = PhysicsWorld()
    assert p.answer("how do I write a for loop") == []
    assert p.covers("how do I write a for loop") < Worlds.TAU


# ── asking as a way a claim gets settled ──────────────────────────────────
def test_an_answer_settles_the_claim_and_is_what_it_earned():
    seen = []
    led = Hypotheses({"ask": ask_verifier(_mounted())}, trace=lambda k, **d: seen.append((k, d)))
    led.frame(Hypothesis(claim="something up there is coming down", test=f"ask: {FALLING}",
                         verifier="ask", payload={"question": FALLING}, made_at=0, patience=3))
    earned = led.sweep(1)
    assert len(earned) >= 4, "one answer can earn the whole law family"
    assert led.settled("confirmed") and not led.open
    assert [d for k, d in seen if k == "hypothesis_settled"][0]["why"].startswith("physics answered")


def test_a_question_nobody_answers_stays_open_then_is_refused():
    """Not knowing is a legitimate state to sit in. Refusing it after patience
    is a fact about what is askable, not a fact about the world."""
    led = Hypotheses({"ask": ask_verifier(_mounted())})
    led.frame(Hypothesis(claim="the ghost is asleep", test="ask", verifier="ask",
                         payload={"question": "is the ghost asleep right now"},
                         made_at=0, patience=3, if_false="nobody out there knows that"))
    assert led.sweep(1) == [] and led.open, "no answer yet is not a verdict"
    assert led.sweep(3) == ["nobody out there knows that"] and led.settled("refused")


def test_a_world_that_knows_is_still_not_an_authority():
    """The coding world hands back an artefact, and the runner checks it before
    he holds it. Ask-then-verify is two verifiers composing, not a new gate."""
    told = CodingWorld().answer_with_test("write hello world in python")
    assert told and told["facts"]
    led = Hypotheses({"runner": runner_verifier(timeout=20.0)})
    led.frame(Hypothesis(claim="that is how hello world goes", test="run it",
                         verifier="runner", if_true=told["facts"],
                         payload={"code": told["code"], "expected": told["expected"]}))
    assert led.sweep(1) == told["facts"]
    led2 = Hypotheses({"runner": runner_verifier(timeout=20.0)})
    led2.frame(Hypothesis(claim="a lie about hello world", test="run it", verifier="runner",
                          if_true=["never earned"], if_false="that is not what it prints",
                          payload={"code": 'print("goodbye")\n', "expected": "hello world"}))
    assert led2.sweep(1) == ["that is not what it prints"]


# ── the agent, in a world that is not cubby-man ───────────────────────────
def test_the_asking_belongs_to_the_agent_so_any_world_gets_it():
    man = CubbyMan(ToyVerse(seed=0), probe=0.0)
    assert isinstance(man.other_worlds, Worlds) and "ask" in man.guesses.verifiers
    man.other_worlds.mount(PhysicsWorld())
    assert man.ask_elsewhere(FALLING, claim="something up there is coming down", now=0)
    learned = man.settle(1)
    assert len(learned) >= 4 and all(f in man.world for f in learned)
    assert man.other_worlds.asked and man.other_worlds.asked[0]["world"] == "physics"


def test_he_asks_once_and_then_he_knows():
    """The kill criterion. A second ask means nothing was learned."""
    man = CubbyMan(ToyVerse(seed=0), probe=0.0)
    man.other_worlds.mount(PhysicsWorld())
    man.ask_elsewhere(FALLING, now=0)
    man.settle(1)
    asks = len(man.other_worlds.asked)
    assert man.already_know(FALLING), "the answer is in his own map now"
    assert man.ask_elsewhere(FALLING, now=2) is None, "he does not ask what he knows"
    man.settle(3)
    assert len(man.other_worlds.asked) == asks, "and nothing went back out to physics"


def test_the_law_is_his_not_the_worlds_it_came_up_in():
    """One map. The laws arrive while he is in ToyVerse — which has no falling
    anything — and they are plain facts in the same store as the maze he walks,
    reachable from a question worded nothing like the one he asked."""
    man = CubbyMan(ToyVerse(seed=0), probe=0.0)
    man.other_worlds.mount(PhysicsWorld())
    man.ask_elsewhere(FALLING, now=0)
    man.settle(1)
    for q in ("what happens to a falling thing", "something is falling toward me"):
        assert any("fall" in fact for _, fact in man.world(q, 3)), q
    assert man.already_know("where will a falling thing land") is None, \
        "a DIFFERENT question is still his to ask — knowing one answer is not knowing the domain"


def test_a_fresh_agent_restored_from_his_map_does_not_ask_again():
    """Across runs, not just within one: the facts are the memory, so an agent
    that starts holding them never sends the question out."""
    first = CubbyMan(ToyVerse(seed=0), probe=0.0)
    first.other_worlds.mount(PhysicsWorld())
    first.ask_elsewhere(FALLING, now=0)
    held = first.settle(1)

    later = CubbyMan(ToyVerse(seed=0), probe=0.0)
    later.other_worlds.mount(PhysicsWorld())
    for f in held:
        later.world.add(f)
    assert later.ask_elsewhere(FALLING, now=0) is None
    assert later.other_worlds.asked == [], "nothing was asked at all"


# ── and in cubby-man, where the question actually comes from ──────────────
def _pac(**kw):
    from pacman import CubbyGhost, GhostVerse
    env = GhostVerse(level=1, ghost_free_levels=9, fallers_from_level=1, **kw)
    return CubbyGhost(env, seed=0, probe=0.0, memory=None, blank=True), env


def test_the_world_says_where_a_thing_is_never_that_it_is_falling():
    """The percept is a position. 'Falling' is an inference, and the whole
    point is that he cannot make it until somebody tells him how."""
    man, env = _pac()
    env.fallers.append({"at": (0, 2, 0), "let_go": (0, 2, 0), "since": 0})
    seen = env.senses(env.cell(0, 0, 0), 3)
    assert seen["things"] == [(0, 2, 0)]
    assert all(not isinstance(t, dict) for t in seen["things"]), "no velocity, no label, no intent"


def test_before_he_asks_a_thing_overhead_means_nothing():
    man, env = _pac()
    man.place = env.cell(0, 0, 0)
    env.fallers.append({"at": (0, 2, 0), "let_go": (0, 3, 0), "since": 0})
    man._sense()
    env.steps += 1
    env.fallers[0]["at"] = (0, 1, 0)
    man._sense()
    assert not man.knows_falling()
    assert man.falling_at_me() is None, "he can see it; he has no reason to think it is coming"


def test_after_he_asks_the_same_two_sightings_are_a_prediction():
    man, env = _pac()
    man.place = env.cell(0, 0, 0)
    man.ask_elsewhere(man.FALLING_Q, now=env.steps)
    man.settle(env.steps)
    assert man.knows_falling(), "physics is mounted in cubby-man, so the question gets answered"
    env.fallers.append({"at": (0, 2, 0), "let_go": (0, 3, 0), "since": 0})
    man._sense()
    env.steps += 1
    env.fallers[0]["at"] = (0, 1, 0)
    man._sense()
    assert man.falling_at_me() == 1


def test_a_prediction_needs_the_law_that_licenses_it():
    """Not a rule in the file: each inference checks the law it reads. Give him
    a map with the OTHER law in it and the inference is not available, though
    every percept is identical and he has plainly been answered."""
    told, env = _pac()
    told.place = env.cell(0, 0, 0)
    told.ask_elsewhere(told.FALLING_Q, now=env.steps)
    laws = told.settle(env.steps)
    assert told.LAW_ONE_UP in laws and told.LAW_NEARER in laws

    half, env2 = _pac()
    half.place = env2.cell(0, 0, 0)
    for f in laws:
        if f != half.LAW_ONE_UP:                         # everything he was told except the one
            half.world.add(f)
    assert half.knows_falling(), "he asked and was answered; this is about WHICH law he holds"

    for man, env_, expect in ((told, env, 1), (half, env2, None)):
        env_.fallers.append({"at": (0, 1, 0), "let_go": (0, 3, 0), "since": 0})
        man._sense()
        assert man.falling_at_me() == expect, "the law was doing the work, not the code"
