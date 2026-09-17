"""Three-factor plasticity: a word is learned only when all three line up.

  PRE    the word was used              `Coach.hear` -> `Eligibility.mark`
  POST   it is still eligible           the decaying trace
  MOD    a neuromodulator moved         `Coach.outcome(magnitude=…)`

GrillCheese had the first two and never multiplied them by the third, which is
why its plasticity weights are written every turn and read by nothing. The
difference here is a world that delivers real pain and real reward at known
instants, so the third factor is a measurement rather than a schedule.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from coach import Coach  # noqa: E402
from lexicon import AffectLexicon, Eligibility  # noqa: E402
from world import CLEAR, FAIL, HURT  # noqa: E402


# ── the trace itself ────────────────────────────────────────────────────────

def test_a_trace_decays_and_is_eventually_dropped():
    e = Eligibility()
    e.mark("something", step=0)
    assert len(e) == 1
    e.decay_to(40)
    assert len(e) == 0, "a lexicon that never forgets fills up with superstition"


def test_decay_is_idempotent_per_step():
    """A caller may tick and mark in either order without double-decaying."""
    a, b = Eligibility(), Eligibility()
    a.mark("x", step=0)
    b.mark("x", step=0)
    a.decay_to(5)
    b.decay_to(5); b.decay_to(5); b.decay_to(5)
    assert a.active()[0][1] == pytest.approx(b.active()[0][1])


def test_recent_beats_distant():
    e = Eligibility()
    e.mark("old", step=0)
    e.mark("new", step=8)
    order = [t for t, _ in e.active()]
    assert order[0] == "new"


def test_the_held_set_is_bounded():
    """A chatty player must not be able to grow it without limit."""
    e = Eligibility()
    for i in range(200):
        e.mark(f"message {i}", step=i // 10)
    assert len(e) <= Eligibility.MAX_HELD


# ── all three factors are required ──────────────────────────────────────────

def test_no_outcome_teaches_nothing():
    """PRE and POST without MOD is the open loop GrillCheese was left with."""
    c = Coach()
    c.hear("banana", step=0)
    assert "banana" not in c.lex.ema
    for s in range(1, 6):
        c.tick(s, ghost_near=False)
    assert "banana" not in c.lex.ema, "nothing happened, so nothing is learned"


def test_an_outcome_with_no_magnitude_teaches_nothing():
    c = Coach()
    c.hear("banana", step=0)
    assert c.outcome(HURT, step=1, magnitude=0.0) == 0
    assert "banana" not in c.lex.ema


def test_an_outcome_teaches_what_was_still_eligible():
    c = Coach()
    c.hear("banana", step=0)
    c.outcome(HURT, step=1, magnitude=1.0)
    assert c.lex.ema["banana"][0] < 0, "a catch teaches the words before it"


def test_something_said_long_ago_is_not_blamed():
    """The cliff the fixed window had, replaced by a slope that reaches zero."""
    c = Coach()
    c.hear("banana", step=0)
    for s in range(1, 45):
        c.tick(s, ghost_near=False)
    c.outcome(HURT, step=45, magnitude=1.0)
    assert "banana" not in c.lex.ema


# ── graded, which the fixed window was not ──────────────────────────────────

def test_credit_is_graded_by_recency():
    near, far = Coach(), Coach()
    near.hear("banana", step=9)
    far.hear("banana", step=0)
    for c in (near, far):
        c.outcome(HURT, step=10, magnitude=1.0)
    assert abs(near.lex.ema["banana"][0]) > abs(far.lex.ema["banana"][0])


def test_credit_is_graded_by_how_much_the_body_moved():
    """The third factor doing its job: the same words, the same instant, a
    different amount of dopamine."""
    big, small = Coach(), Coach()
    for c, mag in ((big, 1.0), (small, 0.15)):
        c.hear("banana", step=0)
        c.outcome(HURT, step=1, magnitude=mag)
    assert abs(big.lex.ema["banana"][0]) > abs(small.lex.ema["banana"][0])


def test_the_sign_comes_from_the_event_kind():
    good, bad = Coach(), Coach()
    good.hear("banana", step=0)
    bad.hear("banana", step=0)
    good.outcome(CLEAR, step=1, magnitude=1.0)
    bad.outcome(FAIL, step=1, magnitude=1.0)
    assert good.lex.ema["banana"][0] > 0 > bad.lex.ema["banana"][0]


def test_the_sdk_event_names_and_the_pac_aliases_agree():
    """A second world raises `world.EVENTS`; pac says "caught". Same result."""
    a, b = Coach(), Coach()
    a.hear("banana", step=0)
    b.hear("banana", step=0)
    a.outcome(HURT, step=1, magnitude=1.0)
    b.outcome("caught", step=1, magnitude=1.0)
    assert a.lex.ema["banana"] == b.lex.ema["banana"]


def test_an_unknown_event_is_ignored_rather_than_guessed():
    c = Coach()
    c.hear("banana", step=0)
    assert c.outcome("something_else", step=1) == 0
    assert "banana" not in c.lex.ema


# ── it still cannot be farmed ───────────────────────────────────────────────

def test_plasticity_does_not_open_a_route_to_reward():
    """The trace scales the LEARNING RATE. It must never touch the drives."""
    c = Coach()
    got = c.hear("you can do it!!", step=0, need=1.0)
    assert "reward" not in got["drives"]
    c.outcome(CLEAR, step=1, magnitude=1.0)
    assert "reward" not in c.hear("you can do it!!", step=2, need=1.0)["drives"]


def test_a_settled_word_is_harder_to_move_than_a_new_one():
    """Momentum falls once a word is seen enough: quick to learn, slow to be
    argued out of."""
    fresh, settled = AffectLexicon({}), AffectLexicon({})
    settled.count["banana"] = AffectLexicon.SETTLED_AT + 5
    settled.ema["banana"] = [0.0, 0.0, 0.0]
    fresh.learn("banana", valence=1.0)
    settled.learn("banana", valence=1.0)
    assert fresh.ema["banana"][0] > settled.ema["banana"][0]
