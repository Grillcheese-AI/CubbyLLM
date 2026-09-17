"""The path from chemistry to behaviour, and the three places it must not reach.

Until this existed, `self.chem` was read twice in the whole decision path, both
times for `craving`. Threat could spike, the compass could read fear, he could
say frightened things — and take exactly the same step he would have taken
calm. The affect stack was decorative at the point where it should have cost
something.

The three guards below are the interesting half. A transient that widens the
berth must not reach anything that MEASURES, or the alarm rates its own
outcome, teaches its own lesson, and grades its own prophecy.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pacman  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402


def _agent():
    a = pacman.CubbyGhost(seed=7, blank=True)
    a.chem = Neurochemistry()
    return a


def _alarm(a, frames=6):
    for _ in range(frames):
        a.chem.update(threat=1.0)


# ── the path exists ─────────────────────────────────────────────────────────

def test_a_calm_body_adds_nothing():
    a = _agent()
    assert a.wariness == 0
    assert a.danger_radius == a.learned_radius


def test_an_alarmed_body_keeps_more_room():
    a = _agent()
    before = a.danger_radius
    _alarm(a)
    assert a.wariness > 0
    assert a.danger_radius > before, "threat now costs him something"


def test_being_warned_by_a_credible_voice_widens_the_berth():
    """End to end: a person says a thing, and he walks differently."""
    a = _agent()
    before = a.danger_radius
    a.hear("careful, ghost!")
    assert a.danger_radius > before


def test_wariness_is_bounded_and_decays_back():
    a = _agent()
    _alarm(a, frames=40)
    assert a.wariness <= pacman.CubbyGhost.MAX_WARY
    for _ in range(80):
        a.chem.update()
    assert a.wariness == 0, "caution is transient; it never becomes what he knows"


def test_any_source_of_threat_does_it_not_just_the_coach():
    """Driven by noradrenaline, so a ghost he saw and a person shouting reach
    it the same way. Wiring the coach straight to the berth would have been
    one feature's demo rather than a mechanism."""
    seen, told = _agent(), _agent()
    _alarm(seen)
    told.hear("careful, ghost!")
    assert seen.wariness > 0 and told.wariness > 0


# ── and the three places it must not reach ──────────────────────────────────

def test_alarm_does_not_teach_a_wider_berth():
    """`_on_caught` reads `learned_radius`. Otherwise a catch taken while
    alarmed bakes the alarm into the lesson, the next alarm builds on that,
    and the radius walks to its ceiling calling it experience."""
    calm, scared = _agent(), _agent()
    for a in (calm, scared):
        a._threat_seen_at = None
    _alarm(scared)
    assert scared.danger_radius > calm.danger_radius, "the arms differ before the catch"
    calm._on_caught()
    scared._on_caught()
    assert scared.caught_at == calm.caught_at, "but they learn the same thing from it"


def test_alarm_does_not_rate_how_badly_a_catch_hurt():
    calm, scared = _agent(), _agent()
    _alarm(scared)
    assert scared.hurt_of(1) == calm.hurt_of(1)


def test_a_warning_cannot_widen_the_window_it_is_judged_in():
    """The self-fulfilling one. A warning raises NE, which widens
    `danger_radius`; settling against that would mean shouting "ghost!"
    enlarges the window in which the shout counts as correct."""
    a = _agent()
    a.env.ghosts = []
    here = a.env.coords(a.place)
    a.hear("careful, ghost!")                   # alarmed, so danger_radius is up
    assert a.wariness > 0
    # a ghost placed just outside the LEARNED berth but inside the alarmed one
    reach = a.learned_radius + 1
    a.env.ghosts = [(here[0] + reach + 1, here[1], here[2])]
    a.env.steps += 1
    a._coach_tick()
    assert a.coach.settled["right"] == 0, "it must not score itself correct"
