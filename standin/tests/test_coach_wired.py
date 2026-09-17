"""The coach inside the running game: a warning gets scored by the maze itself.

`test_coach.py` pins the policy against a hand-built world. These run the real
`CubbyGhost`, so they test the WIRING — that a message reaches the ODE, that
the settlement reads `env.ghosts`, that the outcome hooks fire, and that
conversation is the default route through `handle`.

`chem` is attached explicitly. It is optional on the agent by design (every
call site is guarded by `if self.chem is not None`) and a blank agent has none
until a brain is built, which needs the VM — so attaching one here tests the
coach wiring rather than the VM's availability. The full `step()` loop does
need the VM and lives in the one `live` test at the bottom.
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


def _run(a, steps):
    """Advance the clock and settle, without the VM the full step needs."""
    for _ in range(steps):
        a.env.steps += 1
        a._coach_tick()


def test_talking_to_him_moves_his_body():
    a = _agent()
    before = a.chem.oxytocin
    a.hear("lâche pas, t'es capable!")
    assert a.chem.oxytocin > before, "a kind word reaches the chemistry"
    assert a.chem.pain == 0.0, "and nothing a person says hurts him"


def test_a_warning_raises_threat_and_plain_talk_does_not():
    a = _agent()
    warn = a.hear("careful, ghost behind you!")
    talk = a.hear("how's it going in there")
    assert warn["kind"] == "warn" and warn["drives"]["threat"] > 0.0
    assert talk["kind"] == "talk" and talk["drives"].get("threat", 0.0) == 0.0


def test_the_maze_settles_the_warning_by_itself():
    """No hand-fed ground truth: `_coach_tick` reads the world and scores it."""
    a = _agent()
    a.hear("careful!")
    assert a.coach.pending, "the claim is outstanding"
    _run(a, pacman.Coach.WARN_WINDOW + 2)
    assert not a.coach.pending, "the maze closed it one way or the other"
    assert a.coach.settled["right"] + a.coach.settled["wrong"] == 1


def test_warnings_are_scored_against_real_ghosts_not_his_beliefs():
    """The whole value of a warning is the case where the player can see
    further than he can, so `_coach_tick` reads `env.ghosts` and not
    `believed_ghosts()` — otherwise you only ever get credit for telling him
    what he already knew."""
    a = _agent()
    a.ghost_belief.clear()                      # he believes in nothing at all
    a.env.ghosts = [a.env.coords(a.place)]      # and yet one is right on top of him
    a.hear("careful!")
    _run(a, 1)
    assert a.coach.settled["right"] == 1, "he was warned about something he could not see"


def test_a_false_alarm_costs_standing():
    a = _agent()
    a.env.ghosts = []                           # nothing out there at all
    before = a.coach.credibility
    a.hear("ghost!")
    _run(a, pacman.Coach.WARN_WINDOW + 1)
    assert a.coach.settled["wrong"] == 1 and a.coach.credibility < before


def test_being_caught_labels_what_was_said_just_before_it():
    """The world supplies the target GrillCheese had to be handed by a caller."""
    a = _agent()
    a.hear("this level is a nightmare")
    assert "nightmare" not in a.coach.lex.ema, "hearing only READS; it teaches nothing"
    a._on_caught()
    taught = a.coach.lex.ema.get("nightmare")
    assert taught is not None, "a catch teaches the words that preceded it"
    assert taught[0] < 0, "and it teaches them the sign the world just supplied"


def test_chat_is_the_default_and_commands_are_the_narrow_case():
    a = _agent()
    said = a.handle("you can do it!!")
    assert said["meta"]["heard"] == "talk", "not a command, so it is conversation"
    assert said["offered"] and isinstance(said["offered"][0], str), "and he answers"


def test_the_player_can_see_what_he_holds_about_them():
    a = _agent()
    a.hear("careful!")
    _run(a, 3)
    coach = a.affect().get("coach")
    assert coach and set(coach) >= {"credibility", "base_rate", "warnings"}


def test_live_a_real_run_keeps_the_ledger_consistent():
    """The full step loop, VM and all."""
    a = pacman.CubbyGhost(seed=7, blank=True)
    a.hear("careful out there")
    for _ in range(25):
        a.step()
    st = a.coach.state()
    assert st["pending"] == 0, "a warning cannot stay open across a whole run"
    assert 0.0 <= st["base_rate"] <= 1.0
