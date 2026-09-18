"""Talking to Cubby-Man while he plays: anything gets through, one thing is checked.

The first version of `coach.py` matched two phrase lists and dropped the rest,
which the owner correctly called a two-message channel rather than a chat. The
first test here is the one that pins the fix.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from coach import WARN_AT, Coach  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402


# ── it is a chat channel, not two buttons ───────────────────────────────────

def test_any_message_reaches_him():
    """Not on either phrase list, and it must still arrive: somebody spoke."""
    c = Coach()
    for text in ("how's it going down there", "that maze looks horrible",
                 "ok I'm watching", "quelle horreur ce labyrinthe"):
        got = c.hear(text)
        assert got["kind"] == "talk", text
        assert got["drives"], f"{text!r} produced nothing at all"


def test_only_a_warning_raises_threat():
    """The narrow exception to the afferent rule, and it stays narrow."""
    c = Coach()
    assert c.hear("you can do it!!")["drives"]["threat"] == 0.0
    assert c.hear("I'm scared for you")["drives"]["threat"] == 0.0
    assert c.hear("careful, ghost behind you")["drives"]["threat"] > 0.3


def test_shouting_at_a_maze_is_excitement_not_menace():
    """`appraise` scores `!!` as threat, which is right for a stranger in a
    text box and wrong for someone yelling GO GO GO."""
    c = Coach()
    got = c.hear("GO GO GO!!")
    assert got["drives"]["threat"] == 0.0
    assert got["drives"]["surge"] > 0.0, "the shout is arousal"


def test_kind_words_are_never_reward():
    """`reward` builds tolerance and belongs to the world. A sentence may not
    deliver it, or the channel is a cheat code."""
    c = Coach()
    for text in ("you can do it!!", "lâche pas!", "amazing, well done!"):
        assert "reward" not in c.hear(text)["drives"], text


def test_french_encouragement_lands_without_an_english_equivalent():
    """The lexicon earns its keep against a neutral message at the same need:
    `appraise`'s general FR list has none of these, so without the entry
    "lâche pas la patate" would arrive as flat as "ok I'm watching"."""
    warm = Coach().hear("lâche pas la patate, t'es capable!", step=1, need=0.8)
    flat = Coach().hear("je regarde le labyrinthe", step=1, need=0.8)
    assert warm["drives"]["valence"] > 0.0
    assert warm["drives"]["social"] > flat["drives"]["social"] * 2


# ── it cannot be farmed ─────────────────────────────────────────────────────

def test_repeating_yourself_stops_working():
    c = Coach()
    first = c.hear("you can do it", step=1, need=0.8)["drives"]["social"]
    for s in range(2, 8):
        c.hear("you can do it", step=s, need=0.8)
    last = c.hear("you can do it", step=8, need=0.8)["drives"]["social"]
    assert last < first / 4, "the tenth one in a row is noise and reads as noise"


def test_a_gap_resets_it():
    c = Coach()
    c.hear("you can do it", step=1, need=0.8)
    c.hear("you can do it", step=2, need=0.8)
    worn = c.hear("you can do it", step=3, need=0.8)["drives"]["social"]
    fresh = c.hear("you can do it", step=40, need=0.8)["drives"]["social"]
    assert fresh > worn


def test_encouragement_lands_in_proportion_to_need():
    """Worth little to someone having a fine time; worth a lot after the
    fourth death on level three."""
    fine = Coach().hear("you can do it", step=1, need=0.0)["drives"]["social"]
    sunk = Coach().hear("you can do it", step=1, need=1.0)["drives"]["social"]
    assert sunk > fine * 2


# ── the warning ledger: trust with a referent ───────────────────────────────

def test_being_right_about_a_ghost_earns_standing():
    c = Coach()
    before = c.credibility
    c.hear("careful!", step=10)
    out = c.tick(11, ghost_near=True)
    assert out and out[0]["verdict"] == "right"
    assert c.credibility > before


def test_being_wrong_costs_it():
    c = Coach()
    before = c.credibility
    c.hear("careful!", step=10)
    for s in range(11, 11 + Coach.WARN_WINDOW + 1):
        c.tick(s, ghost_near=False)
    assert c.settled["wrong"] == 1 and c.credibility < before


def _drive(c, steps: int, near_every: int, warn_every: int):
    """Run a maze where a ghost is near every `near_every` steps, with a
    warning every `warn_every` steps."""
    for s in range(1, steps + 1):
        if warn_every and s % warn_every == 0:
            c.hear("ghost!", step=s)
        c.tick(s, ghost_near=(s % near_every == 0))


def test_a_random_shouter_cannot_farm_standing_by_volume():
    """The regression test for a bug that was MEASURED, not argued.

    In `docs/coach_ab.md` the `noise` arm — warnings fired at random moments
    about nothing — finished at 0.97 against a truthful oracle's 0.99, because
    standing was a random walk with a per-warning gain and ~100 warnings
    saturate it whatever their accuracy. Here a shouter fires constantly in a
    maze where ghosts are near most of the time, so he is right most of the
    time and has told nobody anything."""
    c = Coach()
    _drive(c, steps=150, near_every=1, warn_every=2)
    assert c.settled["right"] > 40, "he is right constantly, which is the point"
    assert c.credibility < 0.15, "and it must buy him almost nothing"


def test_being_right_where_ghosts_are_rare_is_what_earns_standing():
    """Same hit rate, different worlds. The advantage over chance is the whole
    measure, so a warning is worth what it could not have been guessed."""
    crowded, quiet = Coach(), Coach()
    _drive(crowded, steps=150, near_every=1, warn_every=3)   # always near
    _drive(quiet, steps=150, near_every=7, warn_every=7)     # near, and warned for
    assert quiet.credibility > crowded.credibility * 2


def test_a_long_honest_record_beats_a_short_one():
    """Shrinkage toward the prior: one lucky call is not a reputation."""
    short, long_ = Coach(), Coach()
    _drive(short, steps=14, near_every=7, warn_every=7)
    _drive(long_, steps=210, near_every=7, warn_every=7)
    assert long_.credibility > short.credibility


def test_a_discredited_voice_stops_being_heard():
    """The spam collapse: shout about ghosts that are not there and the
    warnings stop arriving."""
    c = Coach()
    step = 0
    for _ in range(12):                              # twelve false alarms
        c.hear("GHOST!", step=step)
        for _ in range(Coach.WARN_WINDOW + 1):
            step += 1
            c.tick(step, ghost_near=False)
    assert c.credibility < Coach.STRANGER, "standing falls"
    # And the sharper thing, which was not designed and fell out of putting the
    # learned lexicon underneath: the WORD stops being a warning word. Every
    # false alarm teaches `ghost` toward threat 0, so after enough of them the
    # message no longer crosses `WARN_AT` and never becomes a claim at all.
    # Spam does not merely lose its credit — it stops parsing as a warning.
    assert c.lex.read("GHOST!")["threat"] < WARN_AT
    assert c.hear("GHOST!", step=step)["kind"] == "talk"


# ── the afferent rules still hold inside the game ───────────────────────────

def test_a_worried_player_does_not_make_him_miserable():
    """No mirroring, same rule as chat — it is the same `Afferent` underneath,
    not a second copy of the policy."""
    c = Coach()
    got = c.hear("oh no this is awful, I hate this level")
    assert got["drives"]["valence"] == 0.0, "their bad time is not his"
    assert got["drives"]["social"] > 0.0, "it is concern"


def test_through_the_real_body_encouragement_warms_him():
    c = Coach()
    cheered, alone = Neurochemistry(), Neurochemistry()
    d = c.hear("you can do it!!", step=1, need=0.9)["drives"]
    for _ in range(6):
        cheered.update(**d)
        alone.update()
    assert cheered.oxytocin > alone.oxytocin
    assert cheered.pain == 0.0


def test_the_ledger_is_small_enough_to_show_the_player():
    c = Coach()
    c.hear("careful!", step=1)
    c.tick(2, ghost_near=True)
    st = c.state()
    assert set(st) == {"credibility", "base_rate", "warnings", "messages",
                       "pending", "vocabulary", "eligible"}
