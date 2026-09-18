"""The filter, tested with no world anywhere in sight.

That is the point of the file. If these tests needed a maze, the mechanism
would not be the global one the owner asked for — it would be a maze feature
with a general-sounding name. Every option here is five numbers and a label,
and the same body bends the same way whether those numbers came from a
corridor, a sentence or a shell command.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from choosing import Option, breadth, choose, gains, neutral, weigh  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402


def _rested():
    c = Neurochemistry()
    for _ in range(200):
        c.update()
    return c


def _alarmed(frames=10):
    c = _rested()
    for _ in range(frames):
        c.update(threat=1.0)
    return c


def _delighted(frames=10):
    c = _rested()
    for _ in range(frames):
        c.update(valence=0.9, reward=0.8, novelty=0.6)
    return c


# the same three options, described the way any world would describe them
# The first draft used value 0.9 / risk 0.7, which nets 0.2 against a plain
# 0.5 before novelty — the two options were genuinely a coin-flip under neutral
# weighting, so "a rested body takes the bold one" was a claim about rounding.
# The code was right and the test's premise was wrong. Separated properly, the
# rested choice is unambiguous and the flip under alarm means something.
SAFE_DULL = Option("safe", value=0.50, risk=0.0, novelty=0.0)
BOLD_RISKY = Option("bold", value=0.95, risk=0.5, novelty=0.8)
UNKNOWN = Option("unknown", value=0.3, risk=0.2, novelty=0.9)


# ── neutral means neutral ───────────────────────────────────────────────────

def test_a_resting_body_bends_nothing():
    """The `quiescent()` lesson, applied to choice. A declared midpoint that is
    not the real one would leave a resting agent permanently biased — and a
    permanent bias with no cause looks exactly like a personality."""
    g = gains(_rested().modulation())
    from choosing import BASE
    for term, base in BASE.items():
        assert g[term] == pytest.approx(base, abs=0.02), term


def test_neutral_is_measured_not_assumed():
    n = neutral()
    assert not all(abs(v - 0.5) < 1e-9 for v in n.values()), \
        "a resting body does not sit at 0.5 on every knob, and assuming it does is the bug"


# ── the body bends the choice ───────────────────────────────────────────────

def test_an_alarmed_body_weighs_risk_harder():
    calm, scared = gains(_rested().modulation()), gains(_alarmed().modulation())
    assert scared["risk"] > calm["risk"]


def test_alarm_changes_which_option_wins():
    """Same options, same world, different body — a different answer."""
    opts = [SAFE_DULL, BOLD_RISKY]
    calm, _ = choose(opts, _rested().modulation())
    scared, _ = choose(opts, _alarmed(14).modulation())
    assert calm.key == "bold", "a rested body takes the better option"
    assert scared.key == "safe", "an alarmed one pays for the risk"


def test_a_pleased_body_reaches_for_the_unfamiliar():
    calm = gains(_rested().modulation())
    happy = gains(_delighted().modulation())
    assert happy["novelty"] > calm["novelty"]


# ── the two rules it may not break ──────────────────────────────────────────

def test_no_state_can_make_risk_attractive():
    """A body that can talk itself into the fire is not modulated, it is
    broken. The state sets how loudly the world's numbers speak, never their
    sign."""
    lethal = Option("lethal", value=1.0, risk=1.0)
    harmless = Option("harmless", value=1.0, risk=0.0)
    for chem in (_rested(), _alarmed(), _delighted(), _alarmed(40)):
        g = gains(chem.modulation())
        assert g["risk"] > 0.0
        scored = {o.key: s for o, s in weigh([lethal, harmless], chem.modulation())}
        assert scored["harmless"] > scored["lethal"]


def test_the_world_still_decides_what_is_good():
    """Affect re-ranks; it does not invent value."""
    good = Option("good", value=0.9)
    bad = Option("bad", value=0.1)
    for chem in (_rested(), _alarmed(), _delighted()):
        winner, _ = choose([good, bad], chem.modulation())
        assert winner.key == "good"


# ── the shape of thinking, not just the weights ─────────────────────────────

def test_urgency_narrows_the_field():
    """Under pressure you do not weigh everything slightly differently — you
    stop looking at most of it."""
    calm = breadth(10, _rested().modulation())
    rushed = breadth(10, _alarmed(20).modulation())
    assert rushed < calm


def test_a_panicking_body_still_does_something():
    assert breadth(10, _alarmed(60).modulation()) >= 1
    assert breadth(1, _alarmed(60).modulation()) == 1


def test_a_settled_body_sticks_with_what_it_was_doing():
    """Serotonin as persistence. Staying is a property of a history, not of
    an option, so it is applied to the previous choice rather than to a term."""
    a = Option("a", value=0.50)
    b = Option("b", value=0.52)                 # barely better
    settled = _rested()
    for _ in range(30):
        settled.update(valence=0.5)             # calm and content: 5-HT up
    winner, _ = choose([a, b], settled.modulation(), previous="a")
    assert winner.key == "a", "a small improvement does not move a settled body"


# ── it is the same filter whatever the options are about ────────────────────

def test_the_same_body_bends_a_maze_and_a_sentence_identically():
    """The whole claim, stated as a test. Two worlds, the same five numbers,
    the same body — the bends must be identical, because the filter never
    learns what the numbers are about."""
    chem = _alarmed(12)
    maze = [Option("corridor", value=0.9, risk=0.7), Option("wall", value=0.5, risk=0.0)]
    talk = [Option("bold_claim", value=0.9, risk=0.7), Option("hedge", value=0.5, risk=0.0)]
    m, mr = choose(maze, chem.modulation())
    t, tr = choose(talk, chem.modulation())
    assert [round(s, 6) for _, s in mr] == [round(s, 6) for _, s in tr]
    assert (m.key, t.key) == ("wall", "hedge")


def test_a_world_that_scores_nothing_is_left_alone():
    """An agent with no notion of risk is not made reckless by this module, it
    is made indifferent — which is the honest reading of not knowing."""
    blank = [Option("x"), Option("y")]
    winner, ranked = choose(blank, _alarmed().modulation())
    assert winner is not None
    assert ranked[0][1] == ranked[-1][1] == 0.0
