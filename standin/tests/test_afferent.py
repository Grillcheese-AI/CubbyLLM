"""The user's state reaches Cubby without becoming Cubby's state.

The old `perception.PETAL_DRIVES` mapped a sad user to `valence: -0.7` and an
angry user to `threat: 0.6` — a mirror and a flinch. These tests pin the
asymmetry that replaced it, and the last one runs it through the real ODE,
because a table that looks right and a body that behaves right are different
claims.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from afferent import Afferent, behavioural, drives, stated  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402


# ── the asymmetry: the thing this module exists for ─────────────────────────

def test_a_distressed_user_never_produces_negative_valence():
    """Concern, not contagion. The single most important line in the module."""
    d = drives({"valence": -0.9, "arousal": 0.5})
    assert d["valence"] == 0.0, "their bad day must not become his bad day"
    assert d["social"] > 0.5, "it must become caring"
    assert d["focus"] > 0.3, "and attention"


def test_an_angry_user_is_not_a_threat():
    """`anger -> threat 0.6` meant Cubby was afraid of his user. Never again."""
    d = drives({"valence": -0.8, "arousal": 0.9})
    assert d["threat"] == 0.0
    assert d["surge"] > 0.4, "their intensity is contagious; their anger is not"


def test_shared_joy_transfers_but_attenuated():
    """The one asymmetry in the other direction, and it is smaller than his own."""
    d = drives({"valence": 0.9, "arousal": 0.5})
    assert 0.0 < d["valence"] < 0.9
    assert d["social"] > 0.3


def test_arousal_contagion_carries_no_sign():
    """A furious user and a delighted user move the same NE input."""
    hot_bad = drives({"valence": -0.8, "arousal": 0.9})
    hot_good = drives({"valence": 0.8, "arousal": 0.9})
    assert hot_bad["surge"] == pytest.approx(hot_good["surge"])


def test_pain_and_reward_are_never_afferent():
    """No sentence may reach the somatic layer. It would be counterfeitable."""
    for v, a in ((-1.0, 1.0), (1.0, 1.0), (-1.0, 0.0)):
        d = drives({"valence": v, "arousal": a})
        assert "pain" not in d and "reward" not in d
        assert d["threat"] == 0.0


# ── the stated channel ──────────────────────────────────────────────────────

def test_stated_needs_a_self_report_frame():
    """"the client is frustrated" is a fact about a third party."""
    assert stated("the client is frustrated") is None
    assert stated("I'm frustrated")["valence"] < 0


def test_stated_separates_arousal_where_a_word_lexicon_cannot():
    """`tanné` is spent, `en calvaire` is hot. Both negative, and the third
    axis is the only thing that tells them apart — which is exactly what the
    VAD check says a general word lexicon cannot do."""
    spent = stated("chu tanné")
    hot = stated("chu en calvaire")
    assert spent and hot
    assert spent["valence"] < 0 and hot["valence"] < 0
    assert spent["arousal"] < 0.3 < hot["arousal"]


def test_quebecois_self_reports_fire():
    """Not decoration: the three commonest self-reports in the work corpus,
    none of which survives translation with its arousal intact."""
    for t in ("chu à boutte", "je suis brûlé", "chu vidé"):
        got = stated(t)
        assert got is not None, t
        assert got["valence"] < 0 and got["arousal"] < 0.3


def test_stated_outranks_lexical():
    a = Afferent()
    a.read("I'm exhausted", lexical_valence=0.8)     # the words look cheerful
    assert a.valence < 0, "the user's own report wins"


# ── consent ─────────────────────────────────────────────────────────────────

def test_timing_is_off_until_granted():
    a = Afferent()
    assert not a.may_read_timing
    a.read("ok", burst=3, gap_s=2.0, words=1)
    assert a.arousal == 0.0, "reading cadence without consent is surveillance"
    a.grant()
    a.read("ok", burst=3, gap_s=2.0, words=1)
    assert a.arousal > 0.0


def test_declining_is_a_state_not_a_silence():
    a = Afferent()
    a.decline()
    assert a.timing_consent == Afferent.DECLINED
    a.read("ok", burst=5)
    assert a.arousal == 0.0


def test_forget_clears_everything():
    a = Afferent()
    a.read("I'm furious")
    assert a.valence != 0
    a.forget()
    assert a.state()["valence"] == 0.0 and a.state()["source"] is None


# ── the behavioural channel reads arousal and only arousal ──────────────────

def test_behavioural_returns_none_rather_than_calm():
    """A caller with no timing knows nothing, which is not the same as calm."""
    assert behavioural() is None


def test_behavioural_never_reports_valence():
    got = behavioural(burst=4, gap_s=1.0)
    assert got is not None and "valence" not in got


def test_late_hour_reads_down_not_up():
    """Still up at 2am is spent, not keyed up — the owner's own curve."""
    late = behavioural(hour=2, burst=1)
    early = behavioural(hour=11, burst=1)
    assert late["arousal"] < early["arousal"]


# ── decay ───────────────────────────────────────────────────────────────────

def test_a_stated_mood_fades_over_a_few_turns():
    a = Afferent()
    a.read("I'm furious")
    first = abs(a.valence)
    for _ in range(6):
        a.read("ok")                                 # nothing stated, nothing lexical
    assert abs(a.valence) < first * 0.2, "a mood is not a setting"


# ── and now the real body ───────────────────────────────────────────────────

def test_a_sad_user_warms_cubby_rather_than_saddening_him():
    """The end-to-end claim, through the actual ODE.

    Against a CONTROL rather than against the starting value: a fresh instance
    starts at the declared resting levels and drifts toward where the ODE
    actually settles, so "serotonin went down" is true of a Cubby who received
    nothing at all. The claim is about the difference the user makes.
    """
    quiet, concerned = Neurochemistry(), Neurochemistry()
    a = Afferent()
    a.read("I'm so sad today")
    for _ in range(6):
        quiet.update()
        concerned.update(**a.drives())
    assert concerned.oxytocin > quiet.oxytocin, "concern is oxytocin"
    assert concerned.dopamine > quiet.dopamine, "leaning in, not flattening"
    assert concerned.pain == 0.0, "and nothing hurts him"
    # Serotonin comes down about 0.02, and that residual is ATTENTION, not
    # contagion: concern sets `focus`, focus drives noradrenaline, and
    # `NE_suppresses_5HT` does the rest. Attending closely to someone is
    # mildly activating and mildly costly, which is true of people too. The
    # next test is what bounds it.
    assert concerned.serotonin >= quiet.serotonin - 0.03


def test_the_old_mirror_mapping_would_have_caught_it():
    """The regression guard, stated as the comparison it is.

    `perception.PETAL_DRIVES["sadness"]` was `{"valence": -0.7, "social": 0.2}`.
    Run both and the difference is the point of the module: same user, same
    six frames, one Cubby steady and one Cubby sinking with them.
    """
    quiet, mirror, concerned = Neurochemistry(), Neurochemistry(), Neurochemistry()
    a = Afferent()
    a.read("I'm so sad today")
    for _ in range(6):
        quiet.update()
        mirror.update(valence=-0.7, social=0.2)      # the old mapping, verbatim
        concerned.update(**a.drives())
    assert concerned.serotonin > mirror.serotonin, "the mirror caught it; he should not"
    assert concerned.oxytocin > mirror.oxytocin, "and he should care MORE, not less"
    assert concerned.dopamine > mirror.dopamine, "engaged, where the mirror is flattened"
    # The bound on what caring costs: attention is cheap, catching it is not.
    cost_of_concern = quiet.serotonin - concerned.serotonin
    cost_of_mirroring = quiet.serotonin - mirror.serotonin
    assert cost_of_concern < cost_of_mirroring / 2
