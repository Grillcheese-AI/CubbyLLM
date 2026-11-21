"""cubbyllm/affect/mood.py (order 0079): eight scales from the hormones, their names and the GoEmotions shade."""
from __future__ import annotations

from cubbyllm.affect import mood
from cubbyllm.affect.neurochem import Neurochemistry


def test_the_checks_0079_sets_pass():
    assert mood.validate() == []


def test_the_label_tables_equal_the_builders_checked_ones():
    from validation.builders.goemotions_cube import TIER
    from validation.sft.chat_sft_v6 import GOEMOTIONS_PETAL
    assert mood.LABEL_PETAL == dict(GOEMOTIONS_PETAL) and mood.LABEL_TIER == dict(TIER)


def test_a_fresh_body_is_calm():
    sc = mood.scales(Neurochemistry())
    assert set(sc) == set(mood.ORDER) and all(v == 0 for v in sc.values())
    assert mood.named(sc) == ["calm"] and mood.shade(sc) == []
    assert mood.line(sc) == "Mood: joy 0, trust 0, fear 0, surprise 0, sadness 0, disgust 0, anger 0, anticipation 0"


def test_scales_are_bounded_and_signed():
    q = Neurochemistry.quiescent()
    for pos in [(0, 0, 0), (1, 1, 1), (0.2, 0.9, 0.4)]:
        for ot in (0.05, q["OT"], 0.85):
            sc = mood.scales_at(pos, ot, q["OT"], Neurochemistry._BAND["OT"])
            assert all(-10 <= v <= 10 for v in sc.values())
    assert mood.scales_at((0.5, 0.5, 0.5), 0.85, q["OT"], Neurochemistry._BAND["OT"])["trust"] == 10
    assert mood.scales_at((0.5, 0.5, 0.5), 0.05, q["OT"], Neurochemistry._BAND["OT"])["trust"] == -10


def test_names_tiers_and_dyads():
    sc = dict.fromkeys(mood.ORDER, 0) | {"joy": 6, "trust": 5, "anticipation": 2}
    assert mood.named(sc)[:3] == ["love", "joy", "trust"]
    assert mood.named(dict.fromkeys(mood.ORDER, 0) | {"anger": 9}) == ["rage"]
    assert mood.line(sc).startswith("Mood: joy +6, trust +5, fear 0")


def test_the_shade_follows_the_strong_petals_and_leans_only_on_plutchiks_dyads():
    sc = dict.fromkeys(mood.ORDER, 0) | {"joy": 9, "trust": 5}
    assert mood.shade(sc)[0] == "love"                     # joy at its outer tier, leaning on trust
    sc = dict.fromkeys(mood.ORDER, 0) | {"sadness": 5, "disgust": 4}
    assert mood.shade(sc)[0] == "remorse"
    sc = dict.fromkeys(mood.ORDER, 0) | {"sadness": 5}
    assert mood.shade(sc) == ["sadness"]
    assert mood.shade(dict.fromkeys(mood.ORDER, 3)) == []   # nothing at +4 or more: no shade


def test_a_neighbour_does_not_light_up_from_the_axes_it_shares():
    """0079, 2026-11-21: near joy (5-HT and DA high, NE low) read boredom too; attentive (NE up, 5-HT just above the
    centre) read annoyance beside interest. A positive level needs the state on its corner's side on every axis."""
    from cubbyllm.affect import mood
    from cubbyllm.affect.neurochem import Neurochemistry
    q, band = Neurochemistry.quiescent(), Neurochemistry._BAND["OT"]
    content = mood.scales_at((0.79, 0.74, 0.14), q["OT"], q["OT"], band)
    assert content["joy"] >= 4 and content["disgust"] <= 0
    attentive = mood.scales_at((0.54, 0.60, 0.59), q["OT"], q["OT"], band)
    assert attentive["anticipation"] >= 1 and attentive["anger"] <= 0
    assert mood.validate() == []
