"""Pins for the neurochemistry port (standin/neurochem.py) and the world
store (standin/worlds.py). No VM, no model — pure python.
Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import identity as idn  # noqa: E402
from neurochem import Neurochemistry, appraise  # noqa: E402
from worlds import FactStore, route_world  # noqa: E402


def test_threat_spikes_noradrenaline_fast_and_cortisol_slowly():
    n = Neurochemistry()
    c0, ne0 = n.cortisol, n.noradrenaline
    n.step_message({"novelty": 0.5, "threat": 1.0, "focus": 0.0, "valence": 0.0, "social": 0.0})
    assert n.noradrenaline > ne0 + 0.25, "NE is the fast per-frame stress signal"
    assert n.cortisol < c0 + 0.05, "cortisol is a slow EMA, not a per-frame spike"
    for _ in range(60):                                  # sustained threat -> HPA integrates
        n.update(threat=1.0)
    assert n.cortisol > c0 + 0.10


def test_social_warmth_raises_oxytocin_and_stays_in_identity_bands():
    n = Neurochemistry()
    d = n.step_message({"novelty": 0.5, "threat": 0.0, "focus": 0.0, "valence": 0.4, "social": 0.8})
    assert d["oxytocin"] > 0.30
    for h, (lo, hi) in idn.HORMONE_RANGE.items():
        assert lo <= d[h] <= hi


def test_modulate_threshold_direction():
    n = Neurochemistry()
    base = 0.30
    for _ in range(80):
        n.update(threat=1.0)                             # stressed: cortisol up
    assert n.modulate_threshold(base) > base * 0.98, "stress must not LOWER routing caution"
    m = Neurochemistry()
    for _ in range(10):
        m.update(novelty=1.0, valence=0.8)               # excited: dopamine up
    assert m.modulate_threshold(base) < n.modulate_threshold(base)


def test_appraise_is_bilingual_and_novelty_decays_with_vocabulary():
    seen: set[str] = set()
    en = appraise("HELP ME NOW!!", seen)
    assert en["threat"] >= 0.8
    fr = appraise("Merci beaucoup, bonjour !", seen)
    assert fr["social"] >= 0.4 and fr["valence"] > 0
    seen.update({"the", "capital", "of", "france"})
    first = appraise("the capital of france", set())
    again = appraise("the capital of france", seen)
    assert first["novelty"] > again["novelty"] == 0.0


def test_factstore_adds_dedupes_and_retrieves_incrementally():
    w = FactStore(["berlin is the capital of germany"])
    assert len(w) == 1 and not w.add("berlin  is the capital   of germany")
    assert w.add("Quuxville is the capital of Fnordovia")
    assert "Quuxville is the capital of Fnordovia" in w
    hits = w("capital of Fnordovia", 2)
    assert hits[0][1] == "Quuxville is the capital of Fnordovia"


def test_route_world_picks_the_best_scoring_world():
    geo = FactStore(["Quuxville is the capital of Fnordovia"], name="geo")
    space = FactStore(["mars is the planet of olympus mons"], name="space")
    name, score = route_world({"geo": geo, "space": space}, "which planet has olympus mons?")
    assert name == "space" and score > 0
