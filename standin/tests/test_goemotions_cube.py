"""The GoEmotions -> cube mapping has to survive the agent's own classifier.

A coordinate table is the worst place for a mistake: it is 58k rows of corpus
downstream, all of it plausible-looking, and nothing in training will tell you.
So the table is derived from two tables already in the tree and then checked
against the real nearest-corner function rather than eyeballed.

These tests are the shape of the checks the hand-written table proposed on
2026-09-15 would have failed:

  * its `calculate_nearest_corner` had `(0,1,1)` twice in one dict literal, so
    ANGER_CORNER was unreachable and every angry row was labelled anticipation;
  * `neutral` — the LARGEST class in GoEmotions — binarised to a key that was
    not in the table at all and fell through to a `SADNESS_CORNER` default;
  * it binarised on `>= 0.5` rather than measuring distance, which is the
    threshold-versus-geometry mismatch that caused the original contradiction,
    moved into the data where it is much harder to see.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_chat_sft import GOEMOTIONS, GOEMOTIONS_PETAL  # noqa: E402
from goemotions_cube import (CENTRE, LABEL_CUBE, PETAL_CORNER, PLACED,  # noqa: E402
                             UNPLACED, label_coords, nearest_corner, style_bank_key, validate)
from neurochem import Neurochemistry  # noqa: E402


def test_every_placed_label_lands_on_the_corner_its_petal_implies():
    """The whole point. A mapping that cannot survive the classifier the agent
    actually uses is a mapping that will disagree with him at runtime."""
    errors = [p for p in validate() if not p.startswith("note:")]
    assert errors == [], errors


def test_every_corner_is_reachable():
    """The duplicate-key bug: ANGER was written into the recipe's table and was
    still unreachable, because a later key overwrote it and Python said
    nothing. Reached here by construction, checked here anyway."""
    landed = {nearest_corner(v) for lab, v in PLACED.items() if lab != "neutral"}
    for corner in ("joy", "warm", "anxious", "surprise", "sad", "contempt", "angry"):
        assert corner in landed, f"no GoEmotions label reaches {corner!r}"


def test_neutral_is_the_centre_and_not_some_corners_fallback():
    """`neutral` is ~28% of GoEmotions. The recipe's table binarised it to a
    key that was absent and dropped it into sadness, which would have made a
    third of the corpus sad."""
    assert label_coords("neutral") == CENTRE
    assert style_bank_key("neutral") is None, "the centre belongs to no corner"


def test_anticipation_labels_are_left_unplaced_rather_than_rounded():
    """Plutchik has eight petals and Lövheim has eight corners and they are not
    the same eight: `anticipation` is an override in `_classify_emotion`, not a
    vertex. Labels with nowhere to go say so instead of landing on a
    neighbour."""
    assert set(UNPLACED) == {"curiosity", "desire", "optimism"}
    for lab in UNPLACED:
        assert GOEMOTIONS_PETAL[lab] == "anticipation"
        assert label_coords(lab) is None and style_bank_key(lab) is None


def test_intensity_orders_within_a_petal():
    """annoyance sits inside anger, not beside it: same corner, less far out."""
    def reach(lab):
        v = label_coords(lab)
        return sum((x - 0.5) ** 2 for x in v) ** 0.5
    assert reach("annoyance") < reach("anger")
    assert reach("nervousness") < reach("fear")
    assert reach("sadness") < reach("grief")
    assert reach("approval") < reach("caring") < reach("admiration")


def test_the_table_covers_the_whole_label_set():
    assert set(LABEL_CUBE) == set(GOEMOTIONS) == set(GOEMOTIONS_PETAL)
    assert len(PLACED) + len(UNPLACED) == len(GOEMOTIONS) == 28


def test_it_is_distance_and_not_a_threshold():
    """The mismatch that started all this. A point just over 0.5 on one axis
    belongs to whichever corner is NEARER, which binarising cannot express."""
    v = (0.52, 0.9, 0.85)                                # barely high on 5-HT
    assert nearest_corner(v) == "joy"
    binarised = tuple(1.0 if x >= 0.5 else 0.0 for x in v)
    assert binarised == (1.0, 1.0, 1.0)                  # agrees here...
    v2 = (0.48, 0.9, 0.85)                               # ...and a hair the other way
    assert nearest_corner(v2) == "angry"
    assert Neurochemistry._CORNERS["angry"] == (0, 1, 1)
