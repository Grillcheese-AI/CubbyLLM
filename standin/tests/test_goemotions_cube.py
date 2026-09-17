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
    # Corner names follow the corrected `_CORNERS` (2026-09-17). `spent` is not
    # in this list on purpose: it is the depletion vertex, and GoEmotions has no
    # label for being out of fuel - depletion is a state you reach by exhausting
    # yourself, not one anybody posts a Reddit comment in. It is reachable from
    # the ODE, which is the reachability that matters, and `cube_occupancy.py`
    # is where that gets checked.
    for corner in ("joy", "fear", "surprise", "distress", "disgust", "anger", "interest"):
        assert corner in landed, f"no GoEmotions label reaches {corner!r}"


def test_neutral_is_the_centre_and_not_some_corners_fallback():
    """`neutral` is ~28% of GoEmotions. The recipe's table binarised it to a
    key that was absent and dropped it into sadness, which would have made a
    third of the corpus sad."""
    assert label_coords("neutral") == CENTRE
    assert style_bank_key("neutral") is None, "the centre belongs to no corner"


def test_trust_labels_are_left_unplaced_rather_than_rounded():
    """This test used to assert the opposite, and the flip is the finding.

    It read: anticipation has no vertex, so curiosity/desire/optimism stay
    unplaced. True at the time, but only because JOY had been written onto
    Lövheim's interest/excitement corner - anticipation was homeless because
    something else was standing on its coordinate.

    With the table corrected, anticipation has (1,1,1) and TRUST is the petal
    with nowhere to go. That is a better kind of gap: trust is oxytocin, and
    oxytocin is not an axis of a cube built from serotonin, dopamine and
    noradrenaline. No relabelling can produce it, and the honest move is to say
    so rather than round admiration and gratitude onto a neighbour."""
    assert set(UNPLACED) == {"admiration", "approval", "caring", "gratitude"}
    for lab in UNPLACED:
        assert GOEMOTIONS_PETAL[lab] == "trust"
        assert label_coords(lab) is None and style_bank_key(lab) is None


def test_intensity_orders_within_a_petal():
    """annoyance sits inside anger, not beside it: same corner, less far out."""
    def reach(lab):
        v = label_coords(lab)
        return sum((x - 0.5) ** 2 for x in v) ** 0.5
    assert reach("annoyance") < reach("anger")
    assert reach("nervousness") < reach("fear")
    assert reach("sadness") < reach("grief")
    # The trust ladder (approval < caring < admiration) is gone from this check:
    # those three are UNPLACED now, and `label_coords` returns None for them, so
    # the old line raised TypeError rather than failing an assertion. Curiosity
    # inside desire is the same ordering on the petal that gained a corner.
    assert reach("curiosity") < reach("desire")


def test_the_table_covers_the_whole_label_set():
    assert set(LABEL_CUBE) == set(GOEMOTIONS) == set(GOEMOTIONS_PETAL)
    assert len(PLACED) + len(UNPLACED) == len(GOEMOTIONS) == 28


def test_it_is_distance_and_not_a_threshold():
    """The mismatch that started all this. A point just over 0.5 on one axis
    belongs to whichever corner is NEARER, which binarising cannot express."""
    v = (0.52, 0.9, 0.85)                                # barely high on 5-HT
    assert nearest_corner(v) == "interest"                # (1,1,1), not `joy` any more
    binarised = tuple(1.0 if x >= 0.5 else 0.0 for x in v)
    assert binarised == (1.0, 1.0, 1.0)                  # agrees here...
    v2 = (0.48, 0.9, 0.85)                               # ...and a hair the other way
    assert nearest_corner(v2) == "anger"
    assert Neurochemistry._CORNERS["anger"] == (0, 1, 1)
    # All-high is interest/excitement, not joy: this one line is the remap in
    # miniature. Joy is (1,1,0) - consummatory, low noradrenaline - and putting
    # it on the all-high vertex is what left anticipation with no corner and
    # forced the novelty override that has now been deleted.
    assert Neurochemistry._CORNERS["interest"] == (1, 1, 1)
    assert Neurochemistry._CORNERS["joy"] == (1, 1, 0)
