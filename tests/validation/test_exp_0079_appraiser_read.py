"""Order 0079's trunk read, --loose (Cowork, 2026-11-23): the first petal named anywhere in the answer, else the petal of
the first GoEmotions label named; the strict parse (perception.parse_emotion, as served) is used whenever it finds a
petal, and the answers in the asked "emotion -- petal" form are counted apart."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import exp_0079_appraiser_read as X  # noqa: E402

from cubbyllm.affect.perception import ModelAppraiser  # noqa: E402


def test_loose_petal():
    assert X.loose_petal("The Plutchik petal is a symbol of grief and loss") == "sadness"     # a GoEmotions label
    assert X.loose_petal("a symbol of fear and uncertainty, not of joy") == "fear"            # the first petal named
    assert X.loose_petal("Happiness and fulfillment.") is None


class Says:
    def __init__(self, answers):
        self.answers = list(answers)

    def emit(self, prompt, **kw):
        return self.answers.pop(0)


def test_reader_strict_first_and_the_form_counted_apart():
    answers = ["gratitude, joy — trust", "It is a symbol of grief and loss", "nothing to say"]
    for loose, want in ((False, ["trust", None, None]), (True, ["trust", "sadness", None])):
        cap = X.Capture(Says(answers))
        r = X.Reader(ModelAppraiser(cap, facts={}), cap, loose)
        assert [r.petal(t) for t in ("a", "b", "c")] == want
        assert (r.in_form, r.reads) == (1, 3)
