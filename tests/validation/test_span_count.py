"""Order 0051 A: the span finder (``validation/span_count.py``): word-aligned, leftmost-longest, at most 55 pieces,
no overlaps, deterministic; the two readings (every match / proper names) and the count's arithmetic."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import span_count as S  # noqa: E402

PIECES = ["The", " Eiff", "el", " Tower", " is", " in", " Par", "is", ",", " France", "."]


def test_words_start_at_spaces_and_punctuation():
    st = S.word_starts(PIECES)
    assert st == [True, True, False, True, True, True, True, False, True, True, True]


def test_the_longest_known_run_wins_and_runs_never_overlap():
    known = {"eiffel", "eiffel tower", "the eiffel tower", "tower is", "paris", "france"}
    ok = lambda t: S.fold(t) in known                                     # noqa: E731
    got = S.spans(S.candidates(PIECES, S.word_starts(PIECES)), ok)
    assert got == [(0, 4), (6, 8), (9, 10)]          # "The Eiffel Tower" (not "Eiffel"), "Paris", "France"
    assert all(a[1] <= b[0] for a, b in zip(got, got[1:]))
    assert got == S.spans(S.candidates(PIECES, S.word_starts(PIECES)), ok)   # deterministic


def test_a_span_is_capped_at_55_pieces_and_8_words():
    long = ["A"] + ["b"] * 60                                              # one word of 61 pieces
    assert S.candidates(long, S.word_starts(long)) == []
    words = [" w"] * 12
    cs = S.candidates(words, S.word_starts(words))
    assert max(j - i for i, j, _ in cs) == S.MAX_WORDS


def test_proper_names_keep_language_words_out():
    assert S.proper("The Eiffel Tower") and S.proper("Paris")
    assert not S.proper("is") and not S.proper("the tower") and not S.proper("1990")


class _Tk:
    def __init__(self, pieces):
        self.p = pieces

    def decode(self, ids):
        return self.p[ids[0]]


def test_the_count_folds_only_runs_of_two_or_more_pieces():
    import numpy as np
    known = {"eiffel tower", "the eiffel tower", "paris", "france", "is"}
    res = S.count({"x": np.arange(len(PIECES))[None, :]}, _Tk(PIECES), known_keys=known)["x"]
    a, b = res["a_every_match"], res["b_proper_names"]
    assert (a["folded"], a["saved"]) == (6, 4)        # "The Eiffel Tower" 4 pieces, "Paris" 2; "is", "France" one piece
    assert (b["folded"], b["saved"]) == (6, 4) and b["tokens"] == len(PIECES)


def test_a_capitalised_word_opening_a_sentence_is_no_name_alone():
    p = ["See", " also", " Paris", " is", ".", "\n", "The", " Eiffel", " Tower", ".", " Rome"]
    assert S.initial_at(p, 0) and not S.initial_at(p, 2) and S.initial_at(p, 6) and S.initial_at(p, 10)
    assert not S.proper("See", initial=True) and S.proper("Paris", initial=False)
    assert S.proper("The Eiffel Tower", initial=True)                       # two words: the last one says name
    assert all(p[i].strip() and p[j - 1].strip() for i, j, _ in S.candidates(p, S.word_starts(p)))
