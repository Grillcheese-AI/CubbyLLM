"""Order 0073, step 3 (validation/builders/view_compose.py): a long instruction's text with what each check says.
Each connective is said before and after its partner and its check carries the tag (``prev`` or ``next``, the
neighbour it ties to; "either" on the first of its two); the segments tile the text; the parts are numbered as the
text states them; a check's said forms are read back from how it was worded, and a wording with no description
gives none."""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT)]
import view_compose as C  # noqa: E402
import view_said as S  # noqa: E402

A = {"q": "n", "view": "sentence", "op": ">=", "v": 3}
B = {"q": "n", "view": "word", "op": "<", "v": 40}
BEGINS = {"col": "first_word", "op": "==", "v": "so"}
PHRASES = {"en": [("warm", "Keep it friendly.")], "fr": [("warm", "Reste aimable.")]}


def made(seed, parts, groups, lang="en", share=0.0):
    text, segs, order = C.compose(random.Random(seed), parts, groups, lang, PHRASES, share)
    return (text, *C.renumber(parts, groups, segs, order))


def checks(segs):
    return [s for s in segs if s["kind"] == "check"]


@pytest.mark.parametrize("kind,lang,first,after", [("if", "en", "If ", " if "), ("if", "fr", "Si ", " si "),
                                                   ("unless", "en", "Unless ", ", unless "),
                                                   ("unless", "fr", "Sauf si ", ", sauf si ")])
def test_each_connective_is_said_before_and_after_its_partner(kind, lang, first, after):
    group = ("if", 0, 1) if kind == "if" else ("unless", 1, 0)             # part 0 is the condition in both
    seen = Counter()
    for seed in range(40):
        text, parts, groups, segs = made(seed, [A, B], [group], lang)
        one, two = checks(segs)
        tagged = one if one.get("join") else two
        assert tagged.get("role") == "condition" and tagged["how"] == "cond" and tagged["join"][0] == kind
        assert sum(1 for s in segs if s.get("join")) == 1
        said = text[tagged["start"]:tagged["end"]]
        if tagged["join"][1] == "next":                                    # the connective first: it ties to the next check
            assert tagged is one and said.startswith(first) and parts == [A, B]
            assert groups == [group]
        else:
            assert tagged is two and said.startswith(after) and parts == [B, A]
            assert groups == [(kind, 1, 0) if kind == "if" else (kind, 0, 1)]
        seen[tagged["join"][1]] += 1
    assert seen["next"] >= 8 and seen["prev"] >= 8


def test_either_is_tagged_on_the_first_of_its_two_checks():
    for lang, word in (("en", "Either "), ("fr", "Soit ")):
        for seed in range(10):
            text, parts, groups, segs = made(seed, [A, B], [("either", 0, 1)], lang)
            one, two = checks(segs)
            assert one["join"] == ["either", "next"] and "join" not in two and text[one["start"]:].startswith(word)
            assert parts == [A, B] and groups == [("either", 0, 1)]


def test_segments_tile_the_text_and_parts_are_numbered_in_its_order():
    for seed in range(30):
        parts = [A, B, {"q": "any", "view": "sentence", "where": [BEGINS]}, {"not": A}]
        groups = [("one", 0), ("if", 1, 2), ("one", 3)] if seed % 2 else [("either", 0, 1), ("unless", 3, 2)]
        text, parts, groups, segs = made(seed, parts, groups, "fr" if seed % 3 == 0 else "en")
        at = 0
        for s in segs:
            assert s["start"] >= at and text[at:s["start"]].strip() == "" and text[s["start"]:s["end"]].strip()
            at = s["end"]
        assert text[at:].strip() == ""
        assert [s["part"] for s in checks(segs)] == list(range(4))
        assert sorted(x for g in groups for x in g[1:]) == list(range(4))


def test_said_forms_are_read_from_how_each_check_was_worded():
    text, parts, groups, segs = made(3, [A, {"not": {"q": "any", "view": "sentence", "where": [BEGINS]}}],
                                     [("one", 0), ("one", 1)])
    said = C.said_forms(parts, segs, "en")
    assert [s["how"] for s in said] == ["imp", "imp"]
    for part, s in zip(parts, said):
        assert s["w"] == S.said(part, "en", s["how"]) and s["m"] == S.VS.fold(s["w"]) and S.marks(s["m"]) == 0
    hows = Counter()
    for seed in range(40):                                                 # goal first where there is such a wording
        text, parts, groups, segs = made(seed, [A, B], [("one", 0), ("one", 1)], "en", share=1.0)
        hows.update(s["how"] for s in checks(segs))
        assert C.said_forms(parts, segs, "en") is not None
    assert hows["inverted"] == 80


def test_a_wording_with_no_description_gives_no_said_forms():
    two = {"q": "all", "view": "sentence", "where": [BEGINS, {"col": "n_word", "op": ">", "v": 3}]}
    text, parts, groups, segs = made(4, [A, {"not": two}], [("one", 0), ("one", 1)])   # "... or ...": an or-filter
    assert C.said_forms(parts, segs, "en") is None
