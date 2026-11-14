"""Order 0073, step 2 (validation/builders/view_said.py): the said form of a wording. Amounts are written as said
("at least one" is ``any``, "no" is ``none``, numbers never carry a ``not``); a ``not`` sits only where a verb is
negated, so English and French can differ on the same part; "if there is one" is ``absent: pass``; arm M's form is
the fold of arm W's and never carries a ``not``."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders")]
import view_said as S  # noqa: E402

BEGINS = {"col": "first_word", "op": "==", "v": "so"}
NOT = {"not": True}


def both(part, lang, how="imp"):
    text, w, m = S.say(part, lang, random.Random(0), how)
    if w is not None:
        assert S.marks(w) == S.negations(text, lang) and S.marks(m) == 0
    return text, w, m


def atom(q, where, **kw):
    return dict({"q": q, "view": "sentence", "where": where}, **kw)


def test_amounts_are_written_as_said():
    assert both(atom("count", [BEGINS], op=">", v=0), "en")[1] == atom("any", [BEGINS])         # "at least one"
    assert both(atom("count", [BEGINS], op="!=", v=0), "fr")[1] == atom("any", [BEGINS])
    assert both(atom("count", [BEGINS], op=">=", v=3), "en")[1] == atom("count", [BEGINS], op=">=", v=3)
    assert both({"not": atom("count", [BEGINS], op=">", v=3)}, "en")[1] == atom("count", [BEGINS], op="<=", v=3)
    assert both({"q": "n", "view": "sentence", "op": "<", "v": 1}, "en")[1] == {"q": "n", "view": "sentence", "op": "==", "v": 0}
    long = atom("any", [{"col": "n_word", "op": ">", "v": 0}])                                 # "that has at least one word"
    assert both(long, "en")[1]["where"] == [{"col": "n_word", "op": ">=", "v": 1}]
    assert both({"not": atom("all", [{"col": "n_word", "op": ">", "v": 5}])}, "en")[1]["where"] == [
        {"col": "n_word", "op": "<=", "v": 5}]                                                  # "at most 5": no not


def test_no_is_a_not_only_where_a_verb_is_negated():
    text, w, m = both(atom("none", [BEGINS]), "en")
    assert text.startswith("Do not include any") and w == atom("any", [BEGINS], **NOT) and m == atom("none", [BEGINS])
    text, w, m = both(atom("none", [BEGINS]), "fr")
    assert text.startswith("Ne mets aucune") and w == m == atom("none", [BEGINS])
    word = {"q": "none", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}]}
    for lang, start in (("en", "Do not use"), ("fr", "N'utilise pas")):
        text, w, m = both(word, lang)
        assert text.startswith(start) and w == dict(word, q="any", **NOT) and m == word
    assert both(atom("count", [BEGINS], op="<", v=1), "en")[1] == atom("any", [BEGINS], **NOT)   # "no" from a count


def test_a_negated_condition_carries_the_not_and_every_is_said_as_no_in_english():
    differs = atom("all", [dict(BEGINS, op="!=")])
    text, w, m = both(differs, "en")
    assert text.startswith("No sentence should begin") and w == m == atom("none", [BEGINS])
    text, w, m = both(differs, "fr")
    assert "ne commence pas" in text and w == atom("all", [dict(BEGINS, **NOT)]) and m == differs
    text, w, m = both({"not": atom("all", [BEGINS])}, "en")
    assert "that does not begin" in text and w == atom("any", [dict(BEGINS, **NOT)]) and m == atom("any", [dict(BEGINS, op="!=")])
    two = {"not": atom("all", [BEGINS, {"col": "n_word", "op": "<", "v": 4}])}
    assert both(two, "en")[0] and both(two, "en")[1:] == (None, None)                           # "... or ...": no form


def test_if_there_is_one_is_absent_pass():
    last = {"q": "at", "view": "sentence", "pos": -1, "col": "end_mark", "op": "==", "val": "?"}
    text, w, m = both({"not": last}, "en")
    assert "if there is one, should not end" in text
    assert w == dict(last, absent="pass", **NOT) and m == dict(last, op="!=", absent="pass")
    text, w, m = both(last, "fr")
    assert "s'il y en a" not in text and w == m == last


def test_a_list_line_and_closing_punctuation_are_said_one_way():
    line = {"q": "any", "view": "line", "where": [{"col": "item", "op": "==", "v": "no"}]}
    text, w, m = both(line, "en")
    assert "is not part of a list" in text and w["where"] == [{"col": "item", "op": "==", "v": "yes", "not": True}]
    assert m["where"] == [{"col": "item", "op": "==", "v": "no"}]
    assert both(dict(line, q="all"), "en")[1] == {"q": "none", "view": "line", "where": [{"col": "item", "op": "==", "v": "yes"}]}
    ends = atom("any", [{"col": "end_mark", "op": "!=", "v": "none"}])
    text, w, m = both(ends, "en")
    assert "ends with closing punctuation" in text and w == m == ends


def test_a_condition_and_a_goal_first_wording_have_their_own_forms():
    text, w, m = both(atom("none", [BEGINS]), "en", "cond")
    assert text.startswith("your reply has no sentence") and w == m == atom("none", [BEGINS])
    word = {"q": "none", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}]}
    text, w, m = both(word, "fr", "cond")
    assert "n'emploie pas le mot" in text and w == dict(word, q="any", **NOT) and m == word
    text, w, m = both({"not": atom("count", [BEGINS], op="<", v=2)}, "en", "inverted")
    assert text.startswith("At least 2 sentences") and w == m == atom("count", [BEGINS], op=">=", v=2)
    text, w, m = both(atom("none", [BEGINS]), "en", "inverted")
    assert text.startswith("No sentence that") and w == m == atom("none", [BEGINS])
    assert both({"q": "n", "view": "sentence", "op": "==", "v": 0}, "fr", "inverted") == (None, None, None)
    assert both(word, "en", "inverted") == (None, None, None)                                   # said as its own noun


def test_a_quoted_value_is_not_a_negation_word():
    assert S.negations('Do not use the word "not".', "en") == 1
    assert S.negations("N'utilise pas le mot « pas ».", "fr") == 1
    assert S.negations("Ne mets aucune phrase qui finit par un point.", "fr") == 0
    quote = {"q": "none", "view": "char", "where": [{"col": "char", "op": "==", "v": '"'}, {"col": "char", "op": "!=", "v": "M"}]}
    text, w, _ = both(quote, "en")                                          # the value is the quote mark itself
    assert '"""' in text and S.negations(text, "en") == 2 == S.marks(w)
    assert S.negations("Ne mets aucun caractère qui est le caractère « » » et qui n'est pas le caractère « M ».", "fr") == 1
