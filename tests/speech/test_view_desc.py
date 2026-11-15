"""Order 0073, step 4 (cubbyllm/speech/view_desc.py): a check's description on one line. Every said form writes to a
line that reads back to it, in both spellings; arm M's grammar has no ``not``; a slot outside its set, a column of
another view, a comparison its column's type does not have, a number or a string past its bound are refused with
the slot named."""
from __future__ import annotations

import pytest

from cubbyllm.speech import view_desc as D
from cubbyllm.speech import view_spec as VS

SO = {"col": "first_word", "op": "==", "v": "so"}
LONG = {"col": "n_word", "op": ">", "v": 5}
FORMS = [
    ({"q": "n", "view": "word", "op": ">=", "v": 40}, "n word >= 40"),
    ({"q": "count", "view": "sentence", "op": "<", "v": 3, "where": [LONG]}, "count sentence < 3 where n_word > 5"),
    ({"q": "any", "view": "sentence", "where": [SO, LONG]}, 'any sentence where first_word == "so" and n_word > 5'),
    ({"q": "none", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}]}, 'none word where lower == "so"'),
    ({"q": "all", "view": "line", "where": [{"col": "item", "op": "==", "v": "yes"}]}, 'all line where item == "yes"'),
    ({"q": "at", "view": "sentence", "pos": 1, "col": "first_word", "op": "==", "val": "so"}, 'at sentence 1 first_word == "so"'),
    ({"q": "at", "view": "paragraph", "pos": -1, "col": "n_sentence", "op": "<=", "val": 2, "absent": "pass"},
     "at paragraph last n_sentence <= 2 absent pass"),
    ({"q": "any", "view": "char", "where": [{"col": "char", "op": "==", "v": '"'}]}, 'any char where char == "\\""'),
]
SAID = [
    ({"q": "any", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}], "not": True}, 'not any word where lower == "so"'),
    ({"q": "any", "view": "sentence", "where": [dict(SO, **{"not": True})]}, 'any sentence where not first_word == "so"'),
    ({"q": "at", "view": "sentence", "pos": 2, "col": "first_word", "op": "==", "val": "so", "not": True, "absent": "pass"},
     'at sentence 2 not first_word == "so" absent pass'),
]


@pytest.mark.parametrize("form,line", FORMS)
def test_a_folded_form_writes_to_a_line_that_reads_back(form, line):
    assert D.write(form) == line
    assert D.read(line, "m") == form == D.read(line, "w")
    assert D.read_at("check " + line + " | the text", 6, "m") == (form, 6 + len(line))


@pytest.mark.parametrize("form,line", SAID)
def test_a_said_form_keeps_its_not_and_folds_to_a_line_without_one(form, line):
    assert D.write(form) == line and D.read(line, "w") == form
    with pytest.raises(ValueError):
        D.read(line, "m")                                   # arm M's grammar has no `not`
    folded = D.write(VS.fold(form))
    assert "not " not in folded and D.read(folded, "m") == VS.fold(form)


@pytest.mark.parametrize("line", [
    "some word >= 40",                                      # a quantifier outside its set
    "n words >= 40",                                        # a view outside its set
    'any word where first_word == "so"',                    # a column of another view
    'any sentence where first_word > "so"',                 # a comparison a string does not have
    "any sentence where n_word == \"5\"",                   # a string where the column is a number
    "n word >= 12345",                                      # a number past four digits
    "n word >= 040",                                        # a number written two ways
    "n word >= 40 where len > 2",                           # `n` has no filter
    "count sentence < 3",                                   # `count` has one
    'any sentence where first_word == "so" and n_word > 5 and n_word < 9',   # three conditions
    "at sentence 0 n_word > 5",                             # a position starts at 1
    "not at sentence 1 n_word > 5",                         # a position's `not` goes before its column
    'any sentence where first_word == "' + "x" * 41 + '"',   # a string past forty characters
    'any sentence where first_word == "caf\\u00e9"',        # a string written two ways
    "n word >= 40 ",                                        # something after the description
])
def test_a_line_outside_the_grammar_is_refused(line):
    with pytest.raises(ValueError):
        D.read(line, "w")
