"""Order 0073, step 1: a part written as its words say it (a ``not`` on a condition, on the atom, or around it;
``absent: pass`` on ``at``) and `view_spec.fold`, which writes it with no ``not``. The fold keeps the truth on replies
that have the row and on replies that do not; a two-valued column has one folded spelling; ``not all`` of two
conditions has none; the compiler takes folded forms only, and ``absent: pass`` has its own text."""
from __future__ import annotations

import pytest

pytest.importorskip("regex")
from cubbyllm.speech import view_compile as VC  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

BEGINS = {"col": "first_word", "op": "==", "v": "so"}
LAST = {"q": "at", "view": "sentence", "pos": -1, "col": "end_mark", "op": "==", "val": "?"}
NONE, ONE = {"sentence": [], "line": []}, {"sentence": [{"end_mark": "?", "first_word": "so", "n_word": 2}],
                                             "line": [{"item": "yes", "first_word": "so", "n_word": 2}]}
TWO = {"sentence": ONE["sentence"] + [{"end_mark": ".", "first_word": "we", "n_word": 5}],
       "line": ONE["line"] + [{"item": "no", "first_word": "we", "n_word": 5}]}
REPLIES = (NONE, ONE, TWO)


def same(part) -> dict:
    """`fold(part)`, checked to have `part`'s truth on the three replies and to carry no ``not``."""
    f = VS.fold(part)
    assert "not" not in f and not any("not" in c for c in f.get("where", []))
    assert [VS.evaluate(f, v) for v in REPLIES] == [VS.evaluate(part, v) for v in REPLIES]
    return f


def test_a_not_on_a_condition_turns_its_comparison():
    said = {"q": "any", "view": "sentence", "where": [dict(BEGINS, **{"not": True})]}
    assert same(said) == {"q": "any", "view": "sentence", "where": [dict(BEGINS, op="!=")]}
    n = {"q": "all", "view": "sentence", "where": [{"col": "n_word", "op": ">", "v": 3, "not": True}]}
    assert same(n)["where"] == [{"col": "n_word", "op": "<=", "v": 3}]


def test_a_not_on_the_quantifier_or_the_comparison_folds_into_it():
    any_ = {"q": "any", "view": "sentence", "where": [BEGINS]}
    assert same({"not": any_}) == dict(any_, q="none") == same(dict(any_, **{"not": True}))
    assert same({"not": dict(any_, q="none")}) == any_ == same({"not": {"not": any_}})
    count = {"q": "count", "view": "sentence", "op": ">", "v": 1, "where": [BEGINS]}
    assert same({"not": count})["op"] == "<=" and same({"not": {"q": "n", "view": "sentence", "op": "==", "v": 2}})["op"] == "!="


def test_not_all_of_one_condition_is_any_of_its_complement_and_of_two_has_no_fold():
    every = {"q": "all", "view": "sentence", "where": [BEGINS]}
    assert same({"not": every}) == {"q": "any", "view": "sentence", "where": [dict(BEGINS, op="!=")]}
    with pytest.raises(ValueError):
        VS.fold({"not": dict(every, where=[BEGINS, {"col": "n_word", "op": "<", "v": 4}])})


def test_not_at_holds_when_there_is_no_such_row():
    f = same({"not": LAST})
    assert f == dict(LAST, op="!=", absent="pass")
    assert [VS.evaluate(f, v) for v in REPLIES] == [True, False, True]      # no sentence; ends with "?"; ends with "."
    assert same(dict(LAST, absent="pass", **{"not": True})) == f            # as said: "if there is one, should not"
    assert same({"not": f}) == LAST                                         # and back: the plain atom fails on none
    assert [VS.evaluate(LAST, v) for v in REPLIES] == [False, True, False]


def test_a_two_valued_column_has_one_folded_spelling():
    for said in ({"col": "item", "op": "==", "v": "yes", "not": True}, {"col": "item", "op": "!=", "v": "yes"},
                 {"col": "item", "op": "!=", "v": "no", "not": True}, {"col": "item", "op": "==", "v": "no"}):
        f = same({"q": "all", "view": "line", "where": [said]})
        assert f["where"] == [{"col": "item", "op": "==", "v": "no"}]
    assert same({"q": "any", "view": "sentence", "where": [dict(BEGINS, op="!=")]})["where"] == [dict(BEGINS, op="!=")]


def test_the_compiler_takes_folded_forms_only_and_writes_absent_pass():
    with pytest.raises(ValueError):
        VC.body({"q": "any", "view": "sentence", "where": [dict(BEGINS, **{"not": True})]})
    with pytest.raises(ValueError):
        VC.body(dict(LAST, **{"not": True}))
    assert VC.body(LAST) == ("create c1: number; assign c1 = 0;\n"
                             "create p2: number; assign p2 = n_sentence; sub p2, 1;\n"
                             'if (p2 >= 0) { if (p2 < n_sentence) { let x3 = sentence_end_mark[p2]; '
                             'if (x3 == "?") { assign c1 = 1; } } }\n'
                             "return c1;")
    assert VC.body(VS.fold({"not": LAST})) == (
        "create c1: number; assign c1 = 0;\n"
        "create p2: number; assign p2 = n_sentence; sub p2, 1;\n"
        "assign c1 = 1;\n"
        'if (p2 >= 0) { if (p2 < n_sentence) { assign c1 = 0; let x3 = sentence_end_mark[p2]; '
        'if (x3 != "?") { assign c1 = 1; } } }\n'
        "return c1;")
