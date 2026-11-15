"""Order 0073, step 3 (validation/builders/view_tell.py): how much a row's replies tell. A part's mutants reverse one
slot each and a group's change its join once; a reply that another part fails tells nothing at the row's verdict
though it tells the part run alone; the replies added measure N - 1, N and N + 1 and a tied pair's four
combinations, a reply on which the row's verdict turns on that part or pair alone taken first."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT)]
import view_tell as T  # noqa: E402

AT_LEAST_3 = {"q": "n", "view": "sentence", "op": ">=", "v": 3}
HAS_SO = {"q": "any", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}]}
FEW_WORDS = {"q": "n", "view": "word", "op": "<", "v": 3}
ONES = [("one", 0), ("one", 1)]


def reply(sentences: int, words=()) -> dict:
    """A reply's views: so many sentences, these words."""
    return {"sentence": [{"n_word": 4, "first_word": "so", "end_mark": "."}] * sentences,
            "word": [{"lower": w, "len": len(w)} for w in words]}


def test_a_part_mutant_reverses_one_slot():
    assert T.part_mutants(AT_LEAST_3) == [dict(AT_LEAST_3, op="<"), dict(AT_LEAST_3, op="<=")]   # negated, mirrored
    assert T.part_mutants(dict(AT_LEAST_3, op="==")) == [dict(AT_LEAST_3, op="!=")]
    assert T.part_mutants(HAS_SO) == [dict(HAS_SO, q="none"),
                                      dict(HAS_SO, where=[{"col": "lower", "op": "!=", "v": "so"}])]
    at = {"q": "at", "view": "sentence", "pos": 1, "col": "n_word", "op": ">", "val": 5}
    got = T.part_mutants(at)
    assert got == [dict(at, op="<="), dict(at, op="<"), dict(at, absent="pass")]
    assert {k: v for k, v in at.items()} in T.part_mutants(dict(at, absent="pass"))             # absent toggled back
    count = {"q": "count", "view": "sentence", "op": "<", "v": 2, "where": [{"col": "n_word", "op": ">", "v": 5}]}
    assert len(T.part_mutants(count)) == 4 and count not in T.part_mutants(count)


def test_a_group_mutant_changes_one_join():
    groups = [("one", 0), ("either", 1, 2), ("one", 3)]
    got = T.group_mutants(groups, 4)
    assert [("one", 0), ("if", 1, 2), ("one", 3)] in got and [("one", 0), ("if", 2, 1), ("one", 3)] in got
    assert [("one", 0), ("one", 1), ("one", 2), ("one", 3)] in got                               # untied
    assert sorted(got[-2]) == sorted([("one", 3), ("one", 2), ("either", 1, 0)])                  # tied to part 0
    assert sorted(got[-1]) == sorted([("one", 0), ("one", 1), ("either", 3, 2)])                  # tied to part 3
    assert len(got) == 5 and groups not in got
    for alt in got:
        assert sorted(x for g in alt for x in g[1:]) == [0, 1, 2, 3]
    turned = T.group_mutants([("if", 0, 1)], 2)                                                  # no free neighbour
    assert turned == [[("either", 0, 1)], [("if", 1, 0)], [("one", 0), ("one", 1)]]
    assert T.group_mutants([("unless", 0, 1)], 2)[:2] == [[("if", 0, 1)], [("if", 1, 0)]]       # "unless" is "or"


def test_a_reply_another_part_fails_tells_the_part_alone_only():
    parts = [AT_LEAST_3, HAS_SO]
    lost = T.told(parts, ONES, [reply(2)])                    # two sentences, no "so": the row fails either way
    assert (lost["parts"], lost["parts told"], lost["parts told alone"]) == (4, 0, 3)   # no word: "a word other than so" fails too
    turns = T.told(parts, ONES, [reply(2, ["so"])])           # the row's verdict is now part 1's
    assert (turns["parts told"], turns["parts told alone"]) == (2, 4)
    pair = T.told(parts, [("either", 0, 1)], [reply(2, ["so"]), reply(3), reply(2)])
    assert (pair["groups"], pair["groups told"], pair["groups told alone"]) == (3, 3, 3)
    assert T.told(parts, [("either", 0, 1)], [reply(3, ["so"])])["groups told"] == 0             # both hold: every join passes


def test_replies_are_added_at_n_minus_1_n_and_n_plus_1():
    assert T.number(AT_LEAST_3) == 3 and T.number(HAS_SO) is None
    assert T.measured(AT_LEAST_3, reply(4)) == 4
    kept = [reply(5, ["so"]), reply(1)]
    pool = [reply(2), reply(2, ["so"]), reply(3), reply(3, ["so"]), reply(4, ["so"]), reply(9, ["so"])]
    add, reach = T.extras(random.Random(0), [AT_LEAST_3, HAS_SO], ONES, kept, pool)
    assert sorted(add) == [1, 3, 4] and reach == {"numbers": [[0, [2, 2, 2]]], "groups": []}     # where "so" holds
    add, reach = T.extras(random.Random(0), [AT_LEAST_3, HAS_SO], ONES, kept, [reply(2), reply(3)])
    assert sorted(add) == [0, 1] and reach["numbers"] == [[0, [1, 1, 0]]]                        # none where it holds
    add, reach = T.extras(random.Random(0), [AT_LEAST_3, HAS_SO], ONES, [reply(3, ["so"])], [reply(3, ["so"])])
    assert add == [] and reach["numbers"] == [[0, [0, 2, 0]]]                                    # kept already; nothing edited


def test_a_tied_pair_gets_its_four_combinations_and_a_tied_part_a_partner_that_leaves_it_the_verdict():
    parts, groups = [AT_LEAST_3, HAS_SO, FEW_WORDS], [("either", 0, 1), ("one", 2)]
    pool = [reply(3, ["so"]), reply(3), reply(2, ["so"]), reply(2), reply(4), reply(3, ["so", "a", "b", "c"])]
    add, reach = T.extras(random.Random(1), parts, groups, [], pool)
    assert reach["groups"] == [[0, [2, 2, 2, 2]]]
    assert reach["numbers"][0] == [0, [2, 2, 2]]              # part 0 at 2, 3 and 4 sentences with no "so" and few words
    assert reach["numbers"][1] == [2, [0, 0, 2]] and sorted(add) == [0, 1, 2, 3, 4, 5]   # four words: part 2's N + 1
    add, reach = T.extras(random.Random(1), parts, groups, [], [reply(2, ["so"])])
    assert reach["numbers"][0] == [0, [1, 0, 0]]              # "so" holds: the pair passes whatever part 0 says
    add, reach = T.extras(random.Random(1), [AT_LEAST_3, HAS_SO], [("if", 1, 0)], [], [reply(2, ["so"]), reply(2)])
    assert reach["numbers"] == [[0, [2, 0, 0]]]               # an `if`'s condition must hold for part 0 to decide
    assert reach["groups"] == [[0, [0, 2, 0, 2]]]             # ("so", three sentences): holds and fails; fails and fails
