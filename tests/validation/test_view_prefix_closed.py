"""validation/exp_view_prefix_closed.py: an answer whose every prefix its grammar allows is one the screened decoder
(`ec7044c`) writes unchanged; a prefix the grammar refuses is found, counted and placed. Pinned on stub grammars and
on the real typed grammar: the generator's own answer has no refused prefix, the same answer with a literal of the
wrong type has one, at the literal."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT / "tests" / "speech")]
import exp_view_prefix_closed as P  # noqa: E402
from test_view_types import GOLD  # noqa: E402

INSTR = 'Do not include any sentence that has a number of words other than 2 and that begins with the word "-".'
GOOD = f"\ncheck 1 | {INSTR}\n---\n" + GOLD


class Stub:
    def __init__(self, viable):
        self.viable = viable


def test_no_refused_prefix_when_the_grammar_allows_each():
    assert P.refused(Stub(lambda t: "abc".startswith(t)), "abc") == (None, 0)


def test_a_refused_prefix_the_answer_walks_through_is_found():
    assert P.refused(Stub(lambda t: t != "ab"), "abc") == (2, 1)           # allowed whole, refused on the way


def test_every_refused_prefix_is_counted_from_the_first():
    assert P.refused(Stub(lambda t: len(t) < 3), "abcd") == (3, 2)


def test_the_generators_answer_has_no_refused_prefix_under_the_typed_grammar():
    assert P.refused(P.grammar_class("typed")(INSTR), GOOD) == (None, 0)


def test_a_literal_of_the_wrong_type_is_refused_at_the_literal():
    bad = GOOD.replace('x4 == "-"', "x4 == 9")
    first, count = P.refused(P.grammar_class("typed")(INSTR), bad)
    assert count >= 1 and bad[:first].endswith("x4 == 9")                  # refused where the wrong literal starts


def test_the_summary_counts_by_split_and_among_the_counted_ids():
    results = [{"id": "a", "split": "dev", "prefixes": 10, "first": None, "refused": 0},
               {"id": "b", "split": "dev", "prefixes": 12, "first": 4, "refused": 2},
               {"id": "c", "split": "test", "prefixes": 7, "first": None, "refused": 0}]
    s = P.summary(results, {"dev": {"b"}, "test": {"c"}})
    assert s["dev"]["answers"] == 2 and s["dev"]["every prefix viable"] == 1 and s["dev"]["a refused prefix"] == 1
    assert s["dev"]["prefixes"] == 22 and s["dev"]["refused"] == [{"id": "b", "first": 4, "of": 12}]
    assert s["dev"]["counted rows"] == 1 and s["dev"]["counted with a refused prefix"] == 1
    assert s["test"]["counted rows"] == 1 and s["test"]["counted with a refused prefix"] == 0
    assert "test_long" not in s
