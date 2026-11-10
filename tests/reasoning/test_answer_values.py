"""Order 0055 (Cowork's Marie Curie test of change 3): a question that names the class of its answer narrows a last hop
with many values to that class's values, behind ``CB_ANSWER_CLASS=1`` (`cubbyllm/reasoning/answer_values.py`).

Pinned:
- off (the default): the walk is unchanged, a last hop with several values is refused, naming them;
- on: "Which Nobel Prize ..." keeps the Nobel Prizes; one left is the answer, two are said together (each branch
  certified), in their labels' order; the trace holds each value's own fact;
- a question naming no class, or a class no value is of, is refused as before; more than SAY_MAX left: refused, the
  narrowed values named;
- a branch the VM rejects keeps the whole refusal (nothing said for the others);
- only the last hop: a mid-chain hop's values are never narrowed by the answer's class;
- the class is read from capitalised words only ("which city" names none).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from cubbyllm.reasoning import answer_values as V  # noqa: E402
from test_branching import ask, echo_vm  # noqa: E402

AW = "award received"
CURIE = [f"{v} is the {AW} of marie" for v in
         ("Davy Medal", "Nobel Prize in Physics", "Matteucci Medal", "Nobel Prize in Chemistry", "Albert Medal")]
Q = f"Which Nobel Prize is the {AW} of marie?"


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("CB_ANSWER_CLASS", "1")


def test_off_a_last_hop_with_several_values_is_refused_as_before(monkeypatch):
    monkeypatch.delenv("CB_ANSWER_CLASS", raising=False)
    r = ask(Q, CURIE)
    assert r.reason == "ambiguous_hop" and not r.verified and len(r.refused["objects"]) == 5


def test_the_values_of_the_class_asked_are_said_together(on):
    r = ask(Q, CURIE)
    assert r.verified and r.answer == "Nobel Prize in Chemistry and Nobel Prize in Physics"
    assert [h.fact for h in r.trace] == [f"Nobel Prize in Chemistry is the {AW} of marie",
                                         f"Nobel Prize in Physics is the {AW} of marie"]
    assert {b["answer"] for b in r.branches} == {"Nobel Prize in Chemistry", "Nobel Prize in Physics"}


def test_one_value_of_the_class_is_the_answer(on):
    r = ask(Q, [f for f in CURIE if "Chemistry" not in f])
    assert r.verified and r.answer == "Nobel Prize in Physics" and len(r.trace) == 1


def test_no_class_named_or_none_of_it_is_refused_as_before(on):
    assert ask(f"What is the {AW} of marie?", CURIE).reason == "ambiguous_hop"
    assert ask(f"Which Grammy Award is the {AW} of marie?", CURIE).reason == "ambiguous_hop"


def test_more_than_say_max_left_is_refused_naming_them(on):
    many = CURIE + [f"Nobel Prize in {x} is the {AW} of marie" for x in ("Peace", "Literature")]
    r = ask(Q, many)
    assert r.reason == "ambiguous_hop" and len(r.refused["objects"]) == 4
    assert all("nobel prize" in o for o in r.refused["objects"])


def test_a_branch_the_vm_rejects_keeps_the_refusal(on):
    r = ask(Q, CURIE, vm=echo_vm(reject=("Nobel Prize in Chemistry",)))
    assert r.reason == "ambiguous_hop" and not r.verified


def test_only_the_last_hop_is_narrowed(on):
    facts = ["Nobel Prize Land is the country of citizenship of marie", "Ruritania is the country of citizenship of marie",
             "Paris is the capital of Nobel Prize Land", "Strelsau is the capital of Ruritania"]
    r = ask("Which Nobel Prize is the capital of the country of citizenship of marie?", facts)
    assert not r.verified and r.reason == "ambiguous_hop" and r.refused["hop"] == 0


def test_the_class_is_read_from_capitalised_words_only(on):
    assert V.wanted("Which city is the capital of the Philippines?", "capital of the Philippines") == ()
    assert V.wanted("Which Nobel Prizes did Marie Curie win?", f"{AW} of Marie Curie") == ("nobel", "prize")
    assert V.fits("Nobel Prize in Physics", ("nobel", "prize")) and not V.fits("Nobelium", ("nobel",))
