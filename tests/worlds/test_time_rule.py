"""The time rule and the same-name guard (order 0012, Cowork's review) on synthetic books: a value that changes over
its subject's life (a capital) is said flat only with a date -- undated, it is hedged in time, and refused for a year;
a dated one (a reign) is said as before; a book's person whose dates do not fit the subject's life is another person.
Run: python -m pytest standin/tests/test_time_rule.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ask import AskLoop  # noqa: E402
from book_source import BookSource, fits  # noqa: E402
from books import BookIndex, Mentions  # noqa: E402
from question_forms import TemplateEmitter  # noqa: E402
from test_books import KNOWN, NAMES, TimedStore, _book, _library, exact_vm  # noqa: E402
from test_search_learn import STORE, DictSource  # noqa: E402
from time_rule import at_one_point, changes, undated  # noqa: E402

MORE = ["Kush", "Kerma", "Samuel Fuller", "Redenhall"]


def _loop(tmp_path, extra=(), world_facts=(), tense=None):
    roots, events = _library(tmp_path)
    for rel, text in extra:
        _book(roots[1][0], rel, text)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES + MORE))
    world = TimedStore(STORE)
    for r in ("founded by", "place of birth", "date of birth", "head of state", "capital"):
        world.index.declare_relation(r)
    for f in list(KNOWN) + list(world_facts):
        world.add(f)
    return AskLoop(TemplateEmitter(), world=world, source=DictSource({}), lexicon=False, run_fn=exact_vm(),
                   books=BookSource(idx), tense=tense)


KUSH = [("EPUB/Nubia - Ann Lee/Ann Lee - Nubia (v1.0).epub.txt", "Kerma, the capital of Kush, stood above the third cataract.\n"),
        ("EPUB/Kush - Tom Hart/Tom Hart - Kush (v1.0).epub.txt", "Kerma, the capital of Kush, traded with Egypt.\n")]


def test_the_rule_names_the_relations_that_change_and_the_witnesses_that_date_them():
    assert changes("capital") and changes("head of state") and changes("spouse") and changes("population")
    assert not changes("date of birth") and not changes("place of birth") and not changes("founded by")
    assert undated("capital", [{"when": {}}, {"when": {"point": "1700"}}])          # every witness must date it
    assert not undated("capital", [{"when": {"start": "1154"}}]) and not undated("place of birth", [{"when": {}}])
    assert not undated("capital", [])                                               # no witness: not the books' fact
    assert at_one_point("Kerma", "capital", "Kush") == "Kerma was a capital of Kush at one point."


def test_an_undated_capital_on_two_books_is_hedged_in_time_and_refused_for_a_year(tmp_path):
    loop = _loop(tmp_path, KUSH)
    rec = loop.ask("What is the capital of Kush?")
    assert rec["verified"] and rec["answer"] == "Kerma" and rec["tier"] == "at_one_point"
    assert rec["reply"] == "Kerma was a capital of Kush at one point."            # two authors, still not flat
    rec = loop.ask("What was the capital of Kush in 1700?")                        # the tense check refuses it first
    assert not rec["verified"] and rec["reason"] == "time_mismatch" and "carries no dates" in rec["refused"]["why"]
    off = _loop(tmp_path / "b", KUSH, tense=False)                                  # and the rule, with that check off
    rec = off.ask("What was the capital of Kush in 1700?")
    assert not rec["verified"] and rec["reason"] == "time_undated" and rec["refused"]["answer"] == "Kerma"


def test_a_dated_value_that_changes_is_said_as_before(tmp_path):
    loop = _loop(tmp_path)
    rec = loop.ask("Who ruled England?")                                            # a reign: dated by its years
    assert rec["verified"] and rec["answer"] == "Henry II" and rec["tier"] == "attributed"
    assert rec["reply"] == "One account says Henry II."


def test_a_books_person_whose_dates_do_not_fit_the_subjects_life_is_another_person(tmp_path):
    film = ["1997 is the death date of Samuel Fuller"]                               # the world's Samuel Fuller
    book = [("EPUB/Mayflower - Kim Dale/Kim Dale - Mayflower (v1.0).epub.txt", "Samuel Fuller was born in 1580 in Redenhall.\n")]
    loop = _loop(tmp_path, book, film)
    rec = loop.ask("When was Samuel Fuller born?")
    assert not rec["verified"] and rec["answer"] is None
    assert loop.books.unfit.get("samuel fuller", loop.books.unfit.get("Samuel Fuller", 0)) == 1
    other = _loop(tmp_path / "b", book)                                             # a world that knows no dates
    assert other.ask("When was Samuel Fuller born?")["answer"] == "1580"
    assert fits((None, 1997), "", [("date of birth", "1900")]) and not fits((None, 1997), "", [("date of birth", "1580")])
    assert fits((1769, 1821), "AncientClassical", [])                              # an era is a book's subject, not a test
    assert fits((None, None), "Prehistory", [("date of birth", "1580")])            # nothing known: nothing to guard


def test_a_worlds_own_undated_changing_fact_is_said_at_one_point_or_as_of_the_world(tmp_path):
    """Order 0015, Addendum 2: the rule reads every mounted world's facts, not only the books'. The wiki world dates
    nothing: its capital is said "at one point". The hdc world is a 2019 snapshot: its capital carries "As of 2019"."""
    from test_ask import FakeEmitter
    from test_hdc_store import FACTS, LABELS, PIDS
    from hdc_store import HdcWorld, Stores
    from hdc_world import write_world
    plans = {"What is the capital of France?": ("France", ["capital"]), "What is the capital of Ruritania?": ("Ruritania", ["capital"])}
    world = TimedStore(STORE + ["Paris is the capital of France"])
    world.index.declare_relation("capital")
    on = AskLoop(FakeEmitter(plans), world=world, source=DictSource({}), lexicon=False, run_fn=exact_vm(), time_rule=True)
    rec = on.ask("What is the capital of France?")
    assert rec["verified"] and rec["tier"] == "at_one_point" and rec["reply"] == "Paris was a capital of France at one point."
    off = AskLoop(FakeEmitter(plans), world=world, source=DictSource({}), lexicon=False, run_fn=exact_vm(), time_rule=False)
    assert "tier" not in off.ask("What is the capital of France?")                      # off: as before
    db = tmp_path / "hdc.sqlite"
    write_world(db, FACTS, LABELS, {}, PIDS)
    both = Stores(TimedStore(STORE), HdcWorld(db))
    loop = AskLoop(FakeEmitter(plans), world=both, source=DictSource({}), lexicon=False, run_fn=exact_vm(), time_rule=True)
    rec = loop.ask("What is the capital of Ruritania?")
    assert rec["verified"] and rec["tier"] == "as_of" and rec["reply"] == "As of 2019, Strelsau."
