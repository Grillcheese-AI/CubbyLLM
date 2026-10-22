"""The books' bulk pass (validation/books_bulk_read.py, order 0013) on the synthetic library of test_books: every
paragraph read once, 0012's tiers over every read, a world per era; a re-run with no new book changes nothing, a
new book is the only one read; Wikidata's value corroborates, another value of a one-valued relation holds; the
loop loads a world and speaks one author's fact hedged.
Run: python -m pytest standin/tests/test_books_bulk.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import books_bulk_read as bulk  # noqa: E402
from books import BookIndex, Mentions  # noqa: E402
from test_books import NAMES, TimedStore, _book, _library, exact_vm  # noqa: E402
from test_search_learn import STORE, DictSource  # noqa: E402


def _index(tmp_path):
    roots, events = _library(tmp_path)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES))
    idx.db.close()
    return roots, events


def _worlds(out):
    return {p.stem: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] for p in (out / "worlds").glob("*.jsonl")}


def test_every_read_takes_its_tier_and_its_era_and_a_rerun_changes_nothing(tmp_path):
    roots, events = _index(tmp_path)
    wiki = {("napoleon bonaparte", "BORN_IN"): ["Ajaccio"], ("rome", "FOUNDED_BY"): ["Remus"]}
    rep = bulk.run(tmp_path / "books.sqlite", tmp_path / "bulk", workers=0, wiki=wiki, log=lambda m: None)
    w = _worlds(tmp_path / "bulk")
    by = {f["fact"]: f for fs in w.values() for f in fs}
    born = by["Ajaccio is the place of birth of Napoleon Bonaparte"]
    assert born["tier"] == "corroborated" and born["source"].endswith("+wiki") and born["era"] in w and born in w[born["era"]]
    held = {f["fact"] for f in w["_held"]}
    assert {"1769-08-15 is the date of birth of Napoleon Bonaparte", "1768 is the date of birth of Napoleon Bonaparte"} <= held
    assert "Romulus is the founded by of Rome" in held                          # Wikidata says another founder: held
    henry = by["Henry II is the head of state of England"]
    assert henry["tier"] == "attributed" and henry["source"] == "book:alice smith (attributed)" and henry["era"] == "MiddleAges"
    assert rep["books_read_now"] == rep["books_done"] == 5 and rep["tiers"]["held"] >= 3
    again = bulk.run(tmp_path / "books.sqlite", tmp_path / "bulk", workers=0, wiki=wiki, log=lambda m: None)
    assert again["books_read_now"] == 0 and again["files"] == rep["files"] and again["reads"] == rep["reads"]
    _book(roots[1][0], "EPUB/Rome - Dan Grey/Dan Grey - Rome (v1.0).epub.txt", "Rome was founded by Romulus, the son of Mars.\n")
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES))
    idx.db.close()
    third = bulk.run(tmp_path / "books.sqlite", tmp_path / "bulk", workers=0, wiki={}, log=lambda m: None)
    assert third["books_read_now"] == 1                                          # only the new book is read
    rome = {f["fact"]: f for fs in _worlds(tmp_path / "bulk").values() for f in fs}["Romulus is the founded by of Rome"]
    assert rome["tier"] == "corroborated" and rome["witnesses"] == 2              # two authors now, no Wikidata


def test_the_loop_loads_a_world_and_speaks_one_authors_fact_hedged(tmp_path):
    from ask import AskLoop
    from test_ask import FakeEmitter
    _index(tmp_path)
    bulk.run(tmp_path / "books.sqlite", tmp_path / "bulk", workers=0, wiki={}, log=lambda m: None)
    plans = {"Who founded Rome?": ("Rome", ["founded by"])}
    loop = AskLoop(FakeEmitter(plans), world=TimedStore(STORE), source=DictSource({}), lexicon=False, run_fn=exact_vm())
    assert loop.load_learned(tmp_path / "bulk" / "worlds" / "AncientClassical.jsonl") > 0
    rec = loop.ask("Who founded Rome?")
    assert rec["verified"] and rec["answer"] == "Romulus" and rec.get("tier") == "attributed"
    assert rec["reply"] == "One account says Romulus."


def test_a_changing_value_from_the_bulk_store_is_said_at_one_point(tmp_path):
    """The time rule reads a store's book facts too (no BookSource behind them): two books' undated capital."""
    from ask import AskLoop
    from test_ask import FakeEmitter
    roots, events = _library(tmp_path)
    for rel, text in [("EPUB/Nubia - Ann Lee/Ann Lee - Nubia (v1.0).epub.txt", "Kerma, the capital of Kush, stood above the cataract.\n"),
                      ("EPUB/Kush - Tom Hart/Tom Hart - Kush (v1.0).epub.txt", "Kerma, the capital of Kush, traded with Egypt.\n")]:
        _book(roots[1][0], rel, text)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES + ["Kush", "Kerma"]))
    idx.db.close()
    bulk.run(tmp_path / "books.sqlite", tmp_path / "bulk", workers=0, wiki={}, log=lambda m: None)
    kerma = {f["fact"]: f for fs in _worlds(tmp_path / "bulk").values() for f in fs}["Kerma is the capital of Kush"]
    assert kerma["tier"] == "corroborated" and "time" not in kerma
    loop = AskLoop(FakeEmitter({"What is the capital of Kush?": ("Kush", ["capital"])}), world=TimedStore(STORE),
                   source=DictSource({}), lexicon=False, run_fn=exact_vm())
    for p in (tmp_path / "bulk" / "worlds").glob("*.jsonl"):
        if not p.stem.startswith("_"):
            loop.load_learned(p)
    rec = loop.ask("What is the capital of Kush?")
    assert rec["verified"] and rec["tier"] == "at_one_point" and rec["reply"] == "Kerma was a capital of Kush at one point."


def test_a_book_whose_person_does_not_fit_the_subjects_life_is_set_aside(tmp_path):
    _index(tmp_path)
    wiki = {("napoleon bonaparte", "DIED_ON"): ["1450-01-01"]}                      # a world whose Napoleon died in 1450
    bulk.run(tmp_path / "books.sqlite", tmp_path / "bulk", workers=0, wiki=wiki, log=lambda m: None)
    facts = {f["fact"] for fs in _worlds(tmp_path / "bulk").values() for f in fs}
    assert not any(f.endswith("of Napoleon Bonaparte") for f in facts)               # every book's birth is 1768-1769
    assert "Henry II is the head of state of England" in facts                       # the others stand
