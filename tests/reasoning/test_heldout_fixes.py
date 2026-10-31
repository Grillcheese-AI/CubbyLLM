"""Order 0047 -- the six held-out wrong answers of 0046's R0, on canned claims (no network), ``CB_HELDOUT_FIXES`` on (the default):

- the four titles, asked with their article in English and French, resolve to the item asked or refuse -- never to
  the item labelled with the shorter name; off, the shorter one wins (the defect, pinned so the flag is what fixes it);
- "In which city did John F. Kennedy die?" refuses with ``class_mismatch`` (the walk off) and answers Dallas through
  ``located in`` (``CB_CLASS_WALK=1``);
- Big Ben's height carries its unit; a height with no unit is not served; a count is said bare.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation", ROOT / "tests" / "reasoning", ROOT / "tests" / "sources", ROOT / "tests" / "worlds"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from test_ask import STORE, FakeEmitter  # noqa: E402
from test_books import exact_vm  # noqa: E402
from test_search_learn import LookupStore  # noqa: E402
from test_wikidata_namesakes import LABELS, Fake, hit, item, lab  # noqa: E402

def loop_with(src, plans):
    """The namesakes test's loop, its first world also holding an "author" and a "genre" fact: relations the walk knows,
    so a title's own "of" ("The Ides of March") is no chain boundary -- as on the serving path, where the tables know them."""
    from cubbyllm.reasoning.ask import AskLoop
    store = STORE + ["someone is the author of some book", "some genre is the genre of some film"]
    return AskLoop(FakeEmitter(plans), world=LookupStore(store), source=src, lexicon=False, run_fn=exact_vm(), time_rule=False)


TITLES = [("The Last Dragon", "Q101", "Q102", "Qdemari", "Qmcd"), ("The Ides of March", "Q103", "Q104", "Qwilder", "Qmanf"),
          ("The Interpretation of Dreams", "Q105", "Q106", "Qfreud", "Qanti")]


def titles_source():
    search, ents = {}, {**LABELS, "P50": lab("author"), "P136": lab("genre"), "Qnovel": lab("novel")}
    for full, asked_q, short_q, a1, a2 in TITLES:
        short = full[4:]
        both = [hit(asked_q, full, "the work asked"), hit(short_q, short, "another work")]
        search[full.lower()] = both
        search[short.lower()] = [both[1]]                              # the shorter name finds its own item first
        ents[asked_q] = {"claims": {"P31": [item("Qnovel")], "P50": [item(a1)]}}
        ents[short_q] = {"claims": {"P31": [item("Qnovel")], "P50": [item(a2)]}}
        ents[a1], ents[a2] = lab(f"author of {full}"), lab(f"author of {short}")
    search["l'interprétation des rêves"] = [hit("Q105", "The Interpretation of Dreams", "the work asked", "L'Interprétation des rêves")]
    search["interprétation des rêves"] = [hit("Q106", "interprétation des rêves", "another work")]
    search["the skin"] = [hit("Q107", "The Skin", "film"), hit("Q108", "Skin", "singer")]
    search["skin"] = [hit("Q108", "Skin", "singer")]
    ents["Q107"] = {"claims": {"P31": [item("Qnovel")], "P136": [item("Qdrama")]}}
    ents["Q108"] = {"claims": {"P31": [item("Qhuman")], "P136": [item("Qrock")]}}
    ents["Qdrama"], ents["Qrock"] = lab("drama"), lab("rock music")
    return Fake(search, ents)


QS = [("Who wrote The Last Dragon?", "author of The Last Dragon"), ("Who authored The Ides of March?", "author of The Ides of March"),
      ("Who is the author of the interpretation of dreams?", "author of The Interpretation of Dreams"),
      ("Qui est l'auteur de l'interprétation des rêves ?", "author of The Interpretation of Dreams"),
      ("what genre is the skin", "drama")]


# the reader's seeds, the article dropped as the everyday table drops it (a title's own "of" is no chain)
SEEDS = {"Who wrote The Last Dragon?": "Last Dragon", "Who authored The Ides of March?": "Ides of March",
         "Who is the author of the interpretation of dreams?": "interpretation of dreams",
         "Qui est l'auteur de l'interprétation des rêves ?": "interprétation des rêves", "what genre is the skin": "skin"}


@pytest.mark.parametrize("q,right", QS)
def test_the_name_as_asked_resolves_to_the_item_asked_never_the_shorter_one(q, right, monkeypatch):
    monkeypatch.setenv("CB_HELDOUT_FIXES", "1")
    rec = loop_with(titles_source(), {q: (SEEDS[q], ["author" if "genre" not in q else "genre"])}).ask(q)
    assert rec["verified"] and rec["answer"] == right, rec            # the item asked, never the shorter-named one


def test_off_the_shorter_name_still_wins_the_defect_the_flag_fixes(monkeypatch):
    monkeypatch.setenv("CB_HELDOUT_FIXES", "0")
    rec = loop_with(titles_source(), {}).ask("Who wrote The Last Dragon?")
    assert rec["answer"] == "author of Last Dragon"


def test_items_named_as_asked_that_lack_the_relation_leave_the_shorter_name_unable_to_win_alone(monkeypatch):
    monkeypatch.setenv("CB_HELDOUT_FIXES", "1")
    src = titles_source()
    src._entities["Q101"] = {"claims": {"P31": [item("Qnovel")]}}      # the work asked, its author not recorded
    rec = loop_with(src, {}).ask("Who wrote The Last Dragon?")
    assert not rec["verified"] and rec["answer"] is None


def jfk_source():
    search = {"john f. kennedy": [hit("Q9696", "John F. Kennedy", "president")],
              "city": [hit("Q515", "city", "large human settlement"), hit("Q999", "City", "a film")]}
    ents = {"Q9696": {"claims": {"P31": [item("Qhuman")], "P20": [item("Q1")]}},
            "Q1": {**lab("Parkland Memorial Hospital"), "claims": {"P31": [item("Q16917")], "P131": [item("Q16557")]}},
            "Q16557": {**lab("Dallas"), "claims": {"P31": [item("Q1093829")]}},
            "Q1093829": {**lab("city of the United States"), "claims": {"P279": [item("Q515")]}},
            "Q16917": {**lab("hospital"), "claims": {"P279": [item("Q4287745")]}},
            "Q515": {**lab("city"), "claims": {"P279": [item("Q486972")]}}, "Q999": lab("City"),
            "P20": lab("place of death"), "P131": lab("located in the administrative territorial entity"), **LABELS}
    return Fake(search, ents)


def test_an_answer_of_another_class_refuses_or_walks_up_to_the_class_asked(monkeypatch):
    monkeypatch.setenv("CB_HELDOUT_FIXES", "1")
    q = "In which city did John F. Kennedy die?"
    rec = loop_with(jfk_source(), {}).ask(q)
    assert not rec["verified"] and rec["reason"] == "class_mismatch" and "a hospital" in rec["refused"]["why"]
    monkeypatch.setenv("CB_CLASS_WALK", "1")
    rec = loop_with(jfk_source(), {}).ask(q)
    assert rec["verified"] and rec["answer"] == "Dallas" and rec["reply"] == "Dallas, where Parkland Memorial Hospital is."
    monkeypatch.setenv("CB_HELDOUT_FIXES", "0")
    monkeypatch.delenv("CB_CLASS_WALK")
    rec = loop_with(jfk_source(), {}).ask(q)
    assert rec["verified"] and rec["answer"] == "Parkland Memorial Hospital"           # (off: as before)


def quantities(unit):
    snak = {"mainsnak": {"datatype": "quantity", "datavalue": {"type": "quantity", "value": {"amount": "+96", "unit": unit}}}}
    pop = {"mainsnak": {"datatype": "quantity", "datavalue": {"type": "quantity", "value": {"amount": "+8900000", "unit": "1"}}}}
    allowed = {"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q21514353"}}}}
    ents = {"Q41225": {"claims": {"P2048": [snak], "P1082": [pop]}},
            "P2048": {**lab("height"), "claims": {"P2302": [allowed]}}, "P1082": lab("population"),
            "Q11573": {**lab("metre"), "claims": {"P5061": [{"mainsnak": {"datavalue": {"value": {"text": "m", "language": "en"}}}}]}}}
    return Fake({"big ben": [hit("Q41225", "Big Ben", "tower")]}, ents)


def test_a_quantity_carries_its_unit_a_missing_one_is_not_served_a_count_is_bare(monkeypatch):
    monkeypatch.setenv("CB_HELDOUT_FIXES", "1")
    got = {t.rel: t.obj for t in quantities("http://www.wikidata.org/entity/Q11573").facts("Big Ben")}
    assert got == {"height": "96 m", "population": "8900000"}
    got = {t.rel: t.obj for t in quantities("1").facts("Big Ben")}
    assert got == {"population": "8900000"}                                            # a height without its unit: not served
    monkeypatch.setenv("CB_HELDOUT_FIXES", "0")
    got = {t.rel: t.obj for t in quantities("http://www.wikidata.org/entity/Q11573").facts("Big Ben")}
    assert got["height"] == "96"                                                        # (off: as before)


@pytest.mark.parametrize("q,want", [("In which city did John F. Kennedy die?", ("city", "en")),
                                    ("What country is Lyon in?", ("country", "en")),
                                    ("Dans quelle ville est né Victor Hugo ?", ("ville", "fr")),
                                    ("what kind of government does spain have now?", None),
                                    ("what type of music did mozart write?", None), ("What is the capital of France?", None)])
def test_the_class_word_is_the_kind_of_the_answer_never_a_word_about_kinds(q, want):
    from cubbyllm.reasoning.ask_class import class_word
    assert class_word(q) == want


def test_a_class_word_the_table_reads_as_the_relation_asked_is_not_checked(monkeypatch):
    """"what nationality is X?" plans country of citizenship: "nationality" names the relation, and a country is no
    instance of a class called nationality -- no check, the answer stands (the scoreboard, 2026-10-31)."""
    from cubbyllm.reasoning import ask_class
    monkeypatch.setenv("CB_HELDOUT_FIXES", "1")

    class Table:
        def relations(self, text):
            return {"nationality": ["country of citizenship"], "city": []}.get(text, [])

    class Loop:
        source, _everyday = Fake({}, {}), Table()

    class Hop:
        triple, fact = ("x", "country of citizenship", "United States"), "United States is the country of citizenship of x"

    class LR:
        class result:
            trace = [Hop()]

    rec = {"answer": "United States", "plan": ["country of citizenship"], "verified": True}
    assert ask_class.check(Loop(), "what nationality is jermaine paul?", LR(), rec) and rec["verified"]


def test_the_fixes_are_on_by_default_and_zero_turns_them_off(monkeypatch):
    from cubbyllm.reasoning import heldout
    monkeypatch.delenv("CB_HELDOUT_FIXES", raising=False)
    monkeypatch.delenv("CB_CLASS_WALK", raising=False)
    assert heldout.on() and not heldout.walk_on()
    monkeypatch.setenv("CB_HELDOUT_FIXES", "0")
    assert not heldout.on()
