"""validation/fact_store/ -- order 0055 step 0's coverage audit, pinned on fixtures (no network).

Pinned: a statement's value states a gold or does not (an item by Q-id; a date at the gold's precision, a less
precise value counted apart; a number exactly, never as a float; words); the classes in order (on the item, the
reverse, elsewhere, not found; yes/no is not a value); every item a name names is asked, and ambiguity is
counted; no candidate list is ordered by fame (the audit keeps the full exact-name set the search returns).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fact_store import audit as A  # noqa: E402
from fact_store.audit_values import dates, match, numbers  # noqa: E402


def _t(time, prec):
    return {"type": "time", "value": {"time": time, "precision": prec}}


def test_values_state_a_gold_or_do_not():
    item = {"type": "wikibase-entityid", "value": {"id": "Q90"}}
    assert match(item, {"qids": ["Q90"], "text": ""}) == "exact" and match(item, {"qids": ["Q1"], "text": "Paris"}) is None
    assert match(_t("+1972-12-14T00:00:00Z", 11), {"qids": [], "text": "14 December 1972"}) == "exact"
    assert match(_t("+1972-12-14T00:00:00Z", 11), {"qids": [], "text": "December 1972"}) == "exact"
    assert match(_t("+1972-00-00T00:00:00Z", 9), {"qids": [], "text": "December 14, 1972"}) == "coarser"
    assert match(_t("+1972-12-15T00:00:00Z", 11), {"qids": [], "text": "1972-12-14"}) is None
    assert match(_t("+2010-03-01T00:00:00Z", 11), {"qids": [], "text": "1 mars 2010"}) == "exact"
    q = {"type": "quantity", "value": {"amount": "+4.10", "unit": "1"}}
    assert match(q, {"qids": [], "text": "4.1"}) == "exact" and match(q, {"qids": [], "text": "4.11"}) is None
    s = {"type": "string", "value": "Pozzaglia Sabina"}
    assert match(s, {"qids": [], "text": "Pozzaglia Sabina."}) == "exact"
    assert match({"type": "string", "value": "Sabina"}, {"qids": [], "text": "Pozzaglia Sabina, Lazio, Italy"}) is None
    assert dates("44 BC") == [(-44, None, None)] and numbers("1,234 people") == [__import__("decimal").Decimal("1234")]


class FakeWd:
    """The audit's Wikidata, from dicts: names -> items, items -> entity json."""

    def __init__(self, names, ents):
        self.names, self.ents, self.searched = names, ents, []

    def search_exact(self, name, lang):
        self.searched.append(name)
        return list(self.names.get(name.lower(), [])), False

    def entities(self, qids, props="claims", langs=""):
        return {q: self.ents[q] for q in qids if q in self.ents}


def _claim(pid, qid=None, time=None):
    dv = {"type": "wikibase-entityid", "value": {"id": qid}} if qid else _t(time, 11)
    return {pid: [{"mainsnak": {"snaktype": "value", "datavalue": dv}, "rank": "normal"}]}


def _row(q, golds, set_="simpleqa", asked=None, complexity=None):
    return {"id": "x", "set": set_, "lang": "en", "question": q, "question_fr": None, "asked": asked,
            "golds": golds, "extra": {"complexity": complexity}}


def test_every_item_a_name_names_is_asked_and_ambiguity_is_counted():
    ents = {"Q1": {"claims": _claim("P569", time="+1950-05-02T00:00:00Z")}, "Q2": {"claims": {}}}
    wd = FakeWd({"john smith": ["Q2", "Q1"]}, ents)
    r = A.classify(wd, _row("When was John Smith born?", [{"qids": [], "text": "2 May 1950"}]))
    assert r["class"] == "on_item" and r["ambiguous"] and r["items_hit"] == 1 and r["asked_n"] == 2


def test_the_classes_in_order():
    rev = {"Q5": {"claims": _claim("P50", qid="Q7")}, "Q7": {"claims": {}}}
    wd = FakeWd({"the book": ["Q7"], "ann": ["Q5"]}, rev)
    assert A.classify(wd, _row("Who wrote The Book?", [{"qids": [], "text": "Ann"}]))["class"] == "on_item_rev"
    lab = {"Q7": {"claims": _claim("P50", qid="Q8")}, "Q8": {"labels": {"en": {"value": "Bob Smith"}}}}
    r = A.classify(FakeWd({"the book": ["Q7"]}, lab), _row("Who wrote The Book?", [{"qids": [], "text": "Bob Smith"}]))
    assert r["class"] == "on_item" and r["by_label"]                       # no item named so: its label states it
    wd = FakeWd({"the book": ["Q7"], "bob": ["Q9"]}, {"Q7": {"claims": {}}, "Q9": {"claims": {}}})
    assert A.classify(wd, _row("Who wrote The Book?", [{"qids": [], "text": "Bob"}]))["class"] == "elsewhere"
    assert A.classify(wd, _row("Who wrote The Book?", [{"qids": [], "text": "1999"}]))["class"] == "not_found"
    m = _row("Is X big?", [{"qids": [], "text": "True"}], set_="mintaka", asked=["Q7"], complexity="yesno")
    assert A.classify(wd, m)["class"] == "not_a_value"


def test_mintaka_uses_its_own_ids_and_reads_the_name_in_both_languages():
    ents = {"Q7": {"labels": {"en": {"value": "The Book"}, "fr": {"value": "Le Livre"}},
                   "claims": _claim("P50", qid="Q5")}}
    wd = FakeWd({}, ents)
    row = {**_row("Who wrote The Book?", [{"qids": ["Q5"], "text": ""}], set_="mintaka", asked=["Q7"]),
           "question_fr": "Qui a écrit Le Livre ?"}
    r = A.classify(wd, row)
    assert r["class"] == "on_item" and r["name_en"] and r["name_fr"] and not wd.searched


def test_names_are_found_longest_first_and_an_instance_of_is_not_the_reverse():
    spans = [t for _, _, t in A.name_spans("Who received the IEEE Frank Rosenblatt Award in 2010?", "en")]
    assert spans[0] == "IEEE Frank Rosenblatt Award" and "IEEE Frank Rosenblatt Award in 2010" not in spans
    assert not any(t.startswith(("received", "the ")) for t in spans)
    assert all(not t.startswith("Who") for _, _, t in A.name_spans("Who wrote The Book?", "en"))
    wd = FakeWd({"frank rosenblatt award": ["Q1"], "frank": ["Q2"]}, {})
    assert list(A.asked_by_name(wd, "Who won the Frank Rosenblatt Award?", "en")["asked"]) == ["Q1"]
    assert "frank" not in [s.lower() for s in wd.searched]                       # inside a name found: not looked up
    assert A.pointing_at({"Q5": {"claims": _claim("P31", qid="Q7")}}, {"Q7": ("x", 1)}) == []


def test_gold_names():
    assert A.gold_names("In Berlin.") == ["In Berlin", "Berlin"]
