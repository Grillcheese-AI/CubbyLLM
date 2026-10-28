"""Order 0043 -- a shared name at the source decided by the question's relation (``sources.wikidata_namesakes``), on
canned search hits and claims (no network), behind ``CB_SOURCE_NAMESAKES=1``.

Pinned:
- one fits (a person and a city named alike, asked a place of birth): the source resolves to it, the other is named
  by its kind;
- two kinds both carry the relation (Georgia: a state and a country, asked a capital): no pick, both to be answered;
- two of a kind (two people named Ben Franklin, one described 30 times as fully): never one picked;
- the kind carries the relation through the property's subject-type constraint, read from the property's own cached
  claims, one `subclass of` step up ("may");
- a candidate of no kind whose claims place it competes for a place relation;
- a cue word that is a word of a kind's name rules the other kinds out; the asker's pin resolves without the ledger;
- off, the strict rule as before (ambiguous, no decision).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from cubbyllm.sources.wikidata import WikidataSource  # noqa: E402


class Fake(WikidataSource):
    def __init__(self, search, entities):
        super().__init__(cache_dir=None, sleep_s=0.0, offline=False)
        self._search, self._entities = search, entities

    def _get(self, params):
        self.calls += 1
        if params.get("action") == "wbsearchentities":
            return {"search": list(self._search.get(params["search"].lower(), []))}
        if params.get("action") == "wbgetentities":
            ids = params["ids"].split("|")
            return {"entities": {i: self._entities[i] for i in ids if i in self._entities}}
        return None


def hit(qid, label, desc, match=None):
    return {"id": qid, "label": label, "description": desc, "match": {"type": "label", "text": match or label}}


def item(qid):
    return {"mainsnak": {"datatype": "wikibase-item", "datavalue": {"type": "wikibase-entityid", "value": {"id": qid}}}}


def lab(text):
    return {"labels": {"en": {"value": text}}}


def constraint(*classes):
    return {"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q21503250"}}},
            "qualifiers": {"P2308": [{"datavalue": {"value": {"id": c}}} for c in classes]}}


LABELS = {"P31": lab("instance of"), "P36": lab("capital"), "P19": lab("place of birth"), "P17": lab("country"),
          "P421": lab("located in time zone"), "P131": lab("located in the administrative territorial entity"), "P279": lab("subclass of"), "P106": lab("occupation"),
          "Qstate": lab("U.S. state"), "Qcountry": lab("country"), "Qname": lab("female given name"), "Qcity": lab("city"),
          "Qbig": lab("big city"), "Qband": lab("musical group"), "Qhuman": lab("human"), "Qatl": lab("Atlanta"),
          "Qtbi": lab("Tbilisi"), "Qusa": lab("United States of America"), "Qsmall": lab("Smallville"),
          "Qde": lab("Germany"), "Qmun": lab("municipality"), "Qquarter": lab("quarter"), "Qbos": lab("Boston"),
          "Qcar": lab("Carthage"), "Qpol": lab("politician"), "Qtz": lab("Central Time Zone")}


def georgia():
    search = {"georgia": [hit("Q1", "Georgia", "state"), hit("Q2", "Georgia", "given name"), hit("Q3", "Georgia", "country")]}
    ents = {"Q1": {"claims": {"P31": [item("Qstate")], "P36": [item("Qatl")], "P17": [item("Qusa")]}},
            "Q2": {"claims": {"P31": [item("Qname")]}},
            "Q3": {"claims": {"P31": [item("Qcountry")], "P36": [item("Qtbi")]}}, **LABELS}
    return search, ents


def test_two_kinds_that_both_carry_the_relation_are_both_to_be_answered_and_off_it_is_ambiguous(monkeypatch):
    src = Fake(*georgia())
    assert src.facts("georgia", relations=["capital"]) == [] and len(src.last["ambiguous"]) == 2 and src.namesake_decision("georgia") is None
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    src = Fake(*georgia())
    assert src.facts("georgia", relations=["capital"]) == [] and src.last["each"] == ["Q1", "Q3"]
    d = src.namesake_decision("georgia")
    assert d["fit"] == ["Q1", "Q3"] and d["how"] == {"Q1": "holds", "Q3": "holds"} and d["kinds"]["Q2"] == ["female given name"]
    assert d["others"] == [("Q2", "Georgia", "female given name")]


def test_a_cue_word_that_names_a_kind_rules_the_other_kinds_out(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    src = Fake(*georgia())
    with src.cued(["state"]):
        out = src.facts("georgia", relations=["capital"])
    assert ("Atlanta", "capital") in [(t.obj, t.rel) for t in out] and src.last["how"] == "namesake rule (holds)"
    assert src.namesake_decision("georgia")["ruled_out"] == {"Q3": "state"}


def test_one_fits_and_the_other_is_named_by_its_kind(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"olmo": [hit("Q5", "Olmo", "city in Italy"), hit("Q6", "Olmo", "singer")]}
    ents = {"Q5": {"claims": {"P31": [item("Qcity")], "P17": [item("Qusa")]}},
            "Q6": {"claims": {"P31": [item("Qhuman")], "P19": [item("Qsmall")]}}, **LABELS}
    src = Fake(search, ents)
    out = src.facts("olmo", relations=["place of birth"])
    assert [(t.obj, t.rel, t.subj) for t in out] == [("Smallville", "place of birth", "Olmo")]
    d = src.namesake_decision("olmo")
    assert d["fit"] == ["Q6"] and d["others"] == [("Q5", "Olmo", "city")]


def test_two_of_a_kind_are_never_separated_by_how_fully_one_is_described(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"ben franklin": [hit("Q7", "Benjamin Franklin", "polymath", match="Ben Franklin"), hit("Q8", "Ben Franklin", "politician")]}
    ents = {"Q7": {"claims": {"P31": [item("Qhuman")], "P19": [item("Qbos")], **{f"P{900 + i}": [item("Qusa")] for i in range(30)}}},
            "Q8": {"claims": {"P31": [item("Qhuman")], "P19": [item("Qcar")]}}, **LABELS}
    src = Fake(search, ents)
    assert src.facts("ben franklin", relations=["place of birth"]) == [] and src.last["each"] == ["Q7", "Q8"]
    with src.pinned("ben franklin", "Q8", "Ben Franklin"):     # the loop answering each: no search, no ledger
        out = src.facts("ben franklin", relations=["place of birth"])
    assert [(t.obj, t.subj) for t in out] == [("Carthage", "Ben Franklin")] and src.last["how"] == "pinned (answered each)"
    assert src.choices == {}


def test_the_kind_carries_the_relation_through_the_propertys_constraint_one_subclass_step_up(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"rivertown": [hit("Q9", "Rivertown", "city"), hit("Q10", "Rivertown", "band")]}
    ents = {"Q9": {"claims": {"P31": [item("Qbig")]}}, "Q10": {"claims": {"P31": [item("Qband")]}},
            "Qbig": {**LABELS["Qbig"], "claims": {"P279": [item("Qcity")]}},
            "P421": {**LABELS["P421"], "claims": {"P2302": [constraint("Qcity")]}}, **{k: v for k, v in LABELS.items() if k not in ("Qbig", "P421")}}
    src = Fake(search, ents)
    assert src.constraint_classes("P421") == {"Qcity"}
    assert src.carries(ents["Q9"]["claims"], ["Qbig"], ["P421"]) == "may" and src.carries(ents["Q10"]["claims"], ["Qband"], ["P421"]) is None
    src.facts("rivertown", relations=["located in time zone"])         # the city fits (may) and holds no such fact
    d = src.namesake_decision("rivertown")
    assert d["fit"] == ["Q9"] and d["how"] == {"Q9": "may"} and src.last["qid"] == "Q9"


def test_a_candidate_of_no_kind_that_is_placed_competes_for_a_place_relation(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"kata": [hit("Q11", "Kata", "village"), hit("Q12", "Kata", None)]}
    ents = {"Q11": {"claims": {"P31": [item("Qcity")], "P131": [item("Qusa")]}}, "Q12": {"claims": {"P17": [item("Qde")]}}, **LABELS}
    src = Fake(search, ents)
    assert src.facts("kata", relations=["located in the administrative territorial entity"]) == [] and src.last["each"] == ["Q11", "Q12"]
    assert src.namesake_decision("kata")["how"] == {"Q11": "holds", "Q12": "placed"}
    src = Fake(search, ents)
    src.facts("kata", relations=["capital"])                            # asked a capital: the bare one does not
    assert src.namesake_decision("kata") is None or "Q12" not in src.namesake_decision("kata")["fit"]


def test_the_items_own_rows_tell_namesakes_apart(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    src = Fake(*georgia())
    rows = src.item_rows("Q1", "Georgia")
    assert [(t.rel, t.obj) for _f, t in rows] == [("instance of", "U.S. state"), ("country", "United States of America")]


def test_a_kind_that_may_carry_the_relation_yields_to_another_kind_that_holds_it_but_not_within_a_kind(monkeypatch):
    """Live, 2026-10-27: the novel Pride and Prejudice holds an author; three editions (another kind) only may, by the
    property's constraint, and the question was refused as four that fit. Two people named Harry Truman, one holding a
    place of death and one that only may: each answered (the James Young rule)."""
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"pride": [hit("Q31", "Pride", "novel"), hit("Q32", "Pride", "edition"), hit("Q33", "Pride", "edition")],
              "harry truman": [hit("Q34", "Harry S. Truman", "president", match="Harry Truman"), hit("Q35", "Harry Truman", "rugby player")]}
    ents = {"Q31": {"claims": {"P31": [item("Qnovel")], "P50": [item("Qjane")]}},
            "Q32": {"claims": {"P31": [item("Qedition")]}}, "Q33": {"claims": {"P31": [item("Qedition")]}},
            "Q34": {"claims": {"P31": [item("Qhuman")], "P20": [item("Qkc")]}}, "Q35": {"claims": {"P31": [item("Qhuman")]}},
            "P50": {**lab("author"), "claims": {"P2302": [constraint("Qnovel", "Qedition")]}},
            "P20": {**lab("place of death"), "claims": {"P2302": [constraint("Qhuman")]}},
            "Qnovel": lab("literary work"), "Qedition": lab("version, edition or translation"), "Qjane": lab("Jane Austen"),
            "Qkc": lab("Kansas City"), **LABELS}
    src = Fake(search, ents)
    out = src.facts("pride", relations=["author"])
    assert [(t.obj, t.rel) for t in out] == [("Jane Austen", "author")]
    d = src.namesake_decision("pride")
    assert d["fit"] == ["Q31"] and d["yields"] == ["Q32", "Q33"] and d["how"] == {"Q31": "holds"}
    assert src.facts("harry truman", relations=["place of death"]) == [] and src.last["each"] == ["Q34", "Q35"]
    assert src.namesake_decision("harry truman")["how"] == {"Q34": "holds", "Q35": "may"}


def test_the_label_decides_among_different_kinds_that_fit_never_among_two_of_a_kind(monkeypatch):
    """Live, 2026-10-27: "Spain", "Second Spanish Republic" and "New Spain" all carry a currency; the item LABELLED
    Spain is the one meant (the source's own tier since 2026-09-14). Two people named alike, one by label and one by
    alias, stay two (the James Young rule, pinned above); two labelled alike (Georgia) stay two."""
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"spain": [hit("Q41", "Spain", "country"), hit("Q42", "Second Spanish Republic", "republic", match="Spain"),
                        hit("Q43", "Spain", "family name")]}
    ents = {"Q41": {"claims": {"P31": [item("Qcountry")], "P38": [item("Qeuro")]}},
            "Q42": {"claims": {"P31": [item("Qrepublic")], "P38": [item("Qpeseta")]}},
            "Q43": {"claims": {"P31": [item("Qname")]}}, "P38": lab("currency"), "Qeuro": lab("euro"), "Qpeseta": lab("peseta"),
            "Qrepublic": lab("historical country"), **LABELS}
    src = Fake(search, ents)
    out = src.facts("spain", relations=["currency"])
    assert [(t.obj, t.rel) for t in out] == [("euro", "currency")] and src.last["how"] == "namesake rule (holds + label over alias)"
    d = src.namesake_decision("spain")
    assert d["fit"] == ["Q41"] and d["label"] == "Q41" and d["each"] == ["Q41", "Q42"] and "each" not in src.last
    src = Fake(*georgia())
    assert src.facts("georgia", relations=["capital"]) == [] and src.last["each"] == ["Q1", "Q3"]
