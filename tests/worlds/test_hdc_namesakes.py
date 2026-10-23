"""Namesakes never merge (order 0033) -- on a synthetic hdc world: three items labelled "Globe Theatre", two "Bambi",
two "Mercia" (a kingdom and a ship), two "Ann Reed" (both people), two "Wantage" (a township and a civil parish).

Pinned: a name two or more items share gives none of their facts, and each item's facts stay its own; an alias that
names one item gives that item's facts, not its label's namesakes'; a fact is still held (its provenance names the
world) whichever namesake it is about; the ask loop answers "where is X?" by asking which one, with each item named;
the asker's choice, or a qualifier the question carries, decides; a qualifier that is no value of exactly one item
does not. A relation question: the relation decides only for a holder of known kind among items of other kinds, none
of a kind that carries the relation elsewhere (a ship has no capital; a civil parish has a location even when this
one's is not kept; two people are two of a kind); else it is refused with the items named, and the choice walks one.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests" / "worlds", ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from cubbyllm.reasoning.planner import QuestionPlan  # noqa: E402
from cubbyllm.worlds.hdc_store import HdcWorld, Stores  # noqa: E402
from cubbyllm.worlds.hdc_world import write_world  # noqa: E402

LOC, CIT, KIND = "located in the administrative territorial entity", "country of citizenship", "instance of"
LABELS = {"Q10": ("Globe Theatre", ""), "Q11": ("Southwark", ""), "Q12": ("Globe Theatre", ""), "Q13": ("Boston", ""),
          "Q14": ("Globe Theatre", ""), "Q15": ("England", ""), "Q20": ("Bambi", ""), "Q21": ("film", ""),
          "Q22": ("Bambi", ""), "Q23": ("Angola", ""), "Q30": ("Zenda", ""),
          "Q40": ("Mercia", ""), "Q41": ("kingdom", ""), "Q42": ("Tamworth", ""), "Q43": ("Mercia", ""), "Q44": ("ship", ""),
          "Q50": ("Ann Reed", ""), "Q51": ("human", ""), "Q52": ("Ann Reed", ""),
          "Q60": ("Wantage", ""), "Q61": ("township", ""), "Q62": ("Sussex County", ""), "Q63": ("Wantage", ""),
          "Q64": ("civil parish", ""), "Q65": ("Didcot", ""), "Q66": ("Oxfordshire", ""),
          "Q70": ("Springfield", ""), "Q71": ("Springfield", ""), "Q72": ("Moe's", ""), "Q73": ("Shelbyville Mill", "")}
FACTS = {("Q10", LOC, "Q11"): "P131", ("Q12", LOC, "Q13"): "P131", ("Q14", "country", "Q15"): "P17",
         ("Q20", KIND, "Q21"): "P31", ("Q22", "country", "Q23"): "P17", ("Q30", "country", "Q15"): "P17",
         ("Q40", KIND, "Q41"): "P31", ("Q40", "capital", "Q42"): "P36", ("Q43", KIND, "Q44"): "P31",
         ("Q50", KIND, "Q51"): "P31", ("Q50", CIT, "Q15"): "P27", ("Q52", KIND, "Q51"): "P31",
         ("Q60", KIND, "Q61"): "P31", ("Q60", LOC, "Q62"): "P131", ("Q63", KIND, "Q64"): "P31",
         ("Q65", KIND, "Q64"): "P31", ("Q65", LOC, "Q66"): "P131",
         ("Q72", LOC, "Q70"): "P131", ("Q73", LOC, "Q71"): "P131"}             # two Springfields, read as objects
PIDS = {LOC: "P131", "country": "P17", KIND: "P31", "capital": "P36", CIT: "P27"}
ALIAS = {"the wooden o": "Q10", "twin globe": "Q12"}
SOUTHWARK, BOSTON, ENGLAND = (f"Southwark is the {LOC} of Globe Theatre", f"Boston is the {LOC} of Globe Theatre",
                              "England is the country of Globe Theatre")


@pytest.fixture
def hdc(tmp_path):
    db = tmp_path / "hdc_world.sqlite"
    write_world(db, FACTS, LABELS, ALIAS, PIDS)
    import sqlite3
    with sqlite3.connect(db) as c:                       # an alias two items carry
        c.execute("INSERT INTO names VALUES ('twin globe', 'Q14')")
    return HdcWorld(db)


def test_a_name_several_items_share_gives_none_of_them_and_keeps_each_items_facts_its_own(hdc):
    assert hdc._by_subj.get("Globe Theatre") == []
    by = hdc.namesakes("globe theatre")
    assert {q: [f for f, _t in rows] for q, rows in by.items()} == {"Q10": [SOUTHWARK], "Q12": [BOSTON], "Q14": [ENGLAND]}
    assert [f for f, _t in hdc.item_facts("Globe Theatre", "Q12")] == [BOSTON]
    assert [f for f, _t in hdc._by_subj.get("the wooden o")] == [SOUTHWARK]          # an alias of one item: its facts only
    assert [f for f, _t in hdc._by_subj.get("Q12")] == [BOSTON]                      # a Q-id: that item
    assert hdc._by_subj.get("twin globe") == [] and set(hdc.namesakes("twin globe")) == {"Q12", "Q14"}
    assert [f for f, _t in hdc._by_subj.get("Zenda")] == ["England is the country of Zenda"] and hdc.namesakes("Zenda") == {}
    assert BOSTON in hdc and SOUTHWARK in hdc and "Boston is the country of Globe Theatre" not in hdc
    plan = QuestionPlan(relations=[None], tail=f"{LOC} of Globe Theatre", n_hop=1)
    assert hdc.hop(plan, 0, None) == [] and set(hdc.withheld("globe theatre")) == {"Q10", "Q12"}


def test_a_qualifier_decides_only_when_it_is_a_value_of_exactly_one_item(hdc):
    assert hdc.pick_by("Globe Theatre", "Boston", place=True) == "Q12"
    assert hdc.pick_by("Globe Theatre", "London", place=True) is None                 # no item's value
    assert hdc.pick_by("Bambi", "film") == "Q20"
    assert hdc.pick_by("Bambi", "film", place=True) is None                          # a kind says nothing of where
    assert hdc.pick_by("Zenda", "England", place=True) is None                       # one item: nothing to decide


def test_the_hops_relation_decides_only_where_the_others_kinds_cannot_carry_it(hdc):
    def hop(rel, name):
        return [t.obj for _f, t in hdc.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)]
    assert hop("capital", "Mercia") == ["Tamworth"]                                  # a kingdom and a ship: a ship has no capital
    assert hdc.applies("civil parish", LOC) and not hdc.applies("ship", "capital")
    assert hop(LOC, "Wantage") == [] and hdc.withheld("wantage") == ["Q60"]          # a parish has a place; this one's is not kept
    assert hop(CIT, "Ann Reed") == [] and hdc.withheld("ann reed") == ["Q50"]        # two people: two of a kind
    assert hop("country", "Globe Theatre") == []                                     # the one holder is of no known kind


def _loop(hdc):
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.question_forms import Forms, TemplateEmitter
    from test_books import exact_vm
    from test_search_learn import DictSource, ProvStore
    forms = Forms({r: ("state", {"present": [f"What is the {r} of {{S}}?"]}) for r in ("country", "capital", CIT)})
    w = ProvStore(["Pella is the capital of Macedon"])
    w.times = {}
    return AskLoop(TemplateEmitter(forms), world=Stores(w, hdc), source=DictSource({}), lexicon=False, run_fn=exact_vm())


def test_the_loop_asks_which_one_and_the_choice_or_the_questions_own_context_decides(hdc):
    loop = _loop(hdc)
    rec = loop.ask("Where is the Globe Theatre?")
    assert rec["reason"] == "ambiguous_entity" and rec["answer"] is None and not rec["profile"]
    assert {c["item"]: c["detail"] for c in rec["clarify"]["choices"]} == {
        "Q10": "located in: Southwark", "Q12": "located in: Boston", "Q14": "country: England"}
    rec = loop.ask("Where is the Globe Theatre?", item="Q12")
    assert [l["value"] for l in rec["profile"]] == ["Boston"] and rec["namesake"]["how"] == "chosen by the asker"
    rec = loop.ask("Where is the Globe Theatre in Boston?")
    assert [l["value"] for l in rec["profile"]] == ["Boston"] and "Boston" in rec["namesake"]["how"]
    assert loop.ask("Where is the Globe Theatre in London?")["reason"] == "ambiguous_entity"
    rec = loop.ask("Where is Zenda?")                                                # one item: answered as before
    assert [l["value"] for l in rec["profile"]] == ["England"]


def test_a_relation_question_on_a_shared_name_is_decided_by_its_relation_or_refused_with_the_items(hdc):
    loop = _loop(hdc)
    rec = loop.ask("What is the capital of Mercia?")
    assert rec["verified"] and rec["answer"] == "Tamworth"
    rec = loop.ask("What is the country of Globe Theatre?")
    assert not rec["verified"] and rec["reason"] == "ambiguous_entity"
    assert [c["item"] for c in rec["clarify"]["choices"]][0] == "Q14"                # the item that holds it, first
    assert {c["item"] for c in rec["clarify"]["choices"]} == {"Q10", "Q12", "Q14"}
    rec = loop.ask("What is the country of Globe Theatre?", item="Q14")
    assert rec["verified"] and rec["answer"] == "England"
    rec = loop.ask(f"What is the {CIT} of Ann Reed?")
    assert not rec["verified"] and rec["reason"] == "ambiguous_entity"
    rec = loop.ask("What is the capital of Bambi?")                                  # no Bambi holds one: not a question of which
    assert not rec["verified"] and rec["reason"] != "ambiguous_entity"
    rec = loop.ask("What is the country of Zenda?")
    assert rec["verified"] and rec["answer"] == "England" and rec["reason"] is None


def test_an_object_label_several_items_share_reads_as_none_of_them(hdc):
    """Order 0036, item 1: the object side under 0033's rule (an inverse read never mixes two items)."""
    assert hdc._by_obj.get("Springfield") == []                                      # Moe's and the mill point at two
    assert [t.subj for _f, t in hdc._by_obj.get("Oxfordshire")] == ["Didcot"]          # one item: its rows, as before
    assert [t.subj for _f, t in hdc._by_obj.get("Q71")] == ["Shelbyville Mill"]        # a Q-id: that item


def test_the_kinds_table_answers_as_the_scan_and_the_world_reads_it(hdc, tmp_path):
    """Order 0036, item 3: which relations each kind carries, built once; the same decisions as the scan."""
    from cubbyllm.worlds.hdc_kinds import Table, build, path_for
    built = build(hdc.path)
    assert built["out"] == path_for(hdc.path) and built["pairs"] > 0
    table = Table(built["out"])
    assert table.applies("civil parish", LOC) and not table.applies("ship", "capital") and table.applies("kingdom", "capital")
    scan = HdcWorld(hdc.path)
    scan._kinds = None
    with_table = HdcWorld(hdc.path)
    assert with_table._kinds is not None
    for rel, name in (("capital", "Mercia"), (LOC, "Wantage"), (CIT, "Ann Reed"), ("country", "Globe Theatre")):
        plan = QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1)
        assert with_table.hop(plan, 0, None) == scan.hop(plan, 0, None), (rel, name)
    import sqlite3
    with sqlite3.connect(built["out"]) as c:                 # a table built from another state of the world: not read
        c.execute("UPDATE meta SET v = '1' WHERE k = 'world_bytes'")
    assert HdcWorld(hdc.path)._kinds is None
