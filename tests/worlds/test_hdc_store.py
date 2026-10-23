"""The hdc world on disk and the store that puts it behind the wiki world (standin/hdc_store.py,
standin/data/hdc_world.py; order 0014) -- on a synthetic world.

Pinned: a subject's facts come back by its label, an alias or its Q-id; a hop is served as `TripleIndex.hop`
serves it; the merged store puts the wiki world's facts first, writes only there, and names a hdc fact's
provenance by its snapshot; the ask loop answers from the hdc world, says a changing value "as of 2019", answers
it flat when 2019 is asked, refuses another year; a contradiction between the worlds is never said.
Run: python -m pytest standin/tests/test_hdc_store.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from cubbyllm.reasoning.planner import QuestionPlan  # noqa: E402
from hdc_store import HdcWorld, Stores  # noqa: E402
from hdc_world import write_world  # noqa: E402

LABELS = {"Q1": ("Ruritania", "Ruritanie"), "Q2": ("Zenda", "Zenda"), "Q3": ("Rudolf Rassendyll", ""),
          "Q4": ("Strelsau", "Strelsau"), "Q5": ("Ruritanian Party", ""), "Q6": ("Old Town", "")}
FACTS = {("Q1", "capital", "Q4"): "P36", ("Q1", "head of state", "Q3"): "P35", ("Q2", "country", "Q1"): "P17",
         ("Q3", "member of political party", "Q5"): "P102", ("Q6", "country", "Q1"): "P17"}
PIDS = {"capital": "P36", "head of state": "P35", "country": "P17", "member of political party": "P102"}


@pytest.fixture
def hdc(tmp_path):
    db = tmp_path / "hdc_world.sqlite"
    write_world(db, FACTS, LABELS, {"rurtania": "Q1", "the kingdom": "Q1"}, PIDS)
    return HdcWorld(db)


def test_a_subjects_facts_come_back_by_its_label_an_alias_or_its_qid(hdc):
    want = {"Strelsau is the capital of Ruritania", "Rudolf Rassendyll is the head of state of Ruritania"}
    for name in ("Ruritania", "ruritania", "rurtania", "Q1"):
        assert {f for f, _t in hdc._by_subj.get(name)} == want, name
    assert hdc.label("rurtania") == "Ruritania" and hdc.label("nowhere") is None
    assert {t.subj for _f, t in hdc._by_obj.get("Ruritania")} == {"Zenda", "Old Town"}
    assert "Strelsau is the capital of Ruritania" in hdc and "Zenda is the capital of Ruritania" not in hdc
    assert hdc.relations()["head of state"] == 1 and len(hdc) == 5
    assert hdc.time_sensitive == {"head of state", "member of political party"} and hdc.snapshot == "2019"
    plan = QuestionPlan(relations=[None], tail="capital of Ruritania", n_hop=1)
    assert [f for f, _t in hdc.hop(plan, 0, None)] == ["Strelsau is the capital of Ruritania"]


def test_the_merged_store_puts_the_wiki_world_first_writes_there_and_names_a_hdc_facts_snapshot(hdc):
    from test_search_learn import ProvStore
    wiki = ProvStore(["Old Strelsau is the capital of Ruritania"])
    s = Stores(wiki, hdc)
    plan = QuestionPlan(relations=[None], tail="capital of Ruritania", n_hop=1)
    assert [f for f, _t in s.lookup(plan, 0, None)] == ["Old Strelsau is the capital of Ruritania",
                                                         "Strelsau is the capital of Ruritania"]
    assert s.add("Zenda is the town of Ruritania") and "Zenda is the town of Ruritania" in wiki
    assert s.held_by("Strelsau is the capital of Ruritania") is hdc and s.held_by("Zenda is the town of Ruritania") is None
    assert s.provenance.get("Strelsau is the capital of Ruritania").startswith("wikidata")
    s.provenance["Zenda is the town of Ruritania"] = "test"
    assert wiki.provenance["Zenda is the town of Ruritania"] == "test"
    assert len(s.index._by_subj.get("ruritania")) == 4 and s.index.relations()["capital"] == 2


def _loop(hdc, wiki_facts=()):
    from ask import AskLoop
    from question_forms import Forms, TemplateEmitter
    from test_books import exact_vm
    from test_search_learn import DictSource, ProvStore
    forms = Forms({"head of state": ("state", {"present": ["Who is the head of state of {S}?"],
                                               "past": ["Who was the head of state of {S}?"]}),
                   "capital": ("state", {"present": ["What is the capital of {S}?"]})})
    w = ProvStore(list(wiki_facts) or ["Pella is the capital of Macedon"])
    w.times = {}
    loop = AskLoop(TemplateEmitter(forms), world=Stores(w, hdc), source=DictSource({}), lexicon=False, run_fn=exact_vm())
    return loop


def test_the_loop_answers_from_the_hdc_world_and_says_a_changing_value_as_of_its_snapshot(hdc):
    loop = _loop(hdc)
    rec = loop.ask("What is the capital of Ruritania?")
    assert rec["verified"] and rec["answer"] == "Strelsau" and "tier" not in rec        # a capital: said flat
    rec = loop.ask("What is the capital of rurtania?")                                 # the set's own alias for it
    assert rec["verified"] and rec["answer"] == "Strelsau"
    rec = loop.ask("Who is the head of state of Ruritania?")
    assert rec["verified"] and rec["tier"] == "as_of" and rec["reply"] == "As of 2019, Rudolf Rassendyll."
    rec = loop.ask("Who was the head of state of Ruritania in 2019?")
    assert rec["verified"] and "tier" not in rec                                         # the snapshot's own year
    rec = loop.ask("Who was the head of state of Ruritania in 2010?")
    assert not rec["verified"] and rec["reason"] == "time_mismatch" and "2019" in rec["refused"]["why"]


def test_a_contradiction_between_the_worlds_is_never_said(hdc):
    loop = _loop(hdc, ["Zenda is the capital of Ruritania"])
    rec = loop.ask("What is the capital of Ruritania?")
    assert not rec["verified"] and rec["answer"] is None                                 # Zenda or Strelsau: refused


def test_a_fact_whose_name_holds_a_no_break_space_is_held_and_its_changing_value_said_as_of(tmp_path):
    """Order 0036, found on the way: "Washington University in St.<no-break space>Louis is the employer of Juan Rosai"
    was answered but not found held (the fact's spaces were normalized on one side only), so it was said flat."""
    labels = dict(LABELS, Q3=("Rudolf Rassendyll", ""))
    db = tmp_path / "hdc_world.sqlite"
    write_world(db, FACTS, labels, {}, PIDS)
    hdc = HdcWorld(db)
    assert "Rudolf Rassendyll is the head of state of Ruritania" in hdc
    rec = _loop(hdc).ask("Who is the head of state of Ruritania?")
    assert rec["verified"] and rec["tier"] == "as_of" and rec["reply"].startswith("As of 2019, ")
