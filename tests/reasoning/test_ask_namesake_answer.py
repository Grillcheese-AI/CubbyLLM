"""Order 0037 -- how the answer reads when the question's relation decided a shared name, through the ask loop, on
test_hdc_relation's synthetic world (with ``CB_NAMESAKE_RELATION=1``).

Pinned:
- one item fits: its value is said, then the other namesakes are named by their kind (at most two; never one of no
  known kind), with an offer;
- two that fit and a fact tells apart (a person and a film, both awarded) are each answered; the question's own words
  narrow it ("Which Nobel Prize is the award received of Mara Vale?" reads the person, not
  the film; a phrasing the plan covers, as the plan check asks of every question), and a
  value the question's description of its answer rules out is not said (her one award is another: not what was asked);
- two of a kind a fact tells apart: each answered from its own walk, labelled by that fact, in their ids' order, and
  each walk passes the loop's checks (the tense check, "as of") as any walk does;
- two of a kind nothing tells apart, and every question with the flag off: asked which, as since order 0036.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests" / "worlds", ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from cubbyllm.worlds import hdc_cues, hdc_kinds  # noqa: E402
from cubbyllm.worlds.hdc_store import HdcWorld, Stores  # noqa: E402
from cubbyllm.worlds.hdc_world import write_world  # noqa: E402
from test_hdc_relation import AWARD, CIT, PIDS, _world  # noqa: E402


@pytest.fixture
def made(tmp_path, monkeypatch):
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.question_forms import Forms, TemplateEmitter
    from test_books import exact_vm
    from test_search_learn import DictSource, ProvStore
    monkeypatch.setattr(hdc_kinds, "SPREAD", 2)
    facts, labels, ids = _world()
    db = tmp_path / "hdc_world.sqlite"
    write_world(db, facts, labels, {}, PIDS)
    hdc_kinds.build(str(db))
    hdc_cues.build(str(db), hdc_kinds.for_world(str(db)))
    forms = Forms({"capital": ("state", {"present": ["What is the capital of {S}?"]}),
                   CIT: ("state", {"present": [f"What is the {CIT} of {{S}}?"]}),
                   AWARD: ("state", {"present": [f"What is the {AWARD} of {{S}}?", f"Which Nobel Prize is the {AWARD} of {{S}}?"]})})
    w = ProvStore(["Pella is the capital of Macedon"])
    w.times = {}
    loop = AskLoop(TemplateEmitter(forms), world=Stores(w, HdcWorld(db)), source=DictSource({}), lexicon=False, run_fn=exact_vm())
    return loop, ids


def test_one_fits_its_value_is_said_and_the_other_namesakes_are_offered_by_their_kind(made, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    loop, ids = made
    rec = loop.ask("What is the capital of Fredonia?")
    assert rec["verified"] and rec["answer"] == "Freedonia City"
    assert rec["reply"] == ("As of 2019, Freedonia City. If you meant something else, there is also a constituency named "
                            "Fredonia. Want to know about it?")        # the snapshot's "as of" first, then the others
    assert rec["namesakes"]["item"] == ids["fr"] and [o["item"] for o in rec["namesakes"]["others"]] == [ids["const"]]


def test_the_questions_own_words_pick_the_item_and_a_value_it_rules_out_is_not_said(made, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    loop, ids = made
    rec = loop.ask(f"What is the {AWARD} of Mara Vale?")              # a person and a film, both awarded: each, by kind
    assert rec["namesakes"]["rule"] == "several fit"
    assert [(a["item"], a["answer"]) for a in rec["answers"]] == [(ids["mara_p"], "Elliott Medal"), (ids["mara_f"], "Golden Bear")]
    rec = loop.ask(f"Which Nobel Prize is the {AWARD} of Mara Vale?") # "Nobel" goes with people, never films
    assert rec["namesakes"]["item"] == ids["mara_p"] and "nobel" in rec["namesakes"]["cues"]
    assert not rec["verified"] and rec["answer"] is None and rec["reason"] == "not_what_was_asked"
    assert rec["refused"]["answer"] == "Elliott Medal" and not rec.get("reply")   # not a Nobel Prize: nothing said


def test_two_of_a_kind_a_fact_tells_apart_are_each_answered_from_their_own_walk(made, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    loop, ids = made
    rec = loop.ask(f"What is the {CIT} of Max Roe?")
    assert rec["verified"] and rec["namesakes"]["rule"] == "several fit"
    assert rec["reply"] == ("Max Roe (occupation: baseball player): Ruritania Land; "
                            "Max Roe (occupation: politician): Grand Fenwick.")
    assert [(a["item"], a["answer"]) for a in rec["answers"]] == [(ids["max_b"], "Ruritania Land"), (ids["max_p"], "Grand Fenwick")]


def test_each_namesake_walked_apart_passes_the_loops_checks(made, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    loop, ids = made
    monkeypatch.setattr(loop, "_time_check", lambda q, lr: "it ended" if lr.result.answer == "Grand Fenwick" else None)
    rec = loop.ask(f"What is the {CIT} of Max Roe?")
    assert rec["reply"] == ("Max Roe (occupation: baseball player): Ruritania Land; "
                            "Max Roe (occupation: politician): I do not know.")
    assert [(a["verified"], a["reason"]) for a in rec["answers"]] == [(True, None), (False, "time_mismatch")]


def test_two_nothing_tells_apart_and_the_flag_off_are_asked_which(made, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    loop, ids = made
    rec = loop.ask(f"What is the {CIT} of Ann Lee?")                  # the same value, no fact of their own apart
    assert not rec["verified"] and rec["reason"] == "ambiguous_entity"
    monkeypatch.delenv("CB_NAMESAKE_RELATION")
    for q in ("What is the capital of Fredonia?", f"What is the {CIT} of Max Roe?"):
        rec = loop.ask(q)
        assert not rec["verified"] and rec["reason"] == "ambiguous_entity" and "namesakes" not in rec
