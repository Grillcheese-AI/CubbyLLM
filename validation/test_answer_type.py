"""test_answer_type -- the typed answer class: the question's ask, the value's kind, the refusal.

What is pinned (2026-09-12, after exp_r11 lev6/lev6b):
  * `ask_type`: "in what year" / "on what day, month, and year" / "when" -> date; "how many"
    -> number; "who" -> name; "what is the capital of X" -> none;
  * `value_kinds`: an ISO date is a date; a bare year is a date OR a count; "29" is a
    number; "Paris" is a name;
  * the residual rule: 'year' left over in "in what year ..." is the ask, not a dropped
    hop, whatever the store's vocabulary holds -- coverage no longer depends on what the
    loop learned earlier;
  * the pipeline: a VM-certified chain whose value is a name where a date was asked is
    REFUSED (`answer_type_mismatch`, both kinds named), never spoken; the same chain for a
    question that asks for a name is spoken; a bare year satisfies "how many".
Run: python -m pytest validation/test_answer_type.py -q
"""
from __future__ import annotations

import pathlib
import sys

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.pipeline import answer  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations, answer_type_mismatch, ask_type, covers, value_kinds  # noqa: E402
from cubbyllm.reasoning.planner import QuestionPlan  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import LookupStore  # noqa: E402

NO_SEARCH = lambda q, k: []


def test_ask_type_reads_the_question():
    assert ask_type("In what year did Jean marry Marie?") == "date"
    assert ask_type("On what day, month, and year was Masaki Tsuji born?") == "date"
    assert ask_type("When was the Penny Crane Award established?") == "date"
    assert ask_type("How many U.S. patents did Chadwell O'Connor receive?") == "number"
    assert ask_type("Who is the father of Jean?") == "name"
    assert ask_type("What is the capital of France?") is None
    # a place (2026-09-13): 'where', or a place-class noun as the ask
    assert ask_type("Which district is Kafr al-Awamid located in?") == "place"
    assert ask_type("Where was Masaki Tsuji born?") == "place"
    assert ask_type("In which city did Frank Dobson die?") == "place"
    assert ask_type("What country is Jean a citizen of?") == "place"
    assert answer_type_mismatch("Where was Masaki Tsuji born?", "1932-03-23") == ("place", "date")
    assert answer_type_mismatch("Where was Masaki Tsuji born?", "Nagoya") is None
    # the leading interrogative decides: a where-question that mentions 'when' asks for a place (an LLM
    # wording the gate caught answered with a year); a class noun two words on still reads as a place
    assert ask_type("Where was John Joseph Sirica when he died?") == "place"
    assert answer_type_mismatch("Where was John Joseph Sirica when he died?", "1992") == ("place", "date/number")
    assert ask_type("In what Orkney parish was the poet Edwin Muir born?") == "place"
    assert ask_type("Alois Alzheimer, the German psychiatrist, was born in which Bavarian town?") == "place"
    assert ask_type("Madame de Stael was born in which year?") == "date"


def test_value_kinds_by_shape():
    assert value_kinds("1932-03-23") == {"date"}
    assert value_kinds("2000") == {"date", "number"}
    assert value_kinds("29") == {"number"}
    assert value_kinds("Paris") == {"name"}
    assert answer_type_mismatch("In what year was X founded?", "2000") is None
    assert answer_type_mismatch("How many people voted?", "1234") is None
    assert answer_type_mismatch("In what year did Jean marry Marie?", "marie") == ("date", "name")
    assert answer_type_mismatch("Who is the spouse of Jean?", "1932-03-23") == ("name", "date")
    assert answer_type_mismatch("What is the capital of France?", "1932") is None      # no ask, no rule


def test_the_asks_words_are_frame_not_a_dropped_hop_whatever_the_store_holds():
    store = ["marie is the spouse of jean", "2001 is the fiscal year of acme"]       # the vocabulary holds 'year'
    known = StoreRelations(store)
    plan = QuestionPlan(relations=[None], tail="spouse of jean", n_hop=1)
    assert covers("What is the spouse of Jean?", plan, known)
    assert covers("In what year was the spouse of Jean?", plan, known)          # 'year' is the ask, covered
    assert not covers("What is the fiscal year of the spouse of Jean?", plan, known)   # a real dropped hop still fails


def test_a_certified_chain_of_the_wrong_kind_is_refused_never_spoken():
    store = LookupStore(["marie is the spouse of jean", "1932-03-23 is the date of birth of jean"])
    known = StoreRelations(store.texts)
    plan = QuestionPlan(relations=[None], tail="spouse of jean", n_hop=1)
    r = answer("In what year was the spouse of Jean?", NO_SEARCH, faithful_vm([]), tau_vm=0.4, tau_ret=0.0,
               lookup=store.lookup, known=known, plan=plan)
    assert not r.verified and r.answer is None and r.reason == "answer_type_mismatch"
    assert r.refused["asked"] == "date" and r.refused["got"] == "name" and r.refused["value"] == "marie"
    assert r.refused["facts"] == ["marie is the spouse of jean"]                   # the chain the VM certified, on record
    r2 = answer("Who is the spouse of Jean?", NO_SEARCH, faithful_vm([]), tau_vm=0.4, tau_ret=0.0,
                lookup=store.lookup, known=known, plan=plan)
    assert r2.verified and r2.answer == "marie"
    plan_dob = QuestionPlan(relations=[None], tail="date of birth of jean", n_hop=1)
    r3 = answer("What is the date of birth of Jean?", NO_SEARCH, faithful_vm([]), tau_vm=0.4, tau_ret=0.0,
                lookup=store.lookup, known=known, plan=plan_dob)
    assert r3.verified and r3.answer == "1932-03-23"
