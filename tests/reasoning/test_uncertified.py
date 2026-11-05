"""cubbyllm/reasoning/uncertified.py -- the value a walk reached, certified or not (order 0054's C-VM arm).

Pinned: off (the default) the ask's record is the record it was, key for key; on, a verified answer's
``uncertified`` is the answer; a refusal keeps the value it reached when it reached one (a failed VM check on a chain
that reached every hop, an ambiguous hop's first object, a typed mismatch's value, an answer a check withheld) and
None when it reached none (no plan, a chain that stopped short).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
for p in (ROOT, ROOT / "standin", ROOT / "validation", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning import uncertified as unc  # noqa: E402
from cubbyllm.reasoning.pipeline import CoTResult, HopTrace  # noqa: E402
from cubbyllm.reasoning.planner import Triple  # noqa: E402
from test_ask import make_loop  # noqa: E402
from test_search_learn import DictSource  # noqa: E402

Q = "What is the capital of the country of citizenship of Marie?"
PLANS = {Q: ("marie", ["country of citizenship", "capital"])}
SRC = {"marie": ["canada is the country of citizenship of marie"], "canada": ["ottawa is the capital of canada"]}


def _hop(s, r, o):
    return HopTrace(query="", fact=f"{o} is the {r} of {s}", triple=Triple(obj=o, rel=r, subj=s), ret_score=1.0)


def test_off_the_record_is_unchanged_and_on_it_carries_the_answer(monkeypatch):
    monkeypatch.delenv("CB_RECORD_UNCERTIFIED", raising=False)
    off = make_loop(PLANS, DictSource(SRC)).ask(Q)
    monkeypatch.setenv("CB_RECORD_UNCERTIFIED", "1")
    on = make_loop(PLANS, DictSource(SRC)).ask(Q)
    assert "uncertified" not in off
    assert on["verified"] and on["uncertified"] == on["answer"]
    drop = {"uncertified", "wall_s", "graph", "hash", "learned"}       # the hash covers the record; learned is timed
    assert {k: v for k, v in on.items() if k not in drop} == {k: v for k, v in off.items() if k not in drop}
    assert [l["fact"] for l in on["learned"]] == [l["fact"] for l in off["learned"]]


def test_no_plan_reaches_no_value(monkeypatch):
    monkeypatch.setenv("CB_RECORD_UNCERTIFIED", "1")
    rec = make_loop({}, DictSource({})).ask("Who painted the ceiling of nowhere?")
    assert not rec["verified"] and rec.get("uncertified") is None


def test_refusals_keep_the_value_they_reached():
    full = [_hop("marie", "country of citizenship", "canada"), _hop("canada", "capital", "ottawa")]
    assert unc.value(CoTResult(answer=None, verified=False, trace=full, reason="vm_verify_failed"), 2) == "ottawa"
    short = full[:1]
    assert unc.value(CoTResult(answer=None, verified=False, trace=short, reason="vm_verify_failed"), 2) is None
    amb = CoTResult(answer=None, verified=False, reason="ambiguous_hop", refused={"hop": 1, "objects": ["a", "b"]})
    assert unc.value(amb, 2) == "a"
    typed = CoTResult(answer=None, verified=False, reason="answer_type_mismatch", refused={"value": "1999"})
    assert unc.value(typed, 1) == "1999"
    assert unc.value(CoTResult(answer=None, verified=False, reason="retrieval_exhausted"), 1) is None
    assert unc.value(CoTResult(answer="ottawa", verified=True, trace=full), 2) == "ottawa"
