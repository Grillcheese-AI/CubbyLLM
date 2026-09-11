"""Pins for `pipeline.answer(..., known=)` -- the plan disposer in the forward
path. Fake retriever, fake VM, fake vocabulary: no store, no encoder, no
subprocess. Run: python -m pytest validation/test_pipeline_plan_refusal.py -q

What is pinned:
  * with `known`, a plan asking for a relation the store lacks is refused
    BEFORE any retrieval or VM call, with reason "unknown_relation" and the
    offending relations carried on the result;
  * with `known=None` the same question walks exactly as before -- the
    disposer is opt-in and the old path is byte-for-byte unchanged;
  * a plan that verifies is NOT touched: the walk runs, the VM is called,
    and the result is identical with or without `known`.
"""
from __future__ import annotations

import pathlib
import re
import sys

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.pipeline import answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402

STORE = [
    "france is the country of citizenship of jean",
    "paris is the capital of france",
]


class Vocab:
    def __init__(self, rels):
        self._r = {normalize(r) for r in rels}
    def __contains__(self, rel):
        return normalize(rel) in self._r
    def words(self):
        from cubbyllm.reasoning.plan_verify import _relation_words
        return _relation_words(self._r)


KNOWN = Vocab({"capital", "country of citizenship"})


def make_retrieve(calls):
    def retrieve(query, k):
        calls.append(query)
        q = normalize(query)
        return [(1.0 if any(w in f for w in q.split()) else 0.0, f) for f in STORE][:k]
    return retrieve


def faithful_vm(calls):
    """Recovers what the program bound for the queried role (the pattern from
    test_cot_counterfactuals.py); records every call."""
    def run(source, fn):
        calls.append(fn)
        mm = re.search(rf"function {fn}\([^)]*\): str \{{(.*?)\n    \}}", source, re.S)
        body = mm.group(1)
        role = re.search(r"recover\(frame, (\w+)\)", body).group(1)
        bm = re.search(rf'bind frame, {role}, "((?:[^"\\]|\\.)*)";', body)
        if bm is None:
            return {"ok": True, "result": None, "similarity": None}
        return {"ok": True, "result": bm.group(1), "similarity": 0.9}
    return run


def run(q, known):
    ret_calls, vm_calls = [], []
    r = answer(q, make_retrieve(ret_calls), faithful_vm(vm_calls), tau_vm=0.5, tau_ret=0.5,
               top_k=3, max_repairs=1, known=known)
    return r, ret_calls, vm_calls


def test_unknown_relation_is_refused_before_any_retrieval_or_vm_call():
    q = "What is the award received by the director of photography of Some Film?"
    r, ret_calls, vm_calls = run(q, KNOWN)
    assert not r.verified and r.answer is None
    assert r.reason == "unknown_relation"
    assert r.refused["unknown_relations"] == ["award received by the director of photography"]
    assert r.refused["covers"] is True
    assert ret_calls == [] and vm_calls == []          # zero cost: the whole point
    assert r.trace == [] and r.source is None


def test_without_known_the_old_path_is_unchanged():
    q = "What is the award received by the director of photography of Some Film?"
    r, ret_calls, vm_calls = run(q, None)
    assert r.refused is None
    assert r.reason == "retrieval_exhausted"           # it walks, and dies the old way
    assert ret_calls                                    # ... having paid for retrieval


def test_a_verifying_plan_walks_identically_with_or_without_known():
    q = "What is the capital of the country of citizenship of Jean?"
    r0, ret0, vm0 = run(q, None)
    r1, ret1, vm1 = run(q, KNOWN)
    assert r0.verified and r1.verified
    assert normalize(r0.answer) == normalize(r1.answer) == "paris"
    assert r1.refused is None
    assert ret0 == ret1 and vm0 == vm1                  # same calls, same order
    assert [h.fact for h in r0.trace] == [h.fact for h in r1.trace]


def test_plan_that_does_not_cover_is_refused_with_that_reason(monkeypatch):
    """Force a plan/question mismatch by making the parser return a plan for a
    different question -- the emitter failure mode the disposer exists for."""
    import cubbyllm.reasoning.pipeline as pl
    from cubbyllm.reasoning.planner import parse_question
    other = parse_question("What is the capital of the country of citizenship of Jean?")
    monkeypatch.setattr(pl, "parse_question", lambda q: other)
    r, ret_calls, vm_calls = run("What is the population of Mars?", KNOWN)
    assert r.reason == "plan_does_not_cover_question" and r.refused["covers"] is False
    assert ret_calls == [] and vm_calls == []


def test_injected_plan_replaces_the_grammar_and_is_disposed_of_the_same_way():
    """`answer(..., plan=)`: an emitter's plan goes through the disposer and the
    walk exactly like a grammar plan; the grammar is not consulted."""
    from cubbyllm.reasoning.planner import QuestionPlan
    q = "What is the capital of the country of citizenship of Jean?"
    good = QuestionPlan(relations=[None, "capital"], tail="country of citizenship of jean", n_hop=2)
    r, ret_calls, vm_calls = run(q, KNOWN)
    r2, ret2, vm2 = answer(q, make_retrieve([]), faithful_vm([]), tau_vm=0.5, tau_ret=0.5,
                           top_k=3, max_repairs=1, known=KNOWN, plan=good), None, None
    assert r2.verified and normalize(r2.answer) == "paris"
    assert [h.fact for h in r2.trace] == [h.fact for h in r.trace]
    # a plan for a DIFFERENT question: coverage fails first, and the unknown relation is still reported
    other = QuestionPlan(relations=[None, "capital"], tail="award received by the creator of jean", n_hop=2)
    r3 = answer(q, make_retrieve([]), faithful_vm([]), tau_vm=0.5, tau_ret=0.5,
                top_k=3, max_repairs=1, known=KNOWN, plan=other)
    assert r3.reason == "plan_does_not_cover_question"
    assert r3.refused["unknown_relations"] == ["award received by the creator"]
    # a plan OF this question that asks for an edge the store lacks: unknown_relation, zero cost
    q2 = "What is the award received by the creator of Jean?"
    bad = QuestionPlan(relations=[None], tail="award received by the creator of jean", n_hop=1)
    calls = []
    r4 = answer(q2, make_retrieve(calls), faithful_vm(calls), tau_vm=0.5, tau_ret=0.5,
                top_k=3, max_repairs=1, known=KNOWN, plan=bad)
    assert r4.reason == "unknown_relation" and calls == []
