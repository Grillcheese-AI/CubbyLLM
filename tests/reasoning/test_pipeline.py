"""Pipeline paths with a fake retriever and fake VM (no subprocess)."""
import pytest

from cubbyllm.reasoning.pipeline import answer

Q3 = ("What is the continent of the country of the country of "
      "citizenship of cynthia basinet?")
F1 = "united stated is the country of citizenship of cynthia basinet"
F2 = "united stated is the country united stated is in"
F3 = "oceania portal is the continent of united stated"


def good_retriever(query, k):
    ql = query.lower()
    if "cynthia" in ql:
        return [(0.9, F1)]
    if "continent" in ql:
        return [(0.9, F3)]
    return [(0.9, F2)]


def good_vm(source, fn):
    # echoes the bound object for each hop role; empty for the control
    want = {"solve": "united stated", "hop_2": "united stated",
            "hop_3": "oceania portal"}
    if fn == "control":
        return {"ok": True, "result": None, "similarity": None}
    return {"ok": True, "result": want[fn], "similarity": 0.93}


def test_happy_three_hop_walk_is_verified():
    r = answer(Q3, good_retriever, good_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is True
    assert r.answer == "oceania portal"
    assert len(r.trace) == 3
    assert all(h.similarity >= 0.5 for h in r.trace)


def test_unparseable_question_fails_honestly():
    r = answer("Tell me about cheese.", good_retriever, good_vm,
               tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False and r.answer is None
    assert r.reason == "unparseable"


def test_missing_hop_repairs_then_fails_honestly():
    def no_hop2(query, k):
        ql = query.lower()
        if "cynthia" in ql:
            return [(0.9, F1)]
        return [(0.9, "totally unrelated text with no template")]
    r = answer(Q3, no_hop2, good_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False and r.answer is None
    assert r.repairs_used == 3
    assert len(r.trace) >= 1                       # partial trace preserved


def test_low_similarity_recover_fails_honestly():
    def weak_vm(source, fn):
        out = good_vm(source, fn)
        if fn == "hop_3":
            out = {"ok": True, "result": "oceania portal", "similarity": 0.1}
        return out
    r = answer(Q3, good_retriever, weak_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False


def test_control_violation_fails():
    def leaky_vm(source, fn):
        if fn == "control":
            return {"ok": True, "result": "ghost", "similarity": 0.95}
        return good_vm(source, fn)
    r = answer(Q3, good_retriever, leaky_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False


def test_claimed_answer_invariant():
    r = answer(Q3, good_retriever, good_vm, tau_vm=0.5, tau_ret=0.2)
    if r.verified:
        assert all(h.similarity is not None and h.similarity >= 0.5
                   for h in r.trace)
