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


def test_control_leak_with_result_no_similarity():
    """Control returning result with no similarity is a leak."""
    def leaky_result_vm(source, fn):
        if fn == "control":
            return {"ok": True, "result": "ghost", "similarity": None}
        return good_vm(source, fn)
    r = answer(Q3, good_retriever, leaky_result_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False


def test_control_low_similarity_symbol_passes():
    """Control returning a symbol with low similarity is the expected absent-role
    outcome (noise from cosine cleanup on populated frame) and passes verification.
    High similarity would indicate leaked information; None result is unverifiable.
    """
    def real_vm_behavior(source, fn):
        if fn == "control":
            # Real VM's nearest-neighbor cleanup: returns (noise_symbol, low_sim)
            return {"ok": True, "result": "noise_word", "similarity": 0.2}
        return good_vm(source, fn)
    r = answer(Q3, good_retriever, real_vm_behavior, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is True
    assert r.answer == "oceania portal"


def test_verify_fail_twice_single_repair_used():
    """Double verify failure uses exactly 1 repair (ban + retry, then fail).

    Both walks succeed without retries (retriever provides alternates after ban),
    but verify fails both times due to weak VM similarity.
    """
    F1_alt = "usa is the country of citizenship of cynthia basinet"
    F2_alt = "usa is the country usa is in"
    F3_alt = "oceania is the continent of usa"

    fact_index = {"solve": 0, "hop_2": 0, "hop_3": 0}

    def retriever_with_backup(query, k):
        ql = query.lower()
        if "cynthia" in ql:
            # Return both F1 and F1_alt; walk will use F1 first, then F1_alt after ban
            return [(0.9, F1), (0.8, F1_alt)]
        if "continent" in ql:
            return [(0.9, F3), (0.8, F3_alt)]
        # For "country" queries
        return [(0.9, F2), (0.8, F2_alt)]

    def weak_vm_both(source, fn):
        # All hops return weak similarity to trigger verify failure both times
        if fn == "control":
            return {"ok": True, "result": None, "similarity": None}
        return {"ok": True, "result": good_vm(source, fn)["result"],
                "similarity": 0.1}  # Below tau_vm=0.5

    r = answer(Q3, retriever_with_backup, weak_vm_both, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False
    assert r.repairs_used == 1  # ban + retry on attempt 0, then fail again


def test_walk_exhausts_budget_then_verify_fails():
    """Regression test: walk's retries exhaust budget, verify fails — no budget underflow.

    Scenario: max_repairs=1, walk uses that 1 repair on retrieval retry,
    succeeds with triples, but verify fails (weak similarity). Should not
    decrement budget to -1 (which would make repairs_used > max_repairs).
    Assert: verified=False, repairs_used == 1 (not 2), no exception.
    """
    retriever_call_count = [0]

    def retriever_one_then_valid(query, k):
        """First call returns nothing (triggers retry), second returns valid fact."""
        ql = query.lower()
        if "cynthia" in ql:
            retriever_call_count[0] += 1
            if retriever_call_count[0] == 1:
                # First call: return nothing to force a retry
                return []
            else:
                # Second call (after query refinement): return valid fact
                return [(0.9, F1)]
        if "continent" in ql:
            return [(0.9, F3)]
        # "country" queries always work
        return [(0.9, F2)]

    def vm_all_weak(source, fn):
        """All hops return weak similarity to fail verify."""
        if fn == "control":
            return {"ok": True, "result": None, "similarity": None}
        # All hops weak to trigger verify failure
        return {"ok": True, "result": "dummy", "similarity": 0.1}

    # max_repairs=1: walk uses it on retry, verify fails, no budget left
    r = answer(Q3, retriever_one_then_valid, vm_all_weak,
               tau_vm=0.5, tau_ret=0.2, max_repairs=1)
    assert r.verified is False
    assert r.repairs_used == 1  # Not 2; budget never goes negative
    assert r.reason == "vm_verify_failed"
    # Trace must be preserved even when budget exhausted on verify failure
    assert len(r.trace) >= 1, "Trace should be populated from the walk"
    assert all(h.similarity is not None for h in r.trace), \
        "Trace entries should have similarities from verify step"


def test_repair_recovers_with_alternate_fact():
    """Verify fails on attempt 0, fact is banned, attempt 1 finds alternate.

    Tests that when verify fails on a weak hop, that hop's fact is banned,
    and attempt 1 can find an alternate fact and succeed.
    repairs_used == 1 (the ban+retry counts as one repair).
    """
    # F2_alt: different phrasing, same meaning as F2 (same obj for downstream)
    F2_alt = "united stated is the country of united stated"

    def retriever_with_backup(query, k):
        """Provides both primary and alternate facts."""
        ql = query.lower()
        if "cynthia" in ql:
            return [(0.9, F1)]
        if "continent" in ql:
            return [(0.9, F3)]
        # For country queries: provide both F2 and F2_alt
        return [(0.9, F2), (0.85, F2_alt)]

    hop2_verify_count = [0]

    def vm_weak_then_strong(source, fn):
        """Weak on first verify, strong on retry."""
        if fn == "control":
            return {"ok": True, "result": None, "similarity": None}
        if fn == "solve":
            return {"ok": True, "result": "united stated", "similarity": 0.93}
        if fn == "hop_2":
            hop2_verify_count[0] += 1
            if hop2_verify_count[0] == 1:
                # First verify: weak similarity causes failure
                return {"ok": True, "result": "united stated", "similarity": 0.1}
            else:
                # Second verify: strong similarity succeeds
                return {"ok": True, "result": "united stated", "similarity": 0.93}
        if fn == "hop_3":
            return {"ok": True, "result": "oceania portal", "similarity": 0.93}
        return {"ok": True, "result": "unknown", "similarity": 0.93}

    r = answer(Q3, retriever_with_backup, vm_weak_then_strong,
               tau_vm=0.5, tau_ret=0.2, max_repairs=3)
    assert r.verified is True
    assert r.answer == "oceania portal"
    assert r.repairs_used == 1
