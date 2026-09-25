"""Graph-of-thought at an ambiguous hop (2026-09-24): explore every branch, certify
each in the VM, speak only when they all agree. Fake VM, no subprocess."""
import re

from cubbyllm.reasoning.index import TripleIndex
from cubbyllm.reasoning.pipeline import answer

Q2 = "What is the continent of the country of citizenship of marie?"
Q3 = "What is the currency of the continent of the country of citizenship of marie?"
FR = "france is the country of citizenship of marie"
BE = "belgium is the country of citizenship of marie"
EU_FR = "europe is the continent of france"
EU_BE = "europe is the continent of belgium"


def no_search(query, k):
    return []


def echo_vm(reject=(), sim=0.93):
    """Recovers the object each hop function binds; `reject` objects come back below tau."""
    def run(source, fn):
        body = source.split(f"function {fn}(", 1)[1].split("}", 1)[0]
        role = re.search(r"recover\(frame, (\w+)\)", body).group(1)
        if role == "ABSENT_CTRL":
            return {"ok": True, "result": None, "similarity": None}
        obj = re.search(rf"bind frame, {role}, \"([^\"]*)\"", body).group(1)
        return {"ok": True, "result": obj, "similarity": 0.1 if obj in reject else sim}
    return run


def ask(q, facts, **kw):
    ix = TripleIndex(facts)
    return answer(q, no_search, kw.pop("vm", echo_vm()), tau_vm=0.5, tau_ret=0.2,
                  lookup=ix.hop, **kw)


def test_branches_that_agree_are_spoken_and_say_how():
    r = ask(Q2, [FR, BE, EU_FR, EU_BE])
    assert r.verified is True and r.answer == "europe" and r.reason is None
    assert [b["status"] for b in r.branches] == ["verified", "verified"]
    assert sorted(b["via"][0]["object"] for b in r.branches) == ["belgium", "france"]
    assert r.trace[0].source == "branch" and all(h.similarity >= 0.5 for h in r.trace)
    assert all(not any(k.startswith("_") for k in b) for b in r.branches)


def test_branches_that_disagree_stay_an_ambiguous_refusal_that_names_both():
    r = ask(Q2, [FR, BE, EU_FR, "atlantis is the continent of belgium"])
    assert r.verified is False and r.answer is None and r.reason == "ambiguous_hop"
    assert r.refused["hop"] == 0 and r.refused["objects"] == ["belgium", "france"]
    assert r.refused["verified_answers"] == ["atlantis", "europe"] and r.refused["overflow"] is False
    assert {b["answer"] for b in r.branches} == {"atlantis", "europe"}
    assert r.trace == []                       # the partial trace, as before branching


def test_a_last_hop_set_is_never_spoken():
    """Two values for the LAST hop are two answers: the poisoned-capital case."""
    r = ask("What is the capital of the country of citizenship of marie?",
            [FR, "paris is the capital of france", "lyon is the capital of france"])
    assert r.reason == "ambiguous_hop" and r.refused["hop"] == 1
    assert r.refused["verified_answers"] == ["lyon", "paris"]
    assert len(r.trace) == 1 and r.trace[0].fact == FR


def test_a_branch_the_store_cannot_finish_blocks_the_others_in_an_open_world():
    r = ask(Q2, [FR, BE, EU_FR])
    assert r.reason == "ambiguous_hop"
    st = {b["via"][0]["object"]: b for b in r.branches}
    assert st["france"]["status"] == "verified" and st["belgium"]["status"] == "pruned"
    assert st["belgium"]["stalled"] == "belgium" and st["belgium"]["answer"] is None
    # a store declared complete refutes the branch that cannot finish
    c = ask(Q2, [FR, BE, EU_FR], closed_world=True)
    assert c.verified is True and c.answer == "europe"


def test_a_branch_the_vm_rejects_blocks_even_in_a_closed_world():
    r = ask(Q2, [FR, BE, EU_FR, EU_BE], vm=echo_vm(reject={"belgium"}), closed_world=True)
    assert r.reason == "ambiguous_hop"
    bad = [b for b in r.branches if b["status"] == "vm_failed"]
    assert len(bad) == 1 and bad[0]["via"][0]["object"] == "belgium" and bad[0]["clauses"]


def test_a_nested_ambiguity_branches_again_and_still_needs_agreement():
    facts = [FR, BE, EU_FR, EU_BE, "eurasia is the continent of belgium",
             "euro is the currency of europe", "euro is the currency of eurasia"]
    r = ask(Q3, facts)
    assert r.verified is True and r.answer == "euro" and len(r.branches) == 3
    assert sorted(len(b["via"]) for b in r.branches) == [1, 2, 2]
    d = ask(Q3, facts[:-1] + ["tenge is the currency of eurasia"])
    assert d.reason == "ambiguous_hop" and d.refused["verified_answers"] == ["euro", "tenge"]


def test_the_branch_bound_is_an_overflow_and_an_overflow_never_speaks():
    r = ask(Q2, [FR, BE, EU_FR, EU_BE], branch=1)
    assert r.reason == "ambiguous_hop" and r.refused["overflow"] is True and len(r.branches) == 1


def test_branch_zero_is_the_refuse_on_sight_walk():
    r = ask(Q2, [FR, BE, EU_FR, EU_BE], branch=0)
    assert r.reason == "ambiguous_hop" and r.branches is None
    assert set(r.refused) == {"hop", "objects", "facts"}


def test_a_choice_narrows_to_an_offered_branch_and_never_adds_one():
    facts = [FR, BE, EU_FR, "atlantis is the continent of belgium"]
    r = ask(Q2, facts, choose={0: "belgium"})
    assert r.verified is True and r.answer == "atlantis"
    assert r.trace[0].fact == BE and r.branches is None
    x = ask(Q2, facts, choose={0: "germany"})             # not a candidate: ignored
    assert x.reason == "ambiguous_hop"


def test_branches_do_not_share_hop_records():
    r = ask(Q3, [FR, BE, EU_FR, EU_BE, "eurasia is the continent of belgium",
                 "euro is the currency of europe", "tenge is the currency of eurasia"])
    assert r.reason == "ambiguous_hop"
    facts = [tuple(b["facts"]) for b in r.branches]
    assert len(set(facts)) == 3 and all(len(f) == 3 for f in facts)
