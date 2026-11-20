"""Order 0076, step 5 (validation/exp_disagreements.py): the pairs a record touched (its trace and an ambiguous hop's
values), the store's best-rank values of a pair (deprecated and non-best left out), and what the record names."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import exp_disagreements as X  # noqa: E402


def _st(guid, item="Q829", pid="P421"):
    return {"item": item, "pid": pid, "guid": guid}


def _rec(trace=(), hop=None, spoken=True, reason=None):
    cert = {"facts": [{"worlds": [{"statements": [s]}]} for s in trace]}
    if hop:
        cert["hop"] = {"facts": [{"worlds": [{"statements": [s]}]} for s in hop]}
    return {"spoken": spoken, "reason": reason, "cert": cert}


ENTITY = {"statements": [
    {"guid": "a", "pid": 421, "best": True, "rank": 1, "value": {"entity": "Q1"}, "references": [{}]},
    {"guid": "b", "pid": 421, "best": True, "rank": 1, "value": {"entity": "Q2"}},
    {"guid": "c", "pid": 421, "best": False, "rank": 1, "value": {"entity": "Q3"}},
    {"guid": "d", "pid": 421, "best": True, "rank": 0, "value": {"entity": "Q4"}},
    {"guid": "e", "pid": 38, "best": True, "rank": 1, "value": {"entity": "Q5"}}]}


def test_best_values_keep_best_rank_and_drop_deprecated():
    vals = X.best_values(lambda req: ENTITY, "Q829", "P421", {})
    assert [g for g, _, _ in vals] == ["a", "b"] and vals[0][2] == 1


def test_an_answer_names_one_value_and_a_hop_names_every_value():
    dis = {("Q829", "P421"): X.best_values(lambda req: ENTITY, "Q829", "P421", {})}
    spoken = _rec(trace=[_st("a")])
    got = X.classify(spoken, X.pairs(spoken), dis)
    assert got == {"did": "answered with one", "named": ["one value"], "hop listed": False}
    hop = _rec(hop=[_st("a"), _st("b")], spoken=False, reason="ambiguous_hop")
    got = X.classify(hop, X.pairs(hop), dis)
    assert got["did"] == "refused: ambiguous_hop" and got["named"] == ["every value"] and got["hop listed"]


def test_a_property_meant_to_hold_several_values_is_not_a_disagreement():
    props = {"P1082": {"statements": [{"pid": 2302, "value": {"entity": "Q19474404"}}]},
             "P37": {"statements": [{"pid": 2302, "value": {"entity": "Q21502404"}}]}}
    ask = lambda req: props.get(req["id"], {})                                            # noqa: E731
    assert X.single(ask, "P1082", {}) and not X.single(ask, "P37", {})
    hop = {"spoken": False, "reason": "ambiguous_hop"}
    no = {"same": False, "part": False, "links": []}
    assert X.step6(hop, [no], [False]).startswith("answers with every value")
    assert X.step6(hop, [no], [True]).startswith("a conflict refusal")
    assert X.step6({"spoken": False, "reason": "profile"}, [no], [True]).startswith("keeps its reason")


def test_values_the_store_says_are_the_same_are_linked_and_part_of_is_apart():
    ents = {"Q1": {"statements": [{"pid": 460, "rank": 1, "value": {"entity": "Q2"}, "guid": "x"}]},
            "Q2": {"statements": []}, "Q3": {"statements": [{"pid": 361, "rank": 1, "value": {"entity": "Q2"}, "guid": "y"}]}}
    ask = lambda req: ents[req["id"]]                                                       # noqa: E731
    v = lambda q: (q, '{"entity": "%s"}' % q, 0)                                           # noqa: E731
    assert X.linked(ask, [v("Q1"), v("Q2")], {})["same"]
    got = X.linked(ask, [v("Q1"), v("Q2"), v("Q3")], {})
    assert not got["same"] and got["part"]
