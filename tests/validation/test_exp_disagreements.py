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
