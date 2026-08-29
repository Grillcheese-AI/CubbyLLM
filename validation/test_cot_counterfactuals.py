"""Unit pins for the counterfactual-neighborhood harvest in
exp_m3_cot_pipeline.py (harvest-schema `counterfactuals[]`). Fake VM only --
no subprocess, no corpus, no table. Run: python -m pytest validation/test_cot_counterfactuals.py -q
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

import numpy as np  # noqa: E402

import exp_m3_cot_pipeline as m  # noqa: E402
from cubbyllm.reasoning.planner import Triple, normalize  # noqa: E402

TRIPLES = [
    Triple(obj="united stated", rel="country of citizenship", subj="cynthia basinet"),
    Triple(obj="north america", rel="continent", subj="united stated"),
    Triple(obj="earth", rel="planet", subj="north america"),
]
RELS = [t.rel for t in TRIPLES]
CAL_STORE = [
    "berlin is the capital of germany",
    "oceania portal is the continent of australia",
    "paris is the capital of france",
    "mars is the planet of olympus mons",
]


def _fn_body(source: str, fn: str) -> str:
    mm = re.search(rf"function {fn}\([^)]*\): str \{{(.*?)\n    \}}", source, re.S)
    assert mm is not None, fn
    return mm.group(1)


def faithful_vm(source, fn):
    """Recovers exactly what the program bound for the queried role --
    the real VM's behaviour on a single-binding frame (fidelity != truth)."""
    body = _fn_body(source, fn)
    role = re.search(r"recover\(frame, (\w+)\)", body).group(1)
    bm = re.search(rf'bind frame, {role}, "((?:[^"\\]|\\.)*)";', body)
    if bm is None:
        return {"ok": True, "result": None, "similarity": None}
    return {"ok": True, "result": bm.group(1), "similarity": 0.9}


def run(vm, triples=TRIPLES, budget=None, floor=0.5, seed=0):
    budget = [10_000] if budget is None else budget
    rng = np.random.default_rng(seed)
    return m._counterfactual_neighborhood(triples, RELS, CAL_STORE, rng, vm, floor, budget)


def test_outcome_helper_classifies_every_path():
    # content wrong -> symbol_mismatch wins even at high similarity
    assert m._cf_outcome("berlin", 0.95, "paris", 0.5) == (True, False, True, "symbol_mismatch")
    # faithful symbol but garbled recovery -> below_floor
    assert m._cf_outcome("paris", 0.2, "paris", 0.5) == (False, True, True, "below_floor")
    # nothing recovered at all
    assert m._cf_outcome(None, None, "paris", 0.5) == (True, True, True, "symbol_mismatch")
    # faithful AND confident -> the fault ESCAPED (would-be false accept)
    assert m._cf_outcome("Paris", 0.9, "paris", 0.5) == (False, False, False, None)


def test_faithful_vm_catches_every_planted_fault_by_content():
    recs, n_calls, n_moji, truncated = run(faithful_vm)
    assert recs and n_calls == len(recs) and n_moji == 0 and not truncated
    assert {r["cls"] for r in recs} == set(m.FAULT_CLASSES)      # 3-hop chain: all four classes
    for r in recs:
        assert r["caught"] is True and r["caught_by"] == "symbol_mismatch", r
        assert r["n_hop"] == 3 and 0 <= r["hop"] < 3
        assert set(r["planted"]) == {"obj", "rel", "subj"}
        assert normalize(r["symbol"]) != normalize(r["compare_obj"])
    # only the corrupted hop is attempted per instance (innocent bystanders
    # are never counted): wrong_entity/inverted -> 1 hop each, hop_order -> 2
    per_cls = {c: [r for r in recs if r["cls"] == c] for c in m.FAULT_CLASSES}
    assert len(per_cls["wrong_entity"]) <= m.MAX_FAULTS_PER_CLASS_PER_CHAIN
    assert len(per_cls["inverted_direction"]) <= m.MAX_FAULTS_PER_CLASS_PER_CHAIN
    assert len(per_cls["wrong_hop_order"]) == 2 * m.MAX_FAULTS_PER_CLASS_PER_CHAIN
    # wrong_entity plants an object that is NOT the chain's true object
    for r in per_cls["wrong_entity"]:
        assert normalize(r["planted"]["obj"]) != normalize(r["compare_obj"])


def test_lying_vm_records_escaped_faults_uncaught():
    """A VM that returns the TRUE object regardless of what was bound is the
    false-accept case: the fault must be logged as escaped (caught=False)."""
    truth = {m.build_chain_program(TRIPLES, RELS)[1][i]: t.obj for i, t in enumerate(TRIPLES)}

    def lying_vm(source, fn):
        return {"ok": True, "result": truth.get(fn), "similarity": 0.9}

    recs, *_ = run(lying_vm)
    binding_faults = [r for r in recs if r["cls"] != "wrong_relation"]
    assert binding_faults
    assert all(r["caught"] is False and r["caught_by"] is None for r in binding_faults)
    # wrong_relation compares hop k's TRUE object against hop k+1's -> still a mismatch
    assert all(r["caught"] for r in recs if r["cls"] == "wrong_relation")


def test_budget_truncates_and_reports_it():
    budget = [3]
    recs, n_calls, _n_moji, truncated = run(faithful_vm, budget=budget)
    assert n_calls == 3 and len(recs) == 3 and truncated is True and budget[0] == 0


def test_non_ascii_hop_is_excluded_not_run():
    triples = [Triple(obj="münchen", rel="capital", subj="bavaria"), TRIPLES[1], TRIPLES[2]]
    calls = []

    def vm(source, fn):
        calls.append(fn)
        return faithful_vm(source, fn)

    recs, n_calls, n_moji, _ = run(vm, triples=triples)
    assert n_moji > 0
    # no record may carry the mojibake hop, and no VM call was spent on it
    assert all(r["planted"]["obj"].isascii() and r["planted"]["subj"].isascii() for r in recs)
    assert n_calls == len(recs) == len(calls)


def test_wrong_relation_skips_hops_whose_objects_coincide():
    """Regression (v3cf run, 2026-08-28): a chain with the same object at two
    hops made wrong_relation compare a recovery against itself -- a no-op
    fault that then looked like an 'escape'. Such hops must not be planted."""
    triples = [
        Triple(obj="united stated", rel="country of citizenship", subj="cynthia basinet"),
        Triple(obj="united stated", rel="country", subj="united stated"),
        Triple(obj="oceania portal", rel="continent", subj="united stated"),
    ]
    rels = [t.rel for t in triples]
    rng = np.random.default_rng(0)
    recs, *_ = m._counterfactual_neighborhood(triples, rels, CAL_STORE, rng, faithful_vm, 0.5, [10_000])
    wr = [r for r in recs if r["cls"] == "wrong_relation"]
    assert wr, "shift-2 instance (hop0 vs hop2) must still be planted"
    for r in wr:
        assert normalize(r["planted"]["obj"]) != normalize(r["compare_obj"]), r
        assert r["caught"] is True
