"""The VM half of the plan disposer: `QUERY` over the store's relation
vocabulary answers "is this relation one the store holds?".

Skips cleanly when no cubelang build is reachable (see
`bridges.cubelang_client.find_cubelang_exe`); the host-side pins in
test_plan_verify.py never need the VM. Run:
    python -m pytest validation/test_plan_verify_vm.py -q
"""
from __future__ import annotations

import pathlib
import sys

import pytest

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.bridges.cubelang_client import CubelangNotFound, find_cubelang_exe  # noqa: E402
from cubbyllm.reasoning.plan_verify import (  # noqa: E402
    StoreRelations, VMRelations, verify_plan, write_vocab_jsonl)
from cubbyllm.reasoning.planner import parse_question  # noqa: E402

FACTS = [
    "paris is the capital of france",
    "france is the country of citizenship of jean",
    "bob is the director of photography of some film",
    "unparseable line",
]


def test_vocab_jsonl_is_one_record_per_distinct_relation(tmp_path):
    n = write_vocab_jsonl(FACTS + ["berlin is the capital of germany"], tmp_path / "v.jsonl")
    lines = (tmp_path / "v.jsonl").read_text(encoding="utf-8").splitlines()
    assert n == 3 and len(lines) == 3
    import json
    recs = [json.loads(l) for l in lines]
    assert {r["key"] for r in recs} == {"capital", "country of citizenship", "director of photography"}
    assert all(r["source"] for r in recs)          # knowledge.rs refuses an unsourced fact


def test_vm_relations_memoizes_and_counts_calls(tmp_path):
    """No VM needed: the transport is injected. Pins the contract the real
    transport must meet -- a list result, membership = len > 0, one call per
    distinct normalized relation."""
    calls = []

    def fake_run(program, fn, args, exe, knowledge):
        calls.append(args[0])
        return {"ok": True, "result": [{"text": args[0]}] if args[0] == "capital" else []}

    vm = VMRelations(knowledge=tmp_path / "v.jsonl", run=fake_run)
    assert "capital" in vm and "Capital" in vm and "  capital " in vm
    assert "population" not in vm
    assert vm.n_calls == 2 and calls == ["capital", "population"]

    # the shipped program returns the VM's own verdict integer (ground_min's branch)
    def verdict_run(program, fn, args, exe, knowledge):
        return {"ok": True, "result": {"capital": 1, "mercury": 2}.get(args[0], 0)}
    vm2 = VMRelations(knowledge=tmp_path / "v.jsonl", run=verdict_run)
    assert "capital" in vm2 and "mercury" in vm2 and "population" not in vm2


def _exe():
    try:
        return str(find_cubelang_exe())
    except CubelangNotFound as e:
        pytest.skip(str(e))


def test_vm_agrees_with_host_on_membership(tmp_path):
    exe = _exe()
    kn = tmp_path / "vocab.jsonl"
    write_vocab_jsonl(FACTS, kn)
    host = StoreRelations(FACTS)
    vm = VMRelations(knowledge=kn, exe=exe)
    for rel in ("capital", "country of citizenship", "director of photography",
                "award received by the director of photography", "population", "Capital"):
        assert (rel in vm) == (rel in host), rel
    # six questions, FIVE queries: "Capital" normalizes to "capital", which was memoized.
    # (First run on 2026-09-11 pinned 6 and failed -- the memo was right, the pin was not.)
    assert vm.n_calls == 5


def test_vm_refuses_the_compound_relation_plan(tmp_path):
    exe = _exe()
    kn = tmp_path / "vocab.jsonl"
    write_vocab_jsonl(FACTS, kn)
    vm = VMRelations(knowledge=kn, exe=exe)
    q = "What is the award received by the director of photography of Some Film?"
    v = verify_plan(q, parse_question(q), vm)
    assert not v and v.reason == "unknown_relation"
    q2 = "What is the capital of the country of citizenship of Jean?"
    assert verify_plan(q2, parse_question(q2), vm).ok


# ---- resident process (CubelangSession) -- needs a cubelang build --------

def test_session_serves_many_requests_and_agrees_with_one_shot(tmp_path):
    exe = _exe()
    from cubbyllm.bridges.cubelang_client import CubelangSession, run_program_proto
    src = (ROOT / "cubbyllm" / "bridges" / "programs" / "reasoning_bridge.cube").read_text(encoding="utf-8")
    one_shot = {fn: run_program_proto(src, fn=fn, args=["_"], exe=exe) for fn in ("solve", "wrong_role", "unbound")}
    with CubelangSession(exe=exe) as s:
        resident = {fn: s.run(src, fn=fn, args=["_"]) for fn in ("solve", "wrong_role", "unbound")}
        again = s.run(src, fn="solve", args=["_"])
        assert s.n_requests == 4
    for fn in one_shot:
        assert resident[fn]["result"] == one_shot[fn]["result"], fn
        assert resident[fn]["similarity"] == one_shot[fn]["similarity"], fn
    assert again == resident["solve"]                       # fresh VM per request: order-independent


def test_session_knowledge_path_grounds_membership(tmp_path):
    exe = _exe()
    from cubbyllm.bridges.cubelang_client import CubelangSession
    kn = tmp_path / "vocab.jsonl"
    write_vocab_jsonl(FACTS, kn)
    host = StoreRelations(FACTS)
    with CubelangSession(exe=exe) as s:
        vm = VMRelations(knowledge=kn, session=s)
        for rel in ("capital", "country of citizenship", "director of photography",
                    "award received by the director of photography", "population", "Capital"):
            assert (rel in vm) == (rel in host), rel
        assert vm.n_calls == 5 and s.n_requests == 5          # memoized, one process
        # the paraphrase tier reads the same file the VM was given
        assert vm.match("director of the photography") == "director of photography"
        assert vm.match("population") is None
