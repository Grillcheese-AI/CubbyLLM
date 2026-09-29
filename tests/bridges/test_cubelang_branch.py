"""The branch on the wire (2026-09-28): a RunRequest carries a `Branch` (keys to exclude, facts to assume) that
the VM applies to that request's own clone of the knowledge store; every result carries a `BranchReport` (ops,
jumps, every QUERY with its hits, keys excluded, facts assumed). The fork the branch runner (D) is built on:
"X did not happen" lives in one VM and nowhere else. Needs cubelang.exe built with the 2026-09-28 proto."""
from __future__ import annotations

import pytest

from cubbyllm.bridges import cubelang_client as cc

COUNT = """
program Count implements ISolve {
    @external
    public function solve(mention: str): number {
        query mention;
        create ctx : number;
        pop ctx;
        let n = ctx.len();
        return n;
    }
}
"""


def _ready():
    try:
        cc.find_cubelang_exe()
    except Exception as e:
        pytest.skip(f"cubelang.exe not found ({e})")
    pb2 = pytest.importorskip("cubbyllm.bridges.reasoning_pb2")
    if not hasattr(pb2, "Branch"):
        pytest.skip("reasoning_pb2 predates the Branch message")


@pytest.fixture
def facts(tmp_path):
    p = tmp_path / "history.jsonl"
    p.write_text('{"key":"siege of vienna","text":"1683","source":"store"}\n'
                 '{"alias":"vienna 1683","of":"siege of vienna"}\n'
                 '{"key":"treaty of karlowitz","text":"1699","source":"store"}\n', encoding="utf-8")
    return str(p)


def test_a_branch_forks_the_store_for_one_request_only(facts):
    _ready()
    without = {"id": "without_siege", "exclude": ["Siege of Vienna"]}
    with cc.CubelangSession() as s:
        actual = s.run(COUNT, args=["siege of vienna"], knowledge_path=facts)
        branch = s.run(COUNT, args=["siege of vienna"], knowledge_path=facts, branch=without)
        alias = s.run(COUNT, args=["vienna 1683"], knowledge_path=facts, branch=without)
        rest = s.run(COUNT, args=["treaty of karlowitz"], knowledge_path=facts, branch=without)
        after = s.run(COUNT, args=["siege of vienna"], knowledge_path=facts)
    assert [r["result"] for r in (actual, branch, alias, rest, after)] == ["1", "0", "0", "1", "1"]
    rep = branch["report"]
    assert rep["branch_id"] == "without_siege" and rep["excluded"] == 2 and rep["assumed"] == 0
    assert rep["queried"] == [("siege of vienna", 0)] and rep["ops"] > 0
    assert actual["report"]["queried"] == [("siege of vienna", 1)] and actual["report"]["branch_id"] == ""


def test_a_branch_assumes_and_a_bad_assumption_fails_only_that_branch(facts):
    _ready()
    with cc.CubelangSession() as s:
        assumed = s.run(COUNT, args=["peace of 1684"], knowledge_path=facts,
                        branch={"id": "with_peace", "assume": [{"key": "peace of 1684", "text": "assumed", "source": "branch"}]})
        plain = s.run(COUNT, args=["peace of 1684"], knowledge_path=facts)
        with pytest.raises(cc.CubelangRunError) as e:
            s.run(COUNT, args=["x"], knowledge_path=facts, branch={"id": "bad", "assume": [{"key": "x", "text": "no source"}]})
        assert "branch bad: assume" in str(e.value) and e.value.report["branch_id"] == "bad"
        again = s.run(COUNT, args=["siege of vienna"], knowledge_path=facts)
    assert (assumed["result"], plain["result"], again["result"]) == ("1", "0", "1")
    assert assumed["report"]["assumed"] == 1
