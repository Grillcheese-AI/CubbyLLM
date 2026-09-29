"""VM step 0 on the wire (2026-09-29): an absent role is Null structurally (`recover_reason == "absent_role"`,
no similarity), a present role reports its runner-up and margin, the request carries a `Budget` (ops, queries,
wall, jumps) the VM refuses to cross, and `@internal`/`@system` are refused as the host's entry. Needs
cubelang.exe built with the 2026-09-29 proto."""
from __future__ import annotations

import pytest

from cubbyllm.bridges import cubelang_client as cc

FRAME = """
use vsa;
program Frame implements ISolve {
    @external
    public function solve(x: str): str {
        create frame: number;
        bind frame, SUBJECT, "cat";
        bind frame, OBJECT, "mouse";
        bind frame, VERB, "chased";
        return recover(frame, SUBJECT);
    }
    @external
    public function absent(x: str): str {
        create frame: number;
        bind frame, SUBJECT, "cat";
        bind frame, OBJECT, "mouse";
        bind frame, VERB, "chased";
        return recover(frame, LOCATION);
    }
    @internal
    function helper(x: str): str { return x; }
}
"""

LOOP = """
program Loop implements ISolve {
    @external
    public function solve(x: str): number {
        create count : number;
        assign count = 1000;
        create total : number;
        assign total = 0;
        while (count > 0) {
            add total, 1;
            sub count, 1;
        }
        return total;
    }
}
"""


def _ready():
    try:
        cc.find_cubelang_exe()
    except Exception as e:
        pytest.skip(f"cubelang.exe not found ({e})")
    pb2 = pytest.importorskip("cubbyllm.bridges.reasoning_pb2")
    if not hasattr(pb2, "Budget"):
        pytest.skip("reasoning_pb2 predates the Budget message")


def test_present_role_reports_runner_up_and_absent_role_is_null_with_reason():
    _ready()
    with cc.CubelangSession() as s:
        present = s.run(FRAME, args=["x"])
        absent = s.run(FRAME, fn="absent", args=["x"])
    assert present["result"] == "cat" and present["recover_reason"] == "recovered"
    assert present["runner_up"] is not None and present["runner_up"] < present["similarity"]
    hop = present["report"]["recovers"][0]
    assert hop["winner"] == "cat" and hop["role"] == "SUBJECT" and hop["pool"] == 3
    assert abs(hop["margin"] - (present["similarity"] - present["runner_up"])) < 1e-12
    assert absent["result"] is None and absent["similarity"] is None and absent["runner_up"] is None
    assert absent["recover_reason"] == "absent_role"
    assert absent["report"]["recovers"][0]["reason"] == "absent_role"


def test_internal_is_refused_as_the_entry_and_the_error_carries_the_report():
    _ready()
    with cc.CubelangSession() as s, pytest.raises(cc.CubelangRunError) as e:
        s.run(FRAME, fn="helper", args=["x"])
    assert "@internal" in str(e.value) and e.value.report is not None


def test_budgets_are_applied_echoed_and_refused_by_name():
    _ready()
    with cc.CubelangSession() as s:
        free = s.run(LOOP, args=["x"])
        assert free["result"] == "1000" and free["report"]["budget"]["max_ops"] == 0
        assert free["report"]["budget"]["max_jumps"] == 1_000_000
        with pytest.raises(cc.CubelangRunError) as ops:
            s.run(LOOP, args=["x"], budget={"max_ops": 500})
        assert "op budget" in str(ops.value) and ops.value.report["budget"]["max_ops"] == 500
        assert ops.value.report["ops"] == 501
        with pytest.raises(cc.CubelangRunError) as jumps:
            s.run(LOOP, args=["x"], budget={"max_jumps": 10})
        assert "jump budget" in str(jumps.value) and jumps.value.report["max_jumps"] == 10
        walled = s.run(LOOP, args=["x"], budget={"max_wall_ms": 10_000})
        assert walled["result"] == "1000" and walled["report"]["wall_ms"] < 10_000
