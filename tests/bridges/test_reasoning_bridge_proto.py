"""M2: re-run M1's three reasoning-bridge assertions over the protobuf
transport (cubelang's `run-proto` stdio mode), plus a cross-check that the
JSON (`run_program`) and protobuf (`run_program_proto`) transports agree.
Same program as test_reasoning_bridge.py (M1); only the wire changes.
"""
import pathlib
import pytest

from cubbyllm.bridges import cubelang_client as cc

PROG_PATH = pathlib.Path(cc.__file__).parent / "programs" / "reasoning_bridge.cube"
PROG = PROG_PATH.read_text()


def _skip_unless_ready():
    try:
        cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")
    pytest.importorskip("cubbyllm.bridges.reasoning_pb2")


def test_m2_matches_m1_over_protobuf():
    _skip_unless_ready()
    assert cc.run_program_proto(PROG, fn="solve", args=["_"])["result"] == "cat"
    assert cc.run_program_proto(PROG, fn="wrong_role")["result"] == "mouse"
    assert cc.run_program_proto(PROG, fn="unbound")["result"] is None


def test_m2_proto_agrees_with_json_transport():
    _skip_unless_ready()
    json_out = cc.run_program(str(PROG_PATH), fn="solve", args=["_"])
    proto_out = cc.run_program_proto(PROG, fn="solve", args=["_"])
    assert json_out["ok"] is True and proto_out["ok"] is True
    assert json_out["result"] == proto_out["result"] == "cat"


# ── Task 9 (Feature A, Python side): similarity separation ─────────────────
#
# A recovered symbol alone doesn't say how confident the match was. These
# assert the three controls separate on `similarity` the same way they
# separate on `result`: a real recovery (right role or wrong role -- both
# are clean single-pair binds, just of different roles) surfaces a high
# similarity; an absent binding surfaces none at all, not a low number.
# 0.3 mirrors cubelang's own `tests/proto_stdio.rs` threshold rationale: a
# 2-pair MAP-Bipolar bundle's recovered-role cosine centers near 0.5 by
# construction, so 0.3 sits well clear of noise on the low side with real
# headroom below the observed ~0.49-0.52 on the high side.

def test_solve_similarity_present_and_high():
    _skip_unless_ready()
    out = cc.run_program_proto(PROG, fn="solve", args=["_"])
    assert out["result"] == "cat"
    assert out["similarity"] is not None, "a clean recover() must surface a similarity"
    assert out["similarity"] > 0.3, f"expected high-confidence similarity, got {out['similarity']}"


def test_wrong_role_similarity_present_and_high():
    _skip_unless_ready()
    out = cc.run_program_proto(PROG, fn="wrong_role")
    assert out["result"] == "mouse"
    assert out["similarity"] is not None, "wrong_role is still a clean recover(), just of the other role"
    assert out["similarity"] > 0.3, f"expected high-confidence similarity, got {out['similarity']}"


def test_unbound_similarity_is_none():
    _skip_unless_ready()
    out = cc.run_program_proto(PROG, fn="unbound")
    assert out["result"] is None
    assert out["similarity"] is None, "no winning match means no similarity, not e.g. 0.0"


# ── Task 9: verify-before-execute over the protobuf transport ──────────────
#
# `run-proto` always strict-compiles (cubelang Task 8) -- there is no flag to
# thread here, unlike `run_program`'s JSON path. This proves the Python side
# actually surfaces that rejection as a `CubelangRunError`, not a silent
# no-op or a mis-parsed success.

STRICT_VIOLATION_PROGRAM = """
program Test implements ISolve {
    public function solve(input: str): void {
        infer x;
    }
}
"""


def test_strict_violation_raises_through_proto_transport():
    _skip_unless_ready()
    with pytest.raises(cc.CubelangRunError):
        cc.run_program_proto(STRICT_VIOLATION_PROGRAM, fn="solve", args=["_"])
