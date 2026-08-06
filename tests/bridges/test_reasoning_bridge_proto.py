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
