import pathlib, pytest
from cubbyllm.bridges import cubelang_client as cc

PROG = pathlib.Path(cc.__file__).parent / "programs" / "reasoning_bridge.cube"

def _client_or_skip():
    try:
        cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")

def test_m1_recovers_the_planted_filler():
    _client_or_skip()
    out = cc.run_program(str(PROG), fn="solve", args=["_"])
    assert out["result"] == "cat", "recover(frame, SUBJECT) must return the planted filler"

def test_m1_wrong_role_control_differs():
    _client_or_skip()
    out = cc.run_program(str(PROG), fn="wrong_role")
    assert out["result"] == "mouse"
    assert out["result"] != "cat", "wrong-role control must recover a different filler"

def test_m1_absent_role_control_recovers_nothing():
    _client_or_skip()
    out = cc.run_program(str(PROG), fn="unbound")
    assert out["result"] is None, "an unbound frame must recover no symbol"
