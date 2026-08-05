import pytest
from cubbyllm.bridges import cubelang_client as cc

def _exe_or_skip():
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e}); build it with `cargo build --release` in the cubelang repo")

def test_run_program_returns_parsed_json(tmp_path):
    _exe_or_skip()
    prog = tmp_path / "id.cube"
    prog.write_text(
        'program Id implements ISolve {\n'
        '    public function solve(input: str): str { return "ok"; }\n'
        '}\n'
    )
    out = cc.run_program(str(prog), fn="solve", args=["x"])
    assert out["ok"] is True
    assert out["result"] == "ok"

def test_run_error_raises(tmp_path):
    _exe_or_skip()
    prog = tmp_path / "bad.cube"
    prog.write_text("program Broken implements ISolve { this is not valid }\n")
    with pytest.raises(cc.CubelangRunError):
        cc.run_program(str(prog), fn="solve", args=["x"])
