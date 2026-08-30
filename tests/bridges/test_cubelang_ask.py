"""ASK / resume over run-proto (2026-08-30) — the live contract through the
real cubelang binary. Skips when the exe is not built."""
import pytest

from cubbyllm.bridges import cubelang_client as cc

ASK_PROGRAM = """
program AskMin implements ISolve {
    storage { asked: mutable u64 = 0; }

    @system @once
    public function constructor() { assign asked = 0; }

    @external
    public function solve(input: str): number {
        create luna : number;
        assign luna = 1959;
        create apollo : number;
        assign apollo = 1969;
        ask "which moon landing did you mean", luna, apollo;
        create chosen : number;
        pop chosen;
        sum chosen;
        return chosen;
    }
}
"""


def _exe_or_skip():
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e}); build it with `cargo build --release` in the cubelang repo")


def test_ask_surfaces_as_a_suspension_not_an_error():
    _exe_or_skip()
    out = cc.run_program_proto(ASK_PROGRAM, fn="solve")
    assert out["ok"] and out["suspended"] is True and out["result"] is None
    assert out["question"] == "which moon landing did you mean"
    assert out["candidates"] == [1959, 1969]          # decoded Values, not JSON strings
    assert out["program"] == "AskMin" and out["function"] == "solve"


def test_resume_returns_the_chosen_candidate():
    _exe_or_skip()
    first = cc.run_program_proto(ASK_PROGRAM, fn="solve")
    out = cc.resume_program_proto(ASK_PROGRAM, fn="solve", answers=[first["candidates"][1]])
    assert out["ok"] and out["suspended"] is False and out["result"] == "1969"


def test_resume_rejects_an_invented_answer():
    _exe_or_skip()
    with pytest.raises(cc.CubelangRunError, match="not among"):
        cc.resume_program_proto(ASK_PROGRAM, fn="solve", answers=[1968])


def test_non_asking_program_reports_suspended_false():
    _exe_or_skip()
    src = ("use vsa;\nprogram R implements ISolve {\n    public function solve(input: str): str {\n"
           "        create frame: number;\n        bind frame, SUBJECT, \"cat\";\n        return recover(frame, SUBJECT);\n    }\n}\n")
    out = cc.run_program_proto(src, fn="solve")
    assert out["suspended"] is False and out["result"] == "cat"
