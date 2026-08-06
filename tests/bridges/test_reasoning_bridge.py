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


# ── Task 9 (Feature A, Python side): similarity separation, JSON transport ──
#
# Same separation as the protobuf side (test_reasoning_bridge_proto.py): a
# real recovery surfaces a high similarity, an absent binding surfaces none.
# `run_program` returns cmd_run's JSON verbatim, so `.get("similarity")` is
# how a caller reads it defensively (some JSON shapes, e.g. a suspended run,
# omit the key entirely rather than sending null).

def test_solve_similarity_present_and_high():
    _client_or_skip()
    out = cc.run_program(str(PROG), fn="solve", args=["_"])
    assert out["result"] == "cat"
    assert out.get("similarity") is not None, "a clean recover() must surface a similarity"
    assert out["similarity"] > 0.3, f"expected high-confidence similarity, got {out.get('similarity')}"


def test_wrong_role_similarity_present_and_high():
    _client_or_skip()
    out = cc.run_program(str(PROG), fn="wrong_role")
    assert out["result"] == "mouse"
    assert out.get("similarity") is not None, "wrong_role is still a clean recover(), just of the other role"
    assert out["similarity"] > 0.3, f"expected high-confidence similarity, got {out.get('similarity')}"


def test_unbound_similarity_is_none():
    _client_or_skip()
    out = cc.run_program(str(PROG), fn="unbound")
    assert out["result"] is None
    assert out.get("similarity") is None, "no winning match means no similarity, not e.g. 0.0"


# ── Task 9: verify-before-execute over the JSON transport ──────────────────
#
# Unlike `run_program_proto` (whose `run-proto` transport is unconditionally
# strict server-side), `run_program`'s `--json` path only verifies when the
# new `strict` parameter asks it to. It defaults to True specifically so the
# reasoning bridge -- which always passes strict cleanly -- gets
# verify-before-execute by default without every caller opting in.

STRICT_VIOLATION_PROGRAM_SRC = """
program Test implements ISolve {
    public function solve(input: str): void {
        infer x;
    }
}
"""


def test_strict_violation_raises_by_default(tmp_path):
    _client_or_skip()
    prog = tmp_path / "strict_violation.cube"
    prog.write_text(STRICT_VIOLATION_PROGRAM_SRC)
    with pytest.raises(cc.CubelangRunError):
        cc.run_program(str(prog), fn="solve", args=["_"])


def test_strict_false_allows_the_same_program_through(tmp_path):
    # Proves `strict` actually threads to the CLI rather than being a dead
    # parameter: the identical program that Task 9's default-strict test
    # above rejects must succeed (as a silent no-op on the trace-only
    # `infer`) once strict verification is explicitly turned off.
    _client_or_skip()
    prog = tmp_path / "strict_violation.cube"
    prog.write_text(STRICT_VIOLATION_PROGRAM_SRC)
    out = cc.run_program(str(prog), fn="solve", args=["_"], strict=False)
    assert out["ok"] is True
    assert out["result"] is None
