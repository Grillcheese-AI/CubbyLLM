"""Package-level success criteria (spec §7)."""
import subprocess
import sys


def test_import_and_version():
    import cubbyllm
    assert cubbyllm.__version__ == "0.0.0"


def test_import_is_torch_free():
    """`import cubbyllm` must not drag in torch (TYPE_CHECKING-only usage).

    Run in a fresh interpreter so no prior test's torch import pollutes the check.
    """
    code = (
        "import sys, cubbyllm; "
        "assert 'torch' not in sys.modules, 'import cubbyllm pulled in torch'; "
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=__import__("pathlib").Path(__file__).resolve().parent.parent,
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
