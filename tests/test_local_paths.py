"""Order 0030: no committed text file carries a local path -- a drive-letter folder, a user folder, the local venv's
name -- by the patterns ``tools/scrub_local_paths.py`` rewrites (it names the file to run on a hit). A planted path is
caught, so the guard can fail; the planted strings are built here at run time, so this file holds none.
Run: python -m pytest tests/test_local_paths.py
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scrub_local_paths", ROOT / "tools" / "scrub_local_paths.py")
scrub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scrub)


def committed() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.splitlines() if Path(f).suffix.lower() in scrub.TEXT and f != "tools/scrub_local_paths.py"]


def test_no_committed_text_file_carries_a_local_path():
    bad = []
    for f in committed():
        p = ROOT / f
        if p.is_file() and scrub.hit(p.read_text(encoding="utf-8", errors="replace")):
            bad.append(f)
    assert not bad, f"local paths in {bad[:10]} ({len(bad)} files): run python tools/scrub_local_paths.py --write <files>"


@pytest.mark.parametrize("planted", [
    "E" + ":" + "\\" + "datasets\\hdc",                     # a data folder on a drive
    "C" + ":/" + "Users/someone/Documents/GitHub/CubbyLLM",  # a checkout under a user folder
    "I" + ":" + "\\\\" + "My Drive\\\\cubbyllm",             # JSON-escaped, the synced Drive
    "/c/" + "Users/someone/x",                              # a Git Bash user folder
    "python ." + "venv-dml/Scripts/python.exe",             # the local venv's name
])
def test_a_planted_path_is_caught_and_rewritten(planted):
    assert scrub.hit(planted)
    assert not scrub.hit(scrub.rewrite(planted, ".md"))


@pytest.mark.parametrize("fine", ["https://example.org/a:b", "Q: what?\\nA: yes", "a dict {k: v}", "ratio 3:1",
                                  "$CUBBY_DATASETS/hdc", "/content/drive/MyDrive/cubbyllm"])
def test_ordinary_text_is_not_a_local_path(fine):
    assert not scrub.hit(fine)


def test_python_defaults_keep_resolving(tmp_path):
    src = 'import os\nPATH = os.environ.get("X", r"C' + ':\\CUBBY-TRAINED-MODELS\\m.pt")\n'
    out = scrub.python(src, 1)
    assert '_lp(r"$CUBBY_MODELS\\m.pt")' in out and "from cubbyllm.local_paths import expand as _lp" in out
    compile(out, "x.py", "exec")
