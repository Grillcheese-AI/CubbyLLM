"""Order 0076, step 1: CheezML at a pinned commit (`vendor/cheezml`, a git submodule), checked by its own
`tools/validate.py`: the shapes parse, every positive example conforms and every negative one fails for its stated
reason. The pin is the commit below; moving it is a commit here that changes this line."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CZ = ROOT / "vendor" / "cheezml"
PIN = "bc85026"


def _have():
    if not (CZ / "tools" / "validate.py").exists():
        pytest.skip("vendor/cheezml is not checked out (git submodule update --init)")
    for m in ("rdflib", "pyshacl"):
        pytest.importorskip(m)


def test_the_submodule_is_at_its_pin():
    _have()
    head = subprocess.run(["git", "-C", str(CZ), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head.startswith(PIN), head


def test_cheezml_validates_at_its_pin():
    _have()
    got = subprocess.run([sys.executable, str(CZ / "tools" / "validate.py"), str(CZ / "src")], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=600)
    assert got.returncode == 0 and "RESULT: ALL CHECKS PASS" in got.stdout, got.stdout[-2000:] + got.stderr[-2000:]
    assert "08-stored-statement.jsonld" in got.stdout
