"""Every run sure of its VM (Nick, 2026-11-13): `find_cubelang_exe` reads ``CUBELANG_EXE`` from the environment, then
``paths.local.json`` (`local_paths.value`); a binary older than its own sources is refused (`CubelangStale`, naming both
times) unless ``CB_ALLOW_STALE_VM=1`` lets it through for that run. Fixtures only: a fake checkout with a fake binary."""
from __future__ import annotations

import os
import time

import pytest

from cubbyllm import local_paths
from cubbyllm.bridges import cubelang_client as cc


def checkout(tmp_path, binary_older: bool):
    """A cubelang checkout: Cargo.toml, src/main.rs, target/release/cubelang.exe; the binary older or newer."""
    root = tmp_path / "cubelang-x"
    (root / "src").mkdir(parents=True)
    (root / "target" / "release").mkdir(parents=True)
    (root / "Cargo.toml").write_text("[package]\n")
    exe, src = root / "target" / "release" / "cubelang.exe", root / "src" / "main.rs"
    exe.write_bytes(b"fake")
    src.write_text("fn main() {}\n")
    now = time.time()
    os.utime(exe, (now - 3600, now - 3600) if binary_older else (now, now))
    os.utime(src, (now, now) if binary_older else (now - 3600, now - 3600))
    return exe


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("CUBELANG_EXE", raising=False)
    monkeypatch.delenv("CB_ALLOW_STALE_VM", raising=False)
    monkeypatch.setattr(local_paths, "_cache", {})


def test_the_key_in_paths_local_json_is_used_when_the_environment_is_unset(tmp_path, no_env, monkeypatch):
    exe = checkout(tmp_path, binary_older=False)
    monkeypatch.setattr(local_paths, "_cache", {"CUBELANG_EXE": str(exe)})
    assert cc.find_cubelang_exe() == exe


def test_the_environment_wins_over_the_file(tmp_path, no_env, monkeypatch):
    a, b = checkout(tmp_path / "a", False), checkout(tmp_path / "b", False)
    monkeypatch.setattr(local_paths, "_cache", {"CUBELANG_EXE": str(a)})
    monkeypatch.setenv("CUBELANG_EXE", str(b))
    assert cc.find_cubelang_exe() == b


def test_a_binary_older_than_its_sources_is_refused(tmp_path, no_env, monkeypatch):
    exe = checkout(tmp_path, binary_older=True)
    monkeypatch.setattr(local_paths, "_cache", {"CUBELANG_EXE": str(exe)})
    with pytest.raises(cc.CubelangStale, match="older than"):
        cc.find_cubelang_exe()
    with pytest.raises(cc.CubelangStale):
        cc.find_cubelang_exe(str(exe))                    # named outright: refused all the same
    assert issubclass(cc.CubelangStale, cc.CubelangNotFound)    # callers that skip "no VM" skip a stale one too


def test_the_flag_lets_a_stale_binary_through_with_the_warning(tmp_path, no_env, monkeypatch, capsys):
    exe = checkout(tmp_path, binary_older=True)
    monkeypatch.setenv("CB_ALLOW_STALE_VM", "1")
    assert cc.find_cubelang_exe(str(exe)) == exe
    assert "older than" in capsys.readouterr().err
