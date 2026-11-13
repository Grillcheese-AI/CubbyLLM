"""cubbyllm/bridges/vm_identity.py -- which VM a run used, for its summary (Nick, 2026-11-13: every run names its VM).

`identity()` -> {"path", "modified", "sha256", "commit", "older_than_sources"} for the binary `cubelang_client` resolves
(``CUBELANG_EXE`` from the environment or ``paths.local.json``, else the sibling checkout's release build; a binary
older than its sources is refused unless ``CB_ALLOW_STALE_VM=1``). The path is written relative to the folder that holds
this repo (``cubelang-cowork/target/release/cubelang.exe``), never with a drive or a user folder, so a committed log
can carry it. ``commit`` is the HEAD of the cubelang worktree the binary sits in (read locally with git; None outside
a checkout); ``older_than_sources`` is True when a source file in that worktree is newer than the binary.

Wired: STANDALONE (the run scripts in ``validation/`` call it into their summaries).
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import time

from . import cubelang_client as cc

__wiring__ = "STANDALONE"

HOME = pathlib.Path(__file__).resolve().parents[3]          # the folder that holds CubbyLLM and its siblings


def _rel(p: pathlib.Path) -> str:
    try:
        return p.resolve().relative_to(HOME).as_posix()
    except ValueError:
        return p.name


def identity(exe: str | None = None) -> dict:
    p = cc.find_cubelang_exe(exe)
    data = p.read_bytes()
    root = next((d for d in (p.parent, *p.parents[1:3]) if (d / "Cargo.toml").is_file()), None)
    commit = None
    if root is not None:
        try:
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    src = cc.newest_source(p)
    return {"path": _rel(p), "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
            "sha256": hashlib.sha256(data).hexdigest(), "commit": commit,
            "older_than_sources": bool(src is not None and src.stat().st_mtime > p.stat().st_mtime + 1)}


__all__ = ["identity"]
