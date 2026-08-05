"""cubelang_client — subprocess transport to the cubelang reasoning VM.

Wired: STANDALONE — a bridge, not the forward path.

Shells to a freshly-built ``cubelang.exe`` and drives a CubeLang program via
``run --json``, parsing the symbolic result. The boundary is symbolic by design
(spec 2026-08-05): concepts in (a program + args), concepts out
(``{"result": symbol}``) — no raw hypervector crosses, because the trunk's
grilly algebra and cubelang's Rust VSA are different families. Supersedes, for
this path, cubemind's regex-scraping ``model/cubby/cubelang_bridge.py``.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

from ..core.protocols import Wiring

__wiring__ = Wiring.STANDALONE


class CubelangNotFound(RuntimeError):
    """The cubelang executable could not be located."""


class CubelangRunError(RuntimeError):
    """cubelang reported an error (ok: false) or exited non-zero."""


def find_cubelang_exe(explicit: str | None = None) -> pathlib.Path:
    """Locate cubelang.exe: explicit arg, then $CUBELANG_EXE, then the sibling
    repo's release build. Raises CubelangNotFound if none exist."""
    candidates = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    env = os.environ.get("CUBELANG_EXE")
    if env:
        candidates.append(pathlib.Path(env))
    # sibling repo: <...>/CubbyLLM/../cubelang/target/release/cubelang(.exe)
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]  # cubbyllm/bridges/cubelang_client.py -> CubbyLLM/
    for name in ("cubelang.exe", "cubelang"):
        candidates.append(repo_root.parent / "cubelang" / "target" / "release" / name)
    for c in candidates:
        if c.is_file():
            return c
    raise CubelangNotFound(
        f"set $CUBELANG_EXE or build the cubelang repo; tried: {[str(c) for c in candidates]}"
    )


def run_program(
    program_path: str,
    fn: str = "solve",
    args: list[str] | None = None,
    exe: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Run a CubeLang program's function via `cubelang run … --json`; return the
    parsed JSON. Raises CubelangRunError on ok:false or a non-zero exit."""
    exe_path = find_cubelang_exe(exe)
    cmd = [str(exe_path), "run", program_path, "--fn", fn, "--json"]
    for a in args or []:
        cmd += ["--arg", a]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise CubelangRunError(f"cubelang exited {proc.returncode}: {proc.stderr.strip()}")
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise CubelangRunError(f"unparseable cubelang output: {proc.stdout!r} / {proc.stderr!r}") from e
    if not out.get("ok", False):
        raise CubelangRunError(f"cubelang error: {out.get('error', out)}")
    return out
