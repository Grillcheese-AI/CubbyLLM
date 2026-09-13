"""cubelang_client — subprocess transport to the cubelang reasoning VM.

Wired: STANDALONE — a bridge, not the forward path.

Shells to a freshly-built ``cubelang.exe`` and drives a CubeLang program via
``run --json``, parsing the symbolic result. The boundary is symbolic by design
(spec 2026-08-05): concepts in (a program + args), concepts out
(``{"result": symbol}``) — no raw hypervector crosses, because the trunk's
grilly algebra and cubelang's Rust VSA are different families. Supersedes, for
this path, cubemind's regex-scraping ``model/cubby/cubelang_bridge.py``.

``run_program_proto`` is a second, wire-compatible transport (M2): instead of
a JSON line on ``run --json``'s stdout, it speaks ``run-proto`` — a
length-prefixed ``RunRequest``/``RunResult`` protobuf pair over stdin/stdout
(``cubelang/proto/reasoning.proto``). Same return shape as ``run_program``
(``{"ok": bool, "result": <symbol|None>, "similarity": <float|None>}``), so
the same assertions apply to either transport. ``similarity`` (Task 9,
Feature A) is the cosine similarity of the winning ``recover()``/``UNBIND``
match — present and high for a real recovery, ``None`` (not ``0.0``) when
nothing was bound. Regenerate the pinned stub after changing the proto::

    python -m grpc_tools.protoc -I <cubelang>/proto \\
        --python_out=cubbyllm/bridges <cubelang>/proto/reasoning.proto

No manual edit is needed afterwards — ``tests/test_guards.py`` excludes
``*_pb2.py`` modules from the ``__wiring__`` guard (they're protoc output,
fully overwritten on every regen, never a forward path), so the old
"re-append the ``__wiring__`` tail by hand" step is gone.

``run_program``'s ``--json`` path also takes a ``strict`` kwarg (Feature B,
verify-before-execute; default ``True``) that passes ``--strict`` through to
``cubelang run``, rejecting non-executing constructs at compile time instead
of silently no-op'ing them. ``run_program_proto``'s ``run-proto`` transport
has no equivalent flag: it is unconditionally strict server-side, so every
protobuf request is already verified before it executes.
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
            _warn_if_stale(c)
            return c
    raise CubelangNotFound(
        f"set $CUBELANG_EXE or build the cubelang repo; tried: {[str(c) for c in candidates]}"
    )


def newest_source(exe: pathlib.Path) -> pathlib.Path | None:
    """The most recently modified `.rs` under the sibling repo's `src/`, if the exe
    lives in a cubelang checkout (its root, or its target/<profile>/)."""
    for parent in (exe.parent, *exe.parents[1:3]):
        src = parent / "src"
        if (parent / "Cargo.toml").is_file() and src.is_dir():
            files = list(src.rglob("*.rs"))
            return max(files, key=lambda p: p.stat().st_mtime) if files else None
    return None


def _warn_if_stale(exe: pathlib.Path) -> None:
    """2026-09-12: a `cubelang.exe` from May sat at the checkout's root beside sources
    from August; nothing here resolved to it (target/release is what is searched),
    but a hand-set $CUBELANG_EXE or `--exe` could. A binary older than the newest
    source is measured with a VM the sources no longer describe -- say so, once."""
    try:
        src = newest_source(exe)
        if src is not None and src.stat().st_mtime > exe.stat().st_mtime + 1:
            import sys, time
            fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
            print(f"[cubelang_client] WARNING: {exe} ({fmt(exe.stat().st_mtime)}) is older than {src} "
                  f"({fmt(src.stat().st_mtime)}) -- rebuild with `cargo build --release`", file=sys.stderr)
    except OSError:
        pass


def run_program(
    program_path: str,
    fn: str = "solve",
    args: list[str] | None = None,
    exe: str | None = None,
    timeout: float = 30.0,
    strict: bool = True,
    knowledge: str | None = None,
) -> dict:
    """Run a CubeLang program's function via `cubelang run … --json`; return the
    parsed JSON (plus a normalized `"similarity"` key -- see below). Raises
    CubelangRunError on ok:false or a non-zero exit.

    `knowledge` (2026-09-11, plan_verify) passes `--knowledge <jsonl>` so the
    VM's QUERY has a store to ground against; without it every QUERY abstains
    (empty chunk array). `run-proto` has no equivalent field yet.

    `strict` (Task 8/9, verify-before-execute) passes `--strict` to `cubelang
    run`, so non-executing constructs (trace-only ext ops, `match`, ...) fail
    loudly at compile time instead of silently compiling to no-ops. Defaults
    to True: the reasoning bridge's program always passes strict cleanly, so
    verify-before-execute should be the default a caller has to opt out of,
    not opt into. `run_program_proto`'s `run-proto` transport has no
    equivalent flag -- it is unconditionally strict server-side (cubelang
    Task 8), so there's nothing to thread on that side.
    """
    exe_path = find_cubelang_exe(exe)
    cmd = [str(exe_path), "run", program_path, "--fn", fn, "--json"]
    if strict:
        cmd.append("--strict")
    if knowledge:
        cmd += ["--knowledge", str(knowledge)]
    for a in args or []:
        cmd += ["--arg", a]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise CubelangRunError(f"cubelang timed out after {timeout}s: {cmd}") from e
    if proc.returncode != 0 and not proc.stdout.strip():
        raise CubelangRunError(f"cubelang exited {proc.returncode}: {proc.stderr.strip()}")
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise CubelangRunError(f"unparseable cubelang output: {proc.stdout!r} / {proc.stderr!r}") from e
    if not out.get("ok", False):
        raise CubelangRunError(f"cubelang error: {out.get('error', out)}")
    # Task 9 (Feature A, similarity-surfacing): cmd_run's `--json` carries
    # `"similarity"` on a normal Ok/Return result, but the shape omits the
    # key entirely on other ok:true paths (e.g. a suspended run's JSON has
    # no recover() outcome to report). Normalize so callers can always
    # index `out["similarity"]` instead of needing their own `.get(...)`.
    out["similarity"] = out.get("similarity")
    return out


def _json_loads_lenient(s: str):
    """Decode a JSON-encoded Value from the wire; a bare (non-JSON) string is
    returned as-is so an older binary's plain-text question still reads."""
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def resume_program_proto(
    program_source: str,
    fn: str = "solve",
    answers: list | None = None,
    args: list[str] | None = None,
    exe: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Resume a program that ASKed, by re-execution (2026-08-30): the same
    request goes back with `answers` — one per ASK, in order, each EXACTLY
    one of the `candidates` the suspension offered (the VM rejects an
    invented answer: "chosen, not invented"). `run-proto` is one-shot, so
    the program re-runs from scratch and consumes the answers at each ASK;
    deterministic programs reach the same ASK, so no VM state crosses the
    wire. Returns the same dict shape as `run_program_proto` — including
    another `suspended: True` if the program asks again."""
    return run_program_proto(program_source, fn=fn, args=args, exe=exe, timeout=timeout,
                             answers=list(answers or []))


def run_program_proto(
    program_source: str,
    fn: str = "solve",
    args: list[str] | None = None,
    exe: str | None = None,
    timeout: float = 30.0,
    answers: list | None = None,
    knowledge_path: str | None = None,
) -> dict:
    """Run CubeLang source's function via `cubelang run-proto`'s stdio
    transport: a u32-big-endian-length-prefixed `RunRequest` written to
    stdin, a length-prefixed `RunResult` read back from stdout (both encoded
    per `cubelang/proto/reasoning.proto`). Unlike `run_program`, this takes
    program SOURCE directly (no filesystem path — `run-proto` compiles
    in-memory, matching the proto's `program` field doc).

    Returns `{"ok": bool, "result": <symbol|None>}` — the `None` case is the
    oneof left unset (e.g. `recover()` finding no bound filler), not a
    stringified "null". Raises CubelangRunError when the decoded
    `RunResult.ok` is false, or when the process didn't produce a decodable
    result at all (spawn failure, timeout, truncated/undecodable output).

    Gates on the *parsed* `RunResult.ok`, never on the subprocess exit code:
    `run-proto` intentionally exits 0 even when the run itself errored.
    """
    from . import reasoning_pb2  # lazy: keep `import cubbyllm` protobuf-free

    exe_path = find_cubelang_exe(exe)
    request = reasoning_pb2.RunRequest(
        program=program_source, args=list(args or []), fn_name=fn,
        answers=[json.dumps(a) for a in (answers or [])],
        knowledge_path=str(knowledge_path or ""),
    )
    payload = request.SerializeToString()
    framed_request = len(payload).to_bytes(4, "big") + payload

    cmd = [str(exe_path), "run-proto"]
    try:
        proc = subprocess.run(
            cmd, input=framed_request, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise CubelangRunError(f"cubelang timed out after {timeout}s: {cmd}") from e
    except OSError as e:
        raise CubelangRunError(f"cubelang failed to spawn: {cmd}: {e}") from e

    out = proc.stdout
    if len(out) < 4:
        raise CubelangRunError(
            f"cubelang run-proto produced no length-prefixed RunResult "
            f"(exit {proc.returncode}): stdout={out!r} stderr={proc.stderr!r}"
        )
    result_len = int.from_bytes(out[:4], "big")
    body = out[4 : 4 + result_len]
    if len(body) != result_len:
        raise CubelangRunError(
            f"cubelang run-proto RunResult truncated: expected {result_len} "
            f"bytes, got {len(body)} (exit {proc.returncode}): stderr={proc.stderr!r}"
        )

    return _decode_run_result(body)


def _decode_run_result(body: bytes) -> dict:
    """`RunResult` bytes -> the dict shape `run_program_proto` documents. Shared
    by the one-shot transport and `CubelangSession` so both agree byte-for-byte."""
    from . import reasoning_pb2  # lazy: keep `import cubbyllm` protobuf-free

    result = reasoning_pb2.RunResult()
    try:
        result.ParseFromString(body)
    except Exception as e:  # google.protobuf.message.DecodeError
        raise CubelangRunError(f"undecodable cubelang RunResult: {e}") from e

    which = result.WhichOneof("result")
    if not result.ok:
        err = result.error if which == "error" else "cubelang run-proto reported ok:false"
        raise CubelangRunError(f"cubelang error: {err}")
    if which == "suspended":
        # The third outcome (2026-08-30): the program grounded several
        # candidates and is asking. Not an error, not a result. Candidates
        # and the question are JSON-encoded Values; decode them so the
        # caller sees `1969`, not `"1969"`. Resume with
        # `resume_program_proto(..., answers=[<chosen candidate>])`.
        s = result.suspended
        return {
            "ok": True, "result": None, "similarity": None, "suspended": True,
            "question": _json_loads_lenient(s.question),
            "candidates": [_json_loads_lenient(c) for c in s.candidates],
            "program": s.program, "function": s.function,
        }
    return {
        "ok": result.ok,
        "suspended": False,
        "result": result.symbol if which == "symbol" else None,
        # Task 7 (cubelang): `similarity` is `optional double`, OUTSIDE the
        # `result` oneof -- a side-channel confidence score for `symbol`,
        # not an alternative to it. `HasField` (proto3 explicit presence,
        # not a truthiness/zero check) distinguishes "no winning match"
        # (unset -> None) from a real match that happened to score 0.0.
        "similarity": result.similarity if result.HasField("similarity") else None,
    }


class CubelangSession:
    """A RESIDENT `cubelang run-proto` process: one spawn, many requests.

    2026-09-11. `run_program_proto` spawns a process per call, and on the
    plan-disposer harvest 181 spawns for 181 membership questions cost more
    wall time than the 161 walks they prevented (exp_m3 lookup_vp: 59.4 ->
    65.7 ms/question). `run-proto` now serves until stdin closes, so this
    keeps the pipe open and streams length-prefixed requests down it, reading
    one length-prefixed `RunResult` back per request, in order.

    Same wire, same decoder (`_decode_run_result`), same semantics: every
    request still runs on a fresh VM inside the process, so determinism and
    verify-before-execute are untouched -- only the process boundary is
    amortized. `knowledge_path` rides on the request; the process caches the
    parsed store by (path, mtime), so a vocabulary is parsed once per
    session, not once per question.

    Use as a context manager, or call `close()`. A dead process (any read
    that comes back short) raises `CubelangRunError` with whatever the
    process wrote to stderr; the session is then unusable and a new one
    must be opened -- it never silently respawns, because a respawn would
    also silently drop the knowledge cache and the caller's assumptions
    about it.
    """

    def __init__(self, exe: str | None = None, timeout: float = 30.0) -> None:
        import subprocess as _sp
        self.exe = str(find_cubelang_exe(exe))
        self.timeout = timeout
        self._proc = _sp.Popen([self.exe, "run-proto"], stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE)
        self.n_requests = 0

    # -- transport ---------------------------------------------------------
    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                err = b""
                try:
                    self._proc.kill()
                    err = self._proc.stderr.read() or b""
                except Exception:
                    pass
                raise CubelangRunError(
                    f"cubelang run-proto closed the pipe after {len(buf)} of {n} bytes "
                    f"(exit {self._proc.poll()}): stderr={err[-2000:]!r}")
            buf += chunk
        return buf

    def run(self, program_source: str, fn: str = "solve", args: list[str] | None = None,
            answers: list | None = None, knowledge_path: str | None = None) -> dict:
        """Same contract as `run_program_proto` (dict shape, errors, suspension)."""
        import threading
        from . import reasoning_pb2  # lazy: keep `import cubbyllm` protobuf-free

        if self._proc.poll() is not None:
            raise CubelangRunError(f"cubelang run-proto session is dead (exit {self._proc.returncode})")
        request = reasoning_pb2.RunRequest(
            program=program_source, args=list(args or []), fn_name=fn,
            answers=[json.dumps(a) for a in (answers or [])],
            knowledge_path=str(knowledge_path or ""),
        )
        payload = request.SerializeToString()
        timer = threading.Timer(self.timeout, self._proc.kill) if self.timeout else None
        if timer:
            timer.start()
        try:
            self._proc.stdin.write(len(payload).to_bytes(4, "big") + payload)
            self._proc.stdin.flush()
            length = int.from_bytes(self._read_exact(4), "big")
            body = self._read_exact(length)
        finally:
            if timer:
                timer.cancel()
        self.n_requests += 1
        return _decode_run_result(body)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        p = self._proc
        if p.poll() is None:
            try:
                p.stdin.close()          # clean EOF at a frame boundary -> the loop exits 0
                p.wait(timeout=5)
            except Exception:
                p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
