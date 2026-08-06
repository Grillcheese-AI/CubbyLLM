# Reasoning Bridge Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove a real Python→cubelang→Python VSA reasoning roundtrip — plant a `(role → filler)` binding, recover it through cubelang's real `UNBIND`+cosine-cleanup, with wrong-role and absent-role controls — first over JSON (M1), then over a protobuf boundary (M2).

**Architecture:** A thin CubbyLLM-side Python bridge client (`cubbyllm/bridges/`) shells to a freshly-built `cubelang.exe` and drives a `use vsa; recover(reg, role)` program. The boundary is **symbolic** — concepts in (roles/fillers/program), concepts out (`{"result": symbol}`); no raw hypervector ever crosses, because the trunk's grilly algebra and cubelang's Rust VSA are non-interoperable families. M1 reuses `cubelang run --json`; M2 swaps the transport to length-prefixed protobuf-over-stdio, re-running the identical assertions to prove the swap is behaviour-preserving.

**Tech Stack:** CubeLang source (`use vsa; recover`); Python 3.12 (`cubbyllm/bridges`, `subprocess`, `json`; protobuf stubs for M2); Rust (cubelang: `prost` + `prost-build` + `build.rs` for M2 only).

**Spec:** `docs/superpowers/specs/2026-08-05-reasoning-bridge-slice-design.md` (reconciled 2026-08-05: the `recover()` surface the foundation cycle built supersedes the spec's original "add an `unbind` keyword" task — M1 needs **zero Rust change**, and the full roundtrip + both controls are already proven end-to-end via the CLI).

## Global Constraints

- **Symbolic boundary only.** Messages carry roles / fillers / tokens / a CubeLang program and return `(symbol)`. **No raw hypervector crosses.** (Spec §2.)
- **M1 = zero Rust change.** The `recover(reg, role)` surface (`use vsa;`) already exists on cubelang `main` (@ `ea8f41e`) and is tested (`cubelang/tests/asm_vsa.rs::VSA_RECOVER`). Do not add an `unbind` keyword; do not route through `opcode-vsa-rs` (encoder only) or `cubemind`'s Python `vm.py` (dict-lookup cheat).
- **Source on the wire, not bytecode.** A `use`-ing program cannot serialize to `.cubebin` (the format does not carry capability/`use` info — it errors *"run from source instead"*). Consequently the reasoning `.cube` **must not** live in `cubelang/examples/` (that dir is compile-swept and would fail the `.cubebin` write). It lives CubbyLLM-side.
- **Controls are mandatory.** A recovery only means something against a wrong-role control (different filler) and an absent-role control (no symbol). Same standard as the needle test's shuffled control.
- **Commit trailers.** CubbyLLM commits end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav`. **cubelang commits use cubelang's own conventional style with NO CubbyLLM trailers.**
- **Wiring guard.** Every new `cubbyllm/` module declares `__wiring__` (a CI-guard test enforces it). The bridge client is `Wiring.STANDALONE` (an attachment, not the forward path) — follow `cubbyllm/bridges/world_model.py`'s form: module docstring stating the wiring, `from ..core.protocols import Wiring`, `__wiring__ = Wiring.STANDALONE`.
- **Green suites.** cubelang `cargo test` stays green (238+ tests). CubbyLLM `python -m pytest tests -q` stays green — the bridge client is **torch-free** (subprocess + json only), so its tests must not import torch; tests that shell to `cubelang.exe` **skip with a clear message** when the exe is absent (`pytest.skip`), never fail.
- **Exe discovery.** The client locates the exe via env `CUBELANG_EXE`, else the default `<repo-root>/../cubelang/target/release/cubelang.exe`. The fresh exe already exists (rebuilt during recon 2026-08-05).
- **M1 is a valid stopping point.** M1 (Tasks 1–2) satisfies the spec's kill criterion on its own — a working reasoning bridge over JSON. M2 (Tasks 3–5) is transport hardening: a protobuf schema authored to double as a later tonic/gRPC schema. Do M2 only if the persistent-server / gRPC path is wanted soon (YAGNI otherwise).

---

## File Structure

- `cubbyllm/bridges/cubelang_client.py` (**create**, M1) — the transport: locate the exe, shell `cubelang run … --json`, parse the result. `STANDALONE`.
- `cubbyllm/bridges/programs/reasoning_bridge.cube` (**create**, M1) — the proven bind→recover program + its two control functions.
- `tests/bridges/test_cubelang_client.py` (**create**, M1) — client transport unit test.
- `tests/bridges/test_reasoning_bridge.py` (**create**, M1) — the M1 kill-criterion test (recovery + both controls).
- `cubelang/proto/reasoning.proto` (**create**, M2) — `RunRequest`/`RunResult` schema.
- `cubelang/build.rs` (**create**, M2) — `prost-build` codegen.
- `cubelang/Cargo.toml` (**modify**, M2) — add `prost` + `[build-dependencies] prost-build`.
- `cubelang/src/main.rs` (**modify**, M2) — a protobuf-stdio subcommand alongside `run`.
- `cubbyllm/bridges/cubelang_client.py` (**modify**, M2) — add the protobuf transport path.
- `CUBBYLLM_HYPOTHESES.md` (**modify**, Task 6) — update H-B3, note the H-F2 first step.

---

## Task 1: The Python cubelang transport client (M1)

**Files:**
- Create: `cubbyllm/bridges/cubelang_client.py`
- Test: `tests/bridges/test_cubelang_client.py`

**Interfaces:**
- Produces: `find_cubelang_exe() -> pathlib.Path` (raises `CubelangNotFound` if absent); `run_program(program_path, fn="solve", args=None, exe=None, timeout=30.0) -> dict` — shells `cubelang run <program_path> --fn <fn> [--arg <a>]... --json`, returns the parsed JSON dict `{"ok": bool, "result": <symbol|None>, ...}`, raising `CubelangRunError` on `ok: false` or non-zero exit.

- [ ] **Step 1: Write the failing test.** `tests/bridges/test_cubelang_client.py`:

```python
import json, pathlib, pytest
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
```

- [ ] **Step 2: Run it, verify it fails.** `python -m pytest tests/bridges/test_cubelang_client.py -q` → FAIL (`ModuleNotFoundError: cubbyllm.bridges.cubelang_client`).

- [ ] **Step 3: Implement `cubbyllm/bridges/cubelang_client.py`.**

```python
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
```

- [ ] **Step 4: Run the test, verify it passes** (with a built exe) or skips cleanly (without one). `python -m pytest tests/bridges/test_cubelang_client.py -q`.

- [ ] **Step 5: Confirm the guard test still passes** (`__wiring__` present): `python -m pytest tests/test_guards.py -q`.

- [ ] **Step 6: Commit.**

```bash
git add cubbyllm/bridges/cubelang_client.py tests/bridges/test_cubelang_client.py
git commit -m "feat(bridges): subprocess cubelang transport client (M1)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 2: The reasoning example + M1 roundtrip proof (M1)

**Files:**
- Create: `cubbyllm/bridges/programs/reasoning_bridge.cube`
- Test: `tests/bridges/test_reasoning_bridge.py`

**Interfaces:**
- Consumes: `cubelang_client.run_program` (Task 1).
- Produces: the canonical demonstration program the bridge drives; `REASONING_PROGRAM` path constant re-exported from the test module is not required — the test locates the `.cube` via `pathlib` relative to the package.

**Note:** This is the spec's kill criterion (§5) expressed as a test. The program shape is the proven `VSA_RECOVER` fixture from `cubelang/tests/asm_vsa.rs` — do not invent a new one.

- [ ] **Step 1: Author the CubeLang program.** `cubbyllm/bridges/programs/reasoning_bridge.cube`:

```cubelang
use vsa;

# Reasoning-bridge demonstration: plant (role -> filler) bindings, recover by
# role through cubelang's real UNBIND + cosine cleanup. Runs from source
# (a `use`-ing program cannot compile to .cubebin). Proven shape: matches
# cubelang/tests/asm_vsa.rs::VSA_RECOVER.
program ReasoningBridge implements ISolve {
    public function solve(mention: str): str {
        create frame: number;
        bind frame, SUBJECT, "cat";
        bind frame, OBJECT, "mouse";
        return recover(frame, SUBJECT);
    }

    # control: a different role in the same frame recovers a different filler
    public function wrong_role(): str {
        create frame: number;
        bind frame, SUBJECT, "cat";
        bind frame, OBJECT, "mouse";
        return recover(frame, OBJECT);
    }

    # control: an unbound frame recovers no symbol
    public function unbound(): str {
        create frame: number;
        return recover(frame, SUBJECT);
    }
}
```

- [ ] **Step 2: Write the failing test.** `tests/bridges/test_reasoning_bridge.py`:

```python
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
```

- [ ] **Step 3: Run it, verify it fails** first for the right reason (before the `.cube` exists / before Task 1: import or file-missing), then — with Task 1 landed and the exe built — passes. `python -m pytest tests/bridges/test_reasoning_bridge.py -q`.
  Expected with a built exe: 3 passed. Expected without: 3 skipped (clear message).

- [ ] **Step 4: Verify the roundtrip by hand once** (sanity, not a committed step): `<exe> run cubbyllm/bridges/programs/reasoning_bridge.cube --fn solve --arg _ --json` → `{"...","result":"cat"}`. (Confirmed during recon; re-confirm on the committed file.)

- [ ] **Step 5: Commit.**

```bash
git add cubbyllm/bridges/programs/reasoning_bridge.cube tests/bridges/test_reasoning_bridge.py
git commit -m "feat(bridges): M1 reasoning roundtrip proof — recover + 2 controls

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

**■ M1 COMPLETE HERE** — the spec's kill criterion is met: a planted binding is recovered through the full Python→cubelang→Python path over JSON, with both controls separated. Tasks 3–5 (M2) are optional transport hardening.

---

## Task 3: The protobuf schema + prost build (M2, cubelang)

**Files:**
- Create: `cubelang/proto/reasoning.proto`, `cubelang/build.rs`
- Modify: `cubelang/Cargo.toml`

**Interfaces:**
- Produces: generated Rust types `RunRequest { program: String, args: Vec<String>, fn_name: String }` and `RunResult { ok: bool, result: Option<run_result::Result> }` with `oneof Result { Symbol(String), Error(String) }`, available via `include!(concat!(env!("OUT_DIR"), "/reasoning.rs"))` (module `reasoning`).

**Note:** `fn` is a Rust keyword — name the proto field `fn_name` (or `func`). Similarity is intentionally omitted (`recover`/`UNBIND` discards it — spec §8.1); the schema is authored to also serve a later tonic/gRPC upgrade, so keep messages service-agnostic.

- [ ] **Step 1: Write the failing test.** Add to `cubelang/tests/proto_roundtrip.rs`:

```rust
// Requires the `reasoning` module generated by build.rs (Step 3).
use cubelang::reasoning::{RunRequest, RunResult, run_result};
use prost::Message;

#[test]
fn run_request_and_result_roundtrip_through_prost() {
    let req = RunRequest { program: "program P {}".into(), args: vec!["x".into()], fn_name: "solve".into() };
    let mut buf = Vec::new();
    req.encode(&mut buf).unwrap();
    let back = RunRequest::decode(&buf[..]).unwrap();
    assert_eq!(back.fn_name, "solve");
    assert_eq!(back.args, vec!["x".to_string()]);

    let res = RunResult { ok: true, result: Some(run_result::Result::Symbol("cat".into())) };
    let mut b2 = Vec::new();
    res.encode(&mut b2).unwrap();
    match RunResult::decode(&b2[..]).unwrap().result {
        Some(run_result::Result::Symbol(s)) => assert_eq!(s, "cat"),
        other => panic!("expected Symbol(cat), got {:?}", other),
    }
}
```

- [ ] **Step 2: Run it, verify it fails** (`cargo test proto_roundtrip` → unresolved `cubelang::reasoning`).

- [ ] **Step 3: Author `cubelang/proto/reasoning.proto`.**

```protobuf
syntax = "proto3";
package reasoning;

message RunRequest {
  string program = 1;   // CubeLang source (source on the wire, not bytecode)
  repeated string args = 2;
  string fn_name = 3;    // 'fn' is a Rust keyword; field is fn_name
}

message RunResult {
  bool ok = 1;
  oneof result {
    string symbol = 2;   // the recovered symbol (similarity deferred, spec §8.1)
    string error = 3;
  }
}
```

- [ ] **Step 4: Add deps to `cubelang/Cargo.toml`.**

```toml
[dependencies]
prost = "0.13"
# ...existing (blake3, serde_json)...

[build-dependencies]
prost-build = "0.13"
```

- [ ] **Step 5: Author `cubelang/build.rs`.**

```rust
fn main() {
    prost_build::compile_protos(&["proto/reasoning.proto"], &["proto/"]).unwrap();
    println!("cargo:rerun-if-changed=proto/reasoning.proto");
}
```

- [ ] **Step 6: Expose the generated module** in `cubelang/src/lib.rs` (add near the other `pub mod`s):

```rust
pub mod reasoning {
    include!(concat!(env!("OUT_DIR"), "/reasoning.rs"));
}
```

- [ ] **Step 7: Run the test, verify it passes** (`cargo test proto_roundtrip`). Then the full suite: `cargo test` — still green, `opcode_sync` intact.

- [ ] **Step 8: Commit** (cubelang style, NO CubbyLLM trailers):

```bash
git -C ../cubelang add proto/reasoning.proto build.rs Cargo.toml Cargo.lock src/lib.rs tests/proto_roundtrip.rs
git -C ../cubelang commit -m "feat(proto): RunRequest/RunResult schema + prost build (reasoning bridge M2)"
```

---

## Task 4: Protobuf-stdio mode in main.rs (M2, cubelang)

**Files:**
- Modify: `cubelang/src/main.rs`
- Test: `cubelang/tests/proto_stdio.rs` (create)

**Interfaces:**
- Consumes: `reasoning::{RunRequest, RunResult}` (Task 3); the existing compile+run path used by `cmd_run`.
- Produces: a `run-proto` subcommand that reads a single length-prefixed (`u32` big-endian) `RunRequest` from stdin, compiles+runs `program`'s `fn_name` with `args`, and writes a length-prefixed `RunResult` to stdout. `run --json` is untouched.

- [ ] **Step 1: Write the failing integration test.** `cubelang/tests/proto_stdio.rs`:

```rust
use cubelang::reasoning::{RunRequest, RunResult, run_result};
use prost::Message;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

fn call_proto(req: RunRequest) -> RunResult {
    let exe = env!("CARGO_BIN_EXE_cubelang");
    let mut child = Command::new(exe).arg("run-proto")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    let mut buf = Vec::new(); req.encode(&mut buf).unwrap();
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(&(buf.len() as u32).to_be_bytes()).unwrap();
    stdin.write_all(&buf).unwrap();
    drop(stdin);
    let mut out = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut out).unwrap();
    child.wait().unwrap();
    let len = u32::from_be_bytes(out[..4].try_into().unwrap()) as usize;
    RunResult::decode(&out[4..4 + len]).unwrap()
}

#[test]
fn proto_stdio_runs_a_recover_program() {
    let program = r#"
use vsa;
program R implements ISolve {
    public function solve(input: str): str {
        create frame: number;
        bind frame, SUBJECT, "cat";
        return recover(frame, SUBJECT);
    }
}
"#;
    let res = call_proto(RunRequest { program: program.into(), args: vec!["_".into()], fn_name: "solve".into() });
    assert!(res.ok);
    assert!(matches!(res.result, Some(run_result::Result::Symbol(ref s)) if s == "cat"));
}
```

- [ ] **Step 2: Run it, verify it fails** (`cargo test --test proto_stdio` → unknown subcommand / no output).

- [ ] **Step 3: Implement the `run-proto` subcommand in `main.rs`.** Add a match arm next to `"run"` (`main.rs:32`) dispatching to a new `cmd_run_proto()` that: reads the u32-prefixed `RunRequest` from stdin; compiles `req.program` (from source, via the same `compiler::compile`/loader path `cmd_run` uses — **note: no `.cubebin`, source only**, which is correct since `use`-ing programs can't serialize); builds `args` as `Value::Str`; calls the VM's `call(program, fn_name, args)`; maps `Ok(Value::Str(s))`/other → `RunResult{ok:true, Symbol(...)}` (stringify non-Str via the same shape as `value_to_json`), and any compile/run error → `RunResult{ok:false, Error(e)}`; writes the u32-prefixed encoded `RunResult` to stdout. Reuse `cmd_run`'s program-loading and call logic — factor the shared part into a helper if it reduces duplication, but do not alter `cmd_run`'s `--json` behaviour.

- [ ] **Step 4: Run the test, verify it passes** (`cargo test --test proto_stdio`). Then full `cargo test` — green, and re-confirm `run --json` still works via the CLI on `reasoning_bridge.cube`.

- [ ] **Step 5: Commit** (cubelang style, NO CubbyLLM trailers):

```bash
git -C ../cubelang add src/main.rs tests/proto_stdio.rs
git -C ../cubelang commit -m "feat(cli): run-proto length-prefixed protobuf stdio mode (reasoning bridge M2)"
```

---

## Task 5: Python protobuf transport + re-run M1 assertions (M2)

**Files:**
- Modify: `cubbyllm/bridges/cubelang_client.py`
- Create: `cubbyllm/bridges/_reasoning_pb2.py` (generated), `tests/bridges/test_reasoning_bridge_proto.py`
- Modify: `pyproject.toml` (dev dep: `grpcio-tools` for `protoc`, or `protobuf` runtime)

**Interfaces:**
- Consumes: `cubelang/proto/reasoning.proto` (Task 3), the `run-proto` mode (Task 4).
- Produces: `cubelang_client.run_program_proto(program_source, fn="solve", args=None, exe=None, timeout=30.0) -> dict` returning `{"ok": bool, "result": <symbol|None>}`, shape-compatible with `run_program`'s return so the SAME assertions apply.

- [ ] **Step 1: Generate the Python stubs.** From the proto: `python -m grpc_tools.protoc -I cubelang/proto --python_out=cubbyllm/bridges cubelang/proto/reasoning.proto` → `cubbyllm/bridges/reasoning_pb2.py`. (Add `grpcio-tools` to `pyproject.toml` `[project.optional-dependencies] dev`; document the regen command in the client docstring. Pin the generated file in-tree so tests don't need protoc at runtime.)

- [ ] **Step 2: Write the failing test.** `tests/bridges/test_reasoning_bridge_proto.py` — the SAME three assertions as Task 2, but via `run_program_proto`, plus a cross-check that JSON and protobuf agree:

```python
import pathlib, pytest
from cubbyllm.bridges import cubelang_client as cc

PROG = (pathlib.Path(cc.__file__).parent / "programs" / "reasoning_bridge.cube").read_text()

def _skip_unless_ready():
    try: cc.find_cubelang_exe()
    except cc.CubelangNotFound as e: pytest.skip(f"cubelang.exe not found ({e})")
    pytest.importorskip("cubbyllm.bridges.reasoning_pb2")

def test_m2_matches_m1_over_protobuf():
    _skip_unless_ready()
    assert cc.run_program_proto(PROG, fn="solve", args=["_"])["result"] == "cat"
    assert cc.run_program_proto(PROG, fn="wrong_role")["result"] == "mouse"
    assert cc.run_program_proto(PROG, fn="unbound")["result"] is None
```

- [ ] **Step 3: Run it, verify it fails** (`run_program_proto` undefined).

- [ ] **Step 4: Implement `run_program_proto`** in `cubelang_client.py`: build a `RunRequest` (program source, args, fn_name), spawn `<exe> run-proto`, write the `u32`-length-prefixed encoded request to stdin, read the length-prefixed `RunResult` from stdout, decode; return `{"ok": res.ok, "result": res.symbol if res.HasField("symbol") else None}`, raising `CubelangRunError` on `error`. Keep `import`s of the generated `reasoning_pb2` lazy (inside the function) so top-level `import cubbyllm` stays dependency-light.

- [ ] **Step 5: Run the test, verify it passes** (with exe + stubs) or skips. Then the full CubbyLLM suite `python -m pytest tests -q` — green.

- [ ] **Step 6: Commit.**

```bash
git add cubbyllm/bridges/cubelang_client.py cubbyllm/bridges/reasoning_pb2.py tests/bridges/test_reasoning_bridge_proto.py pyproject.toml
git commit -m "feat(bridges): protobuf transport, re-run M1 assertions (M2)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 6: Update H-B3 + note the H-F2 first step (doc)

**Files:**
- Modify: `CUBBYLLM_HYPOTHESES.md`

- [ ] **Step 1: Update H-B3.** Add a dated Validation-result line: real unbind exists in cubelang's Rust VM (`op::UNBIND` + cosine cleanup), is reachable from a CubeLang program via `use vsa; recover(reg, role)` (foundation cycle), and is now exposed to Python over a **symbolic** boundary — JSON (M1) and, if built, protobuf (M2) — proven with wrong-role and absent-role controls (`tests/bridges/`). Correct any lingering "no real unbind exists" phrasing.

- [ ] **Step 2: Note the H-F2 first step.** The reasoning bridge is the first concrete, tested step toward the Cubby↔CubeMind bridge (H-F2); the full bidirectional context/specialist-handle bridge (`cubbyllm/bridges/world_model.py`'s `WorldModelBridge` Protocol) remains the later cycle.

- [ ] **Step 3: Commit.**

```bash
git add CUBBYLLM_HYPOTHESES.md
git commit -m "docs(hypotheses): H-B3 real unbind exposed via symbolic bridge; H-F2 first step

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Self-review notes

- **Spec coverage:** M1 (Tasks 1–2) = spec §4 M1 + §5 kill criterion. M2 (Tasks 3–5) = spec §4 M2 (proto §3.2, stdio §3.2, Python §3.3). §7 result-updates-H-B3 = Task 6. Deferred §6 items (full H-F2 bridge, grilly-in-Rust, opcode-vsa-rs PyO3, raw-hypervector crossing, cubelang drift cleanup) are **out of scope** — not tasked, by design.
- **Type consistency:** `run_program`/`run_program_proto` both return `{"ok", "result"}` so Task 2's assertions are reused verbatim in Task 5. `fn_name` (not `fn`) throughout the proto/Rust to avoid the Rust keyword.
- **Cross-repo:** Tasks 1–2, 5–6 are CubbyLLM (CubbyLLM trailers); Tasks 3–4 are cubelang (cubelang style, no trailers). The plan lives in CubbyLLM.
- **Placement constraint honored:** the `.cube` lives in `cubbyllm/bridges/programs/`, never `cubelang/examples/` (which is compile-swept and would fail the `.cubebin` write on a `use`-ing program).

---

## M2 Extension — Features A + B (added 2026-08-05, from design investigation)

User chose "full expanded M2." Both features are **Size S** (wiring, not redesign). **Execution order: Task 7 → 8 → 9 → 6 (doc) → whole-branch review.** Tasks 7-8 are cubelang (both touch `src/main.rs`, so sequential); Task 9 is CubbyLLM and needs the exe rebuilt with 7+8. Line numbers are ~approx guides from the investigation (may drift).

### Task 7: Feature A — similarity-surfacing (cubelang)
**Files:** `src/vm/engine.rs`, `proto/reasoning.proto`, `src/main.rs`, `tests/proto_stdio.rs`.
**Approach — ADDITIVE. Do NOT change `recover()`'s bare-string (`Value::Str`) return** — that would break `tests/asm_vsa.rs` (`b1`/`b3`/`b5`) and M1's `result == "cat"`. Instead surface similarity as a side-channel:
- `engine.rs`: add `pub last_recover_similarity: Option<f64>` to `VM` (struct ~132-175), init `None` in `VM::new()` (~184-216), and SET it inside the shared chokepoint `vsa_unbind_cleanup()` (~1288-1310, where the winning `cosine_sim` is computed ~1303) — one point covers both `op::UNBIND` and `recover()`.
- `proto/reasoning.proto`: add `optional double similarity = 4;` to `RunResult`, **OUTSIDE the `oneof`** (symbol/error untouched → existing Python `result.symbol` unaffected).
- `main.rs`: `run_request_result` (~629-664) reads `vm.last_recover_similarity` after the call and sets `RunResult.similarity`; `cmd_run`'s JSON object (~481-486) gets the same `similarity` key for parity.
- `tests/proto_stdio.rs`: assert `solve`/`wrong_role` carry a similarity that is present + high; `unbound` has none.
- Document the "last recover this run" semantics — correct here because `run_request_result` builds a fresh `VM::new()` per request and the reasoning program does one `recover` per fn.

### Task 8: Feature B — verify-before-execute (cubelang)
**Files:** `src/main.rs` (+ a test). Key finding: `compile_strict`/`compile_ast_strict` (compiler.rs ~1965-1989) run on SOURCE — they do NOT need `.cubebin`. The reasoning program passes strict cleanly; the gap is only that the from-source entry points don't CALL strict.
**Approach — wire the existing engine. Do NOT touch `.cubebin`/the wire proto** (the `.cubebin` capability-carrying upgrade is separate, larger (M) work, OUT OF SCOPE):
- `run_request_result` (main.rs ~630): swap `compiler::compile(&req.program)` → `compiler::compile_strict(&req.program)` (~1 line; the `Err(String)` already flows to the existing `Error` arm → surfaces over the wire, no schema change). Protobuf reasoning path becomes always-verified.
- `cmd_run`: add a `--strict` flag (mirror `cmd_compile`'s existing 88-94/123-130 pattern) branching `compile_ast`/`compile_ast_strict`, so the JSON path can verify too.
- Optional: teach `cmd_check` (~323-357, currently parse-only) to call `compile`/`compile_strict` for a standalone "will it run safely" preflight (~10-15 lines).
- One-line pre-existing doc-drift fix if you're in the function: `compile_strict`'s doc-comment (~1954-1957) claims `for` is strict-rejected; the code (`strict_check_stmt` ~1000-1001) exempts it.
**Risk (intended — document it):** strict rejects the trace-only ext ops (`infer`/`score`/`analogy`/`discover`/`predict`/`debate`, currently silent no-ops per engine.rs ~935-940) — a future reasoning program reaching for them fails LOUDLY instead of silently no-op'ing. That is the point of verify-before-execute.

### Task 9: Feature A — Python side (CubbyLLM)
**Files:** `cubbyllm/bridges/cubelang_client.py`, `cubbyllm/bridges/reasoning_pb2.py` (regen), `tests/bridges/test_reasoning_bridge_proto.py`.
**Approach:**
- Rebuild the cubelang release exe (now carries A+B from Tasks 7-8).
- Regenerate `reasoning_pb2.py` (regen command in the module docstring; **RE-APPEND the `__wiring__` tail** per its comment).
- `run_program_proto` (~line 158): also return `"similarity"` from `RunResult` (`res.similarity if res.HasField("similarity") else None`). `run_program` (JSON) returns `similarity` too (cmd_run's JSON now carries it). Recommend `run_program`/`run_program_proto` also request `--strict` verification of the reasoning path where applicable.
- Tests: strengthen the controls with similarity separation — `solve`→cat HIGH sim, `wrong_role`→mouse HIGH sim, `unbound`→None with no/low sim.
- CubbyLLM trailers (Opus 5 + Claude-Session).

### Task 6 (doc) — executes AFTER 7-9
Reflect the FINAL bridge in H-B3: real unbind exposed to Python over a symbolic boundary (JSON + protobuf), now carrying **confidence (similarity)** AND **verify-before-execute** (strict compile from source). H-F2 first-step note unchanged.
