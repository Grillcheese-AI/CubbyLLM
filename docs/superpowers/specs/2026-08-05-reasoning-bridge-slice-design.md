# Reasoning bridge — a vertical slice: real unbind over a symbolic protobuf boundary

**Date:** 2026-08-05 · **Status:** design, approved to plan · **Hypotheses:** H-B3 (VM-side real unbind) + H-F2 (Cubby↔CubeMind bridge) — first concrete step on both

## 1. Why this exists

The reasoning pillar ("gen 2 hybrid" = LM + a symbolic VM) needs a *real* Python→VM
path where the VM does genuine VSA reasoning and hands a result back. Two recons
(2026-08-05) reshaped what that means:

- **The real unbind already exists — in `cubelang`'s Rust VM.** `op::BIND_ROLE`
  (`engine.rs:639`) does real bind+bundle; `op::UNBIND` (`engine.rs:855`) does real
  unbind + cosine nearest-neighbour cleanup against the symbol table, returning
  `(symbol, similarity)`. This **corrects H-B3's premise**: it is not true that "no
  real unbind exists anywhere" — it is built, in cubelang, in Rust. The Python
  `cubemind/reasoning/vm.py::_unbind_role` is the dict-lookup cheat; `opcode-vsa-rs`
  is only an *encoder* (no executor). So the reasoning engine is **cubelang**.
- **The two VSA algebras are non-interoperable, so raw vectors must NOT cross the
  boundary.** grilly `BlockCodeOps` (the decided H-B5 trunk substrate: sparse block
  codes, k=80×l=128=D=10240, per-block circular convolution, PCG64) and the Rust VSA
  (`cubelang`/`opcode-vsa-rs`: dense bipolar `Vec<i8>`, Hadamard bind, D≈4096/8192,
  `SmallRng`) are different families end to end. A grilly-bound vector cannot be
  unbound in Rust (silently returns garbage, not an error).

The resolution (approved): the boundary is **symbolic**, not vectorial. The trunk
hands the VM *concepts* (roles, fillers, tokens, a CubeLang program); the VM does VSA
in its own algebra and returns *concepts* (`symbol, similarity`). grilly stays the
trunk substrate (H-B5 intact); cubelang stays the reasoning VSA; they never have to
agree on a bit layout because the contract is semantic. This is also the right
abstraction — an LM should hand its reasoner resolved concepts, not noisy
hypervectors — and it matches the codebase's own plan (`cubelang/src/main.rs:568`:
*"protobuf is a later transport swap"*).

This slice proves **one** thing end to end: a real Python→cubelang→Python reasoning
roundtrip, with a real unbind, over a symbolic protobuf boundary — reusing the one
piece already wired (cubelang's `run`). Everything larger (the full bidirectional
H-F2 bridge, grilly-in-Rust) is deferred (§6).

## 2. Settled design decisions (approved 2026-08-05)

- **`cubelang` is the canonical reasoning VM.** Do not rebuild its unbind; do not
  route through `opcode-vsa-rs` (encoder only) or the Python `vm.py` (cheat).
- **grilly stays the trunk substrate (H-B5).** No interop with the Rust algebra is
  required or attempted.
- **The boundary is symbolic.** Messages carry roles / fillers / tokens / a CubeLang
  program and return `(symbol, similarity)`. **No raw hypervector ever crosses.**
- **Transport: protobuf-over-stdio on the existing `cubelang run` subprocess** — the
  code's own stated direction. One-shot, isolated, not in the torch graph (matches
  the bridge's existing shape).
- **De-risk in two milestones (§4): JSON first, then protobuf.**
- **UPDATE 2026-08-05 (post-foundation-cycle) — the M1 unbind surface is already
  built; no compiler patch needed.** The original recon budgeted a 5-file patch to
  add an `unbind` keyword because `unbind` had no surface syntax. The CubeLang
  foundation cycle (merged to cubelang `main` @ `ea8f41e`) since built
  **`recover(reg, role)`** — a `use vsa;` registry helper that runs the real
  `BIND_ROLE`/`UNBIND`+cosine-cleanup internally and returns the recovered
  `Value::Str` symbol. That *is* the intended surface, and it matches the owner's
  decision that generated programs call `recover`, never a raw `unbind` (raw opcodes
  are `asm`-only, for building helpers). **Proven end-to-end via the CLI (recon
  2026-08-05):** `use vsa; … bind frame, SUBJECT, "cat"; … return recover(frame,
  SUBJECT);` run through `cubelang run --fn solve --json` returns `{"result":"cat"}`;
  the wrong-role control returns `{"result":"mouse"}`; the unbound control returns
  `{"result":null}`. `value_to_json` already renders `Value::Str` as a bare string.
  **M1 therefore needs zero Rust change** — it collapses to authoring the example +
  the Python client.
- **Note (non-blocking, settles §8.3):** a `use`-ing program cannot be compiled to
  `.cubebin` (the binary format does not yet carry capability/`use` info — it errors
  *"run from source instead"*). M1 runs from **source** via `run --json`, so this
  doesn't block the slice, and it decides the wire format: **source, not bytecode.**
  It also means the reasoning example cannot live in cubelang's compile-swept
  `examples/` dir — it needs a run-from-source home (plan §placement). Serializing
  capability info into `.cubebin` is deferred (only needed to cache/verify a
  reasoning program as a binary artifact).

## 3. Architecture

Three components; the slice spans two repos (`cubelang` Rust + a Python client) — the
spec lives in CubbyLLM because it is CubbyLLM's reasoning pillar.

### 3.1 The CubeLang program (in `cubelang`)
A small CubeLang source program that binds a `(role, filler)` into a register, then
unbinds by role and returns the cleaned-up symbol. It exercises `BIND_ROLE` +
`UNBIND` inside the Rust VM. (Plan must confirm the CubeLang *surface syntax* that
compiles to those opcodes, or whether we hand-assemble bytecode — see §8.)

### 3.2 The transport (in `cubelang`)
- **M1:** none — reuse `cubelang run <prog> --json`, parse stdout JSON.
- **M2:** a minimal `.proto` (`RunRequest { program, args, fn }` →
  `RunResult { ok, sym{symbol, similarity} | error }`), a `prost` dep + `build.rs`,
  and a protobuf stdio mode in `main.rs` that reads a length-prefixed `RunRequest`
  on stdin and writes a `RunResult` on stdout. The `.proto` is authored to also be
  the schema for a later tonic/gRPC upgrade, so no rewrite is needed if call volume
  ever justifies a persistent server.

### 3.3 The Python reasoning client (CubbyLLM or cubemind side — §8)
Serialises the request, shells to a **freshly-rebuilt** `cubelang.exe` (the root
binary is 6 weeks stale — missing `validate`/`--strict`; rebuild + repoint), and
parses the response. M1: build request as CubeLang source + `--json`, parse JSON.
M2: serialise a `RunRequest` protobuf, read a `RunResult` protobuf. Replaces the
regex-scraping in `cubemind/model/cubby/cubelang_bridge.py` for this path.

## 4. Milestones

### M1 — prove the roundtrip over JSON (zero Rust code change — confirmed 2026-08-05)
1. Rebuild `cubelang` (`cargo build --release`); repoint the client at the fresh exe.
   *(Done during recon — the fresh exe is at `cubelang/target/release/cubelang.exe`.)*
2. Author the bind→recover CubeLang example (§3.1) using `use vsa; recover(reg, role)`
   — the surface the foundation cycle built (no `unbind` keyword). Proven shape:
   `tests/asm_vsa.rs::VSA_RECOVER`.
3. Python client shells to the fresh exe, drives the example via
   `run --fn solve --json`, and parses the recovered symbol from `{"result": <sym>}`
   (symbol-only; `recover`/`UNBIND` discards the similarity float — surfacing
   confidence is a deferred larger change, symbol suffices for M1).
4. **Assert recovery + controls:** `recover(frame, SUBJECT)` returns the planted
   `"cat"`; the wrong-role control `recover(frame, OBJECT)` returns a *different*
   symbol (`"mouse"`); the absent-role control (unbound frame) returns `null`. (The
   controls are what make a hit mean something — same standard as the needle test's
   shuffled control. All three already pass via the CLI, recon 2026-08-05.)

### M2 — swap the transport to protobuf
5. Author the `.proto`; add `prost` + `build.rs`; add the protobuf-stdio mode to
   `main.rs` (leave `--json` intact).
6. Swap the Python client to protobuf; re-run the **same** M1 assertions unchanged.
   The result must match M1 to prove the transport swap is behaviour-preserving.

## 5. Success / kill criterion

**Pass:** a planted `(role → filler)` binding is recovered by cubelang's real
`UNBIND`+cleanup, through the full Python→Rust→Python path, at high similarity, with
the wrong-role control clearly separated — first over JSON (M1), then byte-for-byte
the same over protobuf (M2).

**Kill / stop-and-rethink:** if cubelang's real unbind cannot recover a planted
binding above the control (the VM's VSA doesn't actually work at usable capacity for
this shape), or if the CubeLang surface cannot express bind/unbind without
hand-assembling bytecode (the language isn't ready to be driven this way) — surface
it before building the protobuf layer on top.

## 6. Scope — explicitly deferred (YAGNI)

- **The full H-F2 bridge.** Bidirectional context/task-embedding + specialist-handle
  carriage, and fixing `NoveltyToWorldBridge`'s discarded return
  (`hebbian.py:373-374`), is the real H-F2 redesign — a later cycle. This slice is
  the minimal proof the path works at all.
- **grilly-in-Rust.** Reimplementing grilly's block-code algebra in Rust so raw
  block vectors cross intact is the big build; unnecessary while the boundary is
  symbolic. Only revisit if a future requirement genuinely needs the trunk's *vector*
  (not its concepts) unbound in Rust.
- **`opcode-vsa-rs` PyO3.** The repo's own stated future work, and the right eventual
  home for a fast in-process path — but only once real bind/unbind execution lives
  there (today it's cubelang's). Not needed for a subprocess slice.
- **Shipping raw hypervectors across the boundary** — rejected by design (§2).
- **Cleaning cubelang's drift** (stale root exe, orphan `src/cubelang/` dir, the
  `MEM_DIM=4096` vs `DEFAULT_DIM=8192` comment) — note it, fix only what the slice
  touches (the rebuild).

## 7. Placement & naming

- `cubelang` (Rust): the CubeLang program fixture, the `.proto`, `build.rs`, and the
  `main.rs` protobuf-stdio mode.
- Python client: a thin reasoning-bridge module (repo TBD — §8). Reuses/So supersedes
  the `run` path `cubemind/model/cubby/cubelang_bridge.py` marks "later."
- Spec + plan live in CubbyLLM `docs/superpowers/`. Result **updates H-B3** in
  `CUBBYLLM_HYPOTHESES.md` (real unbind exists in cubelang, exposed to Python via a
  symbolic protobuf boundary) and is the first checked step toward H-F2.

## 8. Open questions for the plan

1. **RESOLVED — and superseded by the foundation cycle (2026-08-05):** the original
   recon recommended adding an `unbind` keyword via a 5-file compiler patch. That is
   **no longer needed.** The foundation cycle built **`recover(reg, role)`** (a
   `use vsa;` registry helper doing real `UNBIND`+cleanup, returning the `Value::Str`
   symbol) — the intended user-facing surface (raw `unbind` stays `asm`-only, for
   helper-building). Planting is the existing `bind` statement. The full
   `bind` → `recover` roundtrip is proven end-to-end through `run --json` (recon
   2026-08-05: solve→"cat", wrong-role→"mouse", unbound→null). `recover`/`UNBIND`
   still discards the similarity float — symbol-only, which suffices for M1's
   recovery + controls; surfacing confidence is a deferred larger change.
2. **Client repo.** Does the Python reasoning client live in CubbyLLM (the trunk's
   own reasoning bridge) or cubemind (extend `cubelang_bridge.py`)? Default: a thin
   CubbyLLM-side client for the slice, since the point is *CubbyLLM* reaching the VM;
   fold into cubemind's bridge later if H-F2 wants it there.
3. **Program vs bytecode on the wire.** `RunRequest` carries CubeLang *source*
   (simplest, reuses `run`) vs precompiled bytecode (faster, but couples the schema
   to the bytecode format). Default: source for the slice.
