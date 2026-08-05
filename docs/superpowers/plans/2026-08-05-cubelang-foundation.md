# CubeLang Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make CubeLang a verified, capability-based, LM-writable reasoning VM — fixing the `--strict` verification hole, adding the `use` capability system + `import` for user files, validated `implements`, a `vsa` helper (real unbind behind a natural call), the `quantity→number` rename, conformance, and polish.

**Architecture:** All work is in the **`cubelang` Rust repo** (`C:\Users\grill\Documents\GitHub\cubelang`), on a branch, `cargo test` green before merge. Design: `docs/superpowers/specs/2026-08-05-cubelang-foundation-design.md`. Two module mechanisms — `use` (VM-internal core, tamper-proof) and `import` (external user files). Deny-by-default verification is the foundation everything else stacks on.

**Tech Stack:** Rust (edition 2021, cargo 1.94.1 / rustc 1.94.1, both on PATH). No new deps expected except possibly the loader (Task 5) uses only `std`.

## Global Constraints

- **`cargo test` green is the gate for every task.** Never commit a red test. The suite currently passes; keep it passing (plus new tests).
- **Branch first** — a `cubelang` feature branch (e.g. `feat/foundation`); never commit to the cubelang default branch directly.
- **Preserve the 3-way opcode sync** — `tests/opcode_sync.rs` textually cross-checks `../opcode-vsa-rs/src/ir.rs`. Don't change opcode bytes; if a task adds an opcode, update all three tables + the sync test.
- **Don't break legitimate programs** — after Task 0, run the whole `examples/` suite under `--strict`; every currently-valid program must still compile. (Task 7 fixes the ones the *new* rules intentionally break.)
- **Rebuild is the source of truth** — the root `cubelang.exe` and `target/release` are stale; use `cargo build`/`cargo test` fresh. `target/debug/cubelang.exe` (2026-07-09) is current for manual checks.
- **Commit messages** match the cubelang repo's existing style (clean conventional commits; this repo does not use the CubbyLLM `Co-Authored-By`/`Claude-Session` trailers — check `git log` and match).
- Each task confirms exact syntax/AST/API against the *current* grammar before writing — the recon file:line refs below are the starting point, not a guarantee they haven't moved.

## File Structure (primary files, from the recons)

- `src/lexer.rs` — keyword table (~346), tests (~695,735,765).
- `src/token.rs` — `TokenKind` (~101-207).
- `src/ast.rs` — decls/stmts (`OpcodeStmt` ~548, `Stmt::Import/Export/Exec` ~349, `ImportStmt` ~502, `InterfaceDecl` ~26-37, `ProgramDecl.implements` ~44).
- `src/parser.rs` — top-level `parse()` loop (~319-339), `parse_program` (~375-418), opcode stmts (~1276-1298), `parse_interface` (~343-371), `parse_stmt` (~924-987).
- `src/compiler.rs` — `mod op` opcode table (~11-78), operand tags (~83-95), `compile()` (~443-451), `compile_program` (~453-482), Bind emission (~726-761), `strict_check_stmt` (~505-525), `Stmt::Assign` (~602-616), `emit_named/emit_role/record_symbol` (~1098-1226), `compile()` public (~1284).
- `src/vm/engine.rs` — `BIND_ROLE` (~639), `UNBIND` (~855), `vsa_bind_into`/`vsa_unbind_cleanup` (~1108-1154), `load()` (~284-303), tests (~2201-2331), `MEM_DIM` (~12-14).
- `src/vm/knowledge.rs`, `src/vm/codebook.rs`, `src/vm/hypervec.rs` — VM VSA + knowledge store.
- `src/main.rs` — CLI + `value_to_json` (~569-583), read sites (~103,318,449).
- `docs/SPEC.md` — grammar (~2720-2792), types (~274-283, ~2693), opcode table (~2625-2681).
- `examples/*.cube`, `tests/*.rs`.
- **New:** `src/loader.rs` (Task 5), a VM helper/interface registry module (Task 3, e.g. `src/vm/registry.rs`).

---

### Task 0: `--strict` deny-by-default (verification foundation)

**Files:** `src/compiler.rs` (`strict_check_stmt` ~505-525, `Stmt::Assign` ~602-616), `tests/` (new strict tests).

**Interfaces:** Produces: `--strict` now *rejects* any statement that compiled to zero bytecode and isn't an explicitly-recognized no-op, and rejects unsupported assignment targets. Consumed by Task 4/7 (interface conformance runs under strict).

- [ ] **Step 1: Failing test — unknown statement must error under strict.** Add a Rust test that compiles `program T { public function solve(): number { frobnicate d; return 0; } }` with `compile_strict` and asserts it returns `Err` (today it returns `Ok`). Also one for `program T { … arr[0] = x; … }` (unsupported assignment target). Base the harness on the audit's live repro.
- [ ] **Step 2: Run — verify both currently PASS-compile (i.e. the test fails).** Confirms the hole exists.
- [ ] **Step 3: Implement the whitelist.** In `strict_check_stmt`, replace the blacklist (only inspects `Match`/`ExtOp`/`Unify`) with: a statement is valid under strict iff it is a recognized construct that produced ≥1 real opcode OR is an explicitly-whitelisted no-op. Concretely: flag any statement that falls through to `Stmt::Expr`→`compile_expr_discard` with zero emitted bytecode, and any `Stmt::Assign` whose target isn't a handled form, as a strict error with a clear message (statement kind + span/line). Default to the **explicit whitelist of known statement kinds** (spec §8.1) — deny anything not on it.
- [ ] **Step 4: Run — both new tests pass; the whole `examples/` suite still compiles under `--strict`.** (Any legitimate example that now fails is a real gap — investigate, don't suppress.)
- [ ] **Step 5: Commit.**

---

### Task 1: Safe polish / cleanup

**Files:** repo root (scratch files, stale binaries, `.gitignore`), `src/` (dead stubs), `src/vm/engine.rs` (`MEM_DIM` comment), lexer tests.

- [ ] **Step 1: Remove cruft.** Delete root scratch files (`_t_*.cube`, `_verify*.ps1`) and the empty `src/cubelang/` orphan dir. Ensure stale binaries aren't tracked; `.gitignore` covers `target/` and any built `.exe`. (No behavior change — no test needed beyond `cargo build` clean.)
- [ ] **Step 2: Delete genuinely-dead AST stubs.** Remove `Stmt::Export`, `Stmt::Exec`, `Stmt::Gate`, **the dead ES-module `Stmt::Import` / `ImportStmt`** (wrong shape — file import will be a NEW top-level decl in Task 5, not an in-body statement), and the `bytecode wasm import` cluster (`BytecodeKind::WasmImport`/`CodebookLoad`/`CodebookExport`) + their tokens/lexer entries. **Also remove Task 0's now-orphaned `Stmt::Import` arm in `strict_check_stmt`** (the exhaustive match must still cover every remaining `Stmt` variant with no wildcard). Verify (grep) each variant is genuinely unused before deleting; compile-check that nothing references the deleted variants.
- [ ] **Step 3: Fix drift comments.** `engine.rs:12-14` `MEM_DIM=4096` comment claims it matches opcode-vsa-rs (which is 8192) — correct the comment to state the real value and that cross-repo dim reconciliation is out of scope. Fix other stale dimension doc-comments the audit flagged. (No behavior change.)
- [ ] **Step 4: Test hygiene.** Remove/upgrade the lexer-only false-confidence tests for the deleted tokens; keep `Import` tokenization test (still used). `cargo test` green.
- [ ] **Step 5: Commit.**

---

### Task 2: `quantity` → `number` rename

**Files:** `src/*` (lexer test strings, compiler test strings, validate test strings), `examples/*.cube`, `tests/*.rs`, `docs/SPEC.md`, remaining scratch — everywhere the *type name* `quantity` appears. **Leave the `QUANTITY` semantic role** (`SPEC.md:285`) untouched.

- [ ] **Step 1: Inventory.** Grep `quantity` (lowercase, as a type — `: quantity`, `= quantity`, `Output = quantity`) vs the `QUANTITY` role and any unrelated `Quantity`. Confirm (per audit) it lexes as a plain `Ident` with `ty: String`, so no keyword-table change — but verify no Rust logic string-matches `"quantity"` as a type (if it does, rename there too).
- [ ] **Step 2: Rename** every type-name occurrence to `number`. Update `docs/SPEC.md` (its type-inference lattice already uses `number` — converge on it; add `number` to the Base Types list if the primitives list is authoritative).
- [ ] **Step 3: Run `cargo test`** — green (many inline-string test programs reference the type; they must all still compile/run).
- [ ] **Step 4: Commit.**

---

### Task 3: `use` — the VM-internal capability/module system

**Files:** new `src/vm/registry.rs` (or similar); `src/lexer.rs`/`src/token.rs` (a `use` keyword if not present), `src/ast.rs` (a `TopLevel::Use` + `function … override` marker), `src/parser.rs` (top-level `use` + `override`), `src/compiler.rs`/`src/vm/engine.rs` (resolution).

**Interfaces:** Produces: a VM registry mapping `module_name → {functions: {name → (impl, overridable)}}`, versioned with the VM; `use <name>;`; call resolution `override > VM-impl > error`; `function name() override {…}` validated. Consumed by Tasks 4 (interfaces are registry entries) and 6 (`vsa` is a registry helper).

- [ ] **Step 1: Failing test — `use` + resolution.** Register a trivial test helper module in the registry (e.g. `demo` with an overridable `greet()` returning a known value). Test: (a) `program … { use demo; … return greet(); }` returns the VM impl's value; (b) with `function greet() override { … }` it returns the override; (c) `override` on a non-overridable or absent name errors; (d) calling `greet()` without `use demo;` errors.
- [ ] **Step 2: Run — fails** (no `use`/registry yet).
- [ ] **Step 3: Implement.** (a) The registry type + a versioned, name-addressed table seeded with the demo module; (b) `use <name>;` as a `TopLevel` decl (add keyword/token/AST/parser arm — mirror the existing top-level `interface`/`struct` handling at `parser.rs:319-339`); (c) an `override` marker on functions (AST + parser); (d) name resolution in the compiler/VM implementing `override > registry-impl > error`, with the override-validity checks. Keep it deny-by-default (only `use`'d modules' names resolve).
- [ ] **Step 4: Run — all four cases pass.**
- [ ] **Step 5: Commit.**

---

### Task 4: Standard interfaces (VM-internal) + validated, required `implements`

**Files:** the registry (Task 3) — add `ISolve`/`ISolver` interface entries; `src/compiler.rs`/`src/validate.rs` (a validation pass); `src/parser.rs:380-384` (required check).

**Interfaces:** Consumes: the registry (Task 3). Produces: `use isolve;`/`use isolver;`; `implements` required + validated against the registry.

- [ ] **Step 1: Failing tests.** (a) `program P { }` (no `implements`) → error; (b) `program P implements ISolver { use isolver; … }` missing a required `abstract` fn → error; (c) a program implementing an interface it never `use`d / that isn't in the registry → error; (d) a valid program with `use isolver;` + all three functions → ok. (Interfaces are registry entries, so no inline interface decl needed.)
- [ ] **Step 2: Run — fails** (implements is decorative today).
- [ ] **Step 3: Implement.** Add `ISolve {abstract solve}` and `ISolver {abstract parse; abstract solve; abstract verify}` to the registry. In the parser (`~380-384`), reject empty `implements`. Add a validation pass (natural home: `compiler.rs::compile()` before `compile_program`, or `validate.rs`): for each name in `prog.implements`, look it up in the registry; error if absent; for each `abstract` (non-`optional`) member, confirm the program body has a matching function (name/arity). All under the Task-0 strict discipline.
- [ ] **Step 4: Run — all cases pass.** (Existing programs that break here are Task 7's job — expect some example/test failures until then; keep *this task's* new tests self-contained so the suite for this task is green.)
- [ ] **Step 5: Commit.**

---

### Task 5: `import` — external user files

**Files:** new `src/loader.rs`; `src/main.rs` (call the loader at the 3 read sites ~103,318,449 instead of bare `read_to_string`+`parse`).

**Interfaces:** Produces: a new `TopLevel::Import(path)` variant + `src/loader.rs`; `import "path.cube";` resolving external user files. (The old in-body `Stmt::Import`/`ImportStmt` was deleted in Task 1 as wrong-shaped.)

- [ ] **Step 1: Failing test.** A fixture `tests/fixtures/lib.cube` defining a helper `fn`/interface; a program `import "lib.cube"; …` that uses them → compiles+runs. Plus: duplicate-name across two imported files → error; an import cycle (`a`→`b`→`a`) → detected/error; a path relative to the importing file resolves.
- [ ] **Step 2: Run — fails** (parser rejects `import` as a statement today; make it a `TopLevel::Import`).
- [ ] **Step 3: Implement.** (a) Add a new `TopLevel::Import(path)` variant and parse `import "path";` as a **top-level** decl (parser arm in the `parse()` loop, sibling to `interface`/`struct`). (b) `src/loader.rs`: parse the entry file → walk `TopLevel::Import` items → resolve each relative to the importing file (`Path::new(current).parent().join(path)`) → recursively load → **AST-merge** top-level items (interfaces/structs/types/functions), tracking a visited-set for cycles and erroring on duplicate names. Keep `parser::parse`/`compiler::compile` `&str` signatures; the loader wraps them. (c) Imported code runs through the same `--strict`/validation. (d) Wire the loader into `main.rs`'s read sites.
- [ ] **Step 4: Run — all cases pass; `examples/` still build.**
- [ ] **Step 5: Commit.**

---

### Task 6: `asm` opcode block + the `vsa` helper

**Files:** `src/lexer.rs`/`token.rs`/`ast.rs`/`parser.rs`/`compiler.rs` (the `asm` block), the registry (the `vsa` helper), `src/main.rs`/`value_to_json` (already serializes `Value::Str` — verify).

**Interfaces:** Consumes: the op table + operand tags (`compiler.rs:83-95`), the VM's `UNBIND`+cleanup (`engine.rs:855`, already real). Produces: `asm { OP args… }`; `use vsa; recover(frame, ROLE)`. This is what the **reasoning-bridge slice** (separate spec) writes against.

- [ ] **Step 1: Failing tests.** (a) `asm`: a program `create frame:number; asm { BIND_ROLE frame, SUBJECT, "cat"; UNBIND frame, SUBJECT } return frame;` runs and returns `"cat"` (mirrors the hand-assembled `engine.rs:2277-2330` test, but via source). (b) `vsa` helper: `program P implements ISolve { use vsa; function solve(): number { … recover(frame, SUBJECT) … } }` recovers the planted filler; a wrong/absent role returns a different/no symbol (control).
- [ ] **Step 2: Run — fails.**
- [ ] **Step 3: Implement.** (a) `asm { … }` block: lex/parse a block of `MNEMONIC operand,operand…` lines; compile each by mapping mnemonic→op byte (op table) and operands→tagged bytes reusing `emit_named`/`emit_role`/`emit_global` (bareword=`ROLE`, quoted=`GLOBAL`-literal, ident=`NAMED`-register) — matching the layout `engine.rs`'s UNBIND test hand-assembles. (b) The `vsa` registry helper: `recover(reg, role)` (+ `bind` if wanted) that internally performs the VM's `BIND_ROLE`/`UNBIND`+cleanup and returns the `Value::Str` symbol. Generated user programs call `recover`; `asm` is only for building helpers.
- [ ] **Step 4: Run — asm roundtrip + `vsa::recover` (with control) pass.**
- [ ] **Step 5: Commit.**

---

### Task 7: Conformance migration (~12 programs)

**Files:** the six `ISolver`-duplicate `.cube` files (`recall_min`, `decision_min`, `compare_min`, `loop_min`, `qc_decision`, `conversation_min`), `fibonacci.cube`, `ground_min.cube`, `ask_min.cube`, the 4-method variants (`gsm8k`, `conversation_agent`), and the inline test programs in `tests/query_grounding.rs`/`tests/ask_suspend.rs`.

**Interfaces:** Consumes: `use isolve/isolver` (Task 4), `import` (Task 5) if any user interface is shared via file.

- [ ] **Step 1:** For each program, replace its inline/duplicate/phantom interface with `use isolver;` (or `use isolve;` for solve-only programs like `ground_min`/`ask_min`), and ensure it provides the interface's required functions (add real `parse`/`verify` only where the program genuinely does them; otherwise it implements `ISolve`). Update the inline test programs likewise.
- [ ] **Step 2: Run `cargo test` + the `examples/` suite under `--strict`** — everything green. This is the task that turns the whole repo valid under the new required-`implements` rule.
- [ ] **Step 3: Commit.**

---

### Task 8: Ergonomics / de-ceremony (minimal)

**Files:** `src/parser.rs`/`docs/SPEC.md` + the examples touched.

**Interfaces:** Produces: reduced boilerplate for the common program shape.

- [ ] **Step 1:** Pick the *minimal* high-value trims (spec §3.G / §8.3) — e.g. defaulting/eliding the `@external public function solve(): number` ceremony for the common case — that don't churn every example twice. Write a failing test showing the terser form parses/compiles equivalently.
- [ ] **Step 2–4:** Implement, run (terser + existing forms both work), migrate the examples that benefit, `cargo test` green, commit. (A deeper ergonomics pass is a separate later cycle — keep this bounded.)

---

## Post-plan

Once this lands, the **reasoning-bridge slice** (`2026-08-05-reasoning-bridge-slice-design.md`) writes its M1 program against `use vsa; recover(…)` + a `use isolver;` interface — natural, no `asm`, verified under `--strict`. The security/guardian/`net` layer and the codegen model are their own later cycles (spec §4).
