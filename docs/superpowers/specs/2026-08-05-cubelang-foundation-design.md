# CubeLang foundation — a verified, capability-based, LM-writable reasoning VM

**Date:** 2026-08-05 · **Status:** design, in review · **Repo:** implementation in `C:\Users\grill\Documents\GitHub\cubelang` (Rust); spec/plan tracked here in CubbyLLM. **Relates to:** H-B3 (real unbind), H-F2 (bridges), and the reasoning-bridge slice spec (`2026-08-05-reasoning-bridge-slice-design.md`), which sits on top of this.

## 1. Why this exists

CubeLang is CubbyLLM's reasoning VM. The north star (owner, 2026-08-05): it should be a **living, verified execution substrate for AI** — the tech shape of an EVM (safe, deterministic, verified-before-execution; *not* financial), evolved into an **OS for AI**. Programs are either a pre-built library or **generated on demand by a separate specialist model** (not trunk0). Two hard requirements fall out of that:

- **LM-writable + human-readable.** A trained model has to be able to emit correct CubeLang, and a human has to be able to read it. Today some `.cube` constructs are esoteric and ceremonious; that has to change.
- **Safe to run code something else wrote.** The verifier must *reject* bad programs, the core must be *tamper-proof*, and every boundary must be *mediated*. This is a capability system, not a file system.

A polish audit (2026-08-05) found the load-bearing crack: **`--strict` — the "verified before it runs" gate — is blacklist-shaped and silently accepts any unrecognized statement** (live-reproduced: `frobnicate d;` compiles clean under `--strict` and runs). Everything above leans on `--strict` meaning "valid," and today it means "didn't hit a known-bad pattern." Fixing that is the foundation of this whole cycle.

## 2. Design principles (the through-line)

- **Deny-by-default, everywhere.** The verifier whitelists known-good; capabilities grant nothing by default; the future guardian layer only ever *restricts*. Every layer that could be allow-by-default is a hole.
- **Verify containment, not correctness.** You cannot prove an arbitrary (LM-generated) program *correct*; you *can* prove it can't escape the sandbox or touch the protected core. Aim there.
- **Tamper-proof core, extend by permission.** Core helpers and standard interfaces live *in the VM* (Rust), like EVM precompiles — versioned, name-addressed, unmodifiable. Programs `use` them and may `override` only where explicitly marked overridable.
- **Natural over esoteric.** Reduce ceremony; push low-level machinery behind named helpers. If a construct is hard for an LM to emit or a human to read, it's a bug.

## 3. In scope — this cycle

Ordered roughly by dependency. Each is TDD'd; `cargo test` green is the gate.

### A. Verification foundation — `--strict` deny-by-default  *(task 0, foundational)*
- Flip `strict_check_stmt` from a **blacklist** (inspects only `Match`/`ExtOp`/`Unify`) to a **whitelist**: any statement that compiled to **zero bytecode** and isn't an explicitly-recognized no-op is a **hard error under `--strict`**. Catches unknown-statement typos (`frobnicate d;`) and the `Stmt::Expr`-discard fall-through.
- Fix the sibling silent-drop: `Stmt::Assign` to a non-`Ident`/non-`self.field` target (`arr[0] = x`, nested access) currently emits zero bytecode with no error — make it a hard error (or implement it; error first).
- **Test:** a program with an unrecognized statement, and one with an unsupported assignment target, each **fail** `--strict` with a clear message. This is the regression that proves "verified" now means "valid."

### B. `use` — the capability/module system
- A VM-internal **registry** (Rust) of built-in modules — helpers (`vsa`, `math_level1`, …) and standard interfaces (§C) — **versioned with the VM** (a program targets a VM version; `use vsa` means the same thing every run) and **name-addressed** (readable, not EVM-style fixed addresses).
- **`use <module>;`** — a top-level declaration bringing a VM-internal namespace into scope. No external file; the VM maps it to the internal impl. (Supersedes external `.cube` file import — see §5.)
- **Call resolution** for `name()` after `use m;`: (1) program has `function name() override {…}` **and** `m::name` exists **and** is `overridable` → the override; (2) else `m::name` exists → the VM-internal impl; (3) else → error.
- **`override`** is validated: overriding a name not in any `use`'d module, or not marked overridable, is a compile error. Core stays protected; the intelligent/self-patch story later builds on this.

### C. Standard interfaces (VM-internal) + validated, required `implements`
- Bake the standard interfaces into the registry: a minimal **`ISolve { abstract solve }`** for solve-only programs, and the full **`ISolver { abstract parse; abstract solve; abstract verify }`** where all three genuinely exist. `use isolver;` / `use isolve;` brings them in.
- **`implements` becomes required and validated**: a `program` with no `implements` is an error; and for each implemented interface, the program must provide every `abstract` (non-`optional`) function with matching name/arity — checked against the VM registry (tamper-proof). Kills today's phantom (`fibonacci implements ISolver` with `ISolver` declared nowhere) and the six duplicate copy-pasted `ISolver` bodies.

### D. `asm` + the `vsa` helper — real unbind, hidden behind a natural call
- **`asm { OP args…; … }`** — the low-level hatch that emits raw opcodes (reusing the op table + operand tags `bind` already emits: `NAMED`/`ROLE`/`GLOBAL`). This is how helpers are *built*; generated user programs never write it.
- **`vsa` VM-internal helper** wrapping bind/unbind/cleanup: `use vsa; … recover(frame, ROLE)` compiles to the VM's real `UNBIND` + cosine cleanup (`engine.rs` already implements it) and returns the recovered symbol. This is the natural surface the reasoning slice writes against — no `asm` in sight.

### E. `quantity` → `number`
- Rename the value-type name across `src/`, `examples/`, `tests/`, `docs/` (leave the `QUANTITY` *role* alone). SPEC.md's own type-inference lattice already uses `number` — this converges code onto SPEC's type theory rather than inventing a name.

### F. Conformance migration (~12 programs)
- Bring every example/test program into the new rules: `use` the right standard interface, satisfy it, drop inline/duplicate interface bodies. Covers the six `ISolver` duplicates **plus** the non-conformers required-`implements` newly breaks: `ground_min`/`ask_min` (no `implements` → `use isolve;` + a real `solve`), `fibonacci` (phantom → `use isolver;` + provide the functions), and the inline test programs in `query_grounding.rs`/`ask_suspend.rs`.

### G. Ergonomics / naturalness
- Reduce the boilerplate every example carries (`@external public function solve(): …` ceremony) toward something an LM emits reliably and a human skims. Concrete forms to be decided in the plan (sensible defaults, keyword trims) — the criterion is "reads naturally, generates reliably."

### H. Polish (audit's safe cleanups)
- Remove root scratch files (`_t_*.cube`, `_verify*.ps1`) and the empty `src/cubelang/` orphan dir; drop stale binaries + fix `.gitignore` so a clean `cargo build --release` is the source of truth.
- Delete genuinely-dead AST stubs (`Stmt::Export`/`Exec`/`Gate`, the `bytecode wasm import` cluster) — but **keep `Stmt::Import`** (external file import is in scope, §3.I / §5) and anything else on the roadmap.
- Fix drift: the `MEM_DIM=4096` comment that claims to match opcode-vsa-rs (8192); stale dimension doc-comments. **Do not** reconcile VSA dims across repos here (bigger cross-repo call — just stop the comments lying).
- Test hygiene: the `Import`/`Export`/`BytecodeKw` **lexer-only** tests create false confidence (they prove tokenization, not that the parser does anything) — remove or upgrade. Add the missing coverage around `use`/override/`asm`/UNBIND/interface-conformance the audit flagged.

### I. `import` — external user files (modularity)
- A loader pre-pass (§5): given the entry file, parse it, walk top-level `import "path.cube";` decls, resolve each **relative to the importing file**, recursively load + **AST-merge** their top-level items (interfaces/structs/types/**and helper functions**), with **cycle detection** and **error-on-duplicate-name**. Keep `parse`/`compile`'s `&str` signatures untouched (the loader wraps them, preserving spans for error line numbers). Imported code is verified (`--strict`, validated `implements`) exactly like inline code — never a bypass.
- **Test:** a program `import`s a user file's helper function + interface and uses them; duplicate-name across files errors; an import cycle is detected; a relative path resolves. (Untrusted-author path-allowlisting is deferred to the guardian layer — §4.)

## 4. Deferred — captured, next cycles (not built here)

- **The security / ingestion layer:** the `net` capability (allowlist + anti-SSRF + sandboxed fetch + inert-data + provenance/trust), the **guardian / reference-monitor** layer (deny-by-default reference monitors at every entry/exit; intelligent layer *restricts-never-grants*, tiered by risk), and the **learning-gate** (Cubby-side; poisoning defense on real-time learning). Its own cycle — needs real security engineering + Cubby-side work. This foundation (verification + capability core) is what it builds on.
- **The codegen specialist model** (writes CubeLang; separate from trunk0) — future; this cycle just makes the language it will target safe and writable.
- **Real-time learning from live sources** — Cubby-side (H0-adjacent), future.
- **Governing *untrusted* `import`** — *who* may import *what* when the author is the codegen model, not a human (path-allowlisting, sandboxed project roots). External `import` itself is IN scope this cycle (§3.I, §5) for the trusted-author case; policing it for generated code is part of the deferred guardian layer.
- **Self-generating virtual processors, self-patch-as-a-call** — the far vision; `override`-with-permission is the seed, not the deliverable.
- **Reconciling VSA dims / algebras across grilly / opcode-vsa-rs / cubelang** — separate cross-repo decision.

## 5. Two module mechanisms: `use` (core) + `import` (your files)

Two complementary mechanisms — the split is **who owns the code:**

- **`use <name>;` — VM-internal core.** Helpers and standard interfaces baked into the VM (Rust), tamper-proof, versioned, deny-by-default — like EVM precompiles. This is where the six-duplicate-`ISolver` collapse happens (the interfaces are core) and where safety-critical machinery lives; nothing a program does can modify it.
- **`import "path.cube";` — external user files.** For splitting a large program across files and reusing *your own* authored functionality (interfaces, structs, types, **and helper functions**). Built with the loader pre-pass from the interface/module recon: AST-level merge (preserves line numbers for errors), resolve **relative to the importing file**, cycle detection, error-on-duplicate-name. Imported code is **verified like any code** — it passes `--strict` and validation; `import` is modularity, never a trust bypass.

Security split: the *core* (dangerous to tamper with) is VM-internal and unmodifiable; *your files* are your own trusted code. The one remaining question — **who may `import` what** when the author is the codegen model, not a human (path-allowlisting, sandboxed project roots) — is a capability governed by the guardian layer (§4, deferred). For now `import` assumes a trusted author. So `Stmt::Import` is **kept** and the loader is **built**; only the genuinely-dead stubs (`Export`/`Exec`/`Gate`, the wasm-import cluster) go.

## 6. Testing

`cargo test` green is the gate for every item. Specifically new: the `--strict` regression (§A — unknown statement + bad assignment target must now **error**); `use`/override resolution (override chosen when present+overridable; VM impl otherwise; error when neither; overriding a non-overridable name errors); `implements` validation (missing → error; missing required fn → error; phantom interface → error; valid → ok); the `vsa`/`asm` roundtrip (bind → `recover` → symbol, with a wrong-role control); the conformance suite (all migrated programs compile+run). No red tests committed; the `--strict` fix must not break a legitimate existing program (run the example suite under `--strict`).

## 7. Placement & sequencing

Implementation lives in the `cubelang` repo, executed on a branch, `cargo test` green before merge. Rough order: **A (verify) → H-safe-cleanup + E (rename) → B (`use`) → C (interfaces + implements) → D (`asm`/`vsa`) → F (conformance) → G (ergonomics)**, with the reasoning-bridge slice (separate spec) stacking on top once D lands. Cross-repo note: the reasoning slice's Python client is the only non-cubelang piece; this cycle is entirely Rust.

## 8. Open questions for the plan

1. **`--strict` whitelist shape:** enumerate the allowed statement kinds explicitly (cleanest, deny-by-default) vs a "produced ≥1 real opcode or is a recognized no-op" check (looser). Default: explicit whitelist.
2. **`use`/override syntax exactness:** confirm `use <name>;` placement (top-level vs in-program) and the `override` marker form against the current grammar.
3. **Ergonomics scope (§G):** how far to trim ceremony in *this* cycle vs a follow-up — risk is churning every example twice. Default: minimal trims now (the ones the conformance migration touches anyway), a dedicated ergonomics pass later.
4. **`number` vs a full type-system tighten:** the rename is mechanical; actually *enforcing* numeric-vs-symbol typing is a bigger change — out of scope here (rename only).
