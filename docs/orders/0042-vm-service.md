# 0042 — The brain uses the VM as a service
status: proposed
written: 2026-10-25 by Cowork · approved: — · hypothesis: none

## Why
The VM now runs as its own service: `cubby-stack` (the repo beside this one) starts CubeLang's
`run-proto --listen 127.0.0.1:8793` with its own key, TLS 1.3 only, a SecP384r1MLKEM1024 key
exchange, AES-256-GCM, and both sides pinned (CubeLang `a10c58d`, `docs/CONTRACTS.md`, "Over a
network"). Only the brain's key may connect. Today the brain still starts its own `run-proto`
children (one per call, or one per resident session), so the service sits unused. With this order the brain talks to it instead,
through CubeLang's own relay (`run-proto --remote`), which speaks the same frames, so nothing above
the transport changes.

*In plain words: Cubby's rational side becomes a separate, locked room. The language side passes it
programs through one guarded door, and anything else that knocks is turned away.* What it brings
CubbyLLM: one VM process the stack owns and monitors (it shows on the admin page), its knowledge and
history caches kept warm across every brain restart, and the same door working unchanged when the
VM later moves to another machine. Over a conventional model: the reasoning step runs in an isolated,
audited process reachable only by pinned keys, not inside the model's own runtime.

## Build
- `cubbyllm/bridges/cubelang_client.py`: one helper, `run_proto_argv(exe) -> list[str]`:
  - `CUBBY_VM_REMOTE` unset (the default): `[exe, "run-proto"]`, exactly as today;
  - set: `[exe, "run-proto", "--remote", $CUBBY_VM_REMOTE, "--tls-key", $CUBBY_VM_TLS_KEY,
    "--server-pin", $CUBBY_VM_SERVER_PIN]`. The relay opens the brain's key with
    `CUBELANG_KEY_PASSPHRASE`, which the stack puts in the brain's environment (its own
    passphrase, never the host's). A missing key or pin, with `CUBBY_VM_REMOTE` set, is an error at
    the first call, never a silent fall back to a local VM.
- Use it in both proto transports: `run_program_proto` (one call per process) and the resident
  session class (the `Popen([self.exe, "run-proto"], ...)` at about line 410).
- `run_program` (`cubelang run --json`) stays local: it is a different command, used by tests and
  tools, not the serving path.
- `knowledge_path` and `history_path` ride on the request and are read on the VM's machine. The stack
  runs both on one machine, so the paths stay valid; note it in the docstring.
- The environment the stack gives the brain: `CUBBY_VM_REMOTE=127.0.0.1:8793`,
  `CUBBY_VM_TLS_KEY=<stack home>/brain.key.json`, `CUBBY_VM_SERVER_PIN=<the VM's pin>`,
  `CUBELANG_KEY_PASSPHRASE`, `CUBELANG_EXE` (the stack's CubeLang build). Nothing else to configure.

## Acceptance (pre-registered)
- A unit test pins `run_proto_argv` both ways (unset: today's argv, byte for byte; set: the relay's
  argv), and the error when the key or the pin is missing.
- An integration test, skipped without `CUBELANG_EXE`, makes two throwaway identities
  (`cubelang keygen ... --plaintext`), starts `run-proto --listen` on a free loopback port with the
  client pinned, and runs the same programs through `run_program_proto` and the resident session,
  locally and remotely: the results are identical. A key that is not pinned is refused.
- The 92 tests that drive the binary pass unchanged with `CUBBY_VM_REMOTE` unset, and pass with it set
  against a VM service started by a fixture.
- Report, on this machine, the time per request (median of 200) for: local one-call, remote one-call,
  local resident, remote resident. Gate: the remote resident session adds at most 2 ms per request
  over the local one. Kill line: more than 5 ms. Then stop and report instead of tuning.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Off by default: with `CUBBY_VM_REMOTE` unset, every path is byte for byte what it is today.
- No key, pin or passphrase in the repository, in a log line, or in an error message (name the
  variable, never its value).

## Out of scope
- Starting or stopping the VM: the stack does that (`python cubby.py start vm`, or the admin page).
- `use history;` over the sealed stores through the VM service: the VM would need the host key; a
  later order, if wanted.
- The stack's side (keys, pins, environment): done in `cubby-stack`.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
