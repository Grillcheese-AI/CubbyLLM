# 0062 — Skills as code: Cubby writes a CubeLang helper in a sandbox, and adopts it only if it reproduces the whole record
status: open (queue: after 0049)
written: 2026-11-07 by Cowork · approved: 2026-11-07 by Nick ("yes", after his cloned-agent example: a sandboxed tutor's percentage skill, caught multiplying by 15 without dividing by 100) · hypothesis: H-A13's level 3 ("rules first, then CubeLang helpers"), with H-A15 (0049)

## Why
H-A13's rule library learns composition rules overnight and adopts one only if it re-derives every past episode with
zero counterexamples. Its stated next step is CubeLang helpers: reusable skills written as code. Nick's example shows
the shape. A copy of an agent writes a new skill in a sandbox, the VM tests it, the broken skill is rejected, and the
sandbox returns to its last good state.

**What it brings, in plain words:** Cubby can teach itself a new skill, but the skill has to pass every test Cubby has
ever passed before it is used for real. A broken skill is caught in a sandbox and kept on record as a lesson.
- **Over a conventional model:** when a model learns, its weights change and nobody can see what broke. Here the new
  skill is readable code, tested against the whole record and rolled back at once if it fails.

## Build
0. **First, check what CubeLang supports today:** `extend`, the permission attributes, instruction limits and
   `@hook`. CubeLang's own README lists `@hook` as a strict compile error today. Missing features go to the cubelang
   repo as their own job, with Nick's yes. Until then, the host runs the sandbox and the rollback.
1. **A sandbox per candidate:** a cloned VM session (exp_r20 measured 1.69 MB and 19 ms per session). Inside it:
   - no network, no file access, no writes to the knowledge base;
   - an instruction budget per call and a time limit, so a looping skill stops.
2. **Who proposes:** Cubby's own proposers only.
   - The program seat writes a helper from a cluster of verified episodes that share a shape.
   - The rule miner lifts a family of rules into one function.

   A helper written by a person or by Claude Code goes through the normal order process instead.
3. **The form:** a helper adds a function. It never replaces a function of a program marked immutable; inside the
   sandbox it may shadow one, for the test only.
4. **The gate:** H-A13's gate, applied to code.
   - The helper must reproduce every verified answer on the record that falls in its scope. Expected values come
     from the record or from exact arithmetic, never from a model.
   - It must also pass held-out episodes, with zero counterexamples.
   - It must change no answer outside its scope: the whole record is re-run with and without it.
5. **Promotion:** the host promotes, never the sandbox. Each promoted helper is kept with its hash, its scope, the
   episodes it was tested on, and its before-and-after rates, under 0049's promotion rule and keep-list.
6. **Rejection:** retire, never delete. A rejected helper is kept with the case that broke it as a "never this"
   record (0049's T3), and the sandbox resets to its last good snapshot.
7. **Use:** every call of an adopted helper still runs through the VM, and its answers are certified like any other.

## Acceptance (pre-registered)
- **Planted faults:** seeded broken helpers (wrong scaling, off by one, a flipped sign, a wrong unit), with Nick's
  percentage example first. The gate rejects each one.
- **A real cycle:** helpers proposed from the arithmetic step loop's verified episodes (H-E19) and from CLUTRR's
  rules.
  - Reported: helpers proposed, adopted and rejected, and the right answers they gained.
  - Line: 0 wrong.
- **Sandbox escapes:** attempts at the network, a file, a knowledge-base write and an endless loop are all blocked.
- **Kill:** any wrong answer from an adopted helper. Helpers are then switched off until the gate is fixed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model writes helpers in the loop.
- **CubeLang changes go to the cubelang repo,** only with Nick's yes, and the main checkout's `target/release` binary
  is never overwritten.
- Claude Code never starts a paid run.
- No local path in a committed file.

## Out of scope
- Helpers that call the network or outside tools.
- Sharing helpers between users. Each user's VM stays isolated; a shared library is a later decision.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Cowork's addendum (2026-11-07): watching the proposers, and what the CubeLang compiler accepts today
Nick: "yes", with his drafts of an orchestrator (`orchestrator.cube`) and a bridge between containers
(`bridge_mesh.cube`).

**What it brings, in plain words:** Cubby keeps an eye on its own skill-writers. One that keeps getting it wrong is
slowed down; one that tries to break out is stopped at once, and Nick is told.
- **Over a conventional model:** when an agent misbehaves, you find out from the damage. Here every attempt is on
  record and counted.

### Two standing checks over the records
- **Quality:** each proposer's rejection rate, meaning helpers rejected divided by helpers proposed over a recent
  window, never a raw count. Counts are distinct helpers, so one rejected helper is one, whatever its opcodes or
  failed checks.
- **Safety:** sandbox escape attempts per proposer (network, file, knowledge-base write, budget overrun). This should
  always be 0.
- **The rules:**
  - one escape attempt pauses that proposer at once and alerts Nick;
  - a rejection rate above a threshold Nick sets slows the proposer (fewer proposals per cycle);
  - a paused proposer returns only with Nick's yes.
- **The records:** paused and slowed proposers stay in the records with their history (retire, never delete).
- **Where they run:** as queries over 0062's records on QLever (0055's reference engine) or the in-process store,
  shown in the 0054 report. An answer refusal ("fact missing") is never counted against a proposer; only its helpers'
  gate results are.

### The compiler, measured on Nick's drafts (Cowork, 2026-11-07)
Measured with `cubelang check` on the main checkout's release binary, built 2026-09-29. The source has moved since
(last commit 2026-10-31), so step 0 re-runs this against a build from the current source, in a separate target
directory: the main `target/release` binary is never overwritten.
- **`orchestrator.cube` fails to compile.** The errors found, in order, each fixed in a scratch copy to reach the
  next:
  1. `@hook(*.error)`: no wildcard in a hook target ("expected identifier, got Star"). `@hook` itself is listed as a
     strict compile error in CubeLang's README.
  2. A parameter named `event`: a reserved word.
  3. `??` (null coalescing) is not parsed, though the README uses it in an example.
  4. `revoke ... from ...` inside a function body is not parsed, though the README documents it.

  Later lines were not reached.
- **`bridge_mesh.cube` passes `check`, but compiles to nothing.** `compile` warns "no programs found", so the
  `container`, its `io` block, `vm.start_from_file`, `vm.wire` and `vm.bridge` produce no bytecode. A strict check
  that passes on code that never runs is itself a defect, and it is reported to the cubelang job.
- **The design notes** for when these compile:
  - the orchestrator should count a rate over a window, not a lifetime total, and treat an escape attempt as an
    immediate pause;
  - its `cooldown_period_ms` is declared but never used;
  - quarantine needs a way back (Nick's yes);
  - `revoke ... from SwarmRootOrchestrator` names a program that doesn't exist (the program is `SwarmOrchestrator`);
  - routing by cosine to the single words "algebra" and "calculus" at 0.65 should use MoWM's centroid-and-margin
    rule, with thresholds calibrated per deployment;
  - the fallback solver's answers still go through the VM.
- **CubeLang changes** go to the cubelang repo as their own job, only with Nick's yes.
