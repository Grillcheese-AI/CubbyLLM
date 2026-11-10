# 0065 - World calls from the VM: the host answers them with MoWM and records them
status: open (queue: after cubelang P5-1 and P5-2's first three calls; with 0064's writers)
written: 2026-11-10 by Cowork — approved: 2026-11-10 by Nick ("we also need to code the missing function like to create a world and the other ones"; worlds owned by MoWM) — hypothesis: MoWM routing (route or spawn), H-E15

## Why
CubeLang programs cannot touch MoWM's worlds today. SPECIALIZE and 24 other opcodes compute nothing, and the world
functions in Nick's drafts exist nowhere. Nick chose to keep the worlds in MoWM: the VM calls the host, the host
answers with MoWM, and both sides record the call. Three pieces: the channel and the calls (cubelang P5), the
handlers (mowm, `docs/specs/2026-11-10-vm-world-calls.md`), and this order, the host in between.

**What it brings, in plain words:** Cubby's world decisions (which world answers, when a new one is created, when
one splits) run as programs and are recorded like answers, so each can be replayed and checked.
- **Over a conventional model:** routing between experts happens inside the network, with no record of why.

## Build
1. **The relay.** CubbyLLM's `run-proto` client answers `HostCall` messages for the `mowm` module by calling MoWM's
   handlers in process, and returns `HostResult` with the world state's hash.
2. **One world tree per tenant,** and one per story world (0060), never shared. A call names its tenant's tree; a
   call reaching another tenant's tree is refused.
3. **The records.** Each `route` call becomes a `mowm:RoutingDecision`, each `split` a `mowm:WorldSplit`, written by
   the host from the call and its result (0064). The thresholds carry their calibration ids, or are marked as code
   defaults.
4. **Replay.** A run's recorded calls replay without MoWM attached and give the same report hash (P5-1).
5. **The first programs,** written against the compiler (0064 step 5, with `qc_decision.cube` as the pattern):
   route-or-spawn, and the split check. Each passes `cubelang check --strict` and runs on the VM in a test.

## Acceptance (pre-registered)
- **Records:** every `route` and `split` call in a test run gives a record that conforms to CheezML's shapes, and
  matches the call's result field by field.
- **Rules:** a split at z_max is refused and recorded as refused; no call deletes a world.
- **Replay:** 100 recorded runs replay to the same report hash.
- **Isolation:** a call to another tenant's tree is refused, in a test.
- **Kill:** a record that conforms but disagrees with the call it came from. The relay stops until it is fixed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No neural model on the routing path. SCORE and the other learned opcodes are proposals, never certified alone.
- CubeLang changes go to the cubelang repo and MoWM changes to the mowm repo, each with Nick's yes (given for P5
  and the handlers). The main cubelang checkout's `target/release` binary is never overwritten.
- No local path in a committed file.

## Out of scope
- Containers, `vm.wire` and `vm.bridge` (cubelang P3-1).
- P5-6's opcodes that have no semantics yet.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
