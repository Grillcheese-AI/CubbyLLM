# CLAUDE.md — CubbyLLM

Read this first: who does what, where the state of the work lives, and the rules every change keeps. Machine paths,
drives and accounts are in `CLAUDE.local.md` (git-ignored); this repo is public, so no local path goes in a committed
file. This file holds no status (the queue, results and hypotheses each live in one place, below), so it does not go
stale. Earlier versions: `docs/archive/CLAUDE_*.md`.

## Who does what

| | Role | Does | Does not |
|---|---|---|---|
| **Nick** | owner | decides; approves work orders and compute spend; writes the papers' prose | — |
| **Cowork** (Claude in the desktop app) | research and dispatch | literature and design; experiment plans with pre-registered gates; Colab runs in Nick's browser; reads results; records them in the hypotheses and `validation/logs/`; writes work orders; reviews reports; merges `agent-b` | write library code, kernels or notebook builders (it dispatches them) |
| **Claude Code** (Code A and Code B, on Nick's machine) | coders | take work orders; implement, test, commit and push; fill in each order's report | start GPU/Colab runs; change a hypothesis's status; widen an order's scope on their own |

### The work-order loop

Orders live in `docs/orders/`, one file each, `NNNN-slug.md`, written from `docs/orders/_TEMPLATE.md`; their format
and lifecycle are in `docs/orders/README.md`. In short:

1. **Cowork** writes an order: why (with the hypothesis and the logs it rests on), what to build, acceptance tests,
   constraints, what is out of scope. Status `proposed`.
2. **Nick** approves it: status `open`.
3. **Claude Code** takes the order its status names (else the lowest `open` one naming no one), sets `in progress`,
   builds, tests, fills in **Report** (commits, files, tests, what it could not verify), sets `review`, pushes.
4. **Cowork** checks the report and what only a GPU can, records any measured number, and sets `done`, or writes
   what is missing and sets it back to `open`.

`blocked` is valid at any step, with the reason. If an order's argument turns out wrong while building it, **stop and
say so in the report**: the reasoning is the deliverable, not the diff. The live queue: `python tools/queue.py`
(`--for code-a`, `--for code-b`); `/start-order` and `/finish-order` (`.claude/commands/`) walk an order's two ends.

### Two Claude Code sessions (Nick, 2026-11-13)

- **Code A** works in this checkout on `master`. **Code B** works in a second checkout (a `git worktree`; its folder is
  in `CLAUDE.local.md`) on the branch `agent-b`. Each takes only the orders whose status names it; an order that names
  neither is Code A's. Neither edits the other's order files while they are in progress.
- Code B commits on `agent-b`, never on `master`, and never moves `master` from its own checkout. It brings `master`
  in with `git merge master`: no rebase, no force. When its order reaches `review`, it merges `master` in, runs the
  tests and leaves a note; Cowork merges `agent-b` into `master` in this checkout, checks the order there, and pushes.
- Shared machine: at most six of the twelve CPU threads a session; the home card (the GPU) is shared, so say so in the
  inbox before a long run. A timing line is measured while the other session runs nothing, and says so. Both use the
  same Python and data folders (`paths.local.json` is copied) and run scripts from their own checkout's root.
- **The inboxes:** each checkout has its own `docs/inbox.local.md`; Cowork writes to both. Keep it under ~20 KB:
  move handled notes older than two weeks into `docs/inbox_archive_<from>_to_<to>.local.md` beside it.

## What CubbyLLM is

An infrastructure in which the language model is the smallest, most replaceable layer: a trunk proposes, a host
disposes, a VM (CubeLang) decides; facts live in stores and worlds, not in weights; capability gains come from
programs, stores and adapters, not retraining. The kill line everywhere: **0 wrong answers spoken** (a refusal is a
result; a wrong answer is a defect).

The trunk (`cubbyllm/model/`): MinGRU recurrence with sliding-window attention every 3rd layer (RoPE, window 512), an
optional episodic read (`CB_MEM_EVERY`), a 128k byte-level BPE vocabulary, a top-K head; its decode state is fixed-size.

## Where the state lives

| What | Where |
|---|---|
| the queue: what is open, in progress, in review, waiting | `python tools/queue.py`; each order's `status:` line |
| every claim, its evidence and its status | `docs/hypotheses/INDEX.md`, then the group's file; the rules for using them (held-out discipline) are section 0 of `CUBBYLLM_HYPOTHESES.md` |
| the measured results, in one place | `docs/RESULTS.md`; every number links its `validation/logs/` file |
| what Cubby serves (base and adapters) | cubby-stack's built-in `cubby` preset; an adapter serves only after its order's gate and Nick's yes |
| notes between sessions | `docs/inbox.local.md` in each checkout (local, never committed) |

## Rules every change keeps

**Code**
- **Flag-gated.** A new behaviour is off by default (an env `CB_*` flag, or a meta key absent at its default); off,
  outputs are bit-identical and old checkpoints load. Say what the flag does where it is read.
- **A killed flag leaves the code** (Nick, 2026-11-27): once a flag-gated experiment is killed and its order is
  `done`, its runtime hook is removed from `cubbyllm/`; the order's commit keeps the experiment reproducible.
- **Files under 300 lines.** Split rather than grow; a file already over it does not get longer without a reason in
  the commit message.
- **Every module declares `__wiring__`** (`WIRED` only if production code imports it). Only `cubbyllm/ops/` imports
  grilly. No `model.py` / `models/` anywhere. Torch imports stay lazy so `import cubbyllm` is torch-free.
- **Tests pin the claim**, not the implementation: values against a reference, gradients, decode = forward, off =
  unchanged. A GPU kernel also gets a CPU path under `TRITON_INTERPRET=1`.
- **Custom ops** (`torch.library.custom_op`): the fake returns the kernel's strides; `opcheck` non-contiguous inputs.
- **Wire it or say so.** A component nothing calls is a trap; its docstring states its wiring.

**Evidence**
- Validate before committing to a design: a small, falsifiable experiment with a kill criterion. A kill criterion
  never treats a refusal and a wrong answer as the same event.
- Every number in a doc links the `validation/logs/` file that produced it (dated name). Status tags stay honest
  (VERIFIED / PLAUSIBLE / SPECULATIVE; OPEN / KILLED / PARKED); toy-scale is not scale.
- A new architectural claim goes into its group's file in `docs/hypotheses/` as claim, source, validation, kill
  criterion, by Cowork, from a report or a run; `INDEX.md` gets its row.

**Boundaries**
- No external LLM on Cubby's serving path, ever (external models only build datasets). Keys live in git-ignored
  files (`validation/.env`); never in a command, a log or a commit. Nothing from the private BlockWorld repo comes in.
- **Where code goes (Nick, 2026-10-22):** what makes Cubby think goes in `cubbyllm/` (Apache-2.0); the product layer
  (server, board, harness, vault, ledger) in `cubby_serve/` (BSL 1.1); `standin/` holds only the GGUF stand-in and its
  data builders, and gets no new feature. `cubbyllm/` never touches the network: what fetches lives in `cubby_serve/`
  and hands `cubbyllm/` a fetch function. Data builders go to `validation/`. `tests/test_package_rules.py` pins these
  lines. An older order's `standin/...` path means that module's new home (`tools/moves_0031.json`).
- No local drive paths, venv names or account names in committed files. A committed file names a folder
  (`$CUBBY_MODELS`, `$CUBBY_DATASETS`, `$CUBBY_DRIVE`, ...); `cubbyllm/local_paths.py` resolves it from the
  environment, then the git-ignored `paths.local.json`. `tests/test_local_paths.py` fails on a local path, and
  `tools/scrub_local_paths.py --write <files>` rewrites them.
- No permanent deletion: move things aside (`_discard/`, `docs/archive/`). Large external downloads need Nick's yes.
- **Sources are internal** (Nick, 2026-10-22): no reply to a regular user names a source, author, title, site or
  dataset, or quotes a passage. The record keeps every source, and the audit view (owner or auditor) shows it
  (`docs/WORLDS.md`). Cubby paraphrases in its own words: no run of 8+ words from a source, no imitation of a style.

## Working here

- **Tests:** `python -m pytest tests` from the root (`-q` is already set; a second one hides the summary). Torch here
  is a CPU build: Triton and flex tests skip, and run on a GPU via `notebooks/kernels_check.ipynb`.
- **Notebooks are generated:** edit `notebooks/_build_<name>.py`, run it, commit both; never hand-edit an `.ipynb`.
- **Shards:** `<name>.u32` (+ `.pos.i16`, `.role.u8`) with `_sizes.json`, `_align.json`, `_holdout.json`.
- **Logs:** `validation/logs/<what>_<date>.<ext>`. Files starting `_` there are scratch (untracked).
- **Commits:** the subject says what changed, the body why, with the numbers; `[NNNN] ...`; long messages by
  `git commit -F`. Run git on the Windows side, never through a mounted copy of the repo (it leaves `index.lock`).

## Where to look

| | |
|---|---|
| `docs/orders/`, `docs/RESULTS.md`, `docs/hypotheses/` | the queue, the results, the claims |
| `docs/WORLDS.md`, `docs/COLAB.md` | every dataset into a world; running on Colab GPUs |
| `docs/WORK_ORDERS.md`, `docs/PATH.md`, `docs/archive/` | history: the serve-stack era's specs and lessons; retired docs |
| `cubbyllm/model/` | the trunk: `assembly.py`, `backbone/`, `spatial.py`, `recall/`, `grow.py` |
| `cubbyllm/reasoning/`, `worlds/`, `sources/` | the planner, the disposer, the walk, the ask and step loops; the stores; the sources' logic |
| `cubbyllm/affect/`, `speech/`, `emit/`, `plugins/`, `brain.py` | feeling; the chat turn and identity; the seats (the 450M on grilly2); the plugins; CubbyBrain |
| `cubby_serve/` | the server (BSL 1.1): api, board, harness, ledger, keys, `sources/` (the fetchers) |
| `validation/` | experiments, training, probes, data builders (never imported by `cubbyllm/`) |
| `standin/` | the GGUF stand-in (the control arm) and the old names as shims (order 0091 retires them) |
