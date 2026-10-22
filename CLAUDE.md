# CLAUDE.md — CubbyLLM

Read this first. It says who does what, where the work stands, and the rules every change keeps.
Machine-specific paths, drives and accounts are in `CLAUDE.local.md` (git-ignored, loaded with this
file); this repo is public, so no local path goes in a committed file.

The previous version of this file (the validation-campaign history) is
`docs/archive/CLAUDE_2026-10-21.md`. `CUBBYLLM_HYPOTHESES.md` is the authoritative record of every
architectural claim, its evidence and its status; this file is a map, not a substitute.

## Who does what

| | Role | Does | Does not |
|---|---|---|---|
| **Nick** | owner | decides; approves work orders and compute spend; writes the papers' prose | — |
| **Cowork** (Claude in the desktop app) | research and dispatch | literature and design; experiment plans with pre-registered gates; Colab runs in Nick's browser; reads results; records them in `CUBBYLLM_HYPOTHESES.md` and `validation/logs/`; writes work orders; reviews reports; runs the GPU checks | write library code, kernels or notebook builders (it dispatches them) |
| **Claude Code** (in this repo, on Nick's machine) | coder | takes work orders; implements, tests, commits and pushes; fills in each order's report | start GPU/Colab runs; change a hypothesis's status; widen an order's scope on its own |

### The work-order loop

Orders live in `docs/orders/`, one file each, `NNNN-slug.md`, written from `docs/orders/_TEMPLATE.md`.
Their format and lifecycle are in `docs/orders/README.md`. In short:

1. **Cowork** writes an order: why (with the hypothesis and the logs it rests on), what to build,
   acceptance tests, constraints, what is out of scope. Status `proposed`.
2. **Nick** approves it: status `open`.
3. **Claude Code** takes the lowest-numbered `open` order unless Nick names one; sets `in progress`
   (commit); builds; runs the tests; fills in **Report** (commits, files, test summary, what it could
   not verify on this machine, questions); sets `review`; commits and pushes.
4. **Cowork** checks the report and what only a GPU can (Colab), records any measured number, and
   sets `done` — or writes what is missing and sets it back to `open`.

`blocked` is a valid status at any step, with the reason in the file. If the argument for an order
turns out wrong while building it, **stop and say so in the report**: the reasoning is the
deliverable, not the diff. List the queue: `grep -H "^status:" docs/orders/0*.md`.

## What CubbyLLM is

An infrastructure in which the language model is the smallest, most replaceable layer: a trunk
proposes, a host disposes, a VM (CubeLang) decides; facts live in stores and worlds, not in
weights; capability gains come from programs, stores and adapters, not retraining. The kill line
everywhere: **0 wrong answers spoken** (a refusal is a result; a wrong answer is a defect).

The trunk (`cubbyllm/model/`): a hybrid backbone — MinGRU recurrence, sliding-window attention every
3rd layer (RoPE, window 512) — with an episodic memory read (`CB_MEM_EVERY`), a context router, a
hybrid 128k byte-level BPE vocabulary and a top-K retrieval head. Decode state is fixed-size (a
recurrent vector per layer plus a ring of `window` keys), not a growing KV cache.

## Where things stand (2026-10-21)

- **Bases.** `base450m_cont` — the 450M (D1024, L32, 4 heads) continued past its first run; the
  reference trunk for every experiment below. A grown ~1.3B with the episodic read every 2 layers
  (`notebooks/grow_mem_continue.ipynb`): round 1 trained, round 2 built and not run.
- **Spatial link / loci tokens (H-S12, H-K10; called hypertokens until 2026-10-22).** `cubbyllm/model/spatial.py`: places bound into the
  input (`e ⊛ S(p)`), a causal scene per document, a gated read mid-trunk; roles (a thing WRITES, a
  question ASKS), `query="placed"`, `bind="writes"`. Three 450M runs (`notebooks/spatial_facts_450m.ipynb`):
  villages read 0.793 → 0.776 → **0.953** exact (run 3); facts 800 tokens past the window read
  0.021 in run 3, though the read held them (0.965 whole answers among the document's writers). Order 0001
  added a copy path (`cubbyllm/model/spatial_copy.py`, `CB_SPATIAL_COPY=ring`): the read cleaned up
  against a fixed ring of the last 256 written ids, mixed into the output with a read-aware gate. Run 4: far
  facts **0.761**, villages 0.953, but the whole model drifted on text. Run 3's trunk with run 4's copy
  (~1k parameters) reads **0.844**, with its text unchanged. Next: train the copy alone (order 0005).
- **Speed.** Triton kernels: the fused MinGRU recurrence (`CB_MINGRU_TRITON=1`, 72 → 18 ms a step on
  the G4) and banded flash attention (`CB_ATTN_TRITON=1`; exact, 3.7× the dense mask on an A100; not
  yet measured in a G4 step). Flex attention with tiles for 99 KB cards (`CB_FLEX_OPTS`).
  `validation/speed_check.py` measures the variants on the run's own card and the run takes the
  fastest: run 3 trained at **40.6% MFU** (run 2: 32.2%). GEMMs already run at ~75% of peak; the
  rest of the step is where the time is.
- **Open hypotheses (Group K, §9d).** K1 fact map, K2 surprise-gated writing, K3 continual learning,
  K4 clock, K5 BlockWorld hidden-rule bench, K6 binding heads, K7 exposure law, K8 holographic state,
  K9 flow-map jumps, K10 loci tokens. Read their entries before touching the matching code.
- **Serve stack.** `standin/` (an external GGUF stand-in in the emitter and talk seats) stays the A/B
  control until base450m takes both seats; `docs/WORK_ORDERS.md` holds that era's specs and its
  measured lessons (read its "Before you touch anything").
- **Known red:** `tests/test_guards.py::test_every_module_declares_wiring` fails on
  `cubbyllm.spatial` and `cubbyllm.reasoning.step_loop` (order 0002, proposed).
- **Queue (2026-10-22):** 0001 done; 0004 (rename to loci tokens) open; 0002, 0003, 0005 proposed.

## Rules every change keeps

**Code**
- **Flag-gated.** A new behaviour is off by default (an env `CB_*` flag, or a meta key absent at
  its default); off, outputs are bit-identical and old checkpoints load. Say what the flag does
  where it is read.
- **Files under 300 lines.** Split rather than grow; a file already over it does not get longer
  without a reason in the commit message.
- **Every module declares `__wiring__`** (`WIRED` only if production code imports it). Only
  `cubbyllm/ops/` imports grilly. No `model.py` / `models/` anywhere. Torch imports stay lazy so
  `import cubbyllm` is torch-free.
- **Tests pin the claim**, not the implementation: values against a reference, gradients, decode =
  forward, off = unchanged. A GPU kernel also gets a CPU path under `TRITON_INTERPRET=1`.
- **Custom ops** (`torch.library.custom_op`): the fake must return the strides the kernel returns —
  check with `opcheck` on non-contiguous inputs too (a transposed view broke the compiled band
  kernel on the G4 while every contiguous test passed; `e1185e6`).
- **Wire it or say so.** A component nothing calls is a trap; its docstring states its wiring.

**Evidence**
- Validate before committing to a design: a small, falsifiable experiment with a kill criterion. A
  kill criterion never treats a refusal and a wrong answer as the same event.
- Every number in a doc links the `validation/logs/` file that produced it (dated name). Status
  tags stay honest (VERIFIED / PLAUSIBLE / SPECULATIVE; OPEN / KILLED); toy-scale is not scale.
- A new architectural claim goes into `CUBBYLLM_HYPOTHESES.md` as claim → source → validation →
  kill criterion — by Cowork, from a report or a run.

**Boundaries**
- No external LLM on Cubby's serving path, ever (external models only build datasets). Keys live in
  git-ignored files (`validation/.env`); never in a command, a log or a commit.
- The BlockWorld repo is private: nothing from it comes into this public repo.
- No local drive paths, venv names or account names in committed files.
- No permanent deletion: move things aside (`_discard/`, `docs/archive/`) instead.
- Downloads of large external data need Nick's yes.

## Working here

- **Tests:** `python -m pytest tests` from the repo root. `pyproject.toml` already sets `addopts = -q`;
  adding `-q` again gives `-qq`, which prints no summary line. On Nick's machine torch is a CPU build
  without Triton or flex: those tests skip there and run on a GPU via `notebooks/kernels_check.ipynb`.
- **Training and measurement:** `validation/train_base.py` (configured entirely by `CB_*` env vars;
  resumes from its newest checkpoint), `validation/profile_base.py` (device time by kind),
  `validation/speed_check.py`, `validation/bench_kernels.py`, `validation/exp_hypertoken_recall.py`,
  `validation/diag_fact_read.py`.
- **Notebooks are generated:** edit `notebooks/_build_<name>.py`, run it to regenerate
  `notebooks/<name>.ipynb`, commit both; never hand-edit an `.ipynb`. Colab cells stream
  subprocess output with `Popen` (a captured `subprocess.run` shows nothing until it ends).
- **Shards** (the token cache): `<name>.u32` (+ `.pos.i16` places, `.role.u8` roles) with
  `_sizes.json`, `_align.json`, `_holdout.json`; the Corpus skips sync-tool duplicates named
  `<name> (1)`. Builders: `validation/make_spatial_shard.py`, `validation/make_fact_shard.py`.
- **Logs:** `validation/logs/<what>_<date>.<ext>`. Files starting `_` there are scratch (untracked).
- **Commits:** to `master`, pushed. Subject says what changed; body says why and carries the
  measured numbers; reference the order (`[0001] ...`). Long messages from a file (`git commit -F`).
  Run git on the Windows side, never through a mounted copy of the repo (it leaves `index.lock`).

## Where to look

| | |
|---|---|
| `CUBBYLLM_HYPOTHESES.md` | every claim, its evidence, its status (Group K at §9d) |
| `docs/orders/` | the work queue (this loop) |
| `docs/WORK_ORDERS.md`, `docs/PATH.md` | the serve-stack era's specs and lessons |
| `docs/COLAB.md` | running on Colab GPUs |
| `docs/superpowers/specs/` | design specs (whole-brain, sensory cortex, episodic memory, …) |
| `cubbyllm/model/` | trunk: `assembly.py`, `backbone/` (hybrid, MinGRU, kernels), `spatial.py`, `recall/`, `grow.py` |
| `validation/` | experiments, training, probes (never imported by `cubbyllm/`) |
| `standin/` | the stand-in serve stack (control arm) |
