# standin/ — the stand-in trunk

**Status:** tooling for a disposable stand-in, 2026-08-30. Nothing here is CubbyLLM.

A small open model (currently `LiquidAI/LFM2.5-2.6B`, chosen by a 2026-08-30 Hub
check — see `docs/research/2026-08-28-oracle-competition-scored.md` §3.2 for why a
stand-in exists at all) is fine-tuned with Unsloth on Colab and served locally as a
GGUF, so the parts of the serve stack that need *some* trunk — the CubeLang program
emitter, the learning gate as a live nightly loop, the chat/grounded-answer surface,
the H-F2 bridge contract — can be built and measured while the 2B CubbyLLM trunk
trains. It is replaced the day the 2B checkpoint exists.

## The two guardrails

1. **Nothing the stand-in measures is a CubbyLLM result.** It says nothing about the
   hybrid backbone, θ=f(c), bounded state, the exact fork, the episodic store or the
   retrieval head. Every number it produces is tagged `[stand-in]`, lives in this
   directory's docs or `TODO.md`, and enters `CUBBYLLM_HYPOTHESES.md` only as a
   pointer. This is the sibling repos' "self-reported numbers presented as
   measurements" failure with a new face — do not let it in.
2. **A swap, not a fork.** The stand-in sits behind the trunk interface
   (`standin/emitter.py::Emitter`); `cubbyllm/` never imports `standin/` (pinned by
   `standin/tests/test_build_emitter_sft.py::test_cubbyllm_never_imports_standin`).
   The 2B checkpoint drops in by implementing the same interface, and nothing above
   it changes.

## What is here

| Path | What |
|---|---|
| `data/build_emitter_sft.py` | Builds the emitter SFT set from `cubemind/sandbox/regen` (arithmetic / role-binding / kernels; dialect shim for the current VM's `ISolver` contract) + our 517 verified chain programs; **re-verifies every program through the real Rust VM**; excludes GSM8K *test* (H-G4's eval) outright; caps role-binding; deterministic 95/5 split; writes a manifest with input hashes and exclusion counts. Output in `data/out/` (gitignored). |
| `emitter.py` | The trunk-facing interface (`Emitter` protocol) and `LlamaServerEmitter`, an HTTP client for a local `llama-server` (llama.cpp, Vulkan build for the AMD card) serving the exported GGUF. |
| `eval_emitter_vm.py` | The **verified** read: runs generated programs through `cubelang.exe`, compares to gold (arithmetic / kernels / chains) or to execution (role-binding). Consumes the Colab notebook's `val_generations.json` or generates live through `LlamaServerEmitter`. |
| `tests/` | Unit pins for the builder's pure helpers + the import guard. `python -m pytest standin/tests -q` |
| `../notebooks/standin_emitter_sft.ipynb` | Unsloth LoRA SFT on Colab, format-level exact-match eval, GGUF export to Drive. |

## The built set (2026-08-30, `data/out/emitter_sft.manifest.json`, output sha256 `5efc50f2…`)

| task | records | train / val | verification on the Rust VM |
|---|---|---|---|
| arithmetic (GSM8K *train*-derived) | 6,148 | 5,819 / 329 | 6,148 execute, **6,148 match gold** |
| kernel (decision / compare / loop / recall) | 593 | 564 / 29 | 593 execute, **593 match gold** |
| role_binding (capped from 32k; 4,992 dups removed) | 4,000 | 3,780 / 220 | 4,000 execute (no gold exists) |
| chain (ours, `ISolve`/`recover`, 1–3 hops) | 517 | 492 / 25 | 517 execute, **517 match gold** — scored on the last hop's function (`hop_N`), not `solve()` |
| **total** | **11,258** | 10,655 / 603 | 0 dropped; VM pass 293 s |

Excluded by rule: **1,057 + 4,220 GSM8K-test-derived programs** (H-G4's eval), 24,997 role-binding programs over the cap. The first build dropped all 180 multi-hop chains as "gold mismatch" because it called `solve()` (hop 1) on every program — fixed (`answer_fn`, pinned by a test); the mistake is the kind the VM pass exists to catch.

## Data facts worth knowing (2026-08-30 audit of `cubemind/sandbox/regen`)

- 38.6k programs in `cubby_aug_v4.txt`: 32k role-binding (`Evt`/`Ev`), 4.2k GSM
  arithmetic, 4×600 kernels; generator sources are gone (`.pyc` only).
- **Every GSM program in the aug corpus is a GSM8K *test* question** (4,228/4,228);
  `multitask_v4_arith_xl.jsonl` adds 6,148 that are exactly GSM8K *train*. Test is
  excluded from SFT; the manifest counts the exclusion.
- The current VM's `ISolver` requires `parse(raw)` + `pure verify(input, output)`;
  the kernels ship both, arithmetic/role-binding do not. With the two-function shim
  (verify is a `return true` stub, as the kernels ship it) arithmetic re-verified
  300/300 to gold and kernels 60/60 in the audit; role-binding executes (no gold).
- The 517 chain programs are the `ISolve`/`recover` dialect — the one the oracle
  actually needs — and are kept whole; role-binding is capped so the emitter does
  not learn to bind ACTION/AGENT and little else.
