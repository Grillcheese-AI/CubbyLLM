# CubbyLLM package skeleton — design spec

Date: 2026-07-23
Status: approved (design), pending spec review
Scope: **package skeleton only** — interface stubs + mirrored test tree. No
corpus prep, no tokenizer build, no working model, no training run. Those are
separate later cycles (see "Explicitly out of scope").

## 1. Purpose

Stand up the `cubbyllm/` package structure so every decision the validation
campaign settled has a home, *before* any of it is implemented. The skeleton's
one job: make the validated architecture the path of least resistance, and make
each of the sibling repos' four documented failure patterns
(`cubby-model-environment-map.md` §4) impossible to reproduce by accident.

Grounding decisions this skeleton encodes (all from `VALIDATION_REPORT.md` /
`CUBBYLLM_HYPOTHESES.md`, 2026-07-23):
- **θ=f(c) is the unifying mechanism** (H0) — and works *only* hardened; naive
  generation was worse than baseline. Context *inference*, not generation, is
  the bottleneck (H0b), so the context source must be offline-pretrained and
  frozen at inference.
- **Binding algebra = grilly `BlockCodeOps`** (H-B5, user decision).
- **Vocab = hybrid** — fixed core + retrieval output head + gated dynamic tail
  (H-C6, user decision). Retrieval head is mandatory, not optional (H-C3).
- **Backbone is OPEN** (H-D1: MinGRU leads at small scale, not chosen). The
  skeleton must NOT smuggle in a backbone choice.
- **Real unbind does not exist yet** (H-B3) — it is from-scratch, VM/binding-side.

## 2. Interface design: thesis-first (approach A)

θ=f(c) is made a type-level constraint, not a convention, so a forward path that
ignores context is a type error and the "structural bypass" the campaign warned
about is hard to write by accident.

### core/ — the thesis abstractions

- **`context.py`**
  - `Context` — frozen dataclass: `vector: Tensor` (the c), `source_id: str`,
    `confidence: Tensor | None`. The carrier threaded through every forward path.
  - `ContextSource` (Protocol): `infer(x) -> Context`; property
    `is_frozen -> bool`. Encodes H0b: the contract expects an offline-pretrained,
    frozen-at-inference source. A runtime assert in eval flags an unfrozen source.
- **`generation.py`**
  - `GeneratedParams` — carrier for generated weights (opaque to consumers).
  - `ParameterGenerator` (Protocol): `generate(ctx: Context) -> GeneratedParams`.
    The hypernetwork seam.
  - `Hardener` (Protocol): `snapshot(gen, contexts) -> None`;
    `penalty(gen) -> Tensor`. The anti-drift countermeasure H0 proved essential —
    first-class, never buried inside a training loop.
- **`protocols.py`**
  - `Generable` (Protocol): `forward_generated(x: Tensor, ctx: Context) -> Tensor`.
    Memory, binding, and vocab heads all implement it.
  - `Wiring` enum: `WIRED | STANDALONE`. Every module sets module-level
    `__wiring__`. A helper `declare_wiring(status)` for clarity.
  - **Two orthogonal axes, kept explicitly separate** (this is precisely the
    distinction the sibling repos collapsed): `__wiring__` declares *intent* —
    is this component designed to sit in the default forward path (`WIRED`) or
    to be a standalone attachment (`STANDALONE`)? A stub body raising
    `NotImplementedError` + a skipped test declares *build status* — is it
    implemented yet? So `memory/base.py` is `__wiring__ = WIRED` (intent: yes,
    default path) AND `NotImplementedError` (status: not built). A component is
    only ever "done" when its intent is WIRED, its body is real, `assembly.py`
    actually calls it, AND its test is un-skipped and green — never on the
    strength of `__wiring__` alone.
- **`config.py`** — `CubbyConfig` dataclass (d_model, n_layers, vocab sizes,
  ctx_dim, seeds…), project constants, the canonical seed. No model logic.
- **`probing.py`** — `WrongContextProbe` (Protocol): given a `Generable`, a right
  `Context`, and wrong `Context`s, return the P5 delta. The validation seam,
  in-package so it can gate training (see training/probing.py).

### ops/ — the single grilly seam

- **`vsa.py`** — the ONLY module permitted to import grilly. Wraps
  `grilly.experimental.vsa.block_ops.BlockCodeOps` behind a 3-tier fallback
  (Vulkan bridge → grilly python → numpy), mirroring cubemind's proven
  `BlockCodes` pattern. Public surface: `bind/unbind/bundle/similarity/
  codebook` over (k, l) block codes. Everything else imports `cubbyllm.ops`.

### model/ — the components

- **`backbone/base.py`** — `Backbone` (Protocol): `forward(tokens) -> hidden`.
  Interface ONLY. Explicitly the open decision; no implementation, no default.
- **`memory/base.py`** — `MemoryLayer(Generable)`: hardened θ=f(c) memory. Body
  raises `NotImplementedError`. `__wiring__ = WIRED` (it is meant to be in the
  default path once built).
- **`binding/base.py`** — `BindingHead` (bind/bundle over ops.vsa) and `Unbinder`
  (Protocol) — the from-scratch unbind. Stub bodies.
- **`vocab/embedding.py`** — `EmbeddingSource` (Protocol): the hybrid — a static
  core embedding plus an optional generated tail (a `ParameterGenerator`).
- **`vocab/output_head.py`** — `RetrievalHead` (Protocol): ANN top-K candidate
  retrieval + local softmax (the mandatory full-softmax bypass, H-C3).
- **`assembly.py`** — `CubbyModel`: the single place the forward path is
  assembled. Pulls a `Context` from a `ContextSource`, threads it through
  backbone → memory (Generable) → binding → vocab head. Body raises
  `NotImplementedError`; its *signature* fixes how context flows.

### bridges/ — first-class cross-repo interface

- **`world_model.py`** — `WorldModelBridge` (Protocol) carrying (a) a context
  embedding both directions, (b) novelty events, (c) specialist handles — the
  first-class interface H-F2 found the current narrow duck-typed bridges lack.

### training/ — pipeline seams (stubs)

- **`data.py`** — `DataPipeline` (Protocol): manifest-pinned corpus source
  (H-G3's reproducibility lesson) + tokenizer handle. Interface only.
- **`loop.py`** — `TrainLoop` stub.
- **`probing.py`** — `P5WrongContextProbe`: concrete wiring of core.probing into
  a training gate, so the campaign's P5 discipline is available from day one.

## 3. Data flow (the assembled forward path, once built)

```
tokens
  → ContextSource.infer(x)  ─────────────► Context c   (frozen at inference)
  → Backbone.forward(tokens) ───────────► hidden h
  → MemoryLayer.forward_generated(h, c) ─► h'          (θ=f(c), hardened)
  → BindingHead(h', …) over ops.vsa ────► bound repr
  → EmbeddingSource / RetrievalHead(c) ─► logits over top-K candidates
```
Context `c` is an explicit argument at every generated stage — omitting it does
not type-check. `Hardener` snapshots/penalizes the `ParameterGenerator`(s)
across previously-seen contexts during sequential training.

## 4. Testing, tooling, enforcement

- **`tests/` mirrors `cubbyllm/` 1:1.** One placeholder test per module: imports
  it, asserts the declared interface exists and `__wiring__` is set. Stub-bodied
  modules get an `xfail`/`skip` marked "not implemented yet" so the count of
  unfinished pieces is always visible in the test run.
- **Three CI-guard tests** (anti-sprawl rules as executable checks):
  - `test_no_model_naming_collision` — fails on a stray `model.py` or `models/`.
  - `test_every_module_declares_wiring` — every `cubbyllm/**` module sets
    `__wiring__`.
  - `test_ops_is_sole_grilly_importer` — only `cubbyllm/ops/**` imports grilly.
- **`pyproject.toml`** — import name `cubbyllm`, pytest configured. No new heavy
  deps pinned (torch/numpy/grilly assumed present, as in `validation/`).

## 5. How this blocks the four sibling failure patterns

| Sibling failure | Blocked by |
|---|---|
| Built but never wired in | `__wiring__` declaration + `test_every_module_declares_wiring`; stubs are loud `NotImplementedError` + visible skipped tests |
| Docs describing mechanisms code lacks | Interfaces ARE the doc; no prose-only mechanism can exist without a typed seam |
| Structural duplication / naming collision | `test_no_model_naming_collision`; single `ops/` grilly seam via `test_ops_is_sole_grilly_importer` |
| Self-reported numbers as measurements | No numbers in the skeleton; validation numbers live in `validation/logs/` (separate, already established) |

## 6. Explicitly out of scope (later cycles, each its own spec)

- Corpus cleanup / NSFW-scope decision / manifest pin (H-G3, the pre-training gate).
- Tokenizer build (128k–256k BPE — G1 showed ~5 min).
- Any real component implementation (backbone choice, memory, binding, heads).
- The training run itself.

The skeleton is deliberately inert: it compiles and its tests run (mostly
skipped), but it trains nothing yet. That is the intended end state of this cycle.

## 7. Success criteria

- `pip install -e .` (or equivalent) succeeds; `import cubbyllm` works.
- `pytest` runs green: guard tests pass, per-module smoke tests pass or skip
  with an explicit "not implemented" reason.
- Every module declares `__wiring__` and a docstring stating Wired vs Standalone.
- No component is implemented; every stub raises `NotImplementedError`.
- A reader can see the full validated architecture from the interfaces alone,
  and cannot write a context-ignoring forward path without a type error.
