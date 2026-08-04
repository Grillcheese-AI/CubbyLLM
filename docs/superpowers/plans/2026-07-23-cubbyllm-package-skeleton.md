# CubbyLLM Package Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `cubbyllm/` package as thesis-first interface stubs with a mirrored `tests/` tree, so every validated architecture decision has a typed home and the sibling repos' four failure patterns are each blocked by an executable guard — implementing nothing.

**Architecture:** θ=f(c) is encoded as type-level protocols (`Context`, `ContextSource`, `ParameterGenerator`, `Hardener`, `Generable`) so a context-ignoring forward path is a type error. Every module declares `__wiring__` (intent) separately from build status (stub bodies raise `NotImplementedError`). A single `ops/` module is the only grilly importer. Tests assert interfaces + guards; stub behavior is `xfail`/`skip`.

**Tech Stack:** Python 3.12, pytest, torch/numpy (present, used only in type annotations here via `TYPE_CHECKING`), grilly (imported only in `ops/`).

## Global Constraints

- Import name: `cubbyllm` (lowercase, no hyphen).
- Every `cubbyllm/**` module sets module-level `__wiring__ = Wiring.WIRED | STANDALONE` and a docstring line stating Wired vs Standalone.
- Only `cubbyllm/ops/**` may import `grilly`.
- No `model.py` file and no `models/` directory anywhere.
- No component is implemented: every non-protocol method body raises `NotImplementedError`. Protocols use `...`.
- No torch import at module top level — use `from __future__ import annotations` + `TYPE_CHECKING` so `import cubbyllm` stays dependency-light.
- Repo is NOT git-tracked: the "Commit" step in each task is replaced by "run the full suite green as a checkpoint." No `git` commands.
- Binding algebra = grilly `BlockCodeOps` (k, l block codes). Backbone is OPEN — interface only, no implementation, no default.

---

## File Structure

```
pyproject.toml                      package metadata, pytest config
cubbyllm/__init__.py                exports Wiring, Context, version
cubbyllm/core/__init__.py
cubbyllm/core/protocols.py          Wiring enum, declare_wiring, Generable
cubbyllm/core/context.py            Context, ContextSource
cubbyllm/core/generation.py         GeneratedParams, ParameterGenerator, Hardener
cubbyllm/core/config.py             CubbyConfig, SEED
cubbyllm/core/probing.py            WrongContextProbe
cubbyllm/ops/__init__.py
cubbyllm/ops/vsa.py                 BlockCodeVSA facade over grilly (sole seam)
cubbyllm/model/__init__.py
cubbyllm/model/backbone/__init__.py
cubbyllm/model/backbone/base.py     Backbone protocol (OPEN)
cubbyllm/model/memory/__init__.py
cubbyllm/model/memory/base.py       MemoryLayer(Generable)
cubbyllm/model/binding/__init__.py
cubbyllm/model/binding/base.py      BindingHead, Unbinder
cubbyllm/model/vocab/__init__.py
cubbyllm/model/vocab/embedding.py   EmbeddingSource
cubbyllm/model/vocab/output_head.py RetrievalHead
cubbyllm/model/assembly.py          CubbyModel
cubbyllm/bridges/__init__.py
cubbyllm/bridges/world_model.py     WorldModelBridge
cubbyllm/training/__init__.py
cubbyllm/training/data.py           DataPipeline
cubbyllm/training/loop.py           TrainLoop
cubbyllm/training/probing.py        P5WrongContextProbe
tests/ (mirrors cubbyllm/ 1:1, one test module per source module)
tests/test_guards.py                the three anti-sprawl CI guards
```

Convention for every source module: `from __future__ import annotations` first
line after docstring; `from cubbyllm.core.protocols import Wiring`; set
`__wiring__`; `TYPE_CHECKING` block for `torch.Tensor`.

---

### Task 1: Package scaffolding, pyproject, and the three CI guards

**Files:**
- Create: `pyproject.toml`, `cubbyllm/__init__.py`, `cubbyllm/core/__init__.py`, `cubbyllm/core/protocols.py`
- Test: `tests/test_guards.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `Wiring` (Enum: `WIRED`, `STANDALONE`), `declare_wiring(status) -> Wiring`, `Generable` (Protocol: `forward_generated(self, x, ctx) -> Tensor`). `cubbyllm.__version__: str`.

- [ ] **Step 1: `pyproject.toml`** — name `cubbyllm`, requires-python >=3.12, pytest config with `testpaths = ["tests"]`, and register `xfail_strict = false`.
- [ ] **Step 2: `cubbyllm/core/protocols.py`**

```python
"""Core protocols. Wired: STANDALONE (pure type definitions, no forward path)."""
from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor


class Wiring(Enum):
    WIRED = "wired"            # intent: belongs in the default forward path
    STANDALONE = "standalone"  # intent: standalone attachment


def declare_wiring(status: Wiring) -> Wiring:
    """Explicit re-export helper so `__wiring__ = declare_wiring(...)` reads clearly."""
    return status


@runtime_checkable
class Generable(Protocol):
    """Anything whose active weights are generated from context (theta=f(c))."""
    def forward_generated(self, x: "Tensor", ctx: "Context") -> "Tensor": ...


__wiring__ = Wiring.STANDALONE
```

(Note: `Context` referenced in annotation only; lazy via `from __future__ import annotations`.)

- [ ] **Step 3: `cubbyllm/__init__.py`** — `__version__ = "0.0.0"`; re-export `Wiring`, `declare_wiring`; lazy re-export `Context` (import inside a function or at bottom to avoid cycles — import `Context` from `.core.context` at module end).
- [ ] **Step 4: `tests/test_guards.py`** — the three guards, walking `cubbyllm/`:

```python
import ast, importlib, pathlib, pkgutil
import cubbyllm
ROOT = pathlib.Path(cubbyllm.__file__).parent

def _modules():
    for m in pkgutil.walk_packages([str(ROOT)], prefix="cubbyllm."):
        if not m.ispkg:
            yield m.name

def test_no_model_naming_collision():
    bad = [p for p in ROOT.rglob("*") if p.name == "model.py" or (p.is_dir() and p.name == "models")]
    assert not bad, f"forbidden model.py/models present: {bad}"

def test_every_module_declares_wiring():
    missing = []
    for name in _modules():
        mod = importlib.import_module(name)
        if not hasattr(mod, "__wiring__"):
            missing.append(name)
    assert not missing, f"modules missing __wiring__: {missing}"

def test_ops_is_sole_grilly_importer():
    offenders = []
    for py in ROOT.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        if "grilly" in src and "ops" not in py.parts:
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in getattr(node, "names", [])] + [getattr(node, "module", "") or ""]
                    if any(n and n.startswith("grilly") for n in names):
                        offenders.append(str(py))
    assert not offenders, f"non-ops modules importing grilly: {offenders}"
```

- [ ] **Step 5: Run** `pytest tests/test_guards.py -v`. Expected: `test_no_model_naming_collision` PASS, `test_every_module_declares_wiring` PASS (only core.protocols exists so far, it declares wiring), `test_ops_is_sole_grilly_importer` PASS.
- [ ] **Step 6: Checkpoint** — full suite green.

---

### Task 2: `core/` thesis abstractions

**Files:**
- Create: `cubbyllm/core/context.py`, `cubbyllm/core/generation.py`, `cubbyllm/core/config.py`, `cubbyllm/core/probing.py`
- Test: `tests/core/test_context.py`, `tests/core/test_generation.py`, `tests/core/test_config.py`, `tests/core/test_probing.py`

**Interfaces:**
- Consumes: `Wiring` from Task 1.
- Produces:
  - `Context(vector, source_id, confidence=None)` — frozen dataclass.
  - `ContextSource` (Protocol): `infer(x: Tensor) -> Context`; property `is_frozen: bool`.
  - `GeneratedParams(weights, meta=None)` — frozen dataclass carrier.
  - `ParameterGenerator` (Protocol): `generate(ctx: Context) -> GeneratedParams`.
  - `Hardener` (Protocol): `snapshot(gen: ParameterGenerator, contexts: list[Context]) -> None`; `penalty(gen: ParameterGenerator) -> Tensor`.
  - `CubbyConfig` — frozen dataclass: `d_model: int, n_layers: int, ctx_dim: int, vocab_core: int, vocab_dynamic: bool, block_k: int = 80, block_l: int = 128`. `SEED = 0xC0DEB00C`.
  - `WrongContextProbe` (Protocol): `probe(target: Generable, x: Tensor, right: Context, wrong: list[Context]) -> dict[str, float]`.

- [ ] **Step 1: Write `context.py`** — `Context` frozen dataclass (`vector: Tensor`, `source_id: str`, `confidence: Tensor | None = None`), `ContextSource` Protocol with `infer` + `is_frozen`. Set `__wiring__ = Wiring.STANDALONE`. Docstring states STANDALONE (types + seam).
- [ ] **Step 2: Write `generation.py`** — `GeneratedParams`, `ParameterGenerator`, `Hardener` as above. `__wiring__ = Wiring.STANDALONE`.
- [ ] **Step 3: Write `config.py`** — `CubbyConfig` frozen dataclass + `SEED`. `__wiring__ = Wiring.STANDALONE`.
- [ ] **Step 4: Write `probing.py`** — `WrongContextProbe` Protocol. `__wiring__ = Wiring.STANDALONE`.
- [ ] **Step 5: Tests** — `tests/core/test_context.py` etc. Each: import the module; assert the class/protocol exists; construct the concrete dataclasses with dummy values (no torch tensor needed — pass a plain list/object for `vector`, since annotations are lazy) and assert field access; assert `hasattr(mod, "__wiring__")`.

```python
# tests/core/test_context.py
from cubbyllm.core.context import Context, ContextSource
from cubbyllm.core.protocols import Wiring
import cubbyllm.core.context as mod

def test_context_is_frozen_dataclass():
    c = Context(vector=[0.0], source_id="test", confidence=None)
    assert c.source_id == "test"
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.source_id = "x"

def test_wiring_declared():
    assert mod.__wiring__ in (Wiring.WIRED, Wiring.STANDALONE)
```

- [ ] **Step 6: Run** `pytest tests/core -v`. Expected: all PASS.
- [ ] **Step 7: Checkpoint** — full suite green (guards still pass now that core modules all declare wiring).

---

### Task 3: `ops/vsa.py` — the sole grilly seam

**Files:**
- Create: `cubbyllm/ops/__init__.py`, `cubbyllm/ops/vsa.py`
- Test: `tests/ops/test_vsa.py`

**Interfaces:**
- Produces: `BlockCodeVSA` class with `bind`, `unbind`, `bundle`, `similarity`, `codebook` methods (all raise `NotImplementedError`), and a `backend() -> str` reporting which of the 3 tiers is active. `__wiring__ = Wiring.WIRED`.

- [ ] **Step 1: Write `ops/vsa.py`** — `BlockCodeVSA` with the method signatures over `(k, l)` block codes, docstring naming the 3-tier fallback (grilly Vulkan bridge → `grilly.experimental.vsa.block_ops.BlockCodeOps` → numpy). The grilly import is attempted inside `__init__`/`backend()` in a `try/except` (so absence degrades, doesn't crash import), but the *only place `grilly` is named in the whole package*. Method bodies raise `NotImplementedError("skeleton")`.
- [ ] **Step 2: Test** — assert `BlockCodeVSA` importable, methods present, `backend()` returns a str in `{"grilly_bridge","grilly_python","numpy"}`, calling `bind` raises `NotImplementedError`. Confirm `__wiring__ == Wiring.WIRED`.
- [ ] **Step 3: Run** `pytest tests/ops -v`; then `pytest tests/test_guards.py::test_ops_is_sole_grilly_importer -v` (must still PASS — grilly named only here).
- [ ] **Step 4: Checkpoint** — full suite green.

---

### Task 4: `model/` interfaces

**Files:**
- Create: `cubbyllm/model/__init__.py`, `backbone/__init__.py`, `backbone/base.py`, `memory/__init__.py`, `memory/base.py`, `binding/__init__.py`, `binding/base.py`, `vocab/__init__.py`, `vocab/embedding.py`, `vocab/output_head.py`, `assembly.py`
- Test: mirrored tests under `tests/model/...`

**Interfaces:**
- Consumes: `Context`, `Generable`, `ParameterGenerator`, `CubbyConfig`, `BlockCodeVSA`.
- Produces:
  - `Backbone` (Protocol): `forward(tokens: Tensor) -> Tensor`. `__wiring__ = WIRED`. NO implementation — module docstring states "OPEN decision (H-D1), interface only".
  - `MemoryLayer` (implements `Generable`): `forward_generated(x, ctx)` raises `NotImplementedError`; also holds a `ParameterGenerator` and `Hardener` by composition (typed attributes, not built). `__wiring__ = WIRED`.
  - `BindingHead`: `bind(a, b)`, `bundle(vs)` delegating to a `BlockCodeVSA` (raise `NotImplementedError`). `Unbinder` (Protocol): `unbind(composite, known) -> Tensor`. `__wiring__ = WIRED`.
  - `EmbeddingSource` (Protocol): `embed(token_ids: Tensor, ctx: Context | None) -> Tensor` — hybrid static-core + generated-tail. `__wiring__ = WIRED`.
  - `RetrievalHead` (Protocol): `logits(query: Tensor, k: int) -> Tensor` — ANN top-K + local softmax. `__wiring__ = WIRED`.
  - `CubbyModel`: `__init__(self, config: CubbyConfig, context_source: ContextSource, backbone: Backbone, memory: MemoryLayer, binding: BindingHead, embedding: EmbeddingSource, head: RetrievalHead)`; `forward(self, tokens: Tensor) -> Tensor` raises `NotImplementedError` but its body is a *typed skeleton* showing the context-threaded path in comments/pseudocode. `__wiring__ = WIRED`.

- [ ] **Step 1** Write each interface file per the signatures above; bodies raise `NotImplementedError("skeleton")`; protocols use `...`.
- [ ] **Step 2** `assembly.py` `CubbyModel.forward` — raise `NotImplementedError` but include the exact context-threading order as a docstring so the data flow is legible:
  `ctx = context_source.infer(tokens)` → `h = backbone.forward(tokens)` → `h = memory.forward_generated(h, ctx)` → `bound = binding.bind(h, ...)` → `return head.logits(bound, k)`.
- [ ] **Step 3** Tests: one per module. Assert protocol/class exists, `__wiring__ == WIRED`, and that calling a concrete stub method raises `NotImplementedError` (use a minimal dummy subclass where needed). Backbone test asserts it is a Protocol with `forward` and that NO concrete Backbone is exported from the package (guard against smuggled backbone choice):

```python
# tests/model/backbone/test_base.py
import cubbyllm.model.backbone.base as b
def test_backbone_is_interface_only():
    assert hasattr(b, "Backbone")
    # no concrete implementations live here
    concretes = [n for n in dir(b) if n != "Backbone" and n[0].isupper()
                 and not n.startswith("_")]
    assert concretes == [], f"backbone must stay interface-only, found {concretes}"
```

- [ ] **Step 4** Run `pytest tests/model -v`. Expected: all PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 5: `bridges/world_model.py`

**Files:**
- Create: `cubbyllm/bridges/__init__.py`, `cubbyllm/bridges/world_model.py`
- Test: `tests/bridges/test_world_model.py`

**Interfaces:**
- Produces: `WorldModelBridge` (Protocol): `push_context(ctx: Context) -> None`; `pull_context() -> Context | None`; `emit_novelty(event: object) -> None`; `register_specialist(handle: object) -> None`. `__wiring__ = STANDALONE` (a bridge, not the forward path). Docstring cites H-F2: first-class, carries context both ways + novelty + specialist handles.

- [ ] **Step 1** Write the Protocol.
- [ ] **Step 2** Test: assert the four methods exist on the Protocol, `__wiring__` declared.
- [ ] **Step 3** Run `pytest tests/bridges -v`. Checkpoint — full suite green.

---

### Task 6: `training/` stubs + final full-suite gate

**Files:**
- Create: `cubbyllm/training/__init__.py`, `training/data.py`, `training/loop.py`, `training/probing.py`
- Test: mirrored tests under `tests/training/`

**Interfaces:**
- Produces:
  - `DataPipeline` (Protocol): `manifest_hash() -> str`; `batches(batch_size: int, seq_len: int)` (iterator) — the manifest-pinned corpus source (H-G3). `__wiring__ = STANDALONE`.
  - `TrainLoop`: `__init__(model, data, hardener)`; `step()` raises `NotImplementedError`. `__wiring__ = STANDALONE`.
  - `P5WrongContextProbe` (implements `WrongContextProbe`): `probe(...)` raises `NotImplementedError`; docstring ties it to the training gate. `__wiring__ = STANDALONE`.

- [ ] **Step 1** Write the three modules per signatures.
- [ ] **Step 2** Tests per module (interface exists, `__wiring__` declared, stub raises).
- [ ] **Step 3** Run the FULL suite `pytest -v`. Expected: everything green; guards pass; every module declares wiring; grilly named only in `ops/vsa.py`; no `model.py`/`models/`.
- [ ] **Step 4** Sanity: `python -c "import cubbyllm; print(cubbyllm.__version__)"` works with torch NOT imported (assert via `import sys; 'torch' not in sys.modules` inside a test).
- [ ] **Step 5: Final checkpoint** — full suite green, `import cubbyllm` clean and torch-free.

---

## Self-review

- **Spec coverage:** core thesis interfaces (Task 2) ✓; ops sole seam (Task 3) ✓; model components incl. OPEN backbone (Task 4) ✓; world-model bridge (Task 5) ✓; training stubs + manifest-pin seam (Task 6) ✓; tests mirror + 3 guards (Task 1 + per-task) ✓; pyproject (Task 1) ✓; wiring intent-vs-status distinction (encoded: WIRED memory with NotImplementedError, asserted by tests) ✓; success criteria (import works, pytest green, all NotImplementedError, torch-free import) → Task 6 Steps 3–5 ✓.
- **Placeholders:** none — every interface signature is spelled out.
- **Type consistency:** `Context`, `Generable`, `ParameterGenerator`, `Hardener`, `BlockCodeVSA`, `Backbone`, `MemoryLayer`, `BindingHead`, `Unbinder`, `EmbeddingSource`, `RetrievalHead`, `CubbyModel`, `WorldModelBridge`, `DataPipeline`, `TrainLoop`, `WrongContextProbe`/`P5WrongContextProbe` — names used consistently across tasks.
