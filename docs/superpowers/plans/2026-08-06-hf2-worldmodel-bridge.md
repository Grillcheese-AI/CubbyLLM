# H-F2 World-Model Bridge (M1: Automatic Specialization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove automatic specialization across the Cubby↔MoWM bridge — CubbyLLM poses a challenge, MoWM routes it to an expert world or **spawns a new specialist when none fits**, over a Protocol-first bridge with a raw grilly block-code boundary.

**Architecture:** CubbyLLM ships the `WorldModelBridge` Protocol + a dependency-free `FakeWorldModel` + a thin `attempt_challenge()` flow (tested fast against the fake). MoWM ships the real `CubbyBridge` adapter that imports CubbyLLM's Protocol and drives `MoWMRouter`/`AxiomLibrary`/`World` (route-vs-spawn decided deterministically by cosine-to-world-signature + a challenge→specialist memory, spawn via the real `maybe_spawn`). Dependency points **mowm→cubbyllm only**.

**Tech Stack:** Python 3.12; CubbyLLM (`cubbyllm/bridges/`, torch/mowm-free at import); mowm (`mowm/bridges/`, imports `cubemind`+`grilly`, `uv`-managed); grilly block-codes `(k,l)` float32 shared by both.

## Global Constraints

- **Block-code boundary, no hard link.** A `Challenge` carries a `(k,l)` grilly block-code; a `ChallengeResult` carries a `(k,l)` block-code + `world_id: str` + `confidence: float` + `spawned: bool`. CubbyLLM ships ONLY the Protocol + `FakeWorldModel`; the mowm→cubemind→grilly stack is imported ONLY on the mowm side. `import cubbyllm` and every `cubbyllm/bridges/` module stay torch-free + import-light (lazy imports).
- **Wiring.** Every new `cubbyllm/` module declares `__wiring__ = Wiring.STANDALONE` (`from ..core.protocols import Wiring`); a CI-guard test enforces it. `mowm/bridges/` follows mowm's own conventions (no `__wiring__`).
- **Dependency direction:** mowm imports CubbyLLM's Protocol; CubbyLLM never imports mowm/cubemind.
- **Toy dimensions** `k=4, l=32` in tests. mowm may use **DirectML** (`torch-directml`) for the torch/GPU path.
- **Commit trailers.** CubbyLLM commits end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav`
  **mowm commits use mowm's own conventional style with NO CubbyLLM trailers.**
- **Green suites.** CubbyLLM `python -m pytest tests -q`. mowm `uv run pytest tests/ -q` (738 green as of `30bf828`).

---

## Task 1: Protocol types + `FakeWorldModel` (CubbyLLM)

**Files:**
- Modify: `cubbyllm/bridges/world_model.py` (add `Challenge`, `ChallengeResult`, and `attempt()` to the Protocol)
- Create: `cubbyllm/bridges/fake_world_model.py`
- Test: `tests/bridges/test_worldmodel_bridge.py`

**Interfaces:**
- Produces: `Challenge(context, tag=None)`; `ChallengeResult(result, world_id, confidence, spawned)`; `WorldModelBridge.attempt(challenge: Challenge) -> ChallengeResult`; `FakeWorldModel(seeded_tags=())` implementing the Protocol, with a `.registered: list[str]` of specialist world_ids.

- [ ] **Step 1: Write the failing test.** `tests/bridges/test_worldmodel_bridge.py`:

```python
import numpy as np
from cubbyllm.core.protocols import Wiring
from cubbyllm.bridges.world_model import Challenge, ChallengeResult, WorldModelBridge
from cubbyllm.bridges import fake_world_model as fwm


def _ctx():
    return np.zeros((4, 32), dtype=np.float32)


def test_fake_spawns_on_novel_then_reuses():
    fake = fwm.FakeWorldModel()
    r1 = fake.attempt(Challenge(context=_ctx(), tag="Z"))
    assert isinstance(r1, ChallengeResult)
    assert r1.spawned is True                      # novel -> spawn a specialist
    r2 = fake.attempt(Challenge(context=_ctx(), tag="Z"))
    assert r2.spawned is False                     # same challenge -> reuse
    assert r2.world_id == r1.world_id              # ...the same specialist
    assert fake.registered == [r1.world_id]        # exactly one registered


def test_fake_routes_seeded_tag_without_spawning():
    fake = fwm.FakeWorldModel(seeded_tags=("X",))
    r = fake.attempt(Challenge(context=_ctx(), tag="X"))
    assert r.spawned is False                       # known domain -> no spawn
    assert fake.registered == []


def test_fake_satisfies_protocol():
    assert isinstance(fwm.FakeWorldModel(), WorldModelBridge)


def test_wiring_is_standalone():
    assert fwm.__wiring__ is Wiring.STANDALONE
```

- [ ] **Step 2: Run it, verify it fails.** `python -m pytest tests/bridges/test_worldmodel_bridge.py -q` → FAIL (`ImportError`/`AttributeError`: `Challenge`, `fake_world_model` not defined).

- [ ] **Step 3: Add the types + `attempt()` to `cubbyllm/bridges/world_model.py`.** Insert after the existing imports (keep the existing `WorldModelBridge` docstring + `push_context`/`pull_context`/`emit_novelty`/`register_specialist` methods; add the dataclasses above the Protocol and `attempt` as its first method):

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Challenge:
    """A task CubbyLLM offloads to the world model."""
    context: object            # (k, l) grilly block-code — the task/query
    tag: str | None = None     # optional symbolic label (deterministic tests/audit)


@dataclass(frozen=True)
class ChallengeResult:
    """The world model's answer to a Challenge."""
    result: object             # (k, l) block-code prediction the world produced
    world_id: str              # which expert world answered
    confidence: float          # routing/prediction confidence in [0, 1]
    spawned: bool              # True iff a NEW specialist world was grown for this
```

Add this method as the first member of the `WorldModelBridge` Protocol body (before `push_context`):

```python
    def attempt(self, challenge: "Challenge") -> "ChallengeResult":
        """Route the challenge to an expert world; spawn a specialist if none
        fits; return its block-code result + world_id + confidence + spawned."""
        ...
```

- [ ] **Step 4: Create `cubbyllm/bridges/fake_world_model.py`:**

```python
"""FakeWorldModel — a deterministic, dependency-free WorldModelBridge for tests.

Wired: STANDALONE — a test double, not the forward path.

Implements the challenge round-trip without importing mowm/cubemind/torch: it
mints a new world_id (spawned=True) the first time it sees a challenge tag,
routes to the recorded world (spawned=False) on repeats, and treats a set of
seeded tags as already-existing worlds (route, never spawn). Lets CubbyLLM's
attempt_challenge() flow + its assertions run under plain pytest, no GPU.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .world_model import Challenge, ChallengeResult

__wiring__ = Wiring.STANDALONE


class FakeWorldModel:
    def __init__(self, seeded_tags: tuple[str, ...] = ()) -> None:
        self._worlds: dict[str, str] = {t: f"seed:{t}" for t in seeded_tags}
        self._spawn_counter = 0
        self.registered: list[str] = []

    def _key(self, challenge: Challenge) -> str:
        return challenge.tag if challenge.tag is not None else repr(challenge.context)

    def attempt(self, challenge: Challenge) -> ChallengeResult:
        key = self._key(challenge)
        if key in self._worlds:                       # known / already-spawned
            return ChallengeResult(
                result=challenge.context, world_id=self._worlds[key],
                confidence=1.0, spawned=False,
            )
        self._spawn_counter += 1                      # novel -> spawn a specialist
        world_id = f"spawn:{self._spawn_counter}"
        self._worlds[key] = world_id
        self.register_specialist(world_id)
        return ChallengeResult(
            result=challenge.context, world_id=world_id,
            confidence=0.5, spawned=True,
        )

    # underlying channels (minimal; attempt() drives register_specialist)
    def push_context(self, ctx) -> None: ...
    def pull_context(self): return None
    def emit_novelty(self, event: object) -> None: ...
    def register_specialist(self, handle: object) -> None:
        self.registered.append(str(handle))
```

- [ ] **Step 5: Run the tests, verify they pass.** `python -m pytest tests/bridges/test_worldmodel_bridge.py tests/test_guards.py -q` → all pass (4 new + guards green; `import cubbyllm` stays torch-free — the fake imports only numpy in the test, not the module).

- [ ] **Step 6: Commit.**

```bash
git add cubbyllm/bridges/world_model.py cubbyllm/bridges/fake_world_model.py tests/bridges/test_worldmodel_bridge.py
git commit -m "feat(bridges): Challenge/ChallengeResult + attempt() Protocol + FakeWorldModel (H-F2 M1)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 2: `attempt_challenge()` client flow (CubbyLLM)

**Files:**
- Create: `cubbyllm/bridges/world_model_client.py`
- Test: add to `tests/bridges/test_worldmodel_bridge.py`

**Interfaces:**
- Consumes: `WorldModelBridge`, `Challenge`, `ChallengeResult` (Task 1).
- Produces: `attempt_challenge(bridge, context, tag=None) -> ChallengeResult` — builds a `Challenge` and drives the bridge; `newly_specialized(result) -> bool` — thin predicate `result.spawned`.

- [ ] **Step 1: Write the failing test.** Append to `tests/bridges/test_worldmodel_bridge.py`:

```python
def test_attempt_challenge_drives_the_bridge_and_flags_new_specialists():
    from cubbyllm.bridges.world_model_client import attempt_challenge, newly_specialized
    fake = fwm.FakeWorldModel(seeded_tags=("X",))
    known = attempt_challenge(fake, _ctx(), tag="X")
    assert newly_specialized(known) is False
    novel = attempt_challenge(fake, _ctx(), tag="Z")
    assert newly_specialized(novel) is True
    again = attempt_challenge(fake, _ctx(), tag="Z")
    assert again.world_id == novel.world_id and again.spawned is False


def test_import_cubbyllm_stays_light():
    import sys, cubbyllm  # noqa: F401
    assert "torch" not in sys.modules
    assert not any(m == "mowm" or m.startswith("mowm.") for m in sys.modules)
```

- [ ] **Step 2: Run it, verify it fails.** `python -m pytest tests/bridges/test_worldmodel_bridge.py -q` → FAIL (`ImportError: world_model_client`).

- [ ] **Step 3: Create `cubbyllm/bridges/world_model_client.py`:**

```python
"""world_model_client — CubbyLLM's side of the world-model bridge.

Wired: STANDALONE — a bridge helper, not the forward path.

Builds a Challenge from a block-code context and drives any WorldModelBridge
(the FakeWorldModel in tests; MoWM's CubbyBridge in real use). Import-light: no
mowm/cubemind/torch import here — the concrete bridge is passed in by the caller.
"""
from __future__ import annotations

from ..core.protocols import Wiring
from .world_model import Challenge, ChallengeResult

__wiring__ = Wiring.STANDALONE


def attempt_challenge(bridge, context, tag: str | None = None) -> ChallengeResult:
    """Pose `context` (a (k,l) block-code) to `bridge` as a Challenge."""
    return bridge.attempt(Challenge(context=context, tag=tag))


def newly_specialized(result: ChallengeResult) -> bool:
    """True iff the world model grew a new specialist for this challenge."""
    return result.spawned
```

- [ ] **Step 4: Run the tests, verify they pass.** `python -m pytest tests/bridges/ tests/test_guards.py -q` → all pass.

- [ ] **Step 5: Commit.**

```bash
git add cubbyllm/bridges/world_model_client.py tests/bridges/test_worldmodel_bridge.py
git commit -m "feat(bridges): attempt_challenge() client flow (H-F2 M1)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 3: `CubbyBridge` adapter over real MoWM (mowm repo)

**Files:**
- Create: `mowm/bridges/__init__.py`, `mowm/bridges/cubby_bridge.py` (in `C:\Users\grill\Documents\GitHub\mowm`)

**Interfaces:**
- Consumes: CubbyLLM's `Challenge`/`ChallengeResult`/`WorldModelBridge` (Task 1, imported from the sibling cubbyllm package); mowm's `AxiomLibrary`, `MoWMRouter`, `World`, `seed_all`; `WorldEncoder`.
- Produces: `CubbyBridge(library, router, tau_match=0.35)` implementing `attempt(challenge) -> ChallengeResult`, with `.registered: list[int]` (specialist world_ids) and `.num_worlds` (→ `router.num_active_worlds`).

**Note:** CubbyLLM must be importable from mowm's venv (`uv pip install -e ../CubbyLLM` or add it as a path dependency). The adapter's route-vs-spawn is deterministic: (a) a per-challenge memory `tag → world_id` guarantees a repeated challenge reuses its world; (b) otherwise cosine of `challenge.context` to each active world's axiom signature ≥ `tau_match` routes to the best match; (c) otherwise `router.maybe_spawn(challenge.context, library)` grows a specialist. The world's axiom signature is `world._axiom_vec` — if it isn't public, add a `@property axiom_vec` to `mowm/world.py` returning `self._axiom_vec` as part of this task.

- [ ] **Step 1: Author `mowm/bridges/__init__.py`:**

```python
from .cubby_bridge import CubbyBridge

__all__ = ["CubbyBridge"]
```

- [ ] **Step 2: Author `mowm/bridges/cubby_bridge.py`:**

```python
"""CubbyBridge — MoWM's adapter implementing CubbyLLM's WorldModelBridge Protocol.

Dependency points mowm -> cubbyllm (never the reverse): CubbyLLM ships the
Protocol + Challenge/ChallengeResult; this adapter drives MoWM's router/library
to answer a challenge, spawning a specialist world when none fits.
"""
from __future__ import annotations

import numpy as np

from cubbyllm.bridges.world_model import Challenge, ChallengeResult

from ..axiom_library import AxiomLibrary
from ..router import MoWMRouter


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


class CubbyBridge:
    def __init__(self, library: AxiomLibrary, router: MoWMRouter,
                 tau_match: float = 0.35) -> None:
        self.library = library
        self.router = router
        self.tau_match = tau_match
        self._specialist_for: dict[str, int] = {}   # challenge key -> world_id
        self.registered: list[int] = []

    @property
    def num_worlds(self) -> int:
        return self.router.num_active_worlds

    def _key(self, challenge: Challenge) -> str:
        return challenge.tag if challenge.tag is not None else repr(
            np.asarray(challenge.context).tobytes()
        )

    def _world_by_id(self, world_id: int):
        for w in self.router._worlds.values():
            if w.world_id == world_id:
                return w
        return None

    def _best_match(self, ctx: np.ndarray):
        best, best_sim = None, -1.0
        for w in self.router._worlds.values():
            sim = _cosine(ctx, np.asarray(w.axiom_vec))
            if sim > best_sim:
                best, best_sim = w, sim
        return best, best_sim

    def attempt(self, challenge: Challenge) -> ChallengeResult:
        ctx = np.asarray(challenge.context, dtype=np.float32)
        action = np.zeros_like(ctx)                  # neutral action for predict
        key = self._key(challenge)

        # (a) exact reuse of a previously-answered challenge
        if key in self._specialist_for:
            world = self._world_by_id(self._specialist_for[key])
            pred, conf = world.predict(ctx, action)
            return ChallengeResult(pred, str(world.world_id), float(conf), False)

        # (b) a known world whose signature matches the challenge
        best, best_sim = self._best_match(ctx)
        if best is not None and best_sim >= self.tau_match:
            self._specialist_for[key] = best.world_id
            pred, conf = best.predict(ctx, action)
            return ChallengeResult(pred, str(best.world_id), float(conf), False)

        # (c) novel: grow a specialist from the nearest axioms
        world = self.router.maybe_spawn(ctx, self.library)
        if world is None:                            # pool full / no axioms
            raise RuntimeError("CubbyBridge: could not spawn a specialist")
        self._specialist_for[key] = world.world_id
        self.registered.append(world.world_id)
        pred, conf = world.predict(ctx, action)
        return ChallengeResult(pred, str(world.world_id), float(conf), True)

    # underlying channels (attempt() drives register_specialist via maybe_spawn)
    def push_context(self, ctx) -> None: ...
    def pull_context(self): return None
    def emit_novelty(self, event: object) -> None: ...
    def register_specialist(self, handle: object) -> None:
        self.registered.append(handle)
```

- [ ] **Step 3: Add a `axiom_vec` property to `mowm/world.py`** if not already public (near the other `@property` accessors, e.g. after `obs_count`):

```python
    @property
    def axiom_vec(self) -> np.ndarray:
        """The world's bundled-axiom signature (used for challenge routing)."""
        return self._axiom_vec
```

- [ ] **Step 4: Manual smoke check** (not committed): from mowm's venv, `uv run python -c "from mowm.bridges import CubbyBridge; print(CubbyBridge)"` imports clean (confirms cubbyllm is importable from mowm's env). If `ImportError: cubbyllm`, install it: `uv pip install -e ../CubbyLLM`.

- [ ] **Step 5: Commit** (mowm style, NO CubbyLLM trailers):

```bash
git -C ../mowm add mowm/bridges/__init__.py mowm/bridges/cubby_bridge.py mowm/world.py
git -C ../mowm commit -m "feat(bridges): CubbyBridge adapter implementing CubbyLLM's WorldModelBridge"
```

---

## Task 4: The M1 kill-criterion test against real MoWM (mowm repo)

**Files:**
- Create: `mowm/tests/test_cubby_bridge.py` (in the mowm repo)

**Interfaces:**
- Consumes: `CubbyBridge` (Task 3); `AxiomLibrary`, `MoWMRouter`, `World`, `seed_all` (mowm); `Challenge` (cubbyllm).

**Note:** the known-domain control constructs the known challenge's context AS an existing world's `axiom_vec` (cosine 1.0 → routes, never spawns); the novel challenge uses `library.encoder._hash_to_vec("hf2::novel_challenge")` (a hash unrelated to the seeded axioms → below `tau_match` → spawns).

- [ ] **Step 1: Write the test.** `mowm/tests/test_cubby_bridge.py`:

```python
"""H-F2 M1 kill-criterion: automatic specialization across the Cubby<->MoWM bridge."""
from __future__ import annotations

import numpy as np

from cubbyllm.bridges.world_model import Challenge

from mowm import AxiomLibrary, MoWMRouter, World
from mowm.bridges import CubbyBridge
from mowm.domains import seed_all

K, L = 4, 32


def _bridge():
    lib = AxiomLibrary(k=K, l=L)
    import mowm.domains.physics  # noqa: F401
    import mowm.domains.logic    # noqa: F401
    seed_all(lib, domains=["physics", "logic"])
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    # seed two known worlds from disjoint axiom slices
    phys = lib.select(lib.bc.random_discrete(seed=1), top_n=3)
    logi = lib.select(lib.bc.random_discrete(seed=2), top_n=3)
    router.add_world(World(world_id=100, k=K, l=L, n_hylas=2, z_depth=0,
                           z_max=2, tau=0.5, axioms=phys, seed=100))
    router.add_world(World(world_id=200, k=K, l=L, n_hylas=2, z_depth=0,
                           z_max=2, tau=0.5, axioms=logi, seed=200))
    return CubbyBridge(lib, router)


def test_novel_challenge_spawns_a_specialist_then_reuses_it():
    bridge = _bridge()
    before = bridge.num_worlds
    ctx = bridge.library.encoder._hash_to_vec("hf2::novel_challenge")

    r1 = bridge.attempt(Challenge(context=ctx, tag="novel"))
    assert r1.spawned is True                              # grew a specialist
    assert bridge.num_worlds == before + 1                 # exactly one world added
    assert bridge.registered == [int(r1.world_id)]

    r2 = bridge.attempt(Challenge(context=ctx, tag="novel"))
    assert r2.spawned is False                             # reuse, no re-spawn
    assert r2.world_id == r1.world_id
    assert bridge.num_worlds == before + 1                 # single-spawn control


def test_known_challenge_routes_without_spawning():
    bridge = _bridge()
    before = bridge.num_worlds
    known_ctx = np.asarray(bridge._world_by_id(100).axiom_vec, dtype=np.float32)

    r = bridge.attempt(Challenge(context=known_ctx, tag="known-physics"))
    assert r.spawned is False                              # reuse control
    assert r.world_id == "100"
    assert bridge.num_worlds == before                     # no spurious spawn
```

- [ ] **Step 2: Run it, verify it passes** (rebuild not needed — pure Python). `uv run pytest tests/test_cubby_bridge.py -q` in the mowm repo → 2 passed. If the novel challenge does NOT spawn (cosine ≥ tau_match by chance), lower `tau_match` in the `CubbyBridge(...)` construction in `_bridge()` until the hash-context is below it while the axiom-vec context (cosine 1.0) stays above — document the chosen value.

- [ ] **Step 3: Run the full mowm suite, verify still green.** `uv run pytest tests/ -q` → 740 passed (738 baseline + 2 new).

- [ ] **Step 4: Commit** (mowm style, NO CubbyLLM trailers):

```bash
git -C ../mowm add tests/test_cubby_bridge.py
git -C ../mowm commit -m "test(bridges): H-F2 M1 kill-criterion — spawn specialist + reuse + controls"
```

---

## Task 5: Update H-F2 in the hypotheses doc (CubbyLLM)

**Files:**
- Modify: `CUBBYLLM_HYPOTHESES.md` (the H-F2 section)

- [ ] **Step 1: Update the H-F2 "first step" note.** Find the `**H-F2 first step (2026-08-05).**` paragraph (added by the reasoning-bridge slice) and append a dated M1 result after it: the full bidirectional bridge's **first slice is built** — automatic specialization is demonstrated end-to-end over the block-code boundary. CubbyLLM poses a challenge; MoWM (real, 740-green) routes it to an expert world or **spawns a new specialist when none fits**, proven with the reuse control (a known challenge routes, never spawns) and the single-spawn control (a novel challenge spawns exactly one world). Protocol-first coupling: CubbyLLM ships `WorldModelBridge` + `FakeWorldModel` (`cubbyllm/bridges/`); MoWM ships `CubbyBridge` (`mowm/bridges/`), dependency mowm→cubbyllm only. Cross-world data-sharing, SR axiom discovery, imagination rollouts, and multimodal challenges remain later slices.

- [ ] **Step 2: Commit.**

```bash
git add CUBBYLLM_HYPOTHESES.md
git commit -m "docs(hypotheses): H-F2 M1 — automatic specialization across the world-model bridge

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Self-review notes

- **Spec coverage:** spec §2 (Protocol-first, block-code boundary, mowm→cubbyllm dep) → Tasks 1+3. §3 kill-criterion (spawn + reuse + single-spawn controls) → Task 4. §4 interfaces (`Challenge`/`ChallengeResult`/`attempt`, `FakeWorldModel`, `CubbyBridge`) → Tasks 1+3. §6 testing (CubbyLLM-against-fake, mowm-against-real, toy dims, DirectML) → Tasks 1/2 (fake) + 4 (real). §7 constraints → Global Constraints. §8 deferrals → not tasked (correct). §9 file structure → Tasks 1–5.
- **Type consistency:** `attempt(challenge) -> ChallengeResult` on both `FakeWorldModel` (Task 1) and `CubbyBridge` (Task 3); `ChallengeResult(result, world_id: str, confidence: float, spawned: bool)` used identically in the fake, the adapter, and the tests; `world_id` is a `str` in `ChallengeResult` (adapter stringifies `world.world_id`), and the known-control asserts `r.world_id == "100"` accordingly.
- **Cross-repo:** CubbyLLM tasks carry CubbyLLM trailers; mowm tasks (3, 4) use mowm's style (no trailers) and `git -C ../mowm`. Dependency direction mowm→cubbyllm only (adapter imports `cubbyllm.bridges.world_model`; nothing in CubbyLLM imports mowm).
- **Deterministic kill-criterion:** the route-vs-spawn decision is cosine + a challenge→world memory (not the untrained DSelect-k gate), so the spawn/reuse/known behavior is reproducible; Task 4 Step 2 documents the `tau_match` fallback if the hash-context similarity lands high.
