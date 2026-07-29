# Cubby / CubeMind — Model & Environment Architecture Map

Verified against actual source (not just docs) on 2026-07-23: `cubby-lm/cubby/{config.py,ROADMAP.md,trunk_torch/*}` and `cubemind/cubemind/{reasoning,execution,ops,perception,core}/*`. Where the code and the prose docs (CLAUDE.md, why_sparse_cubby.md, vm.md) disagree, that's called out explicitly rather than silently trusting the docs.

---

## 1. The model — Cubby trunk (`cubby-lm`)

### Current frozen shape

The trunk is **frozen** at the shape below as of July 2026 — it cannot change without retraining from scratch (per `ROADMAP.md`'s "0.0.1 REDEFINED + FROZEN" note):

- `d_model=2560, n_layers=22, d_ffn=6848` — wide-shallow by design, because reasoning is meant to live off the token axis (in the VM, see §2), so the trunk spends its budget on representational width, not depth for in-token reasoning it isn't supposed to do.
- Params: config.py's inline comment says ~1.674B; ROADMAP.md's frozen note says ~1.95B. Same shape, two slightly different estimates in two docs — not reconciled, but both describe the same `d2560/L22/d_ffn6848` configuration.
- Tokenizer: `grillcheese_spm32k_v2`, 32,000-vocab SentencePiece — a **frozen contract**. Reserved special tokens include chat roles (`<|system|>` `<|user|>` `<|assistant|>` `<|tool|>` `<|image|>` `<|audio|>`) and structure tags (`[MY_STATE]` `[INSTRUCTION]` `[THINKING]` `[MEMORY]` `[SPECIALIST]`), several of which are reserved for rungs that don't exist yet (`[MEMORY]`→0.0.5, `[SPECIALIST]`→0.1.0 WorldManager routing, `[THINKING]`/`[INSTRUCTION]`→SFT+CubeLang).
- Two backends implement (nominally) the same architecture: `cubby/trunk/` (grilly/Vulkan, deploy target) and `cubby/trunk_torch/` (PyTorch, the currently-active training stack, ported from `cubemind/model/cubby/`). This map is built from `trunk_torch`, since that's where training is actually happening.

### Version ladder (`config.py` / `ROADMAP.md`)

One flag-gated component per semver rung; an unbuilt component raises rather than silently no-op'ing. Gate is generation quality, not just perplexity.

| ver | adds | status |
|---|---|---|
| 0.0.0 | MinGRU + tied-linear + SwiGLU substrate | done |
| 0.0.1 | scale to production shape (redefined: see above) | done, frozen |
| 0.0.2 | chunked sliding-window attention (W=512, every 3rd layer) | done |
| 0.0.3 | sparse MoE-MinGRU (top-2-of-4 + 3 shared experts) | built in code |
| 0.0.4 | Hebbian growth (Oja + lateral inhibition → spawn expert) | built in code |
| 0.0.5 | SegmentMemory (compressive, 128k effective context) | built, **not wired into the default forward path** |
| 0.0.6 | VSA binding head (frozen MAP-bipolar codebook, D=10240) | built in code |
| 0.0.7 | MTP attachment (decode-time only, frozen trunk) | built, **separate attachment, not called from model_core.py** |
| 0.0.8 | CubeLang head + VM + WorldManager arena | built, self-labeled **"SCAFFOLD"** |
| 0.0.9 | MindForge adapter bank + cross-trunk fusion | not yet reached |
| 0.1.0 | Afferent SNN gate (input triage → CNS reflex or cortical router → WorldManager dispatch) | design note only |

### Forward pass (verified in `trunk_torch/model_core.py` + `blocks.py`)

```
tokens
  │
  ▼
embed  (shared by reference with MTP heads, CubeLang head, and the tied output head)
  │
  ▼
┌── HybridBlock × n_layers ─────────────────────────────────────────────┐
│  x += mix(rmsnorm(x))     MinGRU recurrence (parallel log-domain scan) │
│                            — or MoEMinGRULayer if enable_moe:          │
│                              top-k routing + shared experts +          │
│                              DeepSeek-style bias rebalance.             │
│                              Hebbian growth hook lives HERE — can       │
│                              spawn a new expert synchronously           │
│                              mid-forward on sustained novelty.          │
│                                                            [ALWAYS]     │
│  x += attn(rmsnorm(x))    sliding-window attention, W=512              │
│                                       [only every attn_every_n layer]  │
│  x += mem_gate(EpisodicMemory.read(mean(x)))                          │
│                                       [only every mem_every_n layer,   │
│                                        and only once memory.size > 0]  │
│  x += ffn(rmsnorm(x))     SwiGLU                            [ALWAYS]   │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
final RMSNorm
  │
  ▼
head:  tied Linear(d_model, vocab)   OR   VSABindingHead
       (chosen once at construction — cosine similarity vs a frozen,
        deterministically-seeded MAP-bipolar codebook, D=10240)
  │
  ▼
logits
```

Two implementation details worth knowing before touching this:
- `use_hybrid` (whether layers are `HybridBlock` vs bare `MinGRUBlock`) is decided **once for the whole model** at construction — `enable_moe or enable_attention or enable_memory`, not a per-layer choice.
- Hebbian growth and gradient checkpointing are **mutually exclusive in practice**: checkpoint recompute needs a pure forward, and growth mutates `gate.weight` mid-forward, so growth is force-suppressed whenever `block_grad_checkpoint=True`.

### Memory subsystems — two systems, only one is live

| | `EpisodicMemory` | `SegmentMemory` |
|---|---|---|
| Wired into `model_core.py`? | **Yes** — read every forward pass at gated layers | **No** — never instantiated; reachable only via the alternate `forward_with_memory()` entry point, which nothing currently calls |
| Write gate | "Dopamine" gate — only writes if loss improved | Unconditional write |
| Read | Top-k by similarity×utility, softmax-weighted | Single-bucket lookup by `argmax(signature)` |
| Consolidation | Explicit "sleep phase" (merge/decay/prune) | None — pure FIFO eviction |
| Meant for | Within-run associative recall | Cross-chunk long-document processing (the "unlimited context" story) |

### Attachments that exist in code but sit outside the default forward call

These are fully implemented, self-tested modules that a trainer must invoke explicitly — none of them run just because you call `model(tokens)`:

- **MTP** (`mtp.py`) — `k` extra frozen-trunk decode heads; frozen-trunk discipline is asserted by its own smoketest (zero trunk gradient, non-zero MTP gradient).
- **CubeLangHead** (`cubelang_head.py`) — frozen-trunk decoder emitting CubeLang program token streams. Its own docstring says **"Status: SCAFFOLD"**: the compile/execute reward ladder and grammar-constrained decoding are both stubbed; `generate()` today is plain unconstrained greedy decode.
- **`MinGRUMultiTask`/`MindForgeLoRAHead`** (`heads.py`) — wraps the trunk as a backbone from the outside; looks like an entry point for a training script that isn't in this file set.
- **The `brain/` SNN** (`event_snn.py`, repo root, not under `cubby/`) — an event-driven spiking module, GPU-accelerated via grilly's `spike_propagate_batch`, built as a drop-in for `cubemind.brain.Synapsis`/`SNNFFN` but standalone (doesn't import cubemind). This is the planned CNS-reflex half of the 0.1.0 afferent gate — currently a parallel track, not connected to the trunk at all.
- One specific dead-on-arrival path even when enabled: `HebbianGrouping` (in `group_routing.py`, used only if `enable_group_routing` + `grouping_strategy="hebbian"`) needs the model to call `set_basis()` on it; nothing in `blocks.py`/`model_core.py` does, so it silently falls back to "every token in its own group."

---

## 2. The environment — CubeMind world-model + CubeLang VM

### The world arena (`execution/world_manager.py::WorldManager`)

Not a database or object graph — **one pre-allocated dense array**: `self._arena` shape `(max_worlds=1024, k=80, l=128)` float32, plus a parallel observation-count array. An entry's only identity is its row index. Growth logic (`process_transition` / `register_specialist`):

1. Compute a "rule" vector — `unbind(state_after, state_before)`, or take the caller-supplied vector directly (the novelty-bridge path, see §3).
2. Cosine-similarity search against every active row (brute-force `argmax`, no hashing/ANN).
3. `sim < 0.65` → **spawn**: normalize, write to the next free row. If the arena is already full, this **raises `RuntimeError`** — there is no eviction policy.
4. `sim >= 0.65` → **consolidate**: online Oja's-rule update, plus a "geometric disentanglement" step that pushes the updated vector away from the mean of every *other* active specialist (keeps concepts from bleeding into each other).

A `purify_arena()` method (full QR re-orthonormalization of all active rows) exists but is **never called automatically** by anything — manual maintenance only, and it `print()`s instead of using the codebase's normal logger.

### Grounding — how raw input gets into block-code space at all

Two separate encoders exist, and **they don't call each other**:
- `perception/encoder.py::Encoder` — per-word SHA-256 hash → position-bind → bundle (numpy fallback path; there's also an optional GPU dense-embedding path via grilly).
- `execution/world_encoder.py::WorldEncoder` — coarser: BLAKE2b hash of the *entire string* at once. This is also where role-binding for structured attributes lives (`encode_state({"color":"red",...})` → `bind(role_vec, value_vec)` → `bundle`).

Both docstrings point to a shared caller, "Decision Oracle" — which turns out to live in `cubemind/_archive/execution/decision_oracle.py` (archived, not part of the live package). So as currently staged, there's no single wired path from "text comes in" to "it's in the world model" — two parallel front doors, neither connected to the arena or to each other.

### VSA algebra substrate (`ops/block_codes.py::BlockCodes`)

Everything above is built from five primitives, each with a 3-tier fallback (grilly Vulkan → Python GPU → numpy, always available):
- `bind` / `unbind` — per-block circular convolution / correlation (FFT-based)
- `bundle` — elementwise sum (normalized by element-sum, not L2 — worth knowing since `WorldManager` normalizes by L2 instead)
- `similarity` — `(1/k)·Σ(a·b)`
- `discretize` — argmax per block, snap to a clean symbol

**No `permute` operation exists.** Ordering (e.g. sequence position) is achieved entirely by binding against distinct position vectors, not a first-class cyclic-permutation primitive.

### The CubeLang VM (`reasoning/vm.py::VSAVM`)

A program is a plain **Python `list[tuple]`** — `("ASSIGN", "john", 5)` — not compiled bytecode (that framing belongs to the Rust siblings, `opcode-vsa-rs`/`cubelang`, which are meant to mirror this VM's opcode set but are a different implementation). Dispatch is a 44-case `match opcode:` block. State is a hybrid register file: block-codes in `self.registers`, exact Python ints in a parallel `self._values` dict (arithmetic opcodes operate on the exact ints and re-derive the vector after).

Safety guards, confirmed actually implemented (not just documented): `max_instructions` (default 10000, top-level step budget — note a `LOOP` body only costs **one** unit against this budget no matter how many `max_iter` iterations it runs internally, a separate counter), unknown `JMP`/`CALL`/`POP` targets are real no-ops, `DIV` by zero assigns 0.

`DISCOVER`/`DISCOVER_SEQUENCE` — the file's own docstring calls this "the core innovation" — induces a rule from (input, output) examples via `unbind` + greedy similarity clustering, no hardcoded instruction sequence needed.

---

## 3. How the model and the environment actually connect

There are exactly two bridges, and they're very different in kind:

**Bridge 1 — Hebbian novelty → world model (data bridge, always-on when enabled)**

```
MoEMinGRULayer spawns a new expert (trunk-side novelty)
  → HebbianGrowthLayer fires a NoveltyEvent (component_idx, direction vector, ...)
  → an external "sink" callback (duck-typed — cubby never imports cubemind)
      ──────────────────────────────────────────────────────────►
        execution/novelty_bridge.py :: NoveltyToWorldBridge
          · projects the direction vector through a FIXED SEEDED
            Gaussian (same 0xC0DEB00C seed used across the stack)
          · reshapes to (k, l), L2-normalizes to match WorldManager's convention
          · calls WorldManager.register_specialist(vector)
```
This direction is genuinely one-way: `novelty_bridge.py` imports `WorldManager` but never anything from `model.cubby`; Cubby-side code is expected to hold and call the bridge object, not the other way around. (Verified from cubemind's side only — the matching `model/cubby/hebbian.py` this docstring refers to wasn't in the reviewed set.)

**Bridge 2 — CubeLang program emission → compiler/VM (text/subprocess boundary, scaffold status)**

```
CubeLangHead decodes a token stream (same vocab as the LM)
  → detokenized into CubeLang source text (a Python str)
      ──────────────────────────────────────────────────────────►
        cubby/trunk_torch/cubelang_bridge.py
          · writes a temp .cube file
          · shells out to a real compiled cubelang.exe (check / compile / disasm)
          · regex-parses the result into a CheckResult / CompileResult
```
This is **not a tensor or hypervector boundary** — it's files and a subprocess call to an external executable, and it's outside the torch training graph entirely by design. `cubelang_bridge.py` itself is self-documented as not yet wired to the actual VM execution/reward loop.

Also worth flagging: there are **three distinct hyperdimensional spaces** in play, and `config.py` itself warns they get "historically conflated": the word-vocab VSA binding head (D=10240), the neural↔symbolic opcode codebook (D=4096, canonical in the Rust side), and the world-arena block-codes (k=80×l=128=10240, dimensionally the same as the word-vocab head but generated by a different, not-yet-cross-repo-hash-identical seed path).

---

## 4. Documented vs. actually-implemented — gaps worth knowing before you rely on either doc

- **HyperSeed is dead code.** `vm.md` describes integers as encoded via HyperSeed (a fractional-power-encoding scheme with real VSA-arithmetic properties). It's fully implemented in `vm.py` but never instantiated — the VM's real integer encoding (`_val_vec`) is a much simpler seeded hash with none of HyperSeed's claimed properties; arithmetic only works because exact ints are tracked separately.
- **"SDLS duality gate"** — described in `vm.md` as guarding the `FORGE` opcode. Zero matches for "duality" or "SDLS" anywhere in `vm.py`. If it exists at all, it's inside the unreviewed `mindforge.py`, but the doc frames it as belonging to the VM itself.
- **`UNBIND_ROLE` isn't algebraic.** Despite the name, it's a plain dict lookup against an exact side-table recorded at bind time — it never calls the VM's own `unbind`.
- **Two role-vector systems that don't interoperate**: the VM's fixed 8 roles (AGENT/ACTION/OBJECT/…) vs. `WorldEncoder`'s open, caller-defined roles — same concept, different seed derivation, incompatible vectors.
- **`DECODE` bypasses `execution/decoder.py`** — `vm.py` reimplements the same logic inline instead of calling the standalone `Decoder` class that exists for exactly this.
- Active Inference / DecisionOracle, described in `why_sparse_cubby.md` as part of the reasoning story, currently lives in `cubemind/_archive/` (gitignored, not importable) — both grounding encoders above point to it as their intended common caller, so the described end-to-end pipeline (perception → decision oracle → world model → VM) reads as more aspirational than currently wired.

---

## 5. Component reference (appendix)

**`cubby-lm/cubby/trunk_torch/`**

| File | Owns | Wired into default forward? |
|---|---|---|
| `model_core.py` | `MinGRUModel` — the trunk itself | — |
| `blocks.py` | `HybridBlock`, `MinGRUBlock` | yes |
| `layers.py` | `MinGRULayer`, `GLUChannelMix` | yes |
| `nn_primitives.py` | `RMSNorm`, `prefix_scan_causal` | yes |
| `attention.py` | `LocalCausalAttention` | yes, conditional |
| `moe.py` | `MoEMinGRULayer` | yes, if `enable_moe` |
| `group_routing.py` | grouping strategies + `GroupedMoEBlock(Bias)` | opt-in, off by default |
| `hebbian.py` | `HebbianGrowthLayer`, `NoveltyEvent` | yes, if `enable_hebbian_growth` |
| `episodic_memory.py` | `EpisodicMemory` | yes, if `enable_memory` |
| `segment_memory.py` | `SegmentMemory` | **no** |
| `vsa_binding_head.py` | `VSABindingHead` | yes, if `head_type="vsa"` |
| `mtp.py` | `MTPModule` | **no** — external attachment |
| `cubelang_head.py` | `CubeLangHead` | **no** — external attachment, scaffold |
| `cubelang_bridge.py` | `check_program`, `compile_program` | **no** — subprocess boundary, not torch graph |
| `heads.py` | `MindForgeLoRAHead`, `MinGRUMultiTask` | **no** — wraps trunk from outside |

**`cubemind/cubemind/{reasoning,execution,ops,perception}/`**

| File | Owns | Role |
|---|---|---|
| `reasoning/vm.py` | `VSAVM`, `CleanupMemory`, `HyperSeed` (unused) | the CubeLang interpreter |
| `execution/world_manager.py` | `WorldManager` | the world arena |
| `execution/world_encoder.py` | `WorldEncoder` | text/role → block-code (independent of perception/encoder.py) |
| `execution/novelty_bridge.py` | `NoveltyToWorldBridge` | Cubby → WorldManager adapter |
| `execution/decoder.py` | `Decoder` | standalone decode (unused by the VM's own DECODE) |
| `execution/hyla.py` | `HYLA` | VSA-embedding → LoRA-weight hypernetwork |
| `ops/block_codes.py` | `BlockCodes` | the VSA algebra everything sits on |
| `perception/encoder.py` | `Encoder` | text → block-code (independent of world_encoder.py) |
