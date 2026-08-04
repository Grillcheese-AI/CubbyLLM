# Episodic memory — recall past the attention window

**Date:** 2026-08-04 · **Status:** design, approved to plan · **Hypothesis:** H-D5 (to be added)

## 1. Why this exists

The H-D4 A/B (2026-08-04) settled the backbone: a MinGRU + sliding-window-attention
hybrid recalls ~100% of a planted needle **inside its window** at bounded decode
state, and pure recurrence does not (98% vs 32%). The honest bound is in the same
data — **beyond the window, recall falls to the floor.** At length 4096 the
needle is recallable only when it sits in the final ~512 tokens; everywhere else
the model is at chance.

This component is what carries recall *past* the window. It is the rung the whole
long-context claim now rests on, and it is the component with the most documented
failure history in this project: `MEMORY_PROBE.md` (cubby-lm) is a post-mortem of
why the last episodic memory died. Those failures are this design's hard
constraints (§4).

**Success metric (already have it):** the existing needle test
(`validation/exp_needle_recall.py`). The memory succeeds when the **beyond-window
depths** — currently at chance (e.g. length 4096, depths 0.0–0.75) — become
recallable, at **bounded per-step compute**. Nothing else in the test changes; we
are lifting the part of the curve the windowed hybrid provably cannot reach.

## 2. Settled design decisions

Three forks were decided during brainstorming; each is load-bearing.

- **(A) Trained end-to-end.** The memory read is a differentiable associative
  lookup, learned jointly with the trunk — not an inference-time attach on a
  frozen trunk. Rationale: the windowed attention won *because* gradient shaped
  it; the last memory failed partly because its read was never shaped by training.
- **(1) One vector per token.** Each past token is stored (its learned key/value)
  and retrieved by similarity. Rationale: it is the only option that preserves an
  *exact* needle; chunk-summaries and fixed slots lose it. The store grows with
  document length but lives off-GPU and is accessed sparsely (top-K), so **per-step
  compute and GPU state stay bounded** even as the store grows. This is "bounded
  *compute*, cheap growing *store*," stated honestly — not "bounded total memory."
- **(i) HDC as the index, not the representation.** Each token gets a *learned*
  dense key/value (`k = W_k·h`, `v = W_v·h`); the key is binarized to a hypervector
  (`sign` → bipolar) so retrieval is **O(1) Hamming** content-addressing — the
  CubeMind FAISS-binary trick. Memories still *are* hypervectors and the lookup is
  still content-addressable ("1 vector = 1 neuron, retrieved by Hamming"), but the
  values read out are learned and differentiable. Role-filler *unbinding* (VSA
  representation-ii) is deferred to the structured/symbolic memory (the VM side,
  H-B3) where its structure is actually queried; forcing token recall through
  unbinding only buys the Group-B crosstalk/capacity limits for structure this job
  does not need.

## 3. Architecture

Four units with clean boundaries. Each is understandable and testable alone.

### 3.1 `EpisodicStore` (`cubbyllm/model/recall/store.py`)
Holds the memory for one sequence (batched): dense `keys (N,d_k)`, dense
`values (N,d_v)`, and their binary `codes (N,d_k)` (bipolar `sign(keys)`).
- `write(k, v)` — append a token's key/value (and its code). Per-sequence.
- `retrieve(q, topk)` — return the top-K keys+values for a query. **Two paths that
  must agree on which set they return:** training uses dense cosine over the
  live set (differentiable, O(N) — fine at training S); inference uses Hamming on
  the codes (O(1)-ish via popcount / an index). The *read* over the returned set
  is identical in both.
- `state_dict()` / `load_state_dict()` — the store is an explicit, **checkpointed**
  object (trap 2). It is not held in `__dict__`.

### 3.2 `MemoryRead` (`cubbyllm/model/recall/read.py`)
Given a per-position query `q` and a retrieved `(k, v)` set, computes a soft
attention read `r` = `softmax(q·kᵀ/√d)·v`, projected back to `d_model`. This is the
differentiable part — a modern-Hopfield / kNN-attention read. **Per-position,
per-sequence** by construction (trap 1): every query position gets its own
retrieval and its own read. Adds residually: `h ← h + r`.

### 3.3 Integration into `HybridBackbone` (a third interleaved mixer)
`HybridBackbone` already interleaves MinGRU and windowed attention. Add memory as
a **third interleaved read** on a subset of layers (`mem_every`), matching
cubby-lm's block shape (mix → attn → mem → ffn). The windowed-attention mixer is
left unchanged; memory is its own optional per-layer branch, so the two roles stay
separable and independently testable.

**The retrieval is masked to the beyond-window causal past.** At query position
`i`, the memory retrieves top-K from positions `j < i − window` — i.e. *only what
the window cannot already see*. This forces the memory to learn long-range recall
rather than duplicate the window, and it targets the exact part of the needle
curve we are trying to lift.

### 3.4 Write/read data flow
- **Training (parallel):** compute all keys/values for the sequence; for each
  query `i`, retrieve top-K from the beyond-window causal past via a masked dense
  cosine (an O(S²) similarity like attention, top-K selected — trains the
  retrieval); read (§3.2). Straight-through on the top-K selection so gradient
  flows to keys and the read.
- **Inference (decode):** each new token writes its (k, v, code) to the store;
  each query retrieves top-K by O(1) Hamming over codes, reads the same way. The
  carried recurrent/window state stays bounded; the store grows off-GPU.

## 4. The four traps (hard constraints, from `MEMORY_PROBE.md`)

1. **Per-position, per-sequence reads** — never a batch-row-0 DC offset broadcast
   to all positions. Enforced by §3.2 (every position retrieves and reads its own).
2. **Persisted** — the store is a checkpointed object (§3.1), not `__dict__` state
   wiped on resume.
3. **No recency collapse** — v1 writes every token and evicts nothing during a
   sequence; there is no half-life decay masquerading as importance. (Pruning, if
   ever needed at very long context, is surprise-gated and deferred — §6.)
4. **Cheap** — retrieval is top-K (sparse), O(1)-ish at inference via Hamming; the
   read is over K vectors, not N.

## 5. Testing

- **Falsifiable gate FIRST (`validation/exp_d5_episodic.py`), before the real
  trunk.** A toy: a small model with the memory read, trained on the needle task
  with the needle planted **beyond the window**. **Success:** beyond-window recall
  climbs off the chance floor at bounded top-K. **Kill criterion:** if the toy
  memory cannot beat the windowed hybrid's beyond-window chance floor, the
  approach is wrong and we stop before touching the real model. Report against the
  same controls as `exp_needle_recall` (shuffled, untrained).
- **Package unit tests (`tests/model/recall/`)**, mirroring the backbone's:
  store write/retrieve round-trip; the dense-training path and the Hamming-inference
  path return the *same* top-K set on the same data (the training/serving-skew
  guard); `MemoryRead` is per-position (two positions with different queries get
  different reads); the store round-trips through `state_dict`; decode with memory
  still matches a full forward to tolerance where the store is fixed.

## 6. Scope — explicitly deferred (YAGNI)

- **Surprise-gated pruning / eviction.** v1 keeps everything for a sequence. Only
  build eviction when a real run shows the store is too large to hold, and make it
  importance-based (not recency).
- **A production ANN index.** v1's inference path is exact Hamming (popcount over
  codes) — correct and cheap enough to validate. A FAISS/`IndexBinaryFlat`-grade
  index is a later optimization, not a correctness dependency.
- **Structured VSA-unbind memory (representation-ii).** That is the symbolic/VM
  memory (H-B3), a different job; not in this component.
- **Cross-sequence / persistent-across-documents store.** v1 is per-sequence
  (matches the needle test). A store that carries across documents is a later,
  separately-gated addition.

## 7. Placement & naming

New package `cubbyllm/model/recall/` (`store.py`, `read.py`, `__init__.py`), to
keep it distinct from the existing `cubbyllm/model/memory/MemoryLayer`, which is
the **parametric** θ=f(c) memory (H0) — a different mechanism. Every module
declares `__wiring__`; the read is `WIRED` once interleaved into the default
`HybridBackbone`, `STANDALONE` until the toy gate passes. Only `ops/` imports
grilly (the binary-code path uses plain torch `sign`/popcount, no grilly import
outside the seam).

## 8. Open question for the plan

Whether the memory read shares the windowed-attention layers' query projection or
learns its own. Default: its own `W_q/W_k/W_v` (cleanest boundary), revisit if
params matter. To be settled in the implementation plan.
