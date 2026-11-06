# 0056 — Many adapters in one batch on grilly2: each request with its own LoRA
status: blocked (waits for 0049's first promoted adapter; then open)
written: 2026-11-06 by Cowork · approved: 2026-11-06 by Nick ("yes record it", to build when 0049 promotes its first adapters) · hypothesis: H-A15 (T5 routing, T6 generated adapters), H-K3

## Why
Nick's January plan for a `grilly.agentic` module was reviewed on 2026-11-06. It proposed:
- hypernetwork-spawned agents and runtime experts;
- cellular automata, Hebbian fast weights and nested learning;
- an agent registry.

Its mechanisms are now CubbyLLM hypotheses with their own lines: H0 and H-A15's T6, H-A15's adapters on demand, H-A1
and H-A16. A compute framework provides operations, not architectures. **One operation from that plan is needed:
applying low-rank weights, trained or generated, per request.**

- **Today:** grilly2's `grilly.infer.lora.apply_lora_` wraps the projections with one LoRA pair, and a seat copies a
  role's pair in before decoding a batch (`cubbyllm/ops/grilly_seats.py`). Every request in a batch shares one
  adapter.
- **Soon:** 0049's router picks an adapter per skill signature, and T6 generates more. A batch will then hold requests
  for different adapters. Swapping per request costs a copy per adapter and breaks batching.
- **Prior art** (to cite once checked): S-LoRA (Sheng et al., 2023) and Punica (Chen et al., 2023), a batched
  matrix-vector product over a stack of adapters, gathered by an index per row.

**What it brings, in plain words:** Cubby can use many small skills at once, each question with its own add-on,
without swapping them in and out.
- **Over a conventional model:** there it is one fine-tuned model per task, or adapters merged into one. Here hundreds
  of skills share one trunk in one pass.

## Build
In grilly2, on a branch Nick reviews; grilly2's `main` stays untouched.
1. **The operation:**
   - a stack of K adapters per target projection, each with A (r x d_in) and B (d_out x r);
   - an adapter index per row, where -1 means none;
   - per row, y = W x + s * B[i] (A[i] x);
   - ranks may differ, padded or grouped by rank (Claude Code's choice, recorded).
2. **A Vulkan kernel for it,** through `grilly_core` (never grilly v1, never the deprecated paths), with the torch
   reference beside it.
3. **Adapter memory:** adapters loaded on use and evicted by a least-recently-used rule under a byte budget. This is
   H-A15's lifecycle: latent on disk, loaded when used. A generated adapter (T6) enters the same stack.
4. **On CubbyLLM's side:** `grilly_seats` passes an adapter index per request, and 0049's router fills it. Behind a
   flag, off by default.

## Acceptance (pre-registered)
- **Tests:**
  - parity with torch per row, within 1e-4 relative in fp32: mixed adapters, a row with none, different ranks;
  - index -1 equals the base, bit for bit;
  - a batch with one adapter equals today's single-adapter path.
- **Measured:**
  - decode tokens per second at batch 8 and 32, with 1, 4 and 16 distinct adapters, against today's swap-per-adapter
    path;
  - memory per resident adapter.
- **Line:** with 16 distinct adapters at batch 32, at least 3x the swap path's throughput, and within 20% of the
  single-adapter batch.
- **Kill:** no better than swapping. Serving then stays one adapter per batch, grouped by the router.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- grilly2 only, on a branch Nick reviews. The Vulkan code is linked through `grilly_core`.
- **grilly2 is open source:** the operation is general (any peft LoRA), not Cubby-specific, and its tests live in
  grilly2's own suite.
- Nothing reaches the serving default until Cowork has read the report.
- No local path in a committed file.

## Out of scope
- **The rest of the `grilly.agentic` plan** (hypernetwork agents, runtime experts, cellular automata, Hebbian fast
  weights, nested learning, the agent registry): CubbyLLM hypotheses, not grilly2 modules.
- **The spiking neuron step on Vulkan:** its own order, if 0053's learner passes and needs serving speed.
- **The hypernetwork itself (T6):** trained in CubbyLLM, once the library of adapters exists.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
