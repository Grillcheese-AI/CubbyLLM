# 2026-09-12 — the hippocampal side cortex: certified chains, remembered, proposed

**The ask.** A hippocampal memory like GrillCheese's, compressed, as a side cortex that gathers past facts.
First job chosen: certified-chain recall (an episode is the question, the plan that covered it, the chain the
VM certified, the answer, the provenance; a recalled chain is a *plan candidate* through the same gate).
Substrate chosen: SimHash 256-bit codes, what `EpisodicStore` already uses.

**Why not a weight memory.** The July record settles it: `exp_a_forgetting`, `exp_a5_sparse`, `exp_g2_nyt`
(2026-07-23) ran Hebbian, NLMS, SDM and DG-sparse storage at D=1024–2048 — every rule loses recall between
512 and 1,024 associations (dense 100% at 256 → 0% at 512; DG sparsity, the pattern-separation mechanism,
100% at 512 → 0% at 1,024; on correlated real-text keys all four under 20% at 1,024). The world holds
552,297 facts and the loop learned 1,715 more in an afternoon. A compressed *weight* memory of past facts
would forget them. An explicit store of *codes* does not: 256 bits an episode, 18 MB for the whole world.

**What was built** (`cubbyllm/reasoning/hippocampus.py`, Apache-2.0 trunk, stdlib only):

- *DG* — `encode`: the content words of the question, the plan's relation labels and its seed (frame words
  out), each hashed to a 256-bit bipolar vector, bundled by majority, binarised. Two questions that differ
  only in their frame land at Hamming distance 0; two chains about different entities land apart.
- *CA3* — `recall`: nearest live episodes by Hamming distance (a scan over Python ints; ~50 ms at world
  scale). `recall_shape`: nearest *distinct chain shapes* (relation labels alone, entity-free), one episode
  per shape. `propose` completes the pattern into plans: the recalled episode's own plan, then the nearest
  shapes with the seed **rebound** to the entity the new question names (`residual_entity`: the whole
  leftover span, never a fragment, so a shorter entity's facts cannot be walked for a longer one).
- *consolidation* — the episode's facts live in the world store (they went through the gate); the
  hippocampus keeps the code, the plan, the provenance and a utility count (`reinforce` on a verified
  recall); `consolidate(keep)` retires the low-utility episodes beyond `keep` to the cold list. Retire,
  never delete. `save`/`load`: one jsonl line per episode.
- The invariant: **memory proposes, the host disposes, the VM is the truth gate.** Every candidate goes
  through `covers()`, the walk and the VM exactly as an emitter's plan does. Pinned (`test_hippocampus.py`,
  7): distance 0 across the four forms; recall + verify from every form; rebinding to a new entity;
  a recalled plan for a question it does not cover (a dropped hop, an extra hop) is refused; consolidation
  retires by utility and deletes nothing; save/load round-trip; the residual entity is the whole span.

**Measured** (`exp_r13_hippocampus.py`, the exp_r9 matched pairs: same world, 200 functional two-hop
chains × four surface forms, same disposer, walk, resident VM, tau floors; the chains split in two —
100 *seen*, whose canonical question the grammar plans and the VM certifies, written as episodes; 100
*unseen*, whose entities no episode names).

| phase | form | verified · correct · **wrong** | of | how |
|---|---|---|---|---|
| A — seen chains, out of basin | have | 91 · 91 · **0** | 100 | recalled |
| | relative | 91 · 91 · **0** | 100 | recalled |
| | possessive | 90 · 90 · **0** | 100 | recalled |
| B — unseen chains, all four forms | each form | 7 · 7 · **0** | 100 | rebound (ceiling 11: the unseen chains whose shape a seen chain shares) |

Out of basin on the seen chains: **272 of 300 correct, 0 wrong** — the 28 missing are the 9 chains the
grammar did not certify in phase 0 (no episode to recall; the other chains' plans were proposed and
refused, 54–59 per form) and one possessive. Beside it, exp_r9's emitter on the same forms: 228 of 600,
possessive **21 of 200**; here possessive is **90 of 100**. The possessive hole was the emitter folding the
inner hop into the seed on a shape it never saw; a certified chain remembered has no hops to fold.

Phase B is ceiling-bound by the sample, not the mechanism: 85 distinct shapes are remembered and only 11 of
the 100 unseen chains share one; 7 of those 11 verify by rebinding, 0 wrong (`What is the part of the part
of Ancient Higher Learning Institutions?` → *Observatories*, via the episode for *Chikushi Mountains*).
1,200 VM walks of memory-proposed plans in the whole run, **0 wrong**; 37 s wall, no model in the loop.

**What this is and is not.** It is the loop's own record given a recall path: the harvest jsonl was always
the episodic memory, with no way to ask it. It is not a source of facts (the facts are the world store's)
and not a judge (the disposer and the VM are). What it buys: the out-of-basin forms for free once a chain is
certified; analogical transfer of a chain's shape to a new entity where the shapes overlap; and, next, the
context-to-entity binding the James Young case needs (an episode remembers which item "James Young" +
"Missouri politician" resolved to, with provenance — the second job).

**Hops, on complexity (Nick, 2026-09-12: "depending on the complexity of thinking we can allow more
hops").** The hop budget belongs to the question, not to a fixed cap: the disposer's residual rule already
names the relation words a plan left unbound, so a recalled 2-hop shape plus the question's leftover
relation is a 3-hop candidate — composition, with the certified sub-chain's facts known-good and only the
new hop uncertain. The tau floors exist for 1–3 hops (the harvest's frame-size floors); 4 and beyond need
their own calibration run before a 4-hop chain can be spoken. Not built today; the API (`propose`) is where
it goes.

`exp_r13_hippocampus{,_shape}.*`, `exp_r13_episodes{,_shape}.jsonl`.
