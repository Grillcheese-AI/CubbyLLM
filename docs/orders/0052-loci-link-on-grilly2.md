# 0052 — The loci link on grilly2: far facts read by their place, served at home
status: in progress
written: 2026-11-05 by Cowork · approved: 2026-11-05 by Nick ("yes") · hypothesis: H-K10 (with H-S12)

## Why
0051's run 6 settled the link at the 450M:
- far facts read at 0.804 exact through their place, against 0.019 for the same text read the ordinary way, at equal
  tokens from the same start;
- the text mix unharmed (2.2398 against 2.2491).

**It cannot serve yet.** `export_base.py` refuses a `spatial` meta, so the link runs only in torch on CUDA (H-S12's
stated cost).

**Serving has a second constraint.** Every link checkpoint so far trained its trunk too, so its trunk is not
`base450m_cont`. 0048's guard would then refuse the programs adapter on it. For the link to join the served stack
without retraining every adapter, it has to work on a frozen trunk.

**What it brings, in plain words:** at home, Cubby can answer about something said 800 tokens ago by going back to
where it was put, instead of losing it once it scrolls out of view.
- **Over a conventional model:** that far back, a model either keeps the whole conversation in a cache that grows with
  every token, or forgets. Here the fact is reached by its address, at a fixed cost.

## Build
**A. The port** (grilly2, never grilly v1; behind a serving flag, off by default):
1. **The export:** `export_base.py` writes the `spatial` meta and the link's weights: the encoding's seed and scale,
   the read's projections and gate, and the copy ring's tau and beta.
2. **grilly2 runs the link:**
   - the place encoding S(p);
   - the scene's writes;
   - the asking token's read at layer L/2;
   - the copy path into the logits.

   How the encoding is computed is Claude Code's choice, recorded: precomputed on the host, or kept in the frequency
   domain so that decode needs no FFT. FFTs on this card are untested.
3. **`cubby_serve` loads a spatial export.** The host supplies each fact's place as the shards do, and a question
   carries its address.

**B. The link on the served trunk:** only the link's and the copy's parameters are trained, on a frozen
`base450m_cont` (`CB_TRAIN_ONLY` with both), on the far-facts shards.
- About an hour on an A100, in a Colab notebook Nick runs.
- If it holds, the trunk's hash is unchanged, so the programs and talk adapters stay valid under 0048's guard, and the
  link can be added to any served trunk later (0045's control final included).

## Acceptance (pre-registered)
- **Tests:**
  - **parity:** torch against grilly2 on a spatial export, logits within 1e-4 relative in fp32 (the talk adapter's
    parity was 1.1e-4);
  - the scene's writes and reads match torch's exactly;
  - with the flag off, outputs are bit-identical, and an export without `spatial` is unchanged.
- **A, on run 6's link checkpoint, at home:**
  - far facts exact on grilly2 within 0.01 of torch's 0.804, on the same held-out documents;
  - decode time per token with the link against without, at batch 1. **Line:** at most +15%.
- **B:**
  - **line:** far facts at least 0.70 exact with the trunk frozen (run 6, which trained the trunk, reads 0.804). The
    text mix cannot move, since the trunk is frozen.
  - **kill:** under 0.50. Then the link rides a trunk retrain (0045's control, or a later continuation), and the
    adapters are retrained on it.
- **The served stack, at home, with B's export:**
  - 0048's guard accepts the existing programs adapter (the same trunk hash);
  - with nothing placed, the programs seat's greedy programs and E8 reworded are unchanged.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Nothing reaches the serving default until Cowork has read the report.
- grilly2 only on the local card. Long GPU jobs go to Colab, which Nick runs.
- No local path in a committed file. BlockWorld's villages stay in its private repo; the far-facts shards are the
  synthetic wiki ones.

## Out of scope
- **The host's writer for live conversations:** which facts get a place as a conversation runs (a stated fact, a value
  the VM returned). That is the next order, from this one's numbers.
- **Questions without their address** (the route alone), and in-window villages.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## B, built (Claude Code, 2026-11-05)
- **`CB_TRAIN_ONLY=link,copy`** (`validation/train_base.py`):
  - the whole link trains: the read's query, gate, output and norm, and the copy path;
  - its peak is `CB_LINK_LR` (default 1e-3, a choice of mine, recorded);
  - every trunk tensor stays the checkpoint's.
  - `train_base.py` was already over 300 lines; it grows by 9 for this.
- **The test** (`tests/model/test_train_only.py`): one step moves only the link's tensors, every trunk tensor stays bit
  for bit, and the peak is `CB_LINK_LR`. 9 pass.
- **The notebook:** `spatial_facts_450m.ipynb`, now `RUN = 7`.
  - Run 6's link and copy flags (placed, writes binding, the route loss, the copy ring with tau 3.04 and beta 0.115),
    added to `base450m_cont_final.pt`, the served trunk.
  - Only the link trains, for run 6's step count (1,203, read from its metrics), on its two shards.
  - The readout reads far facts on the 400 held-out documents, against line 0.70 and kill 0.50. It also checks that
    every trunk tensor and the router are bit-identical to the start.
  - The builder was already over 300 lines (426); it grows by 27, because run 7 shares its corpus, probes and readout
    with runs 1-6.
- **A:** next, at home on grilly2.

## A, steps 1 and 2: the export and the link's decode math (Claude Code, 2026-11-05)
- **How the encoding is computed (the order's open choice):** precomputed on the host.
  - grilly2 has no FFT kernel and no complex type (its holographic bind is a direct O(d²) convolution).
  - At d = 1,024, a real DFT is a (d/2+1) x d matrix product, about 1M multiply-adds a transform, against the
    trunk's ~450M a token. So the transforms are two precomputed real matrices each way, and the rest is elementwise
    or a small linear.
  - The place angles are computed in float32, as torch computes them; everything else in float64 on the host.
- **`cubbyllm/model/spatial_decode.py`** (numpy, no torch): `SpatialDecode`, the link's four decode calls in
  `CubbyModel.step`'s order:
  - `bind_input`: a writer's input bound to its place; a non-writer's input is untouched, bit for bit;
  - `write`: a document start empties the scene and the ring; a writer adds its bound input and its id;
  - `read`: at layer `at`, after the mixer, before the FFN. It adds exactly zero while nothing is written;
  - `add_copy`: the ring copy at asking rows with a candidate.
  - Per stream it keeps the scene (d) and the ring (256 ids): fixed-size, as torch's.
- **`validation/export_base.py`:** a linked base no longer refuses.
  - The link goes **beside** the trunk, in `spatial.json` and `spatial.safetensors`. `config.json` and
    `model.safetensors` stay the trunk's alone, because grilly2's loader refuses a tensor it does not know.
  - An export without a link is unchanged.
  - The variants the port has not taken (a text key, the pointer query, copy-or-refuse, the table copy) refuse.
- **Tests** (`tests/model/test_spatial_decode.py`, 5 pass; `test_export_grilly.py` still 2 pass in grilly2's
  environment):
  - the DFT matrices equal numpy's rfft / irfft;
  - **parity:** the torch trunk stepped with the numpy link where grilly2 will call it gives torch's own step's
    logits at every token, with roles, a document restart, a ring smaller than the writes and the copy's gate open.
    Worst relative gap: 7.3e-6 without the copy, 1.4e-6 with it (line 1e-4);
  - with nothing placed, the logits are bit-identical to the trunk alone;
  - the export's file contract and its refusals.
- **Next, step 3:**
  - grilly2's `CubbyForCausalLM` takes an optional link object: the input bound before layer 0, the read at layer
    `at`, the final stream returned for the copy;
  - `cubbyllm/ops/grilly_seats.py` passes positions and roles through;
  - `cubby_serve` loads `spatial.json` behind a flag (`--spatial`, off by default);
  - then run 6's link exported and read at home (far facts within 0.01 of 0.804), and decode time with and without.

## A, step 3: grilly2 runs the link (Claude Code, 2026-11-05)
- **grilly2** (branch `cubby-spatial-link`, `b80de06`, pushed for Nick's review; grilly2's `main` is untouched, per its
  working agreement):
  - `CubbyForCausalLM.forward(..., spatial=None)` binds the writing tokens' inputs after the context, adds the read at
    layer `spatial.at` between the mixer (and memory) and the FFN, and mixes the copy into the logits;
  - `spatial=None` leaves the trunk as before. grilly2's `tests/python/test_cubby.py` still passes (26).
- **CubbyLLM:**
  - `SpatialDecode.call`: one trunk call over T tokens, a prompt or a step, sharing one stream state;
  - `cubbyllm/ops/grilly_spatial.py` (`GrillyLink`): the three hook points, numpy in and out of grilly2's tensors.
- **Tests:**
  - `tests/model/test_spatial_decode.py`, 7 pass on CPU. A call over a whole prompt is torch's forward, and it leaves
    the state the steps leave.
  - `tests/model/test_spatial_grilly.py`, 3 pass with grilly2's branch, and skip without it. A linked base exported,
    then run by grilly2 with `GrillyLink`, gives torch's logits:
    - over a prompt, 7.2e-6 relative without the copy and 1.7e-6 with it;
    - step by step after a prompt, within 1e-4 (the order's parity line);
    - with nothing placed, grilly2's logits are its own, bit for bit.
- **Not yet:**
  - **run 6's link at home:** far facts on grilly2 within 0.01 of 0.804 on the 400 held-out documents, and decode
    time per token with the link against without (line +15%). This needs grilly2's branch checked out locally and
    run 6's checkpoint (2.3 GB, on Drive) exported. It is next, in short chunks on the local card.
  - **`cubby_serve --spatial`:** loading the link behind a flag, after the read above.
