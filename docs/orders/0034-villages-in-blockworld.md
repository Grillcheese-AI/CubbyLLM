# 0034 — Villages in BlockWorld: the spatial link and the scene read on real villages, and a plan placed in the world (H-S12 gate 3)
status: in progress
written: 2026-10-23 by Cowork · approved: 2026-10-23 by Nick ("yes") · hypothesis: H-S12 (gate 3), H-K10

## Why
Nick (2026-10-23): "we never tested the villages in blockworld". True: every village number so far is on the
synthetic shard (`make_spatial_shard.py`: 8-32 things of six kinds in a 24x24x3 box). Run 5 reads those at 0.953, but
0001's review showed what that number is: 76% of the questions answer "nothing", and the answers come from a closed
set of kind words, so copying a name was never tested on a village. H-S12's gate 3 (a real village through
`AskLoop(scene=)`, and a plan placed through BlockWorld's orders on the cell the scene computed) was never run.
BlockWorld's village sim makes real villages: buildings, roads, named villagers at work and the resources round
them.

What it brings: the first check that Cubby's sense of place works on a world it did not generate for itself, and that
what it reads it can act on. A conventional model reads a map as text and has no cell to land a plan on.
In plain words: Cubby looks at a real village, answers where things are, and puts something exactly where it said.

## Build
**BlockWorld stays private.** Nothing from its code or its worlds comes into this repo. The dependency runs one way:
BlockWorld's side may import `cubbyllm/`, never the reverse. Scene files, logs with layouts and the runner that
touches BlockWorld live in the private repo (or off-repo); this repo gets only generic pieces and aggregate numbers.
- **Public (this repo):** a scene file format and `Scene.from_json` / `to_json` in `cubbyllm/spatial/` (things:
  name, kind, cell; regions; nothing BlockWorld-specific), if they are not there; `validation/exp_s12_scene_files.py`,
  which reads scene files from a folder named by the environment, writes each as a document the way the shard does
  (along the Hilbert curve, places on the writing tokens, the question's `here` placed), asks the questions, and scores
  (a) the host, `AskLoop(scene=)`, and (b) the trunk, run 5's checkpoint with its link and copy, against `Scene`'s
  exact reading. Its logs go to that off-repo folder.
- **Private (BlockWorld):** an exporter that runs the village sim headless (several seeds, several days) and writes
  one scene file per village and day, with the villagers named; and the act arm: Cubby's plan ("put a marker 3 east of
  the store") resolved by `AskLoop(scene=)` to a cell, sent as one of BlockWorld's placement orders, and checked in
  the world.
- **Questions** (the shard's shapes): position, the n-th neighbour in a direction, the nearest thing of a kind, the
  region, what is at a cell, and "where is <villager>?". Fixed before any run; at least 20 villages, 30 questions each.
- **Size:** villages bigger than the shard's box are cut to a box round the village centre with at most 32 things (the capacity
  law's comfortable range at d = 1024); the whole villages are read too and reported apart, as a capacity check.

## Acceptance (pre-registered)
- **Host:** every question the loop's grammar parses is verified by the VM, 0 wrong spoken; report how many it parses
  (BlockWorld's kinds and names the grammar does not know are a finding, not a failure).
- **Trunk, three splits reported apart:** answers that are a kind or a position at least 0.90 exact; answers that are
  a villager's name (open vocabulary: the copy) at least 0.80; "nothing" answers kept separate, so the rate is not
  carried by empty cells.
- **Act:** 20 plans, 20 placements standing on the computed cell, 0 placed anywhere else; a target that is occupied or has
  nothing under it is refused before the order is sent.
- **Kill:** the trunk below 0.75 on kinds and positions: the synthetic villages did not transfer. The next step is
  then a private shard of exported villages to continue on, not more synthetic ones.

## Constraints
- Rules in `CLAUDE.md` hold. BlockWorld's privacy rule above. Runs on the local card or CPU; the link has no grilly2
  export yet (H-S12), so the trunk runs through torch.

## Out of scope
- Training anything; the Unity view; Cubby v0's own map (`vsa.py`) and the trunk's scene in one space (later).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
