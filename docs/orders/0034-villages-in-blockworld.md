# 0034 — Villages in BlockWorld: the spatial link and the scene read on real villages, and a plan placed in the world (H-S12 gate 3)
status: review
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
- **Commits.**
  - This repo:
    - `e9c32da`: scene files, the runner, the question generator, and the ask loop counting a scene's relations;
    - `ef43668`: the act line, counted;
    - `656a5fd`: the measurements and the diagnosis;
    - and this report.
  - BlockWorld (private, its own remote):
    - `d8de104`: the exporter and the act arm;
    - `5fca8e4`: each plan records how it was drawn and the cell expected;
    - `cf3ccc3`: a village that cannot take an order is skipped.
- **Files.**
  - Public:
    - `cubbyllm/spatial/scene_file.py` (the `cubby.scene/1` format; `crop`);
    - `cubbyllm/spatial/scene.py` (`Scene.from_json` / `to_json`);
    - `cubbyllm/reasoning/ask_relation.py` (a mounted scene's relations are relations);
    - `validation/s12_questions.py` (the 30 questions, fixed);
    - `validation/exp_s12_scene_files.py` (the runner);
    - `validation/s12_act.py` (the act counts);
    - `validation/diag_s12_names.py` (the diagnosis);
    - `tests/spatial/test_scene_file.py`, `tests/spatial/test_s12_scene_files.py`.
  - Private (BlockWorld):
    - the exporter, the act arm and their test;
    - the scene files, the fixed questions, the detail log and the act log, all in its git-ignored run folder.
    - Nothing from BlockWorld is in this repo: its logs here are counts.
- **Tests:** `python -m pytest tests standin/tests`: **1,232 passed, 1 failed, 6 skipped**
  ([log](../../validation/logs/0034_tests_2026-10-22.json)).
  - Against 0033's file for file, nothing lost; the new ones are 11.
  - The red is the known one (order 0002).
  - BlockWorld's new test: 3 passed (the real VM; it skips without CubbyLLM and cubelang beside it).
- **Measured** ([log](../../validation/logs/exp_s12_scene_files_2026-10-22.json), counts only).
  - **Villages:** 60 = 20 villages × 3 days, exported headless by the world's own sim.
  - **Questions:** 3,600 = 30 per village and view, fixed before the run (`questions.json`, sha256 `80b60cc3…`).
    The generator was committed (`e9c32da`) before any village existed.
  - **Cut:** 4–23 things (median 9; median 9 left out by the 3 levels round the village centre and the 32-thing budget).
  - **Whole:** 8–43 things (median 20).
  - **Host, `AskLoop(scene=)` with the CubeLang VM: 0 wrong spoken in 3,600.**
    - It routes 1,020 of 1,800 per view: position, the one-step neighbour, the nearest of a kind, the region.
    - Of those, every question with an answer was spoken right: 779 of 779 cut, 793 of 793 whole.
    - Every question whose answer is "nothing" was refused: 241 cut, 227 whole.
    - It routes none of "*n* steps away", "what is at *x y z*" or "where is *X*": 780 per view, since the grammar
      has no route from those words to the scene. The order counts these as a finding, not a failure.
  - **Trunk, run 5 (link + copy, torch on the CPU), on the placed shapes it was trained on (neighbour, offset, at):**

    | answer class (cut) | exact |
    |---|---|
    | things (the kind split) | **0.006** (2 of 329) |
    | names | **0.000** (0 of 120) |
    | "nothing" (kept apart) | 0.972 (380 of 391) |

    - Whole villages: 0.006 / 0.000 / 0.951.
    - **The kill line is met: kinds below 0.75. The synthetic villages did not transfer.**
    - The four shapes run 5 never saw, asked unplaced and reported apart: positions 0.137 exact (74 of 540).
  - **Diagnosis**, apart from the gate ([log](../../validation/logs/diag_s12_names_2026-10-22.json)): the same
    villages, cells and fixed questions, every thing renamed to a shard word of its kind.
    - Things: **0.581** (261 of 449), against 0.0045 as exported; "nothing" 0.954.
    - So the miss is mostly the **words**: the trunk opens a thing it never saw as a village thing with "nothing".
      Teacher-forced, it often finishes the right word after that wrong first token ("nothing" then "bench"), but
      the copy does not give the first token.
    - The **places** cost too: 0.58, not the shard's 0.95.
  - **Act** (the world's act arm; counted by `s12_act.py`): **20 villages that could take a placement order, 20
    markers standing on the computed cell, 0 placed anywhere else.**
    - The host's cell was the test's expected cell in all 44 plans.
    - **22 of 22 unplaceable targets were refused before any order was sent:** 12 occupied, 9 with nothing under
      them, 1 where a villager stood.
    - Two more plans went to villages still too young to take the order. The world refused them; they are
      counted and kept. The act arm now checks first and skips such a village (`cf3ccc3`), and two more villages
      made up the 20.
- **Not verified here.**
  - Nothing needed a GPU: the trunk ran through torch on the CPU, 439 s for all 3,600 answers.
  - What a scene holds is the exporter's choice, and so mine:
    - left out: plain rock and worn paths (ground, hundreds of cells), and a second villager on an occupied cell;
    - in: the structures where they stand, the villagers, trees, exposed resources and fields.
  - The trunk was not run with the questions stripped of their address (the learned route); every placed question
    carried its cell, as in run 5's training.
- **Found on the way.**
  1. **The world gives its villagers no names.** The exporter names them from each body's fixed identity (two lists
     of invented names), so the names split is on names no model has seen. That is what the copy was meant for;
     it did not fire.
  2. **`AskLoop(scene=)` routed no relation question before this order.** "What is the east neighbour of the well?"
     was read as a profile of "east neighbour of the well", because the relation check never asked the scene. Fixed
     in `e9c32da`: only with a scene mounted, and nothing changes without one.
  3. **43% of the questions have no host route** ("*n* steps away", "at a cell", "where is X"). These are the
     shapes run 5 reads best when the host gives it the address.
  4. **The kind-or-position line.** The order's split is "a kind or a position", but run 5 was trained only on the
     placed shapes, whose answers are things. A position answer comes only from a shape it never saw. I gated on
     the placed shapes and reported positions apart, and pre-registered that in the runner's docstring before the
     run (question 3).
  5. **The "nothing" prior, seen directly.** Even on synthetic shard-style villages (a smoke test, not logged),
     neighbour and offset questions whose cell holds a thing were mostly answered "nothing". The shard asked them
     where the answer was nearly always nothing; the draws here point half of them at things.
  6. **The world's one placement order exists only in a developed village.** Two orders were refused for that
     before the check was added.
- **Questions.**
  1. The kill's next step is a private shard of exported villages. Given the diagnosis, should it teach the words
     (a thing's name copied from the village written before it), the layouts, or both? Is run 5's copy, trained on
     far facts, meant to fire on village names?
  2. Routes for the host's three missing shapes ("*n* steps *d* of X", "what is at *x y z*", "where is X"): the
     scene already answers each exactly. An order?
  3. Is gating the trunk on the placed shapes, with positions reported apart, the reading you meant by "a kind or a
     position"?
