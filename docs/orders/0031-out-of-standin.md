# 0031 — Out of the stand-in: what is Cubby moves into cubbyllm/, the product layer into cubby_serve/, and standin/ keeps only the stand-in
status: in progress (step 1: the move map)
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("both, in that order"; "pause standin additions"; licence "split by kind") · hypothesis: none · **before any new feature; 0032 follows**

## Why
Nick (2026-10-22): "code is implementing what we need ... but in the standin instead of the real model". Since the
loci-token orders, the model-side work went into `cubbyllm/model/` (0001, 0004-0006, 0021, 0022), but everything about
knowledge and serving went into `standin/`: the books and worlds (0012-0015), the audit rules (0016), learning on a
miss (0007), the news (0024), the board (0027), the plugins (0028). Cowork's own orders named those paths. `standin/`
is, by its own README, "the stand-in serve stack (control arm) ... Nothing here is CubbyLLM ... replaced the day the
trunk exists", and it is under the Business Source License, where `cubbyllm/` is Apache-2.0. Meanwhile the trunk does
sit in its seats: the 450M serves the emitter and talk roles on grilly2. So the code is real and runs with the real
model; it lives in the wrong place, under the wrong name and licence. This order moves it, with no behaviour change.
Order 0032 then makes the trunk itself read the worlds.

What it brings: one package that is Cubby, a product layer kept apart, and a `standin/` that can be deleted the day
the GGUF seats retire. In plain words: Cubby's brain lives in Cubby's own home, and the temporary scaffolding is
labelled as such.

## Build
- **Three homes, by kind (Nick's licence choice):**
  - `cubbyllm/` (**Apache-2.0**), what makes Cubby think. New subpackages, names to confirm in the move map:
    `cubbyllm/worlds/` (the stores and their routing, the `Knows` / `CubbyPlugin` surface, the manifest, ingest,
    adapters, fiction worlds, the hdc store, books and their frames and reader, the encyclopedia, wikitext, the history
    graph and its lookup, the time rule, capsules, the self source, world hygiene); `cubbyllm/sources/` (Wikidata, the
    web source, robots rules, the news archive and its parts: parse, events, GDELT, Global Voices, backfill, reply,
    trigger, the news source); `cubbyllm/reasoning/` (exists: the ask loop, question forms, the context graph, the loci
    reader join it); `cubbyllm/affect/` (neurochem, the ear and its lexicon, the learned lexicon, feeling, perception,
    choosing, saying, afferent); `cubbyllm/speech/` (grounding, the explicit gate, the audience rules of 0016, the
    VM-mediated chat turn); `cubbyllm/emit/` (the emitter contract, the 450M's seats: `emitter_450m`, `emitter_talk`);
    `cubbyllm/plugins/` (index, route, guard, load, the plugin world, the question encoder, the CLI);
    `cubbyllm/brain.py` (`CubbyBrain` and its cortices, out of `serve.py`); `standin/cortex/` if it is the sensory
    cortex the specs describe.
  - `cubby_serve/` (new package, **BSL 1.1**, its own `LICENSE` copied from `standin/LICENSE`), the product layer: the
    HTTP server (`serve_api` and its `get` / `post` / `cli` / `events` parts, `build_serve`), the harness and its
    wiring, the board's routes and data, the plugins' server part, the vault, the ledger. `dashboard/` stays where it
    is and is served from here.
  - `standin/` keeps only the stand-in: the GGUF seat (`emitter_llama`), the LFM as a source, the OpenRouter proposer,
    the stand-in's evaluation, `models/`, the SFT data builders that only serve the GGUF adapters. **The cubby-man game**
    (`pac*`, `verse`, `world`, `coach`, `forge`, `programs`, `hypothesis`) stays put in this order: where it belongs (with
    the cubbyverse, or `cubbyllm/play/`) is Nick's call later.
- **Step 1, the move map, before any move:** a table of every module in `standin/` (and `standin/data/`): its new
  path, or "stays" and why; who imports it; which tests cover it. Post it in this order's report and wait for Cowork's
  review (Nick is told) before step 2.
- **Step 2, the moves, in slices** (worlds; sources; affect and speech; reasoning; emit; plugins; the brain;
  `cubby_serve`), one slice a commit:
  - `git mv`, so history follows each file; the move commit changes paths and imports only, byte for byte otherwise.
    A file over 300 lines is split in the next commit, not during the move (`ask.py` 1,694, `web_source.py` 1,025,
    `harness.py` 873, `neurochem.py` 816, `serve.py` 811 first).
  - Flat imports (`from audience import ...` with `standin/` on `sys.path`) become package imports.
  - For one cycle, `standin/<name>.py` stays as a shim that re-exports from the new home with a `DeprecationWarning`,
    so notebooks, scripts and local notes keep working; the shims go in a later cleanup.
  - The tests move with their code (`tests/worlds/`, `tests/sources/`, ...) and `pyproject.toml` installs `cubbyllm`
    and `cubby_serve`.
- **Guards:** `cubbyllm/` imports nothing from `cubby_serve/` or `standin/`; `cubby_serve/` may import `cubbyllm/`;
  `validation/` may import both; a test pins each rule, and another lists what `standin/` may still hold.
- **Licences:** the README's licence section and each package's `LICENSE` say which is which; a file's header names
  its package's licence where headers exist.
- **Docs:** `CLAUDE.md`'s "Where to look" table, `PACKAGE_LAYOUT.md`, `docs/WORLDS.md` and the README's map point at
  the new paths; the orders keep their old paths as history.

## Acceptance (pre-registered)
- Every test that passed before passes after, by count, per test file (the moved ones in their new place); the red list
  is unchanged.
- The real stack gives the same answers: 20 fixed questions through `/ask` (the 0027 real-stack check's setup)
  byte-identical replies before and after, and every `/board/*` response equal but for times.
- The old entry points still start the server through the shims (`python standin/serve_api.py ...`), and the new one
  (`python -m cubby_serve ...`) does the same.
- The import guards pass; `standin/` holds only what the move map says stays.

## Constraints
- Rules in `CLAUDE.md` hold (`__wiring__` moves with each module; < 300 lines a file after the split commits).
- **No new feature in `standin/` from now on** (Nick, 2026-10-22). Work in an order that would add to `standin/`
  waits for its slice to move, or goes straight into the new home.
- No history rewrite; no local paths in moved files (0030's guard covers them).

## Out of scope
- Behaviour changes of any kind; the game's home; deleting `standin/` (it goes when the GGUF seats retire).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Move map:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
