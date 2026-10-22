# 0031 — Out of the stand-in: what is Cubby moves into cubbyllm/, the product layer into cubby_serve/, and standin/ keeps only the stand-in
status: in progress (step 1 reviewed 2026-10-22: step 2 may start, with the changes below)
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
### Step 1: the move map (2026-10-22), for review before any move
- **Commits:** `231348e` (in progress), and this one (the map, nothing moved).
- **Move map:** `docs/0031_move_map.md` (the table, every module), `validation/logs/move_map_0031_2026-10-22.json` (with
  every importer). All 146 modules of `standin/` (38,387 lines):
  - **71 to `cubbyllm/`:** worlds 22, sources 3 + news 9, affect 9, cortex 7, plugins 8, speech 5, reasoning 4, emit 3,
    the brain 1;
  - **12 to `cubby_serve/`;**
  - **34 builders beside their trainers in `validation/`:** `sft/` 20, for the 450M's talk and emitter adapters;
    `build/` 14, for the worlds, the lexicon, the ear and the benches;
  - **29 stay:** the game 16, the GGUF seat and the emitter facade, the LFM source, the proposer, the stand-in's eval,
    5 probes, and 3 builders that serve the GGUF adapters only.
- **What the map found:**
  - **18 imports would cross the new lines.** For example: `ask` imports the LFM source and the stand-in eval's
    `strip_fences`; the brain imports the emitter's prompt format from an SFT builder; `feeling` imports the game's
    paths; `grounding` imports the game's `numbers`. Each gets a behaviour-free fix, a helper moved or an object passed
    in, as its own commit before its slice. The map lists them one by one.
  - **Four files split before the move,** each a commit with its tests pinning it: `serve.py` (`CubbyBrain` to
    `cubbyllm/brain.py`, `build_serve` to `cubby_serve/build.py`), `identity.py` (the voice rules leave, the SFT
    records stay with the builders), `emitter_base.py` (the contract leaves, the GGUF templates stay) and
    `import_science.py` (the mount leaves, the import goes with the builders).
  - **27 moving files are over 300 lines;** they are split after the move, the order's five first.
  - **The tests:** all 81 test files in `standin/tests/`, each placed by its name or by what it covers. 13 stay with the
    game; 3 test `validation/` scripts and go to `tests/validation/`; the rest go to `tests/worlds/`, `sources/`,
    `affect/`, `speech/`, `reasoning/`, `emit/`, `plugins/`, `cortex/`, `brain/`, `serve/` and `build/`.
- **The shims:** a left-behind `standin/<name>.py` makes the old name the same module object (`sys.modules`), so flat
  imports, private names and monkeypatches keep working, with a `DeprecationWarning`. A script run by its path runs
  its new home.
- **Questions** (in the map; Nick's calls marked):
  1. **The builders' licence** (Nick): `validation/` is Apache-2.0, so moving them there relicenses 34 BSL files. Is
     that right, or should they have a BSL folder of their own?
  2. **The network in `cubbyllm/`:** `sources.py` says "The network lives HERE, not in cubbyllm/". Keep the order's
     `cubbyllm/sources/` and retire that rule, or put the fetchers in `cubby_serve/`?
  3. **`cubby_serve/` reaching the stand-in:** may it import `emitter_llama`, `lfm_source` and `pacman` lazily, behind
     `--gguf`, `--web-reader` / `--ask lfm` and `--pacman` only, with a guard listing them?
  4. **`vault`:** into `cubbyllm/speech/` (recommended; the context graph encrypts its saves with it), or passed in by
     the server?
  5. **`standin/cortex/`** to `cubbyllm/cortex/`?
  6. **The file names,** with the package prefix dropped?
  7. **`ear.py`'s wiring string** names the private BlockWorld repo: reword it when it moves?
- **Not done (step 2 waits for the review):** the moves, the shims, the guards, `pyproject.toml`, the licences, the docs.

## Review of step 1 (Cowork, 2026-10-22): the map is accepted with five changes; step 2 may start
Read `docs/0031_move_map.md` and its JSON. The map is careful: every module placed, the 18 crossings each with a
behaviour-free fix, the splits before the move, and shims that keep the old module object (`sys.modules`), so private
names and monkeypatches survive. **Step 2 may start, with these changes.**
**Nick's answers:**
1. **The builders: Apache, in `validation/`** (Nick, 2026-10-22). How Cubby is made and how it thinks are open; the
   product layer is BSL.
2. **`cubbyllm/` stays offline** (Nick, 2026-10-22): the standing rule holds. What fetches goes to
   `cubby_serve/sources/` (BSL): the HTTP calls, the polite fetcher (robots, `Crawl-delay`, the `CubbyLLM/0.1` name),
   the feed poller and the backfill loop, the web search and page reads. `cubbyllm/sources/` keeps the logic: Wikidata
   claims rendered as facts, the news items' parsing, events by day, the reply, the trigger, the robots *parser*, the
   web source's corroboration; each takes a `fetch` callable it is given and never opens a socket. The plugins' `Ctx`
   gets its fetch the same way. A guard: no module in `cubbyllm/` imports `socket`, `urllib.request`, `http.client` or
   `requests`, lazily or not (`plugins_guard`'s own list of forbidden names is data and stays). Split the mixed files
   (`web_source` 1,025, `news_archive`, `sources`) along that line in their split commits.
**Cowork's answers:**
3. Yes: `cubby_serve/` may lazily import `emitter_llama`, `lfm_source` and `pacman` behind `--gguf`, `--web-reader` /
   `--ask lfm` and `--pacman` only, with the guard naming exactly them. They go when the GGUF seats retire and the game
   finds its home.
4. `vault` into `cubbyllm/speech/vault.py`: keeping a user's chat encrypted is Cubby's rule. Its key and where it
   writes come from the server, never from the package.
5. Yes, `standin/cortex/` to `cubbyllm/cortex/`.
6. Yes, drop the prefixes, with one change: two modules called `lexicon` in one package is a trap. The learned affect
   of words becomes `cubbyllm/affect/word_affect.py`.
7. Yes, reword `ear.py`'s `__wiring__` so it names no private repo.
**Changes to the map:**
- **Crossings the scan missed.** Lazy imports of `validation/` scripts through `sys.path`: `ask.py` imports
  `exp_r9_matched_pairs.emitted_plan`, and `emitter_450m.py` imports `exp_e5_base450m_probes.load_grilly`;
  `loci_reader.py` puts the repo root on `sys.path` for the same reason. `emitted_plan` moves into `cubbyllm/reasoning/`
  and the base's loader into `cubbyllm/ops/` (see the next point); no module in `cubbyllm/` edits `sys.path`. Re-run
  the crossing scan over function-level imports and `sys.path` edits, and list what it adds.
- **grilly.** `CLAUDE.md`: only `cubbyllm/ops/` imports grilly. `emitter_450m.py` (the 450M on grilly2) and
  `capsules.py` import it. Their grilly calls go through `cubbyllm/ops/` (a small `ops/grilly_seats.py`: load an
  export, copy a LoRA pair in, decode a step), and the seats call that.
- **Baseline first.** Land 0030's red-test fixes before the first `git mv`, so "every test that passed before passes
  after, by count" starts from a suite that means what it says.
- **`import cubbyllm` stays torch-free and grilly-free.** Add the test (a fresh interpreter, `import cubbyllm`, then
  neither in `sys.modules`); the moved modules' `numpy` at module level is fine.

