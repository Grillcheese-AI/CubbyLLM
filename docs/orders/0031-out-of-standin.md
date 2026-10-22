# 0031 — Out of the stand-in: what is Cubby moves into cubbyllm/, the product layer into cubby_serve/, and standin/ keeps only the stand-in
status: done (2026-10-22)
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

### Step 2: the moves (2026-10-22), for review
**Done.** What makes Cubby think is in `cubbyllm/`, the product in `cubby_serve/`, and `standin/` holds only the stand-in
and the one-cycle shims. Every check is against the baselines taken before the first `git mv`.

**Commits.** `562e66f` (baselines) through `7f3b26a` (the last split): 76 `[0031]` commits on `master`, pushed, and this report. Each slice has
a move commit (`git mv`; paths and imports only; history follows each file) and a shim commit; each fix and each split
has its own commit. `git log --oneline 562e66f^..HEAD | grep 0031` lists them.

| slice | where it went | move, shims |
|---|---|---|
| worlds (21) | `cubbyllm/worlds/` | `589fa14`, `33d7771` |
| sources (12) | `cubbyllm/sources/`, `cubbyllm/sources/news/` | `8415138`, `3787854`, `6bd9104` |
| affect and speech (14) | `cubbyllm/affect/`, `cubbyllm/speech/` | `88c1cae`, `22e8ad2` |
| reasoning (4) | `cubbyllm/reasoning/` | `84d8906`, `340ec6b` |
| emit (3) | `cubbyllm/emit/` | `da9e0a8`, `a398428` |
| plugins (8) | `cubbyllm/plugins/` (`python -m cubbyllm.plugins`) | `5e30232`, `7001926` |
| cortex (7) | `cubbyllm/cortex/` | `86245ac`, `54236dd` |
| the brain (1) | `cubbyllm/brain.py`; `build_serve` and the CLI to `cubby_serve/build.py` | `cf73b96`, `4855789`, `d8b4311` |
| the server (11) | `cubby_serve/` (`python -m cubby_serve`), `cubby_serve/board/` | `2ba7ea5`, `63cef3b` |
| the builders (36) | `validation/sft/`, `validation/builders/` | `62a49e4`, `341ec83`, `4319d85`, `b8c6092` |

**Against each acceptance line:**
- **Every test that passed before passes after, by count, per test file.**
  - The full suite after every slice and every split matched the baseline file for file, with the moved files mapped
    (`validation/test_counts.py --compare`).
  - Now: 1,201 passed, 6 skipped, 1 failed. That is the baseline's 1,189 plus the 12 of the two new guard files.
  - The red list is unchanged: only 0002's wiring test.
  - Three tests were changed, each to follow the code it pins to its new module, with the same assertion:
    - `test_news_archive`'s gzip test patches `urlopen` where the GET now is;
    - `test_web_source`'s key test patches `ENV_FILES` in `web_search`;
    - `test_serve` patches `pipeline_answer` in `brain_reasoning`.
- **The real stack gives the same answers.**
- The stack was booted as the baseline was: the 450M seats on grilly2, wiki and hdc, the ask loop offline, plugins,
    the board. This was at `7f3b26a`, after every slice, the network split and every split.
  - `python standin/serve_api.py`: the 20 fixed questions give **byte-identical replies**; **11 of 13 board routes are
    equal**.
  - `python -m cubby_serve`: the same.
  - In both, `/board/home` and `/board/system` differ only in the order queue. 0028 went from "in progress" to
    "review" in `e7cd011`, and the board reads `docs/orders/`. That is no code difference
    ([log](../../validation/logs/0031_stack_final_2026-10-22.json); after slice 8, the same:
    [log](../../validation/logs/0031_stack_after_s8_2026-10-22.json)).
- **The old entry points start the server through the shims, and the new one does the same.** `python
  standin/serve_api.py ...` and `python -m cubby_serve ...` both booted the stack above. `python standin/serve.py` and
  `python -m cubby_serve.build` run the builder's command line; `python standin/news_archive.py --every 3600` still
  runs the 0024 poller (now `python -m cubby_serve.sources.news`).
- **The import guards pass** (`tests/test_package_rules.py`, each AST scan with lazy imports included):
  - `cubbyllm/` imports nothing from `cubby_serve/`, `standin/` or `validation/`, flat names included;
  - it imports no `socket`, `urllib.request`, `http.client` or `requests`;
  - it edits no `sys.path`;
  - `cubby_serve/` reaches the stand-in only through `emitter_llama`, `lfm_source` and `pacman`, lazily;
  - `import cubbyllm` loads neither torch nor grilly.

  Planted violations made five of them fail with the file and line; the probes were then moved out of the repo.
- **`standin/` holds only what the map keeps there, and the shims** (the sixth guard). Kept: the GGUF seat and its
  facade, the LFM source, the proposer, the stand-in's eval, five probes, the game, and three GGUF-only builders.

**The review's changes, done:**
- builders Apache in `validation/`;
- `cubbyllm/` offline (below);
- the three named stand-in imports, guarded;
- the vault in `cubbyllm/speech/`, its key's place named by the server (`cubby_serve/keys.py`);
- the cortex moved;
- the prefixes dropped, and `word_affect` named;
- `ear`'s wiring reworded;
- the missed crossings fixed (`emitted_plan`, the model builder and the far-fact layout, the seats' grilly calls
  through `cubbyllm/ops/grilly_seats.py`, the fact pool, `step_loop`'s `vm_run`);
- 0030's red tests fixed first;
- the torch- and grilly-free test.

**`cubbyllm/` offline (Nick's rule).** Five modules opened sockets. Each now takes a `fetch` the server hands in from
`cubby_serve/sources/`:
- the wiki world's Hub download (`hub.py`, with a command to fetch the export deliberately);
- Wikidata (`http.wikidata()`);
- the web source's transport (`http.web`);
- the news archive's GET, its poller and its CLI (`sources/news.py`);
- the news source's feeds and robots.txt (`http.news()`).

Built without a fetch, a source says so: an error naming what to hand in, or a feed treated as down with the reason.
It is never silently offline. `tests/test_offline_seams.py` pins each seam.

**Files over 300 lines.** The order's five were split first:
- `ask` (1,691 lines, now eight modules);
- `web` (1,021, seven);
- `harness` (870, four);
- `neurochem` (816, five);
- `brain` (618, four).

Then the other eleven moved package files over 300: `wikidata`, `history_lookup`, `history_graph`, `chat`, the news
source, `word_affect`, `afferent`, `wikikg`, `encyclopedia`, `harness_wiring`, `context_graph`.

The splits used a new tool, `tools/split_module.py`:
- it moves top-level names, or methods and class constants as a mixin, verbatim;
- it writes each new module's imports from what its code uses;
- the old module re-exports every name;
- it refuses a name that would need the old module imported back, an import cycle, or a double-underscore attribute.

For every split, every name reachable before is reachable after with the same source (`inspect.getsource`). For
`neurochem` I also checked 20 messages through HEAD's class and the split one: the same state at every step. No moved or
new package file is over 300 lines.

**Measured on the models** (Nick: "use grilly2"; the latest model, `spatial_link_5`):
- the 450M seats on grilly2, from the moved module: byte-identical to the recording made before the seats' change, the
  same 3 role swaps ([log](../../validation/logs/0031_seats_grilly2_2026-10-22.json));
- the loci bench's phase 2 on `spatial_link_5_final.pt`, through the moved ask loop, loci reader and model builder: 100
  of 100 rows equal to the pre-move log ([log](../../validation/logs/0031_loci_bench_after_2026-10-22.json)).

**Packaging, licences, docs.**
- `pyproject.toml` installs `cubbyllm` and `cubby_serve` (with its `LICENSE`), and its licence text names both.
- `cubby_serve/LICENSE` is a verbatim copy of `standin/LICENSE`.
- Updated:
  - the README's layout, run commands and licensing section;
  - `CLAUDE.md`'s "Where to look" table and its rule pointing at the guard and at `tools/moves_0031.json` (every old
    path and its new one);
  - `PACKAGE_LAYOUT.md`'s current layout;
  - `docs/WORLDS.md`'s code line.

**Not verified here:** nothing ran on Colab. The 450M seats and the real stack ran on the local card with grilly2.

**Found on the way:**
1. **`loci_reader` writes two reasons the sleep cycle has no route for** (`loci_mismatch`, `loci_miss`). They land
   in "unrouted". The routing test keeps its old coverage until they are routed (see question 1).
2. **A regression of mine, caught and fixed.** `b315242` dropped the trunk's grilly handle, which `test_plugins` fakes.
   The next full run caught it; it was fixed in `3b1e4e0` and checked again on the card.
3. **Data places the package still names under `standin/data/out`:** the identity facts, the wiki world's cache, the
   web and news caches, the Wikidata cache, the plugins' lock and data root. They are paths, not imports; the vault's
   key no longer is one of them.
4. Unchanged, as reported at the baselines: "Where is the Globe Theatre?" merges two namesakes; the Brave
   rate-limit timing test is flaky (5 in 20 before the move, 3 in 20 after).

**Questions:**
1. **The two `loci_reader` reasons:** where should the sleep cycle route them?
2. **The polite fetcher.** The socket, the poller and the backfill loop are in `cubby_serve/`. The archive's
   politeness (robots, `Crawl-delay`) stays in `cubbyllm/` as logic over the fetch it is handed, as the web source's
   robots check does. Is that the line you meant, or should the politeness move too?
3. **Question 8, still open:** the serve stack loads mowm's `FastWordEncoder` by file path. The link is where it was,
   lazily, in `cubby_serve/build.py`. Port it into `cubbyllm/`, or leave it?
4. **`cubby_serve/LICENSE`** is the verbatim copy the map asked for, so its "Licensed Work" line names "the standin/
   directory". Should it name `cubby_serve/`? Nick's text to change, not mine.
5. **Ten builder scripts in `validation/` are still over 300 lines** (`build_program_first` 1,005, `build_history_graph`
   569, `build_gen3` 472, `build_emitter_sft` 451, `import_science` 376, `build_ground_sft` 366, `import_tinygsm` 364,
   `build_history_sft` 347, `build_hdc_sft` 322, `convert_books` 302). They are scripts run by their path, so a split
   needs absolute imports rather than the package's relative ones. Split them in a follow-up, or leave them as scripts?
6. **The data places above:** should they move under one data root the server names, as the vault's key now is?
7. **The shims:** they go in a later cleanup. When?

## Review of step 2 (Cowork, 2026-10-22): done
Re-ran the guards (pass) and read the logs: the suite matches the baseline file for file (1,189 + 12 new guards); the
20 fixed answers are byte-identical through both entry points; the 450M seats on grilly2 and the loci bench on
`spatial_link_5` read exactly as before the move. `cubbyllm/` opens no socket, edits no `sys.path`, imports nothing
from `cubby_serve/`, `standin/` or `validation/`, and `import cubbyllm` loads neither torch nor grilly. Every split was
checked name by name (`inspect.getsource`), and the regression in `b315242` was caught by the next full run and fixed.
**Done.** The pause on new work is lifted: it continues in `cubbyllm/` and `cubby_serve/`.
**Answers:**
1. **The two loci reasons get routes** (the table grows by decision): `loci_mismatch` -> a new queue `loci_review`,
   "the trunk's read and the store disagree; nothing was said" (0029's boundary errors and the store's own doubtful
   values both land here); `loci_miss` -> `source_gaps`, "the trunk read nothing at the address and no source
   supplied the fact". A test pins both.
2. **That is the line:** the policy (robots, `Crawl-delay`, the name) is logic in `cubbyllm/` over the fetch it is
   handed. Add one thing on the other side: `cubby_serve/`'s fetch sets `CubbyLLM/0.1` itself and accepts no other
   name, so no caller can change it.
3. **Port mowm's `FastWordEncoder` into `cubbyllm/`** (port, don't link): the serve stack should not need a sibling
   repo on disk at run time. A test pins the port against mowm's output on a fixed word list, while mowm is there.
4. **`cubby_serve/LICENSE`'s "Licensed Work" line** is Nick's text: Cowork is asking him.
5. **The ten builder scripts over 300:** leave them; split each when it is next changed (the rule's "does not get
   longer without a reason").
6. **Yes, one data root** named by the server (`$CUBBY_DATA_ROOT`, defaulting to today's place), so nothing the
   package uses lives under `standin/`, which is meant to go. Move the caches, never re-download them.
7. **The shims go when nothing uses them:** a test counts in-repo imports of `standin/` names that are now shims; when it
   reads 0 and the Colab notebooks are updated, they are removed. Target: with 0032's report.
**Found on the way, to order 0033:** "Where is the Globe Theatre?" answering "London Borough of Southwark, Boston" is a
wrong answer spoken (two namesakes merged), against the project's first rule; it goes first. The flaky Brave timing
test goes with it.
