# Order 0031, step 1: the move map

Every module in `standin/` (146 files, 38,387 lines), where it goes, who imports it and which tests cover it. Nothing has
moved yet: this map waits for Cowork's review (Nick told) before step 2. Machine-readable, with every importer listed:
`validation/logs/move_map_0031_2026-10-22.json`.

| destination | modules | |
|---|---|---|
| `cubbyllm/` (Apache-2.0) | 71 | worlds 22, sources 3 + news 9, affect 9, cortex 7, plugins 8, speech 5, reasoning 4, emit 3, the brain 1 |
| `cubby_serve/` (BSL 1.1) | 12 | the server's five parts, the board's two, the plugins' server part, harness and its wiring, vault, ledger |
| `validation/` (builders, see question 1) | 34 | `validation/sft/` 20: the 450M's talk and emitter data; `validation/builders/` 14: worlds, lexicon, ear, benches |
| stays in `standin/` | 29 | the game 16, the GGUF seat and its facade 2, the LFM source, the proposer, the stand-in's eval, 5 probes, 3 GGUF-only builders |

## How it was decided
- **The order's three homes, by kind.** What makes Cubby think goes to `cubbyllm/`, the product layer to `cubby_serve/`,
  and the stand-in stays. The game stays put, as the order says.
- **The data builders follow the order's rule** ("standin keeps the SFT data builders that only serve the GGUF
  adapters"). `talk_v2_2`, the talk adapter the 450M serves, is built by `build_talk_mix` from nine family builders. The
  450M's emitter (v12e onward) comes from `build_gen2_partition`, `build_gen3`, `build_program_first`,
  `import_tinygsm`, `build_v13f` and `gen3_llm_wordings`, with `build_emitter_sft`'s helpers. Those leave `standin/`.
  Only `build_game_sft` (v4), `gap_families` (v9) and `build_profile_sft` serve the GGUF adapters alone.
- **Builders cannot go into `cubbyllm/`.** Six of them call the frontier proposer (`openrouter`, which stays: datasets
  only), and `cubbyllm/` may import nothing from `standin/`. So they go beside their trainers, which already live in
  `validation/` (`train_talk_torch.py`, `train_emitter_torch.py`).
- **Names inside a package drop the package's prefix:**
  - `plugins_load` -> `cubbyllm/plugins/load.py`, `news_archive` -> `cubbyllm/sources/news/archive.py`,
    `serve_api` -> `cubby_serve/api.py`, `worlds_fiction` -> `cubbyllm/worlds/fiction.py`;
  - `worlds.py` -> `cubbyllm/worlds/base.py`, re-exported by `cubbyllm/worlds/__init__.py`;
  - `sources.py` -> `cubbyllm/sources/wikidata.py` (it holds `WikidataSource` and the property aliases);
  - `emitter_450m` -> `cubbyllm/emit/seats_450m.py`.

  The old names live on through the shims.
- **Two names collide with `cubbyllm/reasoning/`:** `lexicon` and `programs`. `programs` is the game's and stays.
  `lexicon` becomes `cubbyllm/affect/word_affect.py`, a different module from `cubbyllm/reasoning/lexicon.py` (the learned
  affect of words, not the reasoning lexicon).

## What crosses the new lines (18 imports), and the fix for each
`cubbyllm/` may import nothing from `cubby_serve/`, `standin/` or `validation/`. Each fix is behaviour-free: a helper
moves to the side that may own it, or an object is passed in instead of imported. Each fix is its own commit before
its slice moves.

| import | fix |
|---|---|
| `ask` -> `eval_emitter_vm.strip_fences` | `strip_fences` moves to `cubbyllm/emit/base.py`; the eval imports it from there |
| `ask`, `chat` -> `emitter.clean_reply` | import from `cubbyllm/emit/base.py` |
| `ask` -> `lfm_source.LfmSource` (`--ask lfm`) | the server builds the LFM source and passes it to `AskLoop(source=...)`; the loop keeps resolving the names it owns (`wikidata`, `wikidata-offline`) |
| `book_reader` -> `lfm_source.grounded` | `grounded` moves to `cubbyllm/speech/grounding.py`; `lfm_source` imports it |
| `context_graph` -> `vault` (its `save` / `load` encrypt the conversation) | **question 4**: `vault` into `cubbyllm/speech/` (Cubby's own memory is encrypted, a rule of Cubby's), or `save` / `load` take the writer from the server |
| `feeling` -> `pacpaths.PLUTCHIK_JSON` (module level) | the table's place becomes `$CUBBY_GITHUB/cubbyverse/...` through `cubbyllm.local_paths` in `feeling`; `pacpaths` reads `feeling`'s constant |
| `grounding` -> `forge.numbers` (the game's) | `numbers` moves into `grounding`; `forge` imports it |
| `hdc_world` -> `build_hdc_sft.Templates` | `Templates` (how an hdc fact is worded) moves into `cubbyllm/worlds/hdc_world.py`; the builder imports it |
| `import_science` -> `build_program_first.make_writer` | `import_science` splits: `mount_science` (the server's `--science`) -> `cubbyllm/worlds/science.py`; the import of MegaScience rows -> `validation/builders/import_science.py` |
| `serve` -> `build_emitter_sft.answer_fn`, `shim_isolver`, `wrap_role_prompt` (module level) | the emitter's prompt and answer format, used at serve time -> `cubbyllm/emit/format.py`; the builder imports them |
| `serve` -> `emitter.ContextualEmitter` | `cubbyllm/emit/talk.py` |
| `serve` -> `pacman.CubbyPac` (`--pacman`) | lands in `cubby_serve/build.py` with `build_serve` after the split (below) |
| `plugins_serve` -> `emitter.Cubby450mTrunk` | `cubbyllm/emit/seats_450m.py` |
| `worlds_adapters` -> `build_books_index.world_names` | `world_names` moves into `cubbyllm/worlds/books.py`; the builder imports it |
| `serve_cli`, `serve_get` -> `pacman` (`--pacman`); `serve_cli` -> `lfm_source.LfmReader` (`--web-reader`) | **question 3**: allowed as lazy imports behind those flags only, with a guard test naming exactly them |

## Splits before the move (one commit each, behaviour unchanged, the existing tests pinning it)
- **`serve.py` (811 lines):** `CubbyBrain` and its cortices -> `cubbyllm/brain.py`; `build_serve` (the seats, the
  stand-in, the game) -> `cubby_serve/build.py`.
- **`data/identity.py` (523 lines):** the runtime rules the brain and chat use -> `cubbyllm/speech/identity.py`.
  Those are `voice_ok`, `is_identity_question`, `is_identity_reply`, `is_model_guard`, `identity_system`,
  `load_facts` and `sample_state`. The identity SFT record builders stay with the builders.
- **`emitter_base.py` (213 lines):** the contract, `clean_reply`, the decode lock and `ReplayEmitter` ->
  `cubbyllm/emit/base.py`. The llama-server emitter, the chat templates, the script ban and `NON_LATIN` stay with the
  stand-in.
- **`data/import_science.py` (395 lines):** as in the table above.

## Splits after the move (files over 300 lines, one commit each, as the order says)
In `cubbyllm/` and `cubby_serve/`:
- the order's first five: `ask` 1,694, `web_source` 1,025, `harness` 873, `neurochem` 816, `serve` 811 (split above);
- then `sources` 556, `history_lookup` 490, `history_graph` 474, `chat` 413, `news_source` 389, `lexicon` 375,
  `afferent` 362, `wikikg` 346, `encyclopedia` 338, `harness_wiring` 332, `context_graph` 319.

In `validation/` (the builders): `build_program_first` 1,005, `build_history_graph` 569, `build_emitter_sft` 477,
`build_gen3` 472, `build_ground_sft` 366, `build_hdc_sft` 364, `import_tinygsm` 363, `build_history_sft` 347,
`convert_books` 302.

## The moves, in slices (one commit each; the order's sequence, the builders last)
worlds; sources; affect and speech; reasoning; emit; plugins; the brain; `cubby_serve`; the builders. Each slice:
`git mv` (history follows the file), then that commit changes paths and imports only, byte for byte otherwise.

**The shims.** Each `standin/<name>.py` left behind makes the old name the same module object: it sets
`sys.modules[__name__] = importlib.import_module("<new path>")`, with a `DeprecationWarning`. So old flat imports,
private names (`board_api._cache`) and monkeypatches keep working. A file that was run as a script also runs its new
home (`runpy`, `__main__`), so `python standin/serve_api.py ...` still starts the server.

**The guards (new tests):**
- `cubbyllm/` imports no `cubby_serve`, `standin` or `validation` module, lazy imports included;
- `cubby_serve/` imports `standin/` only through the lazy imports named in question 3;
- `standin/` holds exactly the 29 modules this map keeps there (plus the shims).

`pyproject.toml` installs `cubbyllm` and `cubby_serve`. `cubby_serve/LICENSE` is a copy of `standin/LICENSE`.

## Questions for the review
1. **The builders' home and licence.** `validation/` is under the root licence (Apache-2.0), so 34 builders would change
   from BSL to Apache by moving there. They build the training sets of the 450M's adapters and the worlds' data, and
   six of them call the frontier proposer. Is Apache right for them (Nick's call)? Or should they keep BSL in a folder
   of their own (`builders/` with its own `LICENSE`)?
2. **The network in `cubbyllm/`.** `sources.py` states a standing rule: "The network lives HERE, not in cubbyllm/:
   the package chooses and gates, the serve stack fetches under policy." The order puts the Wikidata, web and news
   fetchers in `cubbyllm/sources/`. Keep the order, and retire that rule in the files? Or put the fetchers in
   `cubby_serve/sources/`, keeping `cubbyllm/` network-free with only the Source protocol and the gates?
3. **`cubby_serve/` reaching the stand-in.** The server still has to run the GGUF seat (`--gguf`), the LFM reader
   (`--web-reader`, `--ask lfm`) and the game (`--pacman`). Allow lazy imports of `emitter_llama`, `lfm_source` and
   `pacman` behind those flags only, with a guard test listing exactly them?
4. **`vault`.** It encrypts what is kept of a user's chat (its owner's rule, 2026-09-04), and `context_graph`'s
   `save` / `load` use it. Into `cubbyllm/speech/vault.py` (recommended: it is how Cubby's own memory is kept), or kept
   in `cubby_serve/` with the context graph given its writer?
5. **`standin/cortex/` -> `cubbyllm/cortex/`.** It is the sensory cortex of
   `docs/superpowers/specs/2026-10-10-sensory-cortex-design.md`, and only its test imports it today. Confirm?
6. **File names.** The prefixes are dropped inside packages (`cubbyllm/plugins/load.py`,
   `cubbyllm/sources/news/archive.py`, `cubby_serve/api.py`). Confirm, or keep the old names?
7. **`ear.py`'s `__wiring__`** reads `"BlockWorld --hearing"`. It names the private repo but carries nothing from it.
   Reword it when it moves?

## After the review (2026-10-22): the changes, and what the rescan added
Cowork accepted the map with five changes, and Nick answered questions 1 and 2. This section supersedes the rows it
names.
- **The builders: Apache, in `validation/`,** as mapped (Nick).
- **`cubbyllm/` stays offline** (Nick). What opens a socket goes to `cubby_serve/sources/` (BSL): the HTTP calls,
  the polite fetcher (robots, `Crawl-delay`, the `CubbyLLM/0.1` name), the feed poller, the backfill loop, the web
  search and page reads. The logic stays in `cubbyllm/sources/` and takes a `fetch` callable it is given.
  - **Split along that line, in their split commits:** `sources.py` (Wikidata claims as facts in `cubbyllm`, the
    HTTP call in `cubby_serve`), `web_source.py`, `news_archive.py`, `news_source.py`.
  - **`data/wikikg.py`:** its Hub download (`--wiki auto`, when the export is not cached) becomes a fetch handed in.
  - **The plugins' `Ctx`** gets its fetch the same way.
  - **A guard:** no module in `cubbyllm/` imports `socket`, `urllib.request`, `http.client` or `requests`, lazily or
    not.
- **`vault` -> `cubbyllm/speech/vault.py`;** the key and the place it writes come from the server.
- **`standin/lexicon.py` -> `cubbyllm/affect/word_affect.py`,** not `lexicon.py`, so two modules called `lexicon`
  don't share a package tree.
- **grilly only through `cubbyllm/ops/`:** a new `cubbyllm/ops/grilly_seats.py` loads an export, copies a LoRA pair
  in and decodes a step. `emitter_450m` (the seats) and `capsules` (`grilly.vsa`) call it.
- **`ear.py`'s `__wiring__`** is reworded so it names no private repo.
- **`import cubbyllm` stays torch-free and grilly-free:** a test checks it in a fresh interpreter.

**The rescan.** It covered function-level imports, imports of `validation/` scripts, `sys.path` edits, grilly and
network imports. It adds these crossings, each with its fix: the helper moves into `cubbyllm/` and its old place
imports it back.

| module (bound for) | imports | fix |
|---|---|---|
| `ask` (cubbyllm) | `exp_r9_matched_pairs.emitted_plan` | -> `cubbyllm/reasoning/` (with `ask`'s split) |
| `emitter_450m` (cubbyllm) | `exp_e5_base450m_probes.load_grilly`, `grilly`, `grilly.infer.lora` | -> `cubbyllm/ops/grilly_seats.py` |
| `emitter_450m` (cubbyllm) | `emitter_data.PROMPT_HEAD`, `PROMPT_TAIL`, `decode_program` | -> `cubbyllm/emit/format.py`, beside `answer_fn`, `shim_isolver`, `wrap_role_prompt` |
| `capsules` (cubbyllm) | `grilly.vsa`, `grilly` | through `cubbyllm/ops/` |
| `loci_reader` (cubbyllm) | `train_base.build_model` (the 450M from a checkpoint's meta) | -> `cubbyllm/model/build.py`; `train_base` imports it |
| `loci_reader` (cubbyllm) | `make_fact_shard.address`, `words` (the far-fact layout) | -> `cubbyllm/model/fact_layout.py`; the shard builder imports them |
| `worlds_ingest` (cubbyllm) | `books_bulk_read.run` (the books' bulk read) | the function -> `cubbyllm/worlds/book_reader.py`; the script keeps its CLI |
| `serve` -> `build_serve` (cubby_serve) | `exp_m3_cot_pipeline.load_sample`, `V4_TABLE`; `exp_m3_domain_routing._load_semantic_words` | the serve stack's fact pool and word table -> `cubbyllm/worlds/fact_pool.py`; question 8 |
| `import_science`, `wikikg` (cubbyllm) | `urllib.request` | the import part -> `validation/builders/`; wikikg's download -> a fetch handed in |
| `emitter_base` (cubbyllm) | `urllib.request` (`LlamaServerEmitter`) | stays in `standin/` with the split |
| 21 modules bound for `cubbyllm/` | `sys.path` edits | removed: package imports; a guard checks that no `cubbyllm/` module edits `sys.path` |

**Question 8 (new).** `_load_semantic_words` loads the `FastWordEncoder` class from the sibling `mowm` checkout by file
path, so the serve stack depends on another repo at run time. This is true today, not new. Moving it into `cubbyllm/`
would carry that link into the package. Should the encoder class be ported into `cubbyllm/` (the rule "port, don't
link")? Or should the link stay in `cubby_serve/build.py` for now?

**Before the first `git mv`:**
- **Baselines.** The test counts per file, and the real stack's answers to 20 fixed questions with its `/board`
  responses. These are what "byte-identical after" is checked against.
- **The fixes and splits above, one commit each.** No path moves in them.

## The map, module by module
"Imported by" leaves out tests. "Tests" lists the test files that import the module directly; a module exercised through another (`plugins_check` through the loader) shows "none". The full lists are in the JSON.

### cubby_serve/ (9)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `harness.py` (873 **>300**) | `cubby_serve/harness.py` | `harness_wiring.py`, `serve_cli.py` | `test_harness`, `test_harness_wiring` |
| `harness_wiring.py` (332 **>300**) | `cubby_serve/harness_wiring.py` | `serve_cli.py` | `test_harness_wiring` |
| `ledger.py` (168) | `cubby_serve/ledger.py` | `harness_wiring.py`, `plugins_serve.py`, `serve_cli.py` | `test_harness`, `test_harness_wiring`, `test_ledger_vault` +1 |
| `plugins_serve.py` (192) | `cubby_serve/plugins.py` | `board_api.py`, `serve_cli.py`, `serve_post.py` +1 | `test_plugins_check`, `test_plugins_serve` |
| `serve_api.py` (130) | `cubby_serve/api.py` | `serve_cli.py`, `v/board_timing.py` | `test_audience`, `test_board`, `test_plugins_serve` +1 |
| `serve_cli.py` (215) | `cubby_serve/cli.py` | `serve_api.py` | none |
| `serve_events.py` (57) | `cubby_serve/events.py` | `serve_api.py`, `serve_get.py`, `serve_post.py` | none |
| `serve_get.py` (168) | `cubby_serve/get.py` | `serve_api.py` | none |
| `serve_post.py` (86) | `cubby_serve/post.py` | `serve_api.py` | none |

### cubby_serve/board/ (2)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `board_api.py` (192) | `cubby_serve/board/api.py` | `serve_cli.py`, `serve_get.py` | `test_board`, `test_board_review`, `test_plugins_serve` |
| `board_data.py` (273) | `cubby_serve/board/data.py` | `board_api.py` | `test_board`, `test_board_review` |

### cubbyllm/ (1)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `serve.py` (811 **>300**) | `cubbyllm/brain.py` (CubbyBrain and its cortices) + cubby_serve/build.py (build_serve), after a split | `forge.py`, `scripts/question_style_probe.py`, `scripts/talk_probe.py` +4 | `test_pacman`, `test_plugins_check`, `test_serve` +1 |

### cubbyllm/affect/ (9)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `afferent.py` (362 **>300**) | `cubbyllm/affect/afferent.py` | `chat.py`, `coach.py` | `test_afferent` |
| `choosing.py` (232) | `cubbyllm/affect/choosing.py` | `chat.py`, `pac_affect.py`, `pac_nav.py` +2 | `test_choosing`, `test_saying` |
| `ear.py` (267) | `cubbyllm/affect/ear.py` | `data/build_ear.py`, `data/build_ear_bulk.py` | `test_ear` |
| `ear_lexicon.py` (96) | `cubbyllm/affect/ear_lexicon.py` | `data/build_ear.py`, `v/plugins_routing.py` | `test_ear` |
| `feeling.py` (181) | `cubbyllm/affect/feeling.py` | `pac_affect.py`, `pac_speech.py`, `pacman.py` | none |
| `lexicon.py` (375 **>300**) | `cubbyllm/affect/word_affect.py` | `coach.py` | `test_plasticity` |
| `neurochem.py` (816 **>300**) | `cubbyllm/affect/neurochem.py` | `chat.py`, `choosing.py`, `coach.py` +7 | `neurochemistry_probe`, `test_afferent`, `test_choosing` +9 |
| `perception.py` (159) | `cubbyllm/affect/perception.py` | `serve_cli.py` | `test_perception` |
| `saying.py` (265) | `cubbyllm/affect/saying.py` | `chat.py`, `pac_affect.py` | `test_said_back`, `test_saying` |

### cubbyllm/cortex/ (7)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `cortex/__init__.py` (24) | `cubbyllm/cortex/__init__.py` | none | none |
| `cortex/contract.py` (145) | `cubbyllm/cortex/contract.py` | none | none |
| `cortex/forward.py` (115) | `cubbyllm/cortex/forward.py` | none | `test_cortex` |
| `cortex/retina.py` (82) | `cubbyllm/cortex/retina.py` | none | `test_cortex` |
| `cortex/salience.py` (97) | `cubbyllm/cortex/salience.py` | none | `test_cortex` |
| `cortex/what.py` (116) | `cubbyllm/cortex/what.py` | none | `test_cortex` |
| `cortex/where.py` (240) | `cubbyllm/cortex/where.py` | none | `test_cortex` |

### cubbyllm/emit/ (3)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `emitter_450m.py` (230) | `cubbyllm/emit/seats_450m.py` | `emitter.py`, `emitter_talk.py` | none |
| `emitter_base.py` (213) | `cubbyllm/emit/base.py` (after a split: the contract, clean_reply, replay move; the llama-server emitter, chat templates, script ban stay) | `emitter.py`, `emitter_llama.py` | none |
| `emitter_talk.py` (208) | `cubbyllm/emit/talk.py` | `emitter.py` | none |

### cubbyllm/plugins/ (8)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `plugins.py` (103) | `cubbyllm/plugins/__main__.py` | none | none |
| `plugins_check.py` (70) | `cubbyllm/plugins/check.py` | `plugins_load.py`, `plugins_serve.py` | none |
| `plugins_guard.py` (288) | `cubbyllm/plugins/guard.py` | `plugins.py`, `plugins_check.py`, `plugins_load.py` +3 | `test_plugins`, `test_plugins_check`, `test_plugins_serve` |
| `plugins_index.py` (145) | `cubbyllm/plugins/index.py` | `plugins.py`, `plugins_serve.py`, `v/plugins_checks.py` +2 | `test_plugins`, `test_plugins_check`, `test_plugins_route` +1 |
| `plugins_load.py` (295) | `cubbyllm/plugins/load.py` | `plugins_serve.py`, `v/plugins_checks.py`, `v/plugins_talk_e2e.py` | `test_plugins`, `test_plugins_check` |
| `plugins_route.py` (148) | `cubbyllm/plugins/route.py` | `plugins_load.py`, `v/plugins_routing.py` | `test_plugins_route` |
| `plugins_world.py` (59) | `cubbyllm/plugins/world.py` | `plugins_load.py` | `test_persons_list` |
| `question_encoder.py` (99) | `cubbyllm/plugins/question_encoder.py` | `plugins_route.py`, `v/plugins_routing.py` | `test_plugins_route` |

### cubbyllm/reasoning/ (4)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `ask.py` (1694 **>300**) | `cubbyllm/reasoning/ask.py` | `exp_e6_host_view.py`, `data/build_chat_sft.py`, `data/build_explicit_sft.py` +15 | `test_ask`, `test_ask_branching`, `test_ask_relation` +11 |
| `context_graph.py` (319 **>300**) | `cubbyllm/reasoning/context_graph.py` | `v/exp_e9_context_followups.py` | `test_context_graph` |
| `loci_reader.py` (214) | `cubbyllm/reasoning/loci_reader.py` | `ask.py`, `v/bench_loci_learn.py` | `test_loci_reader` |
| `question_forms.py` (244) | `cubbyllm/reasoning/question_forms.py` | `ask.py`, `book_source.py`, `v/bench_loci_learn.py` +5 | `test_books`, `test_hdc_store`, `test_question_forms` +2 |

### cubbyllm/sources/ (3)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `robots_rules.py` (62) | `cubbyllm/sources/robots.py` | `news_archive.py`, `news_source.py`, `plugins_guard.py` | `test_news_archive`, `test_news_backfill` |
| `sources.py` (556 **>300**) | `cubbyllm/sources/wikidata.py` (the logic; its fetching -> cubby_serve/sources/, split along that line) | `ask.py`, `data/build_gen3.py`, `data/fetch_hdc_labels.py` +19 | `test_lfm_source`, `test_property_aliases`, `test_sources` |
| `web_source.py` (1025 **>300**) | `cubbyllm/sources/web.py` (the logic; its fetching -> cubby_serve/sources/, split along that line) | `serve_cli.py`, `v/exp_r33_web_heldout.py` | `test_web_source` |

### cubbyllm/sources/news/ (9)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `news_archive.py` (283) | `cubbyllm/sources/news/archive.py` (the logic; its fetching -> cubby_serve/sources/, split along that line) | `board_api.py`, `board_data.py`, `news_backfill.py` +4 | `test_board`, `test_board_review`, `test_news_archive` +2 |
| `news_backfill.py` (105) | `cubbyllm/sources/news/backfill.py` | `news_archive.py` | `test_news_backfill` |
| `news_events.py` (193) | `cubbyllm/sources/news/events.py` | `news_reply.py`, `news_trigger.py` | `test_news_events` |
| `news_gdelt.py` (171) | `cubbyllm/sources/news/gdelt.py` | `news_archive.py` | `test_news_gdelt` |
| `news_globalvoices.py` (108) | `cubbyllm/sources/news/globalvoices.py` | `news_archive.py`, `news_backfill.py` | `test_news_globalvoices` |
| `news_parse.py` (86) | `cubbyllm/sources/news/parse.py` | `news_archive.py` | none |
| `news_reply.py` (163) | `cubbyllm/sources/news/reply.py` | none | `test_news_reply` |
| `news_source.py` (389 **>300**) | `cubbyllm/sources/news/source.py` (the logic; its fetching -> cubby_serve/sources/, split along that line) | `ask.py`, `news_archive.py`, `news_parse.py` +1 | `test_news_source` |
| `news_trigger.py` (118) | `cubbyllm/sources/news/trigger.py` | `news_archive.py` | `test_news_trigger` |

### cubbyllm/speech/ (6)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `audience.py` (223) | `cubbyllm/speech/audience.py` | `ask.py`, `board_api.py`, `serve.py` +4 | `test_audience`, `test_news_trigger`, `test_plugins_check` |
| `chat.py` (413 **>300**) | `cubbyllm/speech/chat.py` | `serve.py`, `verse.py` | `test_chat`, `test_perception`, `test_saying` |
| `data/identity.py` (523 **>300**) | `cubbyllm/speech/identity.py` (after a split: the voice rules move, the SFT record builders stay) | `docs/archive/test_build_chat_sft_v6_2026-10-22.py`, `chat.py`, `data/build_emitter_sft.py` +11 | `test_build_chat_sft`, `test_build_emitter_sft`, `test_chat` +9 |
| `explicit_gate.py` (90) | `cubbyllm/speech/explicit_gate.py` | `data/build_chat_sft.py`, `data/build_explicit_sft.py`, `v/exp_e7_explicit_gate.py` | `test_explicit_gate` |
| `grounding.py` (293) | `cubbyllm/speech/grounding.py` | `news_reply.py`, `pac_affect.py`, `pac_speech.py` +1 | `test_news_reply`, `test_said_back` |
| `vault.py` (110) | `cubbyllm/speech/vault.py` | `context_graph.py`, `data/gap_families.py`, `harness_wiring.py` +2 | `test_harness`, `test_harness_wiring`, `test_ledger_vault` |

### cubbyllm/worlds/ (22)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `book_frames.py` (177) | `cubbyllm/worlds/book_frames.py` | `book_reader.py`, `book_source.py`, `data/build_reader_sft.py` +3 | `test_books` |
| `book_reader.py` (121) | `cubbyllm/worlds/book_reader.py` | `data/build_reader_sft.py`, `v/books_bulk_read.py` | `test_book_reader` |
| `book_source.py` (164) | `cubbyllm/worlds/book_source.py` | `ask.py`, `v/books_bulk_read.py`, `v/exp_books_bench.py` +1 | `test_books`, `test_time_rule` |
| `books.py` (289) | `cubbyllm/worlds/books.py` | `book_source.py`, `data/build_books_index.py`, `data/build_reader_sft.py` +8 | `test_audience`, `test_books`, `test_books_bulk` +2 |
| `capsules.py` (269) | `cubbyllm/worlds/capsules.py` | `data/import_science.py`, `v/exp_capsule_topk.py` | `test_capsules`, `test_possibility` |
| `data/hdc_world.py` (222) | `cubbyllm/worlds/hdc_world.py` | `hdc_store.py` | `test_hdc_store`, `test_time_rule`, `test_worlds` |
| `data/import_science.py` (395 **>300**) | `cubbyllm/worlds/science.py` | `serve_cli.py` | `test_import_science` |
| `data/wikikg.py` (346 **>300**) | `cubbyllm/worlds/wikikg.py` | `ask.py`, `book_reader.py`, `data/build_books_index.py` +33 | `test_wikikg`, `test_worlds` |
| `data/world_hygiene.py` (57) | `cubbyllm/worlds/hygiene.py` | `data/build_same_place.py`, `data/wikikg.py`, `hdc_store.py` | none |
| `encyclopedia.py` (338 **>300**) | `cubbyllm/worlds/encyclopedia.py` | `v/exp_r11_search_learn.py`, `v/exp_r16_encyclopedia.py` | `test_encyclopedia` |
| `hdc_store.py` (272) | `cubbyllm/worlds/hdc_store.py` | `news_gdelt.py`, `serve_cli.py`, `worlds_store.py` +2 | `test_hdc_store`, `test_time_rule`, `test_worlds` |
| `history_graph.py` (474 **>300**) | `cubbyllm/worlds/history_graph.py` | `context_graph.py`, `data/build_history_graph.py`, `data/build_history_sft.py` +4 | `test_context_graph`, `test_history_graph`, `test_history_lookup` |
| `history_lookup.py` (490 **>300**) | `cubbyllm/worlds/history_lookup.py` | `context_graph.py`, `v/build_e8_questions.py`, `v/exp_e8_history_retrieval.py` +1 | `test_context_graph`, `test_history_lookup` |
| `self_source.py` (144) | `cubbyllm/worlds/self_source.py` | `ask.py` | `test_self_source` |
| `time_rule.py` (55) | `cubbyllm/worlds/time_rule.py` | `ask.py`, `v/exp_worlds_bench.py` | `test_time_rule` |
| `wikitext.py` (198) | `cubbyllm/worlds/wikitext.py` | `web_source.py`, `v/exp_r11_search_learn.py` | `test_wikitext` |
| `worlds.py` (272) | `cubbyllm/worlds/base.py` | `capsules.py`, `data/import_science.py`, `data/wikikg.py` +14 | `test_asking`, `test_harness_wiring`, `test_neurochem` +3 |
| `worlds_adapters.py` (197) | `cubbyllm/worlds/adapters.py` | `plugins_load.py`, `worlds_fiction.py`, `worlds_ingest.py` | `test_worlds` |
| `worlds_fiction.py` (240) | `cubbyllm/worlds/fiction.py` | `plugins_load.py`, `plugins_world.py`, `worlds_ingest.py` +1 | `test_persons_list`, `test_worlds` |
| `worlds_ingest.py` (182) | `cubbyllm/worlds/ingest.py` | `plugins_load.py` | `test_worlds` |
| `worlds_manifest.py` (82) | `cubbyllm/worlds/manifest.py` | `worlds_ingest.py`, `v/exp_worlds_bench.py` | `test_worlds` |
| `worlds_store.py` (81) | `cubbyllm/worlds/store.py` | `v/exp_worlds_bench.py` | none |

### validation/builders/ (14)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `data/build_books_index.py` (100) | `validation/builders/build_books_index.py` | `worlds_adapters.py` | none |
| `data/build_ear.py` (140) | `validation/builders/build_ear.py` | none | none |
| `data/build_ear_bulk.py` (115) | `validation/builders/build_ear_bulk.py` | none | none |
| `data/build_history_graph.py` (569 **>300**) | `validation/builders/build_history_graph.py` | `data/build_nyt_events.py`, `v/build_e8_questions.py` | none |
| `data/build_history_records.py` (112) | `validation/builders/build_history_records.py` | none | none |
| `data/build_lexicon.py` (88) | `validation/builders/build_lexicon.py` | none | none |
| `data/build_nyt_events.py` (210) | `validation/builders/build_nyt_events.py` | none | none |
| `data/build_property_aliases.py` (171) | `validation/builders/build_property_aliases.py` | none | none |
| `data/build_same_place.py` (96) | `validation/builders/build_same_place.py` | none | none |
| `data/build_webq_bench.py` (111) | `validation/builders/build_webq_bench.py` | none | none |
| `data/build_word_table.py` (132) | `validation/builders/build_word_table.py` | none | none |
| `data/convert_books.py` (302 **>300**) | `validation/builders/convert_books.py` | none | none |
| `data/fetch_hdc_labels.py` (123) | `validation/builders/fetch_hdc_labels.py` | none | none |
| `data/goemotions_cube.py` (187) | `validation/builders/goemotions_cube.py` | `v/cube_vs_va.py`, `v/emotions_corpus.py`, `v/vad_check.py` | `test_goemotions_cube` |

### validation/sft/ (20)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `data/build_chat_sft.py` (272) | `validation/sft/build_chat_sft.py` | `docs/archive/test_build_chat_sft_v6_2026-10-22.py`, `data/build_explicit_sft.py`, `data/build_webqa_sft.py` +3 | `test_build_chat_sft`, `test_chat`, `test_goemotions_cube` |
| `data/build_emitter_sft.py` (477 **>300**) | `validation/sft/build_emitter_sft.py` | `data/build_game_sft.py`, `data/build_program_first.py`, `data/import_tinygsm.py` +10 | `test_build_emitter_sft` |
| `data/build_events_sft.py` (223) | `validation/sft/build_events_sft.py` | none | none |
| `data/build_explicit_sft.py` (143) | `validation/sft/build_explicit_sft.py` | none | none |
| `data/build_gen2_partition.py` (213) | `validation/sft/build_gen2_partition.py` | none | none |
| `data/build_gen3.py` (472 **>300**) | `validation/sft/build_gen3.py` | `data/build_profile_sft.py`, `data/gen3_llm_wordings.py`, `v/exp_r17_gen3_heldout.py` +5 | none |
| `data/build_ground_sft.py` (366 **>300**) | `validation/sft/build_ground_sft.py` | `exp_e6_host_view.py`, `data/build_events_sft.py`, `data/build_hdc_sft.py` +6 | none |
| `data/build_hdc_sft.py` (364 **>300**) | `validation/sft/build_hdc_sft.py` | `data/hdc_world.py` | `test_hdc_sft` |
| `data/build_history_sft.py` (347 **>300**) | `validation/sft/build_history_sft.py` | `v/build_e8_questions.py` | none |
| `data/build_passage_sft.py` (160) | `validation/sft/build_passage_sft.py` | `data/build_quotes_sft.py`, `data/build_webqa_sft.py` | none |
| `data/build_program_first.py` (1005 **>300**) | `validation/sft/build_program_first.py` | `data/import_science.py` | `test_build_program_first` |
| `data/build_quotes_sft.py` (104) | `validation/sft/build_quotes_sft.py` | none | none |
| `data/build_reader_sft.py` (246) | `validation/sft/build_reader_sft.py` | none | `test_book_reader` |
| `data/build_talk_mix.py` (159) | `validation/sft/build_talk_mix.py` | none | none |
| `data/build_v13f.py` (100) | `validation/sft/build_v13f.py` | none | none |
| `data/build_webqa_sft.py` (138) | `validation/sft/build_webqa_sft.py` | none | none |
| `data/chat_sft_v6.py` (106) | `validation/sft/chat_sft_v6.py` | `data/build_chat_sft.py` | none |
| `data/gen3_llm_wordings.py` (228) | `validation/sft/gen3_llm_wordings.py` | none | none |
| `data/hf_rows.py` (62) | `validation/sft/hf_rows.py` | `data/gap_families.py` | `test_hf_rows` |
| `data/import_tinygsm.py` (363 **>300**) | `validation/sft/import_tinygsm.py` | none | `test_import_tinygsm` |

### stays in standin/ (29)

| module (lines) | new path | imported by | tests |
|---|---|---|---|
| `coach.py` (384) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | `test_coach`, `test_plasticity` |
| `data/build_game_sft.py` (419) | stays: builds data for the GGUF adapters only | none | `test_build_game_sft` |
| `data/build_profile_sft.py` (107) | stays: builds data for the GGUF adapters only | none | none |
| `data/gap_families.py` (1228) | stays: builds data for the GGUF adapters only | `docs/archive/test_build_chat_sft_v6_2026-10-22.py`, `eval_emitter_vm.py`, `scripts/tool_call_probe.py` | `test_build_chat_sft` |
| `emitter.py` (39) | stays: the facade its 30 importers use; re-exports cubbyllm.emit and the stand-in's llama emitter | `ask.py`, `chat.py`, `data/build_program_first.py` +27 | `test_build_emitter_sft`, `test_emitter_450m`, `test_exp_he19_memory` +2 |
| `emitter_llama.py` (125) | stays: the GGUF stand-in | `emitter.py` | none |
| `eval_emitter_vm.py` (255) | stays: the GGUF stand-in (its evaluation) | `ask.py`, `v/exp_r10_simpleqa.py`, `v/exp_r11_search_learn.py` +12 | `test_build_emitter_sft` |
| `forge.py` (182) | stays: the cubby-man game (Nick decides its home later) | `grounding.py`, `pac_forge.py`, `pacman.py` +1 | `test_pacman` |
| `hypothesis.py` (284) | stays: the cubby-man game (Nick decides its home later) | `pac_forge.py`, `pac_sense.py`, `verse.py` | `test_asking`, `test_hypothesis`, `test_portable` |
| `knowledge.py` (161) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | `test_asking` |
| `lfm_source.py` (286) | stays: the GGUF stand-in (the LFM as a source) | `ask.py`, `book_reader.py`, `serve_cli.py` +2 | `test_lfm_source`, `test_web_source` |
| `openrouter.py` (177) | stays: the GGUF stand-in (the frontier proposer, datasets only) | `data/build_ear_bulk.py`, `data/build_ground_sft.py`, `data/build_history_graph.py` +10 | none |
| `pac_affect.py` (417) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | none |
| `pac_forge.py` (343) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | none |
| `pac_nav.py` (386) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | none |
| `pac_sense.py` (310) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | none |
| `pac_speech.py` (453) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | none |
| `pacman.py` (771) | stays: the cubby-man game (Nick decides its home later) | `data/build_game_sft.py`, `data/gap_families.py`, `data/goemotions_cube.py` +8 | `test_asking`, `test_coach_wired`, `test_ledger_vault` +5 |
| `pacpaths.py` (43) | stays: the cubby-man game (Nick decides its home later) | `feeling.py`, `pacman.py`, `pacui_threejs.py` | none |
| `pacui_threejs.py` (430) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | none |
| `pacworld.py` (596) | stays: the cubby-man game (Nick decides its home later) | `pac_affect.py`, `pac_forge.py`, `pac_nav.py` +4 | none |
| `programs.py` (211) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | none |
| `scripts/forge_probe.py` (78) | stays: the GGUF stand-in (a probe) | none | none |
| `scripts/probe_v3.py` (68) | stays: the GGUF stand-in (a probe) | none | none |
| `scripts/question_style_probe.py` (134) | stays: the GGUF stand-in (a probe) | none | none |
| `scripts/talk_probe.py` (130) | stays: the GGUF stand-in (a probe) | none | none |
| `scripts/tool_call_probe.py` (103) | stays: the GGUF stand-in (a probe) | none | none |
| `verse.py` (663) | stays: the cubby-man game (Nick decides its home later) | `pacman.py` | `test_asking`, `test_portable`, `test_serve` +1 |
| `world.py` (226) | stays: the cubby-man game (Nick decides its home later) | `coach.py`, `pac_forge.py`, `pac_nav.py` +4 | `test_plasticity` |

### Where the tests go

| new home | test files |
|---|---|
| `standin/tests/ (stays)` | test_asking (mixed), test_build_game_sft, test_coach (mixed), test_coach_wired (mixed), test_exp_he19_memory, test_hypothesis, test_lfm_source (mixed), test_pacman (mixed), test_pacman_percepts (mixed), test_plasticity (mixed), test_portable, test_said_back (mixed), test_verse (mixed), test_wariness (mixed) |
| `tests/affect/` | neurochemistry_probe, test_afferent, test_choosing, test_ear, test_neurochem, test_perception, test_saying |
| `tests/brain/` | test_serve (mixed) |
| `tests/builders/` | test_build_chat_sft (mixed), test_build_emitter_sft (mixed), test_build_program_first, test_goemotions_cube (mixed), test_hdc_sft, test_hf_rows, test_import_tinygsm, test_property_aliases |
| `tests/cortex/` | test_cortex |
| `tests/emit/` | test_emitter_450m |
| `tests/plugins/` | test_plugins (mixed), test_plugins_check (mixed), test_plugins_route |
| `tests/reasoning/` | test_ask, test_ask_branching, test_ask_relation, test_claim_check, test_context_graph, test_loci_reader, test_question_forms, test_sleep_loop |
| `tests/serve/` | test_board (mixed), test_board_review (mixed), test_harness, test_harness_wiring (mixed), test_ledger_vault (mixed), test_plugins_serve (mixed) |
| `tests/sources/` | test_news_archive, test_news_backfill, test_news_events, test_news_gdelt, test_news_globalvoices, test_news_reply, test_news_source, test_news_trigger, test_sources, test_web_source (mixed) |
| `tests/speech/` | test_audience (mixed), test_chat (mixed), test_explicit_gate, test_identity |
| `tests/worlds/` | test_book_reader (mixed), test_books, test_books_bulk, test_capsules, test_encyclopedia, test_hdc_store, test_history_graph, test_history_lookup, test_import_science, test_persons_list, test_self_source, test_time_rule, test_wikikg, test_wikitext, test_worlds |
| `tests/validation/` | test_emitter_decode, test_exp_he18_ab, test_exp_he19_gate_a (they test `validation/` scripts, not `standin/` modules) |
