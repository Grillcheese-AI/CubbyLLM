# 0033 — Follow-ups from the 0030 and 0031 reviews: a merged namesake spoke a wrong place, and six small fixes
status: review
written: 2026-10-22 by Cowork, from the reviews of 0030 and 0031 · approved: — (fixes from a review; Nick told) · hypothesis: none

## Why
0031's baselines found "Where is the Globe Theatre?" answered "located in: London Borough of Southwark, Boston": two
items sharing a label were merged, and the wrong one's place was spoken. The project's first rule is that a wrong
answer is never spoken; a refusal or a question back is always allowed. The rest are small things the reviews found.
In plain words: when two things share a name, Cubby asks which one you mean instead of mixing them up.

## Build
1. **Namesakes never merge (first).** Find where the Globe Theatre's two items meet (the hdc world's label lookup, the
   wiki world, or the ask loop's entity resolution) and fix it there: a label that names several items either is
   decided by the question's own context (another name in it, a type) or answers `ambiguous_entity` with the items
   named, as "what is Ottawa?" already does. Probe: 50 labels the hdc world shares between two or more items, asked
   "where is X?" / "what is X?": 0 answers mixing two items.
2. **The flaky Brave timing test** (5 in 20 before 0031, 3 in 20 after): make it deterministic (a fake clock), or
   mark it with its reason.
3. **Two local-path leftovers** (0030's review): the `.stdout` log and the doubly escaped JSON; the guard learns both
   forms (`.stdout` files; `\\\\`-escaped drive paths) and fails on a planted one.
4. **The data root** (0031's answer 6): every data place the package names under `standin/data/out` resolves under
   `$CUBBY_DATA_ROOT` (the default is today's place); the caches move, nothing is fetched again.
5. **The loci reasons' routes** (0031's answer 1) and **the fetch's fixed name** (answer 2).
6. **mowm's `FastWordEncoder` ported** into `cubbyllm/` (answer 3), pinned against mowm's output.
7. **The shim counter** (answer 7): a test that reports in-repo imports of shimmed `standin/` names.

## Acceptance (pre-registered)
- The namesake probe: 0 mixed answers in 50; "Where is the Globe Theatre?" says one place or asks which; 0014's bench
  unchanged (177 of 177, 0 wrong).
- The Brave test passes 20 of 20 runs; the path guard fails on both planted forms and finds 0 in `git ls-files`.
- The real stack's 20 fixed answers unchanged, except the Globe Theatre's if it is among them (report it).

## Constraints
- Rules in `CLAUDE.md` hold. Nothing new in `standin/`.

## Report
- **Commits:** one per item.
  - 1 `85039fc`: namesakes;
  - 2 `574114d`: Brave;
  - 3 `2668c40`: the path guard;
  - 4 `d197454`: the data root;
  - 5 `3a82735`: the loci routes and the fetch's name;
  - 6 `178b571`: the encoder's port;
  - 7 `2f57ce1`: the shim counter;
  - and this report.
- **Files:**
  - 1:
    - `cubbyllm/worlds/hdc_namesakes.py` (new: a mixin of `HdcWorld` and `Stores`);
    - `cubbyllm/worlds/hdc_store.py` (272 → 288 lines);
    - `cubbyllm/reasoning/ask_namesakes.py` (new), `ask.py` (278), `ask_profile.py` (271);
    - `validation/namesake_probe.py`; `tests/worlds/test_hdc_namesakes.py`.
  - 2: `cubbyllm/sources/web_search.py`, `tests/sources/test_web_source.py`.
  - 3:
    - `tools/scrub_local_paths.py`, `tests/test_local_paths.py`;
    - the three files the guard then found, scrubbed: the two named in the order, and
      `validation/logs/exp_he19_cvl_train_v1.out`, read once `.out` was.
  - 4:
    - `cubbyllm/local_paths.py` (`data_dir`), and the 16 places that now read it, in 15 modules;
    - `cubby_serve/api.py` + `cli.py` (`--data-root`);
    - `tools/move_data_root.py`; `tests/test_data_root.py`.
  - 5:
    - `cubbyllm/reasoning/sleep.py` (580 → 583: the routing table is one table) and `tests/reasoning/test_sleep.py`;
    - `cubby_serve/sources/news.py`, `http.py` and `tests/test_offline_seams.py`.
  - 6:
    - `cubbyllm/worlds/fast_words.py`, `tests/worlds/test_fast_words.py`, `validation/pin_fast_words.py`;
    - `cubby_serve/build.py` and validation's `_load_semantic_words` now use the port.
  - 7: `tools/shim_count.py`, `tests/test_shim_count.py`.
  - Nothing new in `standin/`.
- **Tests:** `python -m pytest tests standin/tests`: **1,221 passed, 1 failed, 6 skipped**
  ([log](../../validation/logs/0033_tests_2026-10-22.json)).
  - Against 0031's baseline file for file, every test that passed passes. 0031 left 1,201; 0033 adds 20:
    namesakes 5, the port 4, the data root 4, the shim counter 2, the local-path guard 3, the loci
    routes 1, the fetch's name 1.
  - The one red is the known one (`test_guards.py`'s wiring, order 0002).
- **Measured:**
  - **The namesake probe**
    ([before](../../validation/logs/namesake_probe_2026-10-22_before.json),
    [after](../../validation/logs/namesake_probe_2026-10-22_after.json)).
    - The 50 labels are drawn by seed from those the hdc world shares between two or more items with a
      where/what fact. Each is asked "Where is X?" and "What is X?" through the ask loop (wiki + hdc, offline,
      no model).
    - **74 of 100 answers mixed items before; 0 after.**
    - Now 94 ask which item, with each item named and described by its own values. 6 are names with " of " that
      the profile reader leaves to the emitter (no_plan), as before.
  - **"Where is the Globe Theatre?"** asks which of 3:
    - `country: United Kingdom` (Q5571009);
    - `located in: London Borough of Southwark` (Q272434);
    - `located in: Boston` (Q5571005).

    Choosing Q5571005, or asking "…in Boston?", answers `located in: Boston`. "…in London?" still asks: no
    item's value *is* London, and a word inside a name is not the name.
  - **0014's bench:** **179 of 179 spoken correct, 0 wrong**, against 177 of 177 before
    ([log](../../validation/logs/exp_hdc_bench_2026-10-22_0033.json)).
    - The 177 are unchanged, row for row.
    - The 2 gained, "William Owen (architect)" and "Nautical/April/21/Selected article", are aliases of one
      item. Before, the hop read every namesake of that item's label and came back multi-valued.
    - Not "unchanged" to the letter, so I say it here. The time-sensitive part draws new subjects each run
      (`ORDER BY RANDOM()`); it said 0 flat.
  - **The real stack** (the 450M seats on grilly2, wiki + hdc, the ask loop offline, plugins, the board;
    `python -m cubby_serve`): **19 of the 20 fixed answers identical**
    ([log](../../validation/logs/0033_stack_2026-10-22.json)).
    - The Globe Theatre changed: from the merged answer to `ambiguous_entity` with its 3 items.
    - The board differs by the order queue; by the journey and answer counts of that one question; and in
      `/health`, by the talk adapter's one paraphrase call, which the merged answer had used.
  - **Relation hops on real shared names** (in the probe's log). Read:
    - France's capital (Paris);
    - Japan's currency (Japanese yen);
    - Canada's capital (Ottawa) and Italy's (Rome);
    - Terry White's kind (human).

    Withheld, so the asker is asked:
    - Georgia's capital: two items hold one;
    - the Globe Theatre's country: its one holder is of no known kind;
    - Wantage's location: a civil parish has a location, this one's is just not kept;
    - Brazil's official language and the Eiffel Tower's country: conservative.

    Toronto's population: no item holds one. That keeps its own refusal (a source gap), not "which one?".
  - **Brave:** 20 of 20, each run a fresh pytest process
    ([log](../../validation/logs/0033_brave_20runs_2026-10-22.json)).
  - **The path guard:** it fails on both planted forms (escaped twice; a `.stdout` log under a user folder) and
    finds 0 in `git ls-files` (`tests/test_local_paths.py`, 16 passed).
  - **The encoder's port:** **2,020 of 2,020 codes equal** to mowm's on the production table
    (`fastword_table_v4.npz`, 60,151 words); max abs diff 0.0, same meta, same idf
    ([log](../../validation/logs/pin_fast_words_2026-10-22.json)).
  - **The shim counter:** 338 imports of shimmed names in code and 1 in a notebook, over 116 shims.
    - The most used: `ask` 37, `sources` 29, `wikikg` 28, `neurochem` 23, `worlds` 16.
    - Mostly in tests/ and validation/.
    - The test fails only if the count grows (`CEILING`).
- **Not verified here:**
  - The online Wikidata source with a shared name: the stack runs offline. The hdc clarify's choices carry real
    Q-ids, so a choice should reach the online source too; not run.
  - A clarify choice clicked in the board's panel: the loop takes the item (`item=`, tested); the panel's button
    sends it, not clicked here.
  - The server booted with `--data-root` set to another folder: the test reads all 16 places in a fresh process
    with `$CUBBY_DATA` set, but the server was not booted that way.
  - The Colab notebooks were not touched (the counter's one notebook import is still there).
  - Nothing in this order needs a GPU check.
- **Found on the way:**
  1. **The strict rule cost two right answers.** I first had only the asker and a qualifier decide. That refused
     "What is the capital of France?" and "What is the currency of Japan?", two of the stack's three spoken
     answers: Wikidata5M shares almost every famous name (3 Frances, 8 Japans, 34 Parises). It also overwrote 11
     other refusals' reasons with `ambiguous_entity`
     ([log](../../validation/logs/0033_stack_strict_rule_2026-10-22.json), 6 of 20 same).
     - Hence the hop's relation tier: the online source's rule, under three guards (`hdc_namesakes.py`).
     - Hence too: a refusal is overwritten only when the world withheld facts the hop would have read.
     - The third guard (applicability) came from the probe: without it "Where is Wantage?" went to the New
       Jersey township. Profile asks name no relation, so for them only the asker or a qualifier decides.
  2. **The data root's name:** the answer named `$CUBBY_DATA_ROOT`, but since 0030's scrub that name means "any
     other drive root".
     - 23 committed files use it that way, and code defaults resolve through it (`fact_pool.py`'s parquet,
       validation's); this machine's `paths.local.json` sets it to a drive root.
     - I used **`$CUBBY_DATA`** (question 1).
  3. **The applicability scan costs time.** It reads up to 2,000 members of a kind, once per (kind, relation),
     cached: 396 ms for Japan's currency, under 20 ms for the others.
     - A relation that only a handful of a large kind carry could be missed, letting the hop decide.
  4. **Inverse questions are not guarded.** Objects (`_by_obj`, "what has X as its capital?") still read every
     namesake of a shared label. Not this order's (question 3).
  5. **A chosen item's facts are stored under the bare label.** With an online source, the facts fetched for a
     clarify choice go into the wiki world under the bare label, so a later bare ask reads that item as if the
     name were one thing. "What is Ottawa?" already does this; not changed.
  6. **My slip:** re-running 0031's digest script rewrote that order's committed stack log. It was restored from
     git before any commit.
- **Questions:**
  1. Keep `$CUBBY_DATA`, or rename the scrub's generic drive root (23 files) so that `$CUBBY_DATA_ROOT` can mean
     the data root as the answer said?
  2. The hdc relation tier: keep the three guards (France and Japan answered; Brazil's language, the Eiffel
     Tower's country withheld), or go strict (the asker or a qualifier only), and refuse "capital of France"
     until the asker picks?
  3. Two follow-ups, if wanted:
     - a kind → relations table built with the hdc world, in place of the scan (complete and instant);
     - the same namesake rule for objects (inverse questions).
  4. 0034 is open: next, or 0029 step 2 first? Cowork's note says they may run side by side; I would take
     0034's fixed questions first.
