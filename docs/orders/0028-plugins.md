# 0028 — Plugins: one folder format and a loader for adapters, worlds, sources and tools, and board panels, loaded on demand like skills
status: in progress (review 2026-10-22: two fixes, the ask loop's gate, the held-out routing set)
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick (asked for it; chose all four kinds, on-demand loading, code with permissions) · hypothesis: none · **after 0027 P1**

## Why
Nick (2026-10-22): "we should have something to load plugins / adapters (similar to skills)". The pieces exist, but each
is wired by hand at startup:
- `worlds.CubbyPlugin` (MindForge style: `worlds()`, optional `cortices()`, `on_turn()`, `bind(brain)`) mounted by a
  flag (`--pacman`; `verse` is the reference); `worlds.Knows` (`covers(question)`, `answer(question)`) with
  `Worlds.route` and its `TAU = 0.35`: a world that claims a question only when it covers it;
- the LoRA roles on one 450M base (`emitter.Cubby450mTrunk`: one adapter per role, swapped in place, one layout);
- the ingest routes of 0015 (`worlds_adapters`: passages, events, quotes, table, mount, classify);
- `learn.Source` (`facts(entity)`), the feed manifest and the polite fetcher of 0024 (robots, `Crawl-delay`, the
  `CubbyLLM/0.1` name).

What it brings: a new ability is a folder you drop in. Cubby keeps one line about each plugin in mind and loads the
plugin only when a question needs it, the way skills work; every answer still names, in the audit view, the plugin that
supplied it. A conventional model gains a skill by retraining or by stuffing its prompt; Cubby mounts it, uses it,
and can drop it again, and nothing it adds is mistaken for what the base knows.
In plain words: add a skill, a dataset or a data source to Cubby by dropping in a folder; it only wakes up when needed.

## Build
- **The format.** A plugin is a folder with `PLUGIN.md`: YAML front matter, then plain documentation.
  ```yaml
  name: envcanada-alerts            # unique; becomes plugin:<name>
  version: 0.1.0
  description: Weather warnings for Canadian places from Environment Canada, live.   # what the router reads
  examples: ["is there a storm warning in Lévis?", "any weather alerts in Québec today?"]   # must activate it
  not: ["what is the capital of Canada?"]                                                # must not
  kinds: [source, panel]            # any of: adapter, world, source, tool, panel
  audience: public                  # or internal (0016): an internal plugin never shows in a public response
  licence: OGL-Canada
  requires: {plugins: [], base: null}          # base: the 450M export's hash, for an adapter
  permissions: {network: [dd.weather.gc.ca], write: true, gpu: false}
  entry: {source: source.py:Alerts, panel: panel.json}
  ```
- **The kinds:**
  - `adapter`: `<role>_lora.safetensors` + `<role>_lora.json`, registered as a new role on the 450M base when loaded;
    refused, with the reason, when its layout (rank, alpha, targets) or its `base` hash differs from the base's.
  - `world`: a data pack plus its 0015 manifest row (route, world, tier); ingested once into the plugin's own store
    (under the run's data root, `plugins/<name>/`), mounted as `plugin:<name>` through `CubbyBrain.mount`. The time
    rule, the same-name guard and copy-or-refuse apply as to any world.
  - `source` and `tool`: Python classes. A source implements `learn.Source` or `worlds.Knows`; a tool implements
    `run(args) -> dict` with a JSON schema for `args`. Plugins never produce replies: what they return goes through
    the same gates as any fact (and a `Knows` world answers in knowledge, never in state).
  - `panel`: `panel.json`, a card or panel for the 0027 board (title, the plugin's data call, a chart type the board
    already draws). No plugin JavaScript runs in the board.
- **Discovery** (`standin/plugins_index.py`): at startup, behind `--plugins DIR[,DIR]` (default off), read every
  `PLUGIN.md`'s front matter and nothing else. That index is the only thing kept in memory.
- **On demand** (`standin/plugins_load.py`): each plugin's `description` and `examples` are encoded the way worlds are
  routed; a question that scores at or above `tau_plugin` (start at `Worlds.TAU`, 0.35; tune it on the examples) loads
  the plugin (import, mount, register the role), once. A loaded adapter that is idle for `--plugin-idle` minutes
  (default 30) is unloaded; worlds and sources stay until restart. The board can pin a plugin (always loaded) or
  disable it.
- **Code and its permissions** (`standin/plugins_guard.py`):
  - Code runs only after Nick approves it: `python -m standin.plugins approve <name>` (or the board's Approve button,
    audit audience only) records each file's SHA-256 and the permissions approved in a local, git-ignored lock file.
    A file that changes afterwards blocks the plugin, with the reason, until approved again.
  - A plugin's code gets a context object: `ctx.fetch` (the 0024 fetcher: the declared hosts only, robots and
    `Crawl-delay` honoured, always `CubbyLLM/0.1`), `ctx.data_dir` (its only writable folder), `ctx.log`. `check`
    refuses code that imports network, process or shell modules itself (`socket`, `urllib`, `http`, `requests`,
    `subprocess`, `os.system`, `ctypes`). This is review plus a pin, not a sandbox: the docstring says so.
- **CLI** (`python -m standin.plugins`): `list`, `check <name>`, `approve <name>`, `new <name> --kinds ...` (a scaffold).
- **Board** (0027): a Plugins view in System: name, kinds, status (idle, loaded, pinned, disabled, blocked and why),
  last used, permissions, audience; Approve for a new or changed code plugin.
- **Four plugins that prove the format** (in `plugins/` in the repo, public):
  1. `envcanada-alerts` (source + panel): the warnings 0025 already reads, as a plugin;
  2. `gutenberg-plays` (world): the public-domain plays of 0020 as a fiction world pack;
  3. `talk` (adapter): today's talk adapter repackaged, so the role loads from a plugin instead of a flag;
  4. `units` (tool, no network): unit conversion, the smallest code plugin.

## Acceptance (pre-registered)
- Startup with 50 dummy plugins reads only their front matter: under 200 ms added, no payload loaded (report it).
- Routing: for each of the four plugins, at least 9 of 10 `examples` load it and at most 1 of 10 `not` lines does
  (the lines are written before tuning `tau_plugin`; report the confusion table).
- Adapters: a plugin adapter with another layout or base hash is refused with its reason; loading and unloading the
  `talk` plugin leaves the emitter role's outputs bitwise identical on a fixed prompt.
- Code: a plugin file changed after approval is blocked; `ctx.fetch` to an undeclared host is refused; a plugin
  importing `urllib` or `subprocess` fails `check`. Each is a test.
- Audience: an `internal` plugin's name and outputs are absent from every public response (the 0016 test, extended).
- Worlds: `plugin:gutenberg-plays` answers under copy-or-refuse like any world; unloading it leaves the others unchanged.
- With `--plugins` off, nothing in the server's answers changes.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Plugins sit on top, never inside (`CubbyBrain.mount`); they never write a reply.
- The lock file and any internal plugin's folder stay out of git; plugin roots that are not the repo's `plugins/` are
  named in the local notes, not in committed files.

## Out of scope
- A shared catalogue or installing from the network (later; local folders only now).
- Plugin JavaScript in the board; a real process sandbox (a later order, if code plugins from others ever arrive).
- Agents (they will use tools through this loader, in their own order).

## Report
- **Commits:** bea2b5f (in progress), 0327cba (the format, index, router, guard, loader, CLI, the trunk's plugin roles,
  the four plugins, the routing test), a922d20 (the server, the board's Plugins view, the checks, two fixes).
- **Files:** `standin/plugins_index.py` (front matter only), `plugins_route.py` (the router), `plugins_guard.py` (the pin,
  the review, `Ctx`), `plugins_load.py` (the loader, one handler per kind), `plugins_serve.py` (the server and the board),
  `plugins.py` (the CLI), `standin/emitter.py` (`Cubby450mTrunk.layout / base_hash / add_role / remove_role`),
  `serve_api.py` (`--plugins`, `--plugin-idle`, the wake on `/ask` and `/turn`, the board's POST actions), `audience.py`
  (`plugins`, `plugin_facts` internal), `board_api.py`, `dashboard/board/plugins.js`; `plugins/units`,
  `plugins/envcanada-alerts`, `plugins/gutenberg-plays`, `plugins/talk`; tests `standin/tests/test_plugins.py` (7),
  `test_plugins_serve.py` (3); `validation/plugins_routing.py`, `validation/plugins_checks.py`.
- **Tests:** the 10 above pass, with `test_board.py`, `test_audience.py`, `test_ask.py` (48 together). `pytest tests`:
  only the known `test_every_module_declares_wiring` (order 0002) red.
- **Measured:**
  - **Startup** (`validation/logs/plugins_checks_2026-10-22.json`): 50 dummy plugins, each with a 200 KB payload and
    code that would stop the process if imported: `discover` adds 69.8 ms median (108.2 ms worst of 5; 44.6 ms in an
    earlier run on an idle CPU -- this one shared it with a 450M job); nothing imported, no payload opened. Under
    200 ms.
  - **Routing** (`validation/logs/plugins_routing_enc_2026-10-22.json`, with the serve stack's fastword encoder;
    `plugins_routing_2026-10-22.json` with word overlap alone). The ten examples and ten not-lines of each plugin were
    written before any tuning. At the start value 0.35 the not-lines leak (alerts 3/10 with the encoder: population,
    distance, "is Lévis a city"; plays 4/10 with word overlap); tuned to **0.4** with the encoder (0.475 without):

    | plugin | examples wake it | not-lines wake it | examples, leave-one-out |
    |---|---|---|---|
    | envcanada-alerts | 10/10 | 0/10 | 10/10 |
    | gutenberg-plays | 10/10 | 0/10 | 2/10 |
    | talk | 10/10 | 0/10 | 4/10 |
    | units | 10/10 | 0/10 | 2/10 |

    No example of one plugin wakes another. The order's line (in-sample) is met; **leave-one-out** (an example scored
    without its own line in the index) shows what it means: a new phrasing of a need the examples cover often does not
    wake the plugin -- ten lines per plugin are a thin description. More examples per plugin, or the description
    scored by an encoder trained for questions, would lift it.
  - **Adapters:** a plugin adapter whose LoRA layout (rank, alpha, targets) or `requires.base` hash differs from the
    base's is refused with the reason, and the base and every other role keep their weights; loading a plugin role,
    using it and unloading it leaves the emitter role's wrapped weights bit for bit as before (`test_plugins.py`, on
    the trunk's own `use` / `add_role` / `remove_role` with its loaded state faked).
  - **Code:** a file changed after approval blocks the plugin, naming the file; `urllib`, `subprocess` and
    `os.system` fail `check`; `ctx.fetch` to an undeclared host is refused before any request (tests). A CRLF checkout
    of an approved plugin does not block it (text files are hashed with LF endings; binaries byte for byte).
  - **Audience:** an internal plugin's name and facts are in no public board response (`/board/plugins`, the router's
    scores, its PLUGIN.md, Home, System) and not in the public `/ask` record; the audit audience gets them (test).
  - **Worlds:** a world plugin answers a question naming its work with the work's facts, refuses a relation the work
    does not state, covers nothing when no work is named, and unloading it leaves the other plugins as they were
    (test, synthetic play). On the real pack (the 78 Gutenberg plays `CUBBY_PLAYS` names), 0015's fiction reader finds
    **1** in-fiction fact: it reads prose like "Juliet's mother, Lady Capulet", which these First Folio texts almost
    never contain, and only 4 of the 78 state a relation in a persons list. So `plugin:gutenberg-plays` refuses its ten
    examples (0 answered, 0 wrong).
  - **The proof plugins on this machine:** `units` 10/10 conversions exact; `envcanada-alerts` live (the Datamart's CAP
    alerts, `dd.weather.gc.ca` only): 53 alerts in effect, read in 4.7 s, a place with one answered ("A yellow wind
    warning is in effect for Avalon Peninsula Southeast until 2026-10-23 05:25 UTC."), an invented place refused.
  - **Off:** with `--plugins` off, `/health`, `/state`, `/worlds`, `/board/home`, `/board/system` and an unknown path
    answer exactly as with it on and no question asked (test).
  - **The board** (in Chrome, a stand-in brain with the four plugins): the Plugins view lists them with status,
    permissions and audience; "which plugin would wake?" scores a question against the line; Pin loads
    `envcanada-alerts` live and its panel lists the alerts in effect; Approve / Pin / Disable answer the audit
    audience only (403 otherwise).
- **Not verified here:**
  - The `talk` plugin on the real 450M: adding the talk adapter as a plugin role and the emitter's output on a fixed
    prompt before and after need the base on grilly2 (the local GPU) beside a running 450M CPU job; the mechanism is
    pinned by the test above, the end-to-end check is for a run with the base loaded.
  - The idle unload (`--plugin-idle`) in a long-running server (pinned by `Loader.sweep`'s logic, not watched for 30
    minutes).
- **Found on the way:**
  - The alerts plugin, run live, said an ended advisory was "in effect" -- an update whose headline says "ended" is now
    dropped; that would have been a wrong answer.
  - Plugin sources and worlds answer through the loader's own router (`Loader.ask`), and the facts reach the record's
    audit fields; they are not yet fed through the ask loop's gate into its walk. That wiring (a plugin world as a
    source of the ask loop, with its provenance `plugin:<name>`) is the natural next step; the order asks that
    plugins never write replies, which holds.
  - `serve_api.py` grew by 25 lines (the dispatch), `emitter.py` by 55 (the trunk owns its weights): both were already
    over 300.
- **Questions:**
  - A better pack for `gutenberg-plays`: modern-spelling Gutenberg editions with persons lists (Romeo and Juliet
    `1513`, which order 0020 already uses), and a reader for a persons list ("JULIET, daughter to Capulet")?
  - Should a plugin's facts go through the ask loop's gate now (a plugin world as one of the loop's sources), or wait
    for its own order?

## Review (Cowork, 2026-10-22): accepted in part; two fixes and two additions before done
Checked: the tests pass (with 0027's, 20 together); the lock file sits under `standin/data/out/` (git-ignored); the
plugin folders carry no weights or texts (`CUBBY_TALK_ADAPTER`, `CUBBY_PLAYS` name them off-repo); no local paths.
The format, the index, the loader, the pin and the board's view are accepted. The ended-advisory catch in
`envcanada-alerts` is a good find: it would have been a wrong answer.
**Fixes:**
1. **The code check misses most simple escapes.** Probe (12 cases, `validation/logs/_review_0027_28.py`, not
   committed): caught `from os import system` and `importlib.import_module`; missed `eval`, `exec`,
   `getattr(os, "sys" + "tem")`, `builtins.__import__`, `os.posix_spawn`, `os.spawnlp`, `os.execlpe`,
   `sys.modules["socket"]`, `open(..., "w")` outside the plugin's folder and `shutil.rmtree`. The order says review
   plus a pin, not a sandbox, and that stands; but the scan should catch these: `eval`, `exec`, `compile`, `getattr` /
   `setattr` on a module with a non-literal name, `builtins`, `sys.modules`, every `os.exec*`, `os.spawn*` and
   `os.posix_spawn*` by prefix, `shutil`'s deleting and moving calls, and `open` / `Path.write_*` with a write mode
   (code writes through `ctx.data_dir`). A test per case.
2. **Approve shows the code.** Approval is the real control, so the board's Approve shows the plugin's files (or the
   diff since the last approval) before the button works, and the CLI's `approve` prints the same.
**Additions:**
3. **Plugin facts through the ask loop's gate, in this order** (Cowork's answer, 2026-10-22): a plugin world or source
   is one of the loop's sources, tagged `plugin:<name>` in the audit chain, under copy-or-refuse; nothing it returns
   reaches long-term memory except through the learn gate. Test: a plugin fact that contradicts a held fact is
   refused and ledgered, not stored.
4. **Routing beyond the examples.** The order's line is met in-sample at `tau_plugin` 0.40 (10/10 examples, 0/10
   not-lines), but leave-one-out is 2/10 (plays, units) and 4/10 (talk): a new phrasing often does not wake a
   plugin. Cowork wrote a held-out set (10 new phrasings and 10 near-misses per plugin, some in French), handed over
   off-repo; it is not to be tuned on. Add examples per plugin (aim for 30, varied, both languages) and try scoring
   the description with a question encoder; report on the held-out set once, at the end.
**Questions answered:** the plays pack: Gutenberg's modern-spelling editions with persons lists, and a persons-list
reader whose facts are in-fiction, checked against the speaker tags (Folger's texts are CC BY-NC: not for a public
plugin). The `talk` adapter end to end waits for the base on grilly2 after 0022's job; run it then.
**Noted:** `serve_api.py` (559 lines) and `emitter.py` (750) were already over 300 and grew; a split belongs to its own
order.

