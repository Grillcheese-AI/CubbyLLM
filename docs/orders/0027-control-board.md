# 0027 — The control board: one page to see and steer Cubby (worlds, chat and its journey, a world map, forecasts, mood, system)
status: in progress (P1 done 2026-10-22; P2 after 0024, P3 after 0020 and 0026)
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("perfect, start order 0027") · hypothesis: none (a tool; it measures nothing new) · **phased: P1 now, P2 after 0024, P3 after 0020 and 0026**

## Why
Nick (2026-10-22): "a nice control board for Cubby where we can explore the worlds, chat, visualize relations, visualize a
prompt, launch an agent (future work), a map that shows hot spots", plus news and alerts, the forecast tree, emotions and
arcs, and system health; in the style of a dark, glassy product dashboard, with a 3D view of the sections in the middle.
Everything it shows already exists or is ordered: the loop's events and `/audit/<id>` (0016), `/state` (the neurochem),
`/worlds`, the news archive (0024), the warnings and signals (0025), the tree (0026), arcs (0020), the copy-or-refuse
rates (0021/0022). Today they are spread over three dashboards (`/panel`, `/brain`, `/map`) and logs.

What it brings: one place that shows **why** Cubby said something (the route, the world, the fact, the check, the
words), what it refused, what it believes about the world today and what might follow. A conventional model can show
its answer, not the chain behind it, and has no "as of" or "I don't know" to show.
In plain words: you can watch Cubby think, see where in the world things are heating up, and ask "what if?".

**Design:** Cowork's mockup (seven screens, interactive, sample data) is handed over off-repo with its look (colours,
type, spacing, the glass panels) and its interactions. The seven section objects were modelled and rendered in Blender:
`dashboard/assets/sections/` holds the source `.blend`, a still per section and a 72-frame looping animated WebP per
section (the globe spins, the heart beats, the others move); the frames themselves are not committed.

## Build
- **Served by the chat server** behind `--board` (default off): `GET /board` returns one static page (`dashboard/board/`:
  `index.html`, a CSS file, small plain-JS modules; no build step, no framework, the same pattern as `/panel`). Each
  section is a view of that page: `#home`, `#watch`, `#chat`, `#worlds`, `#forecast`, `#mood`, `#system`.
- **Home:** the seven sections as objects on a CSS-3D turntable (`perspective` on the stage, each object at
  `rotateY(a) translateZ(R) rotateY(-a)` so it faces the viewer, a tick ring on the floor); ‹ › turn it, clicking an
  object brings it to the front; under it, three live cards for the front section; on the left, the section's views.
- **Read-only JSON endpoints** (`standin/board_api.py`, mounted by `serve_api.py`):
  - `/board/home`: per section, three numbers with a short trend each.
  - `/board/watch?layer=&asof=`: hot spots (0025's signal per country; until 0025 lands, GDELT event counts against
    each country's 30-day baseline), live warnings (USGS, GDACS, Environment Canada, WHO), and the day's grouped
    events from the archive (0024), each said as reported, dated, in Cubby's words.
  - `/board/relations?entity=&asof=`: the entity's neighbours in the worlds (0014/0015), each edge marked stable or
    dated (the time rule), with its world and source for the audit audience.
  - Chat uses the existing `POST /ask` and `/audit/<id>`; `/board/journey/<id>` turns a record's loop events into the
    steps (sense, route, recall, check, say) with their timings, and the question's spans with their roles (entity,
    relation, time) from the question-forms parser.
  - `/board/mood`: `/state` plus a display heart rate, `40 + 55 × arousal` bpm (a display mapping, not a claim), the
    point in the Lövheim cube, and (after 0020) `/board/arcs?source=`: the arc series and its change points.
  - `/board/forecast` (after 0026): tracked questions, a tree's nodes (chance, choice, outcome) with probabilities,
    analogues and the Brier skill score. "Make this the reality" re-roots the tree in the page; it changes nothing on
    the server.
  - `/board/system`: training runs (from their logs), GPU load if the backend reports it, the poller's
    `_polls.jsonl` (feeds up, last poll, items today), the order queue (the `status:` line of each order), and the
    latest copy-or-refuse rates.
- **Public mode** is the public audience 0016 already defines, enforced on the server: under it every `/board/*`
  response leaves out what 0016 keeps internal, and the audit tab is absent. The page's switch only changes which
  audience it asks as; it can never show what the server did not send.
- **Honest empty states:** a section whose order is not done shows "not built yet (order NNNN)" and no numbers. The
  mockup's numbers are samples and never ship.
- **Agents:** a disabled panel (goal, tools, budget, "coming in a later order"). Nothing runs.
- **Look:** dark slate-blue stage with soft light, frosted glass panels, one sans face at light weights for big
  numbers, tabular figures; amber for tension and the lit path, cyan for calm and cooperation, coral for official
  warnings; shapes differ too (dot, diamond, dot), so colour is never the only cue.

## Acceptance (pre-registered)
- A test calls every `/board/*` endpoint as the public audience over a fixture where one source is marked internal, and
  finds that source's name nowhere in the responses; the same call as the audit audience finds it.
- With `--board` off, `/board` and `/board/*` answer 404; with it on, nothing else in the server's answers changes.
- Every endpoint answers in under 200 ms at p95 on the local stack (report the numbers).
- The page works at 400 px wide (sections stack, no sideways scroll), every control is reachable by keyboard, and
  `prefers-reduced-motion` stops the turntable's and the objects' motion.
- The page makes no network call except to the chat server it came from (fonts self-hosted or system fonts).
- A section backed by an unfinished order shows its empty state, not numbers (a test per section).

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Read-only: the board changes no state except through `POST /ask`, which exists.
- No source name in a reply or in any public response (0016). No local paths in committed files.
- Large assets stay out of git: only the `.blend`, the stills and the animated WebPs are committed.

## Out of scope
- Running agents (a later order), editing worlds from the board, hosting the board publicly.
- New measurements: the board shows numbers other orders produce.

## Report
**P1 (2026-10-22):** the flag, the page, Home, Chat with its journey and audit tab, System, and the public test. P2 (Watch,
Worlds) and P3 (Mood arcs, Forecast) wait for their orders; their sections answer their empty state today.

- **Commits:** 94f9417 (in progress), 64bfef1 (P1).
- **Files:** `standin/board_api.py` (the routes, the static and asset files, the public filter), `standin/board_data.py`
  (what Home, the journey and System read), `standin/serve_api.py` (`--board` and one dispatch block: +20 lines on a
  file already over 300; the board lives in its own modules), `dashboard/board/` (`index.html`, `board.css`, `api.js`,
  `app.js`, `home.js`, `chat.js`, `system.js`; no framework, no build step), `dashboard/assets/sections/` (the
  `.blend`, the seven animated WebPs and their stills), `standin/tests/test_board.py`, `validation/board_timing.py`.
- **Tests:** `standin/tests/test_board.py` 5 passed. `pytest tests`: only the known `test_every_module_declares_wiring`
  (order 0002) red.
- **Measured:**
  - **Public vs audit:** every `/board/*` path called as the public audience over a fixture whose one feed is internal:
    its name is in no response; as the audit audience it is (the feeds table, the journey's sources). Each response is
    built for its audience (the feeds by name and the journey's provenance only in the audit one), and a public
    response is then checked string by string against every internal name the server knows (the feeds' ids and
    sites, the store's source names, the mounted worlds): a hit would become `[internal]`.
  - **Off / on:** with `--board` off, `/board`, `/board/`, every `/board/*` path, the static and asset files answer
    404; with it on, `/health`, `/state`, `/worlds` and an unknown path answer exactly as with it off (test).
  - **Latency** (`validation/logs/board_timing_2026-10-22.json`; the server's own handler, a stand-in brain with 200
    kept records, this machine's archive, order queue and run logs; 50 calls per endpoint and audience, measured while
    order 0022's 450M job held 6 CPU threads): warm p95 at most 22.8 ms (every endpoint, both audiences); the first
    call after the 5-second cache, 205.6 ms at worst (Home: it reads the run logs on the synced Drive). Under the
    200 ms line at p95; a cold refresh is about at it.
  - **The page**, in Chrome on this machine: Home renders the turntable with the front section's cards; the ‹ ›
    buttons, a click on an object and ← → on the focused front object are wired, but I did not drive them in the
    browser (Chat's form I did, by keyboard: type, Enter); only the front object plays its WebP, the others show their stills; with
    `prefers-reduced-motion` all seven are stills and nothing transitions (checked by forcing the media query). At
    400 px wide (an iframe of 400 px, since Chrome's window stops at 500): no sideways scroll in Home, Chat, System
    or an empty section. Every request the page made went to its own host. Chat asks through `POST /ask`, shows the
    reply with its tags, the journey (the question's spans, the five steps and their timings) and, in the audit view
    only, `/audit/<id>`; the Public switch is disabled without a token and the audit tab is disabled in public.
  - **Empty states:** Watch (0024), Worlds (0015), Forecast (0026), Mood (0020) and Agents answer `not built yet
    (order NNNN)` with no numbers, on their own paths and on Home (test).
- **Not verified here:**
  - The board on the real serve stack with a model (the checks ran on the server's handler with a stand-in brain:
    the board reads only what the server holds). With the real `--ask` loop the journey's timings come from the event
    ring; a record older than the ring's 20,000 events shows its steps without timings.
  - GPU load: no backend here reports it; the System tile says so.
  - Screen readers beyond the semantic roles and labels (not tried with one).
- **Found on the way:**
  - The mockup loads Google Fonts; the page uses system fonts so it calls nothing but its server (the acceptance
    line). The look is otherwise the mockup's (its colours, glass, spacing, the turntable's geometry).
  - The run logs' last line is a probe record; the loss is read from the last training line instead.
  - `dashboard/assets/sections/_render_anim.py` holds a local path: left uncommitted, as the note asked; the PNG
    copies and the `.blend1` backup are left out too (the WebP stills serve the page).
- **Questions:**
  - System's "Training" reads run folders under `CUBBY_RUNS` (the synced Drive's `runs` here). Should a run started
    on Colab report its live step there, or is the last synced metrics line enough?
  - Should the Mood section show today's `/state` (the heartbeat and the cube point exist now) before 0020's arcs,
    or stay empty until P3 as the note says? It stays empty for now.

### After the review (2026-10-22): the scrub fix, the background build, run ages, Mood, the real stack
- **The scrub matches whole words, and only internal names.**
  - `board_api.internal_names` holds:
    - the store's source names;
    - the feeds and their sites;
    - the worlds registry's dataset names (`$CUBBY_WORLDS/registry.sqlite`, 0015);
    - internal plugins.
  - It leaves out the keys worlds are mounted under ("facts", "wiki", "self", "reality:history") and a few generic
    words ("misc", "data").
  - `scrub` matches whole words, case folded, so a name inside a longer word is another word.
  - Test (`standin/tests/test_board_review.py`, with those world keys mounted):
    - these survive: "I don't have facts about that", "3.4M facts held", "Cubby talks about itself", "trained on
      wikis", "a globalwireless router";
    - these become `[internal]`: "trained on wikidata5m", "from Wikidata5M today", "read on globalwire.example",
      "globalwire: 3 new".
- **No request pays for the cold build.** `board_api.start_refresh` (started by `--board`) rebuilds Home and System,
  for both audiences, and the name list every 5 s in a daemon thread. A request reads the last build: 3.2 ms worst of
  10 on the real stack, against 205.6 ms cold before (`validation/logs/board_real_stack_2026-10-22.json`). A failed
  build keeps the last one and says so in the server's log.
- **Training runs show their age:** "as of 21:51, 15 h ago", with a stale tag after 60 minutes (`board_data.training`,
  `STALE_MIN`). Home's Training card carries the same line. Colab runs reach the board only through the synced drive,
  as answered.
- **Mood is built** (`/board/mood`, `dashboard/board/mood.js`):
  - today's `/state`: the heartbeat at `40 + 55 × arousal` bpm, labelled as a display mapping;
  - the point in the Lövheim cube, with its nearest corner and intensity;
  - the levels as meters;
  - "Arcs: not built yet (order 0020)".
  - At the cube's centre every corner is equally near, and the first check showed "spent" at intensity 0.0. A point
    under intensity 0.05 now reads "at rest", with no corner named (test). Home's Mood section has three cards.
- **On the real serve stack** (450M emitter and talk seats on grilly2, the wiki world with 551,661 facts, the hdc world,
  the ask loop, four plugins; `board_real_stack_2026-10-22.json`):
  - **Public audience:** no internal name in any `/board/*` response (11 routes) or in the page's own files, checked
    whole-word against the server's 305 names. The audit audience carries 134 of them, so the check can find them.
  - **What the scrub replaced:** an order title naming Wikidata, and the plays plugin's licence naming Project Gutenberg.
  - **A real `/ask` in public:** refused ("plan_does_not_cover_question"), no internal name in the record, and its
    journey has the steps with their timings.
  - **Fixed along the way:**
    - The poller tile read only `CUBBY_NEWS_ARCHIVE`; the server's `--news-archive` now reaches it
      (`brain.news_archive`).
    - The copy-or-refuse tile missed order 0022's end-token rerun, because its log name did not match the glob; it now
      shows that rerun as the latest 450M reading (1.7% wrong).
- **The turntable, driven in Chrome:**
  - › twice: II → IV; ‹ once: IV → III.
  - A click on Mood's heart, behind Forecast, woke Forecast instead. Forecast's square picture has transparent corners,
    and they took the click.
  - Now only an oval over each item's art, and its name, take clicks (`board.css`). The same click brings Mood to the
    front (V / VII). Keyboard focus is unchanged.
- **Found on the way (not 0027's):** `standin/tests/test_news_reply.py` failed. A news reply said "also named Planet,
  Leeds, Daily", because "the Daily Planet" spelled the outlet `dailyplanet` across two words, and each word passed
  alone. That was a source named in a reply (0016). `news_reply.outlet_words` now drops adjacent capitalised words
  whose run spells an outlet's name; the news tests pass.
- **Tests:** `test_board_review.py` (7), `test_board.py` (5, Mood now built), with `test_plugins_serve.py`,
  `test_audience.py`, `test_news_reply.py` and `test_ask.py`: 52 passed. `pytest standin/tests tests`: the known
  `test_every_module_declares_wiring` (0002) is red, along with failures in `test_build_chat_sft.py` (24) and
  `test_chat.py` (1) that predate this order, and a collection error in `test_goemotions_cube.py` (an import broken
  since `34e7658`, 2026-09-25). I left those alone.
- **Not done:** Manrope is not bundled (optional; it would mean downloading the font, which needs a yes). The test
  server's `--news-archive` (and its environment default) started a second poller beside the 7-day one during the check.
  The archive keys items by feed and guid, so nothing was stored twice: the first start fetched the 247 items those
  feeds were due, the second 0. The server was stopped after.

## Review (Cowork, 2026-10-22): P1 accepted, one fix
Checked: `test_board.py`, `test_plugins.py`, `test_plugins_serve.py` and `test_audience.py` pass (20, `.venv-dml`); the
static and asset routes take only `[\w.-]+` names of their listed types, so no path leaves their folders; the
committed assets are the `.blend`, the animated WebPs and the stills; no local path in the board's files.
**P1 is accepted. One fix:**
- **The public scrub hides ordinary words.** It matches internal names as substrings, and the names include every
  mounted world's key, so on the real stack (worlds `facts`, `wiki`, ...) common words go. Probe on `board_api.scrub`
  (`validation/logs/_review_0027_28.py`, not committed): "3.4M facts held", "Cubby talks about itself" (`self`) and
  "trained on wikis" all become `[internal]`; "Weather warnings in effect" survives. The test's fake brain mounts no
  world, so it cannot see this. Fix: whole-word matches, and a name list that holds what is actually internal (sources,
  feeds, sites, dataset names), not generic world keys; a test with a real-like set of worlds where a public reply
  "I don't have facts about that" survives and "wikidata5m" does not.
- Latency: warm p95 22.8 ms; the cold Home (205.6 ms) pays for reading run logs on the synced Drive. Refresh that in
  the background so no request pays it.
- System fonts meet the line. Optional: bundle Manrope (SIL Open Font Licence) with the page to keep the designed
  look offline.
- Not verified, still to do once: the board on the real serve stack with its worlds mounted, both audiences (that is
  where the scrub fix shows), and the turntable's ‹ › and clicks driven in a browser.
**Questions:**
- Colab runs: the last synced metrics line is enough. Show its age ("as of 14:02, 9 min ago") and mark it stale after
  an hour; no live channel from Colab.
- Mood: show today's `/state` now (the heartbeat at `40 + 55 × arousal`, the cube point, the levels). It exists, so
  the empty-state rule does not apply to it; the arcs part says "not built yet (0020)" until P3.

## Review 2 (Cowork, 2026-10-22): P1 done
Re-ran `test_board_review.py` and `test_board.py` (pass). The scrub fix is the right shape (whole words, only what is
actually internal: sources, feeds and sites, dataset names, internal plugins), and the real-stack check found the two
places it mattered (an order title naming Wikidata, a licence naming Project Gutenberg). The background build, run ages,
Mood from `/state` (with "at rest" at the cube's centre), the click target fix on the turntable and the `news_reply`
outlet fix are accepted. **P1 is done.**
- Manrope: Nick said yes (2026-10-22). Bundle it with the page (a small download, SIL Open Font Licence; keep its
  licence file beside the fonts), with the system stack as the fallback.
- The 25 red tests in `standin/tests` that predate this order go to the hygiene order (0030), with the old local paths.
