# 0027 — The control board: one page to see and steer Cubby (worlds, chat and its journey, a world map, forecasts, mood, system)
status: in progress
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
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
