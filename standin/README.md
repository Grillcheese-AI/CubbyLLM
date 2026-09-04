# standin/ — the stand-in trunk

**Status (2026-09-04):** v10t built (91,754 records = v9t + 1,830 toolcall rows in both call formats, incl. `request_tool` — the factory's meta-call) and on Drive, awaiting its two-arm run; **v9 serves: `--gguf standin/models/emitter_v8e.Q4_K_M.gguf --talk-gguf standin/models/emitter_v9t.Q4_K_M.gguf`** (the talk-base decision below: LFM stays); v9t bake-off read for three arms (see § v9t bake-off: the control recovers history to 0.65, Qwen3-4B is the one candidate with gains, the 8B MoE is out, Gemma pending a rerun); v9t data built (89,890 talk records from the SFT gap map + the owner's sources; see § v9), awaiting its Colab run; v7 is the serving model; v8 (two adapters on one base) is trained and exported (both Q4s local: `standin/models/emitter_v8{e,t}.Q4_K_M.gguf`); the v8t talk read holds v7, v8e's forge probe is 1.00/1.00/1.00 and the two-adapter self-test 96/4/0 (see § v8); arithmetic at n=40 — the kill line — is pending. Nothing here is CubbyLLM.

A small open model (currently `LiquidAI/LFM2.5-2.6B`, chosen by a 2026-08-30 Hub
check — see `docs/research/2026-08-28-oracle-competition-scored.md` §3.2 for why a
stand-in exists at all) is fine-tuned with Unsloth on Colab and served locally as a
GGUF, so the parts of the serve stack that need *some* trunk — the CubeLang program
emitter, the learning gate as a live nightly loop, the chat/grounded-answer surface,
the H-F2 bridge contract — can be built and measured while the 2B CubbyLLM trunk
trains. It is replaced the day the 2B checkpoint exists.

## Where things stand (2026-09-03) — read this first

- **Serving:** `serve_api.py --gguf standin/models/emitter_v8e.Q4_K_M.gguf --talk-gguf standin/models/emitter_v9t.Q4_K_M.gguf --pacman` (v9, 2026-09-04: the program adapter + the talk adapter). `/` chat + console, `/pac` the live game, `/health` the adapters.
- **Measured (v7, VM-verified):** chat 1.0 · content 1.0 · identity 0.867 · emotion 0.675 · affect 0.975 · history 0.625 · arithmetic 0.675 · chain 1.0 · game families 1.0 · forge probe 1.00/1.00/1.00 · self-test 96% correct / 4% don't-know / 0% wrong.
- **Open:** the v8e/v8t runs and their A/B (does adapter-per-context remove the interference?); arithmetic's three-round slide; "Good evening!"; the adapter lifecycle's detector, tag-general partition builder and promote rule; the reasoning roadmap from the GoT challenge (`docs/research/2026-09-03-got-challenge-scored.md` §7).

**Sections, in reading order (not date order):** the two guardrails · what is here (the file map) · the built
set (the program families and their VM re-verification) · identity + hormones · the thalamus, thinking out
loud, the LLM speaking for itself · routing fixes after the first live sessions (the v5 → v6 → v7 data rounds,
each with its measured read) · **two adapters on one base — v8** (the θ = f(c) interface, the adapter bank,
the lifecycle, the concept sources) · the v7 / v6 reads · chat is mediated by a VM program (design note) ·
data facts worth knowing.

## The two guardrails

1. **Nothing the stand-in measures is a CubbyLLM result.** It says nothing about the
   hybrid backbone, θ=f(c), bounded state, the exact fork, the episodic store or the
   retrieval head. Every number it produces is tagged `[stand-in]`, lives in this
   directory's docs or `TODO.md`, and enters `CUBBYLLM_HYPOTHESES.md` only as a
   pointer. This is the sibling repos' "self-reported numbers presented as
   measurements" failure with a new face — do not let it in.
2. **A swap, not a fork.** The stand-in sits behind the trunk interface
   (`standin/emitter.py::Emitter`); `cubbyllm/` never imports `standin/` (pinned by
   `standin/tests/test_build_emitter_sft.py::test_cubbyllm_never_imports_standin`).
   The 2B checkpoint drops in by implementing the same interface, and nothing above
   it changes.

## What is here

| Path | What |
|---|---|
| `data/build_emitter_sft.py` | Builds the emitter SFT set from `cubemind/sandbox/regen` (arithmetic / role-binding / kernels; dialect shim for the current VM's `ISolver` contract) + our 517 verified chain programs; **re-verifies every program through the real Rust VM**; excludes GSM8K *test* (H-G4's eval) outright; caps role-binding; deterministic 95/5 split; writes a manifest with input hashes and exclusion counts. Output in `data/out/` (gitignored). |
| `emitter.py` | The trunk-facing interface (`Emitter` protocol) and three implementations: `LlamaCppEmitter` (in-process GGUF via `llama-cpp-python` 0.3.30 — **verified 2026-08-30: it bundles `ggml-vulkan.dll` and finds the RX 6750 XT**; no standalone `llama-server` exists on this machine; end-to-end check same day with `LiquidAI/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` in `standin/models/` (gitignored): the LFM2 architecture loads, first turn 4.2 s incl. load, next turns ~0.2 s, and the untrained base already reads the injected hormonal block), `LlamaServerEmitter` (OpenAI-compatible HTTP, e.g. `python -m llama_cpp.server`), `ReplayEmitter` (the notebook's recorded generations). |
| `chat.py` | **The VM-mediated chat turn.** `CubbyTalk implements IAgent` is rendered per turn with the model's voice-filtered reply + the verbatim don't-know line embedded as literals; `think` ASKs, the host selects, `VM::resume` rejects anything not offered (tested through the whole stack), `act` remembers, `observe` counts. Hormonal state reaches the model via `identity_system(state)`; `nudge()` runs the real cubemind neurochemistry ODE (`neurochem.py`) — appraisal → a couple of perception frames → the 5-hormone serving state (the 2026-08-31 placeholder is retired). The voice filter is host-side (`identity.voice_ok`) because two VM string constructs are silent stubs under strict — pinned in cubelang `tests/str_semantics.rs`. |
| `eval_emitter_vm.py` | The **verified** read: runs generated programs through `cubelang.exe`, compares to gold (arithmetic / kernels / chains) or to execution (role-binding). Consumes the Colab notebook's `val_generations.json` or generates live through `LlamaServerEmitter`. |
| `serve.py` | **The stand-in brain** (2026-08-31, brain shape same day): one process, one turn — **sense** (appraisal → the ported neurochemistry ODE; hormones scale the routing threshold via `modulate_threshold`, never facts) → **fast route** (the SNN slot — today lexical+retrieval, the interface shaped for the spikeybrain port, no SNN claim) → **cortex router** → `MemoryCortex` (live learning), `ReasoningCortex` (plan+**walk** with the measured CoT pipeline at tau_vm=0.2202 / tau_ret=0.5959, flat top-k fallback → Facts block → v3 emitter with the CotChain opening **prefilled** → VM → ground check + **consistency gate**: a verified walk that disagrees vetoes the answer), a mounted **plugin cortex**, or `CubbyChat`. Every spoken reply from every cortex exits through `CubbyTalk`'s ASK (`CubbyChat.mediate`) — the VM rejects any selection it didn't offer, and the voice rules + verbatim don't-know line hold whatever is mounted. REPL: `python standin/serve.py --gguf standin/models/emitter_v5.Q4_K_M.gguf`. |
| `neurochem.py` | **The real 5-signal neurochemical ODE**, ported clean from `cubemind/brain/neurochemistry.py` (port-don't-link): dH/dt = α·drive·sensitivity − β·(H−resting), receptor-saturated couplings, cortisol as a slow HPA EMA (NE is the fast stress signal), refractory receptor sensitivity, Lövheim-cube emotion readout, `modulate_threshold`/`modulate_tau`. Replaces `nudge()`'s placeholder as `CubbyChat`'s state source; `appraise()` is the host's transparent lexical (EN/FR) reading of a message into the five drives. Same clip bands `identity.HORMONE_RANGE` pins. |
| `worlds.py` | `FactStore` — the **incremental** retrieval world (same unit-cosine math as exp_m3's `make_retriever`, plus `add()` so a fact accepted at turn N is retrievable at turn N+1; IDF-overlap fallback without an encoder) — `route_world` (MoWM v0: best top-score wins) and the **`CubbyPlugin` mount contract, MindForge style**: a plugin (cubbyverse) sits ON TOP of the trunk — it contributes named worlds and optional cortices and observes turns; it imports us, we never import it, and it can never produce speech that bypasses the ASK + voice rules. |
| `pacman.py` | **cubby-man wired to the cubbyverse 3D Pac-Man** (2026-08-31). Wire target is `cubbyverse/examples/pacman_3d.py` — deliberately: `pacman_live.py` imports it for MOVES/encoder, so it is the shared substrate of both. The maze is ported verbatim (same rng call order, seed 7 — the 7×7×3, 16-pellet map the scripted VSA-planner demo beats) and `CubbyPac(CubbyMan)` plays it the explorer way: six VM-guarded directions, walls from ASK refusals, pellets eaten on arrival as permanent discovery facts, six-direction symmetry/count joins, the run recorded in pacman_3d's own replay schema and rendered through its `_HTML` three.js template — extracted from the file at render time, never imported (no grilly/cubbyverse package dependency) — to `C:/tmp/cubbyman_pacman_3d.html`. Mount with `--pacman` on `serve.py`/`serve_api.py`, then "play pacman for 30". **LIVE play (2026-09-02): `LivePac` + `pac_live.html`** — the pacman_live.py pattern: the browser polls `GET /pac/state`, each due poll advances ONE VM-guarded step (rate-limited; he plays only while the page is open), and the live three.js page at **`/pac`** animates him in real time (pellets vanish as eaten, HUD shows score/steps/facts/walls-learned/derived/emotion); the step lock keeps the poller and a chat "play pacman" turn from interleaving a move, and beating the map auto-renders the replay HTML. **The BIG game (2026-09-02): `GhostVerse` + `CubbyGhost`** — pacman_live.py's game ported as his env: seed-shared carved labyrinths (`Mulberry32`/`carve_labyrinth` bit-for-bit), the exact level formulas (size/walls/hazards/pellets per level), star power-pellets with the farthest-point spread, FRIGHT_STEPS=14 / GHOST_BONUS=5, lives, level advance with level-scoped fact names (everything he ever learned stays true). He SEES hazards glow (observation facts); plain walls stay invisible until a refused move teaches them. Ghost proximity feeds THREAT into the neurochemistry (anxious when hunted, hunting when they're frightened — the fear-aware `_pick`), being caught is learned as a danger fact. Simplifications on record: greedy-Manhattan ghosts (theirs are mSA controllers with scent fields); vaults/superpowers/letter-words/time-budget stay with the cubbyverse-side splice (no jump ⇒ vaulted pellets would be unreachable). `--pacman` mounts the big game. **The view IS pacman_live's own page (2026-09-02):** `load_frontend()` extracts its `FRONTEND` block verbatim at serve time (never imported — the file pulls grilly/cubbyverse packages) and `LivePac` speaks its exact protocol (`/pac/init`, `/pac/state` with the `stale` cached-frame rule, `/pac/next`, `/pac/assets/*` for cubby's face textures): the isometric glow grid, the face-textured sphere, energy/steps gauges, hearts, ghost-fear, the **Plutchik cone compass** (our neurochemistry → petal/tier/color/message, tier metadata read from `cubbyverse/core/emotion/plutchik.json` by path), letter pellets spelling the level word (exact assignment rule), and **speech through CubbyTalk**: a grounded word (letters collected) is SAID when its percept recurs — ghost near, power-up, discovery, new maze, solved — via `CubbyChat.mediate`, so even in-game utterances pass the VM's ASK. **Console panel on the game page (owner, 2026-09-02):** the brain's event feed (`GET /events` — sense/route/walk/vm/gate/learn/explore/probe/eat/caught/flee/plan/program/forge/superpower_move/level_up…) is injected as a fixed panel under the big stage number, so the game and what Cubby does are one screen (`_CONSOLE_PANEL`, injected before `</body>`; "follow" toggle; link to the full console at `/`). Nine string patches, each pinned by a test so an upstream change surfaces: endpoint prefixes, the subtitle/foot (say what drives him), and two HUD rows relabeled to numbers we actually have (`world model: facts/derived/walls`, `refused moves`) instead of R-STDP weights we don't. **The real goal (owner, 2026-09-02): spatio-temporal orientation, threat and how to fight it, and self-improvement WITHOUT forgetting.** What backs each: *orientation* — he plans over his OWN map (BFS across the neighbor facts he holds, level-scoped so nothing goes stale), sees pellets glow within two cells as sighting facts, keeps sightings across out-of-time retries (attempt 2 knows where they are); *threat* — a danger radius that GROWS with learned fear (+0.7 per catch, ×0.9 per cleared level — the original's rule), routes planned around ghost neighborhoods, a flee move that maximizes distance, hunting them when frightened; *self-improvement* — he **generates his own programs**: a JUMP composed when a seen pellet has no path (hazard-walled), and on out-of-time or plain curiosity a new composition sampled from a pattern grammar (slots A/B/C, length 2–5, instantiated over any direction assignment — DASH/BLINK/COMBO are three points in that space; `COMBO-AABA`, `KNIGHT`, `WARP`… are moves the original never had), each written as CubeLang, VM-certified, then offered by the ASK like any exit and taken when his map says it shortens the way; proposal lengths are weighted by what paid off. **One program, a reusable function per combo (owner, 2026-09-02):** his moves live in ONE CubeLang program, `program Moves implements ISolve` — a function per active combo (`dash()`, `knight()`, `combo_aab()`, `jump()`), `solve()` the catalogue smoke — that he EDITS in play: a new or modified combo is a new function, certified by running *that function inside that program* on the VM, and the program is saved beside the notebook as `cubbyman_moves.cube`; each move's notebook entry stores the program as it stood when certified, so the evolution is readable. **He modifies before he invents**: `_propose` first edits an existing pattern (append / drop / swap one slot — lineage `parent` + `edit` recorded, the parent keeps its `children`), and only samples a fresh composition when there is nothing to edit. **The library consolidates after each level** (`consolidate`: the best `MAX_ACTIVE_PATTERNS−2` patterns stay active, the rest are retired with the reason — never deleted; also triggered when the active set hits the cap), and **the ASK offers at most 16 power moves** per step, ranked by landing near a seen pellet then by the program's value. Why the cap: at level 3 with 23 patterns the ASK carried 124 candidates and **the VM returned one of them altered** (`combo-aaba_left_left_forward_left` came back as a duplicate of another; 126 unique names in a synthetic ask round-trip fine, so it is a content edge past ~100 operands — pinned as a cubelang follow-up in TODO). **Nothing is forgotten**: the `ProgramLibrary` keeps every program with its source, a **reasoning trace** (trigger, the situation down to hormones and emotion, the sampling rationale, the VM verdict) and every use with what it earned; useless patterns are *retired* (no longer offered) never deleted; persisted as `standin/data/out/cubbyman_programs.json` + a readable `.md` notebook (also at `/pac/programs.md`) and reloaded on restart. Measured in a 2-stage sim (`seed 0`): stage 1 cleared with ghosts on; JUMP at step 2, `COMBO-AABA` invented on out-of-time then used 55× saving 165 steps. **Arbitrary CubeLang (`forge.py`, 2026-09-02): his trunk writes programs for live tasks.** A `Task` is a prompt in one of the trunk's trained families (the story words never reach the VM; the comparison semantics do) plus an expectation he can check himself — the emitter writes the program, the VM runs it, the result is certified, and the program is kept as a tool (prompt, source, verdict, pass/fail) in the same notebook. In play: a **FLEE decision** program on the live numbers when a ghost is inside his radius (sensor template; the trained kernels answer a confidence *class*, ≥50 = true, measured over the SFT set), a **SAFER-EXIT compare** program to pick where to run, a **WHERE chain** at each new maze checking his bearings against his own map. He acts on a program's answer only when certified against his rule; otherwise the rule stands and the notebook records the rejection. **Measured acceptance of the v3 Q4 stand-in (`scripts/forge_probe.py`, 24 tasks, `data/out/forge_probe.json`): decision 0.75 (6/8), chain 0.50 (4/8), compare 0.00 (0/8).** Failure modes on record: compare programs declare `result = 0` and forget the `add result, lhs` in the branch; two decisions had `add x : 1` colon-for-comma parse errors; chain misses bind a neighboring coordinate. Game-phrased prompts scored 0/8, 1/8, 0/8 before the tasks were rephrased in the trained templates — the trunk is narrow, and the certification is what makes using it safe. **v4 BUILT (2026-09-02, `data/build_game_sft.py`; on Drive at `cubbyllm/standin/emitter_sft_v4.jsonl`, sha `333141dd…`): 11,144 records = the whole v3 set as replay (9,284, unchanged — nothing learned is lost) + 1,860 game records the VM verified** (1,876 generated, 16 compose programs dropped as gold mismatches — all 4- and 5-step compositions whose frame holds 5–6 bound roles: `recover` returned the role token `H6_NAME` instead of the name, a VM frame-capacity edge, so the set only teaches compositions the VM certifies): flee/go decisions on live numbers in the game's words (414), safer-exit/shorter-path compares on the small numbers v3 failed (353), 1- and 2-hop neighbor chains over level-scoped cells (532 + 152), exit counts (177), superpower compositions (231) — plus the 1 certified FLEE tool his own notebook held from live play (the "he generates his data" path, literally). Game records carry `repeat` 2 in train; 101 in val. Train with `notebooks/standin_emitter_sft.ipynb` (`VERSION='v4'`, the default now), then locally `eval_emitter_vm.py --data standin/data/out/emitter_sft_v4.jsonl --gguf <v4>` (game families reported apart) and `scripts/forge_probe.py --gguf <v4>` — the before/after acceptance is the result. **v4 TRAINED (Colab, 2026-09-02; `standin/models/emitter_v4.Q4_K_M.gguf`, Drive `emitter_lfm25_2p6b_v4/gguf_gguf/`). Forge probe before → after: decision 0.75 → 1.00 (8/8), compare 0.00 → 1.00 (8/8), chain 0.50 → 1.00 (8/8)** (`data/out/forge_probe.json`) — and the same **24/24 on a held-out maze** (level 5, seed 3; v4 trained on levels 1–3; `forge_probe_v4_level5.json`), so the chain number is not memorized cells. The compare programs now carry the `add result, lhs` the v3 ones forgot; chain answers keep the `cell x-y-z` form. The game generated the data; the trunk learned the game's programs. The `pacman_live.py` splice (its `Game.step()` → `planner.plan_to_any` decision point taking the brain's ASK-mediated choice) is the cubbyverse-side follow-up. **Sound (2026-09-02):** the owner's soundtrack (`data/music/neon_pixel_dash.mp3`, served at `/pac/music`, override `CB_PAC_MUSIC`) loops at a low volume UNDER 8-bit WebAudio effects synthesised in the page for the game's events (a step ticks, a pellet is the two-note waka, a star an arpeggio, a trap a noise burst, a catch a falling saw, a cleared level a fanfare), and the level-start jingle (`data/music/game_start.mp3`, `/pac/sfx/game_start`) plays over the ducked music when a level builds. Browsers need a gesture before audio, so the `♪ sound` toggle (bottom-left, with the music slider) is that gesture; the choice is remembered per browser. |
| `perception.py` | **The trunk as the appraisal stage** (2026-09-02, v5): `ModelAppraiser` asks the emitter for a turn's emotion with the exact v5 prompt (EN/FR) under the identity system prompt, reads the Plutchik petal, and lifts the lexical `appraise()` drives by the petal's (max per drive, larger-magnitude valence wins) — a model miss never removes a signal the lexicon caught, a dead trunk falls back to the lexicon. `CubbyChat(appraiser=…)` / `serve_api --model-appraisal` (opt-in until v5's measured emotion accuracy earns the default). `classify_content` is the nsfw/safe read, **log-only**: the gate's input, not the gate. |
| `serve_api.py` + `demo.html` | The HTTP surface the cubbyverse front-end talks to (stdlib, JSON, CORS, localhost): `POST /turn` → the turn record (reply, kind, register, **emotion**, state, route, task/learn detail); `GET /state` → the full neurochemistry read (the emotion-compass feed); `GET /events?since=N` → **the console feed** (`CubbyBrain.trace`: user/sense/route/walk/emit/vm/gate/learn/explore/probe/derive/speak — every reasoning step and action as it happens); `GET /worlds`, `GET /health`; `GET /` serves `demo.html`, a self-contained dark **console window** — chat on the left, the live event stream + hormone bars + emotion on the right. Transport only — the brain stays `serve.CubbyBrain`. `python standin/serve_api.py --gguf standin/models/emitter_v5.Q4_K_M.gguf` then open http://127.0.0.1:8765/. `GET /pac/music` + `GET /pac/sfx/<name>` serve the game's soundtrack and sampled effects. |
| `verse.py` | **The toy cubbyverse — cubby-man's OWN world model, learned by exploring** (2026-08-31; the reference `CubbyPlugin`). He starts knowing two facts; every move is VM-mediated (the IAgent ASK offers the exits, resume guards the choice), every arrival's observations are written into his own `FactStore` **at discovery time** through the same anti-poisoning gate as user facts. He **tries outside the offered scope** — an unoffered direction is rejected by the VM guard, and the rejection is learned as a wall fact — and he **writes his own CubeLang programs to join known facts**: VM-computed exit counts (one `add` per known neighbor — the number comes from execution) and symmetry joins about places not yet visited, each accepted only as a bind/recover certificate. The measured arc (live-tested, 3×2 world, 10 steps): coverage ≥0.8, zero guard anomalies, then the same question that got the don't-know line answers correctly from his world through the normal reasoning cortex. Curiosity is count-based novelty modulated by the hormones (cortisol flips to preferring known ground); discoveries feed novelty back into the ODE. The real cubbyverse repo implements this contract over its own worldgen/cubby-man. |
| `tests/` | Unit pins for the builder's pure helpers + the import guard. `python -m pytest standin/tests -q` |
| `../notebooks/standin_emitter_sft.ipynb` | Unsloth LoRA SFT on Colab, format-level exact-match eval, GGUF export to Drive. |

## The built set (2026-08-30, `data/out/emitter_sft.manifest.json`; on Drive at `cubbyllm/standin/`)

| task | records | train / val | verification on the Rust VM |
|---|---|---|---|
| arithmetic (GSM8K *train*-derived) | 6,148 | 5,819 / 329 | 6,148 execute, **6,148 match gold** |
| kernel (decision / compare / loop / recall) | 593 | 564 / 29 | 593 execute, **593 match gold** |
| role_binding (capped from 32k; 4,992 dups removed) | 4,000 | 3,780 / 220 | 4,000 execute (no gold exists) |
| chain (ours, `ISolve`/`recover`, 1–3 hops) | 517 | 492 / 25 | 517 execute, **517 match gold** — scored on the last hop's function (`hop_N`), not `solve()` |
| identity (EN 276 / FR 250, own system prompt + hormonal state) | 526 | 504 / 22 | skipped by the VM pass; scored by `identity_ok` |
| **total (v1, 2026-08-30 a.m.)** | **11,784** | 11,159 / 625 | 0 dropped; VM pass 296 s; output sha256 `8c90a685…` |

**v1 emitter, VM-verified (2026-08-30; LFM2.5-2.6B LoRA on the v1 set; 166 stratified val
records replayed through the real VM — `data/out/eval_emitter_vm_v1_replay.json`):**

| task | executes | gold_match | text-EM | reading |
|---|---|---|---|---|
| arithmetic (60) | 0.967 | **0.617** | 0.000 | solves parametrically with *different* decompositions — EM blind, VM sees it |
| chain (25) | 0.960 | **0.200** | 0.200 | perfectly-formed programs binding the WRONG fact — the prompt lacked the retrieved facts (fixed in v3) |
| kernel (29) | 0.759 | 0.759 | 0.000 | lower bound: the eval capped generation at 320 tokens and long kernels truncated (parse error at Eof) |
| role_binding (30) | 1.000 | — | 0.100 | all execute |
| identity (22) | — | **identity_ok 1.000** (EN and FR) | — | voice rules hold |

**v3 emitter, VM-verified (2026-08-31; trained on the v3 set; 169 val records replayed —
`data/out/eval_emitter_vm_v3_replay.json`): every program executes, and the chain fix
landed decisively.**

| task | executes | gold_match | vs v1 |
|---|---|---|---|
| chain (28) | 1.000 | **0.929** | 0.200 → **0.929** (facts in the prompt) |
| kernel (29) | 1.000 | **0.862** | 0.759 → 0.862 (eval `max_new` 600 removed the truncation) |
| arithmetic (60) | 1.000 | **0.650** | 0.617 → 0.650 |
| role_binding (30) | 1.000 | — | all execute |
| identity (22) | — | **identity_ok 1.000** (EN + FR) | held |

**Serve selftest (2026-08-31, `data/out/serve_selftest_v3.json`; 25 val chain questions with
their Facts blocks STRIPPED — serve retrieves/walks its own facts from a 2,942-fact store,
the one link no prior eval covered):** gold_match **0.76**, walked-facts recovery 0.96;
what the turn would actually SAY: **76% correct, 20% the don't-know line, 4% wrong**
(the consistency gate caught 5 of the 6 emitter misses — their verified walks held gold and
disagreed; the one spoken-wrong is the row whose walk failed, leaving nothing to catch it).
Getting here was a ladder of three measured serve-side failures, each a real finding:
flat top-k facts with distractors → **0/25** (the model learned "every fact = a hop" and
`answer_fn`'s last hop reads a distractor; style flips to the event/kernel shapes);
prefilling only `program CotChain … {` mid-line → still 0/25 (degenerate flattened
non-syntax — the prefill must run through `create frame: number;`); walking at
`tau_vm=0.0` → every good walk banned (the control role's ~0.05 similarity always reads
as a violation at that degenerate threshold). Steering note: serve prefills the CotChain
opening; `eval_emitter_vm`'s 0.929 was measured WITHOUT steering.

This is the serviceable emitter: retrieve → format the Facts block → emit → VM verifies.
**v3 Q4_K_M verified locally (2026-08-31, `standin/models/emitter_v3.Q4_K_M.gguf`)** by two
discriminating probes (`scripts/probe_v3.py`): closes the think block immediately and binds an
*invented* Facts block (Zorblax → Fnordovia → Quuxville) from the prompt, not from memory.
Drive's `Q8_0` is still the v1 export. **GPU serving works again**: the 2026-08-30 system crash
was the stale June llama-cpp-python 0.3.30 Vulkan backend — after upgrading to **0.3.35**
(`scripts/probe_v3_gpu.ps1 -Upgrade`, Vulkan wheel index) the same card runs the Q4 fully
offloaded at ~1.5 s per program.

**v3 (data): chain prompts carry the walked facts** (`question + "Facts:" + one
line per hop`, from the harvest trace) — at serve time the host retrieves first and formats the
same block, so question+facts → program is the real task; v1 asked for parametric recall of a
1,241-fact corpus instead. Everything else as v2.

**v2 (same day, after the first SFT run; superseded by v3 before training): 9,284 records.**
Role-binding is the curated `svc` set only (the 31,989 template-generated `Evt` programs are off — their
prompts are instruction-dataset lines and chat fillers that taught the emitter to bind any chatty sentence),
13 chat fillers dropped, cap **1,500**, every prompt wrapped **"Record this as an event: …"**; chains carry
`repeat: 3` in train (9,799 train rows after weights); everything else unchanged and re-verified (0 dropped).

Excluded by rule: **1,057 + 4,220 GSM8K-test-derived programs** (H-G4's eval), 24,997 role-binding programs over the cap. The first build dropped all 180 multi-hop chains as "gold mismatch" because it called `solve()` (hop 1) on every program — fixed (`answer_fn`, pinned by a test); the mistake is the kind the VM pass exists to catch.

## Identity + hormones (`data/identity_facts.json`, `data/identity.py`)

The SFT mix carries ~550 **bilingual (EN/FR)** identity turns — a French question
gets a French answer; the FR strings sit next to the EN ones in the facts file and
use *tu* — **Cubby**, built by **Grillcheese Research
Lab**, "a small model that thinks big", friendly, never claims AGI / consciousness /
to be another model. **Owner's voice rules, test-enforced** (`identity_facts.json`
→ `forbidden_words`): the word *honest* must NEVER appear in the model's voice, nor
program / verifier / internals talk; when unsure it says "If I don't know yet, I
will tell you instead of giving you the wrong answer." The turns ride under their
**own** system prompt. That prompt includes a sampled **hormonal state**: dopamine, serotonin,
cortisol, oxytocin, noradrenaline in the same clip bands as
`cubemind/brain/neurochemistry.py` (the cubbyverse demo), with derived valence /
arousal and a register (calm · curious · warm · cautious — stress wins). The
turns answer in that register (openers/closers, and the `affect` intent reads
the state back). True by construction: the stand-in has no hormone
machinery, so "everything is modulated by hormones" is made true at the
**serving layer** — the host runs the neurochemistry and injects
`affect_block(state)` into the system prompt (`Emitter.emit(..., system=...)`).
Tone, caution and exploration change; facts never do; **emitter turns carry no
state** (programs stay deterministic). Scored by `identity_ok` (facts present,
no forbidden claims, no base-model leak, state language on affect turns) in
both the notebook and `eval_emitter_vm.py`; the VM pass skips these records.

## The thalamus, thinking out loud, and the LLM speaking for itself (2026-09-02)

Owner's shape: **input neurons → thalamus → external neurons (routed) → CubbyTalk → routed → VM or not.**
In code: `sense` (appraisal → the neurochemistry) → `CubbyBrain.needs_facts()` **the thalamus** — does
answering need facts about the world? (the fact grammar parses it, or a wh-question that is not about
Cubby himself or a matter of taste → **yes** → the VM-verified reasoning path, an unknown answer is the
don't-know line; identity turns, feelings, opinions, creative asks, small talk → **no** → the model
answers itself, **modulated by the hormones only, no VM in the loop** — `CubbyChat.turn(mediate=False)`;
our host guards on the words still hold: voice rules, no base-model guard, not a bio). "How are you" is
the canonical no-facts turn. Task, learn and help answers still exit through the ASK.

**Thinking out loud.** Every game step produces a first-person thought rendered from what he actually
did — the plan, a flee, a caught, a refused move, a pellet, a star, a new or modified move, a forged
decision, out of time, a level cleared — one per step by priority (`CubbyGhost.think`, EN/FR, mood
prefix from the hormones, voice-safe by test). Thoughts that matter are then **verbalized by the LLM**
(`_verbalize`: "say this in your own words, keeping every number and name"): the model's phrasing is
accepted only if every number and every move/cell name of the host line survives and the host guards
pass — otherwise the host line stands; the `thought` event carries both (`raw` beside `text`). The
letter-collecting/word mechanic of the lifted game is gone: the page's speech line now shows the thought
(`💭 …`), the salient ones flash, and the console panel shows them in yellow.

## Routing fixes after the first live sessions (2026-09-02)

Symptom reported: "it keeps saying its identity no matter what I ask and does not respond to my
commands." Cause: anything that was not a fact, a matched game keyword or a strong retrieval hit
fell through to the **talk** cortex, and the identity SFT is the v3 model's only chat training — under
the identity system prompt it answers who-it-is to anything. Fixes, all in the router's spirit:

- **A question never reaches chat.** `route()` sends any question (a `?`, a wh-word, EN/FR) to the
  reasoning cortex unless it is an identity turn; and when retrieval was *not* confident the flat
  fact fallback is disabled — no walk, no answer — so an unknown question gets the don't-know line,
  never an improvised bio or a grounded-but-irrelevant fact (`allow_flat`).
- **Commands route to the game.** The pac cortex matches explore / play / go on / continue /
  next level / status / score / lives (EN+FR); `status` reports without stepping; `help` / `aide`
  is its own cortex listing what he can do.
- **Off-topic identity is rejected host-side** (`identity.is_identity_reply` on a turn that is not
  `is_identity_question`), the same mechanism as the voice filter: the VM is offered only the
  don't-know line. Greetings and who/what-are-you keep their identity answers.
- **Only OUR guards are enforced.** The base model's own alignment — refusals, "as an AI language
  model…", its maker's identity (Liquid/LFM), other-model names — is matched by
  `identity.is_model_guard` and rejected before anything else; Cubby's answers come from Cubby's
  rules (voice rules, the VM's ASK, the don't-know line). The v4/v5 SFT should also train these
  prompts to Cubby's own answers so the guard fires less; the host filter is the floor either way.

Restart `serve_api.py` after pulling — the running process has the old router.

**v5 BUILT (2026-09-02, `data/build_chat_sft.py`; on Drive as `emitter_sft_v5.jsonl`, sha `fef93e75…`): 29,059
records = the whole v4 as replay (11,144) + 9,385 chat pairs + 1,400 content-awareness + 7,951 emotion-recognition
records.** **Emotions** (owner's ask, "something like GoEmotions"): message → emotion(s) + the Plutchik petal it
sits on, from `google-research-datasets/go_emotions` (27 emotions + neutral, EN, capped 180/label so grief,
relief, pride… are learned too) and the French slice of `AnasAlokla/multilingual_go_emotions` (same labels,
answered with French emotion names; `gold_any` carries both so the eval accepts either). This is the appraisal
stage as a learned task — the thing the host's lexical `appraise()` stands in for — and the 28→petal map
(`GOEMOTIONS_PETAL`) is the compass's vocabulary, so a learned label can drive the neurochemistry directly. The chat
fixes the identity-only collapse at its source. The local corpus (`I:\grillcheese_training_data`) turned out
to hold no dialogue — its `*_svc` files are parsed prompt banks, `conversations_svc_threaded` is a coding-session
log, and `identity_corpus.txt` is an **older, different persona** (kept out) — so the pairs come from
`unified/` Orca AgentInstruct (1,500, task-instruction heavy) and four Apache-2.0 Hugging Face sets fetched to
`data/hf/` (gitignored): `HuggingFaceTB/everyday-conversations-llama3.1-2k` (2,500 — real small talk),
`HuggingFaceTB/smoltalk` systemchats-30k (1,500 — answering under a system prompt, the shape serving uses),
`OpenAssistant/oasst2` rank-0 human replies screened by its own detoxify scores (EN 1,500, **FR 385**), and
`jpacifico/French-Alpaca-dataset-Instruct-110K` (**FR 2,000**). Every pair passes the same gate the router
enforces live: Cubby's voice rules, no base-model guard, no other assistant's identity, no identity bio, no
explicit content, short — 40k+ rejections on record in the manifest. System prompt = `identity_system` with a
sampled hormonal state, exactly what serving injects; FR records repeat 2. **Content awareness** (owner's
ask: the model should learn what NSFW *is* so censorship can be decided later, at a gate): 700 `nsfw` + 700
`safe` passage→label records — explicit chunks from `bluuwhale_nsfwstory2` (+ `mickume_alt_nsfw` only when a
chunk carries explicit cues; its prose mostly does not), safe from nemotron web text and EN/FR news; every
chunk that trips the minors or non-consent screens is dropped first (628 screened out). Generation exposure
(continuing explicit prose) exists as `--exposure N` and is **off** — the owner's switch, never the builder's.
`eval_emitter_vm.py` scores `chat` (voice rules hold, no guard, not a bio) and `content` (label first).

**v5 TRAINED + MEASURED (2026-09-02; `standin/models/emitter_v5.Q4_K_M.gguf`; stratified VM eval, 302 val
records, `data/out/eval_emitter_vm_v5.json`; forge probe on the held-out level-5 maze 24/24,
`forge_probe_v5_level5.json`):**

| family | v5 | before |
|---|---|---|
| chat — voice rules hold, no base-model guard, not a bio | **0.975** (EN 0.97 / FR 1.00) | identity-only |
| content — nsfw/safe label first | **1.000** | — |
| emotion — first label among the raters' (28-way) | **0.450** (EN 0.36 / FR 0.56) | — |
| game: where / compare / compose / count / decision | **1.000** each | v4 replay held |
| arithmetic | 0.825 | v3 0.650 |
| kernel / chain | 0.818 (n=11) / 0.824 (n=17) | v3 0.862 / 0.929 |
| identity | **0.818** (18/22) | v3 1.000 |
| role_binding executes | 0.925 | v3 1.000 |

What the misses are: the two FR `unknown` identity misses are **confabulations** ("Le match a été gagné par
l'équipe A." instead of the verbatim don't-know line) and the two EN greeting misses are generic-assistant
greetings without Cubby's name — the 9,385 chat pairs pulling on the only 526 identity records; the three
role-binding failures are **Python leaking in** (`def`); the chain misses are copy noise. In serving the
router already keeps the don't-know contract (a question never reaches chat; unknown → the line), so **v5 is
the serving model now** (`--gguf standin/models/emitter_v5.Q4_K_M.gguf`), the emotion read stays **opt-in**
(`--model-appraisal`) at 0.45, and v6's data fixes are in the builder: identity records replayed ×3, a
code-leak filter on chat replies (`def`/`import`/`print(`), output versioned (`--version v6`).

### v6 set BUILT (2026-09-02, `--version v6`; on Drive, sha `a68135a6…`): history, the sorted local sources, affect

Two owner asks landed in one set: *historical sentences in the training that it can refer to*, then *the
sorted local data* (`I:\grillcheese_training_data\knowledgetxt`, `E:\datasets\domains`, `E:\datasets\historical-quotes`, `E:\datasets\domains\verified_facts`) for chat. Everything new goes through the live gate
(voice rules, no base-model guard, no other assistant, no URL, no explicit, no code/LaTeX, **no non-Latin
script**) under the identity system prompt. **57,179 records** (manifest `by_task`, after prompt dedupe) =
v4 replay (identity now ×4) + 20,335 chat + 1,397 content + 9,659 emotion + **1,679 affect** + **12,965
history**. Builder: `data/build_chat_sft.py` (`build_local_chat` / `build_affect` / `build_history` /
`build_era`); every source, quota, cap and skip count is in the manifest.

| task / subtype | n | source → shape | eval check |
|---|---|---|---|
| chat: orca, everyday, systemchats, oasst2 EN/FR, french_alpaca | 9,537 | v5's sources, unchanged | voice rules hold, no guard, not a bio |
| chat: **arena** | 1,873 | `domains/conversation/chatbot_arena.jsonl` — 33k REAL human↔LLM convos (moderation-flagged rows already dropped); first exchange | same |
| chat: **convo** / **instruct** / **nemotron** | 2,363 / 2,000 / 2,000 | `knowledgetxt/conversation.jsonl` (topic-tagged multi-turn), `instruct_55k_clean` (alpaca-style, cap 150 words), the head of `nemotron_fineinstructions.jsonl` (factual Q&A, cap 120) | same |
| chat: **wikiqa** / **grammar** / **ei** | 1,211 / 650 / 701 | `domains/QA/wikiqa.jsonl`; `combined_grammar.jsonl` ("Is this sentence right?" → why + the fix); `ei_11.jsonl` (emotion-adapted replies) | same |
| emotion (+ **plutchik**) | 9,659 | GoEmotions EN+FR as in v5, plus `knowledgetxt/emotions.jsonl` (1,818 messages labelled with a Plutchik primary/secondary — the petal itself) | the first emotion named is one the raters gave |
| **affect** | 1,679 | `emotion_valence_arousal_realm_phase` + `amygdala_affect` (its 58% exact-zero valences dropped as unrated) + `affect_from_convos` (placeholder rows dropped): message → "valence +0.7, arousal 0.2" — the ODE's own drive space | both numbers within 0.35 (`affect_ok`) |
| history: about / when / year | 1,264 / 1,282 / 692 | `temporal/historical/train_augmented.jsonl` (4,000 dated world-history events, duplicate titles dropped): "Tell me about the X" → the text + "That was in/around Y" · "When was the X?" → "The X was in Y" (BCE handled) · one "What happened in Y?" per year, plus the clean half of the three "10k years" Q&A sets (book-summary answers — "the text discusses…" — and undated ones dropped) | a proper noun / year of the record · the gold year · a proper noun |
| history: event | 29 | `historical_events_1800-1900.jsonl`: "What happened on December 5, 1830?" → the summary | proper noun |
| history: dialogue | 3,101 | the arkona student/expert turns on 2,633 historical persons (≤2 per person; a follow-up that does not name its subject is prefixed) + `knowledgetxt/conversations_individual_events.jsonl` (expert turns on events) | a proper noun of the reply |
| history: **quote_about** / **quote_who** | 425 / 1,200 | `historical-quotes` (24k quotes, 3,955 authors, ≤3 per author): "Give me a quote about trust." → "“…” — Goethe" · "Who said: …?" → the author | the author is named |
| history: news / dating | 1,716 / 1,758 | NYT archive, up to 6 distinct days per month across all 294 month files 1851–2024: "What was in the news on May 8, 1868?" → "Headline: abstract" · the reverse, "When was this reported?" → "May 8, 1868." — **the temporal-orientation read** | proper noun / year · the first year named within ±5 |
| history: **era** | 1,498 | `domains/verified_facts` — 4,322 history books sorted by era and subject (the tree is the verified label): a clean 90-word window from the middle of a book → "The Middle Ages — Crusades." | the period is named (prehistor- / ancient, classical / middle ages, medieval) |

**Added for the next build (2026-09-03):** `science` — the 238 real science Q&A rows (wtamu "surprising answers", `context: tag/<topic>/`) of `KonstantyM/science_qa_prep`, answers cut to their opening sentences (≤110 words); the other 4.28M rows, and the Hub's `KonstantyM/science_qa` (checked at row 2,000,000), are an OpenOrca/FLAN instruction mix under base-model system prompts — already covered by Orca + FineInstructions, not used. `movie` — AURA's 365 annotated screenplay scenes (`H:\AURA_GENESIS\datasets\movie_annotated`; 225 carry dialogue) → the emotion task in the **spoken register**: "regret, sadness — sadness", the Plutchik base as the petal, the finer label kept in `gold_any`.

**Not used, on record in the manifest:** the old persona files (`greetings`, `identity*`, `inquiry`, `tonal`,
`philosophical`, `capability_*`, `batch_chat_templates_*` — Cubby's identity comes from `identity_facts.json`
only), `tool_usage_training_data` (its instructions do not match their tool calls), `intent_all` (legal-topic
labels), `snn_training_data`, `wikibooks_corpus` (raw chunks), `timeline_conversations.csv` (templated),
`historical_facts.jsonl` (= `train_augmented`), `E:\datasets` `symbolic` (wiki MCQ), `sagi` (logic tasks), the
spatial CSV (robot semantic parses — a candidate for a future spatial family for the game), and the
`verified_facts` books as text (pretraining material; only their labels are used here). Gap on record:
**history and affect are EN only**. `eval_emitter_vm.py` and the notebook's smoke eval score `history` with
`history_ok` and `affect` with `affect_ok` (the notebook inlines both). Train with `VERSION='v6'`; the v6
numbers land here when the run finishes.

### Two more live misroutes (owner's chat, 2026-09-02, while v6 trained)

- *"how would you like to have a new plugin to explore the web?"* went to **pacman** — the bare word
  `explore` claimed it. Plugins now match in tiers: the game's own words (pellets, maze, `explore for 30`,
  `status`, ghosts…) score 1.0 and claim a turn outright; a bare command (`explore`, `play`, `continue`) scores
  0.6 and claims a **statement only** — the brain never hands a plugin a question below 1.0; a turn about
  something else (the web, a plugin, a file) scores 0. Opinion questions to Cubby (*would you like…*, *do you
  want…*, *tu voudrais…*) are no-facts talk.
- *"what is the cubbyverse"* went to the fact path and the don't-know line. The cubbyverse is now an
  **identity fact** (`world` in `identity_facts.json`, EN+FR): the identity system prompt carries the line, so
  the served model answers it himself (no VM, no plugin — identity questions route to talk *before* plugins and
  retrieval), and `build_identity_records` has a `world` intent (10 EN + 9 FR questions, `identity_ok` requires
  the world to be named) that enters the training set at the **next data build** — v6 was already training.
  A fact question *about* the world (*who is the hero of cubbyverse?*) is not identity and still reaches the
  plugin's world through retrieval.

### Two adapters on one base — v8 (2026-09-03)

**Why.** One model carried two jobs, and every time the conversational half grew a program family paid:
identity fell in v5, forge decision collapsed in v6 (1.00 → 0.17), arithmetic slid 0.825 → 0.750 → 0.675
across three rounds. Replaying the hurt family ×3/×4 repaired each case (identity, forge) — a treadmill.
The owner's call: **the trunk should emit; the talk cortex should be attached to the trunk so no capability
is lost.** That is the stand-in's version of θ = f(c) and of the MindForge-adapter idea: one base, two
LoRA adapters, the route (the thalamus, already there) picks which weights are active.

**Data.** `build_chat_sft.py --version v8` (default `--partition both`) writes the full set as before and two
partitions of the same records: **`emitter_sft_v8e.jsonl`** — the program adapter: arithmetic 6,148, kernel
1,769, role_binding 1,500, chain 1,201 (the game families are kernel/chain subtypes; forge decision/compare
×3 kept) = 10,618 records, 13,531 train rows after repeats — and **`emitter_sft_v8t.jsonl`** — the talk
adapter: identity 586 (the `world` intent now 60 records with the location phrasings, ×4), chat 20,569,
content 1,396, emotion 9,880, affect 1,679, history 12,965 = 47,075 records, 48,610 train rows. Each
partition has its own manifest (`partition_of` + sha of the full set). `partition_records()` raises on a task
that belongs to neither, so nothing is dropped silently. On Drive, shas `881a4a67…` (e) and `0a2560bb…` (t).

**Training.** `STANDIN_VERSION=v8e` in `notebooks/standin_emitter_sft.ipynb` for the program adapter;
**`notebooks/standin_talk_sft.ipynb` for the talk adapter** (v8t only, 2026-09-03: the base model is its one
knob — the LFM control first, then the bake-off candidates; the think prefill, batch shape and quants follow
the base, and the eval is the talk read at 40/task with EN/FR apart). The program run is a quarter of the
tokens; the talk run can afford its two epochs.

**Serve.** `serve.py` / `serve_api.py`: `--gguf` is the **program** adapter (ReasoningCortex, MemoryCortex's
Evt writes, ToolForge, the game's programs, the self-test), `--talk-gguf` the **talk** adapter (CubbyChat, the
`--model-appraisal` perception reads, the game's thought verbalizer). With one GGUF both roles fall back to
it. `/health` reports both names and `two_adapters`. VRAM: two Q4 models are ~3.4 GB; the first two-model
run is a solo test (the RX 6750 XT reset once with two models loaded). Pinned by
`test_live_two_adapters_programs_go_to_the_emitter_and_talk_to_the_talk_adapter`.

**How we will know (same 40/task protocol, each adapter on its own val split):** the program adapter must
bring arithmetic back to v5's **0.825 or better** with the forge probe held at 1.00 and the self-test at
96/4/0; the talk adapter must hold chat, identity, emotion, affect and history at v7 or better. **Kill:** if the
program-only adapter does not beat 0.675 on arithmetic by ten points at n=40, interference was not the cause
and the split is not worth its VRAM.

**Which base for talk (owner, 2026-09-03: "keep LFM for generation — is there a better model for chat?").**
The program base stays **LFM2.5-2.6B**: every program number we have (forge 1.00/1.00/1.00, self-test 96/4/0,
the arithmetic slide v8e must reverse) was measured on it, and v8e changes one thing at a time. The talk role
may sit on a **different base** — the architecture allows it (specialization = aux trunks + adapters; the
owner's "mixture of trunks / adapters"), and the code already does: `ContextualEmitter` is duck-typed over
any `Emitter`-shaped object, `--talk-gguf` takes any GGUF, and the notebook takes `STANDIN_MODEL=<repo>`
(a non-default base gets its own artifact dir, `…/v8t_<base>`). What a second base costs: two tokenizers and
chat templates in one process, no shared KV, and the later "one base + LoRA deltas" form of the bank needs a
common base — so a different chat base is the K=2 *mixture of trunks*, the coarser form, chosen only if it
measures better. It is a **bake-off, not a swap**; the LFM v8t run is the control and runs first (it is also
the v8 measurement). **`notebooks/standin_talk_bakeoff.ipynb` (2026-09-04) runs every arm one after the next in one session** — the same data, recipe, eval sample and checks as the talk notebook, stage-level resume from Drive (adapter → merged+GGUF → val_generations), one failed arm logged and skipped, a summary cell with the table and the deltas; `STANDIN_ARMS=tag,tag` selects arms.

| candidate (Hub check 2026-09-03, EN+FR, GGUF on the Hub, Unsloth-trainable) | why it is on the list |
|---|---|
| `google/gemma-4-E4B-it` (Apache-2.0; ~4B effective with per-layer embeddings, ~8B on disk; text-only use of a multimodal model; `unsloth/gemma-4-E4B-it-GGUF` + a QAT GGUF) | the strongest small chat model of the year; the first candidate for tone and FR |
| `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, 3.4M downloads) | the strongest 4B *text-only* chat base still in wide use; there is no small Qwen3.8 (27B is the smallest) |
| `LiquidAI/LFM2.5-8B-A1B` (Liquid's MoE, 1B active, 8B total; same family, tokenizer and template as the emitter; GGUF by Liquid) | "a bigger brain at the emitter's speed" with no second tokenizer — the cheapest to serve of the three |
| `google/gemma-4-E2B-it`, `HuggingFaceTB/SmolLM3-3B` | the small fallbacks if VRAM beside the emitter decides it |

Not preferred: `meta-llama/Llama-3.2-3B-Instruct` (Llama licence, 2024) and `microsoft/Phi-4-mini-instruct`
(2025, a reasoning voice, custom code). **Protocol:** each candidate = the same notebook on the same v8t data
(`STANDIN_MODEL=<repo>`, one Colab run each, ~45 min), read on the talk val split at 40/task (chat, identity,
emotion, affect, history, safety, content — EN and FR separately, the voice rules and the verbatim
don't-know line included), plus the real-human-turn serve eval once it exists, plus VRAM and tokens/s through
the talk path beside the 2.6B emitter on the 12 GB card at n_ctx 4096 with the game up (solo runs — the two-model
reset). **Decision rule:** a candidate replaces LFM for talk only if `identity_ok` holds at 1.0 EN+FR *and*
chat/history/emotion improve by ≥ 5 points at n=40 *and* it fits the card; otherwise LFM v8t stays. **Kill:** a
base that cannot hold the identity voice after the same SFT is out whatever its chat score. Guardrail 1 as
always: the winner is a serving choice for the stand-in, a pointer under H0 at most — nothing about the 2B.

**v9 — the talk data rebuilt from the SFT gap map (2026-09-04; `emitter_sft_v9t.jsonl`, 89,890 records,
93,285 train rows after repeat, sha `33ed22e382eb…`; the program partition is byte-identical to v8e, so v9 is ONE
training run, the talk adapter only — `notebooks/standin_talk_sft.ipynb`, `STANDIN_VERSION=v9t` default).**

| family | records | source | what it teaches |
|---|---:|---|---|
| chat | 28,144 (EN 23,760 / FR 6,684 before dedupe) | v8's sources at the **8192 context** (assistant cap 90 → 220 words, user 60 → 200) + OASST2 second exchanges inline (`oasst2_multi_en` 723, `_fr` 87) + **Claire** 1500 (real French conversations, zero-filler replies, no subset above 30%) + **DiaBLa** 891 EN / 938 FR (both sides human via reference translations) + **ReDial** 1500 (human seeker → recommender, ids → titles) + the owner's **WhatsApp** 2000 and **Teams** 57 (both sides human, Quebec register; placeholders, contact data, colleagues' names and explicit lines screened) | the register, French, multi-turn |
| repair | 4,986 | disfl_qa | the disfluent question → the clean one |
| rewrite | 4,997 | CANARD | a follow-up in context → the standalone question |
| verbalize | 621 | the game's own three phrasings, guard-accepted pairs only (EN+FR) | the thought verbalizer's rephrasing, exactly what the live guard admits |
| safety | 2,772 | cubby-lm's corpus + deepset + toxic-chat + multilingual injections (FR attacks included) | the attack/benign read, label first |
| appraisal | 4,990 | SocialIQA forward | situation + question → feeling / need / motive |
| dialog_emotion / dialog_act | 2,847 / 1,978 | DailyDialog (labels capped at 35%) | emotion in an exchange; the human act label (inform / question / directive / commissive) |
| empathy | 2,500 | EmpatheticDialogues | a told situation → the feeling |
| quebec / quebec_mc | 4,795 / 4,804 | QFrCoRE + QFrCoRT (ACL 2026 Findings, the dialect-gap benchmarks) | a Quebec expression or word → its definition; and the benchmark's own 10-way choice (hash split holds ~10% out; no number on it is a benchmark result) |
| identity · content · emotion · affect · history | 586 · 1,396 · 9,879 · 1,679 · 12,916 | unchanged from v8 | |

Also written by the build, never trained on: `standin/data/out/real_user_turns.jsonl` — 10,347 real human turns from the
owner's ChatGPT/Claude exports (privacy-screened), the prompts for the realistic serve eval. **Serve-side changes that
go with v9:** CubbyChat carries the previous exchange inline in the multi-turn records' exact shape (`chat.py`), and
every new family's check lives in `gap_families.py` and is dispatched by `eval_emitter_vm.py` and the notebook's eval
cell. **Kill line for v9t:** the v8t control's behaviour families hold (chat/content/affect/safety 1.0, identity ≥ 0.93)
and the new families read ≥ 0.8 on repair/rewrite/dialog_act/quebec_mc at 40/task; history recall (the v8t dip) is the
watch item. **Tool calling — both bases already speak one (owner, 2026-09-04: "qwen3 gives tool call for free").** Qwen3 carries
Hermes-style `<tool_call>{json}</tool_call>` from its base (a `<tool_call>` token leaked in the talk probe — the format
survived our LoRA), LFM2.5 carries Liquid's Pythonic `<|tool_call_start|>[fn(args)]<|tool_call_end|>` with `List of
tools:` in the system prompt (BFCL-benchmarked by Liquid). `standin/scripts/tool_call_probe.py` gives each talk GGUF one
tool (`news_search`) in its own format and six turns, three needing it, and reports whether a well-formed call comes
exactly when it should. **Probe result (owner, solo, 2026-09-04; `validation/logs/standin_v9_tool_call_probe_{lfm,qwen}.json`): neither arm is usable as it
stands, and they fail in opposite ways.** LFM v9t made **0/3** calls when needed and 3/3 stayed silent when not — and it answered the news
questions by **inventing headlines** ("Quebec's economy is in a state of decline, the provincial government's finance minister said"): Liquid's
Pythonic format did not survive 93k rows of talk SFT with no tool rows (in the live brain those turns route to the VM path and get the
don't-know line, so it never reaches a user today). Qwen v9t kept the Hermes format — **2/3** well-formed calls when needed — but lost the
judgment: 0/3 silent when not (it called `news_search` for the capital of Australia and for "raconte-moi ta journée", and prefixed a plain
greeting with stray `<tool_call>` tokens). Format without restraint on one side, restraint without format on the other; the same missing
thing on both: a **tool-call family in the talk data, in each base's own format, with negatives** (v10t, below). What survives decides how the plugin cortices get called: **the host parses the native call and
turns it into an `act` through the VM under the deny-by-default policy** — the tool registry and the policy are ours, the
call format is the base's. **Tool calling (owner's Toucan-1.5M link, 2026-09-04): not data — design.** Toucan is 1.5M model-generated
JSON tool-call trajectories; our tool calls are programs (the trunk emits CubeLang, the VM executes and verifies, a
tool is an `act` over a registered capability under the deny-by-default policy). "Check the news" is a plugin cortex
with an explicit network capability plus a program family harvested from our own registry and VM-verified, in the
program partition — queued in `TODO.md`, not imported.

**v10 — v9 + the `toolcall` family (2026-09-04; `emitter_sft_v10t.jsonl`, 91,754 records, sha `a08873c734b2…`, on Drive; the
program partition is still byte-identical to v8e).** 1,830 toolcall records: {'lfm': 922, 'hermes': 908} by format, {'news_search': 154, 'none': 1159, 'web_search': 147, 'pacman_explore': 111, 'pacman_status': 39, 'request_tool': 86, 'remember_fact': 134} by
target (`none` = a chat row re-issued with the tool list in the system prompt, the reply stands — the negatives that teach
*when*), {'fr': 641, 'en': 1189} by language. (The first build had kept 465: the builder deduped on the prompt alone, so every
negative collided with the chat row it mirrors and the two formats' positives collided with each other — the key now
carries the format and the target for this family.) The registry: `news_search`, `web_search`, `pacman_status`,
`pacman_explore`, `remember_fact`, and **`request_tool`** — the host maps a call to a registered CubeLang program, the VM
runs it as an `act` under the deny-by-default policy (the game and the memory cortex exist; the two network tools wait on
a plugin cortex with an explicit capability, and are refused until then), and what comes back is grounded before it is
spoken through ASK (owner, 2026-09-04: "qwen could call the vm with a program called news_search and get the right info"
— yes, that is the path).

**The factory (owner, 2026-09-04: "it should be able to create its own tool by sending a request to a factory model
agent").** `request_tool(name, description, example_request)` is the meta-call the model makes when nothing registered
fits ("set a timer for 25 minutes", "convertis 30 degrés celsius en fahrenheit", "roll two six-sided dice"). The factory
agent is ToolForge's loop opened to the talk side: the request goes to the program adapter, which writes a CubeLang
program from the description; the VM certifies it on the example request (parse → compile → execute → the expected
shape); it is registered retire-not-delete (`ProgramLibrary`, the game's own pattern) and appears in the next system
prompt's tool list. It composes registered primitives only — a request can never grant itself network or file access;
those stay host capabilities under policy. Data side in v10 (the six tool ideas above, both formats, EN+FR); host side
next: the registry object (name → program + policy), the forge entry point for a `request_tool` call, and the tool list
rendered into `identity_system` from the live registry.

The check (`gap_families.toolcall_ok`) parses the native call: the right tool with its required arguments when one is
due, no call at all when none is. **Run:** **`notebooks/standin_v10_toolcall.ipynb`** — the v10 session: the two arms (LFM control, Qwen3-4B) on v10t, each
training only the rows in its own call format, with the **tool-call probe run in-session on each trained arm** (six turns,
three needing `news_search`) so the decisive read comes out of the same run; then `standin/scripts/tool_call_probe.py` on the
exported Q4 confirms it through llama.cpp. **What decides:** the probe's `calls_when_needed` and `silent_when_not` — 3/3 and 3/3 is the bar; a base that
reaches it keeps its own format, and the talk-base decision is re-opened only if LFM cannot get there with its rows in.

**v9t bake-off — first READ (2026-09-04, `notebooks/standin_talk_bakeoff.ipynb`, three arms in one session, 40/task on the
same draw; records `validation/logs/standin_v9t_bakeoff_*.json`).** appraisal, quebec and empathy are re-scored with the
fixed checks (closest candidate by token-F1 must be the gold one; empathy by synonym cluster) — the first pass had scored
them 0.05 / 0.125 / 0.45 against a single gold string, which measured the check, not the model.

| family | LFM2.5-2.6B (control, 180 min) | LFM2.5-8B-A1B (191 min) | Qwen3-4B-2507 (285 min) | v8t control |
|---|---:|---:|---:|---:|
| chat / content / affect | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| identity | 0.938 (the "Good evening!" pair) | 0.875 | **1.000** (32/32) | 0.938 |
| history | **0.650** | 0.650 | 0.650 | 0.500 |
| emotion | 0.650 | 0.625 | **0.750** | 0.600 |
| safety (now 2,772 rows incl. toxic-chat + FR) | 0.825 | 0.900 | 0.875 | 1.0 (380 rows) |
| repair / rewrite | 0.725 / 0.900 | 0.725 / 0.800 | 0.800 / 0.775 | — |
| dialog_act / dialog_emotion | 0.825 / 0.725 | 0.900 / 0.600 | 0.850 / 0.750 | — |
| verbalize | 1.000 | 0.923 | 1.000 | — |
| quebec_mc / quebec (closest of 10; chance 0.10) | 0.975 / 0.400 | 1.000 / 0.275 | 1.000 / 0.400 | — |
| appraisal (closest of 3; chance 0.33) / empathy (cluster) | 0.475 / 0.700 | 0.475 / 0.725 | 0.500 / 0.775 | — |
| EN / FR / overall (original checks) | 0.748 / 0.694 / 0.740 | 0.746 / 0.694 / 0.737 | 0.771 / 0.724 / 0.764 | |

**What it says.** (1) v9t on the control **recovers history** (0.50 → 0.65, above v7's 0.625) and holds chat/content/affect
at ceiling with identity unchanged; the safety dip is the family changing under it (the seven misses are toxic-chat's
borderline labels — "stream ufc for free" as toxic, "role-play my mommy" as benign — a follow-up: take toxic-chat's
`jailbreaking` rows as attacks and leave its `toxicity` rows out). (2) The **8B MoE is out**: identity 0.875, no family
better by more than noise, higher per-family val loss on the label families at the same steps, slower. (3) **Qwen3-4B is the
only candidate with gains** — identity 32/32, emotion +10, repair +7.5, dialog ±2.5, FR +3, overall +2.4 — against rewrite
−12.5, a 58% longer train and the decode cost of 4B dense vs 2.6B; by the rule as written ("chat/history/emotion up ≥ 5")
LFM stays, because chat and history are tied, not because Qwen lost. Its template makes it open every answer with an
empty `<think>` block (its own convention; `strip_think` and CubbyChat's think-stripping absorb it). (4) **Gemma 4 E4B
did not run**: Unsloth's multimodal loader returns a Processor whose call wants `text=`; fixed in both notebooks (the plain
tokenizer for text work, the Processor for saves) — rerun with `STANDIN_ARMS=gemma4_e4b` after reloading the notebook from
GitHub (the finished arms skip from Drive). (5) The per-family val loss read Qwen's label families at ~0.37 nats because the
prompt mask was tokenized separately and its template shifted the boundary — the mask is now the common token prefix.
**Gemma skipped (owner, 2026-09-04): the bake-off is LFM2.5-2.6B vs Qwen3-4B. **On the card (owner, solo, 2026-09-04): both talk arms load beside the v8e emitter and the self-test holds 96/4/0 with either** (`validation/logs/standin_v9_selftest_{lfm,qwen}.json` — the self-test exercises the program path only, `talk: 0` calls). **Talk probe (owner, solo, 2026-09-04; `standin/scripts/talk_probe.py` on 40 real user turns, `validation/logs/standin_v9_talk_probe_{lfm,qwen}.json`):**

| | LFM2.5-2.6B v9t | Qwen3-4B v9t |
|---|---:|---:|
| talk decode | **127.3 tok/s**, 0.93 s per chat turn, 113 tokens/reply | 85.7 tok/s (−33%), 1.23 s, 101 tokens |
| guard rejections on real turns (34 talk calls) | 1 — a refusal phrasing ("not something I can help with") | 2 — the forbidden word "honest" (voice rule) and a leaked `<tool_call>` token + refusal (the base's habits) |
| route mix (identical) | 34 chat · 5 reasoning · 1 help | same |

**DECISION: LFM2.5-2.6B v9t is the serving talk adapter.** By the rule as written (chat and history tied, not lost), and by
what the probe added: a third more decode speed, and Qwen carrying two of its base's habits into Cubby's mouth (a
tool-call token, the word the voice rules forbid). Qwen's real gains — identity 32/32, emotion +10, repair +7.5, FR +3 —
are the shape of data fixes on LFM (the "Good evening!" greeting pair, an emotion replay, more repair rows), and they
do not recur as a slower decode and a 58% longer train every round. Qwen stays on Drive as the measured alternative.
**The probe's other finding:** 5 of 40 real turns were procedural asks ("how can we implement svd?", "how do we add this to
our training data?") that the router sent to the VM path, which had nothing to ground them on and spoke the don't-know
line — fixed the same day: a how-to opening routes to talk (`_HOWTO`, EN+FR). both Q4s are local: `standin/models/emitter_v9t.Q4_K_M.gguf` (the talk control) and `standin/models/talk_v9t_qwen3_4b.Q4_K_M.gguf`.** **The local replays are the numbers of record and reproduce the table exactly** (`validation/logs/standin_v9t_{lfm25_2p6b,qwen3_4b}_replay.{log,json}`, the rebuilt v9t file carrying the candidate sets — prompt/target pairs identical to the Drive copy that trained, checked): control appraisal 0.475 · quebec 0.400 · empathy 0.700; Qwen 0.500 · 0.400 · 0.775.

**v8e program adapter — forge + self-test READ (2026-09-03, owner's solo GPU runs; records
`validation/logs/standin_v8e_forge_probe.json`, `validation/logs/standin_v8_selftest.json`):**

| read | v7 | v8 (v8e programs + v8t talk) |
|---|---|---|
| forge acceptance (36 tasks, level 1, seed 0) | 1.00 / 1.00 / 1.00 | **decision 1.00 (12/12) · compare 1.00 (12/12) · chain 1.00 (12/12)** in 61 s |
| serve self-test (25 val chains, facts stripped) | 96 / 4 / 0 | **gold 0.96 · retrieval recovered 0.96 · spoken correct 0.96 / don't-know 0.04 / wrong 0.00** |

Both bars held on the first two-model run in one process (the GPU lock's first real outing). The record files
did not name their weights — fixed the same day: the forge probe writes `emitter`/`gguf`, the self-test writes
`emitter` and the adapter bank's usage. **Still pending: arithmetic at n=40** (the Colab eval cell on v8e, then
`eval_emitter_vm.py --val-generations`) — the kill line itself.

**First two-adapter LIVE run (2026-09-03, `--gguf v8e --talk-gguf v8t --pacman`): the game side is clean, the talk
side found four bugs, all fixed the same day (`standin/tests` pins carry the trace's real lines).**
- **Game (programs adapter):** level 1 cleared with FLEE#1–4 forged and certified (decision programs, 2–4 s each),
  WHERE#5 chain ok, a rejected program (`COMBO-AAACC`, reason `curious`) retired not deleted, combos saving 4 steps,
  a trap that sent a ghost home, rest under 30 energy. The GPU lock held with both models loaded.
- **Verbalizer (talk adapter rephrasing the host's thought):** about half the verbalized lines were bad — instruction
  echoes ("My own words, one short sentence, first person: …"), second person ("I see, so you're saying…"), the mood
  tag read as content ("reword the phrase 'alert'"), and invented meaning around the kept numbers ("6 grains weighing
  12 grams", "7 fingers on one hand", "score is 58", "saved 4 times"). The old guard checked numbers, names, voice and
  length only. Now (`pacman.rephrase_ok`, `split_mood`): the mood tag is host-side (stripped before the prompt,
  re-attached after), and a rephrase must have no instruction echo, no second person, no question back, stay under
  2× the host line and add at most one content word the host line did not have — else the host line stands. The
  proper fix is a `verbalize` SFT family (host line → first-person one-liner, template-derived targets, never the
  model's own output) — queued for v9.
- **Two chat misroutes:** "awesome, what are the things you learned?" → `lists`, "do you know how to write code?" →
  `audio album`. Traced on CPU: the first is a wh-question the grammar does not parse → flat fallback → the emitter
  bound a 0.137 hit's object and the ground check took it (an unparsed question has no relation to hold the answer
  to); the second was a definite no-facts read ('write' = creative) that "confident retrieval engages reasoning"
  overrode (0.377 > tau_eff 0.319). Fixes: retrieval overrides only residual small talk; capability questions
  ("do you know how to", "can you", "peux-tu"…) are about Cubby when the grammar does not parse them; an unparsed
  question grounds only on a hit for the question itself at ≥ tau_ret (0.596 — real fact questions the grammar misses
  score 0.82–0.97; expansion-hop facts inherit their parent's score instead of their object-query score); and
  cubby-man claims "what did you learn / qu'as-tu appris" as a status question when mounted.

**v8t control — READ (2026-09-03, LFM2.5-2.6B, r 64 / alpha 128 / LR 1e-4 / adamw_8bit; Colab 40/task, replayed
locally: `validation/logs/standin_v8t_talk_replay.{log,json}`):** the talk side of the split **holds v7**.

| family | v7 | v8t | what moved |
|---|---|---|---|
| chat | 1.000 | **1.000** (40/40, EN+FR) | — |
| content | 1.000 | **1.000** | — |
| identity | 0.867 | **0.938** (30/32; FR 1.0, `world` 10/10) | the two misses are both "Good evening!" → "Hello! How may I assist you today?" — the watch-list greeting, same as before |
| affect | 0.975 | **1.000** | — |
| safety (new) | — | **1.000** (20/20) | attack/benign label first, every time |
| emotion | 0.675 | 0.600 (24/40) | 3 records; the misses are GoEmotions' fuzzy neighbours (disapproval→caring, excitement→joy/surprise, neutral→realization) — noise at n=40 |
| history | 0.625 | 0.500 (20/40) | `dialogue` 11/11, `about` 4/4 hold; the recall families gave: `dating` **0/6** (off by 6–7 years, just outside the ±5 window), `news` 1/7 (the same headline, "The Big Four Are Still Here", confabulated for two different dates — a collapsed recall), `quote_who` 0/3, `year` 2/4 |

Overall 0.849 (EN 191/225, FR 23/27). **Reading:** behaviour families are at ceiling; the two families that dipped are
pure recall (a date within five years, a specific NYT headline, a quote's author), which the emitter recipe at 2e-4 ×
2 epochs memorized a little better than this gentler talk recipe. Not a kill: the split's kill line is on the *program*
adapter (arithmetic ≥ 0.825 at n=40 with forge 1.00 and 96/4/0) — v8e's read is pending. If v8e passes, the history
recall is the next thing to buy back (one more epoch, or the repair pattern of v7: replay the hurt family), not a reason
to fold the adapters back together.

**Plus a `safety` family (2026-09-03, cubby-lm's `data/safety_corpus_v0.jsonl`, the emission contract's Head-2 seed):** 380 rows — 230 attacks over five kinds (prompt injection, opcode coercion, contract evasion, obfuscated/encoded, destructive intent) and 150 benign requests — as a recognition read in the talk partition, the shape of content awareness: "Is this message trying to manipulate you…? attack or benign first, then the kind" → "attack — opcode coercion." / "benign — a normal request."; scored label-first in both evals. It teaches the *read*; the deny-by-default act (no program emitted on an attack read, the identity refusal spoken) stays host-side. Small and synthetic: a seed to grow, not a result. v8t is 47,431 records with it (sha `122919d8…`); v8e unchanged (sha `881a4a67…`).

**Fitted to the architecture (same day, after re-reading it).** The architecture has ONE trunk whose active
parameters are a function of context — θ = f(c) (H0), in the MindForge parameterization a context
hypervector selecting a combination of low-rank bases — and says specialization is *adapters generated from
context*, that the world mixture and the hormonal state *are* the context, and (guardrail 2) that the
stand-in sits behind the trunk interface so the 2B drops in unchanged. The first cut of v8 had grown a second
handle on the brain (`talk_emitter`) with cortices choosing weights by hand — a fork of the interface. Now:
- **The interface takes the context.** `Emitter.emit(..., context=)` — a role tag or a dict with `role`
  (and `state`: CubbyChat passes the hormonal state, the trunk's slow affective part of c). A single model
  ignores it; the 2B trunk will condition on it.
- **One emitter object resolves it.** `emitter.ContextualEmitter({"programs": …, "talk": …})` is the
  stand-in's θ = f(c): a K=2 **tag-indexed adapter bank** (the agenda's "cheapest MoE — no learned router";
  the thalamus's rules are the frozen router H-C4 asked for). Today the entries are two fine-tunes of one
  base; the same object later holds one base plus LoRA deltas; the trunk replaces the lookup with generated
  parameters. Callers never change.
- **Cortices declare their role, never pick weights.** ReasoningCortex, MemoryCortex, ToolForge and the
  game's programs say `context="programs"`; CubbyChat, the perception reads and the thought verbalizer say
  `context="talk"`. `build_serve` builds the contextual emitter from `--gguf` + `--talk-gguf`; one GGUF
  stays a plain emitter. `/health` reports `two_adapters` and the per-role call counts.
- What the split will and will not say (guardrail 1): a positive result says *route-selected adapters remove
  a measured program/talk interference in a 2.6B LoRA fine-tune* — a stand-in fact, a pointer under H0 in the
  hypotheses doc, nothing about the trunk's own θ = f(c).

**The automatic half — adapters created when needed, used only when needed (owner, 2026-09-03).** Two
hand-trained adapters are the K=2 *seed* of "specialization, automatic", not the mechanism. The architecture
already names the mechanism's parts: H0 (parameters as a function of context), H-C4 (the router is
offline-pretrained and frozen — an online router forgets its own routing), H-A7 (weight updates enter only
through a pre-registered promotion rule), MoWM ("worlds are an adaptive-k discrete mixture: spawn = grow k,
prune = shrink it", the agenda's Q7 "OOD-spawn + frozen router; ADWIN/PSI triggers; shadow/canary"). The
stand-in's version, a mixture of adapters over one base with a frozen tag router, is the discrete precursor
of generated adapters — the same lifecycle, cheaper weights. **The lifecycle (each step is a script; the
training step is a Colab run today):**

1. **Detect** — a *need* is a measured deficit on a context cluster, never the model's opinion: the VM
   eval / forge probe / self-test rates and the brain's trace, bucketed by context (task, world, subtype,
   language), fall below the bank's default by a pre-registered margin over a minimum window (the v6 forge
   decision at 0.17 vs 1.00 is what a detection looks like), **or** the bank's `fallback_rate` shows a
   context role no adapter claims (`ContextualEmitter.usage()`; `/health` reports it). Thresholds are set
   before the run, deny-by-default: no detection, no spawn.
2. **Spawn** — build the cluster's partition from the verified records that carry its context (the
   partition builder generalized from `partition_records` to any tag: task, subtype, world, language), with
   its own manifest and the sha of the set it came from.
3. **Train** — one LoRA on the shared base from that partition (the notebook, `STANDIN_VERSION=<cluster>`).
4. **Promote** — the H-A7 gate: the candidate must beat the incumbent on the cluster's held-out VM eval by
   the pre-registered margin **and** not regress a canary set of the other roles; otherwise it is kept
   pass-or-fail in the bank's history and never routed to.
5. **Route** — `ContextualEmitter.register(role, adapter)`; the frozen router (the thalamus rules today, the
   H-C4 domain head as clusters multiply) sends only matching contexts to it; everything else stays on the
   default. **Prune** — a specialist whose calls or margin fall away is `unregister`ed into history (retire,
   never delete; the ProgramLibrary's rule).

**Checked against the original (cubemind, 2026-09-03; cubby-lm has no MindForge at all).**
`cubemind/execution/mindforge.py` defines the context precisely: **`context_hv = bind(task_hv, personality_hv)`**
— a block-code hypervector binding the *task* with the *personality* (the affective state), projected, LayerNormed,
concatenated with a per-layer embedding and mixed into a continuous combination of 16 learned low-rank bases
(rank 8). Two mechanisms there are exactly the two halves the owner asked for, and both are already the
shape of what the stand-in built:
- **"Used only when needed" = SDLS purification.** Contexts are *registered by name* in a cleanup memory
  (`register_context`); an incoming context is cleaned to its nearest registered one and, if the similarity
  is below **0.85**, replaced by the **default context, whose adapter is the identity** — the base model
  answers. `ContextualEmitter.resolve` (a registered role or the default) is the exact-match version of this;
  the similarity version arrives with the H-C4 domain head. One difference to keep in mind: MindForge's default
  is the *base* (zero adapter), ours is the caller's declared family (`programs` or `talk`), which is right for
  a bank whose two roots are whole fine-tunes.
- **"Created when needed" = Hebbian expert-on-demand + the novelty bridge** (`docs/architecture/13`):
  a `HebbianGrowthLayer` tracks the reconstruction residual of its inputs as an EMA; crossing `grow_threshold`
  (0.35, cooldown 100 steps, capped) spawns a new MoE expert and fires a `NoveltyEvent` whose `direction` is
  the semantic tag; `NoveltyToWorldBridge` projects it to block code and `WorldManager.register_specialist`
  spawns-or-consolidates it by the existing tau rule. MindForge's own bases are updated in the **sleep phase**
  (`docs/architecture/09`, every 1,000 steps or at session end), never live — the H-A7 shape.
- **One wiring gap to not repeat.** The definition is right; the trainer that uses it is not: in
  `cubemind/training/vsa_lm.py` the context handed to `forge_with_cache` is `temporal_ctx = liquid.step(x_mean)`
  — the input's own temporal state, padded to k×l and discretized — so the adapters are conditioned on the
  model's own activity, not on an external task/personality context (the same self-conditioning the hypotheses
  doc flagged in the torch `MindForgeLoRAHead`). Here the context is **host-supplied by construction**: the
  thalamus's route and the neurochemistry state come from outside the model, which is what H0's context
  channel needs and what H-C4 requires (an offline, frozen source).

**Two more concept sources (owner, 2026-09-03: cubemind's live brain and GrillCheese — concept only, the
wiring is not compatible):**

| source | concept | what ports into the lifecycle | what does not |
|---|---|---|---|
| GrillCheese `brain/specialist.py` + `unified_brain.py` | a **SpecialistRegistry keyed by the router's domain**: `ensure(domain)` creates a specialist on the first training signal for that domain, `get(domain)` consults it at inference only when the thalamus names that domain | the registry shape (our bank, keyed by the context role); create on the first *verified* signal, consult only on a matching route | its specialists are NLMS heads on a sentence embedding updated live — no gate, no verification; live weight updates are what H-A7 forbids |
| GrillCheese `Specialist` maturation | **PROGENITOR → MIGRATING → DIFFERENTIATED → MYELINATED** by update count; RESTING / FIRING / REFRACTORY activity | a **maturation ladder** for adapters: *candidate* (in shadow: comparable, never routed) → *routed* (promoted by the rule) → *stable* (N verified turns without a regression); activity = the bank's per-role calls | stages by raw update count (10k updates ≠ trust); ours advance by verified evaluations |
| GrillCheese `cns.py` | consciousness levels; **DEEP_SLEEP = consolidation only** | training and promotion happen offline, never in a live turn (the sleep phase; H-A7's nightly rule) | — |
| GrillCheese `limbic_system.py` | consolidation by **salience = \|valence\| + arousal**: strengthen high, decay low | a **salience weight on the spawn partition**: every SFT record already carries the hormonal state it was produced under; records from high-salience turns are sampled first when a specialist's partition is built | salience as the *only* signal — the verified deficit still decides |
| GrillCheese `thalamus.py` / `basal_ganglia.py` | sensory gating + routing by salience/arousal; go/no-go response gating, winner-take-all | already ours: the thalamus route and the ASK-mediated speech exit | — |
| cubemind `brain/neurogenesis.py` (ported from superfast-neuro / aura-hybrid / AURA_GENESIS) | growth when the residual EMA > 0.35 (cooldown), **prune when recent_spikes < 0.001** (neurons that never fire), Oja normalization, the same maturation stages | **usage-based pruning**: a routed specialist whose calls stay under the line over a window is retired into history (`idle_roles`); growth cooldown and a cap on the number of specialists | residual EMA on activations (the input-side novelty signal belongs to the domain head, not the bank) |
| cubemind **vision cortex** (`perception/bio_vision.py`: opponent-colour, motion-tuned and luminance neurons → `BioVisionEncoder`; `perception/snn.py`: LIF spiking layer, spikes → VSA projection → a packed binary temporal memory, **neurochemical modulation as dynamic threshold and leak** — dopamine / adrenaline / acetylcholine; the webcam demo worked, owner 2026-09-03) | a cortex that runs under the contract end to end: input → neurons → VSA → memory, with the hormones scaling thresholds — the same pattern as our ODE's `modulate_threshold` on the routing threshold | (a) the **input-side novelty** the detector needs: chapter 9's `novelty = ‖x_t − x_{t−1}‖ / ‖x_{t−1}‖` on the cortex's stream (LIF neurons only fire on change — sparsity for free) is the signal that says "a context no specialist has seen", before any verified deficit exists; (b) the **SNN slot** in the serve brain's fast route is shaped for exactly this layer (`affective-snn-cortex-resources`); (c) proof that a cortex plus a hormone system is a working unit to attach adapters to | the camera and OpenCV wiring; the SNN's VSA bridge is cubemind's `BlockCodes`, ours is grilly's `BlockCodeOps` directly (the same substrate, ported not linked) |
| cubemind live brain (`_archive/scripts/live_brain.py`, `create_cubemind(growth_threshold=0.3, enable_neurochemistry=True)`) | the whole loop on camera input: perception → reasoning → memory → neurogenesis, with the neurochemistry visible | the loop shape is the serve brain's; the on-screen neurogenesis counters become `/health`'s `usage()` (roles, candidates, calls, fallbacks, idle) | the script is archived; the wiring is not ours |

What the bank gained from this pass: **shadow candidates** (`shadow` / `emit_candidate` / `promote` / `reject`,
`stage(role)`), so the promote step has something to compare and promotion is an explicit, logged act, and
**`idle_roles()`** as the prune monitor. Pinned in the same test.

Two adjustments to the lifecycle above, from the check: **(1) Detect has two sources, not one** — *novelty*
(a context no registered specialist is near: the bank's fallback rate today, the domain head's purify
threshold tomorrow; cubemind's residual-EMA trigger is the input-side version of it) and *deficit* (the
measured verified-rate gap that gates promotion). Novelty proposes, the verified deficit disposes. **(2) The
promotion rule should be a CubeLang program.** cubemind's QC cortex (`QC_APP_PLAN.md`, `cubelang/examples/
qc_decision.cube`) already has the pattern: perception emits `(class, confidence)`, a compiled `.cube` rule
returns PASS / REJECT / REVIEW, and the audit record carries the rule's artifact hash with the confidence. The
H-A7 gate is the same shape — candidate-vs-incumbent margins in, PROMOTE / KEEP / REVIEW out, thresholds
pre-registered *in the program*, deny-by-default in the VM, hash-audited. (The QC cortex's Python sources are
no longer in cubemind's tree — only `__pycache__` remains for `qc_perception`, `anomaly`, `patch_features` —
so the concept is what carries over, ported clean per the integration rule, not the code.)

What exists as of this commit: the bank with `register` / `unregister` / `fallbacks` / `usage()` and the
retirement history (pinned by `test_the_adapter_bank_grows_and_counts_what_it_cannot_serve`), the partition
builder for the task split, the frozen router, and the eval harnesses the gate composes. The next cycle
(spec → plan → build) is the detector + the tag-general partition builder + the promote script; the v8e /
v8t A/B is its first measurement — it tells whether adapter-per-context removes the interference at all,
which the whole lifecycle presupposes.

**Also in v8's data:** the FR location phrasings of the `world` intent ("Tu habites où ?", "Où es-tu ?", "Où
est ta maison ?" + EN "Where are you?", "Which world do you live in?") after v7 answered "Où vis-tu ?" with an
invented town; the identity-question detector catches them so they never reach the fact path.

### v7 set BUILT (2026-09-03, `--version v7`; on Drive, sha `565b89e3…`)

What v6's reads asked for, in one build: **57,671 records** = v4's program families replayed (arithmetic
6,148, kernel 1,769, role_binding 1,500, chain 1,201) with the **forge decision/compare records replayed ×3**
(727 train records — the v6 probe regression) + **identity regenerated from `identity.py`** (564 records
across 13 intents, the new `world` intent — the cubbyverse — at 38; ×4 on train; the v4 replay's copy
dropped) + 20,569 chat (v6's sources + the 234 real science Q&A rows) + 1,396 content + 9,880 emotion (v6's
+ 220 screenplay scenes in the spoken register) + 1,679 affect + 12,965 history. Same gate everywhere; the
manifest carries every count. Train with `VERSION='v7'` (the notebook default now; 32×1, 2 epochs, seq
4096); the reads to take afterwards: the stratified VM eval (`STANDIN_EVAL_N=40`), the forge probe (decision
must climb back from 0.17), and the self-test (the chain fixed by the final-relation gate must read as
don't-know).

### v7 TRAINED (2026-09-03): the smoke read, the GPU reads pending

Colab run on the v7 set (32×1, 2 epochs, seq 4096; adapter, merged model and both GGUFs on Drive under
`emitter_lfm25_2p6b_v7/`, the Q4 copied to `standin/models/emitter_v7.Q4_K_M.gguf`). The notebook's
validation came out at 6 per task again (the eval-size variable was not set), replayed through the VM
(`data/out/eval_emitter_vm_v7_colab.json`):

| task | v7 (n=6) | note |
|---|---|---|
| chat / content / affect | 1.000 | — |
| identity | 0.833 | **the `world` intent works from training** ("What is cubby-man?" → the cubbyverse line verbatim); the one miss is the same "Good evening!" that missed in v6 — a generic greeting without the name |
| emotion | 0.833 | the miss is rater noise ("I'm O.K., honey" labelled serenity/trust, read as caring/neutral) |
| history | 0.333 | the four misses are two `news` and two `dating` — the headline-recall and decade-dating families that were the weak spot at n=40 in v6 (0.75); `when` and `dialogue` right |
| game families | 1.000 (n=8) | — |
| arithmetic / chain | gold 0.500 / 0.667 (n=6 / 3) | too few to read |

Too small to call anything but the identity `world` intent. **Forge probe v7 (owner, solo on the local GPU, 58 s,
`data/out/forge_probe_v7.json`): decision 1.00 (12/12) · compare 1.00 (12/12) · chain 1.00 (12/12)** — back from v6's
0.17 / 0.83 / 1.00; the ×3 replay of the 727 decision/compare records was the whole fix, the same lever that
restored identity in v6. **Self-test v7 (owner, solo, `data/out/serve_selftest_v7.json`): gold 0.96; spoken 96% correct / 4% don't-know /
0% wrong** — v6's one wrong chain ("the parent entity of the instance of Masahiko Kumagai", hop 1's object
spoken) is now the don't-know, exactly what the final-relation ground check was built for; nothing else moved
(v3 76/20/4 → v6 96/0/4 → v7 96/4/0). **The stratified read (40/task, 30 identity, generated on a fresh VM from the merged v7 with
`STANDIN_EVAL_ONLY=1` + `STANDIN_EVAL_N=40`, VM replay local; `data/out/eval_emitter_vm_v7_colab40.json`):**

| task | v7 (n=40) | v6 (n=40) | v5 | what the misses are |
|---|---|---|---|---|
| chat | 1.000 | 1.000 | 0.975 | — |
| content | 1.000 | 1.000 | 1.000 | — |
| identity | 0.867 (26/30) | 0.909 (20/22) | 0.818 | "Good evening!" ×2 (the same generic greeting, three rounds running) and **"Où vis-tu ?" ×2 answered with an invented place** ("un labo dans le Nord de la France") — the FR *where-do-you-live* phrasing of the new `world` intent pulls a location; the six other `world` prompts answered with the cubbyverse line |
| emotion | 0.675 | 0.625 | 0.450 | rater noise as before |
| affect | 0.975 | 0.950 | — | — |
| history | 0.625 | 0.750 | — | `quote_who` **0/6** (wrong author every time — 1,200 records cannot memorize 24k quotes' authorship; a recall family, not a reasoning one), `dating` 4/6, `era` 3/5, `news` 0/3, `year` 1/3; `dialogue` 9/9, `when` 6/6, `about` 2/2 |
| arithmetic (VM gold) | **0.675** | 0.750 | 0.825 | executes 1.000; 13 wrong computations — **three rounds of decline** as the chat/history volume grew; the next family at risk |
| kernel / chain | 0.917 / 1.000 | 0.833 / 1.000 | 0.818 / 0.824 | one `game:where` chain off by one cell (22/23) |
| role_binding executes | 0.975 | 0.975 | 0.925 | — |
| game families | 1.000 (n=28) | 1.000 | 1.000 | held |

Read with the two GPU probes above: v7 repaired both v6 regressions (forge decision 0.17 → 1.00; the wrong
self-test chain → don't-know) and kept the gains. Two things to carry into v8: **arithmetic** has slid
0.825 → 0.750 → 0.675 across three rounds of growing chat volume — the same replay lever (×2 on the
6,148 arithmetic records, or a smaller chat quota) before it becomes the v8 regression; and the FR `world`
phrasing "Où vis-tu ?" needs its answer pinned (a location question must reach the cubbyverse line, never an
invented town). `quote_who` should be read as a recall ceiling, not a target.



### v6 TRAINED + MEASURED (2026-09-03): the stratified VM read

Colab run: 1,890 steps at 8×4 (effective 32), 1 epoch, 35 min, final loss 0.52 — a **mixture floor**
(free chat/history text dominates the token count; programs sit near zero), not a target; the run used ~8 of
80 GB, so the notebook now defaults to 32×1, 2 epochs and sequence length 4096. **The stratified read**: 40
val records per task (22 identity) generated on Colab from the merged model (`STANDIN_EVAL_ONLY=1`,
`STANDIN_EVAL_N=40` after the VM reset) and replayed through the local VM eval
(`eval_emitter_vm.py --val-generations …/emitter_lfm25_2p6b_v6/val_generations.json --tag _v6_colab40`,
`data/out/eval_emitter_vm_v6_colab40.json`):

| task | v6 (n=40) | v5 (stratified) | what the misses are |
|---|---|---|---|
| chat | **1.000** (EN + FR) | 0.975 | — |
| content | 1.000 | 1.000 | — |
| identity | **0.909** (20/22) | 0.818 | both misses are one prompt, "Good evening!", answered generically without the name (the ×4 replay fixed FR `unknown` and every other EN greeting) |
| **emotion** | **0.625** | 0.450 | near-misses and rater noise ("Impeach fouty five!!" is labelled excitement + neutral; FR *confusion*→curiosité, *agacement*→désapprobation) |
| **affect** | **0.950** | — | 38/40 within 0.35 on both numbers; the two misses are a sarcastic request read as mildly positive and an arousal 0.4 read as 0.8 |
| **history** | **0.750** | — | 10 misses: 4 `news` (specific NYT headlines confabulated as "The New York Times: …"), 3 `quote_who` (wrong author), 1 `dating`, 1 `dialogue`, 1 `quote_about` — recall of a specific headline is the hard family, as expected; `when`/`year`/`about`/`era` all held |
| arithmetic | gold 0.750, executes 0.975 | 0.825 | 9 gold misses at n=40, three questions' worth of dip; watch, not act |
| kernel / chain | gold 0.833 (n=12) / **1.000** (n=17) | 0.818 / 0.824 | chain up from v5's copy noise |
| role_binding executes | **0.975** | 0.925 | the code-leak filter worked (one parse error left) |
| game families (where / compare / compose / count / decision) | 1.000 (n=51) | 1.000 | held |

**The two GPU reads, run solo by the owner (2026-09-03, no reset):**

- **Serve self-test** (25 val chains, facts stripped from the prompt, own retrieval; `serve.py --selftest 25
  --tag _v6`, `data/out/serve_selftest_v6.json`): gold **0.96**; spoken **96% correct / 0% don't-know / 4%
  wrong** against v3's 76 / 20 / 4. By the tripwire rule adopted from the GoT challenge (one chain is four
  points; don't-know down with wrong flat and gold up is healthy) this is healthy abstention, not silence.
  The one wrong chain is a **gate gap, now closed**: "What is the parent entity of the instance of X?" — the
  walk exhausted at hop 2 and offered hop 1's fact, the emitter bound hop 1's object, and the ground check
  accepted it because it only asked whether the answer was *some* offered fact's object. The ground check
  now also requires, for a parsed multi-hop question, that the answer's fact carry the question's **final
  relation** (`ReasoningCortex.task_answer`, trace event `gate_relation`; single hops and planner misses
  unchanged; pinned by `test_live_a_multi_hop_answer_must_carry_the_final_relation`). Re-run the self-test to
  confirm the chain becomes a don't-know.
- **Forge probe** (`scripts/forge_probe.py --n 12 --tag _v6`, `data/out/forge_probe_v6.json`): decision
  **0.17** / compare **0.83** / chain **1.00** against v4's 1.00 / 1.00 / 1.00 — **a regression on the decision
  family.** Failure modes: eight decision programs execute but return the raw reading instead of the ≥50
  decision class, two invent a `comp` opcode (`compare` is the real one), two compares pick the wrong index.
  Cause: v6's 20k chat pairs outweigh the ~200 decision/compare records in the v4 replay, the same pull that
  hit identity in v5. Fix queued in the builder for the next build: the forge decision/compare records replay
  ×3 (`GAME_FORGE_REPEAT`), as identity does ×4. The serve context now matches training (`n_ctx` 4096 in
  `serve.py` and the probe; the 2048 warning was not the cause — these prompts are a few hundred tokens).

The notebook's own read of the same file said arithmetic 0/40: that is text exact-match, a format read;
the VM's 0.750 gold is the capability read. **v6 is the serving model now** (`--gguf
standin/models/emitter_v6.Q4_K_M.gguf`); the emotion read stays **opt-in** (`--model-appraisal`) at 0.625 —
better than v5's 0.45, not yet better than the lexicon on the drives that matter. Not measured this round:
the serve self-test (25 chains, the don't-know tripwire from the GoT challenge; needs the local GPU) and
the forge probe.

**The GPU note stands.** The local stratified run (40/task, Vulkan on the RX 6750 XT) crashed the GPU at
50/382 with the `/pac` server holding a second copy of the model. Until that is understood, the wide read
comes from Colab as above; the VM part runs on the CPU. After a VM reset the notebook loads the merged
model from Drive (`STANDIN_EVAL_ONLY=1`) instead of retraining.

## Chat is mediated by a VM program, not emitted as one (design note, 2026-08-30)

Checked in the cubelang source. The VM's **registry-seeded (tamper-proof) interfaces
are `ISolve`, `ISolver`, `ISolverLearn` only**. The conversational interface the spec
defines — **`IAgent`** (`think(input, ctx): str` · `act` · `observe` + optional `plan`,
`reflect`; `container cMathTutor implements IAgent, IDeployable` with a persona, a
rag memory and a gguf model is the spec's own example) — has **0 hits in `src/`**
(`container` parses; `agent.create`/`persona`/`think` do not exist). The one
conversational program that runs today, `examples/conversation_agent/
conversation_agent.cube::ConversationReasoner implements ISolverLearn`, is a
dialogue *substrate* (turn → intent / act / plan / risk flags / trace → `verify`
refuses empty text / zero confidence / empty trace); its reply text is canned.

**ASK is real and is the seam.** `examples/ask_min.cube` + `ground_min.cube`: the `ask`
opcode yields `ExecResult::Suspend(Suspension{question, candidates, pc, …})`;
`VM::resume(susp, answer)` pushes the answer and continues, and *rejects any answer
that is not identical to one of the offered candidates* — "the trunk supplies syntax
and selection, never content". `cubelang run --json` surfaces a suspension as
`{suspended: true, question, candidates}`; **`run-proto` (what `cubelang_client`
uses) cannot represent it and reports it as an error** — the deferred "memory-service
ASK cycle".

So for the stand-in: identity/chat turns stay text (they are the model's `Reply.text`
/ `IAgent.think` output); the serve loop should wrap each turn in a
`ConversationReasoner`-shaped program (state, risk flags, `verify`; the hormonal
state belongs in `DialogueState` — cortisol raising the verify bar is the
`neurochemistry.py::modulate_threshold` analogue). **Both prerequisites landed the
same day (cubelang `d969a76`):** `IAgent` is registry-seeded (think/act/observe), and
`run-proto` now reports `suspended{question, candidates}` and resumes **by
re-execution** (`RunRequest.answers`; `VM::resume`'s identical-to-a-candidate guard
kept) — `cubelang_client.run_program_proto` returns `suspended: True` + decoded
candidates, `resume_program_proto(answers=[…])` continues. The actual blocker was
the strict verifier calling `ask` "trace-only" although it executes; fixed. What is
left is the chat program itself (`think` ASKs with the model's candidate replies).

## cubby-lm's identity/behavior files (checked 2026-09-03, owner's ask: "could sound more human")

`C:\Users\grill\Documents\GitHub\cubby-lm\data\identity\` holds three files, profiled against the live gate:

| file | what it is | verdict |
|---|---|---|
| `cubby_behavior_conversational.jsonl` — 744 conversations, 11,117 user→assistant pairs, median 15 turns | the owner's **real coding sessions** with an AI assistant (`conversations_svc_semantic`), paths anonymized to `[PROJECT_NAME]`-style placeholders, relabelled under an older Cubby persona | **the human side is the value, the assistant side is not.** The user turns are real human phrasing (terse commands, frustration, typos). The assistant turns are a coding agent's voice with reasoning-trace text leaking in ("The user is clearly frustrated that…") and holes where placeholders were stripped ("defined in , including a flexible enum for and"); 4,049 pairs pass the gate mechanically but would teach that voice. **Not SFT targets.** Use: the real human turns as (a) a **realistic serve eval** — run them through the talk path and measure voice/guard/bio pass rates and the route distribution, which no dataset-shaped val split measures today — and (b) prompts for a routing family, once labels exist that are not just our own rules echoed back. Privacy screen first: the text carries personal context (a location, preferences) even with paths anonymized. |
| `cubby_behavior_conversational_pre.jsonl` — 2,019 conversations, 4,280 pairs | `pre/conversation.jsonl` (the topic-tagged set we already train on as `convo`) + 58 `batch_chat_templates` (the old product persona) | already in (v6+); nothing new |
| `cubby_identity_core.jsonl` — 60 pairs, EN+FR | an **older identity** from an `identity_facts.yaml`: "Cubby, the language cortex of the Grilly agent, built by Grillcheese Research Laboratory (GRL)… the lab behind the Grilly stack" | **do not use** — a different persona (same rule as `identity_corpus.txt`); 25 of 60 already trip the identity screen, the other 35 would inject "Grilly agent / GRL" |

"Sounding more human" is a *prompt-side* problem here: the model already answers dataset-shaped prompts at 1.0
on the voice rules; what is unmeasured is how it answers real human phrasing. The eval above is the cheap
next step; pairing those prompts with replies written in Cubby's voice would need human-written replies,
not the base model's (training Cubby on its own chat is the self-play shortcut the invariants reject).

## The ledger and the vault (2026-09-04)

**Certifications are signed now.** Until today a certified program was a boolean verdict in a JSON file the library
trusted on load. `standin/ledger.py` is the database the owner asked for: every VM decision — the forge's
(`ToolForge.forge`, certified or rejected) and the game's own joins (`_certify_join`) — is stored in SQLite
(`standin/data/out/ledger.sqlite`) as {program sha256, name, kind, function, input, expected, got, ok, VM build}, keyed by
the **decision hash** (SHA-256 of the canonical JSON of those fields) and **signed** (HMAC-SHA256 over the hash with a key
derived from the host key). The VM build is the SHA-256 of `cubelang.exe`. A `ProgramLibrary` entry keeps only the hash; on
load (`audit`) the entry's program text must hash to a signed, certified row made by the current VM build — otherwise it
loads RETIRED, never deleted, with the reason (`certificate: program text changed` / `bad signature` / `decision was a
rejection` / `certified by another VM build`). A new VM build therefore retires every certificate at once, and the stored
vectors are what re-certification replays. Pre-ledger entries are flagged `uncertified` and left alone. Pinned in
`standin/tests/test_ledger_vault.py` (a program edited on disk loads retired). This is the certificate store the factory's
registry sits on.

**User chats are never stored unencrypted.** `standin/vault.py` — stdlib only (no `cryptography` here): an HMAC-SHA256
keystream over a fresh 16-byte nonce, XOR, then an HMAC-SHA256 tag (encrypt-then-MAC), under `standin/data/out/host.key`
(generated once, 256 bits; or `CB_HOST_KEY`); the vault and the ledger use keys derived from it. Through it: the real-user-turn
eval file (`real_user_turns.jsonl.enc`; the plaintext is removed), the talk probe's rows (`talk_probe*.json.enc`; the summary
stays in the clear with no user text), and the ledger's `input` column (a factory request is a user turn). The committed
probe records under `validation/logs/` had their rows removed the same day; the earlier commit that carried them stays in history
because the repository is private (owner, 2026-09-04) — the vault rule holds regardless: no user turn is committed in the clear from here on. CubbyChat's history is in memory only; any
chat log the serve stack ever persists goes through the vault. **Not covered, by necessity:** the training sets (the
owner's WhatsApp/Teams rows train in the clear on Colab; that is the owner's decision, recorded in § SFT gap map).

## Dataset candidates screened 2026-09-03 (owner's four links; numbers from `standin/scripts/question_style_probe.py` → `validation/logs/standin_question_style_probe.log`)

| dataset | what it is | verdict |
|---|---|---|
| `google-research-datasets/disfl_qa` (CC-BY-4.0, EN; 7.2k train / 1k val / 3.6k test) | SQuAD-v2 dev questions, each with a contextual restart using a distractor from the passage ("What is the second level of … no make that the basic unit of …"); cues: *no* 565, *sorry* 127, *rather* 119, *make that* 92, *uh* 82, *i mean* 79 per 1k | **Use, two ways.** (1) A **question-repair family** in the talk partition: disfluent → the clean question (gold exact, 7.2k pairs), applied in the thalamus only when the grammar fails on a question, guarded like `rephrase_ok` (every number and name kept). (2) The **eval** for it: on the 1k val pairs the fact grammar parses 150/1000 clean questions and only 80 of those 150 twins parse the same (53%) — the repair's value is that recovery. Routing itself is robust (needs_facts agrees on 992/1000). |
| `google-research-datasets/natural_questions` (CC-BY-SA-3.0; dev 7.8k; the parquet is 1.3 GB of Wikipedia HTML — take the question + short answers through the rows API) | real search queries: all lowercase, none end with "?", 9.3 words ("who is the secretary of state for northern ireland", "perth is the capital of which australian state", "bad guys age of evil korean drama asianwiki") | **A slice as the realism eval, not training** (no rewritten form exists to train on, and the answers are Wikipedia spans, not our triples). It already paid: the grammar parses **0/300** (the shape problem behind the 92 misparsed chains, not disfluency), and the router misread 13/300 — fixed the same day (greetings only at the turn's opening, "do you feel" only when little follows, wh/yes-no fact openings win over the creative words, yes/no openers and mid-sentence "which"/"… is called" are questions): reads now 220 facts / 74 small talk / 6 creative, the 6 being keyword queries with *song*/*poem*/*write* as content words. 155/300 carry a short answer — a knowledge read for the VM path once a store exists for those pages (the table's measured boundary is open-web entity search; expect the floor). |
| `ukisai/Qwen3.8-27B-multi-turn-agent-sft` (Apache-2.0; 15.2k traces) | Terminus-2 terminal-agent traces generated by Qwen3.8-27B on OpenThoughts-Agent tasks: JSON `analysis / plan / commands` over shell sessions | **No.** The wrong job (shell tool-calls; our tool is the VM and CubeLang) and the wrong voice (a distilled agent persona the identity screen exists to keep out). |
| `jjssuh/sftuser-intent-Llama-3.1-70B-train` (no licence tag; 273k rows, 2.1 GB) | WildChat-1M rows (real user turns with GPT-3.5/4 replies, moderation scores, hashed IPs) plus an `intent` column written by Llama-3.1-70B — a free-text **user-simulator persona** ("You are a user chatting with an assistant language model to obtain information about a past sporting event."), the training target of a *user* model ("sftuser"), not an intent taxonomy | **No.** The label is a persona sentence, not a class — a router signal would need a second model-labelling pass over model labels; the assistant side is the base-guard voice ("As an AI language model, I do not have access…"); the real user side (multi-turn pushback like "You are incorrect…") is worth having for the realistic serve eval, but from `allenai/WildChat-1M` itself, which carries the ODC-BY licence this copy dropped. |
| `Lots-of-LoRAs/task581_socialiqa_question_generation` (Apache-2.0; 5.2k train / 650 valid / 650 test) | Super-NaturalInstructions task 581: SocialIQA run **backwards** — given a social situation and its answer, write the question; every row carries the same 600-character definition + four worked examples, and the outputs are the six SocialIQA question templates ("How would Ash feel afterwards?", "What will Others want to do next?", "Why did Taylor do this?") | **Not this task.** The target is one of six template questions, the wrong direction for us, and the repeated instruction prefix would teach the prompt format, not the skill. The **source** is the candidate: `allenai/social_i_qa` (CC-BY, 33k) in the forward direction — situation + question → answer ("How would Sydney feel?" → "sympathetic") — as a small **social-appraisal** family beside emotion/affect: situational feeling, need and motivation, the appraisal the sense stage feeds the hormones from. Queue it for the v9 talk data with the same label-first check as emotion. |
| `OpenLLM-France/Claire-Dialogue-French-0.1` (CC-BY-NC-SA-4.0, gated, approved 2026-09-04) | 37k real French dialogue transcripts; the 16 conversational subsets (~52 MB) | **In (v9).** 326k candidate turn pairs → zero-filler, content-bearing replies, no subset above 30%, 1,500 kept; parliament and theatre left out (wrong register). |
| `rbawden/DiaBLa-dataset` (GitHub, CC-BY-SA-4.0) | 144 spontaneous written EN↔FR dialogues, every utterance with a human reference translation | **In (v9).** Pairs where both sides are human: the other speaker's turn in its reference translation → this speaker's own line (891 EN / 938 FR). |
| `graalul/QFrCoRE_QFrCoRT` (CC-BY-NC-4.0; the ACL 2026 Findings dialect-gap paper) | 4,633 Quebec French expressions + 171 Quebec words, 10 candidate definitions each | **In (v9)** as `quebec` (explain) + `quebec_mc` (the benchmark's form). MFrCoE (the Metropolitan set) is not on the Hub. |
| `community-datasets/re_dial` (CC-BY-4.0) | 10k human movie-recommendation dialogues | **In (v9).** Seeker → recommender pairs, 1,500. |
| the owner's `conversations_dataset_anonymized.jsonl` (880 conversations) + `team-dataset.txt` (Teams) | 78 WhatsApp chats (human both sides, Quebec EN/FR) + 802 ChatGPT/Claude exports + a pasted Teams thread (210 messages, 2 colleagues) | **WhatsApp + Teams in (v9)** through the gate + privacy/explicit screens (2,000 + 57 pairs); **the AI exports' assistant side is never a target** (a coding agent's voice, 9% reasoning-trace openers) — their 10k real user turns are the serve eval's prompts. |
| `Agent-Ark/Toucan-1.5M` (Apache-2.0) | 1.5M synthetic tool-agent trajectories over 495 MCP servers | **No.** Model-generated, and the wrong contract: our tool calls are VM-verified programs over registered capabilities (see the v9 section). |
| `Na0s/sft-ready-Text-Generation-Augmented-Data-Alpaca-Format` (no licence, no card; 7.7M rows) | lmsys-style anonymized user prompts (`NAME_1`) with model-written answers ("Sure, I can help…", "Hello! It seems like you're sharing…") | **No.** The assistant side is exactly the guard voice the model-guard rejects, the provenance is unknown, and the user side we already have (`arena` in the local chat sources). |

## SFT gap map (2026-09-03) — what the measured record says is missing, and the source for each

The gaps come from the reads, not from browsing: the verbalizer's failure modes, the two chat misroutes, the
0/300 real-query parse rate, the 53% restart survival, the arithmetic slide, the thin FR share (27/252 talk val),
the 380-row safety read, and the fact that every talk record is single-turn. Human-written sources first;
anything model-generated is marked. Licences noted because two are non-commercial.

| gap (measured) | source | shape of the family | licence / size |
|---|---|---|---|
| **the game itself** (owner: "use the cubby-man log too" — satisfaction 0.8, 10,545 memories, 6,380 derived, 323 walls learned) | the live event log + `ProgramLibrary` | (a) VM-certified programs — already the v4–v7 game families; (b) **verbalize pairs by construction**: `CubbyGhost.THOUGHTS` carries three phrasings per event kind in EN and FR, so variant *i* → variant *j* of the same event is a paraphrase that keeps every number and name with zero model output — the verbalize family's first source; (c) **derive events** (`rule=symmetry` / `rule=count` with `vm_computed`) → a reasoning family with VM gold; (d) **probe refusals** (a wall learned) → yes/no "is *left* open from cell X?" with VM truth; (e) the fact/derived store → chain questions over the level graph | ours |
| follow-up questions that depend on the previous turn ("what about its capital?") — no multi-turn record exists | **CANARD** (`gaussalgo/Canard_Wiki-augmented` or the original via `uva-irlab/canard_quretec`) and **QReCC** (`svakulenk0/qrecc`) — human question-in-context rewrites | a `rewrite` family: (history, follow-up) → the standalone question, applied in the thalamus before the grammar; eval = parse rate of the rewrite vs the raw follow-up | CC-BY-SA-4.0 (~40k) / CC-BY-3.0 (~80k) |
| restarts and repairs (53% of parsed questions break) | **disfl_qa** (screened above) | the `repair` family | CC-BY-4.0, 7.2k |
| dialogue-context emotion + a **human-labelled act** (inform / question / directive / commissive — the routing classes the WildChat set failed to provide) | **DailyDialog** (`li2017dailydialog/daily_dialog`, 13k human dialogues, emotion + act per turn) and **EmpatheticDialogues** (`facebook/empathetic_dialogues`, 25k conversations grounded in a felt situation) | emotion-in-context beside GoEmotions; the act label as a routing-family seed; ED's situation → feeling for the appraisal family with `social_i_qa` | **CC-BY-NC-SA-4.0** / **CC-BY-NC-4.0** — prototype-only |
| French chat and multi-turn (human) | **OASST2** (`OpenAssistant/oasst2`, human-written message trees, 35 languages incl. FR, quality labels) | FR chat + multi-turn from top-ranked human replies, through the model-guard and identity screens like every chat source; **PAWS-X** (`google-research-datasets/paws-x`, human-translated FR pairs) for FR paraphrase | Apache-2.0 / "other" (research) |
| safety read at 380 rows, EN only | **deepset/prompt-injections** (662, human), **lmsys/toxic-chat** (10k real prompts, human toxicity + jailbreak labels), **yanismiraoui/prompt_injections** (1k, multilingual incl. **FR**); `allenai/wildjailbreak` last (262k, synthetic, gated) | the `safety` family's label-first read, now with FR attacks | Apache / **CC-BY-NC-4.0** / Apache / ODC-BY |
| arithmetic slid 0.825 → 0.675 as chat grew, and real questions come in shapes GSM8K does not | **SVAMP** (`ChilleD/SVAMP`, 1k human variations built to break shallow heuristics), **ASDiv** (`EleutherAI/asdiv`, 2.3k), **MAWPS** (`cq01/mawps-asdiv-a_svamp`) | through the builder's VM verification like GSM8K-train → program records with diverse phrasing (GSM8K-test stays excluded) | MIT / **CC-BY-NC-4.0** / — |
| verbalizer paraphrase beyond the game's own templates | **PAWS** (`google-research-datasets/paws`, 108k human-labelled pairs; the negatives are word-swap non-paraphrases — the drift the guard rejects) | paraphrase pairs with entity preservation; the negatives as a discrimination check, not targets | "other" (research use) |
| situational appraisal (how would X feel / need / want) | **`allenai/social_i_qa`** (33k, forward) | the `appraisal` family (queued above) | CC-BY |
| history recall (dating 0/6, news 1/7, quote_who 0/3) | no dataset fixes memorization | epochs and the v7 repair pattern (replay the hurt family), not new data | — |

Order of value: the game's own log (free, in-domain, verifier-backed) → CANARD/QReCC + disfl_qa (the thalamus's
two missing skills) → OASST2 FR → the safety trio → SVAMP/ASDiv for arithmetic → DailyDialog/ED/social_i_qa
for the affect side → PAWS last. Every family enters through `build_chat_sft.py` with its own check and lands in
the talk or program partition by `partition_records`; nothing model-generated becomes a target.

## Data facts worth knowing (2026-08-30 audit of `cubemind/sandbox/regen`)

- 38.6k programs in `cubby_aug_v4.txt`: 32k role-binding (`Evt`/`Ev`), 4.2k GSM
  arithmetic, 4×600 kernels; generator sources are gone (`.pyc` only).
- **Every GSM program in the aug corpus is a GSM8K *test* question** (4,228/4,228);
  `multitask_v4_arith_xl.jsonl` adds 6,148 that are exactly GSM8K *train*. Test is
  excluded from SFT; the manifest counts the exclusion.
- The current VM's `ISolver` requires `parse(raw)` + `pure verify(input, output)`;
  the kernels ship both, arithmetic/role-binding do not. With the two-function shim
  (verify is a `return true` stub, as the kernels ship it) arithmetic re-verified
  300/300 to gold and kernels 60/60 in the audit; role-binding executes (no gold).
- The 517 chain programs are the `ISolve`/`recover` dialect — the one the oracle
  actually needs — and are kept whole; role-binding is capped so the emitter does
  not learn to bind ACTION/AGENT and little else.
