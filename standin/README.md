# standin/ — the stand-in trunk

**Status:** tooling for a disposable stand-in, 2026-08-30. Nothing here is CubbyLLM.

A small open model (currently `LiquidAI/LFM2.5-2.6B`, chosen by a 2026-08-30 Hub
check — see `docs/research/2026-08-28-oracle-competition-scored.md` §3.2 for why a
stand-in exists at all) is fine-tuned with Unsloth on Colab and served locally as a
GGUF, so the parts of the serve stack that need *some* trunk — the CubeLang program
emitter, the learning gate as a live nightly loop, the chat/grounded-answer surface,
the H-F2 bridge contract — can be built and measured while the 2B CubbyLLM trunk
trains. It is replaced the day the 2B checkpoint exists.

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
| `serve.py` | **The stand-in brain** (2026-08-31, brain shape same day): one process, one turn — **sense** (appraisal → the ported neurochemistry ODE; hormones scale the routing threshold via `modulate_threshold`, never facts) → **fast route** (the SNN slot — today lexical+retrieval, the interface shaped for the spikeybrain port, no SNN claim) → **cortex router** → `MemoryCortex` (live learning), `ReasoningCortex` (plan+**walk** with the measured CoT pipeline at tau_vm=0.2202 / tau_ret=0.5959, flat top-k fallback → Facts block → v3 emitter with the CotChain opening **prefilled** → VM → ground check + **consistency gate**: a verified walk that disagrees vetoes the answer), a mounted **plugin cortex**, or `CubbyChat`. Every spoken reply from every cortex exits through `CubbyTalk`'s ASK (`CubbyChat.mediate`) — the VM rejects any selection it didn't offer, and the voice rules + verbatim don't-know line hold whatever is mounted. REPL: `python standin/serve.py --gguf standin/models/emitter_v3.Q4_K_M.gguf`. |
| `neurochem.py` | **The real 5-signal neurochemical ODE**, ported clean from `cubemind/brain/neurochemistry.py` (port-don't-link): dH/dt = α·drive·sensitivity − β·(H−resting), receptor-saturated couplings, cortisol as a slow HPA EMA (NE is the fast stress signal), refractory receptor sensitivity, Lövheim-cube emotion readout, `modulate_threshold`/`modulate_tau`. Replaces `nudge()`'s placeholder as `CubbyChat`'s state source; `appraise()` is the host's transparent lexical (EN/FR) reading of a message into the five drives. Same clip bands `identity.HORMONE_RANGE` pins. |
| `worlds.py` | `FactStore` — the **incremental** retrieval world (same unit-cosine math as exp_m3's `make_retriever`, plus `add()` so a fact accepted at turn N is retrievable at turn N+1; IDF-overlap fallback without an encoder) — `route_world` (MoWM v0: best top-score wins) and the **`CubbyPlugin` mount contract, MindForge style**: a plugin (cubbyverse) sits ON TOP of the trunk — it contributes named worlds and optional cortices and observes turns; it imports us, we never import it, and it can never produce speech that bypasses the ASK + voice rules. |
| `pacman.py` | **cubby-man wired to the cubbyverse 3D Pac-Man** (2026-08-31). Wire target is `cubbyverse/examples/pacman_3d.py` — deliberately: `pacman_live.py` imports it for MOVES/encoder, so it is the shared substrate of both. The maze is ported verbatim (same rng call order, seed 7 — the 7×7×3, 16-pellet map the scripted VSA-planner demo beats) and `CubbyPac(CubbyMan)` plays it the explorer way: six VM-guarded directions, walls from ASK refusals, pellets eaten on arrival as permanent discovery facts, six-direction symmetry/count joins, the run recorded in pacman_3d's own replay schema and rendered through its `_HTML` three.js template — extracted from the file at render time, never imported (no grilly/cubbyverse package dependency) — to `C:/tmp/cubbyman_pacman_3d.html`. Mount with `--pacman` on `serve.py`/`serve_api.py`, then "play pacman for 30". **LIVE play (2026-09-02): `LivePac` + `pac_live.html`** — the pacman_live.py pattern: the browser polls `GET /pac/state`, each due poll advances ONE VM-guarded step (rate-limited; he plays only while the page is open), and the live three.js page at **`/pac`** animates him in real time (pellets vanish as eaten, HUD shows score/steps/facts/walls-learned/derived/emotion); the step lock keeps the poller and a chat "play pacman" turn from interleaving a move, and beating the map auto-renders the replay HTML. **The BIG game (2026-09-02): `GhostVerse` + `CubbyGhost`** — pacman_live.py's game ported as his env: seed-shared carved labyrinths (`Mulberry32`/`carve_labyrinth` bit-for-bit), the exact level formulas (size/walls/hazards/pellets per level), star power-pellets with the farthest-point spread, FRIGHT_STEPS=14 / GHOST_BONUS=5, lives, level advance with level-scoped fact names (everything he ever learned stays true). He SEES hazards glow (observation facts); plain walls stay invisible until a refused move teaches them. Ghost proximity feeds THREAT into the neurochemistry (anxious when hunted, hunting when they're frightened — the fear-aware `_pick`), being caught is learned as a danger fact. Simplifications on record: greedy-Manhattan ghosts (theirs are mSA controllers with scent fields); vaults/superpowers/letter-words/time-budget stay with the cubbyverse-side splice (no jump ⇒ vaulted pellets would be unreachable). `--pacman` mounts the big game. **The view IS pacman_live's own page (2026-09-02):** `load_frontend()` extracts its `FRONTEND` block verbatim at serve time (never imported — the file pulls grilly/cubbyverse packages) and `LivePac` speaks its exact protocol (`/pac/init`, `/pac/state` with the `stale` cached-frame rule, `/pac/next`, `/pac/assets/*` for cubby's face textures): the isometric glow grid, the face-textured sphere, energy/steps gauges, hearts, ghost-fear, the **Plutchik cone compass** (our neurochemistry → petal/tier/color/message, tier metadata read from `cubbyverse/core/emotion/plutchik.json` by path), letter pellets spelling the level word (exact assignment rule), and **speech through CubbyTalk**: a grounded word (letters collected) is SAID when its percept recurs — ghost near, power-up, discovery, new maze, solved — via `CubbyChat.mediate`, so even in-game utterances pass the VM's ASK. Nine string patches, each pinned by a test so an upstream change surfaces: endpoint prefixes, the subtitle/foot (say what drives him), and two HUD rows relabeled to numbers we actually have (`world model: facts/derived/walls`, `refused moves`) instead of R-STDP weights we don't. **The real goal (owner, 2026-09-02): spatio-temporal orientation, threat and how to fight it, and self-improvement WITHOUT forgetting.** What backs each: *orientation* — he plans over his OWN map (BFS across the neighbor facts he holds, level-scoped so nothing goes stale), sees pellets glow within two cells as sighting facts, keeps sightings across out-of-time retries (attempt 2 knows where they are); *threat* — a danger radius that GROWS with learned fear (+0.7 per catch, ×0.9 per cleared level — the original's rule), routes planned around ghost neighborhoods, a flee move that maximizes distance, hunting them when frightened; *self-improvement* — he **generates his own programs**: a JUMP composed when a seen pellet has no path (hazard-walled), and on out-of-time or plain curiosity a new composition sampled from a pattern grammar (slots A/B/C, length 2–5, instantiated over any direction assignment — DASH/BLINK/COMBO are three points in that space; `COMBO-AABA`, `KNIGHT`, `WARP`… are moves the original never had), each written as CubeLang, VM-certified, then offered by the ASK like any exit and taken when his map says it shortens the way; proposal lengths are weighted by what paid off. **Nothing is forgotten**: the `ProgramLibrary` keeps every program with its source, a **reasoning trace** (trigger, the situation down to hormones and emotion, the sampling rationale, the VM verdict) and every use with what it earned; useless patterns are *retired* (no longer offered) never deleted; persisted as `standin/data/out/cubbyman_programs.json` + a readable `.md` notebook (also at `/pac/programs.md`) and reloaded on restart. Measured in a 2-stage sim (`seed 0`): stage 1 cleared with ghosts on; JUMP at step 2, `COMBO-AABA` invented on out-of-time then used 55× saving 165 steps. **Arbitrary CubeLang (`forge.py`, 2026-09-02): his trunk writes programs for live tasks.** A `Task` is a prompt in one of the trunk's trained families (the story words never reach the VM; the comparison semantics do) plus an expectation he can check himself — the emitter writes the program, the VM runs it, the result is certified, and the program is kept as a tool (prompt, source, verdict, pass/fail) in the same notebook. In play: a **FLEE decision** program on the live numbers when a ghost is inside his radius (sensor template; the trained kernels answer a confidence *class*, ≥50 = true, measured over the SFT set), a **SAFER-EXIT compare** program to pick where to run, a **WHERE chain** at each new maze checking his bearings against his own map. He acts on a program's answer only when certified against his rule; otherwise the rule stands and the notebook records the rejection. **Measured acceptance of the v3 Q4 stand-in (`scripts/forge_probe.py`, 24 tasks, `data/out/forge_probe.json`): decision 0.75 (6/8), chain 0.50 (4/8), compare 0.00 (0/8).** Failure modes on record: compare programs declare `result = 0` and forget the `add result, lhs` in the branch; two decisions had `add x : 1` colon-for-comma parse errors; chain misses bind a neighboring coordinate. Game-phrased prompts scored 0/8, 1/8, 0/8 before the tasks were rephrased in the trained templates — the trunk is narrow, and the certification is what makes using it safe. **v4 BUILT (2026-09-02, `data/build_game_sft.py`; on Drive at `cubbyllm/standin/emitter_sft_v4.jsonl`, sha `333141dd…`): 11,144 records = the whole v3 set as replay (9,284, unchanged — nothing learned is lost) + 1,860 game records the VM verified** (1,876 generated, 16 compose programs dropped as gold mismatches — all 4- and 5-step compositions whose frame holds 5–6 bound roles: `recover` returned the role token `H6_NAME` instead of the name, a VM frame-capacity edge, so the set only teaches compositions the VM certifies): flee/go decisions on live numbers in the game's words (414), safer-exit/shorter-path compares on the small numbers v3 failed (353), 1- and 2-hop neighbor chains over level-scoped cells (532 + 152), exit counts (177), superpower compositions (231) — plus the 1 certified FLEE tool his own notebook held from live play (the "he generates his data" path, literally). Game records carry `repeat` 2 in train; 101 in val. Train with `notebooks/standin_emitter_sft.ipynb` (`VERSION='v4'`, the default now), then locally `eval_emitter_vm.py --data standin/data/out/emitter_sft_v4.jsonl --gguf <v4>` (game families reported apart) and `scripts/forge_probe.py --gguf <v4>` — the before/after acceptance is the result. **v4 TRAINED (Colab, 2026-09-02; `standin/models/emitter_v4.Q4_K_M.gguf`, Drive `emitter_lfm25_2p6b_v4/gguf_gguf/`). Forge probe before → after: decision 0.75 → 1.00 (8/8), compare 0.00 → 1.00 (8/8), chain 0.50 → 1.00 (8/8)** (`data/out/forge_probe.json`) — and the same **24/24 on a held-out maze** (level 5, seed 3; v4 trained on levels 1–3; `forge_probe_v4_level5.json`), so the chain number is not memorized cells. The compare programs now carry the `add result, lhs` the v3 ones forgot; chain answers keep the `cell x-y-z` form. The game generated the data; the trunk learned the game's programs. The `pacman_live.py` splice (its `Game.step()` → `planner.plan_to_any` decision point taking the brain's ASK-mediated choice) is the cubbyverse-side follow-up. |
| `serve_api.py` + `demo.html` | The HTTP surface the cubbyverse front-end talks to (stdlib, JSON, CORS, localhost): `POST /turn` → the turn record (reply, kind, register, **emotion**, state, route, task/learn detail); `GET /state` → the full neurochemistry read (the emotion-compass feed); `GET /events?since=N` → **the console feed** (`CubbyBrain.trace`: user/sense/route/walk/emit/vm/gate/learn/explore/probe/derive/speak — every reasoning step and action as it happens); `GET /worlds`, `GET /health`; `GET /` serves `demo.html`, a self-contained dark **console window** — chat on the left, the live event stream + hormone bars + emotion on the right. Transport only — the brain stays `serve.CubbyBrain`. `python standin/serve_api.py --gguf standin/models/emitter_v3.Q4_K_M.gguf` then open http://127.0.0.1:8765/. |
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
