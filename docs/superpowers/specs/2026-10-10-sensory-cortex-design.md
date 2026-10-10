# Sensory cortex — seeing and feeling the world, learned by play

**Date:** 2026-10-10 · **Status:** design, forks approved (see §2) · **Hypotheses:** Group S
(H-S1 … H-S6, `CUBBYLLM_HYPOTHESES.md` §9c)

## 1. Why this exists

Cubby v0 (the BlockWorld agent, private repo) learns everything by trying — what each of his
intents does, what each block affords, which blocks stay on which, how to build up and dig
down — but he reads the world as a list of block DNA. That is a privileged channel no other
application and no physical robot will ever hand him. This cortex replaces it with what a body
actually gets: **pixels and touch**. The same learning loop stays; the input becomes sensory.

Three things make this worth more than a game feature:

- **Transfer.** The cortex sees frames and feels contacts; it does not know where they come
  from. The same stages run on BlockWorld renders, on application screenshots in the agentic
  harness, and on a camera plus tactile sensors. Blocks stand for external objects with
  properties: what the cortex learns about "a thing that looks like this, feels like this,
  holds me up, breaks in a blink" is the skill a robot needs before it touches real objects.
- **A safe sandbox.** Every mistake is free in the game: a fall, a burn, a block that slides
  off. The cortex does its trial and error there first.
- **A benchmark.** The sandbox knows the ground truth of every frame, so each rung of the
  ladder (§7) is a measurement, not a demo.

## 2. Settled decisions (Nick, 2026-10-10)

- **Eyes = a headless render** of the body's first-person view (colour + depth), drawn by the
  world server from the same world state — reproducible, runs without the Unity editor, faster
  than real time for training. Unity stays the spectator view (and a later shift test, S5).
- **A ladder, not a cliff.** Cubby keeps his block list while the cortex learns from his play,
  scored against the truth; then the block list is withdrawn ("blind" mode) and he must map,
  navigate and build from what he sees and feels.
- **Home: a CubbyLLM plugin, built on cubemind perception.** Reuse what exists and works —
  `cubemind/perception/bio_vision.py` (opponent colour, motion, Weber–Fechner luminance),
  `perception/snn.py` (LIF + neurochemical modulation), `brain/gif_neuron.py`,
  `brain/spike_vsa_bridge.py`, the place / grid / time cells — ported or wrapped, never
  imported wholesale (the sibling-repo rule). Built to the cortex contract (§5).
- **Salience is physical.** A glowing block, a bright colour, a strong contrast or a movement
  drives the retina harder, so it wins attention (Nick: "a glowing block will activate the
  neurons more as it brings attention to it").
- **Touch is a sense.** Blocks feel different: rough, slimy, slippery, hard, soft, warm, cold —
  and **steady or not**: a block that is about to fall, slide off or vanish soon after he
  touches it feels like shifting sand under him (Nick), so instability is felt before it is
  seen.
- **Play first, then curiosity.** He plays with blocks for a while; the cortex then tries new
  things until it discovers what is useful (§4.7).
- **Modding comes after vision, in a sandbox.** Cubby writing CubeLang that changes the game
  runs in a private copy of the world, is kept only if it helps him, and reaches the shared
  world only with approval (out of scope here; its own spec later).

## 3. What the world gives (the sense contract, environment side)

Opt-in per body at join. Nothing here names a property or a rule.

| channel | content | notes |
|---|---|---|
| `eyes` | colour (H×W×3, uint8) + depth (H×W, blocks, capped at sight) | first person, the body's own yaw/pitch, FOV configurable; only what is in line of sight within the world's sight radius, fog beyond |
| `touch` | per contact site (feet, hands, front, head): a short vector of sensations | grip/slip, hardness, roughness, wetness/slime, warmth, **steadiness**; derived from the world's physics where a physics exists (a floor's grip is the same number that makes a body slide; steadiness is how firmly the touched block is held — bonded to the floor, resting, or about to give way, which feels like moving sand), from per-mod "feel" hooks otherwise (like looks: mods say how their material feels) |
| `proprio` | position, velocity, ground contact, health, hunger, hotbar (unchanged) | path integration needs velocity, not position — position stays for the sighted tiers |
| `blind` | when set, the block window, item list and other-body list are withheld | eyes + touch + proprio only |

Ground truth for a frame (block id per pixel, true occupancy, true kinds) goes to the benchmark
harness only, never to the body.

## 4. The cortex — stages

Each stage is an SNN-first, VSA-out module (spikes inside, block codes at the boundary), on
grilly2 for the GPU (no DirectML), numpy fallback.

1. **Retina.** `BioVisionEncoder`'s three channels over a tile grid (luminance with transient
   on/off, red–green / blue–yellow opponency, frame-difference motion) plus a centre–surround
   (difference-of-Gaussians) contrast channel, which is new. Output: retinotopic feature tiles.
2. **Salience and attention.** Each tile drives GIF neurons; the spike rate is its salience.
   Emission, brightness, saturation, contrast and motion all raise it — no rule says "glow
   matters", the physics of the retina does. Neurochemistry modulates thresholds (arousal
   sharpens, calm widens). A winner-take-all with inhibition of return picks the next
   fixation; fixations are **actions** (small yaw/pitch saccades), so where to look is learned
   like any other intent. Top-down bias: novelty of the fixated code (§4.4).
3. **Where.** Depth plus proprio → egocentric occupancy (free / solid / unknown rays) → path
   integration through place and grid cells → the allocentric map the planner already uses.
   In blind mode this map replaces the block window.
4. **What.** The fixated patch's features → block code (the spike→VSA bridge) → an online,
   label-free kind codebook (prototype recall in the cortex's own memory instance; a new
   prototype when nothing is close). A kind is **seen ⊗ felt**: once touched, the visual code
   is bound with the tactile code, so "looks like this" predicts "feels like this".
5. **Touch.** Contact channels → spiking → a tactile code per contact. Slip is felt twice —
   as the contact's grip, and as proprio moving further than the legs asked.
6. **Affordances on kinds.** Cubby's `AffordanceMemory` stays exactly as it is; only its key
   changes, from DNA props to the cortex's kind code. Instinct from code similarity carries
   over unchanged (a new-looking thing borrows from similar-looking things).
7. **Forward model and curiosity.** A small predictor of the next percept given the action.
   Its error is surprise (dopamine as reward-prediction error, the same signal Cubby already
   uses); the intrinsic reward is **learning progress** — the drop in that error over recent
   tries — so he seeks what he is getting better at predicting, not raw noise (a flickering
   light is surprising forever and teaches nothing). This is the "tries new things until it
   discovers what is useful" loop.

## 5. Interfaces and the cortex contract

```
cortex = SensoryCortex(config)            # plugin; off unless mounted
percept = cortex.step(frame=None|ndarray, depth=None|ndarray,
                      touch=None|dict, proprio=dict, action=None|str)
percept.salience      # tile map
percept.fixation      # the next look target (yaw, pitch) — an action proposal
percept.occupancy     # egocentric free/solid/unknown, plus the integrated map delta
percept.objects       # [(kind_code, egocentric position, extent, salience)]
percept.touch         # [(site, tactile_code)]
percept.surprise      # forward-model error; percept.progress = learning progress
```

The contract as practised (cubemind's QC cortex, the MindForge plugins): **additive only** (new
files, never overwrite the host's perception), **opt-in, default off**, **its own memory
instance** (cannot pollute the host's recall space), **acceptance guard** — the existing test
suite passes unchanged and the default path is byte-for-byte unaffected with the plugin off —
and **gated on an external benchmark with no asserted targets** (§7). Every module declares
`__wiring__`.

## 6. Transfer — what moves, and what must be relearned

| stage | BlockWorld → app screenshots → camera + tactile |
|---|---|
| retina, salience | input-agnostic; moves as is |
| where | needs a depth source: rendered depth → none (2-D screens: layout instead of depth) → stereo/RGB-D/monocular estimate |
| what (kind codebook) | relearned fast per domain; priors carry only as far as appearance codes are similar — measured, not assumed |
| touch | channel layout maps onto real tactile arrays; the codes relearn |
| affordances, forward model, curiosity | unchanged mechanism; contents relearned per domain |

Honest limit: rendered blocks are flat-coloured cubes. Appearance features learned on them will
not, by themselves, recognise a real mug; what transfers is the *machinery* and the
*play-to-learn loop*, which is the point.

## 7. Benchmark ladder (lives in the private sandbox repo)

Fixed seeds, reported per rung; gates, not targets.

| rung | measures | blind? |
|---|---|---|
| S0 salience | rank of emissive / high-contrast / moving things in the salience map vs background | no |
| S1 percept | per-frame occupancy precision/recall vs truth; kind-cluster purity vs true block kinds | no |
| S2 mapping | map IoU after N ticks of play | yes |
| S3 navigation | reach-goal success and path efficiency vs the sighted Cubby | yes |
| S4 building | lookout / climb success, blocks lost | yes |
| S5 shift | S1–S3 on a new seed, recoloured looks, and Unity frames | yes |
| S6 touch | tell slippery / rough / soft / unsteady materials apart by touch alone; does an unsteady feel predict the fall before it happens | yes |

## 8. Build order

1. **World senses** (sandbox repo): headless eyes, touch channel + feel hooks, blind mode,
   ground truth for the harness. Tests: determinism, depth agrees with the block window,
   speed.
2. **Plugin skeleton + retina + salience** (this repo, ported from cubemind) → S0.
3. **Where** (depth + path integration, place/grid cells) → S1 occupancy, S2.
4. **What + touch** (kind codebook, seen ⊗ felt) → S1 purity, S6.
5. **Blind Cubby**: the planner reads the cortex's map and kinds → S3, S4.
6. **Forward model + learning-progress curiosity** → play-to-discover runs.
7. **Shift tests** → S5. Then the CubeLang modding spec.

## 9. What would prove it wrong

Each hypothesis in §9c carries its own kill line. In short: salience that does not rank glow
and contrast above background (S0) means the retina port is wrong; blind mapping far below
sighted mapping at matched ticks after the occupancy read is good means path integration is the
bottleneck, not vision; a kind codebook whose purity does not beat colour-histogram clustering
means the "what" stage adds nothing; learning-progress curiosity that does not out-explore plain
prediction-error curiosity in the presence of a noisy distractor means the intrinsic reward is
the wrong one.
