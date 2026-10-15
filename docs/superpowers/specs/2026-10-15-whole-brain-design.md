# The whole brain: senses, cortices and language learning together

**Date:** 2026-10-15 · **Status:** design, the principle approved (Nick, 2026-10-15) ·
**Hypotheses:** H-S8 … H-S11 (`CUBBYLLM_HYPOTHESES.md` §9c) · **Builds on:** the sensory cortex
(`2026-10-10-sensory-cortex-design.md`), the base and its growth
(`docs/research/2026-09-25-grow-450m.md`)

## 1. Why this exists

Nick, 2026-10-15: *"Our brain does not train one cortex at a time; it is the sum of all the senses
/ cortex information that drives learning. Input → neuronal stimuli → neurons learn to recognize
the stimuli and route to other cortex (eyes, ears, a recognition cortex, linked to the LM cortex
after) → the sum of all cortex residuals."*

Today the parts learn apart. The sensory cortex learns kinds and feels online, with no gradient
from language; the language trunk learns from text alone, with no sense of anything. A word for
a glowing block and the look of a glowing block never meet. This design puts every cortex on one
shared workspace and one learning signal, so that what each part learns is shaped by all the
others, as in a brain.

### What it is for: the advisor loop (Nick, 2026-10-15)

1. **Given** G, a goal; P, past trials toward it (possibly none); H, a history of past choices,
   related to the goal or not.
2. **Asked:** "I want to do G, I tried P, given that in the past I did H: what are my options, and
   which would be best for me?"
3. **Cubby answers in detail:** the valid options that reach G, each with its pros and cons; the
   one he thinks best by the user's criteria; or, when the choice depends on criteria he does not
   know yet, the question that would settle it.

Every part of this design serves that loop:

| Step | Done by |
|---|---|
| G, the goal | a wanted consequence (§4.7) |
| P, past trials | the postmortem memory (§4.8): what was tried, what happened, which cause was blamed |
| H, past choices | the user's criteria as revealed by their choices, and the context they set |
| the options | conditional mode (§4.8): the inverse head proposes acts that would bring G about, the forward head plays each out; a trial in P is dropped, or kept with what changes and why |
| pros and cons | each branch's predicted consequences, criterion by criterion (time, cost, risk, reversibility, what is learned) |
| the best option | the branch that scores highest under the user's criteria |
| a question instead | asked when the top branches swap places on a criterion the user has not weighted: the one question whose answer changes the pick |

It is learned where every answer can be checked: in the sandbox, each listed option and each
stated pro or con is run in a fork of the world (§5), so an option is valid because it reached G,
not because it sounds plausible. Outside the sandbox the same answer is checked by what can verify
it there (the reasoning VM, the knowledge store, MoWM's worlds), and Cubby says so when nothing can.

## 2. Settled decisions (Nick, 2026-10-15)

- **One brain, trained together.** Every cortex, language included, learns from the first step
  of the same run. No stage trains one cortex alone and freezes it before the next.
- **The sum of the cortices is the state.** Each cortex adds what it knows into one residual
  stream (the workspace); every later part reads the sum.
- **Neurons route.** What a stimulus is decides which cortex handles it: first by the organ it
  came through (eyes, ears, skin, body), then by what the stimulus is (§4.4).
- **Training on Colab when it is faster.** The sandbox stays on the local machine (it is private);
  only the exported data goes to private Drive. Training and its evaluation run on a Colab A100.

## 3. What the brain analogy asks for, and the one place it needs care

- **It holds:** the senses learn together because they keep predicting each other (a cup seen
  predicts the cup felt). Cubby's cortex already does a small version: sight predicts touch
  (H-S3, `validation/logs/s1p_s6_what_2026-10-10.json`).
- **It needs care:** a brain does not open every area at full plasticity at once; primary sensory
  areas mature before association and language areas. In machine learning, modalities trained
  jointly from scratch compete, and the one with the strongest gradient starves the others
  (Wang, Tran & Feiszli 2020, arXiv 1905.12681; Huang et al. 2022, arXiv 2203.12221). So
  everything learns from step 0, with **balanced losses** and **zero-exit admission** (§4.5), not
  a cortex at a time.

## 4. The design

### 4.1 The path of a stimulus

```
organs (fixed, no learning)   retina · depth · cochlea · skin · proprio · needs (soma)
  → stimuli                   spike rates / feature maps per organ, per tick
  → sensory cortices          small learned nets per sense: what is this (kind, sound, place)
  → workspace                 each cortex ADDS its contribution: the sum of the cortices
  → expert routing            per-sense experts, by organ, later by content (§4.4)
  → language cortex           the trunk (the 450M base, or its grown 1.29B form), reading the
                              workspace through gated cross-attention, with Cubby's acts and
                              their consequences on the same clock (§4.7)
```

### 4.2 Organs (fixed)

The retina, salience, depth and touch channels of `standin/cortex/` as they are (H-S1, H-S4), and
the body's needs (air, food, warmth, health) as the soma reports them. **Ears** (rung 2): the
sounds of the sandbox's events rendered headlessly from the same recorded effects the client
plays, panned by direction and attenuated by distance, read as log-mel frames.

### 4.3 Sensory cortices (learned)

One small network per sense turns a tick's stimuli into a few **percept tokens**, vectors the
width of the trunk: eyes (colour + depth + salience) → 8 tokens per frame; skin + body + needs →
2 tokens; ears → 2 tokens (design values, to be sized by the first read). The online kind
codebook and the forward model of the view stay; their outputs (kind code, surprise, progress)
are features these cortices read.

### 4.4 Workspace and routing

Each tick, every cortex adds its percept tokens into the **sensory workspace** (the sum of the
cortices). The language trunk runs on its own sequence, the words plus one `<tick>` position per
tick of an episode so it keeps the world's clock, and reads the workspace through **gated
cross-attention** every few layers (Flamingo, Alayrac et al. 2022, arXiv 2204.14198). Percept
tokens are not interleaved into the trunk's own sequence: the shared attention and the MinGRU
recurrence would carry them into every later word, and day 0 would no longer be the language
model (§4.5). Each sense has its own **expert** on the way in (Mixture-of-Transformers, Liang et
al. 2024, arXiv 2411.04996; BEiT-3's multiway layers, arXiv 2208.10442):

- **Rung 1, routed by organ.** A sense's tokens pass through that sense's expert; the trunk reads
  all of them, summed.
- **Rung 2, routed by content.** A router reads the trunk's state and the workspace and picks
  which experts a position reads, by what the stimulus is (the neurons that recognise it): a word
  about a slippery floor reads the touch expert, a felt slip wakes language. Kept only if it beats
  rung 1 at equal compute.

### 4.5 Zero-exit admission

Every new path joins with its **exit at zero**: each cross-attention read is scaled by tanh(α)
with α = 0 (Flamingo's gate; a scalar gate is ReZero's, arXiv 2003.04887), and each expert's
output projection starts at 0, the operator that made growth exact (`cubbyllm/model/grow.py`;
exact to 1e-6 nats, `validation/logs/grow_step0_*`). On text, day 0 is the trained language
model exactly; on an episode, the trunk starts by ignoring the senses, and the `<tick>` positions
are the only new input. From then on the senses earn their place.

### 4.6 One learning signal: prediction everywhere

L = λ_text · (next word) + Σ_sense λ_sense · (next percept of that sense)
  + λ_x · (each sense predicting another: seen → felt, heard → seen, felt → named)

- Every term is a prediction of what comes next, the forward-model principle of the cortex
  (`standin/cortex/forward.py`) applied to every sense and to language.
- **Balance:** the λ follow each term's own uncertainty (Kendall, Gal & Cipolla 2018, arXiv
  1705.07115), with a floor so no sense's share of the gradient falls below a fixed fraction.
- **Language is protected:** text replay from the base's token cache in every batch (design
  value 50%), and gate G3 (§6).
- Later, Cubby's needs join as value: what relieved a need is remembered as worth predicting
  (H-S5's learning progress decides which experiences are replayed).

### 4.7 Event → action → consequence (Nick's ARC loop)

Nick, 2026-10-15, after a sketch from Gemini, modified. The loop is about **Cubby's own acts**:

- **Event E_t:** what the senses and the needs report at tick t (the workspace).
- **Action A_t:** what Cubby did at t, his motor intent (walk, jump, attack, use, eat …) as an
  action token at the tick position: his act, not the stimulus.
- **Consequence C_t+k:** what the world and his body report after: the next percepts, the change
  in his needs, the events (broke, ate, fell, got hurt).

Heads at each tick position read the trunk there (which has read the workspace):
- **forward** (E, A) → C, predicted before it is seen and scored after: §4.6's next-percept term,
  now conditioned on the act;
- **inverse** (E, C) → A: which act brings this about, how a wanted consequence becomes an act;
- **value** C → its effect on the needs (relief or strain, the soma's currency): what a
  consequence is worth;
- **language:** E, A and C in words.

**Counterfactuals.** At sampled ticks the sandbox forks the world and tries the other actions
from the same state, so the data holds the true consequence of each act, not only of the one
taken: interventions, which no text corpus contains.

Gemini's mapping (stimulus = action, routing = reaction, residual = consequence) names steps of
the internal computation; trained on it, the brain would learn what follows what, not what an act
causes. Here the reaction is the act the brain chose; at rung 3 it chooses the act by the
predicted consequence and its value (the one-currency choice proposed for Cubby's mind, which
then replaces the hand-written goal recipes).

### 4.8 Critical thinking: conditional and postmortem modes

Nick, 2026-10-15: after an act or a world event, a wrong prediction is examined and corrected for
similar events (**postmortem**), and before acting the possibilities are weighed in branches
(**conditional**). Three modes, chosen by uncertainty and stakes (habit against deliberation,
arbitrated by uncertainty: Daw, Niv & Dayan 2005, Nature Neuroscience 8:1704):

- **Habit (default).** Act on the forward head's best prediction. No branches.
- **Conditional (before).** When the forward head is unsure (its own predicted error is high) or
  the stakes are (a need could drop, the act cannot be undone: stepping off a ledge, mining the
  block underfoot), branch: for each candidate act, and for each likely outcome where the world
  itself is uncertain, roll the forward model k ticks from a fork of the state (exact and cheap,
  H-P1), score each branch by value + learning progress − risk, keep the top few (MoWM's
  "a future per world, plausibility × value, top-k kept"; Group P's branching futures, opened
  only where uncertainty is high, H-P2). The branches are kept for the postmortem. In words: "if
  I mine this I drop into water; if I go round, I lose time but stay dry".
- **Postmortem (after).** When a consequence differs from its prediction by more than that
  prediction's own uncertainty (a surprise):
  1. **Locate** the error: which channel was wrong (a percept, a need, an event).
  2. **Blame** a cause: which feature of the event or the act the prediction leaned on, found by
     re-entering the moment with that feature changed (counterfactual re-entry, H-P5; the
     sandbox's forks give real answers). The arithmetic school's rule, blame at the first wrong
     step and name the error's kind (H-E19), applied to the world.
  3. **Correct, twice:** a fast **exception memory** keyed by the event's code, which the forward
     head consults at the next similar event (one-shot, no gradient), and a **replay priority**
     so the slow weights learn it (prioritized replay, Schaul et al. 2015, arXiv 1511.05952;
     H-S5's learning progress decides which lessons are still worth replaying).
  4. **Say it**, checked against the sandbox's truth: "I expected the floor to hold; it was sand
     with nothing under it; sand like that falls."
  - **Noise is not an error.** If similar events disagree with each other, the lesson is "this
    is uncertain" (the predicted spread widens), not a flipped prediction: the noisy-TV lesson
    (H-S5) applied to hindsight.
  - **Conditional mode has its own postmortem:** did the chosen branch's predicted value come
    true? That calibrates the branching itself.

**Prototype first in Cubby's current mind** (local, no GPU): his move operators and affordance
memory already make predictions and check them. The postmortem there is blame plus an exception
memory over those predictions; the conditional is branch rollouts over the learned operators at
risky decisions. His present "death lessons" (a raised setpoint, a shorter safe drop) are a
hand-written postmortem; this generalises them. The same two modes then move into the brain's
heads.

## 5. Data

- **Aligned multisensory episodes**, exported from the private sandbox: per tick the organ
  outputs (frames every 4 ticks), the body's needs, the events, and Cubby's talk. Sighted and
  blind agents, many seeds. The dataset stays on private Drive and never enters this repo (it
  carries the sandbox's mechanics).
- **Grounded descriptions, generated from the sandbox's ground truth by templates** ("a glowing
  block two steps ahead", "the floor is slippery", "he is cold"), so every sentence is true of
  its moment and checkable. No language model writes them.
- **Text replay** from the base's 13-shard token cache.
- First read: 2,000 episodes of 3,000 ticks (design value), exported on the local machine with
  one process per core.

## 6. How it is measured (pre-registered with H-S8)

Three arms at equal compute:
- **J, joint:** everything learns from step 0 (this design).
- **S, separate then glued:** each sensory cortex trained alone on its own prediction loss, then
  attached to a frozen trunk with only the projections trained.
- **T, text only:** the trunk continued on the text replay alone.

Gates, on held-out episodes and held-out seeds:
- **G1, grounded language:** from the percept tokens of a moment, answer templated questions
  (what is ahead, what it feels like, is he hungry or cold, is something hurting). T can only
  guess. **J ≥ S + 5 points.**
- **G2, language helps perception:** kinds named in the descriptions are recognised from fewer
  views, and the feel of a kind touched for the first time is predicted better than the cortex
  alone does today (first-touch error 0.175, `validation/logs/s1p_s6_what_2026-10-10.json`).
  **J better than S.**
- **G3, no loss of language:** J's held-out text loss within +0.02 nats of T's.
- **Kill:** J no better than S on both G1 and G2, or G3 failed. Then the cortices stay separate
  and are joined by the cheaper route (S).

For the ARC loop (H-S9), against **J-shuffled** (J with the action tokens shuffled across ticks in
training, so the act carries no information about its consequence):
- **G4, acts predict their consequences:** on held-out forks, the forward head's error on the
  consequence of each tried act is at least 20% below J-shuffled's.
- **G5, consequences name their acts:** the inverse head picks the act from (E, C) at least twice
  as often as chance, and above J-shuffled.
- **Kill:** no gain over J-shuffled on G4; the brain learned what follows what, not what an act
  causes.

For critical thinking (H-S10), first in Cubby's current mind, then in the brain:
- **G6, postmortem corrects:** on held-out surprise pairs (a kind that looks like a familiar one
  but behaves differently, met twice), the error on the second encounter is at least 50% lower
  with the postmortem than with replay alone at equal compute; on pure-noise events the
  postmortem widens the predicted spread instead of flipping the prediction (calibration no
  worse than without it).
- **G7, conditional pays:** at risky decisions (ledges, hazards, water, unknown kinds), branching
  cuts harms plus deaths by at least 30% against acting on the best prediction alone, within the
  same decision-time budget, and the chosen branch's predicted value correlates with what it
  delivered.
- **Kill:** the postmortem is no better than replay alone on G6, or it flips predictions on
  noise; or branching is no better than habit on G7.

For the advisor loop (H-S11), on held-out sandbox scenarios (G, P, H) with every option and
claim checked in forks, against two baselines at equal compute: the trunk answering without
branching, and a general language model answering from the same description:
- **A1, valid:** at least 90% of the listed options reach G in a fork.
- **A2, complete:** at least 80% of the distinct ways the forks find to reach G (within the step
  budget) are listed.
- **A3, true:** at least 90% of the stated pros and cons hold in the forks.
- **A4, the right pick or the right question:** with the criteria given, the pick is the
  best option under them at least 80% of the time; without them, Cubby asks when the top options
  swap places on an unweighted criterion (at least 80% of such cases) and does not ask when they
  do not (at most 10%).
- **A5, uses P and H:** a failed trial from P is proposed again without saying what changes in at
  most 5% of answers; the pick agrees with the preference revealed by H at least 70% of the time.
- **Kill:** no better than the no-branching trunk on A1 and A3; or no gain from P and H over the
  same answers given without them.

## 7. Build order

1. **Exporter** (private sandbox, local): aligned episodes (percepts, needs, events, Cubby's acts)
   + grounded descriptions + counterfactual forks → shards on private Drive. Determinism test:
   the same seed exports the same bytes.
2. **Cortices and workspace** (`cubbyllm/model/senses/`): per-sense encoders, organ-routed
   experts, gated cross-attention, zero-exit admission. Test: with the gates at zero, the text
   logits equal the base's, and on an episode the senses change nothing.
3. **Trainer** (`validation/train_brain.py`): the balanced loss, the ARC heads, text replay, the
   arms, checkpoints that resume across Colab sessions, G1–G5 evaluated in the run.
4. **Colab notebook**, thin, as for the base: arms J, S, T, J-shuffled.
5. **Rung 2:** ears, then content routing.

Compute (computed, not measured): at the base's measured 46.4k tok/s on an A100
(`docs/research/2026-09-25-grow-450m.md`), 2,000 episodes × ~10k tokens plus as much text
replay is ~40M tokens, ~15 minutes per pass; a few passes per arm keep each arm to about an
A100-hour.

## 8. Risks

- **Modality competition** despite balancing: watched per term, every eval.
- **Template language is narrow:** G1 can be passed by a pattern-matcher. Held-out seeds and
  question forms the training never used guard it.
- **The sandbox is not the world:** rendered blocks are flat-coloured cubes (H-S6 tests the shift).
- **The private mechanics must not leak:** data and captions stay on private Drive; this repo
  holds code and numbers only.

## 9. What it brings to CubbyLLM

A language cortex whose words are tied to what is seen, felt and needed, not only to other words,
learned from the same prediction signal as every sense, and admitted without costing the
language it already has.

**Advantage over conventional models:** multimodal LLMs attach separately pretrained encoders to a
text model and fine-tune the adapters between them; the senses never shape each other, and every
model learns from observation only. Here the brain also learns from its own acts and from the
sandbox's counterfactuals, what an act causes, which observational text cannot teach. Here every
part learns from one prediction objective on one workspace, each cortex can be read on its own
(what it sees, predicts, gets wrong), and the brain starts exactly as good as the language model
that passed the gates.
