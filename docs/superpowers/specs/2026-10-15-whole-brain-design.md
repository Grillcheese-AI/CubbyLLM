# The whole brain: senses, cortices and language learning together

**Date:** 2026-10-15 · **Status:** design, the principle approved (Nick, 2026-10-15) ·
**Hypothesis:** H-S8 (`CUBBYLLM_HYPOTHESES.md` §9c) · **Builds on:** the sensory cortex
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
  → workspace                 the trunk's residual stream: each cortex ADDS its contribution
  → expert routing            shared attention; per-sense and language experts (§4.4)
  → language cortex           the trunk (the 450M base, or its grown 1.29B form)
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

Percept tokens are interleaved with text tokens in time order, so the trunk's attention sees
"what happened" and "what was said" on one clock. Each trunk layer gets **experts per sense**
next to its language FFN, attention stays shared (Mixture-of-Transformers, Liang et al. 2024,
arXiv 2411.04996; BEiT-3's multiway layers, arXiv 2208.10442):

- **Rung 1, routed by organ.** A percept token goes to its sense's expert, a text token to the
  language FFN. Every expert writes back into the same stream, so each later layer reads the sum.
- **Rung 2, routed by content.** A router reads the workspace and picks experts by what the
  stimulus is (the neurons that recognise it), so a word about touch can use the touch expert and
  a felt slip can wake a language expert. Kept only if it beats rung 1 at equal compute.

### 4.5 Zero-exit admission

Every new cortex, projection and expert joins with its **exit at zero** (its output into the
residual stream starts at 0), the operator that made growth exact
(`cubbyllm/model/grow.py`; exact to 1e-6 nats, `validation/logs/grow_step0_*`). On day 0 the
brain is the trained language model, unchanged; from then on the senses earn their place.

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

## 7. Build order

1. **Exporter** (private sandbox, local): aligned episodes + grounded descriptions → shards on
   private Drive. Determinism test: the same seed exports the same bytes.
2. **Cortices and workspace** (`cubbyllm/model/senses/`): per-sense encoders, organ-routed
   experts, zero-exit admission. Test: with percepts present and exits at zero, the text logits
   equal the base's.
3. **Trainer** (`validation/train_brain.py`): the balanced loss, text replay, the three arms,
   checkpoints that resume across Colab sessions, G1–G3 evaluated in the run.
4. **Colab notebook**, thin, as for the base: arms J, S, T.
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
text model and fine-tune the adapters between them; the senses never shape each other. Here every
part learns from one prediction objective on one workspace, each cortex can be read on its own
(what it sees, predicts, gets wrong), and the brain starts exactly as good as the language model
that passed the gates.
