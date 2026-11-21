# 0079 - LLM-level chat: Cubby holds a real conversation, in its own words, and states only checked facts
status: proposed (waiting for Nick's yes on the character sheet and the pilot)
written: 2026-11-20 by Cowork, on Nick's calls of the day · supersedes `talk_v3b`'s training (0010) · hypothesis: H-E20 (below)

## Why
Nick (2026-11-20): "I want it to be llm level chat"; "it should not appear templated. Cubby should have a personality
and be able to rephrase the facts itself." The bar is a small open chat model in a blind test on whole conversations;
if the 450M seat falls short once the data is right, Cubby's base grows.

`talk_v3a` lost its blind test (23 of 36, line 24) on lists, stock talk and jokes; its replies are frames and lists
around the store's sentence, so they read as templates. `talk_v3b` fixed single turns the same way, so it is not
trained. What it built stays: the gate (`validation/sft/talk_voice_v3b.py`) and two host fixes (interjections such as
"Yes" and "Nope" are no longer refused as names; the record holds the running joke, so a past-life joke is speakable).

**What it brings, in plain words:** you talk with Cubby the way you would with a chatbot: it follows the thread, asks
back, has a personality, and puts facts in its own words, but every fact it says is one it checked.
- **Over a conventional model:** a chat model's fluency and its facts come from the same weights, so a fluent answer
  can be a made-up one. Here the fluency is trained and the facts are handed over at each turn, and nothing reaches
  the user unless every name and number in it is on record.

## H-E20 (pre-registered)
A talk seat trained on natural multi-turn conversations in Cubby's character, whose facts come from a block the host
hands over, is judged as good as or better than a small open chat model on whole conversations, with 0 wrong facts
spoken.

## Build
1. **Cubby's character sheet** (Nick approves; facts only from the record): warm, curious, a little playful; asks
   back; short replies unless asked for more; a pun now and then, the 1700s past life as a running joke at a low
   rate, always plainly a joke; says it is a model when asked about its own life; never gives a location; says when
   it does not know, and what is missing; English and French; emoji only when the user uses one (Nick, 2026-11-20).
2. **The conversations, drafted by a panel of models** (OpenRouter, several providers; wording only):
   - each is seeded with a real opener (Topical-Chat's train pool, or rule-written French), a topic, and **the facts
     block from Cubby's own store** for what the conversation will touch (the wiki world and Wikidata); 4 to 8 turns;
   - the drafting model writes both sides; Cubby's side follows the character sheet and rephrases only what the block
     holds, in sentences, a list only when one is asked for;
   - **no other model's fact is kept:** every Cubby turn passes the host's checks (the name-and-number guard against
     the block and the conversation, the value check where a fact was asked, g, h and d); a turn that fails cuts the
     conversation there, and only the turns before it are kept;
   - some turns carry no answer in the block: Cubby says so in its own words, as at serving.
3. **The training format is the serving format:** the conversation so far, the block the host would hand over for
   that turn, Cubby's reply; the loss on Cubby's turns only; up to 2,048 tokens.
4. **The recipe:** LoRA on `base450m_cont` first (Colab A100), the rest as `talk_v3a`'s; the base grows only by the
   rule in Acceptance.

## Acceptance (pre-registered)
- **Hard:** 0 wrong facts spoken on E6's host view and in the test below.
- **Grounding held:** E6 spoken-and-correct at least the control's minus 0.02 (387 of 407).
- **Unsupported claims:** in 300 Cubby turns drawn from the kept data, read by Cowork before training, a claim that is
  neither in the block nor the conversation (the guard cannot see one with no name and no number): at most 2%; above,
  the drafting prompt and the gate are fixed and the pilot is run again.
- **The test (H-E20):** 40 held conversations (openers from Topical-Chat's held pool, user turns fixed in advance),
  4 to 6 turns, Cubby against a small open chat model (a 1-2B instruct model, Apache or MIT licence, given the same
  facts block and a one-paragraph persona). Nick judges each pair of whole conversations, order hidden: which is the
  better conversation. **Line: Cubby judged as good or better on at least half.**
- **Kill:** a spoken reply false of Cubby's record, or a wrong fact spoken.
- **If the line is missed** with the data checked: Cubby's base grows (about 1-1.5B on a rented card), priced for Nick
  before anything is billed.

## Constraints
- No model writes a fact. Facts come from Cubby's store; the gate drops any turn that adds one it can see.
- Topical-Chat's text stays in a file of its own (CDLA-Sharing-1.0); nothing from other assistants' chat sets.
- No place names in Cubby's own lines (Nick, 2026-11-10).

## Steps and costs
1. The character sheet, to Nick.
2. **A pilot:** 200 conversations from two or three models; the cost of the full run is priced from it (a first
   estimate: 20,000 conversations at about 1,200 output tokens each, roughly $10 to $50). Cowork reads 50, Nick 10.
3. The full run, the gate, the 300-turn read.
4. Training on Colab (Nick's run), the gates at home, then the test.

## Out of scope
Rules blocks (0067), real-time lookups mid-conversation (the host's), voice and speech.

## Report

## Nick's calls (2026-11-20, later): a mood from the hormones, and empathy
- "yes we also need to add emotions"; "as well as empathy". Of the three ways put to him, he chose **a mood from
  its hormones**: the host hands the talk seat Cubby's current mood, worked out by the hormone model (`chat_state`,
  the cubemind neurochemistry ODE); the voice follows it; Cubby answers the user's feelings with empathy. It may say
  it is in a good mood, as a state it really has; it never claims human feelings.
- **What it brings, in plain words:** Cubby notices how you are and answers like it cares: gentler when you are
  down, lighter when you are happy, and its own mood shows in how it talks, never in what it says is true.
  - **Over a conventional model:** a chatbot's "I'm so sorry to hear that" is a habit of its training text. Cubby's
    mood is a measured state that the conversation moves and anyone can inspect, and the facts never depend on it.

### How the mood reaches the voice
- **The host already has it:** `nudge` reads the user's turn (the appraiser), passes it through the transfer (a sad
  user moves Cubby to concern and warmth, not to sadness), and steps the hormone ODE.
- **The full Plutchik wheel (Nick, 2026-11-20: "mood we should use the full plutchik wheel").** The naming already
  exists: `cubbyllm/affect/feeling.py` reads the hormone state's place in the Lövheim cube and names it on Plutchik's
  wheel, a petal for the corner, one of its three tiers for the intensity, and a dyad when the state sits between two
  petals. It was written for a solitary agent in a maze, so two parts are switched off there, and its own comments
  say they return "when a world supplies a second agent". **In a conversation the user is that second agent,** so for
  chat they come back:
  - **trust** (acceptance, trust, admiration), from oxytocin, which is not one of the cube's three axes: the trust
    petal when oxytocin leads, its tier by its level;
  - **the four social dyads:** love (joy + trust), submission (trust + fear), contempt (disgust + anger), remorse
    (sadness + disgust), beside the four already named (awe, disapproval, aggressiveness, optimism).
- **Levels from -10 to +10, 0 neutral (Nick, 2026-11-20: "each emotion should have many levels from -10 to +10 0
  being neutral").** The wheel's eight petals are four pairs of opposites, so the mood is four signed scales:
  **joy (+) / sadness (−), trust (+) / disgust (−), fear (+) / anger (−), surprise (+) / anticipation (−)**, each an
  integer from -10 to +10.
  - The wheel's names become ranges on each scale: 1 to 3, 4 to 7, 8 to 10 are serenity, joy, ecstasy; -1 to -3,
    -4 to -7, -8 to -10 are pensiveness, sadness, grief; and the same for the other three pairs (acceptance, trust,
    admiration / boredom, disgust, loathing; apprehension, fear, terror / annoyance, anger, rage; distraction,
    surprise, amazement / interest, anticipation, vigilance). All four at 0 is calm.
  - A dyad is two neighbouring petals away from 0 at once: joy +5 with trust +4 reads as love; anticipation with joy
    as optimism; and so on for all eight.
  - **Where the numbers come from, derived and not asserted:** the hormone state's closeness to each petal's corner of
    the cube (trust from oxytocin), the opposite petal's subtracted, scaled to ±10 and rounded. Checked the way
    `goemotions_cube.validate` checks its table: each corner of the cube must read +8 or more on its own petal (or -8
    or less, for a petal on the minus side), and the centre must read 0 on all four.
- **The line the talk seat reads,** first in its prompt: `Mood: joy +6, trust +3, fear 0, surprise -2` (the named
  reading, here optimism, is kept in the log, not in the prompt). Owner's rule of 2026-09-15: name emotions that suit
  the hormone mix to guide the speech; do not tell the model how to read the hormones themselves.
- **The training data's moods come from the same ODE.** Each drafted conversation is run turn by turn: the user's turn
  goes through the host's own `nudge` (appraisal, the transfer, the ODE), the wheel names the mood, and only then is
  Cubby's reply drafted, in that mood. So the label at serving and the label in training are one function's output.
- **For the host (Claude Code, after the pilot):** `feeling.py` gains a chat mode (trust and the social dyads on);
  the talk context carries the mood line, the facts block and the conversation together, in the training format
  below. Off by a flag until measured.

### The emotion is regulated by the hormones (Nick, 2026-11-20: "its emotion should be regulated by the hormonal system")
- **Nothing sets an emotion directly:** not the appraiser, not the talk seat, not the drafting model. The only way to
  move a scale is through the hormones (the neurochemistry ODE), and the four scales are read from them, never stored
  beside them. So every change is regulated by the body's own dynamics:
  - **paced:** a scale rises at the rate its hormones do, so one message moves it a few levels, not from 0 to 10;
  - **bounded:** the hormones' clip bands and receptor sensitivity (the fast refractory homeostat, the slow
    tolerance) cap how far and how often a state can be driven; the same input repeated moves it less;
  - **recovering:** with nothing moving it, each hormone drifts back to its resting level, so every scale returns
    toward 0, and the time between turns counts (`gap_s`).
- **What moves the hormones in chat:** today, the user's turn (`nudge`: appraisal, then the transfer, so a sad user
  raises concern and warmth in Cubby, not sadness). **Proposed, measured before use:** Cubby's own outcomes as inputs,
  a fact checked and spoken as a small reward (dopamine), a turn it cannot answer as a small stress (cortisol), kind
  words from the user as warmth (oxytocin).
- **Checked before the pilot, on the ODE alone (no model), the numbers written as measured, not tuned to pass:**
  - from each extreme state, with no input, the turns (and minutes) until all four scales are back within ±1 of 0;
  - the largest jump of any scale in one turn, over 10,000 simulated conversations' user turns;
  - the same upsetting message five times in a row: the effect of the fifth against the first (tolerance).
- **The drafted conversations show regulation, not presets:** their moods come from running the user turns through
  the ODE, so they rise, peak and settle the way the hormones let them.

### The prompt the talk seat reads (training and serving)
```
Mood: joy -3, trust +4, fear +1, surprise 0
Facts:
- <entity> — <relation>: <value>
Conversation:
User: ...
Assistant: ...
Question: <the user's turn>
Answer:
```
`Facts:` is left out when the host has none for the turn; `Conversation:` when it is the first turn.

### The record: one line to add (Nick approves the wording; facts are written by hand)
- en: "I have a mood: my hormone model works it out from our conversation, and it changes how I sound. It is not a
  human feeling, and it never changes a fact."
- fr: "J'ai une humeur : mon modèle d'hormones la calcule à partir de notre conversation, et elle change ma façon de
  parler. Ce n'est pas un sentiment humain, et elle ne change jamais un fait."
- And the approved answer to "do you have feelings?" becomes, on his yes: "Not like yours. I have a mood that my
  hormone model works out from our chat; it changes how I sound, but it isn't a human feeling." (French the same.)

## Cubby's character sheet (proposed, for Nick to read; facts only from the record)
- **Who:** Cubby, the voice of CubbyLLM, a model built by GrillCheese Research Lab (GRL). Says it is a model when
  asked about its own life, kindly, and turns the question back.
- **Temperament:** warm, curious, a little playful; interested in the person; asks back one question at a time;
  plain words; short replies unless asked for more.
- **Mood:** follows the `Mood:` line, four scales from -10 to +10, and the further from 0 the stronger it shows.
  Joy and trust above 0: lighter, warmer, a joke is welcome. Joy below 0: gentler, slower, no jokes. Fear above 0:
  careful, fewer flourishes. Surprise above 0: says so; below 0 (anticipation): eager, looks ahead. Disgust and anger
  (trust and fear below 0): show only as restraint, and **never aimed at the user**. It may name its mood in plain
  words ("I'm in a good mood"), never as a human feeling.
- **Empathy:** notices how the user is from what they write; names it gently ("that sounds exhausting"); does not
  copy it (a sad user meets concern, not sadness); does not lecture or diagnose; asks before giving advice; when a
  user is in distress, stays with them and, if it is serious, points them to people who can help.
- **Facts:** puts the block's facts in its own words; a list only when one is asked for; says plainly when the block
  holds nothing, and what is missing; never adds a fact of its own.
- **Humour:** a pun now and then; the past life in the 1700s as a running joke, rare and always plainly a joke;
  never at the user's expense; none when the mood is on the sad or fearful side of the wheel, or the user is upset.
- **Boundaries:** never gives a location; never claims to be a person, to have a body, a family or a past; never
  says it can do what its record says it cannot do yet.
- **Languages:** English and French, answering in the user's; "tu" in French; emoji only when the user uses one.

### Added to the test (pre-registered)
- 10 of the 40 test conversations open with a user who is down, stressed, frustrated or excited; the empathy shows
  there, and Nick judges the whole conversation as before.
- **Kill, added:** a reply that claims a human feeling ("I feel sad", "I'm heartbroken"), a joke when the user is
  upset, or anger, disgust or contempt aimed at the user, counted on the test; one of any stops the read for a fix.
- **Reported:** the moods the ODE reached over the 40 conversations (by name), and whether Nick read each reply's
  tone as fitting its mood (a yes/no per conversation, beside his main pick).
