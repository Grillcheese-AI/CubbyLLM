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
