# 0010 — Casual prose from Cubby's own 450M: a talk adapter taught by human dialogue, no external model anywhere
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("fix prose so we don't need an external model to word
everything in casual sentences"; data route chosen: open human dialogue corpora + rules) · hypothesis: H-E10 (+ H-E15)

## Why
The 450M already holds both seats with nothing third-party in the loop (H-E15, 2026-10-17: `Cubby450mTalk`, adapter
`talk_v2_2`), and spoke 0 wrong. But it answers like a form. Per H-E10, the adapter sees the host's canonical ask, 63%
of its mix is fact questions in one sentence shape, and the value check allows one value. So the LFM stand-in still
does the casual wording. H-E10's voice step planned "local dialogue sources, and a one-time model rewrite". Nick's
call, 2026-10-22: **no model rewrite**. The casual voice comes from what people wrote, and the facts come from the
host, checked as today.

- **What it brings:** Cubby talks casually in its own words, from its own 450M, with the stand-in retired from the
  talk seat.
- **Advantage over conventional models:** a chat model is fluent because it learned to say anything. Here the
  fluency is learned from people, and every fact in a reply is still tied to the host's block by the claim check,
  so sounding natural never buys a made-up fact.
- **In plain words:** it chats like a person, and it still only tells you facts it actually has.

## Build
- **Sources, licenses first.** Before any row is used, read each dataset's license file and record it in a
  manifest (`standin/data/talk_v3.manifest.json`). Prefer licenses that allow commercial use, and ask Nick about
  any non-commercial one (DailyDialog and EmpatheticDialogues may be NC: check, don't assume). Candidates:
  - **Topical-Chat** (Amazon; 10,784 human conversations grounded in Wikipedia, fun facts and news: a knowledge
    snippet beside the casual sentence that uses it, which is the supervision this needs; check its DATALICENSE);
  - **OASST2** (Apache-2.0; human-written replies; keep the human-written turns only);
  - **Claire** (French, already wired in `gap_families.py`; per-subset licenses);
  - **WikiLarge** (Simple-English Wikipedia sentences aligned with the original sentences they rewrite: human
    rewrites, CC BY-SA; Nick has access on the Hub). This is the family for explaining in its own words: a
    reply that composes facts in Cubby's voice, never copying a source;
  - the existing chat sources.

  Exclude every machine-generated set (SODA, synthetic chats).
- **Records (`standin/data/build_talk_mix.py --mix v3`).** Three kinds:
  1. **Knowledge → casual.** From Topical-Chat, the snippet, rewritten as the host's facts block, and the human
     turn that states it as the target. Kept only if the target passes the claim check against that block.
  2. **Casual chat** (no facts) from the human corpora, EN and FR, under the existing voice rules.
  3. **Today's fact records**, reweighted down from 63%. The share is chosen to hold `talk_v2_2`'s
     spoken-and-correct, not to change the mix for its own sake.

  Every fact target passes the name-and-number guard and the claim check. No target is model-written.
- **Training:** a LoRA talk adapter over `base450m_cont`, the same recipe as `talk_v2_2` (`notebooks/talk_sft.ipynb`;
  the builder `_build_talk.py`), as `talk_v3`. The notebook is for Cowork to run on Colab. Serving changes nothing
  but the adapter name.

- **An optional arm, best of N:** draft N replies and keep the most natural one that passes the claim check.
  This is the "natural selection" step of `hybrid.py` in Nick's `empirical_grilly_next` project (a checkout beside this one; path in `CLAUDE.local.md`). Report its cost against its gain on the blind A/B.

## Acceptance (pre-registered)
- **The battery:** 300 held-out turns. 200 are fact turns (150 EN, 50 FR) from the H-E8/H-E10 dev sets and the
  wiki world; 100 are chat turns.
- **Hard lines:** **0 wrong spoken** with the guards on. Spoken-and-correct on fact turns ≥ `talk_v2_2`'s on the same
  battery, minus 0.02.
- **Casual** (the point):
  - **Nick's blind A/B** on 60 pairs (`talk_v3` against the LFM stand-in, order hidden): `talk_v3` judged as casual
    or more casual on ≥ 50%. The harness shows the pairs and records his choices.
  - **Proxies**, reported, not gates: the share of fact replies in the canonical one-sentence shape (target ≤ 30%,
    against `talk_v2_2`'s, measured first); distinct openings per 100 replies; contractions per reply; mean length.
- **Kill:** spoken-and-correct drops more than 0.05, or any wrong spoken. Fluency bought with grounding is not
  this order.
- Tests: the builder's records all pass the guards; the manifest lists every source with its license; full
  `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold: no external LLM on the serving path, **and here none in the data either** (Nick). No
  bulk downloads without Nick's yes: Topical-Chat and OASST2 are small (tens of MB); confirm before fetching.

## Out of scope
- The emitter seat. Multi-turn dialogue shape (H-E10 step 2, its own order). The loci copy inside prose: short fact
  blocks sit in-window, so attention copies them (run 1's in-window text read 0.987).

## Addendum (2026-10-22, Cowork): an emotional-support family
Nick named ESConv and ExTES.
- **ESConv** (1,053 conversations; the supporters are crowdworkers trained on the ESC framework, the seekers describe
  real problems; 8 strategy labels on supporter turns): human-written, so it qualifies as a candidate for a supportive
  register, the supporter turns as targets. CC BY-NC 4.0 and "academic research use only": non-commercial, so ask Nick before any
  row is used, as this order already requires.
- **ExTES** (11,177 dialogues written by gpt-3.5-turbo from 87 human seeds): excluded by this order's own rule (no
  model-written targets).
- **Before any supportive family is served:** Cubby is not a therapist. One of ESConv's problem types is ongoing
  depression; a reply to someone at risk must point to professional or crisis help, never role-play treatment. That
  route is a prerequisite of serving this family, and it is not built in this order: ask Nick before training it.

## Addendum 2 (2026-10-22, Cowork): MENTOS, the supporter's plan
MENTOS (KUNLP; github.com/KUNLP/MENTOS) annotates 12,917 supporter turns of ESConv (train 8,959 / dev 1,907 / test 2,051), each with its four-turn context, with the supporter's own mental state: a belief and an intent in free text, and an emotion on Plutchik's eight (with their opposites) plus 14 mixed emotions (dyads such as hope = joy + anticipation, guilt = joy + fear), each 0-3. The labels were written by gpt-4o; people checked 100 dialogues (Gwet's AC1 0.6-0.8). CC BY-NC 4.0, research use only (it follows ESConv's licence).
- For the supportive family (Addendum 1), its intent per supporter turn is a plan stated before the words, the way
  the emitter plans a program before the talk adapter speaks (H-E18). A candidate record: context -> intent ->
  ESConv's human-written supporter turn, the turn as the target and the intent as conditioning. The intents are
  model-written, so they are conditioning only, never a target. Same prerequisite as Addendum 1: a crisis route first.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Addendum 3 (2026-11-06, Cowork): the queue, the data on disk, and the licence rule
Nick, 2026-11-06: "fluency and conversation, we have plenty of chat datasets".
- **Queue:** right after 0055, then 0059 (conversation), then 0020 and 0057. The knowledge base stays first.
- **The licence rule** (Nick's choice):
  - non-commercial sets (CC BY-NC and similar) are allowed in research runs only, never in an adapter that is served;
  - served adapters use commercial-safe sets only;
  - every licence is read and recorded in the manifest before a row is used, as this order already says.
- **On disk, human-written, candidates** (each licence checked first; NC is expected where marked):
  - OASST2 (`standin/data/hf/oasst2_train.parquet`): the human-written turns only;
  - DiaBLa (`standin/data/hf/diabla/`): English-French human dialogues;
  - Claire (`standin/data/hf/claire/`): French, per-subset licences;
  - DailyDialog (`roskoN/dailydialog`): NC expected, research only;
  - EmpatheticDialogues (`Estwld/empathetic_dialogues_llm`): NC expected, research only. Check first that its turns
    are the original human turns, not a model rewrite.
  - ReDial (`community-datasets/re_dial`): movie-recommendation dialogues.
- **Still to fetch, small:** Topical-Chat (the knowledge-grounded family), with Nick's yes as this order says.
- **On disk, excluded** (model-written, by this order's rule):
  - SystemChat (`systemchats_train.parquet`);
  - French Alpaca (`french_alpaca_train.parquet`);
  - open-instruct;
  - the model rollouts in the instruct folder (all-rollouts, deepseek, gemini).

  They may serve as evaluation prompts only, never as training records.
- **Evaluation only:** lmsys toxic-chat and the two prompt-injection sets feed the guards' tests, not the talk mix.

## Cowork's addendum (2026-11-10): talk templates may be model-written; facts never are
Nick, 2026-11-10: "the talk templates could be model writen", "facts are not model writen". This changes one line of
this order ("No target is model-written") and nothing about the facts.

**In plain words:** Cubby's turns of phrase, greetings and jokes can be drafted by other models and picked by us.
What Cubby tells you as a fact still comes only from its own checked records.
- **Over a conventional model:** a model's wording and its facts come from the same place, so a fluent sentence can
  carry an invented fact. Here the sentence has a hole, and only a checked fact can fill it.

### The rule now
- **A template** is wording with no fact in it: a greeting, a reaction, a way of saying "I don't know", a question
  back with a `{name}` slot, a sentence that hands over an answer with a `{fact}` slot, a pun, a line of the
  past-life joke. A template may be model-written.
- **A fact** is whatever fills a slot, and any statement about the world. A fact is never model-written. It comes
  from the host's block, and the name-and-number guard and the claim check run on every fact target as before.
- **A template carries no fact of its own.** A candidate with a real name, place, date, number or event outside its
  slot is dropped. The past-life joke stays vague or absurd.
- **Each row records where its wording came from:** human dialogue (this order's sources), a model-written template
  (with the model's id) or a rule. The manifest counts each.
- **The human-dialogue route stays** for how a conversation flows. Templates add Cubby's own character on top.

### The first candidates
`validation/panel_talk_lines.py` asked 8 models for 100 lines each in English and French (Nick: puns, and a running
joke about a past life in the 1700s): 1,602 lines, 1,573 with no flag, $0.78. They sit in
`standin/data/out/talk_lines/` (git-ignored), in nine kinds: greet, small, frame, unknown, which, pun, pastlife,
cheer, bye.

### To settle before training (Cowork with Nick)
- **Which lines are kept:** Nick reads and picks. The voice is his character.
- **How often the jokes come:** a pun or a past-life line is a flourish. It never replaces an answer and never sits
  in a refusal about something serious. The rate is a setting; Cowork proposes a low default and Nick sets it.
- **Provider terms:** some providers restrict training on their models' outputs. The manifest keeps each line's
  model, so one provider's lines can be dropped in one step. Nick decides.
- **Where it trains:** rented serverless GPUs (Runpod), run by Cowork, with the price stated and Nick's go before
  anything billable starts.

### For the paper
The preprint says no external model writes training records. From the version that uses this adapter, it reads: no
external model writes a fact or a reasoning record; the talk adapter's wording templates were drafted by external
models and picked by a person.

### Nick's target replies (2026-11-10: "example of what I want cubby to say")
Two replies from his chat mockup are the target for tone and shape. They are kept in `chat/target-replies.md` in the
CubbyUXReference checkout beside this one: a greeting that uses the person's name, what Cubby can do as a short
list, a wrong assumption corrected kindly and directly, several facts as a numbered list, a warm closing line.

- **Second round of templates, for whole replies:** `validation/panel_talk_shapes.py`, same panel, English and
  French: 963 templates, 954 with no flag, $0.77 (`standin/data/out/talk_lines/shapes.jsonl`, git-ignored). Kinds:
  hello_name, about, correct, unconfirmed, answer_list, invite, remark. Every name, fact and ability arrives through
  a slot (`{user}`, `{subject}`, `{facts}`, `{abilities}`).
- **What must differ from the mockup's wording, because facts are not model-written:**
  - **What Cubby can do** is filled from Cubby's own record of its abilities, approved by Nick. Two lines of the
    example are not true of Cubby today (explaining any difficult topic; writing, planning and organising).
  - **Every fact in a reply is in the store.** "The first person to win two" and "the only person to win in two
    fields" are claims about everyone: Cubby says them only when it can derive them by counting over the store.
  - **"Did not" needs a contradiction.** Cubby corrects a wrong assumption when its records contradict it (here,
    her recorded citizenships). When the records are silent it says it cannot confirm it (0017; the `unconfirmed`
    templates).

### The identity set that exists, and what it needs (Nick: "we already made some ... needs modifications tho")
The cubby-lm checkout beside this one holds `identity/identity_facts.yaml` (July 2026) and four files under
`data/identity/`: `cubby_identity_core.jsonl` (60 rows, English and French, expanded from that YAML by rule) and
three behaviour files (744, 2,019 and 4,665 conversations).
- **The YAML is the right base:** facts about Cubby written by Nick, expanded by rule. It becomes Cubby's own
  record of itself, the source for `{abilities}` and for every "who are you" answer.
- **Statements to bring up to date** (each for Nick's yes):
  - the role, "the language cortex of the Grilly agent", and "the lab behind the Grilly stack";
  - "I run and train on Grilly ... no torch in my trunk" (the trunk is trained with torch and served on Vulkan);
  - "what can you do": today it names no real ability;
  - "I'm small right now and still scaling up" (CubbyLLM is never described as a small model);
  - the greeting "What are we building today?", against Nick's "What's on your mind today?".
- **To add:** greeting by name, the past-life joke and the puns as part of the character, corrections of a wrong
  assumption, list answers.
- **The three behaviour files are not used for this adapter.** Their replies are other assistants' answers, facts
  included, so they fall outside "facts are not model-written". They also hold private conversations and stay out
  of any shared dataset.
