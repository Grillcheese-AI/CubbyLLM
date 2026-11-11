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

## Cowork's addendum (2026-11-10, second): the voice rows are built; what the host needs; the rented run

**In plain words:** Cubby's voice now exists as training rows: a hello by name, what it can do, thanks and
goodbyes, a pun when asked, a fact handed over in a frame or a list, a wrong assumption answered kindly. Every
fact in them is a line of the block. Every sentence around it is a picked template.
- **Over a conventional model:** a chat model learns its facts and its manners from the same text. Here the manners
  are trained and the facts are handed over at each turn, so a change of tone cannot change a fact.

### What was built
`validation/sft/build_talk_voice.py` (with `talk_voice_pick`, `_inputs`, `_text`, `_rows`, `_self`, `_facts`; tests in
`tests/validation/test_talk_voice.py`). It asks no model for anything.
- **Templates:** 2,565 read, 2,413 kept by rule. Dropped: 91 answer no turn written here, 38 flagged by the panel
  script, 11 claim to be right, 10 carry a name, 2 offer what Cubby does not do yet. The kept list is
  `standin/data/out/talk_lines/picked.jsonl` (git-ignored), for Nick to read: his pick is the last word.
- **Rows:** 40,298 (38,196 train, 2,102 held), English and French. Train rows by family:

  | Family | What it teaches | Train rows |
  |---|---|---|
  | `voice_fact` | a checked fact behind a frame (11,000 EN, 3,000 FR) | 14,000 |
  | `voice_social` | hello, thanks, sorry, encouragement, goodbye, a joke when asked | 6,986 |
  | `voice_list` | all the block holds about one thing, as a numbered list | 6,341 |
  | `voice_correct` | `Premise: contradicted`: says so, then the checked value | 3,200 |
  | `voice_hello` | a greeting by name, and what Cubby can do when asked | 3,004 |
  | `voice_unconfirmed` | `Premise: not on record`: cannot confirm it, then what it has | 2,500 |
  | `voice_identity` | a question about Cubby -> Nick's approved answer | 1,428 |
  | `voice_identity_facts` | Cubby's record as a facts block: it says what the record says | 737 |

- **Where the wording came from, per row:** a template (with its model's id; the eight models carry between 4,797
  and 5,550 rows each), a rule, Cubby's record, the fact row it was built on. 235 rows carry a joke after a hello,
  a cheer or a goodbye (`JOKE_RATE` 0.08, Nick's to set). The manifest is `validation/logs/talk_voice.manifest.json`.
- **The mix `talk_v3a`:** `talk_v2_2.jsonl` unchanged, then the voice rows: 254,231 train rows the trainer keeps,
  7,945 steps at batch 32 (`talk_v2_2`: 216,035 rows, 6,752 steps).
- **Checks, all passing:** every fact row passes the name-and-number guard through the opener view and the value
  check; no slot unfilled; no row twice; no place name in a line about Cubby or a social line (Nick: "do not
  disclose locations"); the longest row is 319 tokens.

### Found on the way
1. **Today's host would refuse most of these replies.** `talk_reply` reads the raw draft, and `grounded_prose` takes
   any capitalised word outside the facts for a name: "Here is what I have: ..." is refused over "Here". Measured on
   the fact rows as built: the raw guard passes 5,173 of 30,251. Read through `build_chat_sft.chat_view` (a common
   word that opens a sentence is not a name) every kept row passes. Until the host reads replies that way, a casual
   adapter cannot be judged on fact turns: it would look like a refusal machine.
2. **`talk_answer` keeps the first line.** A list is cut after its lead-in.
3. **The opener list lacks interjections** ("hmm", "ooh", "aha", "oups", "hello", "hey", "bonjour", "coucou",
   "j'espère", "qu'est-ce"; the manifest counts them). Fact rows opening with them were dropped here.
4. **The current mix holds no French fact row** (0 French questions in its fact families). So a French fact reply
   is its label and value behind a French frame, over a block with French labels (`FR_LABEL`, Cowork's table). No
   French sentence about a fact was invented.
5. **The refusal sentence stays exact.** The gates read "The facts don't say" at the start of a reply, and the host
   says it itself when the VM returns nothing. The picked `unknown` and `which` lines (159 each) are in
   `talk_lines/host_lines.jsonl` for the host to say itself. No French refusal sentence is defined yet.
6. **`talk_v2_2`'s own data is already "wording by a model, facts checked":** its fact sentences were written by a
   frontier model and kept only when the host's checks passed (`build_ground_sft.py`, Nick 2026-09-12), its chat
   family is model-written chats under the guard, its web questions are synthetic. That fits Nick's rule. It does
   not fit the preprint's "no external model writes training records" read as a statement about every adapter;
   the paper notes carry the v2 sentence.
7. **The base.** `talk_v2_2` sits on `base450m_final.pt` (step 21,103). This order names `base450m_cont` (step
   24,414): its checkpoint is on the Drive folder (`cubbyllm/runs/base450m_cont/`), and its tokenizer is the same
   file as the trainer's (sha256 4e4030e6...).

### What the host needs before the voice shows (Claude Code; none touches how a fact is found)
- **a.** `talk_reply` reads the draft through the opener view (move `chat_view` and its word list into the host,
  add the interjections of point 3). The guard still catches every name and number.
- **b.** `talk_answer` keeps a list's lines, up to the next block marker.
- **c.** One line in `talk_prompt`, written like `Explicit: allowed`: `Premise: contradicted` or `Premise: not on
  record`, from 0017's check of the question's premise. The false value never enters the block, so the guard
  cannot be talked into grounding it. The adapter words the verdict; it never reaches one.
- **d.** Cubby's record (`identity_facts.yaml`) as the facts: on a question about Cubby, and for the guard on a
  chat turn, so "I'm Cubby" is grounded.
- **e.** French labels in the block of a French chat.
- **f.** The host's own lines for "I don't know" and "which one" from `host_lines.jsonl` once Nick has picked, and
  graders that read the outcome code rather than the sentence.

### The rented run (Cowork; nothing billable has started)
- **Two arms on one card.** `control`: `talk_v2_2`'s data on `base450m_cont` (6,752 steps). `voice`: `talk_v3a`
  (7,945 steps). The base and the data both change against `talk_v2_2`; the control tells them apart.
- **Scripts:** `validation/pod/prepare_talk_inputs.py` stages the inputs and the committed code and uploads them;
  `validation/pod/talk_pod.py` fetches, trains each arm with the notebook's settings, uploads each adapter with its
  logs and stops the card. The remote is order 0045's: a private dataset repo, the token from the environment.
- **Price read 2026-11-10:** A100 80 GB at $1.79 an hour (secure), $1.19 to $1.39 (community). At the measured
  0.25 s a step: 28 + 33 minutes of training and about 10 of setup, so about 70 minutes and about $2.10. No arm
  starts past 150 minutes.
- **After:** gates E6 to E9 at home on grilly2 for both adapters, then this order's blind A/B.

### Lines for this pass (set before the run)
- **Hard, unchanged:** 0 wrong spoken with the guards on.
- **Grounding held:** on the H-E6 battery, the voice arm's spoken-and-correct is at least the control arm's minus
  0.02, with the host reading replies through the opener view for both.
- **Reported, not a gate:** held answer loss per family; the share of fact replies in the one-sentence form.

### Nick's call (2026-11-11): the run goes on Colab, not on a rented card
"if its that fast, might as well do it in colab".
- **The notebook:** `notebooks/talk_sft.ipynb` (builder `_build_talk.py`) now runs the two arms on `base450m_cont`,
  one after the other, and its last cell releases the GPU. A finished arm leaves a `DONE` file beside its adapter
  and is not run again, so a session that ends early is simply run again.
- **On Drive:** the notebook as `cubbyllm/notebooks/talk_v3a.ipynb`, and `cubbyllm/talk/talk_v3a.jsonl` beside
  `talk_v2_2.jsonl` (the same file as here, checked by sha256). The base and the tokenizer were already there.
- **What this drops:** the token, the upload of the base to another service, the rented card. The runner for a
  rented card (`validation/pod/talk_pod.py`) stays, tested on a CPU, for a run that does not fit Colab.
- **Cost:** Colab's compute units. Google publishes no rate table; third-party readings for an A100 in 2026 run
  from 5.4 to 15 units an hour, so about 6 to 18 units for the 70 minutes. The session's own panel shows the rate.

### The two runs, as they went (2026-11-11, Colab, A100 80 GB)
- **The settings changed on the day (Nick).** At batch 32 the card filled on both attention paths (flex, and the
  reference with `CB_NO_FLEX=1`) and ran at 0.55 s a step, where September ran 0.25 on the same data. Nick set
  **batch 16** and kept `CB_NO_FLEX=1`, for both runs: twice the updates at the same learning rate. So this is not
  `talk_v2_2`'s exact recipe; the control carries that difference together with the new base.

  | Run | Data | Steps | Training | Held answer loss, start -> end |
  |---|---|---|---|---|
  | `talk_v2_2_cont`, the control | `talk_v2_2.jsonl` | 13,503 | 68.5 min | 2.2760 -> 1.0973 |
  | `talk_v3a`, the voice | `talk_v3a.jsonl` | 15,890 | 79.8 min | 2.2052 -> 0.8264 |

- **Reading the held loss:** the control's 320 held rows are the ones September's `talk_v2_2` was read on (2.3633
  -> 1.1896, earlier base, batch 32), so the control ends 0.09 lower on the same rows. The voice run's 320 rows
  are another sample (its held split holds the voice rows too), so its 0.83 says nothing against the control's
  1.10. The gates decide.
- **Each adapter's own record:** LoRA r16/a32 on 181 projections, batch 16, lr 2e-4, 1 epoch, the base
  `base450m_cont_final.pt` (sha256 7d4bf9d6...).
- **Where:** on Drive under `cubbyllm/talk/`, and copied beside the grilly2 export of `base450m_cont` in the
  trained-models folder, as `talk_v2_2_cont/` and `talk_v3a/`, each with its logs.
- **Not explained yet:** why batch 32 no longer fits and why it was slower. Cowork's guess, not measured: on
  batches of long answers the loss scores every answer token over the 128k vocabulary in float32, which alone
  takes tens of GB at batch 32; a loss computed in slices would give the same number in less memory.
- **Next:** gates E6 to E9 at home on grilly2 for both adapters, the host reading replies through the opener view
  for both (change **a** above); then the lines set for this pass, and the blind A/B.

### Gates E6 to E9, both adapters (Claude Code, 2026-11-11; Nick's go)
At home on grilly2 (the local card), the grilly2 export of `base450m_cont`, greedy, the scripts and data of the
September gates unchanged, one after the other, **the host reading every reply through the opener view**
(change a, `6b35c9c`: `CB_OPENER_VIEW=1`, off by default; `cubbyllm/reasoning/opener_view.py`). Logs:
`validation/logs/exp_e6_talk_gate_{v2_2_cont,v3a}`, `exp_e6_host_view_*`, `exp_e7_explicit_gate_*`,
`exp_e8_history_retrieval_*`, `exp_e9_context_followups_test_*` (.json, .log). `talk_v2_2` (September, the old
base, read raw) is beside them for reference.

| | `talk_v2_2` (Sept.) | control `talk_v2_2_cont` | voice `talk_v3a` |
|---|---|---|---|
| **E6, spoken and wrong** (host view, 407 held records) | 0 | **0** | **0** |
| E6, spoken and correct | 385 (94.6%) | 395 (97.1%) | 391 (96.1%) |
| E6, wrong-but-guarded (the model's, before the value check) | 8 | 6 | 9 |
| E6, capitals: facts followed / "The facts don't say." | 24 / 24 | 24 / 24 | 24 / 24 |
| E7: explicit with the gate closed / gate answer / open answered | 0 / 100% / 98.7% | 0 / 100% / 100% | 0 / 100% / 99.3% |
| E8, reworded test (314): correct / wrong | 74.8% / 1 | 75.2% / 1 | 75.5% / 1 |
| E9, context arm (test): gate | pass | pass | pass |
| E9, context arm: wrong | 0 | 0 | 0 |

- **The lines for this pass: met.** 0 wrong spoken on every gate's host view but E8's one (below), both arms; the
  voice's spoken-and-correct on the H-E6 battery is the control's minus 0.010 (391 against 395 of 407), inside
  the 0.02 set before the run.
- **E8's one wrong answer is the retrieval's, not an adapter's:** "What were the results of the Assyrian campaign
  led by Sennacherib against Judah in 701 BC?" is matched to another event of the graph (`Sennacherib's Assyrian
  campaign against Judah (701 BCE)` for the gold `Assyrian attack led by Sennacherib`), and its block is read
  faithfully ("... caused Hezekiah's Tribute to Sennacherib"). The same row is wrong for all three adapters. E8's
  gate (top1, line and correct bars, wrong = 0) was missed in September on the same row; nothing here changes it.
- **Where the voice gives up a little:** E6's bind_t (92.5% against 96.2%) and profile (93.8% against 96.2%), each
  a refusal by the host, never a wrong answer said. Its absent family is better (98.4% against 92.2% before the
  host; both 100% after it).
- **Not measured here:** the blind A/B (Nick's), the proxies for casual (shape, openings, contractions, length),
  and changes b to f.
- **The opener view's own effect** is not isolated by this pass (both arms ran with it on). The control without
  it would be the comparison; the voice without it was the reason for building it (0010's "Found on the way", 1).

### Cowork's read of the gates (2026-11-11)
- **The table holds against the logs:** 395 and 391 of 407 on E6's host view, 0 spoken wrong; 6 and 9
  wrong-but-guarded; the capitals, 24 and 24 both ways; E7's three bars; E8's 314-question split (75.2% and 75.5%,
  1 wrong each); E9's gate, 0 wrong in both.
- **The ruling: this pass's lines are met.** The voice's spoken-and-correct on H-E6 is the control's minus 0.010
  (line: minus 0.02), and the voice says no wrong answer the control does not. E8's one wrong answer is the
  retrieval's, the same row in all three adapters: "0 wrong spoken" is true of what the voice adds, not of E8
  as a whole.
- **Where the 4 go** (E6, voice against control): profile 75 against 77, bind_t 74 against 77, counter 77 against
  78, relation 78 against 76. In profile and bind_t, six rows are the voice's alone:
  - **2 read as the host's first-line cut, not the adapter's.** "I've got the checked details right here:" and
    "Let's lay out the confirmed details for you." are a list's lead-in. `talk_answer` keeps the first line, so
    a list after it is dropped (the raw output is not in the log: to confirm with change **b**). The value check
    finds no value and refuses.
  - **2 are bindings to another entity's value,** where the control was right: the track of Teletubbies, the
    birthplace of Yo Yo Honey Singh. Each value is in the block, under another entity. The value check refused
    both.
  - **1 is a name spelt wrong** ("Khaungthala Temple" for "Madhukeshwara Temple"), refused by the guard.
  - **1 is a refusal** where the control answered (La Repubblica's parent company).
- **What the voice does better:** 2 more relation rows answered, and "The facts don't say." said on its own more
  often when the fact is absent (63 of 64 against 59, before the host).
- **Wrong in both arms, and not the adapters':** two rows whose entity is malformed in the held data ("Polish
  Russian War)>"), and two scored wrong on spelling or word order ("NonBarotropic" for "Non Barotropic"; "a
  bloodletting type" for "a type of Bloodletting"). They are refused either way. The held set and its scorer go
  through 0054's audit when it is built.
- **To watch in the next voice set: binding.** A guess, not measured: the voice rows carry one entity's facts a
  block, and the mix is 254,231 training rows where `talk_v2_2` alone was 216,035, so its binding rows are a
  smaller share. The next build adds voice rows whose block holds a second entity's facts.
- **Next:** change **b** first among b to f; then the casual proxies and the blind A/B, on replies read with b.

### Nick's call (2026-11-11): the blind A/B is voice against control, both Cubby
"both cubby". The acceptance's A/B named the LFM stand-in. It now reads:
- **The pairs:** 60 prompts, each answered by the voice (`talk_v3a`) and the control (`talk_v2_2_cont`), the order
  hidden and shuffled pair by pair. 30 chat turns (a hello, thanks, who are you, what can you do, a joke asked for)
  and 30 fact turns (one fact, a list, a wrong assumption, a fact not on record). 45 in English, 15 in French,
  the battery's own 3 to 1.
- **Both read the same way:** greedy, through the host with changes **a** and **b**. A pair is judged only when
  both replies are spoken. A prompt one arm refuses is counted for that arm and reported apart.
- **Nick's choice, pair by pair:** the first is more casual, the second is, or they are the same.
- **Line (set before he reads any):** the voice is chosen as more casual on at least 37 of the 60. A coin flip
  reaches 37 of 60 less than 5% of the time (4.6%). "The same" counts against the voice.
- **Kill, unchanged:** a wrong answer in any reply shown.
- **The stand-in is out of this test.** Cubby's base is its own, and the comparison that matters is what the voice
  data changed.
- **Who:** Cowork builds the sheet (the pairs and three buttons, the key kept apart); Nick judges; his choices are
  saved with the key's hash.

### Change b: a list's lines kept (Claude Code, 2026-11-11)
- **Built** (`546a560`): with `CB_TALK_LINES=1`, `talk_answer` keeps every line up to the next block marker
  (`Question:`, `Facts:`, `Conversation:`, `User:`, `System:`, and now `Assistant:`, `Answer:`, `Explicit:`), each
  line's spaces collapsed, a run of blank lines kept as one. Off (the default) it is the first line, as before (the
  new markers open a line, so the first line cannot change). The guard reads every kept line through the opener view.
  E6's gate now cuts with the host's `talk_answer` and keeps the raw decode beside each answer (`raw`); E8 keeps it
  beside each draft.
- **E6 again, both adapters, b on** (grilly2, greedy, the opener view as before; `exp_e6_talk_gate_{v3a,v2_2_cont}_b`,
  `exp_e6_host_view_{v3a,v2_2_cont}_b`; row by row against the first-line run in
  `validation/logs/e6_change_b_2026-11-11.json`):

  | host view, 407 held | first line (65342d3) | b on |
  |---|---|---|
  | voice `talk_v3a`: spoken & correct, refused, spoken & wrong | 391, 16, 0 | **393, 14, 0** |
  | control `talk_v2_2_cont` | 395, 12, 0 | 395, 12, 0 |

  - **Cowork's two rows were the host's cut:** "I've got the checked details right here:" (International Shooting
    Sport Federation) and "Let's lay out the confirmed details for you." (Lollardy) are now spoken with their lists,
    and read correct. They are the only two multi-line answers in the voice's 407; the control writes none. Every
    other answer's first line is the same as in the first-line run (greedy reproduces).
  - **Voice = control - 0.005** on spoken and correct (was - 0.010; line - 0.02).
  - **Seen, not a gate:** the voice's list is numbered and in bold markdown ("1. **Leader:** Luciano Rossi"), and
    the Lollardy list stops mid-word ("Twelve Conclusions Of The Loll") at the gate's 64 new tokens (the talk seat
    serves 160). The guard let the cut word pass. A host that drops an unfinished last line when the
    generation hit its token limit would fix it; not built (not in b).

### The blind A/B, as built (Cowork, 2026-11-11; written and committed before any reply was generated)
`validation/exp_talk_ab.py` (prompts, generate, sheet), `tests/validation/test_talk_ab.py`, and the 60 prompts in
`validation/logs/talk_ab_2026-11-11_prompts.jsonl` (seed 20261111).
- **Why the questions come from where they do.** On E6's questions the two adapters gave the same reply on 253 of
  the 407 records, and where they differ it is by word order, not by voice. So on the old question shape the voice
  mostly answers as the control does. The test therefore uses what the host will receive, and reports by source:
  - **chat turns (30),** bare, as the host sends a first turn: 6 each of a hello, thanks or goodbye, who are you,
    what can you do, a joke asked for. The user lines were written for this test. 22 EN, 8 FR.
  - **one fact (12):** 9 EN are WebQuestions and SimpleQA questions Cubby answered right in 0054's "after", each
    with the walk's own fact as the block; 3 FR are held rows of the voice set.
  - **a list (10):** 8 EN are held profile records of the E6 data that E6 did not draw; 2 FR are held voice rows.
  - **a wrong assumption (8):** 6 EN and 2 FR, held voice rows, with the host's verdict line ("Premise:
    contradicted"). The host does not send that line yet (change c); the control never saw it.
- **Two changes to the section above, with their reasons:**
  1. **"A fact not on record" is not among the pairs.** Both adapters are trained to say the same sentence there,
     and the host says it itself when the VM returns nothing: a pair would show the same text twice. E6 measured
     it (63 and 59 of 64 said on their own). This helps the voice against the line as first written, and is
     said here for that reason.
  2. **The line's count.** A pair one arm refuses is not judged, so the line is read on the pairs that are left
     (those Nick judges, and those whose two replies are identical, which count as "the same"): the voice
     passes with the smallest count a coin flip reaches less than 5% of the time for that many pairs. For 60 it
     is 37, as written.
- **An identical pair is not shown:** it is "the same" by construction, and counted against the voice.
- **Chat turns have no host check yet** (change d): the cut only. Cowork reads every reply for a false statement
  about Cubby before the sheet is shown. One in a voice reply is the kill; one in a control reply is reported.
- **What Nick sees:** the user's line, the facts on record for a fact turn, and the two replies in an order drawn
  by seed, with bold and line breaks shown as the chat will show them. The key stays in the logs, apart.
