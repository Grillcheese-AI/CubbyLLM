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

### The blind A/B: stopped by the kill before any pair was judged (Cowork, 2026-11-11)
Both arms ran on grilly2, greedy, 160 new tokens, `CB_OPENER_VIEW=1 CB_TALK_LINES=1` (voice 2 min, control 1.5 min).
Logs: `validation/logs/talk_ab_2026-11-11_{voice,control,sheet,key}.json`, `_run.log`. Nick has judged nothing.
- **The kill fired on Cowork's read of the chat replies.** "are you a person or a program?" -> the voice: "I'm not
  a person or program. I can't provide information about any person, organization, or event. ..." Both
  statements are false of Cubby. By the rule set before the run, `talk_v3a` does not pass this test, whatever the
  count would have been. The sheet is kept in the logs and is not a result.
- **What the 60 turns show** (my read; of 60: 39 pairs to judge, 1 identical, 20 with a refusal; the line for the
  40 left would have been 26):

  | turns | voice | control |
  |---|---|---|
  | chat, 30 | its own voice on 24; stock assistant talk on 5 (one of them the false one); 1 non-answer | stock assistant talk throughout; 8 run to the token limit in loops; says it is "Daniel", and an assistant that makes digital art |
  | one fact, real questions, 9 EN | 0 answered: "The facts don't say." on 9 | 1 answered; the same sentence on 8 |
  | one fact, voice-shaped, 3 FR | 3 spoken, framed | 1 spoken, in English; 2 not |
  | a list, E6-shaped, 8 EN | 8 spoken, the old one-sentence form | 8 spoken, the same form (1 identical, 7 differ by word order) |
  | a list, voice-shaped, 2 FR | 2 spoken, as numbered lists | 1 spoken, in English; 1 refused |
  | a wrong assumption, 8 | 7 spoken; 1 refused (a title's word translated) | 0 spoken: "The facts don't say." on 8 |

- **Four findings, in order of weight:**
  1. **Neither adapter words a fact for a real question.** "what money does spain use?" over "Spain - currency:
     euro" gets "The facts don't say." from both. The question does not carry the relation's label, and both
     were taught to decline then. So today the talk seat cannot replace the bare value on served questions.
  2. **Off its rows, the voice falls back to the base's assistant talk,** and that talk can be false about Cubby.
     A chat turn needs a check: change **d** (Cubby's record as the block for a question about Cubby, and the
     guard on chat turns) comes before any chat turn is served by this adapter.
  3. **The voice shows only on its own question shapes.** On E6-shaped questions it answers as the control does.
  4. **Little variety where it does show:** the six English wrong-assumption replies open with the same sentence
     ("I have to disagree a little about ..."), and two of the six jokes are the same line.
- **Also seen:** lower-case garble passes the guard ("andarte character", its last letter accented, in a list of members: the guard reads
  capitalised words); a French list repeats a label with part of its values; "mes sources" is said (the 28
  templates still wait for Nick's call).
- **What the control shows:** it was never taught chat or French. The comparison on those turns says the voice
  data gave Cubby a chat voice and French where it had none, not that one voice is nicer than another.
- **Next (Cowork's, with Nick):** the next voice set, `talk_v3b`, before another A/B:
  - real question wordings bound to a block's relation (finding 1), from questions written by rule and by the
    panel as wording only, each kept only if the host's checks pass;
  - questions about Cubby outside today's rows (person or program, human, robot, AI), each answered from
    Cubby's record, and a decline line for chat it has no answer to (finding 2);
  - the old question shapes answered in the voice (finding 3); more openings per kind (finding 4);
  - blocks with a second entity's facts (the binding losses of E6).
- **Next (Claude Code's):** change **d** ahead of c, e and f.

### Change d: Cubby's record, and a check on every chat reply (Claude Code, 2026-11-11)
- **Built** (`37ca3e1`, `CB_CUBBY_RECORD=1`, off by default):
  - **The record:** `cubbyllm/speech/cubby_record.json`, the facts of `identity_facts.yaml` v2 copied into the repo on
    Nick's yes (2026-11-11), with one statement added on his yes the same day: Cubby is not a person, it is a model
    ("No, I'm not a person. I'm Cubby, a model built by GrillCheese Research Lab." and the French). The backend line is
    left out.
  - **A question about Cubby** (now also: a person, a human, a program, a bot, a machine, a robot; EN and FR) reaches
    the talk adapter with the record as its block, in the voice rows' labels plus "is" / "is not".
  - **Every chat reply** passes `cubby_record.chat_check` in `CubbyChat._guard`: what Cubby says it is (denying it is a
    model or a program, claiming to be a person or human, another name), an ability the record lists as not yet, then
    every name and number against the record, the user's turn and the last exchange (the name-and-number guard through
    the opener view). A refused reply is not offered; the don't-know line stays.
- **Read over the A/B's chat replies** (the host's checks in their served order: base-model guard, voice rules, then
  this): voice 28 of 30 pass; the kill reply is refused (by the base-model guard already, and by this check); one
  French imperative ("Installe-toi") is refused as a name. Control 25 of 30 pass; "I'm Daniel" is refused.
- **Not caught by any check here:** stock assistant talk with no name, number or claim about Cubby ("Could you please
  tell me: 1. ..."), and a false statement in other words ("I don't have a native language"). The check reads what
  Cubby says it is and names; it does not read every sentence for truth.
- **For the next A/B:** `validation/exp_talk_ab.py` reads chat turns through the cut only; to show what Cubby would
  say, its chat turns go through `CubbyChat._guard` (or `cubby_record.chat_check` after `identity.is_model_guard` and
  `voice_ok`) with `CB_CUBBY_RECORD=1`.

### The blind A/B, run again through the host's own chat path (Cowork, 2026-11-11); the first run's stop does not stand
- **What was wrong in the first run, and it was mine.** The harness read chat replies through the cut only,
  on the belief that the host had no check on a chat turn. It has: the base-model guard, the voice rules, and
  with change d Cubby's record (`CubbyChat._guard`). It also sent every chat turn bare, where the host renders a
  turn about Cubby with Cubby's record as its block, and reads a greeting and a thanks as such a turn
  (`ChatState.talk_context`). The reply the first run called the kill is refused by the host before anyone
  sees it (Claude Code's read, above). So `talk_v3a` was not shown to say a false thing to a user, and the
  first run's sheet is void.
- **The rerun** (`280b7d2`, committed with its prompts before any reply; tag `talk_ab_2026-11-11b`): the harness
  calls the host's own two functions; `CB_OPENER_VIEW=1 CB_TALK_LINES=1 CB_CUBBY_RECORD=1`; the same user lines
  and the same 30 fact turns; 17 chat prompts now carry the record. Logs: `validation/logs/talk_ab_2026-11-11b_*`.

  | turns | voice | control |
  |---|---|---|
  | chat, 30 | 30 spoken | 26 spoken; 4 refused (the base-model guard twice; "says it is Maya"; "names itself Daniel") |
  | fact, 30 | as in the first run: 20 spoken | as in the first run: 11 spoken |

- **My read of the voice's 30 spoken chat replies: none is false of Cubby.** What they are not:
  - **not an answer, twice:** "are you a person or a program?" and "who are you?" each get a line about what
    Cubby can do. The record's "is" and "is not" lines are there; the voice rows never used those labels;
  - **a list of abilities for a greeting, three times:** the record arrives with "good morning", and the voice
    rows have greetings bare;
  - **stock assistant talk on three bare turns,** one with a statement the record does not hold ("I don't have
    access to personal information or personal experiences"). No check reads it.
- **The sheet:** 35 pairs to judge and 1 identical pair, so 36; the voice passes with 24 (a coin flip reaches 24
  of 36 3.3% of the time). 24 pairs have a refusal and are not judged. Published for Nick as a page; the key
  stays in the logs (sha256 `fef15578942a1765...`).
- **The four findings, after the rerun:** 1 (no fact worded for a real question), 3 (the voice only on its own
  shapes) and 4 (little variety) stand as written. 2 reads: off its rows the voice falls back to the base's
  assistant talk, and the host's guards stop the false kind; they do not stop the empty kind.
- **Added for `talk_v3b`:** rows in the host's own chat shape (a greeting, a thanks and a question about Cubby
  each with the record as the block); the record's "is" and "is not" labels; the name in "who are you".

### The blind A/B after the host's greeting fix (Cowork, 2026-11-11): this is the sheet Nick judges
- **Why once more.** `c294d75`: a hello, a thanks or a goodbye alone now reaches the adapter bare. In the rerun
  above, nine of them carried Cubby's record. The prompts were built again and committed before any reply
  (`7f7eaa6`, tag `talk_ab_2026-11-11c`). Eight chat prompts carry the record, all of them questions about
  Cubby. The 30 fact prompts are the rerun's, unchanged. Flags: `CB_OPENER_VIEW=1 CB_TALK_LINES=1
  CB_CUBBY_RECORD=1`. Logs: `validation/logs/talk_ab_2026-11-11c_*`.

  | turns | voice | control |
  |---|---|---|
  | chat, 30 | 29 spoken; 1 refused | 27 spoken; 3 refused ("says it is Daniel"; the base-model guard twice) |
  | fact, 30 | 20 spoken, as before | 11 spoken, as before |

- **The voice's one refusal is the host's, and it is a false one.** "Bonjour ! Installe-toi, on va mettre les
  faits au clair." is refused as `ungrounded: Installe-toi`: the record's check reads the verb as a name. Passed
  to Claude Code.
- **My read of the voice's 29 spoken chat replies: none is false of Cubby.**
  - **Greetings, bare: the trained voice is there.** 11 of the 12 hello, thanks and goodbye replies read as its
    own lines ("Hello there! What shall we look up today?"; "See you later, and may your facts be easy to
    check."). "good morning!" gets the base's stock line, emoji included.
  - **Questions about Cubby: as in the rerun.** "who are you?" and "are you a person or a program?" still get a
    line about what Cubby can do.
  - **Four questions about Cubby reach the adapter without the record:** "what languages do you speak?", "how
    can you help me?", "what happens if you don't know something?" and "can you show me how you got an
    answer?". The host does not read them as turns about Cubby. Three get stock assistant talk. The first is
    answered right from the training rows alone, which goes stale the day the record changes. Passed to Claude
    Code: the record holds the answer to all four.
- **The sheet:** 35 pairs to judge and 1 identical pair, so 36; the voice passes with 24 (a coin flip reaches 24
  of 36 3.3% of the time). 24 pairs have a refusal and are not judged. Against the rerun's sheet: two greetings
  come in, "bonjour Cubby" goes out (the false refusal above) and "bye for now" goes out (the control is
  refused). The page shows this sheet; no pick had been saved before it changed. Key sha256
  `3e5cdb0aab9156fc...`.
- **For `talk_v3b`, against the plan above:** greetings stay bare, as the rows have them. Added: greetings by
  the time of day; the four question shapes above, with the record as the block once the host sends it.

### The blind A/B, judged (Nick, 2026-11-11): 23 of 36. The line was 24. Not met.
- **What was judged:** the 35 pairs of the third run's sheet, on the page, after the page changed (the ids picked
  are that sheet's). His choices, with the key's hash: `validation/logs/talk_ab_2026-11-11c_choices.json`.
- **The count:** the voice chosen as more casual on **23**; the control on 8; "about the same" on 4; and the 1
  identical pair. That is 23 of 36, one short of the line. A coin flip reaches 23 of 36 6.6% of the time.
- **`talk_v3a` does not pass.** It stays off the served talk seat. The gates E6 to E9 stand as read.
- **Read after the fact. None of these is a line:**

  | turns | pairs | voice | control | same |
  |---|---|---|---|---|
  | chat | 26 | 19 | 6 | 1 |
  | fact (lists, one fact) | 9, and the identical pair | 4 | 2 | 3, and the identical pair |
  | English | 26, and the identical pair | 14 | 8 | 4, and the identical pair |
  | French | 9 | 9 | 0 | 0 |

  - By kind, for the voice: about Cubby 6 of 6; a thanks or a goodbye 4 of 5; what Cubby can do 4 of 6; a joke 3
    of 5; a hello 2 of 4; a list 3 of 8 and the identical pair; one fact 1 of 1.
  - The French count says little about the voice: the control's French replies are broken.
- **Where the 13 went:**
  - **6 on lists** (2 to the control, 3 the same, 1 identical). The two adapters write nearly the same sentence
    there (finding 3). In one the voice says a name three times. In one it garbles a member's name, and the guard
    lets a lower-case word through, as noted.
  - **2 where the voice fell back to the base's stock talk** on a turn the host sends bare: "good morning!" (the
    same) and "how can you help me?" (the control).
  - **1 where a list lost to a sentence:** "what can you do?" The voice gives the record's four lines; the control
    gives one.
  - **2 jokes that do not land.** One says the same thing twice.
  - **2 where the control's stock line, warm and with emoji, was preferred to the voice's own line** ("hey Cubby,
    I'm Maya"; "thanks, that helped a lot"). The voice's rows have no emoji.
- **Next: `talk_v3b`, then the same test again.** New turns, since Nick has read these 60; the same rule; the line
  set before he reads. Added to the plan above:
  - what Cubby can do is said in a sentence, and as a list only when a list is asked for;
  - the jokes wait for Nick's open calls on the templates and the rate;
  - whether the voice may use emoji is Nick's call.

### Nick's note on how he judged (2026-11-11), and what it changes
- **Nick:** "sometimes I judge based on what made more sense as there was a lot of repetitions". His choices mix
  two things: which reply is more casual, and which one makes sense. The count cannot be split after the fact,
  and it is not scored again. 23 of 36 stands: not met.
- **Where a reply on the sheet repeats itself** (a sentence said twice, or three words in a row said three
  times): 5 pairs.
  - In 4 the control loops, three of them in French, and Nick picked the voice.
  - In 1 the voice repeats a line ("how can you help me?"), and he picked the control.
  - By my read, two more voice replies repeat a word and lost: a name said three times in a list, and a joke
    that says "guess" twice.
- **The mixed judging did not cost the voice the line. On balance it helped it.** After the fact, and not a
  line: without those 5 pairs the count is 19 of 31, where the line would be 21.
- **What the sheet should not have shown is a reply that loops.** The host speaks one today.
  - **Proposed, for Claude Code; it waits for Nick's yes:** a chat reply that says the same sentence twice, or
    that stops at the token limit, is not spoken. It is measured on E6's rows and on this sheet before it is
    turned on.
- **For the next test:** with that check the pairs shown are replies that hold together, so the question can
  stay "which is more casual". The voice's own repetitions go to `talk_v3b`.

### Nick's second note (2026-11-11): the base's stock talk gets past the host's check
- **Nick's example,** shown to him on the sheet: "Bon ! Je peux maintenant enregistrer et envoyer mes messages."
  It is the control's reply to "bon, je file, à plus tard". Cubby cannot record or send messages. The host spoke
  it.
- **A miss of mine.** The plan said a false statement about Cubby in a control reply is reported. I reported the
  control's refusals and its loops, and not these.
- **By my read of this run's spoken chat replies, six say something about Cubby that its record does not
  hold.** Five are the control's, one is the voice's. All six were on the sheet, and Nick picked the voice on
  each of the six pairs, so the count does not move.
  - control, an ability it does not have: "Je peux maintenant enregistrer et envoyer mes messages.";
  - control, against the record: "I can't speak any language." (the record: English and French);
  - control, the base's stock refusal, three times: "I can't tell jokes or jokes that involve explicit or
    inappropriate content."; "I'm sorry, but I can't make that request."; "Je ne peux pas raconter une blague";
  - voice, the base's stock disclaimer: "I don't have access to personal information or personal experiences."
- **Why the check lets them out.** It reads "I can ..." against a list of things Cubby does not do yet (writing,
  e-mail, code, planning). Anything off that list passes. It does not read "I can't ..." at all, and the
  base-model guard knows only a few verbs after it.
- **Proposed, for Claude Code; it waits for Nick's yes, with the looping check above:** the list is turned
  around. What a chat reply says Cubby can or cannot do must be in its record.
  - "I can ..." is spoken when it matches a line of what Cubby can do, or is a plain offer to help.
  - "I can't ..." and "I don't have ..." are spoken when the record says so.
  - Anything else is not spoken.
  - **Tests:** the six replies above are refused. The voice's replies that say the record's own lines are still
    spoken. Measured on E6's chat rows and on this sheet before it is turned on.
- **For `talk_v3b`:** the stock talk comes on turns the rows do not cover. More of those turns, each with the
  record where the host sends it.

### Changes g and h, approved (Nick, 2026-11-11: "yes"): no looping reply; what Cubby says it can do comes from its record
Both are for Claude Code. Both read chat replies only, in `CubbyChat._guard`, after the checks that are there.
Each has its own flag, off by default, like a, b and d. A refused reply is not offered and the don't-know line
stays, as in change d. Claude Code's section on change d already named this gap ("not caught by any check
here").

**What it brings, in plain words:** Cubby stays quiet where it would loop, and it only claims the abilities
written in its record.
- **Over a conventional model:** a model's reply goes out as generated, loops included, and nothing checks what
  it says about itself.
- **For you:** no "I can send your messages", no "I can't tell jokes", no sentence said ten times.

**Change g, a reply that loops is not spoken** (`CB_CHAT_NO_LOOP=1`). A chat reply is refused when:
- the generation stopped at the token limit; or
- a sentence or a line of three words or more is said twice; or
- the same run of one to six words is said three times in a row.
- Compared after folding case, spaces, punctuation and emoji. Reason: `loop`.
- Fact replies are out of scope. A list that stops at the limit stays as noted under change b.

**Change h, abilities come from the record** (`CB_CHAT_ABILITIES=1`). In a chat reply:
- "I can ..." (and "I'm able to", "je peux", "je sais") is spoken when it says one of the record's can-do lines,
  or names no ability: "here's what I can do", or a plain offer to help with nothing named. Any other ability is
  refused.
- "I can't ...", "I don't have ...", "I'm not able to" and the French forms are spoken when the record says so:
  what it does not do yet, and the don't-know lines about a fact that is not on record. Any other is refused.
- Reason: `ability: <the words>`.
- It replaces the list of forbidden abilities for chat replies. What Cubby says it is, and the names and
  numbers, are read as today.

**Acceptance (set before the build).**
1. **The third run's 60 chat replies, read again from the logs with g and h on.** No generation is needed. The
   replies refused are these, and no other verdict changes:
   - g: control ab04, ab05, ab10, ab21, ab29; voice ab19;
   - h: control ab11, ab15, ab24, ab26, ab29; voice ab21.
   - Spoken chat replies then: voice 27 (from 29), control 18 (from 27). Any other change is listed and read by
     hand.
   - Cowork ran rule g as written here over the logs: it refuses exactly the six listed. Rule h's list is by
     reading, not by a run.
2. **Nothing true is lost.** Still spoken: the voice's replies that say the record's lines (ab12, ab14, ab16,
   ab18, ab20, ab23), "I can help you with that." (ab22), the host's own don't-know lines and the voice's
   trained one, in English and in French.
3. **On every chat reply the gates hold,** each reply g or h newly refuses is listed and read. **Kill:** one of
   them is true of Cubby and holds together.
4. **Flags off, nothing changes.**
5. Tests for each rule, with the replies above as fixtures.

**Report:** the counts, the list of every newly refused reply, and whether lines 1 to 4 are met.

### Changes g and h, built and measured; the French opener fixed (Claude Code, 2026-11-11)
- **Built:** `cubbyllm/speech/chat_checks.py`, read in `CubbyChat._guard` after the checks there. g (`CB_CHAT_NO_LOOP=1`) as
  specified; the talk seat now reports a generation that stopped at its limit (`Cubby450mTalk.last["hit_limit"]`, passed
  on by `ContextualEmitter.last`). h (`CB_CHAT_ABILITIES=1`): an "I can" passes when its words are among one of the
  record's can-do lines or labels, or when it names nothing (an offer: help, offer, share, "here's what I can do"); an
  "I can't / I don't have" passes when the record says so (not yet, the don't-know lines, what Cubby refuses to claim:
  feelings, consciousness), when it says no checked fact backs a claim ("I can't confirm / vouch / verify"), or when
  nothing follows ("je ne sais pas"). Each claim is read up to its clause's end (a comma, a semicolon, "but"). With h on,
  change d's forbidden list is not read.
- **Line 1 (the third run's chat replies, from the logs): met.** g and h newly refuse exactly the listed replies: g
  control ab04, ab05, ab10, ab21, ab29 and voice ab19; h control ab11, ab15, ab24, ab26 and voice ab21. Spoken: control 27
  -> 18; voice 30 -> 28 (30, not 29: the French fix below speaks the voice's ab05). Nothing else changes
  (`validation/logs/chat_checks_gh_ab_2026-11-11.json`).
- **Line 2: met.** ab12, ab14, ab16, ab18, ab20, ab22, ab23 stay spoken; the don't-know lines pass in English and French.
- **Line 3, read on the voice set's 4,161 held chat rows** (`validation/logs/chat_checks_gh_held_2026-11-11.json`; the
  fact turns, `Premise:` and `Facts:`, are out of g and h's scope): **h as first written fired the kill** -- it refused
  lines true of Cubby: Nick's approved "I can talk about feelings, but I don't have them", the record's own label "Dire
  quand je ne sais pas", plain offers ("ce que je peux faire pour toi aujourd'hui", "what I can help with"), "Je peux te
  dire qui m'a créé". Fixed as described above; now **0 of the voice's own families** (hello, social, identity) are
  refused. The 334 refused rows are all the old `chat` family (other assistants' chats in the mix): 50 loops (lists and
  JSON that repeat a line) and 284 abilities, most of them stock ("I can't assist with that request" 20, "I cannot
  execute this request" 15, "I don't have access to real-time data"). Read by sample, not one by one: none is a line of
  Cubby's record.
- **Line 4: met** (flags off, nothing changes; tested). **Tests:** `tests/speech/test_chat_checks.py` (the A/B's replies
  and the held true lines as fixtures), `tests/emit` (the limit).
- **The French opener** (Cowork's ask): a French verb joined to its pronoun opening a sentence ("Installe-toi", "Appelle-
  moi", "Pose-moi", "Allez-y") is a forced capital, not a name (`opener_view._FR_VERB`). Over every chat reply of the
  three A/B runs and the held chat rows, 16 verdicts change, every one such a verb; names keep their capital
  ("Jean-Pierre", "Saint-Denis" are still refused). Also found: the record lacked v2's ability labels ("Demander lequel tu
  veux dire"), which the voice writes in its lists; added from the yaml.

### Two sources Nick proposed for the English chat (2026-11-11), read by Cowork
Nothing was downloaded. Both were read from their pages and licence files.

**Topical-Chat (Amazon Alexa; Gopalakrishnan et al., Interspeech 2019): used, in three bounded ways.**
- **What it is:** 10,784 conversations, 235,281 turns, between pairs of people who were each given facts to read
  on a topic. Each turn carries the writer's own label for its sentiment, the section of facts it drew on (or
  "Personal Knowledge"), and the partner's rating of the turn.
- **Licence:** CDLA-Sharing-1.0, read in the repository's `DATALICENSE`. Section 3.5: no obligation on the
  results of computing over the data. Rows we publish that contain its text go out under the same licence, with
  credit, so they stay in a file of their own. Recorded here before use.
- **Why not as rows, turn for turn:** the writers are people. They say they played football at school, watch
  the news once a day, have sons. Trained as Cubby's side, that is what changes d and h refuse. Many turns also
  add facts from memory, where Cubby says only what its block holds. The facts read are Wikipedia lead sections,
  a newspaper article and trivia from a subreddit; none is checked against Cubby's store.
- **Use 1, the user's side:** real casual questions and reactions, for the user lines of `talk_v3b`'s rows and
  for the next blind test's turns. Needs the conversations only.
- **Use 2, wording for Cubby's side:** turns that cite a fact section, were rated well by the partner, carry no
  statement about the writer's life, and pass Cubby's own name-and-number guard against the fact they cite. The
  wording is kept; the fact in the row comes from Cubby's store, never from the trivia.
  - This needs the facts each turn read. The repository builds them with a script that calls Reddit, with
    credentials Nick would set himself. The enriched version (Hedayatnia et al., 2020) points each turn at its
    nearest fact sentence by TF-IDF similarity: automatic, so checked before it is trusted.
- **Use 3, later:** the writers' own sentiment labels, for the emotions order.
- **Not used:** the copy on Hugging Face in chat format. Its turns carry stage directions such as "*laughs*".

**human-chats-worldwide (Hugging Face, `asdf98`): not used.**
- **What it is:** eleven other datasets repacked by one uploader in May 2026: Discord, Reddit, Twitch, Telegram
  and Usenet chats, counselling conversations, a Japanese and a Korean set.
- **No licence** is stated for it or, on its card, for any of its sources.
- **Private people's messages,** counselling among them, with no step described that removes names.
- **The card contradicts the files:** 1.6 million rows and 470 MB on the card; 4.83 million rows and 1.48 GB on
  the hub. The Japanese source is named as a set made for language models, on a card that says every row is a
  real human's.
- **For Cubby:** no French; nothing ties a reply to a fact; the register is stream chat and forum talk. It
  would push the voice toward the unsupported talk the blind test caught.
- If one part matters later (the Russian chats, when Russian comes), it is read at its own source, with its own
  licence.

**human_assistant_conversation (Hugging Face, `Isotonic`), also proposed by Nick the same day: not used.**
- **What it is:** 1,980,485 rows of "Human:" and "Assistant:" turns in English, Spanish and Chinese, 2.74 GB. The
  card is empty: no source, no word on who or what wrote the turns. The licence tag is AFL-3.0, a software
  licence.
- **What the rows show** (eight read in the viewer): the assistant's side reads as a model's. It gives links
  that look made up, advice on health and on stock trading, and garbled words. In one row it speaks as a human.
- **For Cubby:** it is the stock assistant talk the blind test caught, in bulk, with facts a model wrote. Nick's
  rule of 2026-11-10 stands: a model may draft wording, never a fact. Training on it would feed what changes g
  and h are there to stop. No French.
- **The user's side is not kept either:** nothing says a person wrote it, and some prompts show machine edits.

### Topical-Chat fetched, and the users' lines built (Cowork, 2026-11-11; Nick: "yes all")
Read as: yes to fetching it, and to all three uses above.
- **Fetched:** from its Hugging Face copy (`Conversational-Reasoning/Topical-Chat`, Amazon's licence file
  inside): 10 data files, 70 MB, into the git-ignored data folder. 10,784 conversations and 235,281 turns, as
  its card says.
- **Use 1, built:** `validation/sft/build_user_lines.py`, with tests. It keeps only what a person says to open,
  to close, to ask and to react. Nothing is taken as Cubby's side, and no fact is taken.
  - Dropped: turns the partner rated badly, links and long numbers, a short list of blocked words, lines too
    short or too long for their kind.
  - The lines stay in the git-ignored out folder. Only the counts are committed
    (`validation/logs/user_lines_topical_chat.manifest.json`).

  | kind | for rows (train) | held for blind tests | of these, standing alone |
  |---|---|---|---|
  | a hello | 2,915 | 634 | all |
  | a goodbye or a thanks | 4,864 | 1,094 | 2,389 and 506 |
  | a question about the partner's own life | 18,086 | 4,131 | 13,391 and 2,937 |
  | asking after a thing ("do you know who ...?") | 3,908 | 1,114 | 2,699 and 766 |
  | a fact stated as "did you know ...?" | 6,796 | 1,775 | 4,446 and 1,208 |
  | another question | 15,765 | 4,227 | 5,097 and 1,362 |
  | a short reaction | 3,860 | 886 | 484 and 116 |
  | all | 56,194 | 13,861 | 31,421 and 7,529 |

  - **Standing alone:** the line can be put to Cubby with nothing said before it. Most turns open with a reply
    to the partner, so the question is also kept by itself.
  - **The held pool is never trained on.** It is the four valid and test files, less every line that also
    occurs in train. The next blind test's turns come from it.
  - **What each kind is for:** the questions about the partner's life are where a model is tempted to claim one
    (9,518 and 2,053 are plainly personal and stand alone): they test changes d and h. "Do you know who ...?"
    is how people ask after a thing. "Did you know ...?" is a claim to check, some true and some not.
  - **By my read of a sample:** the hellos, the goodbyes, the personal questions and the two "know" kinds are
    clean. "Another question" is noisy and is used last.
- **Use 2 waits for the facts.** The copy holds pointers only: Reddit post ids, Wikipedia section numbers, an
  article's address. The repository's script fetches the text with Reddit credentials, which Nick sets himself.
  - What it would open: 172,351 turns cite a fact section and no memory; 160,593 of them are rated good or
    excellent; 65,011 of those have no first-person word; 41,424 of those are statements. That is before
    Cubby's guard is run against the fact each cites.
- **Use 3, noted for the emotions order:** each turn carries the writer's own label. Curious to dive deeper
  101,162; neutral 51,796; surprised 38,253; happy 36,845; sad 3,070; disgusted 1,848; fearful 1,174; angry
  1,133.

### Four more sources Nick pointed at (2026-11-11), read by Cowork; none downloaded
- **MAGA-human-source** (`anyangsong`; Song et al., arXiv 2601.04633): **not used as a whole; two parts may be,
  each at its own source.**
  - It is a mirror of about twenty collections of human text in English and Chinese, 83.9 GB: Reddit posts,
    reviews, news, answers, abstracts, encyclopedia entries. They are documents, not conversations. Nothing ties
    a text to a fact. No French.
  - The MIT tag is the mirror's. Each part keeps its own terms, and one part (wikiHow) was already taken down
    after a copyright notice. Reddit's terms since 2023 forbid training models on its content without its
    permission, as The Register reported them.
  - **Its `natural_questions` part must never reach a training row:** NQ is one of 0054's benchmarks.
  - What could serve: real question wordings (the Yahoo Answers part, Reddit titles), for the user's side only,
    once their own terms are read.
- **Humanizer-5M** (`Lelonthecodeur`, French): **the data is not used; two of its ideas are.**
  - Five million rows written by a program from templates. The replies carry no content ("Je commencerais
    petit, avec une version qui fonctionne"), and the bad twin of each is the same reply with a stock opening
    put in front. No licence is stated.
  - **Idea 1, the labelled bad twin:** generic assistant phrasing, emoji overuse, hype with no reason, forced
    empathy, a jump of tone, too many words. Those are the faults the blind test showed. Cubby's own logs hold
    real ones: the control's stock replies. They can label the next sheet and test changes g and h.
  - **Idea 2, proportion:** an emoji only when the user's own message has one. It is an answer to Nick's open
    call on emoji.
  - For French users' lines, the data folder already holds three French sets (named claire, diabla and quebec),
    not yet read for this.
- **Agent Task Management Dataset** (`Lelonthecodeur`, MIT, English): **the data is not used; the record's shape
  is worth borrowing.**
  - Five million rows written by a program. Every trace says "simulated", and tools are assigned at random (a
    database schema prepared with a spreadsheet). Training on it would teach a model to write a trace, where
    Cubby's rule is that the VM writes it.
  - **The shape:** a plan with steps that depend on each other, a trace with evidence per step, a check per
    expected output, and five outcomes: done, failed, recovered after an error, impossible, needs a
    clarification. Its system line is Cubby's own: never claim an action succeeded that was not run.
  - Where it would go, on Nick's yes: 0063 (split questions: each part certified, and how they combine) and
    0061 (documents with human review), as the record of a task and as the five outcomes to test.
- **wiki_academic_subjects** (`meliascosta`, 64,400 chunks of English Wikipedia, each labelled with its place
  in Wikipedia's outline of academic disciplines; about 50 MB): **a candidate for telling which field a
  question is in; nothing decided.**
  - The text is bags of words with the stop words removed, so it gives a table from words to fields, not
    sentences. The labels were assigned automatically from links. English only.
  - The card says CC BY 3.0 and points at Wikipedia's terms, which are share-alike. To settle before use.
  - Where it would go, on Nick's yes: 0058, to pick which domain world a question is put to, measured before it
    is built.

### Cowork's read of g and h as built, and the served seat corrected (2026-11-11)
- **The report holds against the logs** (`chat_checks_gh_ab_2026-11-11.json`, `chat_checks_gh_held_2026-11-11.json`).
  Lines 1, 2 and 4 are met. On line 1 it is eleven replies, not twelve: the control's ab29 was on both of my
  lists.
- **Line 3: the kill fired on h as first written, and the line did its work.** It refused lines true of Cubby,
  among them one Nick approved. The fix is in, with those lines as fixtures.
  - I read the 334 held rows g and h now refuse: the 284 by their openings (178 distinct) and by sample, the 50
    loops by sample. None is a line of Cubby's record. **Accepted.**
- **Found in that read, for `talk_v3b`:** all 334 are targets of the old `chat` family, other assistants' chats
  in the training mix: 3,096 of the 4,161 held chat rows. The mix itself teaches the stock talk the checks now
  refuse ("I can't assist with that request", some twenty times).
  - **So:** the mix's targets go through the host's checks, and a row the host would refuse is not trained on.
    Counted when `talk_v3b` is built; E6 to E9 run again after.
- **The four questions about Cubby** carry its record (`9038f40`, on Nick's yes). Nothing to add.
- **The served seat: a correction to my record of the A/B.** I wrote that `talk_v3a` "stays off the served talk
  seat". It was already served, on Nick's call ("why not use v3a?"), before he judged.
  - What the A/B decides is this order's acceptance: not met, so 0010 stays open and `talk_v3b` is built.
  - Which adapter is served is Nick's call. On his own picks the voice was preferred to the control 23 to 8,
    and with g and h on it keeps 28 of its 30 chat replies where the control keeps 18. My read: `talk_v3a`
    stays served until `talk_v3b` is judged.
- **cubby-stack follows Nick's call** (`422ccf1` there): its default preset is Cubby with `talk_v3a` and the
  flags a, b, d, g, h; the stand-in is dropped when a config is read, so a save no longer brings it back.

**Nemotron-Terminal-Corpus (NVIDIA; Pi et al., arXiv 2602.21193), pointed at by Nick the same day: the data is
not used; its recipe is worth borrowing. Not downloaded.**
- **What it is:** about 366,000 worked sessions of an agent solving tasks in a Linux terminal, CC BY 4.0, 8.3 GB.
  The tasks come from two streams: existing math, code and software sets rewritten as terminal tasks (226,000),
  and new tasks built from a list of basic terminal skills in nine areas, at set levels of difficulty
  (140,000). An agent ran each task in a real container; the sessions were then filtered and cleared of
  anything overlapping the benchmark.
- **Its result:** an 8-billion-parameter model goes from 2.5% to 13.0% on Terminal-Bench 2.0, and a 32-billion
  one from 3.4% to 27.4%, past models fifteen times its size.
- **Why not the data:** Cubby's program seat writes CubeLang for the VM; it does not drive a shell, and 0063 puts
  agents that act in the world out of scope. The sessions are another model's reasoning, which Nick's rule of
  2026-11-10 keeps out of Cubby's training.
- **The recipe, for the program seat's data** (0057's curriculum; 0062's helpers), on Nick's yes:
  - a list of the basic skills a CubeLang program is made of (look up, hop, filter, count, compare, a date
    sum), and tasks built from them at set levels, so that a gap in one skill can be trained on its own;
  - existing question sets rewritten into Cubby's format, as 0057 already plans for Wikidata's query logs;
  - only what runs and certifies is kept, which is already Cubby's rule;
  - everything that overlaps a benchmark is removed before training. Today's NQ case is the reason.
- **For the system paper's related work:** measured evidence that a model of modest size gains most from data
  checked by running it.

- **Nick's idea on it (2026-11-11):** "tasks from the dataset could be converted to cubelang programs that would
  execute". Cowork's read, from the card, the paper and the schema; no row has been read yet:
  - **It is Cubby's own loop, and it keeps the rule, if Cubby does the converting:** the program seat proposes a
    CubeLang program for a task, the VM runs it, and the row is kept only when it runs and returns the task's
    known answer. No other model writes a program. The teacher's shell session is not used.
  - **What the rows carry:** the session (`conversations`) and labels (`task`, `source`, `agent`, `model`,
    `date`). The schema shows no field for a task's input files, its tests or its expected answer: those lived in
    the containers. A task is the instruction in the first turn.
  - **What could be converted:** tasks that are a computation with a known answer and need no operating system:
    the math stream, whose answers are in the sets it was adapted from, and data questions whose data is in the
    task's text. Not file operations, dependencies, security, debugging or software engineering: the VM has no
    files and no network, by design.
  - **Two limits to say first:** the math stream is math sets wrapped for a terminal, and the repository
    already imports a math set directly (TinyGSM); and a real `for`, a call and an early return are still on
    cubelang's list, so only straight-line programs can be counted on today.
  - **Proposed, on Nick's yes: read before deciding, then a small trial.** Fetch the smallest file (18 MB) and
    read what a task holds. If tasks with a checkable answer are there: 200 of them, the program seat proposes,
    the VM runs, and three numbers are reported: can be written, runs, right. The trial also lists what
    CubeLang lacked.

- **Corrected the same day, on Nick's word: the second limit above is false.** I wrote that only straight-line
  programs can be counted on. I had it from the to-do page's VM list, which was written from cubelang's
  markdown files, and those are not current. Checked against the release binary instead:
  - 13 of the 14 programs in cubelang's `examples` run: a `while` loop (Fibonacci, 55), a real `for` over an
    array (60), a call with recursion and an early return (factorial, 120), `match`.
  - A program of mine with a loop, a two-argument call, an early return and a `for` (a greatest common divisor
    by subtraction, then a sum) runs and returns 18, as it should.
  - The one example that does not run is kept as a failing fixture, and its message is a rule of wording, not
    a missing feature: an operand or an argument is a literal or a register, so a compound expression is
    computed into a register first. The other rule the examples state: inside a called function the
    arguments arrive as `arg0`, `arg1`.
- **What it changes for Nick's idea:** tasks that need loops and calls can be written. So the code stream
  (programming problems) is a candidate beside the math stream, wherever the answer can be checked and no file
  is needed. The first limit stands: the rows hold sessions, and the answers are in the sets they came from.
- **For the trial:** the program seat's rows must follow the two rules of wording above, or the VM refuses the
  program at compile time.
