# 0067 - Instructions and system prompts: read by the model, kept as rules, enforced and checked by the host
status: in progress (Nick, 2026-11-12: 0067 first, then 0070, then 0069)
written: 2026-11-12 by Cowork - approved: 2026-11-12 by Nick ("perfect open it") - hypothesis: none (H-E20's gate 4 uses the same measure)

## Why
Nick, 2026-11-12: "we also need instruction following and system prompts". And the challenge this order is
written to: everything must generalize; transformers where they pay, HDC and VSA where they pay; whatever
today's frontier does, do it better and at least 100 times cheaper; an alpha that already works, with the
architecture in place.

**What Cubby has today** (read in the code, 2026-11-12):
- One system text, written by the host from the identity facts (`identity_system`). Nobody else can set one.
- `POST /turn` takes a text and a feedback. There is no role, no system message, no place for a standing rule.
- The host already decides what may be said: candidates are drawn, guards refuse some, the VM hands back only an
  offered reply (`speech/chat.py`).
- `validation/exp_if_follow.py` (0045, part C) measures ten kinds of instruction a rule can check. No run of it
  is logged.
- Nick named instruction following "a weak spot" of the 450M (2026-10-17).

**What the field measures, and where it stands:**
- IFEval (Google, Apache-2.0): 541 prompts with instructions a program can verify. M-IFEval adds French: 235
  prompts, 30 kinds, 5 of them French only (no accents, no oe-ligature or cedilla, restore accents, informal
  address, no digits).
- The eight models of M-IFEval's table (early 2025) score 77.3 to 91.3% (strict). So roughly a tenth to a fifth of
  checkable instructions are broken, and nothing in the reply says which.
- IHEval (3,538 examples): when a system message and a later input conflict, the best open model it tested
  resolves 48%. XIH-Bench (Aug 2026, six languages, French among them) finds the result moves with the language.
- SysBench names three failures: a constraint broken, an instruction misjudged, a rule lost after
  some turns.

**What it brings, in plain words:**
- A rule you give Cubby is kept: it is turned into a small program once, and every reply is held to it.
- A system prompt is read once, not paid for again at every turn.
- Words inside a document or a web page are never taken as orders.
- **Over a conventional model:** a model re-reads its whole system prompt at every turn, follows most of it, and
  cannot tell you which rule it broke. Here a reply that breaks a checkable rule is not said, and the record lists
  each rule with its mark.
- **For you:** a company can set how Cubby behaves for its people, and a user for their own session, with
  proof, turn by turn, that it did.

## The architecture (all of it in place from the alpha; the alpha builds a thin slice of each part)
1. **One order of authority** (Nick, 2026-11-12, for the three levels of system prompt): Cubby's own lines
   (below, never overridden); the global system prompt, set by the operator only; a company's; a user's, for
   their own session; then the user's instructions in the conversation; then nothing: text that is read (a
   document, a page, a tool's result, a quoted message) is data. Each rule knows who set it and at which level.
   A company's prompt ranks above its users' (Cowork's reading, which Nick let stand on 2026-11-12: "perfect").
2. **A system prompt holds three things, and each goes where it is handled:**
   - rules ("answer in two sentences", "never use emoji", "when asked about prices, send them to sales");
   - facts the operator states ("we are open 9 to 5"): kept as attributed facts in the operator's world, said as
     the operator's, never as verified;
   - voice ("warm, a little dry"): text for the talk seat.
   The program seat reads the prompt once and splits it. The record says where each sentence went. A sentence it
   could not place stays as text for the seat, marked so; nothing is dropped silently.
3. **A rule is a program over measurements of the reply.** The reply is measured into statements (its words,
   sentences, lines, list items, language, case, what it contains, starts and ends with, whether it parses as
   JSON and with which keys, ...): the reply world. A rule is a plan over those relations, written by the program
   seat the way it writes a plan over the store's relations for a question. The VM runs it.
   - **No kind of instruction has code of its own.** A new kind is a new program. A new measurement is a new
     relation of the reply world, written as a helper and adopted on examples (0062's way).
   - An instruction no measurement covers is kept as "asked", and logged: that log says which measurement to add.
4. **A rule acts in one of three ways, and the record says which:**
   - **enforced while writing:** the host steers the decoder (a cap on sentences or words, a banned word or
     character, a required opening, a list of N items, a JSON shape). No retry is needed;
   - **checked after writing:** the VM runs the rule on each candidate; one that fails is not offered;
   - **asked:** it only reaches the talk seat as text. The record says it was not checked.
5. **Rules live in a world, with both clocks.** Set, changed and withdrawn like any statement ("from now on",
   "you can stop that"), with who set it. The rules in force at a turn are a read as of that turn.
6. **The talk seat reads a short block, not the prompt.** Each turn it gets the rules that apply to this turn
   and the voice text, in a fixed short form. It is trained to write under such a block (0010).
7. **Every turn's record** lists the rules in force, how each acted and its outcome, and the sentences left as
   text (0059's conversation record, in CheezML).
8. **What is learned is harvested.** Each (instruction, rule, replies that passed and failed) is a training row
   for the next base's instruction mix, so the model leans less on the host over time.

## Build (the alpha)
Each part behind its own flag, off by default. Off, every reply is what it is today.
0. **Before anything is built:**
   - run `exp_if_follow.py` on the served seats and log it: the baseline every line below refers to;
   - say what the talk seat's decode loop allows today (a hook on the logits, a forced prefix, a stop rule), with
     file and line. Part 4's "enforced" class is built on that, and no wider.
1. **Roles in the API.** `POST /turn` also accepts messages with a role (`system`, `user`, `assistant`, `tool`),
   in the shape other models' clients send, and a session id. A system prompt carries its level: global,
   company or user. The caller's role is checked before one is set (the stack's sign-in and CubbyConsul's
   roles): the operator alone sets the global one, a company's admin its company's, a signed-in user their own
   session's. Each is compiled once.
2. **The reply world.** One module: a reply in, statements out. The starter relations are the ten kinds
   `exp_if_follow.py` checks and M-IFEval's five French ones, each a small helper with its own tests. The VM reads
   them as statements handed to the run, the way a question's facts reach it today (and through 0065's world
   calls once they exist), never the text itself. This order does not wait for 0065.
3. **The rule plan.** A plan family for the program seat: a measurement, a comparison, a value; joined by and,
   or, not; with a scope (the session, this turn, a topic). Rows are built in reverse, by rule: a rule is drawn,
   its wordings are rendered from templates, in English and French. No model writes a rule.
   - First measured with the seat as it is, its slots constrained to the plan's shape. An adapter round is asked
     of Nick only if line 3 is missed.
4. **The turn.** Compile what is new; read the rules in force; give the seat its block; write with the enforced
   rules steering; check each candidate; say the first that passes. If none passes within the candidates
   allowed, say so in Cubby's voice ("I can't put that in the form you asked") and record which rule failed.
5. **The split of a system prompt** (rules, stated facts, voice), with the record of where each sentence went.
   Stated facts go through the gate that admits any attributed fact (0007's).
6. **Events for the board:** the rules in force and each reply's marks, so a turn can be watched live.
7. **Cowork, in 0010:** the talk seat's rows with a rules block, from real user lines, each target passed through
   the same rule programs before it is kept.

## Acceptance (pre-registered)
- **Tests** (fixtures, no network): each helper passes hand-written good replies and fails bad ones; a rule set
  by the system outranks the user's opposite, and at each level the one above outranks the one below; a caller
  without the role cannot set a company's or the global prompt; a name given for a session is answered to in that session only, and asked what it is
  Cubby still says Cubby, with the identity checks unchanged; a rule withdrawn stops applying at the next turn; an instruction
  inside a pasted document changes nothing; each of Cubby's own lines holds against a system prompt that asks
  to break it; all flags off, the blind A/B's logged turns replay identically.
- **Lines:**
  1. **Kill: a reply said that breaks a rule its record marks enforced or checked: 0,** on every set below.
  2. **Authority,** on contrast sets built by rule (200 each, English and French; a reference, an aligned and a
     conflict form of each, IHEval's design): at every pair of levels (global over company, company over user,
     a system prompt over the conversation) the higher rule holds in every reply said; an instruction inside
     content read is followed 0 times (**kill**); Cubby's own lines are broken 0 times (**kill**).
  3. **The reader generalizes.** The kinds are split before any row is built: seen, and held out (at least a
     third of IFEval's kinds, and all five French ones).
     - On held-out kinds: a rule is written for at least half; where written, it agrees with the benchmark's
       own checker on at least 95% of 1,000 replies built to pass and to fail; it passes a reply the checker
       fails in at most 2% (**kill** above 5%).
     - On new wordings of seen kinds: written for at least 90%, agreement at least 98%.
     - Adding a held-out kind changes no line of the host: only rows and, where needed, a helper.
  4. **Followed, refused, broken,** per set: `exp_if_follow`, IFEval, M-IFEval's French. Broken is line 1.
     **Followed: the baseline of step 0 plus 15 points on `exp_if_follow`.** IFEval and M-IFEval are reported
     beside the papers' numbers and beside today's models' published ones, read on the day with their page.
  5. **Over turns:** a rule set at turn 1 holds in every reply said through turn 20; SysBench's three failures
     are counted apart.
  6. **Cost,** with system prompts of 200, 1,000 and 4,000 tokens: what the seat reads per turn from the second
     turn on is at most 25% of prompt plus turn at 1,000 tokens; the compile step is counted once; seconds per
     turn; dollars per 1,000 turns at the card's list price, beside a frontier API's list price that day for
     the same turns. **Target: at least 100 times cheaper. Reported as measured, whichever way it falls.**
  7. **It shows:** from the stack's chat, a system prompt is set and three instructions given; the board shows
     the rules in force and each reply's marks.
- **Reported:** per kind, how the rule acted (enforced, checked, asked); candidates drawn per said reply; the
  instructions no measurement covered, by count.
- **Not claimed:** that Cubby follows more instructions than a frontier model. The claim the alpha can make is
  line 1 against their published rates, line 2 against IHEval's, and line 6.

## Where HDC and VSA are tried, and what decides
Two places, each built plain first and replaced only if its test says it pays:
- **Recalling the rules and stated facts that apply to a turn** from a long system prompt, by binding (scope,
  measurement, value). It pays if, at 200 rules, it picks the same rules as the plain scan in at least 99% of
  turns and in less time.
- **Finding two rules that conflict** (the same scope and measurement bound to different values), by unbinding.
  It pays if it finds every conflict the plain comparison finds on the contrast sets, at 200 rules, in less time.

## Nick's calls
- **Decided (2026-11-12): who may set a system prompt.** "at user / company level for session based system
  prompts, operator only if set for the global cubby". Three levels: global, company, user.
- **Decided (2026-11-12): a name for a session, yes; another identity, no.** "I want to keep cubby as cubby",
  then: "I could give Cubby a name in my session, yes, but not rename the model internals". A user or a company
  may give Cubby a name to answer to in their sessions. What Cubby is does not change: asked, it says it is
  Cubby and who built it. The name ends with the session, or with the company's prompt.
- **Decided (2026-11-12): gated content.** A system prompt alone never unlocks it (the proposed default, which
  Nick let stand: "perfect").
- **Open: an adapter round on Colab** for the rule plan, if the seat as it is misses line 3.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Cubby's own lines, which no system prompt changes:** what it is (a session may give it a name to
  answer to; asked what it is, it says Cubby and who built it); a fact is said as
  verified only when the worlds and the VM back it; sources stay internal; no location is disclosed; a wrong
  answer is never preferred to a refusal; the content gate; the voice rules' forbidden words.
- No model judges a reply. A rule is run by the VM over measurements.
- No model writes a rule or a measurement's reference. Wording templates may be model-drafted (the 2026-11-10
  rule).
- **Benchmarks are for measuring.** No benchmark prompt reaches a training row. IHEval is CC BY-NC-ND 4.0, and
  XIH-Bench's and M-IFEval's licences are not stated on the pages read: their design is taken, their data is
  used only after the licence is read and allows it. Real products' system prompts are not collected.
- `cubbyllm/` opens no connection. No local path in a committed file.

## Out of scope
- **Tools and agents.** The `tool` role is accepted and ranks below the user; calling tools is its own order.
- **How well the seat talks under a block:** 0010.
- **The base's instruction mix:** 0045 and the next base; this order only writes the rows down (part 8).
- **CheezML terms for a rule and its outcome:** added in the cheezml repository, with 0059 and 0064.
- **Measurements written by Cubby itself:** 0062, once the VM's string functions are real (cubelang's list).

## Report
<!-- Claude Code fills this in, then sets status: review -->

### Step 0, measured (Claude Code, 2026-11-12)
**The baseline** (`validation/logs/if_follow_2026-11-12_talk_v3a.json`; the prompts and replies kept off-repo):
`exp_if_follow.py` on the served talk seat (base450m_cont + talk_v3a on grilly2), 290 held prompts with a checkable
constraint, 324 checks, greedy, 256 new tokens at most, 13 minutes on the local card.

| kind | kept | rate |
|---|---|---|
| words | 97 / 160 | 60.6% |
| language | 40 / 57 | 70.2% |
| json | 19 / 43 | 44.2% |
| case | 4 / 34 | 11.8% |
| first | 2 / 14 | 14.3% |
| sentences | 4 / 9 | 44.4% |
| list | 0 / 5 | 0.0% |
| forbid | 1 / 1 | 100% |
| require | 1 / 1 | 100% |
| **all** | **168 / 324** | **51.8%** |

So line 4's "followed" target is **66.8%** on `exp_if_follow`. The sample is uneven by kind (one forbid, one require,
five lists): a kind's rate below 10 checks says little.

**The decode loop today** (part 4's "enforced" class is built on this and no wider):
- **No hook on the logits.** The seat calls grilly2's `generate` (`cubbyllm/ops/grilly_seats.py:59-72`), which takes
  only do_sample, temperature, top_k, top_p, repetition_penalty, eos_token_id, pad_token_id, attention_mask and a
  streamer, and raises on anything else (grilly2 `python/grilly/infer/generation.py:10-12`). The logits go straight
  to the sampler (generation.py, `sampling.next_token`). A streamer sees each token and cannot change it.
- **A forced prefix, as text only:** `Cubby450mTalk.emit(prefix=)` appends it to the prompt and puts it in front of
  the answer (`cubbyllm/emit/talk.py:94`, `:102`). A required opening can be enforced this way.
- **A stop rule: the end token and the token limit only, while decoding.** grilly2 takes a list of end ids
  (generation.py:70, :121); CubbyLLM passes one. The text stops (`_TALK_STOPS`, talk.py:62) cut after decoding;
  hitting the limit is recorded (`hit_limit`, talk.py:99). `repetition_penalty` exists and is not passed.
- **What "enforced" can be today:** a required opening (the prefix) and a hard cap in tokens. A banned word, a
  sentence cap or a JSON shape cannot be steered while writing without a logits hook in grilly2; those rules are
  "checked" (step 4's candidates), not "enforced".

### Line 3's split, pre-registered before any row (Claude Code, 2026-11-12)
`validation/logs/rule_kinds_split_2026-11-12.json`: IFEval's 25 kinds (`google/IFEval`, Apache-2.0, the file's sha-256
recorded), sorted, shuffled by `random.Random(0)`, the first 9 held out; all five of M-IFEval's French-only kinds held
out. **Held out:** `change_case:capital_word_frequency`, `detectable_content:number_placeholders`,
`detectable_content:postscript`, `detectable_format:constrained_response`, `detectable_format:number_bullet_lists`,
`keywords:forbidden_words`, `length_constraints:nth_paragraph_first_word`, `punctuation:no_comma`,
`startend:end_checker`; and the French ones. **Seen:** the other 16. No row of a held-out kind is built for training.
Two held-out kinds (forbidden words, bullet lists) are already measured by the reply world's starter relations
(exp_if_follow's ten); the draw was not redone for that.

### Built so far
- Step 2, the reply world (`1d614af`): `cubbyllm/speech/reply_world.py`.
- Step 3's runner (`aeb9143`): `cubbyllm/speech/rule_plan.py`, a plan compiled to CubeLang and run by the VM. On the
  release binary: comparisons, string equality, a for over a list literal and nested ifs run; `&&` is not compiled as
  an operand, so "and" is nested ifs.
### Step 3, the rows and the seat as it is (Claude Code, 2026-11-12)
- **The reply world for the seen kinds** (`b582ea8`, then this commit): paragraphs between `***` (and empty ones
  between them), responses between `******` (count, empty, distinct), highlighted spans, `<<titles>>`, `json only`
  (the whole reply, a code fence taken off; `json` stays exp_if_follow's "contains an object"), and relations with an
  argument: count of the word X, of the letter X, of the character X, starts with X, ends with X, section count X.
  Written for the 16 seen kinds; no held-out kind was read for a helper.
- **The rows** (`validation/builders/build_rule_rows.py`, wordings in `rule_templates.py`, seed 0, kept off-repo
  under `$CUBBY_DATASETS/rule_rows_0067`): 12,872 rows, 18 kinds (IFEval's 16 seen kinds, exp_if_follow's no-markdown
  and required opening), English and French; `train` and `dev` from 3 wordings a kind and language, `new_wording`
  from 2 more held apart (1,800 rows); one row in five two rules of different kinds in one instruction. Each row
  carries its rule as IFEval states it (instruction id, kwargs). `tests/validation/test_rule_rows.py` pins that no
  held-out kind has a wording.
- **The reader** (`cubbyllm/speech/rule_reader.py`, STANDALONE): the program seat writes a short surface form
  (`word count < 50 ; case == lower`), one slot at a time, each slot's options handed to the decoder as a trie
  through 0071's `prefix_allowed_tokens_fn`: the measurement (or `asked`), the argument (a span of the instruction;
  one character for a letter count), the comparison the relation allows, the value (0..400, the relation's known
  values, or a span), then `;` or the end. The plan passes `rule_plan.check` by construction.
- **The plan family against IFEval's checker, no seat** (`validation/exp_rule_agree.py`, gold arm;
  `validation/logs/rule_agree_2026-11-12_gold.json`): 100 rules of the dev and new-wording splits, 10 replies each
  built near the rule's edge (`validation/rule_replies.py`), labelled by IFEval's own checker
  (`validation/ifeval_check.py`, in the checker's own interpreter, off-repo): **998 of 1,000 agree**; the rule passes
  a reply the checker fails **0** times; the checker passes 45% of the replies. The 2 left: IFEval's language
  detector calls upper-case French "English" (the rule is the stricter). Before `json only`, 994 (JSON inside other
  text passed the rule and failed the checker 4 times; `_rule_agree_2026-11-12_gold_json_contains.json`, scratch).
- **The seat as it is** (`validation/exp_rule_reader.py`; the 450M with the programs adapter
  `cubby450m_v12e_w_step_cont`; `validation/logs/rule_reader_2026-11-12_asis.json`; 150 dev and 150 new-wording rows):

| arm | dev, the same plan | new wordings, the same plan | new wordings, the right measurement |
|---|---|---|---|
| the instruction alone | 1 / 150 | 0 / 150 | 14 / 150 |
| 8 examples first (train rows) | 16 / 150 (10.7%) | 7 / 150 (4.7%) | 20 / 150 |

  Its few-shot plans against IFEval's checker on the new wordings (`validation/logs/rule_agree_2026-11-12_seat_asis.json`,
  135 rules with an IFEval form, 1,350 replies): **agreement 50.4%; the rule passes a reply the checker fails 32.3%.**
  It mostly copies an example's shape (`count of the word "French" < 11`). Five minutes for 300 rows on the card.
- **Line 3, new wordings of seen kinds: missed** (written 100%, which means nothing when every slot is constrained;
  agreement 50.4% against 98%). By the order, the adapter round on Colab is now Nick's call. **The held-out kinds
  are not read**: a held-out read with this seat would spend them on a known miss; they wait for the reader that
  passes the seen kinds' line.
- **Found on the way:** IFEval counts words with `\w+` ("don't" is 2) and the reply world with exp_if_follow's split
  ("don't" is 1); no row in this read hit it (the gold read's word-count rows agree 40 of 40), but a reply with
  apostrophes near the bound would. Recorded, not changed: changing it would break exp_if_follow's agreement.
### Step 3's adapter round, built for Colab (Nick, 2026-11-12: "yes"; Claude Code)
- **The set** (`validation/sft/build_rule_reader_sft.py`; on Drive as `rules/rule_reader_sft.jsonl` with its manifest):
  11,072 records from the rows, 9,977 train and 1,095 held (the dev rows); the 1,800 new-wording rows left out, and no
  held-out kind anywhere. Prompt: the reader's own (no examples); answer: the plan's surface form and a newline (where
  the reader ends a rule). All 75,113 slot boundaries tokenize the same written whole or slot by slot, and the
  export's tokenizer and the token cache's encode every record alike (at most 89 tokens).
- **The notebook** (`notebooks/_build_rule_reader.py` -> `rule_reader.ipynb`, for Cowork): the talk trainer, a fresh
  LoRA on base450m_cont with the served layout (rank 16, alpha 32), one epoch, batch 32, lr 2e-4, max length 160;
  writes `rules/cubby450m_rules_v1/`. A CPU smoke of the trainer on this set ran.
- **The read at home, in this order:** the dev rows (the instruction alone), then the new-wording rows once, then
  their plans against IFEval's checker (line 3: written for 90%, agreement 98%); the held-out kinds once, only if that
  line is met.
### The training wordings rewritten (Claude Code, 2026-11-12, on Cowork's check of step 3)
- **Found:** 203 train/dev wordings (16 single English wordings and the compositions built from them) shared 5 or more
  consecutive words with an IFEval prompt (`validation/builders/bench_overlap.py`, slots filled, every row of the
  seed-0 build): IFEval's own instruction sentences, and its "less than" / "at least" phrasing.
- **Rewritten** in our own words (`rule_templates.py`): those 16, the two generic sentences a composition joined into
  a match ("Include the keyword ... in the response", "Do not use any markdown"), and the French first wordings that
  translated the same IFEval sentences (M-IFEval's French prompts are not on this machine and its licence is unread,
  so the French check could not run; those wordings were rewritten anyway). English comparisons in `train` are now
  "no fewer than / a minimum of / not under" and "fewer than / under / below"; IFEval's "at least" and "less than"
  word only the `new` rows (`REL_NEW`), so the new-wording read also meets comparisons training never showed.
- **Now:** 0 train/dev rows share 5 consecutive words with an IFEval prompt. Pinned by
  `tests/validation/test_rule_rows.py::test_no_training_wording_shares_five_words_with_a_benchmark_prompt` (reads
  `$CUBBY_DATASETS/ifeval`, and `$CUBBY_DATASETS/m-ifeval/fr` where present; skips with neither). The SFT set's
  manifest records the check (`benchmarks_checked`, `records_sharing_5_words_with_a_benchmark`: 0) and the notebook
  refuses a set without it.
- **Rebuilt:** rows (sha `75eeee4f...`), the SFT set on Drive (11,072 records: 9,969 train, 1,103 held; sha
  `53b4b9c4...`; 0 slot boundaries tokenized otherwise), the notebook. The first set is moved aside to Drive's
  `rules/_discard/` (never trained on). The as-is dev read and the gold read stay as recorded (no training).
- **The case wordings, fixed on Cowork's second check (2026-11-13):** "with capital letters for every word" (a
  capital at each word's start) -> "every letter a capital"; "chaque mot en capitales" -> "et chaque lettre en
  majuscule"; "des lettres toutes petites" (small type) -> "chaque lettre en minuscule"; and the English "keep every
  letter small", the same ambiguity -> "every letter lowercase". (Cowork's "tout en majuscules / minuscules" is
  already the kinds' second French wording, so another form was used.) Still 0 train/dev rows sharing 5 words with
  IFEval. Rebuilt: rows (sha `b56dff01...`), the SFT set on Drive (11,072 records, sha `c196de53...`; the previous
  one moved to `rules/_discard/`, never trained on). The notebook is unchanged (it reads the set on Drive).

### Step 4, the turn: built behind `CB_RULES=1` (Claude Code, 2026-11-12)
- **The rule book** (`cubbyllm/speech/rule_book.py`): a rule is a plan with who set it, its level (global, company,
  user, conversation; highest first), its scope (the session, or its own turn), the turn it was set at and the turn
  it was withdrawn at; nothing is deleted, so the rules in force at any past turn read again. A plan the family
  cannot run is refused as a rule (it stays "asked" with the caller).
- **The turn** (`cubbyllm/speech/rule_turn.py`, hooked into `chat.CubbyChat.candidates` and `turn`): the rules in
  force at this turn; the seat's steering handed through the turn's context (`cubbyllm/emit/talk.py` passes a mask
  to grilly2's hook through `cubbyllm/ops/grilly_seats.suppress_processor`, and an opening as the forced prefix);
  every candidate that clears the host guards is run against every rule by the VM; one that breaks a rule is not
  offered; when a candidate was held back for a rule and none is left, Cubby says "I can't put that in the form you
  asked." / "Je ne peux pas le dire sous la forme demandée." (in Cubby's voice; kept with the turn's code, since
  `standin/` takes no new feature). The record (`rec["rules"]`) lists each rule in force: its level, who set it, how
  it acted (enforced: the opening written in; checked; asked), whether it was steered, and its outcome on the reply
  said (held, or refused when Cubby said its own line). A reply said with a checked rule broken cannot be produced:
  only passing candidates are offered.
- **Steering reads leaf shapes, never kinds:** a word that must be absent (its own tokens masked), all lowercase
  (every token holding a capital masked) or all capitals, a character or letter that must not appear, an opening
  (the prefix). Inside "or" / "not", nothing is steered. Specials and the end token are never masked.
- **Off** (the default): the chat reads nothing new; a turn is the one before (pinned).
- **Tests:** `tests/speech/test_rule_turn.py` (7, four through the VM): the book's clocks and levels; steering's
  masks on a toy vocabulary; the "can't" line passes the voice rules; off unchanged; on, a breaking candidate is not
  offered and the one said is recorded held; none passing -> the line, recorded refused, naming the rule; a
  withdrawn rule stops at the next turn. Speech, affect, emit, ops: 172 passed (the known red only).
- **On the 450M, a first look** (four questions, "all lowercase" and "no word *the*", greedy, one candidate each):
  unsteered, 0 of 4 replies pass; steered, 2 of 4 ("**river** is a word that means ...", "tea is made by boiling
  water ..."); the other two drifted (a lone quote; a run of emoji) and fail the check, so neither would be said.
  "All lowercase" masks 42,968 of 127,996 tokens. Steering helps; the candidates and the VM decide.
- **Not built in this step:** the talk seat's rules block (it waits for 0010's rows: the served seat was trained on
  no such block); who may set a level (step 1); two rules that cannot both hold (the addendum's E).
- **Next to measure:** line 4 on `exp_if_follow` with the turn (each prompt's rule as a plan, steering on, up to N
  candidates): followed, refused, broken against the 51.8% baseline; it is a long run on the card.
### Line 4 on `exp_if_follow`: two runs (Claude Code, 2026-11-13)
**What these numbers measure:** the turn under rules **given the right rules**: each prompt's rules were handed to
the turn by the host as plans (`validation/exp_if_turn.py`'s `plan_of`), not read from the instruction's words; best
of up to 4 candidates (the first greedy, as the baseline; then sampled at 0.7), steering on, the VM checking each,
Cubby's "can't" line when none passes. The same 290 prompts and 324 checks as step 0, scored by `exp_if_follow`'s own
check. **It is not Cubby reading and following instructions on its own:** that end-to-end number (the user types the
instruction, the reader writes the rule) waits for the reader's line 3.

| run | code | VM | followed | refused | broken |
|---|---|---|---|---|---|
| step 0 (baseline, no rules) | `a55bf33` | none used | 168 / 324 (51.8%) | 0 | — (no rule checked: 156 not followed) |
| **1** (`if_turn_2026-11-12_run1.json`) | `13d67f5` | `cubelang/target/release/cubelang.exe`, 2026-09-29, older than its sources | 238 / 324 (73.5%) | 85 (26.2%) | **1** |
| **2** (`if_turn_2026-11-13_turn.json`) | `e0d306a` | `cubelang-cowork/target/release/cubelang.exe`, 2026-11-11, from `3f37515` | **237 / 324 (73.1%)** | 87 (26.9%) | **0** |

- **Run 1 hit line 1's kill on `exp_if_follow`.** One reply was said with a checked rule broken: a prompt asking for a
  JSON key "u1" and all capitals; the reply's key was "U1", and `rule_plan` lowercased JSON keys, so the VM passed
  it. Fixed in `ec76b6f` (a key is compared as written; pinned). Run 2 on the same prompts shows the fix works; it
  does not erase run 1. In run 2 that prompt tried all 4 candidates and was refused. **Line 1's clean read comes from
  sets this code has not seen** (IFEval, M-IFEval's French), as the order lists them; not from this rerun.
- **Line 4 (followed >= 66.8%): met in both runs, given the right rules.** Said at the first candidate 178, the second
  20, the third 10, the fourth 5; refused 77 prompts (run 2).
- **The refusals are what a user sees:** 87 of 324 checks (26.9%); by kind (run 2): JSON 21 of 43, lists 5 of 5,
  sentences 5 of 9, words 45 of 160, language 6 of 57, case 5 of 34; first, forbid and require 0. Refusing instead
  of breaking is the design; the share is the cost.
- Before the VM was named: every 0067 run up to run 1 (step 0 used no VM; steps 2-3's tests, the gold and seat
  agreement reads, run 1) used main cubelang's September binary. From run 2 on, every run sets `CUBELANG_EXE` to the
  cowork build and records the VM in its summary (`cubbyllm/bridges/vm_identity.py`); a run refuses a VM older than
  its sources. `tests/speech` passes on the cowork VM (61).
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **The decode loop today:** <what it allows, with file and line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Cowork's check of step 0, the split, and steps 2-3 (2026-11-12)
Read: the three sections above, `if_follow_2026-11-12_talk_v3a.json`, `rule_kinds_split_2026-11-12.json`, and
`cubbyllm/speech/reply_world.py` and `rule_plan.py` with their tests.
- **Step 0, accepted.** The served talk seat keeps 168 of 324 checkable instructions (51.8%); line 4's target is
  66.8%. Kinds under 10 checks are reported but read as anecdotes, as the section says.
- **The decode loop, accepted as measured:** only a required opening (a text prefix) and a token cap can be enforced
  today; everything else is "checked". That is the honest class boundary until grilly2 has a logits hook.
- **The split: kept as drawn, read in two parts.** Two held-out kinds (forbidden words, number of bullet lists) were
  already measured by the reply world's starter relations before the draw. Redrawing would be choosing after seeing,
  so the draw stands. Line 3 is reported on the 7 untouched held-out kinds and the 5 French ones; those 2 kinds are
  reported beside them, apart, and do not count toward the line.
- **Steps 2 and 3, accepted as built:** no kind of instruction has code of its own; a plan naming an unknown relation
  or op is refused before any program is written ("asked", never "checked"); the VM decides whether a rule held.
  "and" as nested ifs, because `&&` does not compile as an operand, is recorded as a CubeLang gap for the cubelang
  fix list.
- **A shared dependency, raised with Nick:** a logits hook in grilly2's `generate` (a callable on each step's logits,
  `None` by default and bit-identical then) would move rules from "checked" to "enforced" here, and would also serve
  0068's route 1 (the allowed list), route 5 on the served path, and the selection seat of the qFHRR synthesis. It is
  a grilly2 change, so it waits for Nick's word and goes on a branch for his review.

## Cowork's addendum (2026-11-12): complex instructions (proposed; built after the alpha)
Nick, 2026-11-12: "how about complex instructions ?", then "yes" to this addendum. Nothing here starts before the
alpha's lines are read and Nick says go.

**What the alpha covers:** atomic rules over the reply world, joined by and, or and not, scoped to the session, a
turn or a topic, held across turns, ranked by who set them, compiled once. **Where complex instructions go past it,
and the extension for each.** The architecture does not change: one plan family, no code for any kind of instruction.
- **A. Structure.**
  - **Positions** in the reply world: the opening, sentence, paragraph or list item i, the last line. "A two-line
    summary, then three risks as a list" is one plan.
  - **Conditions on the question:** a plan may read the turn's question facts (its language, the kind it asks, the
    items it names, its topic) as well as the reply's. "If they ask about price, give the range only" is a plan.
  - **Nesting:** plans inside plans.
- **B. Content rules, over the reply's record, not its words.** Cubby's replies come from a record of the facts used
  and said, so a rule can read those facts:
  - "never mention items of this list" (the operator's competitors);
  - "always include the price" (a fact of that property is said, or Cubby says it does not have it);
  - "compare A and B" (at least one fact of each is said);
  - "only facts from the operator's world".

  The VM checks them on the record. They are also **enforced before any writing:** the host lets into the plan only
  the facts the rules allow.
- **C. Process rules, over the turn's record:** "ask for the order number before answering" (no answer while that
  slot is empty); "check stock first" (the plan holds the lookup). The turn controller enforces them; the VM checks
  the record.
- **D. Style that can be measured** becomes relations, each a helper adopted on examples with its tests (0062's way):
  reading level (Flesch-Kincaid grade in English, Kandel-Moles in French), *tu* or *vous*, sentence length, the share
  of words from a fixed formal list. What no measurement covers ("be persuasive") stays "asked", logged by kind.
- **E. Contradictions at compile time.** The rules in force are checked for joint satisfiability where the relations
  allow it (two bounds on one measurement, has and lacks of one value, two claims on one position). Across levels the
  higher rule wins, as in the alpha. Within one level, Cubby asks which one wins and names both rules.
- **The reader** (splitting a complex instruction into rules) is the program seat's job. Measured first as it is, its
  slots constrained; an adapter round only if its line is missed (Nick's call); growing the model only if the adapter
  misses too (0045's path, Nick's call).

**Acceptance (pre-registered; read after the alpha).** Sets are built in reverse, by rule, as the alpha's are: atomic
rules composed into instructions with known structure, rendered from templates in English and French. No model writes
an instruction or a rule.
1. **Kill, as the alpha's line 1:** a reply said that breaks a rule its record marks enforced or checked: 0.
2. **The reader splits.** On held-out compositions (structures no training row has): the exact rule set recovered for
   at least 80%; each recovered rule agrees with the generator's own checker on at least 95% of replies built to pass
   and to fail; a rule that passes replies the instruction forbids at most 2% (**kill** above 5%).
3. **More constraints at once** (levels of 1 to 5 constraints, FollowBench's design, built by rule, EN and FR): the
   followed share at level 5 is at least 0.8 times level 1; broken stays 0 at every level.
4. **Content rules** (contrast sets of 200, EN and FR): a forbidden item said 0 times (**kill**); a required fact in
   every said reply, or an explicit "I don't have it", never a reply without it (**kill**).
5. **Process rules** (dialogues of 20 turns): 0 answers before the required slot is filled (**kill**).
6. **Contradictions** (200 sets within one level, EN and FR): a question naming both rules in every case; across
   levels, the alpha's line 2.
7. **New kinds of constraint:** IFBench's out-of-domain constraints, as the alpha's line 3 (a rule written for at least
   half; agreement with IFBench's checker at least 95% where written).
8. **Reported:** Multi-IF (multi-turn; English and French) and InFoBench's decomposed requirements, on their parts a
   program can score, beside their published numbers; the share of real users' instructions left "asked", by kind.
- **Benchmarks are for measuring:** licences checked first; no benchmark prompt in a training row; parts those sets
  grade with a model judge are not scored here (no model judges a reply).
- **Not claimed:** that Cubby follows more complex instructions than a frontier model. The claims are the kill lines
  and the content and process checks, which a model that writes free text cannot show.

- **E's first case (Claude Code, 2026-11-13):** line 4's run 1 on `exp_if_follow` held a prompt whose two rules
  cannot both hold: a JSON key "u1" and all capitals. Run 1 said a reply breaking one of them (a JSON-key bug, fixed);
  run 2 refused it after 4 candidates. With E, the compile step would have found the two bounds on one reply and asked
  which wins, before any writing.

## Cowork's check of step 3 and its adapter round (2026-11-12, late): hold the round until the wordings are clean
Read: the two sections above, `rule_agree_2026-11-12_gold.json`, `rule_reader_2026-11-12_asis.json`,
`validation/builders/rule_templates.py`, `validation/sft/build_rule_reader_sft.py`.
- **Accepted:** the reply world's measurements for the seen kinds; the plan family against IFEval's own checker,
  998 of 1,000 agree, and the rule never passes a reply the checker fails; the reader with constrained slots through
  0071's hook; the seat as it is missing line 3 (50.4% agreement, 32.3% false passes); the held-out kinds left unread.
- **Not accepted yet: the training wordings.** In `rule_templates.py`, the first `train` wording of most seen kinds
  is IFEval's own instruction sentence, word for word or nearly: "Your entire output should be wrapped in JSON
  format."; "Your entire response should be in English, and in all lowercase letters. No capital letters are
  allowed."; "In your response, the word {kw} should appear {rel} {n} times."; "First repeat the request word for
  word without change, then give your answer"; "Wrap your entire response with double quotation marks."; the
  sections, highlight, title, capitals, language, letter and two-responses wordings likewise. That puts the
  benchmark's prompts into training rows, which the order's constraints forbid ("No benchmark prompt reaches a
  training row"), and line 4's IFEval read would measure recall of IFEval's sentences, not the reader.
- **Before the adapter round runs:**
  1. Every `train` wording that shares 5 or more consecutive words with an IFEval prompt (slots filled) is rewritten
     in our own words. The same check against M-IFEval's French prompts for the French wordings.
  2. A test pins it: no `train` or `dev` wording shares 5 consecutive words with any IFEval or M-IFEval prompt. It
     reads the benchmarks from their off-repo place (an environment variable) and skips where they are absent.
  3. IFEval's own phrasings are then a true new-wording read: IFEval itself. The `new` split may keep wordings close
     to IFEval's, since nothing trains on it.
  4. The SFT set and the notebook are rebuilt from the clean wordings; the dev read of the seat as it is stays as
     recorded (it trained on nothing).
- **For Nick:** the Colab notebook `rule_reader.ipynb` should not be run until this is done. An adapter already
  trained from the current set would be set aside, not used.

## Cowork's check of the rewritten wordings, the VM, and line 4's two runs (2026-11-13)
- **Line 4, accepted as recorded:** run 1 (73.5%, broken 1) and run 2 (73.1% followed, 26.9% refused, 0 broken, on
  the cowork VM) side by side; both labelled "given the right rules", the end-to-end number waiting for the reader;
  line 1's clean read left to IFEval and M-IFEval's French. Line 4 is met for the turn; line 1 was hit once on
  `exp_if_follow` and its cause fixed.
- **The VM, accepted:** every run names its VM (path without drive or user folder, time, sha256, commit) and refuses
  one older than its sources; the served brain was already on the cowork build.
- **The wordings, accepted, with two fixes before the round runs:** 0 train or dev rows share 5 words with an IFEval
  prompt (pinned; the first set set aside, never trained on). Two new wordings say something other than their rule:
  1. `english_capital`: "with capital letters for every word" / "chaque mot en capitales" reads as a capital at the
     start of each word, not every letter in capitals (and is close to `capital_word_frequency`, a held-out kind).
     Say every letter: "in English, every letter a capital" / "en anglais, tout en majuscules".
  2. `english_lowercase` in French: "des lettres toutes petites" reads as small type. "Tout en minuscules".
- **Then the hold is lifted:** once those two are rebuilt into the set, Nick may run `rule_reader.ipynb`.
- **M-IFEval's French check skips** (the set is not on this machine; its licence unread). It runs when the set is
  fetched, after the licence is read; until then the French wordings are ours, translated from our English.
