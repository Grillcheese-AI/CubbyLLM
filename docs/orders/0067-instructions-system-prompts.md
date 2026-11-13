# 0067 - Instructions and system prompts: read by the model, kept as rules, enforced and checked by the host
status: in progress (Nick, 2026-11-12: 0067 first, then 0070, then 0069; 2026-11-13: Code A stays on 0067, and 0070 then 0069 go to Code B, a second Claude Code session, in parallel)
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
### The reader's adapter round, read (Claude Code, 2026-11-13)
- **The adapter** (`cubby450m_rules_v1`, Colab, on the clean set `c196de53`; one epoch, 312 steps of 32, 1.4 minutes;
  held answer loss 5.39 -> 0.0078), read on grilly2 with the instruction alone (no examples), slots constrained.
- **Dev** (150 rows, the training wordings, other rules; `rule_reader_2026-11-13_v1_dev.json`): first read 140 exact;
  two slot faults were fixed on dev (`b94dbce`: word and opening values as the reply world reads words, so no
  punctuation; up to 6 leaves, the rows compose 5); second read **149 of 150 exact** (EN 59/60, FR 90/90).
- **New wordings, read once on `b94dbce`** (150 rows, wordings and comparison phrases training never showed;
  `rule_reader_2026-11-13_v1_new.json`): **100 of 150 exact (66.7%)**, the measurement right on 109 (EN 56/79, FR
  44/71). Against IFEval's checker (`rule_agree_2026-11-13_v1_new.json`, 135 rules with an IFEval form, 1,350 replies,
  the cowork VM): **agreement 82.3%; the rule passes a reply the checker fails 8.7%.** Lowest: highlighted sections
  0.59, two responses 0.68, all capitals 0.70 (it drops the capitals leaf from "every letter capitalised"), frequency
  0.75, existence 0.77.
- **Line 3, new wordings of seen kinds: missed** (agreement 82.3% against 98%; written 100% means nothing with the
  slots constrained). The 8.7% false passes is above the 5% that line 3's kill sets for held-out kinds. **The held-out
  kinds stay unread.** The gap is the wordings: three training wordings a kind are learned almost exactly (dev 99%)
  and generalize partly (66.7%).
- **What a next round would need** (Nick's call): many more wordings a kind. The order allows wording templates to be
  model-drafted (the 2026-11-10 rule; rules never are), each checked against the benchmarks (`bench_overlap.py`) and
  the new-wording templates kept apart; a fresh new-wording split would be drawn, since this one is now read.
### The reader's second round, built for Colab (Nick, 2026-11-13: "go ahead"; Claude Code)
- **Wordings** (`validation/builders/rule_wordings_v2.py`): 10 training wordings a kind and language (round 1's 3,
  round 1's 2 spent new wordings, 5 more), and a **fresh new-wording split** of 3 a kind and language, in no training
  list and never read in round 1. Drafted by hand here, not through a paid model (the 2026-11-10 rule allows
  model-drafted wordings; no rule is). Round 1's new wordings that sat close to IFEval are reworded on their way into
  training (`REWORD`); round 1's own rows stay as they were read. 0 train/dev rows share 5 words with an IFEval
  prompt; pinned (`test_round_two_trains_clean_and_reads_fresh_wordings`).
- **Rows** (`build_rule_rows.py --wordings v2 --seed 1`, off-repo `rule_rows_0067_v2`, sha `a806725f...`): 12,870,
  1,800 new-wording rows. `--wordings v1` still rebuilds round 1's rows bit for bit.
- **The set** (on Drive, `rules/rule_reader_sft_v2.jsonl`, sha `6bc5c027...`): 11,070 records (9,927 train, 1,143
  held); 0 sharing 5 words with IFEval; 0 slot boundaries tokenized otherwise.
- **The notebook** (`rule_reader.ipynb`, rebuilt): the same settings as round 1, the round 2 set, writes
  `rules/cubby450m_rules_v2/` (round 1's adapter untouched).
- **The read at home:** dev (`--rows` the v2 rows), then the fresh new wordings once, then IFEval's checker on those
  plans; the held-out kinds once, only if line 3 is met.
### The reader's second round, read (Claude Code, 2026-11-13)
- **The adapter** (`cubby450m_rules_v2`, on `rule_reader_sft_v2.jsonl`, hash checked; 311 steps, 1.3 minutes; held
  answer loss 5.56 -> 0.0055), read on grilly2 with the instruction alone, the cowork VM.
- **Dev** (150 rows of the v2 rows; `rule_reader_2026-11-13_v2_dev.json`): **149 of 150 exact** (the miss: an opening
  value that ran on, "yes Answer in English without").
- **The fresh new wordings, read once** (150 rows; `rule_reader_2026-11-13_v2_new.json`): **131 of 150 exact (87.3%)**,
  the measurement right on 141 (EN 67/78, FR 64/72). Against IFEval's checker (`rule_agree_2026-11-13_v2_new.json`,
  132 rules, 1,320 replies): **agreement 92.7%; the rule passes a reply the checker fails 3.9%.** Round 1: 66.7%,
  82.3%, 8.7%.
- **Where it misses:** all capitals 0.62 (35 of the 51 false passes: "with no lowercase letters" read as lowercase;
  "CAPS", "fully uppercase" losing the case leaf); letter counts 0.75 (the new rows word "<" as "less than", which
  training never shows by design, and the reader writes ">="); title 0.81; number of words 0.91; paragraphs 0.92.
  Without the all-capitals kind the agreement would be about 96%.
- **Line 3, new wordings of seen kinds: missed** (92.7% against 98%). The false passes (3.9%) are under the 5% that
  line 3's kill sets for held-out kinds. **The held-out kinds stay unread.** This split is spent; nothing was tuned on it.
- **What remains is narrow:** saying "capital" without the word (CAPS, uppercase, "no lowercase") and comparisons in
  words training never had. A third round, if Nick wants one, would add wordings for those two and draw a fresh split.
### The reader's third round, built for Colab (Nick, 2026-11-13: "yes build it"; Claude Code)
- **Aimed at round 2's two gaps** (`validation/builders/rule_wordings_v3.py`): all capitals and all lowercase said
  without the word (caps, uppercase, block letters, "no lowercase", "no uppercase", "no caps"; 7 more a kind and
  language, each kind the other's contrast), more letter- and word-count wordings, and more training comparisons
  ("not below", "short of", "not as many as", "not fewer than"; French alike). "No less than" and "at minimum" were
  tried and left out: each met an IFEval prompt in 5 words. IFEval's "at least" / "less than" stay out of training and
  still word the new rows.
- Training: round 2's 10 a kind, round 2's spent new wordings (one reworded on its way in), and the additions; a
  **fresh new-wording split** (3 a kind and language, in no training list and no earlier round).
- **Rows** (`--wordings v3 --seed 2`, off-repo `rule_rows_0067_v3`, sha `d72b9f71...`): 12,883; rounds 1 and 2 still
  rebuild bit for bit. 0 train/dev rows share 5 words with IFEval (pinned:
  `test_round_three_trains_clean_and_reads_fresh_wordings`).
- **The set** on Drive (`rules/rule_reader_sft_v3.jsonl`, sha `bba37b82...`): 11,083 records (9,945 train, 1,138
  held); 0 overlaps; 0 slot boundaries tokenized otherwise. The notebook trains it into `rules/cubby450m_rules_v3/`
  with round 1's settings; earlier adapters untouched.
### The reader's third round, read (Claude Code, 2026-11-13)
- **The adapter** (`cubby450m_rules_v3`, on `rule_reader_sft_v3.jsonl`, hash checked; 311 steps; held answer loss
  0.0090), read on grilly2, the instruction alone, the cowork VM.
- **Dev:** 148 of 150 exact, the measurement right on 150 (`rule_reader_2026-11-13_v3_dev.json`).
- **The fresh new wordings, read once** (`rule_reader_2026-11-13_v3_new.json`, `rule_agree_2026-11-13_v3_new.json`):
  133 of 150 exact (EN 74/78, FR 59/72); against IFEval's checker (132 rules, 1,320 replies) **agreement 91.7%, the
  rule passes a reply the checker fails 4.8%.**

| round | training wordings a kind | new wordings exact | agreement | false passes |
|---|---|---|---|---|
| 1 | 3 | 100 / 150 | 82.3% | 8.7% |
| 2 | 10 | 131 / 150 | 92.7% | 3.9% |
| 3 | 13-20 | 133 / 150 | 91.7% | 4.8% |

- **All capitals improved** (0.62 -> 0.78). **The comparisons did not:** word frequency 0.60, sentences 0.71, letters
  0.75; nearly every remaining miss is "less than" written as ">=". Training words "<" as fewer than / under / below
  / short of, and now ">=" partly by negations (not fewer than, not below, not under); "less than", which training
  never shows by the order's split, falls on the wrong side.
- **Line 3, new wordings of seen kinds: missed again** (91.7% against 98%); false passes under 5%. The held-out kinds
  stay unread. The split is spent; nothing was tuned on it.
- **The question this leaves (Nick's or Cowork's call, not taken here):** whether everyday comparison phrases ("at
  least", "less than") may appear in training wordings. They were kept out because IFEval uses them, but they are
  ordinary English, and the 5-word overlap check would still keep IFEval's sentences out. If they may, the new-wording
  split would need other unseen comparisons ("a maximum of" for "<=" is not the same rule, so the choice needs care).
  If not, more training comparisons alone did not close the gap this round.
### The reader's fourth round, built for Colab (Nick, 2026-11-13: "yes allow them, build round 4"; Claude Code)
- **Nick's call:** everyday comparisons may enter training wordings; the 5-word check still keeps IFEval's sentences
  out. Round 4's training comparisons are round 3's plus "at least" / "less than" (`rule_wordings_v4.py`); its
  training wordings are round 3's plus round 3's spent new wordings (one reworded). Its **fresh new-wording split**
  (3 a kind and language) words comparisons as training never did: English "at a minimum" / "not any fewer than"
  (">=") and "lower than" / "beneath" ("<"), each the same rule as the phrase it stands for; French "au minimum" /
  "en deçà de" (never trained in any round).
- **Numbers can make a clean wording meet IFEval** ("appear at least 4 times", "use less than 300 words"), so for
  round 4 the builder draws a train/dev row again when it would share 5 words with a benchmark prompt (where the
  prompts are on this machine). Rounds 1-3 still rebuild bit for bit.
- **Rows** (`--wordings v4 --seed 3`, `rule_rows_0067_v4`, sha `57353bd8...`): 12,857; "at least" in 158 and "less than"
  in 110 English train/dev rows; the unseen comparisons in 200 English new-wording rows. 0 train/dev overlaps (pinned:
  `test_round_four_trains_clean_and_reads_comparisons_it_never_trained_on`).
- **The set** on Drive (`rules/rule_reader_sft_v4.jsonl`, sha `cb7a7260...`): 11,057 records (9,964 train, 1,093
  held); 0 overlaps; 0 slot boundaries tokenized otherwise. The notebook trains it into `rules/cubby450m_rules_v4/`.
### Round 4 with Nick's annotated instruction file folded in (Nick, 2026-11-13: "fold it in now"; Claude Code)
- **The source:** Nick's annotated instruction file (`$CUBBY_INSTRUCT_SVC`): Stanford Alpaca's prompts (text CC BY-NC
  4.0: internal research builds, not a commercial release) with Nick's own annotations (subject / verb / complement,
  parts of speech, lemmas, dependencies). In his split, a reply constraint sits in the complement ("in one sentence"),
  the task's content in the subject.
- **Natural length rules** (`validation/builders/alpaca_rows.py`): whole prompts whose one length phrase a strict
  grammar reads in full ("in a maximum of N words" -> word count <= N; "in N words or less" -> <= N; "in under N
  words" -> < N; "in one sentence" -> == 1; "in 1 to 2 sentences" -> 1..2); an ambiguous phrase ("in 50 words"), two
  length phrases, or any other rule kind -> left out. **This departs from the order's "built in reverse"** (a rule
  drawn, then worded): here the wording is real and the host's grammar reads the rule. No model fixes a rule.
- **Rule-free prompts:** no hit from the grammar or `exp_if_follow.rules_of`, no constraint word, no number in Nick's
  annotated complement; their target is the reader's new **`none`** (the instruction sets no rule), kept apart from
  `asked` (a rule no measurement covers). `rule_reader.read` takes `none` at its first slot.
- **Counts:** 581 natural rules (116 held out as `natural_test`), 2,000 rule-free prompts (300 held out as
  `none_test`); 48 train/dev prompts dropped for sharing 5 words with IFEval. Round 4's rows (`--wordings v4 --seed 3
  --alpaca`, sha `d86c007c...`): 15,390. The set on Drive (`rules/rule_reader_sft_v4.jsonl`, sha `deefbe82...`):
  13,174 records (templated 9,964 / 1,093; natural 419 / 32; none 1,483 / 183, train / held), 0 overlaps, 0 slot
  boundaries tokenized otherwise, at most 96 tokens. The set without them is moved to `rules/_discard/`.
- **The read adds two lines of its own:** the natural rules held out (exact; IFEval's checker where it can state the
  rule) and the rule-free prompts held out (**how often the reader invents a rule**: a rule invented on a plain task
  would make the turn refuse a good answer).
### The reader's fourth round, read (Claude Code, 2026-11-13)
- **The adapter** (`cubby450m_rules_v4`, on `rule_reader_sft_v4.jsonl` with Nick's annotated prompts, hash checked;
  371 steps, 1.5 minutes; held answer loss 0.0077), read on grilly2, the instruction alone, the cowork VM. Each held-out
  split read once.

| read | rows | exact | IFEval's checker: agreement | false passes |
|---|---|---|---|---|
| dev (templated and Alpaca dev rows) | 150 | 148 (26 rule-free: 0 invented) | — | — |
| fresh new wordings | 150 | 136 (90.7%; EN 80/81, FR 56/69) | **90.5%** (130 rules, 1,300 replies) | 5.1% |
| Nick's natural length rules, held out | 116 | 110 (94.8%) | **97.7%** (30 rules IFEval states, 300 replies) | 2.3% |
| Nick's rule-free prompts, held out | 300 | 299 said `none` | — | **1 rule invented (0.3%)** |

- **All capitals fixed** (0.955; round 2: 0.62). **The comparisons are still the gap on new wordings:** word frequency
  0.33, letters 0.71, words 0.80, sentences 0.85; the new split's comparisons are phrasings no round trained on
  ("lower than", "beneath", "en deçà de"), and the reader still writes ">=" for most of them. Three rounds show the
  reader learns the shape of rules across wordings, and does not learn comparison words it never saw.
- **On real prompts** (Nick's annotated Alpaca prompts, held out): 97.7% agreement, 2.3% false passes; 2 of 116 read
  `none` where there was a rule (the rule is lost; nothing is broken). **On plain tasks:** 1 rule invented in 300.
- **Line 3, new wordings of seen kinds: missed** (90.5% against 98%; written 100%). The held-out kinds stay unread.
  Logs: `rule_reader_2026-11-13_v4_{dev,new,alpaca}.json`, `rule_agree_2026-11-13_v4_{new,natural}.json`.
- **For Nick:** the new-wording line, as built, asks the reader to read comparison words it has never been shown; four
  rounds have not met it that way. Real prompts read at 97.7%. Whether to read the held-out kinds now (line 3's
  second part), change how the new-wording split words its comparisons, or wire the reader as it is behind its flag
  with these rates recorded, is his call.
### Line 3's new-wording read with training's comparisons (Nick, 2026-11-13: "option 2, then read the held-out kinds")
- **Nick's change:** a new-wording row's comparisons are worded as training words them ("at least", "less than",
  "fewer than", ...). So that this is not a re-scoring of templates already read, the split is **fresh**: 108 new
  templates (`validation/builders/rule_wordings_v4b.py`), in no training list and no earlier split (checked);
  training identical to round 4's (checked). Rows `--wordings v4b --seed 5` (sha `49299be7...`). Read once with the
  round-4 adapter as trained.
- **Result** (`rule_reader_2026-11-13_v4b_new.json`, `rule_agree_2026-11-13_v4b_new.json`): 128 of 150 exact; **a rule
  written for 144 of 150 (96%; line: 90%)**; 6 read `none`. Where written, against IFEval's checker (134 rules, 1,280
  replies): **agreement 95.7% (line: 98%), the rule passes a reply the checker fails 0.16%** (2 of 1,280). Lowest:
  quotation 0.72 (an extra leaf, `count of the word "end"`, on a right rule), number of words 0.89.
- **Line 3, new wordings: written met, agreement missed by 2.3 points**; false passes nearly nil. The held-out kinds
  are read next, as Nick said.
### Line 3's held-out read: the kill line is hit (Claude Code, 2026-11-13)
- **The read** (`validation/exp_rule_heldout.py`, once; `rule_heldout_2026-11-13_v4.json`; the round-4 adapter, the
  cowork VM): each of the 9 held-out IFEval kinds in IFEval's own wording, 30 instructions a kind with arguments the
  checker drew from seeds 0-29 (`ifeval_check.py --describe`); where a rule was written, 5 replies built near its
  edge (`rule_replies.held_out`), labelled by IFEval's checker, judged by the VM. M-IFEval's French kinds: not read
  (its prompts are not on this machine).

| line 3, held-out | line | the 7 untouched kinds | forbidden words (apart) | bullet lists (apart) |
|---|---|---|---|---|
| a rule written | >= 50% | **85.7%** (180 / 210) | 73% | 100% |
| agreement where written | >= 95% | **47.8%** (900 replies) | 42.7% | 64.0% |
| rule passes, checker fails | <= 2%, **kill > 5%** | **6.4%: kill** | 1.8% | 0% |

- **What it wrote:** the nearest familiar shape, confidently: "refrain from the use of any commas" -> the quotation
  rule (30 of 30, though `count of the character ","` exists); "finish your response with this exact phrase" ->
  `starts with` (not `ends with`); all-capital words -> `count of the word "words"` (false passes 37%); placeholders ->
  quotation / highlights; a postscript -> `opening has P S`; forbidden words -> `count of the word "not" >= 1`;
  "answer with one of the following options" -> `none` (30 of 30). It said `asked` 0 times in 270.
- **Why, as far as the rows show:** no training row ever had `asked` as its answer. Every row either carried a rule the
  reader could write or (round 4) none at all, so the reader never learned "a rule I cannot measure". On an unseen
  kind it forces the nearest plan. That is the opposite of what the turn needs: a wrong rule is checked faithfully,
  so its errors reach the user as wrong refusals or, here 6.4% of the time, as passes the instruction forbids.
- **Line 3: killed on held-out kinds** (and missed on new wordings by 2.3 points). **The reader is not wired into the
  turn**; `CB_RULES` stays off by default; rules reach the turn only as plans the host was given.
- **What a next attempt would need** (Nick's call): rows whose answer is `asked` (instructions about tone, style,
  persuasion, formality ... that no measurement covers), so abstaining is learned; and a fresh set of held-out kinds,
  since these 9 are now read. An abstention read (how often an unseen kind gets `asked` instead of a wrong rule) would
  join line 3.
### The reader's fifth round, built for Colab (Nick, 2026-11-13: "build it"; Claude Code)
- **Rows answering `asked`** (`validation/builders/asked_rows.py`): 32 instructions no measurement covers, each in 2
  English and 2 French wordings (tone, register, style, rhetoric; rhyme, passive voice, tense, questions only, figures
  of speech, British spelling); a seeded quarter of them held out whole (`asked_test`); each training phrase also
  joined to a measurable rule, whose answer is that rule alone; and Nick's annotated prompts carrying an explicit style
  phrase ("in a formal, academic style", "in a positive way"; "a persuasive essay" and "in a different way" left out:
  they are tasks). `rule_reader`, the SFT builder and `exp_rule_reader` tell `asked` from `none`.
- **Rows** (`--wordings v4 --seed 3 --alpaca --asked`, `rule_rows_0067_v4_asked`, sha `c248d954...`): 15,851; 92 held
  out as `asked_test` (32 phrases, 60 annotated prompts). 0 train/dev overlaps with IFEval.
- **The set** on Drive (`rules/rule_reader_sft_v5.jsonl`, sha `b4a9b2cb...`): 14,032 records; the `asked` records
  repeated 4 times (`--repeat-asked 4`) so abstaining weighs like `none`: train 10,552 rules, 1,483 `none`, 652 `asked`.
  0 overlaps; 0 slot boundaries tokenized otherwise. The notebook trains it into `rules/cubby450m_rules_v5/`.
- **A fresh held-out set:** IFEval's 25 kinds are all spent, so IFBench's test set (AllenAI, 300 prompts, 58 kinds no
  row was built from; data ODC-BY-1.0, checker code Apache-2.0) is fetched off-repo (`$CUBBY_DATASETS/ifbench`, 94 KB).
  Running its checker code (downloaded) was held by this machine's permission check: it waits for Nick's yes.
- **The read planned:** dev; `asked_test` (abstention on unmeasurable instructions never trained on); the regression
  splits (round 4's natural and rule-free tests, the v4b new wordings: second reads, labelled as such); IFBench (how
  often an unseen kind gets `asked` instead of a wrong rule; agreement and false passes where written).
### The reader's fifth round, read (Nick, 2026-11-13: "colab ran"; Claude Code)
Trained 397 steps on an A100 (held answer loss 0.0079). Read at home on grilly2, the cowork VM, slots constrained.
- **Dev** (`rule_reader_2026-11-13_v5_dev.json`): 146/150 exact; 0 rules invented on 26 rule-free rows; 2/2 `asked`.
- **Abstention, unmeasurable instructions never trained on** (`asked_test`, 92; `..._v5_heldout_splits.json`): `asked`
  73 (79.3%), `none` 13 (no rule either: mostly Nick's style-phrase prompts), **a rule written 6 (6.5%)**, 5 of them
  French: British spelling read as `language == en` (3), a word of the phrase made a count ("registre", "deux"), "un ton
  sérieux" read as an opening.
- **Second reads (regression, labelled as such):** natural length rules 110/116 exact, IFEval's checker **97.7%**
  agreement, **2.3%** false passes (round 4: the same); rule-free tasks **1 rule invented in 300** (round 4: 1), 3
  answered `asked`; v4b new wordings **96.2%** / **0.54%** (round 4: 95.7% / 0.16%; quotation read with the wording's
  own words in 31 of 110 pairs, none passing falsely).
- **Round 5's own new wordings** (unseen comparisons, a v4-style split): 139/150 exact; **92.1%** / **4.7%** (round 4:
  90.5% / 5.1%). Comparisons no round trained on still read as `>=`.
- **IFBench, fresh held-out kinds** (`rule_ifbench_2026-11-13_v5.json`; 300 prompts read whole, 58 kinds): a rule
  written **77 (25.7%)**, `none` 211 (70.3%), `asked` 12 (4.0%). Agreement not read (the checker waits). Read by eye,
  most written rules are invented: numbers and words not in the prompt ("sentence count == 22 ; paragraph count == 33"
  for a keyword position; "count of the word "life" >= 15"; a word range 57-61 read as two `>=`). Line 3's held-out
  part is not met on written share (25.7% < 50%), and the written rules look like false-pass risk.
- **IFBench's checker** (Nick allowed downloaded code, 2026-11-13; `validation/ifbench_check.py`,
  `exp_rule_ifbench_agree.py`, `rule_ifbench_agree_2026-11-13_v5.json`): the 77 written rules, 10 replies each (half
  aimed at the rule's own edge, half plain), labelled by IFBench's checker (every instruction of the prompt) and judged
  by the VM running the rule. Agreement **70.8%**, false passes **21.7% (kill > 5%)**; of the replies a written rule
  passed, 88.8% fail the checker. **Line 3's held-out kill line is hit on IFBench too.** The worst kinds: unique word
  count, a keyword in a sentence, word-count ranges, stop-word ratios, alphabet and consonant rules: each read as the
  nearest familiar measurement.
- **Reading:** abstention is learned on short instructions (0/270 -> 79% on unseen short phrases) but not on long real
  prompts: there the reader mostly says `none` and, a quarter of the time, invents. Training rows are short (instructions
  alone, or Nick's prompts at most 300 characters); IFBench's prompts are a task with the constraint inside (median
  293 characters, up to 904). Unwired; CB_RULES off.
- **Next (Nick's call):** rows built from long, real, human-written prompts with known licences (a constraint joined inside a
  prompt; `asked` and `none` on long prompts). Nick proposed `crumb/Clean-Instruct-3M`: no licence and no card on the
  Hub, its sources unlisted (the preview shows templated tasks, a StackOverflow post, answers that read model-written),
  1.17 GB; not fetched. Alternatives with known licences and human prompts: OASST1 (Apache-2.0), Dolly-15k (CC BY-SA
  3.0).
### The reader's sixth round, built for Colab (Nick, 2026-11-13: "both"; Claude Code)
Nick chose OASST1 and Dolly-15k (and is asking about CubeLang programs in the set: the rows keep plans, so the answer's
form can change before training). Both fetched off-repo (`$CUBBY_DATASETS/oasst1`: the first-turn prompts file, 13 MB,
Apache-2.0, its licence beside it; `$CUBBY_DATASETS/dolly15k`: 13 MB, CC BY-SA 3.0, its card beside it). Prompts only.
- **Rows from real prompts** (`validation/builders/real_rows.py`): OASST1 first turns (English, French; not deleted,
  reviewed, not synthetic) and Dolly instructions (the instruction screened; its Wikipedia passage, when it has one,
  put before or after it: numbers in a passage are not counts to keep). Every prompt kept carries no constraint
  (round 4's screens and a French one), so each row's rule is known by construction; each prompt is used once:
  alone -> `none`; with a training-wording rule put after it, before it, or after its first sentence -> that rule
  (15% with a phrase beside it); with a phrase no measurement covers -> `asked`. Prompts under 200 characters thinned
  (English median 144 characters; 2,470 English prompts of 300+, IFBench's median 293).
- **Structure phrases** (`struct_phrases.py`): 24 instructions on content or structure the reply world does not measure
  (a dialogue, an analogy, step by step, three examples, the five senses, ...), 2 English and 2 French wordings,
  kept clear of IFBench's 58 kinds and IFEval's 9 held-out kinds; a seeded quarter held out whole.
- **Held out, never trained:** `real_none_test`, `real_rule_test` (rules keep IFEval's form for its checker),
  `real_asked_test` (held-out structure and tone phrases), 300 English + 60 French each; IFBench stays the fresh kinds.
- **The overlap check now includes IFBench** for round 6 (`bench_overlap.WITH_IFBENCH`, set by `--real`): 657 real
  rows dropped. Off by default, so rounds 1-5 rebuild to their recorded hashes (round 5's `c248d954...` checked).
- **Rows** (`--wordings v4 --seed 3 --alpaca --asked --real`, `rule_rows_0067_v4_asked_real`, sha `80d5d215...`):
  26,267 (10,426 real). **The set** (`rules/rule_reader_sft_v6.jsonl`, sha `d7cb9146...`, `--repeat-asked 2`): 24,726
  records (train 22,458); 0 sharing 5 words with IFEval or IFBench; 0 slot boundaries tokenized otherwise.
- **Found on the way:** the trainer skips a record longer than `--max-len` without a word (`talk_data.encode_records`);
  round 5 lost none at 160, but long prompts would have been dropped. The notebook trains at `MAX_LEN` 640 (the
  longest prompt is 546 tokens) and the SFT manifest now counts records over it (0).
- `build_rule_rows.py` reached 316 lines: its extra row sets moved to `rule_rows_extra.py`.
- **The first Colab try ran out of memory** (2026-11-13, on a 22 GB L4, not an A100): a batch of 32 long prompts at
  step 1. `train_talk_torch.py --micro-tokens N` cuts a batch into chunks of at most N padded tokens, each chunk's
  loss weighted by its share of the answer tokens (the same gradient: `tests/validation/test_talk_micro_batches.py`;
  off, one chunk weighted exactly 1.0, the smoke run's losses identical). The notebook sets 4,096 on a card under
  70 GB, off otherwise, and refuses a set with records over `MAX_LEN`.
- **The read planned:** dev; the three real test splits (real rules through IFEval's checker); the regression splits
  (second reads, labelled); IFBench again (written share, then agreement with its checker).
### The reader's sixth round, read (Nick, 2026-11-13; Claude Code, Code A)
Trained on Colab's third try (an A100 80 GB; an L4 and an A100 40 GB ran out of memory first, the third try ran
without `--micro-tokens`): 702 steps, 3.0 min, held answer loss 0.0057. Read at home: grilly2 `03279a0` (the hook,
merged to main as `d593633`), the cowork VM, slots constrained. Licences beside the adapter (`row_licences.json`).
- **Dev** (`rule_reader_2026-11-13_v6_dev.json`): 147/150 exact; 13/13 `asked` rows abstained; 0 invented.
- **Real prompts, held out** (`..._v6_real.json`, 360 each, 300 EN + 60 FR):
  - no rule: **1 rule invented in 360**, 15 `asked`, 344 `none`;
  - a rule inside: 357/360 exact; IFEval's checker **99.7%** agreement, **0 false passes** in 3,190 replies
    (`rule_agree_2026-11-13_v6_real_rule.json`);
  - a phrase no measurement covers (held-out phrases): 315 `asked`, 42 `none`, **3 rules written** (0.8%).
- **Second reads (regression):** round 5's unmeasurable test 79/92 `asked`, **0 rules written** (round 5: 6); natural
  115/116, 97.7% / 2.3% (unchanged); rule-free tasks **0 invented in 300** (round 5: 1); v4b new wordings 137/150,
  97.8% / 0.15% (round 5: 96.2% / 0.54%).
- **IFBench, a diagnostic** (spent by round 5; `rule_ifbench_2026-11-13_v6.json`, `rule_ifbench_agree_2026-11-13_v6.json`):
  a rule written for **66.0%** (round 5: 25.7%), `asked` 15.3%, `none` 18.7%. Where written, agreement **56.8%**, false
  passes **36.3%** (round 5: 21.7%); of the replies a written rule passed, 87% fail IFBench's checker. Worst: stop-word
  ratios, palindromes, a keyword in a given sentence, consonant rules, unique word counts, each read as a count the
  reply world keeps ("keyword paradox seven times" -> `count of the word "paradox" >= 1`; a keyword position ->
  `sentence count >= 22`).
- **Reading:** on the kinds it knows the plan reader now reads real, long prompts nearly perfectly and abstains on
  unmeasurable phrases; on constraint kinds it has no measurement for, round 6 taught it that a constraint inside a
  long prompt is to be written, so it writes the nearest measurement more often. A fixed-form reader cannot say a
  kind it has no form for is new; the abstention it learned is for phrases, not for structures. That is the case for
  the second reader (CubeLang, the addendum), which this read unblocks as its baseline. The plan reader stays unwired;
  CB_RULES off.
- **A host guard worth testing (proposed, not built):** every number and quoted word in a written plan must appear in
  the instruction (as digits or a number word), else the plan is `asked`. It would refuse the invented numbers
  ("sentence count >= 22" from a prompt with no 22) but not a wrong measurement with a right number ("count of the
  word paradox >= 1" from "seven times"). Measured on round 6's IFBench plans before any decision.
### The host guard, built and measured (Cowork, 2026-11-13: "build and measure it, no model call"; Code A)
`cubbyllm/speech/rule_guard.py` (unwired; CB_RULES off): a written plan is `asked` unless (1) every count it compares
above 1 is in the instruction as digits or a number word (EN, FR), (2) every argument it carries is in the
instruction, (3) its measurement's unit is named there (a lexicon per measurement, EN and FR). The lexicon was
tuned on the train and dev rows' own plans only: 0 of 15,903 asked. Tests: `tests/speech/test_rule_guard.py`.
Read once (`validation/exp_rule_guard.py`, `rule_guard_2026-11-13_v6.json`):
- **The line (seen kinds in real prompts):** round 6's real-prompt rule test: the guard asks on **0 of 357** plans
  that are right today (line: at most 1 point) and catches 1 of the 3 wrong ones. Natural prompts 0 of 115.
- Also read: v4b's new wordings (not real prompts, so outside the line): 4 of 137 right plans asked (2.9%), two
  wordings the lexicon lacks ("une première réponse et une seconde": two without the word; "be certain to write
  travel": no word for "word"). Not added: the split is read; a later lexicon change takes a fresh read. Wrong plans
  on the held-out unmeasurable phrases: 2 of 3 caught; the 1 rule invented on 360 rule-free real prompts: not caught.
- **IFBench (a diagnostic):** the guard asks on 58 of the 198 written plans (part 1: 17, part 2: 4, part 3: 49) and
  catches **167 of 718** false passes (part 1: 15, part 2: 14, part 3: 155); false passes 36.3% -> **27.8%**, still far
  above the kill line. It would also turn 20 right verdicts into "asked". Most false passes keep the instruction's own
  unit and word with another meaning ("keyword paradox seven times" -> the word "paradox" at least once): a check on
  words cannot see a measurement read as a neighbour. The guard is a floor for the plan reader, not a fix; the second
  reader is.
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

## Cowork's addendum (2026-11-13): a second reader that writes CubeLang, splits long instructions and joins the parts
Nick, 2026-11-13: "for instructs we should maybe create a dataset based on what we use with cubelang code samples so
it learns to generate unseen commands", then "it should be able to split long instructions and join them as parts
into logical steps too". Queue: after the plan reader's Colab round is read, which is this reader's baseline. It is
the reader the complex-instructions addendum's line 2 asks for.

**Why a second reader.** The plan reader fills a fixed form (one of about twenty measurements, a comparison, a
value), so a kind of instruction nobody wrote a measurement for can only be "asked". A reader that writes the check
in CubeLang composes a few primitive views of the reply instead; an unseen instruction is a new composition, not new
host code. The 450M's programs seat already writes CubeLang for questions; this reader starts from it.

**Build:**
0. **The VM, checked against a build** (never the docs): which operations the generator below needs exist (loops
   over a list, counts, filters, all and any, string equality, prefix and suffix, case, numeric comparison, `&&` and
   `||` as operands). What is missing goes on cubelang's fix list; a cubelang change waits for Nick's yes.
1. **Primitive views in the reply world:** the reply's characters, words, sentences, lines, paragraphs, list items,
   JSON keys and values, each as a list with positions, handed to the VM as statements. The existing measurements
   stay; new kinds use the views.
2. **A program generator:** a grammar over the views and CubeLang's constructs (count, filter, all, any, position,
   compare; and, or, not, if-then, unless, an order of parts). Each program is run on replies built for it and kept
   only if some pass and some fail.
3. **Long instructions, built in reverse.** 2 to 8 parts, each a program with its wording from templates (EN and FR),
   joined by connective wordings ("and then", "unless", "if ... otherwise", "the same in French"), the parts in varied
   order, with sentences that are not instructions mixed in. Each row's target:
   - **the split:** every sentence or clause of the instruction mapped to a part (with its span and its kind: a check
     on the reply, a content rule, a process step, "asked") or to "not an instruction";
   - **the join:** one CubeLang program, a function per part and a main that joins them in the stated logic and order.
4. **CubeLang code as a syntax curriculum:** the repo's examples and tests and the programs seat's rows. No benchmark.
5. **The reader:** decoding constrained to CubeLang's grammar through 0071's hook, so every program parses; a part
   whose program does not compile is "asked", never "checked".
6. **The record:** each part with its source span, how it acts (enforced, checked, asked) and the joined program, so
   a user can see how their instruction was understood. Nothing is dropped silently.

**Acceptance (pre-registered; each held-out read once):**
1. **Every sentence accounted for:** a part, "asked", or "not an instruction". Dropped silently: 0 (**kill**).
2. **The split:** each part's span and kind recovered for at least 90% of parts on held-out compositions.
3. **The join:** the reader's program agrees with the generator's on at least 95% of replies built to pass and to
   fail; it passes a reply the generator's program fails at most 2% (**kill** above 5%).
4. **Longer than seen:** trained on up to 4 parts, read on 5 to 8; agreement at least 90% of the in-length rate.
5. **Unseen kinds:** 0067's held-out kinds and IFBench's out-of-domain constraints: a compiling program for at least
   half; where written, agreement with the benchmark's own checker at least 95% and false passes at most 2% (**kill**
   above 5%).
6. **Against the plan reader:** the same held-out reads, side by side.
7. **Reported:** the share left "asked" by kind, program lengths, compile failures, seconds per instruction.

**Constraints:** no model writes a rule or a program (rows built in reverse, every program run before it is kept);
no benchmark sentence in a training row (the overlap test); CubeLang's capabilities read from a build; cubelang
changes only in the cubelang repo with Nick's yes.

**Nick's calls before the build:** the cubelang fixes step 0 finds; whether a model may draft wording templates
(paraphrases only; the programs stay generated and run; his rule of 2026-11-10 allows model-written wording, never a
fact); the Colab round for this reader when its set is built.

### Step 0, read: the generator's operations on the VM build (Nick, 2026-11-13: "go ahead with step 0"; Code A)
`validation/vm_ops_probe.py` runs one small `ISolve` program per operation on the cowork build (`3f37515`, sha
`3120403f...`) and counts it present only when it returns the expected value (`vm_ops_probe_2026-11-13.json`).
- **Present (run, right values):** list literals and `for (let x of xs)`; a list's length and an index, each bound
  first (`let n = xs.len()`, `let v = xs[i]`); lists of lists; maps (`let v = m["k"]`); a string's length (`let n =
  s.len()`); counting with a filter; all and any (a flag in a loop); the six numeric comparisons; `==` / `!=` between
  strings (variable to literal and variable to variable); functions with parameters; `while`; `add` / `sub` / `mul`;
  an `else` branch and nested `if`s; a string handed in by the host.
- **Missing, with the form that works today:**
  1. `&&`, `||`, `!` as operands of a condition (compile error "BinOp/UnaryOp cannot be encoded as an operand"):
     nested `if`s, a flag set by two `if`s, an `else` branch (what `rule_plan` already writes).
  2. `else if` (parse error): `else { if ... }`.
  3. Arithmetic inside an expression (`a + b * 2`): `add` / `mul` statements. A call, a method call or an index only
     as a `let`'s right side, never inside a condition or a `return`.
  4. `break`: a `while` on a flag.
  5. Whole-number division and modulo: `div` gives 3.5 for 7/2; modulo only as a subtracting loop.
  6. String helpers (prefix, suffix, upper/lower case, contains, concatenation): none; and they cannot be composed
     in the VM, because **indexing a string returns null** (`"hello"[1]`, no error). Workable through the host's
     views (step 1): a word's characters handed in as a list (indexing a list of strings works), a character's case
     handed in as a view.
- **A defect, kill-line relevant:** a missing value (null: an index past a list's end, a string index, a misspelled
  variable name) or the number 0 **compares equal to any string literal** (`==` true, `!=` false); against a number it
  acts as 0 (`< 7` true). A generated program with an off-by-one index or a typo would pass a string check. Today's
  rule programs never reach it: the reply world measures a value for every relation (112 argument-relation checks on
  empty and odd replies: 0 empty measurements). Until the VM is fixed, the generator binds every value it compares
  from the host and never indexes past a view's length.
- **For cubelang's fix list (waits for Nick's yes):** (a) comparing null, or a string with a number, is an error (the
  rule is then not checked: "asked"), never equal; (b) `s[i]` on a string returns its character, or an error; (c)
  `&&`, `||`, `!` and `else if` in conditions; (d) expressions with calls, indexes and arithmetic as operands; (e)
  whole-number division and `%`; (f) `break`; (g) string helpers (`starts_with`, `ends_with`, `upper`, `lower`,
  `contains`, concatenation). (a) is the one that blocks the second reader's kill line; (b)-(g) make programs
  shorter and easier to constrain to a grammar, and each has a working form today.
- **Found on the way (fixed):** `rule_plan.check` did not check a value's type: a count compared with "yes" crashed
  while the program was written, instead of coming back "asked". It is refused now (test added); 0 of the 2,855
  plans the readers wrote in the logs are newly refused.

### Step 1, built: the reply's primitive views (Cowork, 2026-11-13: "step 1: go"; Code A)
`cubbyllm/speech/reply_views.py` (unwired; the generator, step 2, reads it): the reply as seven tables, one row per
item in reading order, each row with its place (`i`) and its characters (`start`, `end`):
- `char` (kind, case), `word` (lower, len, first, last, their lowercase, case, the sentence / line / paragraph it sits
  in), `sentence` (word count, first and last word, end mark), `line` (word count, first word, indent, a list marker
  or not), `paragraph` (between blank lines or `***`; word and sentence counts, first word), `item` (marker, level,
  text, word count), `json` (path, key, value, type; when the whole reply, fences off, is JSON).
- Words, sentences, lines and list items are the reply world's own splits, so the views agree with its measurements
  (a lone "-" is a word there too). The existing measurements stay.
- **Every value is computed in Python; the VM only reads lists** (cubelang's P6 not yet on the build: a string is
  never indexed in the VM, a missing value never compared). `literals(text, views)` binds each column as a parallel
  list (`word_first_lower[i]`) with `n_<view>`; an empty view still binds every column as `[]`, so no name a program
  reads is ever unbound. `statements` gives the record's lines ("<value> is the <column> of word 3 of the reply").
- Tests (`tests/speech/test_reply_views.py`, 18): counts agree with the reply world; every item is found where it says;
  lists, JSON and case; an empty reply binds every column; programs over the views run in the VM and agree with Python
  (upper-case words, words starting with a letter, questions); quotes, backslashes and newlines survive the literal.
- Next: step 2, the program generator over these views.

### Step 2, built: the program generator (Nick, 2026-11-13: "go ahead with step 2"; Code A)
- **Specs** (`validation/builders/view_programs.py`): an atom asks one view (count, all, any, none, how many, the item
  at a position) with up to two column conditions; specs join one or two atoms (and, or, not, if-then, unless). Values
  and thresholds are drawn from a pool of replies built for the spec (`view_replies.py`: prose, lists, mixed, JSON; EN
  and FR), so every spec holds for a fifth to four fifths of its pool. A Python evaluator gives the truth.
- **Held out, never drawn** (`held_out`): forms stating IFEval's held-out kinds or IFBench kinds the grammar can state
  (no commas or brackets, a word absent, list-item counts, a paragraph's first word, capitalised words counted, the last
  word, punctuation counted, every word title-cased, a word at a position; indents and list levels are no filter).
- **The compiler** (`view_compile.py`): only the forms step 0 found running (index loops bounded by `n_<view>`, values
  read with `let`, nested ifs and 0/1 flags; an `at` checks its index before reading). `body(spec)` is what the
  reader will write; the host prepends the views (`reply_views.literals`).
- **Kept only when run** (`build_view_programs.py`): 8 replies a program (half pass, half fail by the evaluator); the
  VM's verdict must be the evaluator's on every one (a disagreement stops the build and is saved whole), and some pass
  and some fail. **The set** (`$CUBBY_DATASETS/view_programs_0067`, seed 0, sha `8229de7a...` (rebuilt in step 5: values on one line),
  `view_programs_2026-11-13_manifest.json`): 2,000 programs, 1,148 shapes, joins: 912 single atoms, 242 not, 225 or,
  208 and, 207 unless, 206 if-then; **15,948 VM runs, 0 disagreements** (the rebuild: 15,956, 0). Tests: `tests/validation/test_view_programs.py`.
- **Found on the way (VM, for cubelang's fix list):**
  1. **A list of 254+ strings returns nothing; one of exactly 255 numbers errors** (cubelang's P6-8 now). The views
     refuse more than 250 rows (`reply_views.TooLong`, the check is "asked").
  2. **Names and string literals share 16-bit keys.** The VM keys every identifier and string literal by BLAKE3's
     first two bytes (cubelang `compiler.rs` `name_hash`, `engine.rs` `name_table`): two strings with one key read as
     one spelling, so they compare equal (`"thing" == "government"` holds: both `15d0`; a searched pair, "eptwcv" and
     "rmcens", too), and two variables with one key share a register. 121 of the first 500 programs of an unguarded
     build carried such a pair; the build's only disagreements were these (seeds 4-9 rerun with the guard: 12,000
     programs, 0 disagreements). Guard: `cubbyllm/bridges/vm_names.collisions` (fails closed without `blake3`);
     `rule_plan.run` now returns "asked" for a program with a shared key (before: "the reply lacks 'government'" was
     judged failed on "a thing here"), and the build leaves such replies out. **Every VM program in the serve path is
     exposed** (question programs, facts). Nick (2026-11-13): "should be fixed in Code B cubelang version": the fix
     (full-length keys or interning by the full string) is Code B's, in cubelang; no client guard in the serve path.
     The host guard stays in `rule_plan.run` and the step-2 build until the fixed build is the one `CUBELANG_EXE` names.
  3. The draw was not reproducible (set order follows the process's string hashing): values are sorted now; the same
     seed draws the same programs under any `PYTHONHASHSEED`. An earlier unseeded build is in `_discard`.

### Step 3, built: long instructions with their split and join (Code A, 2026-11-13)
- **Wording** (`validation/builders/view_wording.py`): every atom said in English and French by hand-written templates,
  as an instruction (`say`), its negation (`say_not`, the negation carried into the atom: "fewer than", "no ...",
  "the second sentence, if there is one, should not ...") and a condition (`say_cond`, for "if" and "unless"). Counts
  said as people say them ("at least one", "no", "a number of lines other than 3"). Read by eye on a sample and fixed
  where a wording said another rule (a numeric "!=" was said "exactly"); 0 French "aucun" without its "ne" in 8,538.
- **The join** (`view_join.py`): one function per part, taking the view columns it reads as arguments (a called
  function cannot read the caller's values on this build: probed), and a `solve` that calls the parts in the stated
  order and joins them: all groups must hold; "A unless B" (A or B), "if B, A", "either A or B".
- **Rows** (`build_long_rows.py`, `view_long_pools.py`): parts drawn one by one from the replies the parts so far still
  pass; a negated part checked against the held-out forms as the atom it equals; groups; then a task sentence (Nick's
  task words, `task_verbs` reads each one back), phrases no measurement covers (round 6's held-out phrases only in
  held-out rows), a process step, sentences that are not instructions, in EN or FR. Targets: `segments` (each sentence
  or clause, its character span and kind: check (its part, and "condition" for an if/unless clause), task, asked,
  process, none) and `join`. Parts numbered as the text states them. Kept when the joined program's VM verdict is the
  evaluator's on all 8 replies (4 pass, 4 fail); train/dev rows sharing 5 words with IFEval or IFBench dropped (34).
- **The set** (`$CUBBY_DATASETS/long_rows_0067`, seed 0, sha `ca2e6a0e...` (rebuilt in step 5), `long_rows_2026-11-13_manifest.json`):
  3,900 rows: train 3,000, dev 300, test 300 (2-4 parts, unseen compositions), test_long 300 (5-8 parts, line 4);
  EN 2,269, FR 1,631; segments: 12,884 checks, 3,109 tasks, 3,881 asked, 1,557 process, 3,777 not instructions;
  groups: 8,696 single, 1,075 either, 559 if, 460 unless. **31,091 VM runs, 0 disagreements** (the rebuild: 31,087, 0).
  Tests: `tests/validation/test_long_rows.py`.
- Next: step 4 (CubeLang as a syntax curriculum: the repo's examples and tests, the programs seat's rows), step 5
  (the reader, grammar-constrained through 0071's hook), then its Colab round; 0072 after step 3, as Nick set.

### Step 4, built: CubeLang as a syntax curriculum (Code A, 2026-11-13; 0072 done first, as Cowork set)
`validation/builders/build_cubelang_curriculum.py` -> `$CUBBY_DATASETS/cubelang_curriculum_0067/programs.jsonl`
(sha `a95da01d...` after the step-5 rebuild, `cubelang_curriculum_2026-11-13_manifest.json`): every program compiled on the cowork build
(`cubelang compile`) before it is kept; duplicates once; IFEval / IFBench 5-word overlap dropped. **No benchmark:** the
programs seat's GSM8K-derived arithmetic and every TinyGSM row are left out (the served programs seat's own set is all
such step fragments, so none of it is used); its identity rows are text. Kept 4,568: the cubelang repo's examples 14
(6 do not compile today), programs inside its tests 74 (30 do not: tests of errors, mostly), the programs seat's role
binding, reasoning kernels and verified chains 2,431 (27 shared 5 words with a benchmark), step 2's checks 2,000 (each
on its shortest reply), 0072's verb programs 49.

### Step 5, built: the reader, its set and its grammar, ready for Colab (Code A, 2026-11-13)
- **The form** (`cubbyllm/speech/view_reader.py`): after the instruction, one line per sentence or clause in order,
  `<kind>[ <part>][ condition] | <text copied exactly>`, then `---`, then the join. `parse` finds each segment's span
  in order; a segment the instruction does not hold makes the read "asked". All 3,900 rows round-trip exactly.
- **The grammar** (`cubbyllm/speech/view_grammar.py`, through 0071's hook): the split's labels from the closed set and
  its text copied forward from the instruction; the join's lines exactly the generator's shapes, in order (head, part
  functions, `solve`, close); end of answer only when it is complete. `cubbyllm/ops/grilly_seats.ranked_processor`
  lets it pick the best-scored token that keeps the answer viable (the top 32, then 512, then all). Every target is
  complete under it and every prefix viable; a program the shapes allow but the compiler refuses is "asked".
- **Found and fixed:** the generator drew values with a line break (the character "
", a JSON value with one); a
  wording put them in an instruction, which no person writes in one line and the split's one-line form cannot hold.
  Values with a line break, a tab or only spaces are no longer drawn; steps 2, 3 and 4 rebuilt from seed 0 (the
  earlier sets in `_discard`; the numbers above carry both).
- **The set** (`validation/sft/build_view_reader_sft.py` -> `rules/view_reader_sft_v1.jsonl` on Drive, sha
  `feeb7c6a...`): 9,018 records (train 8,402, held 616): long instructions with 2-4 parts 3,300 (train and dev only;
  test and test_long never trained), single checks said in EN/FR 1,165, the CubeLang curriculum as continuations 4,553;
  7 records sharing 5 words with IFEval/IFBench dropped; 15 longer than 1,280 tokens dropped (counted).
- **The run** (`notebooks/view_reader.ipynb`): an A100, rank 16, three epochs, `MAX_LEN` 1,280 (micro-batches on a
  smaller card); writes `rules/cubby450m_view_reader_v1/`. Nick's call (the addendum: "the Colab round for this reader
  when its set is built").
- **The read** (`validation/exp_view_reader.py`): test (2-4 parts) and test_long (5-8) once each; the split's span and
  kind per part; uncovered text flagged "asked" (nothing dropped silently); each program bound to a reply's views,
  compiled and run against the generator's verdict (agreement, false passes, the checked share); 5-8 over 2-4; seconds
  per instruction (a plumbing run: about 40 ms a token, so about 40 s an instruction).

### The P6 rerun (Code A, 2026-11-13; the cowork VM rebuilt with cubelang `676df1d`)
- **Steps 2 and 3, rerun once, every change listed** (`validation/rerun_vm_sets.py`, `vm_sets_rerun_2026-11-13_p6.json`):
  step 2 15,956 of 15,956 verdicts the same; step 3 31,087 of 31,087 the same. **No change.**
- **Step 0's probe again** (`vm_ops_probe_2026-11-13_p6.json`): a string's character by index now works (`let c =
  s[0]`), so prefix, suffix, case and contains compose right in the VM (each was a wrong value before: null compared
  equal); only string concatenation is still missing. The compact forms (`&&`, `else if`, a method call inside a
  condition) are unchanged.
- **The 2-byte name-key guard is out of the build path:** `build_view_programs`, `build_long_rows`, `exp_view_reader`
  and `rule_plan.run` no longer call `vm_names`; its test flipped (the VM tells "thing" from "government"; "lacks
  'government'" on "a thing here" is checked and holds). `vm_names` stays only to find 2-byte pairs in old logs.
- **The build is slower** (3.5x on the same programs; 0072's time line now fails: see 0072). The reader's read at home
  runs on this build; its seconds per instruction will carry it.

## Cowork's note (2026-11-13): a real long instruction to test the split and the join on
Nick pointed to the OpenThoughts sets in the training data (`unified/`). Read for this order:
- **AgentTrove** (`open_thoughts_agent_trove.jsonl`, already in the decay mix's agent shards): all 20,000 rows read
  open with the same long system prompt: reply as JSON with required fields (`analysis`, `plan`, `commands`), an
  optional `task_complete`, each command an object with `keystrokes` and `duration`, and rules on durations. It is a
  real long instruction with about ten parts.
  - **Use:** a test for the second reader, never a training row. The reader splits it and joins the parts into one
    program; the program runs on the trajectories' own replies. Report how many pass and show the ones that fail, read
    by hand: a rule the program got wrong, or a reply that broke the format.
- **OpenThoughts3** (`open_thoughts_3_600k.jsonl`, part of `unified`, already in the base's text): problems with long
  worked solutions (median about 62,000 characters a row). It holds no pairs of an instruction and its check, so it
  adds nothing direct to the reader's rows.

## Nick's call (2026-11-13): a model may draft wording variations for the readers' rows
Nick: "yes", to an outside model drafting extra phrasings of instructions for the readers' training rows. How it
runs:
- **Wording only.** The drafting model gets one template at a time ("answer in fewer than {n} words") and returns
  other ways to say it, in the same language, blanks kept. It never sees or writes a rule or a program; those still
  come from the generator and are run before they are kept.
- **Every phrasing filtered before it is kept:** the blanks intact; the benchmark-overlap test (no 5-word run shared
  with an IFEval or M-IFEval prompt); a hand-read sample of at least 10% per kind, and a kind whose sample shows a
  changed meaning is dropped whole and redrafted. Kept and dropped counts reported per kind and language.
- **Test wordings stay ours.** The `new` wordings that measure the readers are written by hand (or are the
  benchmarks' own, unseen); no drafting model writes them.
- **Which models:** only ones whose terms allow training on their output, checked on the day and recorded with the
  model ids. Two drafters, for variety. The script, its prompt and every output are kept off the training rows'
  path until filtered, and logged.
- **Cost:** stated before the run (estimated under $5) and spent on Nick's go, as for the panels.
- **Where it applies:** the second reader's rows first. The plan reader's first Colab round runs as built: it is the
  baseline. Drafted wordings join the plan reader only in a later round, so their effect can be told apart.

## Cowork's check of the reader's rounds 1 to 6 (2026-11-13)
Read: every round's sections above and their logs' summaries. The work is careful: each new-wording split fresh, each
held-out read once, the kill lines recorded where hit, the reader left unwired. Three things to settle before round
6's read, and one proposal.
- **IFBench is spent.** Round 5 read it once (written 25.7%; agreement 70.8%; false passes 21.7%: the kill line).
  Round 6 was built in answer to that read, so reading IFBench again is a second read, a diagnostic, and is labelled so;
  it cannot pass line 3. A clean held-out read needs kinds no round has seen: M-IFEval's five French kinds (fetched
  off-repo once its licence is read) or a set of new kinds written by hand with their programs, before any row of the
  next round is built.
- **The licences ride along in the adapters.** From round 4 on, the reader trains on Stanford Alpaca's prompts (CC
  BY-NC 4.0: non-commercial); round 6 adds Dolly-15k (CC BY-SA 3.0: share-alike). Each adapter's json and this order
  record the licences of its rows. `cubby450m_rules_v4` to `v6` are for research and demos; a reader for a commercial
  Cubby is trained without Alpaca, and with Dolly only if Nick accepts share-alike for it (his call; a lawyer's question
  if it is ever sold).
- **`--micro-tokens`** (round 6) keeps the gradient of a full batch in chunks that fit the card. The talk trainer can
  use it to bring the talk adapters back to batch 32 (0010's question); why batch 32 stopped fitting is still open.
- **Proposed (Nick's call): say the rule back.** Five rounds show the reader, on a kind it has not seen, writes the
  nearest familiar rule with confidence. Whoever set the rule can catch that: a system prompt's compiled rule book is
  shown once, in plain words, to the person who set it, before it applies ("Rule 3: every reply starts with P.S."), and
  is fixed there; an instruction in the conversation is said back in Cubby's voice ("Got it: under 50 words."). A rule
  shown and not corrected is applied; a rule corrected is recompiled. Recorded as a step of the turn if Nick agrees.

## Cowork's addendum (2026-11-13): say the rule back (Nick: "yes")
Why: five rounds show the reader, on a kind of instruction it has not seen, writes the nearest familiar rule with
confidence. The person who set the rule can catch that if Cubby says what it understood. Part of the turn (step 4)
and of the split of a system prompt (step 5); behind `CB_RULES`, as they are.
- **Rendered, not written:** each rule is said in words by templates from its plan (one per relation and comparison,
  English and French), never by a model, so what is said is exactly what will be checked. A rule kept as "asked" is
  said as such ("I'll keep it formal; I can't check that one").
- **A system prompt:** compiling it returns the rule book in plain words to whoever set it (the operator, a company's
  admin, a user for their session), each rule beside the sentence it came from, and where every other sentence went
  (a stated fact, the voice, "asked", not an instruction). Through the API, the rule book is in the compile's reply.
  The setter may correct a rule; a correction recompiles that rule. A rule shown and not corrected is applied. A
  company may ask for "confirm before apply", which holds the new rules until confirmed.
- **In the conversation:** when an instruction sets, changes or withdraws a rule, Cubby says it back once, in one short
  line in its voice ("Got it: under 50 words, all lowercase."), not again at every turn. "No, 50 words in all" from
  the user recompiles the rule at the next turn.
- **The record** keeps the said-back text, who saw it, and whether it was corrected; the board shows it.
- **Lines:**
  - every rule applied was shown to whoever set it: 100% (**kill** below);
  - a correction replaces the rule from the next turn on (tests, both levels);
  - the said-back line is template text only (tests: no model call on that path);
  - **reported:** on the reader's wrong rules from the held-out reads (round 4's IFEval kinds, IFBench), the share whose
    said-back sentence shows the error to a reader at a glance, judged by hand on a sample of 100 and written down
    before the next round's results are seen.

### Say the rule back, built and measured (Code A, 2026-11-13, while the view reader's Colab run waits)
- **Rendered, not written** (`cubbyllm/speech/rule_say.py`): every relation and comparison of the reply world said by
  templates, English and French ("under 50 words", "all lowercase", "without the word "river"", "the word "paradox" at
  least once"); "and" as ", ", "or", "not (...)". All 2,855 plans the readers wrote in the logs are said in both
  languages without a gap. Cubby's lines: "Got it: ..." / "Got it, from now on: ..." / "Got it: no more ..." / an asked
  rule "I'll keep "be formal" in mind; I can't check that one." (FR, with tu: "C'est noté : ..."). The module imports
  nothing but the standard library: no seat, no model on this path.
- **The rule book** (`rule_book.py`): `show` says rules back to whoever set them (Cubby's line) and `show_book` gives a
  system prompt's book to its setter (each rule beside its sentence; every other sentence with where it went); a rule
  applies only once shown to its setter (`in_force` checks it), so no unseen rule is ever applied; `correct` says the
  new reading back now and replaces the old one from the next turn (linked both ways); a level in `confirm` holds its
  new rules until `confirm`. The chat (behind `CB_RULES`) says a rule set in the conversation back once, in the user's
  language, in the turn's record (`said_back`); a system rule never shown to its setter does not apply.
- **Lines** (`validation/exp_rule_say.py`, `rule_say_2026-11-13.json`, tests `tests/speech/test_rule_say.py`):
  - every rule applied was shown to whoever set it: **21,954 of 21,954** rule-turns in 1,000 simulated sessions (set at
    every level, shown to the setter or to someone else or not yet, corrected, confirmed, withdrawn): **100%**, kill not hit;
  - a correction replaces the rule from the next turn, at both levels (tests: a system prompt's book and the chat);
  - the said-back line is template text only (test: the module imports nothing that could reach a model);
  - **reported, judged on a sample of 100** of the reader's wrong rules (round 4 on IFEval's held-out kinds: 215 wrong;
    round 6 on IFBench: 170 wrong; 50 drawn from each by seed; `rule_say_sample_2026-11-13.jsonl`): the said-back line
    shows the error to its setter at a glance in **94 of 100** (IFEval 49 of 50, IFBench 45 of 50). Not shown: three
    "unique words" read as "words", a word range said back right while counted otherwise, a paragraph's first word
    said as "at least once", a three-sentence rule whose other parts were dropped. Judged by Claude Code reading each
    pair, written down before the view reader's results; a human re-judgment is Cowork's or Nick's call.
- The existing callers show their rules first (`exp_if_turn.py`: the host's rules shown to "the prompt" that set
  them; the step 4 tests through a `put` helper).

## Nick's call (2026-11-13): the VM's comparisons are fixed in cubelang
Nick: "yes", to comparing a missing value being an error, and to the fix list. Cowork wrote the seven items as
cubelang's **P6** (`CUBELANG_FIXES.md`, cubelang `805feda`, docs only, not pushed): P6-1 is (a) above, comparing null
or a string with a number fails with `TypeMismatch`, an index past the end fails with `IndexOutOfRange`, an undeclared
name is a compile error; P6-2 is (b); P6-3 to P6-7 are (c) to (g). Each probe program that fails today becomes one of
its tests. Who builds it, and when, is Nick's next call.
- **Until P6-1 runs on the build `CUBELANG_EXE` names:** no generated program runs unless every value it compares is
  bound from the reply world's views by the host, and nothing indexes a string in the VM (the rule above stands). The
  second reader's kill line is read only on a build with P6-1.

## Cowork's check (2026-11-13): say the rule back, accepted (`74292c1`)
- Line 1 met: 21,954 of 21,954 applied rule-turns shown to their setter first (1,000 simulated sessions). A correction
  applies from the next turn; no model on the path; "confirm before apply" holds new rules.
- Reported: 94 of 100 wrong readings visible at a glance in the said-back line (`rule_say_sample_2026-11-13.jsonl`),
  judged by Code A. Cowork re-read 6 drawn at random (seed 7): all 6 judgments hold (e.g. "the word \"words\" at least
  8 times" for "words in all capitals at least 8 times"; "under 2 words" for alternating syllables).
- Open: the API that shows a system prompt's rule book to its setter (step 1; `RuleBook.show_book` is ready).

### Step 1, built: roles in the API (Code A, 2026-11-13, while the view reader's Colab run waits)
Behind `CB_API_ROLES=1` (off: `POST /turn` reads `{"text"}` as before, `/rules/` is unknown; a test pins it).
- **`POST /turn` takes messages** with a role (`system`, `user`, `assistant`, `tool`), the shape other clients send, and
  a session (`cubby_serve/rule_api.py`). A `system` message carries its level: `global` (the permission `rules:global`,
  the operator), `company` (cubby-auth's `tenant_role` "admin" of the caller's company), `user` (any signed-in caller,
  for their own session; the default). The role is checked before anything is set; with no cubby-auth sign-in nobody
  can set a prompt (closed, never open). The last `user` message is the turn's text; `assistant` and `tool` set nothing.
- **Compiled once, shown to its setter:** a prompt is split into sentences, each read once by the plan reader into a
  rule of its level, or kept "asked" / "none" (`cubbyllm/speech/rule_stack.py`); the same text from the same setter is
  not read again; a new prompt at the same place withdraws the old one's rules from that turn. The reply carries the
  rule book in plain words (`RuleBook.show_book`) to whoever set it; a company may ask `"confirm_before_apply": true`
  (`POST /rules/confirm`, its admin). `POST /rules/correct` recompiles one rule, by its setter only, said back ("Got
  it, from now on: ...") and applied from the next turn.
- **Sessions:** made by the server on the first call (an unguessable id) and returned; a signed-in creator's session is
  theirs (another caller 403, an id the server did not make 404). A turn's rules are a stack: global, the session's
  company, the session's own prompt, the conversation; highest first. A turn without a session reads the global rules.
- **What a user sees:** a turn's rule rows above the user's own level show how each acted and its outcome only, not
  the operator's or the company's sentence nor who set it; the audit view shows all.
- **Tests** (`tests/serve/test_rule_api.py`, a running server, a fake reader and a fake sign-in, 8): off = as before;
  who may set which level; compiled once and shown to its setter; a session is its creator's; each level reaches only
  its own sessions and a user sees no higher sentence; a new prompt replaces the old and a correction applies from the
  next turn, by its setter only; "confirm before apply"; assistant and tool messages set nothing. Full suite: 1,644
  passed, 12 skipped, the known red only (order 0002); on the cowork VM build (`676df1d`).
- **Not done, said so:** nothing in the server's boot mounts the plan reader yet (`brain.rule_reader`; its adapter waits
  for the view reader's read): with none, every sentence of a prompt is kept "asked", and the rule book says so. The
  chat's history was still shared by every session (done since: below). Stated facts and the voice are not split out of a prompt yet (step 5). Rules from the user's own turns
  (the conversation level) are step 4's "compile what is new"; the API does not read a user line into a rule.

### Step 1, finished: no conversation shared across sessions (Code A, 2026-11-13; Cowork: "a leak between people")
- Under `CB_API_ROLES=1` each server-made session has its own conversation: the chat's history (the last exchange in
  the seat's context, the replies a candidate is ranked against), the words its user has used (the appraisal's
  novelty), its last choice and last rejection are swapped into the one chat for the session's turn and out after,
  under one lock (`cubby_serve/rule_api.py`). A turn without a session uses the chat's own conversation, as before.
- **One clock for the rules:** a session's history counts its own turns, so every book now counts the server's turns
  (`RuleStack(..., now)` reads the server's clock instead of the chat's turn); a global rule set at the server's turn
  t applies in every session from there.
- **Tests** (2 more, 10 in `tests/serve/test_rule_api.py`): two users, two sessions, turns interleaved through the real
  `CubbyChat` with a seat that records every prompt: each turn's context carries its own session's last line and
  nothing of the other's, in the context or the record; each session's history holds its own lines only; the chat's
  own conversation is untouched. With the swap disabled the test fails (checked once). Rules: two sessions' own
  prompts, interleaved over three turns, each turn reads only its own. Off: unchanged (the first test). Full suite:
  1,646 passed, 12 skipped, the known red only.
- **Still one Cubby, said so:** the hormones (the neurochemistry's state and its emotion) stay the one Cubby's across
  sessions, as Cowork's note leaves them; and what a user teaches the worlds (the memory cortex) goes through the
  worlds' gate into the shared stores, as before: a stated fact, not a conversation. Either is a call for Nick if it
  should be per session too.
