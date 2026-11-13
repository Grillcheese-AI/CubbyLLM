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
