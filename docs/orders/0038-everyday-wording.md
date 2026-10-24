# 0038 — Everyday wording: the host reads "Who wrote X?", and a scoreboard of the questions people ask
status: review (built by Cowork while Claude Code was off; Claude Code reviews, lands and commits it)
written: 2026-10-24 by Cowork · approved: 2026-10-24 by Nick ("yes", then "can you do it"; the live lookup: "Live Wikidata lookup") · hypothesis: none

## Why
Nick asked how far Cubby is from a useful model. On the real stack's 20 questions it answered 2 (order 0036's run).
15 of the 18 misses were question-understanding: the model in the programs seat did not turn the everyday wording
into a plan. The host already reads its own grammar ("What is the capital of France?"); it did not read "Who wrote
Pride and Prejudice?", "What currency does Greece use?" or "Où est né Victor Hugo ?".

**What it brings, in plain words:** Cubby answers the common questions people actually ask, in English and French,
without waiting for the model to learn them. The rule stays that nothing wrong is said. The scoreboard is 187
WebQuestions questions (North America and Europe) plus the stack's 20, with every spoken answer read by hand.
- **Over a conventional model:** a model answers all 187, and some answers are confidently wrong (an old currency, a
  namesake's birthplace). Here an answer comes only from a fact that was looked up, dated and checked, and the
  scoreboard shows each one.

## Build (done, in the worktree; see the inbox note of 2026-10-24)
1. `cubbyllm/reasoning/everyday.py` (new): `EverydayForms`, a table of everyday wordings → relation (EN + FR, the
   French articles expanded), merged with the hdc relation templates (the hand forms win over a duplicate wording).
   `restate()` turns a question it reads into the host's grammar, "What is the {relation} of {entity}{ year}?", and
   strips end-of-question noise ("on a map", "right now", "wikipedia"). Values that change and need a date ("who is
   the mayor", "what team") are in `NOT_READ` and go to the model as before.
   - Why restate: the source's alias table maps "wrote" to `notable work`. Binding the question's own wording
     lost the fact; the restated plan walks `author`.
2. `cubbyllm/reasoning/ask.py`: `AskLoop(everyday=None)`, on with `CB_EVERYDAY=1`, **off by default**. A question
   the table reads goes through the scene's path (no emitter call); everything else is unchanged. The record
   carries `restated`.
3. Three fixes the live lookup needed (each with a test that fails without it):
   - `pipeline._latest_dated`: a value whose end the source stated never beats one still standing, and when every
     value ended, nothing is picked (a refusal naming them). Live: France → Bayeux (June to September 1944), on its
     later start date alone.
   - `learn.learn_and_answer`: a fetched fact that only a read-only world behind already holds is also put in the
     first world, so the walk sees it beside its siblings. Live: Italy → Florence. Rome was a duplicate in the hdc
     world, which reads no hop on the shared name "Italy" (0036). The same happened with the Czech koruna.
   - `wikidata_resolve.resolve`: the label beats an alias only among different kinds of thing (the James Young
     rule). Live: "Ben Franklin" → Carthage, from a minor item *labelled* Ben Franklin, over Benjamin Franklin, who
     only has it as an alias. `CB_NAMESAKE_TIER=1` keeps the old reach.
4. `validation/exp_everyday_webq.py` (new, STANDALONE): the scoreboard. Arms: `--everyday on|off`,
   `--emitter none|450m`, `--online` (the Wikidata source live, as `--ask wikidata` mounts it), `--tense on|off`
   (order 0012's check), `--stack`, `--parsed-only`.

## Acceptance (pre-registered)
- Tests: `tests/reasoning/test_everyday.py` (9), and new tests in `test_ask.py`, `test_hdc_namesakes.py` and
  `test_sources.py`. The reasoning, worlds and sources suites pass on 571b035 with this work (exit 0).
- **No wrong answer** on the scoreboard with the stack's settings and the tense check on, read by hand.

## Report
- **Files:** the table (`everyday.py`); its tests; the loop flag (`ask.py`); three fixes (`pipeline.py`, `learn.py`,
  `wikidata_resolve.py`) and their tests; the scoreboard; logs in `validation/logs/everyday_*_2026-10-24.json`.
- **Measured** (no model in the programs seat, so the table alone; strict namesakes):

  | arm | stack 20 | WebQ 187 the table reads | wrong, by hand |
  |---|---|---|---|
  | table off | 0 | 0 | 0 |
  | table on, offline cache, before the fixes | 0 (4 with the tier) | 5 | 0 |
  | table on, live lookup, before the fixes | 5 + 1 wrong (Bayeux) | 42 spoken | 6 (Florence, Valladolid for Spain, Carthage for Ben Franklin, the drachma, the Czechoslovak koruna twice) |
  | table on, live, fixes, tense check on | **9** | **47** | **0** |
  | table on, live, fixes, tense check off | 9 | 54 | 6 (old currencies, below) |
  | table on, live, fixes, tense on, `CB_NAMESAKE_TIER=1` | — | 53 | 2 (Ben Franklin, "As of 2019, Bonn") |

  - The stack's 9: Paris, London (Ada Lovelace), Jane Austen, Ottawa, Leonardo da Vinci, Alexander Fleming, yen,
    Nile, 1945-09-02. The stack recorded 2 of 20 with the 450M (0036).
  - The 9 WebQ "mismatches" with the tense check on are right, read by hand:
    - two name the hospital where Truman and Nixon died;
    - two give the time zone's database name (America/New_York, America/Nassau);
    - three say the key's answer in other words (parliamentary monarchy, North America, pound sterling);
    - Italica for Hadrian is Wikidata's and most historians' answer;
    - "Canada" for Nina Dobrev is one of her two citizenships (true, not the whole answer).
  - With the tense check off, six answers are old currencies: peseta (twice), Deutsche Mark (three times), Italian
    lira. The euro's Wikidata item has no English label right now (only German "Euro"; checked live on 2026-10-24,
    the item last edited 2026-10-05). So the source serves only the ended currency, and with one candidate the walk
    says it. The tense check refuses those (`time_mismatch`, 7).
  - The cache filled by the live runs answers the same as live (offline, tense on: 47, 0 wrong).
- **Not verified here:** the 450M arm with the table on (about 6 s a question; the old baseline run was stopped, as it
  would also have filled the cache).
- **Found on the way:**
  1. **The stack as recorded runs with the tense check off** (`--books` absent). With the live source, an ended value
     is spoken in the present. Turning `--tense` on, or on whenever the source is live, is Nick's call.
  2. `ask_profile` orders a relation's dated values by start date only: the same Bayeux error, in "tell me about
     France". Not changed here.
  3. Several values still standing are picked by the latest start (Switzerland → Romansh, Norway → Sámi): true, but
     not the whole answer. Answering with all of them belongs with 0037's "answer each".
  4. The as-of snapshot says "As of 2019, X" for terms that ended; currency (P38) is not in `TIME_SENSITIVE`.
  5. The online cache never refreshes a stored response (the euro's empty label stays cached), and the live runs
     added about 1,000 files to it. Offline recordings made after today read them.
  6. "capital city of germany now" → "As of 2019, Bonn" under the tier: a test case for 0037.
  7. City qualifiers ("cleveland ohio", "denver co") could use the existing "Paris, Texas" qualifier reading.
  8. The worlds on the D: drive (registry 0015/0022) are not mounted by the stack.
- **Questions:** keep `CB_EVERYDAY` off by default until the 450M arm is measured, or turn it on now (the table
  only takes questions the model was missing)?

## Landing (Claude Code, 2026-10-24)
- **Reviewed against 0037:** all three fixes kept as built.
  - The resolve change refuses two of a kind at the source, which is 0037's default (asked which).
  - The learn fix admits a duplicate once: `held_by()` is None once the first world holds it, so no turn repeats.
  - Neither touches 0037's hop rule. With `CB_NAMESAKE_RELATION=1` the hdc world reads Q38 for Italy's capital
    itself.
- **Commits:** the fixes and their tests in one, the table, scoreboard, logs and this record in the next, both on
  top of 0037's rule (`05c87c6`).
- **Tests:** whole suite 1095 passed. The one failure is the known red, `test_every_module_declares_wiring` (order
  0002).
- **Correction from Cowork's later note:** the euro's missing English label is Wikidata's `mul` label (the same name
  in every language), not damage on the item. The fix (ask again with `mul` for ids that come back without `en`) is
  the next item in the inbox.
- **The question** (`CB_EVERYDAY` on now, or after the 450M arm) is Nick's; asked in chat.
- **Nick's answers (2026-10-24, in Claude Code's chat):**
  - push: yes;
  - `CB_EVERYDAY` on by default now (`CB_EVERYDAY=0` turns it off);
  - the tense check on always, books or not (`--no-tense` turns it off);
  - the time rule follows it, as Cowork measured.
- **Turning them on surfaced three things, fixed in the same commit:**
  1. The table took nested questions ("the capital of the country of citizenship of Marie") as one hop. A seed that
     starts with a relation the table names is now left to the model's plan.
  2. The tense check refused the host's own grammar for an event: "What is the date of birth of X?" read "is" as
     the present. Now an event named as a noun asks no tense (`question_forms.noun_tense`). "Where is X born?" is
     still refused, and a state such as currency is still checked.
  3. 0037's answers now pass the same checks. The checks moved into `AskMemory._checked`; each "several fit" walk
     passes them, and the namesakes' line comes after the "as of".
- **Re-measured** with the new code, offline on the cache the live runs filled:
  - WebQ: 47 spoken, 38 matched, 9 mismatched, the same 9 rows read as right
    (`validation/logs/everyday_webq_on_offline_tense_newdefaults_2026-10-24.json`);
  - the stack's 20: 9 spoken (`validation/logs/everyday_stack_on_offline_tense_newdefaults_2026-10-24.json`);
  - 0 rows changed against Cowork's runs.

## After landing: Cowork's sorting of the 76 refusals (Claude Code, 2026-10-24)
Items 1, 2, 4 and 5 of the inbox note; item 3 belongs elsewhere (see the inbox). Live, the table on, tense on, every
changed row read by hand:

| step | WebQ spoken | wrong | stack's 20 | log |
|---|---|---|---|---|
| landed (Cowork's best arm) | 47 | 0 | 9 | `everyday_webq_on_live_tense_2026-10-24.json` |
| 1. `mul` labels (`09806a4`) | 54 | 0 | 10 | `everyday_webq_on_live_tense_mul_2026-10-24.json` |
| 2. a bare year, noise words, the place qualifier* (`a4bff41`) | 63 | 0 | – | `everyday_webq_on_live_tense_reading_2026-10-24.json` |
| 4.-5. name matching*, a retry, the siblings' seed (this commit) | 65 | 0 | 11 | `everyday_webq_on_live_tense_names2_2026-10-24.json`, `everyday_stack_on_live_tense_names2_2026-10-24.json` |

\* behind `CB_PLACE_QUALIFIER=1` and `CB_NAME_MATCH=1`, both off by default. The rows above were run with them on.
- "where was stephen g. breyer born?" was not punctuation. It failed in the run and passed alone. After 79 questions the
  store held `birthplace`, and the siblings step rebuilt the plan's tail from the normalized seed ("stephen g breyer").
  Fixed in `learn.resolve_siblings`.
- One call to Wikidata failing once dropped that item's facts. A failed call is now tried once more.
- **Nick (2026-10-24): both on by default.** `CB_PLACE_QUALIFIER=0` and `CB_NAME_MATCH=0` turn them off.
