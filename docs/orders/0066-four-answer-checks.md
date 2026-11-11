# 0066 - Four checks before an answer is said (from the read of 0054's "after")
status: review
written: 2026-11-11 by Cowork - approved: 2026-11-11 by Nick ("yes to all 4") - hypothesis: the system paper's wrong-answer line (C at most 1% wrong among spoken)

## Why
0054's "after" has 8 wrong answers among 86 spoken on SimpleQA, NQ and WebQuestions (0 of 6, 4 of 15, 4 of 65;
`validation/logs/system_eval_after_2026-11-10.json`, and both reads in 0054). Six are Cubby's own:
- the plan answered another relation (3): Dirty Dancing's remake, Germany's culture region, Auschwitz;
- a work named like the asked one was taken for it (2): the Gospel according to John, Jana Gana Mana;
- an alias ended the name search (1): "Emperor Constantine" resolved to Constantine II alone.

Two are the source's value against the gold; no check here claims them. A ninth answer was right as a count and
wrong as a date: a 2020 census count said "As of 2025".

Each check below stops one of those causes. None picks an answer: each lets the walk's answer through, hands the
choice back to the rules that exist, asks, or refuses.

**What it brings, in plain words:**
- **Check 1:** Cubby stops answering "where" with a film, or "who" with a place.
- **Check 2:** asked who wrote a famous work, Cubby answers about the work, not about a book on it.
- **Check 3:** "Emperor Constantine" is asked back ("which Constantine?") instead of taken for one of them.
- **Check 4:** a number is said with the year it was counted, not the year of Cubby's copy of the records.
- **Over a conventional model:** a model has no step where an answer's kind, the item it is about, or a value's
  date is checked against a record; it says the most likely string. Each check here is a rule that passes or
  refuses, and its verdict is in the certificate.
- **For you:** an answer of the wrong kind becomes "I can't answer that from my records", not a confident mistake.

## Build
Four checks, each behind its own flag, off by default. Off, every answer is what it is today.

### 1. The answer is of the kind the question asks (`CB_QWORD_KIND=1`)
An extension of `ask_class` (order 0047): the same check, the same `class_mismatch` refusal, two more ways to
read the kind asked.
- **1a. From the question word,** when the question names no class: "where" wants a place; "who" or "whom" a
  person, a group of people or an organisation; "when" a date. French: "où", "qui", "quand". Only the word that
  opens the question (after an optional preposition).
- **1b. From the head noun,** when the question names a class the source has no class for: "what culture region"
  is read as "region". Today such a class word is left unchecked.
- **The kind is read from the store, never from the label's words:**
  - a date: the value's own type (a time value);
  - a place: the item has coordinates, or its class is under "geographic location" by `subclass of`. A hospital,
    a park, a neighbourhood and a village are places by this rule;
  - a person, a group or an organisation: the item's class is "human", or under "group of humans",
    "organization" or "fictional character" by `subclass of`.
- **The classes under each root** are built once from the store into a small table, stamped with the store's
  snapshot, as `wd_store_cues` builds its own for "fictional entity". No id is taken from this order: each is
  read from the store and listed in the report.
- **What it does not do:**
  - it never narrows or picks among values (that stays `answer_values`);
  - a question that names a class the source knows ("which city", "what Nobel Prize") is checked as today;
  - a value whose kind no world holds (a plain string, an item with no class) is not checked, and is counted
    apart as "kind unknown".

### 2. The original work before works about it (`CB_ORIGINAL_FIRST=1`)
- **First, the record:** how each of the two names resolved in the "after" (which tier, which items, which rule
  picked). The rule below is written for either way.
- **The rule:** when a name's candidates, label tier and alias tier read together for this check, hold a work
  and other works that point at it, the one pointed at is the original, and the name stands for it. "Points at"
  is a statement on the later work: based on, edition or translation of, main subject, or the original's own
  "derivative work". The properties are read from the store and listed in the report.
- **Else,** two or more works share the name and no statement orders them: Cubby asks which one, naming each by
  its description (kind, author, year), as a shared name is asked about today.
- **Only among works** (a class under "work" by `subclass of`). A person and a work sharing a name stay with
  the existing rules.
- **Never** by fame, and never by age alone: the older work is not the original without a statement.

### 3. An alias match does not end the name search (`CB_ALIAS_OPEN=1`)
- **The case:** "Emperor Constantine" is an English alias of Constantine II alone in the store. Several emperors
  are named Constantine.
- **The rule:** a name that matches no label, matches exactly one item by alias, and opens with a role word is
  not resolved yet.
  - **A role word** is the label of an item the store classes as a position, a title or a rank ("emperor",
    "king", "pope", "president"). The list is built from the store, capped and reported, not written by hand.
  - **The bare name** ("Constantine") is read too. Its items are narrowed to those that hold or held the role
    (position held, noble title, occupation; or a role under it by `subclass of`). The alias's item joins them.
  - **Then the existing rules decide** (`wd_store_refine`, the kinds table, the question's cue words, a year the
    question names). One item left: the name stands for it. Several: Cubby asks. None but the alias's item: the
    alias stands, as today.
- **What it does not do:** it does not reopen every alias. "Big Apple" stays New York City: no role word.
- Claude Code may propose a simpler rule that passes the same fixtures; it goes in the report before it is
  built.

### 4. A value with its own date is said with that date (`CB_VALUE_DATE=1`)
- **The case:** "As of 2025, 208501." for Fayetteville's population: the 2020 census count, dated with the dump's
  year (`ask_memory._snapshot`).
- **The rule:** when the statement the answer rests on carries its own date (a point in time; or a start, for a
  value that holds from then), the reply says that date: "As of 2020, 208501." The dump's date is said only for
  a time-sensitive value whose statement carries none.
- **A question that names a year** is compared with the statement's date, not the dump's. The time check (0012,
  0014) is otherwise unchanged.
- **The certificate** records the statement's date and the dump's date apart.
- **If the hot store does not keep the qualifier:** that is checked first. If it does not, the report gives
  what keeping it would cost (size, build time), and this check stops there. Nothing is fetched live to fill it.

## Acceptance (pre-registered)
- **Tests** (fixtures, no network; each flag on alone, and all four on):
  - **check 1:** a film and a book are refused for "where"; a hospital, a park, a neighbourhood, a village and a
    city pass; a place is refused for "who"; a person, a band and a company pass; a place is refused for "when";
    "what culture region" refuses a company and passes a region; "which city" and "what Nobel Prize" are
    untouched;
  - **check 2:** a commentary and its Gospel under one name: the Gospel; an anthem and its arrangement: the
    anthem; two works with no statement between them: asked, both named;
  - **check 3:** "Emperor Constantine" with one alias item and several emperors of that name: asked; with a year
    only one fits: that one; an alias with no role word: resolved as today;
  - **check 4:** a count with a 2020 point in time and a 2025 dump: "As of 2020"; a value with no date of its
    own: the dump's year, as today; a question naming 2023 against a 2020 value: the time check's refusal;
  - **all flags off:** the "after"'s records replay identically (every question of C: the same verdict and the
    same reply).
- **The rerun:** 0054's arm C as the "after" ran it (the same snapshot, programs and three flags), with the four
  new flags on. Each refusal carries its check's reason, so the four are reported apart from one run.
- **Lines:**
  1. No answer read right in the "after" becomes wrong. A check refuses, asks or redates; "the original first"
     and "one item left" are the two places an answer can change, and each changed answer is read by hand.
  2. Of Cubby's own 6 wrong answers, at least 4 are no longer said wrong (refused, asked or right): Dirty
     Dancing, the Gospel, Jana Gana Mana, Constantine. Germany and Auschwitz are reported whichever way they
     fall, with what the store holds for the item said.
  3. Right answers lost (read right in the "after", refused or asked now): at most 2, over the 64 matched by the
     normalizer and those read right. Each is listed.
  4. 0055's picking audit (`validation/audit_wd_kinds.py`), rerun with the four flags on: its line holds (at
     most 1 wrong pick in 50), and each pick that changed is read.
- **Kill, per check:** a check that loses more right answers than the wrong ones it stops stays off, and the
  report says so. A check that turns a right answer into a wrong one is off until fixed.
- **Reported:** wrong among spoken per benchmark, before and after, with the 95% interval; each check's
  refusals, asks and changed answers; the "kind unknown" count; the time per question, before and after.
- **Not claimed here:**
  - **the 1% line.** The two source-against-gold answers remain, and these benchmarks speak too few answers for
    it (0054's addendum of 2026-11-11: 0 wrong needs 381 spoken answers);
  - **a result on these 8 answers.** The checks were written from them, so stopping them is a fixture. The
    result is the next benchmark the checks have not seen: 0054's gold-SPARQL sets.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No neural model in any check. Nothing is ranked by fame, popularity, links or age.
- A kind, a role and a date are read from the store's statements. Class and property ids are verified against
  the store and listed in the report.
- `cubbyllm/` opens no connection: a check reads through the mounted world.
- No local path in a committed file.

## Out of scope
- **The proposer** (no plan for 3,759 of 4,326 SimpleQA questions): its own order, once 0054's C-oracle arm says
  how much of the gap is its.
- **The two source-against-gold answers:** 0054's audit of the golds.
- **How a number is written** (208501 against 208,501).
- **The wording of an ask or a refusal:** the talk seat's, order 0010.

## Report
- **Commits:** `5b8345b` (first build from the four bullets, before the order), `3e0311c` + `25ea6b0` (to the order's
  spec; the first carried only a rename), `cc82f96` (a regression of mine fixed, below), `7f5541d` (the roots as
  Cowork's allow-list), `ccf8e68`, `cd674ab` (check 1 reads a store fact's item and classes in the store), `6b19487`
  (arm C's record keeps what the checks said). cubelang-cowork `a351be8` (local): `wdstore subclasses`.
- **Files:** `cubbyllm/reasoning/ask_qword.py` (check 1a; `store_item`, `store_is_a`), `ask_class.py` (1a's hook, 1b,
  the store-linked read), `ask_memory.py` (check 4's year and `rec["dates"]`), `ask_namesakes.py` (check 3's year);
  `cubbyllm/worlds/wd_store_roots.py` (the class-roots table), `wd_store_names.py` (checks 2, 3, 4's records, 1's
  kinds), `wd_store_world.py`, `hdc_store.py` (a hop reads the rows of an item a name gathered under another label);
  `cubby_serve/wd_world.py` (mounts the table, stops without it); `validation/builders/build_wd_roots.py`;
  `validation/system_eval/run_c.py`; `tests/worlds/test_wd_store_names.py`.
- **The class-roots table:** from the store's 5,157,087 best-rank subclass-of edges (`wdstore subclasses`, 580 s), on
  Cowork's ids, each checked to be a class carrying its label (build stops otherwise): place Q27096213 (93,368
  classes), person Q5 (2,116), group Q16334295 (110,730), organisation Q43229 (105,037), character Q95074, Q21070598
  (7,263), work Q15621286, Q732577 (229,200), role Q4164871, Q216353, Q355567, Q56019, Q16138157 (107,018; with
  278,166 items that are instances of a role class). Properties read: based on P144, edition or translation of P629,
  main subject P921, derivative work P4969 (check 2); position held P39, noble title P97, occupation P106, start P580,
  end P582, birth P569, death P570 (check 3); point in time P585, start P580 (check 4); coordinates P625 (check 1).
- **Tests:** `tests/worlds/test_wd_store_names.py`, 9: the order's fixtures for each check on alone, all four on,
  each off as before; an unknown value given as a plain string; a store-linked value read in the store's dump when
  live Wikidata disagrees. tests/worlds, tests/reasoning, system_eval, package rules, local paths: pass; the known red
  `test_every_module_declares_wiring` (step_loop, spatial; order 0002) only.
- **Measured** (`validation/logs/answer_checks_0066_2026-11-11.json`: the counts and every changed answer, read by
  hand; runs in `run_0066_final`, flags-off in `run_0066_off2`; audit `validation/logs/audit_wd_kinds_2026-11-11-0066.{json,md}`):
  - **All flags off: the "after" replays identically** on SimpleQA, WebQuestions and AA (every question: the same
    verdict and reply); on NQ every answer is the same, and 6 of Cubby's own tally replies ("this loop -- asked ...")
    differ by one count, because the Zimbabwe prompt now has a program (the second round) and its refusal is
    counted under another reason.
  - **Wrong among spoken** (the read; 0054's for the "after", with Cowork's Obama Sr. change):

    | | after: spoken, wrong (+debatable) | checks on: spoken, wrong (+debatable) | 95%, checks on |
    |---|---|---|---|
    | SimpleQA | 6, 0 | 6, 0 | 0-39% |
    | NQ | 15, 4 | 13, 2 | 4.3-42.2% |
    | WebQuestions | 65, 4 (+2) | 64, 2 (+3) | 0.9-10.7% |
    | AA-Omniscience | 0 | 0 | - |

  - **Every answer that changed** (7), by check:
    - check 1a: "where was the remake of Dirty Dancing filmed" -> refused (a film is no place);
    - check 1b: "what culture region is Germany a part of" -> refused ("region"; The Beers of Europe is a book);
    - check 2: "who wrote the Gospel according to John" -> refused (the name now stands for the Gospel; its author
      hop holds several values); "where was Elvis Costello born" -> London as before, and the closing line now names a
      written work instead of a painting and a lithograph (question 1 below); "when does the day of the dead end" ->
      refused before and after (another reason: the VM's type check);
    - check 3: "where did Emperor Constantine die" -> each Constantine who held the role, labelled: "Constantine II
      (father: Constantine the Great): Aquileia; Constantine (father: Leo V the Armenian): I do not know; Constantine
      (father: Theophilos): I do not know." Read **debatable**: each statement is true of its item, and Constantine
      the Great (the gold) is not among them (his label is not "Constantine");
    - check 4: Fayetteville: "As of 2025, 208501." -> "As of 2020, 208501." (`dates`: value 2020, dump 2025).
  - **Line 1 (no answer read right becomes wrong): met.** Two right answers changed (Fayetteville, redated; Elvis
    Costello, the closing line only); neither is wrong.
  - **Line 2 (of Cubby's own 6 wrong, at least 4 no longer said wrong): 3 certain, 4 if Cowork reads Constantine as
    asked rather than wrong.** Refused: Dirty Dancing, Germany, the Gospel. Constantine: labelled per item
    (debatable). **Still said:** Jana Gana Mana: the store now stands for the anthem, which holds no author, so the
    walk falls through to live Wikidata, whose own namesake rule gives the score's author (Herbert Murrill); 0066's
    checks act on the store. Auschwitz: "Gas Chambers" is said; check 1 found no item the store linked the value to
    (the store gives nothing for that hop now; the walk reached it on another path), so its kind is unknown and it
    was not checked; the store holds Auschwitz (Q7341) with "has part" items, the gas chambers among them.
  - **Line 3 (right answers lost, at most 2): 0.** (The first run on the fixed table lost Beatrice, Nebraska -- found
    and fixed: `ccf8e68`, `cd674ab`; the final run keeps it.)
  - **Line 4 (the picking audit, at most 1 wrong pick in 50): met, 0.** With the flags on, the set of shared names
    changes, so the audit drew 30 other rows than the 2026-11-10 run; every single pick (5) is the one item whose kind
    carries the relation (Flashforward the technique, Chen the state, Pearl Jam the band, CBGB the film, The Toxic
    Avenger the film); 7 "every item agrees"; no probe picks another item.
  - **Kill, per check:** no check loses a right answer; none turns one wrong.
  - **"Kind unknown":** 3 (SimpleQA 2 "who", both refused by the time check anyway; WebQuestions 1 "where",
    Auschwitz, spoken).
  - **Time** (the sum over each benchmark, CPU): SimpleQA 491 -> 380 s, NQ 336 -> 287, WebQuestions 366 -> 238, AA 17
    -> 17. Not a gain of the checks: the store's cache was warm from the runs before (not controlled).
- **Not verified here:** French question words on real questions (no French benchmark yet); check 3's year path and
  check 4's start-time branch on benchmark questions (fixtures only); the 1% line (not claimed, as the order says).
- **Found on the way:**
  - **A regression of mine, caught by the flags-off check** (`cc82f96`): the store's new bookkeeping read an unknown
    birth date (a plain string) as a dict, with the flags off, since `5b8345b`; 6 questions ended in `error` where
    the "after" refused them for another reason. Nothing wrong was said.
  - **Roots by label alone caught every sense of the word** (physics work, labour, a poker position, fictional human
    races); "work" Q386724 reached cities and hospitals. Cowork picked the ids; the build now checks them.
  - **Live Wikidata has moved since the dump:** "city in the United States" is no longer a subclass of "city" there.
    A store fact's classes are now read in the store.
- **Questions:**
  1. **Check 2's strength.** As built (the order's "the name stands for it"), the original stands with the items that
     are no work; unrelated works of the name leave too. For Elvis Costello the painting and the lithograph left the
     closing line because an edition points at the written work. The alternative: only the works that point at another
     leave, the original and unrelated works stay, and the existing rules decide or ask. Which?
  2. **Constantine's labelled answer:** wrong, or asked?
  3. **Jana Gana Mana:** the live source's namesake rule (order 0043) gives the score. Should check 2 reach the live
     source too (its own order)?

## Cowork's read of the report (2026-11-11)
- **The report holds against the log.** The 7 changed answers were read in `answer_checks_0066_2026-11-11.json`;
  the counts and the intervals were recomputed (NQ 2 of 13: 4.3 to 42.2%; WebQuestions 2 of 64: 0.9 to 10.7%).
- **The lines:**
  1. No right answer becomes wrong: **met.**
  2. At least 4 of Cubby's own 6 no longer said wrong: **not met, 3 of 6.** Refused: Dirty Dancing, Germany, the
     Gospel. Constantine is debatable, which is neither refused, asked nor right. Jana Gana Mana and Auschwitz
     are still said.
  3. At most 2 right answers lost: **met, 0.**
  4. The picking audit: **met, 0 wrong picks.**
- **No kill fires.** No check loses a right answer, and none turns one wrong.
- **Not a result:** NQ's 2 of 13 and WebQuestions' 2 of 64 are on the answers the checks were written from. They
  are fixtures, as the order says.
- **The three questions:**
  1. **Check 2's strength: the alternative.** Only the works that point at another step aside. The original and
     the unrelated works of the name stay, and the existing rules decide or ask. The order's sentence ("the name
     stands for it") was loose: a work is an original only against the works that point at it. For Elvis
     Costello, the painting and the lithograph stay in the closing line.
  2. **Constantine: debatable, not wrong and not asked.** Each line is true of its item. The cause is the
     order's: the bare name was read by label only, so the emperors whose label is longer (Constantine the
     Great and the numbered ones) were never gathered. **Amended:** for check 3, the bare name gathers its label
     tier and its alias tier together, then the role narrows them, then the existing rules apply (a few:
     labelled; more: asked). The gold itself ("Constantinople") is not where Constantine the Great died; the
     question fits several emperors. Noted for 0054's audit of the golds.
  3. **Jana Gana Mana: yes, here, as one rule.** When the store has made a name stand for an item, the live
     source is asked about that item, never about the name again. Behind `CB_ORIGINAL_FIRST`, and only where
     check 2 decided. That "wrote", for a song, means its lyricist is the relation reader's business, not this
     order's.
- **Auschwitz stays wrong,** as the order allowed. Its cause is the plan's relation ("has part" for "located"),
  which no check here reads. It goes to the relation reader with the proposer's order.
- **Status: open again, for three changes:** question 1's alternative, check 3's two tiers, and check 2's item
  carried to the live source. Then the flags-on run and the audit once more. The lines do not change.

## Report, second run (Claude Code, 2026-11-11): Cowork's three changes
- **Built** (`a1b55eb`): (1) check 2: only the works that point at another of the name's works step aside; the works
  pointed at stand in the label tier; unrelated works and non-works stay; (2) check 3: the bare name's label and alias
  tiers together; (3) when check 2 leaves a name one item (`decided`), the live source is asked about that item, never
  the name (`WikidataSource.resolve`; `AskLoop` hands it every mounted store's decision). Tests: 11 in
  `tests/worlds/test_wd_store_names.py`; the full suite passes but for the known red (order 0002).
- **Measured** (`validation/logs/answer_checks_0066_2026-11-11-r2.json`; runs `run_0066_off3`, `run_0066_r2`; audit
  `audit_wd_kinds_2026-11-11-0066r2`):
  - **Flags off: every answer of all four benchmarks is the "after"'s** (verdict, reply, refusal), NQ included.
  - **Wrong among spoken:** SimpleQA 0 of 6; NQ 2 of 13 (unchanged); WebQuestions **3 (+2 debatable) of 64**; AA 0.
  - **Against the first checks-on run, three answers moved:** Constantine is now refused (both tiers gather
    Constantine the Great; several emperors: asked, `name_unresolved`); Elvis Costello's closing line names the
    painting and the lithograph again (question 1 as you ruled); **the Gospel according to John is said wrong again**
    ("D. A. Carson").
  - **Line 1: met** (Fayetteville "As of 2020"; nothing right turned wrong). **Line 3: met, 0 lost.** **Line 4: met,
    0 wrong picks** (the single picks: Flashforward the technique, Chen the state, Pearl Jam the band, CBGB the film).
  - **Line 2: 3 of 6, not met.** Refused: Dirty Dancing, Germany, Constantine. Said: the Gospel, Jana Gana Mana,
    Auschwitz.
- **Why the Gospel came back, and why change 3 does not reach Jana Gana Mana: the same cause.** Under the amended
  check 2, a name keeps several items once only the pointing works leave:
  - "gospel according to john": the Gospel and three unrelated articles (the commentary and its edition step aside).
    The store asks rather than answer, and the walk goes on to the Wikidata5M world, which holds the name for the
    commentary alone (Q7737359, the same id) and answers "D. A. Carson". In the first run the store stood for the
    Gospel and the walk stopped there (`ambiguous_hop`).
  - "jana gana mana": the anthem, three films and a radio performance (the sheet music and an article step aside).
    Nothing is decided, so the live source is asked by name, and its own namesake rule picks the sheet music
    (Herbert Murrill), the item the store set aside.
- **A proposal, built and measured apart** (`CB_ORIGINAL_ASIDE=1`, off by default; `run_0066_aside`, WebQuestions;
  `answer_checks_0066_2026-11-11-aside.json`): the live source's candidates for a name drop the works the store set
  aside for it. Jana Gana Mana is refused (`fact_missing`); nothing else changes. WebQuestions 2 (+2) of 63; **line 2
  then 4 of 6.** Applied to the other worlds too (Wikidata5M uses the same ids), it would also refuse the Gospel;
  not built.
- **Questions:** (1) Keep `CB_ORIGINAL_ASIDE` as part of check 2? (2) Extend it to the read-only worlds below the
  store, so a name the store holds is never answered by a lower world with an item the store set aside? (3) Or,
  wider: when the store asks about a name, no lower world answers that name by itself (the Gospel's case, whatever
  was set aside).

## Cowork's answers to the second run's three questions (2026-11-11)
- **The report holds against the logs I read.** `answer_checks_0066_2026-11-11-r2.json` and `-aside.json`: the
  counts per benchmark, the three refusals (Dirty Dancing, Germany, Constantine) and the proposal's one change
  (Jana Gana Mana refused, `fact_missing`). I did not re-run the flags-off comparison.
- **The cause is one, as you say:** a name the store has already read is read a second time below it, by a world
  that knows less. The store's reading is the better one, so it is the one that counts.

**What it brings, in plain words:** once the store has set a namesake aside for a question, no other source
brings it back.
- **Over a conventional model:** a model has one reading of a name and no record of the ones it passed over.
- **For you:** an answer about the commentary is not given to a question about the Gospel.

- **Question 1: yes.** `CB_ORIGINAL_ASIDE` is part of check 2. An item the store set aside for a name, in this
  question, is no candidate for that name in the live source.
- **Question 2: yes, and it is the same rule, under the same flag.** The item is no candidate in any world below
  the store either. A lower world may still state facts about the items the store kept: it is asked about them
  by id.
  - **Limit:** worlds that use the store's ids (Wikidata5M, the live source). A world with its own ids is 0058's
    business, not this order's.
  - The flag stays apart for this run, so that it is measured apart. At close it folds into
    `CB_ORIGINAL_FIRST`.
- **Question 3: not built now. First a count, which costs no build.** The wider rule changes how every name with
  several items is read, far beyond works about an original, and its cost in right answers is unknown.
  - **The count,** on the checks-on run: every spoken answer where the store held the asked name, did not settle
    it, and a world below it or the live source then settled the name by itself. How many; of them right, wrong
    and debatable; the read-only worlds and the live source apart.
  - **Why apart:** a read-only world that holds one of the items settles the name by an accident of coverage.
    The live source settles it by a written rule. Under the wider rule the live source would keep its rule, over
    the items the store kept.
  - **The rule, set now:** every answer in the read-only worlds' count would become a refusal. The wider rule is
    built, under its own flag, only if that loses no more right answers than line 3 leaves (2, less what the
    other checks lose). If it would lose more, it is dropped here, and the finding goes to an order of its own:
    the store asks where a written namesake rule could decide.
- **The run, then:** the four checks and `CB_ORIGINAL_ASIDE` (live source and lower worlds), all four
  benchmarks, the flags-off comparison and the picking audit once more. The lines do not change.
  - **What I expect on the six, to be checked against the run:** five no longer said wrong (Dirty Dancing,
    Germany, Constantine, Jana Gana Mana, the Gospel); Auschwitz still said, as the order allowed. If the Gospel
    comes out as an answer and not a refusal, it is read by hand like any changed answer.
- **This is the last pass on these six.** They are fixtures, and this is the third time the checks are adjusted
  on them. Whatever line 2 reads after this run is reported as it stands, and the order closes on it. The
  evidence that the checks help is 0054's next run, on questions they were not written from.
- **Status: open, for one run.**

## Report, last run (Claude Code, 2026-11-11): question 2 built; the order closes on this
- **Built** (`576de44`): `CB_ORIGINAL_ASIDE` reaches the read-only worlds below the store that use its ids (Wikidata5M):
  an item the store set aside for a name is no candidate there; the kept items are read by id
  (`cubbyllm/worlds/below_store.py`). Question 1 kept as built. Question 3 not built (the count below).
- **Measured** (`answer_checks_0066_2026-11-11-r3.json`; runs `run_0066_off4`, `run_0066_r3`; audit
  `audit_wd_kinds_2026-11-11-0066r3`): flags off, every answer of the four benchmarks is the "after"'s.
  Wrong among spoken, all on: SimpleQA 0 of 6; NQ 2 (+1 debatable) of 13; **WebQuestions 1 (+2) of 62**; AA 0.
- **Line 2: 5 of 6, met.** Refused: Dirty Dancing, Germany, Constantine, Jana Gana Mana (the live source loses the
  sheet music), the Gospel (the commentary is no candidate in Wikidata5M; `fact_missing`). Said: Auschwitz.
- **Line 3: 1 right answer lost** (at most 2: met). **"who is the director of nightmare before christmas"**: "Henry
  Selick" in the "after" -> now each same-named item labelled (three TV episodes; "Queenie: Joelle Mae David"), the
  film not among them. Read **debatable** (each line true of its item, as Constantine was read).
  - **Why:** the film (Q322328) is "based on" the poem (Q136673038), so check 2 sets the film aside as the work that
    points; the poem has no director. Before question 2, Wikidata5M (which holds the name for the film alone) gave
    the right answer by coverage; now it may not.
  - **This is check 2's general weakness, not a fixture's:** a film, a series or an opera based on a work of the
    same name is the one asked about as often as the work. "The original first" sets every such adaptation aside.
- **Line 1:** no answer read right became wrong; one became debatable (above). **Kill, per check:** the set-aside
  (questions 1 and 2) stops 2 wrong answers (Jana Gana Mana, the Gospel) and loses 1 right one: it stays by the
  kill's count; whether a debatable answer counts as "turned wrong" is your read.
- **Line 4: met, 0 wrong picks** (the single picks unchanged: Flashforward, Chen, Pearl Jam, CBGB).
- **Question 3, the count** (`validation/logs/count_0066_q3_2026-11-11.json`): spoken answers where the store held the
  name, did not settle it (two or more items in its tier), and a world below or the live source did: **read-only
  worlds 19, all right** (currencies, capitals, time zones, languages: "what money does spain use?", the store holding
  44 items for "spain"); **live source 5, all right**. By your rule (build only if it loses no more right answers
  than line 3 leaves: 1), **the wider rule is dropped**; 19 right answers would become refusals. The finding (the store
  asks where a written namesake rule could decide) goes to its own order.
- **A proposal for check 2, for that order or a follow-up, not built:** a work that points at another steps aside only
  when the item it points at carries the relation asked (the kinds table, as the namesake rule reads it): "who
  directed" keeps the film (the poem has no director); "who wrote" sets the commentary aside (the Gospel has an
  author). Jana Gana Mana would come back ("author": the anthem holds none, the sheet music does).
- **Status: review.** This was the last pass on the six, as you set.
