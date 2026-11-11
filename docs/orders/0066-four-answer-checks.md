# 0066 - Four checks before an answer is said (from the read of 0054's "after")
status: open (changes after Cowork's read of the report: three, then one more run)
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
