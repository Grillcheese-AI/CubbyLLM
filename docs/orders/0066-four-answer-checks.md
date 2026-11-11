# 0066 - Four checks before an answer is said (from the read of 0054's "after")
status: in progress (queue: before 0054's next rerun)
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
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
