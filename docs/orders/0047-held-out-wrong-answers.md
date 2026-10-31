# 0047 — Six wrong answers on held-out questions: the name kept as asked, the class asked, the unit
status: done
written: 2026-10-31 by Cowork · approved: 2026-10-31 by Nick ("yes") · hypothesis: none

## Why
0046's R0 ran today's serving path on 2,650 held-out dev questions. It spoke 68 answers and **6 were wrong**
(the order's report, `retrieval_R0_2026-10-28.json`, every spoken answer read by hand):
- **5 from one defect.** The everyday table's restatement drops a leading "The" from the name, and the
  source's one-exact-hit tier then resolves a different item labelled exactly the shortened name:
  - Freud's *The Interpretation of Dreams* (asked in EN and FR) became Antiphon's *Interpretation of
    Dreams*;
  - Wilder's *The Ides of March* became Manfredi's novel;
  - De Mari's *The Last Dragon* became McDermott's;
  - "the skin" (the film) became the singer Skin.
- **1 of the wrong class.** "In which city did John F. Kennedy die?" was answered "Parkland Memorial
  Hospital": his place of death, a hospital, not a city.

A seventh answer was right but misleading: "How tall is Big Ben?" gave "96" without its unit.

The scoreboard's 0 wrong held only on the rows it was tuned on. These are on the serving path now, so
this order is the one exception to "nothing on the serving path until 0046 has measured it": it only
removes wrong answers.

**What it brings, in plain words:** Cubby answers about the book you named, not another one with a
shorter title. It answers a "which city" question with a city, and says "96 metres", not "96".
- **Over a conventional model:** a model blurs near-identical titles and answers the nearest fact.
  Here the name is kept exactly as asked, and the answer must be the kind of thing the question asks
  for, by the source's own classes.

## Build
1. **The name as asked.**
   - A restatement may change the question's frame. It never changes the words of the name. That covers
     a leading article ("The", and "Le", "La", "Les", "L'" in French), case aside, and any other word of
     the span.
   - When the shortened form is all that matches, it is a candidate beside the full name. It never wins
     alone through the one-exact-hit tier.
   - **Sweep:** list every transform in `reasoning/everyday.py` and `reasoning/question_forms.py` that
     edits a name span. Report how often each fired on R0's dev rows.
2. **The class asked.**
   - When the question names the kind of its answer ("which city", "what country", "in what year",
     "quelle ville", "quel pays"), read the class word through Wikidata's class labels and aliases (EN,
     FR), as data, not a hand list.
   - The value is spoken only if it is an instance of that class: its P31, with one `subclass of` step,
     as `sources.wikidata_namesakes` reads kinds.
   - Otherwise the loop refuses with `class_mismatch`, naming what it found ("his place of death is
     Parkland Memorial Hospital, a hospital").
   - Behind `CB_CLASS_WALK=1` (off until Cowork reads it): walk up `located in` (P131) from the value to
     the first item of the class asked, as a verified chain ("Dallas, where Parkland Memorial Hospital
     is").
3. **The unit.**
   - A quantity is spoken with the unit the source states ("96 metres").
   - A quantity whose property takes a unit but whose statement gives none is refused.
   - A property without units (a count) is said as a number.
4. Items 1 and 3, and item 2's refusal, go behind `CB_HELDOUT_FIXES=1`. They are on by default once the
   acceptance below holds.

## Acceptance (pre-registered)
- **Tests** (canned claims, no network):
  - the four titles, in EN and FR, resolve to the item asked or refuse, never to the shorter-titled one;
  - "In which city did John F. Kennedy die?" refuses with `class_mismatch` (flag off), and answers
    Dallas through the walk (flag on);
  - Big Ben's height carries its unit;
  - a count property is said bare.
- **R0 again, on the same dev sample:** 0 wrong among the spoken answers, every one read by hand. Report
  how many are spoken (62 right before). R0's test split stays unread.
- **The scoreboard (187) and the stack's 20:** 0 wrong. Every changed row read. Report the spoken count
  against today's defaults: 74 on the scoreboard (2026-10-27, `CB_SOURCE_NAMESAKES` off) and 11 on the
  stack's 20.
- **0014's bench and 0033's probe:** 0 wrong, 0 mixed.
- **Report and set `done`** when all of the above hold. Set `review` only if a spoken count falls.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **No new tier and no rule for a row.** The class check reads the source's own classes, and the name
  check keeps what the asker wrote.
- Nothing here anticipates 0046's design. If R2 or R3 replace the readers, these checks stay as
  guarantees around them.

## Out of scope
- How much of a question the reader can read (0046, R3).
- Which item a shared name means (0046, R2).

## Report
<!-- Claude Code, 2026-10-31 -->
- **Commits:** `830b864` (the build, committed by Nick as "order 0046" while the tests were being finished), then the commit
  carrying this report (the relation-word rule, the flag on by default, the R0 driver's tag, the sweep and the logs).
  `fc0b873` (local paths) and CubeLang `2765b58` are 0046's / housekeeping, not this order.
- **Files:**
  - `cubbyllm/reasoning/heldout.py` (the switches; `asked_span`: the name with the article the asker wrote, EN the, FR
    le/la/les/l');
  - `cubbyllm/sources/wikidata_asked.py` (the name as asked searched first, in French when it starts with a French
    article) and its hook in `wikidata_resolve.py`; `wikidata.py` files a fact found through it under the walk's seed;
  - `cubbyllm/reasoning/ask_class.py` (the class asked; the walk up `located in`) and its call in `ask_memory._checked`;
    `sleep.ROUTES` routes `class_mismatch` to `source_gaps` (a decision Cowork may move);
  - `cubbyllm/sources/wikidata_units.py` (the unit: the unit item's P5061 symbol, else its label; a statement with no
    unit whose property has an allowed-units constraint, Q21514353, is not served; a property without one is said bare);
  - `tests/reasoning/test_heldout_fixes.py`; `validation/retrieval_bench/sweep_name_edits.py`; `r0_denominator.py` gains
    `--tag` and writes every spoken answer to a scratch file for reading by hand.
- **Tests:** 17 in `test_heldout_fixes.py` (the four titles in EN and FR, off reproducing the defect; items named as
  asked that lack the relation refuse; JFK refuses with `class_mismatch` and walks to Dallas with `CB_CLASS_WALK=1`;
  Big Ben "96 m", a height with no unit not served, a count bare; the class word; the relation-word rule; on by
  default). Whole suite: 1168 passed, 1 failed (`test_every_module_declares_wiring`, order 0002's known red).
- **Measured** (all live, 2026-10-31, today's defaults otherwise; the off arms run the same day on the same cache):
  - **R0 again, same dev sample** (`retrieval_R0_2026-10-31_heldout.json`; off: `retrieval_R0_2026-10-31_off.json`):
    **76 spoken**: 62 right, 12 unscored, 2 scored wrong (62 right on 2026-10-28; today off: 57 right, 14 unscored,
    6 wrong). **Every spoken answer read by hand: 0 wrong.** The two scored wrong are the unit text against the gold:
    Big Ben "96 m" against "315 feet" (the same height), and Claudia Schiffer "180 cm" against QALD's 180.5 (Wikidata
    states 180 cm; a half-centimetre between sources, Question 2). The six: Freud in French now answered (Sigmund
    Freud); Freud in English, The Ides of March, The Last Dragon and "the skin" refused (several items carry the name as
    asked, or the item asked lacks the relation); JFK refused, `class_mismatch` ("a hospital"). New right answers from
    the name as asked: the Fantastic Mr. Fox, The Poseidon Adventure, The Black Dahlia, The Spoiler, The Chinese Parrot,
    The Magic School Bus... One right answer lost: "Who authored The Mysterious Island?" (several items named so carry an
    author). The 12 unscored read right as the source states them; two worth Cowork's eye: Australia's continent
    "Insular Oceania", George V's cause of death "euthanasia". R0's test split stays unread.
  - **The scoreboard** (`everyday_webq_on_live_tense_0047_2026-10-31.json`; off: `..._0047_off_2026-10-31.json`):
    **75 spoken** (74 on 2026-10-27 and off today), 0 wrong. Every changed row read: one new answer ("what kind of money
    does the uk use?": pound sterling); four refusals change reason (the uk's time zone; three "form of government of
    the united states", now refused because two works titled exactly "The United States" lack the relation, so the
    shorter name may not win alone -- refused before too, as several values).
  - **The stack's 20** (`0047_stack_live_2026-10-31.json`; off: `0047_stack_live_off_2026-10-31.json`): **11 spoken**,
    the same 11, 0 wrong, nothing changed.
  - **0014's bench** (`exp_hdc_bench_2026-10-31_0047.json`): 192 correct, 0 wrong, 0 time-sensitive said flat.
    **0033's probe** (`namesake_probe_2026-10-31_0047.json`): 0 mixed.
  - **The sweep** (`sweep_name_edits_2026-10-31.json`, R0's dev sample, 2,650 rows; the table reads 166 EN and 17 FR;
    the mirror agrees with `EverydayForms.parse` on every row): name-editing transforms and how often each fired --
    `forms.the` (a leading "the" cut, `question_forms.Forms._match`) 42 EN; `everyday.noise` (a noise word cut from the
    span) 5; `forms.as_of` (the year phrase taken out) 2; `everyday.state` ("<state> state", "state of <state>") 1;
    `fr.article` (a French frame's "de l'", "du", "de la", "des", "au", "aux", "à" read as frame) 1;
    `everyday.year_end` 0. The fix keeps the article at the source rather than stopping the reader's cut: the reader
    still reads "the" as grammar, and the source decides with the name as written.
  - **On by default** (item 4): the lines held, so `CB_HELDOUT_FIXES` is on unless `=0`; `CB_CLASS_WALK` stays off.
- **Not verified here:** Cowork's reading of the spoken answers above (the line is that reading).
- **Found on the way:**
  - **The class word that is the relation.** The first scoreboard run with the fixes spoke 71: Turkish and Icelandic
    refused (they reach "language" two `subclass of` steps up, natural language -> human language -> language, and the
    order allows one), and "what nationality is jermaine paul?" refused (a country is no instance of "nationality").
    In each, the class word is a wording the reader's own table maps to the planned relation ("language" -> official
    language; "nationality" -> country of citizenship). Then the relation already names the answer's kind and the check
    stands aside. With that, the scoreboard spoke 75. "city" maps to no relation, so JFK is still checked.
  - **The name as asked, as built:** when no item bears the name as asked, the article is grammar ("the United
    Kingdom") and the reader's name resolves as before. When items bear it and none carries the relation, the question
    is refused (the order's "never wins alone"). Where one carries it, that one wins over the shorter-named item.
  - **A fact found through the name as asked is filed under the walk's seed** (its label differs by the article only,
    and the walk looked for the seed).
- **Questions:**
  1. The class check reads one `subclass of` step, as written. Two steps would read Turkish as a language directly;
     with the relation-word rule it no longer matters on these sets. Keep one?
  2. Claudia Schiffer: "180 cm" against QALD's 180.5. I read it as the source's value, not a wrong answer. Agree?
  3. `class_mismatch` is routed to `source_gaps` in `sleep.ROUTES`; `vm_review` (as `answer_type_mismatch`) is the
     other reading.

## Cowork's check (2026-10-31)
Read: the report, the six refusals and the new answers it lists, the relation-word rule, and `sleep.ROUTES`.
**Done stands.** The line was 0 wrong among the spoken answers, read by hand, and it holds on R0 (76 spoken),
the scoreboard (75), the stack (11), the bench and the probe.

**Read as right:**
- the 12 unscored answers, as the source states them, including Australia's continent "Insular Oceania"
  (Wikidata's label for Oceania) and George V's "euthanasia" (his physician's injection, a matter of record).
  Both are true; the first is a wording cost, like the CFP franc.

**Accepted:**
- **The lost answer:** "Who authored The Mysterious Island?". Several works bear the name as asked and carry an
  author, so R2's arms decide it (answer each, or ask), not a tier.
- **The relation-word rule.** When the reader's table maps the class word to the planned relation, the relation
  already names the answer's kind. It stays pinned by its test, and "city", which maps to no relation, stays
  checked.

**The questions:**
1. **Keep one `subclass of` step.** A false mismatch refuses, which is safe; a wider step could pass the wrong
   kind. The relation-word rule covers the measured cases. 0046's arm B2 reads kinds through the possibility
   worlds and can revisit this with numbers.
2. **Agree:** 180 cm is the source's value, not a wrong answer. Scoring is 0046's to fix (the harness only): compare
   quantities after converting units ("96 m" = "315 feet"), within a stated relative tolerance (0.5%), and report
   those rows as `near`, the match `sleep` already knows.
3. **Keep `source_gaps`,** beside `not_what_was_asked`, its kin. While `CB_CLASS_WALK` is off, the nightly pass
   also records whether the walk up `located in` would have reached the class asked, and verified. Those counts
   are what Cowork reads to switch the walk on.
