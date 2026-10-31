# 0047 — Six wrong answers on held-out questions: the name kept as asked, the class asked, the unit
status: open
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
<!-- Claude Code fills this in, then sets status: done (or review, as above) -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
